"""Sanity-check + factor-decorrelation analysis for the `data_factorial_2c` collection batch
(2x2x2: colour x position x manner, 12 episodes/cell, 96 total).

Context: the 2026-09-05 analysis found that the previous
2-colour batch (data_two_color) had two confounds -- colour was effectively fixed to one board
position, and the deployed checkpoint's predictions collapsed onto whichever colour a given
*manner* happened to be recorded with, at chance accuracy. This new batch crosses colour with
both position (left/right) and manner (normal/firm) independently so neither can stand in for
colour -- see dataset_io.FACTORIAL_CONDITIONS for the session-to-condition mapping (rename its
keys if the actual collected folder names differ).

Reuses analyze_data_two_color.py's already-tested checks wherever possible (repo convention --
see that module's own docstring) rather than reimplementing them: episode-mapping integrity,
raw data integrity, schema compatibility, and the v3 velocity+distance-gated wipe-endpoint
extraction inside its position_confound_analysis. This module adds what that one didn't need:
a decorrelation check across all THREE factors, not just colour.

**What "good" looks like**, and why: holding the other two factors fixed, contrasting one
factor's two levels should give a LARGE, consistent centroid distance for **position** (that's
the real thing the model must select between) and a SMALL distance for **colour** and
**manner** (changing the ink colour or the wipe force at a fixed position-label should target
the *same* physical spot -- if it doesn't, the position label doesn't mean the same thing
across colours/manners, which is a collection-time inconsistency, not the color-vision
confound this design is meant to fix, but just as damaging to training on it).

Usage:
    python3 scripts/analyze_data_factorial.py [--dataset-root PATH]
Writes reports/data_factorial_report.json and reports/per_episode_data_factorial.csv, prints a
training-utility verdict.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_extraction"))  # dataset_io

import dataset_io as dio  # noqa: E402
from analyze_data_two_color import (  # noqa: E402
    check_episode_mapping, check_raw_data_integrity, check_schema_compatibility,
    build_per_episode_table, position_confound_analysis, adverb_analysis,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reports")

FACTORS = ("colour", "position", "manner")


def factor_contrasts(per_session, factor):
    """Holding the other two factors fixed, find the pair of sessions differing only in
    `factor` and report their centroid distance -- one row per (other-factor-combo). With a
    clean 2x2x2 design there are exactly 4 such pairs per factor."""
    other = [f for f in FACTORS if f != factor]
    rows = []
    for _, group in per_session.groupby(other):
        if len(group) != 2 or group[factor].nunique() != 2:
            continue
        a, b = group.iloc[0], group.iloc[1]
        dist = float(np.linalg.norm(
            [a.centroid_x - b.centroid_x, a.centroid_y - b.centroid_y, a.centroid_z - b.centroid_z]
        ))
        rows.append({
            **{o: a[o] for o in other},
            f"{factor}_pair": f"{a[factor]} vs {b[factor]}",
            "centroid_dist_m": dist,
        })
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", default=dio.FACTORIAL_DATASET_ROOT,
                   help="default: dataset_io.FACTORIAL_DATASET_ROOT (../data_factorial_2c)")
    p.add_argument("--sessions", nargs="+", default=dio.FACTORIAL_SESSIONS,
                   help="default: dataset_io.FACTORIAL_SESSIONS (all 8 cells)")
    args = p.parse_args()

    condition_lookup = {s: dio.FACTORIAL_CONDITIONS[s] for s in args.sessions if s in dio.FACTORIAL_CONDITIONS}
    missing = [s for s in args.sessions if s not in dio.FACTORIAL_CONDITIONS]
    if missing:
        print(f"** WARNING: {missing} not in dio.FACTORIAL_CONDITIONS -- their colour/position/"
              "manner will be missing from this report. Update that dict if the real collected "
              "folder names differ from the placeholder convention. **")

    os.makedirs(OUT_DIR, exist_ok=True)
    report = {}

    print("=" * 80)
    print("1. Manifest<->dataset episode-index mapping integrity")
    print("=" * 80)
    mapping_report = {}
    for session in args.sessions:
        n, n_bad, bad = check_episode_mapping(session, dataset_root=args.dataset_root)
        mapping_report[session] = {"n_episodes": n, "n_mismatched": n_bad, "mismatches": bad}
        print(f"  {session:24s} n={n:3d}  mismatched={n_bad}  [{'OK' if n_bad == 0 else 'MISMATCH'}]")
    report["episode_mapping"] = mapping_report

    print("\n" + "=" * 80)
    print("2. Raw data integrity (NaNs, duplicate rows, timestamp ordering)")
    print("=" * 80)
    integrity_report = {}
    for session in args.sessions:
        r = check_raw_data_integrity(session, dataset_root=args.dataset_root)
        integrity_report[session] = r
        status = "OK" if not r["nan_columns"] and not r["duplicate_rows"] and not r["non_monotonic_episodes"] else "ISSUE"
        print(f"  {session:24s} rows={r['n_rows']:5d}  nan_columns={r['nan_columns']}  "
              f"dup_rows={r['duplicate_rows']}  non_monotonic_episodes={r['non_monotonic_episodes']}  [{status}]")
    report["raw_data_integrity"] = integrity_report

    print("\n" + "=" * 80)
    print("3. Schema compatibility with existing training dataset (dataset/demo1)")
    print("=" * 80)
    schema_report = check_schema_compatibility(sessions=args.sessions, dataset_root=args.dataset_root)
    report["schema_compatibility"] = schema_report
    for session, status in schema_report.items():
        print(f"  {session:24s} {status}")

    print("\n" + "=" * 80)
    print("4. Colour x position x manner coverage")
    print("=" * 80)
    df = build_per_episode_table(sessions=args.sessions, dataset_root=args.dataset_root,
                                  condition_lookup=condition_lookup)
    df.to_csv(os.path.join(OUT_DIR, "per_episode_data_factorial.csv"), index=False)
    coverage = (df.groupby(["colour", "position", "manner"])["episode_index"]
                .count().rename("n_episodes").reset_index())
    print(coverage.to_string(index=False))
    report["coverage"] = coverage.to_dict(orient="records")

    print("\n" + "=" * 80)
    print("5. Per-cell wipe-position centroids + within-cell spread (v3 ground truth)")
    print("=" * 80)
    pos_df, per_session, _ = position_confound_analysis(
        sessions=args.sessions, dataset_root=args.dataset_root, condition_lookup=condition_lookup
    )
    print(per_session[["session", "colour", "position", "manner", "n",
                        "max_dist_from_centroid_m"]].to_string(index=False))
    report["per_session"] = per_session.drop(
        columns=[c for c in per_session.columns if c.startswith("std_")]
    ).to_dict(orient="records")

    print("\n" + "=" * 80)
    print("6. Factor decorrelation: centroid-distance contrasts holding the other two factors fixed")
    print("=" * 80)
    print("Want: LARGE + consistent 'position' contrasts (that's the real target selection);")
    print("      SMALL 'colour' and 'manner' contrasts (same nominal position should mean the")
    print("      same physical spot regardless of which colour/manner was recorded there).\n")
    contrasts = {}
    for factor in FACTORS:
        rows = factor_contrasts(per_session, factor)
        contrasts[factor] = rows
        dists = [r["centroid_dist_m"] for r in rows]
        print(f"-- {factor} contrasts ({len(rows)} pairs) --")
        for r in rows:
            other_desc = ", ".join(f"{k}={v}" for k, v in r.items() if k not in ("centroid_dist_m",) and not k.endswith("_pair"))
            print(f"  [{other_desc}] {r[f'{factor}_pair']}: {r['centroid_dist_m']*100:.1f} cm")
        if dists:
            print(f"  -> mean={np.mean(dists)*100:.1f}cm  min={np.min(dists)*100:.1f}cm  max={np.max(dists)*100:.1f}cm\n")
    report["factor_contrasts"] = contrasts

    pos_contrasts = [r["centroid_dist_m"] for r in contrasts["position"]]
    colour_contrasts = [r["centroid_dist_m"] for r in contrasts["colour"]]
    manner_contrasts = [r["centroid_dist_m"] for r in contrasts["manner"]]
    verdict_lines = []
    if pos_contrasts and colour_contrasts:
        ratio = np.mean(colour_contrasts) / np.mean(pos_contrasts) if np.mean(pos_contrasts) > 0 else float("inf")
        colour_verdict = (
            "GOOD -- colour contrast is much smaller than position contrast, position label "
            "looks colour-independent"
        ) if ratio < 0.3 else (
            "WARNING -- colour contrast is not much smaller than position contrast, position "
            "label may not mean the same physical spot across colours"
        )
        verdict_lines.append(f"colour/position contrast ratio = {ratio:.2f} ({colour_verdict})")
    if pos_contrasts and manner_contrasts:
        ratio = np.mean(manner_contrasts) / np.mean(pos_contrasts) if np.mean(pos_contrasts) > 0 else float("inf")
        verdict_lines.append(
            f"manner/position contrast ratio = {ratio:.2f} "
            f"({'GOOD' if ratio < 0.3 else 'WARNING -- manner may be confounded with wipe location'})"
        )
    print("\n".join(verdict_lines))
    report["verdict"] = verdict_lines

    with open(os.path.join(OUT_DIR, "data_factorial_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\nfull report -> {os.path.join(OUT_DIR, 'data_factorial_report.json')}")


if __name__ == "__main__":
    main()
