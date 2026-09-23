"""Sanity-check + cross-operator-adverb + training-utility analysis for the new
`data_two_color` collection batch (2026-09-02).

That batch is 8 single-(colour, manner, operator) LeRobot session dirs living at
`../data_two_color/` (sibling of `dataset/demo{1,3,4}`, not under it -- see
dio.TWO_COLOR_DATASET_ROOT). It only uses red/blue, which is exactly
diagnose_language_grounding.py's DEFAULT_DESCOPE_REFERENTS -- this batch reads as the
data-collection half of that 2026-09-01 descope decision. It also varies `manner`
(gently/normally/firmly), which is a *second* language axis this project already has an
open, gating question about (H4 in the proposal doc / analyze_cross_operator_adverbs.py's
Figure 6 on the old dataset).

This script answers three things, reusing already-tested code wherever it exists rather than
reimplementing it (repo convention -- see analyze_cross_operator_adverbs.py's own docstring):

1. **Is the batch trustworthy to read at all?** Re-runs the manifest<->dataset episode-index
   mapping check documented in language_grounding_issue_handoff.md's "MAJOR CORRECTION" section
   (raw manifest episode_index has discard-gaps; the dataset's own episode_index is a compacted
   0..N-1 renumbering -- matching the two directly silently scored the wrong episode in 0/92
   cases on the old dataset). Reuses diagnose_language_grounding._load_manifest_ordered
   directly, plus checks schema (meta/info.json `features`) against the existing training
   dataset so pooling doesn't need new dataloader code.

2. **Cross-operator adverbs**: reuses analyze_cross_operator_adverbs.py's own approach
   (itself built on fr3_bilateral_teleop/dataset_tools/evaluation/analyze_adverb_separation.py's
   per-episode contact-force + Cohen's-d, already self-tested there) to ask, for *this* batch:
   within each operator, do the mark's the two manners it actually used separate in force? And
   where a manner word is shared across operators (only "firmly" is -- see finding below), do
   the two operators agree?

3. **Colour-position confound**, the same check that falsified this exact hypothesis on the old
   dataset (language_grounding_issue_handoff.md, "What this is NOT" #2) -- but not assumed to
   still hold here, because this batch's collection protocol is structurally different: each
   session is single-colour (mark position isn't reshuffled *within* a session for other
   colours, since there's only one colour per session here). Computes within-session wipe-
   position spread and, where an operator+manner pair has both a red and a blue session,
   the red-vs-blue position distance -- to see whether colour is confounded with the (largely
   session-fixed) board position in this batch specifically.

Usage:
    python3 scripts/analyze_data_two_color.py
Writes reports/data_two_color_report.json, reports/per_episode_data_two_color.csv, and
reports/figure7_data_two_color_adverb_force.png. Prints a training-utility verdict.
"""

import glob
import json
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(PROJECT_ROOT)
EXTERNAL_SCRIPTS = os.path.join(
    PROJECT_ROOT, "hardware", "fr3_bilateral_teleop", "dataset_tools", "evaluation")  # dataset evaluation
sys.path.insert(0, EXTERNAL_SCRIPTS)
sys.path.insert(0, SCRIPT_DIR)

import dataset_io as dio  # noqa: E402
from analyze_adverb_separation import (  # noqa: E402
    manner_from_task_text, per_episode_contact_force, cohens_d, load_episode_table,
)
from diagnose_language_grounding import _load_manifest_ordered  # noqa: E402

OUT_DIR = os.path.join(PROJECT_ROOT, "reports")
CONTACT_FORCE_THRESHOLD = 2.0
TAIL_FRACTION = 0.7  # matches diagnose_language_grounding.load_observation's ground-truth window

ADVERB_COLORS = {"gently": "#4fa35c", "normally": "#2a78d6", "firmly": "#eb6834"}


# ---------------------------------------------------------------------------
# 1. Manifest<->dataset mapping integrity + schema compatibility
# ---------------------------------------------------------------------------

def check_episode_mapping(session, dataset_root=None):
    """Reuses the exact frame_count cross-check that caught the raw-vs-compacted
    episode_index bug on the old dataset (see this module's docstring, part 1).
    Returns (n_checked, n_mismatched, mismatches)."""
    session_path = os.path.join(dataset_root or dio.TWO_COLOR_DATASET_ROOT, session)
    manifest_entries = _load_manifest_ordered(session_path)
    dataset_dir = Path(session_path)
    episodes = load_episode_table(dataset_dir)

    mismatches = []
    for _, row in episodes.iterrows():
        ds_idx = int(row["episode_index"])
        chunk_idx, file_idx = int(row["data/chunk_index"]), int(row["data/file_index"])
        data_file = dataset_dir / "data" / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.parquet"
        n_frames = pq.read_table(
            data_file, columns=["episode_index"]
        ).to_pandas().eval("episode_index == @ds_idx").sum()
        if ds_idx >= len(manifest_entries):
            mismatches.append((ds_idx, "no manifest counterpart"))
            continue
        expected = manifest_entries[ds_idx].get("frame_count")
        if expected != n_frames:
            mismatches.append((ds_idx, f"manifest frame_count={expected} vs dataset={n_frames}"))
    return len(episodes), len(mismatches), mismatches


RAW_INTEGRITY_COLUMNS = [
    "observation.state", "action", "observation.wrench.external_base",
    "observation.leader_pose", "observation.velocity",
]


def check_raw_data_integrity(session, dataset_root=None):
    """NaNs, duplicate (episode_index, frame_index) rows, and non-monotonic per-episode
    timestamps -- basic corruption checks that the frame_count cross-check above doesn't
    catch (that one only verifies episode *length*, not row content)."""
    df = dio.load_frames(
        session, columns=RAW_INTEGRITY_COLUMNS + ["episode_index", "frame_index", "timestamp"],
        dataset_root=dataset_root or dio.TWO_COLOR_DATASET_ROOT,
    )
    nan_cols = {}
    for c in RAW_INTEGRITY_COLUMNS:
        n_nan = int(np.isnan(np.stack(df[c].to_numpy())).sum())
        if n_nan:
            nan_cols[c] = n_nan
    dup_rows = int(df.duplicated(subset=["episode_index", "frame_index"]).sum())
    non_monotonic = 0
    for _, g in df.groupby("episode_index"):
        if not g.sort_values("frame_index")["timestamp"].is_monotonic_increasing:
            non_monotonic += 1
    return {"n_rows": len(df), "nan_columns": nan_cols, "duplicate_rows": dup_rows,
            "non_monotonic_episodes": non_monotonic}


def check_schema_compatibility(sessions=None, dataset_root=None):
    """Compares a batch's meta/info.json `features` block against the existing
    training dataset (dataset/demo1) -- if these differ, this batch cannot be pooled into
    training via the existing dataloader (src/compliance_vla/policy/dataset.py) without code changes."""
    old_features = dio.load_info("demo1")["features"]
    reports = {}
    for session in (sessions or dio.TWO_COLOR_SESSIONS):
        new_features = dio.load_info(session, dataset_root=dataset_root or dio.TWO_COLOR_DATASET_ROOT)["features"]
        reports[session] = "identical" if new_features == old_features else "DIFFERS"
    return reports


# ---------------------------------------------------------------------------
# 2. Cross-operator adverbs (reuses analyze_cross_operator_adverbs.py's approach)
# ---------------------------------------------------------------------------

def build_per_episode_table(sessions=None, dataset_root=None, condition_lookup=None):
    """`condition_lookup`: optional {session: {"colour":..., "position":..., "manner":...}}
    (e.g. dio.FACTORIAL_CONDITIONS) -- fills in `colour`/`position` from the known per-session
    experimental condition instead of (or in addition to) the manifest, for batches like
    data_factorial_2c where position is a session-level design factor never written to
    session_manifest.jsonl or mentioned in the task text (unlike colour and manner, both
    parsed from the task string as before)."""
    dataset_root = dataset_root or dio.TWO_COLOR_DATASET_ROOT
    rows = []
    for session in (sessions or dio.TWO_COLOR_SESSIONS):
        dataset_dir = Path(dio.session_dir(session, dataset_root))
        manifest = {r["episode_index"]: r for r in _iter_manifest_by_dataset_index(session, dataset_root)}
        episodes = load_episode_table(dataset_dir)
        condition = (condition_lookup or {}).get(session, {})
        for _, row in episodes.iterrows():
            ds_idx = int(row["episode_index"])
            task_text = row["tasks"][0] if len(row["tasks"]) else ""
            manner = manner_from_task_text(task_text) or condition.get("manner")
            stats = per_episode_contact_force(dataset_dir, row, CONTACT_FORCE_THRESHOLD)
            manifest_entry = manifest.get(ds_idx, {})
            rows.append({
                "session": session,
                "episode_index": ds_idx,
                "task": task_text,
                "manner": manner,
                "colour": manifest_entry.get("colour") or condition.get("colour"),
                "position": condition.get("position"),
                "operator_id": manifest_entry.get("operator_id"),
                **stats,
            })
    return pd.DataFrame(rows)


def _iter_manifest_by_dataset_index(session, dataset_root=None):
    """Manifest entries keyed by dataset (compacted) episode_index -- see
    _load_manifest_ordered's docstring for why this must be position-in-timestamp-order,
    not the manifest's own raw `episode_index` field."""
    session_path = os.path.join(dataset_root or dio.TWO_COLOR_DATASET_ROOT, session)
    for ds_idx, entry in enumerate(_load_manifest_ordered(session_path)):
        yield {**entry, "episode_index": ds_idx}


def adverb_analysis(df):
    df = df[df["mean_contact_force_n"].notna() & df["operator_id"].notna()].copy()

    crosstab = (df.groupby(["operator_id", "manner"])["episode_index"]
                .count().rename("n_episodes").reset_index())

    summary = (df.groupby(["operator_id", "manner"])["mean_contact_force_n"]
               .agg(n="count", mean_force_n="mean", sd_force_n="std").reset_index())

    effects = {}
    # Within-operator: does that operator's own set of manners separate?
    for op in sorted(df["operator_id"].unique()):
        manners_used = sorted(df[df.operator_id == op]["manner"].unique())
        pairs_used = [(m1, m2) for i, m1 in enumerate(manners_used) for m2 in manners_used[i + 1:]]
        for m1, m2 in pairs_used:
            a = df[(df.operator_id == op) & (df.manner == m2)]["mean_contact_force_n"].to_numpy()
            b = df[(df.operator_id == op) & (df.manner == m1)]["mean_contact_force_n"].to_numpy()
            if len(a) >= 2 and len(b) >= 2:
                d = cohens_d(a, b)
                effects[f"operator_{op}_{m2}_vs_{m1}_d"] = d

    # Cross-operator, for whichever manner words are actually shared.
    manners_by_op = df.groupby("operator_id")["manner"].apply(set).to_dict()
    ops = sorted(manners_by_op)
    shared_manners = set.intersection(*manners_by_op.values()) if len(ops) > 1 else set()
    op_only_manners = {op: manners_by_op[op] - shared_manners for op in ops}
    for manner in sorted(shared_manners):
        by_op = {op: g["mean_contact_force_n"].to_numpy()
                 for op, g in df[df.manner == manner].groupby("operator_id")}
        if len(ops) == 2 and all(len(by_op.get(o, [])) >= 2 for o in ops):
            d = cohens_d(by_op[ops[1]], by_op[ops[0]])
            effects[f"{manner}_operator_{ops[1]}_vs_{ops[0]}_d"] = d

    return {
        "crosstab": crosstab,
        "summary": summary,
        "effects": effects,
        "shared_manners": sorted(shared_manners),
        "operator_only_manners": {op: sorted(v) for op, v in op_only_manners.items()},
    }


def plot_figure7(df, out_path):
    operators = sorted(df["operator_id"].dropna().unique())
    fig, axes = plt.subplots(1, len(operators), figsize=(5.5 * len(operators), 4.4), sharey=True)
    if len(operators) == 1:
        axes = [axes]

    for ax, op in zip(axes, operators):
        sub = df[df.operator_id == op]
        manners = sorted(sub["manner"].unique(), key=lambda m: ["gently", "normally", "firmly"].index(m)
                          if m in ("gently", "normally", "firmly") else 99)
        data = [sub[sub.manner == m]["mean_contact_force_n"].to_numpy() for m in manners]
        bp = ax.boxplot(
            data, positions=range(len(manners)), widths=0.5, patch_artist=True,
            medianprops=dict(color="#0b0b0b", linewidth=2),
            whiskerprops=dict(color="#52514e", linewidth=1.5),
            capprops=dict(color="#52514e", linewidth=1.5),
            boxprops=dict(linewidth=1.5),
            flierprops=dict(marker="o", markersize=4, markerfacecolor="#52514e",
                             markeredgecolor="none", alpha=0.6),
        )
        for patch, m in zip(bp["boxes"], manners):
            color = ADVERB_COLORS.get(m, "#8a8a86")
            patch.set_facecolor(color)
            patch.set_alpha(0.85)
            patch.set_edgecolor(color)

        rng = np.random.default_rng(0)
        for i, vals in enumerate(data):
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals, s=18, color="#0b0b0b",
                       alpha=0.5, zorder=3, linewidths=0)

        ax.set_title(f"Operator {op}  (n={'+'.join(str(len(d)) for d in data)})",
                     fontsize=11, color="#0b0b0b")
        ax.set_xticks(range(len(manners)))
        ax.set_xticklabels(manners, fontsize=10.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#c3c2b7")
        ax.spines["bottom"].set_color("#c3c2b7")
        ax.tick_params(colors="#52514e")
        ax.grid(axis="y", color="#e5e4df", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Mean contact-force magnitude (N)", fontsize=10.5, color="#0b0b0b")
    fig.suptitle("Figure 7: data_two_color -- adverb -> realized contact force, by operator\n"
                  "(gently: operator A only, normally: operator B only, firmly: both)",
                  fontsize=12.5, color="#0b0b0b")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.subplots_adjust(top=0.80)
    fig.savefig(out_path, dpi=200, facecolor="white")


# ---------------------------------------------------------------------------
# 3. Colour-position confound
# ---------------------------------------------------------------------------

SPEED_PAUSE_THRESHOLD_M_S = 0.01  # m/s, matches src/compliance_vla/policy/dataset.py / diagnose_language_grounding.py
NEAR_HOME_RADIUS_M = 0.05  # m, matches src/compliance_vla/policy/dataset.py / diagnose_language_grounding.py


def episode_tail_position(session, ds_idx, chunk_idx, file_idx, dataset_root=None):
    """Ground-truth wipe position for this episode: the last pause (near-zero velocity) that
    is also far from the episode's own start position -- not a blind trailing fraction of
    episode length, and not a contact-force threshold either (both tried and were wrong).

    The data_recorder version that collected data_two_color (commit 2f63ca3) appends
    a still-recording return-to-start_joint_configuration move to every saved episode before
    the 's' key actually stops it (see that repo's README), so TAIL_FRACTION's old assumption
    -- that the trailing 30% of an episode is the wipe endpoint -- instead mostly/entirely
    samples the fixed home pose. A contact-force-threshold fix attempt was ALSO wrong: the
    appended move produces external-wrench estimation transients from the fast commanded
    motion itself that routinely exceed a naive 2N threshold throughout the return trip (this
    function's own per-session centroids, with that fix, converged to the SAME x/y/z across
    every session regardless of operator/colour/manner -- exactly what a single shared
    start_joint_configuration looks like, not 8 different marks -- rather than the tighter,
    per-session-distinct spread a real fix should produce).

    Real wiping is reliably followed by a near-zero-velocity pause (arm lifts off, holds
    still) before the reset motion begins as continuous glide with no further stop until it
    arrives back near the episode's own starting pose (episodes idle at
    start_joint_configuration before wiping begins). Walking backward from episode end and
    requiring BOTH slow and far-from-start skips the final arrival-at-home settling pause
    (also near-zero-velocity) and lands on the real "wipe done" moment instead. See
    diagnose_language_grounding.py's matching _true_wipe_endpoint and src/compliance_vla/policy/dataset.py's
    _trim_to_last_contact for the same fix elsewhere -- kept as a separate copy here since this
    module reads by (chunk_idx, file_idx) rather than an already-loaded episode frame, not
    worth threading a shared helper through for one more caller."""
    data_file = os.path.join(
        dio.session_dir(session, dataset_root or dio.TWO_COLOR_DATASET_ROOT),
        "data", f"chunk-{chunk_idx:03d}", f"file-{file_idx:03d}.parquet",
    )
    df = pq.read_table(
        data_file, columns=["episode_index", "observation.leader_pose", "timestamp"]
    ).to_pandas()
    df = df[df["episode_index"] == ds_idx]
    pos = np.stack(df["observation.leader_pose"].to_numpy())[:, :3]
    t = df["timestamp"].to_numpy().astype(np.float64)
    n = len(pos)
    start_pos = pos[0]
    dist_from_start = np.linalg.norm(pos - start_pos, axis=1)
    dt = np.diff(t)
    with np.errstate(invalid="ignore", divide="ignore"):
        speed = np.linalg.norm(np.diff(pos, axis=0), axis=1) / dt
    k = n - 2
    while k >= 0 and not (speed[k] < SPEED_PAUSE_THRESHOLD_M_S and dist_from_start[k] > NEAR_HOME_RADIUS_M):
        k -= 1
    if k < 0:
        print(f"  ** WARNING: {session} episode {ds_idx} never pauses away from its own start "
              "position -- falling back to trailing-30% ground truth (may itself be "
              "contaminated by appended home motion). **")
        tail = df.iloc[int(len(df) * TAIL_FRACTION):]
        return np.median(np.stack(tail["observation.leader_pose"].to_numpy())[:, :3], axis=0)
    pause_idx = k + 1
    window = df.iloc[max(0, pause_idx - 4):pause_idx + 1]
    return np.median(np.stack(window["observation.leader_pose"].to_numpy())[:, :3], axis=0)


def position_confound_analysis(sessions=None, dataset_root=None, condition_lookup=None):
    dataset_root = dataset_root or dio.TWO_COLOR_DATASET_ROOT
    rows = []
    for session in (sessions or dio.TWO_COLOR_SESSIONS):
        manifest_by_idx = dict(enumerate(_load_manifest_ordered(
            os.path.join(dataset_root, session))))
        episodes = load_episode_table(Path(dio.session_dir(session, dataset_root)))
        condition = (condition_lookup or {}).get(session, {})
        for _, row in episodes.iterrows():
            ds_idx = int(row["episode_index"])
            pos = episode_tail_position(session, ds_idx, int(row["data/chunk_index"]), int(row["data/file_index"]),
                                         dataset_root=dataset_root)
            entry = manifest_by_idx.get(ds_idx, {})
            rows.append({
                "session": session, "episode_index": ds_idx,
                "colour": entry.get("colour") or condition.get("colour"),
                "manner": entry.get("manner") or condition.get("manner"),
                "position": condition.get("position"),
                "operator_id": entry.get("operator_id"),
                "x": pos[0], "y": pos[1], "z": pos[2],
            })
    pos_df = pd.DataFrame(rows)

    per_session = pos_df.groupby("session").agg(
        n=("episode_index", "count"),
        colour=("colour", "first"), manner=("manner", "first"), operator_id=("operator_id", "first"),
        centroid_x=("x", "mean"), centroid_y=("y", "mean"), centroid_z=("z", "mean"),
        std_x=("x", "std"), std_y=("y", "std"), std_z=("z", "std"),
    ).reset_index()
    per_session["max_dist_from_centroid_m"] = per_session.apply(
        lambda r: pos_df[pos_df.session == r["session"]].apply(
            lambda row: np.linalg.norm(
                [row.x - r.centroid_x, row.y - r.centroid_y, row.z - r.centroid_z]
            ), axis=1).max(),
        axis=1,
    )

    # Matched red/blue pairs sharing (operator_id, manner) -- do colours share a board spot?
    matched_pairs = []
    for (op, manner), group in per_session.groupby(["operator_id", "manner"]):
        colours = dict(zip(group["colour"], zip(group["centroid_x"], group["centroid_y"], group["centroid_z"])))
        if "red" in colours and "blue" in colours:
            dist = float(np.linalg.norm(np.array(colours["red"]) - np.array(colours["blue"])))
            matched_pairs.append({"operator_id": op, "manner": manner, "red_vs_blue_centroid_dist_m": dist})

    return pos_df, per_session, matched_pairs


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    report = {}

    print("=" * 80)
    print("1. Manifest<->dataset episode-index mapping integrity")
    print("=" * 80)
    mapping_report = {}
    for session in dio.TWO_COLOR_SESSIONS:
        n, n_bad, bad = check_episode_mapping(session)
        mapping_report[session] = {"n_episodes": n, "n_mismatched": n_bad, "mismatches": bad}
        status = "OK" if n_bad == 0 else "MISMATCH"
        print(f"  {session:16s} n={n:3d}  mismatched={n_bad}  [{status}]")
    report["episode_mapping"] = mapping_report
    total_bad = sum(v["n_mismatched"] for v in mapping_report.values())
    if total_bad:
        print(f"\n  ** {total_bad} episode(s) failed the frame_count cross-check -- "
              f"do not trust ground truth for those without investigating. **")
    else:
        print(f"\n  All {sum(v['n_episodes'] for v in mapping_report.values())} episodes' "
              f"manifest<->dataset mapping verified via frame_count.")

    print("\n" + "=" * 80)
    print("2. Raw data integrity (NaNs, duplicate rows, timestamp ordering)")
    print("=" * 80)
    integrity_report = {}
    for session in dio.TWO_COLOR_SESSIONS:
        r = check_raw_data_integrity(session)
        integrity_report[session] = r
        status = "OK" if not r["nan_columns"] and not r["duplicate_rows"] and not r["non_monotonic_episodes"] else "ISSUE"
        print(f"  {session:16s} rows={r['n_rows']:5d}  nan_columns={r['nan_columns']}  "
              f"dup_rows={r['duplicate_rows']}  non_monotonic_episodes={r['non_monotonic_episodes']}  [{status}]")
    report["raw_data_integrity"] = integrity_report

    print("\n" + "=" * 80)
    print("3. Schema compatibility with existing training dataset (dataset/demo1)")
    print("=" * 80)
    schema_report = check_schema_compatibility()
    report["schema_compatibility"] = schema_report
    for session, status in schema_report.items():
        print(f"  {session:16s} {status}")

    print("\n" + "=" * 80)
    print("4. Colour x manner x operator coverage, and cross-operator adverb agreement")
    print("=" * 80)
    df = build_per_episode_table()
    df.to_csv(os.path.join(OUT_DIR, "per_episode_data_two_color.csv"), index=False)

    coverage = (df.groupby(["colour", "manner", "operator_id"])["episode_index"]
                .count().rename("n_episodes").reset_index())
    print(coverage.to_string(index=False))
    report["colour_manner_operator_coverage"] = coverage.to_dict(orient="records")

    adverb = adverb_analysis(df)
    print("\nPer operator x manner force summary:")
    print(adverb["summary"].to_string(index=False))
    print(f"\nManner words used by operator A only: {adverb['operator_only_manners'].get('A', [])}")
    print(f"Manner words used by operator B only: {adverb['operator_only_manners'].get('B', [])}")
    print(f"Manner words shared by both operators: {adverb['shared_manners']}")
    print("\nCohen's d effect sizes:")
    for k, v in adverb["effects"].items():
        print(f"  {k}: {v:+.2f}")
    report["adverb_analysis"] = {
        "crosstab": adverb["crosstab"].to_dict(orient="records"),
        "summary": adverb["summary"].to_dict(orient="records"),
        "cohens_d": adverb["effects"],
        "shared_manners": adverb["shared_manners"],
        "operator_only_manners": adverb["operator_only_manners"],
    }

    fig_path = os.path.join(OUT_DIR, "figure7_data_two_color_adverb_force.png")
    plot_figure7(df[df["mean_contact_force_n"].notna() & df["operator_id"].notna()], fig_path)
    print(f"\nsaved figure -> {fig_path}")

    print("\n" + "=" * 80)
    print("5. Colour-position confound (within-session + matched red/blue pairs)")
    print("=" * 80)
    pos_df, per_session, matched_pairs = position_confound_analysis()
    print(per_session[["session", "colour", "manner", "operator_id", "n",
                        "max_dist_from_centroid_m"]].to_string(index=False))
    print("\nRed-vs-blue board position, matched by (operator, manner):")
    for mp in matched_pairs:
        print(f"  operator={mp['operator_id']} manner={mp['manner']}: "
              f"red/blue centroid distance = {mp['red_vs_blue_centroid_dist_m']*100:.1f} cm")
    report["position_confound"] = {
        "per_session": per_session.drop(columns=[c for c in per_session.columns if c.startswith("std_")])
                                   .to_dict(orient="records"),
        "matched_red_blue_pairs": matched_pairs,
    }

    with open(os.path.join(OUT_DIR, "data_two_color_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\nfull report -> {os.path.join(OUT_DIR, 'data_two_color_report.json')}")


if __name__ == "__main__":
    main()
