#!/usr/bin/env python3
"""Adverb -> force separation check (a pre-registration item).

Computes Cohen's d / d' between the per-manner contact-force distributions of a LeRobot-format
dataset produced by data_recorder (e.g. dataset_collection/19082601), to check
whether `gently`/`normally`/`firmly` actually produced separated force distributions. If they
did not, H4 (the language-generalization claim) is unfalsifiable and the operator protocol
needs fixing before more demos are collected -- this is a gate, not a nice-to-have plot.

**Manner is read from each episode's own recorded task text (`meta/episodes/*.parquet` ->
`tasks[0]`), not from `session_manifest.jsonl`.** Checked directly against 19082601: the
manifest's `episode_index` is the recorder's own per-session counter (skips index 5, a
rejected/discarded take) while the LeRobot dataset's `episode_index` is a separate, contiguous
0..N-1 re-index assigned when the dataset was built. The two are NOT the same integer space and
silently zipping them together would mislabel every episode after the skipped one. Task text
carries the manner word directly (`_instruction_text()` in lerobot_recorder_node.py appends the
manner into the instruction), so matching on that string is index-space-agnostic and correct by
construction.

**Force metric: ||(fx, fy, fz)|| from `observation.wrench.external_base`, same convention as
`live_force_band_display.py`.** That script's docstring documents why: the operator-facing
adverb bands (gently 3-6 N, normally 8-12 N, firmly 15-22 N) describe the board-normal force,
but board-normal decomposition needs a per-demo contact-frame fit; at firm contact the measured
anisotropy ratio is 7-14x (normal >> lateral), so raw wrench-force magnitude is a reasonable
first-order stand-in without adding a dependency on the frame-fit step. Using the same proxy
here keeps this check comparable to what the operator saw live during collection.

**Multiple dataset directories, and multiple episodes packed into one parquet file.** `normally`
and `firmly` were collected as two separate LeRobot datasets (dataset_collection/19082601 and
.../19082602 as of this writing) -- pass both, or the parent `dataset_collection/` directory and
let it auto-discover every immediate subdirectory that has a `meta/info.json`. Also: 19082601
happens to have one episode per data file, but 19082602 does not (e.g. episodes 2 and 3 both
live in chunk-000/file-002.parquet) -- confirmed by inspecting `data/episode_index` value counts
directly. `per_episode_contact_force` filters rows by the `episode_index` column after loading
rather than assuming "one file == one episode"; that assumption would have silently pooled two
episodes' wrench traces together on 19082602.

Usage:
    ros2 run fr3_bilateral_teleop analyze_adverb_separation.py --dataset-dir dataset_collection
    ros2 run fr3_bilateral_teleop analyze_adverb_separation.py \\
        --dataset-dir dataset_collection/19082601 dataset_collection/19082602
    python3 dataset_tools/evaluation/analyze_adverb_separation.py --self-test
"""
import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

MANNERS = ["gently", "normally", "firmly"]


def manner_from_task_text(task_text: str) -> str:
    lowered = task_text.lower()
    hits = [m for m in MANNERS if m in lowered]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        return "ambiguous:" + "+".join(hits)
    return "unlabeled"


def resolve_dataset_dirs(paths: List[Path]) -> List[Path]:
    """Each input path is either a LeRobot dataset root itself (has meta/info.json) or a
    parent directory to auto-discover dataset roots under (one level down only)."""
    resolved: List[Path] = []
    for p in paths:
        if (p / "meta" / "info.json").exists():
            resolved.append(p)
        else:
            subdirs = sorted(d for d in p.iterdir() if d.is_dir() and (d / "meta" / "info.json").exists())
            if not subdirs:
                raise FileNotFoundError(
                    f"{p} is neither a LeRobot dataset root (no meta/info.json) nor a parent "
                    f"of any (no subdirectory has one either)")
            resolved.extend(subdirs)
    return resolved


def load_episode_table(dataset_dir: Path) -> pd.DataFrame:
    ep_files = sorted(glob.glob(str(dataset_dir / "meta" / "episodes" / "**" / "*.parquet"), recursive=True))
    if not ep_files:
        raise FileNotFoundError(f"no meta/episodes/**/*.parquet under {dataset_dir}")
    frames = [pd.read_parquet(f) for f in ep_files]
    return pd.concat(frames, ignore_index=True)


def per_episode_contact_force(
        dataset_dir: Path, episode_row: pd.Series, contact_force_threshold: float,
) -> Dict:
    chunk_idx = int(episode_row["data/chunk_index"])
    file_idx = int(episode_row["data/file_index"])
    episode_index = int(episode_row["episode_index"])
    data_path = dataset_dir / "data" / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.parquet"
    df = pd.read_parquet(data_path, columns=["observation.wrench.external_base", "episode_index"])
    # A data file can hold more than one episode (confirmed on 19082602) -- filter to this one.
    df = df[df["episode_index"] == episode_index]
    wrench = np.stack(df["observation.wrench.external_base"].values).astype(np.float64)
    force_mag = np.linalg.norm(wrench[:, :3], axis=1)
    in_contact = force_mag > contact_force_threshold
    n_contact = int(in_contact.sum())
    return {
        "n_frames": int(len(force_mag)),
        "n_contact_frames": n_contact,
        "mean_contact_force_n": float(force_mag[in_contact].mean()) if n_contact > 0 else None,
        "peak_contact_force_n": float(force_mag[in_contact].max()) if n_contact > 0 else None,
    }


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Standardized mean difference (mean(a) - mean(b)) / pooled_sd.

    Reported as both "Cohen's d" and "d'" -- for two approximately
    Gaussian force distributions these are the same computation (d' from signal-detection
    theory is exactly this quantity when both distributions share a common sigma estimate);
    no separate d' formula is used.
    """
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        raise ValueError("need at least 2 samples in each group for a pooled-SD effect size")
    var_a, var_b = a.var(ddof=1), b.var(ddof=1)
    pooled_sd = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled_sd == 0:
        return float("inf") if a.mean() != b.mean() else 0.0
    return float((a.mean() - b.mean()) / pooled_sd)


def analyze(dataset_dirs: List[Path], contact_force_threshold: float, min_contact_frames: int) -> Dict:
    per_episode = []
    for dataset_dir in dataset_dirs:
        episodes = load_episode_table(dataset_dir)
        for _, row in episodes.iterrows():
            task_text = row["tasks"][0] if len(row["tasks"]) else ""
            manner = manner_from_task_text(task_text)
            stats = per_episode_contact_force(dataset_dir, row, contact_force_threshold)
            per_episode.append({
                "dataset_dir": str(dataset_dir),
                "episode_index": int(row["episode_index"]),
                "task": task_text,
                "manner": manner,
                **stats,
            })

    groups: Dict[str, List[float]] = {}
    low_contact_episodes = []
    for ep in per_episode:
        if ep["mean_contact_force_n"] is None or ep["n_contact_frames"] < min_contact_frames:
            low_contact_episodes.append(f"{ep['dataset_dir']}#{ep['episode_index']}")
            continue
        groups.setdefault(ep["manner"], []).append(ep["mean_contact_force_n"])

    group_summary = {
        manner: {
            "n": len(vals),
            "mean_n": float(np.mean(vals)) if vals else None,
            "sd_n": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
            "values_n": vals,
        }
        for manner, vals in groups.items()
    }

    pairwise = {}
    present = [m for m in MANNERS if m in groups]
    for i in range(len(present)):
        for j in range(i + 1, len(present)):
            m1, m2 = present[i], present[j]
            a, b = np.array(groups[m1]), np.array(groups[m2])
            key = f"{m1}_vs_{m2}"
            if len(a) < 2 or len(b) < 2:
                pairwise[key] = {
                    "status": "insufficient_data",
                    "n1": len(a), "n2": len(b),
                    "reason": "need n>=2 in both groups for a pooled-SD effect size",
                }
            else:
                pairwise[key] = {
                    "status": "ok",
                    "n1": len(a), "n2": len(b),
                    "cohens_d": cohens_d(a, b),
                }

    missing_manners = [m for m in MANNERS if m not in groups]

    return {
        "dataset_dirs": [str(d) for d in dataset_dirs],
        "contact_force_threshold_n": contact_force_threshold,
        "min_contact_frames": min_contact_frames,
        "n_episodes_total": len(per_episode),
        "n_episodes_excluded_low_contact": len(low_contact_episodes),
        "excluded_episode_indices": low_contact_episodes,
        "per_episode": per_episode,
        "group_summary": group_summary,
        "pairwise_effect_sizes": pairwise,
        "missing_manners": missing_manners,
    }


def print_report(report: Dict) -> None:
    print(f"--- {', '.join(report['dataset_dirs'])} ---")
    print(f"{report['n_episodes_total']} episodes total, "
          f"{report['n_episodes_excluded_low_contact']} excluded "
          f"(< {report['min_contact_frames']} contact frames at "
          f"{report['contact_force_threshold_n']} N threshold)")
    print()
    print("Per-manner contact-force summary (mean-in-contact ||f||, N):")
    for manner in MANNERS + ["unlabeled"]:
        g = report["group_summary"].get(manner)
        if g is None:
            print(f"  {manner:10s}: 0 demos")
            continue
        sd_str = f"{g['sd_n']:.2f}" if g["sd_n"] is not None else "n/a (n=1)"
        print(f"  {manner:10s}: n={g['n']:2d}  mean={g['mean_n']:.2f} N  sd={sd_str}  "
              f"values={[round(v, 2) for v in g['values_n']]}")
    print()
    print("Pairwise Cohen's d / d' (mean-in-contact force, pooled SD):")
    if not report["pairwise_effect_sizes"]:
        print("  (no manner pair has data at all)")
    for key, res in report["pairwise_effect_sizes"].items():
        if res["status"] == "ok":
            print(f"  {key:20s}: d={res['cohens_d']:+.2f}  (n1={res['n1']}, n2={res['n2']})")
        else:
            print(f"  {key:20s}: INSUFFICIENT DATA (n1={res['n1']}, n2={res['n2']}) -- {res['reason']}")
    print()
    if report["missing_manners"]:
        print(f"MISSING MANNERS (0 demos with >= {report['min_contact_frames']} contact frames): "
              f"{report['missing_manners']}")
    n_present_with_data = sum(1 for m in MANNERS if report["group_summary"].get(m, {}).get("n", 0) >= 2)
    if n_present_with_data < 2:
        print("VERDICT: H4 separation check BLOCKED -- fewer than 2 manners have n>=2 demos. "
              "Cannot compute a meaningful three-way d' yet. Collect the missing manner(s) "
              "before treating the adverb axis as validated.")
    else:
        print(f"VERDICT: {n_present_with_data}/3 manners have n>=2 demos; see pairwise d' above. "
              "Still short of a full three-way comparison until all three are present." if n_present_with_data < 3
              else "VERDICT: all three manners have n>=2 demos; see pairwise d' above for separation.")


# ---------------------------------------------------------------------------
# Self-test: synthetic Cohen's d check + manner-extraction check, no dataset needed
# ---------------------------------------------------------------------------

def run_self_test() -> int:
    ok = True

    # manner_from_task_text
    cases = [
        ("wipe the blue mark normally", "normally"),
        ("wipe the red mark", "unlabeled"),
        ("wipe the black mark firmly", "firmly"),
        ("wipe the green mark gently", "gently"),
    ]
    for text, expected in cases:
        got = manner_from_task_text(text)
        if got != expected:
            print(f"FAIL manner_from_task_text({text!r}) = {got!r}, expected {expected!r}")
            ok = False

    # cohens_d against a known analytic case: two unit-variance Gaussians offset by 2 sigma
    rng = np.random.default_rng(0)
    a = rng.normal(loc=10.0, scale=2.0, size=5000)
    b = rng.normal(loc=14.0, scale=2.0, size=5000)
    d = cohens_d(a, b)
    expected_d = (10.0 - 14.0) / 2.0  # -2.0
    if abs(d - expected_d) > 0.1:
        print(f"FAIL cohens_d large-sample check: got {d:.3f}, expected ~{expected_d:.3f}")
        ok = False
    else:
        print(f"cohens_d large-sample check: got {d:.3f}, expected ~{expected_d:.3f} -- OK")

    # insufficient-data path must not raise, must be caught explicitly
    try:
        cohens_d(np.array([1.0]), np.array([2.0, 3.0]))
        print("FAIL cohens_d should have raised on n=1 group")
        ok = False
    except ValueError:
        print("cohens_d correctly raises on n=1 group -- OK")

    # per_episode_contact_force must not pool two episodes packed into one data file
    # (real bug on 19082602: episodes 2 and 3 share chunk-000/file-002.parquet)
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "data" / "chunk-000").mkdir(parents=True)
        n_ep0, n_ep1 = 20, 30
        wrench_ep0 = np.tile([10.0, 0.0, 0.0, 0.0, 0.0, 0.0], (n_ep0, 1))  # ||f|| = 10 N throughout
        wrench_ep1 = np.tile([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], (n_ep1, 1))   # 0 N throughout
        df = pd.DataFrame({
            "episode_index": [0] * n_ep0 + [1] * n_ep1,
            "observation.wrench.external_base": list(wrench_ep0) + list(wrench_ep1),
        })
        df.to_parquet(tmp_path / "data" / "chunk-000" / "file-000.parquet")
        row0 = pd.Series({"data/chunk_index": 0, "data/file_index": 0, "episode_index": 0})
        stats0 = per_episode_contact_force(tmp_path, row0, contact_force_threshold=2.0)
        if stats0["n_frames"] != n_ep0 or stats0["mean_contact_force_n"] != 10.0:
            print(f"FAIL per_episode_contact_force pooled across episodes: got {stats0}")
            ok = False
        else:
            print(f"per_episode_contact_force correctly isolates one episode from a shared file "
                  f"-- OK ({stats0['n_frames']} frames, mean {stats0['mean_contact_force_n']} N)")

    print("SELF-TEST PASS" if ok else "SELF-TEST FAIL")
    return 0 if ok else 1


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dataset-dir", type=Path, nargs="+", default=None,
        help="one or more LeRobot dataset roots, or parent dir(s) to auto-discover roots under")
    parser.add_argument("--contact-force-threshold", type=float, default=2.0,
                         help="N, same default as extract_impedance_labels.py's mask condition (4)")
    parser.add_argument("--min-contact-frames", type=int, default=10,
                         help="episodes with fewer in-contact frames than this are excluded from group stats")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_test()

    if args.dataset_dir is None:
        print("error: --dataset-dir required unless --self-test", file=sys.stderr)
        return 2

    dataset_dirs = resolve_dataset_dirs(args.dataset_dir)
    report = analyze(dataset_dirs, args.contact_force_threshold, args.min_contact_frames)
    print_report(report)

    output_json = args.output_json or (dataset_dirs[0].parent / "adverb_separation_report.json")
    with open(output_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nJSON report written to {output_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
