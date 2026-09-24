#!/usr/bin/env python3
"""Per-axis contact-force traces with phase annotation, for a LeRobot-format demo
(data_recorder output, e.g. dataset_collection/19082601) -- the extraction figure script
(stiffness traces with phase annotation, Figure 3).

**This is NOT Figure 3 as specified, and the gap is deliberate, not an oversight.**
Figure 3 is supposed to show extracted per-axis STIFFNESS K(t) (the extraction regression of
f = K*(x_l - x_f) + D*(...)). That regression needs the LEADER pose x_l(t) as an independent
measurement -- it is the entire point of §2.2(b), the identifiability argument that is this
paper's technical core. `extract_impedance_labels.py` already implements that regression and
already has its own Figure-3 generator (`generate_figure`, see that file) -- but it consumes
CSVs from `record_demo.py`, which logs `x_l, x_f, f` together.

Checked directly against this project's LeRobot recorder (`data_recorder/
lerobot_recorder_node.py`) before writing this script: `pose_topic`/`joint_topic` both default
to the FOLLOWER namespace only, there is no leader-pose parameter at all, and `observation.state`
+ `action` are both built from the follower's own joint/EE state. **The bulk-collection LeRobot
datasets under dataset_collection/ do not contain the leader trajectory**, so
`extract_impedance_labels.py`'s real K(t) regression cannot be run against them -- not today,
not without either (a) patching the recorder to also log leader pose, or (b) switching bulk
collection back to `record_demo.py`'s CSV format. Substituting anything else for x_l (e.g. a
displacement-from-first-contact proxy) would silently reintroduce the exact equilibrium/
stiffness ambiguity the paper exists to resolve -- so this script does not attempt it.

What this script actually plots, and why it's still useful before that gap is closed: per-axis
wrench (`observation.wrench.external_base`) over time, with background shading for a simple
contact-phase state machine (free-space / approach / contact / retract), using the same
||(fx,fy,fz)|| > threshold convention as `live_force_band_display.py` and
`extract_impedance_labels.py`'s mask condition (4). This is real, on-disk data, produced today,
and it is the honest precursor to Figure 3: it shows *when* contact happens and *how much* force
is applied, which is what a reader needs to sanity-check before trusting any later K(t) plot.

**One dataset dir per run, and multiple episodes can share one data file.** Unlike
`analyze_adverb_separation.py`, this script plots one dataset at a time (call it once per
manner's directory, e.g. once for 19082601/`normally` and once for 19082602/`firmly`). Also:
19082601 happens to have one episode per data file, but that is not a general property of this
recorder -- 19082602 packs several episodes into each file (confirmed directly, e.g. episodes 2
and 3 both live in chunk-000/file-002.parquet), so `load_episode_frames` filters rows by the
`episode_index` column after loading rather than assuming "one file == one episode."

Usage:
    ros2 run fr3_bilateral_teleop plot_demo_force_phase_traces.py \\
        --dataset-dir dataset_collection/19082601 --episode-index 2
    ros2 run fr3_bilateral_teleop plot_demo_force_phase_traces.py \\
        --dataset-dir dataset_collection/19082602 --all --output-dir /tmp/figs
    python3 dataset_tools/evaluation/plot_demo_force_phase_traces.py --self-test
"""
import argparse
import glob
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FORCE_AXIS_NAMES = ["fx", "fy", "fz", "tx", "ty", "tz"]
AXIS_UNITS = ["N", "N", "N", "Nm", "Nm", "Nm"]

PHASE_COLORS = {
    "free-space": (0.85, 0.85, 0.85),
    "approach": (1.0, 0.85, 0.4),
    "contact": (1.0, 0.55, 0.3),
    "retract": (1.0, 0.85, 0.4),
}


def load_episode_table(dataset_dir: Path) -> pd.DataFrame:
    ep_files = sorted(glob.glob(str(dataset_dir / "meta" / "episodes" / "**" / "*.parquet"), recursive=True))
    if not ep_files:
        raise FileNotFoundError(f"no meta/episodes/**/*.parquet under {dataset_dir}")
    return pd.concat([pd.read_parquet(f) for f in ep_files], ignore_index=True)


def load_episode_frames(dataset_dir: Path, episode_row: pd.Series) -> pd.DataFrame:
    chunk_idx = int(episode_row["data/chunk_index"])
    file_idx = int(episode_row["data/file_index"])
    episode_index = int(episode_row["episode_index"])
    path = dataset_dir / "data" / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.parquet"
    df = pd.read_parquet(path, columns=["observation.wrench.external_base", "timestamp", "episode_index"])
    # A data file can hold more than one episode (confirmed on dataset_collection/19082602,
    # e.g. episodes 2 and 3 both live in chunk-000/file-002.parquet) -- filter to this one.
    # Each episode's own `timestamp` restarts at 0.0, so no re-basing is needed after filtering.
    return df[df["episode_index"] == episode_index]


def phase_labels(
        t: np.ndarray, force_mag: np.ndarray, contact_force_threshold: float, margin_sec: float,
) -> np.ndarray:
    """Simple threshold state machine: free-space / approach / contact / retract.

    `approach`/`retract` are fixed-duration windows (margin_sec) immediately before/after each
    contiguous contact run, clipped so they never overlap an adjacent contact run or each other.
    This is a labelling convenience for the figure, not a physical detector -- unlike the mask
    conditions in extract_impedance_labels.py, nothing downstream depends on its precision.
    """
    in_contact = force_mag > contact_force_threshold
    labels = np.full(len(t), "free-space", dtype=object)
    labels[in_contact] = "contact"

    # find contiguous contact runs
    edges = np.flatnonzero(np.diff(in_contact.astype(int)))
    starts = [0] if in_contact[0] else []
    ends = []
    idx = 0
    run_start = None
    for i in range(len(t)):
        if in_contact[i] and run_start is None:
            run_start = i
        elif not in_contact[i] and run_start is not None:
            ends.append(i - 1)
            starts.append(run_start)
            run_start = None
    if run_start is not None:
        ends.append(len(t) - 1)
        starts.append(run_start)
    starts, ends = sorted(starts), sorted(ends)

    for s, e in zip(starts, ends):
        t_s, t_e = t[s], t[e]
        approach_mask = (t >= t_s - margin_sec) & (t < t_s) & (labels == "free-space")
        retract_mask = (t > t_e) & (t <= t_e + margin_sec) & (labels == "free-space")
        labels[approach_mask] = "approach"
        labels[retract_mask] = "retract"
    return labels


def plot_episode(
        dataset_dir: Path, episode_row: pd.Series, contact_force_threshold: float,
        margin_sec: float, output_path: Path,
) -> Dict:
    df = load_episode_frames(dataset_dir, episode_row)
    wrench = np.stack(df["observation.wrench.external_base"].values).astype(np.float64)
    t = df["timestamp"].values.astype(np.float64)
    t = t - t[0]
    force_mag = np.linalg.norm(wrench[:, :3], axis=1)
    labels = phase_labels(t, force_mag, contact_force_threshold, margin_sec)
    task_text = episode_row["tasks"][0] if len(episode_row["tasks"]) else ""

    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    for axis in range(6):
        ax = axes[axis // 2, axis % 2]
        for phase, color in PHASE_COLORS.items():
            mask = labels == phase
            if not mask.any():
                continue
            ax.fill_between(t, 0, 1, where=mask, transform=ax.get_xaxis_transform(),
                             alpha=0.25, color=color, step="mid", label=phase)
        ax.plot(t, wrench[:, axis], color="tab:blue", linewidth=1.0)
        ax.set_title(f"{FORCE_AXIS_NAMES[axis]} ({AXIS_UNITS[axis]})", fontsize=10)
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)
        if axis >= 4:
            ax.set_xlabel("t (s)")
    handles, plot_labels = axes[0, 0].get_legend_handles_labels()
    by_label = dict(zip(plot_labels, handles))
    axes[0, 0].legend(by_label.values(), by_label.keys(), fontsize=7, loc="upper right")
    fig.suptitle(
        f"Episode {int(episode_row['episode_index'])}: \"{task_text}\" -- "
        f"contact-force traces, phase-annotated\n"
        f"(precursor to Figure 3 -- true K(t) blocked, see script docstring)",
        fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    n_contact = int((labels == "contact").sum())
    return {
        "episode_index": int(episode_row["episode_index"]),
        "task": task_text,
        "n_frames": len(t),
        "n_contact_frames": n_contact,
        "contact_fraction": n_contact / len(t) if len(t) else 0.0,
        "output_path": str(output_path),
    }


# ---------------------------------------------------------------------------
# Self-test: synthetic wrench trace, no dataset needed
# ---------------------------------------------------------------------------

def run_self_test() -> int:
    ok = True
    fps = 30.0
    duration = 6.0
    t = np.arange(0, duration, 1.0 / fps)
    force_mag = np.where((t > 2.0) & (t < 4.0), 10.0, 0.5)
    labels = phase_labels(t, force_mag, contact_force_threshold=2.0, margin_sec=0.2)

    n_contact = int((labels == "contact").sum())
    n_approach = int((labels == "approach").sum())
    n_retract = int((labels == "retract").sum())
    n_free = int((labels == "free-space").sum())

    expected_contact = int(((t > 2.0) & (t < 4.0)).sum())
    if abs(n_contact - expected_contact) > 1:
        print(f"FAIL contact-phase count: got {n_contact}, expected ~{expected_contact}")
        ok = False
    else:
        print(f"contact-phase count: got {n_contact}, expected ~{expected_contact} -- OK")

    if n_approach == 0 or n_retract == 0:
        print(f"FAIL expected nonzero approach/retract windows, got approach={n_approach} retract={n_retract}")
        ok = False
    else:
        print(f"approach={n_approach} retract={n_retract} frames -- OK")

    if n_free + n_contact + n_approach + n_retract != len(t):
        print("FAIL phase labels don't partition all timesteps")
        ok = False
    else:
        print("phase labels partition all timesteps -- OK")

    # load_episode_frames must not pool two episodes packed into one data file
    # (real bug on dataset_collection/19082602: episodes 2 and 3 share chunk-000/file-002.parquet)
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "data" / "chunk-000").mkdir(parents=True)
        n_ep0, n_ep1 = 15, 25
        wrench_ep0 = np.tile([3.0, 0.0, 0.0, 0.0, 0.0, 0.0], (n_ep0, 1))
        wrench_ep1 = np.tile([9.0, 0.0, 0.0, 0.0, 0.0, 0.0], (n_ep1, 1))
        df = pd.DataFrame({
            "episode_index": [0] * n_ep0 + [1] * n_ep1,
            "timestamp": list(np.arange(n_ep0) / 30.0) + list(np.arange(n_ep1) / 30.0),
            "observation.wrench.external_base": list(wrench_ep0) + list(wrench_ep1),
        })
        df.to_parquet(tmp_path / "data" / "chunk-000" / "file-000.parquet")
        row0 = pd.Series({"data/chunk_index": 0, "data/file_index": 0, "episode_index": 0})
        frames0 = load_episode_frames(tmp_path, row0)
        wrench0 = np.stack(frames0["observation.wrench.external_base"].values)
        if len(frames0) != n_ep0 or not np.allclose(wrench0[:, 0], 3.0):
            print(f"FAIL load_episode_frames pooled across episodes: got {len(frames0)} frames, "
                  f"fx values {np.unique(wrench0[:, 0])}")
            ok = False
        else:
            print(f"load_episode_frames correctly isolates one episode from a shared file -- OK "
                  f"({len(frames0)} frames, fx={wrench0[0, 0]})")

    print("SELF-TEST PASS" if ok else "SELF-TEST FAIL")
    return 0 if ok else 1


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--episode-index", type=int, default=None,
                         help="LeRobot dataset's own episode_index (meta/episodes), not session_manifest's")
    parser.add_argument("--all", action="store_true", help="plot every episode in the dataset")
    parser.add_argument("--contact-force-threshold", type=float, default=2.0)
    parser.add_argument("--approach-retract-margin-sec", type=float, default=0.2)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_test()

    if args.dataset_dir is None:
        print("error: --dataset-dir required unless --self-test", file=sys.stderr)
        return 2
    if not args.all and args.episode_index is None:
        print("error: pass --episode-index N or --all", file=sys.stderr)
        return 2

    output_dir = args.output_dir or (args.dataset_dir / "figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes = load_episode_table(args.dataset_dir)
    if not args.all:
        episodes = episodes[episodes["episode_index"] == args.episode_index]
        if episodes.empty:
            print(f"error: no episode with episode_index={args.episode_index}", file=sys.stderr)
            return 2

    reports: List[Dict] = []
    for _, row in episodes.iterrows():
        out_path = output_dir / f"episode_{int(row['episode_index']):03d}_force_phase.png"
        result = plot_episode(
            args.dataset_dir, row, args.contact_force_threshold,
            args.approach_retract_margin_sec, out_path)
        print(f"episode {result['episode_index']:3d}: {result['n_contact_frames']}/"
              f"{result['n_frames']} contact frames ({result['contact_fraction']:.0%}) "
              f"-> {result['output_path']}")
        reports.append(result)

    return 0


if __name__ == "__main__":
    sys.exit(main())
