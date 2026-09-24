#!/usr/bin/env python3
"""Fits the per-operator log_k calibration src/compliance_vla/policy/labels.py's calibrate_log_k applies.

Why: analyze_cross_operator_adverbs.py (old dataset) and analyze_data_two_color.py (new
batch, post-relabel) both found the same thing -- operators A and B realize systematically
different absolute contact force/stiffness for the *same* manner word (e.g. data_two_color:
firmly d=-2.48, normally d=-1.93 between operators). Trained as-is, "firmly"/"normally"
would supervise a blend of two operator-specific targets rather than one consistent concept.
This is fixed at the *label* level (log_k, not the raw wrench/pose recordings, which stay
exactly as recorded) with a per-axis affine correction:

    corrected = (raw - operator_mean) / operator_std * reference_std + reference_mean

"Equal split" reference (per the 2026-09-02 decision): reference_mean/reference_std for a
manner are the *unweighted* average of each operator's own mean/std for that manner -- e.g.
operator B has fewer episodes than A in data_two_color (49 vs 96), and a sample-count-weighted
pooled mean would let A's larger count dominate the reference, which is exactly the kind of
one-operator-skew this correction is meant to remove. Averaging the two operators' own
per-manner statistics equally, regardless of how many episodes back each one, is what "equal
split" means here.

**Scoped to dio.TWO_COLOR_SESSIONS (data_two_color/*) only, deliberately -- not pooled with
the old dataset/demo{1,3,4}.** An earlier version pooled both, reasoning that both batches
show the same operator disagreement so calibrating on one alone would just swap "inconsistent
across operators" for "inconsistent across dataset batches." That turned out to be wrong in
practice: verification (2026-09-02) found operator B's "normally" reference, once pooled,
was dominated by the *old* dataset's demo3 (only 3 of data_two_color's own 25 B/normally
episodes survive contact-frame fitting -- see MIN_CALIBRATION_EPISODES below), and demo3's
own B/normally force level (7.56N, raw wrench-magnitude proxy) is ~2x data_two_color's
(3.74N) -- a batch-level shift bigger than the operator effect this script exists to correct.
Pooling silently applied a today-batch correction sourced mostly from a different batch's
physical calibration regime. Restricting to today's batch only removes that risk: any
(operator, manner) cell without enough of *today's own* episodes now correctly falls back to
identity (uncalibrated) via MIN_CALIBRATION_EPISODES, rather than being rescued by borrowing
a different batch's numbers.

Consequence, explicit: this calibration file is fit for data_two_color sessions specifically.
Applying it (src/compliance_vla/policy/dataset.py's ComplianceWindowDataset does so automatically, keyed only
on operator_id/manner) to old dataset/demo{1,3,4} episodes in the same training run would
mis-correct them using today's-batch statistics for the same operator/manner label -- don't
mix old-dataset sessions into a run that also relies on this file without addressing that
first (either refit pooled again once the *old* dataset's own batch-level differences are
understood, or keep old-dataset training calibration-free).

Uses each session's own session_manifest.jsonl for operator_id/manner via
diagnose_language_grounding's _load_manifest_ordered (the raw-manifest-vs-compacted-dataset-
index mapping fix already established and tested there) rather than re-deriving that
mapping here.

Usage:
    python -m compliance_vla.policy.fit_manner_force_calibration
Writes data_extraction/manner_force_calibration.json (loaded by compliance_vla.policy.labels.load_manner_calibration).
"""

import json
import os
import sys
from collections import defaultdict

import numpy as np

from compliance_vla.policy._paths import DATA_EXTRACTION_DIR, ensure_on_sys_path  # noqa: E402

ensure_on_sys_path()  # dataset_io / panda_fk are plain scripts under data_extraction/

import dataset_io as dio  # noqa: E402
from compliance_vla.policy.diagnose_language_grounding import _load_manifest_ordered  # noqa: E402
from compliance_vla.policy import labels as lb  # noqa: E402

MIN_CALIBRATION_COUNT = 5  # minimum masked (identifiable) *samples* to trust a fitted mean/std
# Frame/window-level counts alone are misleading: a handful of episodes can each contribute
# hundreds of (correlated, not independent) windows, making a fit look well-supported when
# it's really pseudo-replicated from very few actual demonstrations. Found empirically
# (2026-09-02 calibration verification): operator B's "normally" cell looked like thousands
# of samples but came from only 3 episodes (demo_redB/demo_blueB's other 22 fail the
# extraction pipeline's own contact-frame fit -- see that verification's finding, a real
# upstream data-quality issue in those two sessions, not a bug in this script) -- fitting
# against those 3 episodes' idiosyncratic mean/std made the "correction" worse, not better,
# for several axes. Require a minimum number of *distinct episodes* too.
MIN_CALIBRATION_EPISODES = 8
OUT_PATH = os.path.join(DATA_EXTRACTION_DIR, "manner_force_calibration.json")

# (session, dataset_root) pairs to pool over -- data_two_color (today's batch) only, see
# module docstring for why the old dataset/demo{1,3,4} is deliberately excluded here.
ALL_SESSIONS = [(s, dio.TWO_COLOR_DATASET_ROOT) for s in dio.TWO_COLOR_SESSIONS]


def collect_raw_log_k():
    """(operator_id, manner) -> list of (axis, values, episode_key) tuples, pooled across
    every session and episode, restricted to mask=True (identifiable) entries only -- an
    unidentifiable axis carries no real force information to calibrate against.
    episode_key = (session, episode_index), carried through so fit_stats can count distinct
    contributing *episodes* per axis, not just samples -- see MIN_CALIBRATION_EPISODES."""
    tool_offset = lb.load_tool_offset()
    args = lb.default_extraction_args()
    sigma_f = lb.default_sigma_f()

    buckets = defaultdict(list)
    n_episodes, n_failed = 0, 0
    fail_counts_by_session = defaultdict(int)

    for session, root in ALL_SESSIONS:
        frames = dio.load_frames(session, columns=dio.NON_IMAGE_COLUMNS, dataset_root=root)
        frames = frames.sort_values(["episode_index", "frame_index"])
        manifest_entries = _load_manifest_ordered(dio.session_dir(session, root))

        for ep_idx, g in frames.groupby("episode_index"):
            n_episodes += 1
            g = g.sort_values("frame_index")
            arrays = lb.compute_episode_arrays(g, tool_offset, args, sigma_f)  # raw, no calibration
            if arrays is None:
                n_failed += 1
                fail_counts_by_session[session] += 1
                continue
            if ep_idx >= len(manifest_entries):
                continue
            entry = manifest_entries[ep_idx]
            operator_id = entry.get("operator_id")
            manner = entry.get("manner") or dio.parse_manner_from_task(entry.get("task", ""))
            if operator_id is None or manner is None:
                continue
            log_k, mask = arrays["log_k"], arrays["mask"]
            episode_key = (session, int(ep_idx))
            for axis in range(log_k.shape[1]):
                vals = log_k[mask[:, axis], axis]
                if len(vals):
                    buckets[(operator_id, manner)].append((axis, vals, episode_key))

    print(f"{n_episodes} episodes scanned, {n_failed} failed contact-frame fitting (skipped)")
    if n_failed:
        print("  failures by session:", dict(fail_counts_by_session))
    return buckets


def fit_stats(buckets):
    """buckets: (operator_id, manner) -> list of (axis, values, episode_key) tuples. Returns
    operator_stats[operator][manner] = {mean, std, fitted, n_per_axis, n_episodes_per_axis}
    (6-vectors, `fitted` bool per axis) and reference[manner] = equal-split average across
    operators. An axis is only marked `fitted` with enough *samples* (MIN_CALIBRATION_COUNT)
    AND enough distinct *episodes* (MIN_CALIBRATION_EPISODES) -- see that constant's
    docstring for why sample count alone is misleading."""
    operator_stats = defaultdict(dict)
    per_manner_operator_means = defaultdict(dict)  # manner -> operator -> (mean(6,), std(6,))

    for (operator_id, manner), entries in buckets.items():
        by_axis = defaultdict(list)
        episodes_by_axis = defaultdict(set)
        for axis, vals, episode_key in entries:
            by_axis[axis].extend(vals.tolist())
            episodes_by_axis[axis].add(episode_key)

        mean = np.zeros(6)
        std = np.ones(6)
        fitted = np.zeros(6, dtype=bool)
        counts = np.zeros(6, dtype=int)
        episode_counts = np.zeros(6, dtype=int)
        for axis, vals in by_axis.items():
            counts[axis] = len(vals)
            episode_counts[axis] = len(episodes_by_axis[axis])
            if counts[axis] >= MIN_CALIBRATION_COUNT and episode_counts[axis] >= MIN_CALIBRATION_EPISODES:
                mean[axis] = float(np.mean(vals))
                std[axis] = float(np.std(vals))
                fitted[axis] = std[axis] > 0

        operator_stats[operator_id][manner] = {
            "mean": mean.tolist(), "std": std.tolist(), "fitted": fitted.tolist(),
            "n_per_axis": counts.tolist(), "n_episodes_per_axis": episode_counts.tolist(),
        }
        per_manner_operator_means[manner][operator_id] = (mean, std, fitted)

    reference = {}
    for manner, by_op in per_manner_operator_means.items():
        ops = sorted(by_op)
        # Equal-split: unweighted mean across operators, axis-by-axis, only over operators
        # that actually had a fitted value for that axis (so one operator's missing/unfitted
        # axis doesn't silently pull the reference toward its zero-initialized default).
        ref_mean = np.zeros(6)
        ref_std = np.zeros(6)
        for axis in range(6):
            means_axis = [by_op[op][0][axis] for op in ops if by_op[op][2][axis]]
            stds_axis = [by_op[op][1][axis] for op in ops if by_op[op][2][axis]]
            if means_axis:
                ref_mean[axis] = float(np.mean(means_axis))
                ref_std[axis] = float(np.mean(stds_axis))
            else:
                ref_std[axis] = 1.0  # no operator had this axis fitted -- identity fallback
        reference[manner] = {
            "mean": ref_mean.tolist(), "std": ref_std.tolist(),
            "operators_averaged": ops,
        }

    return dict(operator_stats), reference


def main():
    buckets = collect_raw_log_k()
    operator_stats, reference = fit_stats(buckets)

    print("\nPer operator x manner log_k stats:")
    for op in sorted(operator_stats):
        for manner, stats in sorted(operator_stats[op].items()):
            n_fitted = sum(stats["fitted"])
            print(f"  operator={op} manner={manner}: {n_fitted}/6 axes fitted, "
                  f"n_episodes_per_axis={stats['n_episodes_per_axis']}, "
                  f"n_samples_per_axis={stats['n_per_axis']}")

    print("\nEqual-split reference (unweighted average across operators):")
    for manner, stats in sorted(reference.items()):
        print(f"  manner={manner}: operators_averaged={stats['operators_averaged']}, "
              f"mean={np.round(stats['mean'], 3).tolist()}")

    out = {
        "description": "Per-operator log_k calibration for cross-operator manner grounding "
                        "-- see src/compliance_vla/policy/fit_manner_force_calibration.py docstring.",
        "reference_method": "equal_split",
        "min_calibration_count": MIN_CALIBRATION_COUNT,
        "sessions_used": [s for s, _ in ALL_SESSIONS],
        "operators": operator_stats,
        "reference": reference,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
