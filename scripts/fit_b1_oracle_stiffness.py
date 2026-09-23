"""Day 13: B1 -- oracle per-axis constant-K, grid-searched on the validation
set (proposal §6.1 baseline table): "Same [as B0], oracle-tuned constant per-
axis compliance (grid-searched on validation set) -- H3 -- the cheap control
baseline. Omitting this is the single fastest route to rejection."

B1's *policy* is B0 (position output, no compliance head, no force input --
already trained, Day 12) run through the *same* low-level variable-impedance
controller as every other baseline, but with the controller's per-axis
stiffness pinned to one fixed vector instead of either B0's own default
fixed-high-stiffness or a learned/predicted one. This script's only job is
to produce that one 6-dim K vector -- there is no new neural network here,
by design (B1 exists specifically to show H2/H3's gain isn't just "any
non-trivial impedance"). The vector is consumed by the Week 4 controller
integration, not by this script.

**Search, not the closed-form mean.** offline_stiffness_benchmark.py's (M8,
Day 10) "constant" baseline is the closed-form TRAIN-split mean of log K --
a sanity-check baseline for the *learned* predictor, picked for cheapness,
not for being the best possible constant. B1 is a different, stronger
baseline: a real grid search over candidate constant stiffnesses, clipped to
the controller's realizable range (§4.1: 50-1500 N/m translational, 5-100
N*m/rad rotational -- same K_MIN_LOG/K_MAX_LOG bounds M8 already uses,
reused unmodified so both scripts agree on what's physically commandable),
selected by minimizing error against the VALIDATION session specifically
(not train, per the proposal's literal wording -- this project's frozen
session-level split, build_dataset_splits.py: train=demo4, val=demo1,
test=demo3), which for a scalar-target grid search is a real, if small,
distinction from a closed-form train-mean: it can land at a different point
than the train mean whenever the val session's own masked-log-K distribution
is shifted or skewed relative to train's (checked and reported below, not
assumed away).

Reuses offline_stiffness_benchmark.py's already-tested `build_axis_datasets`
(extraction + per-axis pooling of masked log_k across a session list)
directly rather than re-deriving it -- only the input features (q, qdot,
manner/referent one-hots) it also returns are unused here, since a constant
predictor ignores input by construction.
"""

import argparse
import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(PROJECT_ROOT)
EXTERNAL_SCRIPTS = os.path.join(
    PROJECT_ROOT, "hardware", "fr3_bilateral_teleop", "dataset_tools", "labeling")  # label extraction
sys.path.insert(0, EXTERNAL_SCRIPTS)
sys.path.insert(0, SCRIPT_DIR)

import dataset_io as dio  # noqa: E402
from extract_impedance_labels import DEFAULT_SIGMA_F, FORCE_AXIS_NAMES, parse_args as extraction_parse_args  # noqa: E402
from offline_stiffness_benchmark import K_MAX_LOG, K_MIN_LOG, build_axis_datasets, rmse  # noqa: E402
import run_extraction_on_dataset as red  # noqa: E402

MIN_SAMPLES = 10  # same floor offline_stiffness_benchmark.py uses before trusting an axis's numbers
DEFAULT_GRID_POINTS = 60


def grid_search_axis(y_val, log_k_min, log_k_max, n_grid, metric="rmse", huber_delta=1.0):
    """Evaluate a log-spaced grid of candidate constants against y_val
    (all identifiable/masked log_k values for one axis, pooled across the
    validation session's episodes), return the best (log_k, score) plus the
    full grid so the search itself is inspectable, not just its argmin."""
    candidates = np.linspace(log_k_min, log_k_max, n_grid)
    scores = np.empty(n_grid)
    for i, c in enumerate(candidates):
        if metric == "rmse":
            scores[i] = rmse(np.full_like(y_val, c), y_val)
        elif metric == "huber":
            resid = c - y_val
            delta = huber_delta
            abs_r = np.abs(resid)
            quad = np.minimum(abs_r, delta)
            lin = abs_r - quad
            scores[i] = np.mean(0.5 * quad**2 + delta * lin)
        else:
            raise ValueError(metric)
    best_i = int(np.argmin(scores))
    return candidates, scores, best_i


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grid-points", type=int, default=DEFAULT_GRID_POINTS,
                         help="candidates per axis, log-spaced across the controller-realizable range")
    parser.add_argument("--metric", choices=["rmse", "huber"], default="rmse",
                         help="grid-search selection objective on the validation session")
    parser.add_argument("--huber-delta", type=float, default=1.0, help="only used if --metric huber")
    parser.add_argument("--out-dir", default=os.path.join(PROJECT_ROOT, "reports"))
    cli = parser.parse_args()
    os.makedirs(cli.out_dir, exist_ok=True)

    if not os.path.exists(red.TOOL_OFFSET_PATH):
        raise FileNotFoundError("tool_offset.npy missing -- run calibrate_tool_offset.py first")
    tool_offset = np.load(red.TOOL_OFFSET_PATH)
    extraction_args = extraction_parse_args([])
    sigma_f = np.array(DEFAULT_SIGMA_F)

    with open(os.path.join(dio.DATASET_ROOT, "franka_vla_multimodal", "splits.json")) as f:
        splits = json.load(f)
    split_sessions = {split: [s["session"] for s in stats["sessions"]] for split, stats in splits.items()}
    print(f"Splits used: {split_sessions}")
    print(f"Grid search objective: {cli.metric}, {cli.grid_points} candidates/axis, "
          f"selected on VAL ({split_sessions['val']}), reported also against TRAIN/TEST for context.\n")

    print("Building per-axis datasets (train/val/test)...")
    train_data = build_axis_datasets(split_sessions["train"], tool_offset, extraction_args, sigma_f)
    val_data = build_axis_datasets(split_sessions["val"], tool_offset, extraction_args, sigma_f)
    test_data = build_axis_datasets(split_sessions["test"], tool_offset, extraction_args, sigma_f)

    report = {
        "splits": split_sessions, "metric": cli.metric, "grid_points": cli.grid_points,
        "k_min_max_realizable_range": {
            axis: {"k_min": float(np.exp(K_MIN_LOG[a])), "k_max": float(np.exp(K_MAX_LOG[a]))}
            for a, axis in enumerate(FORCE_AXIS_NAMES)
        },
        "per_axis": {},
    }
    k_vector_log = {}
    k_vector = {}

    for a, axis in enumerate(FORCE_AXIS_NAMES):
        _, y_tr = train_data[axis]
        _, y_va = val_data[axis]
        _, y_te = test_data[axis]
        n_tr, n_va, n_te = len(y_tr), len(y_va), len(y_te)
        print(f"=== axis {axis}: n_train={n_tr} n_val={n_va} n_test={n_te} ===")

        if n_va < MIN_SAMPLES:
            print(f"  insufficient VAL data (< {MIN_SAMPLES}) -- cannot grid-search this axis, skipping")
            report["per_axis"][axis] = {"status": "insufficient_val_data", "n_train": n_tr, "n_val": n_va, "n_test": n_te}
            continue

        candidates, scores, best_i = grid_search_axis(
            y_va, K_MIN_LOG[a], K_MAX_LOG[a], cli.grid_points, metric=cli.metric, huber_delta=cli.huber_delta,
        )
        best_log_k = float(candidates[best_i])
        best_val_score = float(scores[best_i])
        best_k = float(np.exp(best_log_k))
        grid_step = float(candidates[1] - candidates[0]) if cli.grid_points > 1 else float("nan")

        # Context, not selection criteria: same oracle constant scored against train/test too, and against
        # the train-mean closed-form baseline M8 already reports, so a reader can see whether grid-search-on-
        # val actually differs from (and beats) the simpler train-mean choice, rather than assuming it does.
        train_mean_log_k = float(y_tr.mean()) if n_tr > 0 else None
        val_score_train_mean = rmse(np.full_like(y_va, train_mean_log_k), y_va) if train_mean_log_k is not None else None
        train_score = rmse(np.full_like(y_tr, best_log_k), y_tr) if n_tr > 0 else None
        test_score = rmse(np.full_like(y_te, best_log_k), y_te) if n_te >= MIN_SAMPLES else None
        test_score_train_mean = (
            rmse(np.full_like(y_te, train_mean_log_k), y_te) if (train_mean_log_k is not None and n_te >= MIN_SAMPLES) else None
        )

        print(f"  best constant: log_k={best_log_k:.4f} (k={best_k:.2f}) val_{cli.metric}={best_val_score:.4f} "
              f"[grid step {grid_step:.4f} log-units, range ({K_MIN_LOG[a]:.3f}, {K_MAX_LOG[a]:.3f})]")
        if val_score_train_mean is not None:
            beats = "beats" if best_val_score <= val_score_train_mean else "does NOT beat"
            print(f"  vs. train-mean constant (log_k={train_mean_log_k:.4f}): val_{cli.metric}={val_score_train_mean:.4f} "
                  f"-- grid search {beats} the closed-form train mean on VAL")
        if test_score is not None:
            print(f"  held-out TEST rmse at this constant: {test_score:.4f}"
                  + (f" (train-mean constant: {test_score_train_mean:.4f})" if test_score_train_mean is not None else ""))

        k_vector_log[axis] = best_log_k
        k_vector[axis] = best_k
        report["per_axis"][axis] = {
            "status": "ok", "n_train": n_tr, "n_val": n_va, "n_test": n_te,
            "best_log_k": best_log_k, "best_k": best_k,
            f"val_{cli.metric}_at_best": best_val_score,
            "grid_step_log_units": grid_step,
            "train_mean_log_k": train_mean_log_k,
            f"val_{cli.metric}_at_train_mean": val_score_train_mean,
            "grid_search_beats_train_mean_on_val": (
                bool(best_val_score <= val_score_train_mean) if val_score_train_mean is not None else None
            ),
            "train_rmse_at_best": train_score,
            "test_rmse_at_best": test_score,
            "test_rmse_at_train_mean": test_score_train_mean,
        }
        print()

    n_axes_ok = sum(1 for v in report["per_axis"].values() if v["status"] == "ok")
    report["summary"] = {
        "n_axes_evaluated": n_axes_ok, "n_axes_total": len(FORCE_AXIS_NAMES),
        "k_vector_contact_frame": k_vector,  # N/m or N*m/rad depending on axis, per §4.1's contact-frame convention
        "log_k_vector_contact_frame": k_vector_log,
    }

    print("=" * 70)
    print(f"B1 oracle constant stiffness (contact frame): "
          + ", ".join(f"{ax}={k_vector.get(ax, float('nan')):.2f}" for ax in FORCE_AXIS_NAMES))
    print(f"{n_axes_ok}/{len(FORCE_AXIS_NAMES)} axes had enough VAL data to grid-search; "
          "axes without a value here should NOT be silently defaulted at Week-4 controller-integration time "
          "-- see status='insufficient_val_data' entries in the JSON report.")
    print("=" * 70)

    out_path = os.path.join(cli.out_dir, "b1_oracle_constant_stiffness.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
