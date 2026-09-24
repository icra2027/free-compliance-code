"""M8: offline stiffness-prediction benchmark -- learned predictor vs
constant predictor vs nearest-neighbour, on held-out sessions. Also evaluates
the learnability check: learned prediction must beat the constant predictor on
held-out sessions, or the labels are noise and no online experiment will
rescue them.

**Task definition (not fully specified by the method -- a documented choice,
same spirit as evaluate_gate1.py's (ii)/(iii)).** Per axis, per identifiable
(masked) 30Hz timestep, predict log K(t) in the contact frame from a
proprioception + language feature vector: [q (7), q_dot (7), manner one-hot
(2: normally/firmly), referent one-hot (4: red/blue/green/black)] = 20 dims.
Log-space matches how the policy head itself is trained (masked Huber on
log k) and how K spans decades. Deliberately
excludes force/wrench features: the point of M8 is whether compliance is
predictable from context resembling what a policy conditions on, not from
other force channels that would make the check partly circular. Unmasked
(non-identifiable) timesteps are excluded entirely, matching "masked entries
are excluded from the loss, not imputed".

**Splits.** Uses this project's frozen session-level split (build_dataset_splits.py):
train=demo4, val=demo1, test=demo3 (session counts 1/1/1 -- thin, flagged
there and again here, not hidden). Val selects the learned predictor's
hyperparameters (never test); test is the one and only number the learnability check looks at.

**Models:**
  - constant: per-axis mean(log K) over TRAIN's masked timesteps, ignoring
    input entirely -- the trivial baseline the learnability check requires beating.
  - nearest-neighbour: 1-NN in standardized feature space (sklearn
    NearestNeighbors), label copied from the nearest TRAIN point.
  - learned: RFFRidgeBiasModel, reused unmodified from
    fr3_bilateral_teleop/scripts/fit_residual_bias.py (the project's own
    established "small MLP or GP" choice -- a
    random-Fourier-feature ridge regressor, i.e. an approximate GP posterior
    mean, numpy-only). (length_scale, ridge_lambda) are grid-searched on VAL
    (same grid fit_residual_bias.py's own real-data hyperparameter search used),
    then refit on TRAIN only and scored on TEST.

Metric: RMSE in log-space, per axis, plus the mean across axes with enough
test data. Reported alongside improvement_ratio = constant_rmse / model_rmse
(the same convention fit_residual_bias.py already reports for its own
baseline-vs-fitted comparison).
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(PROJECT_ROOT)
EXTERNAL_SCRIPTS = os.path.join(
    PROJECT_ROOT, "hardware", "fr3_bilateral_teleop", "dataset_tools", "labeling")  # label extraction
RIG_SCRIPTS = os.path.join(
    PROJECT_ROOT, "hardware", "fr3_bilateral_teleop", "scripts")  # fit_residual_bias
sys.path.insert(0, RIG_SCRIPTS)
DATA_EXTRACTION_DIR = os.path.join(PROJECT_ROOT, "data_extraction")  # dataset_io, panda_fk, extraction drivers
sys.path.insert(0, EXTERNAL_SCRIPTS)
sys.path.insert(0, DATA_EXTRACTION_DIR)
sys.path.insert(0, SCRIPT_DIR)

import dataset_io as dio  # noqa: E402
from extract_impedance_labels import (  # noqa: E402
    nearest_sample_indices, parse_args as extraction_parse_args, DEFAULT_SIGMA_F, FORCE_AXIS_NAMES,
)
from fit_residual_bias import RFFRidgeBiasModel  # noqa: E402
import run_extraction_on_dataset as red  # noqa: E402

MANNERS = ["normally", "firmly"]
REFERENTS = ["red", "blue", "green", "black"]
K_MIN_LOG = np.log([50.0, 50.0, 50.0, 5.0, 5.0, 5.0])
K_MAX_LOG = np.log([1500.0, 1500.0, 1500.0, 100.0, 100.0, 100.0])

LENGTH_SCALE_GRID = [1, 3, 10, 30, 100, 300]
RIDGE_LAMBDA_GRID = [1e-3, 1e-2, 1e-1, 1.0]


def featurize(manner, referent):
    m = np.zeros(len(MANNERS))
    if manner in MANNERS:
        m[MANNERS.index(manner)] = 1.0
    r = np.zeros(len(REFERENTS))
    if referent in REFERENTS:
        r[REFERENTS.index(referent)] = 1.0
    return m, r


def build_axis_datasets(sessions, tool_offset, args, sigma_f):
    """Returns, per axis: X (n,20), y (n,) log K, for all masked timesteps
    pooled across the given sessions' episodes."""
    per_axis_X = {a: [] for a in FORCE_AXIS_NAMES}
    per_axis_y = {a: [] for a in FORCE_AXIS_NAMES}

    for session in sessions:
        frames = dio.load_frames(session).sort_values(["episode_index", "frame_index"])
        manifest = {r["episode_index"]: r for r in dio.load_session_manifest(session)}
        episodes_meta = dio.load_episodes_meta(session)
        task_lookup = dict(zip(episodes_meta["episode_index"],
                                episodes_meta["tasks"].apply(lambda t: t[0] if len(t) else None)))

        for ep_idx, g in frames.groupby("episode_index"):
            g = g.sort_values("frame_index")
            state = dio.stack_col(g, "observation.state")
            velocity = dio.stack_col(g, "observation.velocity")
            leader_pose = dio.stack_col(g, "observation.leader_pose")
            wrench = dio.stack_col(g, "observation.wrench.external_base")
            timestamp = g["timestamp"].to_numpy()

            demo = red.build_demo_dict(state, leader_pose, wrench, timestamp, tool_offset)
            result = red.extract_demo_30hz(demo, args, sigma_f)
            if result["status"] != "ok":
                continue

            task_text = task_lookup.get(ep_idx, "")
            manner = dio.parse_manner_from_task(task_text)
            referent = dio.parse_referent_from_task(task_text)
            if manner not in MANNERS:
                continue
            m_feat, r_feat = featurize(manner, referent)

            t = timestamp - timestamp[0]
            nearest_idx = nearest_sample_indices(t, result["output_times"])
            q_at_out = state[nearest_idx]
            qdot_at_out = velocity[nearest_idx]
            feat = np.concatenate([
                q_at_out, qdot_at_out,
                np.tile(m_feat, (len(nearest_idx), 1)),
                np.tile(r_feat, (len(nearest_idx), 1)),
            ], axis=1)

            log_k = np.log(np.clip(result["k"], 1e-6, None))
            for a, axis_name in enumerate(FORCE_AXIS_NAMES):
                m = result["mask"][:, a]
                if m.sum() == 0:
                    continue
                per_axis_X[axis_name].append(feat[m])
                per_axis_y[axis_name].append(log_k[m, a])

    out = {}
    for axis_name in FORCE_AXIS_NAMES:
        if per_axis_X[axis_name]:
            out[axis_name] = (
                np.concatenate(per_axis_X[axis_name], axis=0),
                np.concatenate(per_axis_y[axis_name], axis=0),
            )
        else:
            out[axis_name] = (np.zeros((0, 20)), np.zeros(0))
    return out


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def select_hyperparams(X_train, y_train, X_val, y_val):
    best = None
    for ls in LENGTH_SCALE_GRID:
        for rl in RIDGE_LAMBDA_GRID:
            model = RFFRidgeBiasModel(n_features=256, length_scale=ls, ridge_lambda=rl, seed=0)
            model.fit(X_train, y_train.reshape(-1, 1))
            pred = model.predict(X_val).ravel()
            score = rmse(pred, y_val)
            if best is None or score < best[0]:
                best = (score, ls, rl)
    return best[1], best[2], best[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=os.path.join(PROJECT_ROOT, "reports"))
    cli = parser.parse_args()
    os.makedirs(cli.out_dir, exist_ok=True)

    if not os.path.exists(red.TOOL_OFFSET_PATH):
        raise FileNotFoundError("tool_offset.npy missing -- run calibrate_tool_offset.py first")
    tool_offset = np.load(red.TOOL_OFFSET_PATH)
    args = extraction_parse_args([])
    sigma_f = np.array(DEFAULT_SIGMA_F)

    with open(os.path.join(dio.DATASET_ROOT, "franka_vla_multimodal", "splits.json")) as f:
        splits = json.load(f)
    split_sessions = {split: [s["session"] for s in stats["sessions"]] for split, stats in splits.items()}
    print("Splits used:", split_sessions)

    print("Building per-axis train datasets...")
    train_data = build_axis_datasets(split_sessions["train"], tool_offset, args, sigma_f)
    print("Building per-axis val datasets...")
    val_data = build_axis_datasets(split_sessions["val"], tool_offset, args, sigma_f)
    print("Building per-axis test datasets...")
    test_data = build_axis_datasets(split_sessions["test"], tool_offset, args, sigma_f)

    MIN_SAMPLES = 10
    report = {"splits": split_sessions, "per_axis": {}}
    gate2_axis_pass = {}

    for axis in FORCE_AXIS_NAMES:
        X_tr, y_tr = train_data[axis]
        X_va, y_va = val_data[axis]
        X_te, y_te = test_data[axis]
        n_tr, n_va, n_te = len(y_tr), len(y_va), len(y_te)
        print(f"\n=== axis {axis}: n_train={n_tr} n_val={n_va} n_test={n_te} ===")

        if n_tr < MIN_SAMPLES or n_te < MIN_SAMPLES:
            report["per_axis"][axis] = {"status": "insufficient_data", "n_train": n_tr, "n_val": n_va, "n_test": n_te}
            continue

        # constant predictor: train-set mean, ignores input entirely
        const_pred_value = y_tr.mean()
        const_rmse = rmse(np.full(n_te, const_pred_value), y_te)

        # nearest-neighbour: 1-NN in standardized feature space, label copied from train
        x_mean, x_std = X_tr.mean(axis=0), X_tr.std(axis=0)
        x_std[x_std < 1e-8] = 1.0
        nn = NearestNeighbors(n_neighbors=1).fit((X_tr - x_mean) / x_std)
        _, idx = nn.kneighbors((X_te - x_mean) / x_std)
        nn_pred = y_tr[idx.ravel()]
        nn_rmse = rmse(nn_pred, y_te)

        # learned predictor: RFF ridge, hyperparameters selected on val (if enough val data),
        # else fall back to fit_residual_bias.py's own defaults.
        if n_va >= MIN_SAMPLES:
            length_scale, ridge_lambda, val_rmse = select_hyperparams(X_tr, y_tr, X_va, y_va)
            print(f"  selected on val: length_scale={length_scale}, ridge_lambda={ridge_lambda}, "
                  f"val_rmse={val_rmse:.4f}")
        else:
            length_scale, ridge_lambda = 3.0, 1e-2
            print("  insufficient val data -- using fit_residual_bias.py defaults")

        model = RFFRidgeBiasModel(n_features=256, length_scale=length_scale, ridge_lambda=ridge_lambda, seed=0)
        model.fit(X_tr, y_tr.reshape(-1, 1))
        learned_pred = model.predict(X_te).ravel()
        learned_rmse = rmse(learned_pred, y_te)

        beats_constant = learned_rmse < const_rmse
        gate2_axis_pass[axis] = beats_constant

        print(f"  constant_rmse={const_rmse:.4f}  nn_rmse={nn_rmse:.4f}  learned_rmse={learned_rmse:.4f}  "
              f"learned_vs_constant_improvement={const_rmse / max(learned_rmse, 1e-9):.3f}x  "
              f"{'BEATS' if beats_constant else 'DOES NOT BEAT'} constant")

        report["per_axis"][axis] = {
            "status": "ok", "n_train": n_tr, "n_val": n_va, "n_test": n_te,
            "constant_rmse_log_k": const_rmse,
            "nearest_neighbour_rmse_log_k": nn_rmse,
            "learned_rmse_log_k": learned_rmse,
            "learned_vs_constant_improvement_ratio": const_rmse / max(learned_rmse, 1e-9),
            "learned_vs_nn_improvement_ratio": nn_rmse / max(learned_rmse, 1e-9),
            "learned_beats_constant": beats_constant,
            "hyperparameters": {"length_scale": length_scale, "ridge_lambda": ridge_lambda},
        }

    n_axes_evaluated = len(gate2_axis_pass)
    n_axes_passing = sum(gate2_axis_pass.values())
    all_axes_pass = n_axes_evaluated > 0 and n_axes_passing == n_axes_evaluated
    majority_axes_pass = n_axes_evaluated > 0 and n_axes_passing > n_axes_evaluated / 2

    # Primary learnability metric: the check is one pass/fail ("learned
    # prediction must beat the constant predictor"), not six independent ones -- so the
    # headline comparison is the MEAN log-K RMSE across all evaluated axes, learned vs
    # constant. Per-axis pass/fail is still reported in full below rather than folded away,
    # matching this project's established practice of surfacing per-axis heterogeneity
    # (e.g. the fx/ty/tz residual-bias gap) instead of letting an aggregate
    # pass hide it.
    ok_axes = [a for a in FORCE_AXIS_NAMES if report["per_axis"].get(a, {}).get("status") == "ok"]
    mean_const_rmse = float(np.mean([report["per_axis"][a]["constant_rmse_log_k"] for a in ok_axes]))
    mean_learned_rmse = float(np.mean([report["per_axis"][a]["learned_rmse_log_k"] for a in ok_axes]))
    aggregate_pass = mean_learned_rmse < mean_const_rmse

    report["gate2"] = {
        "per_axis_pass": gate2_axis_pass,
        "n_axes_evaluated": n_axes_evaluated,
        "n_axes_passing": n_axes_passing,
        "all_axes_pass": all_axes_pass,
        "majority_axes_pass": majority_axes_pass,
        "mean_constant_rmse_log_k_across_axes": mean_const_rmse,
        "mean_learned_rmse_log_k_across_axes": mean_learned_rmse,
        "aggregate_improvement_ratio": mean_const_rmse / max(mean_learned_rmse, 1e-9),
        "aggregate_pass": aggregate_pass,
    }

    print("\n" + "=" * 70)
    print(f"LEARNABILITY: learned predictor vs constant predictor, held-out session ({split_sessions['test']})")
    print(f"  mean log-K RMSE across {len(ok_axes)} axes: constant={mean_const_rmse:.4f}  "
          f"learned={mean_learned_rmse:.4f}  (improvement {mean_const_rmse / max(mean_learned_rmse, 1e-9):.3f}x)")
    print(f"  per-axis: {n_axes_passing}/{n_axes_evaluated} axes have learned beating constant")
    print(f"  PRIMARY VERDICT (aggregate): {'PASS' if aggregate_pass else 'FAIL'}")
    print("=" * 70)

    report["gate2"]["verdict"] = "PASS" if aggregate_pass else "FAIL"

    with open(os.path.join(cli.out_dir, "offline_stiffness_benchmark_m8.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nWrote {os.path.join(cli.out_dir, 'offline_stiffness_benchmark_m8.json')}")


if __name__ == "__main__":
    main()
