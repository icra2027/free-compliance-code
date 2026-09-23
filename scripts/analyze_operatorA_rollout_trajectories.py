#!/usr/bin/env python3
"""Offline rollout analysis for the b0_seed0_operatorA_two_color checkpoints
(scripts/slurm_train_hpc_operator_a_two_color.sbatch's B0 run: position-only
output, no force input, trained on all 4 operator-A data_two_color sessions
-- demo_blue, demo_blue_firm, demo_red_firm, demo_red_left, 24 episodes each,
no held-out val/test, matching data_recorder/operator_a_lowest_df.csv's
own OPERATOR_A_DATASETS list exactly).

There is no simulator and no real robot in this environment (scripts/README.md's
"real-robot evaluation" section, evaluate_gate3.py's own docstring: rollouts need
the physical rig / rclpy, "not runnable or verified in this environment"), so a
true closed-loop rollout (predicted action -> robot moves -> new camera frame)
isn't possible here. Instead this script does a receding-horizon, teacher-forced
stitch, the same idea scripts/diagnose_language_grounding.py uses for a single
frame, just walked across a whole episode: at every chunk_size=32-frame anchor
in a real recorded episode, feed the checkpoint the REAL (state, scene_rgb,
wrist_rgb) observed at that frame (exactly what deploy_smolvla.py would have
sent had the robot really been there) and take its predicted 32-step x_eq
chunk as the model's plan for that stretch. Concatenating non-overlapping
chunks across all anchors gives one continuous predicted position trajectory
spanning the whole episode. This is *not* the same as what the real robot
would have done if it had actually been driven open-loop by this policy from
frame 0 (errors don't compound the way they would in closed loop, since every
anchor resets to the real recorded state) -- it answers a narrower, well-posed
question instead: "given what the robot actually saw at each point in a real
wipe, does this checkpoint's own plan dip down toward the true contact point
and then rise back up, the way the demonstration did?"

x_eq's first 3 dims are `observation.leader_pose[:3]` in base frame
(src/compliance_vla/policy/labels.py's module docstring), and data_recorder's
operator_a_lowest_df.csv's x/y/z come from the parquet's `action` field --
empirically nearly identical to leader_pose (checked directly: same-index
diff ~1-3mm typically, a few cm worst case, versus tens-of-cm position
scales), so the predicted x_eq and the CSV's lowest point are directly
comparable in the same frame with no transform.

Usage:
    python3 scripts/analyze_operatorA_rollout_trajectories.py
    python3 scripts/analyze_operatorA_rollout_trajectories.py --steps 30000 --episodes-per-session 4
    python3 scripts/analyze_operatorA_rollout_trajectories.py --steps 5000 10000 30000 --n-plot-episodes 4

Writes reports/operatorA_rollout_per_episode.csv (one row per (checkpoint
step, session, episode)), reports/operatorA_rollout_summary.json (aggregated
by step and by session), reports/figure_operatorA_rollout_examples.png
(z(t) + top-down xy for a few example episodes at the final requested step),
and, if more than one --steps value is given, reports/figure_operatorA_rollout_vs_step.png
(does the dip-to-lowest-point distance improve over training).
"""

import argparse
import glob
import json
import os
import sys
import time

import matplotlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SCRIPT_DIR)

import dataset_io as dio  # noqa: E402
import panda_fk as fk  # noqa: E402
from serve_policy import load_policy  # noqa: E402
from diagnose_language_grounding import _load_manifest_ordered, _decode_array3d  # noqa: E402
from compliance_vla.policy.labels import load_tool_offset  # noqa: E402

OPERATOR_A_DATASETS = ["demo_blue", "demo_blue_firm", "demo_red_firm", "demo_red_left"]
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "b0_seed0_operatorA_two_color")
LOWEST_DF_CSV = os.path.join(REPO_ROOT, "data_recorder", "operator_a_lowest_df.csv")
OUT_DIR = os.path.join(PROJECT_ROOT, "reports")

ALL_STEPS = [5000, 10000, 15000, 20000, 25000, 30000]


# ---------------------------------------------------------------------------
# Loading one episode's real observations at every replan anchor
# ---------------------------------------------------------------------------

def _episode_meta(session_path, ds_idx):
    meta_files = sorted(glob.glob(os.path.join(session_path, "meta", "episodes", "chunk-*", "file-*.parquet")))
    meta_cols = ["episode_index", "data/chunk_index", "data/file_index", "tasks"]
    meta = pd.concat(
        pq.read_table(f, columns=meta_cols).to_pandas() for f in meta_files
    )
    ep_meta = meta[meta["episode_index"] == ds_idx]
    if ep_meta.empty:
        raise ValueError(f"no episode_index={ds_idx} in {session_path}'s meta/episodes")
    return ep_meta.iloc[0]


def _episode_data_file(session_path, ep_meta):
    chunk_idx = int(ep_meta["data/chunk_index"])
    file_idx = int(ep_meta["data/file_index"])
    return os.path.join(session_path, "data", f"chunk-{chunk_idx:03d}", f"file-{file_idx:03d}.parquet")


def load_episode(session, ds_idx, tool_offset, chunk_size, dataset_root=None):
    """Real per-frame state/action for the whole episode, plus decoded images only at
    the replan anchors (every chunk_size frames) -- images dominate memory/decode time
    (analyze_data_two_color.py's own docstring) and every other anchor is skipped anyway."""
    dataset_root = dataset_root or dio.TWO_COLOR_DATASET_ROOT
    session_path = os.path.join(dataset_root, session)
    ep_meta = _episode_meta(session_path, ds_idx)
    data_file = _episode_data_file(session_path, ep_meta)
    task_text = ep_meta["tasks"][0]

    cols = dio.NON_IMAGE_COLUMNS + ["observation.images.scene_rgb", "observation.images.wrist_rgb"]
    df = pq.read_table(data_file, columns=cols, filters=[("episode_index", "=", ds_idx)]).to_pandas()
    ep = df.sort_values("frame_index").reset_index(drop=True)
    n = len(ep)

    manifest_entries = _load_manifest_ordered(session_path)
    manifest_entry = manifest_entries[ds_idx] if ds_idx < len(manifest_entries) else {}

    q_all = np.stack(ep["observation.state"].to_numpy()).astype(np.float64)
    qdot_all = np.stack(ep["observation.velocity"].to_numpy()).astype(np.float32)
    real_xyz = np.stack(ep["action"].to_numpy()).astype(np.float32)[:, :3]

    pos_all, rot_all = fk.fk_batch(q_all)
    tip_pos_all = pos_all + np.einsum("nij,j->ni", rot_all, tool_offset)
    tip_rotvec_all = np.stack([fk.rotvec_from_matrix(r) for r in rot_all])
    x_f_all = np.concatenate([tip_pos_all, tip_rotvec_all], axis=1).astype(np.float32)
    state_all = np.concatenate([q_all.astype(np.float32), qdot_all, x_f_all], axis=1)

    anchors = list(range(0, n, chunk_size))
    scene_col = ep["observation.images.scene_rgb"]
    wrist_col = ep["observation.images.wrist_rgb"]
    anchor_obs = {}
    for a in anchors:
        anchor_obs[a] = {
            "state": state_all[a],
            "scene_rgb": _decode_array3d(scene_col.iloc[a]),
            "wrist_rgb": _decode_array3d(wrist_col.iloc[a]),
        }

    return {
        "session": session, "episode_index": ds_idx, "n_frames": n,
        "task": task_text,
        "colour": manifest_entry.get("colour"), "manner": manifest_entry.get("manner"),
        "real_xyz": real_xyz,
        "anchors": anchors,
        "anchor_obs": anchor_obs,
    }


def _to_chw_float(img_hwc_uint8, device):
    t = torch.from_numpy(np.asarray(img_hwc_uint8, dtype=np.uint8))
    return t.permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)


@torch.no_grad()
def predict_stitched_trajectory(policy, tokenizer, device, episode, chunk_size):
    """Runs one predict_action_chunk call per anchor (real, teacher-forced
    observation) and concatenates the non-overlapping chunks into one
    (n_frames, 3) predicted position trajectory."""
    n = episode["n_frames"]
    pred_xyz = np.full((n, 3), np.nan, dtype=np.float32)
    tok = tokenizer(
        [episode["task"]], padding=policy.config.pad_language_to, truncation=True,
        max_length=policy.config.tokenizer_max_length, return_tensors="pt",
    )
    for a in episode["anchors"]:
        obs = episode["anchor_obs"][a]
        batch = {
            "observation.state": torch.as_tensor(obs["state"], dtype=torch.float32, device=device).unsqueeze(0),
            "observation.images.scene_rgb": _to_chw_float(obs["scene_rgb"], device),
            "observation.images.wrist_rgb": _to_chw_float(obs["wrist_rgb"], device),
            "observation.language.tokens": tok["input_ids"].to(device),
            "observation.language.attention_mask": tok["attention_mask"].bool().to(device),
        }
        action_chunk = policy.predict_action_chunk(batch)[0].cpu().numpy()  # (chunk_size, 7)
        take = min(chunk_size, n - a)
        pred_xyz[a:a + take] = action_chunk[:take, :3]
    return pred_xyz


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def analyze_trajectory(pred_xyz, real_xyz, csv_row):
    valid = ~np.isnan(pred_xyz[:, 2])
    idx = np.where(valid)[0]
    pred_lowest_i = idx[np.argmin(pred_xyz[idx, 2])]
    pred_lowest_xyz = pred_xyz[pred_lowest_i]

    real_lowest_i = int(np.argmin(real_xyz[:, 2]))
    real_lowest_xyz = real_xyz[real_lowest_i]

    csv_xyz = np.array([csv_row["x"], csv_row["y"], csv_row["z"]], dtype=np.float32) if csv_row is not None else None

    z0 = pred_xyz[idx[0], 2]
    z_end = pred_xyz[idx[-1], 2]
    z_low = pred_lowest_xyz[2]
    down_then_up = bool(z0 > z_low and z_end > z_low)
    frac_pos = pred_lowest_i / max(1, len(pred_xyz) - 1)

    out = {
        "pred_lowest_frame": int(pred_lowest_i),
        "pred_lowest_frame_frac": float(frac_pos),
        "pred_lowest_x": float(pred_lowest_xyz[0]),
        "pred_lowest_y": float(pred_lowest_xyz[1]),
        "pred_lowest_z": float(pred_lowest_xyz[2]),
        "real_lowest_frame": real_lowest_i,
        "real_lowest_x": float(real_lowest_xyz[0]),
        "real_lowest_y": float(real_lowest_xyz[1]),
        "real_lowest_z": float(real_lowest_xyz[2]),
        "down_then_up": down_then_up,
        "z_drop_m": float(z0 - z_low),
        "z_rise_after_m": float(z_end - z_low),
    }
    if csv_xyz is not None:
        out["csv_x"], out["csv_y"], out["csv_z"] = map(float, csv_xyz)
        out["dist_pred_to_csv_m"] = float(np.linalg.norm(pred_lowest_xyz - csv_xyz))
        out["dist_pred_to_csv_z_m"] = float(abs(z_low - csv_xyz[2]))
        out["dist_real_to_csv_m"] = float(np.linalg.norm(real_lowest_xyz - csv_xyz))
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_examples(results, out_path):
    examples = results if len(results) <= 8 else results[:8]
    n = len(examples)
    fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 7.5), squeeze=False)
    for j, r in enumerate(examples):
        ep = r["episode"]
        n_frames = ep["n_frames"]
        t = np.arange(n_frames)
        ax = axes[0][j]
        ax.plot(t, ep["real_xyz"][:, 2], color="#2a78d6", lw=2, label="real (recorded) z")
        ax.plot(t, r["pred_xyz"][:, 2], color="#eb6834", lw=2, ls="--", label="predicted (stitched) z")
        ax.axhline(r["metrics"]["real_lowest_z"], color="#2a78d6", lw=0.8, alpha=0.4)
        ax.scatter([r["metrics"]["pred_lowest_frame"]], [r["metrics"]["pred_lowest_z"]],
                   color="#eb6834", zorder=5, s=40, marker="v", label="pred lowest")
        if "csv_z" in r["metrics"]:
            ax.scatter([ep.get("csv_frame_index", r["metrics"]["real_lowest_frame"])],
                       [r["metrics"]["csv_z"]], color="#0b0b0b", zorder=5, s=50, marker="*",
                       label="CSV lowest point")
        ax.set_title(f"{ep['session']} ep{ep['episode_index']} ({ep['colour']}, {ep['manner']})", fontsize=9.5)
        ax.set_xlabel("frame index")
        if j == 0:
            ax.set_ylabel("z (m)")
            ax.legend(fontsize=7, loc="best")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax2 = axes[1][j]
        ax2.plot(ep["real_xyz"][:, 0], ep["real_xyz"][:, 1], color="#2a78d6", lw=1.5, alpha=0.8)
        ax2.plot(r["pred_xyz"][:, 0], r["pred_xyz"][:, 1], color="#eb6834", lw=1.5, ls="--", alpha=0.9)
        ax2.scatter([r["metrics"]["pred_lowest_x"]], [r["metrics"]["pred_lowest_y"]],
                    color="#eb6834", zorder=5, s=40, marker="v")
        if "csv_x" in r["metrics"]:
            ax2.scatter([r["metrics"]["csv_x"]], [r["metrics"]["csv_y"]],
                        color="#0b0b0b", zorder=5, s=50, marker="*")
        ax2.set_xlabel("x (m)")
        if j == 0:
            ax2.set_ylabel("y (m)")
        ax2.set_aspect("equal", adjustable="datalim")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

    fig.suptitle("b0_seed0_operatorA_two_color -- real vs. stitched-predicted trajectory\n"
                  "(top: height z(t), bottom: top-down x/y; * = CSV lowest point, v = predicted lowest point)",
                  fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=170, facecolor="white")
    plt.close(fig)


def plot_vs_step(per_episode_df, out_path):
    agg = per_episode_df.groupby("step").agg(
        mean_dist=("dist_pred_to_csv_m", "mean"),
        median_dist=("dist_pred_to_csv_m", "median"),
        frac_down_then_up=("down_then_up", "mean"),
    ).reset_index().sort_values("step")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    ax1.plot(agg["step"], agg["mean_dist"] * 100, "o-", color="#2a78d6", label="mean")
    ax1.plot(agg["step"], agg["median_dist"] * 100, "o--", color="#8a8a86", label="median")
    ax1.set_xlabel("training step")
    ax1.set_ylabel("dist: predicted lowest point -> CSV lowest point (cm)")
    ax1.legend(fontsize=9)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2.plot(agg["step"], agg["frac_down_then_up"] * 100, "o-", color="#eb6834")
    ax2.set_xlabel("training step")
    ax2.set_ylabel("% episodes with down-then-up z profile")
    ax2.set_ylim(0, 105)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle("b0_seed0_operatorA_two_color -- rollout quality vs. training step", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path, dpi=170, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, nargs="+", default=ALL_STEPS,
                   help="which checkpoint step(s) under checkpoints/b0_seed0_operatorA_two_color/ to run")
    p.add_argument("--sessions", nargs="+", default=OPERATOR_A_DATASETS, choices=OPERATOR_A_DATASETS)
    p.add_argument("--episodes-per-session", type=int, default=None,
                   help="cap episodes per session (default: all 24)")
    p.add_argument("--n-plot-episodes", type=int, default=4,
                   help="how many example episodes to plot in detail, for the final --steps value")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", default=OUT_DIR)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tool_offset = load_tool_offset()

    lowest_df = pd.read_csv(LOWEST_DF_CSV)

    print(f"Loading {len(args.sessions)} session(s)' episodes (real observations, cached across checkpoints)...")
    t0 = time.time()
    episodes = []
    for session in args.sessions:
        session_path = os.path.join(dio.TWO_COLOR_DATASET_ROOT, session)
        manifest_entries = _load_manifest_ordered(session_path)
        n_eps = len(manifest_entries)
        if args.episodes_per_session:
            n_eps = min(n_eps, args.episodes_per_session)
        for ds_idx in range(n_eps):
            # chunk_size read per-checkpoint below (all b0 checkpoints share chunk_size=32,
            # but read it off the actual config rather than hardcoding).
            episodes.append((session, ds_idx))
    print(f"  {len(episodes)} episodes queued across {args.sessions} ({time.time()-t0:.1f}s)")

    all_rows = []
    example_results = None
    last_step = None

    for step in args.steps:
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"b0_seed0_step{step}.pt")
        if not os.path.exists(ckpt_path):
            print(f"** skipping step {step}: {ckpt_path} not found **")
            continue
        print(f"\n=== loading checkpoint step {step} ===")
        policy, uses_force = load_policy(ckpt_path, args.device)
        assert not uses_force, "this script is for the b0 (position-only) checkpoints"
        tokenizer = policy.model.vlm_with_expert.processor.tokenizer
        chunk_size = policy.config.chunk_size

        step_results = []
        t_step = time.time()
        for i, (session, ds_idx) in enumerate(episodes):
            ep = load_episode(session, ds_idx, tool_offset, chunk_size)
            pred_xyz = predict_stitched_trajectory(policy, tokenizer, args.device, ep, chunk_size)

            csv_match = lowest_df[(lowest_df["dataset"] == session) & (lowest_df["episode_index"] == ds_idx)]
            csv_row = csv_match.iloc[0] if len(csv_match) else None
            if csv_row is not None:
                ep["csv_frame_index"] = int(csv_row["frame_index"])

            metrics = analyze_trajectory(pred_xyz, ep["real_xyz"], csv_row)
            row = {"step": step, "session": session, "episode_index": ds_idx,
                   "colour": ep["colour"], "manner": ep["manner"], "n_frames": ep["n_frames"], **metrics}
            all_rows.append(row)
            step_results.append({"episode": ep, "pred_xyz": pred_xyz, "metrics": metrics})

            if (i + 1) % 20 == 0 or i == len(episodes) - 1:
                print(f"  [{i+1}/{len(episodes)}] {session} ep{ds_idx}: "
                      f"dist_to_csv={metrics.get('dist_pred_to_csv_m', float('nan'))*100:.1f}cm "
                      f"down_then_up={metrics['down_then_up']}  ({time.time()-t_step:.0f}s elapsed)")

        del policy
        torch.cuda.empty_cache()
        last_step = step
        example_results = step_results

    per_episode_df = pd.DataFrame(all_rows)
    per_episode_csv = os.path.join(args.out_dir, "operatorA_rollout_per_episode.csv")
    per_episode_df.to_csv(per_episode_csv, index=False)
    print(f"\nwrote {per_episode_csv} ({len(per_episode_df)} rows)")

    summary = {
        "by_step": per_episode_df.groupby("step").agg(
            n_episodes=("episode_index", "count"),
            mean_dist_to_csv_m=("dist_pred_to_csv_m", "mean"),
            median_dist_to_csv_m=("dist_pred_to_csv_m", "median"),
            frac_down_then_up=("down_then_up", "mean"),
            mean_z_drop_m=("z_drop_m", "mean"),
            mean_z_rise_after_m=("z_rise_after_m", "mean"),
        ).reset_index().to_dict(orient="records"),
        "by_session_final_step": per_episode_df[per_episode_df["step"] == last_step].groupby("session").agg(
            n_episodes=("episode_index", "count"),
            mean_dist_to_csv_m=("dist_pred_to_csv_m", "mean"),
            frac_down_then_up=("down_then_up", "mean"),
        ).reset_index().to_dict(orient="records"),
    }
    summary_path = os.path.join(args.out_dir, "operatorA_rollout_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"wrote {summary_path}")

    print("\n=== summary by checkpoint step ===")
    print(pd.DataFrame(summary["by_step"]).to_string(index=False))
    print(f"\n=== summary by session (final step={last_step}) ===")
    print(pd.DataFrame(summary["by_session_final_step"]).to_string(index=False))

    if example_results:
        fig_path = os.path.join(args.out_dir, "figure_operatorA_rollout_examples.png")
        # spread examples across sessions rather than taking the first N (all from one session)
        spread = []
        by_session = {}
        for r in example_results:
            by_session.setdefault(r["episode"]["session"], []).append(r)
        i = 0
        while len(spread) < min(args.n_plot_episodes, len(example_results)):
            for sess in args.sessions:
                lst = by_session.get(sess, [])
                if i < len(lst):
                    spread.append(lst[i])
                if len(spread) >= args.n_plot_episodes:
                    break
            i += 1
        plot_examples(spread, fig_path)
        print(f"wrote {fig_path}")

    if len([s for s in args.steps if os.path.exists(os.path.join(CHECKPOINT_DIR, f"b0_seed0_step{s}.pt"))]) > 1:
        fig_path = os.path.join(args.out_dir, "figure_operatorA_rollout_vs_step.png")
        plot_vs_step(per_episode_df, fig_path)
        print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
