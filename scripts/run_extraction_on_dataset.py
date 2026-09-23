"""Run the real, tested §4.1 extraction pipeline (fr3_bilateral_teleop/dataset_tools/
labeling/extract_impedance_labels.py) over the built LeRobot dataset (dataset/demo{1,3,4}).

That pipeline was written against record_demo.py's CSV output, which logs
Cartesian x_l/x_f directly. The LeRobot dataset instead logs
`observation.state` as 7-DoF joint position (no Cartesian follower pose) --
see panda_fk.py / calibrate_tool_offset.py for why and how that gap is closed.
This script is the adapter: for each episode, build the same `demo` dict shape
`extract_demo()` expects (Cartesian leader/follower pose as quaternions,
6D wrench), computing x_f via FK(observation.state) + the calibrated tool
offset, x_l directly from observation.leader_pose, and calls the upstream
`extract_demo` unmodified -- so the regression, mask, and contact-frame-fit
logic are exactly the ones already self-tested in that repo, not a
reimplementation.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)          # compliance-vla/
REPO_ROOT = os.path.dirname(PROJECT_ROOT)            # datasets/ (holds dataset/ and external/)
EXTERNAL_SCRIPTS = os.path.join(
    PROJECT_ROOT, "hardware", "fr3_bilateral_teleop", "dataset_tools", "labeling")  # label extraction
sys.path.insert(0, EXTERNAL_SCRIPTS)
sys.path.insert(0, SCRIPT_DIR)

import dataset_io as dio  # noqa: E402
import panda_fk as fk  # noqa: E402
from extract_impedance_labels import (  # noqa: E402
    fit_contact_frame, compute_pose_error, numerically_differentiate,
    rotate_force_to_contact_frame, extract_axis, compute_mask, nearest_sample_indices,
    parse_args as extraction_parse_args, DEFAULT_SIGMA_E, DEFAULT_SIGMA_F, FORCE_AXIS_NAMES,
    K_MIN, K_MAX,
)

TOOL_OFFSET_PATH = os.path.join(SCRIPT_DIR, "tool_offset.npy")

# extract_demo()'s own extract_axis() call hardcodes min_window_samples=20, sized for
# record_demo.py's ~1kHz raw CSVs (a 300ms window there holds ~300 samples). This dataset's
# frames are already downsampled to 30Hz at build time (matching the policy's 30Hz action-chunk
# rate, per data_recorder), so the same 300ms window only ever holds 9 samples --
# under the hardcoded 20, every window would be skipped and mask coverage would silently read
# 0% everywhere (confirmed: this is exactly what happened before this constant was introduced).
# Verified no dropped frames within any episode across all 3 sessions (frame_index is gapless
# 0..N-1 per episode), so requiring a full 9-sample window is safe -- MIN_WINDOW_SAMPLES_30HZ=8
# tolerates at most one missing sample rather than demanding bit-exact completeness.
MIN_WINDOW_SAMPLES_30HZ = 8

# fit_contact_frame()'s own default min_contact_samples=50 has the same 1kHz-sizing problem:
# at 30Hz, 50 raw samples is 1.67s of sustained >8N contact, which most of these ~5-7s wiping
# episodes never reach even when they contain real, firm contact (checked directly: every one
# of the 92 episodes across all 3 sessions has at least 21 firm-contact (>8N) samples, but the
# per-episode minimum ranges 21-30 across sessions, well under 50). 20 is chosen to sit safely
# below that observed per-episode minimum while still being well above the ~6 points a 2-DOF
# plane orientation strictly needs, for a statistically stable SVD fit.
MIN_CONTACT_SAMPLES_30HZ = 20


def extract_demo_30hz(demo, args, sigma_f):
    """Re-orchestrates extract_impedance_labels.py's extract_demo(), reusing every one of its
    building blocks unmodified, with the one change described above (min_window_samples)."""
    t = demo["t"] - demo["t"][0]
    follower_pos = np.column_stack([demo["fx"], demo["fy"], demo["fz"]])
    wrench_force = np.column_stack([demo["wfx"], demo["wfy"], demo["wfz"]])
    wrench_torque = np.column_stack([demo["wtx"], demo["wty"], demo["wtz"]])

    frame_fit = fit_contact_frame(
        follower_pos, wrench_force, args.frame_fit_force_threshold,
        min_contact_samples=MIN_CONTACT_SAMPLES_30HZ,
        max_planarity_ratio=args.max_planarity_ratio)
    if not frame_fit["ok"]:
        return {"status": "failed", "reason": frame_fit["reason"], "frame_fit": frame_fit}

    sigma_e = DEFAULT_SIGMA_E
    R_contact = frame_fit["R"]
    e = compute_pose_error(demo, R_contact)
    edot = numerically_differentiate(t, e)
    f = rotate_force_to_contact_frame(wrench_force, wrench_torque, R_contact)
    force_mag = np.linalg.norm(wrench_force, axis=1)

    t_end = t[-1]
    output_times = np.arange(args.window_sec, t_end, 1.0 / args.output_rate_hz)
    if len(output_times) == 0:
        return {"status": "failed", "reason": "demo too short for even one output window"}

    nearest_idx = nearest_sample_indices(t, output_times)

    per_axis = []
    for axis in range(6):
        k_min, k_max = (K_MIN[axis], K_MAX[axis])
        per_axis.append(extract_axis(
            t, e[:, axis], edot[:, axis], f[:, axis], output_times, args.window_sec,
            k_min, k_max, args.lam, args.d_max, min_window_samples=MIN_WINDOW_SAMPLES_30HZ))

    condition_number_per_axis = np.stack([pa["condition_number"] for pa in per_axis], axis=1)

    mask = np.zeros((len(output_times), 6), dtype=bool)
    contact_indicator = None
    for axis in range(6):
        axis_mask, axis_contact = compute_mask(
            e[:, axis:axis + 1], f[:, axis:axis + 1], force_mag,
            condition_number_per_axis[:, axis], nearest_idx,
            sigma_e[axis:axis + 1], sigma_f[axis:axis + 1], args.kappa_max,
            args.contact_force_threshold, args.contact_fraction_required,
            t, output_times, args.window_sec,
        )
        mask[:, axis] = axis_mask[:, 0]
        contact_indicator = axis_contact

    mask_coverage = mask.mean(axis=0)
    n_contact_timesteps = int(np.sum(contact_indicator))
    mask_coverage_within_contact = (
        mask[contact_indicator].mean(axis=0) if n_contact_timesteps > 0 else np.zeros(6))

    return {
        "status": "ok",
        "frame_fit": {k: v for k, v in frame_fit.items() if k != "R"},
        "mask_coverage": mask_coverage,
        "mask_coverage_within_contact": mask_coverage_within_contact,
        "n_contact_timesteps": n_contact_timesteps,
        "n_output_timesteps": len(output_times),
        "k": np.stack([pa["k"] for pa in per_axis], axis=1),
        "mask": mask,
        "output_times": output_times,
    }


def rotvec_batch_to_quat(rv):
    """rv: (N,3) rotation vectors -> (N,4) quaternions [x,y,z,w], matching
    extract_impedance_labels.py's quat_to_rotvec/quat_multiply convention."""
    theta = np.linalg.norm(rv, axis=1)
    quat = np.zeros((rv.shape[0], 4))
    quat[:, 3] = 1.0
    nonzero = theta > 1e-12
    axis = np.zeros_like(rv)
    axis[nonzero] = rv[nonzero] / theta[nonzero, None]
    quat[nonzero, :3] = axis[nonzero] * np.sin(theta[nonzero] / 2.0)[:, None]
    quat[nonzero, 3] = np.cos(theta[nonzero] / 2.0)
    return quat


def rotmat_batch_to_quat(R):
    """R: (N,3,3) -> (N,4) quaternions [x,y,z,w], via the rotation-vector route
    (fk.rotvec_from_matrix is already validated against the FK self-test)."""
    rv = np.stack([fk.rotvec_from_matrix(R[i]) for i in range(R.shape[0])], axis=0)
    return rotvec_batch_to_quat(rv)


def build_demo_dict(state, leader_pose, wrench, timestamp, tool_offset):
    follower_pos, follower_rot = fk.fk_batch(state)
    follower_pos = follower_pos + np.einsum("nij,j->ni", follower_rot, tool_offset)
    follower_quat = rotmat_batch_to_quat(follower_rot)
    leader_quat = rotvec_batch_to_quat(leader_pose[:, 3:6])

    return {
        "t": timestamp,
        "lx": leader_pose[:, 0], "ly": leader_pose[:, 1], "lz": leader_pose[:, 2],
        "lqx": leader_quat[:, 0], "lqy": leader_quat[:, 1], "lqz": leader_quat[:, 2],
        "lqw": leader_quat[:, 3],
        "fx": follower_pos[:, 0], "fy": follower_pos[:, 1], "fz": follower_pos[:, 2],
        "fqx": follower_quat[:, 0], "fqy": follower_quat[:, 1], "fqz": follower_quat[:, 2],
        "fqw": follower_quat[:, 3],
        "wfx": wrench[:, 0], "wfy": wrench[:, 1], "wfz": wrench[:, 2],
        "wtx": wrench[:, 3], "wty": wrench[:, 4], "wtz": wrench[:, 5],
    }


def run_session(session, tool_offset, args, sigma_f):
    frames = dio.load_frames(session).sort_values(["episode_index", "frame_index"])
    manifest = {r["episode_index"]: r for r in dio.load_session_manifest(session)}
    episodes_meta = dio.load_episodes_meta(session)
    task_lookup = dict(zip(episodes_meta["episode_index"],
                            episodes_meta["tasks"].apply(lambda t: t[0] if len(t) else None)))

    results = []
    for ep_idx, g in frames.groupby("episode_index"):
        g = g.sort_values("frame_index")
        state = dio.stack_col(g, "observation.state")
        leader_pose = dio.stack_col(g, "observation.leader_pose")
        wrench = dio.stack_col(g, "observation.wrench.external_base")
        timestamp = g["timestamp"].to_numpy()

        demo = build_demo_dict(state, leader_pose, wrench, timestamp, tool_offset)
        result = extract_demo_30hz(demo, args, sigma_f)

        task_text = task_lookup.get(ep_idx, "")
        meta = {
            "session": session, "episode_index": int(ep_idx), "task": task_text,
            "manner": dio.parse_manner_from_task(task_text),
            "referent": dio.parse_referent_from_task(task_text),
            "operator_id": manifest.get(ep_idx, {}).get("operator_id"),
        }
        if result["status"] != "ok":
            results.append({**meta, "status": "failed", "reason": result["reason"]})
            continue

        results.append({
            **meta,
            "status": "ok",
            "planarity_ratio": result["frame_fit"]["planarity_ratio"],
            "n_contact_timesteps": result["n_contact_timesteps"],
            "n_output_timesteps": result["n_output_timesteps"],
            "mask_coverage_within_contact": dict(
                zip(FORCE_AXIS_NAMES, result["mask_coverage_within_contact"].tolist())),
        })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", nargs="+", default=dio.SESSIONS)
    parser.add_argument("--out-dir", default=os.path.join(PROJECT_ROOT, "reports"))
    cli = parser.parse_args()
    os.makedirs(cli.out_dir, exist_ok=True)

    if not os.path.exists(TOOL_OFFSET_PATH):
        raise FileNotFoundError(f"{TOOL_OFFSET_PATH} missing -- run calibrate_tool_offset.py first")
    tool_offset = np.load(TOOL_OFFSET_PATH)

    args = extraction_parse_args([])  # upstream script's own defaults
    sigma_f = np.array(DEFAULT_SIGMA_F)

    all_results = []
    for session in cli.sessions:
        print(f"=== extracting {session} (via fr3_bilateral_teleop/extract_impedance_labels.py) ===")
        res = run_session(session, tool_offset, args, sigma_f)
        n_ok = sum(1 for r in res if r["status"] == "ok")
        n_failed = len(res) - n_ok
        print(f"[{session}] {n_ok} episodes ok, {n_failed} failed contact-frame fit")
        all_results.extend(res)

    df = pd.DataFrame(all_results)
    df.to_json(os.path.join(cli.out_dir, "extraction_per_episode.json"), orient="records", indent=2)

    failed = df[df["status"] == "failed"]
    if len(failed):
        print(f"\n{len(failed)} episodes failed contact-frame fitting:")
        for _, r in failed.iterrows():
            print(f"  {r['session']}#{r['episode_index']} ({r['task']}): {r['reason']}")

    ok = df[df["status"] == "ok"].copy()

    # Pool per-axis, per-task coverage from exact per-episode (numerator, denominator) pairs,
    # not a naive mean-of-per-episode-ratios (which would weight a 20-frame-contact episode
    # the same as a 400-frame one).
    def pooled_coverage(sub):
        rows = []
        for axis in FORCE_AXIS_NAMES:
            n_contact = sub["n_contact_timesteps"].to_numpy()
            cov = sub["mask_coverage_within_contact"].apply(lambda d: d[axis]).to_numpy()
            n_identifiable = np.round(cov * n_contact).astype(int)
            total_contact = int(n_contact.sum())
            total_identifiable = int(n_identifiable.sum())
            rows.append({
                "axis": axis,
                "n_contact_timesteps": total_contact,
                "n_identifiable": total_identifiable,
                "coverage_within_contact": total_identifiable / total_contact if total_contact else float("nan"),
            })
        return pd.DataFrame(rows)

    cov_all = pooled_coverage(ok)
    cov_all["task"] = "__all__"
    per_task = []
    for task, g in ok.groupby("task"):
        c = pooled_coverage(g)
        c["task"] = task
        per_task.append(c)
    coverage_table = pd.concat([cov_all] + per_task, ignore_index=True)
    coverage_table.to_csv(os.path.join(cli.out_dir, "mask_coverage.csv"), index=False)

    headline = cov_all.set_index("axis")["coverage_within_contact"].to_dict()
    print("\nHeadline mask coverage (within contact timesteps), whole dataset "
          "(via fr3_bilateral_teleop's extract_impedance_labels.py):")
    for axis, v in headline.items():
        print(f"  {axis}: {v:.1%}")

    with open(os.path.join(cli.out_dir, "mask_coverage_headline.json"), "w") as f:
        json.dump({
            "per_axis_coverage_within_contact": headline,
            "overall_mean": float(np.mean(list(headline.values()))),
            "n_episodes_ok": int(len(ok)),
            "n_episodes_failed_frame_fit": int(len(failed)),
            "extraction_args": vars(args),
        }, f, indent=2, default=str)


if __name__ == "__main__":
    main()
