"""Calibrate the rigid tool-tip offset (flange -> wiper tip, in the flange
frame) needed to express FK(observation.state) in the same Cartesian
convention as observation.leader_pose / action.

Why this is needed: the dataset logs observation.state as 7-DoF joint
position (not Cartesian follower pose), and the standard Panda flange frame
sits short of the physically mounted rigid eraser by a fixed offset. Rather
than hand-measure it, fit it: pick frames that are genuinely static (near-zero
external force *and* near-zero joint velocity -- not just low force, since
bilateral teleop keeps driving the leader even off-contact, which otherwise
contaminates a low-force-only selection with real tracking lag, not pure
geometry). On those frames x_l ~= x_f exactly, so

    leader_pos(t) = FK_flange_pos(q(t)) + R_flange(q(t)) @ tool_offset

is linear in tool_offset and solvable by least squares pooled across all
sessions (one rigid tool, frozen setup).
"""

import os

import numpy as np

import dataset_io as dio
import panda_fk as fk

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(SCRIPT_DIR, "tool_offset.npy")

FORCE_THRESHOLD_N = 0.5
VEL_THRESHOLD_RAD_S = 0.01


def main():
    rows = []
    for session in dio.SESSIONS:
        df = dio.load_frames(session)
        q = dio.stack_col(df, "observation.state")
        leader_pose = dio.stack_col(df, "observation.leader_pose")
        wrench = dio.stack_col(df, "observation.wrench.external_base")
        vel = dio.stack_col(df, "observation.velocity")
        force_mag = np.linalg.norm(wrench[:, :3], axis=1)
        vel_mag = np.linalg.norm(vel, axis=1)
        static = (force_mag < FORCE_THRESHOLD_N) & (vel_mag < VEL_THRESHOLD_RAD_S)
        print(f"{session}: {static.sum()} static calibration frames / {len(static)} total")
        rows.append((q[static], leader_pose[static]))

    Q = np.concatenate([r[0] for r in rows], axis=0)
    LP = np.concatenate([r[1] for r in rows], axis=0)
    print(f"pooled static frames: {Q.shape[0]}")

    pos, rot = fk.fk_batch(Q)
    A = rot.reshape(-1, 3)
    b = (LP[:, :3] - pos).reshape(-1)
    tool_offset, *_ = np.linalg.lstsq(A, b, rcond=None)
    print("fitted tool_offset (flange frame, meters):", tool_offset)

    pred = pos + np.einsum("nij,j->ni", rot, tool_offset)
    err_mm = np.linalg.norm(pred - LP[:, :3], axis=1) * 1000
    print(f"residual position error (mm): mean={err_mm.mean():.3f} "
          f"p50={np.median(err_mm):.3f} p95={np.percentile(err_mm, 95):.3f} max={err_mm.max():.3f}")

    np.save(OUT_PATH, tool_offset)
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
