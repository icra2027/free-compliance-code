#!/usr/bin/env python3
"""Day 14: pure-numpy helpers for scripts/run_pilot_rollout.py (external/fr3_bilateral_teleop)
-- everything the pilot-rollout harness needs that does NOT depend on ROS2/rclpy, split out
so it can be unit-tested in this venv (which has no rclpy, see scripts/README.md) rather than
only ever exercised on the real rig.

Two independent jobs:

1. Rotating a predicted stiffness vector from the auto-fit CONTACT frame (what B5/B3's
   compliance head outputs, per src/compliance_vla/policy/labels.py: "log_k IS in the auto-fit contact
   frame... not base frame") into the base-frame DIAGONAL `target_stiffness` 6-vector
   variable_impedance_controllers' CartesianController actually consumes (Float64MultiArray, [kx,ky,kz,
   krx,kry,krz], base frame -- confirmed by external/fr3_bilateral_teleop/scripts/
   probe_variable_impedance_sinusoid.py, which publishes to it with no frame parameter).

   K_base_full = R @ diag(K_contact) @ R.T is the correct tensor-rotation formula (R's
   columns are the contact-frame axes expressed in base frame, see
   extract_impedance_labels.fit_contact_frame's docstring/return convention) but is in
   general NOT diagonal once rotated -- the controller only accepts a diagonal, so this
   necessarily drops the off-diagonal (cross-axis coupling) terms of K_base_full. This is
   the exact "Week-4 controller-integration question" src/compliance_vla/policy/labels.py's own docstring
   already flags as out of scope for training; scripts/fit_frozen_contact_frame.py explains
   why a Day-14 PILOT rollout resolves it with a fixed, offline-fit contact frame rather
   than a live per-rollout re-fit. Reported explicitly (see diagonal_dropped_fraction below)
   rather than silently accepted.

2. Deciding, at each control-loop tick, which step of the last-predicted action chunk to
   command -- proposal §4.3: "Policy replans at 10Hz; controller runs at 1kHz," with cubic-
   spline interpolation on x_eq and a log-space stiffness rate limiter (already built,
   external/fr3_bilateral_teleop's variable_impedance_controllers integration, Day 3/5) doing the actual
   smoothing inside the controller. This module's `chunk_step_index` only answers "how many
   30Hz chunk steps has replan_period_sec covered," a plain arithmetic lookup -- the
   controller-side rate limiter and cubic-spline interpolation (already validated on real
   hardware, Day 5's commanded-vs-realized-stiffness figure) do the rest; this is
   deliberately NOT a second implementation of that machinery.
"""

import numpy as np


def rotate_diag_stiffness_to_base(k_contact_diag: np.ndarray, R_contact: np.ndarray):
    """k_contact_diag: (6,) or (N,6) stiffness values IN THE CONTACT FRAME (already
    exp()'d out of log-space -- see src/compliance_vla/policy/labels.py "log_k IS in the auto-fit contact
    frame"). Translational (0:3) and rotational (3:6) sub-blocks are rotated by the SAME
    R_contact (3x3, translational contact axes = rotational contact axes by construction --
    extract_impedance_labels.fit_contact_frame fits one frame, used for both compute_pose_error's
    position AND rotation-vector error columns).

    Returns (k_base_diag, diagonal_dropped_fraction):
      k_base_diag: (6,) or (N,6), the diagonal of R @ diag(k) @ R.T per 3-block -- what
        target_stiffness actually gets published as (Float64MultiArray requires a plain
        6-vector, not a 6x6 matrix).
      diagonal_dropped_fraction: how much of the FULL rotated tensor's Frobenius norm lives
        off-diagonal (0 = the rotation was axis-aligned, nothing lost; near 1 = a large
        fraction of the true anisotropic behaviour cannot be realized as a base-frame
        diagonal and is being dropped) -- report this per rollout, don't silently discard it.
    """
    k = np.asarray(k_contact_diag, dtype=np.float64)
    R = np.asarray(R_contact, dtype=np.float64)
    single = k.ndim == 1
    if single:
        k = k[None, :]
    n = k.shape[0]
    k_base = np.empty((n, 6))
    dropped = np.empty(n)
    for blk, sl in enumerate((slice(0, 3), slice(3, 6))):
        k_blk = k[:, sl]  # (n, 3)
        for i in range(n):
            K_full = R @ np.diag(k_blk[i]) @ R.T  # (3,3), full rotated tensor
            diag_part = np.diag(np.diag(K_full))
            off_norm = np.linalg.norm(K_full - diag_part)
            full_norm = np.linalg.norm(K_full)
            dropped_blk = float(off_norm / full_norm) if full_norm > 1e-12 else 0.0
            k_base[i, sl] = np.diag(K_full)
            dropped[i] = dropped_blk if blk == 0 else max(dropped[i], dropped_blk)  # worst of the two blocks
    if single:
        return k_base[0], float(dropped[0])
    return k_base, dropped


def chunk_step_index(t_since_replan_sec: float, action_rate_hz: float, chunk_size: int) -> int:
    """Which 0-indexed step of the currently-held action chunk should be commanded right
    now, given `t_since_replan_sec` elapsed since the chunk was predicted. Clamped to the
    last available step (holds the final predicted step, rather than indexing out of range)
    if replanning is running behind schedule -- matches ACT-style "hold the last chunk"
    graceful degradation rather than crashing the control loop.
    """
    if action_rate_hz <= 0 or chunk_size <= 0:
        raise ValueError(f"action_rate_hz={action_rate_hz}, chunk_size={chunk_size} must be positive")
    idx = int(t_since_replan_sec * action_rate_hz)
    return min(max(idx, 0), chunk_size - 1)


def _self_test():
    # rotate_diag_stiffness_to_base: identity rotation -> exact passthrough, zero dropped.
    k = np.array([200.0, 250.0, 150.0, 10.0, 12.0, 8.0])
    k_base, dropped = rotate_diag_stiffness_to_base(k, np.eye(3))
    assert np.allclose(k_base, k), f"identity rotation should pass k through unchanged: {k_base}"
    assert dropped < 1e-9, f"identity rotation should drop nothing: {dropped}"

    # A rotation about the z axis mixes only x/y (both translational AND rotational blocks,
    # applied with the SAME R) -- with kx != ky the result should NOT simply equal (kx, ky),
    # and some off-diagonal energy should appear (dropped > 0) unless kx == ky (isotropic in
    # that plane, in which case a z-rotation genuinely changes nothing).
    theta = np.deg2rad(30.0)
    Rz = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    k_aniso = np.array([300.0, 100.0, 150.0, 20.0, 5.0, 8.0])  # anisotropic in-plane (kx != ky)
    k_base2, dropped2 = rotate_diag_stiffness_to_base(k_aniso, Rz)
    assert dropped2 > 1e-6, "anisotropic in-plane stiffness rotated about z should lose some off-diagonal energy"
    assert not np.allclose(k_base2[0:2], k_aniso[0:2]), "rotation should actually change the in-plane values"
    # trace (sum of eigenvalues) is rotation-invariant for the FULL tensor; the diagonal-only
    # k_base sum need not equal the pre-rotation sum in general, but IS exactly the trace of
    # R@diag(k)@R.T (trace is basis-independent), which itself equals sum(k) -- check that
    # invariant directly as a correctness cross-check on the rotation math.
    assert abs(k_base2[0:3].sum() - k_aniso[0:3].sum()) < 1e-8, "trace of the translational block must be rotation-invariant"
    assert abs(k_base2[3:6].sum() - k_aniso[3:6].sum()) < 1e-8, "trace of the rotational block must be rotation-invariant"

    # isotropic in-plane stiffness: a z-rotation should be a no-op (both dropped and value).
    k_iso = np.array([200.0, 200.0, 150.0, 10.0, 10.0, 8.0])
    k_base3, dropped3 = rotate_diag_stiffness_to_base(k_iso, Rz)
    assert np.allclose(k_base3, k_iso, atol=1e-8), f"isotropic in-plane rotation about its own axis should be a no-op: {k_base3}"
    assert dropped3 < 1e-9

    # batched (N,6) input matches per-row single-vector calls.
    K = np.stack([k, k_aniso, k_iso], axis=0)
    Kb_batch, d_batch = rotate_diag_stiffness_to_base(K, Rz)
    for i, row in enumerate(K):
        kb_i, d_i = rotate_diag_stiffness_to_base(row, Rz)
        assert np.allclose(Kb_batch[i], kb_i)
        assert abs(d_batch[i] - d_i) < 1e-10

    # chunk_step_index: basic arithmetic, clamping at both ends.
    assert chunk_step_index(0.0, action_rate_hz=30.0, chunk_size=32) == 0
    assert chunk_step_index(0.5, action_rate_hz=30.0, chunk_size=32) == 15  # 0.5*30=15.0 -> int() = 15
    assert chunk_step_index(10.0, action_rate_hz=30.0, chunk_size=32) == 31, "must clamp to chunk_size-1, not overrun"
    assert chunk_step_index(-1.0, action_rate_hz=30.0, chunk_size=32) == 0, "must clamp negative elapsed time to 0"

    print("scripts/controller_frame_utils.py self-test: PASS")


if __name__ == "__main__":
    _self_test()
