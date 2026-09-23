"""Synthetic bilateral demonstration with known ground-truth stiffness.

Generates a demonstration whose true per-axis K is known by construction: a
contact phase against a tilted plane with a prescribed (by default anisotropic)
stiffness, bracketed by free space where no consistent K exists and no real
contact occurs.

This is what lets the extraction pipeline be checked end to end -- contact-frame
fit, windowed regression, and identifiability mask together -- against ground
truth, with no robot and no recorded data. The whole test suite rests on it, and
so did the original development of the pipeline before it was ever pointed at
real pilot data.

The generator deliberately shares geometry.build_inplane_basis with the fitting
code it is used to test. The in-plane axis choice is gauge-arbitrary (any
rotation about the normal is equally valid), so without one shared convention
the ground truth and the recovered frame would disagree about which in-plane
direction is "x" and the per-axis comparison would be meaningless. An earlier
version of this generator did exactly that and made erx/ery look swapped.

Extracted verbatim from the reference implementation.
"""

from typing import Dict, Tuple

import numpy as np

from .geometry import build_inplane_basis, quat_multiply

__all__ = ["synthetic_demo"]


def synthetic_demo(
        seed: int = 0, true_k: "np.ndarray | None" = None, k_modulation_depth: float = 0.0,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Builds a synthetic demo with a KNOWN per-axis true stiffness during a contact phase,
    and free-space (no consistent K, no real contact) before/after -- so the pipeline can be
    checked against ground truth end-to-end (frame fit, regression, mask) before ever
    pointing it at real pilot data. `true_k` defaults to an anisotropic set (matching the
    real T1 rig's expected "compliant along normal, stiff in-plane" pattern); pass an
    isotropic array (e.g. evaluate_gate1.py's self-test) to check the anisotropy condition
    actually discriminates rather than always passing. `k_modulation_depth` (0-1) adds a
    slow sinusoidal swing to the true stiffness DURING the contact phase (e.g.
    evaluate_gate1.py's self-test uses this to give condition (ii), "within-demo K
    variation," genuine signal to detect -- with depth=0 the true K is constant and that
    condition has nothing but noise to measure, which is a fine test of extraction accuracy
    but not of the variation-vs-noise check)."""
    rng = np.random.default_rng(seed)
    dt = 1e-3
    duration = 8.0
    t = np.arange(0.0, duration, dt)
    n = len(t)

    # Board frame: normal tilted 20 degrees from vertical (matching the real T1 rig),
    # rotated about the base Y axis. Board plane passes through a point 0.4m in front of
    # the base along X.
    tilt = np.radians(20.0)
    normal_true = np.array([np.sin(tilt), 0.0, np.cos(tilt)])
    board_point = np.array([0.4, 0.0, 0.3])

    contact_start, contact_end = 2.0, 6.0
    in_contact = (t >= contact_start) & (t < contact_end)

    if true_k is None:
        true_k = np.array([300.0, 250.0, 800.0, 20.0, 15.0, 40.0])  # N/m x3, Nm/rad x3
    true_d = np.array([15.0, 12.0, 25.0, 1.0, 0.8, 1.5])

    # Follower stays on the board plane while "in contact" (small in-plane wander), lifts
    # off into free space otherwise. Build follower position directly in a frame aligned
    # with normal_true so contact points are exactly planar (a clean test of fit_contact_frame).
    # Uses the SAME in-plane basis convention fit_contact_frame will recover from this data
    # (see build_inplane_basis) -- otherwise ground truth and the pipeline's own recovered
    # frame disagree on which in-plane direction is "x" vs "y" and the per-axis comparison
    # below is meaningless (caught by this self-test's first version: erx/ery looked
    # swapped because of exactly this mismatch).
    x_axis, y_axis = build_inplane_basis(normal_true)
    R_true = np.column_stack([x_axis, y_axis, normal_true])

    e_true = np.zeros((n, 6))
    for axis in range(6):
        e_true[in_contact, axis] = 0.01 * np.sin(2 * np.pi * (0.5 + 0.1 * axis) * t[in_contact])
    edot_true = np.gradient(e_true, t, axis=0)

    k_modulation = np.ones(n)
    if k_modulation_depth > 0.0:
        # One full swing over the contact window -- slow relative to the ~0.5-1.1 Hz probe
        # frequencies above, so within any single 300ms regression window K is close to
        # locally constant (a fair test of the windowed regression), while the WHOLE-DEMO
        # variation is real and well above adjacent-window noise.
        contact_duration = contact_end - contact_start
        k_modulation[in_contact] = 1.0 + k_modulation_depth * np.sin(
            2 * np.pi * (t[in_contact] - contact_start) / contact_duration)

    f_contact_frame = np.zeros((n, 6))
    for axis in range(6):
        f_contact_frame[:, axis] = (
            true_k[axis] * k_modulation * e_true[:, axis]
            + true_d[axis] * edot_true[:, axis])
    f_contact_frame[in_contact] += rng.normal(0.0, 0.03, size=(int(np.sum(in_contact)), 6))
    # Add a purely-normal preload so ||force|| clears the contact threshold even on axes
    # with a small oscillation amplitude (mirrors real wiping: mostly axial preload).
    f_contact_frame[in_contact, 2] += 4.0

    follower_pos_contact = 0.02 * np.stack([
        np.sin(0.3 * t), np.cos(0.2 * t), np.zeros(n)], axis=1)
    follower_pos = board_point + follower_pos_contact @ R_true.T
    follower_pos[~in_contact, 2] += 0.05  # lift off the plane in free space

    leader_pos = follower_pos + e_true[:, 0:3] @ R_true.T

    def const_quat(n):
        return np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (n, 1))

    q_l, q_f = const_quat(n), const_quat(n)
    # Small, known relative rotation during contact (rotational axes 3:6 of e_true), applied
    # as a small-angle approximation (valid since e_true's rotational entries are <= 0.01 rad).
    for axis, local_axis in zip([3, 4, 5], [x_axis, y_axis, normal_true]):
        half_angle = e_true[:, axis] / 2.0
        dq = np.zeros((n, 4))
        dq[:, 0:3] = local_axis[None, :] * np.sin(half_angle)[:, None]
        dq[:, 3] = np.cos(half_angle)
        q_l = np.array([quat_multiply(dq[i], q_l[i]) for i in range(n)])

    wrench_force = f_contact_frame[:, 0:3] @ R_true.T
    wrench_torque = f_contact_frame[:, 3:6] @ R_true.T
    wrench_force[~in_contact] = rng.normal(0.0, 0.05, size=(int(np.sum(~in_contact)), 3))
    wrench_torque[~in_contact] = rng.normal(0.0, 0.02, size=(int(np.sum(~in_contact)), 3))

    demo = {
        "t": t,
        "lx": leader_pos[:, 0], "ly": leader_pos[:, 1], "lz": leader_pos[:, 2],
        "lqx": q_l[:, 0], "lqy": q_l[:, 1], "lqz": q_l[:, 2], "lqw": q_l[:, 3],
        "fx": follower_pos[:, 0], "fy": follower_pos[:, 1], "fz": follower_pos[:, 2],
        "fqx": q_f[:, 0], "fqy": q_f[:, 1], "fqz": q_f[:, 2], "fqw": q_f[:, 3],
        "wfx": wrench_force[:, 0], "wfy": wrench_force[:, 1], "wfz": wrench_force[:, 2],
        "wtx": wrench_torque[:, 0], "wty": wrench_torque[:, 1], "wtz": wrench_torque[:, 2],
    }
    return demo, true_k
