"""Frame conversions between the extraction's contact frame and the controller.

The compliance head predicts log K in the AUTO-FIT CONTACT FRAME, which is where
the labels were extracted. The low-level Cartesian impedance controller consumes
a base-frame DIAGONAL stiffness 6-vector. Bridging the two is not a relabelling:
the correct tensor rotation K_base = R diag(K_contact) R^T is in general NOT
diagonal, so expressing it as the diagonal the controller accepts necessarily
discards the off-diagonal cross-axis coupling terms.

That loss is reported explicitly rather than silently accepted -- see
`diagonal_dropped_fraction` -- so a rollout in which a large fraction of the
extracted anisotropy cannot actually be realized is visible in the logs instead
of looking like a controller that simply underperformed.

Extracted verbatim from the reference implementation.
"""

import numpy as np

__all__ = ["rotate_diag_stiffness_to_base", "chunk_step_index"]


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
