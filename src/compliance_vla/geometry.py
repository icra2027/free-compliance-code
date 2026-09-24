"""Contact-frame geometry: quaternion helpers, plane fitting, and frame rotations.

Implements the contact-frame definition of the paper's Method section
(Eq. "contact-frame SVD"): a contact frame is fitted per demonstration from the
follower's own in-contact positions -- the direction of least variance of those
positions IS the surface normal -- rather than from a dedicated touch
calibration, and is rejected outright (not silently accepted as degenerate)
when the fit fails its planarity gate.

Extracted verbatim from the reference implementation that produced the paper's
numbers; only the module split and imports differ.
"""

from typing import Dict, Tuple

import numpy as np

AXIS_NAMES = ["ex", "ey", "ez", "erx", "ery", "erz"]
FORCE_AXIS_NAMES = ["fx", "fy", "fz", "tx", "ty", "tz"]
TRANSLATIONAL_AXES = [0, 1, 2]
ROTATIONAL_AXES = [3, 4, 5]


# ---------------------------------------------------------------------------
# Quaternion helpers (x, y, z, w convention, matching geometry_msgs/Quaternion)
# ---------------------------------------------------------------------------

def quat_conjugate(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array([-x, -y, -z, w])


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ])


def quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    """SO(3) log map: quaternion -> axis-angle vector (angle in [0, pi])."""
    v = q[:3]
    w = q[3]
    v_norm = np.linalg.norm(v)
    angle = 2.0 * np.arctan2(v_norm, w)
    if v_norm < 1e-9:
        return np.zeros(3)
    return angle * (v / v_norm)


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array([
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)],
    ])

# ---------------------------------------------------------------------------
# Contact-frame fitting (from data, no dedicated calibration motion)
# ---------------------------------------------------------------------------

def build_inplane_basis(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Completes a right-handed (x, y) in-plane basis for a given plane normal. In-plane
    axis CHOICE is inherently gauge-arbitrary (any rotation about the normal is equally
    valid) -- this one fixed convention (project global X onto the plane, or global Y if
    the normal is too close to X) is used by both fit_contact_frame (recovering it from
    real data) and synthetic_demo (generating self-test ground truth), so the two agree on
    which in-plane direction is "x" instead of being compared across two different,
    independently-chosen bases."""
    seed = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(seed, normal)) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    x_axis = seed - np.dot(seed, normal) * normal
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(normal, x_axis)
    return x_axis, y_axis


def fit_contact_frame(
        follower_pos: np.ndarray, wrench_force: np.ndarray,
        contact_force_threshold: float, min_contact_samples: int = 50,
        max_planarity_ratio: float = 0.15) -> Dict:
    """Fits the board's contact frame (z = normal, x/y in-plane) from the follower's own
    positions while in contact, rather than a dedicated touch calibration. See module
    docstring. Returns a dict with 'R' (3x3, contact-frame axes as columns, base frame) and
    quality diagnostics; 'ok' is False (and 'R' is None) if the fit doesn't pass its own
    gate -- refuses to hand back a silently-untrustworthy frame.

    `contact_force_threshold` here is deliberately a HIGHER, firmer threshold than the
    mask's general sustained-contact condition (args.contact_force_threshold, ~2N) -- real
    pilot data showed light/transitional contact near the mask's lower threshold
    (approach, retreat, grazing touches) is genuinely NOT planar (planarity_ratio 0.15-0.24
    at 2N on 4/5 real pilots), while firm, confident contact (>~6-8N) is (0.007-0.13 on the
    same demos at 8N) -- a real geometric distinction, not a bug: a light graze's contact
    point is inherently less precisely localized than solid contact against a rigid board.
    Fitting the PLANE from firm contact only, while still using the lower threshold for the
    mask's SNR/excitation eligibility, uses the right threshold for each different job.
    """
    force_mag = np.linalg.norm(wrench_force, axis=1)
    contact_mask = force_mag > contact_force_threshold
    n_contact = int(np.sum(contact_mask))
    result = {
        "n_contact_samples": n_contact, "contact_force_threshold": contact_force_threshold,
        "ok": False, "R": None, "planarity_ratio": None,
    }
    if n_contact < min_contact_samples:
        result["reason"] = (
            f"only {n_contact} in-contact samples (need >= {min_contact_samples}) -- demo "
            "may not contain real contact, or --contact-force-threshold is too high")
        return result

    pts = follower_pos[contact_mask]
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    # SVD of the centered in-contact points: the smallest singular vector is the direction
    # of least variance -- the plane normal, if these points really are roughly planar.
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    normal = vt[2]
    planarity_ratio = float(s[2] / s[0]) if s[0] > 1e-9 else float("inf")
    result["planarity_ratio"] = planarity_ratio
    if planarity_ratio > max_planarity_ratio:
        result["reason"] = (
            f"planarity_ratio={planarity_ratio:.3f} exceeds max_planarity_ratio="
            f"{max_planarity_ratio} -- in-contact points don't look like they lie on a "
            "plane; refusing to trust the fitted normal")
        return result

    # Orient normal so mean contact force has a POSITIVE component along it: the estimated
    # external wrench is the environment pushing back on the robot, i.e. away from the
    # board, when the robot presses INTO the board -- so a consistent "away from board"
    # convention needs mean(f_contact) . normal > 0.
    mean_force = wrench_force[contact_mask].mean(axis=0)
    if np.dot(mean_force, normal) < 0:
        normal = -normal

    x_axis, y_axis = build_inplane_basis(normal)

    result["ok"] = True
    result["R"] = np.column_stack([x_axis, y_axis, normal])
    result["normal_base_frame"] = normal.tolist()
    return result


# ---------------------------------------------------------------------------
# Pose error e(t) = x_l(t) (-) x_f(t), rotated into the contact frame
# ---------------------------------------------------------------------------

def compute_pose_error(demo: Dict[str, np.ndarray], R_contact: np.ndarray) -> np.ndarray:
    n = len(demo["t"])
    e = np.zeros((n, 6))
    leader_pos = np.column_stack([demo["lx"], demo["ly"], demo["lz"]])
    follower_pos = np.column_stack([demo["fx"], demo["fy"], demo["fz"]])
    e_pos_base = leader_pos - follower_pos
    e[:, 0:3] = e_pos_base @ R_contact  # R^T applied via row-vector convention

    for i in range(n):
        q_l = np.array([demo["lqx"][i], demo["lqy"][i], demo["lqz"][i], demo["lqw"][i]])
        q_f = np.array([demo["fqx"][i], demo["fqy"][i], demo["fqz"][i], demo["fqw"][i]])
        q_rel = quat_multiply(q_l, quat_conjugate(q_f))
        rotvec_base = quat_to_rotvec(q_rel)
        e[i, 3:6] = R_contact.T @ rotvec_base
    return e


def numerically_differentiate(t: np.ndarray, x: np.ndarray, smooth_window: int = 5) -> np.ndarray:
    """Short moving-average smoothing then central difference -- same pattern as
    calibrate_payload.py's numerically_differentiate, applied here to the whole demo (one
    continuous recording, not per-waypoint-segment)."""
    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        x_smooth = np.apply_along_axis(
            lambda col: np.convolve(col, kernel, mode="same"), axis=0, arr=x)
    else:
        x_smooth = x
    return np.gradient(x_smooth, t, axis=0)


def rotate_force_to_contact_frame(
        wrench_force: np.ndarray, wrench_torque: np.ndarray, R_contact: np.ndarray) -> np.ndarray:
    f = np.zeros((len(wrench_force), 6))
    f[:, 0:3] = wrench_force @ R_contact
    f[:, 3:6] = wrench_torque @ R_contact
    return f
