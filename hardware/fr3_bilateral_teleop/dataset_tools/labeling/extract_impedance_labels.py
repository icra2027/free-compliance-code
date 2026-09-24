#!/usr/bin/env python3
"""Impedance label extraction from a recorded bilateral demonstration.

Offline, no ROS dependency (record_demo.py produces the input CSV; this reads it back).
Turns `x_l(t)` (leader pose), `x_f(t)` (follower pose), `f(t)` (follower estimated wrench)
into per-axis, per-timestep stiffness labels `K(t)` with an identifiability mask, in the
task/contact frame.

Design decisions, and why:

**Contact frame, fit from the demo data itself, not a separate calibration.** The method
puts z along the calibrated board normal for wiping. Rather than adding a new dedicated
board-touch calibration step (which would mean more real-hardware motion, and an earlier
session had already had a safety near-miss), this fits the plane directly from the
follower's own position samples during genuine contact (`||f|| > --contact-force-threshold`):
those positions should lie approximately on the 2D board surface, so the direction of least
variance (smallest SVD singular vector) IS the board normal. Zero extra robot motion; reuses
data already being collected for the demo itself. Gated on a planarity quality check -- see
`fit_contact_frame` -- rather than silently trusted, matching this package's established
"quality gate, don't silently trust" pattern (calibrate_payload.py, fit_residual_bias.py).

**Output at 30 Hz, not 1 kHz.** The raw signals are ~1 kHz, but the label this ultimately
supervises is the policy's `log k` output head at the action-chunk rate (30 Hz,
H=32). Computing a windowed regression at 1 kHz would be ~33x more work for no signal the
policy ever sees. Each 30 Hz output timestep uses a TRAILING 300 ms window of the raw 1 kHz
data (the method's window length), i.e. causal, not centered.

**Regression solved via `scipy.optimize.least_squares` with bounds**, not a hand-rolled
solver -- the objective `min_{k_i,d_i} ... + lambda*||log k_i - log k_i^prior||^2` is
nonlinear in `log k_i` (the regularizer) even though it's linear in `(k_i, d_i)` without it,
so this is a genuine bounded nonlinear least-squares problem, not something with a closed
form. `scipy` (already a dependency elsewhere in this environment) gets a well-tested
trust-region-reflective solver instead of a hand-rolled Gauss-Newton loop.

**Mask conditions (1)/(4) are window properties (excitation, sustained contact); (2)/(3) are
instantaneous** (the method's wording, "|e_i| above the pose-error noise floor", reads as
a per-timestep magnitude check, not a windowed one) -- evaluated at the single raw sample
nearest each 30 Hz output timestep.

**Damping is fit freely per window** (d_i is fit freely only for the offline analysis
figure that shows the critical-damping assumption is reasonable -- exactly what the
pilot analysis needs). The POLICY-TARGET constrained damping `D = 2*zeta*sqrt(K*Mhat)` needs a
Cartesian effective-mass estimate (`Mhat`) from the arm's joint-space mass matrix and Jacobian,
which this demo-CSV pipeline does not have (would need q + the payload_model_broadcaster
snapshot recorded alongside every demo, not just x_l/x_f/f) -- deliberately deferred to training
prep, since the pilot analysis doesn't need it.

**sigma_f defaults to the real measured values** (free-space-residual
substitute: fx=0.726 fy=0.418 fz=0.461 N, tx=0.275 ty=0.434 tz=0.155 Nm) -- override
--sigma-f if re-measured later. sigma_e (pose-error noise floor) is NOT a measured quantity
here -- encoder-measured pose error is accurate to ~0.01mm, so a small
fixed constant is used per axis TYPE (translational vs rotational), not fit from data.

Usage:
    ros2 run fr3_bilateral_teleop extract_impedance_labels.py --input demo1.csv demo2.csv
    python3 dataset_tools/labeling/extract_impedance_labels.py --self-test   # synthetic data, no demo needed
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy.optimize import least_squares

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

AXIS_NAMES = ["ex", "ey", "ez", "erx", "ery", "erz"]
FORCE_AXIS_NAMES = ["fx", "fy", "fz", "tx", "ty", "tz"]
TRANSLATIONAL_AXES = [0, 1, 2]
ROTATIONAL_AXES = [3, 4, 5]

# Bound [k_min, k_max] to the controller's realizable range (50-1500 N/m
# translational, 5-100 N*m/rad rotational).
K_MIN = np.array([50.0, 50.0, 50.0, 5.0, 5.0, 5.0])
K_MAX = np.array([1500.0, 1500.0, 1500.0, 100.0, 100.0, 100.0])

# Real, measured (via the free-space-residual substitute, since the
# ground-truth hanging-mass check was not safely completable that session).
DEFAULT_SIGMA_F = np.array([0.726, 0.418, 0.461, 0.275, 0.434, 0.155])

# Not measured -- encoder-measured pose error is accurate to ~0.01 mm.
# Rotational floor is an assumed order-of-magnitude default, not a measured quantity.
DEFAULT_SIGMA_E = np.array([1e-5, 1e-5, 1e-5, 1e-4, 1e-4, 1e-4])


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
# Demo loading
# ---------------------------------------------------------------------------

def load_demo_csv(path: Path) -> Dict[str, np.ndarray]:
    rows = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})
    if len(rows) < 50:
        raise ValueError(f"{path}: only {len(rows)} rows, too few to extract from")
    cols = {k: np.array([r[k] for r in rows]) for k in rows[0].keys()}
    return cols


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


# ---------------------------------------------------------------------------
# Windowed, log-space, box-constrained regression
# ---------------------------------------------------------------------------

def _fit_one_window(
        e_win: np.ndarray, edot_win: np.ndarray, f_win: np.ndarray,
        log_k_prior: float, log_k_min: float, log_k_max: float,
        d_max: float, lam: float) -> Tuple[float, float]:
    sqrt_lam = np.sqrt(lam)

    def residuals(params):
        log_k, d = params
        pred = np.exp(log_k) * e_win + d * edot_win
        return np.concatenate([f_win - pred, [sqrt_lam * (log_k - log_k_prior)]])

    x0 = np.array([log_k_prior, 0.0])
    bounds = ([log_k_min, 0.0], [log_k_max, d_max])
    result = least_squares(residuals, x0, bounds=bounds, method="trf")
    log_k, d = result.x
    return float(np.exp(log_k)), float(d)


def extract_axis(
        t: np.ndarray, e_axis: np.ndarray, edot_axis: np.ndarray, f_axis: np.ndarray,
        output_times: np.ndarray, window_sec: float, k_min: float, k_max: float,
        lam: float, d_max: float, min_window_samples: int = 20) -> Dict[str, np.ndarray]:
    log_k_prior = 0.5 * (np.log(k_min) + np.log(k_max))
    log_k_min, log_k_max = np.log(k_min), np.log(k_max)

    n_out = len(output_times)
    k_out = np.full(n_out, np.nan)
    d_out = np.full(n_out, np.nan)
    cond_out = np.full(n_out, np.inf)
    n_win_samples = np.zeros(n_out, dtype=int)

    for i, t_out in enumerate(output_times):
        window_mask = (t > t_out - window_sec) & (t <= t_out)
        n_win = int(np.sum(window_mask))
        n_win_samples[i] = n_win
        if n_win < min_window_samples:
            continue
        e_win, edot_win, f_win = e_axis[window_mask], edot_axis[window_mask], f_axis[window_mask]
        A = np.column_stack([e_win, edot_win])
        gram = A.T @ A
        cond_out[i] = float(np.linalg.cond(gram))
        k_out[i], d_out[i] = _fit_one_window(
            e_win, edot_win, f_win, log_k_prior, log_k_min, log_k_max, d_max, lam)

    return {
        "k": k_out, "d": d_out, "condition_number": cond_out, "n_window_samples": n_win_samples,
    }


def nearest_sample_indices(t: np.ndarray, output_times: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(t, output_times)
    idx = np.clip(idx, 0, len(t) - 1)
    idx_prev = np.clip(idx - 1, 0, len(t) - 1)
    use_prev = np.abs(t[idx_prev] - output_times) < np.abs(t[idx] - output_times)
    return np.where(use_prev, idx_prev, idx)


# ---------------------------------------------------------------------------
# Identifiability mask (all four conditions)
# ---------------------------------------------------------------------------

def compute_mask(
        e: np.ndarray, f: np.ndarray, force_mag_contact: np.ndarray,
        condition_number: np.ndarray, nearest_idx: np.ndarray,
        sigma_e: np.ndarray, sigma_f: np.ndarray, kappa_max: float,
        contact_force_threshold: float, contact_fraction_required: float,
        t: np.ndarray, output_times: np.ndarray, window_sec: float,
) -> Tuple[np.ndarray, np.ndarray]:
    n_out = len(output_times)

    e_at_out = e[nearest_idx]  # (n_out, 6), instantaneous
    f_at_out = f[nearest_idx]  # (n_out, 6), instantaneous

    cond1 = condition_number[:, None] < kappa_max  # (n_out, 1) broadcasts to 6 axes
    cond2 = np.abs(e_at_out) > sigma_e[None, :]
    cond3 = np.abs(f_at_out) > sigma_f[None, :]

    cond4 = np.zeros(n_out, dtype=bool)
    for i, t_out in enumerate(output_times):
        window_mask = (t > t_out - window_sec) & (t <= t_out)
        n_win = int(np.sum(window_mask))
        if n_win == 0:
            continue
        in_contact_frac = np.mean(force_mag_contact[window_mask] > contact_force_threshold)
        cond4[i] = in_contact_frac >= contact_fraction_required

    mask = cond1 & cond2 & cond3 & cond4[:, None]
    return mask, cond4


# ---------------------------------------------------------------------------
# Top-level per-demo extraction
# ---------------------------------------------------------------------------

def extract_demo(
        demo: Dict[str, np.ndarray], args: argparse.Namespace, sigma_f: np.ndarray) -> Dict:
    t = demo["t"] - demo["t"][0]
    follower_pos = np.column_stack([demo["fx"], demo["fy"], demo["fz"]])
    wrench_force = np.column_stack([demo["wfx"], demo["wfy"], demo["wfz"]])
    wrench_torque = np.column_stack([demo["wtx"], demo["wty"], demo["wtz"]])

    frame_fit = fit_contact_frame(
        follower_pos, wrench_force, args.frame_fit_force_threshold,
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
            k_min, k_max, args.lam, args.d_max))

    # Mask condition (1) is about excitation of THIS axis's (e, edot) pair specifically --
    # use each axis's own condition number, not a pooled one.
    condition_number_per_axis = np.stack([pa["condition_number"] for pa in per_axis], axis=1)

    mask = np.zeros((len(output_times), 6), dtype=bool)
    contact_indicator = None  # identical across axes (global ||f|| contact detector); keep one
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
    # H1's identifiability condition (i) is literally "identifiable on >= 25% of CONTACT timesteps" --
    # report coverage restricted to that denominator, not the whole demo (most of which is
    # free-space transit/reset, which would otherwise dilute the number in an uninformative
    # direction).
    if n_contact_timesteps > 0:
        mask_coverage_within_contact = mask[contact_indicator].mean(axis=0)
    else:
        mask_coverage_within_contact = np.zeros(6)

    return {
        "status": "ok",
        "frame_fit": {k: v for k, v in frame_fit.items() if k != "R"},
        "R_contact": R_contact.tolist(),
        "output_times": output_times,
        "k": np.stack([pa["k"] for pa in per_axis], axis=1),
        "d": np.stack([pa["d"] for pa in per_axis], axis=1),
        "mask": mask,
        "mask_coverage": mask_coverage,
        "mask_coverage_within_contact": mask_coverage_within_contact,
        "n_contact_timesteps": n_contact_timesteps,
        "n_output_timesteps": len(output_times),
        "force_mag": force_mag,
        "t": t,
    }


# ---------------------------------------------------------------------------
# Figure (Figure 3: stiffness traces + mask coverage, phase-annotated by contact)
# ---------------------------------------------------------------------------

def generate_figure(result: Dict, contact_force_threshold: float, output_path: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    t_out = result["output_times"]
    t_raw = result["t"]
    in_contact_raw = result["force_mag"] > contact_force_threshold

    for axis in range(6):
        ax = axes[axis // 2, axis % 2]
        ax.fill_between(
            t_raw, 0, 1, where=in_contact_raw, transform=ax.get_xaxis_transform(),
            alpha=0.12, color="tab:orange", step="mid", label="contact (raw)")
        k = result["k"][:, axis]
        m = result["mask"][:, axis]
        ax.plot(t_out, k, color="tab:gray", linewidth=0.8, alpha=0.5, label="K (unmasked)")
        ax.scatter(t_out[m], k[m], s=6, color="tab:blue", label="K (mask=1)")
        coverage = result["mask_coverage"][axis]
        ax.set_title(f"{AXIS_NAMES[axis]}  (mask coverage {coverage:.0%})", fontsize=10)
        ax.set_yscale("log")
        if axis >= 4:
            ax.set_xlabel("t (s)")
    axes[0, 0].legend(fontsize=7, loc="upper right")
    fig.suptitle("Extracted stiffness K(t) per axis, contact-frame -- Figure 3")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Self-test: synthetic data with known ground truth, no ROS/hardware needed
# ---------------------------------------------------------------------------

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


def run_self_test(args: argparse.Namespace) -> int:
    demo, true_k = synthetic_demo()
    sigma_f = np.array([0.1, 0.1, 0.1, 0.02, 0.02, 0.02])  # tight, synthetic-noise-matched

    result = extract_demo(demo, args, sigma_f)
    if result["status"] != "ok":
        print(f"self-test: FAIL (extraction failed: {result['reason']})")
        return 1

    print(f"frame fit: planarity_ratio={result['frame_fit']['planarity_ratio']:.4f}")
    print(f"mask coverage per axis (all timesteps): {result['mask_coverage'].tolist()}")
    print(
        f"mask coverage per axis (within contact, {result['n_contact_timesteps']}/"
        f"{result['n_output_timesteps']} timesteps) -- what H1 condition (i) thresholds: "
        f"{result['mask_coverage_within_contact'].tolist()}")

    ok = True
    for axis in range(6):
        m = result["mask"][:, axis]
        if m.sum() < 20:
            print(f"axis {AXIS_NAMES[axis]}: FAIL -- only {m.sum()} identifiable timesteps")
            ok = False
            continue
        k_med = np.median(result["k"][m, axis])
        rel_err = abs(k_med - true_k[axis]) / true_k[axis]
        status = "OK" if rel_err < 0.2 else "FAIL"
        if status == "FAIL":
            ok = False
        print(
            f"axis {AXIS_NAMES[axis]}: true_k={true_k[axis]:.1f} median_fit={k_med:.1f} "
            f"rel_err={rel_err:.1%} coverage={m.mean():.1%} -- {status}")

    for axis in range(6):
        cov = result["mask_coverage_within_contact"][axis]
        if cov < 0.25:
            print(
                f"axis {AXIS_NAMES[axis]}: FAIL -- within-contact coverage {cov:.1%} is "
                "below H1 condition (i)'s 25% threshold on data that should clearly pass it")
            ok = False

    # Free-space phase must be mostly masked out on every axis -- no real excitation there.
    free_space_out = result["output_times"] < 1.5
    free_space_mask_rate = result["mask"][free_space_out].mean()
    print(f"free-space mask rate (should be near 0): {free_space_mask_rate:.1%}")
    if free_space_mask_rate > 0.05:
        print("FAIL: mask is passing free-space (no-contact) timesteps as identifiable")
        ok = False

    print(f"self-test: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input", nargs="+", type=Path, default=None,
        help="demo CSV(s) from record_demo.py")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--window-sec", type=float, default=0.3)
    parser.add_argument("--output-rate-hz", type=float, default=30.0)
    parser.add_argument(
        "--lam", type=float, default=0.1,
        help="log-space prior regularization weight, tunable (not a measured constant)")
    parser.add_argument(
        "--d-max", type=float, default=500.0,
        help="loose sanity bound on fitted damping, not a physical constant")
    parser.add_argument(
        "--kappa-max", type=float, default=1e3,
        help="max Gram-matrix condition number for mask condition (1)")
    parser.add_argument(
        "--contact-force-threshold", type=float, default=2.0,
        help="N, mask condition (4)'s sustained-contact / SNR-eligibility threshold")
    parser.add_argument(
        "--frame-fit-force-threshold", type=float, default=8.0,
        help="N, HIGHER firm-contact threshold used only for fitting the contact plane's "
             "geometry -- real pilot data showed light/transitional contact near the "
             "lower --contact-force-threshold is genuinely not planar (see fit_contact_frame "
             "docstring); firm contact is")
    parser.add_argument(
        "--contact-fraction-required", type=float, default=0.8,
        help="fraction of a window that must be in contact for 'sustained contact'")
    parser.add_argument(
        "--max-planarity-ratio", type=float, default=0.15,
        help="max smallest/largest SVD singular-value ratio for the auto-fit contact "
             "plane to be trusted")
    parser.add_argument(
        "--sigma-f", type=float, nargs=6, default=list(DEFAULT_SIGMA_F),
        metavar=("FX", "FY", "FZ", "TX", "TY", "TZ"),
        help="measured wrench noise floor, N/N/N/Nm/Nm/Nm -- defaults to the "
             "real measured values")
    parser.add_argument("--no-figure", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_test(args)

    if not args.input:
        print("error: --input required unless --self-test", file=sys.stderr)
        return 2

    sigma_f = np.array(args.sigma_f)
    output_dir = Path(
        args.output_dir or __import__("os").environ.get(
            "TELEOP_EXTRACTION_OUTPUT_DIR", "/tmp/franka_teleop_extraction"))
    output_dir.mkdir(parents=True, exist_ok=True)

    all_reports = []
    for path in args.input:
        print(f"--- {path} ---")
        demo = load_demo_csv(path)
        result = extract_demo(demo, args, sigma_f)
        if result["status"] != "ok":
            print(f"FAILED: {result['reason']}")
            all_reports.append(
                {"input": str(path), "status": "failed", "reason": result["reason"]})
            continue

        coverage = result["mask_coverage"]
        coverage_contact = result["mask_coverage_within_contact"]
        print(f"mask coverage per axis (all timesteps): "
              f"{dict(zip(FORCE_AXIS_NAMES, coverage.round(3).tolist()))}")
        print(
            f"mask coverage per axis (within contact, {result['n_contact_timesteps']}/"
            f"{result['n_output_timesteps']} timesteps) -- H1 condition (i) threshold is "
            f"25% here: {dict(zip(FORCE_AXIS_NAMES, coverage_contact.round(3).tolist()))}")

        stem = path.stem
        if not args.no_figure:
            fig_path = output_dir / f"{stem}_stiffness_traces.png"
            generate_figure(result, args.contact_force_threshold, fig_path)
            print(f"Wrote {fig_path}")

        report = {
            "input": str(path),
            "status": "ok",
            "frame_fit": result["frame_fit"],
            "R_contact": result["R_contact"],
            "mask_coverage_per_axis": dict(zip(FORCE_AXIS_NAMES, coverage.tolist())),
            "mask_coverage_overall": float(coverage.mean()),
            "mask_coverage_within_contact_per_axis": dict(
                zip(FORCE_AXIS_NAMES, coverage_contact.tolist())),
            "mask_coverage_within_contact_overall": float(coverage_contact.mean()),
            "n_contact_timesteps": result["n_contact_timesteps"],
            "sigma_f": sigma_f.tolist(),
            "n_output_timesteps": int(len(result["output_times"])),
        }
        report_path = output_dir / f"{stem}_extraction_report.json"
        report_path.write_text(json.dumps(report, indent=2))
        print(f"Wrote {report_path}")
        all_reports.append(report)

    summary_path = output_dir / f"extraction_summary_{int(time.time())}.json"
    summary_path.write_text(json.dumps(all_reports, indent=2))
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
