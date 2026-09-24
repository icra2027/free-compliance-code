"""Compliance label extraction from a bilateral demonstration.

Turns the leader pose x_l(t), follower pose x_f(t) and estimated contact wrench
f(t) of one bilateral demonstration into per-axis, per-timestep stiffness labels
K(t) in the contact frame, together with the binary identifiability mask that
says which of those labels may be used.

This is the paper's central mechanism made concrete. Given only (x_f, f) the
impedance law f = K*e + D*e_dot is unidentifiable -- two unknowns per axis, one
observed equation per axis, at every timestep. Bilateral teleoperation supplies
the leader pose as an independent physical measurement of the operator's
intended equilibrium, so e(t) = x_l(t) (-) x_f(t) is directly observed and
(K, D) become identifiable by regression, subject to the excitation conditions
the mask enforces.

Three details that are deliberate design decisions rather than implementation
detail, and are documented as such at their definitions below:

* Labels are produced at the 30 Hz action-chunk rate the policy head is
  supervised at, each from a TRAILING (causal) 300 ms window of the raw ~1 kHz
  signals -- not at 1 kHz, which would be ~33x the work for signal no policy
  ever sees.
* The regression is solved as a bounded NONLINEAR least-squares problem. The
  log-space prior regularizer is nonlinear in k_i, so there is no closed-form
  (ridge) solution, and the box constraint keeps labels interpretable as
  deployable impedance targets.
* Two distinct contact thresholds are used: a firm threshold for fitting the
  contact plane's geometry, and a separate lower sustained-contact threshold for
  the mask's SNR eligibility. Conflating them measurably degrades the frame fit.

Masked-out timesteps are excluded from every downstream fit. They are never
imputed.

Extracted verbatim from the reference implementation that produced the paper's
numbers; the argparse.Namespace the original threaded through has been replaced
by the equivalent frozen ExtractionConfig dataclass, and the plotting/CLI layers
now live alongside in the data_extraction/ directory.
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy.optimize import least_squares

from .geometry import (
    AXIS_NAMES,
    FORCE_AXIS_NAMES,
    compute_pose_error,
    fit_contact_frame,
    numerically_differentiate,
    rotate_force_to_contact_frame,
)

__all__ = [
    "AXIS_NAMES",
    "FORCE_AXIS_NAMES",
    "K_MIN",
    "K_MAX",
    "DEFAULT_SIGMA_F",
    "DEFAULT_SIGMA_E",
    "ExtractionConfig",
    "load_demo_csv",
    "extract_axis",
    "nearest_sample_indices",
    "compute_mask",
    "extract_demo",
]

# Bounds are the controller's realizable stiffness range (50-1500 N/m
# translational, 5-100 N*m/rad rotational). No controller executes these labels
# during extraction, but the bound is retained regardless: it is what keeps the
# extracted labels physically interpretable as DEPLOYABLE impedance targets.
K_MIN = np.array([50.0, 50.0, 50.0, 5.0, 5.0, 5.0])
K_MAX = np.array([1500.0, 1500.0, 1500.0, 100.0, 100.0, 100.0])

# Measured per-axis sensorless-wrench noise floor (N, N, N, Nm, Nm, Nm), from
# free-space residuals on held-out sweep sessions. Consumed by mask condition
# (iii). Override when re-measured on different hardware.
DEFAULT_SIGMA_F = np.array([0.726, 0.418, 0.461, 0.275, 0.434, 0.155])

# NOT a measured quantity: encoder-measured pose error is accurate to ~0.01 mm,
# so a small fixed constant per axis TYPE is used rather than a fit value. The
# rotational floor is an assumed order-of-magnitude default.
DEFAULT_SIGMA_E = np.array([1e-5, 1e-5, 1e-5, 1e-4, 1e-4, 1e-4])


@dataclass(frozen=True)
class ExtractionConfig:
    """Extraction hyperparameters. Defaults are the values used for the paper.

    Every field that is a tuned knob rather than a measured constant says so, so
    a reader can tell which numbers carry physical meaning and which are choices.
    """

    #: Trailing regression window length (s), at the native control rate.
    window_sec: float = 0.3
    #: Label output rate (Hz) -- the policy's action-chunk rate.
    output_rate_hz: float = 30.0
    #: Log-space prior regularization weight. Tunable, not a measured constant.
    lam: float = 0.1
    #: Loose sanity bound on fitted damping. Not a physical constant.
    d_max: float = 500.0
    #: Max Gram-matrix condition number for mask condition (i).
    kappa_max: float = 1e3
    #: N. Mask condition (iv)'s sustained-contact / SNR-eligibility threshold.
    contact_force_threshold: float = 2.0
    #: N. HIGHER firm-contact threshold, used ONLY to fit the contact plane's
    #: geometry -- light/transitional contact is genuinely not planar. See
    #: geometry.fit_contact_frame.
    frame_fit_force_threshold: float = 8.0
    #: Fraction of a window that must be in contact to count as sustained.
    contact_fraction_required: float = 0.8
    #: Max smallest/largest SVD singular-value ratio for the fitted contact
    #: plane to be trusted; a fit above this is rejected, not accepted.
    max_planarity_ratio: float = 0.15


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
        demo: Dict[str, np.ndarray], args: ExtractionConfig, sigma_f: np.ndarray) -> Dict:
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

