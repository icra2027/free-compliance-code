"""Sensorless wrench estimation: residual bias model and noise floor.

The extraction pipeline needs a contact wrench, and this work deliberately takes
it from the manipulator's own joint-torque sensing rather than a dedicated
force/torque or tactile sensor -- at training time and at inference time. That
makes the wrench estimate's residual bias, and the noise floor it leaves behind,
first-class quantities: the noise floor is what mask condition (iii) thresholds
against, so it directly decides which labels are usable.

Model: random-Fourier-feature (RFF) ridge regression per wrench axis. RFF ridge
regression is the finite-dimensional Monte-Carlo approximation of kernel ridge
regression with an RBF kernel -- an approximate GP posterior mean, obtained by a
single closed-form solve, with no iterative training loop and no autodiff.

Splits are by SESSION, never by row. A free-space sweep is slow point-to-point
motion, so consecutive rows within a session are highly correlated and a
row-level split would leak across the split boundary and overstate accuracy.
The same session-level convention is used everywhere else in this work.

Extracted verbatim from the reference implementation; the argparse.Namespace has
been replaced by the equivalent frozen BiasModelConfig, and per_axis_noise_floor
(previously computed by the reporting layer) is exposed as a function so it can
be tested directly.
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

__all__ = [
    "FEATURE_COLS",
    "TARGET_COLS",
    "BiasModelConfig",
    "RFFRidgeBiasModel",
    "load_sweep_csvs",
    "rms_per_axis",
    "session_split",
    "fit_and_report",
    "per_axis_noise_floor",
    "synthetic_dataset",
]

FEATURE_COLS = [f"q{i}" for i in range(1, 8)] + [f"dq{i}" for i in range(1, 8)]
TARGET_COLS = ["fx", "fy", "fz", "tx", "ty", "tz"]


@dataclass(frozen=True)
class BiasModelConfig:
    """Residual bias model hyperparameters. Defaults are the values used for the
    paper (D_f = 512, length scale 30, ridge penalty 0.1 after the real-data
    sweep; the 3.0/1e-2 below are the generic starting point the sweep began
    from, kept as the library default so a fresh rig does not silently inherit
    one specific robot's tuned values)."""

    #: Number of random Fourier features D_f.
    n_features: int = 512
    #: RBF length scale, in STANDARDIZED feature units. If a fit shows little
    #: improvement, sweep this before concluding the signal is noise.
    length_scale: float = 3.0
    #: Ridge penalty lambda.
    ridge_lambda: float = 1e-2
    #: Fraction of SESSIONS (never rows) held out for validation.
    val_fraction: float = 0.3
    #: RNG seed for the random projection and the session shuffle.
    seed: int = 0


def load_sweep_csvs(paths: List[Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (X, Y, session_ids). session_ids disambiguated across files by prefixing the
    file stem, so two sweep runs that both happen to use speed_factor 0.1 aren't merged into
    one session."""
    X_rows, Y_rows, sessions = [], [], []
    for path in paths:
        with path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                X_rows.append([float(row[c]) for c in FEATURE_COLS])
                Y_rows.append([float(row[c]) for c in TARGET_COLS])
                sessions.append(f"{path.stem}::{row['session_id']}")
    if not X_rows:
        raise ValueError(f"No rows loaded from {paths}")
    return np.array(X_rows), np.array(Y_rows), np.array(sessions)


class RFFRidgeBiasModel:
    """One random-Fourier-feature ridge regressor per output axis, sharing one feature
    standardization and one random projection across axes (only the ridge weights differ)."""

    def __init__(
            self, n_features: int = 512, length_scale: float = 3.0,
            ridge_lambda: float = 1e-2, seed: int = 0):
        self.n_features = n_features
        self.length_scale = length_scale
        self.ridge_lambda = ridge_lambda
        self.seed = seed
        self.x_mean_: np.ndarray = None
        self.x_std_: np.ndarray = None
        self.omega_: np.ndarray = None
        self.phase_: np.ndarray = None
        self.weights_: np.ndarray = None  # (n_features*2, n_targets)

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        return (X - self.x_mean_) / self.x_std_

    def _rff(self, Xs: np.ndarray) -> np.ndarray:
        proj = Xs @ self.omega_ + self.phase_
        # cos/sin features rather than cos-with-random-phase-only: halves variance for the
        # same feature count and avoids the (small) bias a random-phase-only estimator has.
        return np.concatenate(
            [np.cos(proj), np.sin(proj)], axis=1) * np.sqrt(1.0 / self.n_features)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        self.x_mean_ = X.mean(axis=0)
        self.x_std_ = X.std(axis=0)
        self.x_std_[self.x_std_ < 1e-8] = 1.0
        Xs = self._standardize(X)

        rng = np.random.default_rng(self.seed)
        n_dims = X.shape[1]
        # omega ~ N(0, 1/length_scale^2) is the standard RFF sampling for an RBF kernel.
        self.omega_ = rng.normal(0.0, 1.0 / self.length_scale, size=(n_dims, self.n_features))
        self.phase_ = np.zeros(self.n_features)  # unused with cos/sin pairing; kept for clarity

        Phi = self._rff(Xs)
        n_feat_total = Phi.shape[1]
        A = Phi.T @ Phi + self.ridge_lambda * np.eye(n_feat_total)
        b = Phi.T @ Y
        self.weights_ = np.linalg.solve(A, b)

    def predict(self, X: np.ndarray) -> np.ndarray:
        Phi = self._rff(self._standardize(X))
        return Phi @ self.weights_

    def save(self, path: Path) -> None:
        np.savez(
            path, x_mean=self.x_mean_, x_std=self.x_std_, omega=self.omega_,
            weights=self.weights_, n_features=self.n_features, length_scale=self.length_scale,
            ridge_lambda=self.ridge_lambda, feature_cols=np.array(FEATURE_COLS),
            target_cols=np.array(TARGET_COLS))

    @classmethod
    def load(cls, path: Path) -> "RFFRidgeBiasModel":
        data = np.load(path, allow_pickle=False)
        model = cls(
            n_features=int(data["n_features"]), length_scale=float(data["length_scale"]),
            ridge_lambda=float(data["ridge_lambda"]))
        model.x_mean_, model.x_std_ = data["x_mean"], data["x_std"]
        model.omega_, model.weights_ = data["omega"], data["weights"]
        model.phase_ = np.zeros(model.n_features)
        return model


def rms_per_axis(residual: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(residual ** 2, axis=0))


def session_split(
        sessions: np.ndarray, val_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    unique = np.unique(sessions)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_val = max(1, int(round(len(unique) * val_fraction)))
    val_sessions = set(unique[:n_val])
    val_mask = np.array([s in val_sessions for s in sessions])
    return ~val_mask, val_mask


def fit_and_report(
        X: np.ndarray, Y: np.ndarray, sessions: np.ndarray, args: BiasModelConfig) -> Dict:
    train_mask, val_mask = session_split(sessions, args.val_fraction, args.seed)
    X_train, Y_train = X[train_mask], Y[train_mask]
    X_val, Y_val = X[val_mask], Y[val_mask]

    baseline_rms = rms_per_axis(Y_val - Y_train.mean(axis=0, keepdims=True))

    model = RFFRidgeBiasModel(
        n_features=args.n_features, length_scale=args.length_scale,
        ridge_lambda=args.ridge_lambda, seed=args.seed)
    model.fit(X_train, Y_train)

    pred_val = model.predict(X_val)
    fitted_rms = rms_per_axis(Y_val - pred_val)

    report = {
        "num_train_samples": int(len(X_train)),
        "num_val_samples": int(len(X_val)),
        "num_train_sessions": int(len(np.unique(sessions[train_mask]))),
        "num_val_sessions": int(len(np.unique(sessions[val_mask]))),
        "target_cols": TARGET_COLS,
        "baseline_rms": baseline_rms.tolist(),
        "fitted_rms": fitted_rms.tolist(),
        "improvement_ratio": (baseline_rms / np.maximum(fitted_rms, 1e-9)).tolist(),
        "hyperparameters": {
            "n_features": args.n_features, "length_scale": args.length_scale,
            "ridge_lambda": args.ridge_lambda, "seed": args.seed,
        },
    }
    return report, model

def per_axis_noise_floor(
        residual_raw: np.ndarray, residual_bias_corrected: np.ndarray) -> np.ndarray:
    """Per-axis noise floor sigma_f, evaluated on HELD-OUT free-space sweep
    sessions:

        sigma_f_i = min(RMS_i^raw, RMS_i^bias-corrected)

    i.e. the bias-corrected estimate is adopted only on axes where it measurably
    improves over the raw estimate, and the raw value stands on the axes where it
    does not. Taking the elementwise minimum rather than always trusting the
    fitted model is the point: a bias model that fails to help on some axis must
    not be allowed to make that axis's floor look better than it is, because
    sigma_f is consumed directly by the identifiability mask and an optimistic
    floor would admit labels that are really noise.

    Both arguments are (n_samples, 6) residual arrays for the same held-out
    sessions. Returns (6,).
    """
    raw = np.asarray(residual_raw, dtype=np.float64)
    corrected = np.asarray(residual_bias_corrected, dtype=np.float64)
    if raw.shape != corrected.shape:
        raise ValueError(
            f"residual arrays must have the same shape, got {raw.shape} and {corrected.shape}")
    return np.minimum(rms_per_axis(raw), rms_per_axis(corrected))


def synthetic_dataset(n_sessions: int = 6, n_per_session: int = 800, seed: int = 0):
    """Generates data with a known smooth f_bias(q, q̇) plus noise, entirely offline -- lets
    the fit/validation pipeline (session split, RFF ridge, error reporting) be checked without
    any hardware or logged CSV, the same "verify the pipeline against synthetic data before the
    real run" approach used for the offline analysis scripts.

    Deliberately additive/low-order in each joint (a handful of single-joint sin/cos terms for
    the gravity-model-residual part, linear single-joint terms for the velocity/friction part)
    rather than a fully entangled random-projection function of all 14 dims at once -- this
    mirrors the actual physics (Coulomb/viscous friction is close to per-joint, and unmodeled
    gravity residual is smooth in each joint angle) and is the regime an additive/smooth kernel
    method is actually meant for. A test that instead demanded fitting an arbitrary 14D
    high-frequency function from ~1-2k free-space samples would fail for curse-of-dimensionality
    reasons that have nothing to do with whether this script's regression code is correct.
    """
    rng = np.random.default_rng(seed)
    X_rows, Y_rows, sessions = [], [], []

    def true_bias(x: np.ndarray) -> np.ndarray:
        q, dq = x[:, :7], x[:, 7:]
        fx = 0.5 * np.sin(q[:, 0]) + 0.3 * np.cos(q[:, 2]) + 0.4 * dq[:, 0]
        fy = 0.4 * np.sin(q[:, 1]) - 0.3 * np.sin(q[:, 4]) + 0.3 * dq[:, 1]
        fz = 0.6 * np.cos(q[:, 3]) + 0.2 * np.sin(q[:, 5]) - 0.4 * dq[:, 2]
        tx = 0.3 * np.sin(q[:, 4]) + 0.2 * dq[:, 4]
        ty = 0.25 * np.cos(q[:, 5]) - 0.2 * dq[:, 5]
        tz = 0.2 * np.sin(q[:, 6]) + 0.15 * dq[:, 6]
        return np.stack([fx, fy, fz, tx, ty, tz], axis=1)

    for s in range(n_sessions):
        q = rng.uniform(-1.0, 1.0, size=(n_per_session, 7))
        dq = rng.uniform(-0.3, 0.3, size=(n_per_session, 7))
        x = np.concatenate([q, dq], axis=1)
        y = true_bias(x) + rng.normal(0.0, 0.05, size=(n_per_session, 6))
        X_rows.append(x)
        Y_rows.append(y)
        sessions.append(np.full(n_per_session, f"synthetic::speed_{s}"))
    return np.concatenate(X_rows), np.concatenate(Y_rows), np.concatenate(sessions)
