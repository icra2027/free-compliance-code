"""Offline compliance-prediction benchmark: are the extracted labels learnable?

This is the paper's label-quality check, and it is deliberately a lower bar than
beating a trained policy: it asks only whether the extracted labels carry real,
learnable signal at all. That is the necessary precondition everything
downstream depends on -- if the labels were noise, no architecture could rescue
them.

Task. Per axis, per identifiable (masked) timestep, predict log K(t) in the
contact frame from a 20-dimensional proprioception + language feature vector:
[q (7), q_dot (7), manner one-hot (2), referent one-hot (4)]. Force and wrench
features are deliberately EXCLUDED: the point is whether compliance is
predictable from the kind of context a policy conditions on, not from other
force channels, which would make the check partly circular. Non-identifiable
timesteps are excluded entirely, never imputed.

Models. `constant` is the per-axis training-split mean of log k, which ignores
its input entirely and is the trivial baseline that must be beaten.
`nearest-neighbour` is 1-NN in standardized feature space. `learned` is the same
RFF ridge regressor used for the wrench bias model, reused unmodified for a
second, distinct regression target.

Splits are session-level: hyperparameters are selected on validation and never
on test, and test is a held-out session from the OTHER operator, so the reported
number is a cross-operator generalization number.

Metric is RMSE in log space, per axis, reported alongside
improvement_ratio = constant_rmse / model_rmse, so a value above 1.0 means the
model beat the trivial baseline.

The 1-NN search here is plain numpy rather than scikit-learn, so that the
reproducible core of this release depends only on numpy and scipy. Brute-force
1-NN is exact, so this is the same predictor, not an approximation of it.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .wrench import RFFRidgeBiasModel

__all__ = [
    "MANNERS",
    "REFERENTS",
    "LENGTH_SCALE_GRID",
    "RIDGE_LAMBDA_GRID",
    "AxisResult",
    "featurize",
    "rmse",
    "constant_predict",
    "nearest_neighbour_predict",
    "select_hyperparams",
    "evaluate_axis",
    "aggregate",
]

MANNERS = ["normally", "firmly"]
REFERENTS = ["red", "blue", "green", "black"]

#: Hyperparameter grid searched on the VALIDATION split only.
LENGTH_SCALE_GRID = [1, 3, 10, 30, 100, 300]
RIDGE_LAMBDA_GRID = [1e-3, 1e-2, 1e-1, 1.0]

#: Number of random features used by the benchmark's learned predictor. Smaller
#: than the bias model's 512: this target has far fewer training rows per axis.
BENCHMARK_N_FEATURES = 256

#: An axis with fewer than this many train or test rows is reported as
#: insufficient_data rather than scored on a handful of points.
MIN_SAMPLES = 10


def featurize(manner: str, referent: str) -> tuple[np.ndarray, np.ndarray]:
    """One-hot encodes the language instruction's manner and referent.

    An unrecognized value yields an all-zero block rather than raising, so an
    episode carrying an out-of-vocabulary instruction contributes its
    proprioception features instead of dropping out of the benchmark entirely.
    """
    m = np.zeros(len(MANNERS))
    if manner in MANNERS:
        m[MANNERS.index(manner)] = 1.0
    r = np.zeros(len(REFERENTS))
    if referent in REFERENTS:
        r[REFERENTS.index(referent)] = 1.0
    return m, r


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def constant_predict(y_train: np.ndarray, n_test: int) -> np.ndarray:
    """The trivial baseline: the training-split mean, repeated. Ignores input."""
    return np.full(n_test, float(np.mean(y_train)))


def _standardize(X_train: np.ndarray, X: np.ndarray) -> np.ndarray:
    x_mean, x_std = X_train.mean(axis=0), X_train.std(axis=0)
    x_std = np.where(x_std < 1e-8, 1.0, x_std)
    return (X - x_mean) / x_std


def nearest_neighbour_predict(
        X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """1-NN in standardized feature space; the label is copied from the nearest
    training point. Brute force, which is exact for 1-NN."""
    train_s = _standardize(X_train, X_train)
    test_s = _standardize(X_train, X_test)
    # (n_test, n_train) squared distances; sqrt is monotone so argmin is unaffected.
    d2 = ((test_s[:, None, :] - train_s[None, :, :]) ** 2).sum(axis=2)
    return np.asarray(y_train)[np.argmin(d2, axis=1)]


def select_hyperparams(
        X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray) -> tuple[float, float, float]:
    """Grid-searches (length_scale, ridge_lambda) on the VALIDATION split.

    Returns (length_scale, ridge_lambda, val_rmse). Test data is never consulted
    here -- selection on validation and reporting on test is what makes the test
    number a clean held-out, cross-operator result.
    """
    best: tuple[float, float, float] | None = None
    for ls in LENGTH_SCALE_GRID:
        for rl in RIDGE_LAMBDA_GRID:
            model = RFFRidgeBiasModel(
                n_features=BENCHMARK_N_FEATURES, length_scale=ls, ridge_lambda=rl, seed=0)
            model.fit(X_train, np.asarray(y_train).reshape(-1, 1))
            score = rmse(model.predict(X_val).ravel(), y_val)
            if best is None or score < best[0]:
                best = (score, ls, rl)
    assert best is not None  # the grid is non-empty by construction
    return best[1], best[2], best[0]


@dataclass
class AxisResult:
    """Per-axis benchmark outcome. `status` is "ok" or "insufficient_data"."""

    axis: str
    status: str
    n_train: int
    n_val: int
    n_test: int
    constant_rmse: float | None = None
    nn_rmse: float | None = None
    learned_rmse: float | None = None
    length_scale: float | None = None
    ridge_lambda: float | None = None
    val_rmse: float | None = None

    @property
    def nn_improvement(self) -> float | None:
        """constant_rmse / nn_rmse. Above 1.0 means it beat the baseline."""
        return _ratio(self.constant_rmse, self.nn_rmse)

    @property
    def learned_improvement(self) -> float | None:
        """constant_rmse / learned_rmse. Above 1.0 means it beat the baseline."""
        return _ratio(self.constant_rmse, self.learned_rmse)


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    return float(numerator / max(denominator, 1e-9))


def evaluate_axis(
        axis: str,
        train: tuple[np.ndarray, np.ndarray],
        val: tuple[np.ndarray, np.ndarray],
        test: tuple[np.ndarray, np.ndarray],
        min_samples: int = MIN_SAMPLES) -> AxisResult:
    """Scores constant / nearest-neighbour / learned predictors on one axis.

    Each of `train`, `val`, `test` is an (X, y) pair, X of shape (n, 20) and y of
    shape (n,) holding log K for that axis at masked timesteps only.

    If validation is too thin to select hyperparameters, the library defaults are
    used and `length_scale`/`ridge_lambda` still report what was actually used --
    a thin split is disclosed, not silently papered over.
    """
    X_tr, y_tr = train
    X_va, y_va = val
    X_te, y_te = test
    n_tr, n_va, n_te = len(y_tr), len(y_va), len(y_te)

    if n_tr < min_samples or n_te < min_samples:
        return AxisResult(axis, "insufficient_data", n_tr, n_va, n_te)

    constant_rmse = rmse(constant_predict(y_tr, n_te), y_te)
    nn_rmse = rmse(nearest_neighbour_predict(X_tr, y_tr, X_te), y_te)

    if n_va >= min_samples:
        length_scale, ridge_lambda, val_rmse = select_hyperparams(X_tr, y_tr, X_va, y_va)
    else:
        length_scale, ridge_lambda, val_rmse = 3.0, 1e-2, None

    model = RFFRidgeBiasModel(
        n_features=BENCHMARK_N_FEATURES, length_scale=length_scale,
        ridge_lambda=ridge_lambda, seed=0)
    model.fit(X_tr, np.asarray(y_tr).reshape(-1, 1))
    learned_rmse = rmse(model.predict(X_te).ravel(), y_te)

    return AxisResult(
        axis, "ok", n_tr, n_va, n_te,
        constant_rmse=constant_rmse, nn_rmse=nn_rmse, learned_rmse=learned_rmse,
        length_scale=length_scale, ridge_lambda=ridge_lambda, val_rmse=val_rmse)


def aggregate(results: Sequence[AxisResult]) -> dict[str, object]:
    """Aggregates per-axis results the way the paper reports them.

    The aggregate improvement ratio is computed from the MEAN per-axis RMSE
    across scored axes, and the per-axis win counts are reported alongside it,
    because a favourable aggregate can hide a mixed per-axis picture -- which is
    exactly what happens on this dataset, and is reported rather than smoothed
    over.
    """
    scored = [r for r in results if r.status == "ok"]
    if not scored:
        return {"n_axes_scored": 0}

    const = float(np.mean([r.constant_rmse for r in scored]))
    nn = float(np.mean([r.nn_rmse for r in scored]))
    learned = float(np.mean([r.learned_rmse for r in scored]))
    return {
        "n_axes_scored": len(scored),
        "constant_rmse": const,
        "nn_rmse": nn,
        "learned_rmse": learned,
        "nn_improvement": _ratio(const, nn),
        "learned_improvement": _ratio(const, learned),
        "n_axes_learned_beats_constant": sum(
            1 for r in scored if r.learned_improvement is not None and r.learned_improvement > 1.0),
        "n_axes_nn_beats_constant": sum(
            1 for r in scored if r.nn_improvement is not None and r.nn_improvement > 1.0),
    }
