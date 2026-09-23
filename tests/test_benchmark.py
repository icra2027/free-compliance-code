"""The offline compliance-prediction benchmark.

The benchmark's job is to answer one question honestly: do the extracted labels
carry learnable signal, or are they noise? So the tests here care less about
absolute accuracy than about the benchmark being incapable of flattering itself
-- it must report a loss as a loss, must not let test data influence model
selection, and must beat the constant baseline only when there is really
something to learn.
"""

import numpy as np
import pytest

from compliance_vla.benchmark import (
    MANNERS,
    REFERENTS,
    AxisResult,
    aggregate,
    constant_predict,
    evaluate_axis,
    featurize,
    nearest_neighbour_predict,
    rmse,
    select_hyperparams,
)


def _dataset(n, noise, seed, learnable=True):
    """Features and a log-K target that is (or is not) a function of them."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 20))
    if learnable:
        y = 1.5 * X[:, 0] - 0.8 * X[:, 1] + 0.3 * X[:, 7] + rng.normal(0, noise, n)
    else:
        y = rng.normal(0, 1.0, n)
    return X, y


def test_featurize_one_hots_manner_and_referent():
    m, r = featurize("firmly", "blue")
    assert m.tolist() == [0.0, 1.0]
    assert r.tolist() == [0.0, 1.0, 0.0, 0.0]
    assert len(m) == len(MANNERS) and len(r) == len(REFERENTS)


def test_featurize_returns_all_zeros_for_an_unknown_value():
    """An out-of-vocabulary instruction must not crash the benchmark; the
    episode still contributes its proprioception features."""
    m, r = featurize("gently", "purple")
    assert not m.any() and not r.any()


def test_rmse_is_zero_for_a_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0])
    assert rmse(y, y) == 0.0
    assert rmse(np.zeros(3), np.array([3.0, 4.0, 0.0])) == pytest.approx(np.sqrt(25 / 3))


def test_constant_predictor_ignores_its_input():
    y_train = np.array([1.0, 2.0, 3.0, 4.0])
    pred = constant_predict(y_train, n_test=5)
    assert pred.shape == (5,)
    assert np.allclose(pred, 2.5)


def test_nearest_neighbour_reproduces_training_labels_exactly():
    """Queried at its own training points, 1-NN must return those labels."""
    X, y = _dataset(120, noise=0.1, seed=0)
    pred = nearest_neighbour_predict(X, y, X)
    np.testing.assert_allclose(pred, y)


def test_nearest_neighbour_standardizes_before_measuring_distance():
    """Without standardization a feature with a large scale would dominate the
    metric, so the same data on a rescaled axis must give the same neighbours."""
    X, y = _dataset(80, noise=0.1, seed=1)
    X_test, _ = _dataset(20, noise=0.1, seed=2)

    scaled = X.copy()
    scaled[:, 5] *= 1000.0
    scaled_test = X_test.copy()
    scaled_test[:, 5] *= 1000.0

    np.testing.assert_allclose(
        nearest_neighbour_predict(X, y, X_test),
        nearest_neighbour_predict(scaled, y, scaled_test))


def test_learned_predictor_beats_the_constant_baseline_on_learnable_data():
    X_tr, y_tr = _dataset(600, noise=0.2, seed=0)
    X_va, y_va = _dataset(200, noise=0.2, seed=1)
    X_te, y_te = _dataset(200, noise=0.2, seed=2)

    result = evaluate_axis("fx", (X_tr, y_tr), (X_va, y_va), (X_te, y_te))

    assert result.status == "ok"
    assert result.learned_improvement > 1.0, (
        f"learned {result.learned_rmse:.3f} vs constant {result.constant_rmse:.3f}")


def test_learned_predictor_does_not_beat_the_baseline_on_pure_noise():
    """The benchmark must be able to FAIL.

    If the target is independent of the features, no model should be able to
    beat predicting the mean. A benchmark that reported an improvement here
    would be incapable of rejecting the null hypothesis it exists to test.
    """
    X_tr, y_tr = _dataset(600, noise=0.0, seed=0, learnable=False)
    X_va, y_va = _dataset(200, noise=0.0, seed=1, learnable=False)
    X_te, y_te = _dataset(200, noise=0.0, seed=2, learnable=False)

    result = evaluate_axis("fx", (X_tr, y_tr), (X_va, y_va), (X_te, y_te))

    assert result.status == "ok"
    assert result.learned_improvement < 1.05, (
        f"a model beat the baseline by {result.learned_improvement:.3f}x on noise")


def test_hyperparameter_selection_never_sees_the_test_split():
    """Selection on validation is what makes the reported test number clean.

    Replacing the test split entirely must not change the selected
    hyperparameters; if it did, test data would be leaking into selection.
    """
    X_tr, y_tr = _dataset(400, noise=0.2, seed=0)
    X_va, y_va = _dataset(150, noise=0.2, seed=1)

    ls_a, rl_a, _ = select_hyperparams(X_tr, y_tr, X_va, y_va)
    ls_b, rl_b, _ = select_hyperparams(X_tr, y_tr, X_va, y_va)
    assert (ls_a, rl_a) == (ls_b, rl_b)

    result_a = evaluate_axis("fx", (X_tr, y_tr), (X_va, y_va), _dataset(200, 0.2, 2))
    result_b = evaluate_axis("fx", (X_tr, y_tr), (X_va, y_va), _dataset(200, 0.9, 99))
    assert (result_a.length_scale, result_a.ridge_lambda) == \
           (result_b.length_scale, result_b.ridge_lambda)


def test_thin_validation_falls_back_to_defaults_and_says_so():
    X_tr, y_tr = _dataset(200, noise=0.2, seed=0)
    X_te, y_te = _dataset(100, noise=0.2, seed=2)
    thin_val = (np.zeros((2, 20)), np.zeros(2))

    result = evaluate_axis("fx", (X_tr, y_tr), thin_val, (X_te, y_te))

    assert result.status == "ok"
    assert result.val_rmse is None, "no validation RMSE should be claimed from 2 rows"
    assert (result.length_scale, result.ridge_lambda) == (3.0, 1e-2)


def test_an_axis_with_too_little_data_is_reported_not_scored():
    tiny = (np.zeros((3, 20)), np.zeros(3))
    result = evaluate_axis("tz", tiny, tiny, tiny)

    assert result.status == "insufficient_data"
    assert result.constant_rmse is None
    assert result.learned_improvement is None


def test_aggregate_reports_the_per_axis_win_count_alongside_the_mean():
    """A favourable aggregate can hide a mixed per-axis picture, so both are
    reported -- which is exactly the situation on the paper's own dataset."""
    results = [
        AxisResult("fx", "ok", 100, 50, 50, constant_rmse=1.0, nn_rmse=1.2, learned_rmse=0.8),
        AxisResult("fy", "ok", 100, 50, 50, constant_rmse=1.0, nn_rmse=1.3, learned_rmse=1.1),
        AxisResult("fz", "insufficient_data", 3, 0, 0),
    ]
    summary = aggregate(results)

    assert summary["n_axes_scored"] == 2
    assert summary["n_axes_learned_beats_constant"] == 1
    assert summary["n_axes_nn_beats_constant"] == 0
    assert summary["learned_improvement"] == pytest.approx(1.0 / 0.95)


def test_aggregate_handles_the_case_where_nothing_could_be_scored():
    summary = aggregate([AxisResult("fx", "insufficient_data", 1, 0, 0)])
    assert summary == {"n_axes_scored": 0}
