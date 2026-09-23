"""Sensorless wrench estimation: the bias model and the noise floor.

The noise floor these produce is what the identifiability mask thresholds
against, so an over-optimistic floor would silently admit labels that are really
noise. Two properties therefore matter more than raw accuracy: the session-level
split must not leak, and the floor must never be better than the raw measurement
on an axis where the model failed to help.
"""

import numpy as np
import pytest

from compliance_vla.wrench import (
    FEATURE_COLS,
    TARGET_COLS,
    BiasModelConfig,
    RFFRidgeBiasModel,
    per_axis_noise_floor,
    rms_per_axis,
    session_split,
    synthetic_dataset,
)


@pytest.fixture(scope="module")
def dataset():
    return synthetic_dataset(seed=0)


def test_feature_and_target_columns_match_the_documented_dimensions():
    assert len(FEATURE_COLS) == 14   # q (7) and q_dot (7)
    assert len(TARGET_COLS) == 6     # 6-DoF wrench


def test_model_fits_a_known_smooth_function_better_than_the_mean(dataset):
    X, Y, sessions = dataset
    train, val = session_split(sessions, val_fraction=0.3, seed=0)

    model = RFFRidgeBiasModel(n_features=512, length_scale=3.0, ridge_lambda=1e-2, seed=0)
    model.fit(X[train], Y[train])

    baseline = rms_per_axis(Y[val] - Y[train].mean(axis=0, keepdims=True))
    fitted = rms_per_axis(Y[val] - model.predict(X[val]))
    ratio = baseline / np.maximum(fitted, 1e-9)

    # Every axis must improve, and the mean improvement must be substantial. The
    # per-axis bar is deliberately only "better than the mean predictor": the
    # synthetic axes span a range of signal-to-noise ratios, as the real wrench
    # axes do, so requiring a uniform large margin on every axis would be a test
    # of the data, not of the regression.
    assert np.all(ratio > 1.0), f"per-axis improvement ratios {ratio}"
    assert ratio.mean() >= 2.0, f"mean improvement {ratio.mean():.2f}x"


def test_session_split_never_puts_one_session_on_both_sides(dataset):
    """A row-level split would leak: a slow sweep's neighbouring rows are nearly
    identical, so the same motion would appear in train and validation."""
    _, _, sessions = dataset
    train, val = session_split(sessions, val_fraction=0.3, seed=0)

    assert not np.any(train & val), "a row landed in both splits"
    assert np.all(train | val), "a row landed in neither split"
    assert set(np.unique(sessions[train])).isdisjoint(set(np.unique(sessions[val])))
    assert val.sum() > 0 and train.sum() > 0


def test_session_split_always_holds_out_at_least_one_session(dataset):
    _, _, sessions = dataset
    _, val = session_split(sessions, val_fraction=0.0, seed=0)
    assert len(np.unique(sessions[val])) >= 1


def test_model_is_deterministic_for_a_fixed_seed(dataset):
    X, Y, _ = dataset
    predictions = []
    for _ in range(2):
        model = RFFRidgeBiasModel(n_features=128, length_scale=3.0, ridge_lambda=1e-2, seed=7)
        model.fit(X[:500], Y[:500])
        predictions.append(model.predict(X[500:600]))
    np.testing.assert_array_equal(predictions[0], predictions[1])


def test_different_seeds_give_different_random_features(dataset):
    """The random projection must actually depend on the seed, or the
    'random' in random Fourier features is doing nothing."""
    X, Y, _ = dataset
    models = []
    for seed in (0, 1):
        model = RFFRidgeBiasModel(n_features=128, length_scale=3.0, ridge_lambda=1e-2, seed=seed)
        model.fit(X[:500], Y[:500])
        models.append(model)
    assert not np.allclose(models[0].omega_, models[1].omega_)


def test_stronger_ridge_shrinks_the_weights(dataset):
    """The ridge penalty must bind in the usual direction."""
    X, Y, _ = dataset
    norms = []
    for ridge_lambda in (1e-4, 1e2):
        model = RFFRidgeBiasModel(
            n_features=128, length_scale=3.0, ridge_lambda=ridge_lambda, seed=0)
        model.fit(X[:800], Y[:800])
        norms.append(np.linalg.norm(model.weights_))
    assert norms[1] < norms[0]


def test_save_and_load_round_trips_exactly(dataset, tmp_path):
    """A model reloaded on the rig must predict what it predicted when fitted."""
    X, Y, _ = dataset
    model = RFFRidgeBiasModel(n_features=128, length_scale=3.0, ridge_lambda=1e-2, seed=0)
    model.fit(X[:800], Y[:800])

    path = tmp_path / "bias_model.npz"
    model.save(path)
    reloaded = RFFRidgeBiasModel.load(path)

    np.testing.assert_array_equal(model.predict(X[800:900]), reloaded.predict(X[800:900]))
    assert reloaded.n_features == model.n_features
    assert reloaded.length_scale == model.length_scale


def test_constant_feature_columns_do_not_produce_nan():
    """A joint that never moves during a sweep has zero variance; standardizing
    by it must not divide by zero."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 14))
    X[:, 3] = 1.234                     # a frozen joint
    Y = rng.normal(size=(300, 6))

    model = RFFRidgeBiasModel(n_features=64, seed=0)
    model.fit(X, Y)
    assert np.all(np.isfinite(model.predict(X)))


def test_noise_floor_takes_the_better_of_raw_and_corrected():
    """sigma_f = min(RMS_raw, RMS_corrected), elementwise.

    This is the rule that stops a bias model which failed on some axis from
    making that axis's floor look better than it really is.
    """
    n = 1000
    rng = np.random.default_rng(0)
    raw = rng.normal(0.0, 1.0, size=(n, 6))
    corrected = raw.copy()
    corrected[:, :3] *= 0.25            # model helped on the first three axes
    corrected[:, 3:] *= 4.0             # and hurt on the last three

    floor = per_axis_noise_floor(raw, corrected)
    raw_rms, corrected_rms = rms_per_axis(raw), rms_per_axis(corrected)

    np.testing.assert_allclose(floor[:3], corrected_rms[:3])
    np.testing.assert_allclose(floor[3:], raw_rms[3:])
    assert np.all(floor <= raw_rms + 1e-12), "the floor must never exceed the raw estimate"


def test_noise_floor_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="same shape"):
        per_axis_noise_floor(np.zeros((10, 6)), np.zeros((10, 3)))


def test_bias_model_config_defaults_are_the_documented_ones():
    cfg = BiasModelConfig()
    assert cfg.n_features == 512
    assert cfg.val_fraction == 0.3
