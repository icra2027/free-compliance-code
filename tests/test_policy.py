"""The compliance-output policy's losses and force-history encoder.

These are the parts of the policy that can be exercised on tiny tensors without
a checkpoint, a dataset or a GPU. The properties they pin down are the ones the
method's claims actually rest on:

* masked entries are EXCLUDED from the loss, never imputed -- the same rule the
  extraction mask enforces, carried through to training;
* a batch with no identifiable timesteps contributes zero loss and a zero
  gradient, rather than a NaN that would poison the whole update;
* log K is supervised with a Huber shape and x_eq with an L1 shape, weighted by
  lambda, exactly as specified;
* force dropout is an augmentation, so it must be active in training and a
  strict no-op at evaluation time.

The whole module is skipped without torch, so the numpy-only core can still be
tested on a machine that has no deep-learning stack. The lerobot-backed policy
classes are not tested here: they need the pretrained SmolVLA tower, which is
out of scope for a unit test.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="policy code requires torch")

from compliance_vla.policy.force_encoder import (  # noqa: E402
    ForceHistoryEncoder,
    force_dropout,
    resample_to_n_samples,
    wrench_bias_injection,
)
from compliance_vla.policy.losses import (  # noqa: E402
    compliance_loss,
    hybrid_loss,
    masked_huber_loss,
    masked_l1_loss,
    position_only_loss,
)

BATCH, HORIZON = 4, 32


# --------------------------------------------------------------------------
# Masked losses
# --------------------------------------------------------------------------

def test_masked_huber_ignores_masked_out_entries_entirely():
    """Changing a masked-out residual must not change the loss at all.

    This is "excluded, never imputed" stated as a test: if masked entries were
    filled with zeros and averaged in, the loss would move here.
    """
    residual = torch.zeros(2, 4)
    mask = torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0]], dtype=torch.bool)
    residual[mask] = 0.5

    before = masked_huber_loss(residual, mask)
    residual = residual.clone()
    residual[~mask] = 1000.0
    after = masked_huber_loss(residual, mask)

    assert before.item() == pytest.approx(after.item())


def test_masked_huber_averages_over_masked_entries_only():
    residual = torch.tensor([[0.5, 0.5, 9.0, 9.0]])
    mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.bool)
    # Huber with delta=1 is quadratic below the knee: 0.5 * 0.5^2 = 0.125.
    assert masked_huber_loss(residual, mask).item() == pytest.approx(0.125)


def test_masked_l1_averages_over_masked_entries_only():
    residual = torch.tensor([[2.0, 4.0, 100.0]])
    mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)
    assert masked_l1_loss(residual, mask).item() == pytest.approx(3.0)


def test_huber_is_less_sensitive_to_outliers_than_l1():
    """log K is given a Huber shape deliberately, for robustness to outliers."""
    mask = torch.ones(1, 2, dtype=torch.bool)
    small = torch.tensor([[0.1, 0.1]])
    large = torch.tensor([[50.0, 50.0]])

    huber_growth = masked_huber_loss(large, mask) / masked_huber_loss(small, mask)
    l1_growth = masked_l1_loss(large, mask) / masked_l1_loss(small, mask)
    assert huber_growth < l1_growth ** 2


@pytest.mark.parametrize("loss_fn", [masked_huber_loss, masked_l1_loss])
def test_an_all_false_mask_gives_zero_loss_and_a_finite_gradient(loss_fn):
    """A batch with no identifiable timesteps must not poison the update.

    Averaging over an empty mask would divide by zero and propagate NaN into
    every other term of the loss.
    """
    residual = torch.randn(3, 5, requires_grad=True)
    mask = torch.zeros(3, 5, dtype=torch.bool)

    loss = loss_fn(residual, mask)
    loss.backward()

    assert loss.item() == 0.0
    assert torch.isfinite(residual.grad).all()
    assert torch.all(residual.grad == 0.0)


def test_masks_may_be_float_as_well_as_bool():
    residual = torch.tensor([[1.0, 5.0]])
    as_bool = masked_huber_loss(residual, torch.tensor([[True, False]]))
    as_float = masked_huber_loss(residual, torch.tensor([[1.0, 0.0]]))
    assert as_bool.item() == pytest.approx(as_float.item())


# --------------------------------------------------------------------------
# Combined training losses
# --------------------------------------------------------------------------

def _compliance_inputs(seed=0):
    torch.manual_seed(seed)
    u_t = torch.randn(BATCH, HORIZON, 13)
    v_t = torch.randn(BATCH, HORIZON, 13)
    x_eq_mask = torch.ones(BATCH, HORIZON, 6, dtype=torch.bool)
    log_k_mask = torch.rand(BATCH, HORIZON, 6) > 0.3
    gripper_mask = torch.zeros(BATCH, HORIZON, 1, dtype=torch.bool)
    return u_t, v_t, x_eq_mask, log_k_mask, gripper_mask


def test_compliance_loss_splits_the_13_dims_as_specified():
    """[x_eq(6), log_k(6), gripper(1)] -- a mis-slice here would silently
    supervise the wrong channels with the wrong loss shape."""
    u_t, v_t, x_eq_mask, log_k_mask, gripper_mask = _compliance_inputs()
    total, parts = compliance_loss(u_t, v_t, x_eq_mask, log_k_mask, gripper_mask, lam=1.0)

    residual = u_t - v_t
    assert parts["loss_x_eq"] == pytest.approx(
        masked_l1_loss(residual[..., 0:6], x_eq_mask).item())
    assert parts["loss_log_k"] == pytest.approx(
        masked_huber_loss(residual[..., 6:12], log_k_mask).item())
    assert parts["loss_gripper"] == pytest.approx(
        masked_l1_loss(residual[..., 12:13], gripper_mask).item())
    assert total.item() == pytest.approx(parts["loss"])


def test_lambda_scales_only_the_log_k_term():
    """lambda is the tuned knob on compliance supervision; it must not move the
    pose term, or tuning it would silently retune pose tracking too."""
    args = _compliance_inputs()
    _, low = compliance_loss(*args, lam=0.1)
    _, high = compliance_loss(*args, lam=10.0)

    assert low["loss_x_eq"] == pytest.approx(high["loss_x_eq"])
    assert low["loss_gripper"] == pytest.approx(high["loss_gripper"])
    assert low["loss_log_k"] == pytest.approx(high["loss_log_k"])
    assert high["loss"] > low["loss"]


def test_compliance_loss_is_differentiable():
    u_t, v_t, x_eq_mask, log_k_mask, gripper_mask = _compliance_inputs()
    v_t = v_t.clone().requires_grad_(True)

    total, _ = compliance_loss(u_t, v_t, x_eq_mask, log_k_mask, gripper_mask, lam=1.0)
    total.backward()

    assert torch.isfinite(v_t.grad).all()
    # The log_k channels must receive gradient only where the mask allows it.
    masked_out = ~log_k_mask
    assert torch.all(v_t.grad[..., 6:12][masked_out] == 0.0)


def test_position_only_loss_has_no_compliance_term():
    """The B0/B2 baselines have no compliance head, so their loss must not
    contain one -- otherwise the comparison the paper rests on is not a
    comparison of output parameterizations."""
    torch.manual_seed(0)
    u_t, v_t = torch.randn(BATCH, HORIZON, 7), torch.randn(BATCH, HORIZON, 7)
    x_eq_mask = torch.ones(BATCH, HORIZON, 6, dtype=torch.bool)
    gripper_mask = torch.zeros(BATCH, HORIZON, 1, dtype=torch.bool)

    total, parts = position_only_loss(u_t, v_t, x_eq_mask, gripper_mask)

    assert "loss_log_k" not in parts
    assert total.item() == pytest.approx(parts["loss_x_eq"] + parts["loss_gripper"])


def test_hybrid_loss_uses_a_direct_regression_residual_for_log_k():
    """B3 decodes log K through a separate regression head, so its log_k
    residual is (prediction - target), not a flow-matching noise residual."""
    torch.manual_seed(0)
    u_t = torch.randn(BATCH, HORIZON, 7)
    v_t = torch.randn(BATCH, HORIZON, 7)
    x_eq_mask = torch.ones(BATCH, HORIZON, 6, dtype=torch.bool)
    gripper_mask = torch.zeros(BATCH, HORIZON, 1, dtype=torch.bool)
    log_k_pred = torch.randn(BATCH, HORIZON, 6)
    log_k_target = torch.randn(BATCH, HORIZON, 6)
    log_k_mask = torch.ones(BATCH, HORIZON, 6, dtype=torch.bool)

    _, parts = hybrid_loss(
        u_t, v_t, x_eq_mask, gripper_mask, log_k_pred, log_k_target, log_k_mask, lam=1.0)

    assert parts["loss_log_k"] == pytest.approx(
        masked_huber_loss(log_k_pred - log_k_target, log_k_mask).item())


# --------------------------------------------------------------------------
# Force-history encoder
# --------------------------------------------------------------------------

def test_resampling_produces_the_architecture_s_fixed_input_width():
    times = np.linspace(0.0, 1.0, 31)
    values = np.tile(np.arange(6, dtype=np.float64), (31, 1))

    out = resample_to_n_samples(times, values, t_end=1.0, window_sec=0.5, n_samples=20)

    assert out.shape == (20, 6)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, np.tile(np.arange(6), (20, 1)), atol=1e-6)


def test_resampling_interpolates_a_known_ramp():
    times = np.linspace(0.0, 1.0, 101)
    values = times.reshape(-1, 1)

    out = resample_to_n_samples(times, values, t_end=1.0, window_sec=1.0, n_samples=11)

    np.testing.assert_allclose(out[:, 0], np.linspace(0.0, 1.0, 11), atol=1e-6)


def test_resampling_pads_by_holding_the_earliest_sample_at_episode_start():
    """Before 500 ms of history exists, the window is padded with real history,
    not with an imputed sensor reading."""
    times = np.array([0.0, 0.01])
    values = np.array([[1.0] * 6, [2.0] * 6])

    out = resample_to_n_samples(times, values, t_end=0.01, window_sec=0.5, n_samples=20)

    assert out.shape == (20, 6)
    assert np.all(out >= 1.0) and np.all(out <= 2.0)
    np.testing.assert_allclose(out[0], 1.0)


def test_resampling_with_a_single_sample_repeats_it():
    out = resample_to_n_samples(
        np.array([0.0]), np.ones((1, 6)), t_end=0.0, window_sec=0.5, n_samples=20)
    assert out.shape == (20, 6)
    np.testing.assert_allclose(out, 1.0)


def test_force_dropout_is_a_no_op_at_evaluation_time():
    """An augmentation that fired at inference would change the policy's
    behaviour at exactly the moment the force-freeze ablation measures it."""
    hist = torch.randn(8, 20, 6)
    torch.testing.assert_close(force_dropout(hist, p=1.0, training=False), hist)


def test_force_dropout_zeroes_whole_examples_when_training():
    """Dropout is per-EXAMPLE: it removes the force channel entirely for that
    sample, which is what defends against modal masking."""
    hist = torch.randn(64, 20, 6)
    dropped = force_dropout(hist, p=1.0, training=True)
    assert torch.all(dropped == 0.0)

    kept = force_dropout(hist, p=0.0, training=True)
    torch.testing.assert_close(kept, hist)


def test_force_dropout_removes_examples_whole_not_piecemeal():
    torch.manual_seed(0)
    hist = torch.randn(256, 20, 6).abs() + 1.0
    out = force_dropout(hist, p=0.5, training=True)

    per_example_zero = (out == 0.0).flatten(1).all(dim=1)
    per_example_any_zero = (out == 0.0).flatten(1).any(dim=1)
    assert torch.equal(per_example_zero, per_example_any_zero), \
        "an example was partially zeroed; dropout must drop the whole window"
    assert 0 < per_example_zero.sum() < 256


def test_wrench_bias_injection_shifts_by_a_bounded_constant():
    torch.manual_seed(0)
    hist = torch.zeros(32, 20, 6)
    out = wrench_bias_injection(hist, bias_range_n=1.0, training=True)

    assert out.shape == hist.shape
    assert out.abs().max() <= 1.0 + 1e-6
    # A BIAS is constant across the window for a given example and channel.
    assert torch.allclose(out.std(dim=1), torch.zeros(32, 6), atol=1e-6)


def test_wrench_bias_injection_is_a_no_op_at_evaluation_time():
    hist = torch.randn(8, 20, 6)
    torch.testing.assert_close(
        wrench_bias_injection(hist, bias_range_n=1.0, training=False), hist)


def test_force_history_encoder_maps_a_window_to_one_token():
    encoder = ForceHistoryEncoder(in_channels=6, hidden_dim=256)
    out = encoder(torch.randn(BATCH, 20, 6))

    assert out.shape == (BATCH, 256)
    assert torch.isfinite(out).all()


def test_force_history_encoder_is_differentiable():
    encoder = ForceHistoryEncoder(in_channels=6, hidden_dim=64)
    hist = torch.randn(2, 20, 6, requires_grad=True)

    encoder(hist).sum().backward()

    assert hist.grad is not None and torch.isfinite(hist.grad).all()
    assert hist.grad.abs().sum() > 0, "the force channel receives no gradient at all"


def test_force_history_encoder_responds_to_its_input():
    """A constant output would mean the force channel is architecturally
    ignored -- the modal-masking failure the augmentations exist to prevent."""
    torch.manual_seed(0)
    encoder = ForceHistoryEncoder(in_channels=6, hidden_dim=64).eval()
    with torch.no_grad():
        a = encoder(torch.zeros(1, 20, 6))
        b = encoder(torch.ones(1, 20, 6) * 5.0)
    assert not torch.allclose(a, b)
