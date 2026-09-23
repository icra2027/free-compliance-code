"""Day 11: force-history token -- last 500ms of 6-DoF wrench, downsampled to
20 samples, 1D-conv encoder, injected post-VLM (proposal §4.2, ForceVLA
finding).

Native data-rate caveat, documented rather than smoothed over: the LeRobot
dataset's wrench channel is already downsampled to 30Hz at build time (Day 10
note in run_extraction_on_dataset.py), i.e. ~15 native samples in a trailing
500ms window, not the 20 the proposal's architecture spec calls for (implying
a ~40Hz history). `resample_to_n_samples` linearly interpolates the native
~15-sample trailing window up to exactly 20 points so the conv encoder's
input shape matches the spec, but the underlying temporal resolution is still
native 30Hz -- interpolation adds no new information, it only satisfies the
architecture's fixed input width.
"""

import numpy as np
import torch
from torch import nn


def resample_to_n_samples(times, values, t_end, window_sec, n_samples):
    """Linearly resample `values` (T, C) sampled at `times` (T,) onto n_samples
    evenly spaced points covering [t_end - window_sec, t_end].

    If `times` has fewer than 2 points inside the window (e.g. right at an
    episode's start, before 500ms of history exists), the window is clipped
    to whatever history actually exists and the earliest available sample is
    held constant to fill the rest -- this is real historical padding
    (episode start), not imputation of a missing sensor reading, so it is not
    the same category of "never imputed" the extraction mask governs.
    """
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    t0 = t_end - window_sec
    query = np.linspace(t0, t_end, n_samples)

    in_window = times <= t_end
    if in_window.sum() == 0:
        return np.repeat(values[:1], n_samples, axis=0).astype(np.float32)

    t_hist = times[in_window]
    v_hist = values[in_window]
    if len(t_hist) == 1:
        return np.repeat(v_hist, n_samples, axis=0).astype(np.float32)

    out = np.empty((n_samples, values.shape[1]), dtype=np.float64)
    for c in range(values.shape[1]):
        out[:, c] = np.interp(query, t_hist, v_hist[:, c], left=v_hist[0, c], right=v_hist[-1, c])
    return out.astype(np.float32)


def force_dropout(hist, p, training, generator=None):
    """Per-example force dropout: with probability p, zero the entire history
    window for that example (§4.2 augmentation, defends against modal
    masking per ForceVLA/ForceFlow 2605.11048). No-op at eval time."""
    if not training or p <= 0.0:
        return hist
    bsize = hist.shape[0]
    keep = torch.rand(bsize, device=hist.device, generator=generator) >= p
    return hist * keep.view(bsize, 1, 1).to(hist.dtype)


def wrench_bias_injection(hist, bias_range_n=1.0, training=True, generator=None):
    """Per-example, per-axis constant wrench bias in [-bias_range_n, +bias_range_n],
    added uniformly across the whole history window (simulates a
    slowly-varying sensor offset, not per-sample noise). Applied to all 6
    wrench channels identically in N-equivalent magnitude -- the proposal
    states "±1 N" without specifying a separate torque unit/magnitude, so
    this is a literal, documented reading rather than an invented
    force/torque split. No-op at eval time."""
    if not training or bias_range_n <= 0.0:
        return hist
    bsize, _, channels = hist.shape
    bias = (torch.rand(bsize, 1, channels, device=hist.device, generator=generator) * 2 - 1) * bias_range_n
    return hist + bias


class ForceHistoryEncoder(nn.Module):
    """1D-conv encoder: (B, 20, 6) wrench history -> (B, hidden_dim) token."""

    def __init__(self, in_channels=6, hidden_dim=256, conv_channels=(32, 64)):
        super().__init__()
        c1, c2 = conv_channels
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, c1, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(c1, c2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.out_proj = nn.Linear(c2, hidden_dim)

    def forward(self, hist):
        """hist: (B, T, C) -> (B, hidden_dim)."""
        x = hist.transpose(1, 2)  # (B, C, T)
        x = self.net(x).squeeze(-1)  # (B, c2)
        return self.out_proj(x)


def _self_test():
    torch.manual_seed(0)
    np.random.seed(0)

    # resample_to_n_samples: exact reproduction of a linear function.
    t = np.linspace(0.0, 0.49, 15)  # ~30Hz native, 15 samples in 500ms
    v = np.stack([2.0 * t + 1.0, -1.0 * t], axis=1)  # 2 channels, known-linear
    out = resample_to_n_samples(t, v, t_end=0.49, window_sec=0.49, n_samples=20)
    assert out.shape == (20, 2)
    query = np.linspace(0.0, 0.49, 20)
    expected = np.stack([2.0 * query + 1.0, -1.0 * query], axis=1)
    assert np.allclose(out, expected, atol=1e-5), "linear interpolation should reproduce a linear function"

    # resample at episode start: fewer than 2 samples in window -> constant fill, no NaN.
    out_start = resample_to_n_samples(np.array([0.0]), np.array([[3.0, -3.0]]), t_end=0.0, window_sec=0.5, n_samples=20)
    assert out_start.shape == (20, 2)
    assert np.allclose(out_start, 3.0 * np.array([1.0, -1.0]))
    assert not np.isnan(out_start).any()

    # force_dropout: p=1.0 zeros everything; p=0.0 is a no-op; eval mode is always a no-op.
    hist = torch.randn(8, 20, 6)
    dropped_all = force_dropout(hist, p=1.0, training=True)
    assert torch.all(dropped_all == 0.0)
    kept_all = force_dropout(hist, p=0.0, training=True)
    assert torch.equal(kept_all, hist)
    eval_noop = force_dropout(hist, p=1.0, training=False)
    assert torch.equal(eval_noop, hist)

    # force_dropout: with p=0.15 over a large batch, roughly 15% of examples fully zeroed.
    big = torch.ones(2000, 20, 6)
    g = torch.Generator().manual_seed(0)
    out_big = force_dropout(big, p=0.15, training=True, generator=g)
    frac_zeroed = (out_big.sum(dim=(1, 2)) == 0).float().mean().item()
    assert 0.10 < frac_zeroed < 0.20, f"expected ~15% dropped, got {frac_zeroed:.3f}"

    # wrench_bias_injection: bounded within [-1, 1] N, constant across the time axis per example.
    hist0 = torch.zeros(16, 20, 6)
    biased = wrench_bias_injection(hist0, bias_range_n=1.0, training=True)
    assert biased.abs().max().item() <= 1.0 + 1e-5
    # constant across time within one example/channel
    assert torch.allclose(biased[:, 0, :], biased[:, -1, :])
    # eval mode no-op
    assert torch.equal(wrench_bias_injection(hist0, bias_range_n=1.0, training=False), hist0)

    # ForceHistoryEncoder: shape, finite output (incl. on the all-zero dropped-out case), gradient flows.
    enc = ForceHistoryEncoder(hidden_dim=128)
    x = torch.randn(4, 20, 6, requires_grad=True)
    y = enc(x)
    assert y.shape == (4, 128)
    assert torch.isfinite(y).all()
    y_zero = enc(torch.zeros(4, 20, 6))
    assert torch.isfinite(y_zero).all(), "encoder must not NaN on an all-zero (dropped-out) window"
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()

    # different histories -> different embeddings (not a degenerate constant encoder).
    y1 = enc(torch.randn(1, 20, 6))
    y2 = enc(torch.randn(1, 20, 6))
    assert not torch.allclose(y1, y2)

    print("src/compliance_vla/policy/force_encoder.py self-test: PASS")


if __name__ == "__main__":
    _self_test()
