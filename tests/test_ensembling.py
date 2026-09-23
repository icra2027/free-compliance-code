"""Temporal ensembling across overlapping action chunks.

The mechanism's value is that it smooths chunk boundaries rather than discarding
the previous chunk wholesale. The properties worth pinning down are therefore
the ones a subtle bug would break silently: the weighting must favour the
freshest prediction, disabling it must reproduce the old behaviour EXACTLY
rather than approximately, and the buffer must not grow without bound.
"""

import numpy as np
import pytest

from compliance_vla.ensembling import ChunkEnsembler

HORIZON, ACTION_DIM = 32, 13


def chunk(value, horizon=HORIZON, dim=ACTION_DIM):
    return np.full((horizon, dim), float(value), dtype=np.float32)


def test_a_single_chunk_is_returned_unchanged():
    ens = ChunkEnsembler()
    ens.buffer_chunk(0, chunk(1.0))
    np.testing.assert_allclose(ens.ensembled_action(0), np.ones(ACTION_DIM))
    np.testing.assert_allclose(ens.ensembled_action(5), np.ones(ACTION_DIM))


def test_returns_none_when_no_chunk_covers_the_step():
    """Callers must be able to tell "no prediction" from a stale one."""
    ens = ChunkEnsembler()
    ens.buffer_chunk(0, chunk(1.0, horizon=4))
    assert ens.ensembled_action(10) is None
    assert ens.ensembled_action(-1) is None


def test_overlapping_chunks_are_blended_by_recency():
    """The blend must lie strictly between the two predictions, nearer the fresher one."""
    ens = ChunkEnsembler(ensemble_m=0.05)
    ens.buffer_chunk(0, chunk(0.0))
    ens.buffer_chunk(8, chunk(1.0))

    blended = ens.ensembled_action(8)[0]
    assert 0.0 < blended < 1.0
    assert blended > 0.5, "the freshly-queried chunk should dominate at its own step"


def test_weights_match_the_exponential_recency_formula():
    """Checked against the formula directly, not just for plausibility."""
    m = 0.05
    ens = ChunkEnsembler(ensemble_m=m)
    ens.buffer_chunk(0, chunk(0.0))
    ens.buffer_chunk(8, chunk(1.0))

    step = 10
    ages = np.array([step - 0, step - 8])          # 10 and 2
    weights = np.exp(-m * ages)
    weights /= weights.sum()
    expected = weights @ np.array([0.0, 1.0])

    assert ens.ensembled_action(step)[0] == pytest.approx(expected, rel=1e-6)


def test_larger_m_concentrates_weight_on_the_freshest_chunk():
    values = {}
    for m in (0.01, 0.5, 5.0):
        ens = ChunkEnsembler(ensemble_m=m)
        ens.buffer_chunk(0, chunk(0.0))
        ens.buffer_chunk(8, chunk(1.0))
        values[m] = ens.ensembled_action(8)[0]

    assert values[0.01] < values[0.5] < values[5.0]
    assert values[5.0] == pytest.approx(1.0, abs=1e-6), (
        "m -> infinity must recover naive single-chunk execution")


def test_m_of_zero_is_a_plain_unweighted_mean():
    ens = ChunkEnsembler(ensemble_m=0.0)
    ens.buffer_chunk(0, chunk(0.0))
    ens.buffer_chunk(8, chunk(1.0))
    assert ens.ensembled_action(8)[0] == pytest.approx(0.5)


def test_disabling_returns_the_newest_chunk_exactly():
    """Disabled must be the exact pre-ensembling behaviour, not an approximation."""
    disabled = ChunkEnsembler(enabled=False)
    for query_step, value in ((0, 0.0), (4, 1.0), (8, 2.0)):
        disabled.buffer_chunk(query_step, chunk(value))

    np.testing.assert_array_equal(
        disabled.ensembled_action(8), np.full(ACTION_DIM, 2.0, dtype=np.float32))


def test_blending_is_elementwise_across_all_action_channels():
    """Position, rotation vector, log K and gripper are all blended the same way."""
    ens = ChunkEnsembler(ensemble_m=0.0)
    a = np.tile(np.arange(ACTION_DIM, dtype=np.float32), (HORIZON, 1))
    b = a + 2.0
    ens.buffer_chunk(0, a)
    ens.buffer_chunk(0, b)
    np.testing.assert_allclose(ens.ensembled_action(0), np.arange(ACTION_DIM) + 1.0)


def test_expired_chunks_are_pruned_so_the_buffer_stays_bounded():
    """A rollout is long; an unpruned buffer would grow for its whole duration."""
    ens = ChunkEnsembler()
    for i in range(200):
        ens.buffer_chunk(i * 8, chunk(i, horizon=8))
    assert len(ens) <= 2, f"buffer grew to {len(ens)} chunks"


def test_pruning_keeps_chunks_that_still_overlap():
    ens = ChunkEnsembler()
    ens.buffer_chunk(0, chunk(0.0, horizon=32))
    ens.buffer_chunk(8, chunk(1.0, horizon=32))
    assert len(ens) == 2, "a chunk still covering future steps must not be dropped"


def test_reset_clears_the_buffer_between_rollouts():
    ens = ChunkEnsembler()
    ens.buffer_chunk(0, chunk(1.0))
    ens.reset()
    assert len(ens) == 0
    assert ens.ensembled_action(0) is None


def test_output_dtype_is_float32_for_the_control_interface():
    ens = ChunkEnsembler()
    ens.buffer_chunk(0, chunk(1.0))
    ens.buffer_chunk(4, chunk(2.0))
    assert ens.ensembled_action(4).dtype == np.float32


@pytest.mark.parametrize(
    "bad_chunk, message",
    [
        (np.zeros((5,)), "2-D"),
        (np.zeros((0, 13)), "at least one step"),
    ],
)
def test_malformed_chunks_are_rejected(bad_chunk, message):
    ens = ChunkEnsembler()
    with pytest.raises(ValueError, match=message):
        ens.buffer_chunk(0, bad_chunk)


def test_negative_decay_rate_is_rejected():
    """A negative m would weight STALE predictions more heavily."""
    with pytest.raises(ValueError, match="non-negative"):
        ChunkEnsembler(ensemble_m=-0.1)
