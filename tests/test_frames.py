"""Contact-frame stiffness -> base-frame diagonal, and chunk step indexing.

The compliance head predicts K in the contact frame; the controller consumes a
base-frame diagonal. The rotation R diag(K) R^T is not diagonal in general, so
the conversion necessarily discards cross-axis coupling. These tests check that
the rotation is done correctly AND that the discarded fraction is reported
honestly, because a silently-dropped anisotropy would look like a controller
that merely underperformed.

The first four cases are ported from the reference implementation's own
self-test, which is why they check invariants (trace preservation) rather than
hard-coded numbers.
"""

import numpy as np
import pytest

from compliance_vla.frames import chunk_step_index, rotate_diag_stiffness_to_base

K_EXAMPLE = np.array([200.0, 250.0, 150.0, 10.0, 12.0, 8.0])
K_ANISOTROPIC = np.array([300.0, 100.0, 150.0, 20.0, 5.0, 8.0])
K_ISOTROPIC_IN_PLANE = np.array([200.0, 200.0, 150.0, 10.0, 10.0, 8.0])


def rotation_about_z(degrees):
    theta = np.deg2rad(degrees)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_identity_rotation_passes_stiffness_through_unchanged():
    k_base, dropped = rotate_diag_stiffness_to_base(K_EXAMPLE, np.eye(3))
    np.testing.assert_allclose(k_base, K_EXAMPLE)
    assert dropped < 1e-9, "an axis-aligned frame discards nothing"


def test_rotating_anisotropic_stiffness_loses_off_diagonal_energy():
    """The honest part: some anisotropy genuinely cannot be realized as a
    base-frame diagonal, and the caller is told how much."""
    k_base, dropped = rotate_diag_stiffness_to_base(K_ANISOTROPIC, rotation_about_z(30.0))

    assert dropped > 1e-6
    assert not np.allclose(k_base[0:2], K_ANISOTROPIC[0:2]), \
        "a rotation about z must actually change the in-plane values"


def test_trace_is_preserved_by_the_rotation():
    """Trace is basis-independent, so the per-block sums must survive the
    rotation exactly -- an independent check on the rotation arithmetic."""
    k_base, _ = rotate_diag_stiffness_to_base(K_ANISOTROPIC, rotation_about_z(30.0))
    assert k_base[0:3].sum() == pytest.approx(K_ANISOTROPIC[0:3].sum(), abs=1e-8)
    assert k_base[3:6].sum() == pytest.approx(K_ANISOTROPIC[3:6].sum(), abs=1e-8)


def test_rotating_about_an_isotropic_plane_is_a_no_op():
    k_base, dropped = rotate_diag_stiffness_to_base(
        K_ISOTROPIC_IN_PLANE, rotation_about_z(30.0))
    np.testing.assert_allclose(k_base, K_ISOTROPIC_IN_PLANE, atol=1e-8)
    assert dropped < 1e-9


def test_batched_input_matches_per_row_calls():
    R = rotation_about_z(30.0)
    stacked = np.stack([K_EXAMPLE, K_ANISOTROPIC, K_ISOTROPIC_IN_PLANE])

    k_batch, dropped_batch = rotate_diag_stiffness_to_base(stacked, R)

    assert k_batch.shape == (3, 6)
    for i, row in enumerate(stacked):
        k_row, dropped_row = rotate_diag_stiffness_to_base(row, R)
        np.testing.assert_allclose(k_batch[i], k_row)
        assert dropped_batch[i] == pytest.approx(dropped_row, abs=1e-10)


def test_dropped_fraction_grows_with_misalignment():
    """Zero at alignment, maximal at 45 degrees for an in-plane anisotropy."""
    fractions = [
        rotate_diag_stiffness_to_base(K_ANISOTROPIC, rotation_about_z(d))[1]
        for d in (0.0, 15.0, 30.0, 45.0)
    ]
    assert fractions == sorted(fractions)
    assert fractions[0] < 1e-9


def test_rotated_stiffness_stays_physically_plausible():
    """A rotation of a positive-definite stiffness must stay positive."""
    k_base, _ = rotate_diag_stiffness_to_base(K_ANISOTROPIC, rotation_about_z(37.0))
    assert np.all(k_base > 0.0)


@pytest.mark.parametrize(
    "elapsed, expected",
    [
        (0.0, 0),
        (0.5, 15),     # 0.5 s * 30 Hz
        (10.0, 31),    # clamped to the last step, not an overrun
        (-1.0, 0),     # clamped, not a negative index
    ],
)
def test_chunk_step_index(elapsed, expected):
    assert chunk_step_index(elapsed, action_rate_hz=30.0, chunk_size=32) == expected


def test_chunk_step_index_holds_the_last_step_when_replanning_is_late():
    """ACT-style graceful degradation: hold the final prediction rather than
    crash the control loop."""
    assert chunk_step_index(3.0, action_rate_hz=30.0, chunk_size=32) == 31


@pytest.mark.parametrize("rate, size", [(0.0, 32), (-30.0, 32), (30.0, 0)])
def test_chunk_step_index_rejects_nonsense_parameters(rate, size):
    with pytest.raises(ValueError):
        chunk_step_index(0.1, action_rate_hz=rate, chunk_size=size)
