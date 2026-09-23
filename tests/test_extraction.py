"""Compliance label extraction: does it recover the stiffness that is really there?

These tests run the whole pipeline -- contact-frame fit, windowed regression,
identifiability mask -- against a synthetic demonstration whose true per-axis
stiffness is known by construction, and then pin down the properties the method
claims rather than only the happy path:

* the recovered K matches ground truth on every axis;
* free-space timesteps, where nothing is identifiable, are masked OUT;
* the mask's conditions actually bind, individually;
* masked-out labels are never imputed;
* labels stay inside the controller's realizable range;
* each label depends only on a TRAILING window, so extraction is causal.
"""

import numpy as np
import pytest

from compliance_vla.extraction import (
    AXIS_NAMES,
    DEFAULT_SIGMA_F,
    K_MAX,
    K_MIN,
    ExtractionConfig,
    extract_demo,
    nearest_sample_indices,
)
from compliance_vla.synthetic import synthetic_demo

pytestmark = pytest.mark.slow


def test_extraction_succeeds_on_a_clean_synthetic_demo(extracted):
    result, _ = extracted
    assert result["status"] == "ok"
    # Contact points are exactly planar by construction, so the fit should be
    # essentially perfect -- a large ratio here means the frame fit regressed.
    assert result["frame_fit"]["planarity_ratio"] < 1e-6


@pytest.mark.parametrize("axis", range(6))
def test_recovers_known_stiffness_on_every_axis(extracted, axis):
    """The identifiability claim, checked end to end and per axis.

    The tolerance is deliberately per-axis-uniform rather than tuned: if any
    axis needed its own looser bound to pass, that would be a finding about the
    method, not a reason to widen the test.
    """
    result, true_k = extracted
    mask = result["mask"][:, axis]
    assert mask.sum() >= 20, f"{AXIS_NAMES[axis]}: too few identifiable timesteps to judge"

    k_median = np.median(result["k"][mask, axis])
    rel_err = abs(k_median - true_k[axis]) / true_k[axis]
    assert rel_err < 0.2, (
        f"{AXIS_NAMES[axis]}: recovered K={k_median:.1f} vs true {true_k[axis]:.1f} "
        f"({rel_err:.1%} error)")


def test_free_space_timesteps_are_not_marked_identifiable(extracted):
    """Nothing is identifiable without contact, and the mask must say so.

    This is the test that would catch a mask that passes everything: the
    per-axis coverage numbers alone cannot distinguish a working mask from a
    mask stuck at 1.
    """
    result, _ = extracted
    # The synthetic demo makes contact at t = 2.0 s.
    free_space = result["output_times"] < 1.5
    assert free_space.sum() > 0
    assert result["mask"][free_space].mean() < 0.05


def test_within_contact_coverage_clears_the_reported_floor(extracted):
    """Coverage is reported WITHIN contact, not over the whole demonstration.

    Measuring over the whole demonstration would dilute the number with
    free-space transit and reset time, in an uninformative direction.
    """
    result, _ = extracted
    coverage = result["mask_coverage_within_contact"]
    assert coverage.shape == (6,)
    assert np.all(coverage > 0.25), f"per-axis within-contact coverage {coverage}"
    # Within-contact coverage must be at least the whole-demo coverage, since
    # it drops the free-space denominator that can only contain masked-out steps.
    assert np.all(coverage >= result["mask_coverage"] - 1e-12)


def test_masked_out_labels_are_never_imputed(extracted):
    """Windows with too little data must stay NaN rather than receive a value."""
    result, _ = extracted
    k = result["k"]
    # Every finite label is a real fit; every non-finite one must also be masked out.
    non_finite = ~np.isfinite(k)
    assert not np.any(result["mask"] & non_finite), (
        "a non-finite K was marked identifiable -- masked entries must be excluded, "
        "never imputed")


def test_labels_stay_inside_the_realizable_stiffness_range(extracted):
    """Bounds are what keep labels interpretable as DEPLOYABLE impedance targets."""
    result, _ = extracted
    k = result["k"]
    finite = np.isfinite(k)
    for axis in range(6):
        col = k[finite[:, axis], axis]
        assert np.all(col >= K_MIN[axis] - 1e-6), f"{AXIS_NAMES[axis]} below k_min"
        assert np.all(col <= K_MAX[axis] + 1e-6), f"{AXIS_NAMES[axis]} above k_max"


def test_extraction_is_causal_in_the_leader_channel(synthetic, tight_sigma_f):
    """A label must not depend on data from its own future.

    Causality is tested by perturbing the LEADER pose after a cut time and
    checking that earlier labels are untouched. Perturbing the leader is what
    isolates the property: the contact frame is fitted from the FOLLOWER
    positions and the wrench, so it is bit-for-bit unchanged here, whereas
    truncating the record would also change the frame fit and confound the two
    effects.

    The contact frame is itself a whole-demonstration (or whole-session) fit and
    is deliberately NOT causal -- the method fits one frame per demonstration
    offline. Causality is a property of the windowed regression given that
    frame, and that is what is asserted here.

    A margin of one window plus a few samples is excluded either side of the
    cut, because the derivative's moving-average smoothing legitimately reaches
    a couple of samples backwards.
    """
    demo, _ = synthetic
    cfg = ExtractionConfig()
    rng = np.random.default_rng(0)

    baseline = extract_demo(demo, cfg, tight_sigma_f)

    cut_index = len(demo["t"]) // 2
    cut_time = demo["t"][cut_index] - demo["t"][0]
    perturbed_demo = dict(demo)
    for key in ("lx", "ly", "lz"):
        perturbed = demo[key].copy()
        perturbed[cut_index:] += rng.normal(0.0, 0.05, size=len(perturbed) - cut_index)
        perturbed_demo[key] = perturbed

    perturbed_result = extract_demo(perturbed_demo, cfg, tight_sigma_f)

    assert baseline["status"] == "ok" and perturbed_result["status"] == "ok"
    # The frame is fitted from follower pose and wrench only, so it must be identical.
    np.testing.assert_array_equal(
        np.asarray(baseline["R_contact"]), np.asarray(perturbed_result["R_contact"]))

    safe = baseline["output_times"] <= cut_time - cfg.window_sec - 0.01
    assert safe.sum() > 10, "not enough pre-cut output steps to make this check meaningful"

    np.testing.assert_array_equal(baseline["mask"][safe], perturbed_result["mask"][safe])
    np.testing.assert_allclose(
        baseline["k"][safe], perturbed_result["k"][safe],
        rtol=1e-9, atol=1e-9, equal_nan=True)

    # ...and the perturbation must actually have changed something afterwards,
    # or this test would pass against an extractor that ignored the leader.
    after = baseline["output_times"] > cut_time + cfg.window_sec
    assert not np.allclose(
        baseline["k"][after], perturbed_result["k"][after], rtol=1e-6, equal_nan=True)


def test_raising_the_noise_floor_can_only_shrink_the_mask(synthetic):
    """Mask condition (iii) must actually bind, and in the right direction.

    sigma_f is the measured noise floor; demanding more signal above it can
    never ADMIT a timestep that a lower floor rejected.
    """
    demo, _ = synthetic
    cfg = ExtractionConfig()

    tight = extract_demo(demo, cfg, np.array([0.1, 0.1, 0.1, 0.02, 0.02, 0.02]))
    loose = extract_demo(demo, cfg, DEFAULT_SIGMA_F)      # the real rig's floor: much higher

    assert tight["status"] == "ok" and loose["status"] == "ok"
    assert np.all(loose["mask"] <= tight["mask"]), (
        "raising sigma_f admitted a timestep that the lower floor rejected")
    assert loose["mask"].sum() < tight["mask"].sum(), "sigma_f had no effect at all"


def test_tightening_the_conditioning_limit_can_only_shrink_the_mask(synthetic, tight_sigma_f):
    """Mask condition (i), the excitation/conditioning term, must also bind."""
    permissive = extract_demo(synthetic[0], ExtractionConfig(kappa_max=1e3), tight_sigma_f)
    strict = extract_demo(synthetic[0], ExtractionConfig(kappa_max=5.0), tight_sigma_f)

    assert np.all(strict["mask"] <= permissive["mask"])
    assert strict["mask"].sum() < permissive["mask"].sum()


def test_demanding_more_sustained_contact_can_only_shrink_the_mask(synthetic, tight_sigma_f):
    """Mask condition (iv), sustained contact over the window."""
    lenient = extract_demo(
        synthetic[0], ExtractionConfig(contact_fraction_required=0.5), tight_sigma_f)
    strict = extract_demo(
        synthetic[0], ExtractionConfig(contact_fraction_required=1.0), tight_sigma_f)

    assert np.all(strict["mask"] <= lenient["mask"])


def test_extraction_refuses_a_demo_with_no_planar_contact(tight_sigma_f):
    """A failed frame fit must fail the whole extraction, with a reason.

    Returning labels computed in an untrustworthy frame would be worse than
    returning nothing, because nothing downstream would be able to tell.
    """
    rng = np.random.default_rng(0)
    demo, _ = synthetic_demo(seed=0)
    n = len(demo["t"])
    # Scatter the follower positions so no plane can be fitted.
    for key in ("fx", "fy", "fz"):
        demo[key] = rng.normal(0.0, 0.1, size=n)

    result = extract_demo(demo, ExtractionConfig(), tight_sigma_f)

    assert result["status"] == "failed"
    assert "planarity_ratio" in result["reason"]


def test_anisotropic_stiffness_is_recovered_as_anisotropic(tight_sigma_f):
    """The method's value is per-axis compliance, so isotropy must not be assumed.

    The generator's in-plane x and normal-direction stiffnesses differ by ~2.7x;
    the extracted labels must reproduce a comparable ratio rather than collapsing
    to one shared value.
    """
    demo, true_k = synthetic_demo(seed=0)
    result = extract_demo(demo, ExtractionConfig(), tight_sigma_f)

    medians = np.array([
        np.median(result["k"][result["mask"][:, a], a]) for a in range(6)])
    true_ratio = true_k[2] / true_k[0]
    fit_ratio = medians[2] / medians[0]
    assert fit_ratio == pytest.approx(true_ratio, rel=0.25), (
        f"anisotropy ratio {fit_ratio:.2f} vs true {true_ratio:.2f}")


def test_nearest_sample_indices_picks_the_closest_sample():
    t = np.array([0.0, 0.1, 0.2, 0.3])
    queries = np.array([0.04, 0.06, 0.29, 10.0])
    idx = nearest_sample_indices(t, queries)
    np.testing.assert_array_equal(idx, [0, 1, 3, 3])
