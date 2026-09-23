"""Controller-side safety: the energy tank and the log-space rate limiter.

These mirror the C++ that runs on the robot at 1 kHz (see the module docstring
in compliance_vla.safety). The invariants asserted here are the ones the
passivity argument rests on, stated so that a future change to either the Python
mirror or the deployed C++ has something concrete to violate.
"""

import numpy as np
import pytest

from compliance_vla.safety import EnergyTank, LogSpaceStiffnessRateLimiter

K_MIN = np.array([50.0] * 3 + [5.0] * 3)
K_MAX = np.array([1500.0] * 3 + [100.0] * 3)
GAMMA = np.array([3.0] * 6)


# --------------------------------------------------------------------------
# EnergyTank
# --------------------------------------------------------------------------

def test_tank_starts_at_its_initial_energy():
    tank = EnergyTank(e0=10.0, e_min=0.0, e_max=20.0)
    assert tank.energy == 10.0
    assert not tank.depleted()


def test_dissipated_power_refills_the_tank_but_not_past_the_cap():
    """A long quiet period must not let the tank grow without bound."""
    tank = EnergyTank(e0=10.0, e_min=0.0, e_max=12.0)
    tank.accumulate_dissipated_power(power_watts=1000.0, dt=1.0)
    assert tank.energy == 12.0


def test_energy_injection_cannot_push_the_tank_below_its_floor():
    tank = EnergyTank(e0=1.0, e_min=0.5, e_max=20.0)
    tank.accumulate_dissipated_power(power_watts=-1000.0, dt=1.0)
    assert tank.energy == 0.5
    assert tank.depleted()


def test_a_withdrawal_within_budget_is_committed():
    tank = EnergyTank(e0=10.0, e_min=0.0, e_max=20.0)
    assert tank.try_withdraw(4.0)
    assert tank.energy == pytest.approx(6.0)


def test_a_withdrawal_beyond_budget_is_refused_and_leaves_the_tank_untouched():
    """Refusal must be atomic: a rejected stiffness increase must not
    half-drain the tank on its way to being rejected."""
    tank = EnergyTank(e0=1.0, e_min=0.5, e_max=20.0)
    assert not tank.try_withdraw(5.0)
    assert tank.energy == 1.0


def test_a_withdrawal_landing_exactly_on_the_floor_is_allowed():
    """Floating-point rounding at the floor must not spuriously refuse a
    withdrawal that is exactly affordable."""
    tank = EnergyTank(e0=1.0, e_min=0.9, e_max=20.0)
    assert tank.try_withdraw(1.0 - 0.9)
    assert tank.energy == pytest.approx(0.9)
    assert tank.depleted()


def test_a_negative_withdrawal_is_refused():
    """Withdrawing negative energy would be a back-door deposit."""
    tank = EnergyTank(e0=10.0, e_min=0.0, e_max=20.0)
    assert not tank.try_withdraw(-5.0)
    assert tank.energy == 10.0


def test_a_depleted_tank_refuses_every_further_withdrawal():
    """This is the property that makes 'the controller can give but not
    stiffen past the budget' true."""
    tank = EnergyTank(e0=0.5, e_min=0.5, e_max=20.0)
    assert tank.depleted()
    assert not tank.try_withdraw(0.001)
    assert tank.try_withdraw(0.0), "a zero-cost withdrawal is still affordable"


def test_tank_rejects_an_inverted_range():
    with pytest.raises(ValueError, match="e_min <= e_max"):
        EnergyTank(e0=1.0, e_min=5.0, e_max=1.0)


# --------------------------------------------------------------------------
# LogSpaceStiffnessRateLimiter
# --------------------------------------------------------------------------

def test_first_step_without_reset_adopts_the_command():
    limiter = LogSpaceStiffnessRateLimiter(GAMMA, K_MIN, K_MAX)
    assert not limiter.initialized
    out = limiter.step(np.array([200.0] * 3 + [20.0] * 3), dt=1e-3)
    np.testing.assert_allclose(out, [200.0] * 3 + [20.0] * 3)
    assert limiter.initialized


def test_log_rate_never_exceeds_gamma():
    """The defining property: |d(log k)/dt| <= gamma, per axis, per step."""
    limiter = LogSpaceStiffnessRateLimiter(GAMMA, K_MIN, K_MAX)
    limiter.reset(np.array([100.0] * 3 + [10.0] * 3))

    dt = 1e-3
    target = np.array([1500.0] * 3 + [100.0] * 3)
    previous = limiter.current()
    for _ in range(500):
        current = limiter.step(target, dt)
        log_rate = np.abs(np.log(current) - np.log(previous)) / dt
        assert np.all(log_rate <= GAMMA + 1e-9), f"log rate {log_rate} exceeded gamma {GAMMA}"
        previous = current


def test_the_limit_is_scale_invariant_in_log_space():
    """One gamma must behave the same at low and high stiffness.

    A linear-rate limiter would move a fixed number of N/m per step, so the same
    gamma would take wildly different times to cover the same RATIO at 50 N/m
    and at 1000 N/m. In log space the two must take equally long.
    """
    dt = 1e-3

    # Uniform, wide bounds on all six axes: the real per-axis ceilings would
    # clamp one of the two cases before it finished doubling and mask the
    # property under test.
    wide_min, wide_max = np.full(6, 1.0), np.full(6, 1e5)

    def steps_to_double(start):
        limiter = LogSpaceStiffnessRateLimiter(GAMMA, wide_min, wide_max)
        limiter.reset(np.full(6, start))
        target = np.full(6, start * 2.0)
        for n in range(1, 100_000):
            if np.all(limiter.step(target, dt) >= start * 2.0 - 1e-9):
                return n
        raise AssertionError("never reached the target")

    assert steps_to_double(60.0) == steps_to_double(600.0)


def test_the_commanded_value_is_reached_eventually():
    limiter = LogSpaceStiffnessRateLimiter(GAMMA, K_MIN, K_MAX)
    limiter.reset(np.array([100.0] * 3 + [10.0] * 3))
    target = np.array([800.0] * 3 + [50.0] * 3)
    for _ in range(5000):
        out = limiter.step(target, dt=1e-3)
    np.testing.assert_allclose(out, target, rtol=1e-9)


def test_output_is_always_clamped_into_the_realizable_range():
    """Range clamping is applied AFTER rate limiting, so an out-of-range
    command cannot be used to bypass the limiter."""
    limiter = LogSpaceStiffnessRateLimiter(GAMMA, K_MIN, K_MAX)
    limiter.reset(K_MAX)
    for _ in range(5000):
        out = limiter.step(np.full(6, 1e9), dt=1e-3)
        assert np.all(out <= K_MAX + 1e-9)
    for _ in range(20000):
        out = limiter.step(np.full(6, 1e-9), dt=1e-3)
    assert np.all(out >= K_MIN - 1e-9)


def test_reset_clamps_an_out_of_range_initial_value():
    limiter = LogSpaceStiffnessRateLimiter(GAMMA, K_MIN, K_MAX)
    limiter.reset(np.full(6, 1e6))
    np.testing.assert_allclose(limiter.current(), K_MAX)


def test_a_zero_gamma_axis_is_frozen():
    gamma = np.array([0.0, 3.0, 3.0, 3.0, 3.0, 3.0])
    limiter = LogSpaceStiffnessRateLimiter(gamma, K_MIN, K_MAX)
    start = np.array([100.0] * 3 + [10.0] * 3)
    limiter.reset(start)
    for _ in range(1000):
        out = limiter.step(np.array([1400.0] * 3 + [90.0] * 3), dt=1e-3)
    assert out[0] == pytest.approx(100.0)
    assert out[1] > 100.0


def test_decreases_are_rate_limited_too():
    """Both directions are bounded; the tank, not the limiter, is what makes
    decreases always permissible."""
    limiter = LogSpaceStiffnessRateLimiter(GAMMA, K_MIN, K_MAX)
    limiter.reset(np.array([1000.0] * 3 + [90.0] * 3))
    out = limiter.step(np.full(6, 50.0), dt=1e-3)
    assert np.all(out > np.array([990.0] * 3 + [80.0] * 3))


@pytest.mark.parametrize("dt", [0.0, -1e-3])
def test_a_non_positive_timestep_is_rejected(dt):
    limiter = LogSpaceStiffnessRateLimiter(GAMMA, K_MIN, K_MAX)
    limiter.reset(np.full(6, 100.0))
    with pytest.raises(ValueError, match="dt must be positive"):
        limiter.step(np.full(6, 200.0), dt)


def test_invalid_bounds_are_rejected():
    with pytest.raises(ValueError, match="0 < k_min <= k_max"):
        LogSpaceStiffnessRateLimiter(GAMMA, K_MAX, K_MIN)
    with pytest.raises(ValueError, match="0 < k_min <= k_max"):
        LogSpaceStiffnessRateLimiter(GAMMA, np.zeros(6), K_MAX)


def test_mismatched_shapes_are_rejected():
    with pytest.raises(ValueError, match="same shape"):
        LogSpaceStiffnessRateLimiter(np.ones(3), K_MIN, K_MAX)
