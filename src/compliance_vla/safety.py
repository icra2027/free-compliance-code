"""Controller-side safety for a policy-commanded, chunk-rate stiffness.

A compliance-output policy emits K once per action chunk, so the stiffness the
low-level controller sees is piecewise-constant and jumps at chunk boundaries.
A jump injects energy in a single control step, which can make the arm buzz or
trip a protective stop. Two mechanisms guard against that, and they are
complementary rather than redundant:

`LogSpaceStiffnessRateLimiter` bounds how FAST K may move,
|d(log K)/dt| <= gamma. The limit is imposed in log space because stiffness
spans decades (tens to ~1500 N/m): one gamma in LINEAR units would be wildly
too aggressive at low K and far too conservative at high K, whereas a log-rate
bound is scale-invariant and a single gamma works across the whole commandable
range.

`EnergyTank` bounds whether the CUMULATIVE moves stay passive. It is the
standard passivity-via-energy-tank formulation: a scalar reservoir that starts
at E0, is replenished by energy the controller dissipates through damping, and
is drawn down by anything that could inject energy -- notably a commanded
stiffness INCREASE, which adds potential energy to the interaction while the
arm is displaced. Once the tank is at its floor, stiffness increases are
refused; decreases are always permitted, since they only remove potential
energy. The controller can therefore always give, but can never forcibly
stiffen its way past the passivity budget.

Rate limiting alone does not bound cumulative energy injection, and a tank alone
does not bound per-step discontinuity, which is why both are present.

THIS MODULE IS A REFERENCE MIRROR, NOT THE DEPLOYED CODE. The implementation
that actually runs on the robot at 1 kHz is the C++ in
`hardware/variable_impedance_controllers/include/variable_impedance_controllers/utils/` (and its twin in
the teleop package), which carries its own C++ unit tests. This Python port
exists so the same invariants can be stated and checked as part of this
release's test suite, on a machine with no ROS 2 toolchain. It mirrors the C++
semantics line for line, including the floor epsilon. If one side changes, change
both.
"""

from collections.abc import Sequence

import numpy as np

__all__ = ["EnergyTank", "LogSpaceStiffnessRateLimiter"]

#: Absorbs floating-point rounding exactly at the tank floor, so a withdrawal
#: that lands precisely on e_min is not spuriously rejected. Deliberately tiny
#: relative to any physically meaningful energy quantum here.
FLOOR_EPSILON = 1e-9


class EnergyTank:
    """Passivity bookkeeping for a variable-stiffness controller.

    Args:
        e0: initial/nominal tank energy (J).
        e_min: floor below which stiffness increases must be refused.
        e_max: cap, so a long quiet (highly damped) period cannot let the tank
            grow without bound. Dissipated energy beyond this is simply not
            banked; that is not an error.
    """

    def __init__(self, e0: float, e_min: float, e_max: float):
        if not (e_min <= e_max):
            raise ValueError(f"require e_min <= e_max, got e_min={e_min}, e_max={e_max}")
        self._energy = float(np.clip(e0, e_min, e_max))
        self._e_min = float(e_min)
        self._e_max = float(e_max)

    @property
    def energy(self) -> float:
        return self._energy

    @property
    def e_min(self) -> float:
        return self._e_min

    @property
    def e_max(self) -> float:
        return self._e_max

    def accumulate_dissipated_power(self, power_watts: float, dt: float) -> None:
        """Call every control cycle with the signed power being dissipated via
        damping -- positive when energy flows OUT of the mechanical interaction
        and INTO the tank. `dt` is the control period in seconds."""
        self._energy = float(
            np.clip(self._energy + power_watts * dt, self._e_min, self._e_max))

    def depleted(self) -> bool:
        """True once drawn down to the floor. The caller must not permit a
        stiffness INCREASE this cycle while this is true. Decreases remain
        permitted regardless."""
        return self._energy <= self._e_min

    def try_withdraw(self, delta_energy: float) -> bool:
        """Attempts to withdraw `delta_energy` (>= 0), e.g. the potential energy
        a requested stiffness increase would add at the current pose error.

        Commits and returns True only if the withdrawal would not push the tank
        below its floor. Otherwise the tank is left untouched and False is
        returned, signalling the caller to reject or clamp the increase.
        """
        if delta_energy < 0.0 or self._energy - delta_energy < self._e_min - FLOOR_EPSILON:
            return False
        self._energy = max(self._energy - delta_energy, self._e_min)
        return True


class LogSpaceStiffnessRateLimiter:
    """Rate-limits a diagonal Cartesian stiffness so |d(log k)/dt| <= gamma.

    Args:
        gamma_per_axis: (6,) max |d(log k)/dt| per axis, 1/s.
        k_min, k_max: (6,) hard bounds on the realizable stiffness range,
            applied AFTER rate limiting so the limiter cannot be bypassed by
            requesting an out-of-range target.
    """

    def __init__(
            self, gamma_per_axis: Sequence[float],
            k_min: Sequence[float], k_max: Sequence[float]):
        self._gamma = np.asarray(gamma_per_axis, dtype=np.float64)
        k_min_arr = np.asarray(k_min, dtype=np.float64)
        k_max_arr = np.asarray(k_max, dtype=np.float64)
        if not (self._gamma.shape == k_min_arr.shape == k_max_arr.shape):
            raise ValueError(
                "gamma_per_axis, k_min and k_max must have the same shape, got "
                f"{self._gamma.shape}, {k_min_arr.shape}, {k_max_arr.shape}")
        if np.any(self._gamma < 0.0):
            raise ValueError("gamma_per_axis must be non-negative")
        if np.any(k_min_arr <= 0.0) or np.any(k_max_arr < k_min_arr):
            raise ValueError("require 0 < k_min <= k_max elementwise")
        self._log_k_min = np.log(k_min_arr)
        self._log_k_max = np.log(k_max_arr)
        self._log_k_state = np.zeros_like(self._gamma)
        self._initialized = False

    def _clamp_log(self, log_k: np.ndarray) -> np.ndarray:
        return np.clip(log_k, self._log_k_min, self._log_k_max)

    @property
    def initialized(self) -> bool:
        return self._initialized

    def reset(self, initial_k: Sequence[float]) -> None:
        """Must be called before the first `step`, since there is no sane
        "previous stiffness" to rate-limit from otherwise. Clamped into
        [k_min, k_max] like every other output."""
        k = np.maximum(np.asarray(initial_k, dtype=np.float64), 1e-9)
        self._log_k_state = self._clamp_log(np.log(k))
        self._initialized = True

    def current(self) -> np.ndarray:
        return np.exp(self._log_k_state)

    def step(self, k_commanded: Sequence[float], dt: float) -> np.ndarray:
        """Advances the internal state by at most gamma*dt (in log space, per
        axis) toward `k_commanded`, returning the rate-limited, range-clamped
        stiffness. `dt` must be positive; the caller (the 1 kHz control loop)
        owns the actual timestep.

        The first call with no prior `reset` adopts the command directly rather
        than rate-limiting from an arbitrary state.
        """
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")
        if not self._initialized:
            self.reset(k_commanded)
            return self.current()
        k = np.maximum(np.asarray(k_commanded, dtype=np.float64), 1e-9)
        log_k_target = self._clamp_log(np.log(k))
        delta = log_k_target - self._log_k_state
        max_step = self._gamma * dt
        delta_clamped = np.clip(delta, -max_step, max_step)
        self._log_k_state = self._clamp_log(self._log_k_state + delta_clamped)
        return self.current()
