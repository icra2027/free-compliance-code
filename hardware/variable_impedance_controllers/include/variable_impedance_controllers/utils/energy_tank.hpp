#pragma once

#include <algorithm>

// Standard passivity-via-energy-tank bookkeeping: a scalar reservoir that starts at E0,
// accumulates energy the controller dissipates (damping doing negative work on the arm),
// and is drawn down by anything that could inject energy (a commanded stiffness INCREASE,
// since raising K while displaced adds potential energy to the interaction). When the tank
// is at its floor, stiffness increases must be refused (decreases are always safe -- they
// only remove potential energy). This is what keeps a piecewise-constant, chunk-boundary-
// jumpy K output from being able to pump energy into the arm/environment interaction
// indefinitely; it complements (does not replace) LogSpaceStiffnessRateLimiter, which
// bounds how FAST K can move but not, on its own, whether the cumulative moves stay
// passive.
//
// Ported (unchanged except namespace) from
// fr3_bilateral_teleop/include/fr3_bilateral_teleop/energy_tank.hpp, where it is unit-tested
// (test/test_energy_tank.cpp). See log_space_stiffness_rate_limiter.hpp in this same
// directory for why this is a plain copy rather than a shared dependency between the two
// packages.
class EnergyTank {
public:
  // e0: initial/nominal tank energy (J). e_min: floor below which stiffness increases are
  // refused. e_max: cap so a long quiet (highly damped) period can't let the tank grow
  // without bound -- excess dissipated energy is simply not banked past this point, not an
  // error.
  EnergyTank(double e0, double e_min, double e_max)
  : energy_(e0), e_min_(e_min), e_max_(e_max) {
  }

  double energy() const {return energy_;}

  // Call every control cycle with the (signed) power the low-level controller is currently
  // dissipating via damping, positive when energy is flowing OUT of the mechanical
  // interaction and INTO the tank (i.e. damping doing positive work removing kinetic
  // energy). dt is the control period in seconds.
  void accumulate_dissipated_power(double power_watts, double dt) {
    energy_ = std::clamp(energy_ + power_watts * dt, e_min_, e_max_);
  }

  // True once the tank has been drawn down to its floor -- the caller must not permit a
  // stiffness INCREASE this cycle if this returns true. Decreases remain permitted
  // regardless.
  bool depleted() const {return energy_ <= e_min_;}

  // Attempts to withdraw delta_energy (>= 0) from the tank, e.g. the potential energy a
  // requested stiffness increase would add at the current pose error. Returns true and
  // commits the withdrawal only if it would not push the tank below its floor; otherwise
  // leaves the tank untouched and returns false, signalling the caller to reject (or clamp)
  // the stiffness increase that would have required it.
  bool try_withdraw(double delta_energy) {
    // kFloorEpsilon absorbs floating-point rounding at the exact floor (e.g. 1.0 - 0.9 !=
    // 0.1 in double precision) so a withdrawal that lands exactly on e_min_ isn't
    // spuriously rejected. Deliberately tiny relative to any physically meaningful energy
    // quantum here.
    constexpr double kFloorEpsilon = 1e-9;
    if (delta_energy < 0.0 || energy_ - delta_energy < e_min_ - kFloorEpsilon) {
      return false;
    }
    energy_ = std::max(energy_ - delta_energy, e_min_);
    return true;
  }

private:
  double energy_;
  double e_min_;
  double e_max_;
};
