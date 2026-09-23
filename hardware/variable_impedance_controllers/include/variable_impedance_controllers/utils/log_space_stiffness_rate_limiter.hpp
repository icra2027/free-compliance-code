#pragma once

#include <cmath>

#include <Eigen/Core>

// Log-space rate limiter for a diagonal Cartesian stiffness: |d(log k)/dt| <= gamma.
// Stiffness spans decades (tens to ~1500 N/m), so limiting the raw (linear) rate would let
// the same gamma be wildly too aggressive at low k and too conservative at high k; limiting
// the LOG rate makes the bound scale-invariant, so one gamma works across the whole
// commandable range. This is what protects the arm from the piecewise-constant K produced
// by chunked policy predictions: without it, a stiffness jump at a chunk boundary injects
// energy in one control step and can make the arm buzz or trip a protective stop.
//
// Ported (unchanged except namespace) from
// fr3_bilateral_teleop/include/fr3_bilateral_teleop/stiffness_rate_limiter.hpp, where it is
// unit-tested (test/test_stiffness_rate_limiter.cpp). Kept in the global namespace here to
// match this package's existing utils/ convention (see e.g. saturateTorqueRate,
// exponential_moving_average) rather than fr3_bilateral_teleop's own `fr3_bilateral_teleop::`
// namespace. If the two diverge, update both -- there is deliberately no shared dependency
// between the two packages so variable_impedance_controllers stays self-contained.
class LogSpaceStiffnessRateLimiter {
public:
  using Vector6d = Eigen::Matrix<double, 6, 1>;

  // gamma_per_axis: max |d(log k)/dt| per axis, 1/s. k_min/k_max: hard bounds on the
  // realizable stiffness range, applied AFTER rate limiting so the limiter can never be
  // bypassed by requesting an out-of-range target.
  LogSpaceStiffnessRateLimiter(
    const Vector6d & gamma_per_axis, const Vector6d & k_min, const Vector6d & k_max)
  : gamma_(gamma_per_axis), log_k_min_(k_min.array().log()), log_k_max_(k_max.array().log()) {
  }

  // Must be called once before the first step() -- there is no sane "previous stiffness" to
  // rate-limit from otherwise. Clamped into [k_min, k_max] like every other output.
  void reset(const Vector6d & initial_k) {
    log_k_state_ = clamp_log(initial_k.array().max(1e-9).log().matrix());
    initialized_ = true;
  }

  bool initialized() const {return initialized_;}

  // Advances the internal state by at most gamma*dt (in log space, per axis) toward
  // k_commanded, and returns the resulting (rate-limited, range-clamped) stiffness. dt must
  // be positive; the caller (the 1 kHz controller loop) owns the actual timestep.
  Vector6d step(const Vector6d & k_commanded, double dt) {
    if (!initialized_) {
      reset(k_commanded);
      return current();
    }
    const Vector6d log_k_target = clamp_log(k_commanded.array().max(1e-9).log().matrix());
    const Vector6d delta = log_k_target - log_k_state_;
    const Vector6d max_step = gamma_ * dt;
    const Vector6d delta_clamped = delta.array().max(-max_step.array()).min(max_step.array());
    log_k_state_ = clamp_log(log_k_state_ + delta_clamped);
    return log_k_state_.array().exp().matrix();
  }

  Vector6d current() const {return log_k_state_.array().exp().matrix();}

private:
  Vector6d clamp_log(const Vector6d & log_k) const {
    return log_k.array().max(log_k_min_.array()).min(log_k_max_.array()).matrix();
  }

  Vector6d gamma_;
  Vector6d log_k_min_;
  Vector6d log_k_max_;
  Vector6d log_k_state_{Vector6d::Zero()};
  bool initialized_{false};
};
