#include <cmath>

#include <gtest/gtest.h>
#include <Eigen/Eigen>

#include "variable_impedance_controllers/utils/log_space_stiffness_rate_limiter.hpp"

namespace {
LogSpaceStiffnessRateLimiter::Vector6d constant_vector(double value) {
  return LogSpaceStiffnessRateLimiter::Vector6d::Constant(value);
}
}  // namespace

using Vector6d = LogSpaceStiffnessRateLimiter::Vector6d;

// reset() should set the internal state directly (no rate limiting -- there is no sane
// "previous stiffness" to limit from yet) and flip initialized() to true.
TEST(LogSpaceStiffnessRateLimiterTest, ResetSetsCurrentAndInitializedFlag) {
  LogSpaceStiffnessRateLimiter limiter(
    constant_vector(3.0), constant_vector(50.0), constant_vector(1500.0));
  EXPECT_FALSE(limiter.initialized());

  limiter.reset(constant_vector(200.0));

  EXPECT_TRUE(limiter.initialized());
  for (int i = 0; i < 6; ++i) {
    EXPECT_NEAR(limiter.current()[i], 200.0, 1e-9);
  }
}

// The very first call to step(), with no prior reset(), must behave like reset(): jump
// straight to the (clamped) commanded value with no rate limiting applied.
TEST(LogSpaceStiffnessRateLimiterTest, FirstStepBehavesLikeReset) {
  LogSpaceStiffnessRateLimiter limiter(
    constant_vector(1.0), constant_vector(50.0), constant_vector(1500.0));
  EXPECT_FALSE(limiter.initialized());

  const Vector6d commanded = constant_vector(900.0);
  const Vector6d result = limiter.step(commanded, 0.001);

  EXPECT_TRUE(limiter.initialized());
  for (int i = 0; i < 6; ++i) {
    EXPECT_NEAR(result[i], 900.0, 1e-6);
  }
}

// After initialization, a single step() toward a target far away must move by at most
// gamma * dt in LOG space -- it must not jump straight to the target in one control cycle.
TEST(LogSpaceStiffnessRateLimiterTest, StepRateLimitsLargeJump) {
  const double gamma = 2.0;  // 1/s
  const double dt = 0.001;   // 1 kHz control loop
  LogSpaceStiffnessRateLimiter limiter(
    constant_vector(gamma), constant_vector(50.0), constant_vector(1500.0));
  limiter.reset(constant_vector(100.0));

  const Vector6d result = limiter.step(constant_vector(1500.0), dt);

  const double expected_log_step = gamma * dt;
  const double expected_value = 100.0 * std::exp(expected_log_step);
  for (int i = 0; i < 6; ++i) {
    EXPECT_NEAR(result[i], expected_value, 1e-6);
    EXPECT_GT(result[i], 100.0);
    EXPECT_LT(result[i], 1500.0);
  }
}

// Repeated step() calls toward a fixed target should converge to it once enough control
// cycles have accumulated more than the needed |d(log k)| budget.
TEST(LogSpaceStiffnessRateLimiterTest, StepConvergesAfterEnoughCalls) {
  const double gamma = 5.0;
  const double dt = 0.01;
  LogSpaceStiffnessRateLimiter limiter(
    constant_vector(gamma), constant_vector(1.0), constant_vector(2000.0));
  limiter.reset(constant_vector(100.0));

  const Vector6d target = constant_vector(400.0);
  Vector6d result = limiter.current();
  for (int i = 0; i < 200; ++i) {
    result = limiter.step(target, dt);
  }

  for (int i = 0; i < 6; ++i) {
    EXPECT_NEAR(result[i], 400.0, 1e-3);
  }
}

// The rate limit is in LOG space specifically so one gamma is equally aggressive/conservative
// at low and high stiffness -- the multiplicative ratio moved in one step should be identical
// regardless of the starting magnitude, as long as neither is near the k_min/k_max clamp.
TEST(LogSpaceStiffnessRateLimiterTest, RateLimitIsScaleInvariant) {
  const double gamma = 1.0;
  const double dt = 0.05;

  LogSpaceStiffnessRateLimiter low_limiter(
    constant_vector(gamma), constant_vector(1.0), constant_vector(1.0e6));
  low_limiter.reset(constant_vector(10.0));
  const Vector6d low_result = low_limiter.step(constant_vector(1.0e5), dt);

  LogSpaceStiffnessRateLimiter high_limiter(
    constant_vector(gamma), constant_vector(1.0), constant_vector(1.0e6));
  high_limiter.reset(constant_vector(10000.0));
  const Vector6d high_result = high_limiter.step(constant_vector(1.0e8), dt);

  const double low_ratio = low_result[0] / 10.0;
  const double high_ratio = high_result[0] / 10000.0;
  EXPECT_NEAR(low_ratio, high_ratio, 1e-6);
  EXPECT_NEAR(low_ratio, std::exp(gamma * dt), 1e-6);
}

// Both reset() and step() must clamp their output into [k_min, k_max] -- the rate limiter
// must never be bypassable by simply commanding an out-of-range target.
TEST(LogSpaceStiffnessRateLimiterTest, OutputClampedToKMinKMax) {
  const Vector6d k_min = constant_vector(50.0);
  const Vector6d k_max = constant_vector(1500.0);
  LogSpaceStiffnessRateLimiter limiter(constant_vector(10.0), k_min, k_max);

  limiter.reset(constant_vector(5.0));
  for (int i = 0; i < 6; ++i) {
    EXPECT_NEAR(limiter.current()[i], 50.0, 1e-6);
  }

  Vector6d result = limiter.current();
  for (int i = 0; i < 500; ++i) {
    result = limiter.step(constant_vector(1.0e7), 0.01);
  }
  for (int i = 0; i < 6; ++i) {
    EXPECT_NEAR(result[i], 1500.0, 1e-3);
    EXPECT_LE(result[i], 1500.0 + 1e-9);
  }
}

int main(int argc, char ** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
