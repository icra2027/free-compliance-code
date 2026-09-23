// Copyright (c) 2026 Franka Robotics GmbH
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cmath>

#include <gtest/gtest.h>
#include <Eigen/Eigen>

#include "fr3_bilateral_teleop/stiffness_rate_limiter.hpp"

using fr3_bilateral_teleop::LogSpaceStiffnessRateLimiter;
using fr3_bilateral_teleop::Vector6d;

namespace
{
Vector6d constant_vector(double value)
{
  return Vector6d::Constant(value);
}
}  // namespace

// reset() should set the internal state directly (no rate limiting -- there is no sane
// "previous stiffness" to limit from yet) and flip initialized() to true.
TEST(StiffnessRateLimiterTest, ResetSetsCurrentAndInitializedFlag)
{
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
// straight to the (clamped) commanded value with no rate limiting applied. Rate-limiting a
// controller's very first output toward some undefined previous state would be meaningless.
TEST(StiffnessRateLimiterTest, FirstStepBehavesLikeReset)
{
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
TEST(StiffnessRateLimiterTest, StepRateLimitsLargeJump)
{
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
    // Must have moved toward the target, but nowhere near reaching it in one small step.
    EXPECT_GT(result[i], 100.0);
    EXPECT_LT(result[i], 1500.0);
  }
}

// Repeated step() calls toward a fixed target should converge to it once enough control
// cycles have accumulated more than the needed |d(log k)| budget.
TEST(StiffnessRateLimiterTest, StepConvergesAfterEnoughCalls)
{
  const double gamma = 5.0;
  const double dt = 0.01;
  LogSpaceStiffnessRateLimiter limiter(
    constant_vector(gamma), constant_vector(1.0), constant_vector(2000.0));
  limiter.reset(constant_vector(100.0));

  const Vector6d target = constant_vector(400.0);
  Vector6d result = limiter.current();
  // |log(400) - log(100)| ~= 1.386; at gamma*dt = 0.05 per step this needs ~28 steps. Give it
  // generous headroom.
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
TEST(StiffnessRateLimiterTest, RateLimitIsScaleInvariant)
{
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
TEST(StiffnessRateLimiterTest, OutputClampedToKMinKMax)
{
  const Vector6d k_min = constant_vector(50.0);
  const Vector6d k_max = constant_vector(1500.0);
  LogSpaceStiffnessRateLimiter limiter(constant_vector(10.0), k_min, k_max);

  // reset() with an out-of-range value clamps immediately.
  limiter.reset(constant_vector(5.0));
  for (int i = 0; i < 6; ++i) {
    EXPECT_NEAR(limiter.current()[i], 50.0, 1e-6);
  }

  // Many steps toward an out-of-range target converge to the bound, never past it.
  Vector6d result = limiter.current();
  for (int i = 0; i < 500; ++i) {
    result = limiter.step(constant_vector(1.0e7), 0.01);
  }
  for (int i = 0; i < 6; ++i) {
    EXPECT_NEAR(result[i], 1500.0, 1e-3);
    EXPECT_LE(result[i], 1500.0 + 1e-9);
  }
}

int main(int argc, char ** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
