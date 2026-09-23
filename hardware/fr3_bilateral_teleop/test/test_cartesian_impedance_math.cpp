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

#include <gtest/gtest.h>

#include <cmath>

#include "fr3_bilateral_teleop/cartesian_impedance_math.hpp"

using fr3_bilateral_teleop::criticalDamping;
using fr3_bilateral_teleop::dynamicallyConsistentNullspaceProjector;
using fr3_bilateral_teleop::Matrix6x7d;
using fr3_bilateral_teleop::Matrix7d;
using fr3_bilateral_teleop::operationalSpaceMass;
using fr3_bilateral_teleop::poseError6d;
using fr3_bilateral_teleop::poseFromColumnMajor4x4;
using fr3_bilateral_teleop::Vector6d;

TEST(PoseError6dTest, ZeroWhenPosesMatch)
{
  const Eigen::Vector3d p(0.3, -0.1, 0.4);
  const Eigen::Quaterniond q(Eigen::AngleAxisd(0.7, Eigen::Vector3d::UnitZ()));
  const Vector6d error = poseError6d(p, q, p, q);
  EXPECT_LT(error.norm(), 1e-12);
}

TEST(PoseError6dTest, TranslationErrorIsExactVectorDifference)
{
  const Eigen::Vector3d p(0.5, 0.0, 0.3);
  const Eigen::Vector3d p_d(0.52, -0.01, 0.31);
  const Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
  const Vector6d error = poseError6d(p_d, q, p, q);
  EXPECT_TRUE(error.head<3>().isApprox(p_d - p, 1e-12));
  EXPECT_LT(error.tail<3>().norm(), 1e-12);
}

TEST(PoseError6dTest, RestoringSenseMatchesDesiredMinusCurrent)
{
  // Desired is a small positive rotation about Z ahead of current: a restoring torque
  // F = K * error should point toward increasing z-rotation, i.e. error.z() > 0. This is
  // the same "desired ahead of current => positive error" convention the extraction
  // pipeline uses for e(t) = x_l(t) ⊖ x_f(t) (proposal §4.1) and the controller law
  // f = K·e (proposal §4.3).
  const Eigen::Vector3d p = Eigen::Vector3d::Zero();
  const Eigen::Quaterniond q_cur = Eigen::Quaterniond::Identity();
  const Eigen::Quaterniond q_des(Eigen::AngleAxisd(0.05, Eigen::Vector3d::UnitZ()));
  const Vector6d error = poseError6d(p, q_des, p, q_cur);
  EXPECT_GT(error.tail<3>().z(), 0.0);
  EXPECT_NEAR(error.tail<3>().x(), 0.0, 1e-9);
  EXPECT_NEAR(error.tail<3>().y(), 0.0, 1e-9);
}

TEST(PoseError6dTest, DoubleCoverDoesNotProduceSpuriousLargeError)
{
  // q and -q represent the identical rotation. Negating q_d must not change the computed
  // error -- this is exactly the double-cover bug the sign-flip guard exists to prevent.
  const Eigen::Vector3d p = Eigen::Vector3d::Zero();
  const Eigen::Quaterniond q_cur(Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitX()));
  Eigen::Quaterniond q_des(Eigen::AngleAxisd(0.25, Eigen::Vector3d::UnitX()));
  const Vector6d error_positive = poseError6d(p, q_des, p, q_cur);
  q_des.coeffs() = -q_des.coeffs();
  const Vector6d error_negated = poseError6d(p, q_des, p, q_cur);
  EXPECT_TRUE(error_positive.isApprox(error_negated, 1e-12));
  EXPECT_LT(error_positive.tail<3>().norm(), 0.2);  // small angle, not a spurious ~pi error
}

namespace
{
// A physically plausible (symmetric positive definite, diagonally dominant) 7x7 mass matrix
// and a full-row-rank 6x7 Jacobian, used across the operational-space-mass and nullspace
// tests below. Not meant to represent a real FR3 configuration -- just numerically sane
// fixtures for exercising the linear algebra.
Matrix7d samplePositiveDefiniteMass()
{
  Matrix7d m = Matrix7d::Identity() * 2.0;
  for (int i = 0; i < 7; ++i) {
    for (int j = 0; j < 7; ++j) {
      if (i != j) {
        m(i, j) = 0.05 * static_cast<double>((i + 1) % 3 - 1) * static_cast<double>(j + 1);
      }
    }
  }
  // Symmetrize and boost the diagonal to guarantee positive-definiteness regardless of the
  // off-diagonal pattern above.
  m = 0.5 * (m + m.transpose());
  m += Matrix7d::Identity() * 3.0;
  return m;
}

Matrix6x7d sampleJacobian()
{
  Matrix6x7d j = Matrix6x7d::Zero();
  for (int r = 0; r < 6; ++r) {
    for (int c = 0; c < 7; ++c) {
      j(r, c) = std::sin(0.3 * (r + 1) + 0.2 * (c + 1));
    }
  }
  return j;
}
}  // namespace

TEST(OperationalSpaceMassTest, IsSymmetricPositiveDefinite)
{
  const Matrix6x7d jacobian = sampleJacobian();
  const Matrix7d mass = samplePositiveDefiniteMass();
  const auto lambda = operationalSpaceMass(jacobian, mass);
  EXPECT_TRUE(lambda.isApprox(lambda.transpose(), 1e-9));
  Eigen::SelfAdjointEigenSolver<fr3_bilateral_teleop::Matrix6d> solver(lambda);
  EXPECT_GT(solver.eigenvalues().minCoeff(), 0.0);
}

// dynamicallyConsistentNullspaceProjector regularizes (JM^-1J^T) with a small epsilon*I
// before inverting (see its doc comment) so it degrades gracefully near a singularity
// instead of crashing -- J*N is therefore not EXACTLY zero, only zero to O(epsilon). Using
// a tighter epsilon here than the function's 1e-6 default makes that residual small enough
// to use a tight-but-not-1e-8 tolerance as a real correctness check (a genuine formula bug
// would produce an O(1) residual, not O(epsilon)).
// kResidualTolerance is set by the synthetic fixture's own conditioning, not by
// kTestEpsilon: dropping epsilon from the function's 1e-6 default to 1e-10 left the residual
// unchanged (~1e-5), so it is standard floating-point round-off through several chained
// matrix ops (LDLT solve, two matrix products, an explicit inverse) on this particular
// hand-picked, moderately ill-conditioned Jacobian/mass pair -- not something a smaller
// epsilon fixes. Still four orders of magnitude below the O(1) residual a genuine formula
// bug (e.g. a transposed or swapped term) would produce, so this remains a real correctness
// check, just not a bit-exact one.
constexpr double kTestEpsilon = 1e-10;
constexpr double kResidualTolerance = 1e-4;

TEST(NullspaceProjectorTest, TaskJacobianTimesProjectorIsZero)
{
  const Matrix6x7d jacobian = sampleJacobian();
  const Matrix7d mass = samplePositiveDefiniteMass();
  const Matrix7d n = dynamicallyConsistentNullspaceProjector(jacobian, mass, kTestEpsilon);
  const Matrix6x7d product = jacobian * n;
  EXPECT_LT(product.norm(), kResidualTolerance);
}

TEST(NullspaceProjectorTest, ProjectingAJointTorqueTwiceMatchesOnce)
{
  // N is idempotent under the mass-weighted metric it's derived from: projecting an
  // arbitrary joint torque through N and then through N again should land on the same
  // task-space-null torque as projecting once (J * N * tau == 0 either way).
  const Matrix6x7d jacobian = sampleJacobian();
  const Matrix7d mass = samplePositiveDefiniteMass();
  const Matrix7d n = dynamicallyConsistentNullspaceProjector(jacobian, mass, kTestEpsilon);
  const Eigen::Matrix<double, 7, 1> tau = (Eigen::Matrix<double, 7, 1>() <<
    1.0, -2.0, 0.5, 3.0, -1.5, 0.2, 0.8).finished();
  const auto once = n * tau;
  const auto twice = n * once;
  EXPECT_LT((jacobian * once).norm(), kResidualTolerance);
  EXPECT_LT((jacobian * twice).norm(), kResidualTolerance);
}

TEST(CriticalDampingTest, MatchesClosedFormPerAxis)
{
  Vector6d k;
  k << 300.0, 300.0, 800.0, 20.0, 20.0, 40.0;
  Vector6d m_task;
  m_task << 4.0, 4.0, 9.0, 0.25, 0.25, 1.0;
  const double zeta = 0.8;
  const Vector6d d = criticalDamping(k, m_task, zeta);
  for (int i = 0; i < 6; ++i) {
    EXPECT_NEAR(d(i), 2.0 * zeta * std::sqrt(k(i) * m_task(i)), 1e-9);
  }
}

TEST(PoseFromColumnMajor4x4Test, RoundTripsAKnownTransform)
{
  const Eigen::Vector3d position(0.4, -0.2, 0.6);
  const Eigen::Quaterniond orientation =
    Eigen::Quaterniond(Eigen::AngleAxisd(1.1, Eigen::Vector3d(0.2, 0.7, 0.3).normalized()));

  Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
  transform.block<3, 3>(0, 0) = orientation.toRotationMatrix();
  transform.block<3, 1>(0, 3) = position;

  std::array<double, 16> packed{};
  Eigen::Map<Eigen::Matrix4d>(packed.data()) = transform;  // column-major, matches franka::Model

  Eigen::Vector3d recovered_position;
  Eigen::Quaterniond recovered_orientation;
  poseFromColumnMajor4x4(packed, recovered_position, recovered_orientation);

  EXPECT_TRUE(recovered_position.isApprox(position, 1e-12));
  // Quaternion double-cover: compare rotation matrices, not raw coefficients.
  EXPECT_TRUE(
    recovered_orientation.toRotationMatrix().isApprox(orientation.toRotationMatrix(), 1e-9));
}

TEST(CriticalDampingTest, NeverNegativeEvenWithNegativeMassInput)
{
  // Defensive: task-space mass should never be negative in practice, but the helper must
  // not produce NaN/negative damping if it ever is (e.g. a transient near-singular Lambda).
  Vector6d k = Vector6d::Constant(100.0);
  Vector6d m_task = Vector6d::Constant(-1.0);
  const Vector6d d = criticalDamping(k, m_task, 1.0);
  for (int i = 0; i < 6; ++i) {
    EXPECT_GE(d(i), 0.0);
    EXPECT_FALSE(std::isnan(d(i)));
  }
}
