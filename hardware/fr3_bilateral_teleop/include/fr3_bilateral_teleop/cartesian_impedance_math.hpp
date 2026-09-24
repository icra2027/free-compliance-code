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

#pragma once

#include <array>

#include <Eigen/Eigen>
#include <Eigen/Geometry>

namespace fr3_bilateral_teleop
{

using Vector6d = Eigen::Matrix<double, 6, 1>;
using Vector7d = Eigen::Matrix<double, 7, 1>;
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Matrix7d = Eigen::Matrix<double, 7, 7>;
using Matrix6x7d = Eigen::Matrix<double, 6, 7>;

// Pure, standalone Cartesian-impedance math. NOT wired into a controller in this package --
// the variable-impedance Cartesian controller is implemented by patching
// the sibling `variable_impedance_controllers` package (see ../../variable_impedance_controllers/, and
// README for the decision record) rather than by a from-scratch controller here,
// since variable_impedance_controllers already provides a mature, FR3-validated Cartesian
// impedance/OSC engine built specifically for VLA policy deployment. This header (and its
// test, test_cartesian_impedance_math.cpp) is kept as an independent cross-check: a second,
// differently-derived implementation of the same pose-error/operational-space-mass/
// nullspace-projector math variable_impedance_controllers computes via Pinocchio, useful for sanity-
// checking that engine's numbers rather than for driving hardware itself. Deliberately has
// no ROS/franka_semantic_components dependency so it can be exercised with plain Eigen
// fixtures, the same "standalone, unit-tested building block" pattern already used by
// stiffness_rate_limiter.hpp and energy_tank.hpp -- both of which now ALSO have copies
// in variable_impedance_controllers/include/variable_impedance_controllers/utils/ (patched in to satisfy
// §4.3's log-space rate-limit + energy-tank requirement, which variable_impedance_controllers' own
// EMA/torque-rate-saturation smoothing does not provide).

// Six-dimensional pose error e = x_d ⊖ x, in the SAME convention as the extraction
// pipeline's e(t) = x_l(t) ⊖ x_f(t) and the controller law
// f = K·e + D·ė: positive e means the desired/equilibrium pose is
// AHEAD of the current one, so F = K·e is a restoring force/torque toward x_d.
//
// Translation: plain vector difference (p_d - p), expressed in the base frame -- accurate
// to encoder precision, no approximation.
//
// Rotation: the standard quaternion-vector-part construction used throughout the
// franka_ros/franka_ros2 example controllers (e.g. the widely-deployed
// cartesian_impedance_example_controller formula), NOT an exact SO(3) log map. For small
// angular errors sin(theta/2) ~= theta/2, so this under-reports the true rotation error by
// a roughly-2x factor near zero and saturates (rather than diverging) as the error grows
// toward pi -- a deliberate, well-precedented choice: it is smooth and singularity-free
// everywhere except a true 180 degree flip, at the cost of not being metrically exact. Callers
// that need the literal log-map convention from §4.1's extraction pipeline for a downstream
// comparison (e.g. commanded-vs-realized-K validation) should account for that ~2x factor
// rather than assuming the two are numerically identical.
inline Vector6d poseError6d(
  const Eigen::Vector3d & p_d, const Eigen::Quaterniond & q_d_in,
  const Eigen::Vector3d & p, const Eigen::Quaterniond & q_in)
{
  Eigen::Quaterniond q_cur = q_in.normalized();
  Eigen::Quaterniond q_des = q_d_in.normalized();
  // Quaternions double-cover SO(3): q and -q represent the same rotation. Without this,
  // a demonstration/rollout that happens to cross the -q/+q boundary would see a
  // spurious ~360 degree "error" and the arm would try to spin the long way around.
  if (q_des.coeffs().dot(q_cur.coeffs()) < 0.0) {
    q_des.coeffs() = -q_des.coeffs();
  }
  const Eigen::Quaterniond q_err = q_cur.inverse() * q_des;
  const Eigen::Vector3d rot_error_base = q_cur.toRotationMatrix() * q_err.vec();

  Vector6d error;
  error.head<3>() = p_d - p;
  error.tail<3>() = rot_error_base;
  return error;
}

// Operational-space (task-space) mass matrix Lambda = (J M^-1 J^T)^-1 (Khatib 1987).
// `epsilon` is a tiny Tikhonov regularizer applied before inversion so a near-singular
// configuration degrades to a large-but-finite effective mass instead of a crash; it should
// never be tuned to matter away from a true singularity.
inline Matrix6d operationalSpaceMass(
  const Matrix6x7d & jacobian, const Matrix7d & mass_matrix, double epsilon = 1e-6)
{
  const Matrix7d mass_inv = mass_matrix.ldlt().solve(Matrix7d::Identity());
  const Matrix6d lambda_inv = jacobian * mass_inv * jacobian.transpose();
  return (lambda_inv + epsilon * Matrix6d::Identity()).inverse();
}

// Dynamically consistent nullspace projector N = I - M^-1 J^T Lambda J (Khatib 1987):
// projecting any joint torque through N^T leaves the task-space wrench it produces
// unchanged (J * N == 0), so a posture-holding torque can be added in the nullspace of the
// Cartesian impedance task without fighting it.
inline Matrix7d dynamicallyConsistentNullspaceProjector(
  const Matrix6x7d & jacobian, const Matrix7d & mass_matrix, double epsilon = 1e-6)
{
  const Matrix7d mass_inv = mass_matrix.ldlt().solve(Matrix7d::Identity());
  const Matrix6d lambda = operationalSpaceMass(jacobian, mass_matrix, epsilon);
  return Matrix7d::Identity() - mass_inv * jacobian.transpose() * lambda * jacobian;
}

// Per-axis critical(-ish) damping D_i = 2*zeta*sqrt(k_i * m_i). `zeta` is
// expected in the [0.7, 1.0] range but is not clamped here -- that policy
// decision belongs to the caller (see the controller's `damping_ratio` parameter).
inline Vector6d criticalDamping(
  const Vector6d & stiffness, const Vector6d & task_space_mass_diag, double zeta)
{
  return 2.0 * zeta * (stiffness.array() * task_space_mass_diag.array().max(0.0)).sqrt().matrix();
}

// Extracts position + orientation from a franka::Model pose matrix (column-major 4x4, as
// returned by e.g. FrankaRobotModel::getPoseMatrix -- see franka::Model::pose). Delegates to
// Eigen::Quaterniond's own rotation-matrix constructor rather than a hand-rolled trace-based
// conversion (contrast payload_model_broadcaster.cpp, which avoids an Eigen dependency on
// purpose and so hand-rolls the same conversion); this file already depends on Eigen
// throughout, so there is no equivalent reason to duplicate that logic here.
inline void poseFromColumnMajor4x4(
  const std::array<double, 16> & transform_matrix,
  Eigen::Vector3d & position, Eigen::Quaterniond & orientation)
{
  const Eigen::Map<const Eigen::Matrix4d> transform(transform_matrix.data());
  position = transform.block<3, 1>(0, 3);
  orientation = Eigen::Quaterniond(transform.block<3, 3>(0, 0));
}

}  // namespace fr3_bilateral_teleop
