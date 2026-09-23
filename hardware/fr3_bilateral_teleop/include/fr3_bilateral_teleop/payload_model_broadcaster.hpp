// Copyright (c) 2025 Franka Robotics GmbH
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
#include <memory>
#include <string>

#include <controller_interface/controller_interface.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include "franka_semantic_components/franka_robot_model.hpp"

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

namespace fr3_bilateral_teleop
{

// Publishes the franka::Model quantities (evaluated at whatever payload is currently configured
// via set_load) needed to identify an UNDECLARED rigid-body payload offline: gravity vector,
// Coriolis vector, mass matrix, and the flange's Jacobian/pose in base frame. Not used during
// normal teleoperation -- only loaded by the payload calibration launch, since franka::Model
// evaluation is too heavyweight to run in every session's control loop.
//
// All quantities for one control cycle are packed into a single Float64MultiArray message
// (std_msgs::msg::Float64MultiArray has no header/stamp field) so a consumer never has to
// time-align several independently-timestamped topics. Layout of `~/model_snapshot.data`
// (length kSnapshotLength), all franka::Model arrays column-major as libfranka returns them:
//   [0]        stamp, seconds (rclcpp::Time, same clock as measured_joint_states)
//   [1:8]      gravity vector g(q)                            (7)
//   [8:15]     Coriolis vector c(q, dq)                        (7)
//   [15:64]    mass matrix M(q), column-major 7x7               (49)
//   [64:106]   flange zero Jacobian, column-major 6x7, base frame (42)
//   [106:109]  flange position (x, y, z), base frame             (3)
//   [109:113]  flange orientation quaternion (x, y, z, w)        (4)
class PayloadModelBroadcaster : public controller_interface::ControllerInterface
{
public:
  static constexpr std::size_t kSnapshotLength = 113;

  [[nodiscard]] controller_interface::InterfaceConfiguration command_interface_configuration()
  const override;

  [[nodiscard]] controller_interface::InterfaceConfiguration state_interface_configuration()
  const override;

  controller_interface::return_type update(
    const rclcpp::Time & time,
    const rclcpp::Duration & period) override;
  controller_interface::CallbackReturn on_init() override;

  controller_interface::CallbackReturn on_configure(
    const rclcpp_lifecycle::State & previous_state) override;

  controller_interface::CallbackReturn on_activate(
    const rclcpp_lifecycle::State & previous_state) override;

  controller_interface::CallbackReturn on_deactivate(
    const rclcpp_lifecycle::State & previous_state) override;

private:
  std::string robot_type_;
  std::string arm_prefix_;
  int publish_decimation_{1};
  int update_counter_{0};

  std::unique_ptr<franka_semantic_components::FrankaRobotModel> franka_robot_model_;

  const std::string k_robot_state_interface_name{"robot_state"};
  const std::string k_robot_model_interface_name{"robot_model"};

  std::shared_ptr<rclcpp::Publisher<std_msgs::msg::Float64MultiArray>> snapshot_publisher_;
};
}  // namespace fr3_bilateral_teleop
