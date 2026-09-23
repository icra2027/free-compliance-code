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

#include <fr3_bilateral_teleop/payload_model_broadcaster.hpp>

#include <algorithm>
#include <cmath>

namespace fr3_bilateral_teleop
{

controller_interface::InterfaceConfiguration
PayloadModelBroadcaster::command_interface_configuration() const
{
  return controller_interface::InterfaceConfiguration{
    controller_interface::interface_configuration_type::NONE};
}

controller_interface::InterfaceConfiguration
PayloadModelBroadcaster::state_interface_configuration() const
{
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (const auto & name : franka_robot_model_->get_state_interface_names()) {
    config.names.push_back(name);
  }
  return config;
}

controller_interface::CallbackReturn PayloadModelBroadcaster::on_init()
{
  try {
    auto_declare<std::string>("robot_type", "fr3");
    auto_declare<std::string>("arm_prefix", "");
    // franka::Model evaluation (mass/coriolis/jacobian) is too heavy to run at the full
    // controller_manager update rate; decimate publishing instead of publishing every update().
    auto_declare<int>("publish_decimation", 10);
  } catch (const std::exception & e) {
    fprintf(stderr, "Exception thrown during init stage with message: %s \n", e.what());
    return CallbackReturn::ERROR;
  }
  return CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn PayloadModelBroadcaster::on_configure(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  robot_type_ = get_node()->get_parameter("robot_type").as_string();
  arm_prefix_ = get_node()->get_parameter("arm_prefix").as_string();
  arm_prefix_ = arm_prefix_.empty() ? "" : arm_prefix_ + "_";
  publish_decimation_ = get_node()->get_parameter("publish_decimation").as_int();
  if (publish_decimation_ < 1) {
    publish_decimation_ = 1;
  }

  franka_robot_model_ = std::make_unique<franka_semantic_components::FrankaRobotModel>(
    franka_semantic_components::FrankaRobotModel(
      arm_prefix_ + robot_type_ + "/" + k_robot_model_interface_name,
      arm_prefix_ + robot_type_ + "/" + k_robot_state_interface_name));

  snapshot_publisher_ = get_node()->create_publisher<std_msgs::msg::Float64MultiArray>(
    "~/model_snapshot", rclcpp::QoS(10).best_effort());

  return CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn PayloadModelBroadcaster::on_activate(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  franka_robot_model_->assign_loaned_state_interfaces(state_interfaces_);
  update_counter_ = 0;
  return CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn PayloadModelBroadcaster::on_deactivate(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  franka_robot_model_->release_interfaces();
  return CallbackReturn::SUCCESS;
}

controller_interface::return_type PayloadModelBroadcaster::update(
  const rclcpp::Time & time,
  const rclcpp::Duration & /*period*/)
{
  if (++update_counter_ % publish_decimation_ != 0) {
    return controller_interface::return_type::OK;
  }

  const std::array<double, 7> gravity = franka_robot_model_->getGravityForceVector();
  const std::array<double, 7> coriolis = franka_robot_model_->getCoriolisForceVector();
  const std::array<double, 49> mass = franka_robot_model_->getMassMatrix();
  const std::array<double, 42> jacobian =
    franka_robot_model_->getZeroJacobian(franka::Frame::kFlange);
  const std::array<double, 16> flange_pose =
    franka_robot_model_->getPoseMatrix(franka::Frame::kFlange);

  // Column-major 4x4: translation is column 3 (indices 12-14); rotation is the top-left
  // 3x3 block (columns 0-2, rows 0-2). Quaternion computed by hand (no Eigen dependency
  // needed here) via the standard trace-based conversion.
  const double r00 = flange_pose[0];
  const double r10 = flange_pose[1];
  const double r20 = flange_pose[2];
  const double r01 = flange_pose[4];
  const double r11 = flange_pose[5];
  const double r21 = flange_pose[6];
  const double r02 = flange_pose[8];
  const double r12 = flange_pose[9];
  const double r22 = flange_pose[10];
  const double trace = r00 + r11 + r22;

  double qw = 0.0;
  double qx = 0.0;
  double qy = 0.0;
  double qz = 0.0;
  if (trace > 0.0) {
    const double s = 0.5 / std::sqrt(trace + 1.0);
    qw = 0.25 / s;
    qx = (r21 - r12) * s;
    qy = (r02 - r20) * s;
    qz = (r10 - r01) * s;
  } else if (r00 > r11 && r00 > r22) {
    const double s = 2.0 * std::sqrt(1.0 + r00 - r11 - r22);
    qw = (r21 - r12) / s;
    qx = 0.25 * s;
    qy = (r01 + r10) / s;
    qz = (r02 + r20) / s;
  } else if (r11 > r22) {
    const double s = 2.0 * std::sqrt(1.0 + r11 - r00 - r22);
    qw = (r02 - r20) / s;
    qx = (r01 + r10) / s;
    qy = 0.25 * s;
    qz = (r12 + r21) / s;
  } else {
    const double s = 2.0 * std::sqrt(1.0 + r22 - r00 - r11);
    qw = (r10 - r01) / s;
    qx = (r02 + r20) / s;
    qy = (r12 + r21) / s;
    qz = 0.25 * s;
  }

  std_msgs::msg::Float64MultiArray msg;
  msg.data.reserve(kSnapshotLength);
  msg.data.push_back(time.seconds());
  msg.data.insert(msg.data.end(), gravity.begin(), gravity.end());
  msg.data.insert(msg.data.end(), coriolis.begin(), coriolis.end());
  msg.data.insert(msg.data.end(), mass.begin(), mass.end());
  msg.data.insert(msg.data.end(), jacobian.begin(), jacobian.end());
  msg.data.push_back(flange_pose[12]);
  msg.data.push_back(flange_pose[13]);
  msg.data.push_back(flange_pose[14]);
  msg.data.push_back(qx);
  msg.data.push_back(qy);
  msg.data.push_back(qz);
  msg.data.push_back(qw);

  snapshot_publisher_->publish(msg);

  return controller_interface::return_type::OK;
}

}  // namespace fr3_bilateral_teleop

#include "pluginlib/class_list_macros.hpp"
// NOLINTNEXTLINE
PLUGINLIB_EXPORT_CLASS(
  fr3_bilateral_teleop::PayloadModelBroadcaster,
  controller_interface::ControllerInterface)
