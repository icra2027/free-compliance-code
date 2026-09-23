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

#include <string>
#include <cassert>
#include <cmath>
#include <exception>

#include <Eigen/Eigen>
#include <fr3_bilateral_teleop/teleop_follower_controller.hpp>
#include <fr3_bilateral_teleop/utils.hpp>

namespace fr3_bilateral_teleop
{

controller_interface::InterfaceConfiguration
TeleopFollowerController::command_interface_configuration() const
{
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  for (unsigned int i = 1; i <= teleop_utils::NUM_JOINTS; ++i) {
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/effort");
  }
  return config;
}

controller_interface::InterfaceConfiguration
TeleopFollowerController::state_interface_configuration() const
{
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (unsigned int i = 1; i <= teleop_utils::NUM_JOINTS; ++i) {
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/position");
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/velocity");
  }
  return config;
}

controller_interface::return_type TeleopFollowerController::update(
  const rclcpp::Time & /*time*/, const rclcpp::Duration & period)
{
  if (loop_period_publisher_ && loop_period_publisher_->trylock()) {
    loop_period_publisher_->msg_.data = period.seconds() * 1.0e6;
    loop_period_publisher_->unlockAndPublish();
  }

  updateJointStates();
  Vector7d q_goal_target = initial_q_;

  const auto input = measured_joint_states_from_leader_buffer_ptr_.readFromRT();

  if (teleop_utils::check_if_rt_buffer_data_is_valid(input)) {
    if (position_channel_latency_publisher_ && position_channel_latency_publisher_->trylock()) {
      position_channel_latency_publisher_->msg_.data =
        teleop_utils::message_age_microseconds(get_node(), input);
      position_channel_latency_publisher_->unlockAndPublish();
    }

    if (teleop_utils::check_if_message_too_old(get_node(), input, input_topic_timeout_)) {
      teleop_utils::set_command_interfaces_to_gravity_compensation(command_interfaces_);

      RCLCPP_ERROR(this->get_node()->get_logger(), "Latest message is too old");
      return controller_interface::return_type::ERROR;
    }

    Eigen::Map<Eigen::VectorXd>(q_goal_target.data(), teleop_utils::NUM_JOINTS) =
      Eigen::Map<Eigen::VectorXd>((*input)->position.data(), teleop_utils::NUM_JOINTS);
  }

  // Slew-rate limit the goal so a discontinuity in q_goal_target (e.g. the raw leader
  // position jumping in the instant this controller reactivates, if the leader moved while
  // this controller was inactive) is ramped into rather than snapped to -- an unbounded
  // snap here turns straight into a torque spike below via k_gains_.
  const double max_step = max_goal_slew_rate_ * period.seconds();
  for (unsigned int i = 0; i < teleop_utils::NUM_JOINTS; ++i) {
    double step = q_goal_target(i) - q_goal_smoothed_(i);
    if (step > max_step) {
      step = max_step;
    } else if (step < -max_step) {
      step = -max_step;
    }
    q_goal_smoothed_(i) += step;
  }
  const Vector7d & q_goal = q_goal_smoothed_;

  // Force channel (leader -> follower): the leader's own sensed external torque
  // (i.e. the force the human operator applies at the leader) is scaled by
  // force_feedforward_gains_ and fed forward directly into the follower's torque
  // command. This is independent of the position channel (k_gains_/d_gains_), so
  // the operator's push has a direct, tunable effect on the follower that does not
  // require raising the position-tracking stiffness. Missing or stale data on this
  // channel degrades gracefully to zero feedforward rather than aborting, since the
  // position channel above remains the safety-critical one.
  Vector7d tau_ff = Vector7d::Zero();
  const auto force_input = external_joint_torques_from_leader_buffer_ptr_.readFromRT();
  if (force_feedforward_channel_latency_publisher_ &&
    teleop_utils::check_if_rt_buffer_data_is_valid(force_input) &&
    force_feedforward_channel_latency_publisher_->trylock())
  {
    force_feedforward_channel_latency_publisher_->msg_.data =
      teleop_utils::message_age_microseconds(get_node(), force_input);
    force_feedforward_channel_latency_publisher_->unlockAndPublish();
  }

  if (teleop_utils::check_if_rt_buffer_data_is_valid(force_input) &&
    !teleop_utils::check_if_message_too_old(get_node(), force_input, input_topic_timeout_))
  {
    Vector7d leader_external_torque;
    Eigen::Map<Eigen::VectorXd>(leader_external_torque.data(), teleop_utils::NUM_JOINTS) =
      Eigen::Map<const Eigen::VectorXd>((*force_input)->effort.data(), teleop_utils::NUM_JOINTS);
    tau_ff = force_feedforward_gains_.cwiseProduct(leader_external_torque);
  }

  // standard joint impedance controller from franka example controllers
  // the target position comes from the leader (position channel, tuned for tracking
  // via k_gains_/d_gains_), plus the force feedforward term above (force channel,
  // tuned for feel via force_feedforward_gains_).
  constexpr double kAlpha = 0.99;
  dq_filtered_ = (1 - kAlpha) * dq_filtered_ + kAlpha * dq_;
  Vector7d tau_d_calculated =
    k_gains_.cwiseProduct(q_goal - q_) + d_gains_.cwiseProduct(-dq_filtered_) + tau_ff;
  for (unsigned int i = 0; i < teleop_utils::NUM_JOINTS; ++i) {
    command_interfaces_[i].set_value(tau_d_calculated(i));
  }
  return controller_interface::return_type::OK;
}

CallbackReturn TeleopFollowerController::on_init()
{
  try {
    auto_declare<std::string>("arm_id", "");
    auto_declare<std::vector<double>>("k_gains", {});
    auto_declare<std::vector<double>>("d_gains", {});
    auto_declare<std::vector<double>>("force_feedforward_gains", {});
    auto_declare<std::string>("force_input_topic", "");
    auto_declare<double>("max_goal_slew_rate", 2.0);
  } catch (const std::exception & e) {
    RCLCPP_ERROR(
      this->get_node()->get_logger(), "Exception thrown during init stage with message: %s \n",
      e.what());
    return CallbackReturn::ERROR;
  }
  return CallbackReturn::SUCCESS;
}

CallbackReturn TeleopFollowerController::on_configure(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  arm_id_ = get_node()->get_parameter("arm_id").as_string();
  input_topic_ = get_node()->get_parameter("input_topic").as_string();
  auto k_gains = get_node()->get_parameter("k_gains").as_double_array();
  auto d_gains = get_node()->get_parameter("d_gains").as_double_array();
  if (k_gains.empty()) {
    RCLCPP_FATAL(get_node()->get_logger(), "k_gains parameter not set");
    return CallbackReturn::FAILURE;
  }
  if (k_gains.size() != static_cast<uint>(teleop_utils::NUM_JOINTS)) {
    RCLCPP_FATAL(
      get_node()->get_logger(), "k_gains should be of size %d but is of size %ld",
      teleop_utils::NUM_JOINTS, k_gains.size());
    return CallbackReturn::FAILURE;
  }
  if (d_gains.empty()) {
    RCLCPP_FATAL(get_node()->get_logger(), "d_gains parameter not set");
    return CallbackReturn::FAILURE;
  }
  if (d_gains.size() != static_cast<uint>(teleop_utils::NUM_JOINTS)) {
    RCLCPP_FATAL(
      get_node()->get_logger(), "d_gains should be of size %d but is of size %ld",
      teleop_utils::NUM_JOINTS, d_gains.size());
    return CallbackReturn::FAILURE;
  }
  auto force_feedforward_gains =
    get_node()->get_parameter("force_feedforward_gains").as_double_array();
  if (force_feedforward_gains.empty()) {
    RCLCPP_FATAL(get_node()->get_logger(), "force_feedforward_gains parameter not set");
    return CallbackReturn::FAILURE;
  }
  if (force_feedforward_gains.size() != static_cast<uint>(teleop_utils::NUM_JOINTS)) {
    RCLCPP_FATAL(
      get_node()->get_logger(),
      "force_feedforward_gains should be of size %d but is of size %ld",
      teleop_utils::NUM_JOINTS, force_feedforward_gains.size());
    return CallbackReturn::FAILURE;
  }
  for (unsigned int i = 0; i < teleop_utils::NUM_JOINTS; ++i) {
    d_gains_(i) = d_gains.at(i);
    k_gains_(i) = k_gains.at(i);
    force_feedforward_gains_(i) = force_feedforward_gains.at(i);
  }
  dq_filtered_.setZero();

  max_goal_slew_rate_ = get_node()->get_parameter("max_goal_slew_rate").as_double();
  if (max_goal_slew_rate_ <= 0.0) {
    RCLCPP_FATAL(get_node()->get_logger(), "max_goal_slew_rate must be positive");
    return CallbackReturn::FAILURE;
  }

  auto parameters_client =
    std::make_shared<rclcpp::AsyncParametersClient>(get_node(), "robot_state_publisher");
  parameters_client->wait_for_service();

  auto future = parameters_client->get_parameters({"robot_description"});
  auto result = future.get();
  if (!result.empty()) {
    robot_description_ = result[0].value_to_string();
  } else {
    RCLCPP_ERROR(get_node()->get_logger(), "Failed to get robot_description parameter.");
  }

  arm_id_ = teleop_utils::getRobotNameFromDescription(robot_description_, get_node()->get_logger());
  input_topic_ = get_node()->get_parameter("input_topic").as_string();
  RCLCPP_INFO(this->get_node()->get_logger(), "Using input_topic: %s", input_topic_.c_str());
  input_topic_timeout_ = get_node()->get_parameter("input_topic_timeout").as_int();
  RCLCPP_INFO(
    this->get_node()->get_logger(), "Using input_topic_timeout: %ld", input_topic_timeout_);

  loop_period_publisher_ = std::make_shared<Float64RtPublisher>(
    this->get_node()->create_publisher<std_msgs::msg::Float64>(
      "~/diagnostics/loop_period_us", rclcpp::SystemDefaultsQoS()));
  position_channel_latency_publisher_ = std::make_shared<Float64RtPublisher>(
    this->get_node()->create_publisher<std_msgs::msg::Float64>(
      "~/diagnostics/position_channel_latency_us", rclcpp::SystemDefaultsQoS()));
  force_feedforward_channel_latency_publisher_ = std::make_shared<Float64RtPublisher>(
    this->get_node()->create_publisher<std_msgs::msg::Float64>(
      "~/diagnostics/force_feedforward_channel_latency_us", rclcpp::SystemDefaultsQoS()));

  measured_joint_states_from_leader_subscriber_ =
    this->get_node()->create_subscription<sensor_msgs::msg::JointState>(
    input_topic_, rclcpp::SystemDefaultsQoS(),
    [this](const sensor_msgs::msg::JointState::SharedPtr msg) {
      // check if message is correct size, if not ignore
      if (msg->position.size() == teleop_utils::NUM_JOINTS) {
        measured_joint_states_from_leader_buffer_ptr_.writeFromNonRT(msg);
      } else {
        RCLCPP_ERROR(
          this->get_node()->get_logger(),
          "Invalid command received for %zu joints, expected "
          "command for %u size",
          msg->position.size(), teleop_utils::NUM_JOINTS);
      }
    });

  force_input_topic_ = get_node()->get_parameter("force_input_topic").as_string();
  RCLCPP_INFO(
    this->get_node()->get_logger(), "Using force_input_topic: %s", force_input_topic_.c_str());

  // Force channel (leader -> follower): subscribes to the leader's own sensed
  // external torque, i.e. the force the human operator applies at the leader.
  external_joint_torques_from_leader_subscriber_ =
    this->get_node()->create_subscription<sensor_msgs::msg::JointState>(
    force_input_topic_, rclcpp::SystemDefaultsQoS(),
    [this](const sensor_msgs::msg::JointState::SharedPtr msg) {
      // check if message is correct size, if not ignore
      if (msg->effort.size() == teleop_utils::NUM_JOINTS) {
        external_joint_torques_from_leader_buffer_ptr_.writeFromNonRT(msg);
      } else {
        RCLCPP_ERROR(
          this->get_node()->get_logger(),
          "Invalid command received for %zu joints, expected "
          "command for %u size",
          msg->effort.size(), teleop_utils::NUM_JOINTS);
      }
    });

  return CallbackReturn::SUCCESS;
}

CallbackReturn TeleopFollowerController::on_activate(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  updateJointStates();
  dq_filtered_.setZero();
  initial_q_ = q_;
  q_goal_smoothed_ = q_;
  elapsed_time_ = 0.0;

  measured_joint_states_from_leader_buffer_ptr_ =
    realtime_tools::RealtimeBuffer<std::shared_ptr<sensor_msgs::msg::JointState>>(nullptr);
  external_joint_torques_from_leader_buffer_ptr_ =
    realtime_tools::RealtimeBuffer<std::shared_ptr<sensor_msgs::msg::JointState>>(nullptr);

  return CallbackReturn::SUCCESS;
}

CallbackReturn TeleopFollowerController::on_deactivate(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  measured_joint_states_from_leader_buffer_ptr_ =
    realtime_tools::RealtimeBuffer<std::shared_ptr<sensor_msgs::msg::JointState>>(nullptr);
  external_joint_torques_from_leader_buffer_ptr_ =
    realtime_tools::RealtimeBuffer<std::shared_ptr<sensor_msgs::msg::JointState>>(nullptr);

  return controller_interface::CallbackReturn::SUCCESS;
}

void TeleopFollowerController::updateJointStates()
{
  for (unsigned int i = 0; i < teleop_utils::NUM_JOINTS; ++i) {
    const auto & position_interface = state_interfaces_.at(2 * i);
    const auto & velocity_interface = state_interfaces_.at(2 * i + 1);

    assert(position_interface.get_interface_name() == "position");
    assert(velocity_interface.get_interface_name() == "velocity");

    q_(i) = position_interface.get_value();
    dq_(i) = velocity_interface.get_value();
  }
}

}  // namespace fr3_bilateral_teleop

#include "pluginlib/class_list_macros.hpp"
// NOLINTNEXTLINE
PLUGINLIB_EXPORT_CLASS(
  fr3_bilateral_teleop::TeleopFollowerController, controller_interface::ControllerInterface)
