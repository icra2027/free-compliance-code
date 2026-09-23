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

#include <algorithm>
#include <exception>
#include <string>

#include <fr3_bilateral_teleop/teleop_leader_controller.hpp>
#include <fr3_bilateral_teleop/utils.hpp>

namespace fr3_bilateral_teleop
{

controller_interface::InterfaceConfiguration
TeleopLeaderController::command_interface_configuration() const
{
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  for (unsigned int i = 1; i <= teleop_utils::NUM_JOINTS; ++i) {
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/effort");
  }
  return config;
}

template<typename T>
auto sgn(T val) -> int
{
  // returns the sign of the value: -1, 0 or +1
  return (T(0) < val) - (val < T(0));
}

controller_interface::InterfaceConfiguration TeleopLeaderController::state_interface_configuration()
const
{
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (unsigned int i = 1; i <= teleop_utils::NUM_JOINTS; ++i) {
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/position");
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/velocity");
  }
  return config;
}

controller_interface::return_type TeleopLeaderController::update(
  const rclcpp::Time & /*time*/, const rclcpp::Duration & period)
{
  if (loop_period_publisher_ && loop_period_publisher_->trylock()) {
    loop_period_publisher_->msg_.data = period.seconds() * 1.0e6;
    loop_period_publisher_->unlockAndPublish();
  }

  auto input = external_joint_torques_from_follower_buffer_ptr_.readFromRT();

  if (force_reflection_channel_latency_publisher_ &&
    teleop_utils::check_if_rt_buffer_data_is_valid(input) &&
    force_reflection_channel_latency_publisher_->trylock())
  {
    force_reflection_channel_latency_publisher_->msg_.data =
      teleop_utils::message_age_microseconds(get_node(), input);
    force_reflection_channel_latency_publisher_->unlockAndPublish();
  }

  std::vector<double> tau_in{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  updateJointStates();

  if (!use_input_topic_) {
    teleop_utils::set_command_interfaces_to_gravity_compensation(command_interfaces_);
    return controller_interface::return_type::OK;
  }

  if (use_input_topic_ && teleop_utils::check_if_rt_buffer_data_is_valid(input)) {
    if (teleop_utils::check_if_message_too_old(get_node(), input, input_topic_timeout_)) {
      teleop_utils::set_command_interfaces_to_gravity_compensation(command_interfaces_);

      RCLCPP_ERROR(this->get_node()->get_logger(), "Latest message is too old");
      return controller_interface::return_type::ERROR;
    }

    tau_in = (*input)->effort;
  }

  // Force channel (follower -> leader): the follower's sensed contact torque is
  // scaled by force_reflection_gains_ and reflected back to the leader so the
  // operator feels contact. This gain is independent of the follower's
  // position-tracking gains (k_gains/d_gains), so "feel" can be tuned without
  // compromising tracking accuracy.
  // The feedback-avoidance term guards against a feedback loop that would
  // otherwise result in the leader "jumping away" after contact of the follower;
  // it is computed on the actually-reflected (gain-scaled) torque so it stays
  // consistent with what is really being commanded.
  std::vector<double> leader_torque_commands(teleop_utils::NUM_JOINTS);
  for (unsigned int i = 0; i < teleop_utils::NUM_JOINTS; ++i) {
    const double reflected_tau = force_reflection_gains_[i] * tau_in[i];
    const double power_in = leader_velocity_[i] * reflected_tau;

    double feedback_avoidance_term = 0.0;

    if (power_in < 0) {
      feedback_avoidance_term =
        -feedback_avoidance_alpha_[i] * sgn(leader_velocity_[i]) * fabs(power_in);
    }

    leader_torque_commands[i] = -reflected_tau + feedback_avoidance_term;
  }

  teleop_utils::set_command_interfaces(command_interfaces_, leader_torque_commands);

  return controller_interface::return_type::OK;
}

CallbackReturn TeleopLeaderController::on_configure(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  arm_id_ = get_node()->get_parameter("arm_id").as_string();
  RCLCPP_INFO(this->get_node()->get_logger(), "Using arm_id: %s", arm_id_.c_str());

  input_topic_ = get_node()->get_parameter("input_topic").as_string();
  RCLCPP_INFO(this->get_node()->get_logger(), "Using input_topic: %s", input_topic_.c_str());

  input_topic_timeout_ = get_node()->get_parameter("input_topic_timeout").as_int();
  RCLCPP_INFO(
    this->get_node()->get_logger(), "Using input_topic_timeout: %ld", input_topic_timeout_);

  use_input_topic_ = get_node()->get_parameter("use_input_topic").as_bool();
  RCLCPP_INFO(
    this->get_node()->get_logger(), "Using use_input_topic: %s",
    use_input_topic_ ? "True" : "False");

  feedback_avoidance_alpha_ = get_node()->get_parameter("alpha").as_double_array();
  std::stringstream stream;
  stream << "Using alpha: [";
  for (unsigned int i = 0; i < teleop_utils::NUM_JOINTS - 1; ++i) {
    stream << feedback_avoidance_alpha_[i] << ", ";
  }
  stream << feedback_avoidance_alpha_[teleop_utils::NUM_JOINTS - 1] << "]";
  RCLCPP_INFO(this->get_node()->get_logger(), stream.str().c_str());

  force_reflection_gains_ = get_node()->get_parameter("force_reflection_gains").as_double_array();
  if (force_reflection_gains_.size() != teleop_utils::NUM_JOINTS) {
    RCLCPP_FATAL(
      get_node()->get_logger(),
      "force_reflection_gains should be of size %d but is of size %ld",
      teleop_utils::NUM_JOINTS, force_reflection_gains_.size());
    return CallbackReturn::FAILURE;
  }
  std::stringstream force_gain_stream;
  force_gain_stream << "Using force_reflection_gains: [";
  for (unsigned int i = 0; i < teleop_utils::NUM_JOINTS - 1; ++i) {
    force_gain_stream << force_reflection_gains_[i] << ", ";
  }
  force_gain_stream << force_reflection_gains_[teleop_utils::NUM_JOINTS - 1] << "]";
  RCLCPP_INFO(this->get_node()->get_logger(), force_gain_stream.str().c_str());

  loop_period_publisher_ = std::make_shared<Float64RtPublisher>(
    this->get_node()->create_publisher<std_msgs::msg::Float64>(
      "~/diagnostics/loop_period_us", rclcpp::SystemDefaultsQoS()));
  force_reflection_channel_latency_publisher_ = std::make_shared<Float64RtPublisher>(
    this->get_node()->create_publisher<std_msgs::msg::Float64>(
      "~/diagnostics/force_reflection_channel_latency_us", rclcpp::SystemDefaultsQoS()));

  if (use_input_topic_) {
    external_joint_torques_from_follower_subscriber_ =
      this->get_node()->create_subscription<sensor_msgs::msg::JointState>(
      input_topic_, rclcpp::SystemDefaultsQoS(),
      [this](const sensor_msgs::msg::JointState::SharedPtr msg) {
        if (msg->effort.size() == teleop_utils::NUM_JOINTS) {
          external_joint_torques_from_follower_buffer_ptr_.writeFromNonRT(msg);
        } else {
          RCLCPP_ERROR(
            this->get_node()->get_logger(),
            "Invalid command received for %zu joints, "
            "expected command for %u size",
            msg->effort.size(), teleop_utils::NUM_JOINTS);
        }
      });
  }

  return CallbackReturn::SUCCESS;
}

CallbackReturn TeleopLeaderController::on_init()
{
  try {
    auto_declare<std::string>("arm_id", "fr3");
    auto_declare<std::vector<double>>(
      "force_reflection_gains", {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0});
  } catch (const std::exception & e) {
    RCLCPP_ERROR(
      this->get_node()->get_logger(), "Exception thrown during init stage with message: %s \n",
      e.what());
    return CallbackReturn::ERROR;
  }

  external_joint_torques_from_follower_buffer_ptr_ =
    realtime_tools::RealtimeBuffer<std::shared_ptr<sensor_msgs::msg::JointState>>(nullptr);

  return CallbackReturn::SUCCESS;
}

CallbackReturn TeleopLeaderController::on_deactivate(
  const rclcpp_lifecycle::State & /*previous_state*/
)
{
  external_joint_torques_from_follower_buffer_ptr_ =
    realtime_tools::RealtimeBuffer<std::shared_ptr<sensor_msgs::msg::JointState>>(nullptr);

  return controller_interface::CallbackReturn::SUCCESS;
}

void TeleopLeaderController::updateJointStates()
{
  for (unsigned int i = 0; i < teleop_utils::NUM_JOINTS; ++i) {
    const auto & velocity_interface = state_interfaces_.at(2 * i + 1);

    assert(velocity_interface.get_interface_name() == "velocity");

    leader_velocity_[i] = velocity_interface.get_value();
  }
}

}  // namespace fr3_bilateral_teleop
#include "pluginlib/class_list_macros.hpp"
// NOLINTNEXTLINE
PLUGINLIB_EXPORT_CLASS(
  fr3_bilateral_teleop::TeleopLeaderController, controller_interface::ControllerInterface)
