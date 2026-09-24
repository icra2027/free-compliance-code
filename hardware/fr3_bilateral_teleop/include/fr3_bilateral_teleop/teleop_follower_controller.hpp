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

#include <string>
#include <memory>

#include <Eigen/Eigen>
#include <controller_interface/controller_interface.hpp>
#include <rclcpp/rclcpp.hpp>
#include <realtime_tools/realtime_buffer.hpp>
#include <realtime_tools/realtime_publisher.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64.hpp>

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

namespace fr3_bilateral_teleop
{

/**
 * The joint impedance example controller moves joint 4 and 5 in a very
 * compliant periodic movement.
 */
class TeleopFollowerController : public controller_interface::ControllerInterface
{
public:
  using Vector7d = Eigen::Matrix<double, 7, 1>;
  [[nodiscard]] controller_interface::InterfaceConfiguration command_interface_configuration()
  const override;
  [[nodiscard]] controller_interface::InterfaceConfiguration state_interface_configuration()
  const override;
  controller_interface::return_type update(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;
  CallbackReturn on_init() override;
  CallbackReturn on_configure(const rclcpp_lifecycle::State & previous_state) override;
  CallbackReturn on_activate(const rclcpp_lifecycle::State & previous_state) override;
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State & previous_state) override;

private:
  std::string arm_id_;
  std::string robot_description_;
  std::string input_topic_;
  std::string force_input_topic_;
  Vector7d q_;
  Vector7d initial_q_;
  Vector7d dq_;
  Vector7d dq_filtered_;
  Vector7d k_gains_;
  Vector7d d_gains_;
  // Slew-rate-limited position goal actually used by update() (see max_goal_slew_rate_
  // below). Kept separate from the raw leader-commanded position so a large jump in that
  // raw position -- e.g. right after this controller reactivates while the leader has
  // drifted away from wherever the follower now is, such as after the lerobot recorder's
  // post-episode home move, which deliberately leaves the leader alone -- is ramped into
  // instead of snapped to.
  Vector7d q_goal_smoothed_;
  // Max per-joint rate (rad/s) at which q_goal_smoothed_ may move toward the raw target.
  // Set well above normal hand-guided teleop speeds so it is a no-op during regular
  // operation, and only engages to smooth out large discontinuities like the reactivation
  // jump described above.
  double max_goal_slew_rate_{2.0};
  // Force channel gain: scales the leader's own sensed external torque (i.e. the
  // force the human operator applies at the leader) before it is fed forward into
  // the follower's torque command. Independent from k_gains_/d_gains_ (the position
  // coupling), so tracking and "feel" can each be tuned on their own gain.
  Vector7d force_feedforward_gains_;
  double elapsed_time_{0.0};
  int64_t input_topic_timeout_ = 2500000;

  realtime_tools::RealtimeBuffer<std::shared_ptr<sensor_msgs::msg::JointState>>
  measured_joint_states_from_leader_buffer_ptr_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr
    measured_joint_states_from_leader_subscriber_;

  realtime_tools::RealtimeBuffer<std::shared_ptr<sensor_msgs::msg::JointState>>
  external_joint_torques_from_leader_buffer_ptr_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr
    external_joint_torques_from_leader_subscriber_;

  // Timing diagnostics (see README "Verifying 1 kHz timing and channel latency"):
  // - loop_period_publisher_ publishes the actual, controller_manager-measured `update()`
  //   period, in microseconds, so a 1 kHz claim can be checked against real jitter rather
  //   than the configured update_rate alone.
  // - position_channel_latency_publisher_ / force_feedforward_channel_latency_publisher_
  //   publish, in microseconds, the age of the leader->follower position message (channel 1)
  //   and force-feedforward message (channel 3) at the moment each is consumed, i.e.
  //   `now - header.stamp`. Meaningful only if the leader and follower clocks are synchronized
  //   (see the time-sync check); it is not a substitute for that check.
  // Publishing uses realtime_tools::RealtimePublisher, which is RT-safe: it never blocks the
  // update() thread and silently drops a sample if the non-RT publish thread is still busy.
  using Float64RtPublisher = realtime_tools::RealtimePublisher<std_msgs::msg::Float64>;
  std::shared_ptr<Float64RtPublisher> loop_period_publisher_;
  std::shared_ptr<Float64RtPublisher> position_channel_latency_publisher_;
  std::shared_ptr<Float64RtPublisher> force_feedforward_channel_latency_publisher_;

  void updateJointStates();
};

}  // namespace fr3_bilateral_teleop
