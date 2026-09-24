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
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <functional>
#include <vector>

#include <controller_manager_msgs/srv/switch_controller.hpp>
#include <fr3_bilateral_teleop/utils.hpp>
#include <geometry_msgs/msg/wrench_stamped.hpp>
#include <rclcpp/rclcpp.hpp>

using namespace std::chrono_literals;

constexpr auto WAIT_TIME = 1s;
constexpr int SWITCH_CONTROLLER_BEST_EFFORT = 1;
constexpr int SWITCH_CONTROLLER_STRICT = 2;
const std::string MOVE_TO_START_CONTROLLER = "move_to_start_example_controller";
const std::string TARGET_POSITION_REACHED_PARAMETER = "process_finished";
const std::string START_JOINT_CONFIGURATION_PARAMETER = "start_joint_configuration";
const std::string EXTERNAL_WRENCH_TOPIC =
  "franka_robot_state_broadcaster/external_wrench_in_base_frame";

// Every session must hold at the fixed free-space start pose long enough for residual
// motion from the move-to-start motion generator to die out (settle) and then long
// enough to average out sensor noise in the wrench estimate (sample). This is what
// makes the re-zero automatic and non-skippable: it runs on the same code path that
// gates activation of the real teleoperation controllers for every session.
constexpr auto REZERO_SETTLE_DURATION = 1s;
constexpr auto REZERO_SAMPLE_DURATION = 1s;
constexpr double REZERO_SAMPLE_RATE_HZ = 100.0;
const std::string REZERO_OUTPUT_DIR_ENV = "TELEOP_REZERO_OUTPUT_DIR";
const std::string REZERO_OUTPUT_DIR_DEFAULT = "/tmp/franka_teleop_rezero";

class ControllerCoordinator
{
public:
  ControllerCoordinator(
    std::shared_ptr<rclcpp::Node> node, const std::string & controller_namespace,
    const std::string & target_controller)
  {
    controller_namespace_ = controller_namespace;
    node_handle_ = node;
    target_controller_ = target_controller;

    move_to_start_controller_parameter_client_ = std::make_shared<rclcpp::SyncParametersClient>(
      node, controller_namespace_ + "/" + MOVE_TO_START_CONTROLLER);

    controller_manager_switch_service_client_ =
      node->create_client<controller_manager_msgs::srv::SwitchController>(
      controller_namespace_ + "/controller_manager/switch_controller");

    wrench_topic_ = controller_namespace_ + "/" + EXTERNAL_WRENCH_TOPIC;
    external_wrench_subscription_ = node->create_subscription<geometry_msgs::msg::WrenchStamped>(
      wrench_topic_, rclcpp::QoS(10).best_effort(),
      [this](const geometry_msgs::msg::WrenchStamped::SharedPtr msg) {
        if (rezero_sampling_active_) {
          rezero_wrench_samples_.push_back(*msg);
        }
      });
  }

  // Clears any samples from a previous session and starts accumulating wrench
  // readings from the fixed free-space pose. Call only once the arm has settled
  // there (move_to_start has reported it reached the target position).
  void begin_rezero_sampling()
  {
    rezero_wrench_samples_.clear();
    rezero_sampling_active_ = true;
  }

  // Stops accumulating samples, computes the per-axis wrench bias measured at the
  // fixed free-space pose, and writes it to <output_dir>/<namespace>_rezero.json.
  // Returns false (and refuses to report success) if no samples were received --
  // that means franka_robot_state_broadcaster isn't publishing, and silently
  // proceeding would let a session start with an unmeasured bias.
  bool finish_rezero_and_report(const std::string & output_dir)
  {
    rezero_sampling_active_ = false;

    if (rezero_wrench_samples_.empty()) {
      RCLCPP_ERROR(
        node_handle_->get_logger(),
        "[%s] Free-space re-zero FAILED: no messages received on %s during the sampling "
        "window. Is franka_robot_state_broadcaster running for this arm?",
        controller_namespace_.c_str(), wrench_topic_.c_str());
      return false;
    }

    std::array<double, 6> sum{};
    for (const auto & sample : rezero_wrench_samples_) {
      sum[0] += sample.wrench.force.x;
      sum[1] += sample.wrench.force.y;
      sum[2] += sample.wrench.force.z;
      sum[3] += sample.wrench.torque.x;
      sum[4] += sample.wrench.torque.y;
      sum[5] += sample.wrench.torque.z;
    }
    const double n = static_cast<double>(rezero_wrench_samples_.size());
    std::array<double, 6> bias{};
    for (std::size_t i = 0; i < bias.size(); ++i) {
      bias[i] = sum[i] / n;
    }

    std::vector<double> free_space_joint_configuration;
    for (auto & parameter : move_to_start_controller_parameter_client_->get_parameters(
        {START_JOINT_CONFIGURATION_PARAMETER}))
    {
      free_space_joint_configuration = parameter.as_double_array();
    }

    const double stamp_sec = node_handle_->now().seconds();

    std::ostringstream json;
    json << std::fixed << std::setprecision(6);
    json << "{\n";
    json << "  \"namespace\": \"" << controller_namespace_ << "\",\n";
    json << "  \"wrench_topic\": \"" << wrench_topic_ << "\",\n";
    json << "  \"stamp_sec\": " << stamp_sec << ",\n";
    json << "  \"num_samples\": " << rezero_wrench_samples_.size() << ",\n";
    json << "  \"free_space_joint_configuration\": [";
    for (std::size_t i = 0; i < free_space_joint_configuration.size(); ++i) {
      json << free_space_joint_configuration[i]
           << (i + 1 < free_space_joint_configuration.size() ? ", " : "");
    }
    json << "],\n";
    json << "  \"external_wrench_bias_base_frame\": {\n";
    json << "    \"force\": [" << bias[0] << ", " << bias[1] << ", " << bias[2] << "],\n";
    json << "    \"torque\": [" << bias[3] << ", " << bias[4] << ", " << bias[5] << "]\n";
    json << "  }\n";
    json << "}\n";

    // "Latest" file: the canonical path the extraction pipeline reads for this arm.
    // Timestamped copy under history/: so session-to-session drift stays inspectable
    // instead of being overwritten every time re-zero runs.
    std::filesystem::create_directories(output_dir);
    std::filesystem::create_directories(output_dir + "/history");
    const std::string file_path = output_dir + "/" + sanitized_namespace() + "_rezero.json";
    const std::string history_path = output_dir + "/history/" + sanitized_namespace() + "_" +
      std::to_string(static_cast<long long>(stamp_sec)) + ".json";

    for (const auto & path : {file_path, history_path}) {
      std::ofstream out(path, std::ios::trunc);
      out << json.str();
    }

    const double force_norm = std::sqrt(bias[0] * bias[0] + bias[1] * bias[1] + bias[2] * bias[2]);
    RCLCPP_INFO(
      node_handle_->get_logger(),
      "[%s] Free-space re-zero OK: |F_bias|=%.3f N over %zu samples -> %s",
      controller_namespace_.c_str(), force_norm, rezero_wrench_samples_.size(),
      file_path.c_str());

    return true;
  }

  bool wait_for_connection() const
  {
    return teleop_utils::sync_wait_for_service(
      move_to_start_controller_parameter_client_, "move_to_start parameter client",
      controller_namespace_, node_handle_->get_logger(), WAIT_TIME) &&
           teleop_utils::sync_wait_for_service(
      controller_manager_switch_service_client_, "switch_controller service client",
      controller_namespace_, node_handle_->get_logger(), WAIT_TIME);
  }

  bool start_move_to_start_controller() const
  {
    return switch_controller({MOVE_TO_START_CONTROLLER}, {});
  }

  bool start_target_controller() const
  {
    return switch_controller({target_controller_}, {MOVE_TO_START_CONTROLLER});
  }

  bool has_reached_target_position() const
  {
    std::stringstream ss;
    ss << "[" << controller_namespace_ << "]";

    bool target_position_reached = false;

    for (auto & parameter : move_to_start_controller_parameter_client_->get_parameters(
        {TARGET_POSITION_REACHED_PARAMETER}))
    {
      ss << " has_reached_target_position: " << parameter.value_to_string();

      target_position_reached = parameter.as_bool();
    }

    RCLCPP_DEBUG(node_handle_->get_logger(), "%s", ss.str().c_str());

    return target_position_reached;
  }

private:
  std::string sanitized_namespace() const
  {
    std::string sanitized = controller_namespace_;
    std::replace(sanitized.begin(), sanitized.end(), '/', '_');
    return sanitized;
  }

  bool switch_controller(
    const std::vector<std::string> & controllers_to_activate,
    const std::vector<std::string> & controllers_to_deactivate) const
  {
    auto request = std::make_shared<controller_manager_msgs::srv::SwitchController::Request>();
    request->activate_controllers = controllers_to_activate;
    request->deactivate_controllers = controllers_to_deactivate;
    request->strictness = SWITCH_CONTROLLER_STRICT;
    request->activate_asap = true;
    request->timeout = rclcpp::Duration(1.0s);

    auto result = controller_manager_switch_service_client_->async_send_request(request);

    rclcpp::spin_until_future_complete(node_handle_, result);

    const bool succeeded = result.get()->ok;

    RCLCPP_INFO(
      node_handle_->get_logger(), "[%s] switch controller %s", controller_namespace_.c_str(),
      (succeeded ? "succeded" : "failed"));

    return succeeded;
  }

  rclcpp::Node::SharedPtr node_handle_;
  std::string controller_namespace_;
  std::string target_controller_;
  rclcpp::SyncParametersClient::SharedPtr move_to_start_controller_parameter_client_;
  rclcpp::Client<controller_manager_msgs::srv::SwitchController>::SharedPtr
    controller_manager_switch_service_client_;

  std::string wrench_topic_;
  rclcpp::Subscription<geometry_msgs::msg::WrenchStamped>::SharedPtr external_wrench_subscription_;
  std::vector<geometry_msgs::msg::WrenchStamped> rezero_wrench_samples_;
  bool rezero_sampling_active_{false};
};

int main(int argc, char ** argv)
{
  setvbuf(stdout, NULL, _IONBF, BUFSIZ);

  std::vector<std::string> arguments = rclcpp::init_and_remove_ros_arguments(argc, argv);
  arguments.erase(arguments.begin());

  auto node = rclcpp::Node::make_shared("teleop_coordinator");

  if (arguments.size() % 2 != 0) {
    RCLCPP_ERROR(
      node->get_logger(),
      "Arguments are not even! Usage: ros2 run fr3_bilateral_teleop teleop_coordinator "
      "<controller_namespace_1> "
      "<target_controller_name_1> [<controller_namespace_2> <target_controller_name_2> ...] ");
    return 1;
  }

  std::vector<ControllerCoordinator> coordinatedControllers;
  // Reserved upfront: the subscription callbacks below capture `this`, so the vector
  // must never reallocate (which would move-construct elements and dangle those
  // pointers) once coordinators start being constructed.
  coordinatedControllers.reserve(arguments.size() / 2);

  for (std::size_t i = 0; i < arguments.size(); i += 2) {
    coordinatedControllers.emplace_back(node, arguments[i], arguments[i + 1]);
    RCLCPP_INFO(
      node->get_logger(), "Controller Interface: %s %s", arguments[i].c_str(),
      arguments[i + 1].c_str());
  }

  rclcpp::Rate loop_rate(10);

  bool running = true;

  for (const ControllerCoordinator & coordinator : coordinatedControllers) {
    running = coordinator.wait_for_connection();
  }

  rclcpp::sleep_for(std::chrono::duration_cast<std::chrono::nanoseconds>(0.5s));

  for (const ControllerCoordinator & coordinator : coordinatedControllers) {
    running = coordinator.start_move_to_start_controller();
  }

  while (rclcpp::ok() && running) {
    bool all_reached = true;
    for (const ControllerCoordinator & coordinator : coordinatedControllers) {
      all_reached = all_reached && coordinator.has_reached_target_position();
    }

    if (all_reached) {
      running = false;
    }

    rclcpp::spin_some(node);
    loop_rate.sleep();
  }

  // Free-space re-zero (mandatory, every session): all arms are now holding at the
  // fixed start_joint_configuration under move_to_start_example_controller. Settle,
  // then sample the estimated external wrench there -- whatever isn't zero is bias
  // (friction, payload mis-specification, thermal drift) that must be captured now,
  // before teleoperation starts, or it silently shows up later as a spurious
  // stiffness trend.
  RCLCPP_INFO(node->get_logger(), "Settling at free-space pose before re-zero sampling...");
  rclcpp::sleep_for(std::chrono::duration_cast<std::chrono::nanoseconds>(REZERO_SETTLE_DURATION));
  rclcpp::spin_some(node);

  for (ControllerCoordinator & coordinator : coordinatedControllers) {
    coordinator.begin_rezero_sampling();
  }

  const char * output_dir_env = std::getenv(REZERO_OUTPUT_DIR_ENV.c_str());
  const std::string rezero_output_dir = output_dir_env ? output_dir_env : REZERO_OUTPUT_DIR_DEFAULT;

  rclcpp::Rate rezero_sample_rate(REZERO_SAMPLE_RATE_HZ);
  const auto rezero_sampling_deadline = node->now() + rclcpp::Duration(REZERO_SAMPLE_DURATION);
  while (rclcpp::ok() && node->now() < rezero_sampling_deadline) {
    rclcpp::spin_some(node);
    rezero_sample_rate.sleep();
  }

  bool rezero_ok = true;
  for (ControllerCoordinator & coordinator : coordinatedControllers) {
    rezero_ok = coordinator.finish_rezero_and_report(rezero_output_dir) && rezero_ok;
  }

  if (!rezero_ok) {
    RCLCPP_FATAL(
      node->get_logger(),
      "Free-space re-zero failed for at least one arm. Refusing to start teleoperation "
      "controllers -- fix the wrench estimate (is franka_robot_state_broadcaster running?) "
      "and restart the session.");
    rclcpp::shutdown();
    return 1;
  }

  for (const ControllerCoordinator & coordinator : coordinatedControllers) {
    coordinator.start_target_controller();
  }

  rclcpp::shutdown();

  return 0;
}
