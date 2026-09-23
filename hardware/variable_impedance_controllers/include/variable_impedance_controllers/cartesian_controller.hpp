#pragma once

/**
 * @file cartesian_controller.hpp
 * @brief Cartesian controller implementation for robot manipulation (supports impedance and OSC)
 */

#include <memory>
#include <string>
#include <unordered_set>

#include <Eigen/Dense>  // NOLINT(build/include_order)

#include <controller_interface/controller_interface.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/wrench_stamped.hpp>
#include <std_msgs/msg/float64.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/multibody/fwd.hpp>
#include <rclcpp/rclcpp.hpp>
#include <realtime_tools/realtime_publisher.hpp>

#include <variable_impedance_controllers/utils/energy_tank.hpp>
#include <variable_impedance_controllers/utils/log_space_stiffness_rate_limiter.hpp>
#include <variable_impedance_controllers/utils/ros2_version.hpp>

#if ROS2_VERSION_ABOVE_HUMBLE
#include <variable_impedance_controllers/cartesian_controller_parameters.hpp>
#else
#include <cartesian_controller_parameters.hpp>
#endif

#include <sensor_msgs/msg/joint_state.hpp>
#include "realtime_tools/realtime_buffer.hpp"

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

namespace variable_impedance_controllers {

/**
 * @brief Controller implementing Cartesian control
 *
 * This controller implements Cartesian control for robotic manipulation,
 * supporting both impedance control and operational space control (OSC),
 * allowing for compliant interaction with the environment while maintaining
 * desired position and orientation targets.
 */
class CartesianController : public controller_interface::ControllerInterface {
public:
  /**
   * @brief Get the command interface configuration
   * @return Interface configuration specifying required command interfaces
   */
  [[nodiscard]] controller_interface::InterfaceConfiguration
  command_interface_configuration() const override;

  /**
   * @brief Get the state interface configuration
   * @return Interface configuration specifying required state interfaces
   */
  [[nodiscard]] controller_interface::InterfaceConfiguration
  state_interface_configuration() const override;

  /**
   * @brief Update function called periodically
   * @param time Current time
   * @param period Time since last update
   * @return Success/failure of update
   */
  controller_interface::return_type
  update(const rclcpp::Time & time, const rclcpp::Duration & period) override;

  /**
   * @brief Initialize the controller
   * @return Success/failure of initialization
   */
  CallbackReturn on_init() override;

  /**
   * @brief Configure the controller
   * @param previous_state Previous lifecycle state
   * @return Success/failure of configuration
   */
  CallbackReturn on_configure(const rclcpp_lifecycle::State & previous_state) override;

  /**
   * @brief Activate the controller
   * @param previous_state Previous lifecycle state
   * @return Success/failure of activation
   */
  CallbackReturn on_activate(const rclcpp_lifecycle::State & previous_state) override;

  /**
   * @brief Deactivate the controller
   * @param previous_state Previous lifecycle state
   * @return Success/failure of deactivation
   */
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State & previous_state) override;

  /*CartesianImpedanceController();*/

private:
  /** @brief Subscription for target pose messages */
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;
  /** @brief Subscription for target joint state messages */
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_;
  /** @brief Subscription for target wrench messages */
  rclcpp::Subscription<geometry_msgs::msg::WrenchStamped>::SharedPtr wrench_sub_;
  /** @brief Subscription for variable stiffness messages */
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr stiffness_sub_;

  /** @brief Proposal §4.3 commanded-vs-realized-stiffness diagnostics. Independent of
   * ROS2 Control Introspection (HAS_ROS2_CONTROL_INTROSPECTION, registered separately above
   * when available) -- these are always-on plain topics, since introspection needs
   * ros2_control >= 4.27.0 and is compiled out on e.g. Humble's stock hardware_interface. */
  using Float64RtPublisher = realtime_tools::RealtimePublisher<std_msgs::msg::Float64>;
  using Float64MultiArrayRtPublisher =
    realtime_tools::RealtimePublisher<std_msgs::msg::Float64MultiArray>;
  std::shared_ptr<Float64MultiArrayRtPublisher> stiffness_target_publisher_;
  std::shared_ptr<Float64MultiArrayRtPublisher> stiffness_applied_publisher_;
  std::shared_ptr<Float64RtPublisher> energy_tank_energy_publisher_;

  /** @brief Flag to indicate if multiple publishers detected */
  bool multiple_publishers_detected_;

  /** @brief Expected maximum number of publishers per topic */
  size_t max_allowed_publishers_;

  /**
   * @brief Set the stiffness and damping matrices based on parameters
   */
  void setStiffnessAndDamping();

  /**
   * @brief Applies log-space rate limiting + energy-tank passivity gating (proposal §4.3) to
   * stiffness_target_diagonal_, producing stiffness_applied_diagonal_, and writes the result
   * into `stiffness`/`damping`. Must be called after `error` and `J` are both up to date for
   * this cycle (needs `error` for the tank's withdrawal-amount calculation and `xdot_task`
   * for its dissipated-power accounting), and before `stiffness`/`damping` are used to
   * compute tau_task. If params_.stiffness_shaping.enabled is false, falls back to the
   * original instantaneous behavior (stiffness_applied_diagonal_ == stiffness_target_diagonal_).
   * @param dt Control period in seconds.
   * @param xdot_task Current task-space (Cartesian) velocity, J * dq.
   */
  void applyStiffnessShaping(double dt, const Eigen::VectorXd & xdot_task);

  /**
   * @brief Get the current state of the robot from hardware interfaces and update internal variables
   * @param initialize If set to true, initialize the exponential moving average filter with the current state 
   */
  void updateCurrentState(bool initialize = false);

  /**
   * @brief Reads the target pose in realtime loop from the buffer and parses it to be used in the controller.
   */
  void parse_target_pose_();

  /**
   * @brief Reads the target joint in realtime loop from the buffer and parses it to be used in the controller.
   */
  void parse_target_joint_();

  /**
   * @brief Reads the target wrench in realtime loop from the buffer and parses it to be used in the controller.
   */
  void parse_target_wrench_();

  /**
   * @brief Reads the target stiffness in realtime loop from the buffer and parses it to be used in the controller.
   */
  void parse_target_stiffness_();

  bool new_target_pose_;
  bool new_target_joint_;
  bool new_target_wrench_;
  bool new_target_stiffness_ = false;
  bool use_topic_stiffness_ = false;

  realtime_tools::RealtimeBuffer<std::shared_ptr<geometry_msgs::msg::PoseStamped>>
    target_pose_buffer_;

  realtime_tools::RealtimeBuffer<std::shared_ptr<sensor_msgs::msg::JointState>>
    target_joint_buffer_;

  realtime_tools::RealtimeBuffer<std::shared_ptr<geometry_msgs::msg::WrenchStamped>>
    target_wrench_buffer_;

  realtime_tools::RealtimeBuffer<std::shared_ptr<std_msgs::msg::Float64MultiArray>>
    target_stiffness_buffer_;

  /** @brief Target position in Cartesian space */
  Eigen::Vector3d target_position_;
  /** @brief Target orientation as quaternion */
  Eigen::Quaterniond target_orientation_;
  /** @brief Target wrench in task space */
  Eigen::VectorXd target_wrench_;
  /** @brief Desired target position in Cartesian space after applying filtering */
  Eigen::Vector3d desired_position_;
  /** @brief Desired target orientation as quaternion after applying filtering */
  Eigen::Quaterniond desired_orientation_;

  /** @brief Parameter listener for dynamic parameter updates */
  std::shared_ptr<cartesian_controller::ParamListener> params_listener_;
  /** @brief Current parameter values */
  cartesian_controller::Params params_;

  /** @brief Frame ID of the end effector in the robot model */
  int end_effector_frame_id;

  /** @brief Pinocchio robot model */
  pinocchio::Model model_;
  /** @brief Pinocchio data for computations */
  pinocchio::Data data_;

  /** @brief Cartesian stiffness matrix (6x6) */
  Eigen::MatrixXd stiffness = Eigen::MatrixXd::Zero(6, 6);
  /** @brief Cartesian damping matrix (6x6) */
  Eigen::MatrixXd damping = Eigen::MatrixXd::Zero(6, 6);
  /** @brief Topic-provided stiffness matrix (6x6 diagonal) */
  Eigen::Matrix<double, 6, 6> topic_stiffness_ = Eigen::Matrix<double, 6, 6>::Zero();

  /** @brief This cycle's un-shaped stiffness target (diag of `stiffness` pre-§4.3 shaping),
   * set by setStiffnessAndDamping() from either topic_stiffness_ or the static task.k_* params. */
  Eigen::Matrix<double, 6, 1> stiffness_target_diagonal_ = Eigen::Matrix<double, 6, 1>::Zero();
  /** @brief Stiffness actually in force this cycle, after log-space rate limiting + the
   * energy tank (proposal §4.3) -- what `stiffness`'s diagonal is set to, and the "realized"
   * half of the commanded-vs-realized-stiffness validation figure. Also this energy tank's
   * bookkeeping reference for the next cycle's withdrawals (see applyStiffnessShaping()). */
  Eigen::Matrix<double, 6, 1> stiffness_applied_diagonal_ = Eigen::Matrix<double, 6, 1>::Zero();
  /** @brief Per-axis damping override from task.d_* params (value if > 0, else -1 meaning
   * "derive critically from the APPLIED, post-shaping stiffness": 2*sqrt(k_applied)). Set by
   * setStiffnessAndDamping(); consumed by applyStiffnessShaping(). */
  Eigen::Matrix<double, 6, 1> damping_override_diagonal_ = Eigen::Matrix<double, 6, 1>::Constant(-1.0);

  /** @brief Log-space rate limiter for stiffness_target_diagonal_ -> stiffness_applied_diagonal_
   * (proposal §4.3, |d(log k)/dt| <= gamma). See stiffness_shaping.gamma. */
  std::unique_ptr<LogSpaceStiffnessRateLimiter> stiffness_rate_limiter_;
  /** @brief Passivity guard on cumulative stiffness INCREASES (proposal §4.3). Reset to
   * stiffness_shaping.energy_tank.e0 on every on_activate(). See stiffness_shaping.energy_tank. */
  std::unique_ptr<EnergyTank> energy_tank_;
  /** @brief Snapshot of energy_tank_->energy(), refreshed every applyStiffnessShaping() call.
   * A plain double (rather than reading energy_tank_->energy() directly) so it has a stable
   * address for ros2_control introspection registration. */
  double energy_tank_energy_ = 0.0;

  /** @brief Nullspace stiffness matrix for posture control */
  Eigen::MatrixXd nullspace_stiffness;
  /** @brief Nullspace damping matrix for posture control */
  Eigen::MatrixXd nullspace_damping;

  /** @brief Current joint positions with dimension nv. */
  Eigen::VectorXd q;
  /** @brief Current joint positions with dimension nq.
   This is size might be different than the actuated dimension of the joint type is different! 
   Check https://github.com/stack-of-tasks/pinocchio/issues/1127
  */
  Eigen::VectorXd q_pin;
  /** @brief Current joint velocities */
  Eigen::VectorXd dq;
  /** @brief Reference joint positions for posture task */
  Eigen::VectorXd q_ref;
  /** @brief Reference joint velocities */
  Eigen::VectorXd dq_ref;
  /** @brief Target joint positions for posture task */
  Eigen::VectorXd q_target;

  /** @brief Previously computed torque */
  Eigen::VectorXd tau_previous;

  /** @brief Current end effector pose */
  pinocchio::SE3 end_effector_pose;
  /** @brief End effector Jacobian matrix */
  pinocchio::Data::Matrix6x J;
  /** @brief End effector Jacobian matrix pseudoinverse */
  Eigen::MatrixXd J_pinv;
  /** @brief Joint-space identity matrix */
  Eigen::MatrixXd Id_nv;

  /** @brief Friction parameters 1 of size nv */
  Eigen::VectorXd fp1;
  /** @brief Friction parameters 2 of size nv */
  Eigen::VectorXd fp2;
  /** @brief Friction parameters 3 of size nv */
  Eigen::VectorXd fp3;

  /** @brief Allowed type of joints **/
  const std::unordered_set<std::basic_string<char>> allowed_joint_types = {
    "JointModelRX",
    "JointModelRY",
    "JointModelRZ",
    "JointModelRevoluteUnaligned",
    "JointModelRUBX",
    "JointModelRUBY",
    "JointModelRUBZ",
  };
  /** @brief Continous joint types that should be considered separetly. **/
  const std::unordered_set<std::basic_string<char>> continous_joint_types = {
    "JointModelRUBX", "JointModelRUBY", "JointModelRUBZ"};

  /** @brief Maximum allowed delta values for error clipping */
  Eigen::VectorXd max_delta_ = Eigen::VectorXd::Zero(6);

  /** @brief Nullspace projection matrix */
  Eigen::MatrixXd nullspace_projection;

  /** @brief Task space error vector (6x1) */
  Eigen::VectorXd error = Eigen::VectorXd::Zero(6);

  /** @brief Task space control torque */
  Eigen::VectorXd tau_task;
  /** @brief Joint limit avoidance torque */
  Eigen::VectorXd tau_joint_limits;
  /** @brief Secondary task torque before nullspace projection */
  Eigen::VectorXd tau_secondary;
  /** @brief Nullspace projected secondary task torque */
  Eigen::VectorXd tau_nullspace;
  /** @brief Friction compensation torque */
  Eigen::VectorXd tau_friction;
  /** @brief Coriolis compensation torque */
  Eigen::VectorXd tau_coriolis;
  /** @brief Gravity compensation torque */
  Eigen::VectorXd tau_gravity;
  /** @brief External wrench compensation torque */
  Eigen::VectorXd tau_wrench;
  /** @brief Final desired torque command */
  Eigen::VectorXd tau_d;

  /** @brief Inverse of the manipulator joint mass projected in Cartesian space (6x6) */
  Eigen::Matrix<double, 6, 6> Mx_inv = Eigen::Matrix<double, 6, 6>::Zero();
  /** @brief the manipulator joint mass projected in Cartesian space (6x6) */
  Eigen::Matrix<double, 6, 6> Mx = Eigen::Matrix<double, 6, 6>::Zero();

  /**
   * @brief Log debug information based on parameter settings
   * @param time Current time for throttling logs
   */
  void log_debug_info(const rclcpp::Time & time);

  /**
   * @brief Check publisher count for a specific topic
   * @param topic_name Name of the topic to check
   * @return true if publisher count is safe (<=1), false otherwise
   */
  bool check_topic_publisher_count(const std::string & topic_name);
};

}  // namespace variable_impedance_controllers
