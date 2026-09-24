#  Copyright (c) 2026 Franka Robotics GmbH
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Brings up a single Franka arm (no teleop pairing) running variable_impedance_controllers'
CartesianController, patched with the log-space stiffness rate limiter + energy tank --
see fr3_bilateral_teleop/README.md ("Variable-impedance controller").

Defaults to `use_fake_hardware:=true` (no physical robot needed, via ros2_control's
mock_components/GenericSystem): this validates the wiring -- controller activation, topic
names, parameter nesting, the stiffness-shaping pipeline's own numerics -- but NOT real
physical dynamics, since fake hardware does not integrate torque commands into motion.
Pass `use_fake_hardware:=false robot_ip:=<ip>` for a real-hardware run once one is available;
nothing else in this launch file or the controller config needs to change.

Run this first, then in a second terminal run the probe script:
    ros2 run fr3_bilateral_teleop probe_variable_impedance_sinusoid.py

For a simple fixed-pose hold (no sinusoidal stiffness sweep, no probe script needed), pass
`target_position` (and optionally `target_orientation`) launch args -- this file then
publishes that pose once to variable_impedance_controller's `target_pose` topic itself, right
after the controller activates.
"""

import base64
import os

import yaml
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, OpaqueFunction,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

LOWER_TORQUE_THRESHOLDS_ACCELERATION = [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0]
UPPER_TORQUE_THRESHOLDS_ACCELERATION = [85.0, 85.0, 85.0, 85.0, 11.0, 11.0, 11.0]
LOWER_TORQUE_THRESHOLDS_NOMINAL = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
UPPER_TORQUE_THRESHOLDS_NOMINAL = [85.0, 85.0, 85.0, 85.0, 11.0, 11.0, 11.0]
LOWER_FORCE_THRESHOLDS_ACCELERATION = [20.0, 20.0, 20.0, 20.0, 20.0, 20.0]
UPPER_FORCE_THRESHOLDS_ACCELERATION = [85.0, 85.0, 85.0, 11.0, 11.0, 11.0]
LOWER_FORCE_THRESHOLDS_NOMINAL = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
UPPER_FORCE_THRESHOLDS_NOMINAL = [85.0, 85.0, 85.0, 11.0, 11.0, 11.0]

# Where calibrate_payload.py writes <namespace>_payload.yaml -- same env var / default dir
# teleop.launch.py and calibrate_payload.py itself use (fr3_bilateral_teleop README, "Payload
# calibration"), so a calibration produced for teleop is picked up here too.
PAYLOAD_OUTPUT_DIR_ENV = "TELEOP_PAYLOAD_OUTPUT_DIR"
PAYLOAD_OUTPUT_DIR_DEFAULT = "/home/robot/franka_ros2_ws/franka_data/franka_teleop_payload/"


def cvt_to_string(values):
    return ", ".join(str(float(v)) for v in values)


def default_set_load_values():
    """No-extra-load default -- used when calibration hasn't been run or didn't succeed, so
    this validation can still start rather than being blocked on it (same reasoning as
    teleop.launch.py's identically-named helper)."""
    return {"mass": 0.0, "center_of_mass": [0.0, 0.0, 0.0], "load_inertia": [0.0] * 9}


def load_payload_calibration(payload_namespace: str):
    """Read <payload_namespace>_payload.yaml written by calibrate_payload.py. Missing/failed/
    unreadable calibration is a loud warning, not a launch failure -- mirrors teleop.launch.py's
    load_payload_calibration() exactly (kept as its own copy here rather than importing that
    launch file as a module, since launch files aren't meant to be imported as a library)."""
    output_dir = os.environ.get(PAYLOAD_OUTPUT_DIR_ENV, PAYLOAD_OUTPUT_DIR_DEFAULT)
    file_namespace = payload_namespace.strip("/").replace("/", "_")
    payload_path = os.path.join(output_dir, f"{file_namespace}_payload.yaml")

    if not os.path.exists(payload_path):
        print(
            f"[validate_variable_impedance.launch] WARNING: no payload calibration file at "
            f"'{payload_path}' for '{payload_namespace}'. Run calibrate_payload.launch.py + "
            f"calibrate_payload.py first. Falling back to a zero/no-extra-load default -- "
            f"gravity (via set_load) and this controller's own Coriolis compensation will not "
            f"account for any attached tool."
        )
        return default_set_load_values()

    try:
        with open(payload_path, "r") as payload_file:
            report = yaml.safe_load(payload_file)
        if report.get("status") != "ok" or "set_load" not in report:
            print(
                f"[validate_variable_impedance.launch] WARNING: payload calibration at "
                f"'{payload_path}' has status={report.get('status')!r} (not 'ok') -- falling "
                f"back to a zero/no-extra-load default for '{payload_namespace}'."
            )
            return default_set_load_values()
        return report["set_load"]
    except (OSError, yaml.YAMLError, AttributeError) as exc:
        print(
            f"[validate_variable_impedance.launch] WARNING: failed to read payload calibration "
            f"at '{payload_path}': {exc}. Falling back to a zero/no-extra-load default for "
            f"'{payload_namespace}'."
        )
        return default_set_load_values()


def create_set_load_process(namespace: str, set_load_values) -> ExecuteProcess:
    """Applies the calibrated payload to Franka firmware's own gravity compensation on the
    torque interface (same franka_msgs/srv/SetLoad call teleop.launch.py's identically-named
    helper uses). This is the ONLY thing that compensates payload gravity here:
    variable_impedance_validation_controllers.yaml deliberately leaves
    use_gravity_compensation false (firmware already does it) so the config carries over to
    real hardware unchanged -- see that file's comment."""
    return ExecuteProcess(
        cmd=[[
            FindExecutable(name="ros2"),
            " service call ",
            f"/{namespace}/service_server/set_load ",
            "franka_msgs/srv/SetLoad ",
            "\"{ ",
            f"mass: {float(set_load_values['mass'])}, ",
            f"center_of_mass: [{cvt_to_string(set_load_values['center_of_mass'])}], ",
            f"load_inertia: [{cvt_to_string(set_load_values['load_inertia'])}] ",
            "}\"",
        ]],
        shell=True,
        name="set_validation_payload",
        output="both",
    )


def create_target_pose_process(namespace: str, frame_id: str, position, orientation) -> ExecuteProcess:
    """One-shot target pose command to variable_impedance_controllers' CartesianController `target_pose`
    topic (geometry_msgs/PoseStamped) -- the same topic
    probe_variable_impedance_sinusoid.py drives continuously, but published here just once
    (via `ros2 topic pub --once`) so the controller has a fixed setpoint to servo to without
    needing that script running."""
    return ExecuteProcess(
        cmd=[[
            FindExecutable(name="ros2"),
            " topic pub --once ",
            f"/{namespace}/target_pose ",
            "geometry_msgs/msg/PoseStamped ",
            "\"{ ",
            f"header: {{frame_id: '{frame_id}'}}, ",
            "pose: { ",
            f"position: {{x: {position[0]}, y: {position[1]}, z: {position[2]}}}, ",
            "orientation: {"
            f"x: {orientation[0]}, y: {orientation[1]}, "
            f"z: {orientation[2]}, w: {orientation[3]}"
            "} ",
            "} }\"",
        ]],
        shell=True,
        name="set_validation_target_pose",
        output="both",
    )


def generate_launch_description():
    namespace = LaunchConfiguration("namespace")
    controllers_yaml = LaunchConfiguration("controllers_yaml")

    single_robot = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("fr3_bilateral_teleop"), "launch",
                "teleop_single_robot.launch.py",
            ])
        ),
        launch_arguments={
            "arm_id": LaunchConfiguration("arm_id"),
            "arm_prefix": LaunchConfiguration("arm_prefix"),
            "namespace": namespace,
            "urdf_file": LaunchConfiguration("urdf_file"),
            "robot_ip": LaunchConfiguration("robot_ip"),
            "load_gripper": "false",  # end_effector_frame is fr3_link8; no gripper needed
            "use_fake_hardware": LaunchConfiguration("use_fake_hardware"),
            "fake_sensor_commands": LaunchConfiguration("use_fake_hardware"),
            "controllers_yaml": controllers_yaml,
            "initial_joint_positions": LaunchConfiguration("initial_joint_positions"),
        }.items(),
    )

    set_collision_behavior = ExecuteProcess(
        cmd=[[
            FindExecutable(name="ros2"),
            " service call ",
            "/", namespace, "/service_server/set_full_collision_behavior ",
            "franka_msgs/srv/SetFullCollisionBehavior ",
            "\"{ ",
            "lower_torque_thresholds_acceleration: ",
            f"[{cvt_to_string(LOWER_TORQUE_THRESHOLDS_ACCELERATION)}], ",
            "upper_torque_thresholds_acceleration: ",
            f"[{cvt_to_string(UPPER_TORQUE_THRESHOLDS_ACCELERATION)}], ",
            "lower_torque_thresholds_nominal: ",
            f"[{cvt_to_string(LOWER_TORQUE_THRESHOLDS_NOMINAL)}], ",
            "upper_torque_thresholds_nominal: ",
            f"[{cvt_to_string(UPPER_TORQUE_THRESHOLDS_NOMINAL)}], ",
            "lower_force_thresholds_acceleration: ",
            f"[{cvt_to_string(LOWER_FORCE_THRESHOLDS_ACCELERATION)}], ",
            "upper_force_thresholds_acceleration: ",
            f"[{cvt_to_string(UPPER_FORCE_THRESHOLDS_ACCELERATION)}], ",
            "lower_force_thresholds_nominal: ",
            f"[{cvt_to_string(LOWER_FORCE_THRESHOLDS_NOMINAL)}], ",
            "upper_force_thresholds_nominal: ",
            f"[{cvt_to_string(UPPER_FORCE_THRESHOLDS_NOMINAL)}] ",
            "}\"",
        ]],
        shell=True,
        name="set_validation_collision_behavior",
        output="both",
    )

    # NOT spawning joint_state_broadcaster here: teleop_single_robot.launch.py's own
    # generate_robot_nodes() already spawns it unconditionally (and, on real hardware,
    # franka_robot_state_broadcaster too -- see variable_impedance_validation_controllers.yaml
    # for why that controller is defined in this package's own config despite variable_impedance_controllers
    # not needing it). A second spawn attempt here raced with that one and failed outright on
    # a real-hardware run -- confirmed empirically, not just reasoned
    # about after the fact.

    pose_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        namespace=namespace,
        arguments=["pose_broadcaster", "--controller-manager-timeout", "30"],
        output="screen",
    )

    def configure_payload(context):
        resolved_namespace = namespace.perform(context)
        load_calibration = (
            LaunchConfiguration("load_payload_calibration").perform(context).lower()
            in ("true", "1", "yes")
        )
        payload_calibration_namespace = LaunchConfiguration(
            "payload_calibration_namespace"
        ).perform(context)

        set_load_values = (
            load_payload_calibration(payload_calibration_namespace)
            if load_calibration else default_set_load_values()
        )

        # Also fold the payload into variable_impedance_controllers' own Pinocchio dynamics model (see
        # variable_impedance_controllers/src/cartesian_controller.yaml's `payload` params). set_load above
        # only reaches firmware's gravity compensation -- it does NOT cover this controller's
        # own Coriolis term (use_coriolis_compensation: true in
        # variable_impedance_validation_controllers.yaml, since firmware doesn't auto-add
        # that on the torque interface), which variable_impedance_controllers computes itself from a
        # Pinocchio model that otherwise only knows the robot's own links.
        start_joint_configuration = [
            float(v) for v in
            LaunchConfiguration("initial_joint_positions").perform(context).split()
        ]

        payload_overrides = {
            "/**": {
                "variable_impedance_controller": {
                    "ros__parameters": {
                        "payload": {
                            "mass": float(set_load_values["mass"]),
                            "center_of_mass": [
                                float(v) for v in set_load_values["center_of_mass"]
                            ],
                            "inertia": [float(v) for v in set_load_values["load_inertia"]],
                        }
                    }
                },
                # Real hardware only -- see move_to_start_example_controller's own
                # description in validate_variable_impedance.launch.py's --help. Harmless if
                # loaded-but-never-activated under fake hardware.
                "move_to_start_example_controller": {
                    "ros__parameters": {
                        "start_joint_configuration": start_joint_configuration,
                    }
                },
            }
        }
        random_string = base64.urlsafe_b64encode(os.urandom(6)).decode().lower()
        payload_param_file = f"/tmp/validate_variable_impedance_payload_{random_string}.yaml"
        with open(payload_param_file, "w") as param_file:
            yaml.dump(payload_overrides, param_file, default_flow_style=False)

        variable_impedance_controller_spawner = Node(
            package="controller_manager",
            executable="spawner",
            namespace=resolved_namespace,
            arguments=[
                "variable_impedance_controller", "--controller-manager-timeout", "30",
                "--param-file", payload_param_file,
            ],
            output="screen",
        )

        # Loaded + configured but left INACTIVE: it claims the same effort interface as
        # variable_impedance_controller, so only one may be active at a time. Reset the arm
        # to initial_joint_positions on demand with:
        #   ros2 control switch_controllers --deactivate variable_impedance_controller \
        #       --activate move_to_start_example_controller
        # then, once it's done (watch the arm, or poll:
        #   ros2 param get /<namespace>/move_to_start_example_controller process_finished ),
        # swap back:
        #   ros2 control switch_controllers --deactivate move_to_start_example_controller \
        #       --activate variable_impedance_controller
        # CAUTION (real hardware): this is a plain joint-space PD controller, not a
        # velocity-limited trajectory generator -- activating it while the arm is far from
        # start_joint_configuration can command a large torque/velocity spike (a real fault
        # this project hit before, see calibrate_payload.py's preflight_delta_check
        # docstring). Don't activate it if the arm is currently far from
        # initial_joint_positions; jog it closer via Desk first if needed.
        move_to_start_spawner = Node(
            package="controller_manager",
            executable="spawner",
            namespace=resolved_namespace,
            arguments=[
                "move_to_start_example_controller", "--controller-manager-timeout", "30",
                "--inactive", "--param-file", payload_param_file,
            ],
            output="screen",
        )

        actions = [
            create_set_load_process(resolved_namespace, set_load_values),
            variable_impedance_controller_spawner,
            move_to_start_spawner,
        ]

        target_position_str = LaunchConfiguration("target_position").perform(context).strip()
        if target_position_str:
            target_position = [float(v) for v in target_position_str.split()]
            target_orientation = [
                float(v) for v in
                LaunchConfiguration("target_orientation").perform(context).split()
            ]
            target_frame_id = LaunchConfiguration("target_frame_id").perform(context)
            target_pose_process = create_target_pose_process(
                resolved_namespace, target_frame_id, target_position, target_orientation)
            # Wait for the spawner to exit (it does so once activation succeeds) rather than
            # publishing immediately: target_pose is a plain reliable topic, not latched, so
            # a publish before variable_impedance_controller has activated and subscribed
            # would just be dropped.
            actions.append(RegisterEventHandler(OnProcessExit(
                target_action=variable_impedance_controller_spawner,
                on_exit=[target_pose_process],
            )))

        return actions

    return LaunchDescription([
        DeclareLaunchArgument("arm_id", default_value="", description="ID of the type of arm"),
        DeclareLaunchArgument("arm_prefix", default_value="", description="Prefix for arm topics"),
        DeclareLaunchArgument(
            "namespace", default_value="follower",
            description="Namespace for the arm under validation"
        ),
        DeclareLaunchArgument(
            "urdf_file", default_value="fr3/fr3.urdf.xacro", description="Path to URDF file"
        ),
        DeclareLaunchArgument(
            "robot_ip", default_value="192.168.101.2",
            description="Hostname or IP address of the robot (ignored when use_fake_hardware)"
        ),
        DeclareLaunchArgument(
            "use_fake_hardware", default_value="true",
            description=(
                "Use fake hardware (mock_components/GenericSystem) -- no physical "
                "robot needed, but no real dynamics either. Set false for a real run."
            ),
        ),
        DeclareLaunchArgument(
            "controllers_yaml",
            default_value=PathJoinSubstitution([
                FindPackageShare("fr3_bilateral_teleop"), "config",
                "variable_impedance_validation_controllers.yaml",
            ]),
            description=(
                "Override for debugging (e.g. a scratch copy with different "
                "stiffness_shaping params); defaults to this package's own config."
            ),
        ),
        DeclareLaunchArgument(
            "initial_joint_positions",
            default_value=(
                "0.0 -0.7853981633974483 0.0 -2.356194490192345 0.0 "
                "1.5707963267948966 0.7853981633974483"
            ),
            description=(
                "Space-separated joint1..7 initial positions (radians) -- only used by "
                "fake hardware's initial_value state interface; real hardware ignores this "
                "(it reads its own encoders instead)."
            ),
        ),
        DeclareLaunchArgument(
            "target_position", default_value="",
            description=(
                "Space-separated target x y z (meters, in target_frame_id) -- if given, "
                "published once to variable_impedance_controller's target_pose topic "
                "(geometry_msgs/PoseStamped) right after it activates, so the controller has "
                "a fixed setpoint to servo to without needing "
                "probe_variable_impedance_sinusoid.py running. Leave empty (default) to "
                "publish nothing here -- the controller then just holds against its own "
                "last/default setpoint until something else publishes one."
            ),
        ),
        DeclareLaunchArgument(
            "target_orientation", default_value="0.0 0.0 0.0 1.0",
            description=(
                "Space-separated target orientation quaternion x y z w, paired with "
                "target_position above (ignored if target_position is empty). Defaults to "
                "identity."
            ),
        ),
        DeclareLaunchArgument(
            "target_frame_id", default_value="fr3_link0",
            description=(
                "frame_id stamped on the published target_pose -- matches base_frame in "
                "variable_impedance_validation_controllers.yaml (variable_impedance_controllers' "
                "CartesianController expects the target expressed in its configured "
                "base_frame)."
            ),
        ),
        DeclareLaunchArgument(
            "load_payload_calibration", default_value="true",
            description=(
                "If true, apply the calibrated payload (mass/center_of_mass/inertia) via "
                "set_load (firmware gravity compensation) and variable_impedance_controllers' own Pinocchio "
                "model (this controller's Coriolis compensation -- see cartesian_controller.yaml "
                "'payload' params). If false, explicitly zero the load instead of leaving "
                "whatever was previously configured on the robot."
            ),
        ),
        DeclareLaunchArgument(
            "payload_calibration_namespace", default_value="franka_teleop/follower",
            description=(
                "Which <namespace>_payload.yaml (see calibrate_payload.py) to load -- "
                "independent of 'namespace' above (this launch's own ROS namespace), since "
                "payload calibration is normally run once under teleop's namespace convention "
                "and reused here, not recalibrated per validation namespace."
            ),
        ),
        single_robot,
        set_collision_behavior,
        pose_broadcaster_spawner,
        OpaqueFunction(function=configure_payload),
    ])
