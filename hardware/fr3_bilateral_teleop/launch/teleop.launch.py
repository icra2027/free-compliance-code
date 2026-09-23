#  Copyright (c) 2025 Franka Robotics GmbH
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

import math
import os
import base64
from typing import Any, Dict, List, Optional
import yaml
from launch import LaunchDescription, LaunchDescriptionEntity
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    ExecuteProcess
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    FindExecutable
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


LEADER_NAMESPACE = "leader"
FOLLOWER_NAMESPACE = "follower"

LEADER_CONTROLLER_NAME = "leader_controller"
FOLLOWER_CONTROLLER_NAME = "follower_controller"

# Both robots publish their own sensed external joint torque and measured joint
# state on this same (per-namespace) topic name via franka_robot_state_broadcaster.
# It is reused for both force channels of the 4-channel bilateral controller:
#   - leader_namespace/EXTERNAL_JOINT_TORQUES_TOPIC:   force channel leader -> follower
#     (the operator's own applied force, fed forward into the follower)
#   - follower_namespace/EXTERNAL_JOINT_TORQUES_TOPIC: force channel follower -> leader
#     (the follower's sensed contact force, reflected back to the operator)
EXTERNAL_JOINT_TORQUES_TOPIC = "franka_robot_state_broadcaster/external_joint_torques"
FOLLOWER_INPUT_TOPIC = "franka_robot_state_broadcaster/measured_joint_states"

LOWER_TORQUE_THRESHOLDS_ACCELERATION = [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0]
LOWER_TORQUE_THRESHOLD_NOMINAL = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
LOWER_FORCE_THRESHOLDS_ACCELERATION = [20.0, 20.0, 20.0, 20.0, 20.0, 20.0]
LOWER_FORCE_THRESHOLDS_NOMINAL = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0]

TIMEOUT_KEY = 'input_topic_timeout'
ALPHA_KEY = 'alpha'
K_GAINS_KEY = 'k_gains'
D_GAINS_KEY = 'd_gains'
# Force channel follower -> leader: scales the follower's sensed contact torque
# before it is reflected back to the leader. Tunes "feel", independent of k_gains/d_gains.
FORCE_REFLECTION_GAINS_KEY = 'force_reflection_gains'
# Force channel leader -> follower: scales the leader's own sensed external torque
# (the operator's applied force) before it is fed forward into the follower.
FORCE_FEEDFORWARD_GAINS_KEY = 'force_feedforward_gains'
MAX_TORQUE_ACCELERATION_KEY = 'upper_torque_thresholds_acceleration'
MAX_TORQUE_NOMINAL_KEY = 'upper_torque_thresholds_nominal'
MAX_FORCE_ACCELERATION_KEY = 'upper_force_thresholds_acceleration'
MAX_FORCE_NOMINAL_KEY = 'upper_force_thresholds_nominal'
LOAD_GRIPPER_KEY = 'load_gripper'
FAKE_HARDWARE_KEY = 'fake_hardware'
# The joint-space pose move_to_start_example_controller drives each arm to before teleop
# activates (also the pose teleop_coordinator's free-space re-zero re-uses, see the
# fr3_bilateral_teleop README's "Automatic free-space re-zero on every session"). Settable per
# base/pair/robot exactly like k_gains/d_gains below -- e.g. a pair whose task fixture sits
# somewhere else on the table can start from a different, still joint-limit-safe, posture.
# NOTE: config/workspace_constraints.yaml's nullspace_posture is a separate, independently
# documented copy of whatever this default used to be (Franka's standard "ready" pose) -- it
# is not read from here, so overriding HOME_POSITION_KEY does not change it; update that file
# too if the nullspace target should track a new home position.
HOME_POSITION_KEY = 'start_joint_configuration'

# Where calibrate_payload.py writes <namespace>_payload.yaml (see fr3_bilateral_teleop README,
# "Payload calibration"). Same env var / default the calibration script itself uses.
PAYLOAD_OUTPUT_DIR_ENV = 'TELEOP_PAYLOAD_OUTPUT_DIR'
PAYLOAD_OUTPUT_DIR_DEFAULT = '/home/robot/franka_ros2_ws/franka_data/franka_teleop_payload/'
URDF_KEY = 'urdf_file'
NAMESPACE_KEY = 'namespace'
BASE_NAMESPACE_KEY = 'base_namespace'
ROBOT_IP_KEY = 'robot_ip'
ARM_ID_KEY = 'arm_id'
ARM_PREFIX_KEY = 'arm_prefix'

default_parameters = {
    TIMEOUT_KEY: 2500000,
    ALPHA_KEY: [3, 3, 3, 3, 1, 1, 1],
    K_GAINS_KEY: [600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0],
    D_GAINS_KEY: [30.0, 30.0, 30.0, 30.0, 10.0, 10.0, 5.0],
    FORCE_REFLECTION_GAINS_KEY: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    FORCE_FEEDFORWARD_GAINS_KEY: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    MAX_TORQUE_ACCELERATION_KEY: [85.0, 85.0, 85.0, 85.0, 11.0, 11.0, 11.0],
    MAX_TORQUE_NOMINAL_KEY: [85.0, 85.0, 85.0, 85.0, 11.0, 11.0, 11.0],
    MAX_FORCE_ACCELERATION_KEY: [85.0, 85.0, 85.0, 85.0, 11.0, 11.0],
    MAX_FORCE_NOMINAL_KEY: [85.0, 85.0, 85.0, 85.0, 11.0, 11.0],
    LOAD_GRIPPER_KEY: False,
    FAKE_HARDWARE_KEY: False,
    # Franka's standard "ready" pose -- matches move_to_start_example_controller's own
    # on_init() default exactly, so leaving this unset changes nothing.
    HOME_POSITION_KEY: [
        0.0, -math.pi / 4, 0.0, -3.0 * math.pi / 4, 0.0, math.pi / 2, -3.0 * math.pi / 4,
    ],
    URDF_KEY: "fr3/fr3.urdf.xacro",
    ARM_ID_KEY: "",
    ARM_PREFIX_KEY: "",
    BASE_NAMESPACE_KEY: None,
    ROBOT_IP_KEY: "you_need_to_the_configure_robot_address",
}


def load_yaml(file_path: str) -> Dict[str, Any]:
    """Load a YAML file and return its content as a dictionary."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def construct_namespace(*names):
    """Construct a namespace from the provided names, filtering out empty strings."""
    return "/".join(filter(None, names))


def cvt_to_string(input_list: List) -> str:
    """Convert a list of numbers to a comma-separated string."""
    return ', '.join(map(str, map(float, input_list)))


def cvt_to_float_list(input_list: List) -> List[float]:
    """Convert a list of strings or numbers to a list of floats."""
    return list(map(float, input_list))


def get_action_remapping(action: str, target_action: str) -> list:
    """To remap an action one has to remap all topics associated with that action."""
    return [
        (f'{action}/_action/feedback', f'{target_action}/_action/feedback'),
        (f'{action}/_action/status', f'{target_action}/_action/status'),
        (f'{action}/_action/cancel_goal', f'{target_action}/_action/cancel_goal'),
        (f'{action}/_action/get_result', f'{target_action}/_action/get_result'),
        (f'{action}/_action/send_goal', f'{target_action}/_action/send_goal'),
    ]


def create_controller_config(
    input_topic_name: str,
    force_input_topic_name: str,
    config
) -> str:
    """Create a YAML configuration file for the teleoperation controllers."""
    random_string: str = base64.urlsafe_b64encode(os.urandom(6)).decode().lower()
    target_file_name = f'/tmp/launch_params_{random_string}'

    input_topic_timeout = config[TIMEOUT_KEY]
    alpha: List[float] = cvt_to_float_list(config[ALPHA_KEY])
    k_gains: List[float] = cvt_to_float_list(config[K_GAINS_KEY])
    d_gains: List[float] = cvt_to_float_list(config[D_GAINS_KEY])
    # Force coupling gains (feel), independent of the position coupling gains above (tracking).
    force_reflection_gains: List[float] = cvt_to_float_list(config[FORCE_REFLECTION_GAINS_KEY])
    force_feedforward_gains: List[float] = cvt_to_float_list(config[FORCE_FEEDFORWARD_GAINS_KEY])
    home_position: List[float] = cvt_to_float_list(config[HOME_POSITION_KEY])
    if len(home_position) != 7:
        raise ValueError(
            f"{HOME_POSITION_KEY} must have exactly 7 joint values, got {len(home_position)}: "
            f"{home_position}"
        )

    config_data = {
        "/**": {
            "leader_controller": {
                "ros__parameters": {
                    "arm_id": "fr3",
                    "input_topic": input_topic_name,
                    "input_topic_timeout": input_topic_timeout,
                    "use_input_topic": True,
                    "alpha": alpha,
                    # Force channel follower -> leader (feel).
                    "force_reflection_gains": force_reflection_gains,
                }
            },
            "follower_controller": {
                "ros__parameters": {
                    "arm_id": "fr3",
                    "input_topic": input_topic_name,
                    "input_topic_timeout": input_topic_timeout,
                    "k_gains": k_gains,
                    "d_gains": d_gains,
                    # Force channel leader -> follower (feel), separate from k_gains/d_gains
                    # (tracking) above.
                    "force_input_topic": force_input_topic_name,
                    "force_feedforward_gains": force_feedforward_gains,
                }
            },
            # Applied to this robot's own move_to_start_example_controller instance (spawned
            # per-namespace in add_robot_launch_config below) via this same generated
            # --param-file, so leader and follower can each get their own home position.
            "move_to_start_example_controller": {
                "ros__parameters": {
                    "start_joint_configuration": home_position,
                }
            },
        },
    }

    with open(target_file_name, 'w') as param_file:
        param_file.write(yaml.dump(config_data, default_flow_style=False))

    return target_file_name


def default_set_load_values() -> Dict[str, Any]:
    """Return the no-extra-load default set_load values.

    Nothing beyond whatever the hardware/driver already assumes (e.g. the Franka Hand's own
    factory defaults if load_gripper is set) -- used when calibration hasn't been run or
    didn't succeed, so teleop can still start rather than being blocked on it.
    """
    return {'mass': 0.0, 'center_of_mass': [0.0, 0.0, 0.0], 'load_inertia': [0.0] * 9}


def load_payload_calibration(namespace: str) -> Dict[str, Any]:
    """Read <namespace>_payload.yaml written by calibrate_payload.py.

    See fr3_bilateral_teleop README, "Payload calibration". Missing/failed/unreadable
    calibration is a loud warning, not a launch failure -- unlike the free-space re-zero,
    this is a per-hardware-config calibration (redo it when the attached tool changes), not
    a per-session one, so gating every teleop launch on it would be the wrong failure mode.
    """
    output_dir = os.environ.get(PAYLOAD_OUTPUT_DIR_ENV, PAYLOAD_OUTPUT_DIR_DEFAULT)
    # Sanitize the same way calibrate_payload.py does -- namespace may be nested (e.g.
    # 'franka_teleop/follower'), and a bare '/' in the filename would need a matching
    # subdirectory that nothing here creates.
    file_namespace = namespace.strip('/').replace('/', '_')
    payload_path = os.path.join(output_dir, f'{file_namespace}_payload.yaml')

    if not os.path.exists(payload_path):
        print(
            f"[teleop.launch] WARNING: no payload calibration file at '{payload_path}' for "
            f"'{namespace}'. Run calibrate_payload.launch.py + calibrate_payload.py first. "
            f"Falling back to a zero/no-extra-load default -- force/compliance data from this "
            f"session will carry whatever bias the undeclared payload introduces."
        )
        return default_set_load_values()

    try:
        with open(payload_path, 'r') as payload_file:
            report = yaml.safe_load(payload_file)
        if report.get('status') != 'ok' or 'set_load' not in report:
            print(
                f"[teleop.launch] WARNING: payload calibration at '{payload_path}' has "
                f"status={report.get('status')!r} (not 'ok') -- falling back to a zero/"
                f"no-extra-load default for '{namespace}'."
            )
            return default_set_load_values()
        return report['set_load']
    except (OSError, yaml.YAMLError, AttributeError) as exc:
        print(
            f"[teleop.launch] WARNING: failed to read payload calibration at "
            f"'{payload_path}': {exc}. Falling back to a zero/no-extra-load default for "
            f"'{namespace}'."
        )
        return default_set_load_values()


def create_set_load_process(namespace: str, set_load_values: Dict[str, Any]) -> ExecuteProcess:
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
        name='set_robot_payload',
        output='both',
    )


def add_robot_launch_config(
        robot_config: Dict[str, Any],
        namespace: str,
        input_topic_name: str,
        force_input_topic_name: str,
        controller_name: str,
        apply_payload_calibration: bool = False,
        load_payload_from_yaml: bool = True,
) -> List[LaunchDescriptionEntity]:
    """Create the shared launch configuration for a single robot."""
    launch_config = []

    config_file = create_controller_config(
        input_topic_name,
        force_input_topic_name,
        robot_config
    )

    launch_config.append(ExecuteProcess(
        cmd=[[
            FindExecutable(name="ros2"),
            " service call ",
            f"/{namespace}/service_server/set_full_collision_behavior ",
            "franka_msgs/srv/SetFullCollisionBehavior ",
            "\"{ ",
            "lower_torque_thresholds_acceleration: ",
            f"[{cvt_to_string(LOWER_TORQUE_THRESHOLDS_ACCELERATION)}], ",
            "upper_torque_thresholds_acceleration: "
            f"[{cvt_to_string(robot_config[MAX_TORQUE_ACCELERATION_KEY])}], ",
            "lower_torque_thresholds_nominal: ",
            f"[{cvt_to_string(LOWER_TORQUE_THRESHOLD_NOMINAL)}], ",
            "upper_torque_thresholds_nominal: ",
            f"[{cvt_to_string(robot_config[MAX_TORQUE_NOMINAL_KEY])}], ",
            "lower_force_thresholds_acceleration: ",
            f"[{cvt_to_string(LOWER_FORCE_THRESHOLDS_ACCELERATION)}], ",
            "upper_force_thresholds_acceleration: ",
            f"[{cvt_to_string(robot_config[MAX_FORCE_ACCELERATION_KEY])}], ",
            "lower_force_thresholds_nominal: ",
            f"[{cvt_to_string(LOWER_FORCE_THRESHOLDS_NOMINAL)}], ",
            "upper_force_thresholds_nominal: ",
            f"[{cvt_to_string(robot_config[MAX_FORCE_NOMINAL_KEY])}] ",
            "}\"",
        ]],
        shell=True,
        name='set_robot_collision_behavior',
        output='both',
    ))

    if apply_payload_calibration:
        set_load_values = (
            load_payload_calibration(namespace) if load_payload_from_yaml
            else default_set_load_values()
        )
        launch_config.append(create_set_load_process(namespace, set_load_values))

    launch_config.append(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('fr3_bilateral_teleop'),
                    'launch',
                    'teleop_single_robot.launch.py'
                ])
            ),
            launch_arguments={
                'arm_id': robot_config[ARM_ID_KEY],
                'arm_prefix': robot_config[ARM_PREFIX_KEY],
                'namespace': namespace,
                'urdf_file': robot_config[URDF_KEY],
                'robot_ip': robot_config[ROBOT_IP_KEY],
                'load_gripper': str(robot_config[LOAD_GRIPPER_KEY]),
                'use_fake_hardware': str(robot_config[FAKE_HARDWARE_KEY]),
                'fake_sensor_commands': str(robot_config[FAKE_HARDWARE_KEY]),
                'joint_sources': ','.join(["joint_states", "franka_gripper/joint_states"]),
                'joint_state_rate': str(30),
            }.items(),
        )
    )

    controller_manager = Node(
        package='controller_manager',
        executable='spawner',
        namespace=namespace,
        arguments=[
            'move_to_start_example_controller',
            controller_name,
            '--controller-manager-timeout', '30',
            '--inactive',
            '--param-file', config_file
        ],
        parameters=[PathJoinSubstitution([
            FindPackageShare('fr3_bilateral_teleop'), 'config', "teleop_controllers.yaml",
        ])],
        output='screen',
    )
    launch_config.append(controller_manager)

    return launch_config


def add_pair(pair_config, load_payload_from_yaml: bool = True) -> List[LaunchDescriptionEntity]:
    """Create the launch configuration for a pair of robots (leader and follower)."""
    try:
        pair_namespace: str = pair_config['namespace']
    except KeyError:
        raise Exception(
            "You need to specify a namespace for each pair!"
        )

    base_namespace: Optional[str] = pair_config.get(BASE_NAMESPACE_KEY, None)

    leader_namespace = construct_namespace(base_namespace, pair_namespace, "leader")
    follower_namespace = construct_namespace(base_namespace, pair_namespace, "follower")

    teleop_coordinator_node = Node(
        package='fr3_bilateral_teleop',
        executable='teleop_coordinator',
        arguments=[
            leader_namespace, LEADER_CONTROLLER_NAME, follower_namespace, FOLLOWER_CONTROLLER_NAME
        ],
        output='both',
    )

    launch_config = [teleop_coordinator_node]

    leader_config = pair_config.pop('leader')
    follower_config = pair_config.pop('follower')

    # Create the launch configuration for the leader robot
    resolved_leader_config = pair_config | leader_config

    launch_config.extend(
        add_robot_launch_config(
            resolved_leader_config,
            leader_namespace,
            f"/{follower_namespace}/{EXTERNAL_JOINT_TORQUES_TOPIC}",
            "",  # the leader controller has no force_input_topic parameter
            LEADER_CONTROLLER_NAME,
        )
    )

    # Create the launch configuration for the follower robot
    resolved_follower_config = pair_config | follower_config

    launch_config.extend(
        add_robot_launch_config(
            resolved_follower_config,
            follower_namespace,
            f"/{leader_namespace}/{FOLLOWER_INPUT_TOPIC}",
            f"/{leader_namespace}/{EXTERNAL_JOINT_TORQUES_TOPIC}",
            FOLLOWER_CONTROLLER_NAME,
            apply_payload_calibration=True,
            load_payload_from_yaml=load_payload_from_yaml,
        )
    )

    if resolved_follower_config[LOAD_GRIPPER_KEY]:
        teleop_gripper_node = Node(
            package='fr3_bilateral_teleop',
            executable='teleop_gripper_node',
            output='both',
            namespace=follower_namespace,
            remappings=[
                (
                    "~/leader/franka_gripper/joint_states",
                    f"/{leader_namespace}/franka_gripper/joint_states"
                ),
                *get_action_remapping(
                    "~/leader/franka_gripper/homing", f"/{leader_namespace}/franka_gripper/homing"
                ),
                *get_action_remapping(
                    "~/follower/franka_gripper/homing",
                    f"/{follower_namespace}/franka_gripper/homing"
                ),
                *get_action_remapping(
                    "~/follower/franka_gripper/grasp",
                    f"/{follower_namespace}/franka_gripper/grasp"
                ),
                *get_action_remapping(
                    "~/follower/franka_gripper/move",
                    f"/{follower_namespace}/franka_gripper/move"
                ),
            ],
        )
        launch_config.append(teleop_gripper_node)

    return launch_config


def generate_launch_configuration_for_all_robots(context) -> List[LaunchDescriptionEntity]:
    """
    Generate the launch configuration for all robots.

    Configuration is based on the provided robot configuration file.
    """
    config_file = LaunchConfiguration('robot_config_file').perform(context)
    base_config = load_yaml(config_file)

    load_payload_from_yaml = LaunchConfiguration('load_payload_calibration').perform(
        context).lower() in ('true', '1', 'yes')

    pairs = base_config.pop('pairs', None)

    assert pairs, "You need to add teleoperation pairs (leader and follower) to your config file!"

    launch_config = []

    resolved_base_config = default_parameters | base_config

    for pair_config in pairs:

        resolved_pair_config = resolved_base_config | pair_config

        launch_config.extend(add_pair(resolved_pair_config, load_payload_from_yaml))

    return launch_config


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'robot_config_file',
            default_value=PathJoinSubstitution([
                FindPackageShare('fr3_bilateral_teleop'), 'config', 'fr3_teleop_config.yaml'
            ]),
            description='Path to the robot configuration file to load',
        ),
        DeclareLaunchArgument(
            'load_payload_calibration',
            default_value='true',
            description=(
                "If true, apply the follower's <namespace>_payload.yaml via set_load at "
                "startup. If false, explicitly zero the load instead (rather than leaving "
                "whatever was previously configured on the robot)."
            ),
        ),
        OpaqueFunction(function=generate_launch_configuration_for_all_robots),
    ])
