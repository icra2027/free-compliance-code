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

"""Brings up a single Franka arm (no teleop pairing) for payload calibration.

Loads joint_state_broadcaster, franka_robot_state_broadcaster (active), and
move_to_start_example_controller + payload_model_broadcaster -- the former inactive
(the calibration script activates/retargets it per waypoint), the latter active
immediately so `~/payload_model_broadcaster/model_snapshot` is available right away.

Run this first, then in a second terminal run the calibration script:
    ros2 run fr3_bilateral_teleop calibrate_payload.py --namespace <namespace>

See fr3_bilateral_teleop/README.md ("Payload calibration") before running against real hardware --
the arm moves through several static poses (and short transits between them) unattended.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, FindExecutable
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

# Same conservative thresholds used by teleop.launch.py -- calibration motions are slow
# point-to-point moves under move_to_start_example_controller, not teleop, but the arm
# should stop under the same collision behavior rather than an even looser default.
LOWER_TORQUE_THRESHOLDS_ACCELERATION = [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0]
LOWER_TORQUE_THRESHOLD_NOMINAL = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
LOWER_FORCE_THRESHOLDS_ACCELERATION = [20.0, 20.0, 20.0, 20.0, 20.0, 20.0]
LOWER_FORCE_THRESHOLDS_NOMINAL = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
UPPER_TORQUE_THRESHOLDS = [85.0, 85.0, 85.0, 85.0, 11.0, 11.0, 11.0]
UPPER_FORCE_THRESHOLDS = [85.0, 85.0, 85.0, 85.0, 11.0, 11.0]


def cvt_to_string(values):
    return ", ".join(str(float(v)) for v in values)


def generate_launch_description():
    namespace = LaunchConfiguration("namespace")

    single_robot = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("fr3_bilateral_teleop"),
                    "launch",
                    "teleop_single_robot.launch.py",
                ]
            )
        ),
        launch_arguments={
            "arm_id": LaunchConfiguration("arm_id"),
            "arm_prefix": LaunchConfiguration("arm_prefix"),
            "namespace": namespace,
            "urdf_file": LaunchConfiguration("urdf_file"),
            "robot_ip": LaunchConfiguration("robot_ip"),
            "load_gripper": LaunchConfiguration("load_gripper"),
            "use_fake_hardware": LaunchConfiguration("use_fake_hardware"),
            "fake_sensor_commands": LaunchConfiguration("use_fake_hardware"),
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
            f"[{cvt_to_string(UPPER_TORQUE_THRESHOLDS)}], ",
            "lower_torque_thresholds_nominal: ",
            f"[{cvt_to_string(LOWER_TORQUE_THRESHOLD_NOMINAL)}], ",
            "upper_torque_thresholds_nominal: ",
            f"[{cvt_to_string(UPPER_TORQUE_THRESHOLDS)}], ",
            "lower_force_thresholds_acceleration: ",
            f"[{cvt_to_string(LOWER_FORCE_THRESHOLDS_ACCELERATION)}], ",
            "upper_force_thresholds_acceleration: ",
            f"[{cvt_to_string(UPPER_FORCE_THRESHOLDS)}], ",
            "lower_force_thresholds_nominal: ",
            f"[{cvt_to_string(LOWER_FORCE_THRESHOLDS_NOMINAL)}], ",
            "upper_force_thresholds_nominal: ",
            f"[{cvt_to_string(UPPER_FORCE_THRESHOLDS)}] ",
            "}\"",
        ]],
        shell=True,
        name="set_calibration_collision_behavior",
        output="both",
    )

    move_to_start_speed_config = PathJoinSubstitution([
        FindPackageShare("fr3_bilateral_teleop"), "config", "calibration_move_to_start_speed.yaml",
    ])

    move_to_start_spawner = Node(
        package="controller_manager",
        executable="spawner",
        namespace=namespace,
        arguments=[
            "move_to_start_example_controller",
            "--controller-manager-timeout", "30",
            "--inactive",
            "--param-file", move_to_start_speed_config,
        ],
        output="screen",
    )

    payload_model_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        namespace=namespace,
        arguments=[
            "payload_model_broadcaster",
            "--controller-manager-timeout", "30",
        ],
        output="screen",
    )

    return LaunchDescription([
        DeclareLaunchArgument("arm_id", default_value="", description="ID of the type of arm"),
        DeclareLaunchArgument("arm_prefix", default_value="", description="Prefix for arm topics"),
        DeclareLaunchArgument(
            "namespace", default_value="follower",
            description="Namespace for the arm being calibrated"
        ),
        DeclareLaunchArgument(
            "urdf_file", default_value="fr3/fr3.urdf.xacro", description="Path to URDF file"
        ),
        DeclareLaunchArgument(
            "robot_ip", default_value="192.168.101.2",
            description="Hostname or IP address of the robot"
        ),
        DeclareLaunchArgument(
            "load_gripper", default_value="true",
            description="Franka Hand attached (the 3D-printed wiper mount screws onto it)"
        ),
        DeclareLaunchArgument(
            "use_fake_hardware", default_value="false", description="Use fake hardware"
        ),
        single_robot,
        set_collision_behavior,
        move_to_start_spawner,
        payload_model_broadcaster_spawner,
    ])
