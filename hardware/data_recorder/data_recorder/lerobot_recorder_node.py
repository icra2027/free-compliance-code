#!/usr/bin/env python3

import argparse
import inspect
import importlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import message_filters
import numpy as np
import rclpy
import torch
from controller_manager_msgs.srv import SwitchController
from cv_bridge import CvBridge
from franka_msgs.msg import FrankaRobotState
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import GetParameters
from rcl_interfaces.srv import SetParameters
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.utilities import remove_ros_args
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, JointState

try:
    # Newer lerobot versions expose datasets under lerobot.datasets.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    try:
        # Fallback for older layouts.
        LeRobotDataset = importlib.import_module(
            "lerobot.common.datasets.lerobot_dataset"
        ).LeRobotDataset
    except ImportError as exc:
        raise ImportError(
            "Could not import LeRobotDataset. Install/upgrade `lerobot` in your ROS 2 Python environment."
        ) from exc


# The T1 whiteboard only ever carries a red and a blue mark (previously red/blue/green/black --
# the extra two were dropped, so a 2-way referent choice is all the instruction ever names).
VALID_COLOURS = ("red", "blue")
# The adverb axis / target force bands.
VALID_MANNERS = ("gently", "normally", "firmly")
# Both {colour} and {manner} are substituted; the resulting wording ("wipe the red mark
# normally") is what compliance-vla's dataset_io.parse_referent_from_task /
# parse_manner_from_task read back out of the recorded task string, so keep the colour word
# and the trailing adverb in place if this is overridden via the language_instruction param.
DEFAULT_INSTRUCTION_TEMPLATE = "wipe the {colour} mark {manner}"


class FrankaLeRobotRecorder(Node):
    def __init__(self, colour: Optional[str] = None, manner: Optional[str] = None) -> None:
        # colour/manner come from the command line (see main()) and are fixed for the whole
        # collection -- one recorder launch is one session is one (colour, manner) pair. They
        # used to be typed by the operator in the GUI before every single episode, which made
        # every episode's label a fresh chance to typo/forget and left the referent word
        # unverifiable after the fact; a per-launch value is checked once, here, against
        # VALID_COLOURS/VALID_MANNERS and then applies identically to every episode.
        super().__init__("data_recorder")

        self.declare_parameter("dataset_root", "/tmp/lerobot_data")
        self.declare_parameter("repo_id", "local/franka_vla_multimodal")
        self.declare_parameter("dataset_fps", 30)
        self.declare_parameter("robot_type", "franka")
        self.declare_parameter("episode_task", "sample_task")
        self.declare_parameter("language_instruction", "")
        self.declare_parameter("use_videos", True)
        self.declare_parameter("record_wrist_camera", True)
        self.declare_parameter("record_wrist_depth", False)
        self.declare_parameter("wrist_rgb_width", 224)
        self.declare_parameter("wrist_rgb_height", 224)
        self.declare_parameter("scene_rgb_width", 224)
        self.declare_parameter("scene_rgb_height", 224)
        self.declare_parameter("joint_topic", "/franka_teleop/follower/franka_robot_state_broadcaster/measured_joint_states")
        self.declare_parameter(
            "pose_topic",
            "/franka_teleop/follower/franka_robot_state_broadcaster/current_pose",
        )
        self.declare_parameter("gripper_joint_topic", "/franka_teleop/follower/joint_states")
        self.declare_parameter(
            "robot_state_topic",
            "/franka_teleop/follower/franka_robot_state_broadcaster/robot_state",
        )
        self.declare_parameter("record_wrench_forces", True)
        # The identifiability argument substitutes x_eq := x_l (the
        # LEADER's pose), which is what makes K(t) regressable at all -- without this, the
        # dataset only has the follower's own pose (twice, as both observation.state and
        # action) and extract_impedance_labels.py's regression has nothing to run against.
        # This was missing entirely before today (confirmed: pose_topic/joint_topic both
        # defaulted to the follower namespace, no leader parameter existed anywhere in this
        # file) -- every episode recorded before this fix landed is permanently follower-only,
        # since the leader's trajectory during those sessions was never subscribed to or
        # logged and cannot be recovered after the fact.
        self.declare_parameter(
            "leader_pose_topic",
            "/franka_teleop/leader/franka_robot_state_broadcaster/current_pose",
        )
        self.declare_parameter("record_leader_pose", True)
        # The policy's proprioceptive input is "(q, q_dot, x_f)" -- q was already
        # covered by observation.state (joint positions, from the same JointState message this
        # just reads .velocity off of instead of .position), but q_dot was entirely absent
        # before this. No new topic/subscription needed.
        self.declare_parameter("record_joint_velocity", True)
        # Default matches franka_mobile_sensors/launch/cameras/realsense_cameras.launch.py's
        # default_sensor_suite, which puts the wrist RealSense under namespace=wrist,
        # camera_name=wrist_camera (topic /wrist/wrist_camera/color/image_raw). Confirm with
        # `ros2 topic list` if the sensor suite config changes.
        self.declare_parameter("wrist_rgb_topic", "/wrist/wrist_camera/color/image_raw")
        self.declare_parameter(
            "wrist_depth_topic", "/wrist/wrist_camera/aligned_depth_to_color/image_raw"
        )
        # Fixed third-person scene view. Default matches orbbec_camera's femto_bolt.launch.py
        # with its default camera_name:=camera (color image topic is /camera/color/image_raw;
        # see run_deets.md). Depth is deliberately not consumed here -- the Femto
        # Bolt's depth engine was found to crash on launch in this container (no working GL/EGL)
        # and T1's extraction pipeline only needs scene RGB anyway.
        self.declare_parameter("scene_rgb_topic", "/camera/color/image_raw")
        # Fallback capture path that bypasses orbbec_camera/the ROS driver entirely and reads the
        # Femto Bolt's color stream straight off its V4L2 UVC node via cv2.VideoCapture. Exists
        # because the ROS driver has repeatedly failed to even open the device in this container
        # (usbEnumerator openUsbDevice failed! status:113 -- see
        # fix_femto_bolt_usb_perms.sh): the kernel's own UVC driver and V4L2
        # generally tolerate this environment's lack of a udev daemon better than the vendor SDK's
        # own USB enumeration does. Off by default -- the ROS driver path is preferred when it
        # works, since it also gives depth/IR/IMU and proper camera_info.
        self.declare_parameter("use_scene_cv2_capture", False)
        self.declare_parameter("scene_cv2_device", 0)
        # If set, takes priority over scene_cv2_device (e.g. "/dev/video6"). The Femto Bolt
        # exposes multiple /dev/video* nodes (one pair for color's alternate UVC formats, one
        # pair for depth/IR) and which numeric index lands on the actual color stream is not
        # guaranteed to be stable across replugs/reboots -- find the right path with:
        #   for v in /dev/video*; do echo "$v: $(cat /sys/class/video4linux/$(basename $v)/name)"; done
        # then confirm it's really color (not just correctly named) by checking the preview in
        # the recorder GUI once running.
        self.declare_parameter("scene_cv2_device_path", "")
        self.declare_parameter("scene_cv2_width", 224)
        self.declare_parameter("scene_cv2_height", 224)
        # The Femto Bolt has been observed to drop off the USB bus and re-enumerate
        # mid-session, sometimes failing to reopen. If no scene frame arrives for longer than
        # this, the GUI flags it and a throttled warning is logged, so a silently-frozen scene
        # feed doesn't get baked into a demo unnoticed.
        self.declare_parameter("scene_camera_timeout_sec", 1.0)
        self.declare_parameter("sync_queue_size", 30)
        self.declare_parameter("sync_slop", 0.05)
        self.declare_parameter("visualize_gui", True)
        self.declare_parameter("gui_window_name", "LeRobot Recorder")
        self.declare_parameter("reset_controller_namespace", "/franka_teleop/follower")
        self.declare_parameter("reset_target_controller", "follower_controller")
        self.declare_parameter("reset_leader_enabled", True)
        self.declare_parameter("reset_leader_namespace", "/franka_teleop/leader")
        self.declare_parameter("reset_leader_target_controller", "leader_controller")
        self.declare_parameter("reset_move_to_start_controller", "move_to_start_example_controller")
        self.declare_parameter("reset_wait_timeout_sec", 20.0)
        # Operator/colour/manner are folded into the task label (e.g. "wipe the red
        # mark gently") and, along with session_id, are written to a session manifest sidecar
        # so the dataset build can split by session without re-deriving groupings from
        # timestamps. session_id defaults to this process's start time if left empty -- one
        # recorder launch is one session.
        #
        # colour/manner are normally given as `--colour`/`--manner` on the command line; these
        # parameters exist so a launch file can supply them instead. The command line wins when
        # both are present.
        self.declare_parameter("operator_id", "A")
        self.declare_parameter("colour", "")
        self.declare_parameter("manner", "")
        self.declare_parameter("session_id", "")

        self.dataset_root = (
            self.get_parameter("dataset_root").get_parameter_value().string_value
        )
        self.repo_id = self.get_parameter("repo_id").get_parameter_value().string_value
        self.dataset_fps = (
            self.get_parameter("dataset_fps").get_parameter_value().integer_value
        )
        self.robot_type = self.get_parameter("robot_type").get_parameter_value().string_value
        self.episode_task = self.get_parameter("episode_task").get_parameter_value().string_value
        self.language_instruction = (
            self.get_parameter("language_instruction").get_parameter_value().string_value
        ).strip()
        self.use_videos = self.get_parameter("use_videos").get_parameter_value().bool_value
        self.record_wrist_camera = (
            self.get_parameter("record_wrist_camera").get_parameter_value().bool_value
        )
        self.record_wrist_depth = (
            self.get_parameter("record_wrist_depth").get_parameter_value().bool_value
        )
        self.record_wrench_forces = (
            self.get_parameter("record_wrench_forces").get_parameter_value().bool_value
        )
        self.record_leader_pose = (
            self.get_parameter("record_leader_pose").get_parameter_value().bool_value
        )
        self.record_joint_velocity = (
            self.get_parameter("record_joint_velocity").get_parameter_value().bool_value
        )
        self.wrist_rgb_width = self.get_parameter("wrist_rgb_width").get_parameter_value().integer_value
        self.wrist_rgb_height = self.get_parameter("wrist_rgb_height").get_parameter_value().integer_value
        self.scene_rgb_width = self.get_parameter("scene_rgb_width").get_parameter_value().integer_value
        self.scene_rgb_height = self.get_parameter("scene_rgb_height").get_parameter_value().integer_value
        self.use_scene_cv2_capture = (
            self.get_parameter("use_scene_cv2_capture").get_parameter_value().bool_value
        )
        self.scene_cv2_device = self.get_parameter("scene_cv2_device").get_parameter_value().integer_value
        self.scene_cv2_device_path = (
            self.get_parameter("scene_cv2_device_path").get_parameter_value().string_value.strip()
        )
        self.scene_cv2_width = self.get_parameter("scene_cv2_width").get_parameter_value().integer_value
        self.scene_cv2_height = self.get_parameter("scene_cv2_height").get_parameter_value().integer_value
        self.scene_camera_timeout_sec = (
            self.get_parameter("scene_camera_timeout_sec").get_parameter_value().double_value
        )
        self.visualize_gui = self.get_parameter("visualize_gui").get_parameter_value().bool_value
        self.gui_window_name = (
            self.get_parameter("gui_window_name").get_parameter_value().string_value
        )
        self.reset_controller_namespace = (
            self.get_parameter("reset_controller_namespace").get_parameter_value().string_value
        ).rstrip("/")
        self.reset_target_controller = (
            self.get_parameter("reset_target_controller").get_parameter_value().string_value
        )
        self.reset_leader_enabled = (
            self.get_parameter("reset_leader_enabled").get_parameter_value().bool_value
        )
        self.reset_leader_namespace = (
            self.get_parameter("reset_leader_namespace").get_parameter_value().string_value
        ).rstrip("/")
        self.reset_leader_target_controller = (
            self.get_parameter("reset_leader_target_controller").get_parameter_value().string_value
        )
        self.reset_move_to_start_controller = (
            self.get_parameter("reset_move_to_start_controller").get_parameter_value().string_value
        )
        self.reset_wait_timeout_sec = (
            self.get_parameter("reset_wait_timeout_sec").get_parameter_value().double_value
        )
        self.operator_id = self.get_parameter("operator_id").get_parameter_value().string_value.strip()
        self.colour = self._resolve_fixed_choice(
            "colour",
            colour,
            self.get_parameter("colour").get_parameter_value().string_value,
            VALID_COLOURS,
        )
        self.manner = self._resolve_fixed_choice(
            "manner",
            manner,
            self.get_parameter("manner").get_parameter_value().string_value,
            VALID_MANNERS,
        )
        self.instruction_template = self.language_instruction or DEFAULT_INSTRUCTION_TEMPLATE
        self._validate_instruction_template()
        session_id_param = self.get_parameter("session_id").get_parameter_value().string_value.strip()
        self.session_id = session_id_param or time.strftime("%Y%m%d_%H%M%S")
        self.session_manifest_path = Path(self.dataset_root) / "session_manifest.jsonl"

        self.scene_capture: Optional[cv2.VideoCapture] = None
        self._stats_aggregation_disabled_logged = False
        self.bridge = CvBridge()
        self._switch_controller_clients: Dict[str, Any] = {}
        self._move_to_start_get_parameters_clients: Dict[str, Any] = {}
        self._move_to_start_set_parameters_clients: Dict[str, Any] = {}

        self.features: Dict[str, Dict[str, Any]] = {
            "observation.state": {
                "dtype": "float32",
                "shape": (7,),
            },
            "action": {
                "dtype": "float32",
                "shape": (7,),
            },
            "observation.images.scene_rgb": {
                "dtype": "uint8",
                "shape": (self.scene_rgb_height, self.scene_rgb_width, 3),
                "type": "video",
                "compressed": True,
            },
        }
        if self.record_wrist_camera:
            self.features["observation.images.wrist_rgb"] = {
                "dtype": "uint8",
                "shape": (self.wrist_rgb_height, self.wrist_rgb_width, 3),
                "type": "video",
                "compressed": True,
            }
            if self.record_wrist_depth:
                self.features["observation.images.wrist_depth"] = {
                    "dtype": "uint16",
                    "shape": (self.wrist_rgb_height, self.wrist_rgb_width, 1),
                    "type": "image",
                    "compressed": False,
                }
        if self.record_wrench_forces:
            self.features["observation.wrench.external_base"] = {
                "dtype": "float32",
                "shape": (6,),
            }
            self.features["observation.wrench.external_stiffness"] = {
                "dtype": "float32",
                "shape": (6,),
            }
        if self.record_leader_pose:
            # x_l(t) -- the leader's EE pose (xyz + rpy, same convention as current_ee_pose_6d
            # below). This is the independent equilibrium measurement the method's
            # identifiability argument substitutes for x_eq; without it,
            # extract_impedance_labels.py's K(t) regression has nothing to run against.
            self.features["observation.leader_pose"] = {
                "dtype": "float32",
                "shape": (6,),
            }
        if self.record_joint_velocity:
            # Follower q_dot, same 7-joint ordering as observation.state's q.
            self.features["observation.velocity"] = {
                "dtype": "float32",
                "shape": (7,),
            }

        self.dataset = self._init_lerobot_dataset()
        self._language_instruction_feature_enabled = self._dataset_supports_feature(
            "language_instruction"
        )
        self._wrench_base_feature_enabled = self._dataset_supports_feature(
            "observation.wrench.external_base"
        )
        self._wrench_stiffness_feature_enabled = self._dataset_supports_feature(
            "observation.wrench.external_stiffness"
        )
        self._leader_pose_feature_enabled = self._dataset_supports_feature(
            "observation.leader_pose"
        )
        if self.record_leader_pose and not self._leader_pose_feature_enabled:
            self.get_logger().warn(
                "Dataset schema does not include 'observation.leader_pose' (an existing "
                "dataset created before this feature was added); disabling leader-pose "
                "recording for this session. Episodes recorded this way cannot feed the real "
                "K(t) extraction pipeline -- see extract_impedance_labels.py."
            )
            self.record_leader_pose = False
        self._joint_velocity_feature_enabled = self._dataset_supports_feature(
            "observation.velocity"
        )
        if self.record_joint_velocity and not self._joint_velocity_feature_enabled:
            self.get_logger().warn(
                "Dataset schema does not include 'observation.velocity' (an existing dataset "
                "created before this feature was added); disabling joint-velocity recording "
                "for this session."
            )
            self.record_joint_velocity = False
        if not self._language_instruction_feature_enabled:
            # Not a problem, just worth stating once per launch: the instruction still reaches
            # every frame as its 'task' label, which is what the downstream referent/manner
            # parsing reads back out anyway.
            self.get_logger().info(
                "Dataset schema has no separate 'language_instruction' column; recording the "
                "instruction as each frame's 'task' label only."
            )
        if self.record_wrench_forces and (
            not self._wrench_base_feature_enabled
            or not self._wrench_stiffness_feature_enabled
        ):
            self.get_logger().warn(
                "Dataset schema does not include wrench features; disabling wrench recording for this session."
            )
            self.record_wrench_forces = False
        self._normalize_legacy_dataset_stats()
        self.recording = False
        self.frame_count = 0
        self.episode_index = 1
        self.saved_episodes = 0
        self.discarded_episodes = 0
        self.last_action = "Idle"
        self.gui_enabled = self.visualize_gui
        self.latest_scene_rgb: Optional[np.ndarray] = None
        self.latest_wrist_rgb: Optional[np.ndarray] = None
        self.current_state: Optional[np.ndarray] = None
        self.current_ee_pose_6d: Optional[np.ndarray] = None
        self.current_leader_pose_6d: Optional[np.ndarray] = None
        self.current_gripper_state: Optional[float] = None
        self.current_external_wrench_base: Optional[np.ndarray] = None
        self.current_external_wrench_stiffness: Optional[np.ndarray] = None
        self._missing_action_source_warned = False
        self._missing_wrench_source_warned = False
        self._missing_leader_pose_source_warned = False
        self._missing_velocity_source_warned = False
        self._last_scene_frame_wall_time: Optional[float] = None
        self._scene_camera_stale_warned = False
        self.rx_counts: Dict[str, int] = {
            "joint": 0,
            "robot_state": 0,
            "wrist_rgb": 0,
            "wrist_depth": 0,
            "scene_ros": 0,
            "scene_cv2": 0,
        }

        joint_topic = self.get_parameter("joint_topic").get_parameter_value().string_value
        pose_topic = (
            self.get_parameter("pose_topic").get_parameter_value().string_value
        )
        gripper_joint_topic = (
            self.get_parameter("gripper_joint_topic").get_parameter_value().string_value
        )
        robot_state_topic = (
            self.get_parameter("robot_state_topic").get_parameter_value().string_value
        )
        leader_pose_topic = (
            self.get_parameter("leader_pose_topic").get_parameter_value().string_value
        )
        wrist_rgb_topic = (
            self.get_parameter("wrist_rgb_topic").get_parameter_value().string_value
        )
        wrist_depth_topic = (
            self.get_parameter("wrist_depth_topic").get_parameter_value().string_value
        )
        scene_rgb_topic = (
            self.get_parameter("scene_rgb_topic").get_parameter_value().string_value
        )
        queue_size = (
            self.get_parameter("sync_queue_size").get_parameter_value().integer_value
        )
        slop = self.get_parameter("sync_slop").get_parameter_value().double_value

        self.joint_sub = message_filters.Subscriber(
            self, JointState, joint_topic, qos_profile=qos_profile_sensor_data
        )
        self.pose_sub = self.create_subscription(
            PoseStamped,
            pose_topic,
            self._pose_callback,
            qos_profile_sensor_data,
        )
        self.gripper_joint_sub = self.create_subscription(
            JointState,
            gripper_joint_topic,
            self._gripper_joint_callback,
            10,
        )
        if self.record_leader_pose:
            self.leader_pose_sub = self.create_subscription(
                PoseStamped,
                leader_pose_topic,
                self._leader_pose_callback,
                qos_profile_sensor_data,
            )
        if self.record_wrench_forces:
            self.robot_state_sub = self.create_subscription(
                FrankaRobotState,
                robot_state_topic,
                self._franka_robot_state_callback,
                10,
            )
        if self.record_wrist_camera:
            self.wrist_rgb_sub = message_filters.Subscriber(self, Image, wrist_rgb_topic)
            if self.record_wrist_depth:
                self.wrist_depth_sub = message_filters.Subscriber(self, Image, wrist_depth_topic)

        if self.use_scene_cv2_capture and self.record_wrist_camera and self.record_wrist_depth:
            self._init_scene_cv2_capture()
            # In cv2 mode, synchronize only ROS topics and read the scene frame directly from capture.
            self.sync = message_filters.ApproximateTimeSynchronizer(
                [
                    self.joint_sub,
                    self.wrist_rgb_sub,
                    self.wrist_depth_sub,
                ],
                queue_size=queue_size,
                slop=slop,
            )
            self.sync.registerCallback(self._sync_callback_cv2)
        elif self.use_scene_cv2_capture and self.record_wrist_camera and not self.record_wrist_depth:
            self._init_scene_cv2_capture()
            self.sync = message_filters.ApproximateTimeSynchronizer(
                [
                    self.joint_sub,
                    self.wrist_rgb_sub,
                ],
                queue_size=queue_size,
                slop=slop,
            )
            self.sync.registerCallback(self._sync_callback_cv2_no_depth)
        elif self.use_scene_cv2_capture and not self.record_wrist_camera:
            self._init_scene_cv2_capture()
            # Only JointState is consumed from ROS in this mode.
            self.joint_only_sub = self.create_subscription(
                JointState,
                joint_topic,
                self._joint_callback_cv2_no_wrist,
                qos_profile_sensor_data,
            )
        elif not self.use_scene_cv2_capture and self.record_wrist_camera:
            self.scene_rgb_sub = message_filters.Subscriber(self, Image, scene_rgb_topic)

            # ApproximateTimeSynchronizer aligns messages arriving close in time across sensors.
            if self.record_wrist_depth:
                self.sync = message_filters.ApproximateTimeSynchronizer(
                    [
                        self.joint_sub,
                        self.wrist_rgb_sub,
                        self.wrist_depth_sub,
                        self.scene_rgb_sub,
                    ],
                    queue_size=queue_size,
                    slop=slop,
                )
                self.sync.registerCallback(self._sync_callback)
            else:
                self.sync = message_filters.ApproximateTimeSynchronizer(
                    [
                        self.joint_sub,
                        self.wrist_rgb_sub,
                        self.scene_rgb_sub,
                    ],
                    queue_size=queue_size,
                    slop=slop,
                )
                self.sync.registerCallback(self._sync_callback_no_depth)
        else:
            self.scene_rgb_sub = message_filters.Subscriber(self, Image, scene_rgb_topic)
            self.sync = message_filters.ApproximateTimeSynchronizer(
                [
                    self.joint_sub,
                    self.scene_rgb_sub,
                ],
                queue_size=queue_size,
                slop=slop,
            )
            self.sync.registerCallback(self._sync_callback_no_wrist)

        # Scene camera watchdog: if the Femto Bolt drops off USB (observed repeatedly
        # mid-session, in both the ROS driver and cv2.VideoCapture paths), the sync/capture
        # callback simply stops producing frames and nothing would otherwise notice. In ROS-topic
        # mode this is an independent subscription (separate from the message_filters.Subscriber
        # above) that only updates a timestamp, never touches the dataset. In cv2 mode there is no
        # separate topic to subscribe to, so _read_scene_frame_from_capture() itself updates the
        # same timestamp on every successful read -- either way _check_scene_camera_liveness()
        # below doesn't need to know which mode is active.
        if not self.use_scene_cv2_capture:
            self._scene_liveness_sub = self.create_subscription(
                Image, scene_rgb_topic, self._scene_liveness_callback, 10,
            )
        self._scene_watchdog_timer = self.create_timer(0.5, self._check_scene_camera_liveness)

        self.get_logger().info(
            f"Recorder initialized: repo_id={self.repo_id}, root={self.dataset_root}, "
            f"session_id={self.session_id}, operator_id={self.operator_id}, "
            f"colour={self.colour}, manner={self.manner}, "
            f"slop={slop:.3f}s, record_wrist_camera={self.record_wrist_camera}, "
            f"record_wrist_depth={self.record_wrist_depth}, "
            f"use_scene_cv2_capture={self.use_scene_cv2_capture}"
        )
        self.get_logger().info(
            "Recorder topics: "
            f"joint={joint_topic}, wrist_rgb={wrist_rgb_topic}, "
            f"wrist_depth={wrist_depth_topic}, scene_rgb={scene_rgb_topic}, "
            f"pose={pose_topic}, gripper_joint={gripper_joint_topic}, "
            f"robot_state={robot_state_topic}, "
            f"leader_pose={leader_pose_topic if self.record_leader_pose else '(disabled)'}"
        )
        self.get_logger().info(
            f"Fixed instruction for this session (every episode): '{self._instruction_text()}'"
            + (
                f" [template override: '{self.instruction_template}']"
                if self.language_instruction
                else ""
            )
        )
        self.get_logger().info(
            "Keyboard controls: r=start recording, s=save successful episode, "
            "d=discard current episode, n=next episode, q=quit"
        )

    def _resolve_fixed_choice(
        self,
        name: str,
        cli_value: Optional[str],
        param_value: str,
        valid: tuple,
    ) -> str:
        """Resolve one session-wide label from --<name> (preferred) or -p <name>:=, and validate.

        Raises rather than warning: an unrecognised colour/manner is silently baked into every
        frame of every episode of the whole collection, and the referent word is exactly what
        the policy is meant to ground on, so a typo here is not recoverable after the fact.
        """
        source = f"--{name}"
        value = (cli_value or "").strip().lower()
        if not value:
            source = f"-p {name}:="
            value = param_value.strip().lower()
        if not value:
            raise ValueError(
                f"{name} is required and was not given -- pass --{name} "
                f"{{{'|'.join(valid)}}} on the command line (or -p {name}:=<value> "
                "as a ROS parameter). It is fixed for the whole collection."
            )
        if value not in valid:
            raise ValueError(
                f"invalid {name} '{value}' (from {source}); expected one of "
                f"{', '.join(valid)}."
            )
        return value

    def _validate_instruction_template(self) -> None:
        """Fail at startup, not mid-episode, on a bad language_instruction override."""
        try:
            rendered = self.instruction_template.format(colour=self.colour, manner=self.manner)
        except (KeyError, IndexError, ValueError) as exc:
            raise ValueError(
                f"language_instruction='{self.instruction_template}' is not a valid template: "
                f"{exc}. It may contain the placeholders {{colour}} and {{manner}} only."
            ) from exc
        if self.colour not in rendered.lower():
            # Not fatal (an operator may deliberately want an unreferenced-referent control
            # condition), but it means the recorded text does not name the mark being wiped,
            # which is the one thing a referent-grounding policy trains on.
            self.get_logger().warn(
                f"Instruction '{rendered}' does not contain the colour word "
                f"'{self.colour}' -- nothing in the recorded language names which mark this "
                "episode is about. Include {colour} in language_instruction unless this is "
                "deliberate."
            )
        if self.manner not in rendered.lower():
            self.get_logger().warn(
                f"Instruction '{rendered}' does not contain the manner word "
                f"'{self.manner}' -- the adverb axis will not be visible to the policy. "
                "Include {manner} in language_instruction unless this is deliberate."
            )

    def _scene_liveness_callback(self, _msg: Image) -> None:
        self._last_scene_frame_wall_time = time.monotonic()
        self._scene_camera_stale_warned = False

    def _check_scene_camera_liveness(self) -> None:
        if self._last_scene_frame_wall_time is None:
            return
        age = time.monotonic() - self._last_scene_frame_wall_time
        if age > self.scene_camera_timeout_sec and not self._scene_camera_stale_warned:
            self.get_logger().warn(
                f"Scene camera (Femto Bolt) has not published for {age:.1f}s "
                f"(> {self.scene_camera_timeout_sec:.1f}s timeout) -- check USB connection; "
                "it is known to drop off the bus and re-enumerate mid-session."
            )
            self._scene_camera_stale_warned = True

    def _scene_camera_is_stale(self) -> bool:
        if self._last_scene_frame_wall_time is None:
            return False
        return (time.monotonic() - self._last_scene_frame_wall_time) > self.scene_camera_timeout_sec

    def _to_bgr(self, image_rgb: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    def _panel(self, title: str, image_bgr: Optional[np.ndarray], size: tuple[int, int]) -> np.ndarray:
        h, w = size
        if image_bgr is None:
            panel = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.putText(panel, f"{title}: waiting...", (18, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2, cv2.LINE_AA)
            return panel

        out = cv2.resize(image_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
        cv2.rectangle(out, (0, 0), (w - 1, 28), (0, 0, 0), -1)
        cv2.putText(out, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
        return out

    def _build_gui_frame(self) -> np.ndarray:
        scene_bgr = self._to_bgr(self.latest_scene_rgb) if self.latest_scene_rgb is not None else None
        wrist_bgr = self._to_bgr(self.latest_wrist_rgb) if self.latest_wrist_rgb is not None else None

        panel_h, panel_w = 360, 640
        scene_title = "Scene RGB (Femto Bolt)"
        if self._scene_camera_is_stale():
            scene_title = "Scene RGB (Femto Bolt) -- STALE/DISCONNECTED"
        top_left = self._panel(scene_title, scene_bgr, (panel_h, panel_w))
        if self._scene_camera_is_stale():
            cv2.rectangle(top_left, (0, 0), (panel_w - 1, panel_h - 1), (0, 0, 220), 4)
        top_right = self._panel("Wrist RGB", wrist_bgr, (panel_h, panel_w))

        status = np.full((panel_h, panel_w, 3), (22, 26, 34), dtype=np.uint8)
        mode = "RECORDING" if self.recording else "IDLE"
        mode_color = (70, 220, 100) if self.recording else (70, 170, 255)
        cv2.putText(status, "Recorder Control", (24, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (245, 245, 245), 2, cv2.LINE_AA)
        cv2.putText(status, f"Mode: {mode}", (24, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.72, mode_color, 2, cv2.LINE_AA)
        cv2.putText(status, f"Episode: {self.episode_index}", (24, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (230, 230, 230), 2, cv2.LINE_AA)
        cv2.putText(status, f"Frames: {self.frame_count}", (24, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (230, 230, 230), 2, cv2.LINE_AA)
        cv2.putText(status, f"Saved: {self.saved_episodes}", (24, 156), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (120, 230, 130), 2, cv2.LINE_AA)
        cv2.putText(status, f"Discarded: {self.discarded_episodes}", (24, 182), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (90, 170, 255), 2, cv2.LINE_AA)
        # Colour/manner are the one thing the operator must not get wrong for a whole session,
        # so draw them prominently, in the mark's own colour, together with the exact text every
        # frame is being labelled with.
        colour_bgr = (60, 60, 235) if self.colour == "red" else (235, 150, 60)
        cv2.putText(status, f"Operator: {self.operator_id}   Session: {self.session_id}", (24, 206), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190, 190, 190), 1, cv2.LINE_AA)
        cv2.putText(status, f"FIXED: {self.colour} / {self.manner}", (24, 234), cv2.FONT_HERSHEY_SIMPLEX, 0.72, colour_bgr, 2, cv2.LINE_AA)
        instruction = self._instruction_text()
        if len(instruction) > 52:  # keep it on one line of the 640px-wide panel
            instruction = instruction[:49] + "..."
        cv2.putText(status, f'"{instruction}"', (24, 258), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(status, f"Action: {self.last_action}", (24, 286), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 240, 120), 2, cv2.LINE_AA)

        # Two columns: the panel is only 360px tall and a single column of six would run off
        # the bottom of it.
        key_lines = [
            ("r: start episode", "n: next episode"),
            ("s: save episode", "m: move to start pose"),
            ("d: discard episode", "q: quit"),
        ]
        y = 304
        for left, right in key_lines:
            cv2.putText(status, left, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 1, cv2.LINE_AA)
            cv2.putText(status, right, (330, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 1, cv2.LINE_AA)
            y += 22

        return np.hstack((top_left, top_right, status))

    def _discard_current_episode_buffer(self) -> None:
        cleared = False
        for method_name in ("clear_episode_buffer", "reset_episode_buffer", "discard_episode"):
            method = getattr(self.dataset, method_name, None)
            if callable(method):
                try:
                    method()
                    cleared = True
                    break
                except TypeError:
                    continue

        if not cleared and hasattr(self.dataset, "episode_buffer"):
            episode_buffer = getattr(self.dataset, "episode_buffer")
            if isinstance(episode_buffer, dict):
                for value in episode_buffer.values():
                    if hasattr(value, "clear"):
                        value.clear()
                        cleared = True

        if not cleared:
            self.dataset = self._init_lerobot_dataset()

    def _save_recording_via_home(self) -> None:
        """Move the follower (and the leader, same as the 'm' key) to their configured
        start_joint_configuration, still recording throughout, then save the episode.

        Frames from the return-to-home motion are appended to the same episode via the normal
        _sync_callback* path -- that path is gated on self.recording (see e.g. _sync_callback
        above), not on which controller currently commands the follower, so it keeps recording
        automatically while _move_follower_to_home() switches the follower onto
        move_to_start_example_controller and back. The episode therefore always ends at the
        same known, repeatable pose instead of wherever the operator happened to release the
        leader, and is only written to disk (stop_recording) once that motion is done.

        A failed/timed-out home move does not discard the demo: the operator's recorded work is
        saved regardless, with whatever frames exist (including any partial home-motion
        frames) -- only the trailing "return to home" segment is missing, which is logged
        loudly so it doesn't go unnoticed.
        """
        self.last_action = f"Moving robot home before saving episode {self.episode_index}"
        if not self._move_follower_to_home():
            self.get_logger().error(
                f"Failed to move robot home before saving episode {self.episode_index}; "
                "saving anyway with whatever frames were already recorded."
            )

        saved = self.stop_recording(task="", save_episode=True)
        if saved:
            self.saved_episodes += 1
            self.last_action = f"Saved successful episode {self.episode_index}"
        else:
            self.last_action = "Nothing to save"

    def _next_episode(self) -> None:
        self.episode_index += 1
        self.frame_count = 0
        self.last_action = f"Ready for episode {self.episode_index}"

    def _handle_key(self, key_code: int) -> bool:
        if key_code < 0:
            return True

        key = chr(key_code & 0xFF).lower()
        if key == "r":
            if self.recording:
                self.last_action = "Already recording"
            else:
                # No metadata prompt any more: colour and manner are fixed for the whole
                # collection at launch, so "r" goes straight to recording.
                self.start_recording()
                self.last_action = f"Recording episode {self.episode_index}"
        elif key == "s":
            if not self.recording:
                self.last_action = "No active recording to save"
            else:
                self._save_recording_via_home()
                self._next_episode()
        elif key == "d":
            if not self.recording:
                self.last_action = "No active recording to discard"
            else:
                self.stop_recording(task="", save_episode=False)
                self.discarded_episodes += 1
                self.last_action = f"Discarded episode {self.episode_index}"
                self._next_episode()
        elif key == "n":
            if self.recording:
                self.last_action = "Finish current recording first (s/d)"
            else:
                self._next_episode()
        elif key == "m":
            if self.recording:
                self.last_action = "Stop/discard current recording before reset (s/d)"
            else:
                moved = self._move_robot_to_start_pose()
                self.last_action = (
                    "Robot moved to start joint configuration"
                    if moved
                    else "Failed to move robot to start configuration"
                )
        elif key == "q":
            self.last_action = "Quit requested"
            return False

        return True

    def update_gui(self) -> bool:
        if not self.gui_enabled:
            return True

        try:
            frame = self._build_gui_frame()
            cv2.imshow(self.gui_window_name, frame)
            key_code = cv2.waitKey(1)
            return self._handle_key(key_code)
        except cv2.error as exc:
            self.get_logger().error(f"OpenCV GUI unavailable; disabling visualization: {exc}")
            self.gui_enabled = False
            return True

    def shutdown_gui(self) -> None:
        if self.gui_enabled:
            cv2.destroyWindow(self.gui_window_name)

    def _init_scene_cv2_capture(self) -> None:
        # Fallback path only -- the Femto Bolt is driven through the proper orbbec_camera ROS
        # driver by default (use_scene_cv2_capture defaults to false). Unlike the previous ZED
        # setup this camera has no stereo pair, so there is no side-crop to do here.
        #
        # Explicit cv2.CAP_V4L2 backend: without it, OpenCV probes GStreamer first on this
        # platform (seen as noisy "pipeline have not been created" warnings) before falling back
        # to V4L2, which is both slower to open and occasionally hangs on isOpened() rather than
        # failing fast -- pinning the backend skips that probe entirely.
        #
        # Explicit YUYV FOURCC, deliberately *not* MJPG: this process also imports cv_bridge,
        # which links against the system/apt libopencv (4.5.4 here) rather than the pip
        # opencv-python this module's own `import cv2` resolves to (4.13.0 here). With both
        # OpenCV builds loaded at once, cv2.VideoCapture.read() on an MJPG-format V4L2 stream
        # goes through opencv-python's imgcodecs imdecode() for the JPEG-> BGR step and hits a
        # `!buf.empty()` assertion on every single read -- 100% reproducible, confirmed by
        # isolating cv_bridge as the trigger (rclpy alone does not cause it). YUYV bypasses that
        # decode path entirely (V4L2 backend converts it via cvtColor instead), and the Femto
        # Bolt's color node here (verified with `v4l2-ctl --list-formats-ext`) offers YUYV at
        # the same 1280x720@30fps as MJPG, so there's no capability loss -- just don't switch
        # this back to MJPG without re-testing against cv_bridge in the same process.
        device = self.scene_cv2_device_path if self.scene_cv2_device_path else self.scene_cv2_device
        self.scene_capture = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if self.scene_capture is not None:
            self.scene_capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
            self.scene_capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.scene_cv2_width)
            self.scene_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.scene_cv2_height)
            self.scene_capture.set(cv2.CAP_PROP_FPS, float(self.dataset_fps))

        if self.scene_capture is None or not self.scene_capture.isOpened():
            raise RuntimeError(
                f"use_scene_cv2_capture is enabled but cv2.VideoCapture could not open device "
                f"{device!r}. If this is a permission error, see "
                "fr3_bilateral_teleop/scripts/fix_femto_bolt_usb_perms.sh. If it opens other "
                "devices but not this one, scene_cv2_device(_path) may be pointing at the wrong "
                "/dev/video* node -- the Femto Bolt exposes several; see the parameter's "
                "declaration comment for how to enumerate them by name."
            )

        # Verify now, not on the first frame mid-recording: an open() that "succeeds" but can't
        # actually deliver the requested format returns ok=False or a None/empty frame on read()
        # -- or, observed directly against this rig's non-color Femto Bolt nodes, raises a bare
        # cv2.error out of OpenCV's own MJPG decoder (imdecode on an empty buffer) instead of
        # failing cleanly. Catch that here so a bad node index is a clear RuntimeError with a
        # pointer to the fix, not an unhandled crash out of node __init__.
        try:
            ok, frame = self.scene_capture.read()
        except cv2.error:
            ok, frame = False, None
        if not ok or frame is None or frame.size == 0:
            self.scene_capture.release()
            self.scene_capture = None
            raise RuntimeError(
                f"use_scene_cv2_capture is enabled and device {device!r} opened, but reading a "
                "verification frame failed. Likely wrong /dev/video* node (picked one of the "
                "Femto Bolt's non-color or metadata nodes) or an unsupported format negotiation -- "
                "try a different scene_cv2_device_path."
            )

        self.get_logger().info(
            f"Using cv2.VideoCapture for scene camera input (device={device!r}, "
            f"size={self.scene_cv2_width}x{self.scene_cv2_height}, verified with a test frame "
            f"of shape {frame.shape})"
        )

    def _service_path(self, namespace: str, suffix: str) -> str:
        ns = namespace.rstrip("/")
        if ns:
            return f"{ns}/{suffix}"
        return f"/{suffix}"

    def _switch_controllers(
        self,
        namespace: str,
        activate_controllers: list[str],
        deactivate_controllers: list[str],
    ) -> bool:
        if namespace not in self._switch_controller_clients:
            self._switch_controller_clients[namespace] = self.create_client(
                SwitchController,
                self._service_path(namespace, "controller_manager/switch_controller"),
            )

        switch_controller_client = self._switch_controller_clients[namespace]
        if not switch_controller_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error(
                "Controller switch service is not available: "
                f"{self._service_path(namespace, 'controller_manager/switch_controller')}"
            )
            return False

        request = SwitchController.Request()
        request.activate_controllers = activate_controllers
        request.deactivate_controllers = deactivate_controllers
        request.strictness = 2  # STRICT
        request.activate_asap = True
        request.timeout.sec = 1
        request.timeout.nanosec = 0

        future = switch_controller_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        if not future.done() or future.result() is None:
            self.get_logger().error("Timed out waiting for controller switch response")
            return False

        return bool(future.result().ok)

    def _has_reached_start_pose(self, namespace: str) -> Optional[bool]:
        if namespace not in self._move_to_start_get_parameters_clients:
            self._move_to_start_get_parameters_clients[namespace] = self.create_client(
                GetParameters,
                self._service_path(
                    namespace,
                    f"{self.reset_move_to_start_controller}/get_parameters",
                ),
            )

        move_to_start_parameters_client = self._move_to_start_get_parameters_clients[namespace]
        if not move_to_start_parameters_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error(
                "Move-to-start parameter service is not available: "
                f"{self._service_path(namespace, f'{self.reset_move_to_start_controller}/get_parameters')}"
            )
            return None

        request = GetParameters.Request()
        request.names = ["process_finished"]

        future = move_to_start_parameters_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
        if not future.done() or future.result() is None:
            return None

        values = future.result().values
        if not values:
            return None

        return bool(values[0].bool_value)

    def _set_move_to_start_process_finished(self, namespace: str, finished: bool) -> bool:
        if namespace not in self._move_to_start_set_parameters_clients:
            self._move_to_start_set_parameters_clients[namespace] = self.create_client(
                SetParameters,
                self._service_path(
                    namespace,
                    f"{self.reset_move_to_start_controller}/set_parameters",
                ),
            )

        move_to_start_set_parameters_client = self._move_to_start_set_parameters_clients[namespace]
        if not move_to_start_set_parameters_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error(
                "Move-to-start set_parameters service is not available: "
                f"{self._service_path(namespace, f'{self.reset_move_to_start_controller}/set_parameters')}"
            )
            return False

        request = SetParameters.Request()
        request.parameters = [
            Parameter(
                name="process_finished",
                value=ParameterValue(
                    type=ParameterType.PARAMETER_BOOL,
                    bool_value=finished,
                ),
            )
        ]

        future = move_to_start_set_parameters_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
        if not future.done() or future.result() is None:
            return False

        results = future.result().results
        return bool(results) and all(result.successful for result in results)

    def _move_arms_to_start_pose(self, arms_to_reset: list[tuple[str, str]]) -> bool:
        """Drive each (namespace, active_controller) pair through move_to_start_example_controller
        and back to its configured start_joint_configuration.

        Shared mechanics for two callers: the manual 'm' reset (_move_robot_to_start_pose,
        below -- all configured arms) and the automatic home-before-save move triggered by 's'
        (_move_follower_to_home, below -- follower only). Blocking: spins this node in a loop
        until every arm reports process_finished or reset_wait_timeout_sec elapses, same as
        before this was split out.
        """
        self.get_logger().info(
            "Starting reset to start_joint_configuration for "
            f"{', '.join(namespace for namespace, _ in arms_to_reset) or '(no arms)'}"
        )

        # Deduplicate in case multiple entries resolve to the same namespace/controller.
        unique_arms: list[tuple[str, str]] = []
        seen_arms = set()
        for namespace, target_controller in arms_to_reset:
            arm_key = (namespace, target_controller)
            if arm_key in seen_arms:
                continue
            seen_arms.add(arm_key)
            unique_arms.append(arm_key)

        activated_arms: list[tuple[str, str]] = []
        for namespace, target_controller in unique_arms:
            if not self._switch_controllers(
                namespace,
                [self.reset_move_to_start_controller],
                [target_controller],
            ):
                self.get_logger().error(
                    f"Failed to activate move-to-start controller for namespace '{namespace}'"
                )
                for rollback_namespace, rollback_target_controller in activated_arms:
                    self._switch_controllers(
                        rollback_namespace,
                        [rollback_target_controller],
                        [self.reset_move_to_start_controller],
                    )
                return False
            activated_arms.append((namespace, target_controller))

        for namespace, _ in activated_arms:
            if not self._set_move_to_start_process_finished(namespace, False):
                self.get_logger().warn(
                    "Could not reset process_finished=false before waiting; "
                    f"completion detection may be unreliable for namespace '{namespace}'"
                )

        pending_namespaces = {namespace for namespace, _ in activated_arms}
        deadline = self.get_clock().now().nanoseconds + int(self.reset_wait_timeout_sec * 1e9)
        while rclpy.ok() and pending_namespaces and self.get_clock().now().nanoseconds < deadline:
            reached_namespaces = set()
            for namespace in pending_namespaces:
                reached = self._has_reached_start_pose(namespace)
                if reached is True:
                    reached_namespaces.add(namespace)
            pending_namespaces -= reached_namespaces
            if not pending_namespaces:
                break
            rclpy.spin_once(self, timeout_sec=0.1)

        restore_ok = True
        for namespace, target_controller in activated_arms:
            if not self._switch_controllers(
                namespace,
                [target_controller],
                [self.reset_move_to_start_controller],
            ):
                self.get_logger().error(
                    f"Failed to reactivate target teleop controller for namespace '{namespace}'"
                )
                restore_ok = False

        if not restore_ok:
            return False

        if pending_namespaces:
            pending_text = ", ".join(sorted(pending_namespaces))
            self.get_logger().error(
                "Timed out waiting for move_to_start controller to finish. "
                f"timeout={self.reset_wait_timeout_sec:.1f}s, pending={pending_text}"
            )
            return False

        self.get_logger().info(
            f"Robot(s) reached configured start_joint_configuration for {len(activated_arms)} namespace(s)"
        )
        return True

    def _move_robot_to_start_pose(self) -> bool:
        """Manual full reset (the 'm' key): follower, plus leader if reset_leader_enabled."""
        arms_to_reset: list[tuple[str, str]] = [
            (self.reset_controller_namespace, self.reset_target_controller)
        ]
        if self.reset_leader_enabled:
            arms_to_reset.append(
                (self.reset_leader_namespace, self.reset_leader_target_controller)
            )
        return self._move_arms_to_start_pose(arms_to_reset)

    def _move_follower_to_home(self) -> bool:
        """Move-to-home run automatically before saving (the 's' key).

        Same reset as the manual 'm' key (_move_robot_to_start_pose): follower, plus leader
        if reset_leader_enabled. This used to move the follower only, leaving the leader
        wherever the operator released it -- when follower_controller reactivates afterwards,
        it reads the leader's (now far away) live position as its impedance target, and the
        large position error snaps the follower across that gap violently. Resetting the
        leader to the same start pose closes that gap so there is nothing to spring back to.
        """
        return self._move_robot_to_start_pose()

    def _read_scene_frame_from_capture(self) -> np.ndarray:
        if self.scene_capture is None:
            raise RuntimeError("Scene camera cv2 capture is not initialized")

        ok, frame_bgr = self.scene_capture.read()
        if not ok or frame_bgr is None:
            raise RuntimeError("Failed to read frame from cv2.VideoCapture")
        self._last_scene_frame_wall_time = time.monotonic()
        self._scene_camera_stale_warned = False

        expected_h, expected_w, _ = self.features["observation.images.scene_rgb"]["shape"]
        scene_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # Femto Bolt is mounted upside down.
        scene_rgb = cv2.rotate(scene_rgb, cv2.ROTATE_180)

        if scene_rgb.shape[0] != expected_h or scene_rgb.shape[1] != expected_w:
            scene_rgb = cv2.resize(scene_rgb, (expected_w, expected_h), interpolation=cv2.INTER_LINEAR)

        return scene_rgb

    def _prepare_state(self, joint_msg: JointState) -> Optional[np.ndarray]:
        if len(joint_msg.position) < 7:
            self.get_logger().warn("JointState has fewer than 7 positions; skipping frame.")
            return None
        return np.asarray(joint_msg.position[:7], dtype=np.float32)

    def _prepare_velocity(self, joint_msg: JointState) -> Optional[np.ndarray]:
        # Synchronous, unlike _prepare_wrench_observation/_prepare_leader_pose_observation --
        # velocity lives on the same already-synchronized joint_msg as _prepare_state's
        # position, not a separately-cached async subscription, so there is nothing to wait on
        # across calls, only a per-message validity check.
        if not self.record_joint_velocity:
            return None
        if len(joint_msg.velocity) < 7:
            if not self._missing_velocity_source_warned:
                self.get_logger().warn(
                    "JointState has fewer than 7 velocities (field unpopulated?); skipping "
                    "frames until it is -- is joint_topic's publisher filling in .velocity?"
                )
                self._missing_velocity_source_warned = True
            return None
        return np.asarray(joint_msg.velocity[:7], dtype=np.float32)

    def _quat_to_euler_xyz(self, x: float, y: float, z: float, w: float) -> np.ndarray:
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (w * y - z * x)
        if abs(sinp) >= 1.0:
            pitch = math.copysign(math.pi / 2.0, sinp)
        else:
            pitch = math.asin(sinp)

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return np.asarray([roll, pitch, yaw], dtype=np.float32)

    def _pose_callback(self, msg: PoseStamped) -> None:
        pos = msg.pose.position
        ori = msg.pose.orientation
        rpy = self._quat_to_euler_xyz(ori.x, ori.y, ori.z, ori.w)
        self.current_ee_pose_6d = np.asarray(
            [pos.x, pos.y, pos.z, rpy[0], rpy[1], rpy[2]],
            dtype=np.float32,
        )

    def _leader_pose_callback(self, msg: PoseStamped) -> None:
        # x_l(t) -- same (xyz, rpy) convention as _pose_callback's follower pose, deliberately,
        # so the two are directly comparable without a re-basing step in the extraction pipeline.
        pos = msg.pose.position
        ori = msg.pose.orientation
        rpy = self._quat_to_euler_xyz(ori.x, ori.y, ori.z, ori.w)
        self.current_leader_pose_6d = np.asarray(
            [pos.x, pos.y, pos.z, rpy[0], rpy[1], rpy[2]],
            dtype=np.float32,
        )

    def _gripper_joint_callback(self, msg: JointState) -> None:
        if not msg.position:
            return

        width: Optional[float] = None
        if msg.name and len(msg.name) == len(msg.position):
            finger_positions = [
                float(pos)
                for name, pos in zip(msg.name, msg.position)
                if "finger" in name or "gripper" in name
            ]
            if len(finger_positions) >= 2:
                width = float(finger_positions[0] + finger_positions[1])
            elif len(finger_positions) == 1:
                width = float(finger_positions[0])

        if width is None:
            if len(msg.position) >= 2:
                width = float(msg.position[0] + msg.position[1])
            else:
                width = float(msg.position[0])

        self.current_gripper_state = width

    def _prepare_action(self) -> Optional[np.ndarray]:
        if self.current_ee_pose_6d is None:
            if not self._missing_action_source_warned:
                self.get_logger().warn(
                    "Waiting for pose messages before recording actions."
                )
                self._missing_action_source_warned = True
            return None

        if self.current_gripper_state is None:
            if not self._missing_action_source_warned:
                self.get_logger().warn(
                    "No gripper_joint messages received yet; defaulting action gripper state to 0.0."
                )
                self._missing_action_source_warned = True
            gripper_state = np.float32(0.0)
        else:
            gripper_state = np.float32(self.current_gripper_state)

        action = np.empty((7,), dtype=np.float32)
        action[:6] = self.current_ee_pose_6d
        action[6] = gripper_state
        return action

    def _franka_robot_state_callback(self, msg: FrankaRobotState) -> None:
        self.rx_counts["robot_state"] += 1
        self.current_external_wrench_base = np.asarray(
            [
                msg.o_f_ext_hat_k.wrench.force.x,
                msg.o_f_ext_hat_k.wrench.force.y,
                msg.o_f_ext_hat_k.wrench.force.z,
                msg.o_f_ext_hat_k.wrench.torque.x,
                msg.o_f_ext_hat_k.wrench.torque.y,
                msg.o_f_ext_hat_k.wrench.torque.z,
            ],
            dtype=np.float32,
        )
        self.current_external_wrench_stiffness = np.asarray(
            [
                msg.k_f_ext_hat_k.wrench.force.x,
                msg.k_f_ext_hat_k.wrench.force.y,
                msg.k_f_ext_hat_k.wrench.force.z,
                msg.k_f_ext_hat_k.wrench.torque.x,
                msg.k_f_ext_hat_k.wrench.torque.y,
                msg.k_f_ext_hat_k.wrench.torque.z,
            ],
            dtype=np.float32,
        )

    def _prepare_wrench_observation(
        self,
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        if not self.record_wrench_forces:
            return None

        if (
            self.current_external_wrench_base is None
            or self.current_external_wrench_stiffness is None
        ):
            if not self._missing_wrench_source_warned:
                self.get_logger().warn(
                    "Waiting for FrankaRobotState wrench messages before recording frames."
                )
                self._missing_wrench_source_warned = True
            return None

        return (
            self.current_external_wrench_base,
            self.current_external_wrench_stiffness,
        )

    def _prepare_leader_pose_observation(self) -> Optional[np.ndarray]:
        if not self.record_leader_pose:
            return None

        if self.current_leader_pose_6d is None:
            if not self._missing_leader_pose_source_warned:
                self.get_logger().warn(
                    "Waiting for leader pose messages before recording frames -- is the leader "
                    "arm's franka_robot_state_broadcaster running under the configured "
                    "leader_pose_topic namespace?"
                )
                self._missing_leader_pose_source_warned = True
            return None

        return self.current_leader_pose_6d

    def _match_feature_shape(
        self,
        image: np.ndarray,
        feature_name: str,
        interpolation: int,
    ) -> np.ndarray:
        expected_h, expected_w, expected_c = self.features[feature_name]["shape"]

        if image.ndim == 2:
            image = image[..., np.newaxis]

        if image.shape[:2] != (expected_h, expected_w):
            image = cv2.resize(image, (expected_w, expected_h), interpolation=interpolation)
            if image.ndim == 2:
                image = image[..., np.newaxis]

        if image.ndim != 3 or image.shape[2] != expected_c:
            raise ValueError(
                f"{feature_name} resolved to shape {image.shape}, expected "
                f"({expected_h}, {expected_w}, {expected_c})"
            )

        return np.ascontiguousarray(image)

    def _prepare_wrist_images(self, wrist_rgb_msg: Image, wrist_depth_msg: Image) -> tuple[np.ndarray, np.ndarray]:
        wrist_rgb = self.bridge.imgmsg_to_cv2(wrist_rgb_msg, desired_encoding="rgb8")
        wrist_rgb = self._match_feature_shape(
            wrist_rgb,
            "observation.images.wrist_rgb",
            cv2.INTER_LINEAR,
        )

        # Preserve depth metric values: keep raw 16-bit data and store uncompressed.
        wrist_depth = self.bridge.imgmsg_to_cv2(
            wrist_depth_msg,
            desired_encoding="passthrough",
        )
        if wrist_depth.dtype != np.uint16:
            wrist_depth = wrist_depth.astype(np.uint16)
        wrist_depth = self._match_feature_shape(
            wrist_depth,
            "observation.images.wrist_depth",
            cv2.INTER_NEAREST,
        )

        return wrist_rgb, wrist_depth

    def _instruction_text(self) -> str:
        # Fixed for the whole session: same colour, same manner, same wording on every frame of
        # every episode. Validated at startup by _validate_instruction_template(), so the
        # .format() here cannot fail mid-recording.
        return self.instruction_template.format(
            colour=self.colour, manner=self.manner
        ).strip()

    def _build_frame(
        self,
        state: np.ndarray,
        action: np.ndarray,
        scene_rgb: np.ndarray,
        wrist_rgb: Optional[np.ndarray] = None,
        wrist_depth: Optional[np.ndarray] = None,
        external_wrench_base: Optional[np.ndarray] = None,
        external_wrench_stiffness: Optional[np.ndarray] = None,
        leader_pose: Optional[np.ndarray] = None,
        velocity: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        instruction = self._instruction_text()
        frame: Dict[str, Any] = {
            "task": instruction,
            "observation.state": torch.from_numpy(state.copy()),
            "action": torch.from_numpy(action.copy()),
            "observation.images.scene_rgb": torch.from_numpy(np.ascontiguousarray(scene_rgb)),
        }
        if self._language_instruction_feature_enabled:
            frame["language_instruction"] = instruction
        if wrist_rgb is not None:
            frame["observation.images.wrist_rgb"] = torch.from_numpy(
                np.ascontiguousarray(wrist_rgb)
            )
        if wrist_depth is not None:
            frame["observation.images.wrist_depth"] = torch.from_numpy(
                np.ascontiguousarray(wrist_depth)
            )
        if external_wrench_base is not None and self._wrench_base_feature_enabled:
            frame["observation.wrench.external_base"] = torch.from_numpy(
                external_wrench_base.astype(np.float32, copy=True)
            )
        if (
            external_wrench_stiffness is not None
            and self._wrench_stiffness_feature_enabled
        ):
            frame["observation.wrench.external_stiffness"] = torch.from_numpy(
                external_wrench_stiffness.astype(np.float32, copy=True)
            )
        if leader_pose is not None and self._leader_pose_feature_enabled:
            frame["observation.leader_pose"] = torch.from_numpy(
                leader_pose.astype(np.float32, copy=True)
            )
        if velocity is not None and self._joint_velocity_feature_enabled:
            frame["observation.velocity"] = torch.from_numpy(
                velocity.astype(np.float32, copy=True)
            )
        return frame

    def _dataset_supports_feature(self, feature_name: str) -> bool:
        dataset_features = getattr(self.dataset, "features", None)
        if isinstance(dataset_features, dict):
            return feature_name in dataset_features

        meta = getattr(self.dataset, "meta", None)
        meta_features = getattr(meta, "features", None)
        if isinstance(meta_features, dict):
            return feature_name in meta_features

        return False

    def _init_lerobot_dataset(self) -> Any:
        # Prefer writable dataset creation API for recording.
        create_fn = getattr(LeRobotDataset, "create", None)
        if callable(create_fn):
            try:
                dataset = create_fn(
                    repo_id=self.repo_id,
                    fps=int(self.dataset_fps),
                    features=self.features,
                    root=self.dataset_root,
                    robot_type=self.robot_type,
                    use_videos=bool(self.use_videos),
                )
                self.get_logger().info(
                    "Created LeRobot dataset with LeRobotDataset.create(repo_id, fps, features, root, ...)"
                )
                return dataset
            except FileExistsError:
                self.get_logger().warn(
                    "LeRobotDataset.create() refused existing dataset_root; "
                    "falling back to opening the dataset in-place."
                )
            except TypeError:
                # Fallback for very old versions where create() does not exist or differs.
                pass

        # Backward-compatibility fallback: this constructor style can open existing datasets.
        init_attempts = [
            {"repo_id": self.repo_id, "root": self.dataset_root},
            {"repo_id": self.repo_id, "dataset_root": self.dataset_root},
            {"repo_id": self.repo_id, "local_dir": self.dataset_root},
        ]
        for kwargs in init_attempts:
            try:
                return LeRobotDataset(**kwargs)
            except TypeError:
                continue

        signature = inspect.signature(LeRobotDataset)
        raise RuntimeError(
            "Unable to initialize LeRobotDataset with known argument patterns. "
            f"Constructor signature: {signature}. "
            "Update _init_lerobot_dataset() to match your installed lerobot version."
        )

    def _normalize_legacy_dataset_stats(self) -> int:
        meta = getattr(self.dataset, "meta", None)
        stats = getattr(meta, "stats", None)
        if not isinstance(stats, dict):
            return 0

        reshaped = 0
        for feature_key, feature_stats in stats.items():
            if "image" not in feature_key or not isinstance(feature_stats, dict):
                continue
            for stat_key, stat_value in feature_stats.items():
                if stat_key == "count":
                    continue
                value_arr = np.asarray(stat_value)
                if value_arr.size == 3 and value_arr.shape != (3, 1, 1):
                    feature_stats[stat_key] = value_arr.reshape(3, 1, 1)
                    reshaped += 1

        if reshaped > 0:
            self.get_logger().warn(
                "Detected legacy image stats shape (3,) in dataset metadata; "
                f"reshaped {reshaped} stats entries to (3,1,1) for compatibility."
            )
        return reshaped

    def start_recording(self) -> None:
        self._discard_current_episode_buffer()
        self.recording = True
        self.frame_count = 0
        self._missing_action_source_warned = False
        self._missing_wrench_source_warned = False
        for key in self.rx_counts:
            self.rx_counts[key] = 0
        self.get_logger().info("Recording started")

    def stop_recording(self, task: str = "", save_episode: bool = True) -> bool:
        self.recording = False

        if not save_episode:
            self._discard_current_episode_buffer()
            self.get_logger().info(
                f"Discarded episode with {self.frame_count} synchronized frames"
            )
            return False

        episode_task = task if task else self._instruction_text()
        if self.frame_count == 0:
            mode = "cv2" if self.use_scene_cv2_capture else "ros"
            self.get_logger().warn(
                "No synchronized frames were recorded; episode will not be saved. "
                f"counts(joint={self.rx_counts['joint']}, robot_state={self.rx_counts['robot_state']}, "
                f"wrist_rgb={self.rx_counts['wrist_rgb']}, "
                f"wrist_depth={self.rx_counts['wrist_depth']}, scene_ros={self.rx_counts['scene_ros']}, "
                f"scene_cv2={self.rx_counts['scene_cv2']}), "
                f"record_wrist_camera={self.record_wrist_camera}, scene_source={mode}."
            )
            return False

        save_sig = inspect.signature(self.dataset.save_episode)
        # Normalize possibly legacy stats right before save, as metadata can be loaded lazily.
        reshaped = self._normalize_legacy_dataset_stats()

        # Some local datasets contain legacy stats that are incompatible with newer
        # lerobot aggregation checks. Disable aggregation to keep recording functional.
        if reshaped > 0:
            meta = getattr(self.dataset, "meta", None)
            if meta is not None:
                setattr(meta, "stats", None)
            if not self._stats_aggregation_disabled_logged:
                self.get_logger().warn(
                    "Disabled dataset stats aggregation for this session due to legacy stats format."
                )
                self._stats_aggregation_disabled_logged = True

        try:
            if "task" in save_sig.parameters:
                self.dataset.save_episode(task=episode_task)
            elif "episode_data" in save_sig.parameters:
                # Newer lerobot versions expect save_episode() to consume self.episode_buffer.
                # If a task override is provided at stop time, update buffered task labels first.
                if episode_task and hasattr(self.dataset, "episode_buffer"):
                    episode_buffer = getattr(self.dataset, "episode_buffer")
                    if episode_buffer is not None and "task" in episode_buffer:
                        task_count = len(episode_buffer["task"])
                        episode_buffer["task"] = [episode_task] * task_count
                self.dataset.save_episode()
            else:
                self.dataset.save_episode()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Failed to save episode: {exc}")
            return False
        self.get_logger().info(
            f"Saved episode with {self.frame_count} synchronized frames (task='{episode_task}')"
        )
        self._append_session_manifest_entry(episode_task)
        return True

    def _append_session_manifest_entry(self, episode_task: str) -> None:
        # Sidecar JSONL, deliberately outside LeRobotDataset's own metadata -- scaffolding for
        # the session-level split in the dataset builder. This doesn't build the splitter
        # itself, it just makes sure every saved episode records which session/operator/manner
        # it belongs to now, while that's still known, so the splitter can group by session_id without reconstructing it from file timestamps later.
        entry = {
            "timestamp": time.time(),
            "session_id": self.session_id,
            "operator_id": self.operator_id,
            # colour/manner are session-wide constants, so every line of a session's manifest
            # carries the same pair -- which is the point: the referent is recoverable per
            # episode without having to re-parse it back out of the task string.
            "colour": self.colour,
            "manner": self.manner,
            "task_family": self.episode_task,
            "episode_index": self.episode_index,
            "task": episode_task,
            "frame_count": self.frame_count,
            "repo_id": self.repo_id,
        }
        try:
            self.session_manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with self.session_manifest_path.open("a") as manifest_file:
                manifest_file.write(json.dumps(entry) + "\n")
        except OSError as exc:
            self.get_logger().error(f"Failed to append session manifest entry: {exc}")

    def _sync_callback(
        self,
        joint_msg: JointState,
        wrist_rgb_msg: Image,
        wrist_depth_msg: Image,
        scene_rgb_msg: Image,
    ) -> None:
        self.rx_counts["joint"] += 1
        self.rx_counts["wrist_rgb"] += 1
        self.rx_counts["wrist_depth"] += 1
        self.rx_counts["scene_ros"] += 1

        state = self._prepare_state(joint_msg)
        if state is None:
            return

        try:
            wrist_rgb, wrist_depth = self._prepare_wrist_images(wrist_rgb_msg, wrist_depth_msg)
            scene_rgb = self.bridge.imgmsg_to_cv2(scene_rgb_msg, desired_encoding="rgb8")
            # Femto Bolt is mounted upside down.
            scene_rgb = cv2.rotate(scene_rgb, cv2.ROTATE_180)
            scene_rgb = self._match_feature_shape(
                scene_rgb,
                "observation.images.scene_rgb",
                cv2.INTER_LINEAR,
            )
            self.latest_scene_rgb = scene_rgb
            self.latest_wrist_rgb = wrist_rgb
            self.current_state = state

            if not self.recording:
                return

            action = self._prepare_action()
            if action is None:
                return

            wrench_data = self._prepare_wrench_observation()
            if self.record_wrench_forces and wrench_data is None:
                return

            wrench_base, wrench_stiffness = (
                wrench_data if wrench_data is not None else (None, None)
            )

            leader_pose_data = self._prepare_leader_pose_observation()
            if self.record_leader_pose and leader_pose_data is None:
                return

            velocity_data = self._prepare_velocity(joint_msg)
            if self.record_joint_velocity and velocity_data is None:
                return

            frame = self._build_frame(
                state,
                action,
                scene_rgb,
                wrist_rgb,
                wrist_depth,
                wrench_base,
                wrench_stiffness,
                leader_pose_data,
                velocity_data,
            )

            self.dataset.add_frame(frame)
            self.frame_count += 1
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Failed to convert or store synchronized frame: {exc}")

    def _sync_callback_cv2(
        self,
        joint_msg: JointState,
        wrist_rgb_msg: Image,
        wrist_depth_msg: Image,
    ) -> None:
        self.rx_counts["joint"] += 1
        self.rx_counts["wrist_rgb"] += 1
        self.rx_counts["wrist_depth"] += 1

        state = self._prepare_state(joint_msg)
        if state is None:
            return

        try:
            wrist_rgb, wrist_depth = self._prepare_wrist_images(wrist_rgb_msg, wrist_depth_msg)
            scene_rgb = self._read_scene_frame_from_capture()
            self.rx_counts["scene_cv2"] += 1
            self.latest_scene_rgb = scene_rgb
            self.latest_wrist_rgb = wrist_rgb
            self.current_state = state

            if not self.recording:
                return

            action = self._prepare_action()
            if action is None:
                return

            wrench_data = self._prepare_wrench_observation()
            if self.record_wrench_forces and wrench_data is None:
                return

            wrench_base, wrench_stiffness = (
                wrench_data if wrench_data is not None else (None, None)
            )

            leader_pose_data = self._prepare_leader_pose_observation()
            if self.record_leader_pose and leader_pose_data is None:
                return

            velocity_data = self._prepare_velocity(joint_msg)
            if self.record_joint_velocity and velocity_data is None:
                return

            frame = self._build_frame(
                state,
                action,
                scene_rgb,
                wrist_rgb,
                wrist_depth,
                wrench_base,
                wrench_stiffness,
                leader_pose_data,
                velocity_data,
            )

            self.dataset.add_frame(frame)
            self.frame_count += 1
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Failed to convert or store synchronized frame: {exc}")

    def _sync_callback_cv2_no_depth(
        self,
        joint_msg: JointState,
        wrist_rgb_msg: Image,
    ) -> None:
        self.rx_counts["joint"] += 1
        self.rx_counts["wrist_rgb"] += 1

        state = self._prepare_state(joint_msg)
        if state is None:
            return

        try:
            wrist_rgb = self.bridge.imgmsg_to_cv2(wrist_rgb_msg, desired_encoding="rgb8")
            wrist_rgb = self._match_feature_shape(
                wrist_rgb,
                "observation.images.wrist_rgb",
                cv2.INTER_LINEAR,
            )
            scene_rgb = self._read_scene_frame_from_capture()
            self.rx_counts["scene_cv2"] += 1
            self.latest_scene_rgb = scene_rgb
            self.latest_wrist_rgb = wrist_rgb
            self.current_state = state

            if not self.recording:
                return

            action = self._prepare_action()
            if action is None:
                return

            wrench_data = self._prepare_wrench_observation()
            if self.record_wrench_forces and wrench_data is None:
                return

            wrench_base, wrench_stiffness = (
                wrench_data if wrench_data is not None else (None, None)
            )

            leader_pose_data = self._prepare_leader_pose_observation()
            if self.record_leader_pose and leader_pose_data is None:
                return

            velocity_data = self._prepare_velocity(joint_msg)
            if self.record_joint_velocity and velocity_data is None:
                return

            frame = self._build_frame(
                state,
                action,
                scene_rgb,
                wrist_rgb,
                None,
                wrench_base,
                wrench_stiffness,
                leader_pose_data,
                velocity_data,
            )

            self.dataset.add_frame(frame)
            self.frame_count += 1
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Failed to convert or store synchronized frame: {exc}")

    def _sync_callback_no_depth(
        self,
        joint_msg: JointState,
        wrist_rgb_msg: Image,
        scene_rgb_msg: Image,
    ) -> None:
        self.rx_counts["joint"] += 1
        self.rx_counts["wrist_rgb"] += 1
        self.rx_counts["scene_ros"] += 1

        state = self._prepare_state(joint_msg)
        if state is None:
            return

        try:
            wrist_rgb = self.bridge.imgmsg_to_cv2(wrist_rgb_msg, desired_encoding="rgb8")
            wrist_rgb = self._match_feature_shape(
                wrist_rgb,
                "observation.images.wrist_rgb",
                cv2.INTER_LINEAR,
            )
            scene_rgb = self.bridge.imgmsg_to_cv2(scene_rgb_msg, desired_encoding="rgb8")
            # Femto Bolt is mounted upside down.
            scene_rgb = cv2.rotate(scene_rgb, cv2.ROTATE_180)
            scene_rgb = self._match_feature_shape(
                scene_rgb,
                "observation.images.scene_rgb",
                cv2.INTER_LINEAR,
            )
            self.latest_scene_rgb = scene_rgb
            self.latest_wrist_rgb = wrist_rgb
            self.current_state = state

            if not self.recording:
                return

            action = self._prepare_action()
            if action is None:
                return

            wrench_data = self._prepare_wrench_observation()
            if self.record_wrench_forces and wrench_data is None:
                return

            wrench_base, wrench_stiffness = (
                wrench_data if wrench_data is not None else (None, None)
            )

            leader_pose_data = self._prepare_leader_pose_observation()
            if self.record_leader_pose and leader_pose_data is None:
                return

            velocity_data = self._prepare_velocity(joint_msg)
            if self.record_joint_velocity and velocity_data is None:
                return

            frame = self._build_frame(
                state,
                action,
                scene_rgb,
                wrist_rgb,
                None,
                wrench_base,
                wrench_stiffness,
                leader_pose_data,
                velocity_data,
            )

            self.dataset.add_frame(frame)
            self.frame_count += 1
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Failed to convert or store synchronized frame: {exc}")

    def _sync_callback_no_wrist(
        self,
        joint_msg: JointState,
        scene_rgb_msg: Image,
    ) -> None:
        self.rx_counts["joint"] += 1
        self.rx_counts["scene_ros"] += 1

        state = self._prepare_state(joint_msg)
        if state is None:
            return

        try:
            scene_rgb = self.bridge.imgmsg_to_cv2(scene_rgb_msg, desired_encoding="rgb8")
            # Femto Bolt is mounted upside down.
            scene_rgb = cv2.rotate(scene_rgb, cv2.ROTATE_180)
            scene_rgb = self._match_feature_shape(
                scene_rgb,
                "observation.images.scene_rgb",
                cv2.INTER_LINEAR,
            )
            self.latest_scene_rgb = scene_rgb
            self.current_state = state

            if not self.recording:
                return

            action = self._prepare_action()
            if action is None:
                return

            wrench_data = self._prepare_wrench_observation()
            if self.record_wrench_forces and wrench_data is None:
                return

            wrench_base, wrench_stiffness = (
                wrench_data if wrench_data is not None else (None, None)
            )

            leader_pose_data = self._prepare_leader_pose_observation()
            if self.record_leader_pose and leader_pose_data is None:
                return

            velocity_data = self._prepare_velocity(joint_msg)
            if self.record_joint_velocity and velocity_data is None:
                return

            frame = self._build_frame(
                state,
                action,
                scene_rgb,
                None,
                None,
                wrench_base,
                wrench_stiffness,
                leader_pose_data,
                velocity_data,
            )

            self.dataset.add_frame(frame)
            self.frame_count += 1
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Failed to convert or store synchronized frame: {exc}")

    def _joint_callback_cv2_no_wrist(self, joint_msg: JointState) -> None:
        self.rx_counts["joint"] += 1

        state = self._prepare_state(joint_msg)
        if state is None:
            return

        try:
            scene_rgb = self._read_scene_frame_from_capture()
            self.rx_counts["scene_cv2"] += 1
            self.latest_scene_rgb = scene_rgb
            self.current_state = state

            if not self.recording:
                return

            action = self._prepare_action()
            if action is None:
                return

            wrench_data = self._prepare_wrench_observation()
            if self.record_wrench_forces and wrench_data is None:
                return

            wrench_base, wrench_stiffness = (
                wrench_data if wrench_data is not None else (None, None)
            )

            leader_pose_data = self._prepare_leader_pose_observation()
            if self.record_leader_pose and leader_pose_data is None:
                return

            velocity_data = self._prepare_velocity(joint_msg)
            if self.record_joint_velocity and velocity_data is None:
                return

            frame = self._build_frame(
                state,
                action,
                scene_rgb,
                None,
                None,
                wrench_base,
                wrench_stiffness,
                leader_pose_data,
                velocity_data,
            )

            self.dataset.add_frame(frame)
            self.frame_count += 1
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Failed to convert or store synchronized frame: {exc}")

    def destroy_node(self) -> bool:
        if self.scene_capture is not None:
            self.scene_capture.release()
            self.scene_capture = None
        return super().destroy_node()


def parse_cli_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="record_lerobot",
        description=(
            "Record synchronized Franka + camera data in LeRobot format. The mark colour and "
            "the wiping manner are fixed for the whole collection: one launch is one session "
            "is one (colour, manner) pair, applied to every episode recorded by it."
        ),
        epilog=(
            "example: ros2 run data_recorder record_lerobot --colour red "
            "--manner gently --ros-args -p dataset_root:=/path/to/data"
        ),
    )
    parser.add_argument(
        "--colour",
        "--color",
        dest="colour",
        choices=VALID_COLOURS,
        help=(
            "colour of the mark to wipe in this collection -- the whiteboard only ever carries "
            "a red and a blue mark. Falls back to -p colour:= if omitted."
        ),
    )
    parser.add_argument(
        "--manner",
        dest="manner",
        choices=VALID_MANNERS,
        help=(
            "wiping manner for this collection (the adverb axis). Falls back "
            "to -p manner:= if omitted."
        ),
    )
    return parser.parse_args(argv)


def main(args: Optional[list[str]] = None) -> None:
    argv = list(sys.argv if args is None else args)
    # remove_ros_args strips "--ros-args ..." so argparse only ever sees this node's own flags;
    # rclpy.init() gets the full argv back and does the mirror-image thing with it.
    cli = parse_cli_args(remove_ros_args(argv)[1:])

    rclpy.init(args=argv)
    try:
        node = FrankaLeRobotRecorder(colour=cli.colour, manner=cli.manner)
    except ValueError as exc:
        # A missing/invalid colour or manner would otherwise surface as an unhandled traceback
        # from inside the constructor; it is an operator input error, so report it as one.
        print(f"record_lerobot: {exc}", file=sys.stderr)
        rclpy.shutdown()
        sys.exit(2)

    try:
        running = True
        while rclpy.ok() and running:
            rclpy.spin_once(node, timeout_sec=0.03)
            running = node.update_gui()
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted by user, stopping recording.")
    finally:
        if node.recording:
            node.stop_recording(task="", save_episode=False)
        node.shutdown_gui()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
