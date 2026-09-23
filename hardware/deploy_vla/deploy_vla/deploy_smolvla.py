"""
Deploys a SmolVLA compliance-VLA checkpoint (compliance-vla,
scripts/serve_policy.py's /act endpoint) on the real Franka follower arm.

Talks to the same server contract as compliance-vla/scripts/client_example.py:
POST {"task", "scene_rgb", "wrist_rgb", "state", "force_history"} -> {"action_chunk"}.
Uses json_numpy.dumps/.loads directly rather than the global json_numpy.patch()
monkeypatch, for the same reason client_example.py does -- see that file's docstring.

state = [q(7), qdot(7), x_f(6)], x_f = follower EE pose as [position(3), rotvec(3)]
in the base frame.
force_history = last 500ms of external wrench (fx,fy,fz,tx,ty,tz), downsampled to
20 samples; only sent for b2/b5 checkpoints (set include_force_history=False for b0).

action_chunk is (chunk_size, action_dim): action_dim 7 = [x_eq(6), gripper(1)]
(b0/b2, fixed/observed stiffness), action_dim 13 = [x_eq(6), log_k(6), gripper(1)]
(b5, compliance output). x_eq(6) is the *absolute* predicted equilibrium pose in
the same [position(3), rotvec(3)] base-frame convention as x_f -- not a delta --
because the whole point of the proposal's identifiability argument is that x_eq is
directly comparable to (and, in training data, extracted from) an absolute pose.
Published straight to variable_impedance_controllers' target_pose/target_stiffness topics, which
already implement the log-space stiffness rate limiter + energy tank (§4.3), so this
script does not re-implement passivity shaping -- it only clips log_k to the
controller-realizable range before publishing.

Scene camera (Femto Bolt) and wrist camera (RealSense) are both read via
cv2.VideoCapture directly off their V4L2 nodes.


ros2 service call /follower/service_server/set_full_collision_behavior \
  franka_msgs/srv/SetFullCollisionBehavior \
"{ lower_torque_thresholds_acceleration: [20,20,20,20,20,20,20],
   upper_torque_thresholds_acceleration: [85,85,85,85,11,11,11],
   lower_torque_thresholds_nominal:      [10,10,10,10,10,10,10],
   upper_torque_thresholds_nominal:      [85,85,85,85,11,11,11],
   lower_force_thresholds_acceleration:  [20,20,20,20,20,20],
   upper_force_thresholds_acceleration:  [85,85,85,11,11,11],
   lower_force_thresholds_nominal:       [10,10,10,10,10,10],
   upper_force_thresholds_nominal:       [85,85,85,11,11,11] }"


"""

import rclpy
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, WrenchStamped
from std_msgs.msg import Float64MultiArray
from control_msgs.action import GripperCommand
from controller_manager_msgs.srv import SwitchController
from franka_msgs.msg import FrankaRobotState
from cv_bridge import CvBridge
from collections import deque
import cv2
import csv
import numpy as np
import os
import requests
import json_numpy
import threading
import time
import tkinter as tk
from tkinter import ttk
from scipy.spatial.transform import Rotation as R, Slerp

from . import score_wipe

# --- Path wiring for tool_offset.npy, same repo-relative layout
# replay_policy_on_data_two_color.py and verify_x_f_frame.py already use (this
# package and compliance-vla are src/ siblings in the same ROS 2 workspace).
# Walked upward from __file__ (rather than a fixed number of dirname() calls) because
# this file's on-disk location differs depending on how it's run: two levels up from
# __file__ is franka_ros2_ws/src when running the src/ copy directly, but ros2 run /
# ros2 launch instead execute colcon's *installed* copy at
# install/deploy_vla/lib/python3.10/site-packages/deploy_vla/deploy_smolvla.py, where
# two levels up is install/deploy_vla/lib/python3.10 -- not a workspace src/ directory
# at all. Searching for the first ancestor that actually contains
# compliance-vla/scripts/tool_offset.npy works under both.
def _find_bookish_scripts_dir(start_dir, max_levels=10):
    candidate = os.path.abspath(start_dir)
    for _ in range(max_levels):
        scripts_dir = os.path.join(candidate, "src", "compliance-vla", "scripts")
        if os.path.isfile(os.path.join(scripts_dir, "tool_offset.npy")):
            return scripts_dir
        parent = os.path.dirname(candidate)
        if parent == candidate:
            break
        candidate = parent
    raise FileNotFoundError(
        "Could not locate compliance-vla/scripts/tool_offset.npy by searching "
        f"upward from {start_dir!r} -- is this workspace missing that src/ package, "
        "or has the workspace been relocated/restructured?"
    )


_BOOKISH_SCRIPTS = _find_bookish_scripts_dir(os.path.dirname(os.path.realpath(__file__)))

# Franka flange -> wiper-tip offset, fit by compliance-vla/scripts/
# calibrate_tool_offset.py against observation.leader_pose -- the exact value
# src/compliance_vla/policy/labels.py bakes into x_f. Loaded once at import time (not per-request)
# since it's a fixed property of one mounted tool, same as replay_policy_on_data_
# two_color.py's own module-level load.
TOOL_OFFSET = np.load(os.path.join(_BOOKISH_SCRIPTS, "tool_offset.npy"))

# Controller-realizable stiffness range from the proposal (§4.1), used to clip
# exp(log_k) before publishing -- keeps a bad prediction from commanding an
# uncommandable stiffness rather than trusting the policy blindly.
K_TRANS_RANGE = (50.0, 1500.0)   # N/m
K_ROT_RANGE = (5.0, 100.0)       # N*m/rad

# Fixed/non-compliant impedance published explicitly the moment variable_impedance_
# controller is switched in (see _start_running) -- matches that controller's own
# static baseline in fr3_bilateral_teleop/config/variable_impedance_validation_
# controllers.yaml (task.k_pos_x/y/z, task.k_rot_x/y/z), set here too rather than
# relying on the controller silently falling back to it: a B0/B2 checkpoint's
# action_dim==7 never publishes target_stiffness at all (see is_compliance_checkpoint
# in run_interactive), so without this the whole rollout would run at whatever that
# yaml happens to say -- correct today, but a silent dependency on a file this script
# doesn't own. A B5 checkpoint's first predicted log_k overwrites this on its first
# published step, same as it always has.
NON_COMPLIANT_STIFFNESS = (300.0, 300.0, 300.0, 20.0, 20.0, 20.0)  # [k_pos_xyz, k_rot_xyz]

# Temporal ensembling across overlapping action chunks (proposal §4.3: "ACT-style
# temporal ensembling across overlapping chunks" -- NOT previously implemented in
# this file; added 2026-09-10 after diagnosing a real local-minimum failure mode).
# Without this, each freshly-queried chunk replaces the previous one wholesale at
# the chunk boundary -- a rigid, non-compliant checkpoint (b0/b2, fixed stiffness,
# see NON_COMPLIANT_STIFFNESS) has nothing to absorb the resulting discontinuity
# and can stall re-chasing a slightly different target every steps_to_execute steps
# instead of making net progress, worst at short steps_to_execute where boundaries
# are frequent. A compliance-output checkpoint (b5) tolerates the same discontinuity
# because it can physically give rather than fight it, which is the leading
# hypothesis for why b0 needed steps_to_execute>8 while b5 was fine at 3 -- see
# language_grounding_issue_handoff.md-adjacent investigation notes, 2026-09-10.
#
# Weighting: a candidate prediction for a given low-level step is weighted
# exp(-ENSEMBLE_M * age), age = how many low-level steps ago the chunk that
# produced it was queried, so the freshest available prediction for that step
# dominates and older ones decay in influence rather than being discarded outright
# at the boundary. ENSEMBLE_M is UNTUNED -- start here, watch settle behavior in
# free space before trusting it on the contact task, and treat this the same way
# the RFF bias-model hyperparameters were treated: verify empirically, don't assume
# the first guess is right.
ENABLE_TEMPORAL_ENSEMBLE_DEFAULT = True
ENSEMBLE_M_DEFAULT = 0.05

# get_action_chunk's requests.post had no timeout: a stalled/hung inference server (as
# opposed to one that's simply down, which fails the connection immediately) blocked this
# call indefinitely, and the operator's 's'/'m'/'q' keys are only polled *between* main-loop
# iterations -- so a hang here also stalled the "m to stop" keypress for however long it
# lasted, with variable_impedance_controllers left holding the last published target in the meantime.
# Bounded to roughly a couple of chunk periods so a slow-but-alive server still gets a
# response through; a request that misses this is treated as failed (see run_interactive).
SERVER_REQUEST_TIMEOUT_SEC = 5.0

# Franka Hand full-open width; gripper action outputs are treated as an absolute
# target width in meters and clipped into this range (see run_evaluation_episode).
GRIPPER_WIDTH_RANGE = (0.0, 0.08)

# Hard (non-skippable) per-step gate, ported from replay_policy_on_data_two_color.py
# after that script's thresholds were validated on real hardware (it correctly
# differentiates mark colors): refuse to publish a predicted target this far from the
# arm's *live* pose rather than trusting the policy blindly. This closed-loop rollout
# has no equivalent check today -- action_scale/constant_orientation only reshape a
# prediction, they never refuse one -- and here a bad prediction also feeds back into
# next step's state, unlike that script's open-loop replay.
MAX_STEP_POSITION_JUMP_M = 0.3
MAX_STEP_ORIENTATION_JUMP_DEG = 90.0

# After this many low-level steps, stop publishing actions and switch the follower
# back to move_to_start_example_controller -- same controller-swap pair the deploy
# runbook (compliance-vla/deploy_vla_on_franka.md) documents as a manual
# `ros2 control switch_controllers` command. Just the declared default for the
# max_steps ROS param (see __init__) -- self.max_steps is what's actually checked.
MAX_STEPS = 200
VARIABLE_IMPEDANCE_CONTROLLER = "variable_impedance_controller"
MOVE_TO_START_CONTROLLER = "move_to_start_example_controller"

# Orbbec Femto Bolt needs a few seconds after cv2.VideoCapture opens before frames
# stabilize (auto-exposure/white-balance settle); block here rather than letting the
# operator hit 's' into a still-warming feed.
CAMERA_WARMUP_SEC = 7.0

# Hardcoded orientation override used when constant_orientation=True -- see the
# "DO NOT CHANGE THIS!!!" comment in _publish_target_pose. Named here too so the
# diagnostic logging in _log_orientation_debug can compare against it without a
# second copy of the literal.
CONSTANT_ORIENTATION_QUAT = (0.92388, -0.38268, 0.0, 0.0)  # (x, y, z, w)

# Reference wipe-motion displacement, in the same base-frame/PoseStamped convention as
# current_pose: mean x_f delta across 33 "wipe the blue mark firmly" episodes in
# data_two_color (demo_blue_firm + demo_blue_firmB; 3/36 excluded -- 2 failed contact-frame
# extraction after trimming, 1 where the trim heuristic found no separable pause and so
# left an untrimmed home-return tail in place, see below). All 33 agree in sign on x
# (range +0.166 to +0.282m, mean +0.207m); y is bimodal (roughly -0.03 to -0.06m for one
# mark position, +0.08 to +0.16m for another) and not a meaningful single-value check, left
# at 0 here.
#
# 2026-09-05 correction: an earlier version of this constant was (-0.022, -0.003) --
# WRONG, computed without src/compliance_vla/policy/dataset.py's _trim_to_last_contact step. data_two_color
# episodes have a documented recorder bug (data_recorder commit 2f63ca3) that
# appends a still-recording return-to-start_joint_configuration move after the operator's
# 's' keypress, and dataset.py's own docstring notes a naive contact-force threshold
# (exactly what was used here) does NOT exclude it -- that return move produces 2-3.6N
# wrench-estimation transients from the commanded motion itself, comfortably over this
# file's 2N threshold, for ~100 raw frames. Every "wipe the blue mark firmly" episode's
# raw (untrimmed) end position converged to the same ~(0.306, 0.001, 0.486) -- the
# fixed home pose, not a wipe endpoint -- which is what should have been the tell.
# That untrimmed analysis produced a false "confirmed bug": a deployed rollout matching
# the untrimmed (wrong) sign looked like a reversal, when it was actually correct.
# Re-verify against this file's _trim_to_last_contact-equivalent methodology, not a bare
# contact-force threshold, before trusting any future disagreement here as a real bug.
REFERENCE_WIPE_DELTA_XY = (0.207, 0.0)

# Same contact definition extract_impedance_labels.py uses (its default
# --contact-force-threshold) to isolate the wipe stroke from the free-space reach/
# descent onto the board -- confirmed necessary empirically: a raw start-of-run-to-
# stop delta on 2026-09-05 was dominated by a ~15cm approach/descent phase that
# swamped the actual post-contact wipe displacement.
CONTACT_FORCE_THRESHOLD_N = 2.0

# Where per-rollout current_pose logs (one CSV per _start_running..._stop_and_home
# window) are written, relative to this file.
POSE_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rollout_pose_logs")

# Fixed vocabulary for the pre-episode instruction picker, matching
# compliance-vla/scripts/dataset_io.py's parse_referent_from_task /
# parse_manner_from_task enums exactly (compliance-vla-icra2027-proposalv2.md §5.2)
# so a GUI-composed instruction always parses back to the same referent/manner the
# training data used. Only red/blue have recorded episodes today; green/black are
# in the enum but untested -- kept here anyway since parse_referent_from_task
# already expects them. Deliberately excludes the proposal's held-out OOD manner
# words (lightly, scrub hard, barely touch it) -- those are eval-only by design.
INSTRUCTION_COLORS = ("left", "right")
INSTRUCTION_MANNERS = ("normally", "firmly")

# Mark colour to segment when *scoring* a run's before/after photos (score_wipe.py's
# DEFAULT_HSV_RANGES key set) -- picked per-episode via the same pre-episode dialog
# as INSTRUCTION_COLORS/INSTRUCTION_MANNERS, but a distinct concept: INSTRUCTION_COLORS
# names *which* mark to wipe (left/right), never a literal colour, and is sent to the
# policy as part of the instruction string; SCORE_COLORS is the physical mark colour,
# used only locally to segment the before/after photos and never sent to the server.
SCORE_COLORS = ("blue", "red")

# Where before/after wipe-scoring photos (ROI-cropped -- see wipe_score_roi) and
# score_wipe.score_and_visualize's scored figure are saved, one set per run -- same
# directory-next-to-this-file convention as POSE_LOG_DIR.
WIPE_SCORE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wipe_scores")

# Fixed wait after move_to_start_example_controller is activated, before capturing
# the wipe-scoring "after" photo -- the switch_controller service call (see
# _stop_and_home) only confirms the controller swap, not that the resulting home
# motion has finished, so capturing immediately after it returns risks a photo taken
# mid-motion.
HOME_SETTLE_WAIT_SEC = 5.0


class SmolVLADeployment(Node):
    def __init__(
        self,
        server_url: str = "http://127.0.0.1:8000/act",
        follower_ns: str = "follower",
        include_force_history: bool = True,
        image_size: int = 224,
        scene_cv2_device=6,
        wrist_cv2_device=4,
        action_scale: float = 1.0,
        camera_warmup_sec: float = CAMERA_WARMUP_SEC,
        steps_to_execute: int = 1,
        enable_temporal_ensemble: bool = ENABLE_TEMPORAL_ENSEMBLE_DEFAULT,
        ensemble_m: float = ENSEMBLE_M_DEFAULT,
        enable_scene_camera: bool = True,
        enable_wrist_camera: bool = True,
        max_steps: int = MAX_STEPS,
        wipe_score_roi=(498, 367, 306, 226),
    ):
        super().__init__("franka_smolvla_client")

        # --- Configuration ---
        self.follower_ns = follower_ns
        self.server_url = server_url
        self.include_force_history = include_force_history
        self.image_size = image_size
        self.camera_warmup_sec = float(camera_warmup_sec)
        # x_eq is an *absolute* target pose (see module docstring), not a delta, so this
        # is NOT the same "scale" sibling scripts (deploy_vla_crisp/cic) multiply the raw
        # action by -- doing that here would move the target toward the base-frame origin,
        # not toward the current pose. Instead this blends the published target a fraction
        # of the way from the current pose to the policy's predicted x_eq each step: 1.0
        # trusts the prediction fully (default, matches prior behavior); e.g. 0.3 commands
        # only 30% of the displacement per step, which -- combined with variable_impedance_controllers'
        # own EMA pose filter -- makes the arm track the policy noticeably more cautiously.
        self.action_scale = float(np.clip(action_scale, 0.0, 1.0))
        self.chunk_hz = 30.0        # matches the policy's action-chunk rate (§4.2)
        # Default 1 = strictly reactive: execute only action_chunk[0] and immediately
        # request a fresh chunk for the next step, rather than executing several cached
        # low-level steps from one server response before requerying. This does NOT stop
        # the arm from holding its last published target while a request is in flight --
        # variable_impedance_controllers keeps tracking whatever target_pose/target_stiffness it last
        # received, there's no "do nothing" command this script can send instead -- it
        # only guarantees no *cached* prediction ever gets executed. Set higher (e.g. 3,
        # ~100ms of receding horizon -> ~10Hz replanning per §4.3) to trade reactivity for
        # fewer server round trips. Declared as a ROS param (rather than just a
        # constructor kwarg) so it can be overridden per-launch from the command line:
        #   ros2 run deploy_vla deploy_smolvla --ros-args -p steps_to_execute:=3
        # The constructor kwarg above only supplies the declared default. Temporal
        # ensembling (below) smooths the discontinuity a chunk boundary introduces
        # regardless of this value -- it does not remove the round-trip-cost
        # tradeoff steps_to_execute itself controls.
        self.declare_parameter("steps_to_execute", steps_to_execute)
        self.steps_to_execute = int(self.get_parameter("steps_to_execute").value)

        # Temporal ensembling toggle/hyperparameter -- see ENABLE_TEMPORAL_ENSEMBLE_
        # DEFAULT's module comment. Declared as ROS params, same reasoning as
        # steps_to_execute, so a quick disable/A-B check doesn't need a redeploy:
        #   ros2 run deploy_vla deploy_smolvla --ros-args -p enable_temporal_ensemble:=false
        #   ros2 run deploy_vla deploy_smolvla --ros-args -p ensemble_m:=0.1
        self.declare_parameter("enable_temporal_ensemble", enable_temporal_ensemble)
        self.declare_parameter("ensemble_m", ensemble_m)
        self.enable_temporal_ensemble = bool(self.get_parameter("enable_temporal_ensemble").value)
        self.ensemble_m = float(self.get_parameter("ensemble_m").value)
        # Buffer of (query_step, action_chunk) pairs still covering at least one
        # low-level step that hasn't executed yet -- see _buffer_chunk/_ensembled_
        # action. query_step is self.step_count at the moment that chunk was
        # requested, i.e. the absolute low-level step chunk[0] is a prediction for.
        self._chunk_buffer = deque()

        # After this many low-level steps, run_interactive stops publishing actions and
        # switches the follower back to move_to_start_example_controller (see MAX_STEPS'
        # module-level comment for why 500). Declared as a ROS param for the same reason
        # steps_to_execute is -- e.g. a scripted rollout runner wanting shorter episodes:
        #   ros2 run deploy_vla deploy_smolvla --ros-args -p max_steps:=200
        self.declare_parameter("max_steps", max_steps)
        self.max_steps = int(self.get_parameter("max_steps").value)

        # Independently disable either camera feed (e.g. running with only one camera
        # plugged in, or an ablation of what the policy actually needs) -- a disabled
        # feed is replaced with a fixed all-zero image rather than left as None, so nothing
        # downstream (run_interactive's readiness gate, get_action_chunk's payload, the
        # preview window) needs its own special-casing; it just always sees an image, a
        # blank one. Declared as ROS params for the same reason steps_to_execute is:
        #   ros2 run deploy_vla deploy_smolvla --ros-args -p enable_scene_camera:=false
        #   ros2 run deploy_vla deploy_smolvla --ros-args -p enable_wrist_camera:=false
        self.declare_parameter("enable_scene_camera", enable_scene_camera)
        self.declare_parameter("enable_wrist_camera", enable_wrist_camera)
        self.enable_scene_camera = bool(self.get_parameter("enable_scene_camera").value)
        self.enable_wrist_camera = bool(self.get_parameter("enable_wrist_camera").value)

        self.force_history_window_sec = 0.5
        self.force_history_samples = 20
        self.constant_orientation = False  # if True, override the policy's predicted orientation with a fixed one
        # Measured 2026-09-05 via _log_orientation_debug, BEFORE get_current_state was
        # fixed (2026-09-08) to feed FK+TOOL_OFFSET instead of current_pose/tool_pose --
        # at the time this was measured, get_current_state's fed-back state was ~44deg
        # rotated from the training distribution (see language_grounding_issue_handoff.md),
        # so this 39.5deg is suspiciously close to being a correction for THAT bug rather
        # than a real, independent tool-mount twist. Left in place because the published
        # orientation still needs *some* correction and this is the last measured value,
        # but treat it as stale: re-run the _log_orientation_debug measurement now that
        # get_current_state feeds the correct (flange-frame) convention, and expect this
        # number to change, possibly toward ~0.
        self.orientation_correction_deg = 39.5
        self.step_count = 0

        # Instruction for the run currently in progress (or the most recently run
        # one, while idle) -- set by _prompt_episode_instruction before each
        # _start_running call, see run_interactive. Remembers the last picked
        # color/manner as the next dialog's default.
        self.current_instruction = None
        self._last_color = INSTRUCTION_COLORS[1]   # "right", matches the prior hardcoded default
        self._last_manner = INSTRUCTION_MANNERS[1]  # "firmly", ditto

        # Wipe-scoring (score_wipe.py integration): mark colour to segment when
        # scoring a run's before/after photos, set per-episode by the GUI dialog
        # below (run_interactive) -- left None (scoring skipped in _start_running/
        # _stop_and_home) for callers like run_scripted_rollout.py that don't set it.
        self.score_color = None
        self._last_score_color = SCORE_COLORS[0]  # "blue"
        # Result of the most recently completed wipe-scoring, set by
        # _stop_and_home (None if scoring was skipped or failed for that run) --
        # exists so a caller like run_final_evaluation.py can read the outcome of
        # the run it just drove instead of only seeing it go by in the log.
        # Dict shape: {"score_color", "percent_wiped", "before_area_px",
        # "after_area_px", "out_path"}.
        self.last_wipe_score = None
        # Pixel ROI restricting the before/after photos to just the wipe board (same
        # x,y,w,h convention as score_wipe.py's --roi) -- find one with
        # `score_wipe.py --roi-tune --live`. None crops nothing (whole scene frame).
        self.wipe_score_roi = score_wipe.parse_roi(wipe_score_roi)
        # "Before" photo captured in _start_running, ROI-cropped, consumed and reset
        # to None by the matching score in _stop_and_home.
        self._score_before_crop = None

        # Tool offset: fixed flange -> wiper-tip translation, loaded once at import time
        # (module-level TOOL_OFFSET, from calibrate_tool_offset.py's fit) -- the exact
        # value src/compliance_vla/policy/labels.py bakes into x_f. No longer a per-launch ROS param: this
        # used to be independently configurable (tool_offset_translation/rotation_deg/
        # include_rotation/apply_to_state below) with a default that both (a) fed the
        # policy a different frame than it was trained on -- current_pose/tool_pose is
        # the O_T_EE *hand* frame, ~44deg rotated from the FK *flange* frame panda_fk.fk()
        # (no hand/TCP offset applied) and training actually use -- and (b) subtracted an
        # extra, untrained-for 8cm from every published target at publish time. Both
        # measured directly (language_grounding_issue_handoff.md, 2026-09-07/08 updates;
        # compliance-vla/scripts/check_state_convention_sensitivity.py). Fixed by
        # dropping tool_pose/current_pose-derived state entirely and mirroring replay_
        # policy_on_data_two_color.py's already hardware-validated FK+TOOL_OFFSET input /
        # raw-x_eq publish path exactly -- see get_current_state/_publish_target_pose.
        self.bridge = CvBridge()

        # --- Scene camera: cv2.VideoCapture straight off the Femto Bolt's V4L2 node,
        # bypassing the orbbec_camera ROS driver entirely -- deliberately, not for lack
        # of trying it the other way. The driver's own USB enumeration has repeatedly
        # failed/re-enumerated mid-session in this environment (see data_recorder's
        # use_scene_cv2_capture fallback and tasks.md Day 2/4/6); the kernel's
        # UVC driver via V4L2 has been more tolerant of it. Explicit cv2.CAP_V4L2 backend
        # (OpenCV probes GStreamer first otherwise on this box) and explicit YUYV FOURCC
        # -- deliberately *not* MJPG, because this process also imports cv_bridge, and
        # mixing MJPG-format V4L2 reads with cv_bridge in the same process has been
        # observed to pull in two different OpenCV builds and produce a bogus cv2.error
        # out of imdecode. Matches data_recorder's _init_scene_cv2_capture
        # exactly for this reason. The wrist camera below uses the identical pattern.
        self.latest_scene_frame = None
        self._scene_frame_lock = threading.Lock()
        self._stop_scene_capture = threading.Event()
        self.scene_capture = None
        self._scene_capture_thread = None
        if self.enable_scene_camera:
            self.scene_capture = cv2.VideoCapture(scene_cv2_device, cv2.CAP_V4L2)
            if self.scene_capture.isOpened():
                self.scene_capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
                self.scene_capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.image_size)
                self.scene_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.image_size)
                ok, _ = self.scene_capture.read()
                if not ok:
                    self.get_logger().error(
                        f"Opened scene_cv2_device={scene_cv2_device!r} but a verification read "
                        "failed -- likely the wrong /dev/video* node (the Femto Bolt exposes "
                        "several); check `for v in /dev/video*; do echo \"$v: $(cat /sys/class/"
                        'video4linux/$(basename $v)/name)"; done` and pass the right one.'
                    )
            else:
                self.get_logger().error(f"Cannot open scene cv2.VideoCapture device {scene_cv2_device!r}")
            self._scene_capture_thread = threading.Thread(
                target=self._scene_capture_loop, daemon=True
            )
            self._scene_capture_thread.start()
        else:
            self.get_logger().info(
                "Scene camera disabled (enable_scene_camera:=false) -- sending an all-zero "
                "scene_rgb to the policy every step."
            )

        # --- Wrist camera: cv2.VideoCapture straight off the RealSense's V4L2 node, same
        # as the scene camera above -- NOT the ROS Image topic
        # (/wrist/wrist_camera/color/image_raw) data_recorder's wrist_rgb_topic
        # default used, and this file itself used until 2026-09-10. Switched after that
        # ROS-topic path was observed, on real hardware, to intermittently corrupt/drop
        # samples under DDS reliable-QoS backlog ("sequence size exceeds remaining
        # buffer") -- worst whenever a caller's control loop went a while without spinning
        # (e.g. across get_action_chunk's blocking HTTP round trip), which then left
        # wrist_img_np silently frozen rather than just occasionally stale. No rotation
        # applied here (unlike the scene camera's Femto-Bolt-is-upside-down fix) -- the
        # wrist camera's mount isn't upside down; this matches what the old
        # _wrist_image_callback did (BGR->RGB only, no rotate).
        self.latest_wrist_frame = None
        self.wrist_capture = None
        if self.enable_wrist_camera:
            self.wrist_capture = cv2.VideoCapture(wrist_cv2_device, cv2.CAP_V4L2)
            if self.wrist_capture.isOpened():
                self.wrist_capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
                self.wrist_capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.image_size)
                self.wrist_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.image_size)
                ok, _ = self.wrist_capture.read()
                if not ok:
                    self.get_logger().error(
                        f"Opened wrist_cv2_device={wrist_cv2_device!r} but a verification read "
                        "failed -- likely the wrong /dev/video* node (the RealSense exposes "
                        "several); check `for v in /dev/video*; do echo \"$v: $(cat /sys/class/"
                        'video4linux/$(basename $v)/name)"; done` and pass the right one.'
                    )
            else:
                self.get_logger().error(f"Cannot open wrist cv2.VideoCapture device {wrist_cv2_device!r}")
        else:
            self.get_logger().info(
                "Wrist camera disabled (enable_wrist_camera:=false) -- sending an all-zero "
                "wrist_rgb to the policy every step."
            )

        # --- Proprioception: q, qdot straight off the follower's own broadcaster,
        # same source record_demo.py / the free-space sweep tooling already use.
        self.joint_q = None
        self.joint_qdot = None
        self.create_subscription(
            JointState,
            f"/{follower_ns}/franka_robot_state_broadcaster/measured_joint_states",
            self._joint_state_callback,
            10,
        )

        # --- Live EE pose + flange pose, both derived from ONE subscription to
        # franka_robot_state_broadcaster's combined robot_state message, not two.
        # current_pose (O_T_EE, hand frame) and flange_pose (O_T_F, flange frame) used
        # to come from separate subscriptions -- a plain current_pose PoseStamped topic
        # plus this robot_state topic -- which meant deserializing/processing the same
        # O_T_EE data twice per broadcaster tick (up to ~1kHz) on this node's single
        # background executor thread, PLUS robot_state's own much larger message on top.
        # That redundant load is the leading suspect for a switch_controller call timing
        # out waiting for its response callback to be serviced (see the investigation
        # after 2026-09-08's panda_fk removal, which is what added the second
        # subscription in the first place) -- consolidating to one callback halves the
        # high-rate topic traffic this node has to process either way, independent of
        # whether it turns out to be the actual root cause.
        #
        # current_pose is kept ONLY for the preflight/step-jump safety distance checks,
        # the pose CSV log, and the (currently dead, action_scale=1.0 in main()) blend
        # reference -- NEVER used to build the policy's state input, see get_current_state
        # which uses flange_pose + TOOL_OFFSET (the flange-frame convention training
        # actually used) instead. flange_pose is O_T_F = O_T_EE * F_T_EE^-1: o_t_ee
        # (O_T_EE, base -> hand frame) and f_t_ee (F_T_EE, flange -> hand frame, bakes in
        # the mounted gripper's NE_T_EE offset) are both in the same message, so this
        # composition needs no joint-angle FK and no separate Panda DH model to keep in
        # sync with the real robot. Deliberately NOT using O_T_EE alone as flange_pose:
        # it's ~44deg rotated / ~8cm offset from the flange frame src/compliance_vla/policy/labels.py's
        # x_f uses (see get_current_state) -- that would be the exact bug fixed on
        # 2026-09-07/08 (language_grounding_issue_handoff.md), just via a different path.
        self.current_pose = None
        self.flange_pose = None
        # Hemisphere-continuity state for _robot_state_callback's quaternion sign fix --
        # see that callback for why.
        self._prev_quaternion = None
        # Per-rollout current_pose CSV logging -- see _start_running/_stop_and_home.
        # _pose_log_lock guards _pose_log_file/_pose_log_writer against a race between
        # _robot_state_callback (background executor thread, up to ~1kHz) and
        # _stop_and_home (main thread): without it, the callback can read a non-None
        # _pose_log_writer, then have _stop_and_home close the underlying file before
        # the callback's writerow() call runs, raising "I/O operation on closed file".
        self._pose_log_lock = threading.Lock()
        self._pose_log_file = None
        self._pose_log_writer = None
        self._pose_log_samples = []  # [(t, position, force_mag), ...] for the current run
        self.create_subscription(
            FrankaRobotState,
            f"/{follower_ns}/franka_robot_state_broadcaster/robot_state",
            self._robot_state_callback,
            10,
        )

        # --- External wrench, for force_history (b2/b5 checkpoints only) ---
        self._wrench_buffer = deque(maxlen=2000)  # ~2s headroom at up to 1kHz
        # _wrench_callback appends on the background executor thread (see __init__'s
        # continuous-spin comment below) while get_force_history() (and the force_mag
        # read in _robot_state_callback) iterate/index it from whatever thread calls
        # them -- unprotected, deque raises "RuntimeError: deque mutated during
        # iteration" if a callback appends mid-iteration. Same producer/consumer
        # hazard _scene_frame_lock guards for the scene camera; same fix here.
        self._wrench_buffer_lock = threading.Lock()
        # Diagnostic counter -- see _start_running/_stop_and_home -- to check whether
        # _wrench_callback is actually being serviced during a run (suspected rclpy
        # spin_once starvation: it services one ready callback per call, and this node's
        # manual polling loop competes wrist-image/joint-state/pose subscriptions against
        # this one, plus yields nothing at all during get_action_chunk's blocking
        # requests.post()).
        self._wrench_callback_count = 0
        self._wrench_callback_count_at_start = 0
        if self.include_force_history:
            self.create_subscription(
                WrenchStamped,
                f"/{follower_ns}/franka_robot_state_broadcaster/external_wrench_in_base_frame",
                self._wrench_callback,
                50,
            )

        # --- Outputs: variable_impedance_controllers' CartesianController (already wired with
        # the log-space stiffness rate limiter + energy tank from §4.3) ---
        self.pose_pub = self.create_publisher(PoseStamped, f"/{follower_ns}/target_pose", 10)
        self.stiffness_pub = self.create_publisher(
            Float64MultiArray, f"/{follower_ns}/target_stiffness", 10
        )

        # --- Gripper ---
        self._gripper_client = ActionClient(
            self, GripperCommand, f"/{follower_ns}/franka_gripper/gripper_action"
        )
        self._last_gripper_width = None
        self._gripper_goal_in_flight = False

        # --- Controller switching, to move home after MAX_STEPS -- same service/
        # request shape as fr3_bilateral_teleop's calibrate_payload.py switch_controller ---
        self.switch_client = self.create_client(
            SwitchController, f"/{follower_ns}/controller_manager/switch_controller"
        )

        # --- Continuous background spinning ---
        # Confirmed 2026-09-05 via _wrench_callback_count: manually polling rclpy.spin_once
        # from run_interactive (a handful of short-timeout calls per loop, competing against
        # wrist-image/joint-state/pose subscriptions, plus zero spins at all during
        # get_action_chunk's blocking requests.post()) was starving the 50Hz wrench
        # subscription almost completely -- force_history was being built from an
        # effectively-frozen buffer despite the real /external_wrench_in_base_frame topic
        # publishing fine (confirmed independently with live_force_band_display.py). Spinning
        # continuously in its own thread -- same fix as the scene camera's dedicated capture
        # thread above, for the same reason -- decouples callback servicing from the main
        # loop's timing entirely. SingleThreadedExecutor is enough (not MultiThreaded): every
        # subscription/client here uses the node's default MutuallyExclusiveCallbackGroup, so
        # callbacks are serialized regardless of executor type -- the fix is running spin()
        # continuously, not running callbacks in parallel. switch_controller waits on this
        # executor's background thread rather than calling rclpy.spin_until_future_complete
        # (see that method), since that helper would add/remove this node from a second,
        # temporary executor and removing it would deregister every callback from this one.
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self)
        self._executor_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._executor_thread.start()

    def shutdown_executor(self):
        self._executor.shutdown()
        self._executor_thread.join(timeout=2.0)

    # --- Callbacks ---

    def _scene_capture_loop(self):
        """Continuously drains the scene camera so a read always returns the most
        recent frame, instead of one queued up while the main loop was busy
        publishing/waiting on robot motion (same pattern as deploy_vla_cic.py's
        primary-camera capture thread)."""
        while not self._stop_scene_capture.is_set():
            ok, frame = self.scene_capture.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Femto Bolt is mounted upside down -- same fix, same order (convert then
            # rotate) as data_recorder's _read_scene_frame_from_capture, which
            # built the training data this policy was trained on from the same camera.
            rgb = cv2.rotate(rgb, cv2.ROTATE_180)
            with self._scene_frame_lock:
                self.latest_scene_frame = rgb

    def get_scene_frame(self):
        if not self.enable_scene_camera:
            return np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        with self._scene_frame_lock:
            if self.latest_scene_frame is None:
                return None
            return self.latest_scene_frame.copy()

    def get_wrist_frame(self):
        """Reads one frame directly off the wrist camera's V4L2 capture -- same
        "called inline from the main loop, no background drain" convention as
        get_scene_frame, so a returned frame can be as stale as one main-loop
        iteration. No rotation (unlike get_scene_frame's Femto-Bolt-upside-down
        fix) -- see this class's wrist-camera setup comment for why."""
        if not self.enable_wrist_camera:
            return np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        ok, frame = self.wrist_capture.read()
        if not ok:
            return self.latest_wrist_frame
        self.latest_wrist_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.latest_wrist_frame.copy()

    def _joint_state_callback(self, msg):
        if len(msg.position) < 7 or len(msg.velocity) < 7:
            self.get_logger().warn(
                "measured_joint_states has fewer than 7 positions/velocities.",
                throttle_duration_sec=2.0,
            )
            return
        self.joint_q = np.asarray(msg.position[:7], dtype=np.float32)
        self.joint_qdot = np.asarray(msg.velocity[:7], dtype=np.float32)

    def _robot_state_callback(self, msg):
        """Single callback for both current_pose (O_T_EE, hand frame) and flange_pose
        (O_T_F, flange frame) -- see the subscription's __init__ comment for why this
        used to be two separate high-rate subscriptions and isn't anymore."""
        o_t_ee_position = np.array([
            msg.o_t_ee.pose.position.x, msg.o_t_ee.pose.position.y, msg.o_t_ee.pose.position.z,
        ])
        o_t_ee_quaternion = np.array([
            msg.o_t_ee.pose.orientation.x, msg.o_t_ee.pose.orientation.y,
            msg.o_t_ee.pose.orientation.z, msg.o_t_ee.pose.orientation.w,
        ])
        # q and -q are the same rotation, but near a 180 deg orientation (w~0 -- exactly
        # where this task's fixed target sits) the raw quaternion can equivalently read as
        # either from one sample to the next, and as_rotvec()'s principal-branch [0, pi]
        # output then flips the whole axis sign for the same physical pose. Pin the sign to
        # match the previous sample so downstream rotvec conversions don't see that
        # spurious flip every time it happens.
        if self._prev_quaternion is not None and np.dot(o_t_ee_quaternion, self._prev_quaternion) < 0.0:
            o_t_ee_quaternion = -o_t_ee_quaternion
        self._prev_quaternion = o_t_ee_quaternion
        o_t_ee_rotation = R.from_quat(o_t_ee_quaternion)
        self.current_pose = {"position": o_t_ee_position, "orientation": o_t_ee_rotation}

        with self._pose_log_lock:
            if self._pose_log_writer is not None:
                t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                # Nearest available external-force reading, for the contact-based wipe-window
                # segmentation in _stop_and_home -- see CONTACT_FORCE_THRESHOLD_N. A raw
                # start-of-run-to-stop delta is dominated by the free-space reach/descent onto
                # the board, which is not the wipe. No lock needed here (unlike
                # get_force_history): this callback and _wrench_callback's append both run on
                # the same single background executor thread (see __init__'s continuous-spin
                # comment), so they're already serialized and never execute concurrently with
                # EACH OTHER -- the actual race _wrench_buffer_lock guards against is a
                # different thread (get_force_history, called from run_interactive's main
                # loop) reading while this thread appends.
                force_mag = float(np.linalg.norm(self._wrench_buffer[-1][1][:3])) if self._wrench_buffer else None
                self._pose_log_writer.writerow([t, *o_t_ee_position, *o_t_ee_quaternion, force_mag])
                if force_mag is not None:
                    self._pose_log_samples.append((t, o_t_ee_position.copy(), force_mag))

        # Flange pose: O_T_F = O_T_EE * F_T_EE^-1. f_t_ee (F_T_EE, flange -> hand frame)
        # bakes in the mounted gripper's NE_T_EE offset, so this needs no joint-angle FK.
        f_t_ee_position = np.array([
            msg.f_t_ee.pose.position.x, msg.f_t_ee.pose.position.y, msg.f_t_ee.pose.position.z,
        ])
        f_t_ee_rotation = R.from_quat([
            msg.f_t_ee.pose.orientation.x, msg.f_t_ee.pose.orientation.y,
            msg.f_t_ee.pose.orientation.z, msg.f_t_ee.pose.orientation.w,
        ])
        flange_rotation = o_t_ee_rotation * f_t_ee_rotation.inv()
        flange_position = o_t_ee_position - flange_rotation.apply(f_t_ee_position)
        self.flange_pose = {"position": flange_position, "orientation": flange_rotation}

    def _wrench_callback(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        wrench = np.array(
            [
                msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z,
                msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z,
            ],
            dtype=np.float32,
        )
        with self._wrench_buffer_lock:
            self._wrench_buffer.append((t, wrench))
        self._wrench_callback_count += 1

    # --- Observation assembly ---

    def get_current_state(self):
        """20-dim proprio matching the server contract: [q(7), qdot(7), x_f(6)].

        x_f = self.flange_pose position + TOOL_OFFSET (tip position), FLANGE rotation --
        the exact mixed convention src/compliance_vla/policy/labels.py and diagnose_language_grounding.
        load_observation build it in (tool_offset.npy was fit as a translation only,
        never a rotation). self.flange_pose comes from franka_robot_state_broadcaster's
        robot_state message (o_t_ee/f_t_ee composed -- see _robot_state_callback), not
        joint-angle FK. Deliberately NOT current_pose (that reports O_T_EE, the robot's
        *hand* frame -- measured ~44deg rotated from the flange frame here, see
        language_grounding_issue_handoff.md's 2026-09-07/08 updates and
        check_state_convention_sensitivity.py) -- feeding that would put every live
        state 44deg + 8cm outside the training distribution on every single step."""
        if self.joint_q is None or self.flange_pose is None:
            return None
        tip_pos = self.flange_pose["position"] + self.flange_pose["orientation"].apply(TOOL_OFFSET)
        x_f = np.concatenate([tip_pos, self.flange_pose["orientation"].as_rotvec()])
        return np.concatenate([self.joint_q, self.joint_qdot, x_f]).astype(np.float32)

    def get_force_history(self):
        """(20, 6) trailing-500ms wrench, resampled to a fixed rate, or None if
        the buffer doesn't yet span the full window."""
        # Snapshot under the lock (a plain list, cheap even at maxlen=2000) rather than
        # iterating self._wrench_buffer directly -- this runs on the caller's thread
        # (run_interactive's main loop / a scripted rollout runner) while _wrench_callback
        # keeps appending on the background executor thread; iterating the live deque
        # unprotected raises "RuntimeError: deque mutated during iteration" the moment a
        # wrench message arrives mid-iteration (hit in practice, not just theoretical).
        with self._wrench_buffer_lock:
            buffer_snapshot = list(self._wrench_buffer)

        if len(buffer_snapshot) < 2:
            return None
        now = buffer_snapshot[-1][0]
        oldest = buffer_snapshot[0][0]
        if now - oldest < self.force_history_window_sec * 0.8:
            return None  # not enough history yet, e.g. right after startup

        times = np.array([t for t, _ in buffer_snapshot])
        values = np.array([w for _, w in buffer_snapshot])
        target_times = np.linspace(
            now - self.force_history_window_sec, now, self.force_history_samples
        )
        resampled = np.stack(
            [np.interp(target_times, times, values[:, i]) for i in range(6)], axis=1
        )
        return resampled.astype(np.float32)

    def _resize(self, img_np):
        if img_np.shape[0] != self.image_size or img_np.shape[1] != self.image_size:
            return cv2.resize(img_np, (self.image_size, self.image_size))
        return img_np

    # --- Server call (mirrors client_example.py's get_action_chunk) ---

    def get_action_chunk(self, scene_rgb, wrist_rgb, state, task, force_history=None):
        payload = {"task": task, "scene_rgb": scene_rgb, "wrist_rgb": wrist_rgb, "state": state}
        if force_history is not None:
            payload["force_history"] = force_history
        try:
            resp = requests.post(
                self.server_url,
                data=json_numpy.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=SERVER_REQUEST_TIMEOUT_SEC,
            )
            resp.raise_for_status()
            return json_numpy.loads(resp.content)["action_chunk"]
        except requests.exceptions.RequestException as e:
            self.get_logger().error(f"Server query failed: {e}")
            return None

    # --- Temporal ensembling across overlapping chunks ---
    # See ENABLE_TEMPORAL_ENSEMBLE_DEFAULT's module comment for why this exists.

    def _buffer_chunk(self, query_step, action_chunk):
        """Records one server response for ensembling and drops any previously
        buffered chunk whose coverage has fully passed. query_step is self.step_
        count at the moment this chunk was requested (chunk[0] is its prediction
        for that absolute low-level step). Called once per get_action_chunk
        response, before that chunk's steps_to_execute steps run -- a single prune
        here is sufficient, since no chunk's coverage can expire again before the
        next call (step_count only advances within what's already covered until
        then)."""
        self._chunk_buffer.append((query_step, action_chunk))
        while self._chunk_buffer and (
            self._chunk_buffer[0][0] + self._chunk_buffer[0][1].shape[0] <= query_step
        ):
            self._chunk_buffer.popleft()

    def _ensembled_action(self, step):
        """Combines every buffered chunk's prediction for absolute low-level step
        `step` via exponential recency weighting (exp(-self.ensemble_m * age), age
        = steps since that chunk was queried), rather than blindly executing
        whichever chunk was queried most recently. Returns None if no buffered
        chunk covers `step` (shouldn't happen given _buffer_chunk is always called
        for the chunk about to be played back, but callers must check).

        self.enable_temporal_ensemble=False returns the newest chunk's raw
        prediction unweighted -- exactly this file's pre-ensembling behavior, not
        an approximation of it, since _chunk_buffer is appended in query order and
        the newest chunk's coverage always includes the step it was just queried
        for.

        Position/rotvec/log-stiffness/gripper are all blended as plain vectors
        (rotvec treated as linear, not via multi-rotation averaging/Slerp) -- a
        reasonable approximation given candidates are different chunks' short-
        horizon predictions for what should be nearly the same target, not
        arbitrary rotations to average; revisit if ensembled orientation looks
        wrong in the _log_orientation_debug output on real hardware."""
        candidates, ages = [], []
        for query_step, chunk in self._chunk_buffer:
            if query_step <= step < query_step + chunk.shape[0]:
                candidates.append(chunk[step - query_step])
                ages.append(step - query_step)
        if not candidates:
            return None
        if not self.enable_temporal_ensemble:
            return candidates[-1]
        weights = np.exp(-self.ensemble_m * np.array(ages, dtype=np.float64))
        weights /= weights.sum()
        stacked = np.stack(candidates, axis=0).astype(np.float64)
        return (weights[:, None] * stacked).sum(axis=0).astype(np.float32)

    # --- Action execution ---

    def _log_orientation_debug(self, predicted_rotation):
        """Diagnostic for the constant_orientation position/orientation tradeoff:
        prints the policy's raw predicted orientation (before any action_scale
        blending or constant_orientation override) as base-frame Euler angles, plus
        its rotation *relative to* the hardcoded CONSTANT_ORIENTATION_QUAT expressed
        in the tool's own frame -- so the xyz components read as "twist about the
        wiper's approach axis" / "tilt off flush" directly, rather than an ambiguous
        base-frame mix you'd have to mentally re-project onto the tool."""
        predicted_euler = predicted_rotation.as_euler("xyz", degrees=True)
        fixed_rotation = R.from_quat(CONSTANT_ORIENTATION_QUAT)
        delta_in_tool_frame = predicted_rotation.inv() * fixed_rotation
        delta_euler = delta_in_tool_frame.as_euler("xyz", degrees=True)
        self.get_logger().info(
            "[orientation debug] predicted (base xyz euler, deg): "
            f"[{predicted_euler[0]:6.1f}, {predicted_euler[1]:6.1f}, {predicted_euler[2]:6.1f}]  "
            "delta to CONSTANT_ORIENTATION_QUAT (tool-frame xyz euler, deg): "
            f"[{delta_euler[0]:6.1f}, {delta_euler[1]:6.1f}, {delta_euler[2]:6.1f}]",
            throttle_duration_sec=1.0,
        )

    def _publish_target_pose(self, x_eq):
        target_position = x_eq[0:3]
        target_rotation = R.from_rotvec(x_eq[3:6])
        self._log_orientation_debug(target_rotation)

        if self.orientation_correction_deg:
            # Twist-only correction in the tool's own frame (right-multiply), measured
            # empirically -- see the comment on self.orientation_correction_deg. Composed
            # onto the policy's own prediction rather than replacing it, so the commanded
            # (and therefore the actually-achieved, fed-back) orientation stays close to
            # what the policy predicted instead of jumping to an unrelated fixed pose.
            target_rotation = target_rotation * R.from_euler(
                "z", self.orientation_correction_deg, degrees=True
            )

        # Blend reference for action_scale < 1.0 -- dead in current production usage
        # (main() constructs this class with action_scale=1.0), kept for the cautious-
        # tracking mode documented on self.action_scale. Uses current_pose (the live
        # O_T_EE hand-frame reading) rather than an FK-derived pose: a frame mismatch
        # here only softens how aggressively a fractional step blends toward the
        # target, it doesn't feed the policy or change what gets published at
        # action_scale=1.0, so it's left as the simpler live reading (same pattern
        # replay_policy_on_data_two_color.py's own current_pose usage already accepts
        # for its non-policy-input purposes).
        if self.action_scale < 1.0 and self.current_pose is not None:
            current_position = self.current_pose["position"]
            position = current_position + self.action_scale * (target_position - current_position)
            slerp = Slerp([0.0, 1.0], R.concatenate(
                [self.current_pose["orientation"], target_rotation]
            ))
            rotation = slerp([self.action_scale])[0]
        else:
            position = target_position
            rotation = target_rotation

        # No tool-offset conversion here: x_eq's position is already the tool-tip
        # position the controller should drive to (see get_current_state), and
        # variable_impedance_controllers' CartesianController takes target_pose directly, same as
        # replay_policy_on_data_two_color.py's publish_target -- publish the (rotation-
        # corrected) prediction as-is.
        quat = rotation.as_quat()
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "fr3_link0"
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = (float(v) for v in position)
        msg.pose.orientation.x, msg.pose.orientation.y = float(quat[0]), float(quat[1])
        msg.pose.orientation.z, msg.pose.orientation.w = float(quat[2]), float(quat[3])

        # DO NOT CHANGE THIS!!!
        if self.constant_orientation:
            msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w =0.92388,-0.38268,0.0, 0.0
        self.pose_pub.publish(msg)

    def _step_jump_ok(self, x_eq_pred) -> bool:
        """Non-skippable per-step safety gate, ported from
        replay_policy_on_data_two_color.py's _step_jump_ok: refuses a predicted target
        this far from the arm's live pose instead of executing it. Checked against the
        policy's raw prediction (before action_scale blending/orientation_correction_deg
        reshape it), since those exist to make a trusted prediction track more
        cautiously/accurately -- not to launder an untrusted one past this gate. Compared
        against self.flange_pose+TOOL_OFFSET (same convention as get_current_state/x_eq),
        NOT current_pose/tool_pose -- x_eq_pred is in the flange-frame convention, and
        current_pose's O_T_EE hand frame is ~44deg off from that (see get_current_state),
        which would make this gate's rotation error meaningless. Returns True (nothing to
        check yet) if flange_pose hasn't arrived -- run_interactive already gates the
        caller on that before it ever calls this."""
        if self.flange_pose is None:
            return True
        live_position = self.flange_pose["position"] + self.flange_pose["orientation"].apply(TOOL_OFFSET)
        live_rotation = self.flange_pose["orientation"]
        pos_err = float(np.linalg.norm(live_position - x_eq_pred[0:3]))
        rot_err_deg = float(np.degrees(np.linalg.norm(
            (live_rotation * R.from_rotvec(x_eq_pred[3:6]).inv()).as_rotvec()
        )))
        if pos_err > MAX_STEP_POSITION_JUMP_M:
            self.get_logger().error(
                f"ABORT -- predicted target is {pos_err * 100:.1f}cm / {rot_err_deg:.1f}deg from "
                f"the arm's live pose (limits: {MAX_STEP_POSITION_JUMP_M * 100:.0f}cm / "
                f"{MAX_STEP_ORIENTATION_JUMP_DEG:.0f}deg) -- refusing to execute this prediction."
            )
            return False
        return True

    def _publish_target_stiffness(self, log_k):
        k = np.exp(log_k)
        k[0:3] = np.clip(k[0:3], *K_TRANS_RANGE)
        k[3:6] = np.clip(k[3:6], *K_ROT_RANGE)
        msg = Float64MultiArray()
        msg.data = [float(v) for v in k]
        self.stiffness_pub.publish(msg)

    def _send_gripper_command(self, width):
        width = float(np.clip(width, *GRIPPER_WIDTH_RANGE))
        if self._gripper_goal_in_flight:
            return
        if self._last_gripper_width is not None and abs(width - self._last_gripper_width) < 0.005:
            return
        if not self._gripper_client.server_is_ready():
            return
        goal = GripperCommand.Goal()
        goal.command.position = width
        goal.command.max_effort = 20.0
        self._gripper_goal_in_flight = True
        self._last_gripper_width = width

        def _done_cb(_future):
            self._gripper_goal_in_flight = False

        future = self._gripper_client.send_goal_async(goal)
        future.add_done_callback(_done_cb)

    def switch_controller(self, activate: list, deactivate: list) -> bool:
        if not self.switch_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("switch_controller service not available")
            return False
        req = SwitchController.Request()
        req.activate_controllers = activate
        req.deactivate_controllers = deactivate
        # BEST_EFFORT, not STRICT: this is called from an interactive tool that doesn't
        # track which controller was left active by a prior run/manual test, so the
        # requested activate/deactivate set may already partially hold (e.g. a previous
        # session exited with variable_impedance_controller still active) -- STRICT
        # rejects the whole request in that case, BEST_EFFORT applies what's still needed.
        req.strictness = SwitchController.Request.BEST_EFFORT
        req.activate_asap = True
        req.timeout = rclpy.duration.Duration(seconds=5.0).to_msg()
        future = self.switch_client.call_async(req)
        # Not rclpy.spin_until_future_complete(self, future, ...): that helper adds this
        # node to a (temporary, default) executor and removes it again once the future
        # resolves -- removing it would deregister every subscription callback from the
        # persistent background executor in __init__. The background thread is already
        # spinning this node continuously, so the future's response callback gets serviced
        # there; just wait for it.
        deadline = time.time() + 10.0
        while not future.done() and time.time() < deadline:
            time.sleep(0.01)
        result = future.result()
        if result is None:
            self.get_logger().error("switch_controller call did not return a response")
            return False
        if not result.ok:
            self.get_logger().error("switch_controller rejected the requested switch")
        return result.ok

    # --- Live preview / keyboard control ---

    def _render_preview(self, status_text: str, hint_color=(0, 255, 0)):
        """Shows whatever scene/wrist frames are currently available, overlaid with
        the current mode and key bindings, and returns the key pressed (or -1 for
        none) so callers can poll for 's'/'m'/'q' without a separate imshow loop."""
        scene_frame = self.get_scene_frame()
        wrist_frame = self.get_wrist_frame()
        blank = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        scene_disp = self._resize(scene_frame) if scene_frame is not None else blank
        wrist_disp = self._resize(wrist_frame) if wrist_frame is not None else blank

        # RGB internally (what the policy expects) -- imshow wants BGR.
        both_images = np.concatenate([scene_disp, wrist_disp], axis=1)
        frame_bgr = cv2.cvtColor(both_images, cv2.COLOR_RGB2BGR)
        cv2.putText(
            frame_bgr, status_text, (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, hint_color, 2,
        )
        cv2.putText(
            frame_bgr, "[s] start VLA   [m] stop + home   [q] quit", (10, frame_bgr.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
        )
        cv2.imshow("Scene+Wrist Cameras", frame_bgr)
        return cv2.waitKey(1) & 0xFF

    def _prompt_episode_instruction(self):
        """Blocking Tkinter dialog, shown before each episode, to pick the mark
        colour and wipe manner from the fixed training vocabulary (see
        INSTRUCTION_COLORS/INSTRUCTION_MANNERS), plus the physical mark colour
        (SCORE_COLORS) score_wipe.py should segment when scoring this run's
        before/after photos -- a separate field from "Mark colour" above, which
        names *which* mark (left/right) to wipe, not what colour it is. Returns
        (instruction, score_color), or None if the operator cancelled/closed the
        dialog (in which case the caller should stay idle). Runs on the main thread
        and blocks it -- fine here since it's a deliberate pause between episodes,
        not during one."""
        result = {"instruction": None, "score_color": None}
        root = tk.Tk()
        root.title("Episode instruction")
        root.attributes("-topmost", True)

        frame = ttk.Frame(root, padding=12)
        frame.grid()

        ttk.Label(frame, text="Mark colour:").grid(column=0, row=0, sticky="w", pady=4)
        color_var = tk.StringVar(value=self._last_color)
        color_box = ttk.Combobox(
            frame, textvariable=color_var, values=INSTRUCTION_COLORS, state="readonly"
        )
        color_box.grid(column=1, row=0, padx=8)

        ttk.Label(frame, text="Manner:").grid(column=0, row=1, sticky="w", pady=4)
        manner_var = tk.StringVar(value=self._last_manner)
        manner_box = ttk.Combobox(
            frame, textvariable=manner_var, values=INSTRUCTION_MANNERS, state="readonly"
        )
        manner_box.grid(column=1, row=1, padx=8)

        ttk.Label(frame, text="Score colour (blue/red):").grid(column=0, row=2, sticky="w", pady=4)
        score_color_var = tk.StringVar(value=self._last_score_color)
        score_color_box = ttk.Combobox(
            frame, textvariable=score_color_var, values=SCORE_COLORS, state="readonly"
        )
        score_color_box.grid(column=1, row=2, padx=8)

        def _confirm():
            result["instruction"] = f"wipe the {color_var.get()} mark {manner_var.get()}"
            result["score_color"] = score_color_var.get()
            self._last_color = color_var.get()
            self._last_manner = manner_var.get()
            self._last_score_color = score_color_var.get()
            root.destroy()

        def _cancel():
            root.destroy()

        button_frame = ttk.Frame(frame)
        button_frame.grid(column=0, row=3, columnspan=2, pady=(8, 0))
        ttk.Button(button_frame, text="Start episode", command=_confirm).grid(column=0, row=0, padx=4)
        ttk.Button(button_frame, text="Cancel", command=_cancel).grid(column=1, row=0, padx=4)
        root.protocol("WM_DELETE_WINDOW", _cancel)
        root.bind("<Return>", lambda _event: _confirm())
        root.bind("<Escape>", lambda _event: _cancel())

        root.mainloop()
        if result["instruction"] is None:
            return None
        return result["instruction"], result["score_color"]

    def _start_running(self) -> bool:
        """Switches the follower onto the variable impedance controller and resets
        step bookkeeping. Returns False (staying idle) if the switch fails."""
        # Non-skippable start-of-run gate, ported from replay_policy_on_data_two_color.py's
        # preflight: refuses to start if another node is already publishing target_pose,
        # since running alongside it would race for control of the same controller input.
        publishers = self.get_publishers_info_by_topic(f"/{self.follower_ns}/target_pose")
        other_publishers = [p for p in publishers if p.node_name != self.get_name()]
        if other_publishers:
            self.get_logger().error(
                f"start FAILED -- {len(other_publishers)} other node(s) already publishing "
                f"target_pose ({[p.node_name for p in other_publishers]}); running alongside "
                "them would race. Stop whatever else is commanding this arm first."
            )
            return False

        self.get_logger().info("Switching to variable impedance controller...")
        if not self.switch_controller(
            activate=[VARIABLE_IMPEDANCE_CONTROLLER], deactivate=[MOVE_TO_START_CONTROLLER]
        ):
            self.get_logger().error("Controller switch failed -- staying idle.")
            return False
        self.step_count = 0

        # Publish the fixed/non-compliant impedance immediately on activation -- see
        # NON_COMPLIANT_STIFFNESS's comment for why this can't just be left to the
        # controller's own static baseline param.
        self._publish_target_stiffness(np.log(NON_COMPLIANT_STIFFNESS))

        # Log current_pose (same topic/frame as x_f, i.e. base-frame PoseStamped) for
        # exactly this running window, so its Δx/Δy can be checked against
        # REFERENCE_WIPE_DELTA_XY once the wipe finishes -- see _stop_and_home.
        os.makedirs(POSE_LOG_DIR, exist_ok=True)
        log_path = os.path.join(
            POSE_LOG_DIR, f"pose_log_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        )
        with self._pose_log_lock:
            self._pose_log_file = open(log_path, "w", newline="")
            self._pose_log_writer = csv.writer(self._pose_log_file)
            self._pose_log_writer.writerow(["t", "x", "y", "z", "qx", "qy", "qz", "qw", "force_mag"])
            self._pose_log_samples = []
        self._wrench_callback_count_at_start = self._wrench_callback_count

        # Wipe-scoring: grab the "before" photo now, right before any action
        # publishing begins, so it reflects the mark's un-wiped state -- see
        # _stop_and_home for the matching "after" capture + scoring. ROI-cropped
        # (self.wipe_score_roi) so only the wipe board -- not the robot/background
        # -- ever gets written to disk, same as score_wipe.py's own --roi.
        self._score_before_crop = None
        if self.score_color is not None:
            before_frame = self.get_scene_frame()
            if before_frame is None:
                self.get_logger().warn(
                    "wipe-scoring: no scene frame available to capture the 'before' "
                    "photo -- skipping scoring for this run."
                )
            else:
                self._score_before_crop, _ = score_wipe.apply_roi(before_frame, self.wipe_score_roi)

        self.get_logger().info(f"VLA prediction + action execution started. Logging pose to {log_path}")
        return True

    def _stop_and_home(self):
        """Stops publishing actions and moves the follower back to its home state
        via move_to_start_example_controller -- used for both a manual 'm' press
        and the automatic MAX_STEPS cutoff."""
        self.get_logger().info("Stopping VLA prediction; switching to move_to_start (home)...")

        wrench_msgs_this_run = self._wrench_callback_count - self._wrench_callback_count_at_start
        self.get_logger().info(
            f"[wrench diagnostic] _wrench_callback fired {wrench_msgs_this_run} times during this "
            f"run ({len(self._pose_log_samples)} pose samples logged) -- expect roughly 50x the pose "
            "sample count if the wrench subscription (50Hz) is being serviced normally."
        )

        # Contact-only Δx/Δy: a raw start-of-run-to-stop delta is dominated by the
        # free-space reach/descent onto the board, not the wipe itself (confirmed
        # empirically 2026-09-05 -- see CONTACT_FORCE_THRESHOLD_N). First/last sample
        # with force above threshold approximates the same contact window
        # extract_impedance_labels.py's contact_mask picks out of training episodes.
        contact_positions = [
            pos for (_, pos, force_mag) in self._pose_log_samples
            if force_mag > CONTACT_FORCE_THRESHOLD_N
        ]
        if len(contact_positions) >= 2:
            start_position, end_position = contact_positions[0], contact_positions[-1]
            dx, dy = end_position[0] - start_position[0], end_position[1] - start_position[1]
            ref_dx, _ = REFERENCE_WIPE_DELTA_XY
            # x only: y is bimodal across data_two_color episodes (depends on which mark's
            # position was wiped), so it has no single reference sign to check against --
            # see REFERENCE_WIPE_DELTA_XY's comment.
            signs_match = np.sign(dx) == np.sign(ref_dx)
            self.get_logger().info(
                f"[wipe delta] contact-only (force>{CONTACT_FORCE_THRESHOLD_N:.0f}N, "
                f"{len(contact_positions)} samples) actual Δx={dx:+.3f}m Δy={dy:+.3f}m vs reference "
                f"Δx={ref_dx:+.3f}m -- x sign {'MATCH' if signs_match else 'FLIPPED'}"
            )
        elif self._pose_log_samples:
            self.get_logger().warn(
                f"[wipe delta] never saw force>{CONTACT_FORCE_THRESHOLD_N:.0f}N during this run -- "
                "no contact detected, skipping Δx/Δy check."
            )
        with self._pose_log_lock:
            if self._pose_log_file is not None:
                self._pose_log_file.close()
                self._pose_log_file = None
                self._pose_log_writer = None
            self._pose_log_samples = []

        home_switch_ok = self.switch_controller(
            activate=[MOVE_TO_START_CONTROLLER], deactivate=[VARIABLE_IMPEDANCE_CONTROLLER]
        )
        if not home_switch_ok:
            self.get_logger().error(
                "Controller switch to move_to_start failed -- verify the follower's state manually."
            )

        # Wipe-scoring: capture the "after" photo once the arm has actually stopped
        # moving (the switch_controller call above only confirms the controller
        # swap, not that the resulting home motion has finished -- see
        # _wait_until_home_settled), then score it against the "before" photo
        # captured in _start_running. Skipped entirely if no score colour was picked
        # (e.g. run_scripted_rollout.py, which never sets self.score_color) or the
        # "before" capture itself failed.
        self.last_wipe_score = None
        if self.score_color is not None and self._score_before_crop is not None:
            if home_switch_ok:
                self._wait_until_home_settled()
            after_frame = self.get_scene_frame()
            if after_frame is None:
                self.get_logger().warn(
                    "wipe-scoring: no scene frame available to capture the 'after' "
                    "photo -- skipping scoring for this run."
                )
            else:
                after_crop, _ = score_wipe.apply_roi(after_frame, self.wipe_score_roi)
                ranges = score_wipe.hsv_ranges_for(self.score_color, None, None)
                try:
                    percent_wiped, before_area, after_area, out_path = score_wipe.score_and_visualize(
                        self._score_before_crop, after_crop, ranges, roi=None,
                        out_dir=WIPE_SCORE_DIR, show=False, min_blob_px=50,
                    )
                    self.get_logger().info(
                        f"[wipe score] {self.score_color} mark: {percent_wiped:.1f}% wiped "
                        f"(before {before_area}px, after {after_area}px) -- saved to {out_path}"
                    )
                    self.last_wipe_score = {
                        "score_color": self.score_color,
                        "percent_wiped": float(percent_wiped),
                        "before_area_px": int(before_area),
                        "after_area_px": int(after_area),
                        "out_path": str(out_path),
                    }
                except Exception as e:
                    self.get_logger().error(f"wipe-scoring failed: {e}")
        self._score_before_crop = None

        self.step_count = 0
        self._last_gripper_width = None

    def _wait_until_home_settled(self):
        """Blocks for HOME_SETTLE_WAIT_SEC (spinning this node so subscriptions keep
        being serviced) before the wipe-scoring "after" photo is captured -- called
        from _stop_and_home right after the move_to_start controller switch, which
        only confirms the controller swap, not that the resulting home motion has
        actually finished.

        Also keeps draining the scene camera via get_scene_frame() every tick of
        this wait, not just spinning ROS callbacks: cv2.VideoCapture/V4L2 queues
        captured frames internally, and if nothing calls .read() for the whole 5s
        wait, the single get_scene_frame() call made right after this returns
        (in _stop_and_home) just dequeues whatever frame was sitting oldest in that
        backlog -- i.e. one from around when the wait *started*, not a current one
        -- which is exactly the stale/pre-move 'after' photo bug this fixes."""
        deadline = time.time() + HOME_SETTLE_WAIT_SEC
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.enable_scene_camera:
                self.get_scene_frame()

    def run_interactive(self):
        # Warmup is specifically for the scene camera's auto-exposure/white-balance
        # settling (see CAMERA_WARMUP_SEC) -- skip the wait entirely when it's disabled.
        if self.enable_scene_camera:
            self.get_logger().info(
                f"Warming up scene camera for {self.camera_warmup_sec:.0f}s -- feed is shown below."
            )
            warmup_start = time.time()
            while rclpy.ok() and time.time() - warmup_start < self.camera_warmup_sec:
                # No rclpy.spin_once here -- the background executor thread (see __init__)
                # services all callbacks continuously now; this loop only needs to pace itself.
                time.sleep(0.01)
                remaining = self.camera_warmup_sec - (time.time() - warmup_start)
                self._render_preview(f"WARMING UP CAMERA ({remaining:0.1f}s)", hint_color=(0, 165, 255))

        self.get_logger().info(
            "Ready. Press 's' to pick colour/manner and start, 'm' to stop/home, 'q' to quit."
        )
        step_period = 1.0 / self.chunk_hz
        running = False

        try:
            while rclpy.ok():
                # No rclpy.spin_once here -- see the warmup loop above / __init__'s
                # background executor thread.
                time.sleep(0.01)

                status_text = f"RUNNING: {self.current_instruction}" if running else "IDLE"
                key = self._render_preview(
                    status_text,
                    hint_color=(0, 0, 255) if running else (0, 255, 0),
                )
                if key == ord('q'):
                    if running:
                        self._stop_and_home()
                    break
                elif key == ord('s') and not running:
                    prompt_result = self._prompt_episode_instruction()
                    if prompt_result is None:
                        continue  # operator cancelled the dialog -- stay idle
                    instruction, score_color = prompt_result
                    self.current_instruction = instruction
                    self.score_color = score_color
                    running = self._start_running()
                    continue
                elif key == ord('m'):
                    if running:
                        self._stop_and_home()
                    running = False
                    continue

                if not running:
                    continue

                scene_frame = self.get_scene_frame()
                wrist_frame = self.get_wrist_frame()
                current_state = self.get_current_state()
                force_history = self.get_force_history() if self.include_force_history else None

                if (
                    scene_frame is None
                    or wrist_frame is None
                    or current_state is None
                    or (self.include_force_history and force_history is None)
                ):
                    self.get_logger().warn(
                        "Waiting for scene/wrist cameras, proprioception and wrench streams...",
                        throttle_duration_sec=2.0,
                    )
                    continue

                scene_rgb = self._resize(scene_frame)
                wrist_rgb = self._resize(wrist_frame)
                # cv2.imshow('scene', cv2.cvtColor(scene_rgb, cv2.COLOR_RGB2BGR))
                # cv2.waitKey(1)  # allow imshow to update
                action_chunk = self.get_action_chunk(
                    scene_rgb, wrist_rgb, current_state, self.current_instruction, force_history
                )
                if action_chunk is None:
                    # get_action_chunk already logged why (timeout or other request
                    # failure). Don't just retry next tick and keep holding the last
                    # published target indefinitely on an untrusted/unresponsive server --
                    # same "refuse rather than trust blindly" posture as _step_jump_ok.
                    self.get_logger().error("Stopping run -- action-chunk request failed.")
                    self._stop_and_home()
                    running = False
                    continue

                action_dim = action_chunk.shape[-1]
                is_compliance_checkpoint = action_dim == 13
                if action_dim not in (7, 13):
                    self.get_logger().error(
                        f"Unexpected action_dim={action_dim} (expected 7 for b0/b2 or "
                        "13 for b5); skipping chunk."
                    )
                    continue

                # Buffer this response for temporal ensembling (see
                # ENABLE_TEMPORAL_ENSEMBLE_DEFAULT's module comment) *before* playing
                # any of it back, so the very first step below already has every
                # still-relevant chunk (this one plus any older ones whose coverage
                # hasn't expired yet) available to blend.
                self._buffer_chunk(self.step_count, action_chunk)

                for _ in range(min(self.steps_to_execute, action_chunk.shape[0])):
                    start_time = time.time()

                    # Poll for a mid-chunk 'm'/'q' so a stop request lands within one
                    # low-level step instead of waiting for the whole chunk to finish.
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('m') or key == ord('q'):
                        self._stop_and_home()
                        running = False
                        break
                    if self.flange_pose is None:
                        # Gates _step_jump_ok's fail-open "nothing to check yet" branch --
                        # without this, a not-yet-populated flange_pose would let that
                        # check silently pass instead of actually comparing against a
                        # live pose. Reaching here with flange_pose still None shouldn't
                        # happen in practice (get_current_state, checked before this loop
                        # starts, already requires it), but keep the gate explicit.
                        break

                    step = self._ensembled_action(self.step_count)
                    if step is None:
                        # Shouldn't happen -- the chunk just buffered above always
                        # covers self.step_count -- but fail closed rather than publish
                        # a stale or garbage target if this invariant is ever violated.
                        self.get_logger().error(
                            "No buffered chunk covers the current step -- stopping."
                        )
                        self._stop_and_home()
                        running = False
                        break
                    x_eq = step[0:6]
                    if not self._step_jump_ok(x_eq):
                        self._stop_and_home()
                        running = False
                        break
                    if is_compliance_checkpoint:
                        log_k, gripper_cmd = step[6:12], step[12]
                        self._publish_target_stiffness(log_k)
                    else:
                        gripper_cmd = step[6]

                    self._publish_target_pose(x_eq)
                    self._send_gripper_command(gripper_cmd)

                    self.step_count += 1
                    print(f"steps so far: {self.step_count}")

                    if self.step_count >= self.max_steps:
                        break

                    elapsed = time.time() - start_time
                    time.sleep(max(0.0, step_period - elapsed))

                if running and self.step_count >= self.max_steps:
                    self.get_logger().info(f"Reached {self.max_steps} steps -- stopping actions and moving home.")
                    self._stop_and_home()
                    running = False

                if key == ord('q'):
                    break

        except KeyboardInterrupt:
            self.get_logger().info("Trial interrupted by user.")
            if running:
                self._stop_and_home()
        finally:
            if self.enable_scene_camera:
                self._stop_scene_capture.set()
                self._scene_capture_thread.join(timeout=2.0)
                self.scene_capture.release()
            if self.enable_wrist_camera and self.wrist_capture is not None:
                self.wrist_capture.release()
            cv2.destroyAllWindows()
            self.shutdown_executor()


def main():
    rclpy.init()
    # Instruction (format: "wipe the <colour> mark <manner>", compliance-vla
    # -icra2027-proposalv2.md §5) is now picked per-episode from a GUI dialog --
    # see _prompt_episode_instruction, triggered by 's' in run_interactive.
    node = SmolVLADeployment(action_scale=1.0)  # 0.3 = cautious tracking of predicted x_eq
    node.run_interactive()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
