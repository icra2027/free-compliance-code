"""Open-loop debug replay: feeds one real, trimmed data_two_color episode's actual
recorded observations (scene/wrist images, state, task string, force history) to the
policy server one anchor at a time -- exactly generate_trajectory_from_policy.py's
teacher-forced query pattern -- but instead of only logging the prediction, actually
publishes it to variable_impedance_controllers' target_pose/target_stiffness topics and drives the
real Franka follower with it.

No live camera/proprioception/wrench feed is ever read into the policy: every
observation sent to the server comes straight from the recorded demo, so this is not
closed-loop control and nothing compounds -- it answers "what does the real arm do if
you actually execute this checkpoint's predictions for a known-good demo," decoupled
from whatever cameras/topics a live rollout (deploy_smolvla.py) would need. current_pose
is still subscribed to, but only for the preflight gate and the per-step safety clamp
below, never as policy input.

Trims the episode with the same _trim_to_last_contact logic src/compliance_vla/policy/dataset.py applies
(copied, not imported, same reason as every sibling script here: keep this real-hardware
-facing script torch-free). See that module's docstring for why a naive contact-force
threshold is not enough to drop data_two_color's appended return-to-start_joint_
configuration tail.

action_chunk decoding matches deploy_smolvla.py exactly: action_dim 7 = [x_eq(6),
gripper(1)] (b0/b2, fixed stiffness), action_dim 13 = [x_eq(6), log_k(6), gripper(1)]
(b5, compliance output) -- only the chunk's first low-level step is used per anchor,
then the loop advances to the next (stride-selected) demo frame.

Two safety nets beyond the shared preflight gate (start-pose check, no other
target_pose publisher):
  - a hard, non-skippable per-step clamp: if a predicted target jumps further from the
    arm's *live* current pose than --max-jump-m / --max-jump-deg, the run aborts and
    switches home rather than executing it. This is the one place live current_pose
    feeds back into control flow (never into the policy's input) -- it exists because,
    unlike replay_data_two_color.py's recorded (and therefore already smooth) x_eq
    sequence, a policy's raw predictions are untested and a bad one two anchors apart
    (especially with --stride > 1) should not be blindly trusted on real hardware.
  - --out-csv optionally records label vs predicted vs actually-published values for
    offline comparison, same shape as generate_trajectory_from_policy.py's.

A cv2 preview window shows the demo's own scene/wrist frames (not a live camera) each
step, so the operator can see what the policy is actually looking at as its predicted
action executes; --no-display turns it off (e.g. no X display on the control machine).

Publishes the same measured orientation twist correction deploy_smolvla.py applies
(--orientation-correction-deg, default 39.5, composed onto the raw predicted
orientation only at publish time -- see ORIENTATION_CORRECTION_DEG), so a published
EE orientation here matches what the live deployment would actually command.

Usage (run on the machine actually controlling the follower, with serve_policy.py's
/act endpoint already up -- see deploy_vla_on_franka.md):
    python3 replay_policy_on_data_two_color.py --session demo_blue_firm --episode 11

    # See what's in a session first:
    python3 replay_policy_on_data_two_color.py --session demo_blue_firm --list-episodes


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

import argparse
import csv
import os
import sys
import threading
import time

import cv2
import numpy as np
import json_numpy
import requests
import rclpy
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray
from control_msgs.action import GripperCommand
from controller_manager_msgs.srv import SwitchController
from scipy.spatial.transform import Rotation as R

# --- sys.path wiring, matching replay_data_two_color.py / generate_trajectory_from_
# policy.py -- these repo-relative paths are specific to how compliance-vla/
# fr3_bilateral_teleop are vendored into *this* workspace (as src/ siblings).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_WS_SRC = os.path.dirname(os.path.dirname(_THIS_DIR))          # franka_ros2_ws/src
_REPO_ROOT = os.path.dirname(_WS_SRC)                          # franka_ros2_ws
_BOOKISH_ROOT = os.path.join(_WS_SRC, "compliance-vla")
_BOOKISH_SCRIPTS = os.path.join(_BOOKISH_ROOT, "data_extraction")
_TELEOP_SCRIPTS = os.path.join(_WS_SRC, "fr3_bilateral_teleop", "dataset_tools", "labeling")
for _p in (_BOOKISH_ROOT, _BOOKISH_SCRIPTS, _TELEOP_SCRIPTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dataset_io as dio  # noqa: E402
from compliance_vla.policy import labels as lb  # noqa: E402

DATA_TWO_COLOR_ROOT = os.path.join(_REPO_ROOT, "data_two_color")
IMAGE_COLUMNS = ["observation.images.scene_rgb", "observation.images.wrist_rgb"]

# Same controller pair deploy_smolvla.py switches between, and the same topics
# variable_impedance_controllers' CartesianController exposes on the follower.
VARIABLE_IMPEDANCE_CONTROLLER = "variable_impedance_controller"
MOVE_TO_START_CONTROLLER = "move_to_start_example_controller"

# Controller-realizable stiffness range (deploy_smolvla.py's own
# constants) -- clip exp(log_k) into this before publishing for 13-dim (b5) chunks.
K_TRANS_RANGE = (50.0, 1500.0)   # N/m
K_ROT_RANGE = (5.0, 100.0)       # N*m/rad

# Used verbatim for 7-dim (b0/b2) chunks, which don't predict stiffness at all --
# same conservative mid-range constant replay_data_two_color.py tracks recorded x_eq
# with.
FIXED_STIFFNESS = np.array([300.0, 300.0, 300.0, 20.0, 20.0, 20.0])

# Franka Hand full-open width; gripper action outputs are treated as an absolute
# target width in meters and clipped into this range (matches deploy_smolvla.py).
GRIPPER_WIDTH_RANGE = (0.0, 0.08)

# Hard (non-skippable) preflight gate: refuse to start if the follower's current pose is
# farther than this from the trimmed episode's first recorded (not predicted) pose --
# an impedance controller snapping at a large step error is exactly the sudden motion
# this guards against, same rationale as replay_data_two_color.py's own preflight.
MAX_START_POSITION_ERROR_M = 0.4
MAX_START_ORIENTATION_ERROR_DEG = 90.0

# Hard (non-skippable) per-step gate: abort mid-run rather than execute a predicted
# target this far from the arm's *live* pose. Looser than the preflight thresholds
# above -- legitimate motion between anchors should trip this rarely -- but it exists
# specifically because, unlike a recorded x_eq sequence, nothing guarantees consecutive
# policy predictions are smooth.
MAX_STEP_POSITION_JUMP_M = 0.3
MAX_STEP_ORIENTATION_JUMP_DEG = 90.0

# Empirically-measured twist correction deploy_smolvla.py composes onto every
# predicted orientation before publishing (see that script's own comment: a
# 2026-09-05 measurement found the raw prediction was consistently off by a pure
# twist about the tool's approach axis, 39-40deg across 14 samples). Applied here
# too, at publish time only -- x_eq_pred as logged/compared and as checked by
# _step_jump_ok below stays the policy's raw output, same separation deploy_smolvla.py
# keeps between _log_orientation_debug/_step_jump_ok (raw) and _publish_target_pose
# (corrected) -- so a published EE orientation from this script matches what the live
# deployment would actually command for the same prediction. Re-measure and update
# if the checkpoint, tool, or its mount changes.
ORIENTATION_CORRECTION_DEG = 39.5

SPEED_PAUSE_THRESHOLD_M_S = 0.01
NEAR_HOME_RADIUS_M = 0.05

FORCE_HISTORY_WINDOW_SEC = 0.5
FORCE_HISTORY_LEN = 20


def _trim_to_last_contact(g, speed_threshold=SPEED_PAUSE_THRESHOLD_M_S,
                           near_home_radius=NEAR_HOME_RADIUS_M, settle_frames=2):
    """Copied verbatim from compliance-vla/src/compliance_vla/policy/dataset.py (not imported --
    that module pulls in torch). Walks backward from episode end for the last point
    that is both slow AND far from the episode's own start pose (i.e. paused out over
    the board, not settling back at rest), to exclude data_two_color's appended
    return-to-start_joint_configuration tail."""
    pos = np.stack(g["observation.leader_pose"].to_numpy())[:, :3]
    t = g["timestamp"].to_numpy().astype(np.float64)
    n = len(pos)
    if n < 10:
        return g
    start_pos = pos[0]
    dist_from_start = np.linalg.norm(pos - start_pos, axis=1)
    dt = np.diff(t)
    with np.errstate(invalid="ignore", divide="ignore"):
        speed = np.linalg.norm(np.diff(pos, axis=0), axis=1) / dt
    k = n - 2
    while k >= 0 and not (speed[k] < speed_threshold and dist_from_start[k] > near_home_radius):
        k -= 1
    if k < 0:
        return g
    cutoff = min(k + 1 + settle_frames, n)
    return g.iloc[:cutoff]


def _decode_array3d(cell):
    """Copied from compliance-vla/src/compliance_vla/policy/dataset.py: the dataset's
    Array3DExtensionType image cells come back from pandas/pyarrow as a doubly-nested
    object ndarray; recursing one level explicitly resolves it."""
    return np.stack([np.stack(row) for row in cell]).astype(np.uint8)


def resample_to_n_samples(times, values, t_end, window_sec, n_samples):
    """Copied from compliance-vla/src/compliance_vla/policy/force_encoder.py. Linearly resamples
    `values` (T, C) onto n_samples evenly spaced points covering [t_end - window_sec,
    t_end], holding the earliest available sample constant to fill any gap -- the exact
    history shape the model was trained against at every anchor."""
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    t0 = t_end - window_sec
    query = np.linspace(t0, t_end, n_samples)

    in_window = times <= t_end
    if in_window.sum() == 0:
        return np.repeat(values[:1], n_samples, axis=0).astype(np.float32)
    t_hist, v_hist = times[in_window], values[in_window]
    if len(t_hist) == 1:
        return np.repeat(v_hist, n_samples, axis=0).astype(np.float32)

    out = np.empty((n_samples, values.shape[1]), dtype=np.float64)
    for c in range(values.shape[1]):
        out[:, c] = np.interp(query, t_hist, v_hist[:, c], left=v_hist[0, c], right=v_hist[-1, c])
    return out.astype(np.float32)


def _resize(img, size):
    if img.shape[0] == size and img.shape[1] == size:
        return img
    return cv2.resize(img, (size, size))


DISPLAY_WINDOW_NAME = "data_two_color demo playback (scene | wrist)"


def _show_demo_frame(scene_rgb, wrist_rgb, status_text):
    """Shows the demo's own scene/wrist frames -- exactly what's being sent to the
    policy this step, not a live camera -- side by side, so the operator can watch
    what the model is "seeing" while its inferred action executes on the real arm.
    Returns the key pressed (or -1 for none)."""
    scene_bgr = cv2.cvtColor(scene_rgb, cv2.COLOR_RGB2BGR)
    wrist_bgr = cv2.cvtColor(wrist_rgb, cv2.COLOR_RGB2BGR)
    combined = np.concatenate([scene_bgr, wrist_bgr], axis=1)
    cv2.putText(combined, status_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(combined, "[q] abort", (10, combined.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.imshow(DISPLAY_WINDOW_NAME, combined)
    return cv2.waitKey(1) & 0xFF


def list_episodes(session):
    frames = dio.load_frames(session, dataset_root=DATA_TWO_COLOR_ROOT)
    meta = dio.load_episodes_meta(session, dataset_root=DATA_TWO_COLOR_ROOT)
    task_lookup = dict(zip(meta["episode_index"], meta["tasks"].apply(lambda t: t[0] if len(t) else "")))
    for ep_idx, g in frames.groupby("episode_index"):
        print(f"  episode {ep_idx:>3}: {len(g):>4} frames  task={task_lookup.get(ep_idx, '')!r}")


def _load_single_episode_frame(session, episode_index):
    """Reads only the one parquet file that holds this episode -- see
    generate_trajectory_from_policy.py's identical helper for why this matters (loading
    every episode's image columns to read one OOM-killed that call during testing)."""
    import pyarrow.parquet as pq

    meta = dio.load_episodes_meta(session, dataset_root=DATA_TWO_COLOR_ROOT)
    row = meta[meta["episode_index"] == episode_index]
    if row.empty:
        raise ValueError(f"episode {episode_index} not found in session {session!r} "
                          f"(available: {sorted(meta['episode_index'])})")
    chunk_idx = int(row.iloc[0]["data/chunk_index"])
    file_idx = int(row.iloc[0]["data/file_index"])
    path = os.path.join(dio.session_dir(session, DATA_TWO_COLOR_ROOT), "data",
                         f"chunk-{chunk_idx:03d}", f"file-{file_idx:03d}.parquet")
    table = pq.read_table(path, columns=dio.NON_IMAGE_COLUMNS + IMAGE_COLUMNS)
    df = table.to_pandas()
    return df[df["episode_index"] == episode_index]


def _compute_state_arrays_without_contact_fit(ep_frames, tool_offset):
    """Same t/frame_index/q/qdot/x_f/x_eq/wrench derivation as
    compliance_vla.policy.labels.compute_episode_arrays -- see that module's docstring: x_eq is just
    observation.leader_pose directly and x_f is plain FK + tool_offset, neither needs a
    contact-frame fit -- but skips extract_demo_30hz's windowed log_k/mask regression
    entirely (the part of compute_episode_arrays that can fail and return None).

    Used as a fallback when compute_episode_arrays returns None: this script never reads
    arrays["log_k"]/["mask"]/["nearest_idx"] (the 13-dim/b5 log_k it publishes always
    comes from the policy's own prediction, never the label), so a contact-frame fit
    failure on a given episode -- e.g. too little/no clean contact segment, unrelated to
    whether its position trajectory is fine -- should not block replaying it."""
    import panda_fk as fk

    state = dio.stack_col(ep_frames, "observation.state")
    velocity = dio.stack_col(ep_frames, "observation.velocity")
    leader_pose = dio.stack_col(ep_frames, "observation.leader_pose")
    wrench = dio.stack_col(ep_frames, "observation.wrench.external_base")
    timestamp = ep_frames["timestamp"].to_numpy()
    frame_index = ep_frames["frame_index"].to_numpy()

    follower_pos, follower_rot = fk.fk_batch(state)
    follower_pos = follower_pos + np.einsum("nij,j->ni", follower_rot, tool_offset)
    follower_rotvec = np.stack(
        [fk.rotvec_from_matrix(follower_rot[i]) for i in range(len(follower_rot))], axis=0
    )
    x_f = np.concatenate([follower_pos, follower_rotvec], axis=1)

    return {
        "t": timestamp,
        "frame_index": frame_index,
        "q": state,
        "qdot": velocity,
        "x_f": x_f,
        "x_eq": leader_pose,
        "wrench": wrench,
    }


def load_trimmed_episode_with_images(session, episode_index, use_tool_offset=True):
    """Returns (arrays, image_lookup, task) for one trimmed episode. arrays has at least
    t, q, qdot, x_f, x_eq, wrench, frame_index (compliance_vla.policy.labels.compute_episode_arrays'
    return dict when its contact-frame fit succeeds, else
    _compute_state_arrays_without_contact_fit's smaller dict -- see that function for
    why this script doesn't care which one it got); image_lookup maps raw frame_index ->
    (scene_rgb, wrist_rgb) uint8 arrays for every frame still present after trimming."""
    g = _load_single_episode_frame(session, episode_index).sort_values("frame_index")
    n_before = len(g)
    g = _trim_to_last_contact(g)
    print(f"[replay] session={session} episode={episode_index}: "
          f"trimmed {n_before} -> {len(g)} frames ({n_before - len(g)} dropped)")

    image_lookup = {
        int(fi): (_decode_array3d(scene), _decode_array3d(wrist))
        for fi, scene, wrist in zip(g["frame_index"], g["observation.images.scene_rgb"],
                                     g["observation.images.wrist_rgb"])
    }

    # tool_offset shifts x_f's FK-derived position by the tool's calibrated offset from
    # the flange (data_extraction/calibrate_tool_offset.py); --no-tool-offset skips loading it
    # (np.zeros(3) instead) for a machine/checkout where tool_offset.npy hasn't been
    # calibrated yet. NOTE this is not just cosmetic: x_f feeds straight into `state`
    # below, which IS sent to the policy, so running without the same tool_offset the
    # checkpoint was trained with means feeding it an out-of-distribution flange pose
    # instead of the tool-tip pose it expects -- use this flag to unblock a quick replay
    # on an uncalibrated machine, not as a default for judging a checkpoint's accuracy.
    # It also shifts the preflight/per-step safety clamps' reference pose by the same
    # amount, since those compare against this same x_f.
    # tool_offset = lb.load_tool_offset() if use_tool_offset else np.zeros(3)
    tool_offset = np.array([0.0, 0.0, 0.2]) if use_tool_offset else np.zeros(3)
    print(f"tool offset: {tool_offset} (use_tool_offset={use_tool_offset})")
    args = lb.default_extraction_args()
    sigma_f = lb.default_sigma_f()
    arrays = lb.compute_episode_arrays(g, tool_offset, args, sigma_f)
    if arrays is None:
        print(f"[replay] {session}#{episode_index}: contact-frame extraction failed after "
              "trimming -- falling back to FK-only state arrays (this script never uses "
              "the label log_k/mask that extraction would have produced).")
        arrays = _compute_state_arrays_without_contact_fit(g, tool_offset)

    meta = dio.load_episodes_meta(session, dataset_root=DATA_TWO_COLOR_ROOT)
    task_lookup = dict(zip(meta["episode_index"], meta["tasks"].apply(lambda t: t[0] if len(t) else "")))
    return arrays, image_lookup, task_lookup.get(episode_index, "")


def get_action_chunk(server_url, scene_rgb, wrist_rgb, state, task, force_history=None):
    payload = {"task": task, "scene_rgb": scene_rgb, "wrist_rgb": wrist_rgb, "state": state}
    if force_history is not None:
        payload["force_history"] = force_history
    resp = requests.post(server_url, data=json_numpy.dumps(payload),
                          headers={"Content-Type": "application/json"})
    resp.raise_for_status()
    return json_numpy.loads(resp.content)["action_chunk"]


def pose_to_xyz_rotvec(msg: PoseStamped) -> np.ndarray:
    pos = msg.pose.position
    ori = msg.pose.orientation
    rotvec = R.from_quat([ori.x, ori.y, ori.z, ori.w]).as_rotvec()
    return np.concatenate([[pos.x, pos.y, pos.z], rotvec])


class PolicyOnDemoReplayer(Node):
    """Publishes to the same variable_impedance_controllers CartesianController topics deploy_smolvla.py
    and replay_data_two_color.py use. current_pose is only ever read here for the
    preflight gate and the per-step safety clamp -- never sent to the policy."""

    def __init__(self, follower_ns: str = "follower",
                 orientation_correction_deg: float = ORIENTATION_CORRECTION_DEG):
        super().__init__("data_two_color_policy_replayer")
        self.follower_ns = follower_ns
        self.orientation_correction_deg = orientation_correction_deg
        self.current_pose = None
        self.create_subscription(
            PoseStamped,
            f"/{follower_ns}/franka_robot_state_broadcaster/current_pose",
            self._pose_callback,
            10,
        )
        self.pose_pub = self.create_publisher(PoseStamped, f"/{follower_ns}/target_pose", 10)
        self.stiffness_pub = self.create_publisher(
            Float64MultiArray, f"/{follower_ns}/target_stiffness", 10
        )
        self._gripper_client = ActionClient(
            self, GripperCommand, f"/{follower_ns}/franka_gripper/gripper_action"
        )
        self._last_gripper_width = None
        self._gripper_goal_in_flight = False
        self.switch_client = self.create_client(
            SwitchController, f"/{follower_ns}/controller_manager/switch_controller"
        )

        # Continuous background spin -- same fix deploy_smolvla.py / replay_data_two_
        # color.py apply: a blocking service call or requests.post with no spinning in
        # between would leave current_pose stale.
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self)
        self._executor_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._executor_thread.start()

    def _pose_callback(self, msg):
        self.current_pose = pose_to_xyz_rotvec(msg)

    def shutdown_executor(self):
        self._executor.shutdown()
        self._executor_thread.join(timeout=2.0)

    def switch_controller(self, activate: list, deactivate: list) -> bool:
        if not self.switch_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("switch_controller service not available")
            return False
        req = SwitchController.Request()
        req.activate_controllers = activate
        req.deactivate_controllers = deactivate
        req.strictness = SwitchController.Request.BEST_EFFORT
        req.activate_asap = True
        req.timeout = rclpy.duration.Duration(seconds=5.0).to_msg()
        future = self.switch_client.call_async(req)
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

    def publish_target(self, x_eq: np.ndarray):
        rotation = R.from_rotvec(x_eq[3:6])
        if self.orientation_correction_deg:
            # Twist-only correction in the tool's own frame (right-multiply), same as
            # deploy_smolvla.py's _publish_target_pose -- composed onto the prediction
            # rather than replacing it, so the commanded orientation stays close to what
            # the policy actually predicted.
            rotation = rotation * R.from_euler("z", self.orientation_correction_deg, degrees=True)
        quat = rotation.as_quat()
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "fr3_link0"
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = (float(v) for v in x_eq[0:3])
        msg.pose.orientation.x, msg.pose.orientation.y = float(quat[0]), float(quat[1])
        msg.pose.orientation.z, msg.pose.orientation.w = float(quat[2]), float(quat[3])
        self.pose_pub.publish(msg)

    def publish_stiffness(self, k: np.ndarray):
        msg = Float64MultiArray()
        msg.data = [float(v) for v in k]
        self.stiffness_pub.publish(msg)

    def send_gripper_command(self, width):
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


def preflight(node: PolicyOnDemoReplayer, first_target: np.ndarray) -> bool:
    """Hard gate, not skippable by any flag: refuses to start if the follower isn't
    already close to the trimmed episode's first *recorded* pose (not a prediction)."""
    deadline = time.time() + 5.0
    while node.current_pose is None and time.time() < deadline:
        time.sleep(0.05)
    if node.current_pose is None:
        node.get_logger().error("preflight: never received a /current_pose reading -- aborting.")
        return False

    pos_err = float(np.linalg.norm(node.current_pose[0:3] - first_target[0:3]))
    rot_err_deg = float(np.degrees(np.linalg.norm(
        (R.from_rotvec(node.current_pose[3:6]) * R.from_rotvec(first_target[3:6]).inv()).as_rotvec()
    )))
    node.get_logger().info(
        f"preflight: current pose is {pos_err * 100:.1f}cm / {rot_err_deg:.1f}deg from the "
        f"episode's first recorded frame (limits: {MAX_START_POSITION_ERROR_M * 100:.0f}cm / "
        f"{MAX_START_ORIENTATION_ERROR_DEG:.0f}deg)"
    )
    if pos_err > MAX_START_POSITION_ERROR_M or rot_err_deg > MAX_START_ORIENTATION_ERROR_DEG:
        node.get_logger().error(
            "preflight FAILED -- move the follower closer to the episode's start pose before "
            "replaying (this check cannot be skipped: an impedance controller snapping at a "
            "large step error is real, sudden motion)."
        )
        return False

    publishers = node.get_publishers_info_by_topic(f"/{node.follower_ns}/target_pose")
    other_publishers = [p for p in publishers if p.node_name != node.get_name()]
    if other_publishers:
        node.get_logger().error(
            f"preflight FAILED -- {len(other_publishers)} other node(s) already publishing "
            f"target_pose ({[p.node_name for p in other_publishers]}); replaying alongside "
            "them would race. Stop whatever else is commanding this arm first."
        )
        return False
    return True


def _step_jump_ok(node: PolicyOnDemoReplayer, x_eq_pred: np.ndarray) -> bool:
    """Non-skippable per-step gate: compares a fresh prediction against the arm's
    *live* current pose (not the demo's), so a prediction that has drifted far from
    where the real arm actually is aborts the run instead of being executed."""
    if node.current_pose is None:
        node.get_logger().error("no live current_pose available -- aborting for safety.")
        return False
    pos_err = float(np.linalg.norm(node.current_pose[0:3] - x_eq_pred[0:3]))
    rot_err_deg = float(np.degrees(np.linalg.norm(
        (R.from_rotvec(node.current_pose[3:6]) * R.from_rotvec(x_eq_pred[3:6]).inv()).as_rotvec()
    )))
    if pos_err > MAX_STEP_POSITION_JUMP_M:
        node.get_logger().error(
            f"ABORT -- predicted target is {pos_err * 100:.1f}cm / {rot_err_deg:.1f}deg from the "
            f"arm's live pose (limits: {MAX_STEP_POSITION_JUMP_M * 100:.0f}cm / "
            f"{MAX_STEP_ORIENTATION_JUMP_DEG:.0f}deg) -- refusing to execute this prediction."
        )
        return False
    return True


def run_policy_replay(node, server_url, session, episode, stride, image_size,
                       include_force_history, skip_confirm, out_csv, show_display=True,
                       use_tool_offset=True):
    arrays, image_lookup, task = load_trimmed_episode_with_images(
        session, episode, use_tool_offset=use_tool_offset
    )
    t, q, qdot, x_f, x_eq, wrench, frame_index = (
        arrays["t"], arrays["q"], arrays["qdot"], arrays["x_f"], arrays["x_eq"],
        arrays["wrench"], arrays["frame_index"],
    )
    n = len(t)
    anchors = list(range(0, n, max(1, stride)))
    print(f"[replay] task={task!r}  {n} trimmed frames, executing {len(anchors)} anchors "
          f"(stride={stride}) from {server_url} onto the REAL follower")

    if not preflight(node, x_f[anchors[0]]):
        return

    if not skip_confirm:
        answer = input(
            f"About to query the policy on {len(anchors)} recorded demo anchors and EXECUTE "
            f"every prediction on the REAL follower arm. Type 'yes' to continue: "
        )
        if answer.strip().lower() != "yes":
            node.get_logger().info("Aborted by operator.")
            return

    node.get_logger().info("Switching to variable impedance controller...")
    if not node.switch_controller(
        activate=[VARIABLE_IMPEDANCE_CONTROLLER], deactivate=[MOVE_TO_START_CONTROLLER]
    ):
        node.get_logger().error("Controller switch failed -- aborting, nothing was executed.")
        return

    node.publish_stiffness(FIXED_STIFFNESS)
    rows = []
    try:
        for step, i in enumerate(anchors):
            state = np.concatenate([q[i], qdot[i], x_f[i]]).astype(np.float32)

            force_history = None
            if include_force_history:
                hist_sel = t <= t[i]
                force_history = resample_to_n_samples(
                    t[hist_sel], wrench[hist_sel], t_end=t[i],
                    window_sec=FORCE_HISTORY_WINDOW_SEC, n_samples=FORCE_HISTORY_LEN,
                )

            scene, wrist = image_lookup[int(frame_index[i])]
            scene, wrist = _resize(scene, image_size), _resize(wrist, image_size)

            if show_display:
                try:
                    key = _show_demo_frame(
                        scene, wrist,
                        f"step {step}/{len(anchors) - 1}  t={t[i]:.2f}s  task={task}",
                    )
                    if key == ord('q'):
                        node.get_logger().info("Display window quit requested -- stopping replay.")
                        break
                except cv2.error as e:
                    node.get_logger().warn(f"Disabling image display after a cv2 error: {e}")
                    show_display = False

            action_chunk = get_action_chunk(server_url, scene, wrist, state, task, force_history)
            action_dim = action_chunk.shape[-1]
            if action_dim not in (7, 13):
                node.get_logger().error(f"unexpected action_dim={action_dim} (expected 7 or 13) "
                                         "-- aborting.")
                break
            is_compliance_checkpoint = action_dim == 13

            low_level = action_chunk[0]
            x_eq_pred = np.asarray(low_level[0:6], dtype=np.float64)
            if is_compliance_checkpoint:
                log_k, gripper_cmd = low_level[6:12], low_level[12]
                k = np.exp(log_k)
                k[0:3] = np.clip(k[0:3], *K_TRANS_RANGE)
                k[3:6] = np.clip(k[3:6], *K_ROT_RANGE)
            else:
                gripper_cmd = low_level[6]
                k = FIXED_STIFFNESS

            print(f"[replay] step {step}/{len(anchors) - 1}  anchor_frame={i}/{n - 1}  "
                  f"t={t[i]:.3f}s  label_x_eq={np.array2string(x_eq[i], precision=4)}  "
                  f"predicted_x_eq={np.array2string(x_eq_pred, precision=4)}")

            if not _step_jump_ok(node, x_eq_pred):
                break

            if is_compliance_checkpoint:
                node.publish_stiffness(k)
            node.publish_target(x_eq_pred)
            node.send_gripper_command(gripper_cmd)

            rows.append((t[i], *x_eq[i], *x_eq_pred, *k, float(gripper_cmd)))

            if step < len(anchors) - 1:
                time.sleep(max(0.0, float(t[anchors[step + 1]] - t[i])))
    except KeyboardInterrupt:
        node.get_logger().info("Replay interrupted by user.")
    finally:
        if show_display:
            cv2.destroyAllWindows()
        node.get_logger().info("Replay done -- switching back to move_to_start (home)...")
        node.switch_controller(
            activate=[MOVE_TO_START_CONTROLLER], deactivate=[VARIABLE_IMPEDANCE_CONTROLLER]
        )

    if rows:
        label_dx, label_dy = rows[-1][1] - rows[0][1], rows[-1][2] - rows[0][2]
        pred_dx, pred_dy = rows[-1][7] - rows[0][7], rows[-1][8] - rows[0][8]
        signs_match = np.sign(pred_dx) == np.sign(label_dx)
        print(
            f"\n[replay] label    Δx={label_dx:+.4f}m Δy={label_dy:+.4f}m\n"
            f"[replay] predicted Δx={pred_dx:+.4f}m Δy={pred_dy:+.4f}m\n"
            f"[replay] x sign {'MATCH' if signs_match else 'FLIPPED'}"
        )

    if out_csv and rows:
        with open(out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["t", "label_x", "label_y", "label_z", "label_rx", "label_ry", "label_rz",
                              "pred_x", "pred_y", "pred_z", "pred_rx", "pred_ry", "pred_rz",
                              "k_x", "k_y", "k_z", "k_rx", "k_ry", "k_rz", "gripper_cmd"])
            writer.writerows(rows)
        print(f"[replay] wrote {len(rows)} rows -> {out_csv}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", required=True, choices=dio.TWO_COLOR_SESSIONS)
    parser.add_argument("--episode", type=int, default=None)
    parser.add_argument("--list-episodes", action="store_true")
    parser.add_argument("--server-url", default="http://127.0.0.1:8000/act")
    parser.add_argument("--stride", type=int, default=1, help="query the policy every Nth trimmed demo frame instead of every one")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--no-force-history", action="store_true", help="omit force_history (b0 checkpoints)")
    parser.add_argument("--follower-ns", default="follower")
    parser.add_argument("--yes", action="store_true", help="skip the confirmation prompt (preflight/per-step safety checks still run and cannot be skipped)")
    parser.add_argument("--out-csv", default=None, help="optional path to save the label-vs-predicted-vs-executed trajectory")
    parser.add_argument("--no-display", action="store_true", help="don't open the scene/wrist preview window (e.g. no X display available)")
    parser.add_argument("--no-tool-offset", action="store_true", help="skip loading tool_offset.npy (use a zero tool offset instead) -- for an uncalibrated checkout; NOTE this changes the x_f actually sent to the policy, not just logging, so don't use it to judge a checkpoint's accuracy")
    parser.add_argument("--orientation-correction-deg", type=float, default=ORIENTATION_CORRECTION_DEG, help="twist correction (deg, about the tool's own z-axis) composed onto every predicted orientation before publishing, matching deploy_smolvla.py; pass 0 to publish the raw predicted orientation")
    cli = parser.parse_args()

    if cli.list_episodes:
        list_episodes(cli.session)
        return
    if cli.episode is None:
        parser.error("--episode is required unless --list-episodes is given")

    rclpy.init()
    node = PolicyOnDemoReplayer(
        follower_ns=cli.follower_ns,
        orientation_correction_deg=cli.orientation_correction_deg,
    )
    try:
        run_policy_replay(
            node, cli.server_url, cli.session, cli.episode, cli.stride, cli.image_size,
            include_force_history=not cli.no_force_history, skip_confirm=cli.yes,
            out_csv=cli.out_csv, show_display=not cli.no_display,
            use_tool_offset=not cli.no_tool_offset,
        )
    finally:
        node.shutdown_executor()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
