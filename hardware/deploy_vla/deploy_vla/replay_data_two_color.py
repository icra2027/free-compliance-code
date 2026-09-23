"""Replays one real data_two_color demonstration's recorded x_eq (leader pose) trajectory
onto the real Franka follower arm, through the same variable_impedance_controllers interface
(target_pose/target_stiffness on variable_impedance_controller) deploy_smolvla.py uses --
so "does the actual recorded wipe look right when executed?" is checkable directly,
independent of the VLA policy.

Trims the episode with the same _trim_to_last_contact logic src/compliance_vla/policy/dataset.py applies
before ever building a training label (copied here rather than imported, so this script
stays torch-free -- see that module's own docstring for why a naive contact-force
threshold is NOT enough to drop data_two_color's appended return-to-start_joint_
configuration tail; extract_impedance_labels.py's contact_mask is exactly that naive
threshold, and using it here would replay ~100 extra frames of the arm coasting home).

Usage (run on the machine actually controlling the follower):
    python3 replay_data_two_color.py --session demo_blue_firm --episode 11

    # See what's in a session first:
    python3 replay_data_two_color.py --session demo_blue_firm --list-episodes


ros2 service call /follower/service_server/set_full_collision_behavior \\
  franka_msgs/srv/SetFullCollisionBehavior \\
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
import os
import sys
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray
from controller_manager_msgs.srv import SwitchController
from scipy.spatial.transform import Rotation as R

# --- sys.path wiring, matching every other ad hoc analysis script written during this
# investigation (verify_x_f_frame.py, the offline contact-window extraction) -- these
# repo-relative paths are specific to how compliance-vla/fr3_bilateral_teleop are
# vendored into *this* workspace (as src/ siblings, not the external/ layout those
# scripts' own REPO_ROOT constants assume).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_WS_SRC = os.path.dirname(os.path.dirname(_THIS_DIR))          # franka_ros2_ws/src
_REPO_ROOT = os.path.dirname(_WS_SRC)                          # franka_ros2_ws
_BOOKISH_ROOT = os.path.join(_WS_SRC, "compliance-vla")
_BOOKISH_SCRIPTS = os.path.join(_BOOKISH_ROOT, "scripts")
_TELEOP_SCRIPTS = os.path.join(_WS_SRC, "fr3_bilateral_teleop", "dataset_tools", "labeling")
for _p in (_BOOKISH_ROOT, _BOOKISH_SCRIPTS, _TELEOP_SCRIPTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dataset_io as dio  # noqa: E402
from compliance_vla.policy import labels as lb  # noqa: E402

DATA_TWO_COLOR_ROOT = os.path.join(_REPO_ROOT, "data_two_color")

# Same controller pair deploy_smolvla.py switches between, and the same topics
# variable_impedance_controllers' CartesianController exposes on the follower.
VARIABLE_IMPEDANCE_CONTROLLER = "variable_impedance_controller"
MOVE_TO_START_CONTROLLER = "move_to_start_example_controller"

# Conservative, fixed mid-range impedance for the whole replay (deploy_smolvla.py's
# K_TRANS_RANGE/K_ROT_RANGE clip bounds) -- this script isn't replaying a per-step
# compliance target (log_k is in the auto-fit contact frame, a Week-4 controller-
# integration question per src/compliance_vla/policy/labels.py, out of scope for a position sanity
# check), just tracking the recorded Cartesian pose with a sane, constant stiffness.
DEFAULT_STIFFNESS = np.array([300.0, 300.0, 300.0, 20.0, 20.0, 20.0])  # [N/m x3, N*m/rad x3]

# Hard (non-skippable) preflight gate: refuse to start if the follower's current pose is
# farther than this from the trimmed episode's first frame -- an impedance controller
# snapping at a large step error is exactly the sudden motion this guards against, same
# rationale as data_recorder/replay_episode.py's own preflight.
MAX_START_POSITION_ERROR_M = 0.05
MAX_START_ORIENTATION_ERROR_DEG = 15.0

SPEED_PAUSE_THRESHOLD_M_S = 0.01
NEAR_HOME_RADIUS_M = 0.05


def _trim_to_last_contact(g, speed_threshold=SPEED_PAUSE_THRESHOLD_M_S,
                           near_home_radius=NEAR_HOME_RADIUS_M, settle_frames=2):
    """Copied verbatim from compliance-vla/src/compliance_vla/policy/dataset.py (not imported --
    that module pulls in torch, which this real-hardware-facing script deliberately does
    not need). See that module's docstring for the full rationale: a naive contact-force
    threshold does not exclude data_two_color's appended return-to-start_joint_
    configuration tail, so this walks backward from episode end for the last point that
    is both slow AND far from the episode's own start pose (i.e. paused out over the
    board, not settling back at rest)."""
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


def load_trimmed_episode(session, episode_index):
    """Returns (t, x_eq) for one trimmed episode: t is (T,) seconds from episode start,
    x_eq is (T, 6) [x, y, z, rx, ry, rz] in base frame -- the same convention
    deploy_smolvla.py publishes to target_pose."""
    frames = dio.load_frames(session, dataset_root=DATA_TWO_COLOR_ROOT)
    episodes = dict(tuple(frames.groupby("episode_index")))
    if episode_index not in episodes:
        raise ValueError(
            f"episode {episode_index} not found in session {session!r} "
            f"(available: {sorted(episodes)})"
        )
    g = episodes[episode_index].sort_values("frame_index")
    n_before = len(g)
    g = _trim_to_last_contact(g)
    n_after = len(g)
    print(f"[replay] session={session} episode={episode_index}: "
          f"trimmed {n_before} -> {n_after} frames ({n_before - n_after} dropped)")

    tool_offset = lb.load_tool_offset()
    args = lb.default_extraction_args()
    sigma_f = lb.default_sigma_f()
    arrays = lb.compute_episode_arrays(g, tool_offset, args, sigma_f)
    if arrays is None:
        raise RuntimeError(
            f"{session}#{episode_index}: contact-frame extraction failed after trimming "
            "-- pick a different episode (see --list-episodes)."
        )
    return arrays["t"], arrays["x_eq"]


def list_episodes(session):
    frames = dio.load_frames(session, dataset_root=DATA_TWO_COLOR_ROOT)
    tasks = dio.load_episodes_meta(session, dataset_root=DATA_TWO_COLOR_ROOT)
    task_lookup = dict(zip(tasks["episode_index"], tasks["tasks"].apply(lambda t: t[0] if len(t) else "")))
    for ep_idx, g in frames.groupby("episode_index"):
        print(f"  episode {ep_idx:>3}: {len(g):>4} frames  task={task_lookup.get(ep_idx, '')!r}")


def pose_to_xyz_rotvec(msg: PoseStamped) -> np.ndarray:
    pos = msg.pose.position
    ori = msg.pose.orientation
    rotvec = R.from_quat([ori.x, ori.y, ori.z, ori.w]).as_rotvec()
    return np.concatenate([[pos.x, pos.y, pos.z], rotvec])


class DataTwoColorReplayer(Node):
    def __init__(self, follower_ns: str = "follower"):
        super().__init__("data_two_color_replayer")
        self.follower_ns = follower_ns
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
        self.switch_client = self.create_client(
            SwitchController, f"/{follower_ns}/controller_manager/switch_controller"
        )

        # Continuous background spin -- see deploy_smolvla.py's own fix (2026-09-05) for
        # why manual rclpy.spin_once starves subscriptions here: a blocking service call
        # (switch_controller) with no spinning in between would leave current_pose stale
        # for the whole preflight check otherwise.
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


def preflight(node: DataTwoColorReplayer, first_target: np.ndarray) -> bool:
    """Hard gate, not skippable by any flag: refuses to start if the follower isn't
    already close to the episode's first recorded frame."""
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
        f"episode's first frame (limits: {MAX_START_POSITION_ERROR_M * 100:.0f}cm / "
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


def run_replay(node: DataTwoColorReplayer, t: np.ndarray, x_eq: np.ndarray, skip_confirm: bool):
    n = len(t)
    if not preflight(node, x_eq[0]):
        return

    if not skip_confirm:
        answer = input(
            f"About to replay {n} steps ({t[-1]:.1f}s of recorded motion) on the REAL follower "
            "arm. Type 'yes' to continue: "
        )
        if answer.strip().lower() != "yes":
            node.get_logger().info("Aborted by operator.")
            return

    node.get_logger().info("Switching to variable impedance controller...")
    if not node.switch_controller(
        activate=[VARIABLE_IMPEDANCE_CONTROLLER], deactivate=[MOVE_TO_START_CONTROLLER]
    ):
        node.get_logger().error("Controller switch failed -- aborting, nothing was replayed.")
        return

    node.publish_stiffness(DEFAULT_STIFFNESS)
    try:
        for i in range(n):
            print(f"[replay] step {i}/{n - 1}  t={t[i]:.3f}s  x_eq={np.array2string(x_eq[i], precision=4)}")
            node.publish_target(x_eq[i])
            if i < n - 1:
                time.sleep(max(0.0, float(t[i + 1] - t[i])))
    except KeyboardInterrupt:
        node.get_logger().info("Replay interrupted by user.")
    finally:
        node.get_logger().info("Replay done -- switching back to move_to_start (home)...")
        node.switch_controller(
            activate=[MOVE_TO_START_CONTROLLER], deactivate=[VARIABLE_IMPEDANCE_CONTROLLER]
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", required=True, choices=dio.TWO_COLOR_SESSIONS)
    parser.add_argument("--episode", type=int, default=None)
    parser.add_argument("--list-episodes", action="store_true")
    parser.add_argument("--follower-ns", default="follower")
    parser.add_argument("--yes", action="store_true", help="skip the confirmation prompt (preflight pose/publisher checks still run and cannot be skipped)")
    cli = parser.parse_args()

    if cli.list_episodes:
        list_episodes(cli.session)
        return
    if cli.episode is None:
        parser.error("--episode is required unless --list-episodes is given")

    t, x_eq = load_trimmed_episode(cli.session, cli.episode)

    rclpy.init()
    node = DataTwoColorReplayer(follower_ns=cli.follower_ns)
    try:
        run_replay(node, t, x_eq, skip_confirm=cli.yes)
    finally:
        node.shutdown_executor()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
