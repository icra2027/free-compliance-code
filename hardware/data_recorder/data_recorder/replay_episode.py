#!/usr/bin/env python3
"""Replay one recorded LeRobot episode's joint trajectory on the real follower Franka arm.

Reuses the follower's own already-tuned, already-active TeleopFollowerController rather than
introducing a new controller: that controller is a joint-space impedance tracker that follows
whatever sensor_msgs/JointState.position it last received on its `input_topic` (normally the
leader's own franka_robot_state_broadcaster/measured_joint_states topic -- see
fr3_bilateral_teleop/src/teleop_follower_controller.cpp, which reads `input->position` directly,
by index, with no name matching). This script queries that `input_topic` from the live
follower_controller node and republishes the episode's recorded `observation.state` (the
follower's own recorded joint positions -- not `action`, which is a Cartesian EE pose/gripper
tuple, not a joint target) onto it, interpolated up to a much higher rate than the dataset's
recording fps so the impedance controller sees a smooth trajectory rather than a stair-step of
sparse targets. In effect this makes the script look like "the leader" for the duration of the
replay.

Because this drives a real robot arm through recorded motion with no human hand on either side,
several checks run before anything moves (see EpisodeReplayer.preflight): the follower_controller
must already be the active controller, nothing else may already be publishing on its input_topic
(a live real leader would race with us), and the follower's *current* joint position must already
be close to the episode's first frame (an impedance controller snapping at a large step error is
exactly the kind of sudden motion this guards against). None of these are skippable via a single
"just do it" flag except deliberately, and there is a mandatory confirmation prompt before any
motion starts unless --yes is passed.

Usage:
    ros2 run data_recorder replay_lerobot_episode \\
        --dataset-root /path/to/dataset --episode 3

    # See what's in a dataset before picking an episode (dataset-local, compacted indices --
    # NOT session_manifest.jsonl's raw episode_index, which includes discarded takes):
    ros2 run data_recorder replay_lerobot_episode \\
        --dataset-root /path/to/dataset --list-episodes
"""

import argparse
import importlib
import sys
import time
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from controller_manager_msgs.srv import ListControllers
from rcl_interfaces.srv import GetParameters
from rclpy.node import Node
from sensor_msgs.msg import JointState

try:
    # Newer lerobot versions expose datasets under lerobot.datasets.
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.datasets.utils import load_nested_dataset
except ImportError:
    try:
        # Fallback for older layouts.
        _lerobot_datasets = importlib.import_module("lerobot.common.datasets.lerobot_dataset")
        _lerobot_utils = importlib.import_module("lerobot.common.datasets.utils")
        LeRobotDatasetMetadata = _lerobot_datasets.LeRobotDatasetMetadata
        load_nested_dataset = _lerobot_utils.load_nested_dataset
    except ImportError as exc:
        raise ImportError(
            "Could not import LeRobotDatasetMetadata/load_nested_dataset. Install/upgrade "
            "`lerobot` in your ROS 2 Python environment."
        ) from exc

try:
    from franka_msgs.action import Grasp, Move
    from rclpy.action import ActionClient
    _GRIPPER_MSGS_AVAILABLE = True
except ImportError:
    _GRIPPER_MSGS_AVAILABLE = False


NUM_JOINTS = 7

# Mirrors fr3_bilateral_teleop's teleop_gripper_node.cpp bang-bang gripper policy exactly (same
# thresholds/goal fields), just driven from replayed recorded widths instead of a live leader
# gripper -- see that node for why these specific numbers.
GRIPPER_MOVE_SPEED = 0.5  # m/s
GRIPPER_GRASP_FORCE = 1.0  # N
GRIPPER_GRASP_EPSILON_INNER = 0.001  # m
GRIPPER_GRASP_EPSILON_OUTER_SCALE = 100.0
RELATIVE_GRASPING_THRESHOLD = 0.5
RELATIVE_OPENING_THRESHOLD = 0.6
# Franka Hand's default max opening -- only used until the episode's own recorded data reveals
# a wider observed value (mirrors teleop_gripper_node's own running max_width_ tracking).
DEFAULT_MAX_GRIPPER_WIDTH = 0.07  # m


def load_dataset(repo_id: str, dataset_root: str) -> LeRobotDatasetMetadata:
    """Load only the dataset's metadata, never the full LeRobotDataset.

    LeRobotDataset.__init__ unconditionally requires every episode's video file (scene_rgb,
    wrist_rgb, ...) to exist on disk for *every* episode in the dataset, even when loading just
    one episode's numeric columns -- if even one is missing (an interrupted recording, a
    videos-not-yet-encoded dummy dataset, ...) it falls back to treating repo_id as a real
    Hugging Face Hub dataset and crashes trying to reach the network for it (a fake local
    repo_id like "local/whatever" 401s/404s there). Replay only ever needs
    observation.state/action, never images, so there is no reason to pay that cost or that
    failure mode at all -- LeRobotDatasetMetadata's own load_metadata() reads meta/info.json,
    meta/tasks.parquet and meta/episodes/*.parquet directly off disk with no video check and no
    network fallback for a local root.
    """
    return LeRobotDatasetMetadata(repo_id, root=dataset_root)


def num_available_episodes(meta: LeRobotDatasetMetadata) -> int:
    """The real, physically-present episode count.

    Deliberately NOT meta.total_episodes (an info.json field): an interrupted or partially
    written recording session can leave info.json claiming more episodes than actually got a
    meta/episodes row written for them, in which case meta.total_episodes overcounts and
    indexing meta.episodes near the end raises IndexError instead of a clear error message.
    """
    return len(meta.episodes)


def list_episodes(meta: LeRobotDatasetMetadata) -> None:
    available = num_available_episodes(meta)
    if available != meta.total_episodes:
        print(
            f"NOTE: info.json claims {meta.total_episodes} episode(s) but only {available} "
            "actually have metadata on disk -- likely an interrupted recording session. "
            f"Only episodes 0..{available - 1} are listed/usable."
        )
    print(f"{'episode':>7}  {'frames':>7}  {'duration_s':>10}  task")
    for i in range(available):
        ep = meta.episodes[i]
        length = int(ep["length"])
        task = ", ".join(ep["tasks"]) if ep["tasks"] else ""
        print(f"{i:>7}  {length:>7}  {length / meta.fps:>10.2f}  {task}")


def load_episode(
    meta: LeRobotDatasetMetadata, episode_index: int
) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
    """Return (states[N,7], actions[N,7] or None, task_str) for one episode.

    `episode_index` here is meta.episodes' own row POSITION (0..N-1, what --list-episodes
    prints) -- NOT necessarily the same number as that row's own "episode_index" FIELD. A
    session with discarded/failed takes can leave meta/episodes compacted to 0..N-1 by position
    while the physical data/*.parquet files still carry the ORIGINAL, uncompacted episode_index
    VALUES (e.g. position 2's own episode_index field can be 4, if raw takes 2 was discarded and
    never flushed) -- confirmed on a real dataset here, and it's the same raw-vs-compacted
    confusion compliance-vla's diagnose_language_grounding.py hit and had to fix. Reading
    ep["episode_index"] (the field) rather than reusing the `episode_index` parameter (the
    position) as the physical filter value, plus the frame-count assertion below, is this
    script's equivalent of that fix.

    Reads only this episode's rows via load_nested_dataset's own PyArrow predicate pushdown, with
    no `features` schema override -- passing one is what makes lerobot treat image/video-typed
    columns specially and decode them; leaving it out reads the parquet's physical columns as-is
    (observation.state/action are stored as plain float arrays regardless), so this never
    touches video data or video files at all, unlike going through LeRobotDataset/its
    __getitem__.
    """
    total = num_available_episodes(meta)
    if not (0 <= episode_index < total):
        raise ValueError(
            f"episode {episode_index} out of range -- {total} episode(s) actually available, "
            f"0..{total - 1}. Use --list-episodes to see what's available."
        )

    ep = meta.episodes[episode_index]
    length = int(ep["length"])
    if length <= 0:
        raise ValueError(f"episode {episode_index} has no frames (length={length})")
    physical_episode_index = int(ep["episode_index"])

    rows = load_nested_dataset(meta.root / "data", episodes=[physical_episode_index])
    has_action = "action" in rows.column_names
    columns = ["observation.state"] + (["action"] if has_action else [])
    rows = rows.select_columns(columns)

    states = np.stack([np.asarray(x, dtype=np.float64) for x in rows["observation.state"]])
    # Hard assertion, not a warning: a mapping error here means silently replaying the wrong
    # (or a truncated) episode's motion on a real robot -- fail loudly instead.
    if len(states) != length:
        raise ValueError(
            f"episode {episode_index} (physical episode_index={physical_episode_index}): its "
            f"own metadata says {length} frames but {len(states)} were found in the data files "
            "-- data/metadata mismatch, refusing to replay a possibly wrong or truncated episode."
        )
    if states.shape[1] < NUM_JOINTS:
        raise ValueError(
            f"observation.state has {states.shape[1]} columns, expected >= {NUM_JOINTS}"
        )
    states = states[:, :NUM_JOINTS]

    actions = None
    if has_action:
        actions = np.stack([np.asarray(x, dtype=np.float64) for x in rows["action"]])

    task_str = ", ".join(ep["tasks"]) if ep["tasks"] else ""
    return states, actions, task_str


def get_node_parameters(
    node: Node, service_node_path: str, names: List[str], timeout_sec: float = 5.0
) -> Optional[dict]:
    client = node.create_client(GetParameters, f"{service_node_path}/get_parameters")
    if not client.wait_for_service(timeout_sec=timeout_sec):
        return None
    request = GetParameters.Request()
    request.names = names
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_sec)
    if not future.done() or future.result() is None:
        return None
    result = {}
    for name, value in zip(names, future.result().values):
        # 4 == PARAMETER_STRING; avoid importing ParameterType just for one constant here.
        result[name] = value.string_value if value.type == 4 else value.double_array_value
    return result


class EpisodeReplayer(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("franka_episode_replayer")
        self.args = args
        self.namespace = args.namespace.strip("/")
        self.controller_manager_path = f"/{self.namespace}/controller_manager"
        self.controller_node_path = f"/{self.namespace}/{args.controller_name}"

        self.gripper_available = False
        self.gripper_max_width = DEFAULT_MAX_GRIPPER_WIDTH
        self.gripper_closed: Optional[bool] = None  # None = unknown yet
        if _GRIPPER_MSGS_AVAILABLE:
            self.move_client = ActionClient(self, Move, f"/{self.namespace}/franka_gripper/move")
            self.grasp_client = ActionClient(
                self, Grasp, f"/{self.namespace}/franka_gripper/grasp"
            )

    def preflight(
        self, states: np.ndarray
    ) -> Optional[Tuple[str, str]]:
        """Run every safety check before anything is allowed to move.

        Returns (input_topic, arm_id) on success, or None (having already logged the reason)
        if replay must not proceed.
        """
        list_controllers_client = self.create_client(
            ListControllers, f"{self.controller_manager_path}/list_controllers"
        )
        if not list_controllers_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error(
                f"controller_manager not reachable at '{self.controller_manager_path}' -- is "
                "the follower's teleop stack (teleop.launch.py, or at least the follower half "
                "of it) running?"
            )
            return None

        future = list_controllers_client.call_async(ListControllers.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if not future.done() or future.result() is None:
            self.get_logger().error("Timed out listing controllers")
            return None

        matching = [c for c in future.result().controller if c.name == self.args.controller_name]
        if not matching:
            self.get_logger().error(
                f"Controller '{self.args.controller_name}' is not loaded under "
                f"'{self.controller_manager_path}'"
            )
            return None
        if matching[0].state != "active":
            self.get_logger().error(
                f"Controller '{self.args.controller_name}' is state='{matching[0].state}', "
                "not 'active' -- replay drives it by publishing to its input_topic, which only "
                "has an effect while it is the active controller commanding the follower."
            )
            return None

        params = get_node_parameters(self, self.controller_node_path, ["input_topic", "arm_id"])
        if params is None:
            self.get_logger().error(
                f"Could not read parameters from '{self.controller_node_path}'"
            )
            return None
        input_topic = params["input_topic"]
        arm_id = params["arm_id"] or "fr3"
        if not input_topic:
            self.get_logger().error(
                f"'{self.controller_node_path}' has no input_topic parameter set"
            )
            return None

        # A live real leader (or another replay) publishing on this same topic would race with
        # us for control of the follower -- refuse by default. Topic discovery needs a brief
        # moment to settle, hence the short spin before checking.
        rclpy.spin_once(self, timeout_sec=0.5)
        existing_publishers = self.get_publishers_info_by_topic(input_topic)
        if existing_publishers:
            publisher_names = ", ".join(
                f"{p.node_namespace}/{p.node_name}" for p in existing_publishers
            )
            message = (
                f"'{input_topic}' already has {len(existing_publishers)} publisher(s): "
                f"{publisher_names}. Replaying while something else (a real leader? another "
                "replay?) is also publishing here means the follower will race between the two "
                "-- refusing to start."
            )
            if not self.args.force:
                self.get_logger().error(message + " Pass --force to override.")
                return None
            self.get_logger().warn(message + " Continuing anyway because --force was given.")

        current_state = self._read_current_joint_positions(arm_id)
        if current_state is None:
            self.get_logger().error(
                f"Never received a message on '{self.args.current_joint_topic}' -- cannot "
                "verify the follower's current position is safe to start replay from."
            )
            return None

        offset = np.abs(current_state - states[0])
        max_offset = float(np.max(offset))
        if max_offset > self.args.max_start_offset_rad:
            self.get_logger().error(
                "Follower's current joint position is too far from this episode's first frame "
                f"(max per-joint offset {max_offset:.3f} rad > "
                f"--max-start-offset-rad {self.args.max_start_offset_rad:.3f} rad) -- an "
                "impedance controller snapping to a large step error is exactly the sudden "
                "motion this check exists to prevent. Move the follower closer to the episode's "
                f"start pose first (current={np.round(current_state, 3).tolist()}, "
                f"target={np.round(states[0], 3).tolist()})."
            )
            return None

        if self.args.replay_gripper and not _GRIPPER_MSGS_AVAILABLE:
            self.get_logger().warn(
                "franka_msgs.action not importable; disabling gripper replay."
            )
        elif self.args.replay_gripper:
            if self.move_client.wait_for_server(timeout_sec=3.0) and self.grasp_client.wait_for_server(
                timeout_sec=3.0
            ):
                self.gripper_available = True
            else:
                self.get_logger().warn(
                    f"franka_gripper action servers not available under '/{self.namespace}/"
                    "franka_gripper' -- disabling gripper replay (was the follower launched "
                    "with load_gripper:=true?)."
                )

        return input_topic, arm_id

    def _read_current_joint_positions(self, arm_id: str) -> Optional[np.ndarray]:
        received: List[JointState] = []
        sub = self.create_subscription(
            JointState, self.args.current_joint_topic, received.append, 10
        )
        deadline = time.monotonic() + 5.0
        while rclpy.ok() and not received and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        self.destroy_subscription(sub)
        if not received:
            return None
        msg = received[-1]
        if len(msg.position) < NUM_JOINTS:
            return None
        return np.asarray(msg.position[:NUM_JOINTS], dtype=np.float64)

    def _send_gripper_goal(self, close: bool) -> None:
        if close:
            goal = Grasp.Goal()
            goal.width = 0.0
            goal.speed = GRIPPER_MOVE_SPEED
            goal.force = GRIPPER_GRASP_FORCE
            goal.epsilon.inner = GRIPPER_GRASP_EPSILON_INNER
            goal.epsilon.outer = GRIPPER_GRASP_EPSILON_OUTER_SCALE * self.gripper_max_width
            self.grasp_client.send_goal_async(goal)
            self.get_logger().info("Gripper: grasp (close)")
        else:
            goal = Move.Goal()
            goal.width = self.gripper_max_width
            goal.speed = GRIPPER_MOVE_SPEED
            self.move_client.send_goal_async(goal)
            self.get_logger().info("Gripper: move (open)")

    def _update_gripper(self, recorded_width: float) -> None:
        if not self.gripper_available:
            return
        # Track the running max exactly like teleop_gripper_node.cpp does for a live leader,
        # since a fixed hardware default may not match the actual attached gripper.
        width = float(np.clip(recorded_width, 0.0, None))
        self.gripper_max_width = max(self.gripper_max_width, width)
        grasp_threshold = RELATIVE_GRASPING_THRESHOLD * self.gripper_max_width
        open_threshold = RELATIVE_OPENING_THRESHOLD * self.gripper_max_width
        if width < grasp_threshold and self.gripper_closed is not True:
            self.gripper_closed = True
            self._send_gripper_goal(close=True)
        elif width > open_threshold and self.gripper_closed is not False:
            self.gripper_closed = False
            self._send_gripper_goal(close=False)

    def replay(
        self,
        states: np.ndarray,
        actions: Optional[np.ndarray],
        input_topic: str,
        arm_id: str,
    ) -> None:
        n = len(states)
        fps = self.args.fps
        speed = self.args.speed
        duration_s = (n - 1) / fps / speed if n > 1 else 0.0
        publish_dt = 1.0 / self.args.publish_rate
        joint_names = [f"{arm_id}_joint{i}" for i in range(1, NUM_JOINTS + 1)]

        publisher = self.create_publisher(JointState, input_topic, 10)
        replay_gripper = self.gripper_available and actions is not None
        if replay_gripper:
            # Unlike teleop_gripper_node's live running max (which only ever grows, seeded from
            # a hardware-default guess), the whole episode is already known here -- use its
            # actual observed peak width up front so the grasp/open thresholds are correct from
            # frame 0, even for an episode whose gripper never approaches the 0.07 m hardware
            # default (e.g. a small object held throughout).
            self.gripper_max_width = max(
                self.gripper_max_width, float(np.max(actions[:, 6]))
            )

        self.get_logger().info(
            f"Replaying {n} frames ({duration_s:.1f}s at speed={speed}x) onto '{input_topic}'"
        )

        last_frame_idx = -1
        t_start = time.monotonic()
        next_tick = t_start
        try:
            while rclpy.ok():
                elapsed_playback = (time.monotonic() - t_start) * speed
                if elapsed_playback >= (n - 1) / fps:
                    break

                virtual_idx = elapsed_playback * fps
                i0 = min(int(np.floor(virtual_idx)), n - 2) if n > 1 else 0
                i1 = min(i0 + 1, n - 1)
                frac = virtual_idx - i0

                target = states[i0] * (1.0 - frac) + states[i1] * frac

                msg = JointState()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.name = joint_names
                msg.position = target.tolist()
                publisher.publish(msg)

                if replay_gripper and i0 != last_frame_idx:
                    last_frame_idx = i0
                    self._update_gripper(float(actions[i0, 6]))

                rclpy.spin_once(self, timeout_sec=0.0)

                next_tick += publish_dt
                sleep_for = next_tick - time.monotonic()
                if sleep_for > 0:
                    time.sleep(sleep_for)
                else:
                    next_tick = time.monotonic()  # fell behind; resync instead of spiraling

            # Publish the exact final recorded position once, so replay always ends precisely
            # at the episode's last frame rather than wherever interpolation happened to land.
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = joint_names
            msg.position = states[-1].tolist()
            publisher.publish(msg)
            self.get_logger().info("Replay finished.")
        except KeyboardInterrupt:
            self.get_logger().warn("Replay interrupted by user; stopped mid-episode.")

        self.get_logger().warn(
            f"No further commands will be published on '{input_topic}'. "
            f"'{self.args.controller_name}' will fall back to gravity compensation "
            "input_topic_timeout after the last message (same as if a real leader disconnected) "
            "unless teleop or another script resumes publishing to it."
        )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="replay_lerobot_episode",
        description=(
            "Replay one recorded LeRobot episode's joint trajectory on the real follower "
            "Franka arm, via the follower's own active TeleopFollowerController."
        ),
    )
    parser.add_argument("--dataset-root", required=True, help="Path to the LeRobot dataset root")
    parser.add_argument(
        "--repo-id",
        default="local/franka_vla_multimodal",
        help="Dataset repo_id (only used as a cache key for a local root; cosmetic)",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=None,
        help=(
            "Dataset-local episode index (0-based, the dataset's own compacted numbering -- "
            "NOT session_manifest.jsonl's raw episode_index, which includes discarded takes). "
            "Required unless --list-episodes is given."
        ),
    )
    parser.add_argument(
        "--list-episodes",
        action="store_true",
        help="Print every episode's index/length/task and exit without replaying anything",
    )
    parser.add_argument(
        "--namespace",
        default="franka_teleop/follower",
        help="Follower's ROS namespace (matches teleop.launch.py's convention)",
    )
    parser.add_argument("--controller-name", default="follower_controller")
    parser.add_argument(
        "--current-joint-topic",
        default=None,
        help=(
            "Topic to read the follower's current joint position from, for the start-offset "
            "safety check. Default: derived from --namespace, matching "
            "data_recorder's own joint_topic default."
        ),
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override the dataset's own recorded fps (default: read from dataset metadata)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help=(
            "Playback speed multiplier. 1.0 = recorded speed. Values > 1.0 exceed the peak "
            "joint velocity actually demonstrated and are logged loudly."
        ),
    )
    parser.add_argument(
        "--publish-rate",
        type=float,
        default=200.0,
        help=(
            "Rate (Hz) to publish interpolated joint targets at. Higher than the dataset's own "
            "recording fps on purpose -- the impedance controller tracks a smooth trajectory "
            "far better than the recorded frames' native (10-30 Hz) step changes."
        ),
    )
    parser.add_argument(
        "--max-start-offset-rad",
        type=float,
        default=0.5,
        help=(
            "Max allowed per-joint difference (radians) between the follower's current position "
            "and the episode's first frame before refusing to start (safety check)."
        ),
    )
    parser.add_argument(
        "--replay-gripper",
        dest="replay_gripper",
        action="store_true",
        default=True,
        help="Replay recorded gripper open/close via franka_gripper Move/Grasp (default: on)",
    )
    parser.add_argument(
        "--no-replay-gripper", dest="replay_gripper", action="store_false"
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip the interactive confirmation prompt before moving the robot",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Proceed even if another publisher is already on the controller's input_topic",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run every check and print the replay plan, but never publish or move anything",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv[1:] if argv is not None else sys.argv[1:])

    if args.current_joint_topic is None:
        args.current_joint_topic = (
            f"/{args.namespace.strip('/')}/franka_robot_state_broadcaster/measured_joint_states"
        )

    if args.speed <= 0:
        print("replay_lerobot_episode: --speed must be > 0", file=sys.stderr)
        sys.exit(2)

    meta = load_dataset(args.repo_id, args.dataset_root)

    if args.list_episodes:
        list_episodes(meta)
        return

    if args.episode is None:
        print(
            "replay_lerobot_episode: --episode is required (use --list-episodes to see options)",
            file=sys.stderr,
        )
        sys.exit(2)

    try:
        states, actions, task_str = load_episode(meta, args.episode)
    except ValueError as exc:
        print(f"replay_lerobot_episode: {exc}", file=sys.stderr)
        sys.exit(2)

    if args.fps is None:
        args.fps = float(meta.fps)
    if args.speed > 1.0:
        print(
            f"WARNING: --speed {args.speed} replays faster than recorded, exceeding the peak "
            "joint velocity actually demonstrated in this episode.",
            file=sys.stderr,
        )
    if args.replay_gripper and actions is None:
        print(
            "NOTE: this dataset has no 'action' feature (older recording), so gripper replay "
            "is unavailable regardless of --replay-gripper.",
            file=sys.stderr,
        )

    n = len(states)
    duration_s = (n - 1) / args.fps / args.speed if n > 1 else 0.0
    print(f"Episode {args.episode}: task='{task_str}', {n} frames, {duration_s:.1f}s at "
          f"speed={args.speed}x (recorded fps={meta.fps})")

    rclpy.init(args=None)
    node = EpisodeReplayer(args)
    try:
        preflight_result = node.preflight(states)
        if preflight_result is None:
            sys.exit(1)
        input_topic, arm_id = preflight_result

        print(f"input_topic='{input_topic}', arm_id='{arm_id}', "
              f"gripper_replay={'on' if node.gripper_available else 'off'}")

        if args.dry_run:
            print("--dry-run given: all checks passed, not moving anything.")
            return

        if not args.yes:
            if not sys.stdin.isatty():
                print(
                    "replay_lerobot_episode: refusing to move the robot without confirmation "
                    "in a non-interactive session. Pass --yes to proceed.",
                    file=sys.stderr,
                )
                sys.exit(1)
            answer = input(
                f"About to move the real follower arm through {n} recorded frames "
                f"({duration_s:.1f}s). Type 'replay' to continue: "
            )
            if answer.strip() != "replay":
                print("Aborted.")
                return

        node.replay(states, actions, input_topic, arm_id)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
