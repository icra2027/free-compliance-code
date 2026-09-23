#!/usr/bin/env python3
"""Record one bilateral teleoperation demonstration to CSV for offline impedance extraction.

Logs leader pose x_l(t), follower pose x_f(t), and the follower's estimated external
wrench f(t) -- exactly the three signals proposal §4.1's extraction regression needs
(`e(t) = x_l(t) ⊖ x_f(t)`, regressed against `f(t)`). No existing demo-recording
infrastructure was in this repo before this script (checked: no rosbag2 record launch
files, no LeRobot dataset builder, nothing) -- this is deliberately a minimal, CSV-based
recorder matching the convention every other script in this package already uses
(collect_free_space_sweep.py, etc.) rather than introducing rosbag2 as a new dependency
for what only needs to feed extract_impedance_labels.py.

Pose comes directly from franka_robot_state_broadcaster's `~/current_pose`
(geometry_msgs/PoseStamped, sourced from `O_T_EE`) on both namespaces -- no FK-from-joints
needed, and it runs at the full 1 kHz `convenience_publish_rate` on this rig (confirmed:
neither franka_robot_state_broadcaster's default nor this package's
config/teleop_controllers.yaml override it down).

Recording is driven off wrench-message arrival (typically the highest-rate of the three in
practice) and paired with the most recently received leader/follower pose within
--pair-tolerance, the same latest-sample-cache pattern collect_free_space_sweep.py already
uses.

Usage (after bringing up bilateral teleop normally, e.g. teleop.launch.py):
    ros2 run fr3_bilateral_teleop record_demo.py --task T1_wiping --operator A \
        --manner gently --duration 25
"""
import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped, WrenchStamped

CSV_FIELDS = [
    "t",
    "lx", "ly", "lz", "lqx", "lqy", "lqz", "lqw",
    "fx", "fy", "fz", "fqx", "fqy", "fqz", "fqw",
    "wfx", "wfy", "wfz", "wtx", "wty", "wtz",
]


def sanitize(text: str) -> str:
    return text.strip("/").replace("/", "_").replace(" ", "_")


class DemoRecorderNode(Node):
    def __init__(self, leader_ns: str, follower_ns: str, pair_tolerance_sec: float):
        super().__init__("record_demo")
        self.leader_ns = leader_ns.strip("/")
        self.follower_ns = follower_ns.strip("/")
        self.pair_tolerance_sec = pair_tolerance_sec

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=200)

        self._latest_leader_pose: Optional[Dict] = None
        self._latest_follower_pose: Optional[Dict] = None
        self._recording = False
        self.rows = []

        self.create_subscription(
            PoseStamped, f"/{self.leader_ns}/franka_robot_state_broadcaster/current_pose",
            self._leader_pose_cb, qos)
        self.create_subscription(
            PoseStamped, f"/{self.follower_ns}/franka_robot_state_broadcaster/current_pose",
            self._follower_pose_cb, qos)
        self.create_subscription(
            WrenchStamped,
            f"/{self.follower_ns}/franka_robot_state_broadcaster/external_wrench_in_base_frame",
            self._wrench_cb, qos)

    @staticmethod
    def _stamp_sec(msg) -> float:
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def _leader_pose_cb(self, msg: PoseStamped) -> None:
        self._latest_leader_pose = {"t": self._stamp_sec(msg), "pose": msg.pose}

    def _follower_pose_cb(self, msg: PoseStamped) -> None:
        self._latest_follower_pose = {"t": self._stamp_sec(msg), "pose": msg.pose}

    def _wrench_cb(self, msg: WrenchStamped) -> None:
        if not self._recording:
            return
        t = self._stamp_sec(msg)
        lp, fp = self._latest_leader_pose, self._latest_follower_pose
        if lp is None or fp is None:
            return
        if (abs(lp["t"] - t) > self.pair_tolerance_sec
                or abs(fp["t"] - t) > self.pair_tolerance_sec):
            return
        l, f = lp["pose"], fp["pose"]
        self.rows.append({
            "t": t,
            "lx": l.position.x, "ly": l.position.y, "lz": l.position.z,
            "lqx": l.orientation.x, "lqy": l.orientation.y,
            "lqz": l.orientation.z, "lqw": l.orientation.w,
            "fx": f.position.x, "fy": f.position.y, "fz": f.position.z,
            "fqx": f.orientation.x, "fqy": f.orientation.y,
            "fqz": f.orientation.z, "fqw": f.orientation.w,
            "wfx": msg.wrench.force.x, "wfy": msg.wrench.force.y, "wfz": msg.wrench.force.z,
            "wtx": msg.wrench.torque.x, "wty": msg.wrench.torque.y, "wtz": msg.wrench.torque.z,
        })

    def wait_for_first_messages(self, timeout_sec: float = 15.0) -> bool:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            if self._latest_leader_pose is not None and self._latest_follower_pose is not None:
                return True
            rclpy.spin_once(self, timeout_sec=0.1)
        return False

    def record_for(self, duration_sec: float) -> None:
        self._recording = True
        deadline = time.monotonic() + duration_sec
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.02)
        self._recording = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--leader-namespace", default="franka_teleop/leader")
    parser.add_argument("--follower-namespace", default="franka_teleop/follower")
    parser.add_argument("--task", required=True, help="e.g. T1_wiping")
    parser.add_argument("--operator", default="A")
    parser.add_argument("--manner", default="", help="e.g. gently/normally/firmly, optional")
    parser.add_argument(
        "--duration", type=float, default=25.0,
        help="seconds, per the proposal's ~25s/demo data budget")
    parser.add_argument(
        "--pair-tolerance", type=float, default=0.02,
        help="max pose/wrench timestamp skew accepted, seconds")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(
        args.output_dir or __import__("os").environ.get(
            "TELEOP_DEMOS_OUTPUT_DIR", "/tmp/franka_teleop_demos"))
    output_dir.mkdir(parents=True, exist_ok=True)

    rclpy.init()
    node = DemoRecorderNode(args.leader_namespace, args.follower_namespace, args.pair_tolerance)
    try:
        node.get_logger().info(
            f"Waiting for leader/follower pose topics (leader={args.leader_namespace}, "
            f"follower={args.follower_namespace})...")
        if not node.wait_for_first_messages():
            node.get_logger().error(
                "No pose messages received -- is teleop.launch.py up and are both arms "
                "publishing franka_robot_state_broadcaster/current_pose?")
            return 1

        print(
            f"Ready. Task={args.task!r} operator={args.operator!r} manner={args.manner!r} "
            f"duration={args.duration}s. Position the leader/follower as the demo should "
            f"start, then press Enter to begin recording.", flush=True)
        input("> ")

        node.get_logger().info(f"Recording for {args.duration}s...")
        node.record_for(args.duration)
        node.get_logger().info(f"Done. {len(node.rows)} paired samples captured.")

        if len(node.rows) < 50:
            node.get_logger().error(
                f"Only {len(node.rows)} samples -- refusing to write a near-empty demo file. "
                "Check that both current_pose topics and the wrench topic are actually "
                "publishing throughout (not just at the start).")
            return 1

        timestamp = int(time.time())
        manner_part = f"_{sanitize(args.manner)}" if args.manner else ""
        filename = (
            f"{sanitize(args.task)}_{sanitize(args.operator)}{manner_part}_{timestamp}.csv")
        csv_path = output_dir / filename
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(node.rows)
        node.get_logger().info(f"Wrote {csv_path}")
        print(str(csv_path))
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
