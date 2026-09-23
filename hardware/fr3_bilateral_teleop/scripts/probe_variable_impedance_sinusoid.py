#!/usr/bin/env python3
"""Sinusoidal stiffness probe for the proposal §4.3 variable-impedance controller.

Publishes a sinusoidal per-axis target stiffness (`target_stiffness`, std_msgs/
Float64MultiArray, the same 6-element [kx,ky,kz,krx,kry,krz] layout
variable_impedance_controllers/CartesianController's `variable_stiffness` topic expects) and a fixed
target pose offset from wherever the arm currently is (`target_pose`, geometry_msgs/
PoseStamped) at `variable_impedance_controllers`' `variable_impedance_controller`, then records that
controller's own `~/diagnostics/{stiffness_target,stiffness_applied,energy_tank_energy}`
topics (added this session specifically for this purpose -- see cartesian_controller.cpp's
`applyStiffnessShaping()`) to CSV and produces the commanded-vs-realized-stiffness +
energy-tank-trace figure the proposal §4.3 requires.

Run launch/validate_variable_impedance.launch.py first (defaults to fake hardware -- no
physical robot needed to exercise the wiring, but no real dynamics either; the arm will not
actually move under fake hardware, so this validates the stiffness-shaping pipeline's own
numerics and the topic/param wiring, NOT a physically realistic force response -- see
tasks.md Day 5 for what a real-hardware run still needs to add). A nonzero, FIXED pose
offset is commanded (not zero) specifically so the energy tank sees genuine nonzero
withdrawal amounts (0.5*delta_k*e^2) as the sinusoid increases stiffness, rather than every
increase being free.

Usage:
    ros2 launch fr3_bilateral_teleop validate_variable_impedance.launch.py
    # separate terminal:
    ros2 run fr3_bilateral_teleop probe_variable_impedance_sinusoid.py --duration 20
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64, Float64MultiArray

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

AXIS_LABELS = ["x", "y", "z", "rx", "ry", "rz"]
CSV_FIELDS = ["t"] + [f"k_target_{a}" for a in AXIS_LABELS] + \
    [f"k_applied_{a}" for a in AXIS_LABELS] + ["energy_tank_j"]


class SinusoidProbeNode(Node):
    def __init__(self, namespace: str, controller: str):
        super().__init__("probe_variable_impedance_sinusoid")
        ns = namespace.strip('/')
        # variable_impedance_controllers' target_pose/target_stiffness/current_pose topic params are
        # plain names (no `~/` private-namespace prefix), so ros2_control resolves them
        # directly under the arm's namespace, NOT under the controller's own sub-namespace
        # -- confirmed against the live `ros2 topic list` output from this launch file, not
        # assumed. Only this controller's own `~/diagnostics/*` topics (added this session)
        # are namespaced under the controller name.
        self.namespace_ns = f"/{ns}"
        self.diagnostics_ns = f"/{ns}/{controller}/diagnostics"

        reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)
        best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=200)

        self._current_pose: Optional[PoseStamped] = None
        self._latest_target = [0.0] * 6
        self._latest_applied = [0.0] * 6
        self._latest_energy = 0.0
        self.rows = []
        self._recording = False
        self._t0 = None

        self.create_subscription(
            PoseStamped, f"{self.namespace_ns}/current_pose", self._pose_cb, best_effort)
        self.target_stiffness_pub = self.create_publisher(
            Float64MultiArray, f"{self.namespace_ns}/target_stiffness", reliable)
        self.target_pose_pub = self.create_publisher(
            PoseStamped, f"{self.namespace_ns}/target_pose", reliable)

        self.create_subscription(
            Float64MultiArray, f"{self.diagnostics_ns}/stiffness_target",
            self._target_cb, best_effort)
        self.create_subscription(
            Float64MultiArray, f"{self.diagnostics_ns}/stiffness_applied",
            self._applied_cb, best_effort)
        self.create_subscription(
            Float64, f"{self.diagnostics_ns}/energy_tank_energy",
            self._energy_cb, best_effort)

    def _pose_cb(self, msg: PoseStamped) -> None:
        self._current_pose = msg

    def _target_cb(self, msg: Float64MultiArray) -> None:
        self._latest_target = list(msg.data)

    def _applied_cb(self, msg: Float64MultiArray) -> None:
        self._latest_applied = list(msg.data)
        # Recording is driven off stiffness_applied's arrival -- it's published every
        # control cycle by applyStiffnessShaping() (nominally 1kHz), the highest-rate of
        # the three diagnostics topics, same "drive off the fastest topic" pattern
        # record_demo.py uses for its wrench/pose triple.
        if self._recording:
            now = time.monotonic()
            self.rows.append(
                [now - self._t0] + list(self._latest_target) + list(self._latest_applied) +
                [self._latest_energy])

    def _energy_cb(self, msg: Float64) -> None:
        self._latest_energy = msg.data

    def wait_for_current_pose(self, timeout_sec: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout_sec
        while self._current_pose is None and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        return self._current_pose is not None

    def publish_offset_target_pose(self, offset_xyz):
        assert self._current_pose is not None
        msg = PoseStamped()
        msg.header.frame_id = self._current_pose.header.frame_id
        msg.pose.orientation = self._current_pose.pose.orientation
        msg.pose.position.x = self._current_pose.pose.position.x + offset_xyz[0]
        msg.pose.position.y = self._current_pose.pose.position.y + offset_xyz[1]
        msg.pose.position.z = self._current_pose.pose.position.z + offset_xyz[2]
        msg.header.stamp = self.get_clock().now().to_msg()
        self.target_pose_pub.publish(msg)

    def publish_sinusoid_stiffness(self, t: float, freq_hz: float):
        import math
        k_mid = [500.0, 500.0, 500.0, 40.0, 40.0, 40.0]
        k_amp = [400.0, 400.0, 400.0, 30.0, 30.0, 30.0]
        # Distinct phase per axis so anisotropy is visible in the trace rather than all six
        # axes moving in lockstep.
        phases = [0.0, 1.05, 2.09, 0.52, 1.57, 2.62]
        msg = Float64MultiArray()
        msg.data = [
            k_mid[i] + k_amp[i] * math.sin(2 * math.pi * freq_hz * t + phases[i])
            for i in range(6)
        ]
        self.target_stiffness_pub.publish(msg)

    def run(self, duration_sec: float, freq_hz: float, pose_offset, publish_rate_hz: float):
        self._recording = True
        self._t0 = time.monotonic()
        period = 1.0 / publish_rate_hz
        next_publish = self._t0
        end_time = self._t0 + duration_sec
        while time.monotonic() < end_time:
            now = time.monotonic()
            if now >= next_publish:
                elapsed = now - self._t0
                self.publish_sinusoid_stiffness(elapsed, freq_hz)
                self.publish_offset_target_pose(pose_offset)
                next_publish += period
            rclpy.spin_once(self, timeout_sec=max(0.0, next_publish - time.monotonic()))
        self._recording = False


def make_figure(rows, output_path: Path):
    if not rows:
        return
    t = [r[0] for r in rows]
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    axes = axes.flatten()
    for i, axis_label in enumerate(AXIS_LABELS):
        ax = axes[i]
        k_target = [r[1 + i] for r in rows]
        k_applied = [r[1 + 6 + i] for r in rows]
        ax.plot(t, k_target, label="commanded (target)", linewidth=1, alpha=0.7)
        ax.plot(t, k_applied, label="realized (applied)", linewidth=1.5)
        unit = "N/m" if i < 3 else "Nm/rad"
        ax.set_title(f"stiffness {axis_label} ({unit})")
        ax.set_xlabel("t (s)")
        if i == 0:
            ax.legend(fontsize=8)
    energy = [r[-1] for r in rows]
    ax = axes[6]
    ax.plot(t, energy, color="tab:red")
    ax.set_title("energy tank (J)")
    ax.set_xlabel("t (s)")
    axes[7].axis("off")
    axes[8].axis("off")
    fig.suptitle(
        "Commanded vs realized stiffness + energy-tank trace "
        "(fake hardware -- wiring/numerics check, not physical dynamics)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--namespace", default="follower")
    parser.add_argument("--controller", default="variable_impedance_controller")
    parser.add_argument("--duration", type=float, default=20.0, help="seconds")
    parser.add_argument(
        "--freq", type=float, default=0.5,
        help="stiffness sinusoid frequency, Hz -- 0.5Hz with gamma=3/s is fast enough that "
             "rate limiting is visible in the trace, slow enough that it still tracks")
    parser.add_argument(
        "--publish-rate", type=float, default=30.0,
        help="Hz -- matches the proposal's action-chunk rate (§4.2), not the raw 1kHz")
    parser.add_argument(
        "--pose-offset", type=float, nargs=3, default=[0.03, 0.0, 0.0],
        metavar=("DX", "DY", "DZ"),
        help="meters, target pose offset from the arm's current pose at start -- kept "
             "nonzero and FIXED so the energy tank sees genuine withdrawals (see module "
             "docstring)")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(
        args.output_dir or os.environ.get(
            "TELEOP_IMPEDANCE_PROBE_OUTPUT_DIR", "/tmp/franka_teleop_impedance_probe"))
    output_dir.mkdir(parents=True, exist_ok=True)

    rclpy.init()
    node = SinusoidProbeNode(args.namespace, args.controller)
    try:
        node.get_logger().info(
            f"Waiting for {node.namespace_ns}/current_pose (pose_broadcaster must be active)...")
        if not node.wait_for_current_pose():
            node.get_logger().error(
                "No current_pose received -- is validate_variable_impedance.launch.py up, "
                "and is pose_broadcaster active?")
            return 1

        node.get_logger().info(
            f"Probing for {args.duration}s at {args.freq}Hz, pose offset {args.pose_offset}...")
        node.run(args.duration, args.freq, args.pose_offset, args.publish_rate)
        node.get_logger().info(f"Done. {len(node.rows)} samples captured.")

        if len(node.rows) < 50:
            node.get_logger().error(
                f"Only {len(node.rows)} samples -- is variable_impedance_controller active "
                "and publishing ~/diagnostics/stiffness_applied?")
            return 1

        timestamp = int(time.time())
        csv_path = output_dir / f"sinusoid_probe_{timestamp}.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_FIELDS)
            writer.writerows(node.rows)
        node.get_logger().info(f"Wrote {csv_path}")

        fig_path = output_dir / f"sinusoid_probe_{timestamp}.png"
        make_figure(node.rows, fig_path)
        node.get_logger().info(f"Wrote {fig_path}")

        print(str(csv_path))
        print(str(fig_path))
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
