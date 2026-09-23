#!/usr/bin/env python3
"""Collect (q, q̇, estimated external wrench) samples in free space for residual bias fitting.

Day 3 of the ICRA27 plan: the Panda's estimated external wrench still carries a
configuration- and velocity-dependent bias after payload identification (Day 2) and
per-session re-zero -- friction and model error that a single free-space-pose bias sample
does not capture. This script drives the follower through free-space motion spanning the
intended working volume at several speeds and logs (q, q̇, wrench) throughout, both during
the point-to-point transits (q̇ != 0) and the static holds between them (q̇ ~= 0). Since
there is no contact anywhere in this sweep, every non-zero wrench sample IS the bias --
fit_residual_bias.py regresses f_bias(q, q̇) directly against this log, no separate
ground-truth force channel needed.

Reuses the same move_to_start_example_controller retargeting mechanism as
calibrate_payload.py (see that script and fr3_bilateral_teleop/README.md "Payload calibration"
for why: it's the one controller already proven safe for repeated unattended point-to-point
motion on this rig). Deliberately does NOT use payload_model_broadcaster's model snapshot --
unlike calibrate_payload.py (which runs before set_load is configured and has to reconstruct
the residual from franka::Model's gravity/Coriolis/mass predictions itself), this script runs
AFTER Day 2's payload calibration is wired into teleop startup, so
franka_robot_state_broadcaster/external_wrench_in_base_frame is already the payload-corrected
signal the rest of the pipeline (re-zero, extraction) consumes. Logging it directly keeps this
script's job to exactly one thing: characterizing what's left after that correction.

PRECONDITION: run this with the follower's normal set_load payload already configured (i.e.
via the standard teleop.launch.py startup path, or by having already run calibrate_payload.py
once) -- this script fits the RESIDUAL after payload correction, not the payload itself.

Usage (after bringing up the follower, e.g. via calibrate_payload.launch.py or teleop.launch.py
with the leader side simply left unused):
    ros2 run fr3_bilateral_teleop collect_free_space_sweep.py --namespace franka_teleop/follower
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import yaml

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped
from controller_manager_msgs.srv import SwitchController
from rcl_interfaces.srv import GetParameters, SetParameters
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType

try:
    from ament_index_python.packages import get_package_share_directory
except ImportError:  # pragma: no cover - only used for the default waypoints path
    get_package_share_directory = None

MOVE_TO_START = "move_to_start_example_controller"
WRENCH_TOPIC = "franka_robot_state_broadcaster/external_wrench_in_base_frame"
JOINT_STATE_TOPIC = "franka_robot_state_broadcaster/measured_joint_states"

CSV_FIELDS = [
    "session_id", "speed_factor", "waypoint_index", "phase", "stamp",
    "q1", "q2", "q3", "q4", "q5", "q6", "q7",
    "dq1", "dq2", "dq3", "dq4", "dq5", "dq6", "dq7",
    "fx", "fy", "fz", "tx", "ty", "tz",
]


def sanitize_namespace(namespace: str) -> str:
    return namespace.strip("/").replace("/", "_")


class FreeSpaceSweepNode(Node):
    def __init__(self, namespace: str, pair_tolerance_sec: float):
        super().__init__("collect_free_space_sweep")
        self.ns = namespace.strip("/")
        self.pair_tolerance_sec = pair_tolerance_sec

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=200)

        self._latest_state: Optional[Dict] = None
        self._logging = False
        self._session_id = ""
        self._speed_factor = 0.0
        self._waypoint_index = 0
        self._phase = "static"
        self.rows: List[Dict] = []

        self.create_subscription(
            JointState, f"/{self.ns}/{JOINT_STATE_TOPIC}", self._state_cb, qos)
        self.create_subscription(
            WrenchStamped, f"/{self.ns}/{WRENCH_TOPIC}", self._wrench_cb, qos)

        self.switch_client = self.create_client(
            SwitchController, f"/{self.ns}/controller_manager/switch_controller")
        self.get_param_client = self.create_client(
            GetParameters, f"/{self.ns}/{MOVE_TO_START}/get_parameters")
        self.set_param_client = self.create_client(
            SetParameters, f"/{self.ns}/{MOVE_TO_START}/set_parameters")

    def _state_cb(self, msg: JointState) -> None:
        self._latest_state = {
            "q": list(msg.position),
            "dq": list(msg.velocity),
            "stamp": msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
        }

    def _wrench_cb(self, msg: WrenchStamped) -> None:
        if not self._logging or self._latest_state is None:
            return
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if abs(stamp - self._latest_state["stamp"]) > self.pair_tolerance_sec:
            return
        q, dq = self._latest_state["q"], self._latest_state["dq"]
        if len(q) != 7 or len(dq) != 7:
            return
        row = {
            "session_id": self._session_id,
            "speed_factor": self._speed_factor,
            "waypoint_index": self._waypoint_index,
            "phase": self._phase,
            "stamp": stamp,
            **{f"q{i + 1}": q[i] for i in range(7)},
            **{f"dq{i + 1}": dq[i] for i in range(7)},
            "fx": msg.wrench.force.x, "fy": msg.wrench.force.y, "fz": msg.wrench.force.z,
            "tx": msg.wrench.torque.x, "ty": msg.wrench.torque.y, "tz": msg.wrench.torque.z,
        }
        self.rows.append(row)

    def wait_for_services(self, timeout_sec: float = 15.0) -> bool:
        for client, name in (
            (self.switch_client, "switch_controller"),
            (self.get_param_client, "get_parameters"),
            (self.set_param_client, "set_parameters"),
        ):
            if not client.wait_for_service(timeout_sec=timeout_sec):
                self.get_logger().error(f"Service {name} not available on namespace {self.ns}")
                return False
        return True

    def set_speed_factor(self, speed_factor: float) -> bool:
        return self._set_double_param("speed_factor", speed_factor)

    def set_start_joint_configuration(self, waypoint: List[float]) -> bool:
        req = SetParameters.Request()
        param = Parameter()
        param.name = "start_joint_configuration"
        param.value = ParameterValue(
            type=ParameterType.PARAMETER_DOUBLE_ARRAY, double_array_value=list(waypoint))
        req.parameters = [param]
        future = self.set_param_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if future.result() is None:
            return False
        return all(r.successful for r in future.result().results)

    def _set_double_param(self, name: str, value: float) -> bool:
        req = SetParameters.Request()
        param = Parameter()
        param.name = name
        param.value = ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=value)
        req.parameters = [param]
        future = self.set_param_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if future.result() is None:
            return False
        return all(r.successful for r in future.result().results)

    def switch_controller(self, activate: List[str], deactivate: List[str]) -> bool:
        req = SwitchController.Request()
        req.activate_controllers = activate
        req.deactivate_controllers = deactivate
        req.strictness = SwitchController.Request.STRICT
        req.activate_asap = True
        req.timeout = rclpy.duration.Duration(seconds=5.0).to_msg()
        future = self.switch_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        if future.result() is None:
            return False
        return future.result().ok

    def _process_finished(self) -> bool:
        req = GetParameters.Request()
        req.names = ["process_finished"]
        future = self.get_param_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        result = future.result()
        return result is not None and bool(result.values) and result.values[0].bool_value

    def log_until_finished(self, timeout_sec: float, poll_period_sec: float = 0.2) -> bool:
        """Log (phase="transit") while polling process_finished, until it reports done or
        timeout_sec elapses. Logging stays on throughout -- it's the transit itself we want,
        not just the moment it ends -- and stops as soon as the arm reports arrival rather than
        padding out to the full timeout."""
        self._phase = "transit"
        self._logging = True
        deadline = time.monotonic() + timeout_sec
        reached = False
        next_poll = time.monotonic()
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.02)
            if time.monotonic() >= next_poll:
                if self._process_finished():
                    reached = True
                    break
                next_poll = time.monotonic() + poll_period_sec
        self._logging = False
        return reached

    def log_for(self, duration_sec: float, phase: str) -> None:
        self._phase = phase
        self._logging = True
        deadline = time.monotonic() + duration_sec
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.02)
        self._logging = False


def default_waypoints_path() -> Path:
    if get_package_share_directory is not None:
        try:
            return Path(get_package_share_directory("fr3_bilateral_teleop")) / "config" / \
                "free_space_sweep_waypoints.yaml"
        except Exception:
            pass
    return Path(__file__).resolve().parent.parent / "config" / "free_space_sweep_waypoints.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--namespace", default="follower")
    parser.add_argument("--waypoints-file", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--settle-time", type=float, default=1.0)
    parser.add_argument(
        "--sample-time", type=float, default=1.5,
        help="static-hold logging window per waypoint, seconds")
    parser.add_argument("--activation-timeout", type=float, default=20.0)
    parser.add_argument(
        "--pair-tolerance", type=float, default=0.02,
        help="max q/wrench timestamp skew accepted, seconds")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    waypoints_path = Path(args.waypoints_file) if args.waypoints_file else default_waypoints_path()
    spec = yaml.safe_load(waypoints_path.read_text())
    waypoints = spec["waypoints"]
    speed_factors = spec["speed_factors"]

    output_dir = Path(
        args.output_dir or os.environ.get(
            "TELEOP_SWEEP_OUTPUT_DIR", "/tmp/franka_teleop_free_space_sweep"))
    output_dir.mkdir(parents=True, exist_ok=True)

    rclpy.init()
    node = FreeSpaceSweepNode(args.namespace, args.pair_tolerance)

    try:
        if not node.wait_for_services():
            return 1

        controller_active = False
        for speed_factor in speed_factors:
            node.get_logger().info(f"--- speed_factor={speed_factor} ---")
            node._speed_factor = float(speed_factor)
            node._session_id = f"speed_{speed_factor}"

            for i, waypoint in enumerate(waypoints):
                node._waypoint_index = i
                node.get_logger().info(f"Waypoint {i + 1}/{len(waypoints)}: {waypoint}")
                if controller_active:
                    if not node.switch_controller([], [MOVE_TO_START]):
                        node.get_logger().error(
                            "Failed to deactivate move_to_start_example_controller")
                        return 1
                    controller_active = False

                if i == 0:
                    # Only while inactive -- see README/Day-2 notes on why
                    # move_to_start_example_controller's parameters must not be touched while
                    # the controller is active and holding.
                    if not node.set_speed_factor(float(speed_factor)):
                        node.get_logger().error("Failed to set speed_factor")
                        return 1

                if not node.set_start_joint_configuration(waypoint):
                    node.get_logger().error("Failed to set start_joint_configuration")
                    return 1

                if not node.switch_controller([MOVE_TO_START], []):
                    node.get_logger().error("Failed to activate move_to_start_example_controller")
                    return 1
                controller_active = True

                reached = node.log_until_finished(args.activation_timeout)
                if not reached:
                    node.get_logger().error(
                        f"Waypoint {i + 1} did not report process_finished -- aborting rather "
                        "than sampling a still-moving arm as if it were a static hold.")
                    return 1

                time.sleep(args.settle_time)
                node.log_for(args.sample_time, phase="static")

            node.get_logger().info(
                f"speed_factor={speed_factor} pass complete, {len(node.rows)} samples so far")

        csv_name = f"{sanitize_namespace(args.namespace)}_free_space_sweep_{int(time.time())}.csv"
        csv_path = output_dir / csv_name
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(node.rows)
        node.get_logger().info(f"Wrote {len(node.rows)} samples to {csv_path}")
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
