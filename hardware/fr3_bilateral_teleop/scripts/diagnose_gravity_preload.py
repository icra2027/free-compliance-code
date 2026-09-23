#!/usr/bin/env python3
"""Isolates whether franka::Model's gravity() output actually responds to set_load the way
calibrate_payload.py assumes -- WITHOUT moving the robot.

Motivation: three real-hardware calibrate_payload.py runs at different --preload-mass values
(0.30, 0.18, 0.15 kg) produced wildly different total_mass estimates (0.28, 0.078, 0.023 kg)
for the same unchanged physical attachment, and two repeat runs at the same preload (0.18 kg)
agreed with each other to within ~3%. That combination -- reproducible, but dependent on
preload -- points at a systematic bug in the gravity-model chain rather than noise or a loose
cable (both of which would show up as poor repeatability at fixed preload, not as tight
repeatability with a preload-dependent trend).

This script isolates that chain directly: hold the arm still, call set_load twice with a known
mass delta, and compare the ACTUAL change in the model_snapshot's "gravity" field against the
CHANGE predicted analytically from the robot's own Jacobian at that pose. franka::Model::gravity()
is a deterministic function of q and the currently configured load -- not a noisy measurement --
so if set_load is really taking effect, actual and expected should match almost exactly. If they
don't, the bug is in the set_load -> robot_state.m_total/F_x_Ctotal -> gravity() chain, not in
calibrate_payload.py's regression math.

Usage (robot can be sitting anywhere static -- teleop.launch.py must NOT be running, same
precondition as calibrate_payload.py):
    python3 diagnose_gravity_preload.py --namespace franka_teleop/follower --test-mass 0.3
"""
import argparse
import sys
import time
from typing import List

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float64MultiArray
from franka_msgs.srv import SetLoad

from calibrate_payload import parse_snapshot, gravity_regressor_row, sphere_fallback_inertia


class GravityDiagnosticNode(Node):
    def __init__(self, namespace: str):
        super().__init__("diagnose_gravity_preload")
        self.ns = namespace.strip("/")
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=200)
        self._log: List[dict] = []
        self._logging = False
        self.create_subscription(
            Float64MultiArray, f"/{self.ns}/payload_model_broadcaster/model_snapshot",
            self._snapshot_cb, qos)
        self.set_load_client = self.create_client(
            SetLoad, f"/{self.ns}/service_server/set_load")

    def _snapshot_cb(self, msg: Float64MultiArray) -> None:
        if not self._logging:
            return
        try:
            self._log.append(parse_snapshot(list(msg.data)))
        except ValueError as exc:
            self.get_logger().warn(f"Dropping malformed model_snapshot: {exc}")

    def wait_for_service(self, timeout_sec: float = 15.0) -> bool:
        if not self.set_load_client.wait_for_service(timeout_sec=timeout_sec):
            self.get_logger().error(f"set_load service not available on namespace {self.ns}")
            return False
        return True

    def call_set_load(self, mass: float) -> bool:
        req = SetLoad.Request()
        req.mass = float(mass)
        req.center_of_mass = [0.0, 0.0, 0.0]
        req.load_inertia = [
            float(v) for v in sphere_fallback_inertia(mass, 0.03).flatten(order="F")
        ] if mass > 0 else [0.0] * 9
        future = self.set_load_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        result = future.result()
        if result is None or not result.success:
            self.get_logger().error(f"set_load failed: {result.error if result else 'no response'}")
            return False
        return True

    def sample_gravity(self, duration_sec: float) -> np.ndarray:
        self._log = []
        self._logging = True
        deadline = time.monotonic() + duration_sec
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
        self._logging = False
        if len(self._log) < 5:
            self.get_logger().error(f"Only {len(self._log)} model_snapshot samples -- is the "
                                     "broadcaster running?")
            return None
        return self._log

    def wait_settle(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default="follower")
    parser.add_argument("--test-mass", type=float, default=0.3,
                         help="kg, the mass delta to apply via set_load between the two samples")
    parser.add_argument("--settle-time", type=float, default=1.0)
    parser.add_argument("--sample-time", type=float, default=1.0)
    args = parser.parse_args()

    rclpy.init()
    node = GravityDiagnosticNode(args.namespace)
    try:
        if not node.wait_for_service():
            return 1

        node.get_logger().info("Setting baseline load (mass=0)...")
        if not node.call_set_load(0.0):
            return 1
        node.wait_settle(args.settle_time)
        baseline = node.sample_gravity(args.sample_time)
        if baseline is None:
            return 1

        node.get_logger().info(f"Setting test load (mass={args.test_mass} kg)...")
        if not node.call_set_load(args.test_mass):
            return 1
        node.wait_settle(args.settle_time)
        loaded = node.sample_gravity(args.sample_time)
        if loaded is None:
            return 1

        # Restore zero load so nothing is left misconfigured.
        node.call_set_load(0.0)

        gravity_baseline = np.mean(np.stack([r["gravity"] for r in baseline]), axis=0)
        gravity_loaded = np.mean(np.stack([r["gravity"] for r in loaded]), axis=0)
        actual_diff = gravity_loaded - gravity_baseline

        jacobian = np.mean(np.stack([r["jacobian"] for r in baseline]), axis=0)
        R = np.mean(np.stack([r["R"] for r in baseline]), axis=0)
        expected_diff = gravity_regressor_row(jacobian, R)[:, 0] * args.test_mass

        node.get_logger().info(f"actual   gravity delta (measured):   {actual_diff.tolist()}")
        node.get_logger().info(f"expected gravity delta (analytical): {expected_diff.tolist()}")
        rel_err = np.abs(actual_diff - expected_diff) / (np.abs(expected_diff) + 1e-6)
        node.get_logger().info(f"per-joint relative error: {rel_err.tolist()}")
        max_rel_err = float(np.max(rel_err))
        if max_rel_err < 0.05:
            node.get_logger().info(
                f"MATCH (max rel error {max_rel_err:.1%}) -- gravity() correctly responds to "
                "set_load. The bug is NOT in this chain; look elsewhere (e.g. friction/tau_J "
                "under different holding torques).")
        else:
            node.get_logger().error(
                f"MISMATCH (max rel error {max_rel_err:.1%}) -- gravity() is NOT responding to "
                "set_load the way calibrate_payload.py assumes. This is the bug.")
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
