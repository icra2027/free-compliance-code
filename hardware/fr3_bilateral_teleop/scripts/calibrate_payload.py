#!/usr/bin/env python3
"""Identify an undeclared rigid-body payload (mass, COM, inertia) without weighing it.

Moves the arm through a set of static joint-space poses (via move_to_start_example_controller,
retargeted between waypoints) plus the point-to-point transits between them, logging measured
joint torque (tau_J) alongside franka::Model's own gravity/Coriolis/mass-matrix/Jacobian
predictions -- evaluated at whatever payload is CURRENTLY configured via set_load, i.e. NOT
including the new attachment. The residual between measured and predicted torque is therefore
explained entirely by the new attachment, and is regressed against it in two stages:

  1. Mass + center of mass, from the static poses only (well-conditioned: dq = ddq = 0, so only
     the gravity term is at play). This is the dominant bias term the calibration exists for.
  2. Full inertia tensor, from the transit motions (approximate: numerically-differentiated
     acceleration, and the added body's own Coriolis coupling is neglected -- reasonable for a
     small, light, near-flange 3D-printed part, but meaningfully less certain than stage 1).
     Falls back to a solid-sphere point-mass approximation if the fit looks physically
     implausible or ill-conditioned.

PRECONDITION: whatever load is configured via set_load when the sweep starts is treated as the
KNOWN baseline -- franka::Model's convenience overloads evaluate gravity/Coriolis/mass-matrix at
that load, and the fit measures the RESIDUAL beyond it. By default that baseline is zero (a
truly undeclared attachment). If `<namespace>_payload.yaml` from a previous run already exists
in the output directory, this script now applies it as a PRELOAD via set_load before sweeping
(override with --preload-from, disable with --no-preload) -- this exists because
move_to_start_example_controller relies on the robot's own internal gravity compensation once a
waypoint's motion reports "finished" (see update()'s zero-torque branch), and with a truly
undeclared payload attached, that compensation is wrong by the full weight of the attachment,
which given several seconds of holding (this script's settle+sample window) is enough to sag
into a joint_velocity_violation reflex -- found on real hardware 2026-08-10 recalibrating with
the eraser/wiper mount attached. A preload close to correct (e.g. a stale prior calibration)
keeps the hold stable. The fit then measures mass/COM/inertia DELTA relative to the preload,
and this script composes preload + delta into the correct absolute total (mass adds directly;
COM via a mass-weighted average; inertia via the parallel-axis theorem, shifting both
sub-bodies' own-COM inertia to the combined COM before summing) -- see compose_rigid_bodies.
With no preload (fresh, first-ever calibration), delta IS the total and nothing changes.

Usage (after `ros2 launch fr3_bilateral_teleop calibrate_payload.launch.py namespace:=follower`):
    ros2 run fr3_bilateral_teleop calibrate_payload.py --namespace follower
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from controller_manager_msgs.srv import SwitchController
from rcl_interfaces.srv import GetParameters, SetParameters
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType
from franka_msgs.srv import SetLoad

try:
    from ament_index_python.packages import get_package_share_directory
except ImportError:  # pragma: no cover - only used for the default waypoints path
    get_package_share_directory = None

MOVE_TO_START = "move_to_start_example_controller"


def sanitize_namespace(namespace: str) -> str:
    """Collapse a (possibly nested, e.g. 'franka_teleop/follower') ROS namespace into a
    filesystem-safe name -- used for filenames only, never for topic/service names."""
    return namespace.strip("/").replace("/", "_")


GRAVITY_VEC = np.array([0.0, 0.0, 9.81])
# Sign verified empirically against franka::Model::gravity()'s actual response to set_load
# (see scripts/diagnose_gravity_preload.py) -- real-hardware 2026-08-10: the naive downward
# [0, 0, -9.81] convention produced a mass regressor exactly the negative of the robot's real
# response (per-joint delta ratios all ~-1.0 across four calibration runs at different preloads),
# which silently flipped the sign of the fitted mass term (delta_com came out fine since it's a
# ratio and the sign cancels there, but the fitted mass/total_mass composition was wrong by
# roughly `2*preload_mass` every time -- see diagnose_gravity_preload.py's docstring for the
# full derivation). Do not "fix" this back to -9.81 without re-running that diagnostic.
SNAPSHOT_LENGTH = 113  # must match PayloadModelBroadcaster::kSnapshotLength


def skew(v: np.ndarray) -> np.ndarray:
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])


def quat_to_rotmat(x: float, y: float, z: float, w: float) -> np.ndarray:
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array([
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)],
    ])


def parallel_axis_shift(inertia: np.ndarray, mass: float, d: np.ndarray) -> np.ndarray:
    """Shift a 3x3 inertia tensor, currently expressed about a body's own center of mass, to
    be expressed about a new reference point offset by d = new_point - center_of_mass (the
    generalized parallel axis theorem, in tensor form rather than the scalar I = I_cm + m*d^2
    form usually taught first)."""
    d = np.asarray(d, dtype=float)
    return inertia + mass * (np.dot(d, d) * np.eye(3) - np.outer(d, d))


def compose_rigid_bodies(
    m1: float, com1: np.ndarray, inertia1: np.ndarray,
    m2: float, com2: np.ndarray, inertia2: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Combine two rigid sub-bodies -- each given as (mass, center of mass, inertia ABOUT ITS
    OWN center of mass, matching Franka's set_load/load_inertia convention) -- into the single
    equivalent rigid body they'd add up to if both were rigidly attached at the same point:
    total mass (sums directly), the mass-weighted-average center of mass, and the total
    inertia about that combined center of mass (shift each sub-body's own-COM inertia to the
    combined COM via the parallel axis theorem, then sum -- summing the two inputs' inertia
    tensors directly, without this shift, is only valid if they already share a COM).

    Used to recover calibrate_payload.py's true absolute mass/COM/inertia when a preload was
    applied via set_load before the sweep: the regression fits the DELTA relative to whatever
    was already configured, and this composes preload + delta back into the total. Verified
    against a hand-derived case (two equal point masses symmetric about the origin) and a
    degenerate case (splitting one body into two identical co-located halves must reproduce it
    exactly) during development -- see the Day-3 (2026-08-10) real-hardware notes in tasks.md.
    """
    total_mass = m1 + m2
    if abs(total_mass) < 1e-9:
        return 0.0, np.zeros(3), np.zeros((3, 3))
    com1, com2 = np.asarray(com1, dtype=float), np.asarray(com2, dtype=float)
    total_com = (m1 * com1 + m2 * com2) / total_mass
    inertia1_at_total_com = parallel_axis_shift(inertia1, m1, total_com - com1)
    inertia2_at_total_com = parallel_axis_shift(inertia2, m2, total_com - com2)
    return total_mass, total_com, inertia1_at_total_com + inertia2_at_total_com


def parse_snapshot(data: List[float]) -> Dict[str, np.ndarray]:
    """Inverse of PayloadModelBroadcaster's packing -- see payload_model_broadcaster.hpp."""
    if len(data) != SNAPSHOT_LENGTH:
        raise ValueError(f"model_snapshot has {len(data)} elements, expected {SNAPSHOT_LENGTH}")
    arr = np.asarray(data, dtype=float)
    stamp = arr[0]
    gravity = arr[1:8]
    coriolis = arr[8:15]
    mass = arr[15:64].reshape(7, 7).T  # column-major -> transpose of row-major reshape
    jacobian = arr[64:106].reshape(7, 6).T  # (6,7): rows = linear(0:3)/angular(3:6)
    flange_pos = arr[106:109]
    flange_quat = arr[109:113]  # x, y, z, w
    return {
        "stamp": stamp,
        "gravity": gravity,
        "coriolis": coriolis,
        "mass": mass,
        "jacobian": jacobian,
        "flange_pos": flange_pos,
        "flange_quat": flange_quat,
        "R": quat_to_rotmat(*flange_quat),
    }


def gravity_regressor_row(jacobian: np.ndarray, R: np.ndarray) -> np.ndarray:
    """7x4 regressor mapping [m, m*cx, m*cy, m*cz] -> added-body gravity torque at this q."""
    j_v = jacobian[0:3, :]
    j_w = jacobian[3:6, :]
    row = np.zeros((7, 4))
    row[:, 0] = j_v.T @ GRAVITY_VEC
    row[:, 1:4] = -(j_w.T @ skew(GRAVITY_VEC) @ R)
    return row


INERTIA_BASIS = [
    np.diag([1.0, 0.0, 0.0]),
    np.diag([0.0, 1.0, 0.0]),
    np.diag([0.0, 0.0, 1.0]),
    np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
]  # Ixx, Iyy, Izz, Ixy, Ixz, Iyz


def inertia_regressor_row(jacobian: np.ndarray, R: np.ndarray, ddq: np.ndarray) -> np.ndarray:
    """Build the 7x6 regressor mapping the 6 independent inertia components to torque.

    Maps to the added-body torque contribution M_add(q) @ ddq, given the body's mass/COM
    contribution has already been subtracted from the residual elsewhere.
    """
    j_w = jacobian[3:6, :]
    row = np.zeros((7, 6))
    for k, basis in enumerate(INERTIA_BASIS):
        row[:, k] = (j_w.T @ (R @ basis @ R.T) @ j_w) @ ddq
    return row


class PayloadCalibrationNode(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("calibrate_payload")
        self.ns = args.namespace.strip("/")
        self.file_ns = sanitize_namespace(args.namespace)
        self.args = args

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=200)

        self._latest_state: Optional[Dict] = None
        self._static_log: List[Dict] = []
        self._transit_log: List[Dict] = []
        self._static_logging = False
        self._transit_logging = False

        self.create_subscription(
            JointState, f"/{self.ns}/franka_robot_state_broadcaster/measured_joint_states",
            self._state_cb, qos)
        self.create_subscription(
            Float64MultiArray, f"/{self.ns}/payload_model_broadcaster/model_snapshot",
            self._snapshot_cb, qos)

        self.switch_client = self.create_client(
            SwitchController, f"/{self.ns}/controller_manager/switch_controller")
        self.get_param_client = self.create_client(
            GetParameters, f"/{self.ns}/{MOVE_TO_START}/get_parameters")
        self.set_param_client = self.create_client(
            SetParameters, f"/{self.ns}/{MOVE_TO_START}/set_parameters")
        self.set_load_client = self.create_client(
            SetLoad, f"/{self.ns}/service_server/set_load")

    def _state_cb(self, msg: JointState) -> None:
        self._latest_state = {
            "q": np.asarray(msg.position, dtype=float),
            "dq": np.asarray(msg.velocity, dtype=float),
            "tau": np.asarray(msg.effort, dtype=float),
            "stamp": msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
        }

    def _snapshot_cb(self, msg: Float64MultiArray) -> None:
        if not (self._static_logging or self._transit_logging):
            return
        if self._latest_state is None:
            return
        try:
            snapshot = parse_snapshot(list(msg.data))
        except ValueError as exc:
            self.get_logger().warn(f"Dropping malformed model_snapshot: {exc}")
            return
        # Pair with the most recent joint-state sample; both broadcasters read the same
        # underlying robot_state once per controller_manager cycle, so skew is at most a
        # cycle or two -- reject anything looser than that instead of silently pairing stale data.
        if abs(self._latest_state["stamp"] - snapshot["stamp"]) > 0.05:
            return
        record = dict(snapshot)
        record.update(self._latest_state)
        if self._static_logging:
            self._static_log.append(record)
        if self._transit_logging:
            self._transit_log.append(record)

    def wait_for_services(self, timeout_sec: float = 15.0) -> bool:
        for client, name in (
            (self.switch_client, "switch_controller"),
            (self.get_param_client, "get_parameters"),
            (self.set_param_client, "set_parameters"),
            (self.set_load_client, "set_load"),
        ):
            if not client.wait_for_service(timeout_sec=timeout_sec):
                self.get_logger().error(f"Service {name} not available on namespace {self.ns}")
                return False
        return True

    def call_set_load(self, mass: float, com: np.ndarray, inertia: np.ndarray) -> bool:
        req = SetLoad.Request()
        req.mass = float(mass)
        req.center_of_mass = [float(c) for c in com]
        req.load_inertia = [float(v) for v in inertia.flatten(order="F")]
        future = self.set_load_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        result = future.result()
        if result is None:
            self.get_logger().error("set_load call did not return a response")
            return False
        if not result.success:
            self.get_logger().error(f"set_load rejected: {result.error}")
            return False
        return True

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

    def wait_until_process_finished(self, timeout_sec: float) -> bool:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            req = GetParameters.Request()
            req.names = ["process_finished"]
            future = self.get_param_client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
            result = future.result()
            if result is not None and result.values and result.values[0].bool_value:
                return True
            rclpy.spin_once(self, timeout_sec=0.1)
        return False

    def sample_for(self, duration_sec: float, log: List[Dict], enable_flag: str) -> None:
        setattr(self, enable_flag, True)
        log.clear()
        deadline = time.monotonic() + duration_sec
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
        setattr(self, enable_flag, False)

    def wait_for_current_joint_state(self, timeout_sec: float = 5.0) -> Optional[np.ndarray]:
        deadline = time.monotonic() + timeout_sec
        while self._latest_state is None and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        return None if self._latest_state is None else np.asarray(self._latest_state["q"])

    def preflight_delta_check(self, waypoint: List[float], max_delta_rad: float) -> bool:
        """Refuses to activate move_to_start_example_controller toward `waypoint` if the arm's
        CURRENT measured joint state is already too far from it.

        move_to_start_example_controller is a plain joint-space PD controller
        (k_gains * (q_desired - q) + ...), not a velocity-limited trajectory generator -- it
        was designed and tested for this file's small (<=0.5 rad) single-joint transits, where
        a small position error naturally keeps the commanded velocity small. If the arm is
        instead sitting somewhere unrelated (e.g. left there by manual jogging, or by an
        earlier aborted transit), the same PD law sees a large error and can command a
        velocity spike large enough to trip a joint_velocity_violation reflex -- exactly the
        failure mode found during Day-3 real-hardware testing (2026-08-10): the very FIRST
        waypoint faulted this way when the arm wasn't already near the base pose, and every
        retry after that started from an equally arbitrary post-fault position, reproducing
        the same fault for what looked like unrelated reasons. Catching this before
        activation -- loudly, before commanding anything -- is cheap and turns a reflex fault
        into a clear instruction instead.
        """
        current_q = self.wait_for_current_joint_state()
        if current_q is None:
            self.get_logger().error(
                "No joint state received -- cannot verify the arm is near the target before "
                "activating move_to_start_example_controller. Is measured_joint_states "
                "publishing?")
            return False
        target_q = np.asarray(waypoint)
        delta = np.abs(current_q - target_q)
        max_delta = float(np.max(delta))
        if max_delta > max_delta_rad:
            worst_joint = int(np.argmax(delta)) + 1
            self.get_logger().error(
                f"Refusing to activate move_to_start_example_controller: joint{worst_joint} is "
                f"{max_delta:.3f} rad ({np.degrees(max_delta):.1f} deg) from the target "
                f"{target_q.tolist()}, exceeding --max-preflight-delta-rad={max_delta_rad}. "
                f"move_to_start_example_controller is a plain PD controller, not a "
                f"velocity-limited trajectory generator, and a jump this large can trip a "
                f"joint_velocity_violation reflex. Manually jog the arm closer to this target "
                f"via Desk first, then re-run.")
            return False
        return True


def average_record(records: List[Dict]) -> Dict:
    keys = ["q", "gravity", "coriolis", "mass", "jacobian", "flange_pos"]
    out = {k: np.mean(np.stack([r[k] for r in records]), axis=0) for k in keys}
    # Average quaternions properly is fiddly; average the rotation matrices instead (fine for
    # the small intra-hold drift this is meant to smooth) and don't bother renormalizing --
    # gravity_regressor_row only needs R to be close, not exactly orthonormal, since it's
    # multiplying an already-small correction term.
    out["R"] = np.mean(np.stack([r["R"] for r in records]), axis=0)
    out["tau"] = np.mean(np.stack([r["tau"] for r in records]), axis=0)
    return out


def fit_mass_com(static_records: List[Dict]) -> Dict:
    rows_A, rows_b = [], []
    for rec in static_records:
        A = gravity_regressor_row(rec["jacobian"], rec["R"])
        b = rec["tau"] - rec["gravity"]
        rows_A.append(A)
        rows_b.append(b)
    A = np.concatenate(rows_A, axis=0)
    b = np.concatenate(rows_b, axis=0)
    theta, _, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    mass = theta[0]
    com = theta[1:4] / mass if abs(mass) > 1e-9 else np.zeros(3)
    residual_rms = float(np.sqrt(np.mean((A @ theta - b) ** 2)))
    condition_number = float(sv[0] / sv[-1]) if sv[-1] > 1e-12 else float("inf")
    return {
        "mass": float(mass),
        "center_of_mass": com.tolist(),
        "rank": int(rank),
        "condition_number": condition_number,
        "residual_rms_nm": residual_rms,
        "num_static_poses": len(static_records),
    }


def numerically_differentiate(transit_segments: List[List[Dict]]) -> List[Dict]:
    """Smooth dq with a short moving average, then central-difference it into ddq.

    Done per segment, never across segment boundaries -- those are separate
    point-to-point motions.
    """
    out = []
    for segment in transit_segments:
        if len(segment) < 5:
            continue  # too short to differentiate meaningfully
        segment = sorted(segment, key=lambda r: r["stamp"])
        t = np.array([r["stamp"] for r in segment])
        dq = np.stack([r["dq"] for r in segment])
        window = 5
        kernel = np.ones(window) / window
        dq_smooth = np.apply_along_axis(
            lambda col: np.convolve(col, kernel, mode="same"), axis=0, arr=dq)
        ddq = np.gradient(dq_smooth, t, axis=0)
        for i, rec in enumerate(segment):
            merged = dict(rec)
            merged["ddq"] = ddq[i]
            out.append(merged)
    return out


def fit_inertia(transit_samples: List[Dict], mass: float, com: np.ndarray) -> Dict:
    rows_B, rows_y = [], []
    for rec in transit_samples:
        A_gc = gravity_regressor_row(rec["jacobian"], rec["R"])
        theta_gc = np.array([mass, mass * com[0], mass * com[1], mass * com[2]])
        tau_grav_add = A_gc @ theta_gc

        j_v = rec["jacobian"][0:3, :]
        j_w = rec["jacobian"][3:6, :]
        r_vec = rec["R"] @ com
        j_v_com = j_v - skew(r_vec) @ j_w
        known_term = mass * (j_v_com.T @ j_v_com) @ rec["ddq"]

        residual = (
            rec["tau"] - rec["gravity"] - tau_grav_add
            - rec["mass"] @ rec["ddq"] - rec["coriolis"] - known_term
        )
        rows_B.append(inertia_regressor_row(rec["jacobian"], rec["R"], rec["ddq"]))
        rows_y.append(residual)

    B = np.concatenate(rows_B, axis=0)
    y = np.concatenate(rows_y, axis=0)
    theta, _, rank, sv = np.linalg.lstsq(B, y, rcond=None)
    ixx, iyy, izz, ixy, ixz, iyz = theta
    inertia = np.array([
        [ixx, ixy, ixz],
        [ixy, iyy, iyz],
        [ixz, iyz, izz],
    ])
    residual_rms = float(np.sqrt(np.mean((B @ theta - y) ** 2)))
    condition_number = float(sv[0] / sv[-1]) if sv[-1] > 1e-12 else float("inf")
    return {
        "inertia": inertia,
        "rank": int(rank),
        "condition_number": condition_number,
        "residual_rms_nm": residual_rms,
        "num_transit_samples": len(transit_samples),
    }


def sphere_fallback_inertia(mass: float, radius_m: float) -> np.ndarray:
    i = (2.0 / 5.0) * mass * radius_m ** 2
    return np.diag([i, i, i])


def mass_com_gate_passed(
        fit: Dict, total_mass: float, total_com: np.ndarray, args: argparse.Namespace) -> bool:
    """Conditioning checks (rank, condition_number) apply to the raw regression fit -- a DELTA
    relative to whatever preload was configured, which can legitimately be small or even
    negative (if the true tool is lighter than the preload guess) without that meaning
    anything is wrong. Physical-plausibility checks (mass positive and bounded, COM bounded)
    apply to the composed TOTAL instead, since those bounds describe the actual physical tool,
    not the delta from an arbitrary baseline."""
    if fit["rank"] < 4:
        return False
    if fit["condition_number"] > args.max_condition_number:
        return False
    if not (0.0 < total_mass <= args.max_mass):
        return False
    if any(abs(c) > args.max_com for c in total_com):
        return False
    return True


def inertia_gate_passed(fit: Dict, total_inertia: np.ndarray, args: argparse.Namespace) -> bool:
    """Same split as mass_com_gate_passed: conditioning on the raw (delta) fit, physical
    plausibility (positive-semidefinite, bounded diagonal) on the composed total."""
    if fit["rank"] < 6:
        return False
    if fit["condition_number"] > args.max_condition_number:
        return False
    eigenvalues = np.linalg.eigvalsh(total_inertia)
    if np.any(eigenvalues < -1e-5):
        return False  # physically invalid: negative principal moment of inertia
    if np.any(np.diag(total_inertia) > args.max_inertia):
        return False
    return True


def default_waypoints_path() -> Path:
    if get_package_share_directory is not None:
        try:
            return Path(get_package_share_directory("fr3_bilateral_teleop")) / "config" / \
                "payload_calibration_waypoints.yaml"
        except Exception:
            pass
    return Path(__file__).resolve().parent.parent / "config" / \
        "payload_calibration_waypoints.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--namespace", default="follower")
    parser.add_argument("--waypoints-file", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--settle-time", type=float, default=2.0)
    parser.add_argument("--sample-time", type=float, default=1.5)
    parser.add_argument("--activation-timeout", type=float, default=20.0)
    parser.add_argument("--skip-inertia", action="store_true")
    parser.add_argument("--max-mass", type=float, default=2.0, help="kg, sanity bound")
    parser.add_argument("--max-com", type=float, default=0.15, help="m, sanity bound per axis")
    parser.add_argument("--max-inertia", type=float, default=0.01, help="kg*m^2, sanity bound")
    parser.add_argument("--max-condition-number", type=float, default=1e4)
    parser.add_argument(
        "--fallback-sphere-radius", type=float, default=0.03,
        help="m, used for the inertia fallback if the dynamic fit is rejected")
    parser.add_argument(
        "--max-preflight-delta-rad", type=float, default=0.6,
        help="refuse to activate move_to_start_example_controller toward a waypoint if any "
             "joint's current measured position is farther than this from the target (rad); "
             "see preflight_delta_check for why")
    parser.add_argument(
        "--preload-from", default=None,
        help="path to an existing <namespace>_payload.yaml to apply via set_load before "
             "sweeping, so move_to_start_example_controller's holds don't sag under an "
             "undeclared payload (see module docstring). Defaults to "
             "<output-dir>/<namespace>_payload.yaml if it exists.")
    parser.add_argument(
        "--no-preload", action="store_true",
        help="disable preloading even if a previous payload file is found -- use for a "
             "genuinely from-scratch calibration (e.g. first-ever run, or the arm is already "
             "confirmed stable near the target without one)")
    parser.add_argument(
        "--preload-mass", type=float, default=None,
        help="apply this mass (kg) as a manual preload via set_load before sweeping, instead "
             "of reading one from a previous payload file -- e.g. a rough scale weighing of "
             "the attachment. Takes priority over --preload-from / auto-detection. COM "
             "defaults to zero (flange origin) unless --preload-com is also given; a rough "
             "mass alone is what matters for keeping move_to_start_example_controller's holds "
             "from sagging, precise COM is not needed for a preload.")
    parser.add_argument(
        "--preload-com", type=float, nargs=3, default=[0.0, 0.0, 0.0], metavar=("X", "Y", "Z"),
        help="center of mass (m, flange frame) to pair with --preload-mass")
    return parser.parse_args()


def load_preload(path: Path) -> Tuple[float, np.ndarray, np.ndarray]:
    """Reads a previous calibrate_payload.py report's set_load block back into
    (mass, center_of_mass, inertia-as-3x3) for use as a preload."""
    report = yaml.safe_load(path.read_text())
    set_load = report["set_load"]
    mass = float(set_load["mass"])
    com = np.asarray(set_load["center_of_mass"], dtype=float)
    inertia = np.asarray(set_load["load_inertia"], dtype=float).reshape(3, 3, order="F")
    return mass, com, inertia


def main() -> int:
    args = parse_args()
    waypoints_path = Path(args.waypoints_file) if args.waypoints_file else default_waypoints_path()
    waypoints = yaml.safe_load(waypoints_path.read_text())["waypoints"]
    output_dir = Path(
        args.output_dir or __import__("os").environ.get(
            "TELEOP_PAYLOAD_OUTPUT_DIR", "/tmp/franka_teleop_payload"))
    output_dir.mkdir(parents=True, exist_ok=True)

    rclpy.init()
    node = PayloadCalibrationNode(args)

    try:
        if not node.wait_for_services():
            return 1

        preload_mass, preload_com, preload_inertia = 0.0, np.zeros(3), np.zeros((3, 3))
        preload_path_str: Optional[str] = None
        if args.preload_mass is not None:
            preload_mass = args.preload_mass
            preload_com = np.asarray(args.preload_com, dtype=float)
            preload_inertia = sphere_fallback_inertia(preload_mass, args.fallback_sphere_radius)
            node.get_logger().info(
                f"Applying manually-specified preload: mass={preload_mass:.4f} kg, "
                f"com={preload_com.tolist()} (--preload-mass/--preload-com) -- so "
                f"move_to_start_example_controller's post-'finished' holds don't sag under an "
                f"undeclared payload. The fit below measures the DELTA from this preload; "
                f"final numbers are composed back into an absolute total.")
            if not node.call_set_load(preload_mass, preload_com, preload_inertia):
                node.get_logger().error("Failed to apply preload via set_load -- aborting.")
                return 1
            preload_path_str = "--preload-mass (manual)"
        elif not args.no_preload:
            preload_path = Path(args.preload_from) if args.preload_from else (
                output_dir / f"{node.file_ns}_payload.yaml")
            if preload_path.exists():
                try:
                    preload_mass, preload_com, preload_inertia = load_preload(preload_path)
                except (OSError, yaml.YAMLError, KeyError) as exc:
                    node.get_logger().error(
                        f"Failed to read preload file {preload_path}: {exc}. Refusing to "
                        f"guess -- pass --no-preload to proceed with a zero baseline instead "
                        f"(only safe if the arm is already confirmed stable near the target).")
                    return 1
                node.get_logger().info(
                    f"Applying preload from {preload_path}: mass={preload_mass:.4f} kg, "
                    f"com={preload_com.tolist()} -- so move_to_start_example_controller's "
                    f"post-'finished' holds don't sag under an undeclared payload. The fit "
                    f"below measures the DELTA from this preload; final numbers are composed "
                    f"back into an absolute total.")
                if not node.call_set_load(preload_mass, preload_com, preload_inertia):
                    node.get_logger().error("Failed to apply preload via set_load -- aborting.")
                    return 1
                preload_path_str = str(preload_path)
            elif args.preload_from:
                node.get_logger().error(f"--preload-from {args.preload_from} does not exist")
                return 1
            else:
                node.get_logger().info(
                    "No existing payload file found -- calibrating from a zero baseline "
                    "(fresh/first-ever calibration).")

        static_records = []
        transit_segments = []

        for i, waypoint in enumerate(waypoints):
            node.get_logger().info(f"Waypoint {i + 1}/{len(waypoints)}: {waypoint}")
            if i > 0:
                if not node.switch_controller([], [MOVE_TO_START]):
                    node.get_logger().error(
                        "Failed to deactivate move_to_start_example_controller")
                    return 1

            if not node.set_start_joint_configuration(waypoint):
                node.get_logger().error("Failed to set start_joint_configuration")
                return 1

            if not node.preflight_delta_check(waypoint, args.max_preflight_delta_rad):
                return 1

            if i > 0:
                node._transit_log = []
                node._transit_logging = True

            if not node.switch_controller([MOVE_TO_START], []):
                node.get_logger().error("Failed to activate move_to_start_example_controller")
                return 1

            reached = node.wait_until_process_finished(args.activation_timeout)

            if i > 0:
                node._transit_logging = False
                if reached and len(node._transit_log) >= 5:
                    transit_segments.append(list(node._transit_log))

            if not reached:
                node.get_logger().error(
                    f"Waypoint {i + 1} did not report process_finished within "
                    f"{args.activation_timeout}s -- aborting rather than sampling a still-moving "
                    "arm.")
                return 1

            time.sleep(args.settle_time)
            node.sample_for(args.sample_time, node._static_log, "_static_logging")
            if len(node._static_log) < 5:
                node.get_logger().error(
                    f"Only {len(node._static_log)} samples collected at waypoint {i + 1} -- "
                    "check that payload_model_broadcaster and franka_robot_state_broadcaster "
                    "are both running.")
                return 1
            static_records.append(average_record(node._static_log))

        node.get_logger().info("Waypoint sweep complete, fitting mass + center of mass...")
        mass_com_fit = fit_mass_com(static_records)
        delta_mass = mass_com_fit["mass"]
        delta_com = np.array(mass_com_fit["center_of_mass"])
        # Mass/COM composition doesn't need real inertia inputs; pass zeros and discard the
        # (meaningless at this point) returned inertia -- the real composition happens below,
        # once an inertia estimate (fitted or fallback) actually exists.
        total_mass, total_com, _ = compose_rigid_bodies(
            preload_mass, preload_com, np.zeros((3, 3)), delta_mass, delta_com, np.zeros((3, 3)))
        mass_com_ok = mass_com_gate_passed(mass_com_fit, total_mass, total_com, args)
        node.get_logger().info(
            f"delta_mass={delta_mass:.4f} kg, delta_com={delta_com.tolist()} (relative to "
            f"preload_mass={preload_mass:.4f} kg) -- total_mass={total_mass:.4f} kg, "
            f"total_com={total_com.tolist()}, "
            f"condition_number={mass_com_fit['condition_number']:.1f}, "
            f"residual_rms={mass_com_fit['residual_rms_nm']:.4f} Nm, gate_passed={mass_com_ok}")

        report = {
            "namespace": node.ns,
            "generated_at_unix": time.time(),
            "preload": {
                "path": preload_path_str,
                "mass": preload_mass,
                "center_of_mass": preload_com.tolist(),
                "load_inertia": preload_inertia.flatten(order="F").tolist(),
            },
            "mass_com_delta_fit": mass_com_fit,
            "quality_gates": {"mass_com_passed": mass_com_ok},
        }

        if not mass_com_ok:
            node.get_logger().error(
                "Mass/COM fit failed its sanity gate -- refusing to write a payload file. "
                "Check waypoint diversity, that the arm actually reached each pose, and that "
                "no external contact happened during calibration.")
            report["status"] = "failed"
            report_path = output_dir / f"{node.file_ns}_payload_report.json"
            report_path.write_text(json.dumps(report, indent=2))
            return 1

        inertia_source = "fallback_sphere_approx"
        # The sphere fallback is a crude placeholder for the WHOLE tool, not a delta -- compute
        # it against the composed total mass directly rather than composing it with the preload.
        total_inertia = sphere_fallback_inertia(total_mass, args.fallback_sphere_radius)
        inertia_fit_report = None

        if not args.skip_inertia and transit_segments:
            transit_samples = numerically_differentiate(transit_segments)
            if len(transit_samples) >= 20:
                # fit_inertia regresses the DELTA body's own contribution (like fit_mass_com),
                # so it needs the delta mass/COM, not the composed total.
                inertia_fit = fit_inertia(transit_samples, delta_mass, delta_com)
                _, _, composed_inertia = compose_rigid_bodies(
                    preload_mass, preload_com, preload_inertia,
                    delta_mass, delta_com, inertia_fit["inertia"])
                inertia_ok = inertia_gate_passed(inertia_fit, composed_inertia, args)
                node.get_logger().info(
                    f"inertia condition_number={inertia_fit['condition_number']:.1f}, "
                    f"residual_rms={inertia_fit['residual_rms_nm']:.4f} Nm, "
                    f"gate_passed={inertia_ok}")
                inertia_fit_report = {
                    "condition_number": inertia_fit["condition_number"],
                    "residual_rms_nm": inertia_fit["residual_rms_nm"],
                    "num_transit_samples": inertia_fit["num_transit_samples"],
                    "gate_passed": inertia_ok,
                }
                if inertia_ok:
                    inertia_source = "fitted"
                    total_inertia = composed_inertia
                else:
                    node.get_logger().warn(
                        "Inertia fit failed its sanity gate -- falling back to a solid-sphere "
                        f"point-mass approximation (radius={args.fallback_sphere_radius} m). "
                        "This is a crude placeholder, not a measurement.")
            else:
                node.get_logger().warn(
                    "Not enough transit samples for an inertia fit -- falling back to a "
                    "solid-sphere approximation.")
        else:
            node.get_logger().info(
                "Inertia fit skipped (--skip-inertia or no transit data) -- using a "
                "solid-sphere approximation.")

        # symmetric: row/col-major agree
        load_inertia_flat = total_inertia.flatten(order="F").tolist()

        report.update({
            "status": "ok",
            "inertia_source": inertia_source,
            "inertia_fit": inertia_fit_report,
            "set_load": {
                "mass": total_mass,
                "center_of_mass": total_com.tolist(),
                "load_inertia": load_inertia_flat,
            },
        })

        payload_path = output_dir / f"{node.file_ns}_payload.yaml"
        payload_path.write_text(yaml.dump(report, default_flow_style=False, sort_keys=False))
        node.get_logger().info(f"Wrote {payload_path}")

        history_dir = output_dir / "history"
        history_dir.mkdir(exist_ok=True)
        history_path = history_dir / f"{node.file_ns}_{int(time.time())}.yaml"
        history_path.write_text(yaml.dump(report, default_flow_style=False, sort_keys=False))

        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
