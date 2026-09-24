#!/usr/bin/env python3
"""Validate the sensorless external-wrench estimate against known hanging masses.

The last step of the sensorless wrench calibration (in
priority order -- payload ID -> re-zero -> residual bias model -> workspace/nullspace
constraints -> THIS). Everything before this step corrects the estimate; this step is the
one independent check that it is actually correct, using ground truth the earlier steps
never had access to: a known, calibrated mass.

Physics -- deliberately simple, and that simplicity is the point: hang a known mass `m` off
the tool by any means (string, hook, calibration fixture) so it settles and hangs freely.
The *force* the mass exerts on the robot is `m * g` straight down, in the BASE frame, and
critically this does **not** depend on where on the tool the mass is attached or on the
arm's current pose -- unlike torque, which depends on the (unmeasured) lever arm from the
wrench reference point to the hang point and rotates with the arm. So this script validates
force channels (fx, fy, fz) against a clean predicted value at every pose; torque channels
(tx, ty, tz) are logged and reported descriptively only, with no independent ground truth
here (see the report's "torque_note").

This assumes the follower's BASE frame is gravity-aligned (mounted level, +Z up) --
`external_wrench_in_base_frame` is already expressed in that frame, so no per-pose rotation
is needed to predict the force. Override --gravity-vector-base if the base is known to be
tilted.

PRECONDITION: run this with the follower's normal tool payload already configured via
set_load (i.e. calibrate_payload.py has already run and teleop.launch.py's/
calibrate_payload.launch.py's startup has applied `<namespace>_payload.yaml`). This script
adds a SEPARATE, additional known mass on top of that -- it must never be declared to
set_load, or the comparison stops being independent (see module docstring above).

SAFETY (read before running against real hardware): each hold relies on the same
finished-state, zero-added-torque mechanism as calibrate_payload.py's and
collect_free_space_sweep.py's static holds -- i.e. it trusts the robot's OWN internal
gravity compensation, which does NOT include whatever extra mass you hang for this script
(by design -- see above). Earlier real-hardware sessions found that holding an undeclared payload this way for
several seconds is enough to sag into a joint_velocity_violation reflex once the undeclared
weight gets large enough. Start with the LIGHTEST calibrated mass first, watch Desk, and
only increase --masses once the lighter ones are confirmed to hold cleanly. Default
--sample-time is short (1.5 s) on purpose.

This is a semi-automated, human-in-the-loop script by necessity: nothing can hang a
calibrated mass on the robot but the operator. It moves the arm to each pose, then pauses
with an interactive prompt at every (pose, mass) combination.

Usage (after bringing up the follower with its normal payload configured, e.g. via
teleop.launch.py or calibrate_payload.launch.py with the leader side left unused):
    ros2 run fr3_bilateral_teleop static_hanging_mass_calibration.py --namespace follower
    ros2 run fr3_bilateral_teleop static_hanging_mass_calibration.py --self-test   # no hardware
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import rclpy  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy  # noqa: E402
from sensor_msgs.msg import JointState  # noqa: E402
from geometry_msgs.msg import WrenchStamped  # noqa: E402
from controller_manager_msgs.srv import SwitchController  # noqa: E402
from rcl_interfaces.srv import SetParameters  # noqa: E402
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType  # noqa: E402

try:
    from ament_index_python.packages import get_package_share_directory
except ImportError:  # pragma: no cover - only used for the default waypoints path
    get_package_share_directory = None

MOVE_TO_START = "move_to_start_example_controller"
WRENCH_TOPIC = "franka_robot_state_broadcaster/external_wrench_in_base_frame"
JOINT_STATE_TOPIC = "franka_robot_state_broadcaster/measured_joint_states"
DEFAULT_GRAVITY_VEC_BASE = np.array([0.0, 0.0, -9.81])


def sanitize_namespace(namespace: str) -> str:
    return namespace.strip("/").replace("/", "_")


def predicted_force(mass_kg: float, gravity_vec_base: np.ndarray) -> np.ndarray:
    """Force a freely-hanging mass exerts on the robot, in the base frame. Independent of
    pose and of where on the tool it's attached -- only torque depends on that lever arm."""
    return mass_kg * gravity_vec_base


def compute_calibration_stats(records: List[Dict], gravity_vec_base: np.ndarray) -> Dict:
    """Pure, offline-testable core: turn a list of per-(pose, mass) samples into bias +
    noise-floor (sigma_f) numbers for the force channels, plus descriptive-only torque
    stats. Each record needs: pose_index, pose_label, mass_kg, raw_forces (n, 3),
    mean_force (3,), within_hold_std_force (3,), mean_torque (3,), n_samples.

    sigma_f is computed by pooling RAW per-sample residuals across every hold, never by
    taking the spread of per-hold MEANS -- averaging n samples within a hold shrinks the
    spread of the mean by sqrt(n) relative to the actual single-sample noise (standard
    error, not the noise floor itself), and the identifiability mask in §4.1 thresholds
    single-timestep |f_i| against sigma_f, not an hold-averaged value. This was caught by
    --self-test: an early version used per-hold means here and recovered a sigma_f about
    6x smaller than the synthetic ground truth (matching sqrt(40) for 40 samples/hold).
    """
    if not records:
        raise ValueError("no records to compute stats from")

    pooled_residuals = []
    for r in records:
        pred = predicted_force(r["mass_kg"], gravity_vec_base)
        pooled_residuals.append(np.asarray(r["raw_forces"], dtype=float) - pred)
    pooled = np.concatenate(pooled_residuals, axis=0)  # (total_samples, 3)

    bias_n = pooled.mean(axis=0)
    sigma_f_debiased = pooled.std(axis=0)
    sigma_f_conservative = np.sqrt((pooled ** 2).mean(axis=0))  # RMS, bias not removed
    within_hold_std = np.stack(
        [np.asarray(r["within_hold_std_force"], dtype=float) for r in records]).mean(axis=0)

    # Config-dependence check: if the hold-mean residual varies a lot across poses, the
    # residual isn't just flat noise -- it's pose-dependent, which is exactly what the
    # workspace/nullspace constraint is meant to keep small. Uses hold means deliberately
    # (unlike sigma_f above) since this is checking for a per-pose SHIFT, not sample noise.
    hold_residual_means = [
        np.asarray(r["mean_force"], dtype=float) - predicted_force(r["mass_kg"], gravity_vec_base)
        for r in records]
    by_pose: Dict[int, List[np.ndarray]] = {}
    for r, res in zip(records, hold_residual_means):
        by_pose.setdefault(r["pose_index"], []).append(res)
    per_pose_mean = np.stack([np.mean(v, axis=0) for v in by_pose.values()])
    pose_bias_spread = (
        per_pose_mean.max(axis=0) - per_pose_mean.min(axis=0)
        if len(by_pose) > 1 else np.zeros(3))

    zero_mass = [r for r in records if abs(r["mass_kg"]) < 1e-9]
    zero_mass_force_bias = (
        np.mean([r["mean_force"] for r in zero_mass], axis=0).tolist() if zero_mass else None)
    zero_mass_torque = (
        np.stack([r["mean_torque"] for r in zero_mass]) if zero_mass else np.zeros((0, 3)))

    return {
        "num_records": len(records),
        "num_poses": len(by_pose),
        "force_axes": ["fx", "fy", "fz"],
        "bias_n": bias_n.tolist(),
        "sigma_f_debiased_n": sigma_f_debiased.tolist(),
        "sigma_f_conservative_rms_n": sigma_f_conservative.tolist(),
        "within_hold_std_n": within_hold_std.tolist(),
        "pose_bias_spread_n": pose_bias_spread.tolist(),
        "zero_mass_force_bias_n": zero_mass_force_bias,
        "torque_note": (
            "tx/ty/tz have no independent ground truth in this procedure (predicting them "
            "requires the unmeasured lever arm from the wrench reference point to the hang "
            "point). Reported descriptively only, at the zero-added-mass condition, as a "
            "sanity check that the payload calibration is holding -- should sit near the "
            "same residual level as the existing payload-corrected estimate, not grow with "
            "pose or with the extra hanging mass."),
        "zero_mass_torque_mean_nm": (
            zero_mass_torque.mean(axis=0).tolist() if len(zero_mass_torque) else None),
        "zero_mass_torque_std_nm": (
            zero_mass_torque.std(axis=0).tolist() if len(zero_mass_torque) else None),
    }


def generate_figure(records: List[Dict], gravity_vec_base: np.ndarray, output_path: Path) -> None:
    masses = np.array([r["mass_kg"] for r in records])
    measured_fz = np.array([r["mean_force"][2] for r in records])
    predicted_fz = np.array([predicted_force(m, gravity_vec_base)[2] for m in masses])
    pose_idx = np.array([r["pose_index"] for r in records])

    residuals = np.stack([
        np.asarray(r["mean_force"]) - predicted_force(r["mass_kg"], gravity_vec_base)
        for r in records])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    for p in np.unique(pose_idx):
        mask = pose_idx == p
        ax.scatter(predicted_fz[mask], measured_fz[mask], label=f"pose {p}", alpha=0.8)
    lo = min(predicted_fz.min(), measured_fz.min())
    hi = max(predicted_fz.max(), measured_fz.max())
    pad = 0.1 * (hi - lo + 1e-6)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", linewidth=1, label="y = x")
    ax.set_xlabel("predicted fz = -m·g (N)")
    ax.set_ylabel("measured fz, external_wrench_in_base_frame (N)")
    ax.set_title("Hanging-mass calibration curve (z-axis)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    axis_labels = ["fx", "fy", "fz"]
    for i, label in enumerate(axis_labels):
        ax.hist(residuals[:, i], bins=max(5, len(records) // 3), alpha=0.5, label=label)
    ax.set_xlabel("residual = measured - predicted (N)")
    ax.set_ylabel("count")
    ax.set_title("Residual distribution -- visual σ_f")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


class HangingMassCalibrationNode(Node):
    def __init__(self, namespace: str, pair_tolerance_sec: float):
        super().__init__("static_hanging_mass_calibration")
        self.ns = namespace.strip("/")
        self.pair_tolerance_sec = pair_tolerance_sec

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=200)

        self._latest_q: Optional[np.ndarray] = None
        self._logging = False
        self._force_log: List[np.ndarray] = []
        self._torque_log: List[np.ndarray] = []

        self.create_subscription(
            JointState, f"/{self.ns}/{JOINT_STATE_TOPIC}", self._state_cb, qos)
        self.create_subscription(
            WrenchStamped, f"/{self.ns}/{WRENCH_TOPIC}", self._wrench_cb, qos)

        self.switch_client = self.create_client(
            SwitchController, f"/{self.ns}/controller_manager/switch_controller")
        self.set_param_client = self.create_client(
            SetParameters, f"/{self.ns}/{MOVE_TO_START}/set_parameters")

    def _state_cb(self, msg: JointState) -> None:
        self._latest_q = np.asarray(msg.position, dtype=float)

    def _wrench_cb(self, msg: WrenchStamped) -> None:
        if not self._logging:
            return
        self._force_log.append(np.array(
            [msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z]))
        self._torque_log.append(np.array(
            [msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z]))

    def wait_for_services(self, timeout_sec: float = 15.0) -> bool:
        for client, name in (
            (self.switch_client, "switch_controller"),
            (self.set_param_client, "set_parameters"),
        ):
            if not client.wait_for_service(timeout_sec=timeout_sec):
                self.get_logger().error(f"Service {name} not available on namespace {self.ns}")
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

    def wait_for_current_joint_state(self, timeout_sec: float = 5.0) -> Optional[np.ndarray]:
        deadline = time.monotonic() + timeout_sec
        while self._latest_q is None and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        return self._latest_q

    def preflight_delta_check(self, waypoint: List[float], max_delta_rad: float) -> bool:
        """See calibrate_payload.py's identically-purposed check: refuse to activate
        move_to_start_example_controller (a plain PD controller, not velocity-limited)
        toward a target far from where the arm actually is -- the failure mode that tripped
        a reflex on the very first real-hardware waypoint."""
        current_q = self.wait_for_current_joint_state()
        if current_q is None:
            self.get_logger().error(
                "No joint state received -- cannot verify the arm is near the target. Is "
                f"{JOINT_STATE_TOPIC} publishing?")
            return False
        delta = np.abs(current_q - np.asarray(waypoint))
        max_delta = float(np.max(delta))
        if max_delta > max_delta_rad:
            worst = int(np.argmax(delta)) + 1
            self.get_logger().error(
                f"Refusing to move: joint{worst} is {max_delta:.3f} rad "
                f"({np.degrees(max_delta):.1f} deg) from target, exceeding "
                f"--max-preflight-delta-rad={max_delta_rad}. Jog the arm closer via Desk "
                "first, then re-run.")
            return False
        return True

    def wait_settle_spin(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)

    def sample_wrench(self, duration_sec: float) -> Optional[Dict]:
        self._force_log, self._torque_log = [], []
        self._logging = True
        deadline = time.monotonic() + duration_sec
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
        self._logging = False
        if len(self._force_log) < 5:
            self.get_logger().error(
                f"Only {len(self._force_log)} wrench samples in {duration_sec}s -- is "
                f"{WRENCH_TOPIC} publishing?")
            return None
        forces = np.stack(self._force_log)
        torques = np.stack(self._torque_log)
        return {
            "raw_forces": forces,
            "mean_force": forces.mean(axis=0),
            "within_hold_std_force": forces.std(axis=0),
            "mean_torque": torques.mean(axis=0),
            "within_hold_std_torque": torques.std(axis=0),
            "n_samples": len(self._force_log),
        }


def default_waypoints_path() -> Path:
    if get_package_share_directory is not None:
        try:
            return Path(get_package_share_directory("fr3_bilateral_teleop")) / "config" / \
                "hanging_mass_calibration_waypoints.yaml"
        except Exception:
            pass
    return Path(__file__).resolve().parent.parent / "config" / \
        "hanging_mass_calibration_waypoints.yaml"


def parse_masses(spec: str) -> List[float]:
    masses = sorted({float(x) for x in spec.split(",")})
    if not masses or abs(masses[0]) > 1e-9:
        masses = [0.0] + masses
        # 0.0 kg (nothing hung) is the mandatory baseline -- it's what makes the zero-mass
        # bias check in compute_calibration_stats possible, and it's the safest condition to
        # start the whole run on.
    return masses


def prompt(text: str) -> str:
    print(text, flush=True)
    return input("> ").strip().lower()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--namespace", default="follower")
    parser.add_argument("--waypoints-file", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--masses", default="0.0,0.1,0.2,0.5,1.0,2.0",
        help="comma-separated kg values from the calibrated mass set. 0.0 (nothing hung) is "
             "always included as the baseline even if omitted. Start a real run with only "
             "the lightest values until they're confirmed to hold cleanly -- see the SAFETY "
             "note in this script's module docstring.")
    parser.add_argument("--settle-time", type=float, default=2.0)
    parser.add_argument("--sample-time", type=float, default=1.5)
    parser.add_argument(
        "--gravity-vector-base", type=float, nargs=3, default=list(DEFAULT_GRAVITY_VEC_BASE),
        metavar=("GX", "GY", "GZ"),
        help="m/s^2, in the follower base frame; default assumes a level, +Z-up mount")
    parser.add_argument("--max-preflight-delta-rad", type=float, default=0.6)
    parser.add_argument(
        "--no-figure", action="store_true", help="skip generating the calibration PNG")
    parser.add_argument(
        "--self-test", action="store_true",
        help="run compute_calibration_stats against synthetic data, no hardware/ROS needed")
    return parser.parse_args()


def synthetic_records(
        true_bias: np.ndarray, true_noise_std: np.ndarray,
        gravity_vec_base: np.ndarray, seed: int = 0) -> List[Dict]:
    rng = np.random.default_rng(seed)
    masses = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0]
    records = []
    for pose_index in range(3):
        for mass in masses:
            true_force = predicted_force(mass, gravity_vec_base) + true_bias
            noisy_samples = true_force + rng.normal(
                0.0, true_noise_std, size=(40, 3))
            records.append({
                "pose_index": pose_index,
                "pose_label": f"synthetic_pose_{pose_index}",
                "mass_kg": mass,
                "raw_forces": noisy_samples,
                "mean_force": noisy_samples.mean(axis=0),
                "within_hold_std_force": noisy_samples.std(axis=0),
                "mean_torque": rng.normal(0.0, 0.05, size=3),
                "n_samples": len(noisy_samples),
            })
    return records


def run_self_test() -> int:
    gravity_vec = DEFAULT_GRAVITY_VEC_BASE
    true_bias = np.array([0.3, -0.2, 0.5])
    true_noise_std = np.array([0.4, 0.35, 0.6])
    records = synthetic_records(true_bias, true_noise_std, gravity_vec)
    stats = compute_calibration_stats(records, gravity_vec)
    print(json.dumps(stats, indent=2))

    recovered_bias = np.array(stats["bias_n"])
    recovered_sigma = np.array(stats["sigma_f_debiased_n"])
    bias_err = np.abs(recovered_bias - true_bias)
    sigma_err = np.abs(recovered_sigma - true_noise_std)
    # Loose tolerances -- this checks the pipeline recovers known values, not tight
    # statistical convergence at n=40 samples/hold, n=18 holds/axis.
    ok = bool(np.all(bias_err < 0.1) and np.all(sigma_err < 0.15))
    print(
        f"self-test: max bias error={bias_err.max():.3f} N, "
        f"max sigma_f error={sigma_err.max():.3f} N -- {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_test()

    gravity_vec_base = np.asarray(args.gravity_vector_base, dtype=float)
    masses = parse_masses(args.masses)

    waypoints_path = Path(args.waypoints_file) if args.waypoints_file else default_waypoints_path()
    spec = yaml.safe_load(waypoints_path.read_text())
    waypoints = spec["waypoints"]
    pose_labels = spec.get("pose_labels", [f"pose_{i}" for i in range(len(waypoints))])

    output_dir = Path(
        args.output_dir or __import__("os").environ.get(
            "TELEOP_HANGING_MASS_OUTPUT_DIR", "/tmp/franka_teleop_hanging_mass_calibration"))
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"About to move real hardware (namespace={args.namespace}) through {len(waypoints)} "
        f"pose(s), sampling masses {masses} kg at each. Confirm the follower's normal tool "
        "payload is already applied via set_load (payload calibration), and that the mass "
        f"you're about to hang is NOT also declared to set_load.", flush=True)
    if prompt("Type 'go' to continue, anything else to abort:") != "go":
        print("Aborted.")
        return 1

    rclpy.init()
    node = HangingMassCalibrationNode(args.namespace, pair_tolerance_sec=0.05)
    controller_active = False
    records: List[Dict] = []

    try:
        if not node.wait_for_services():
            return 1

        for i, waypoint in enumerate(waypoints):
            label = pose_labels[i] if i < len(pose_labels) else f"pose_{i}"
            node.get_logger().info(f"Pose {i + 1}/{len(waypoints)} ({label}): {waypoint}")

            if controller_active:
                if not node.switch_controller([], [MOVE_TO_START]):
                    node.get_logger().error(
                        "Failed to deactivate move_to_start_example_controller")
                    return 1
                controller_active = False

            if not node.set_start_joint_configuration(waypoint):
                node.get_logger().error("Failed to set start_joint_configuration")
                return 1
            if not node.preflight_delta_check(waypoint, args.max_preflight_delta_rad):
                return 1
            if not node.switch_controller([MOVE_TO_START], []):
                node.get_logger().error("Failed to activate move_to_start_example_controller")
                return 1
            controller_active = True
            node.wait_settle_spin(args.settle_time)

            for mass in masses:
                if mass == 0.0:
                    resp = prompt(
                        f"Pose {label}: remove any hanging mass so this is the zero-mass "
                        "baseline, then press Enter (or type 'skip'):")
                else:
                    resp = prompt(
                        f"Pose {label}: hang the {mass} kg mass now, let it settle so it "
                        "hangs freely with no swing, then press Enter (or type 'skip'):")
                if resp == "skip":
                    node.get_logger().info(f"Skipped pose={label} mass={mass}kg")
                    continue

                node.wait_settle_spin(args.settle_time)
                sample = node.sample_wrench(args.sample_time)
                if sample is None:
                    node.get_logger().error(
                        "No wrench data -- aborting rather than recording a gap.")
                    return 1
                node.get_logger().info(
                    f"pose={label} mass={mass}kg mean_force={sample['mean_force'].tolist()} "
                    f"predicted_force={predicted_force(mass, gravity_vec_base).tolist()}")
                records.append({
                    "pose_index": i,
                    "pose_label": label,
                    "mass_kg": mass,
                    "raw_forces": sample["raw_forces"],
                    "mean_force": sample["mean_force"],
                    "within_hold_std_force": sample["within_hold_std_force"],
                    "mean_torque": sample["mean_torque"],
                    "within_hold_std_torque": sample["within_hold_std_torque"],
                    "n_samples": sample["n_samples"],
                })

            prompt(
                f"Pose {label} done. Remove any hanging mass before the arm moves to the "
                "next pose, then press Enter:")

        if controller_active:
            node.switch_controller([], [MOVE_TO_START])

        if len(records) < 2:
            node.get_logger().error("Fewer than 2 recorded samples -- nothing to compute.")
            return 1

        stats = compute_calibration_stats(records, gravity_vec_base)
        report = {
            "namespace": node.ns,
            "generated_at_unix": time.time(),
            "gravity_vector_base": gravity_vec_base.tolist(),
            "masses_kg": masses,
            "waypoints_file": str(waypoints_path),
            **stats,
        }
        report_path = (
            output_dir / f"{sanitize_namespace(args.namespace)}_hanging_mass_calibration.json")
        report_path.write_text(json.dumps(report, indent=2))
        node.get_logger().info(f"Wrote {report_path}")
        print(json.dumps(stats, indent=2))
        print(
            f"sigma_f (debiased, N) = {stats['sigma_f_debiased_n']} -- this is the number "
            "the identifiability mask should use.")

        csv_path = output_dir / f"{sanitize_namespace(args.namespace)}_hanging_mass_samples.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "pose_index", "pose_label", "mass_kg",
                "mean_fx", "mean_fy", "mean_fz", "mean_tx", "mean_ty", "mean_tz",
                "std_fx", "std_fy", "std_fz", "n_samples"])
            for r in records:
                writer.writerow([
                    r["pose_index"], r["pose_label"], r["mass_kg"],
                    *r["mean_force"].tolist(), *r["mean_torque"].tolist(),
                    *r["within_hold_std_force"].tolist(), r["n_samples"]])
        node.get_logger().info(f"Wrote {csv_path}")

        if not args.no_figure:
            fig_path = (
                output_dir / f"{sanitize_namespace(args.namespace)}_hanging_mass_calibration.png")
            generate_figure(records, gravity_vec_base, fig_path)
            node.get_logger().info(f"Wrote {fig_path}")

        return 0
    finally:
        if controller_active:
            node.switch_controller([], [MOVE_TO_START])
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
