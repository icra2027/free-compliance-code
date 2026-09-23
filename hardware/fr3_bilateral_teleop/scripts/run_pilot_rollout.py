#!/usr/bin/env python3
"""Day 14: T1 in-distribution PILOT rollouts for GATE 3 (tasks.md Day 14 / proposal §7):

  "Pilot rollouts on T1 in-distribution, n = 5 per policy. Sanity only --
  these are NOT evaluation rollouts and must not be reported."
  "GATE 3. B5 >= B0 on the in-distribution pilot. If not, check whether
  realized K tracks commanded K before touching the model."

This is the robot-control-side client: lightweight by design (json_numpy +
requests only for the policy call, no torch/lerobot -- same split
scripts/serve_policy.py's docstring already establishes: "the client is the
robot-control machine (no GPU, possibly no direct network path to wherever
the checkpoint lives)"). The actual policy runs on scripts/serve_policy.py,
reached over the SSH-tunnelled HTTP /act endpoint documented there.

One rollout:
  1. Capture a "before" overhead frame (scene camera) for ink-removal scoring.
  2. Loop for --duration seconds: at --replan-hz (10Hz, proposal §4.3),
     build the observation (scene/wrist RGB, [q, qdot, x_f] state, force
     history) and POST it to the served policy for a fresh action chunk;
     between replans, publish `target_pose`/`target_stiffness` to
     variable_impedance_controllers' variable-impedance controller at --publish-hz,
     picking the current chunk step via controller_frame_utils.chunk_step_index
     (compliance-vla/scripts, imported below) and holding it --
     the controller's own log-space stiffness rate limiter + cubic-spline
     pose interpolation (already validated real-hardware, Day 5) do the
     actual smoothing between commands, this script does not re-implement
     that.
  3. Capture an "after" overhead frame, score per-mark ink removal
     (compliance-vla/scripts/score_ink_removal.py) against
     operator-supplied --rois, and write one JSON log consumable by
     compliance-vla/scripts/evaluate_gate3.py.

Policies with no compliance output (b0, b2) get a FIXED, isotropic HIGH
stiffness (extract_impedance_labels.K_MAX -- the controller's own realizable
upper bound, reused verbatim rather than inventing a second "high" constant)
for the whole rollout, matching proposal §6.1's literal description of B0
("fixed high stiffness") and giving B2 the same low-level behaviour B0 has
(B2's own architecture has no stiffness output either). Policies with a
compliance output (b3, b5) get the policy's predicted log_k, exp()'d and
rotated from the auto-fit CONTACT frame into a base-frame diagonal via
scripts/fit_frozen_contact_frame.py's frozen R_contact + this project's
controller_frame_utils.rotate_diag_stiffness_to_base -- see that module's
docstring for exactly what this approximation does and does not preserve
(it is the "Week-4 controller-integration question" src/compliance_vla/policy/labels.py
flags as out of scope for training, given a first, documented pass here
because a PILOT rollout can't skip it the way training could).

**Requires ROI calibration first** (one-time per session, not built here,
out of scope for a Day-14 sanity pilot): the 4 marks' colour->position
mapping is randomized per session (proposal §5), so --rois pixel
coordinates must be read off the CURRENT scene-camera frame by hand (e.g.
`ros2 run image_view image_view image:=/camera/color/image_raw` and eyeball
pixel bounds, or a small dedicated click-4-corners tool -- not built here)
before running this script for real.

**Cannot be run or verified in this environment** -- no ROS2/rclpy install
here (checked: `python3 -c "import rclpy"` fails), no robot. Written to the
same topic/message conventions already proven correct elsewhere in this
package (record_demo.py's pose/wrench subscriptions,
probe_variable_impedance_sinusoid.py's target_pose/target_stiffness
publishers, data_recorder's camera topics/cv_bridge usage) rather
than invented fresh, and the pure-computation pieces it depends on
(controller_frame_utils, score_ink_removal) are unit-tested in isolation
(see compliance-vla/scripts/README.md) -- but the ROS2 wiring itself
is unverified against real hardware, same "no robot access this session"
status as several earlier days in tasks.md.

Usage (after bringing up bilateral teleop + serve_policy.py on the GPU
machine, with an SSH tunnel per serve_policy.py's docstring):
    ros2 run fr3_bilateral_teleop run_pilot_rollout.py \\
        --policy b5 --n-rollouts 5 --referent red --manner firmly \\
        --rois red:120,80,60,60 blue:300,80,60,60 green:120,260,60,60 black:300,260,60,60
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import requests
import json_numpy
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped, WrenchStamped
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64MultiArray
from cv_bridge import CvBridge

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))  # realpath, not abspath: `ros2 run`
# executes this file through install/.../lib/fr3_bilateral_teleop/'s symlink-install symlink,
# so abspath() (which doesn't follow symlinks) would resolve SCRIPT_DIR to the install tree
# instead of src/fr3_bilateral_teleop/scripts, breaking the sibling-package lookup below.
WORKSPACE_SRC = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # .../src
PROJECT_SCRIPTS = os.path.join(WORKSPACE_SRC, "compliance-vla", "scripts")
# extract_impedance_labels lives in this package's dataset_tools/labeling/ in the source tree
# (without symlink-install it is also installed next to this file, in lib/fr3_bilateral_teleop/).
LABELING_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "dataset_tools", "labeling")
for _p in (SCRIPT_DIR, LABELING_DIR, PROJECT_SCRIPTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import panda_fk as fk  # noqa: E402
from extract_impedance_labels import K_MAX, quat_to_rotvec  # noqa: E402 -- reused, not reimplemented

from controller_frame_utils import chunk_step_index, rotate_diag_stiffness_to_base  # noqa: E402
from score_ink_removal import score_rollout  # noqa: E402

CONTACT_FRAME_PATH = os.path.join(PROJECT_SCRIPTS, "contact_frame_t1.npy")
TOOL_OFFSET_PATH = os.path.join(PROJECT_SCRIPTS, "tool_offset.npy")

POLICIES_WITH_COMPLIANCE = ("b3", "b5")
ACTION_RATE_HZ = 30.0  # matches the policy's own chunk rate, proposal §4.2
CHUNK_SIZE = 32  # proposal §4.2 H=32 @ 30Hz


def rotvec_to_quat(rv: np.ndarray) -> np.ndarray:
    """rv: (3,) rotation vector -> (4,) quaternion [x,y,z,w]. Same formula as
    compliance-vla/scripts/run_extraction_on_dataset.py's
    rotvec_batch_to_quat -- duplicated here (rather than imported) so this
    robot-control script stays free of that module's pandas import, the
    same "lightweight client" principle scripts/serve_policy.py's docstring
    already establishes for client_example.py. Keep in sync with the
    canonical batched version if either changes.
    """
    theta = float(np.linalg.norm(rv))
    if theta < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis = rv / theta
    return np.concatenate([axis * np.sin(theta / 2.0), [np.cos(theta / 2.0)]])


def resample_force_history(times: np.ndarray, values: np.ndarray, t_end: float, window_sec: float, n_samples: int) -> np.ndarray:
    """Pure-numpy duplicate of compliance_vla.policy.force_encoder.resample_to_n_samples --
    duplicated (not imported) for the same reason rotvec_to_quat is: importing
    compliance_vla.policy.force_encoder would pull in torch on the robot-control machine.
    Keep in sync with the canonical version if either changes; behaviour must
    match exactly (same self-tested edge cases: <2 samples in window -> hold
    earliest constant, never NaN).
    """
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


class PilotRolloutNode(Node):
    def __init__(self, follower_ns: str, scene_topic: str, wrist_topic: str, tool_offset: np.ndarray):
        super().__init__("run_pilot_rollout")
        self.follower_ns = follower_ns.strip("/")
        self.tool_offset = tool_offset
        self.bridge = CvBridge()

        reliable = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)
        best_effort = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=200)

        self._q: Optional[np.ndarray] = None
        self._qdot: Optional[np.ndarray] = None
        self._pose: Optional[Dict] = None  # {"t":, "pos":, "quat":}
        self._wrench_history: List[Dict] = []  # rolling buffer, trimmed to a few seconds
        self._scene_rgb: Optional[np.ndarray] = None
        self._wrist_rgb: Optional[np.ndarray] = None
        self.peak_force_n = 0.0

        self.create_subscription(
            JointState, f"/{self.follower_ns}/franka_robot_state_broadcaster/measured_joint_states",
            self._joint_cb, best_effort)
        self.create_subscription(
            PoseStamped, f"/{self.follower_ns}/franka_robot_state_broadcaster/current_pose",
            self._pose_cb, best_effort)
        self.create_subscription(
            WrenchStamped, f"/{self.follower_ns}/franka_robot_state_broadcaster/external_wrench_in_base_frame",
            self._wrench_cb, best_effort)
        self.create_subscription(Image, scene_topic, self._scene_cb, best_effort)
        self.create_subscription(Image, wrist_topic, self._wrist_cb, best_effort)

        self.target_pose_pub = self.create_publisher(PoseStamped, f"/{self.follower_ns}/target_pose", reliable)
        self.target_stiffness_pub = self.create_publisher(
            Float64MultiArray, f"/{self.follower_ns}/target_stiffness", reliable)

    @staticmethod
    def _stamp_sec(msg) -> float:
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def _joint_cb(self, msg: JointState) -> None:
        if len(msg.position) >= 7 and len(msg.velocity) >= 7:
            self._q = np.array(msg.position[:7])
            self._qdot = np.array(msg.velocity[:7])

    def _pose_cb(self, msg: PoseStamped) -> None:
        p = msg.pose
        self._pose = {
            "t": self._stamp_sec(msg),
            "pos": np.array([p.position.x, p.position.y, p.position.z]),
            "quat": np.array([p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]),
            "frame_id": msg.header.frame_id,
        }

    def _wrench_cb(self, msg: WrenchStamped) -> None:
        w = msg.wrench
        force = np.array([w.force.x, w.force.y, w.force.z])
        torque = np.array([w.torque.x, w.torque.y, w.torque.z])
        self.peak_force_n = max(self.peak_force_n, float(np.linalg.norm(force)))
        self._wrench_history.append({"t": self._stamp_sec(msg), "wrench": np.concatenate([force, torque])})
        cutoff = self._stamp_sec(msg) - 2.0  # keep 2s, well over the 500ms window the policy needs
        self._wrench_history = [h for h in self._wrench_history if h["t"] >= cutoff]

    def _scene_cb(self, msg: Image) -> None:
        self._scene_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")

    def _wrist_cb(self, msg: Image) -> None:
        self._wrist_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")

    def ready(self) -> bool:
        return all(x is not None for x in (self._q, self._qdot, self._pose, self._scene_rgb, self._wrist_rgb)) \
            and len(self._wrench_history) > 0

    def wait_until_ready(self, timeout_sec: float = 15.0) -> bool:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            if self.ready():
                return True
            rclpy.spin_once(self, timeout_sec=0.05)
        return False

    def x_f(self) -> np.ndarray:
        """Follower Cartesian pose (6D: pos + rotvec, base frame), including the calibrated
        tool offset -- matches src/compliance_vla/policy/labels.py's x_f convention exactly (same tool_offset
        file, same FK-based correction) so the served policy sees the same state
        distribution it was trained on."""
        # Prefer the pose topic's own quaternion for orientation (it's the arm's real O_T_EE,
        # not FK(q) which would double-count the tool offset differently) -- but rotate the
        # TOOL offset by that same orientation, matching src/compliance_vla/policy/labels.py's
        # `follower_pos + einsum(follower_rot, tool_offset)` convention exactly.
        rotvec = quat_to_rotvec(self._pose["quat"])
        rot = fk.matrix_from_rotvec(rotvec)
        pos = self._pose["pos"] + rot @ self.tool_offset
        return np.concatenate([pos, rotvec]).astype(np.float32)

    def force_history_20(self) -> np.ndarray:
        if not self._wrench_history:
            return np.zeros((20, 6), dtype=np.float32)
        times = np.array([h["t"] for h in self._wrench_history])
        values = np.stack([h["wrench"] for h in self._wrench_history])
        return resample_force_history(times, values, t_end=times[-1], window_sec=0.5, n_samples=20)

    def publish_command(self, x_eq: np.ndarray, k_base_diag: np.ndarray) -> None:
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        # Reuse whatever frame_id our own current_pose subscription carries (base frame,
        # confirmed real O_T_EE) rather than guessing a link name -- same "trust the message,
        # don't invent a frame_id string" pattern probe_variable_impedance_sinusoid.py's
        # publish_offset_target_pose already uses.
        pose_msg.header.frame_id = self._pose["frame_id"] if self._pose is not None else ""
        pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z = (float(v) for v in x_eq[0:3])
        quat = rotvec_to_quat(x_eq[3:6])
        pose_msg.pose.orientation.x, pose_msg.pose.orientation.y, pose_msg.pose.orientation.z, pose_msg.pose.orientation.w = \
            (float(v) for v in quat)
        self.target_pose_pub.publish(pose_msg)

        stiff_msg = Float64MultiArray()
        stiff_msg.data = [float(v) for v in k_base_diag]
        self.target_stiffness_pub.publish(stiff_msg)


def call_policy(server_url: str, task: str, scene_rgb, wrist_rgb, state, force_history=None) -> np.ndarray:
    payload = {"task": task, "scene_rgb": scene_rgb, "wrist_rgb": wrist_rgb, "state": state}
    if force_history is not None:
        payload["force_history"] = force_history
    resp = requests.post(server_url, data=json_numpy.dumps(payload), headers={"Content-Type": "application/json"})
    resp.raise_for_status()
    return json_numpy.loads(resp.content)["action_chunk"]


def run_one_rollout(
    node: PilotRolloutNode, args, rollout_index: int, R_contact: Optional[np.ndarray], task: str,
) -> Dict:
    node.peak_force_n = 0.0
    before_img = node._scene_rgb.copy()

    uses_force = args.policy != "b0"
    uses_compliance = args.policy in POLICIES_WITH_COMPLIANCE

    t_rollout_start = time.monotonic()
    t_last_replan = -1e9
    chunk: Optional[np.ndarray] = None
    replan_period = 1.0 / args.replan_hz
    publish_period = 1.0 / args.publish_hz
    t_last_publish = 0.0

    protective_stop = False
    while time.monotonic() - t_rollout_start < args.duration:
        rclpy.spin_once(node, timeout_sec=0.005)
        now = time.monotonic()

        if now - t_last_replan >= replan_period or chunk is None:
            state = np.concatenate([node._q, node._qdot, node.x_f()]).astype(np.float32)
            force_hist = node.force_history_20() if uses_force else None
            try:
                chunk = call_policy(
                    args.server_url, task, node._scene_rgb, node._wrist_rgb, state, force_hist,
                )
            except Exception as e:  # noqa: BLE001 -- a failed HTTP call must not crash the rollout loop
                node.get_logger().error(f"policy call failed: {e}")
            t_last_replan = now

        if node.peak_force_n > args.max_safe_force_n:
            protective_stop = True  # conservative proxy -- see module docstring's limitation note

        if chunk is not None and now - t_last_publish >= publish_period:
            step = chunk_step_index(now - t_last_replan, action_rate_hz=ACTION_RATE_HZ, chunk_size=chunk.shape[0])
            action = chunk[step]
            x_eq = action[0:6]
            if uses_compliance and R_contact is not None:
                log_k = action[6:12]
                k_contact = np.exp(log_k)
                k_base, _dropped = rotate_diag_stiffness_to_base(k_contact, R_contact)
            else:
                k_base = K_MAX.copy()  # fixed high stiffness, §6.1's B0 description -- see module docstring
            node.publish_command(x_eq, k_base)
            t_last_publish = now

    after_img = node._scene_rgb.copy()
    scores = score_rollout(before_img, after_img, args.rois, args.diff_threshold)
    targeted_pct = scores[args.referent]["coverage_pct"]

    return {
        "policy": args.policy,
        "rollout_index": rollout_index,
        "task": task,
        "referent": args.referent,
        "manner": args.manner,
        "duration_sec": args.duration,
        "ink_removal_pct_targeted": targeted_pct,
        "ink_removal_pct_all_marks": {k: v["coverage_pct"] for k, v in scores.items()},
        "success": bool(targeted_pct >= args.success_threshold_pct),
        "peak_force_n": node.peak_force_n,
        "protective_stop": protective_stop,
        "server_url": args.server_url,
    }


def _parse_roi_arg(spec: str):
    name, coords = spec.split(":")
    x, y, w, h = (int(v) for v in coords.split(","))
    return name, (x, y, w, h)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", required=True, choices=("b0", "b2", "b3", "b4", "b5"))
    p.add_argument("--server-url", default="http://localhost:8000/act")
    p.add_argument("--follower-namespace", default="franka_teleop/follower")
    p.add_argument("--scene-topic", default="/camera/color/image_raw")
    p.add_argument("--wrist-topic", default="/wrist/wrist_camera/color/image_raw")
    p.add_argument("--n-rollouts", type=int, default=5, help="tasks.md Day 14: n=5 per policy, sanity only")
    p.add_argument("--referent", required=True, choices=("red", "blue", "green", "black"))
    p.add_argument("--manner", default="normally", choices=("normally", "firmly"), help="see Day 6 scope reset: gently is out of scope")
    p.add_argument("--duration", type=float, default=8.0, help="seconds per rollout")
    p.add_argument("--replan-hz", type=float, default=10.0, help="proposal §4.3: policy replans at 10Hz")
    p.add_argument("--publish-hz", type=float, default=30.0, help="target_pose/target_stiffness publish rate")
    p.add_argument("--rois", nargs="+", required=True, help="name:x,y,w,h per mark -- read off the CURRENT session's frame, see module docstring")
    p.add_argument("--diff-threshold", type=float, default=25.0)
    p.add_argument("--success-threshold-pct", type=float, default=50.0)
    p.add_argument("--max-safe-force-n", type=float, default=40.0, help="conservative protective_stop proxy, see module docstring")
    p.add_argument("--contact-frame", default=CONTACT_FRAME_PATH)
    p.add_argument("--tool-offset", default=TOOL_OFFSET_PATH)
    p.add_argument("--output-dir", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.rois = dict(_parse_roi_arg(s) for s in args.rois)

    output_dir = Path(args.output_dir or os.environ.get("TELEOP_PILOT_ROLLOUTS_OUTPUT_DIR", "/tmp/franka_teleop_pilot_rollouts"))
    output_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(args.tool_offset):
        print(f"error: {args.tool_offset} missing -- run compliance-vla/scripts/calibrate_tool_offset.py first", file=sys.stderr)
        return 1
    tool_offset = np.load(args.tool_offset)

    R_contact = None
    if args.policy in POLICIES_WITH_COMPLIANCE:
        if not os.path.exists(args.contact_frame):
            print(f"error: {args.contact_frame} missing -- run compliance-vla/scripts/fit_frozen_contact_frame.py first", file=sys.stderr)
            return 1
        R_contact = np.load(args.contact_frame)

    task = f"wipe the {args.referent} mark {args.manner}"  # matches dataset_io.py's parse_manner_from_task/parse_referent_from_task convention

    rclpy.init()
    node = PilotRolloutNode(args.follower_namespace, args.scene_topic, args.wrist_topic, tool_offset)
    try:
        node.get_logger().info(f"Waiting for state/wrench/camera topics under /{args.follower_namespace}...")
        if not node.wait_until_ready():
            node.get_logger().error("Timed out waiting for first messages -- is teleop.launch.py up?")
            return 1

        node.get_logger().info(f"Checking server at {args.server_url}...")
        try:
            call_policy(args.server_url, task, node._scene_rgb, node._wrist_rgb,
                        np.concatenate([node._q, node._qdot, node.x_f()]).astype(np.float32),
                        node.force_history_20() if args.policy != "b0" else None)
        except Exception as e:  # noqa: BLE001
            node.get_logger().error(f"serve_policy.py not reachable at {args.server_url}: {e}")
            return 1

        results = []
        for i in range(args.n_rollouts):
            print(f"[run_pilot_rollout] {args.policy} rollout {i+1}/{args.n_rollouts} -- "
                  f"press Enter to reset the mark and start (auto-reset harness is Week 4, Day 16 -- "
                  f"manual reset for these Day-14 pilots).", flush=True)
            input("> ")
            result = run_one_rollout(node, args, i, R_contact, task)
            print(json.dumps(result, indent=2))
            results.append(result)

            timestamp = int(time.time())
            out_path = output_dir / f"{args.policy}_rollout{i}_{timestamp}.json"
            with out_path.open("w") as f:
                json.dump(result, f, indent=2)
            node.get_logger().info(f"wrote {out_path}")

        n_success = sum(r["success"] for r in results)
        mean_ink = sum(r["ink_removal_pct_targeted"] for r in results) / len(results)
        print(f"\n[run_pilot_rollout] {args.policy}: {n_success}/{len(results)} success, "
              f"mean targeted ink-removal {mean_ink:.1f}%. Sanity only -- see tasks.md Day 14, "
              f"do not report these numbers. Copy {output_dir} into "
              f"compliance-vla/reports/pilot_rollouts/ before running evaluate_gate3.py.")
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
