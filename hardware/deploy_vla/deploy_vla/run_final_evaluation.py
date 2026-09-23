"""Final B0-B5 real-hardware evaluation runner (tasks.md Day 18-20: E1/E2, "6
policies x 20 rollouts" etc.) -- built on deploy_smolvla.py's SmolVLADeployment
and mirrors run_scripted_rollout.py's non-interactive episode loop exactly (see
that module's docstring for why the control loop itself is duplicated here
rather than refactored into a shared function: it's private/interactive-loop-
specific in deploy_smolvla.py). If deploy_smolvla.run_interactive's loop ever
changes (e.g. a new safety check), mirror the change in both scripted runners.

The one thing this script adds on top of run_scripted_rollout.py is recording,
per run, into its own timestamped directory:

  1. Every observation actually sent to the served policy at each replan step
     -- scene_rgb, wrist_rgb (exactly the resized arrays get_action_chunk() is
     called with, not a separately-sampled copy), state, force_history and
     the instruction -- as JPEGs + one JSONL log (RunRecorder). This is an
     EVENT log, paced by however long each policy HTTP round trip takes, not
     wall-clock -- for that reason it is a separate thing from (2) below, not
     a substitute for it.
  2. One continuous video for the whole run, wrist | scene | zed stacked
     left-to-right into a single composite frame at --video-fps (default 24)
     -- scene is the Femto Bolt, wrist the RealSense, both already used for
     policy inference; the third panel is an externally connected ZED camera.
     A single pacer thread (PacedVideoRecorder, driven by
     make_stacked_frame_getter) assembles and writes each composite frame on
     one fixed wall-clock schedule, so the three panels are frame-aligned by
     construction -- there's exactly one clock deciding when a composite
     frame gets built, not three independently-paced files stitched together
     after the fact (see PacedVideoRecorder's docstring for the exact
     guarantee and its limits -- this is wall-clock software pacing, not
     hardware genlock). All three cameras are read directly via
     cv2.VideoCapture (V4L2), NOT ROS topics -- scene and wrist both used to
     go through deploy_smolvla.py's own driver-level reads (scene) or a ROS
     Image subscription (wrist), but the ROS-topic path was found, on real
     hardware, to intermittently corrupt/drop samples under DDS reliable-QoS
     backlog ("sequence size exceeds remaining buffer"), worst during
     get_action_chunk's blocking HTTP round trip. Each camera is drained by
     its own dedicated grabber thread (SceneCameraGrabber / WristCameraGrabber
     / ZedGrabber) feeding a shared latest-frame buffer the pacer reads from,
     so a stalled/slow camera read never blocks the control loop or the
     recording. Both the scene and wrist panels' frame_getters read off the
     SAME buffers the control loop itself uses for policy inference
     (SceneCameraGrabber monkeypatches node.get_scene_frame, WristCameraGrabber
     monkeypatches node.get_wrist_frame -- see each class's docstring), so
     what's in those two panels is what the policy saw, not a second
     independent feed.

Metrics (proposal §6.4) this script computes per-rollout vs. what it only
records the signal for, since some of §6.4's metrics are cross-rollout
aggregates that can't be computed from a single run in isolation:

  M1 Binary success       -- COMPUTED here: ink_removal_pct_targeted >=
                              --success-threshold-pct (default 50.0, same
                              threshold fr3_bilateral_teleop/run_pilot_rollout.py
                              uses). The Wilson 95% CI itself is a cross-
                              rollout statistic -- computed by whatever
                              aggregation script pools rollout_summary.json
                              files across a policy/condition cell, not here.
  M2 Continuous task score -- COMPUTED here: ink_removal_pct_targeted, via
                              deploy_smolvla.py's score_wipe integration
                              (node.score_color must be set -- see
                              --score-color -- or this is skipped and
                              ink_removal_pct_targeted is null).
  M3 Peak / RMS contact    -- COMPUTED here: peak_force_n, rms_force_n, from a
     force                    dedicated full-rollout wrench recording (see
                              WrenchRecorder) -- NOT node's own
                              _wrench_buffer, which only keeps ~2s.
  M4 Force-distribution     -- SIGNAL recorded here (wrench.csv, the full
     distance to human           per-sample 6-DoF trace), metric NOT computed
     demos (1-Wasserstein)       here -- needs the training dataset's demo
                              force distributions loaded too, which live in
                              compliance-vla, not this package; compute
                              in a separate aggregation pass over wrench.csv.
  M5 Protective stops /     -- COMPUTED here, but as an explicit PROXY, not
     torque-limit violations    real robot fault telemetry: protective_stop
                              = peak_force_n > --max-safe-force-n (default
                              40.0N, matching run_pilot_rollout.py's same
                              proxy and its documented limitation -- this
                              does not read franka_msgs/FrankaRobotState's
                              actual error/reflex fields, which this project
                              has not verified the exact field names for on
                              this firmware version).
  M6 Adverb -> force effect -- SIGNAL recorded here (peak/RMS force per
     size (Cohen's d)            rollout, tagged with policy/referent/manner
                              in rollout_summary.json); metric NOT computed
                              here -- needs rollouts from >=2 manners pooled,
                              a separate aggregation step.
  M7 Causal force ablation  -- CONDITION supported here (--freeze-force-
     (Delta success, force      history holds force_history fixed at its
     frozen at test time)       first-observed value for the rest of the
                              episode, for b2/b5/b3 checkpoints -- see
                              run_one_recorded_episode), and tagged in the
                              summary (force_history_frozen); the Delta-
                              success comparison itself is computed by
                              diffing two aggregated batches (frozen vs.
                              not), not here.
  M8 Offline stiffness      -- Not applicable to this script at all -- M8 is
     prediction error            computed entirely offline from the labeled
                              dataset, no robot involved.

Output layout, one directory per run:
    <out-root>/<policy>/<referent>_<manner>/run<NNN>_<timestamp>/
        observations/replan_00000_scene.jpg, replan_00000_wrist.jpg, ...
        steps.jsonl                    # one line per replan: state/action/instruction
        wrench.csv                     # full-rollout 6-DoF external wrench @ native rate (M3/M4/M5/M6 signal)
        combined_video.mp4             # wrist | scene | zed, stacked, --video-fps (default 24)
        combined_video.frame_times.csv # per-frame target-vs-actual write timing
        rollout_summary.json           # policy/referent/manner/steps/metrics/paths/counts/sync anchor

Usage (after serve_policy.py is up on the GPU machine and SSH-tunnelled, and
the variable_impedance_controllers stack is launched per compliance-vla/
deploy_vla_on_franka.md -- same follower/controller preconditions as
run_scripted_rollout.py, plus a ZED camera plugged in. NOT the wrist camera's
own ROS launch -- the wrist camera is read directly via cv2.VideoCapture,
see WristCameraGrabber, so only its /dev/video* node needs to exist, not a
running realsense2_camera_node):

    taskset -c 2-15 ros2 run deploy_vla run_final_evaluation \\
        --policy b5 --referent right --manner firmly --n-rollouts 20 \\
        --wrist-cv2-device 4 --zed-device 10 --server-url http://localhost:8000/act

(taskset range matches deploy_vla_on_franka.md's existing guidance for
deploy_smolvla itself -- this script adds several more background threads per
run [three grabbers + one pacer] on top of that node's own camera/GUI load,
which a prior postmortem found capable of starving ros2_control_node's 1kHz
thread when left unpinned; keep it inside the same client-core range.)

B1 (proposal Sect.6.1: "B0's policy, run through the same low-level controller
... with stiffness pinned to one fixed vector") has no checkpoint of its own
-- point --server-url at a B0 checkpoint's serve_policy.py instance and pass
--policy b1 --b1-oracle-stiffness (defaults to compliance-vla/reports/
b1_oracle_constant_stiffness.json) to override the published stiffness every
step with that file's summary.k_vector_contact_frame, instead of B0's own
NON_COMPLIANT_STIFFNESS default. Published as-is, in the SAME no-contact-
frame-rotation convention deploy_smolvla.py already uses for
NON_COMPLIANT_STIFFNESS and for B5's own predicted log_k (see
_publish_target_stiffness) -- NOT rotated by contact_frame_t1.npy the way the
older fr3_bilateral_teleop/run_pilot_rollout.py path does. Documented here
rather than silently assumed correct; revisit if B1 numbers look off.
"""
import argparse
import csv
import json
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import WrenchStamped

from deploy_vla.deploy_smolvla import (
    SmolVLADeployment,
    INSTRUCTION_COLORS,
    INSTRUCTION_MANNERS,
    SCORE_COLORS,
    MAX_STEPS,
)
from deploy_vla.run_scripted_rollout import wait_until_ready

POLICY_CHOICES = ("b0", "b1", "b2", "b3", "b4", "b5")

# M5 protective-stop proxy default -- matches fr3_bilateral_teleop/scripts/
# run_pilot_rollout.py's --max-safe-force-n exactly, same documented
# limitation (peak-force threshold, not real franka_msgs fault/reflex
# telemetry -- see WrenchRecorder).
DEFAULT_MAX_SAFE_FORCE_N = 40.0
# M1 success-from-M2 threshold -- matches run_pilot_rollout.py's
# --success-threshold-pct default exactly, same reasoning: a policy that
# doesn't clear half the target mark's ink hasn't "succeeded" at the task.
DEFAULT_SUCCESS_THRESHOLD_PCT = 50.0

# compliance-vla/scripts is a src/ sibling of this package -- same
# upward-search helper deploy_smolvla.py uses for tool_offset.npy, reused here
# (rather than re-derived) so this script finds the same workspace root no
# matter whether it's run from the src/ copy or colcon's installed copy.
from deploy_vla.deploy_smolvla import _find_bookish_scripts_dir  # noqa: E402
import os  # noqa: E402

_BOOKISH_SCRIPTS_DIR = _find_bookish_scripts_dir(os.path.dirname(os.path.realpath(__file__)))
DEFAULT_OUT_ROOT = os.path.join(os.path.dirname(_BOOKISH_SCRIPTS_DIR), "reports", "final_eval")
DEFAULT_B1_ORACLE_STIFFNESS = os.path.join(
    os.path.dirname(_BOOKISH_SCRIPTS_DIR), "reports", "b1_oracle_constant_stiffness.json"
)


class RunRecorder:
    """Writes, into one run's own directory, every observation actually fed to
    the policy (exactly the scene_rgb/wrist_rgb/state/force_history/instruction
    get_action_chunk() is called with -- hooked at that call site, not
    re-sampled independently) plus the resulting low-level actions, one JSONL
    line per replan."""

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.obs_dir = run_dir / "observations"
        self.obs_dir.mkdir(parents=True, exist_ok=True)
        self._steps_log = open(run_dir / "steps.jsonl", "w")
        self._replan_index = 0
        self.n_observations = 0

    def log_replan(self, scene_rgb, wrist_rgb, state, force_history, instruction, action_chunk):
        idx = self._replan_index
        scene_path = self.obs_dir / f"replan_{idx:05d}_scene.jpg"
        wrist_path = self.obs_dir / f"replan_{idx:05d}_wrist.jpg"
        # Saved images are exactly what was sent over the wire: scene_rgb/
        # wrist_rgb are already the resized (image_size, image_size, 3) uint8
        # RGB arrays get_action_chunk()'s payload uses -- cvtColor here is only
        # RGB->BGR for cv2.imwrite, not a resize/recrop.
        cv2.imwrite(str(scene_path), cv2.cvtColor(scene_rgb, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(str(wrist_path), cv2.cvtColor(wrist_rgb, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        record = {
            "replan_index": idx,
            "wall_time": time.time(),
            "instruction": instruction,
            "state": np.asarray(state, dtype=np.float32).tolist(),
            "force_history": (
                np.asarray(force_history, dtype=np.float32).tolist() if force_history is not None else None
            ),
            "action_chunk": (
                np.asarray(action_chunk, dtype=np.float32).tolist() if action_chunk is not None else None
            ),
            "scene_image": scene_path.name,
            "wrist_image": wrist_path.name,
        }
        self._steps_log.write(json.dumps(record) + "\n")
        self._steps_log.flush()
        self._replan_index += 1
        self.n_observations += 1

    def close(self):
        self._steps_log.close()


class WrenchRecorder:
    """Records the FULL-rollout 6-DoF external wrench trace at native
    publish rate to <run_dir>/wrench.csv, and derives M3 (peak/RMS contact
    force) and the M5 protective-stop proxy from it.

    Deliberately does NOT read node._wrench_buffer (SmolVLADeployment's own
    deque, maxlen=2000, "~2s headroom" per its own comment) -- that buffer
    is sized for force_history's 500ms window, not for a whole rollout, and
    quietly drops everything older than ~2s. A rollout can run for tens of
    seconds (up to max_steps / chunk_hz), so M3/M4/M5/M6 all need the
    complete trace, not a 2s tail of it.

    Subscribes to the wrench topic independently rather than hooking
    node._wrench_callback -- unlike the scene/wrist cameras (a single
    cv2.VideoCapture handle that only one thread may safely .read()), a ROS
    topic supports multiple independent subscribers with no race condition,
    so no monkeypatching/SOLE-reader pattern is needed here the way
    SceneCameraGrabber/WristCameraGrabber need it for their cameras."""

    def __init__(self, node: SmolVLADeployment, run_dir: Path, max_safe_force_n: float):
        self.node = node
        self.max_safe_force_n = float(max_safe_force_n)
        self._csv_path = run_dir / "wrench.csv"
        self._file = open(self._csv_path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["t", "fx", "fy", "fz", "tx", "ty", "tz", "force_mag_n"])
        self._lock = threading.Lock()
        self.n_samples = 0
        self.peak_force_n = 0.0
        self._sq_sum_force = 0.0  # running sum of force_mag_n**2, for RMS
        self._sub = None

    def _wrench_cb(self, msg: WrenchStamped):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        w = msg.wrench
        force = (w.force.x, w.force.y, w.force.z)
        torque = (w.torque.x, w.torque.y, w.torque.z)
        force_mag = float(np.linalg.norm(force))
        with self._lock:
            self._writer.writerow([t, *force, *torque, force_mag])
            self.n_samples += 1
            self.peak_force_n = max(self.peak_force_n, force_mag)
            self._sq_sum_force += force_mag * force_mag

    def start(self):
        self._sub = self.node.create_subscription(
            WrenchStamped,
            f"/{self.node.follower_ns}/franka_robot_state_broadcaster/external_wrench_in_base_frame",
            self._wrench_cb,
            50,
        )

    def stop(self):
        if self._sub is not None:
            self.node.destroy_subscription(self._sub)
            self._sub = None
        self._file.close()

    def summary(self) -> dict:
        with self._lock:
            n, peak, sq_sum = self.n_samples, self.peak_force_n, self._sq_sum_force
        rms = float(np.sqrt(sq_sum / n)) if n > 0 else float("nan")
        return {
            "csv_path": "wrench.csv",
            "n_samples": n,
            "peak_force_n": peak,
            "rms_force_n": rms,
            # M5: conservative proxy, NOT real robot fault/reflex telemetry --
            # see this class's docstring and the module docstring's M5 entry.
            "protective_stop": bool(peak > self.max_safe_force_n),
            "max_safe_force_n": self.max_safe_force_n,
        }


class SceneCameraGrabber:
    """Dedicated thread that becomes the SOLE caller of node.scene_capture.read()
    for as long as it's running. cv2.VideoCapture is not documented safe for
    concurrent .read() from two threads, and deploy_smolvla.py's own control
    loop calls node.get_scene_frame() (-> scene_capture.read()) every replan --
    so a second, independent reader here would race it on the same capture
    object. Instead this monkeypatches node.get_scene_frame for as long as
    it's running to return this thread's continuously-refreshed buffer
    instead, restoring the original method on stop() (this object is created
    fresh per rollout; the node itself persists across --n-rollouts rollouts).
    One side effect, deliberate: the control loop's own policy-inference
    reads now come from the exact same buffer this recorder writes to video,
    not a second independent camera read.

    Deliberately does NOT resize frames to node.image_size before buffering
    them -- that would change get_scene_frame()'s contract for every other
    caller while this is active, in particular _start_running/_stop_and_home's
    wipe-scoring photo capture, which crops a fixed PIXEL roi
    (wipe_score_roi, e.g. (498, 367, 306, 226)) out of the NATIVE-resolution
    frame; resizing here first would make that crop nonsensical. So
    frame_shape (set by prime(), used for the video's VideoWriter) is
    whatever the camera's native resolution actually is, not image_size."""

    def __init__(self, node: SmolVLADeployment):
        self.node = node
        self._lock = threading.Lock()
        self._latest_rgb = None
        self._stop_event = threading.Event()
        self._thread = None
        self._orig_get_scene_frame = node.get_scene_frame
        self.frame_shape = None  # (h, w), set by prime()

    def _read_once(self):
        if not self.node.enable_scene_camera:
            return np.zeros((self.node.image_size, self.node.image_size, 3), dtype=np.uint8)
        ok, frame = self.node.scene_capture.read()
        if not ok:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Femto Bolt mounted upside down -- same fix/order (convert then rotate)
        # as SmolVLADeployment.get_scene_frame itself, which this replaces.
        return cv2.rotate(rgb, cv2.ROTATE_180)

    def prime(self):
        """Synchronous (no thread yet) read to seed the buffer and learn the
        real frame_shape before start() -- PacedVideoRecorder needs a fixed
        shape up front to open its VideoWriter, same reason ZedGrabber.
        open_and_prime() exists. Raises if the camera is enabled but a read
        still fails, same fail-fast posture as ZedGrabber's version."""
        frame = self._read_once()
        if frame is None:
            raise RuntimeError(
                "SceneCameraGrabber.prime(): scene_capture.read() failed -- is "
                "--scene-cv2-device pointed at the right /dev/video* node?"
            )
        self._latest_rgb = frame
        self.frame_shape = frame.shape[:2]

    def _run(self):
        while not self._stop_event.is_set():
            frame = self._read_once()
            if frame is not None:
                with self._lock:
                    self._latest_rgb = frame

    def start(self):
        self.node.get_scene_frame = self._patched_get_scene_frame
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _patched_get_scene_frame(self):
        with self._lock:
            return None if self._latest_rgb is None else self._latest_rgb.copy()

    def get_latest_bgr(self):
        """For PacedVideoRecorder -- cv2.VideoWriter wants BGR, this buffer is RGB
        (the convention the rest of this file/deploy_smolvla.py uses for policy input)."""
        with self._lock:
            frame = self._latest_rgb
        return None if frame is None else cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.node.get_scene_frame = self._orig_get_scene_frame


class WristCameraGrabber:
    """Dedicated thread that becomes the SOLE caller of node.wrist_capture.read()
    for as long as it's running -- exact mirror of SceneCameraGrabber, now that
    deploy_smolvla.py reads the wrist (RealSense) camera the same way it reads
    the scene (Femto Bolt) camera: cv2.VideoCapture straight off its V4L2 node,
    not a ROS Image topic (switched 2026-09-10 after the ROS-topic path was
    observed, on real hardware, to intermittently corrupt/drop samples under
    DDS reliable-QoS backlog -- "sequence size exceeds remaining buffer",
    worst during get_action_chunk's blocking HTTP round trip). Since
    deploy_smolvla.py's own control loop calls node.get_wrist_frame() (->
    wrist_capture.read()) every replan, a second/independent reader here would
    race it on the same capture object -- so, like SceneCameraGrabber, this
    monkeypatches node.get_wrist_frame for as long as it's running to return
    this thread's continuously-refreshed buffer instead, restoring the
    original method on stop(). One side effect, deliberate: the control
    loop's own policy-inference reads now come from the exact same buffer
    this recorder writes to video, not a second independent camera read.

    Scoped to persist across the WHOLE script run (create/prime/start once,
    before wait_until_ready(); stop once at the very end), NOT per-rollout
    like ZedGrabber -- wrist_rgb is a REQUIRED policy observation, not an
    optional recording extra, so it must already be live before the first
    rollout's readiness gate, independent of whether --no-record-video is set."""

    def __init__(self, node: SmolVLADeployment):
        self.node = node
        self._lock = threading.Lock()
        self._latest_rgb = None
        self._stop_event = threading.Event()
        self._thread = None
        self._orig_get_wrist_frame = node.get_wrist_frame
        self.frame_shape = None  # (h, w), set by prime()

    def _read_once(self):
        if not self.node.enable_wrist_camera:
            return np.zeros((self.node.image_size, self.node.image_size, 3), dtype=np.uint8)
        ok, frame = self.node.wrist_capture.read()
        if not ok:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # no rotation -- wrist mount isn't upside down

    def prime(self):
        """Synchronous (no thread yet) read to seed the buffer and learn the
        real frame_shape before start() -- PacedVideoRecorder needs a fixed
        shape up front to open its VideoWriter. Raises if the camera is
        enabled but a read still fails, same fail-fast posture as
        SceneCameraGrabber/ZedGrabber's own prime methods."""
        frame = self._read_once()
        if frame is None:
            raise RuntimeError(
                "WristCameraGrabber.prime(): wrist_capture.read() failed -- is "
                "--wrist-cv2-device pointed at the right /dev/video* node?"
            )
        self._latest_rgb = frame
        self.frame_shape = frame.shape[:2]

    def _run(self):
        while not self._stop_event.is_set():
            frame = self._read_once()
            if frame is not None:
                with self._lock:
                    self._latest_rgb = frame

    def start(self):
        self.node.get_wrist_frame = self._patched_get_wrist_frame
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _patched_get_wrist_frame(self):
        with self._lock:
            return None if self._latest_rgb is None else self._latest_rgb.copy()

    def get_latest_bgr(self):
        """For PacedVideoRecorder -- cv2.VideoWriter wants BGR, this buffer is RGB."""
        with self._lock:
            frame = self._latest_rgb
        return None if frame is None else cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.node.get_wrist_frame = self._orig_get_wrist_frame


class ZedGrabber:
    """Dedicated thread draining an externally connected ZED camera, opened as
    a plain UVC device via cv2.VideoCapture (not the ZED SDK) -- same
    "bypass the vendor driver, read V4L2 directly" choice deploy_smolvla.py
    already makes for the Femto Bolt scene camera. Nothing else in this
    process touches this capture object, so (unlike SceneCameraGrabber) there
    is no monkeypatching to do here -- just drain it continuously into a
    shared latest-frame buffer so a PacedVideoRecorder reading it never blocks
    on a live camera read."""

    def __init__(self, device, fps_hint=24.0, width=None, height=None):
        self.device = device
        self.fps_hint = fps_hint
        self.width = width
        self.height = height
        self._cap = None
        self._lock = threading.Lock()
        self._latest_bgr = None
        self._stop_event = threading.Event()
        self._thread = None
        self.frame_shape = None  # (h, w), set by open_and_prime

    def open_and_prime(self):
        """Opens the device and blocks for one verification frame -- raises if
        it can't get one, same fail-fast posture as SmolVLADeployment's own
        scene-camera open check. Must be called (and must succeed) before
        start(); also the only way frame_shape becomes known, which
        PacedVideoRecorder needs up front to open its VideoWriter."""
        self._cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open ZED capture device {self.device!r}")
        if self.width:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps_hint)

        ok, frame = self._cap.read()
        if not ok:
            self._cap.release()
            raise RuntimeError(
                f"Opened ZED device {self.device!r} but a verification read failed -- check "
                "`for v in /dev/video*; do echo \"$v: $(cat /sys/class/video4linux/$(basename $v)/name)\"; done` "
                "and pass the right one via --zed-device."
            )
        self.frame_shape = frame.shape[:2]
        self._latest_bgr = frame

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop_event.is_set():
            ok, frame = self._cap.read()
            if ok:
                with self._lock:
                    self._latest_bgr = frame

    def get_latest_bgr(self):
        with self._lock:
            return self._latest_bgr  # already BGR straight off cv2, no copy needed (read-only use)

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()


class PacedVideoRecorder:
    """Writes whatever `frame_getter()` returns (already BGR, ready for
    cv2.VideoWriter, and either None or a fixed (h, w) matching `frame_shape`)
    to an mp4 at a fixed `fps`, on its own thread, anchored to a shared
    `start_time` (time.monotonic() domain, shared across a process -- not
    valid across separate processes/machines).

    The synchronization guarantee: multiple PacedVideoRecorders constructed
    with the SAME start_time and fps write frame index k at target wall-clock
    time start_time + k/fps -- always, by construction, regardless of
    per-tick jitter -- so their output files are frame-aligned when played
    back side by side. This deliberately does NOT resync frame_index to "now"
    if a tick runs late (a naive pacer would, to avoid drift in wall-clock
    duration) -- doing that would break the shared start_time+k/fps meaning
    that side-by-side alignment depends on. A late tick instead just writes
    back-to-back with no sleep until it catches up, holding (duplicating) the
    last available frame if frame_getter has nothing newer -- the standard
    tradeoff for wall-clock-locked multi-camera recording without hardware
    genlock. Also writes `<out_path>.frame_times.csv` (frame_index,
    target_monotonic, actual_monotonic, wall_clock_unix) so drift/staleness on
    any one stream can be checked after the fact instead of assumed away."""

    def __init__(self, frame_getter, out_path, fps, start_time, frame_shape, csv_path=None):
        self.frame_getter = frame_getter
        self.out_path = str(out_path)
        self.fps = float(fps)
        self.start_time = start_time
        self.frame_shape = frame_shape  # (h, w)
        self.csv_path = str(csv_path) if csv_path is not None else None
        self._writer = None
        self._stop_event = threading.Event()
        self._thread = None
        self.frames_written = 0
        self._rows = []

    def start(self):
        h, w = self.frame_shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(self.out_path, fourcc, self.fps, (w, h))
        if not self._writer.isOpened():
            raise RuntimeError(f"Could not open VideoWriter for {self.out_path!r}")
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        period = 1.0 / self.fps
        frame_index = 0
        last_frame = None
        while not self._stop_event.is_set():
            target_t = self.start_time + frame_index * period
            now = time.monotonic()
            if now < target_t:
                time.sleep(target_t - now)
                now = time.monotonic()
            frame = self.frame_getter()
            if frame is None:
                frame = last_frame
            if frame is not None:
                self._writer.write(frame)
                last_frame = frame
                self.frames_written += 1
                self._rows.append((frame_index, target_t, now, time.time()))
            frame_index += 1

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._writer is not None:
            self._writer.release()
        if self.csv_path is not None:
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["frame_index", "target_monotonic", "actual_monotonic", "wall_clock_unix"])
                writer.writerows(self._rows)


def _resize_to_height(frame_bgr, height):
    h, w = frame_bgr.shape[:2]
    new_w = max(1, round(w * (height / h)))
    return cv2.resize(frame_bgr, (new_w, height))


def make_stacked_frame_getter(wrist_getter, scene_getter, zed_getter, panel_height):
    """Builds ONE frame_getter (for a single PacedVideoRecorder) that reads all
    three camera sources at every tick and horizontally concatenates them --
    wrist | scene | zed, each resized to a common `panel_height` (native
    aspect ratio preserved per panel, so panel widths differ) with a text
    label burned in -- into one composite BGR frame. Doing the stacking
    INSIDE one pacer's frame_getter (rather than compositing three already-
    separately-recorded video files afterward) is what makes the result
    correct by construction: there's exactly one clock (this pacer's) deciding
    when a composite frame is assembled, no separate alignment step needed
    after the fact.

    Falls back to a black panel of the same target size for whichever source
    has no frame yet (e.g. the very first tick, if a camera's grabber hasn't
    produced one), so one slow/missing source never blanks the other two
    panels -- only its own.

    Returns (frame_getter, (h, w)) -- the fixed composite shape, computed
    once from whatever each source has available right now (call this only
    after all three sources are primed/ready, i.e. after
    scene_grabber.prime(), zed_grabber.open_and_prime(), and
    wait_until_ready() have all already succeeded)."""

    def _panel(getter, label):
        frame = getter()
        panel = (
            np.zeros((panel_height, panel_height, 3), dtype=np.uint8) if frame is None
            else _resize_to_height(frame, panel_height)
        )
        cv2.putText(panel, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return panel

    def frame_getter():
        return np.concatenate(
            [_panel(wrist_getter, "WRIST"), _panel(scene_getter, "SCENE"), _panel(zed_getter, "ZED")], axis=1,
        )

    composite_shape = frame_getter().shape[:2]
    return frame_getter, composite_shape


def load_b1_oracle_stiffness(path: str) -> np.ndarray:
    """[k_pos_x,k_pos_y,k_pos_z,k_rot_x,k_rot_y,k_rot_z] from
    fit_b1_oracle_stiffness.py's output (summary.k_vector_contact_frame),
    same axis order _publish_target_stiffness/NON_COMPLIANT_STIFFNESS use."""
    with open(path) as f:
        summary = json.load(f)["summary"]["k_vector_contact_frame"]
    return np.array(
        [summary["fx"], summary["fy"], summary["fz"], summary["tx"], summary["ty"], summary["tz"]],
        dtype=np.float64,
    )


def run_one_recorded_episode(
    node: SmolVLADeployment, instruction: str, recorder, b1_stiffness=None,
    score_color=None, freeze_force_history=False,
) -> int:
    """Mirrors run_scripted_rollout.run_one_episode's loop exactly, plus a
    recorder.log_replan() call (skipped if recorder is None, i.e.
    --no-record-observations) at the exact point each observation is sent to
    the server, an optional b1_stiffness override (see module docstring)
    applied in place of whatever the server response would otherwise drive,
    score_color (M2 -- which physical mark colour to segment when scoring
    this run's before/after photos, set on node BEFORE _start_running() so
    its "before" capture actually fires -- see deploy_smolvla.py's
    _start_running/_stop_and_home), and freeze_force_history (M7 -- see
    below).

    Reads node.last_wipe_score (set by deploy_smolvla.py's _stop_and_home,
    called in this function's own finally block) after this returns to get
    M2's result -- not returned directly, since _stop_and_home is what
    actually computes it and this function's return value is steps_executed,
    unchanged from before this was added."""
    node.current_instruction = instruction
    node.score_color = score_color
    if not node._start_running():
        return 0
    if b1_stiffness is not None:
        # Overwrite _start_running's NON_COMPLIANT_STIFFNESS publish with B1's
        # oracle constant -- see module docstring for why this is B0's own
        # server response run through a fixed stiffness instead.
        node._publish_target_stiffness(np.log(b1_stiffness))

    step_period = 1.0 / node.chunk_hz
    running = True
    frozen_force_history = None  # M7 -- captured on first successful fetch, then reused, see below
    try:
        while running and rclpy.ok():
            # Drain whatever ROS callbacks have queued up since the last iteration (or
            # since wait_until_ready(), on the first pass): this loop -- copied from
            # run_scripted_rollout.run_one_episode -- never called rclpy.spin_once()
            # anywhere, which means ROS-callback-driven state (joint_q, flange_pose, the
            # wrench buffer -- the wrist image is no longer ROS-driven, see
            # WristCameraGrabber) silently froze at whatever it was when
            # wait_until_ready() last succeeded, for the REST OF THE EPISODE -- confirmed
            # on real hardware as stale proprioception. deploy_smolvla.run_interactive's
            # own loop avoids this by spinning once every iteration; matched here. A
            # single spin_once only services one ready callback, so drain a short bounded
            # burst (cheap: timeout_sec=0.0, each call returns immediately once nothing's
            # left) to actually catch up after get_action_chunk's blocking HTTP round
            # trip below -- which still can't itself be spun through (requests.post has
            # no interleaving hook, same gap run_interactive's own per-iteration call has
            # around its own get_action_chunk call); this closes the gap for the rest of
            # the loop, not that one blocking call.
            for _ in range(20):
                rclpy.spin_once(node, timeout_sec=0.0)

            scene_frame = node.get_scene_frame()
            wrist_frame = node.get_wrist_frame()
            current_state = node.get_current_state()
            force_history = node.get_force_history() if node.include_force_history else None
            if (
                scene_frame is None
                or wrist_frame is None
                or current_state is None
                or (node.include_force_history and force_history is None)
            ):
                node.get_logger().warn(
                    "Waiting for scene/wrist cameras, proprioception and wrench streams...",
                    throttle_duration_sec=2.0,
                )
                time.sleep(0.05)
                continue

            if freeze_force_history and force_history is not None:
                # M7 causal force ablation: hold force_history fixed at its first-
                # observed value for the rest of the episode rather than zeroing it
                # -- zeroing would push the observation out of the training
                # distribution and confound "not using force" with "broke on an OOD
                # input" (see modal-masking risk, proposal §8). Freezing at a real,
                # in-distribution value removes the *causal* information (it no
                # longer tracks what's actually happening at contact) while keeping
                # the input itself plausible -- see module docstring's M7 entry.
                if frozen_force_history is None:
                    frozen_force_history = force_history.copy()
                force_history = frozen_force_history

            scene_rgb = node._resize(scene_frame)
            wrist_rgb = node._resize(wrist_frame)
            action_chunk = node.get_action_chunk(
                scene_rgb, wrist_rgb, current_state, node.current_instruction, force_history
            )
            if recorder is not None:
                recorder.log_replan(
                    scene_rgb, wrist_rgb, current_state, force_history, node.current_instruction, action_chunk
                )
            if action_chunk is None:
                node.get_logger().error("Stopping episode -- action-chunk request failed.")
                break

            action_dim = action_chunk.shape[-1]
            if action_dim not in (7, 13):
                node.get_logger().error(
                    f"Unexpected action_dim={action_dim} (expected 7 for b0/b1/b2 or "
                    "13 for b3/b5); skipping chunk."
                )
                continue
            is_compliance_checkpoint = action_dim == 13 and b1_stiffness is None

            for i in range(min(node.steps_to_execute, action_chunk.shape[0])):
                start_time = time.time()
                if node.flange_pose is None:
                    running = False
                    break

                step = action_chunk[i]
                x_eq = step[0:6]
                if not node._step_jump_ok(x_eq):
                    running = False
                    break
                if is_compliance_checkpoint:
                    log_k, gripper_cmd = step[6:12], step[12]
                    node._publish_target_stiffness(log_k)
                elif b1_stiffness is not None:
                    # B1: fixed stiffness for the whole run, republished every
                    # step (cheap, and avoids depending on the controller
                    # holding _start_running's one-shot publish indefinitely).
                    gripper_cmd = step[6] if action_dim == 7 else step[12]
                    node._publish_target_stiffness(np.log(b1_stiffness))
                else:
                    gripper_cmd = step[6]

                node._publish_target_pose(x_eq)
                node._send_gripper_command(gripper_cmd)
                node.step_count += 1

                if node.step_count >= node.max_steps:
                    running = False
                    break

                elapsed = time.time() - start_time
                time.sleep(max(0.0, step_period - elapsed))
    finally:
        steps_executed = node.step_count
        node._stop_and_home()
    return steps_executed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", required=True, choices=POLICY_CHOICES,
                   help="label only (for output paths/summary) -- the checkpoint actually "
                        "running is whatever --server-url is serving; b1 has no checkpoint "
                        "of its own, see module docstring")
    p.add_argument("--referent", required=True, choices=INSTRUCTION_COLORS)
    p.add_argument("--manner", required=True, choices=INSTRUCTION_MANNERS)
    p.add_argument("--n-rollouts", type=int, default=20, help="tasks.md Day 18: 20 per policy for E1")
    p.add_argument("--skip-confirm", action="store_true",
                    help="don't pause for Enter between rollouts -- default pauses so the "
                         "operator can reset the scene by hand")
    p.add_argument("--ready-timeout-sec", type=float, default=15.0)
    # --- Passed straight through to SmolVLADeployment's constructor ---
    p.add_argument("--server-url", default="http://127.0.0.1:8000/act")
    p.add_argument("--follower-ns", default="follower")
    p.add_argument("--no-force-history", action="store_true", help="omit force_history (b0/b1 checkpoints)")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--scene-cv2-device", type=int, default=6)
    p.add_argument("--wrist-cv2-device", type=int, default=None,
                    help="cv2.VideoCapture device index for the wrist (RealSense) camera -- "
                         "read directly via V4L2 instead of its ROS Image topic (see "
                         "WristCameraGrabber for why); required unless --disable-wrist-camera. "
                         "Find it the same way as --zed-device: `for v in /dev/video*; do echo "
                         "\"$v: $(cat /sys/class/video4linux/$(basename $v)/name)\"; done`")
    p.add_argument("--action-scale", type=float, default=1.0)
    p.add_argument("--steps-to-execute", type=int, default=1,
                    help="low-level steps executed from one action chunk before requerying the "
                         "server for a fresh one -- default 1 is strictly reactive (requery every "
                         "step); higher (e.g. 3, ~100ms of receding horizon at chunk_hz=30) trades "
                         "reactivity for fewer server round trips, see deploy_smolvla.py's own "
                         "steps_to_execute param for the full tradeoff")
    p.add_argument("--max-steps", type=int, default=MAX_STEPS)
    p.add_argument("--disable-scene-camera", action="store_true")
    p.add_argument("--disable-wrist-camera", action="store_true")
    # --- B1 stiffness override ---
    p.add_argument("--b1-oracle-stiffness", default=DEFAULT_B1_ORACLE_STIFFNESS,
                    help="fit_b1_oracle_stiffness.py's output JSON -- only read/applied when --policy b1")
    # --- Metrics (M1/M2/M5/M7, see module docstring) ---
    p.add_argument("--score-color", choices=SCORE_COLORS, default=None,
                    help="M2: physical colour of the mark named by --referent for THIS session's "
                         "layout (--referent is which mark, left/right -- e.g. 'left'; --score-color "
                         "is what colour that mark actually is right now, e.g. 'blue' -- same "
                         "distinction deploy_smolvla.py's own GUI dialog makes). Required unless "
                         "--no-score-wipe is passed; without it, ink_removal_pct_targeted (and "
                         "therefore M1's derived success) is null for every rollout in this run.")
    p.add_argument("--no-score-wipe", action="store_true",
                    help="skip M2 wipe-scoring entirely (e.g. before-photo ROI/HSV ranges aren't "
                         "calibrated yet for this session) -- ink_removal_pct_targeted and success "
                         "are recorded as null rather than silently guessed at")
    p.add_argument("--success-threshold-pct", type=float, default=DEFAULT_SUCCESS_THRESHOLD_PCT,
                    help="M1: ink_removal_pct_targeted >= this -> success=True. Ignored (success "
                         "stays null) when M2 scoring is unavailable for a rollout.")
    p.add_argument("--max-safe-force-n", type=float, default=DEFAULT_MAX_SAFE_FORCE_N,
                    help="M5 protective-stop PROXY threshold -- see WrenchRecorder's docstring for "
                         "why this is a proxy, not real robot fault telemetry")
    p.add_argument("--freeze-force-history", action="store_true",
                    help="M7 causal force ablation: hold force_history fixed at its first-observed "
                         "value for the rest of each episode instead of tracking live wrench -- "
                         "meaningless for --policy b0 (no force input at all) or --policy b1 (b1's "
                         "own module-docstring override already ignores the served stiffness "
                         "regardless of force_history); run one full n-rollouts batch with this off "
                         "and a separate batch with it on per policy/condition to get M7's Delta")
    # --- Recording ---
    p.add_argument("--out-root", default=DEFAULT_OUT_ROOT,
                    help=f"default: {DEFAULT_OUT_ROOT}")
    p.add_argument("--no-record-observations", action="store_true",
                    help="skip saving per-replan scene/wrist JPEGs + steps.jsonl")
    p.add_argument("--no-record-video", action="store_true",
                    help="skip the stacked wrist/scene/ZED video recording entirely -- "
                         "otherwise --zed-device is required")
    p.add_argument("--zed-device", type=int, default=None,
                    help="cv2.VideoCapture device index for the externally connected ZED "
                         "camera (see `for v in /dev/video*; do echo \"$v: $(cat /sys/class/"
                         "video4linux/$(basename $v)/name)\"; done` to find it)")
    p.add_argument("--video-fps", type=float, default=24.0)
    p.add_argument("--video-panel-height", type=int, default=360,
                    help="each of the 3 panels (wrist/scene/zed) is resized to this height "
                         "(native aspect ratio kept) before being stacked left-to-right into "
                         "one combined_video.mp4 frame")
    p.add_argument("--zed-width", type=int, default=None)
    p.add_argument("--zed-height", type=int, default=None)
    p.add_argument("--video-sync-margin-sec", type=float, default=0.5,
                    help="delay between opening captures/the VideoWriter and the pacer's "
                         "start_time -- gives every grabber thread time to actually be "
                         "spun up and producing frames before frame 0's deadline")
    args = p.parse_args()
    if not args.no_record_video and args.zed_device is None:
        p.error("--zed-device is required unless --no-record-video is passed")
    if not args.disable_wrist_camera and args.wrist_cv2_device is None:
        p.error("--wrist-cv2-device is required unless --disable-wrist-camera is passed")
    if not args.no_score_wipe and args.score_color is None:
        p.error("--score-color is required unless --no-score-wipe is passed (see its help for why)")
    return args


def main() -> int:
    args = parse_args()
    instruction = f"wipe the {args.referent} mark {args.manner}"

    b1_stiffness = None
    if args.policy == "b1":
        b1_stiffness = load_b1_oracle_stiffness(args.b1_oracle_stiffness)
        print(f"[run_final_evaluation] B1 fixed stiffness (contact-frame diag, no rotation "
              f"applied -- see module docstring): {b1_stiffness}")

    run_group_dir = Path(args.out_root) / args.policy / f"{args.referent}_{args.manner}"
    run_group_dir.mkdir(parents=True, exist_ok=True)

    rclpy.init()
    node = SmolVLADeployment(
        server_url=args.server_url,
        follower_ns=args.follower_ns,
        include_force_history=not args.no_force_history,
        image_size=args.image_size,
        scene_cv2_device=args.scene_cv2_device,
        wrist_cv2_device=args.wrist_cv2_device,
        action_scale=args.action_scale,
        steps_to_execute=args.steps_to_execute,
        max_steps=args.max_steps,
        enable_scene_camera=not args.disable_scene_camera,
        enable_wrist_camera=not args.disable_wrist_camera,
    )
    wrist_grabber = None
    try:
        if node.enable_wrist_camera:
            # Created and started ONCE, before the readiness gate -- see WristCameraGrabber's
            # docstring for why this can't be scoped per-rollout the way ZedGrabber is:
            # wrist_rgb is a required policy observation, not just something recorded. Wraps
            # node.wrist_capture, which SmolVLADeployment's own __init__ already opened at
            # wrist_cv2_device -- no separate device open here (a second one would conflict
            # with it on the same V4L2 node).
            wrist_grabber = WristCameraGrabber(node)
            wrist_grabber.prime()
            wrist_grabber.start()
            node.get_logger().info(
                f"Wrist camera live via cv2.VideoCapture(device={args.wrist_cv2_device}), "
                f"native frame_shape={wrist_grabber.frame_shape}."
            )

        if node.enable_scene_camera:
            # Spin throughout the warmup wait, NOT a blocking time.sleep() (that was
            # run_scripted_rollout.py's original pattern, copied here initially and then
            # found -- on real hardware -- to overflow the DDS reliable-QoS buffer on the
            # 1kHz robot_state/measured_joint_states subscriptions: "sequence size exceeds
            # remaining buffer", which then left this process's readers wedged for the rest
            # of its life, no matter how long the later wait_until_ready spin-loop waited.
            # run_interactive's own warmup loop already spins for exactly this reason --
            # matching it here instead of inventing a second pattern.
            node.get_logger().info(f"Warming up scene camera for {node.camera_warmup_sec:.0f}s...")
            warmup_deadline = time.monotonic() + node.camera_warmup_sec
            while rclpy.ok() and time.monotonic() < warmup_deadline:
                rclpy.spin_once(node, timeout_sec=0.01)

        node.get_logger().info(f"Waiting up to {args.ready_timeout_sec:.0f}s for sensor streams...")
        if not wait_until_ready(node, args.ready_timeout_sec):
            node.get_logger().error(
                "Timed out waiting for scene/wrist cameras, proprioception, or wrench "
                "streams -- is the wrist camera / controller launch actually up?"
            )
            return 1

        for i in range(args.n_rollouts):
            if not args.skip_confirm:
                print(
                    f"[run_final_evaluation] {args.policy} rollout {i + 1}/{args.n_rollouts} -- "
                    f"instruction={instruction!r}. Reset the scene, then press Enter to start.",
                    flush=True,
                )
                input("> ")

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            run_dir = run_group_dir / f"run{i:03d}_{timestamp}"
            run_dir.mkdir(parents=True, exist_ok=True)

            recorder = RunRecorder(run_dir) if not args.no_record_observations else None

            # M3/M5 (peak/RMS force, protective-stop proxy) + the M4/M6 raw
            # signal -- started before the episode so it doesn't miss the
            # approach/first-contact transient, stopped in the finally block
            # below alongside the other per-rollout recorders.
            wrench_recorder = WrenchRecorder(node, run_dir, args.max_safe_force_n)
            wrench_recorder.start()

            scene_grabber = zed_grabber = None
            combined_video = None
            if not args.no_record_video:
                scene_grabber = SceneCameraGrabber(node)
                scene_grabber.prime()  # needs to succeed before we know it has a frame at all
                zed_grabber = ZedGrabber(
                    args.zed_device, fps_hint=args.video_fps, width=args.zed_width, height=args.zed_height,
                )
                zed_grabber.open_and_prime()  # needs to succeed before we know it has a frame at all

                scene_grabber.start()
                zed_grabber.start()

                # wrist_grabber (created once, before wait_until_ready -- see its docstring)
                # is already live by this point; falls back to a black panel if it's None
                # (--disable-wrist-camera).
                wrist_frame_getter = wrist_grabber.get_latest_bgr if wrist_grabber is not None else (lambda: None)
                stacked_getter, composite_shape = make_stacked_frame_getter(
                    wrist_frame_getter, scene_grabber.get_latest_bgr, zed_grabber.get_latest_bgr,
                    args.video_panel_height,
                )
                # Anchor for this one pacer's frame k -> start_time + k/fps schedule -- see
                # PacedVideoRecorder's docstring. Margin gives the VideoWriter below time to
                # actually open before frame 0's deadline.
                start_time = time.monotonic() + args.video_sync_margin_sec
                combined_video = PacedVideoRecorder(
                    stacked_getter, run_dir / "combined_video.mp4", args.video_fps,
                    start_time, composite_shape, csv_path=run_dir / "combined_video.frame_times.csv",
                )
                combined_video.start()
                node.get_logger().info(
                    f"Stacked wrist|scene|zed recording started (start_time={start_time:.3f} "
                    f"monotonic, {args.video_fps:.1f}fps, shape={composite_shape}) -> {run_dir}"
                )

            node.get_logger().info(f"Starting rollout {i + 1}/{args.n_rollouts}: {instruction!r}")
            t_start = time.time()
            score_color = None if args.no_score_wipe else args.score_color
            try:
                steps_executed = run_one_recorded_episode(
                    node, instruction, recorder, b1_stiffness=b1_stiffness,
                    score_color=score_color, freeze_force_history=args.freeze_force_history,
                )
            finally:
                # Stop the pacer before the grabbers (it may still be mid-tick, calling
                # into stacked_getter -- the grabbers must stay alive until that's done),
                # and SceneCameraGrabber.stop() specifically must run before this rollout's
                # `recorder`/the next rollout so node.get_scene_frame is restored promptly.
                if combined_video is not None:
                    combined_video.stop()
                if scene_grabber is not None:
                    scene_grabber.stop()
                if zed_grabber is not None:
                    zed_grabber.stop()
                wrench_recorder.stop()
                if recorder is not None:
                    recorder.close()
            t_end = time.time()

            # M2: node.last_wipe_score is set by deploy_smolvla.py's _stop_and_home
            # (called inside run_one_recorded_episode's own finally block, which has
            # already run by this point) -- None if scoring was skipped (score_color
            # None) or failed for this run.
            wipe_score = node.last_wipe_score
            ink_removal_pct_targeted = wipe_score["percent_wiped"] if wipe_score is not None else None
            # M1: derived from M2, null (not False) when M2 itself is unavailable --
            # a missing score and a real failed wipe must stay distinguishable
            # downstream, not silently collapsed to the same value.
            success = (
                bool(ink_removal_pct_targeted >= args.success_threshold_pct)
                if ink_removal_pct_targeted is not None else None
            )
            force_summary = wrench_recorder.summary()  # M3 + M5 proxy, see WrenchRecorder

            summary = {
                "policy": args.policy,
                "referent": args.referent,
                "manner": args.manner,
                "task": instruction,
                "rollout_index": i,
                "run_dir": str(run_dir),
                "steps_executed": steps_executed,
                "reached_max_steps": steps_executed >= args.max_steps,
                "duration_sec": t_end - t_start,
                "server_url": args.server_url,
                "n_observations_logged": recorder.n_observations if recorder is not None else 0,
                # --- M1/M2 ---
                "ink_removal_pct_targeted": ink_removal_pct_targeted,
                "success": success,
                "success_threshold_pct": args.success_threshold_pct,
                "wipe_score": wipe_score,  # full detail (score_color, before/after area px, viz path) or null
                # --- M3 + M5 proxy (M4/M6's raw signal is wrench.csv itself) ---
                "wrench": force_summary,
                # legacy top-level aliases -- same field names/meaning
                # scripts/evaluate_gate3.py already parses from
                # fr3_bilateral_teleop/run_pilot_rollout.py's pilot logs, kept so
                # this script's summaries are consumable by that same tool
                # without a schema-specific branch.
                "peak_force_n": force_summary["peak_force_n"],
                "protective_stop": force_summary["protective_stop"],
                # --- M7 ---
                "force_history_frozen": bool(args.freeze_force_history),
                "video": {
                    "path": "combined_video.mp4",
                    "layout": "wrist | scene | zed, left-to-right, each resized to "
                              f"panel_height={args.video_panel_height}px (native aspect kept)",
                    "fps": args.video_fps,
                    "start_time_monotonic": start_time,
                    "frames_written": combined_video.frames_written,
                    "frame_times_csv": "combined_video.frame_times.csv",
                } if not args.no_record_video else None,
                "b1_oracle_stiffness": b1_stiffness.tolist() if b1_stiffness is not None else None,
            }
            with open(run_dir / "rollout_summary.json", "w") as f:
                json.dump(summary, f, indent=2)

            success_str = "n/a (no M2 score)" if success is None else str(success)
            print(
                f"[run_final_evaluation] rollout {i + 1}/{args.n_rollouts} done -- "
                f"{steps_executed} steps executed, success={success_str}, "
                f"peak_force={force_summary['peak_force_n']:.1f}N, "
                f"protective_stop={force_summary['protective_stop']}, recorded to {run_dir}"
            )

        return 0
    finally:
        if wrist_grabber is not None:
            wrist_grabber.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
