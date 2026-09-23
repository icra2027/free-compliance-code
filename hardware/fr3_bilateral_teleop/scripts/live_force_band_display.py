#!/usr/bin/env python3
"""Live contact-force gauge with adverb target bands, for the operator during T1 data collection.

Day 6 task (compliance-vla/tasks.md): "Stand up the live normal-force display with
target bands (gently 3-6 N, normally 8-12 N, firmly 15-22 N). The policy never sees these
numbers." This script is deliberately a separate, standalone process from
data_recorder's `record_lerobot` node -- it never touches the LeRobot dataset, never
writes a frame, and has no coupling to recording state. That separation is what makes "the
policy never sees these numbers" true by construction rather than by convention.

What it actually displays: the magnitude of the follower's estimated external force
(||(fx, fy, fz)||), not a true board-normal force. Proposal §4.1 fits the real per-axis contact
frame (board normal vs. in-plane) offline, from the demo's own in-contact positions
(extract_impedance_labels.py::fit_contact_frame) -- that fit needs a completed demo and is not
available live. For a live operator gauge this is an acceptable approximation, not a shortcut
being smuggled past the reader: Day 5's pilot demos already measured a contact-frame anisotropy
ratio of 7-14x (normal stiffness/force dominates lateral by an order of magnitude) once contact
is firm, so during genuine wiping contact the force magnitude is, to first order, the normal
force. Treat this display as operator feedback only; the extraction pipeline's own offline
per-axis decomposition is what actually goes in the paper.

Usage (after bringing up bilateral teleop normally, e.g. teleop.launch.py):
    ros2 run fr3_bilateral_teleop live_force_band_display.py --target-manner gently
"""
import argparse
import sys
from collections import deque
from typing import Optional

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import WrenchStamped

# (low_n, high_n, BGR color) -- matches proposal §5 / tasks.md Day 6 exactly.
BANDS = {
    "gently": (3.0, 6.0, (110, 220, 110)),
    "normally": (8.0, 12.0, (60, 210, 240)),
    "firmly": (15.0, 22.0, (70, 70, 235)),
}


class ForceGaugeNode(Node):
    def __init__(self, wrench_topic: str, smoothing_window: int):
        super().__init__("live_force_band_display")
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=100)
        self._samples: deque = deque(maxlen=max(1, smoothing_window))
        self.latest_force_n: float = 0.0
        self.have_data = False
        self.create_subscription(WrenchStamped, wrench_topic, self._wrench_callback, qos)

    def _wrench_callback(self, msg: WrenchStamped) -> None:
        f = msg.wrench.force
        magnitude = float(np.sqrt(f.x * f.x + f.y * f.y + f.z * f.z))
        self._samples.append(magnitude)
        self.latest_force_n = float(np.mean(self._samples))
        self.have_data = True


def _band_for(force_n: float) -> Optional[str]:
    for name, (low, high, _color) in BANDS.items():
        if low <= force_n <= high:
            return name
    return None


def _draw_gauge(force_n: float, have_data: bool, max_force_n: float, target_manner: str) -> np.ndarray:
    width, height = 420, 620
    margin_top, margin_bottom = 60, 60
    gauge_h = height - margin_top - margin_bottom
    gauge_x0, gauge_w = 140, 120

    img = np.full((height, width, 3), (24, 24, 24), dtype=np.uint8)
    cv2.putText(img, "Contact Force", (24, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (235, 235, 235), 2, cv2.LINE_AA)
    cv2.putText(img, "(operator display only -- not recorded)", (24, 54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 140, 140), 1, cv2.LINE_AA)

    def y_for(force_val: float) -> int:
        clamped = max(0.0, min(force_val, max_force_n))
        return int(margin_top + gauge_h * (1.0 - clamped / max_force_n))

    # Background track.
    cv2.rectangle(img, (gauge_x0, margin_top), (gauge_x0 + gauge_w, margin_top + gauge_h), (55, 55, 55), -1)

    # Target bands.
    for name, (low, high, color) in BANDS.items():
        y_high = y_for(high)
        y_low = y_for(low)
        thickness = -1 if name == target_manner else 2
        cv2.rectangle(img, (gauge_x0, y_high), (gauge_x0 + gauge_w, y_low), color, thickness)
        label_y = (y_high + y_low) // 2
        cv2.putText(img, f"{name} [{low:g}-{high:g}N]", (gauge_x0 + gauge_w + 14, label_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # Axis ticks.
    for tick_n in range(0, int(max_force_n) + 1, 5):
        ty = y_for(float(tick_n))
        cv2.line(img, (gauge_x0 - 8, ty), (gauge_x0, ty), (150, 150, 150), 1)
        cv2.putText(img, str(tick_n), (gauge_x0 - 40, ty + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (150, 150, 150), 1, cv2.LINE_AA)

    # Current-value needle.
    needle_color = (255, 255, 255) if have_data else (90, 90, 90)
    ny = y_for(force_n)
    cv2.line(img, (gauge_x0 - 20, ny), (gauge_x0 + gauge_w + 20, ny), needle_color, 3)
    cv2.circle(img, (gauge_x0 - 20, ny), 5, needle_color, -1)

    band = _band_for(force_n) if have_data else None
    status_text = f"{force_n:5.1f} N" if have_data else "waiting for wrench..."
    status_color = BANDS[band][2] if band else (235, 235, 235)
    cv2.putText(img, status_text, (24, height - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2, cv2.LINE_AA)
    band_text = f"in band: {band}" if band else "out of band"
    cv2.putText(img, band_text, (200, height - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 1, cv2.LINE_AA)
    if target_manner:
        cv2.putText(img, f"target: {target_manner}", (24, height - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 100), 1, cv2.LINE_AA)

    return img


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--follower-namespace", default="franka_teleop/follower")
    parser.add_argument(
        "--wrench-topic", default=None,
        help="overrides --follower-namespace; full topic name for the follower's "
             "external_wrench_in_base_frame (geometry_msgs/WrenchStamped)")
    parser.add_argument(
        "--target-manner", default="", choices=["", "gently", "normally", "firmly"],
        help="which band to highlight as filled, e.g. for the current collection batch")
    parser.add_argument(
        "--smoothing-window", type=int, default=50,
        help="samples averaged for display (~50ms at the follower's 1kHz wrench rate); "
             "raw per-sample force is too jittery to read")
    parser.add_argument("--max-force", type=float, default=25.0)
    parser.add_argument("--window-name", default="Live Force Bands")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    wrench_topic = args.wrench_topic or (
        f"/{args.follower_namespace.strip('/')}/franka_robot_state_broadcaster/"
        "external_wrench_in_base_frame"
    )

    rclpy.init()
    node = ForceGaugeNode(wrench_topic, args.smoothing_window)
    node.get_logger().info(f"Subscribed to {wrench_topic}; target manner={args.target_manner or '(none)'}")
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.02)
            frame = _draw_gauge(node.latest_force_n, node.have_data, args.max_force, args.target_manner)
            cv2.imshow(args.window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
