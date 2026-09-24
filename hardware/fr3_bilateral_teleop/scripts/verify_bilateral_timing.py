#!/usr/bin/env python3
# Copyright (c) 2025 Franka Robotics GmbH
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Verify the 4-channel bilateral teleop controllers run at a true 1 kHz loop rate.

Subscribes to the timing diagnostics published by TeleopLeaderController and
TeleopFollowerController (see README "Verifying 1 kHz timing and channel
latency"), samples them for a fixed duration, reports achieved loop rate and
per-channel latency statistics, and plots a latency histogram.

The loop-period numbers come from the controller_manager-measured `update()`
period and need no clock synchronization. The channel-latency numbers are
`now - header.stamp` computed on the *consuming* robot and are only meaningful
if the leader and follower hosts' clocks are synchronized -- measure and
report that separately (e.g. with `chrony` / PTP stats) before trusting them.

Example:
    ros2 run fr3_bilateral_teleop verify_bilateral_timing.py \\
        --leader-namespace leader --follower-namespace follower --duration 30
"""
import argparse
import csv
import json
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import rclpy  # noqa: E402
from rclpy.node import Node  # noqa: E402
from std_msgs.msg import Float64  # noqa: E402

LOOP_PERIOD_SUFFIX = "loop_period"


@dataclass
class Channel:
    """One diagnostics topic being sampled, and the samples collected from it."""

    name: str
    topic: str
    samples: List[float] = field(default_factory=list)
    elapsed_s: List[float] = field(default_factory=list)


def build_channels(
    leader_ns: str, follower_ns: str, leader_controller: str, follower_controller: str
) -> "Dict[str, Channel]":
    """Map channel keys to the diagnostics topics published by the controllers."""

    def topic(ns: str, controller: str, suffix: str) -> str:
        return f"/{ns}/{controller}/diagnostics/{suffix}"

    return {
        "leader_loop_period": Channel(
            "leader_controller loop period",
            topic(leader_ns, leader_controller, "loop_period_us"),
        ),
        "follower_loop_period": Channel(
            "follower_controller loop period",
            topic(follower_ns, follower_controller, "loop_period_us"),
        ),
        "position_channel_latency": Channel(
            "channel 1: position (leader -> follower)",
            topic(follower_ns, follower_controller, "position_channel_latency_us"),
        ),
        "force_reflection_channel_latency": Channel(
            "channel 2: force reflection (follower -> leader)",
            topic(leader_ns, leader_controller, "force_reflection_channel_latency_us"),
        ),
        "force_feedforward_channel_latency": Channel(
            "channel 3: force feedforward (leader -> follower)",
            topic(follower_ns, follower_controller, "force_feedforward_channel_latency_us"),
        ),
    }


class BilateralTimingVerifier(Node):
    """Collects diagnostics samples for a fixed wall-clock duration."""

    def __init__(self, channels: "Dict[str, Channel]"):
        super().__init__("verify_bilateral_timing")
        self._channels = channels
        self._start_time = self.get_clock().now()
        self._subscriptions = [
            self.create_subscription(Float64, channel.topic, self._make_callback(channel), 50)
            for channel in channels.values()
        ]

    def _make_callback(self, channel: Channel):
        def _callback(msg: Float64) -> None:
            channel.samples.append(msg.data)
            channel.elapsed_s.append(self.elapsed())

        return _callback

    def elapsed(self) -> float:
        return (self.get_clock().now() - self._start_time).nanoseconds / 1.0e9


def summarize(values: List[float]) -> Optional[Dict[str, float]]:
    if not values:
        return None
    values_sorted = sorted(values)
    n = len(values_sorted)

    def percentile(p: float) -> float:
        idx = min(n - 1, int(round(p / 100.0 * (n - 1))))
        return values_sorted[idx]

    return {
        "count": n,
        "mean": statistics.fmean(values),
        "stdev": statistics.pstdev(values) if n > 1 else 0.0,
        "min": values_sorted[0],
        "p50": percentile(50),
        "p95": percentile(95),
        "p99": percentile(99),
        "max": values_sorted[-1],
    }


def plot_histograms(
    channels: "Dict[str, Channel]", target_rate_hz: float, latency_warn_ms: float,
    output_path: Path
) -> None:
    populated = [(key, ch) for key, ch in channels.items() if ch.samples]
    if not populated:
        return

    cols = 3
    rows = (len(populated) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4 * rows), squeeze=False)

    for i, (key, channel) in enumerate(populated):
        ax = axes[i // cols][i % cols]
        is_loop_period = key.endswith(LOOP_PERIOD_SUFFIX)
        if is_loop_period:
            data = channel.samples
            ax.set_xlabel("update() period (us)")
            ax.axvline(
                1.0e6 / target_rate_hz, color="tab:red", linestyle="--", label="target period")
        else:
            data = [v / 1000.0 for v in channel.samples]
            ax.set_xlabel("latency (ms)")
            ax.axvline(latency_warn_ms, color="tab:red", linestyle="--", label="warn threshold")
        ax.hist(data, bins=60, color="tab:blue", alpha=0.8)
        ax.set_title(channel.name, fontsize=10)
        ax.set_ylabel("count")
        ax.legend(fontsize=8)

    for i in range(len(populated), rows * cols):
        axes[i // cols][i % cols].axis("off")

    fig.suptitle("4-channel bilateral teleop: loop period and channel latency")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_csv(channels: "Dict[str, Channel]", output_path: Path) -> None:
    with open(output_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["channel", "topic", "elapsed_s", "value_us"])
        for key, channel in channels.items():
            for elapsed_s, value in zip(channel.elapsed_s, channel.samples):
                writer.writerow([key, channel.topic, f"{elapsed_s:.6f}", value])


def evaluate(
    channels: "Dict[str, Channel]", target_rate_hz: float, rate_tolerance_pct: float,
    latency_warn_ms: float
) -> "tuple[dict, bool]":
    report: Dict[str, dict] = {}
    ok = True

    for key, channel in channels.items():
        stats = summarize(channel.samples)
        report[key] = {"topic": channel.topic, "stats": stats}

        if stats is None:
            print(f"[FAIL] {channel.name}: no samples received on {channel.topic}")
            ok = False
            continue

        if key.endswith(LOOP_PERIOD_SUFFIX):
            achieved_rate_hz = 1.0e6 / stats["mean"] if stats["mean"] > 0 else 0.0
            rate_error_pct = abs(achieved_rate_hz - target_rate_hz) / target_rate_hz * 100.0
            passed = rate_error_pct <= rate_tolerance_pct
            report[key]["achieved_rate_hz"] = achieved_rate_hz
            ok = ok and passed
            print(
                f"[{'OK' if passed else 'FAIL'}] {channel.name}: {achieved_rate_hz:.1f} Hz "
                f"(target {target_rate_hz:.0f} Hz +/- {rate_tolerance_pct:.1f}%) | "
                f"period mean={stats['mean']:.1f}us p99={stats['p99']:.1f}us "
                f"max={stats['max']:.1f}us n={stats['count']}"
            )
        else:
            p99_ms = stats["p99"] / 1000.0
            passed = p99_ms <= latency_warn_ms
            ok = ok and passed
            print(
                f"[{'OK' if passed else 'WARN'}] {channel.name}: "
                f"mean={stats['mean'] / 1000.0:.3f}ms p99={p99_ms:.3f}ms "
                f"max={stats['max'] / 1000.0:.3f}ms (warn threshold {latency_warn_ms:.1f}ms) "
                f"n={stats['count']}"
            )

    return report, ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--leader-namespace", default="leader")
    parser.add_argument("--follower-namespace", default="follower")
    parser.add_argument("--leader-controller", default="leader_controller")
    parser.add_argument("--follower-controller", default="follower_controller")
    parser.add_argument(
        "--duration", type=float, default=30.0, help="Sampling duration in seconds (default: 30)"
    )
    parser.add_argument("--target-rate-hz", type=float, default=1000.0)
    parser.add_argument(
        "--rate-tolerance-pct", type=float, default=2.0,
        help="Achieved loop rate must be within this %% of --target-rate-hz"
    )
    parser.add_argument(
        "--latency-warn-ms", type=float, default=2.0,
        help="p99 channel latency above this is flagged (sync target: 2ms)"
    )
    parser.add_argument("--output-dir", default="bilateral_timing_report")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    channels = build_channels(
        args.leader_namespace, args.follower_namespace, args.leader_controller,
        args.follower_controller
    )

    rclpy.init()
    node = BilateralTimingVerifier(channels)
    try:
        print(f"Sampling {len(channels)} diagnostics topics for {args.duration:.1f}s:")
        for channel in channels.values():
            print(f"  - {channel.name}: {channel.topic}")
        while rclpy.ok() and node.elapsed() < args.duration:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        rclpy.shutdown()

    print()
    print(
        "Note: channel latencies are `now - header.stamp` on the consuming robot. They are "
        "only valid if leader and follower clocks are synchronized -- verify that separately "
        "(target < 2 ms) before trusting these numbers."
    )
    print()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report, ok = evaluate(
        channels, args.target_rate_hz, args.rate_tolerance_pct, args.latency_warn_ms)

    write_csv(channels, output_dir / "raw_samples.csv")
    plot_histograms(
        channels, args.target_rate_hz, args.latency_warn_ms,
        output_dir / "latency_histogram.png")

    with open(output_dir / "report.json", "w") as report_file:
        json.dump(
            {
                "args": vars(args),
                "pass": ok,
                "channels": report,
            },
            report_file,
            indent=2,
        )

    print()
    print(f"Wrote {output_dir / 'raw_samples.csv'}")
    print(f"Wrote {output_dir / 'latency_histogram.png'}")
    print(f"Wrote {output_dir / 'report.json'}")
    print()
    print("PASS" if ok else "FAIL")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
