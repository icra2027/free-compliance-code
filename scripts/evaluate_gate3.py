#!/usr/bin/env python3
"""Evaluate GATE 3 -- proposal §7 Week 3 plan / tasks.md Day 14:

  "GATE 3. B5 >= B0 on in-distribution T1 pilot. If not, debug the
  controller (is realized K tracking commanded K?) before touching the
  model."

Consumes the per-rollout JSON logs scripts/run_pilot_rollout.py writes for
n=5 T1 in-distribution pilot rollouts per policy (tasks.md Day 14: "Pilot
rollouts on T1 in-distribution, n = 5 per policy. Sanity only -- these are
NOT evaluation rollouts and must not be reported."). Because these pilots
are explicitly sanity-only and n=5 is far below this project's real
evaluation budget (n=20/cell, §6.5), Gate 3's comparison here is a plain
mean-ink-removal comparison, NOT a CI-based statistical test -- Holm-
Bonferroni / Wilson CIs / bootstrap (§6.5) are reserved for the real Week 4
evaluation (E1), and using them here would falsely dress up a 5-rollout
sanity check as a real result. This is a documented design choice, same
"defensible, not proposal-specified" pattern as evaluate_gate1.py's (ii)/
(iii) or the M8 task definition -- stated here rather than silently decided.

On FAIL, this script does NOT touch the model. Per the proposal's own
instruction, it prints the required next diagnostic step -- re-verify
realized-vs-commanded stiffness tracking (external/fr3_bilateral_teleop/
scripts/probe_variable_impedance_sinusoid.py, already validated on real
hardware Day 5) -- and stops there. Whether the controller or the model is
actually at fault is a judgment for whoever runs that diagnostic, not
something this script auto-decides.

Usage:
    python scripts/evaluate_gate3.py --b0-logs 'reports/pilot_rollouts/b0_*.json' \\
                                      --b5-logs 'reports/pilot_rollouts/b5_*.json'
    python scripts/evaluate_gate3.py --self-test   # synthetic rollout logs, no hardware needed
"""
import argparse
import glob
import json
import sys
from typing import Dict, List


def load_rollout_logs(pattern: str) -> List[Dict]:
    paths = sorted(glob.glob(pattern))
    logs = []
    for p in paths:
        with open(p) as f:
            logs.append(json.load(f))
    return logs


def summarize(logs: List[Dict], policy_label: str) -> Dict:
    if not logs:
        return {
            "policy": policy_label, "n": 0, "mean_ink_removal_pct": float("nan"),
            "success_rate": float("nan"), "n_protective_stops": 0,
        }
    ink = [ld["ink_removal_pct_targeted"] for ld in logs]
    success = [bool(ld["success"]) for ld in logs]
    stops = [bool(ld.get("protective_stop", False)) for ld in logs]
    return {
        "policy": policy_label,
        "n": len(logs),
        "mean_ink_removal_pct": sum(ink) / len(ink),
        "ink_removal_pct_per_rollout": ink,
        "success_rate": sum(success) / len(success),
        "n_protective_stops": sum(stops),
    }


DIAGNOSTIC_MESSAGE = """
GATE 3: FAIL -- B5 did not beat B0 on the in-distribution T1 pilot.

Per tasks.md Day 14 / proposal §7 Week 3: "If not, debug the controller (is
realized K tracking commanded K?) before touching the model. The controller
is the likelier culprit."

Next step, in order -- do NOT retrain or otherwise modify the model yet:
  1. Re-run the commanded-vs-realized stiffness validation on the REAL
     follower (not fake hardware):
       ros2 launch fr3_bilateral_teleop validate_variable_impedance.launch.py \\
           use_fake_hardware:=false robot_ip:=<follower_ip>
       ros2 run fr3_bilateral_teleop probe_variable_impedance_sinusoid.py --duration 20
  2. Compare the resulting commanded-vs-realized-stiffness + energy-tank
     figure against Day 5's baseline
     (external/fr3_bilateral_teleop's sinusoid_probe_*.png) -- look
     specifically for stiffness-rate-limiter stalls or energy-tank
     depletion during the pilot rollouts (see this script's
     `diagonal_dropped_fraction` field in each rollout log too -- a large
     value there means the fixed contact-frame rotation
     (scripts/fit_frozen_contact_frame.py) is dropping a lot of the
     predicted anisotropy, which would also suppress B5's advantage
     without the model itself being at fault).
  3. Only once the controller is confirmed tracking correctly should the
     model/training pipeline be treated as a suspect.
""".strip()


def evaluate(b0_summary: Dict, b5_summary: Dict) -> Dict:
    if b0_summary["n"] == 0 or b5_summary["n"] == 0:
        return {"decision": "INCONCLUSIVE", "reason": "missing rollout logs for b0 and/or b5"}
    passed = b5_summary["mean_ink_removal_pct"] >= b0_summary["mean_ink_removal_pct"]
    return {
        "decision": "PASS" if passed else "FAIL",
        "b0_mean_ink_removal_pct": b0_summary["mean_ink_removal_pct"],
        "b5_mean_ink_removal_pct": b5_summary["mean_ink_removal_pct"],
        "margin_pct": b5_summary["mean_ink_removal_pct"] - b0_summary["mean_ink_removal_pct"],
    }


def _print_summary(summary: Dict) -> None:
    print(f"  {summary['policy']}: n={summary['n']}, mean_ink_removal={summary['mean_ink_removal_pct']:.1f}%, "
          f"success_rate={summary['success_rate']:.2f}, protective_stops={summary['n_protective_stops']}")


def run_self_test() -> int:
    def make_logs(policy, ink_values, successes, stops=None):
        stops = stops or [False] * len(ink_values)
        return [
            {"policy": policy, "rollout_index": i, "task": "wipe the red mark firmly",
             "ink_removal_pct_targeted": v, "success": s, "protective_stop": st}
            for i, (v, s, st) in enumerate(zip(ink_values, successes, stops))
        ]

    # PASS case: B5 clearly beats B0.
    b0_logs = make_logs("b0", [40.0, 35.0, 45.0, 38.0, 42.0], [True, False, True, True, False])
    b5_logs = make_logs("b5", [70.0, 65.0, 80.0, 72.0, 68.0], [True, True, True, True, True])
    b0_sum, b5_sum = summarize(b0_logs, "b0"), summarize(b5_logs, "b5")
    result = evaluate(b0_sum, b5_sum)
    print("=== self-test: PASS scenario ===")
    _print_summary(b0_sum)
    _print_summary(b5_sum)
    print(result)
    assert result["decision"] == "PASS", result
    assert abs(result["margin_pct"] - 31.0) < 1e-6, result  # mean(70,65,80,72,68)=71 vs mean(40,35,45,38,42)=40

    # FAIL case: B5 underperforms B0 -- must print the diagnostic, not silently pass.
    b5_logs_bad = make_logs("b5", [20.0, 15.0, 25.0, 18.0, 22.0], [False, False, True, False, False])
    b5_sum_bad = summarize(b5_logs_bad, "b5")
    result_bad = evaluate(b0_sum, b5_sum_bad)
    print("\n=== self-test: FAIL scenario ===")
    _print_summary(b0_sum)
    _print_summary(b5_sum_bad)
    print(result_bad)
    assert result_bad["decision"] == "FAIL", result_bad

    # missing logs -> INCONCLUSIVE, never a false PASS/FAIL.
    empty_sum = summarize([], "b5")
    result_missing = evaluate(b0_sum, empty_sum)
    assert result_missing["decision"] == "INCONCLUSIVE", result_missing

    print("\nscripts/evaluate_gate3.py self-test: PASS")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--b0-logs", help="glob pattern for B0's pilot rollout JSON logs")
    p.add_argument("--b5-logs", help="glob pattern for B5's pilot rollout JSON logs")
    p.add_argument("--b2-logs", help="optional, context only -- not part of the Gate 3 decision")
    p.add_argument("--b3-logs", help="optional, context only -- not part of the Gate 3 decision")
    p.add_argument("--b4-logs", help="optional, context only -- not part of the Gate 3 decision")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()

    if args.self_test:
        return run_self_test()

    if not (args.b0_logs and args.b5_logs):
        print("error: --b0-logs and --b5-logs required unless --self-test", file=sys.stderr)
        return 2

    b0_summary = summarize(load_rollout_logs(args.b0_logs), "b0")
    b5_summary = summarize(load_rollout_logs(args.b5_logs), "b5")
    print("=== GATE 3 pilot summary (sanity only -- NOT evaluation numbers, do not report) ===")
    _print_summary(b0_summary)
    _print_summary(b5_summary)
    for label, pattern in (("b2", args.b2_logs), ("b3", args.b3_logs), ("b4", args.b4_logs)):
        if pattern:
            _print_summary(summarize(load_rollout_logs(pattern), label))

    result = evaluate(b0_summary, b5_summary)
    print(f"\nGATE 3 decision: {result['decision']}")
    if result["decision"] == "INCONCLUSIVE":
        print(f"reason: {result['reason']}")
        return 1
    if result["decision"] == "FAIL":
        print(f"\n{DIAGNOSTIC_MESSAGE}")
        return 1

    print(f"B5 mean ink-removal {result['b5_mean_ink_removal_pct']:.1f}% >= "
          f"B0 mean ink-removal {result['b0_mean_ink_removal_pct']:.1f}% "
          f"(margin {result['margin_pct']:+.1f} pct pts). Proceed per tasks.md Day 15.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
