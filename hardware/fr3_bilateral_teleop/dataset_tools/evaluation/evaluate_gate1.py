#!/usr/bin/env python3
"""Evaluate GATE 1 (H1) from a set of pilot demos -- proposal §3 / tasks.md Day 5.

Runs extract_impedance_labels.py's pipeline (imported, not reimplemented -- same pattern
diagnose_gravity_preload.py already uses against calibrate_payload.py) against each pilot
demo CSV and checks all three Gate 1 conditions:

  (i)   stiffness identifiable on >= 25% of CONTACT timesteps
  (ii)  within-demo K variation > 2x estimator noise
  (iii) contact-frame anisotropy ratio > 2

The proposal states (i) precisely but leaves (ii)/(iii) as qualitative statements with no
exact formula -- extract_impedance_labels.py already gives (i) directly (mask coverage
restricted to contact timesteps). This script defines defensible, documented computations
for (ii) and (iii) so Gate 1 is a single command once real pilot data exists, rather than
something computed by hand from the extraction reports:

  (ii) "estimator noise" is approximated from the SHORT-timescale (adjacent-output-step,
  33ms apart at 30Hz) jitter in the fitted K, on the reasoning that true stiffness cannot
  swing meaningfully between two 33ms-apart windows that mostly overlap (out of 300ms) --
  so consecutive-step differences are dominated by regression/measurement noise, not real
  signal. sigma_noise = std(diff(K)) / sqrt(2) (variance of a difference of two
  ~independent-noise samples is 2*sigma^2). "Within-demo K variation" is the plain std of K
  over all masked (identifiable) timesteps in the demo. Ratio = variation / noise.

  (iii) anisotropy ratio = max/min of the three TRANSLATIONAL axes' median masked K
  (ex, ey, ez) within one demo -- matches §5's own framing for T1 ("compliant along the
  normal, stiff in-plane"), i.e. comparing the board-normal axis against the two in-plane
  axes. Median, not mean, for robustness to the regression's own outlier windows.

Both are reported per-demo AND pooled, with the reasoning kept visible in the output --
this is a judgment call, not something to silently auto-decide, and the Day 5 exit
criterion explicitly says a real decision, not a formality.

Usage:
    ros2 run fr3_bilateral_teleop evaluate_gate1.py --input pilot1.csv pilot2.csv ...
    python3 dataset_tools/evaluation/evaluate_gate1.py --self-test   # synthetic data, no hardware
"""
import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

# extract_impedance_labels lives in ../labeling in the source tree. resolve(), not absolute():
# under a colcon --symlink-install, `ros2 run` executes this file through a symlink in
# install/.../lib/fr3_bilateral_teleop/. (Without symlink-install both scripts are installed
# side by side in that lib/ directory, so the import resolves from there anyway.)
_LABELING_DIR = str(Path(__file__).resolve().parent.parent / "labeling")
if _LABELING_DIR not in sys.path:
    sys.path.insert(0, _LABELING_DIR)

from extract_impedance_labels import (  # noqa: E402
    AXIS_NAMES, DEFAULT_SIGMA_F, extract_demo, load_demo_csv, parse_args as extraction_parse_args,
    synthetic_demo,
)

TRANSLATIONAL = [0, 1, 2]  # ex, ey, ez -- ez is the board-normal axis by construction


def estimator_noise_and_variation(k: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    k_masked = k[mask]
    if len(k_masked) < 10:
        return {"variation": float("nan"), "noise": float("nan"), "ratio": float("nan")}
    variation = float(np.std(k_masked))
    diffs = np.diff(k_masked)
    noise = float(np.std(diffs) / np.sqrt(2.0)) if len(diffs) > 0 else float("nan")
    ratio = variation / noise if noise > 1e-9 else float("inf")
    return {"variation": variation, "noise": noise, "ratio": ratio}


def anisotropy_ratio(k: np.ndarray, mask: np.ndarray) -> Dict:
    medians = []
    for axis in TRANSLATIONAL:
        m = mask[:, axis]
        medians.append(float(np.median(k[m, axis])) if m.sum() >= 5 else float("nan"))
    if any(np.isnan(medians)):
        return {"medians": medians, "ratio": float("nan")}
    lo, hi = min(medians), max(medians)
    ratio = hi / lo if lo > 1e-6 else float("inf")
    return {"medians": dict(zip([AXIS_NAMES[a] for a in TRANSLATIONAL], medians)), "ratio": ratio}


def evaluate_demo(path_or_label: str, result: Dict) -> Dict:
    coverage_contact = result["mask_coverage_within_contact"]
    cond_i_per_axis = (coverage_contact >= 0.25).tolist()
    cond_i = bool(coverage_contact.mean() >= 0.25)

    per_axis_ii = [
        estimator_noise_and_variation(result["k"][:, axis], result["mask"][:, axis])
        for axis in range(6)
    ]
    cond_ii_per_axis = [pa["ratio"] > 2.0 for pa in per_axis_ii]
    valid_ratios = [pa["ratio"] for pa in per_axis_ii if not np.isnan(pa["ratio"])]
    cond_ii = bool(valid_ratios and np.median(valid_ratios) > 2.0)

    aniso = anisotropy_ratio(result["k"], result["mask"])
    cond_iii = bool(not np.isnan(aniso["ratio"]) and aniso["ratio"] > 2.0)

    return {
        "input": path_or_label,
        "mask_coverage_within_contact": dict(
            zip([f"axis_{i}" for i in range(6)], coverage_contact.tolist())),
        "condition_i": {
            "pass": cond_i, "per_axis_pass": cond_i_per_axis,
            "coverage_mean": float(coverage_contact.mean())},
        "condition_ii": {"pass": cond_ii, "per_axis": [
            {"axis": AXIS_NAMES[i], **per_axis_ii[i], "pass": cond_ii_per_axis[i]}
            for i in range(6)]},
        "condition_iii": {"pass": cond_iii, **aniso},
        "all_pass": bool(cond_i and cond_ii and cond_iii),
    }


def run_self_test() -> int:
    print("--- Case A: anisotropic synthetic demo (should PASS all 3 conditions) ---")
    demo, _ = synthetic_demo(seed=0, k_modulation_depth=0.6)
    args = _default_extraction_args()
    sigma_f = np.array([0.1, 0.1, 0.1, 0.02, 0.02, 0.02])
    result = extract_demo(demo, args, sigma_f)
    ev = evaluate_demo("synthetic_anisotropic", result)
    _print_eval(ev)
    ok = ev["all_pass"]

    print("\n--- Case B: isotropic synthetic demo (anisotropy should FAIL) ---")
    isotropic_k = np.array([400.0, 400.0, 400.0, 20.0, 20.0, 20.0])
    demo_iso, _ = synthetic_demo(seed=0, true_k=isotropic_k, k_modulation_depth=0.6)
    result_iso = extract_demo(demo_iso, args, sigma_f)
    ev_iso = evaluate_demo("synthetic_isotropic", result_iso)
    _print_eval(ev_iso)
    if ev_iso["condition_iii"]["pass"]:
        print("FAIL: isotropic synthetic data should NOT pass the anisotropy condition")
        ok = False
    elif not (ev_iso["condition_i"]["pass"] and ev_iso["condition_ii"]["pass"]):
        print("FAIL: isotropic case should still pass (i) and (ii) -- only (iii) should differ")
        ok = False
    else:
        print("OK: isotropic case correctly fails only the anisotropy condition")

    print(f"\nself-test: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def _default_extraction_args():
    return extraction_parse_args([])


def _print_eval(ev: Dict) -> None:
    print(f"  (i)   coverage-within-contact >=25%: {ev['condition_i']['pass']} "
          f"(mean={ev['condition_i']['coverage_mean']:.1%})")
    print(f"  (ii)  K variation > 2x estimator noise (median across axes): "
          f"{ev['condition_ii']['pass']}")
    print(f"  (iii) anisotropy ratio > 2: {ev['condition_iii']['pass']} "
          f"(ratio={ev['condition_iii'].get('ratio', float('nan')):.2f})")
    print(f"  ALL THREE: {ev['all_pass']}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", nargs="+", type=Path, default=None)
    parser.add_argument(
        "--sigma-f", type=float, nargs=6, default=list(DEFAULT_SIGMA_F),
        metavar=("FX", "FY", "FZ", "TX", "TY", "TZ"))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return run_self_test()

    if not args.input:
        print("error: --input required unless --self-test", file=sys.stderr)
        return 2

    extraction_args = _default_extraction_args()
    sigma_f = np.array(args.sigma_f)

    evaluations: List[Dict] = []
    for path in args.input:
        demo = load_demo_csv(path)
        result = extract_demo(demo, extraction_args, sigma_f)
        if result["status"] != "ok":
            print(f"{path}: extraction FAILED -- {result['reason']}")
            continue
        ev = evaluate_demo(str(path), result)
        print(f"\n=== {path} ===")
        _print_eval(ev)
        evaluations.append(ev)

    if not evaluations:
        print("\nNo demos extracted successfully -- GATE 1 cannot be evaluated.")
        return 1

    n_pass = sum(1 for e in evaluations if e["all_pass"])
    print(f"\n=== GATE 1 SUMMARY: {n_pass}/{len(evaluations)} pilot demos pass all 3 "
          f"conditions ===")
    print(
        "Decision per tasks.md: proceed only if Gate 1 holds. This script reports per-demo "
        "results -- the pass/fail call across the pilot SET is a judgment for you to make "
        "explicitly, not an auto-decision.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
