#!/usr/bin/env python3
"""T1's primary metric (M2): per-mark pixel-difference ink-removal
coverage from a fixed overhead camera, scored per mark (not whole-board) so a policy that
wipes the wrong mark -- or wipes indiscriminately -- is penalized rather than credited.

"ink removed = per-mark pixel-difference coverage (a fixed ROI around each of the 4 marks)
... as a percentage. Continuous, automatic, no human labelling." (T1). This reuses this
project's existing camera feed rather than adding a dedicated metric camera: the fixed
scene_rgb topic (`/camera/color/image_raw`, Orbbec Femto Bolt, confirmed fixed/overhead-ish
and already the policy's own scene-observation input -- see data_recorder's
`scene_rgb_topic` default) doubles as the fixed camera this metric needs; §5 does not
specify a separate camera and this project has only ever mounted the two it already uses
(wrist + scene).

Pure image-processing module (numpy/OpenCV only, no ROS dependency) so it can be unit-tested
here and imported directly by scripts/run_pilot_rollout.py (which does the actual
before/after capture on the real rig).

Method: for each mark's ROI, convert BEFORE/AFTER crops to grayscale, threshold the
per-pixel absolute difference against `--diff-threshold` (ink removal changes local
brightness more than typical lighting/shadow drift), and report the fraction of ROI pixels
that changed as the ink-removal percentage for that mark. A fixed, hand-set threshold rather
than a learned one -- consistent with this project's other "documented design choice, not a
formula fixed by the method" pattern (evaluate_gate1.py's (ii)/(iii), offline_stiffness_benchmark.py's
M8 task definition) -- and reported alongside the raw diff so a reviewer/operator can see
what the threshold actually captured, not just the collapsed percentage.

Usage:
    python -m compliance_vla.policy.score_ink_removal --before before.png --after after.png \\
        --rois red:120,80,60,60 blue:300,80,60,60 green:120,260,60,60 black:300,260,60,60
    python -m compliance_vla.policy.score_ink_removal --self-test   # synthetic images, no camera needed
"""
import argparse
import json
import sys

import numpy as np

try:
    import cv2
    _HAVE_CV2 = True
except ImportError:  # pragma: no cover -- degrade gracefully rather than hard-require cv2
    _HAVE_CV2 = False


def _to_gray(img: np.ndarray) -> np.ndarray:
    """img: (H, W, 3) uint8 RGB -> (H, W) float64 grayscale. Uses cv2 if available (matches
    what any real capture pipeline would use); falls back to the standard luma weights
    otherwise so this module has no hard cv2 dependency for the self-test / CI path."""
    img = np.asarray(img)
    if img.ndim == 2:
        return img.astype(np.float64)
    if _HAVE_CV2:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float64)
    weights = np.array([0.299, 0.587, 0.114])
    return (img.astype(np.float64) @ weights)


def crop_roi(img: np.ndarray, roi):
    """roi: (x, y, w, h) in pixels, top-left origin -- matches the metric's "fixed ROI
    around each of the 4 marks" definition and its grid layout (corners + centre)."""
    x, y, w, h = roi
    return img[y:y + h, x:x + w]


def score_mark(before_roi: np.ndarray, after_roi: np.ndarray, diff_threshold: float = 25.0) -> dict:
    """Returns {"coverage_pct":, "mean_abs_diff":, "n_pixels":} for one mark's ROI.

    coverage_pct: fraction of ROI pixels whose |after - before| grayscale intensity exceeds
    `diff_threshold`, as a percentage -- the literal metric ("pixel-difference
    coverage ... as a percentage"). Clipped to [0, 100] is unnecessary by construction (a
    fraction of pixels can't exceed 1.0), stated here rather than defensively re-clamped.
    """
    before_roi = np.asarray(before_roi)
    after_roi = np.asarray(after_roi)
    if before_roi.shape[:2] != after_roi.shape[:2]:
        raise ValueError(f"ROI shape mismatch: before={before_roi.shape[:2]} after={after_roi.shape[:2]}")
    g_before = _to_gray(before_roi)
    g_after = _to_gray(after_roi)
    abs_diff = np.abs(g_after - g_before)
    changed = abs_diff > diff_threshold
    return {
        "coverage_pct": float(100.0 * changed.mean()),
        "mean_abs_diff": float(abs_diff.mean()),
        "n_pixels": int(changed.size),
    }


def score_rollout(before_img: np.ndarray, after_img: np.ndarray, rois: dict, diff_threshold: float = 25.0) -> dict:
    """rois: {mark_name: (x, y, w, h)}. Returns {mark_name: score_mark(...)} for every mark
    -- scored per-mark even for marks the instruction did NOT target, per §5's "so a policy
    that wipes the wrong mark -- or wipes indiscriminately -- is penalized rather than
    credited": the caller (run_pilot_rollout.py) is expected to compare the TARGETED mark's
    coverage against the other three's, not just report the targeted one in isolation.
    """
    return {
        name: score_mark(crop_roi(before_img, roi), crop_roi(after_img, roi), diff_threshold)
        for name, roi in rois.items()
    }


def _parse_roi_arg(spec: str):
    name, coords = spec.split(":")
    x, y, w, h = (int(v) for v in coords.split(","))
    return name, (x, y, w, h)


def run_self_test() -> int:
    rng = np.random.default_rng(0)
    H, W = 480, 640
    before = np.full((H, W, 3), 240, dtype=np.uint8)  # near-white board
    marks = {"red": (120, 80, 60, 60), "blue": (300, 80, 60, 60), "green": (120, 260, 60, 60), "black": (300, 260, 60, 60)}
    for name, (x, y, w, h) in marks.items():
        # Ink blob covers most (not all) of its ROI -- leaves a border so a "fully wiped"
        # after-image still exercises the diff against real background pixels too, not just
        # ink-vs-ink comparisons.
        before[y + 3:y + h - 3, x + 3:x + w - 3] = 30

    # "after": red mark fully wiped (ink -> background), others untouched, plus mild uniform
    # lighting drift everywhere (should NOT register as removal at the default threshold).
    after = before.copy().astype(np.int16)
    rx, ry, rw, rh = marks["red"]
    after[ry:ry + rh, rx:rx + rw] = 240
    after = np.clip(after + rng.integers(-4, 5, size=after.shape), 0, 255).astype(np.uint8)

    scores = score_rollout(before, after, marks, diff_threshold=25.0)
    print(json.dumps(scores, indent=2))

    # ink blob covers 81% of its ROI by construction (54x54 blob in a 60x60 ROI, see above) --
    # a fully-wiped mark should land close to that, not saturate at 100% (background border
    # pixels never had ink, so they never register a diff).
    assert scores["red"]["coverage_pct"] > 75.0, f"fully-wiped mark should score high coverage: {scores['red']}"
    for name in ("blue", "green", "black"):
        assert scores[name]["coverage_pct"] < 5.0, \
            f"untouched mark should score near-zero coverage even under mild lighting drift: {scores[name]}"

    # partial wipe: half the ROI cleared -> coverage should land near 50%, not saturate.
    before2 = before.copy()
    after2 = before.copy().astype(np.int16)
    bx, by, bw, bh = marks["blue"]
    after2[by:by + bh // 2, bx:bx + bw] = 240  # top half only
    after2 = np.clip(after2, 0, 255).astype(np.uint8)
    partial = score_mark(crop_roi(before2, marks["blue"]), crop_roi(after2, marks["blue"]), diff_threshold=25.0)
    assert 40.0 < partial["coverage_pct"] < 60.0, f"half-wiped ROI should score ~50%, got {partial}"

    # shape mismatch -> explicit error, not a silent broadcast/crash somewhere downstream.
    try:
        score_mark(np.zeros((10, 10, 3), dtype=np.uint8), np.zeros((10, 12, 3), dtype=np.uint8))
        raise AssertionError("expected ValueError on ROI shape mismatch")
    except ValueError:
        pass

    print("src/compliance_vla/policy/score_ink_removal.py self-test: PASS")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--before", help="path to the before-rollout overhead image")
    p.add_argument("--after", help="path to the after-rollout overhead image")
    p.add_argument("--rois", nargs="+", help="name:x,y,w,h per mark, e.g. red:120,80,60,60")
    p.add_argument("--diff-threshold", type=float, default=25.0)
    p.add_argument("--out", help="optional path to write the JSON scores to")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()

    if args.self_test:
        return run_self_test()

    if not (args.before and args.after and args.rois):
        print("error: --before/--after/--rois required unless --self-test", file=sys.stderr)
        return 2
    if not _HAVE_CV2:
        print("error: opencv-python required to read image files (only --self-test works without it)", file=sys.stderr)
        return 2

    before_img = cv2.cvtColor(cv2.imread(args.before), cv2.COLOR_BGR2RGB)
    after_img = cv2.cvtColor(cv2.imread(args.after), cv2.COLOR_BGR2RGB)
    rois = dict(_parse_roi_arg(s) for s in args.rois)
    scores = score_rollout(before_img, after_img, rois, args.diff_threshold)
    print(json.dumps(scores, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(scores, f, indent=2)
        print(f"[score_ink_removal] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
