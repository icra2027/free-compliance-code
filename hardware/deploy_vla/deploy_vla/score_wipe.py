"""Standalone (no rclpy) tool for testing automated wipe-scoring: given a "before"
and an "after" photo of the surface, segments the coloured mark by HSV threshold in
each, keeps only the largest matching blob (so stray same-hue pixels elsewhere in
the scene -- robot links, background clutter -- don't get counted), and reports
percent_wiped = 100 * (1 - after_mark_px / before_mark_px).

This is a dev/calibration tool, not something run_scripted_rollout.py depends on --
its only job is to let you check the scoring approach (and dial in HSV thresholds
for your actual mark colour/lighting) against real photos before trusting it.

Usage:
    # score two already-captured photos
    python3 score_wipe.py --before before.jpg --after after.jpg --color blue

    # capture the before/after pair live off the scene camera (same
    # cv2.VideoCapture device deploy_smolvla.py uses) through a GUI window: watch
    # the live feed and press SPACE to capture each of the before/after shots
    python3 score_wipe.py --live --color red --scene-cv2-device 6

    # interactively tune HSV thresholds against a photo before trusting the
    # defaults for your mark colour/lighting -- shows the original photo alongside
    # a live filtered view (only pixels currently passing the threshold), and
    # prints the bounds to pass back in
    python3 score_wipe.py --tune --before before.jpg --color blue

    # same, but tune against the live camera feed instead of a still photo --
    # both windows update continuously, not just the filtered one
    python3 score_wipe.py --tune --live --color blue --scene-cv2-device 6

    # resume refining bounds you already tuned (sliders seed from these instead of
    # the hardcoded --color defaults) rather than starting over from scratch
    python3 score_wipe.py --tune --before before.jpg --color blue --hsv-lower 100 70 50 --hsv-upper 130 255 255

    # interactively find the pixel ROI that frames just the wipe board (drag X/Y/W/H
    # sliders, see the crop live), then reuse the printed --roi on a real run
    python3 score_wipe.py --roi-tune --live --scene-cv2-device 6

    # preview a --roi you already have against the live feed (e.g. after
    # repositioning the camera/board) without going back into slider-tuning
    python3 score_wipe.py --roi-preview --live --roi 100 50 300 300 --scene-cv2-device 6

    # same, but also pass --color: adds live H/S/V sliders plus a third window
    # showing the filtered-only preview (same as --tune) so you can see the
    # detected-pixel mask and dial in thresholds on the spot, no separate --tune
    # pass needed, plus a live wipe-percentage score -- drag the sliders, press
    # 'b' once the mask cleanly covers the clean mark to set the baseline, then
    # wipe and watch it update
    python3 score_wipe.py --roi-preview --live --roi 100 50 300 300 --color blue

Or via the package entry point:
    ros2 run deploy_vla score_wipe --before before.jpg --after after.jpg --color blue
"""
import argparse
import os
import sys
from datetime import datetime

import cv2
import numpy as np

# HSV bounds (OpenCV convention: H in [0,179], S,V in [0,255]) as a starting point --
# red wraps around hue 0, hence two ranges. Real mark material/lighting will very
# likely need retuning; use --tune to find your own bounds rather than trusting these.
DEFAULT_HSV_RANGES = {
    # Re-measured 2026-09-10 against wipe_scores/tune_crop_20260910_114551_293.png --
    # the actual mark is a pale/desaturated pink (S only ~17-57, not a vivid red),
    # so the old S>=70 lower bound matched zero pixels. Hue clusters at 161-172 with
    # a few outliers near 0-10 (the wraparound), V is bright (142-194, light mark on
    # a light board) -- bounds below have margin on both sides of the measured range.
    "red": [((0, 15, 120), (10, 255, 255)), ((150, 15, 120), (179, 255, 255))],
    "blue": [((25, 35, 50), (145, 255, 255))],
}


def load_image_rgb(path: str) -> np.ndarray:
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def open_scene_camera(device: int) -> cv2.VideoCapture:
    """Opens `device` the same way deploy_smolvla.py's own scene camera does:
    explicit CAP_V4L2 backend (this box's OpenCV probes GStreamer first
    otherwise, which has been unreliable here) plus explicit YUYV FOURCC
    negotiation. Without the FOURCC call, cv2.VideoCapture on this camera has
    been observed to hand back a frame that never actually updates (reads keep
    succeeding, but against a stale/degenerate stream) instead of a live feed --
    this is the fix for that, not just a style match. Does a verification read,
    same as deploy_smolvla.py's own scene_capture setup, so a wrong /dev/video*
    node (the Femto Bolt exposes several) fails loudly here instead of quietly
    producing a frozen picture."""
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open cv2.VideoCapture device {device!r}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
    ok, _ = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(
            f"Opened device {device!r} but a verification read failed -- likely the "
            "wrong /dev/video* node (the Femto Bolt exposes several); check `for v in "
            "/dev/video*; do echo \"$v: $(cat /sys/class/video4linux/$(basename $v)/name)\"; "
            "done` and pass the right one via --scene-cv2-device."
        )
    return cap


class CaptureCancelled(Exception):
    """Raised when the operator quits the live-capture GUI (Q) before both the
    before and after shots have been taken."""


def run_capture_gui(device: int):
    """Live cv2 preview window -- same imshow/waitKey idiom as deploy_smolvla.py's
    _render_preview loop -- that walks the operator through capturing the before
    and after shots by watching the feed and pressing a key, instead of blindly
    pressing Enter in a terminal with no visual confirmation of what got captured.

    Controls: SPACE captures the current step's frame, R retakes the previous
    step, Q cancels. Returns (before_rgb, after_rgb).
    """
    cap = open_scene_camera(device)

    win = "score_wipe capture  [SPACE] capture  [R] retake  [Q] cancel"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    steps = [
        ("before", "Step 1/2: position camera over the marked surface, then press SPACE"),
        ("after", "Step 2/2: wipe the mark, then press SPACE to capture the result"),
    ]
    captured = {}
    step_idx = 0
    try:
        while step_idx < len(steps):
            ok, frame_bgr = cap.read()
            if not ok:
                raise RuntimeError(f"Failed to read from device {device!r}")
            # Femto Bolt is mounted upside down -- same fix deploy_smolvla.py's
            # get_scene_frame applies, so a photo captured here matches what a real
            # run actually feeds the scorer.
            frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_180)

            key_name, hint = steps[step_idx]
            disp = frame_bgr.copy()
            cv2.putText(disp, hint, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(
                disp, "[SPACE] capture   [R] retake previous   [Q] cancel",
                (10, disp.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
            )
            cv2.imshow(win, disp)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):
                captured[key_name] = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                print(f"[score_wipe] captured '{key_name}' image")
                # Brief freeze-frame so the operator sees what was captured.
                confirm = frame_bgr.copy()
                cv2.putText(confirm, f"captured '{key_name}'!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.imshow(win, confirm)
                cv2.waitKey(500)
                step_idx += 1
            elif key == ord("r") and step_idx > 0:
                step_idx -= 1
                captured.pop(steps[step_idx][0], None)
            elif key == ord("q"):
                raise CaptureCancelled("capture window closed before both shots were taken")
    finally:
        cap.release()
        cv2.destroyWindow(win)

    return captured["before"], captured["after"]


def parse_roi(roi_arg):
    if roi_arg is None:
        return None
    x, y, w, h = roi_arg
    return int(x), int(y), int(w), int(h)


def apply_roi(img_rgb: np.ndarray, roi):
    if roi is None:
        return img_rgb, (0, 0)
    x, y, w, h = roi
    return img_rgb[y:y + h, x:x + w], (x, y)


def hsv_ranges_for(color: str, hsv_lower, hsv_upper):
    """Explicit --hsv-lower/--hsv-upper (from --tune output) override the built-in
    default ranges for `color` entirely."""
    if hsv_lower is not None or hsv_upper is not None:
        if hsv_lower is None or hsv_upper is None:
            raise ValueError("--hsv-lower and --hsv-upper must be given together")
        return [(tuple(hsv_lower), tuple(hsv_upper))]
    return DEFAULT_HSV_RANGES[color]


def segment_mark(img_rgb: np.ndarray, ranges, roi=None, min_blob_px: int = 50):
    """Returns (full_frame_bool_mask, pixel_area) for the largest connected blob
    matching any of `ranges` within `roi` (default: whole frame). Morphological
    open+close first drops speckle noise before the largest-component filter, so a
    handful of stray matching pixels elsewhere in the scene can't masquerade as (or
    get merged into) the actual mark."""
    cropped, (ox, oy) = apply_roi(img_rgb, roi)
    hsv = cv2.cvtColor(cropped, cv2.COLOR_RGB2HSV)

    raw_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in ranges:
        raw_mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))

    # 3x3, not the more aggressive 5x5 this started as: a 5x5 open erodes away
    # most of a thin/multi-loop mark (measured 2026-09-10 against a coiled-wire-
    # style mark -- 5x5 left 54px of an actual ~424px raw match) while 3x3 still
    # drops single-pixel background speckle just fine.
    kernel = np.ones((3, 3), np.uint8)
    clean = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean, connectivity=8)
    full_mask = np.zeros(img_rgb.shape[:2], dtype=bool)
    if n_labels <= 1:
        return full_mask, 0  # background only -- no matching blob found

    # label 0 is background; pick the largest of the rest.
    areas = stats[1:, cv2.CC_STAT_AREA]
    best_label = 1 + int(np.argmax(areas))
    best_area = int(areas[best_label - 1])
    if best_area < min_blob_px:
        return full_mask, 0

    crop_mask = labels == best_label
    full_mask[oy:oy + crop_mask.shape[0], ox:ox + crop_mask.shape[1]] = crop_mask
    return full_mask, best_area


def overlay_mask(img_rgb: np.ndarray, mask: np.ndarray, rgb_color=(255, 0, 255)) -> np.ndarray:
    out = img_rgb.copy()
    out[mask] = (0.4 * out[mask] + 0.6 * np.array(rgb_color)).astype(np.uint8)
    return out


def draw_hsv_readout(img_bgr: np.ndarray, lo, hi, origin=(10, 25)) -> None:
    """Draws the current H/S/V lo/hi slider values onto `img_bgr` in place, right
    next to wherever the trackbars themselves are (OpenCV attaches trackbar value
    labels to the OS widget, which can be too small/cut off to read at a glance --
    this makes the numbers legible directly on the image). Black outline behind
    green fill so it stays readable over both bright and near-black (filtered)
    backgrounds."""
    text = f"H:[{lo[0]}-{hi[0]}]  S:[{lo[1]}-{hi[1]}]  V:[{lo[2]}-{hi[2]}]"
    cv2.putText(img_bgr, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img_bgr, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)


def save_crop_image(out_dir: str, prefix: str, img_bgr: np.ndarray) -> str:
    """Saves `img_bgr` (an undecorated crop -- no overlay/trackbar text) to
    `out_dir/<prefix>_<timestamp>.png` and returns the path. Millisecond-precision
    timestamp so repeated 's' presses in the same second don't collide."""
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    path = os.path.join(out_dir, f"{prefix}_{stamp}.png")
    cv2.imwrite(path, img_bgr)
    return path


def run_tune(frame_source, roi, seed_lower, seed_upper, out_dir: str) -> None:
    """Interactive HSV trackbar window, seeded from (seed_lower, seed_upper) --
    pass a color's DEFAULT_HSV_RANGES on a first pass, or a previous
    --hsv-lower/--hsv-upper (via hsv_ranges_for) to resume refining values you
    already tuned, instead of resetting to the hardcoded defaults every time.

    `frame_source` is called once per loop iteration and must return a BGR frame
    (same convention as run_roi_tune/run_roi_preview) -- a live cv2.VideoCapture
    read for --live, so the "original" window is a genuinely live feed rather
    than a single frozen snapshot, or a closure replaying one cached photo for
    --before. `roi` (or None) is applied every frame via apply_roi, same as a
    real scoring run would. Shows the (live) original in one window and the
    live filtered result (only pixels currently passing the threshold,
    everything else blacked out) in a second, so you can directly compare what's
    being kept against the source as you drag the sliders. Pressing 's' anytime
    saves the current (ROI-cropped, if `roi` is given) original image to
    `out_dir`. Prints the resulting bounds on exit ('q' or closing the window) so
    they can be passed back via --hsv-lower/--hsv-upper on a real scoring (or
    --roi-preview) run."""
    frame_bgr = frame_source()
    if frame_bgr is None:
        raise RuntimeError("frame_source produced no frame")
    cropped0, _ = apply_roi(frame_bgr, roi)

    win_orig = "tune -- original"
    win_filtered = "tune -- filtered (q to quit)"
    cv2.namedWindow(win_orig, cv2.WINDOW_NORMAL)
    cv2.namedWindow(win_filtered, cv2.WINDOW_NORMAL)
    # A window that auto-sizes tightly around a small image can leave too little
    # room for 6 trackbars to render/respond to drags on some backends (GTK in
    # particular) -- force both windows to a sane minimum size up front.
    disp_w = max(cropped0.shape[1], 400)
    cv2.resizeWindow(win_orig, disp_w, cropped0.shape[0])
    cv2.resizeWindow(win_filtered, disp_w, cropped0.shape[0] + 150)
    for name, val, maxval in [
        ("H lo", seed_lower[0], 179), ("H hi", seed_upper[0], 179),
        ("S lo", seed_lower[1], 255), ("S hi", seed_upper[1], 255),
        ("V lo", seed_lower[2], 255), ("V hi", seed_upper[2], 255),
    ]:
        cv2.createTrackbar(name, win_filtered, val, maxval, lambda _v: None)

    print("[score_wipe --tune] adjust sliders until the filtered image cleanly "
          "shows just the mark. 's' saves the current original crop to a file, "
          "'q' quits. (If the filtered window never changes, watch the console "
          "below -- it logs every slider read-back change to tell you whether "
          "the drag is even being registered.)")
    last_bounds = None
    try:
        while True:
            frame_bgr = frame_source()
            if frame_bgr is None:
                raise RuntimeError("frame_source produced no frame")
            cropped, _ = apply_roi(frame_bgr, roi)
            hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)

            lo = np.array([cv2.getTrackbarPos(n, win_filtered) for n in ("H lo", "S lo", "V lo")])
            hi = np.array([cv2.getTrackbarPos(n, win_filtered) for n in ("H hi", "S hi", "V hi")])
            bounds = (tuple(lo), tuple(hi))
            if bounds != last_bounds:
                print(f"[score_wipe --tune] slider read-back: lo={bounds[0]} hi={bounds[1]}")
                last_bounds = bounds
            mask = cv2.inRange(hsv, lo, hi)
            filtered_bgr = cv2.bitwise_and(cropped, cropped, mask=mask)
            draw_hsv_readout(filtered_bgr, lo, hi)
            cv2.imshow(win_orig, cropped)
            cv2.imshow(win_filtered, filtered_bgr)

            key = cv2.waitKey(30) & 0xFF
            if key == ord("s"):
                path = save_crop_image(out_dir, "tune_crop", cropped)
                print(f"[score_wipe --tune] saved crop to {path}")
            elif key == ord("q"):
                break
    finally:
        cv2.destroyWindow(win_orig)
        cv2.destroyWindow(win_filtered)
    print(f"[score_wipe --tune] chosen bounds: --hsv-lower {lo[0]} {lo[1]} {lo[2]} "
          f"--hsv-upper {hi[0]} {hi[1]} {hi[2]}")


def clamp_roi(x: int, y: int, w: int, h: int, frame_w: int, frame_h: int):
    """Clamps an (x, y, w, h) box so it stays fully inside a frame_w x frame_h
    frame. Pulled out as a pure function so the ROI-tuner's per-frame math is
    testable without a display."""
    x = max(0, min(x, frame_w - 1))
    y = max(0, min(y, frame_h - 1))
    w = max(1, min(w, frame_w - x))
    h = max(1, min(h, frame_h - y))
    return x, y, w, h


def run_roi_tune(frame_source, out_dir: str):
    """Interactive X/Y/W/H trackbar window (debug-mode crop finder) for locating
    the pixel ROI that frames just the wipe board, so it can be reused via --roi
    on a real scoring run without the mark-colour detector picking up same-hue
    clutter elsewhere in the scene (robot links, background).

    `frame_source` is called once per loop iteration and must return a BGR frame
    (a live cv2.VideoCapture read, or a closure returning a single cached photo
    for tuning against a still image instead of the live feed). Draws the crop
    rectangle on the full frame and shows the cropped region in a second window
    so you can see exactly what --roi would keep. Press 's' to save the current
    cropped image (undecorated -- no rectangle/text) to `out_dir`, 'q' to accept
    the current sliders; the chosen box is printed as --roi X Y W H and returned.
    """
    frame_bgr = frame_source()
    if frame_bgr is None:
        raise RuntimeError("frame_source produced no frame")
    frame_h, frame_w = frame_bgr.shape[:2]

    win = "roi tune (q to accept)"
    crop_win = "roi tune -- cropped preview"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.namedWindow(crop_win, cv2.WINDOW_NORMAL)
    # Sliders start at the full frame (x=0,y=0,w=frame_w,h=frame_h) -- drag them
    # inward to shrink the box down onto the wipe board.
    cv2.createTrackbar("X", win, 0, frame_w - 1, lambda _v: None)
    cv2.createTrackbar("Y", win, 0, frame_h - 1, lambda _v: None)
    cv2.createTrackbar("W", win, frame_w, frame_w, lambda _v: None)
    cv2.createTrackbar("H", win, frame_h, frame_h, lambda _v: None)

    print("[score_wipe --roi-tune] drag X/Y/W/H until the cropped-preview window "
          "frames just the wipe board. 's' saves the current crop to a file, "
          "'q' accepts.")
    x, y, w, h = 0, 0, frame_w, frame_h
    crop_bgr = frame_bgr
    try:
        while True:
            frame_bgr = frame_source()
            if frame_bgr is None:
                raise RuntimeError("frame_source produced no frame")

            x = cv2.getTrackbarPos("X", win)
            y = cv2.getTrackbarPos("Y", win)
            w = cv2.getTrackbarPos("W", win)
            h = cv2.getTrackbarPos("H", win)
            x, y, w, h = clamp_roi(x, y, w, h, frame_w, frame_h)
            crop_bgr = frame_bgr[y:y + h, x:x + w]

            disp = frame_bgr.copy()
            cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(disp, f"ROI: x={x} y={y} w={w} h={h}   [S] save crop   [Q] accept",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(win, disp)
            cv2.imshow(crop_win, crop_bgr)

            key = cv2.waitKey(30) & 0xFF
            if key == ord("s"):
                path = save_crop_image(out_dir, "roi_tune_crop", crop_bgr)
                print(f"[score_wipe --roi-tune] saved crop to {path}")
            elif key == ord("q"):
                break
    finally:
        cv2.destroyWindow(win)
        cv2.destroyWindow(crop_win)

    print(f"[score_wipe --roi-tune] chosen ROI: --roi {x} {y} {w} {h}")
    return x, y, w, h


def open_frame_source(live: bool, device: int, before_path):
    """Returns (frame_source, cleanup) for the ROI debug windows: a continuous
    live-camera reader when `live`, otherwise a closure replaying the same cached
    `before_path` photo every call (so --roi-tune/--roi-preview work identically
    against a still image). Caller must call cleanup() when done -- it releases
    the camera for the live case and is a no-op otherwise."""
    if live:
        cap = open_scene_camera(device)

        def frame_source():
            ok, frame = cap.read()
            if not ok:
                return None
            # Femto Bolt is mounted upside down -- same fix deploy_smolvla.py's
            # get_scene_frame applies, so --tune/--roi-tune/--roi-preview --live
            # see the same orientation a real run's scorer would.
            return cv2.rotate(frame, cv2.ROTATE_180)

        return frame_source, cap.release

    frame_bgr = cv2.imread(before_path)
    if frame_bgr is None:
        raise FileNotFoundError(f"Could not read image: {before_path}")
    return (lambda: frame_bgr), (lambda: None)


def run_roi_preview(frame_source, roi, out_dir: str, ranges=None, min_blob_px: int = 50):
    """Streams frame_source() through an already-chosen, fixed `roi` (no crop
    sliders -- use --roi-tune for that) -- for a quick visual sanity check that a
    --roi found earlier still frames the wipe board correctly, e.g. right before a
    real run or after nudging the camera/board. Shows the full frame with the ROI
    rectangle overlay alongside the cropped-only view.

    If `ranges` is given (i.e. --color was also passed), the cropped window adds
    live H/S/V-lo/hi trackbars (seeded from `ranges`' first range -- same
    single-range limitation --tune has, so red's hue-wraparound second range isn't
    reachable via the sliders) and overlays the detected mark using whatever the
    sliders are currently set to -- the same segment_mark() pipeline (threshold +
    open/close + largest-blob) the real scorer runs, so what you see here is what
    a real run would detect. A third window shows the same result as --tune's
    filtered view: only the pixels currently passing the threshold, everything
    else blacked out, for directly comparing the raw detected pixels against the
    tinted overlay. Pressing 'b' snapshots the current mark area as a baseline;
    every frame after that overlays a live
    percent_wiped = 100 * (1 - current_area / baseline_area), so you can dial in
    thresholds and watch the score update in real time while physically wiping the
    board, all in one window -- no separate --tune pass or before/after capture
    step needed. On close, prints the final HSV bounds (--hsv-lower/--hsv-upper)
    to carry into a real run. Without `ranges`, this is just the crop preview (no
    filtered window). Pressing 's' anytime saves the current cropped image
    (undecorated -- no overlay/text) to `out_dir`. Controls: 's' save crop,
    'b' set/reset baseline, 'q' close."""
    frame_bgr = frame_source()
    if frame_bgr is None:
        raise RuntimeError("frame_source produced no frame")
    frame_h, frame_w = frame_bgr.shape[:2]
    x, y, w, h = clamp_roi(*(int(v) for v in roi), frame_w, frame_h)

    win = "roi preview -- full frame (q to close)"
    crop_win = "roi preview -- cropped ROI + detected mask"
    filtered_win = "roi preview -- filtered (detected pixels only)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.namedWindow(crop_win, cv2.WINDOW_NORMAL)

    tunable = ranges is not None
    if tunable:
        lower0, upper0 = ranges[0]
        cv2.namedWindow(filtered_win, cv2.WINDOW_NORMAL)
        # A window auto-sized tightly around a small ROI crop can leave too
        # little room for 6 trackbars to render/respond to drags on some
        # backends (GTK in particular) -- force a sane minimum size up front.
        cv2.resizeWindow(crop_win, max(w, 400), h + 150)
        cv2.resizeWindow(filtered_win, max(w, 400), h)
        for name, val, maxval in [
            ("H lo", lower0[0], 179), ("H hi", upper0[0], 179),
            ("S lo", lower0[1], 255), ("S hi", upper0[1], 255),
            ("V lo", lower0[2], 255), ("V hi", upper0[2], 255),
        ]:
            cv2.createTrackbar(name, crop_win, val, maxval, lambda _v: None)

    control_hint = "[S] save crop   [Q] close" if not tunable else \
        "[S] save crop   [B] set/reset baseline   [Q] close"
    print(f"[score_wipe --roi-preview] previewing ROI x={x} y={y} w={w} h={h} -- "
          + ("press 's' to save the current crop, 'q' to close." if not tunable else
             "drag H/S/V sliders in the cropped window until the mask cleanly "
             "covers the mark, press 'b' once it does to set the baseline, then "
             "wipe and watch the score update live; 's' saves the current crop, "
             "'q' closes. (If nothing visibly changes, watch the console -- it "
             "logs every slider read-back change to confirm the drag is being "
             "registered.)"))

    baseline_area = None
    active_lo = active_hi = None
    last_bounds = None
    try:
        while True:
            frame_bgr = frame_source()
            if frame_bgr is None:
                raise RuntimeError("frame_source produced no frame")
            crop_bgr = frame_bgr[y:y + h, x:x + w]

            area = None
            mask = None
            if tunable:
                active_lo = tuple(cv2.getTrackbarPos(n, crop_win) for n in ("H lo", "S lo", "V lo"))
                active_hi = tuple(cv2.getTrackbarPos(n, crop_win) for n in ("H hi", "S hi", "V hi"))
                bounds = (active_lo, active_hi)
                if bounds != last_bounds:
                    print(f"[score_wipe --roi-preview] slider read-back: "
                          f"lo={active_lo} hi={active_hi}")
                    last_bounds = bounds
                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                mask, area = segment_mark(crop_rgb, [(active_lo, active_hi)], roi=None,
                                           min_blob_px=min_blob_px)
                crop_disp = cv2.cvtColor(overlay_mask(crop_rgb, mask), cv2.COLOR_RGB2BGR)
                draw_hsv_readout(crop_disp, active_lo, active_hi)
                # Same "filtered" view --tune shows: only pixels currently
                # passing the threshold, everything else blacked out -- lets you
                # compare the raw detected pixels against the tinted overlay.
                filtered_disp = cv2.bitwise_and(crop_bgr, crop_bgr, mask=mask.astype(np.uint8) * 255)
                draw_hsv_readout(filtered_disp, active_lo, active_hi)
                cv2.imshow(filtered_win, filtered_disp)
            else:
                crop_disp = crop_bgr

            disp = frame_bgr.copy()
            if mask is not None:
                # Mirror the detected-pixel mask onto the full-frame window too
                # (not just the crop window) so the sliders' effect is visible
                # no matter which window is being watched.
                roi_region = disp[y:y + h, x:x + w]
                roi_region[mask] = (0.4 * roi_region[mask] + 0.6 * np.array([255, 0, 255])).astype(np.uint8)
            cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 255, 0), 2)
            lines = [f"ROI: x={x} y={y} w={w} h={h}"]
            if tunable:
                if baseline_area is None:
                    lines.append(f"mark area: {area}px  (no baseline yet)")
                elif baseline_area == 0:
                    lines.append("baseline had no detected mark -- press B over a clean mark")
                else:
                    pct = max(0.0, min(100.0, 100.0 * (1.0 - area / baseline_area)))
                    lines.append(f"wiped: {pct:.1f}%  (baseline {baseline_area}px, now {area}px)")
            lines.append(control_hint)
            for i, line in enumerate(lines):
                cv2.putText(disp, line, (10, 30 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (0, 255, 0), 2)

            cv2.imshow(win, disp)
            cv2.imshow(crop_win, crop_disp)

            key = cv2.waitKey(30) & 0xFF
            if key == ord("s"):
                path = save_crop_image(out_dir, "roi_preview_crop", crop_bgr)
                print(f"[score_wipe --roi-preview] saved crop to {path}")
            elif key == ord("b") and tunable:
                baseline_area = area
                print(f"[score_wipe --roi-preview] baseline set: {area}px")
            elif key == ord("q"):
                break
    finally:
        cv2.destroyWindow(win)
        cv2.destroyWindow(crop_win)
        if tunable:
            cv2.destroyWindow(filtered_win)

    if tunable and active_lo is not None:
        print(f"[score_wipe --roi-preview] final HSV bounds: "
              f"--hsv-lower {active_lo[0]} {active_lo[1]} {active_lo[2]} "
              f"--hsv-upper {active_hi[0]} {active_hi[1]} {active_hi[2]}")


def score_and_visualize(before_rgb, after_rgb, ranges, roi, out_dir, show, min_blob_px):
    before_mask, before_area = segment_mark(before_rgb, ranges, roi, min_blob_px)
    after_mask, after_area = segment_mark(after_rgb, ranges, roi, min_blob_px)

    if before_area == 0:
        print("[score_wipe] WARNING: no mark detected in the 'before' image -- "
              "check --color / ROI / lighting, or use --tune. Score is undefined.")
        percent_wiped = float("nan")
    else:
        raw = 100.0 * (1.0 - after_area / before_area)
        percent_wiped = max(0.0, min(100.0, raw))
        if raw < 0:
            print(f"[score_wipe] WARNING: detected mark area increased after wiping "
                  f"({before_area}px -> {after_area}px) -- reporting 0% rather than "
                  f"negative; check segmentation.")

    before_vis = overlay_mask(before_rgb, before_mask)
    after_vis = overlay_mask(after_rgb, after_mask)

    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].imshow(before_rgb); axes[0, 0].set_title("before")
    axes[0, 1].imshow(before_vis); axes[0, 1].set_title(f"before mask ({before_area}px)")
    axes[1, 0].imshow(after_rgb); axes[1, 0].set_title("after")
    axes[1, 1].imshow(after_vis); axes[1, 1].set_title(f"after mask ({after_area}px)")
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(f"percent wiped: {percent_wiped:.1f}%", fontsize=16)
    fig.tight_layout()

    out_path = os.path.join(out_dir, f"wipe_score_{stamp}.png")
    fig.savefig(out_path, dpi=150)
    print(f"[score_wipe] saved visualization to {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return percent_wiped, before_area, after_area, out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--before", help="path to the 'before' photo (required unless --live/--tune)")
    p.add_argument("--after", help="path to the 'after' photo (required unless --live)")
    p.add_argument("--color", choices=sorted(DEFAULT_HSV_RANGES), default=None,
                    help="mark colour to segment (required unless --roi-tune; optional with "
                         "--roi-preview -- if given, adds live HSV sliders + a detected-pixel "
                         "mask overlay and a wipe-percentage score baselined by pressing 'b')")
    p.add_argument("--live", action="store_true",
                    help="capture before/after from the scene camera through a live-preview GUI "
                         "window (SPACE to capture each shot) instead of reading files")
    p.add_argument("--tune", action="store_true",
                    help="open an interactive HSV threshold tuner on --before (or the live feed, "
                         "continuously -- not a one-shot grab), showing the original and a live "
                         "filtered view side by side, and print the resulting bounds, instead of "
                         "scoring. Sliders seed from "
                         "--hsv-lower/--hsv-upper if given, else --color's defaults")
    p.add_argument("--roi-tune", action="store_true",
                    help="debug mode: open an interactive X/Y/W/H crop-slider window on --before "
                         "(or the live feed) and print the resulting --roi, instead of scoring")
    p.add_argument("--roi-preview", action="store_true",
                    help="debug mode: preview an already-chosen --roi against --before (or the "
                         "live feed), instead of scoring -- for confirming the crop still frames "
                         "the board, e.g. right before a real run. Pass --color too to add live "
                         "HSV tuning sliders, a detected-pixel mask overlay, a filtered-only "
                         "preview (same as --tune), and a live wipe-percentage score, all "
                         "against the same crop")
    p.add_argument("--scene-cv2-device", type=int, default=6,
                    help="cv2.VideoCapture device index, same convention as deploy_smolvla.py")
    p.add_argument("--roi", type=float, nargs=4, metavar=("X", "Y", "W", "H"), default=None,
                    help="pixel ROI (x y w h) to restrict segmentation to, e.g. to exclude the "
                         "robot arm/background from the frame -- find these interactively with "
                         "--roi-tune")
    p.add_argument("--hsv-lower", type=int, nargs=3, metavar=("H", "S", "V"), default=None)
    p.add_argument("--hsv-upper", type=int, nargs=3, metavar=("H", "S", "V"), default=None)
    p.add_argument("--min-blob-px", type=int, default=50,
                    help="ignore matching blobs smaller than this many pixels (noise floor)")
    p.add_argument("--out-dir", default=os.path.join(os.getcwd(), "wipe_scores"),
                    help="where to save the before/after visualization PNG, and where 's' saves "
                         "a cropped image from --tune/--roi-tune/--roi-preview")
    p.add_argument("--no-show", action="store_true",
                    help="save the visualization but don't open a window (e.g. headless/SSH)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    roi = parse_roi(args.roi)

    roi_debug_mode = args.roi_tune or args.roi_preview
    if not roi_debug_mode and args.color is None:
        print("--color is required (unless --roi-tune/--roi-preview)", file=sys.stderr)
        return 1

    if args.tune:
        if not args.live and not args.before:
            print("--tune needs --before <path> or --live", file=sys.stderr)
            return 1
        try:
            frame_source, cleanup = open_frame_source(args.live, args.scene_cv2_device, args.before)
        except (RuntimeError, FileNotFoundError) as e:
            print(str(e), file=sys.stderr)
            return 1
        try:
            seed_lower, seed_upper = hsv_ranges_for(args.color, args.hsv_lower, args.hsv_upper)[0]
            run_tune(frame_source, roi, seed_lower, seed_upper, args.out_dir)
        finally:
            cleanup()
        return 0

    if roi_debug_mode:
        flag = "--roi-tune" if args.roi_tune else "--roi-preview"
        if not args.live and not args.before:
            print(f"{flag} needs --before <path> or --live", file=sys.stderr)
            return 1
        if args.roi_preview and roi is None:
            print("--roi-preview needs --roi X Y W H", file=sys.stderr)
            return 1
        try:
            frame_source, cleanup = open_frame_source(args.live, args.scene_cv2_device, args.before)
        except (RuntimeError, FileNotFoundError) as e:
            print(str(e), file=sys.stderr)
            return 1
        try:
            if args.roi_tune:
                run_roi_tune(frame_source, args.out_dir)
            else:
                ranges = (hsv_ranges_for(args.color, args.hsv_lower, args.hsv_upper)
                          if args.color is not None else None)
                run_roi_preview(frame_source, roi, args.out_dir, ranges, args.min_blob_px)
        finally:
            cleanup()
        return 0

    if args.live:
        try:
            before_rgb, after_rgb = run_capture_gui(args.scene_cv2_device)
        except CaptureCancelled as e:
            print(f"[score_wipe] {e}", file=sys.stderr)
            return 1
    else:
        if not args.before or not args.after:
            print("Either --live, or both --before and --after are required.", file=sys.stderr)
            return 1
        before_rgb = load_image_rgb(args.before)
        after_rgb = load_image_rgb(args.after)

    ranges = hsv_ranges_for(args.color, args.hsv_lower, args.hsv_upper)
    percent_wiped, before_area, after_area, out_path = score_and_visualize(
        before_rgb, after_rgb, ranges, roi, args.out_dir, not args.no_show, args.min_blob_px
    )

    print(f"[score_wipe] before mark area: {before_area}px, after mark area: {after_area}px")
    print(f"[score_wipe] percent wiped: {percent_wiped:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
