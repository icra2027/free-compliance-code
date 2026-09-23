#!/usr/bin/env python3
"""Day 14: fit ONE frozen T1 contact frame (board-normal rotation, base frame) for the
pilot-rollout harness (scripts/run_pilot_rollout.py, external/fr3_bilateral_teleop) to rotate
a policy's predicted log_k (fit in the auto-fit contact frame, per src/compliance_vla/policy/labels.py's
docstring) into the base-frame diagonal `target_stiffness` variable_impedance_controllers' variable-
impedance controller actually consumes.

Why this is needed and why it's scoped this way: src/compliance_vla/policy/labels.py's own docstring already
flags "Rotating predicted K back to base frame for the low-level controller is a Week-4
controller-integration question, out of scope here" -- during TRAINING that's fine, log_k
never leaves contact-frame numbers. At ROLLOUT time it can't stay out of scope, since the
controller needs a real 6-vector to command. Two ways to get a contact frame at rollout
time: (a) re-fit it live, per-rollout, from the rollout's own in-contact trajectory (what
extract_demo_30hz does per-DEMO, offline, after the fact); or (b) fit it ONCE, offline, from
already-collected real T1 data, and treat it as a fixed per-session calibration constant --
justified because the board is physically rigid and was frozen on Day 6 ("Freeze both task
setups... any change after today invalidates earlier sessions"), so its normal does not
change between episodes or sessions. (b) is what this script does: it is strictly safer for
a Day-14 PILOT rollout (sanity-only, n=5, "must not be reported") than adding a live re-fit
to the control loop, and it reuses fit_contact_frame() -- the exact same already-tested
building block extract_demo_30hz calls per-demo, see run_extraction_on_dataset.py -- rather
than any new numerical logic. A live per-rollout re-fit is a reasonable Week-4 upgrade if
the frozen fit's residuals turn out not to generalize across sessions; not attempted here,
same "out of scope, not silently smoothed over" spirit as labels.py's own note.

Concretely: pools in-contact follower positions + wrench-force samples across every
successfully-extracted episode in the given session(s) (default: the frozen train split,
demo4 -- Day 10/12's TRAIN_SESSIONS), then calls fit_contact_frame() ONCE on the pooled set.
Pooling across many episodes (not just one) gives the SVD plane fit far more spatial
coverage of the board than any single ~5-7s wipe does, which should only tighten the
planarity_ratio, not degrade it, since the true board normal is identical across the
episodes.

Usage:
    python scripts/fit_frozen_contact_frame.py                       # pools demo4 (default)
    python scripts/fit_frozen_contact_frame.py --sessions demo1 demo4
    python scripts/fit_frozen_contact_frame.py --self-test           # synthetic plane, no dataset needed
"""
import argparse
import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(PROJECT_ROOT)
EXTERNAL_SCRIPTS = os.path.join(
    PROJECT_ROOT, "hardware", "fr3_bilateral_teleop", "dataset_tools", "labeling")  # label extraction
for _p in (EXTERNAL_SCRIPTS, SCRIPT_DIR, PROJECT_ROOT):  # PROJECT_ROOT needed for `import compliance_vla.policy.labels`
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dataset_io as dio  # noqa: E402
import panda_fk as fk  # noqa: E402
from run_extraction_on_dataset import MIN_CONTACT_SAMPLES_30HZ  # noqa: E402
from extract_impedance_labels import fit_contact_frame  # noqa: E402

DEFAULT_SESSIONS = ["demo4"]  # matches scripts/train_policy.py's TRAIN_SESSIONS -- the frozen train split
DEFAULT_FRAME_FIT_FORCE_THRESHOLD = 8.0  # N, matches extract_impedance_labels' own default (firm contact only)
OUT_PATH = os.path.join(SCRIPT_DIR, "contact_frame_t1.npy")
REPORT_PATH = os.path.join(PROJECT_ROOT, "reports", "contact_frame_t1.json")


def pooled_contact_points(sessions, tool_offset):
    """Pools (follower_pos, wrench_force) across every frame of every episode in `sessions`
    -- fit_contact_frame's own contact_force_threshold does the in-contact selection, so
    pooling raw (not pre-filtered) frames here is correct and simplest."""
    pos_chunks, force_chunks = [], []
    episodes_used = []
    for session in sessions:
        frames = dio.load_frames(session, columns=dio.NON_IMAGE_COLUMNS)
        frames = frames.sort_values(["episode_index", "frame_index"])
        for ep_idx, g in frames.groupby("episode_index"):
            g = g.sort_values("frame_index")
            state = dio.stack_col(g, "observation.state")
            wrench = dio.stack_col(g, "observation.wrench.external_base")
            follower_pos, follower_rot = fk.fk_batch(state)
            follower_pos = follower_pos + np.einsum("nij,j->ni", follower_rot, tool_offset)
            pos_chunks.append(follower_pos)
            force_chunks.append(wrench[:, 0:3])
            episodes_used.append({"session": session, "episode_index": int(ep_idx), "n_frames": len(g)})
    return np.concatenate(pos_chunks, axis=0), np.concatenate(force_chunks, axis=0), episodes_used


def run_self_test() -> int:
    """Synthetic plane, tilted ~20deg from vertical (matching proposal §5's real board
    geometry) about the base X axis, contact points scattered on it with a bit of normal
    noise -- confirms fit_contact_frame recovers the known normal and this script's own
    pooling/saving logic round-trips, with no dataset on disk required."""
    rng = np.random.default_rng(0)
    tilt = np.deg2rad(20.0)
    true_normal = np.array([0.0, -np.sin(tilt), np.cos(tilt)])  # tilted off +z about x
    true_normal /= np.linalg.norm(true_normal)

    n = 400
    seed = np.array([1.0, 0.0, 0.0])
    x_axis = seed - np.dot(seed, true_normal) * true_normal
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(true_normal, x_axis)
    u, v = rng.uniform(-0.1, 0.1, n), rng.uniform(-0.1, 0.1, n)
    pts = 0.5 * true_normal + u[:, None] * x_axis + v[:, None] * y_axis
    pts += rng.normal(0.0, 0.0005, size=pts.shape)  # small out-of-plane noise
    force = np.tile(true_normal, (n, 1)) * 10.0 + rng.normal(0.0, 0.2, size=(n, 3))  # pushing along +normal

    result = fit_contact_frame(pts, force, contact_force_threshold=1.0, min_contact_samples=50, max_planarity_ratio=0.05)
    assert result["ok"], f"self-test fit failed: {result.get('reason')}"
    recovered_normal = result["R"][:, 2]
    cos_sim = float(np.dot(recovered_normal, true_normal))
    assert abs(cos_sim) > 0.999, f"recovered normal off from ground truth, cos_sim={cos_sim}"
    print(f"[fit_frozen_contact_frame] self-test PASS -- planarity_ratio={result['planarity_ratio']:.5f}, "
          f"normal cos_sim={cos_sim:.6f}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sessions", nargs="+", default=DEFAULT_SESSIONS)
    p.add_argument("--frame-fit-force-threshold", type=float, default=DEFAULT_FRAME_FIT_FORCE_THRESHOLD)
    p.add_argument("--min-contact-samples", type=int, default=MIN_CONTACT_SAMPLES_30HZ)
    p.add_argument("--max-planarity-ratio", type=float, default=0.15)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()

    if args.self_test:
        return run_self_test()

    from compliance_vla.policy.labels import load_tool_offset  # local import: needs PROJECT_ROOT on sys.path, done above
    tool_offset = load_tool_offset()

    follower_pos, wrench_force, episodes_used = pooled_contact_points(args.sessions, tool_offset)
    print(f"[fit_frozen_contact_frame] pooled {len(follower_pos)} frames across "
          f"{len(episodes_used)} episodes from sessions={args.sessions}")

    result = fit_contact_frame(
        follower_pos, wrench_force, args.frame_fit_force_threshold,
        min_contact_samples=args.min_contact_samples, max_planarity_ratio=args.max_planarity_ratio,
    )
    if not result["ok"]:
        print(f"[fit_frozen_contact_frame] FAILED: {result['reason']}", file=sys.stderr)
        return 1

    R = np.asarray(result["R"])
    np.save(OUT_PATH, R)
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump({
            "sessions": args.sessions, "n_episodes": len(episodes_used), "n_frames_pooled": len(follower_pos),
            "n_contact_samples": result["n_contact_samples"], "planarity_ratio": result["planarity_ratio"],
            "frame_fit_force_threshold": args.frame_fit_force_threshold,
            "normal_base_frame": result["normal_base_frame"], "R_contact": R.tolist(),
            "episodes_used": episodes_used,
        }, f, indent=2)

    print(f"[fit_frozen_contact_frame] ok -- planarity_ratio={result['planarity_ratio']:.4f} "
          f"({result['n_contact_samples']} in-contact samples), normal (base frame)={result['normal_base_frame']}")
    print(f"[fit_frozen_contact_frame] wrote {OUT_PATH}, {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
