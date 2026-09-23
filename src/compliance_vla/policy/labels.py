"""Day 11: per-episode training labels for the compliance policy.

Reuses the already-tested §4.1 extraction pipeline (scripts/
run_extraction_on_dataset.py -> hardware/fr3_bilateral_teleop/dataset_tools/
labeling/extract_impedance_labels.py) for log_k/mask rather than re-deriving the
windowed regression, and scripts/panda_fk.py + the calibrated tool offset for
the follower's Cartesian pose x_f, exactly like that script already does for
Gate 2's M8 benchmark.

Two things worth being explicit about, since they're easy to get wrong by
analogy with the extraction script:
  - x_eq is NOT extract_demo_30hz's contact-frame pose error `e`. It is
    `observation.leader_pose` directly: proposal §2.2(b)/§4.1's
    identifiability argument defines x_eq := x_l (the leader's own pose),
    and `observation.leader_pose` is already stored as [x, y, z, rx, ry, rz]
    -- position + rotation vector, base/world frame (confirmed by
    run_extraction_on_dataset.build_demo_dict, which feeds
    leader_pose[:, 3:6] straight into rotvec_batch_to_quat). No FK, no
    contact-frame rotation needed -- it is read off the dataset as-is.
  - log_k IS in the auto-fit contact frame (extract_demo_30hz's `k`), not
    base frame -- matching how M8's offline benchmark already consumes the
    same field. Rotating predicted K back to base frame for the low-level
    controller is a Week-4 controller-integration question, out of scope
    here.
"""

import json
import os

import numpy as np

from ._paths import EXTERNAL_SCRIPTS as _EXTERNAL_SCRIPTS  # noqa: E402
from ._paths import SCRIPTS_DIR as _SCRIPTS_DIR  # noqa: E402
from ._paths import ensure_on_sys_path  # noqa: E402

# The modules imported just below are standalone scripts, not installed
# packages, so they have to be reachable on sys.path before the imports run.
ensure_on_sys_path()
SCRIPTS_DIR = str(_SCRIPTS_DIR)
EXTERNAL_SCRIPTS = str(_EXTERNAL_SCRIPTS)

import dataset_io as dio  # noqa: E402
import panda_fk as fk  # noqa: E402
import run_extraction_on_dataset as red  # noqa: E402
from extract_impedance_labels import (  # noqa: E402
    DEFAULT_SIGMA_F,
    nearest_sample_indices,
    parse_args as extraction_parse_args,
)

TOOL_OFFSET_PATH = os.path.join(SCRIPTS_DIR, "tool_offset.npy")


def load_tool_offset():
    if not os.path.exists(TOOL_OFFSET_PATH):
        raise FileNotFoundError(f"{TOOL_OFFSET_PATH} missing -- run scripts/calibrate_tool_offset.py first")
    return np.load(TOOL_OFFSET_PATH)


def default_extraction_args():
    return extraction_parse_args([])


def default_sigma_f():
    return np.array(DEFAULT_SIGMA_F)


MANNER_CALIBRATION_PATH = os.path.join(SCRIPTS_DIR, "manner_force_calibration.json")


def load_manner_calibration(path=None):
    """Loads the per-operator log_k calibration fit by
    scripts/fit_manner_force_calibration.py -- see that script's docstring for what problem
    this corrects (operator-dependent absolute force/stiffness for the same manner word;
    language_grounding_issue_handoff.md's cross-operator-adverb finding, sharpened by the
    2026-09-02 data_two_color batch). Returns None (meaning "apply no calibration") if the
    file doesn't exist yet, rather than raising -- callers should treat that as the
    pre-calibration default, not an error, so existing runs/tests that never fit this file
    keep working unchanged."""
    path = path or MANNER_CALIBRATION_PATH
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def calibrate_log_k(log_k, mask, operator_id, manner, calibration):
    """Per-axis affine recalibration of one episode's log_k array onto the "equal split"
    cross-operator reference fit by scripts/fit_manner_force_calibration.py: reference_mean/
    reference_std for a given manner are the unweighted (equal-per-operator, not
    sample-count-weighted) average of each operator's own mean/std for that manner -- so
    operator B's smaller episode count doesn't get drowned out by operator A's larger one.

    corrected = (raw - operator_mean) / operator_std * reference_std + reference_mean

    Applied per axis, only where that (operator, manner, axis) cell was actually fit (see
    MIN_CALIBRATION_COUNT in the fitting script) -- axes/combos without a reliable estimate
    are left unchanged (identity), not guessed at. `mask`-false entries are left unchanged
    too (harmless either way, since the loss never reads them, but there's no reason to
    touch what's never supervised). No-op (returns log_k unchanged) if `calibration` is None,
    or if `operator_id`/`manner` aren't present in it.
    """
    if calibration is None or operator_id is None or manner is None:
        return log_k
    op_stats = calibration.get("operators", {}).get(operator_id, {}).get(manner)
    ref_stats = calibration.get("reference", {}).get(manner)
    if op_stats is None or ref_stats is None:
        return log_k

    corrected = log_k.copy()
    fitted = np.array(op_stats["fitted"], dtype=bool)  # (6,) -- which axes had enough data
    op_mean = np.array(op_stats["mean"], dtype=np.float64)
    op_std = np.array(op_stats["std"], dtype=np.float64)
    ref_mean = np.array(ref_stats["mean"], dtype=np.float64)
    ref_std = np.array(ref_stats["std"], dtype=np.float64)

    for axis in range(log_k.shape[-1]):
        if not fitted[axis] or op_std[axis] <= 0:
            continue
        col = mask[:, axis]
        corrected[col, axis] = (
            (log_k[col, axis] - op_mean[axis]) / op_std[axis] * ref_std[axis] + ref_mean[axis]
        )
    return corrected


def compute_episode_arrays(ep_frames, tool_offset, args, sigma_f, operator_id=None, manner=None, calibration=None):
    """ep_frames: one episode's rows from dataset_io.load_frames(session),
    already filtered+sorted by frame_index. Returns None if the episode
    fails contact-frame fitting (same rejection as Gate 1/the offline
    benchmark -- e.g. 1/92 episodes dataset-wide, see reports/extraction_per_episode.json),
    else a dict of raw-frame arrays (length T, native ~30Hz) and
    extraction-output arrays (length N <= T, offset by args.window_sec).

    operator_id/manner/calibration: optional, and only meaningful together -- when all three
    are given, the returned "log_k" has calibrate_log_k applied (see that function's
    docstring). Default None/None/None reproduces the pre-calibration behaviour exactly, so
    existing callers (e.g. scripts/fit_manner_force_calibration.py's own first pass, which
    needs the *raw* log_k to fit the calibration in the first place) are unaffected."""
    state = dio.stack_col(ep_frames, "observation.state")
    velocity = dio.stack_col(ep_frames, "observation.velocity")
    leader_pose = dio.stack_col(ep_frames, "observation.leader_pose")
    wrench = dio.stack_col(ep_frames, "observation.wrench.external_base")
    timestamp = ep_frames["timestamp"].to_numpy()
    frame_index = ep_frames["frame_index"].to_numpy()

    demo = red.build_demo_dict(state, leader_pose, wrench, timestamp, tool_offset)
    result = red.extract_demo_30hz(demo, args, sigma_f)
    if result["status"] != "ok":
        return None

    follower_pos, follower_rot = fk.fk_batch(state)
    follower_pos = follower_pos + np.einsum("nij,j->ni", follower_rot, tool_offset)
    follower_rotvec = np.stack([fk.rotvec_from_matrix(follower_rot[i]) for i in range(len(follower_rot))], axis=0)
    x_f = np.concatenate([follower_pos, follower_rotvec], axis=1)  # (T, 6), base frame

    output_times = result["output_times"]
    nearest_idx = nearest_sample_indices(demo["t"], output_times)  # (N,) indices into the T-length raw arrays

    log_k = np.log(result["k"])  # (N, 6), contact frame
    mask = result["mask"]  # (N, 6) bool, identifiability mask (never imputed)
    if calibration is not None:
        log_k = calibrate_log_k(log_k, mask, operator_id, manner, calibration)

    return {
        "t": demo["t"],  # (T,) seconds from episode start
        "frame_index": frame_index,  # (T,) raw LeRobot frame_index, for image row lookup
        "q": state,  # (T, 7)
        "qdot": velocity,  # (T, 7)
        "x_f": x_f,  # (T, 6)
        "x_eq": leader_pose,  # (T, 6) -- IS x_l already, base frame, see module docstring
        "wrench": wrench,  # (T, 6)
        "nearest_idx": nearest_idx,  # (N,) -> index into the T-length arrays above
        "log_k": log_k,
        "mask": mask,
    }
