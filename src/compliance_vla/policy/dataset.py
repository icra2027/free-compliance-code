"""Day 11: torch Dataset bridging the on-disk franka_vla_multimodal LeRobot
sessions (dataset/demo{1,3,4}, see scripts/dataset_io.py) + the §4.1
extraction labels (src/compliance_vla/policy/labels.py) into batches shaped exactly the way
compliance_vla.policy.compliance_policy.ComplianceSmolVLAPolicy.forward expects.

One training example = one 30Hz anchor timestep inside one episode:
  - observation: current-frame wrist+scene RGB, (q, q_dot, x_f) state, task
    text, trailing 500ms of wrench (-> force_history).
  - target: the next H=32 steps of [x_eq(6), log_k(6), gripper(1)=0], plus
    a per-step/per-axis validity mask (in-episode-bound AND, for log_k, the
    extraction pipeline's identifiability mask -- never imputed).

Deliberately eager per-session loading (image columns included), not a
lazy/streaming reader: this dataset is 3 sessions / ~92 episodes / ~3.8GB
including images (Day 10 note), and Day 11's job is a training *smoke test*
(scripts/train_b5.py --smoke-test), not a throughput-tuned full-scale
dataloader -- that belongs with Day 12's real 3-seeds-per-policy runs, if the
smoke test shows this approach needs it.
"""


import numpy as np
import torch
from torch.utils.data import Dataset

from ._paths import SCRIPTS_DIR as _SCRIPTS_DIR  # noqa: E402
from ._paths import ensure_on_sys_path  # noqa: E402

# dataset_io and diagnose_language_grounding are standalone scripts, not
# installed packages, so sys.path has to be prepared before importing them.
ensure_on_sys_path()
SCRIPTS_DIR = str(_SCRIPTS_DIR)

import dataset_io as dio  # noqa: E402
from diagnose_language_grounding import _load_manifest_ordered  # noqa: E402

from . import labels as lb  # noqa: E402
from .force_encoder import resample_to_n_samples  # noqa: E402

IMAGE_COLUMNS = ["observation.images.scene_rgb", "observation.images.wrist_rgb"]
GRIPPER_PLACEHOLDER = 0.0  # no gripper channel recorded for T1 (rigid wiper mount); never supervised

# 2026-09-03: the data_recorder version that collected data_two_color (commit
# 2f63ca3) appends a still-recording return-to-start_joint_configuration move to every saved
# episode before the 's' key actually stops/saves it (see that repo's README, "s: save
# successful episode"). src/compliance_vla/policy/labels.py's log_k target IS already protected from this --
# extract_demo_30hz's identifiability mask requires real contact force, so a no-contact home
# move naturally comes back masked-out (see extract_impedance_labels.py's mask condition).
# x_eq (leader_pose) has no such protection: it's supervised unconditionally within
# action_valid_mask, which only tracks in-episode-bound, not in-contact. Left untrimmed, any
# training window whose target chunk overlaps the appended tail would supervise the position
# head with "move to the fixed home pose" regardless of which colour the episode was labelled
# with -- see reports/two_color_diagnostic_operatorA_step30000.json's ground_truth_correction
# section, which found and fixed the matching bug in the *evaluation* scripts; this is the
# training-side counterpart.
SPEED_PAUSE_THRESHOLD_M_S = 0.01  # m/s, below this = "stationary" for _trim_to_last_contact
NEAR_HOME_RADIUS_M = 0.05  # a stationary point within this of the episode's own start frame
# counts as "arrived back home", not a genuine mid-task pause -- see docstring below.


def _trim_to_last_contact(g, speed_threshold=SPEED_PAUSE_THRESHOLD_M_S,
                           near_home_radius=NEAR_HOME_RADIUS_M, settle_frames=2):
    """Drops any trailing frames after the last genuine "wipe done, arm paused" moment, so
    compute_episode_arrays never sees an appended home-return tail in the first place.

    NOT force-based, despite the name (kept for call-site continuity) -- a first attempt used
    a contact-force threshold and was WRONG: inspecting a real data_two_color episode's raw
    trace showed the appended return-to-start_joint_configuration move produces external-
    wrench estimation transients from the fast commanded motion itself that routinely exceed a
    naive 2N contact threshold throughout the ~100-frame return trip (2-3.6N observed, vs
    4-10N during real wipe contact) -- "last frame above threshold" landed almost at the raw
    episode end, barely trimming anything.

    A second attempt used velocity alone (walk backward, stop at the first near-zero-speed
    frame) and was ALSO wrong, for a different reason: the return-to-home glide itself ends by
    decelerating and settling AT home, which is also near-zero-speed -- walking backward from
    episode end hits that arrival/settling pause FIRST and stops immediately, before ever
    reaching the real post-wipe pause earlier in the trajectory. Empirically this undertrimmed
    data_two_color (most episodes showed 0 frames dropped) and, since a real demo's own
    natural pauses (repositioning grip, a beat between strokes) look identical to a genuine
    pause by speed alone, badly OVERtrimmed dataset/demo{1,3,4} (some episodes cut to a
    handful of frames).

    Fix: episodes reliably start idle at (approximately) the same pose the appended reset
    returns to -- both are "the arm at start_joint_configuration". So a stationary point is
    only treated as "arrived home" (and skipped) if it's within `near_home_radius` of the
    episode's OWN first frame; the pause we actually want is the LAST stationary point that is
    both slow AND far from that start position -- i.e. the arm paused somewhere out over the
    board, not back at rest. This distinguishes "wipe done, holding position away from home"
    from both "still settling into the final home arrival" and ordinary mid-task pauses in
    episodes that never returned toward start at all (old-dataset episodes typically end far
    from their own start too, since they end at the mark, not back at rest -- so this should
    rarely find anything to trim there).

    No-op (returns g unchanged) if no such pause exists -- e.g. the episode never paused near
    its end at all, or every stationary point found is close to the start pose."""
    pos = np.stack(g["observation.leader_pose"].to_numpy())[:, :3]
    t = g["timestamp"].to_numpy().astype(np.float64)
    n = len(pos)
    if n < 10:
        return g
    start_pos = pos[0]
    dist_from_start = np.linalg.norm(pos - start_pos, axis=1)  # (n,)
    dt = np.diff(t)
    with np.errstate(invalid="ignore", divide="ignore"):
        speed = np.linalg.norm(np.diff(pos, axis=0), axis=1) / dt  # (n-1,) m/s, index k = step k->k+1
    k = n - 2
    while k >= 0 and not (speed[k] < speed_threshold and dist_from_start[k] > near_home_radius):
        k -= 1  # `not (a and b)` treats a NaN speed step (zero-dt) as "moving", i.e. skip it
    if k < 0:
        return g
    cutoff = min(k + 1 + settle_frames, n)
    return g.iloc[:cutoff]


def _decode_array3d(cell):
    """The dataset's Array3DExtensionType image cells come back from
    pandas/pyarrow as a doubly-nested object ndarray (rows of rows of a real
    (3,) uint8 pixel array) -- plain np.array(cell, dtype=uint8) fails
    ("setting an array element with a sequence") because numpy won't
    recurse through two levels of object dtype in one shot. Recursing one
    level explicitly (row-by-row) resolves it; ~15ms/image, fine for a
    training-time __getitem__."""
    return np.stack([np.stack(row) for row in cell]).astype(np.uint8)


class ComplianceWindowDataset(Dataset):
    def __init__(self, sessions, chunk_size=32, force_history_len=20, force_history_window_sec=0.5,
                 dataset_roots=None, apply_manner_calibration=True):
        """dataset_roots: optional {session: root} for sessions living outside the default
        dio.DATASET_ROOT (e.g. dio.TWO_COLOR_DATASET_ROOT for data_two_color sessions) --
        sessions not present in this dict use the default root, so old callers passing plain
        dataset/demo{1,3,4} names need no changes.

        apply_manner_calibration: applies compliance_vla.policy.labels.calibrate_log_k's per-operator
        affine correction (fit by scripts/fit_manner_force_calibration.py) to each episode's
        log_k target, using that episode's own operator_id/manner from session_manifest.jsonl
        (looked up via _load_manifest_ordered's position-in-timestamp-order mapping -- the
        manifest's own raw episode_index field is a different, gappy numbering space, see
        that function's docstring). Silently no-ops per-episode if the calibration file
        hasn't been fit yet (lb.load_manner_calibration() returns None) or if that episode's
        (operator_id, manner) isn't in it -- log_k falls back to its raw, uncalibrated value,
        same as before this option existed."""
        self.chunk_size = chunk_size
        self.force_history_len = force_history_len
        self.force_history_window_sec = force_history_window_sec
        dataset_roots = dataset_roots or {}

        self.tool_offset = lb.load_tool_offset()
        self.extraction_args = lb.default_extraction_args()
        self.sigma_f = lb.default_sigma_f()
        self.calibration = lb.load_manner_calibration() if apply_manner_calibration else None

        self._episode_cache = {}
        self._image_cache = {}
        self._task_lookup = {}
        self.index = []
        self.n_episodes_failed = 0

        for session in sessions:
            root = dataset_roots.get(session)
            frames = dio.load_frames(session, columns=dio.NON_IMAGE_COLUMNS + IMAGE_COLUMNS, dataset_root=root)
            frames = frames.sort_values(["episode_index", "frame_index"])
            manifest_entries = _load_manifest_ordered(dio.session_dir(session, root))

            # Plain dict, not frames.set_index(...).loc[...]: pandas' fast_xs path for a
            # MultiIndex single-row lookup calls find_common_type() across the row's block
            # dtypes, and HF's Array3DExtensionType's __hash__ raises AttributeError there
            # (its own bug, hit non-deterministically depending on which rows happen to share
            # a block) -- confirmed by reproducing it via DataLoader shuffle order picking a
            # different row than a first-row smoke check exercised. A plain dict keyed by
            # (episode_index, frame_index) sidesteps pandas' block-dtype machinery entirely.
            self._image_cache[session] = {
                (int(ep), int(fi)): (scene, wrist)
                for ep, fi, scene, wrist in zip(
                    frames["episode_index"], frames["frame_index"],
                    frames["observation.images.scene_rgb"], frames["observation.images.wrist_rgb"],
                )
            }

            episodes_meta = dio.load_episodes_meta(session, dataset_root=root)
            task_lookup = dict(zip(
                episodes_meta["episode_index"],
                episodes_meta["tasks"].apply(lambda t: t[0] if len(t) else ""),
            ))

            for ep_idx, g in frames.groupby("episode_index"):
                g = g.sort_values("frame_index")
                g = _trim_to_last_contact(g)
                manifest_entry = manifest_entries[ep_idx] if ep_idx < len(manifest_entries) else {}
                operator_id = manifest_entry.get("operator_id")
                manner = manifest_entry.get("manner") or dio.parse_manner_from_task(manifest_entry.get("task", ""))
                arrays = lb.compute_episode_arrays(
                    g, self.tool_offset, self.extraction_args, self.sigma_f,
                    operator_id=operator_id, manner=manner, calibration=self.calibration,
                )
                if arrays is None:
                    self.n_episodes_failed += 1
                    continue
                self._episode_cache[(session, ep_idx)] = arrays
                self._task_lookup[(session, ep_idx)] = task_lookup.get(ep_idx, "")
                n = len(arrays["nearest_idx"])
                self.index.extend((session, ep_idx, start) for start in range(n))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        session, ep_idx, start = self.index[i]
        arr = self._episode_cache[(session, ep_idx)]
        H = self.chunk_size
        N = len(arr["nearest_idx"])

        end = min(start + H, N)
        valid_len = end - start
        raw_idx_chunk = arr["nearest_idx"][start:end]

        x_eq_chunk = np.zeros((H, 6), dtype=np.float32)
        log_k_chunk = np.zeros((H, 6), dtype=np.float32)
        mask_chunk = np.zeros((H, 6), dtype=bool)
        valid_chunk = np.zeros((H,), dtype=bool)

        x_eq_chunk[:valid_len] = arr["x_eq"][raw_idx_chunk]
        log_k_chunk[:valid_len] = arr["log_k"][start:end]
        mask_chunk[:valid_len] = arr["mask"][start:end]
        valid_chunk[:valid_len] = True
        gripper_chunk = np.full((H, 1), GRIPPER_PLACEHOLDER, dtype=np.float32)

        anchor_raw = arr["nearest_idx"][start]
        anchor_t = arr["t"][anchor_raw]

        state = np.concatenate([arr["q"][anchor_raw], arr["qdot"][anchor_raw], arr["x_f"][anchor_raw]]).astype(np.float32)

        t_raw = arr["t"]
        hist_sel = t_raw <= anchor_t
        force_hist = resample_to_n_samples(
            t_raw[hist_sel], arr["wrench"][hist_sel], t_end=anchor_t,
            window_sec=self.force_history_window_sec, n_samples=self.force_history_len,
        )

        anchor_frame_index = int(arr["frame_index"][anchor_raw])
        scene_cell, wrist_cell = self._image_cache[session][(int(ep_idx), anchor_frame_index)]
        scene = _decode_array3d(scene_cell)
        wrist = _decode_array3d(wrist_cell)

        return {
            "state": state,
            "force_history": force_hist,
            "scene_rgb": scene,
            "wrist_rgb": wrist,
            "action": np.concatenate([x_eq_chunk, log_k_chunk, gripper_chunk], axis=1),
            "log_k_mask": mask_chunk,
            "action_valid_mask": valid_chunk,
            "task": self._task_lookup[(session, ep_idx)],
        }


def _to_chw_float(img_hwc_uint8):
    """(H, W, 3) uint8 -> (3, H, W) float32 in [0, 1], matching SmolVLAPolicy.prepare_images's expected input."""
    return torch.from_numpy(img_hwc_uint8).permute(2, 0, 1).float() / 255.0


# Column indices into the dataset's native 13-dim action ([x_eq(6), log_k(6), gripper(1)])
# selecting just [x_eq(6), gripper(1)] = 7 dims, for the B0/B2 position-only baselines (Day 12).
POSITION_ONLY_COLUMNS = [0, 1, 2, 3, 4, 5, 12]


def make_collate_fn(tokenizer, tokenizer_max_length=48, pad_language_to="longest", action_layout="full13"):
    """Batches ComplianceWindowDataset items and tokenizes task text with the
    policy's own SmolVLM2 tokenizer (must match training/inference exactly,
    so this is a factory taking the tokenizer rather than a bare function).

    action_layout:
      - "full13" (default, B5): action stays [x_eq(6), log_k(6), gripper(1)].
      - "position7" (B0/B2, Day 12): action reduced to [x_eq(6), gripper(1)]
        via POSITION_ONLY_COLUMNS -- log_k_mask is still included in the
        batch (harmless, B0/B2's policy classes never read it) rather than
        conditionally omitted, keeping this one collate path simple.

    Also emits `actions_id_pad` (True = padding, i.e. NOT action_valid_mask)
    for compatibility with lerobot's unmodified `SmolVLAPolicy.forward`
    (B0), which reads that exact key name; compliance_vla.policy.compliance_policy's B2/B5
    policies use `action_valid_mask` (True = valid) instead -- both are
    included so one collate function serves all three baselines.
    """
    if action_layout not in ("full13", "position7"):
        raise ValueError(f"unknown action_layout {action_layout!r}")

    def collate(items):
        action_full = torch.stack([torch.from_numpy(it["action"]) for it in items])
        if action_layout == "position7":
            action_full = action_full[:, :, POSITION_ONLY_COLUMNS]
        valid = torch.stack([torch.from_numpy(it["action_valid_mask"]) for it in items])
        batch = {
            "observation.state": torch.stack([torch.from_numpy(it["state"]) for it in items]),
            "force_history": torch.stack([torch.from_numpy(it["force_history"]) for it in items]),
            "observation.images.scene_rgb": torch.stack([_to_chw_float(it["scene_rgb"]) for it in items]),
            "observation.images.wrist_rgb": torch.stack([_to_chw_float(it["wrist_rgb"]) for it in items]),
            "action": action_full,
            "log_k_mask": torch.stack([torch.from_numpy(it["log_k_mask"]) for it in items]),
            "action_valid_mask": valid,
            "actions_id_pad": ~valid,
        }
        tasks = [it["task"] or "" for it in items]
        tok = tokenizer(
            tasks, padding=pad_language_to, truncation=True, max_length=tokenizer_max_length,
            return_tensors="pt",
        )
        batch["observation.language.tokens"] = tok["input_ids"]
        batch["observation.language.attention_mask"] = tok["attention_mask"].bool()
        return batch

    return collate
