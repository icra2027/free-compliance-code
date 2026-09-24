#!/usr/bin/env python3
"""Diagnose whether a served checkpoint is actually conditioning on the
language instruction, independent of anything physical.

Holds one real (scene_rgb, wrist_rgb, state[, force_history]) observation --
pulled straight from a recorded training episode, via the same FK + tool_offset
path src/compliance_vla/policy/labels.py uses for x_f -- completely fixed, and only varies the
`task` colour word across a POST /act call per referent. If the checkpoint is
grounding language, the four predicted x_eq targets should land near the four
differently-coloured, differently-positioned marks actually visible in that
frame. If they cluster together regardless of colour, the model is falling
back to some other cue (most likely a positional prior) instead of parsing
the referent -- this isolates language-conditioning from camera setup, board
layout, and everything else in the real robot loop, since a single frozen
observation is reused for every request.

Only needs numpy/pandas/pyarrow/requests/json_numpy -- same "no torch/lerobot
needed" property as client_example.py, since it just POSTs to an already-
running src/compliance_vla/policy/serve_policy.py.

    python -m compliance_vla.policy.diagnose_language_grounding --session demo1 --episode 3

Point --server-url at wherever serve_policy.py is listening (tunnel first if
it's remote, exactly as client_example.py's docstring describes).
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from compliance_vla.policy._paths import DATA_EXTRACTION_DIR, ensure_on_sys_path  # noqa: E402

ensure_on_sys_path()  # dataset_io / panda_fk are plain scripts under data_extraction/

import dataset_io as dio  # noqa: E402
import panda_fk as fk  # noqa: E402

REFERENTS = ("red", "blue", "green", "black")

# 2-colour descope (2026-09-01 decision):
# every 4-way discrimination attempt (frozen backbone, unfreeze-lm, unfreeze-lm+vision, a
# full-unfreeze overfit on the model's own training episodes, and LoRA through at least step
# 10000) has landed at or below the 25% chance floor, with a persistent collapse onto 1-2
# colours regardless of the true one. Restricting to 2 referents raises chance to 50% and,
# per the positional-prior check (2026-08-28), the existing signal is real but ~3x too weak
# for 4-way -- a 2-way task is a much more plausible bar for it to clear. red/blue chosen for
# maximum hue separation (both far from each other and from the achromatic black in RGB
# space), not because either showed unusual promise in the diagnostic data -- per-session
# colour counts are balanced (~7-8 episodes/colour/session) for every pair, so this is a
# free choice among equals, not a fit to noise.
DEFAULT_DESCOPE_REFERENTS = ("red", "blue")
FORCE_HISTORY_WINDOW_SEC = 0.5
FORCE_HISTORY_SAMPLES = 20

# 2026-09-03: the data_recorder used for data_two_color (external/data_recorder
# commit 2f63ca3) appends a return-to-start_joint_configuration move to every saved
# episode, still recording, before the 's' key actually stops+saves (see that repo's README,
# "s: save successful episode"). A blind trailing-30%-of-episode ground truth therefore mostly
# or entirely samples that fixed home pose, not the real wipe endpoint.
#
# A first fix attempt used a contact-force threshold (last frame above 2N, the convention
# analyze_data_two_color.py / analyze_adverb_separation.py use elsewhere) and was WRONG: the
# appended return-to-home move produces external-wrench estimation transients from the fast
# commanded motion itself that routinely exceed a naive 2N threshold throughout the return trip
# (2-3.6N observed in a real trace, vs 4-10N during real contact) -- "last frame above
# threshold" landed almost at the raw episode end, barely correcting anything (confirmed: the
# "corrected" centroids this produced converged to the SAME x/y/z across every session
# regardless of operator/colour/manner -- exactly what a single shared
# start_joint_configuration looks like, not 8 different marks).
#
# Fix: use velocity instead of force, gated on distance from the episode's own start position.
# Real wiping is reliably followed by a near-zero-velocity pause (arm lifts off, holds still)
# before the reset motion begins as continuous glide with no further stop until it arrives back
# near the episode's own starting pose (episodes idle at start_joint_configuration before
# wiping begins, so "close to frame 0's position" ~= "arrived home"). So: walk backward from
# episode end and find the last pause that is BOTH slow and far from the start position -- this
# skips over the final arrival-at-home settling pause (also near-zero-velocity, which is what
# broke a velocity-only, distance-unaware first attempt at this) and lands on the real "wipe
# done, holding position away from home" moment instead. See src/compliance_vla/policy/dataset.py's
# _trim_to_last_contact for the training-side counterpart of this same fix, with the fuller
# writeup of both wrong attempts before this one.
SPEED_PAUSE_THRESHOLD_M_S = 0.01  # m/s
NEAR_HOME_RADIUS_M = 0.05  # m, matches src/compliance_vla/policy/dataset.py's constant


def get_action_chunk(server_url, scene_rgb, wrist_rgb, state, task, force_history=None, seed=None):
    """Same contract as client_example.py's helper of the same name, plus an optional
    `seed` -- serve_policy.py reseeds torch right before flow-matching sampling when given
    one, so calls that pass the same seed get the same initial noise. Without this, each
    colour word in evaluate_episode gets an independent random noise draw for the ODE
    integration, which confounds "the model doesn't ground language" with "the model's
    flow field isn't very peaked and sampling variance moved the answer" -- see the
    --seed/--n-seeds help text.

    json_numpy/requests are imported here, not at module level, so that importing just
    _resolve_session_dir/pick_balanced_episodes (as src/compliance_vla/policy/train_policy.py's
    --overfit-n-per-referent does) doesn't require them installed -- bit us on HPC's
    offline wheelhouse, which pins only what training itself needs."""
    import json_numpy
    import requests

    payload = {"task": task, "scene_rgb": scene_rgb, "wrist_rgb": wrist_rgb, "state": state}
    if force_history is not None:
        payload["force_history"] = force_history
    if seed is not None:
        payload["seed"] = seed
    resp = requests.post(server_url, data=json_numpy.dumps(payload),
                          headers={"Content-Type": "application/json"})
    resp.raise_for_status()
    return json_numpy.loads(resp.content)["action_chunk"]


def referent_from_task(task_text):
    for colour in REFERENTS:
        if colour in task_text:
            return colour
    return None


def _decode_array3d(cell):
    """Mirrors src/compliance_vla/policy/dataset.py's _decode_array3d: the image cell comes back
    from pandas/pyarrow as a doubly-nested object ndarray."""
    return np.stack([np.stack(row) for row in cell]).astype(np.uint8)


def _resolve_session_dir(session, dataset_root=None):
    """dataset_io.DATASET_ROOT assumes compliance-vla's own parent holds
    dataset/ -- true on the training machine, but this checkout sits one level
    deeper (franka_ros2_ws/src/compliance-vla), where the dataset was
    instead pulled back to franka_ros2_ws/dataset. Try the real DATASET_ROOT
    first (correct on a standalone clone), then one level further up, before
    giving up -- --dataset-root overrides both."""
    if dataset_root:
        return os.path.join(dataset_root, session)
    candidates = [
        dio.session_dir(session),
        os.path.join(os.path.dirname(dio.REPO_ROOT), "dataset", session),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    raise FileNotFoundError(
        f"couldn't find session {session!r} in any of: {candidates} -- pass --dataset-root"
    )


def _load_manifest_ordered(session_path):
    """session_manifest.jsonl's own `episode_index` field is a RAW recording-sequence
    number that includes discarded/failed takes (e.g. demo1's manifest has episode_index
    values 2,3,4,5,7,8,10,... -- 0,1,6,9,... are gaps, discarded before ever reaching the
    exported dataset). The dataset's own `episode_index` (what training and this script's
    parquet reads use everywhere else) is a COMPACTED 0..N-1 renumbering of only the
    episodes that survived, in recording order. These are two different numbering spaces --
    matching on `entry["episode_index"] == dataset_episode_index` (this function used to be
    inlined that way in both load_observation and pick_balanced_episodes) essentially never
    finds the right episode: verified via frame_count cross-check (a near-unique per-episode
    fingerprint) that position-in-timestamp-order is the correct mapping and the raw
    episode_index field coincidentally matches the dataset index in ZERO of 92 episodes
    across demo1/demo3/demo4. This was silently scoring every diagnostic run against a
    DIFFERENT episode's ground truth. Returns manifest entries sorted by timestamp, so
    `entries[dataset_episode_index]` is the correct one -- see load_observation's frame_count
    assertion for a hard guard against this recurring."""
    entries = []
    with open(os.path.join(session_path, "session_manifest.jsonl")) as f:
        for line in f:
            entries.append(json.loads(line))
    entries.sort(key=lambda e: e["timestamp"])
    return entries


def _true_wipe_endpoint(ep, speed_threshold=SPEED_PAUSE_THRESHOLD_M_S,
                         near_home_radius=NEAR_HOME_RADIUS_M, settle_frames=5):
    """Ground-truth wipe target for one episode's own leader_pose trajectory.

    Walks backward from episode end to find the last pause (near-zero velocity) that is also
    far from the episode's own start position -- see this module's SPEED_PAUSE_THRESHOLD_M_S
    comment for why (force-based and distance-unaware-velocity-based attempts at this were both
    tried and were both wrong). Median-filtered over a small trailing window ending there, to
    denoise a single noisy frame. Falls back to the old trailing-30% heuristic (with a loud
    warning) if no such pause is found -- e.g. an episode that never paused near its end --
    rather than silently returning garbage."""
    pos = np.stack(ep["observation.leader_pose"].to_numpy())[:, :3]
    t = ep["timestamp"].to_numpy().astype(np.float64)
    n = len(pos)
    start_pos = pos[0]
    dist_from_start = np.linalg.norm(pos - start_pos, axis=1)
    dt = np.diff(t)
    with np.errstate(invalid="ignore", divide="ignore"):
        speed = np.linalg.norm(np.diff(pos, axis=0), axis=1) / dt
    k = n - 2
    while k >= 0 and not (speed[k] < speed_threshold and dist_from_start[k] > near_home_radius):
        k -= 1
    if k < 0:
        print("  ** WARNING: episode never pauses away from its own start position -- falling "
              "back to trailing-30% ground truth, which may itself be contaminated by appended "
              "home motion. Treat this episode's ground truth with suspicion. **")
        tail = ep.iloc[int(len(ep) * 0.7):]
        return np.median(np.stack(tail["observation.leader_pose"].to_numpy())[:, :3], axis=0)
    pause_idx = k + 1
    window = ep.iloc[max(0, pause_idx - settle_frames + 1):pause_idx + 1]
    return np.median(np.stack(window["observation.leader_pose"].to_numpy())[:, :3], axis=0)


def load_observation(session, episode_index, frame_index, tool_offset, dataset_root=None):
    """Builds the exact (scene_rgb, wrist_rgb, state, force_history) tuple
    deploy_smolvla.py would have sent for this one real recorded frame.

    Reads only the single data file that holds this episode (per meta/episodes'
    data/chunk_index+data/file_index), not the whole session -- the image
    columns dominate a session's size (dataset_io.py's own docstring: ~3.8GB
    across 3 sessions for 92 episodes), and loading all of it at once for a
    single-frame diagnostic OOM-killed this on a modest box."""
    session_path = _resolve_session_dir(session, dataset_root)
    meta_files = sorted(glob.glob(os.path.join(session_path, "meta", "episodes", "chunk-*", "file-*.parquet")))
    meta_cols = ["episode_index", "data/chunk_index", "data/file_index"]
    meta = pa.concat_tables([pq.read_table(f, columns=meta_cols) for f in meta_files]).to_pandas()
    ep_meta = meta[meta["episode_index"] == episode_index]
    if ep_meta.empty:
        raise ValueError(f"no episode_index={episode_index} in {session}'s meta/episodes")
    chunk_idx = int(ep_meta.iloc[0]["data/chunk_index"])
    file_idx = int(ep_meta.iloc[0]["data/file_index"])
    data_file = os.path.join(session_path, "data", f"chunk-{chunk_idx:03d}", f"file-{file_idx:03d}.parquet")

    cols = dio.NON_IMAGE_COLUMNS + ["observation.images.scene_rgb", "observation.images.wrist_rgb"]
    df = pq.read_table(data_file, columns=cols).to_pandas()

    ep = df[df["episode_index"] == episode_index].sort_values("frame_index").reset_index(drop=True)
    if ep.empty:
        raise ValueError(f"no rows for {session} episode_index={episode_index}")
    if frame_index >= len(ep):
        raise ValueError(f"episode has {len(ep)} frames, frame_index={frame_index} out of range")

    row = ep.iloc[frame_index]
    q = np.asarray(row["observation.state"], dtype=np.float64)
    qdot = np.asarray(row["observation.velocity"], dtype=np.float32)

    pos, rot = fk.fk_batch(q[None])
    tip_pos = pos[0] + rot[0] @ tool_offset
    tip_rotvec = fk.rotvec_from_matrix(rot[0])
    x_f = np.concatenate([tip_pos, tip_rotvec]).astype(np.float32)
    state = np.concatenate([q.astype(np.float32), qdot, x_f])

    scene_rgb = _decode_array3d(row["observation.images.scene_rgb"])
    wrist_rgb = _decode_array3d(row["observation.images.wrist_rgb"])

    hist = ep.iloc[max(0, frame_index - 30):frame_index + 1]
    wrench = np.stack(hist["observation.wrench.external_base"].to_numpy()).astype(np.float32)
    times = hist["timestamp"].to_numpy().astype(np.float64)
    if len(hist) >= 2 and times[-1] - times[0] > 0:
        target_times = np.linspace(
            times[-1] - FORCE_HISTORY_WINDOW_SEC, times[-1], FORCE_HISTORY_SAMPLES
        )
        force_history = np.stack(
            [np.interp(target_times, times, wrench[:, i]) for i in range(6)], axis=1
        ).astype(np.float32)
    else:
        force_history = np.zeros((FORCE_HISTORY_SAMPLES, 6), dtype=np.float32)

    # Ground truth for this exact episode: the trailing 30% of leader_pose (the actual
    # physical wipe location that really happened here) and the real task/colour that
    # produced it -- lets the comparison check *accuracy*, not just "did it move". See
    # _load_manifest_ordered's docstring: must map by position (timestamp order), NOT by
    # matching session_manifest.jsonl's own `episode_index` field against this dataset
    # episode_index -- those are different numbering spaces.
    manifest_entries = _load_manifest_ordered(session_path)
    if episode_index >= len(manifest_entries):
        raise ValueError(
            f"{session} dataset episode_index={episode_index} has no manifest counterpart "
            f"({len(manifest_entries)} manifest entries total)"
        )
    manifest_entry = manifest_entries[episode_index]
    # Hard guard against the position-mapping being wrong (recording gaps in a different
    # place than expected, a manifest edited out of order, etc.): frame_count is a
    # near-unique per-episode fingerprint, cheap to check, and this exact assertion would
    # have caught the raw-vs-compacted index bug the day this script was first written.
    if manifest_entry.get("frame_count") != len(ep):
        raise ValueError(
            f"{session} episode_index={episode_index}: manifest entry (raw episode_index="
            f"{manifest_entry.get('episode_index')}) has frame_count={manifest_entry.get('frame_count')}, "
            f"but the dataset episode has {len(ep)} frames -- manifest<->dataset episode mapping "
            "is wrong, do not trust this ground truth"
        )
    true_task = manifest_entry.get("task", "")
    true_referent = referent_from_task(true_task)
    true_target = _true_wipe_endpoint(ep)

    return scene_rgb, wrist_rgb, state, force_history, true_referent, true_task, true_target


def pick_balanced_episodes(session_path, n_per_referent=1, referents=REFERENTS):
    """Auto-picks DATASET episode_index values (0..N-1, position in timestamp order --
    see _load_manifest_ordered) spanning each of `referents`, so a default run gets
    balanced colour coverage without the caller having to know episode indices up front.
    Used to return session_manifest.jsonl's own raw `episode_index` field instead, which
    is a different (pre-discard) numbering space -- see _load_manifest_ordered's docstring
    for why that silently picked/scored the wrong episodes.

    `referents` defaults to all 4; pass e.g. DEFAULT_DESCOPE_REFERENTS to pick only
    episodes whose true colour is one of a 2-colour subset (episodes for the other
    colours are skipped entirely, not folded in as noise)."""
    by_referent = {c: [] for c in referents}
    for dataset_episode_index, entry in enumerate(_load_manifest_ordered(session_path)):
        ref = referent_from_task(entry.get("task", ""))
        if ref in by_referent:
            by_referent[ref].append(dataset_episode_index)
    picked = []
    for c in referents:
        picked.extend(sorted(by_referent[c])[:n_per_referent])
    return sorted(picked)


def evaluate_episode(server_url, session, episode_index, frame_index, manner,
                      tool_offset, force_history_enabled, dataset_root, seeds, referents=REFERENTS):
    scene_rgb, wrist_rgb, state, force_history, true_referent, true_task, true_target = load_observation(
        session, episode_index, frame_index, tool_offset, dataset_root
    )
    if not force_history_enabled:
        force_history = None
    if true_referent not in referents:
        raise ValueError(
            f"{session} episode={episode_index}'s true referent {true_referent!r} is not in the "
            f"active referent set {referents} -- pick episodes via pick_balanced_episodes(..., "
            "referents=referents) or --episode values whose true colour is in --referents, "
            "otherwise the candidate set can't possibly include the right answer."
        )

    print(f"\n=== {session} episode={episode_index} frame={frame_index} -- "
          f"true task={true_task!r} (referent={true_referent!r}) -- candidates={referents} ===")

    positions_by_colour = {c: [] for c in referents}
    per_seed_closest = []
    for seed in seeds:
        predictions = {}
        if len(seeds) > 1:
            print(f"  -- seed={seed} --")
        else:
            print(f"{'colour':<8} {'x_eq pos (LAST predicted step)':<38} {'dist from truth (m)'}")
        for colour in referents:
            task = f"wipe the {colour} mark {manner}"
            action_chunk = get_action_chunk(
                server_url, scene_rgb, wrist_rgb, state, task, force_history, seed=seed
            )
            x_eq_pos = np.asarray(action_chunk[-1][:3], dtype=np.float64)
            predictions[colour] = x_eq_pos
            positions_by_colour[colour].append(x_eq_pos)
            dist_from_truth = np.linalg.norm(x_eq_pos - true_target)
            flag = "  <- actually happened" if colour == true_referent else ""
            if len(seeds) == 1:
                print(f"{colour:<8} {np.array2string(x_eq_pos, precision=4):<38} {dist_from_truth:.4f}{flag}")

        closest = min(predictions, key=lambda c: np.linalg.norm(predictions[c] - true_target))
        per_seed_closest.append(closest)

    # Per-seed argmin vote: with n_seeds=1 this is just that one seed's answer, identical to
    # the original single-shot behaviour. With n_seeds>1 it's still noise-sensitive (each
    # seed's own argmin can flip independently of the others) -- reported mainly to show the
    # flip rate, not trusted as the accuracy number.
    vote_counts = {c: per_seed_closest.count(c) for c in set(per_seed_closest)}
    vote_closest = max(vote_counts, key=vote_counts.get)
    flip_rate = 1.0 - vote_counts[vote_closest] / len(per_seed_closest)

    # Mean-position estimator: average each colour's predicted x_eq across seeds first, THEN
    # take the argmin -- averaging cancels sampling noise (CLT) the way a per-seed vote can't,
    # since a vote only sees which bucket each noisy sample fell into, not how far it moved.
    # This is the metric --n-seeds>1 is actually for; per-seed vote is kept above as a
    # diagnostic on how unstable a single-seed read (the original script's behaviour) is.
    mean_positions = {c: np.mean(positions_by_colour[c], axis=0) for c in referents}
    closest = min(mean_positions, key=lambda c: np.linalg.norm(mean_positions[c] - true_target))
    correct = true_referent is not None and closest == true_referent
    verdict = "CORRECT" if correct else "wrong"
    if len(seeds) > 1:
        print(f"  per-seed closest (noisy):    {per_seed_closest} (flip rate {flip_rate:.0%})")
        print(f"  mean-position-over-seeds dist from truth: " +
              ", ".join(f"{c}={np.linalg.norm(mean_positions[c] - true_target):.4f}" for c in referents))
    print(f"closest prediction to truth: {closest!r} (actual was {true_referent!r}) -- {verdict}")
    return correct, true_referent, closest, flip_rate


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--server-url", default="http://localhost:8000/act")
    p.add_argument("--session", default="demo1", choices=dio.SESSIONS + dio.TWO_COLOR_SESSIONS)
    p.add_argument("--episode", type=int, nargs="+", default=None,
                   help="episode_index(es) within --session; default: one episode per "
                        "referent, auto-picked for balanced colour coverage")
    p.add_argument("--n-per-referent", type=int, default=1,
                   help="when --episode is omitted, how many episodes per referent to sample")
    p.add_argument("--frame-index", type=int, default=5,
                   help="frame within the episode -- early, before the arm occludes/reaches a mark")
    p.add_argument("--manner", default="normally")
    p.add_argument("--seed", type=int, default=0,
                    help="base seed for the flow-matching sampling noise -- held fixed across the 4 "
                         "colour queries within an episode (via serve_policy.py's optional 'seed' field) "
                         "so a difference in prediction can only come from the task text, not an "
                         "independent random noise draw per colour.")
    p.add_argument("--n-seeds", type=int, default=1,
                    help="run each episode at this many consecutive seeds (--seed, --seed+1, ...) and "
                         "take a majority vote, reporting how often the answer flips across noise draws "
                         "alone (same task text) -- use >1 to check whether sampling variance, not "
                         "language grounding, is driving --n-seeds=1's verdict.")
    p.add_argument("--no-force", action="store_true", help="omit force_history (b0 checkpoints)")
    p.add_argument("--dataset-root", default=None,
                   help="override if dataset_io.DATASET_ROOT doesn't match this checkout's layout")
    p.add_argument("--referents", nargs="+", default=None, choices=REFERENTS,
                   help="restrict the candidate colour set (default: all 4). Pass "
                        "'--referents red blue' for the 2026-09-01 2-colour descope -- raises "
                        "chance from 25%% to 50%% and only evaluates/picks episodes whose true "
                        "colour is in this set. See DEFAULT_DESCOPE_REFERENTS.")
    args = p.parse_args()
    referents = tuple(args.referents) if args.referents else REFERENTS

    tool_offset = np.load(os.path.join(DATA_EXTRACTION_DIR, "tool_offset.npy"))
    session_path = _resolve_session_dir(args.session, args.dataset_root)
    episodes = args.episode if args.episode is not None else pick_balanced_episodes(
        session_path, args.n_per_referent, referents=referents
    )
    print(f"Evaluating {len(episodes)} episode(s) from {args.session} (referents={referents}): {episodes}")

    seeds = list(range(args.seed, args.seed + args.n_seeds))
    results = []
    for ep in episodes:
        results.append(evaluate_episode(
            args.server_url, args.session, ep, args.frame_index, args.manner,
            tool_offset, not args.no_force, args.dataset_root, seeds, referents=referents,
        ))

    n_correct = sum(1 for correct, _, _, _ in results if correct)
    print(f"\n=== summary: {n_correct}/{len(results)} episodes -- closest prediction matched "
          "the true colour ===")
    for correct, true_ref, closest, flip_rate in results:
        flip_note = f" (flip rate {flip_rate:.0%})" if args.n_seeds > 1 else ""
        print(f"  true={true_ref:<6} predicted-closest={closest:<6} {'OK' if correct else 'MISS'}{flip_note}")


if __name__ == "__main__":
    main()
