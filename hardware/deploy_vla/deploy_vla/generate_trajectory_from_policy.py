"""
Teacher-forced offline replay: feeds one real, trimmed data_two_color episode's actual
recorded observations (scene/wrist images, state, task string, force history) to the
policy server -- exactly serve_policy.py's /act endpoint deploy_smolvla.py talks to -- one
real anchor frame at a time, and records what the model predicts at each one.

"Teacher-forced" matters here: each anchor's images/state come straight from the real
recording, not from a previous step's predicted pose, so nothing ever compounds and the
model never has to react to a scene it couldn't actually have produced (there's no
simulator to re-render what the wrist camera would see if the arm had gone somewhere the
demo never went). This directly answers "does the model's own prediction match its own
training label, frame by frame" without needing the real robot at all.

Trims the episode with the same _trim_to_last_contact logic src/compliance_vla/policy/dataset.py applies
(copied here, not imported -- that module pulls in torch, and this script is meant to run
anywhere client_example.py can, e.g. a machine with neither a GPU nor the training deps
installed). See that module's docstring for why a naive contact-force threshold is not
enough to drop data_two_color's appended return-to-start_joint_configuration tail.

Usage:
    python3 generate_trajectory_from_policy.py --session demo_blue_firm --episode 11
    python3 generate_trajectory_from_policy.py --session demo_blue_firm --list-episodes

Only needs json_numpy/requests/numpy/pandas/pyarrow (dataset_io's own deps) -- no rclpy,
no torch, no lerobot, and it never touches the real robot.
"""

import argparse
import csv
import os
import sys

import numpy as np
import json_numpy
import requests

# --- sys.path wiring, matching replay_data_two_color.py and every other ad hoc analysis
# script written during this investigation -- these repo-relative paths are specific to
# how compliance-vla/fr3_bilateral_teleop are vendored into *this* workspace.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_WS_SRC = os.path.dirname(os.path.dirname(_THIS_DIR))          # franka_ros2_ws/src
_REPO_ROOT = os.path.dirname(_WS_SRC)                          # franka_ros2_ws
_BOOKISH_ROOT = os.path.join(_WS_SRC, "compliance-vla")
_BOOKISH_SCRIPTS = os.path.join(_BOOKISH_ROOT, "scripts")
_TELEOP_SCRIPTS = os.path.join(_WS_SRC, "fr3_bilateral_teleop", "dataset_tools", "labeling")
for _p in (_BOOKISH_ROOT, _BOOKISH_SCRIPTS, _TELEOP_SCRIPTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dataset_io as dio  # noqa: E402
from compliance_vla.policy import labels as lb  # noqa: E402

DATA_TWO_COLOR_ROOT = os.path.join(_REPO_ROOT, "data_two_color")
IMAGE_COLUMNS = ["observation.images.scene_rgb", "observation.images.wrist_rgb"]

SPEED_PAUSE_THRESHOLD_M_S = 0.01
NEAR_HOME_RADIUS_M = 0.05

FORCE_HISTORY_WINDOW_SEC = 0.5
FORCE_HISTORY_LEN = 20


def _trim_to_last_contact(g, speed_threshold=SPEED_PAUSE_THRESHOLD_M_S,
                           near_home_radius=NEAR_HOME_RADIUS_M, settle_frames=2):
    """Copied verbatim from compliance-vla/src/compliance_vla/policy/dataset.py -- see that
    module's docstring for the full rationale (a naive contact-force threshold does not
    exclude data_two_color's appended return-to-start_joint_configuration tail)."""
    pos = np.stack(g["observation.leader_pose"].to_numpy())[:, :3]
    t = g["timestamp"].to_numpy().astype(np.float64)
    n = len(pos)
    if n < 10:
        return g
    start_pos = pos[0]
    dist_from_start = np.linalg.norm(pos - start_pos, axis=1)
    dt = np.diff(t)
    with np.errstate(invalid="ignore", divide="ignore"):
        speed = np.linalg.norm(np.diff(pos, axis=0), axis=1) / dt
    k = n - 2
    while k >= 0 and not (speed[k] < speed_threshold and dist_from_start[k] > near_home_radius):
        k -= 1
    if k < 0:
        return g
    cutoff = min(k + 1 + settle_frames, n)
    return g.iloc[:cutoff]


def _decode_array3d(cell):
    """Copied from compliance-vla/src/compliance_vla/policy/dataset.py: the dataset's
    Array3DExtensionType image cells come back from pandas/pyarrow as a doubly-nested
    object ndarray; recursing one level explicitly resolves it."""
    return np.stack([np.stack(row) for row in cell]).astype(np.uint8)


def resample_to_n_samples(times, values, t_end, window_sec, n_samples):
    """Copied from compliance-vla/src/compliance_vla/policy/force_encoder.py (which imports torch
    at module level for unrelated nn.Module classes -- not needed here). Linearly
    resamples `values` (T, C) onto n_samples evenly spaced points covering
    [t_end - window_sec, t_end], holding the earliest available sample constant to fill
    any gap -- the exact history shape the model was trained against at every anchor,
    including near an episode's own start."""
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    t0 = t_end - window_sec
    query = np.linspace(t0, t_end, n_samples)

    in_window = times <= t_end
    if in_window.sum() == 0:
        return np.repeat(values[:1], n_samples, axis=0).astype(np.float32)
    t_hist, v_hist = times[in_window], values[in_window]
    if len(t_hist) == 1:
        return np.repeat(v_hist, n_samples, axis=0).astype(np.float32)

    out = np.empty((n_samples, values.shape[1]), dtype=np.float64)
    for c in range(values.shape[1]):
        out[:, c] = np.interp(query, t_hist, v_hist[:, c], left=v_hist[0, c], right=v_hist[-1, c])
    return out.astype(np.float32)


def _resize(img, size):
    if img.shape[0] == size and img.shape[1] == size:
        return img
    import cv2
    return cv2.resize(img, (size, size))


def list_episodes(session):
    frames = dio.load_frames(session, dataset_root=DATA_TWO_COLOR_ROOT)
    meta = dio.load_episodes_meta(session, dataset_root=DATA_TWO_COLOR_ROOT)
    task_lookup = dict(zip(meta["episode_index"], meta["tasks"].apply(lambda t: t[0] if len(t) else "")))
    for ep_idx, g in frames.groupby("episode_index"):
        print(f"  episode {ep_idx:>3}: {len(g):>4} frames  task={task_lookup.get(ep_idx, '')!r}")


def _load_single_episode_frame(session, episode_index):
    """Reads only the one parquet file that holds this episode (data_two_color stores
    exactly one episode per file -- confirmed via load_episodes_meta's data/chunk_index,
    data/file_index columns), instead of dataset_io.load_frames' concat-every-file-in-
    the-session behaviour. That matters here specifically because it's the only place
    this script touches image columns: loading all ~24 episodes' worth of scene/wrist
    frames (~1.7GB of nested-object image cells for demo_blue_firm alone) to read one
    episode OOM-killed this exact call during testing."""
    import pyarrow.parquet as pq

    meta = dio.load_episodes_meta(session, dataset_root=DATA_TWO_COLOR_ROOT)
    row = meta[meta["episode_index"] == episode_index]
    if row.empty:
        raise ValueError(f"episode {episode_index} not found in session {session!r} "
                          f"(available: {sorted(meta['episode_index'])})")
    chunk_idx = int(row.iloc[0]["data/chunk_index"])
    file_idx = int(row.iloc[0]["data/file_index"])
    path = os.path.join(dio.session_dir(session, DATA_TWO_COLOR_ROOT), "data",
                         f"chunk-{chunk_idx:03d}", f"file-{file_idx:03d}.parquet")
    table = pq.read_table(path, columns=dio.NON_IMAGE_COLUMNS + IMAGE_COLUMNS)
    df = table.to_pandas()
    return df[df["episode_index"] == episode_index]


def load_trimmed_episode_with_images(session, episode_index):
    """Returns (arrays, image_lookup, task) for one trimmed episode. arrays is
    compliance_vla.policy.labels.compute_episode_arrays' return dict (t, q, qdot, x_f, x_eq, wrench,
    frame_index, ...); image_lookup maps raw frame_index -> (scene_rgb, wrist_rgb)
    uint8 arrays for every frame still present after trimming."""
    g = _load_single_episode_frame(session, episode_index).sort_values("frame_index")
    n_before = len(g)
    g = _trim_to_last_contact(g)
    print(f"[replay] session={session} episode={episode_index}: "
          f"trimmed {n_before} -> {len(g)} frames ({n_before - len(g)} dropped)")

    image_lookup = {
        int(fi): (_decode_array3d(scene), _decode_array3d(wrist))
        for fi, scene, wrist in zip(g["frame_index"], g["observation.images.scene_rgb"],
                                     g["observation.images.wrist_rgb"])
    }

    tool_offset = lb.load_tool_offset()
    args = lb.default_extraction_args()
    sigma_f = lb.default_sigma_f()
    arrays = lb.compute_episode_arrays(g, tool_offset, args, sigma_f)
    if arrays is None:
        raise RuntimeError(f"{session}#{episode_index}: contact-frame extraction failed "
                            "after trimming -- pick a different episode (see --list-episodes).")

    meta = dio.load_episodes_meta(session, dataset_root=DATA_TWO_COLOR_ROOT)
    task_lookup = dict(zip(meta["episode_index"], meta["tasks"].apply(lambda t: t[0] if len(t) else "")))
    return arrays, image_lookup, task_lookup.get(episode_index, "")


def get_action_chunk(server_url, scene_rgb, wrist_rgb, state, task, force_history=None):
    payload = {"task": task, "scene_rgb": scene_rgb, "wrist_rgb": wrist_rgb, "state": state}
    if force_history is not None:
        payload["force_history"] = force_history
    resp = requests.post(server_url, data=json_numpy.dumps(payload),
                          headers={"Content-Type": "application/json"})
    resp.raise_for_status()
    return json_numpy.loads(resp.content)["action_chunk"]


def generate_trajectory(server_url, session, episode, stride, image_size,
                         include_force_history, out_csv):
    arrays, image_lookup, task = load_trimmed_episode_with_images(session, episode)
    t, q, qdot, x_f, x_eq, wrench, frame_index = (
        arrays["t"], arrays["q"], arrays["qdot"], arrays["x_f"], arrays["x_eq"],
        arrays["wrench"], arrays["frame_index"],
    )
    n = len(t)
    anchors = list(range(0, n, max(1, stride)))
    print(f"[replay] task={task!r}  {n} trimmed frames, replaying {len(anchors)} anchors "
          f"(stride={stride}) against {server_url}")

    rows = []
    for step, i in enumerate(anchors):
        state = np.concatenate([q[i], qdot[i], x_f[i]]).astype(np.float32)

        force_history = None
        if include_force_history:
            hist_sel = t <= t[i]
            force_history = resample_to_n_samples(
                t[hist_sel], wrench[hist_sel], t_end=t[i],
                window_sec=FORCE_HISTORY_WINDOW_SEC, n_samples=FORCE_HISTORY_LEN,
            )

        scene, wrist = image_lookup[int(frame_index[i])]
        scene, wrist = _resize(scene, image_size), _resize(wrist, image_size)

        action_chunk = get_action_chunk(server_url, scene, wrist, state, task, force_history)
        action_dim = action_chunk.shape[-1]
        if action_dim not in (7, 13):
            raise RuntimeError(f"unexpected action_dim={action_dim} (expected 7 or 13)")
        predicted_x_eq = np.asarray(action_chunk[0, 0:6], dtype=np.float64)

        print(f"[replay] step {step}/{len(anchors) - 1}  anchor_frame={i}/{n - 1}  "
              f"t={t[i]:.3f}s  label_x_eq={np.array2string(x_eq[i], precision=4)}  "
              f"predicted_x_eq={np.array2string(predicted_x_eq, precision=4)}")

        rows.append((t[i], *x_eq[i], *predicted_x_eq))

    label_dx = rows[-1][1] - rows[0][1]
    label_dy = rows[-1][2] - rows[0][2]
    pred_dx = rows[-1][7] - rows[0][7]
    pred_dy = rows[-1][8] - rows[0][8]
    signs_match = np.sign(pred_dx) == np.sign(label_dx)
    print(
        f"\n[replay] label    Δx={label_dx:+.4f}m Δy={label_dy:+.4f}m\n"
        f"[replay] predicted Δx={pred_dx:+.4f}m Δy={pred_dy:+.4f}m\n"
        f"[replay] x sign {'MATCH' if signs_match else 'FLIPPED'}"
    )

    if out_csv:
        with open(out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["t", "label_x", "label_y", "label_z", "label_rx", "label_ry", "label_rz",
                              "pred_x", "pred_y", "pred_z", "pred_rx", "pred_ry", "pred_rz"])
            writer.writerows(rows)
        print(f"[replay] wrote {len(rows)} rows -> {out_csv}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", required=True, choices=dio.TWO_COLOR_SESSIONS)
    parser.add_argument("--episode", type=int, default=None)
    parser.add_argument("--list-episodes", action="store_true")
    parser.add_argument("--server-url", default="http://127.0.0.1:8000/act")
    parser.add_argument("--stride", type=int, default=1, help="replay every Nth trimmed frame instead of all of them")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--no-force-history", action="store_true", help="omit force_history (b0 checkpoints)")
    parser.add_argument("--out-csv", default=None, help="optional path to save the label-vs-predicted trajectory")
    cli = parser.parse_args()

    if cli.list_episodes:
        list_episodes(cli.session)
        return
    if cli.episode is None:
        parser.error("--episode is required unless --list-episodes is given")

    generate_trajectory(
        cli.server_url, cli.session, cli.episode, cli.stride, cli.image_size,
        include_force_history=not cli.no_force_history, out_csv=cli.out_csv,
    )


if __name__ == "__main__":
    main()
