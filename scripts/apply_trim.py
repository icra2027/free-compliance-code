import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets.compute_stats import get_feature_stats

DATASET_ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/robot/repos/datasets/data_two_color")

SPEED_PAUSE_THRESHOLD_M_S = 0.01
NEAR_HOME_RADIUS_M = 0.05
SETTLE_FRAMES = 2

IMAGE_FEATURES = ["observation.images.scene_rgb", "observation.images.wrist_rgb"]
VECTOR_FEATURES = [
    "observation.state", "action", "observation.wrench.external_base",
    "observation.wrench.external_stiffness", "observation.leader_pose", "observation.velocity",
]
SCALAR_FEATURES = ["timestamp", "frame_index", "episode_index", "index", "task_index"]


def trim_cutoff(pos, t):
    n = len(pos)
    if n < 10:
        return n
    start_pos = pos[0]
    dist_from_start = np.linalg.norm(pos - start_pos, axis=1)
    dt = np.diff(t)
    with np.errstate(invalid="ignore", divide="ignore"):
        speed = np.linalg.norm(np.diff(pos, axis=0), axis=1) / dt
    k = n - 2
    while k >= 0 and not (speed[k] < SPEED_PAUSE_THRESHOLD_M_S and dist_from_start[k] > NEAR_HOME_RADIUS_M):
        k -= 1
    if k < 0:
        return n
    return min(k + 1 + SETTLE_FRAMES, n)


def decode_images_fast(table, col_name, n):
    col = table.column(col_name).combine_chunks()
    # HF `datasets` (imported transitively via lerobot) registers a pyarrow extension
    # type (Array3DExtensionType) that these image columns are tagged with in the
    # parquet file; once registered, reads wrap them in an ArrayExtensionArray whose
    # nested-list payload lives under `.storage`.
    if hasattr(col, "storage"):
        col = col.storage
    arr = np.asarray(col.flatten().flatten().flatten())
    side = int(round((arr.shape[0] / n / 3) ** 0.5))
    return arr.reshape(n, side, side, 3)


def stack_vector(table, col_name, n):
    col = table.column(col_name).combine_chunks()
    flat = np.asarray(col.flatten())
    return flat.reshape(n, -1).astype(np.float64)


def scalar_col(table, col_name):
    return np.asarray(table.column(col_name).combine_chunks()).astype(np.float64)


def compute_episode_stats_dict(table, cutoff):
    """table: full (untrimmed) pyarrow Table for one episode. Returns {parquet stats
    column name: numpy value}, computed over the first `cutoff` rows, matching the
    dataset's existing per-episode stats shape/axis convention (verified against the
    stored values for an untouched episode before writing anything)."""
    out = {}
    for feat in VECTOR_FEATURES:
        arr = stack_vector(table, feat, table.num_rows)[:cutoff]
        s = get_feature_stats(arr, axis=0, keepdims=False)
        for stat_name, value in s.items():
            out[f"stats/{feat}/{stat_name}"] = np.asarray(value)

    for feat in SCALAR_FEATURES:
        arr = scalar_col(table, feat)[:cutoff]
        s = get_feature_stats(arr, axis=0, keepdims=True)
        for stat_name, value in s.items():
            out[f"stats/{feat}/{stat_name}"] = np.asarray(value)

    for feat in IMAGE_FEATURES:
        imgs = decode_images_fast(table, feat, table.num_rows)[:cutoff]
        a = imgs.transpose(0, 3, 1, 2).astype(np.float64)
        s = get_feature_stats(a, axis=(0, 2, 3), keepdims=False)
        for stat_name, value in s.items():
            out[f"stats/{feat}/{stat_name}"] = np.asarray(value)

    return out


def cast_for_field(value, field_type):
    elem_type = field_type.value_type
    if pa.types.is_integer(elem_type):
        return np.round(np.asarray(value, dtype=np.float64)).astype(np.int64).tolist()
    return np.asarray(value, dtype=np.float64).tolist()


demo_dirs = sorted(
    p for p in DATASET_ROOT.iterdir()
    if p.is_dir() and p.name.startswith("demo_") and not p.name.startswith("_backup")
)

report_rows = []

for demo_dir in demo_dirs:
    print(f"\n=== {demo_dir.name} ===")
    ep_files = sorted((demo_dir / "meta" / "episodes").rglob("*.parquet"))
    assert len(ep_files) == 1, f"expected exactly one episodes file, got {ep_files}"
    ep_file = ep_files[0]
    episodes_table = pq.read_table(ep_file)
    episodes_df = episodes_table.to_pandas()

    per_episode_result = {}
    running_offset = 0
    for _, erow in episodes_df.sort_values("episode_index").iterrows():
        ep = int(erow["episode_index"])
        ci, fi = int(erow["data/chunk_index"]), int(erow["data/file_index"])
        data_file = demo_dir / "data" / f"chunk-{ci:03d}" / f"file-{fi:03d}.parquet"

        table = pq.read_table(data_file)
        assert set(table.column("episode_index").to_pylist()) == {ep}
        n_full = table.num_rows

        pos = stack_vector(table, "observation.leader_pose", n_full)[:, :3]
        t = scalar_col(table, "timestamp")
        cutoff = trim_cutoff(pos, t)

        stats = compute_episode_stats_dict(table, cutoff)

        sliced = table.slice(0, cutoff)
        new_index = pa.array(np.arange(running_offset, running_offset + cutoff), type=pa.int64())
        idx_field = sliced.schema.get_field_index("index")
        sliced = sliced.set_column(idx_field, "index", new_index)
        pq.write_table(sliced, data_file)

        per_episode_result[ep] = {
            "length": cutoff,
            "dataset_from_index": running_offset,
            "dataset_to_index": running_offset + cutoff,
            "stats": stats,
        }
        report_rows.append({
            "demo": demo_dir.name, "episode": ep, "n_full": n_full, "cutoff": cutoff,
            "dropped": n_full - cutoff,
        })
        running_offset += cutoff

    # Rewrite the episodes meta file, in the table's original row order.
    ep_indices_in_order = episodes_table.column("episode_index").to_pylist()

    def col_values(name):
        return [per_episode_result[ep][name] for ep in ep_indices_in_order]

    new_episodes = episodes_table
    for name in ["length", "dataset_from_index", "dataset_to_index"]:
        idx = new_episodes.schema.get_field_index(name)
        new_episodes = new_episodes.set_column(idx, name, pa.array(col_values(name), type=pa.int64()))

    stats_columns = [c for c in episodes_table.schema.names if c.startswith("stats/")]
    for col in stats_columns:
        field_type = episodes_table.schema.field(col).type
        values = [cast_for_field(per_episode_result[ep]["stats"][col], field_type) for ep in ep_indices_in_order]
        idx = new_episodes.schema.get_field_index(col)
        new_episodes = new_episodes.set_column(idx, col, pa.array(values, type=field_type))

    pq.write_table(new_episodes, ep_file)

    total_frames = sum(per_episode_result[ep]["length"] for ep in ep_indices_in_order)
    info_path = demo_dir / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["total_frames"] = int(total_frames)
    info_path.write_text(json.dumps(info, indent=4))

    print(f"updated {len(ep_indices_in_order)} episodes, total_frames {info['total_frames']}")

report = pd.DataFrame(report_rows)
report.to_csv(DATASET_ROOT / "trim_report.csv", index=False)
print(f"\nDone. Wrote {DATASET_ROOT / 'trim_report.csv'}")
print(report[["dropped"]].describe())
