"""Build the LeRobot dataset with session-level splits, and record the
train/val/test session counts.

What's collected so far (2026-08-20) is 3 sessions -- demo1 (operator A,
20260819_133337), demo3 (operator B, 20260819_152423), demo4 (operator A,
20260819_160500), 92 episodes total, all under the shared repo_id
`local/franka_vla_multimodal` per session_manifest.jsonl. Schema (meta/info.json
`features`) is identical across all three, so they are genuinely one dataset
split across directories, not three different datasets.

**Manifest-based merge, not a physical re-chunk.** The image columns
(observation.images.scene_rgb/wrist_rgb) are raw uint8 arrays embedded in the
parquet shards, ~3.8GB combined for these 92 episodes; physically concatenating
and re-chunking that data into a new merged directory would duplicate it for no
analysis benefit (none of the extraction/benchmark analysis touches pixels,
and LeRobot's own dataset abstractions resolve data by chunk/file index already
stored per-episode, so a manifest that points at the existing per-session
shards is sufficient to load any global episode without copying bytes). This
writes `dataset/franka_vla_multimodal/{splits.json,episode_manifest.csv,
info.json}` as that manifest; the raw shards stay in place under
`dataset/demo{1,3,4}/`.

**Split assignment, honestly: 3 sessions is a train/val/test count of (1,1,1),**
nowhere near the eventual target of dozens of sessions -- flagged
explicitly rather than smoothed over. Given only 3, assignment is deliberate
rather than random:
  - test  = demo3 (operator B) -- held out entirely. This also gives the
    cross-operator held-out test (train on A, evaluate on B) for free, rather than wasting the one non-A session on an arbitrary
    train/val role.
  - val   = demo1 (operator A, 28 episodes, the smaller of the two A sessions)
  - train = demo4 (operator A, 32 episodes)
Revisit this assignment once more sessions exist -- with only 3, every session
plays an outsized role in whichever split it lands in.
"""

import csv
import json
import os

import dataset_io as dio

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUT_DIR = os.path.join(dio.DATASET_ROOT, "franka_vla_multimodal")

SPLIT_ASSIGNMENT = {
    "train": ["demo4"],
    "val": ["demo1"],
    "test": ["demo3"],
}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    session_to_split = {s: split for split, sessions in SPLIT_ASSIGNMENT.items() for s in sessions}
    infos = {s: dio.load_info(s) for s in dio.SESSIONS}
    assert all(infos[s]["features"] == infos[dio.SESSIONS[0]]["features"] for s in dio.SESSIONS), \
        "sessions have divergent schemas -- cannot merge as one dataset"

    manifest_rows = []
    global_idx = 0
    canonical_tasks = {}  # task string -> canonical task_index, decoupled from each
                            # session's own local task_index (those differ per session --
                            # see analyze_adverb_separation.py's docstring on why local
                            # per-session integer spaces must never be zipped together).

    split_stats = {split: {"n_sessions": 0, "n_episodes": 0, "n_frames": 0, "sessions": []}
                   for split in SPLIT_ASSIGNMENT}

    for split, sessions in SPLIT_ASSIGNMENT.items():
        for session in sessions:
            manifest = {r["episode_index"]: r for r in dio.load_session_manifest(session)}
            episodes_meta = dio.load_episodes_meta(session)
            n_frames_session = 0
            for _, row in episodes_meta.sort_values("episode_index").iterrows():
                local_idx = int(row["episode_index"])
                task_text = row["tasks"][0] if len(row["tasks"]) else ""
                length = int(row["length"])
                if task_text not in canonical_tasks:
                    canonical_tasks[task_text] = len(canonical_tasks)
                manifest_rows.append({
                    "global_episode_index": global_idx,
                    "split": split,
                    "session": session,
                    "local_episode_index": local_idx,
                    "operator_id": manifest.get(local_idx, {}).get("operator_id"),
                    "session_id": manifest.get(local_idx, {}).get("session_id"),
                    "task": task_text,
                    "task_index": canonical_tasks[task_text],
                    "manner": dio.parse_manner_from_task(task_text),
                    "referent": dio.parse_referent_from_task(task_text),
                    "length": length,
                    "data_chunk_index": int(row["data/chunk_index"]),
                    "data_file_index": int(row["data/file_index"]),
                })
                global_idx += 1
                n_frames_session += length

            split_stats[split]["n_sessions"] += 1
            split_stats[split]["n_episodes"] += len(episodes_meta)
            split_stats[split]["n_frames"] += n_frames_session
            split_stats[split]["sessions"].append({
                "session": session, "n_episodes": int(len(episodes_meta)),
                "n_frames": n_frames_session,
                "operator_id": next(iter(manifest.values()))["operator_id"] if manifest else None,
            })

    manifest_path = os.path.join(OUT_DIR, "episode_manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    splits_path = os.path.join(OUT_DIR, "splits.json")
    with open(splits_path, "w") as f:
        json.dump(split_stats, f, indent=2)

    ref_info = infos[dio.SESSIONS[0]]
    merged_info = {
        "codebase_version": ref_info["codebase_version"],
        "robot_type": ref_info["robot_type"],
        "repo_id": "local/franka_vla_multimodal",
        "total_episodes": global_idx,
        "total_frames": sum(s["n_frames"] for s in split_stats.values()),
        "total_tasks": len(canonical_tasks),
        "fps": ref_info["fps"],
        "features": ref_info["features"],
        "splits_by_session": {
            split: {"sessions": [s["session"] for s in stats["sessions"]],
                    "n_sessions": stats["n_sessions"],
                    "n_episodes": stats["n_episodes"],
                    "n_frames": stats["n_frames"]}
            for split, stats in split_stats.items()
        },
        "note": (
            "Manifest-based merge over dataset/demo{1,3,4}/ -- raw data/video "
            "shards were NOT copied here; resolve (session, local_episode_index, "
            "data_chunk_index, data_file_index) from episode_manifest.csv back "
            "to dataset/<session>/data/chunk-*/file-*.parquet to load frames."
        ),
    }
    info_path = os.path.join(OUT_DIR, "info.json")
    with open(info_path, "w") as f:
        json.dump(merged_info, f, indent=2)

    print("Session-level split (train/val/test session counts):")
    for split, stats in split_stats.items():
        session_names = ", ".join(s["session"] for s in stats["sessions"])
        print(f"  {split:5s}: {stats['n_sessions']} session(s) [{session_names}], "
              f"{stats['n_episodes']} episodes, {stats['n_frames']} frames")
    print(f"\nTotal: {global_idx} episodes, {merged_info['total_frames']} frames, "
          f"{len(canonical_tasks)} canonical tasks")
    print(f"\nWrote {manifest_path}")
    print(f"Wrote {splits_path}")
    print(f"Wrote {info_path}")


if __name__ == "__main__":
    main()
