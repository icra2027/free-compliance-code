#!/usr/bin/env python3
"""Relabel a mislabeled manner word in one or more data_two_color session dirs.

Operator mislabeling report (2026-09-02): operator A's two "gently" sessions
(demo_red_left, demo_blue) were actually "normally" -- confirmed by the user, not
inferred from the data itself (analyze_data_two_color.py's force-separation check can't
distinguish "operator said the wrong word" from "operator really did wipe gently"; both
would show internally-consistent, well-separated force distributions).

The manner/task string is stored in three places per session (checked directly, not
assumed -- data/chunk-*/file-*.parquet does NOT carry the string, only an int
`task_index`, so those ~24 per-session data shards do not need touching):

1. session_manifest.jsonl -- one JSON object per line, `manner` and `task` fields.
2. meta/tasks.parquet -- one row (this batch has exactly one task per session), the task
   string is the pandas index (stored as the `__index_level_0__` column), `task_index` is
   the int every data-shard row's own `task_index` column refers back to.
3. meta/episodes/**/*.parquet -- one row per episode, `tasks` is a list<string> column
   (LeRobot allows multiple task strings per episode; this batch always has exactly one).

`task_index` itself is deliberately left unchanged throughout -- this is a rename of what
task_index 0 *means* for the session, not the introduction of a second task, so nothing
downstream that joins on task_index breaks.

Edits (2) and (3) surgically via pyarrow (read Table, replace only the affected column's
values, write the same Table back) rather than round-tripping through pandas, so every
other column's dtype/shape and the file's pandas index metadata survive byte-for-byte.

Every touched file is copied to `<session>/_backup_<timestamp>/` before any write, so this
is reversible.

Usage:
    python3 data_extraction/relabel_manner.py --session demo_red_left demo_blue \\
        --old-manner gently --new-manner normally
    python3 data_extraction/relabel_manner.py --session demo_red_left demo_blue \\
        --old-manner gently --new-manner normally --dry-run
"""

import argparse
import glob
import json
import os
import shutil
import sys
from datetime import datetime

import pyarrow as pa
import pyarrow.parquet as pq

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(PROJECT_ROOT)
sys.path.insert(0, SCRIPT_DIR)

import dataset_io as dio  # noqa: E402


def backup_file(path, backup_root):
    rel = os.path.relpath(path, backup_root["session_dir"])
    dest = os.path.join(backup_root["dir"], rel)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copy2(path, dest)


def relabel_manifest(session_dir, old_manner, new_manner, backup_root, dry_run):
    path = os.path.join(session_dir, "session_manifest.jsonl")
    backup_file(path, backup_root)
    lines = open(path).read().splitlines()
    n_changed = 0
    new_lines = []
    for line in lines:
        if not line.strip():
            new_lines.append(line)
            continue
        entry = json.loads(line)
        if entry.get("manner") == old_manner:
            entry["manner"] = new_manner
            entry["task"] = entry.get("task", "").replace(old_manner, new_manner)
            n_changed += 1
        new_lines.append(json.dumps(entry))
    if not dry_run:
        with open(path, "w") as f:
            f.write("\n".join(new_lines) + "\n")
    return n_changed, len(lines)


def relabel_tasks_parquet(session_dir, old_manner, new_manner, backup_root, dry_run):
    path = os.path.join(session_dir, "meta", "tasks.parquet")
    backup_file(path, backup_root)
    table = pq.read_table(path)
    idx_col = table.column("__index_level_0__").to_pylist()
    n_changed = sum(old_manner in s for s in idx_col)
    new_idx_col = [s.replace(old_manner, new_manner) for s in idx_col]
    new_table = table.set_column(
        table.column_names.index("__index_level_0__"),
        "__index_level_0__",
        pa.array(new_idx_col, type=pa.string()),
    )
    if not dry_run:
        pq.write_table(new_table, path)
    return n_changed, len(idx_col)


def relabel_episodes_meta(session_dir, old_manner, new_manner, backup_root, dry_run):
    files = sorted(glob.glob(os.path.join(session_dir, "meta", "episodes", "chunk-*", "file-*.parquet")))
    total_changed, total_rows = 0, 0
    for f in files:
        backup_file(f, backup_root)
        table = pq.read_table(f)
        tasks_col = table.column("tasks").to_pylist()  # list[list[str]]
        n_changed = sum(any(old_manner in s for s in row) for row in tasks_col)
        new_tasks_col = [[s.replace(old_manner, new_manner) for s in row] for row in tasks_col]
        new_table = table.set_column(
            table.column_names.index("tasks"),
            "tasks",
            pa.array(new_tasks_col, type=table.schema.field("tasks").type),
        )
        if not dry_run:
            pq.write_table(new_table, f)
        total_changed += n_changed
        total_rows += len(tasks_col)
    return total_changed, total_rows


def relabel_session(session, old_manner, new_manner, dataset_root, dry_run):
    session_dir = dio.session_dir(session, dataset_root)
    if not os.path.isdir(session_dir):
        raise FileNotFoundError(session_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = {
        "session_dir": session_dir,
        "dir": os.path.join(session_dir, f"_backup_{timestamp}"),
    }
    if not dry_run:
        os.makedirs(backup_root["dir"], exist_ok=True)

    print(f"\n=== {session} ===")
    n, total = relabel_manifest(session_dir, old_manner, new_manner, backup_root, dry_run)
    print(f"  session_manifest.jsonl: {n}/{total} rows relabeled")

    n, total = relabel_tasks_parquet(session_dir, old_manner, new_manner, backup_root, dry_run)
    print(f"  meta/tasks.parquet:     {n}/{total} rows relabeled")

    n, total = relabel_episodes_meta(session_dir, old_manner, new_manner, backup_root, dry_run)
    print(f"  meta/episodes/**:       {n}/{total} rows relabeled")

    if dry_run:
        print("  (dry run -- nothing written, no backup created)")
    else:
        print(f"  backup -> {backup_root['dir']}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session", nargs="+", required=True)
    p.add_argument("--old-manner", required=True)
    p.add_argument("--new-manner", required=True)
    p.add_argument("--dataset-root", default=dio.TWO_COLOR_DATASET_ROOT)
    p.add_argument("--dry-run", action="store_true", help="report what would change, write nothing")
    args = p.parse_args()

    for session in args.session:
        relabel_session(session, args.old_manner, args.new_manner, args.dataset_root, args.dry_run)


if __name__ == "__main__":
    main()
