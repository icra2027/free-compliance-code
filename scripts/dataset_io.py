"""Lightweight readers for the franka_vla_multimodal per-session LeRobot dirs.

Deliberately avoids pandas.read_parquet's default full-row materialization:
the image columns (observation.images.scene_rgb/wrist_rgb) are raw uint8
arrays embedded in the parquet files and dominate their size (~3.8GB across
the 3 sessions for ~92 episodes), but none of the impedance-extraction /
adverb / benchmark analysis touches pixels. Reading with an explicit column
list via pyarrow keeps this fast and memory-light.
"""

import json
import glob
import os

import numpy as np
import pyarrow.parquet as pq

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_ROOT = os.path.join(REPO_ROOT, "dataset")

NON_IMAGE_COLUMNS = [
    "observation.state",
    "action",
    "observation.wrench.external_base",
    "observation.wrench.external_stiffness",
    "observation.leader_pose",
    "observation.velocity",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
]

SESSIONS = ["demo1", "demo3", "demo4"]

# Second collection batch (2026-09-02): 2-colour (red/blue only, matching the
# DEFAULT_DESCOPE_REFERENTS in diagnose_language_grounding.py), 2-operator, per-session-fixed
# manner word. See scripts/analyze_data_two_color.py for the cross-operator-adverb / colour-
# position-confound analysis of this batch.
TWO_COLOR_DATASET_ROOT = os.path.join(REPO_ROOT, "data_two_color")
TWO_COLOR_SESSIONS = [
    "demo_red_left", "demo_blue", "demo_blue_firm", "demo_red_firm",
    "demo_redB", "demo_blueB", "demo_blue_firmB", "demo_red_firmB",
]

# Third collection batch (2026-09-08 decision, not yet landed on disk as of this commit): full
# 2 (colour) x 2 (position) x 2 (manner) factorial, 12 episodes/cell, 96 total -- designed to
# fix both confounds language_grounding_issue_handoff.md's 2026-09-05 update found in
# data_two_color: colour was effectively fixed to one board position, and predictions collapsed
# onto whatever colour a given *manner* happened to be recorded with. Crossing colour with both
# position and manner independently means no single other factor can stand in for colour.
# Physical mark position is jittered a few cm across the 12 reps within a cell (confirmed at
# collection time) specifically so the model can't solve this via an 8-way (colour,
# position-label, manner) lookup table instead of actually parsing the visible ink colour --
# see scripts/analyze_data_factorial.py's position-confound check, which verifies this
# quantitatively rather than trusting it was done correctly.
#
# NOTE: session directory names and FACTORIAL_CONDITIONS below are a proposed convention
# (demo_<colour>_<position>_<manner>) written before the data existed locally -- rename this
# dict's keys (and FACTORIAL_SESSIONS, derived from it) to match whatever the actual collected
# folder names turn out to be; nothing else needs to change.
FACTORIAL_DATASET_ROOT = os.path.join(REPO_ROOT, "data_factorial_2c")
FACTORIAL_CONDITIONS = {
    "demo_blue_left_normal":  {"colour": "blue", "position": "left",  "manner": "normal"},
    "demo_blue_right_normal": {"colour": "blue", "position": "right", "manner": "normal"},
    "demo_red_left_firm":     {"colour": "red",  "position": "left",  "manner": "firm"},
    "demo_red_right_firm":    {"colour": "red",  "position": "right", "manner": "firm"},
    "demo_blue_left_firm":    {"colour": "blue", "position": "left",  "manner": "firm"},
    "demo_blue_right_firm":   {"colour": "blue", "position": "right", "manner": "firm"},
    "demo_red_right_normal":  {"colour": "red",  "position": "right", "manner": "normal"},
    "demo_red_left_normal":   {"colour": "red",  "position": "left",  "manner": "normal"},
}
FACTORIAL_SESSIONS = list(FACTORIAL_CONDITIONS)


def session_dir(session, dataset_root=None):
    return os.path.join(dataset_root or DATASET_ROOT, session)


def load_info(session, dataset_root=None):
    with open(os.path.join(session_dir(session, dataset_root), "meta", "info.json")) as f:
        return json.load(f)


def load_session_manifest(session, dataset_root=None):
    path = os.path.join(session_dir(session, dataset_root), "session_manifest.jsonl")
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_tasks(session, dataset_root=None):
    """task_index -> task string, from meta/tasks.parquet (index column carries the string)."""
    t = pq.read_table(os.path.join(session_dir(session, dataset_root), "meta", "tasks.parquet")).to_pandas()
    # tasks.parquet is indexed by the task string with a 'task_index' column.
    if "task_index" in t.columns and t.index.name != "task_index":
        return dict(zip(t.iloc[:, 0], t.index)) if False else {row["task_index"]: idx for idx, row in t.iterrows()}
    return {}


EPISODES_META_COLUMNS = [
    "episode_index", "tasks", "length", "data/chunk_index", "data/file_index",
]


def load_episodes_meta(session, columns=None, dataset_root=None):
    files = sorted(glob.glob(os.path.join(session_dir(session, dataset_root), "meta", "episodes", "chunk-*", "file-*.parquet")))
    cols = columns if columns is not None else EPISODES_META_COLUMNS
    tables = [pq.read_table(f, columns=cols).to_pandas() for f in files]
    import pandas as pd
    return pd.concat(tables, ignore_index=True)


def load_frames(session, columns=None, dataset_root=None):
    """Concatenate all data shards for a session, non-image columns only by default."""
    cols = columns if columns is not None else NON_IMAGE_COLUMNS
    files = sorted(glob.glob(os.path.join(session_dir(session, dataset_root), "data", "chunk-*", "file-*.parquet")))
    tables = [pq.read_table(f, columns=cols) for f in files]
    import pyarrow as pa
    return pa.concat_tables(tables).to_pandas()


def stack_col(df, name):
    return np.stack(df[name].to_numpy())


def parse_manner_from_task(task_text):
    for m in ("gently", "normally", "firmly"):
        if m in task_text:
            return m
    return None


def parse_referent_from_task(task_text):
    for c in ("red", "blue", "green", "black"):
        if c in task_text:
            return c
    return None
