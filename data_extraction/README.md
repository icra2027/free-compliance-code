# data_extraction/

Turns recorded bilateral demonstrations (LeRobot per-session dirs under
`../../dataset/`) into the dataset splits, calibration constants and compliance
labels that everything else consumes. Nothing here needs a GPU, ROS or torch;
`apply_trim.py` alone needs `lerobot` for its feature statistics.

| File | Role |
| --- | --- |
| `dataset_io.py` | Column-selective parquet readers for the per-session dirs (skips the image columns) |
| `panda_fk.py` | Panda/FR3 forward kinematics: `observation.state` (joints) -> Cartesian follower pose |
| `build_dataset_splits.py` | Session-level train/val/test split manifest (`dataset/franka_vla_multimodal/`) |
| `calibrate_tool_offset.py` | Fits the flange -> tool-tip offset, writes `tool_offset.npy` |
| `run_extraction_on_dataset.py` | Runs the §4.1 label extraction over every episode; per-episode report in `../reports/` |
| `fit_frozen_contact_frame.py` | Fits the frozen T1 board-normal frame, writes `contact_frame_t1.npy` |
| `apply_trim.py` | Trims each episode's trailing return-to-home segment and recomputes stats |
| `relabel_manner.py` | Renames a mislabeled manner word across a session's metadata |

The extraction algorithm itself is
`../hardware/fr3_bilateral_teleop/dataset_tools/labeling/extract_impedance_labels.py`.
It stays in that package because the rig-side rollout harness imports it and
colcon installs it, and `../tests/test_provenance.py` checks that
`../src/compliance_vla/` has not diverged from it. `run_extraction_on_dataset.py`
is the adapter that feeds it LeRobot episodes.

## Order of operations

Run from the release root (`compliance-vla/`):

```bash
python data_extraction/build_dataset_splits.py
python data_extraction/calibrate_tool_offset.py     # -> data_extraction/tool_offset.npy
python data_extraction/run_extraction_on_dataset.py # -> reports/extraction_per_episode.json
python data_extraction/fit_frozen_contact_frame.py  # -> data_extraction/contact_frame_t1.npy
```

`tool_offset.npy` and `contact_frame_t1.npy` are shipped, already fit on the
collected data. They are read from this directory by
`compliance_vla.policy` (training labels and diagnostics),
`../smolvla_policy/deploy_vla/` and
`../hardware/fr3_bilateral_teleop/scripts/run_pilot_rollout.py` (on the robot).
Refit them only after a change to the tool mount or the task setup.
