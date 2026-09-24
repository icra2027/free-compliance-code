# Franka joint states
```ros2 launch fr3_bilateral_teleop teleop.launch.py```

# RealSense RGB (wrist camera)
```ros2 launch realsense2_camera rs_launch.py pointcloud.enable:=true align_depth.enable:=true```

# Orbbec Femto Bolt (fixed third-person scene camera). Depth/IR are disabled here on purpose --
# this container's Femto Bolt depth engine crashes on launch (no working GL/EGL; see Day 2 notes
# in compliance-vla/tasks.md), and the recorder only ever consumes the color stream.
```ros2 launch orbbec_camera femto_bolt.launch.py enable_depth:=false enable_ir:=false```

The Femto Bolt's USB permissions are not persistent in this container (no udev daemon) and it has
been observed to drop off the USB bus and re-enumerate mid-session -- if `femto_bolt.launch.py`
fails with `usbEnumerator openUsbDevice failed! status:113`, or the recorder GUI's scene panel
border turns red with "STALE/DISCONNECTED", run the fix script (finds the current node by vendor
ID rather than assuming a fixed bus/device number, since re-enumeration changes it) and restart
the `orbbec_camera` launch before continuing -- do not keep recording through it:
```
sudo src/fr3_bilateral_teleop/scripts/fix_femto_bolt_usb_perms.sh   # run from /ros2_ws
```
If it keeps re-enumerating every few seconds even right after that script runs, that's not a
permissions problem anymore -- it's the separate, harder USB3 connection-stability issue Day 2
documented (try a different port, a shorter/rated-USB3 cable, or a powered hub).

# Recording a collection

**The mark colour and the wiping manner are fixed for a whole collection** and are given as
command-line flags, *before* `--ros-args`. One recorder launch is one session is one
`(colour, manner)` pair: every episode it records is labelled `"wipe the <colour> mark <manner>"`,
with no per-episode typing. The whiteboard only ever carries **two** marks, so `--colour` accepts
`red` or `blue` only (`--color` works too); `--manner` accepts `gently`/`normally`/`firmly`.

```
source install/setup.bash && ros2 run data_recorder record_lerobot \
  --colour red --manner gently \
  --ros-args -p dataset_root:=/home/robot/franka_ros2_ws/data/wiping \
  -p repo_id:=local/franka_vla_multimodal -p visualize_gui:=true \
  -p operator_id:=A -p episode_task:=T1_wiping
```

To cover the grid, relaunch once per `(colour, manner)` cell -- six launches for the full
2 colours x 3 manners, changing only `--colour`/`--manner` (and `session_id` if you are not
letting it auto-generate) between them:

```
--colour red  --manner gently      --colour blue --manner gently
--colour red  --manner normally    --colour blue --manner normally
--colour red  --manner firmly      --colour blue --manner firmly
```

Both flags are required. If either is missing the recorder exits before touching the dataset
rather than recording a session of unlabelled episodes; an invalid value is rejected by
`--help`-style argparse validation (`--colour green` is an error -- there is no green mark).
They can alternatively be supplied as ROS parameters (`-p colour:=red -p manner:=gently`) for
launch files; the command-line flags win when both are present.

Sanity-check the first line of the log before starting a batch -- it prints the exact text every
frame of the session will be labelled with:
```
[INFO] [data_recorder]: Fixed instruction for this session (every episode): 'wipe the red mark gently'
```
The recorder GUI's status panel shows the same `FIXED: red / gently` pair, drawn in the mark's own
colour, for the whole session.

Pair this with `fr3_bilateral_teleop`'s `live_force_band_display.py` (see that package's README) so
the operator has real-time force feedback against the target bands while recording -- that display
is intentionally a separate process and never writes into the dataset (the policy must not see the
target-band numbers).

Keyboard controls in the recorder GUI:
- `r`: start recording (starts immediately -- there is no per-episode instruction/manner prompt any more; both are fixed at launch)
- `s`: save successful episode -- first drives the follower back to its configured `start_joint_configuration` via `move_to_start_example_controller` while still recording (that return-to-home motion is appended to the same episode, so every saved episode ends at the same known pose instead of wherever the operator released the leader), then saves. Blocks the GUI until the move finishes or `reset_wait_timeout_sec` elapses. The leader is left alone (unlike `m` below) since the operator's hand is normally still on it at this point; if the move fails or times out, the episode is still saved with whatever frames were recorded, and an error is logged.
- `d`: discard current episode (no home move -- if you don't want the episode, there's no reason to relocate the arm for it)
- `n`: move to next episode index
- `m`: move robot to `start_joint_configuration` via `move_to_start_example_controller` -- manual, full reset (follower **and** leader if `reset_leader_enabled`), for use when nobody is holding the leader (e.g. before starting a session)
- `q`: quit

Optional reset parameters (defaults match teleop launch conventions):
- `reset_controller_namespace` (default: `/franka_teleop/follower`)
- `reset_target_controller` (default: `follower_controller`)
- `reset_leader_enabled` (default: `true`)
- `reset_leader_namespace` (default: `/franka_teleop/leader`)
- `reset_leader_target_controller` (default: `leader_controller`)
- `reset_move_to_start_controller` (default: `move_to_start_example_controller`)
- `reset_wait_timeout_sec` (default: `20.0`)

Additional recording parameters:
- `language_instruction` (default: empty string) -- an optional **template** overriding the default `"wipe the {colour} mark {manner}"`. The placeholders `{colour}` and `{manner}` are substituted with the session's fixed values, e.g. `language_instruction:="scrub the {colour} mark {manner} with the sponge"`. A bad placeholder is rejected at startup rather than mid-episode; a template that names neither placeholder is accepted but warned about, since the recorded text then never says which mark the episode is about (which is the one thing a referent-grounding policy trains on).
- `episode_task` (default: `sample_task`) -- a task-family label (e.g. `T1_wiping`). It no longer feeds the instruction text; it is recorded per episode in the session manifest as `task_family`.
- `pose_topic` (default: `/franka_teleop/follower/franka_robot_state_broadcaster/current_pose`) used to read end-effector pose for the action vector.
- `gripper_joint_topic` (default: `/franka_teleop/follower/joint_states`) used to read gripper opening for the action vector.
- `record_wrench_forces` (default: `true`) records both external wrench vectors from FR3 robot state into dataset fields `observation.wrench.external_base` and `observation.wrench.external_stiffness`.
- `robot_state_topic` (default: `/franka_teleop/follower/franka_robot_state_broadcaster/robot_state`) source of `O_F_ext_hat_K` and `K_F_ext_hat_K` wrench signals.

Scene camera (Orbbec Femto Bolt, fixed third-person view):
- `scene_rgb_topic` (default: `/camera/color/image_raw`) -- matches `orbbec_camera femto_bolt.launch.py`'s default `camera_name:=camera`. Recorded into dataset feature `observation.images.scene_rgb`.
- `scene_rgb_width` / `scene_rgb_height` (default: `224`/`224`) -- output resolution the driver's native frame is resized to.
- `use_scene_cv2_capture` (default: `false`) -- fallback path that reads the scene camera straight off its V4L2 node via `cv2.VideoCapture` (explicit `cv2.CAP_V4L2` backend, MJPG FOURCC requested to match the driver's own default format) instead of going through `orbbec_camera`/the ROS topic. Use this when the ROS driver itself won't open the device (`usbEnumerator openUsbDevice failed! status:113` -- the driver's own USB enumeration has been unreliable in this container; the kernel's UVC driver tends to be more tolerant of it). It bypasses the driver entirely and gets none of its exposure/white-balance handling. On init the recorder reads and logs a verification frame's shape before recording starts, so a wrong device is caught immediately rather than mid-session.
- `scene_cv2_device` (default: `0`) / `scene_cv2_device_path` (default: empty, e.g. `/dev/video6`) -- which V4L2 node to open; the path form takes priority. **The Femto Bolt exposes several `/dev/video*` nodes** (color's data + metadata pair, depth/IR's data + metadata pair) and the right numeric index is not guaranteed stable across replugs. Find it with:
  ```bash
  for v in /dev/video*; do echo "$v: $(cat /sys/class/video4linux/$(basename $v)/name 2>/dev/null)"; done
  ```
  all four will report the same product name, so name alone doesn't disambiguate -- confirm which is actually color by resolution: the color data node opens and reads real ~1280x720 frames; its metadata sibling fails to open by index at all; the depth/IR pair reads a very different, non-1280x720 shape (or fails). On this rig it was `/dev/video6`, e.g.:
  ```bash
  ros2 run data_recorder record_lerobot --ros-args -p use_scene_cv2_capture:=true -p scene_cv2_device_path:=/dev/video6 ...
  ```
  If it opens the wrong node, `_init_scene_cv2_capture`'s startup verification read will raise a clear error rather than silently recording garbage. If you hit "Permission denied" opening any `/dev/video*` node, run `fr3_bilateral_teleop/scripts/fix_femto_bolt_usb_perms.sh` -- it fixes both the raw USB node (for the ROS driver) and every `/dev/video*` node belonging to the Femto Bolt (for this cv2 path) in one pass.
- `scene_camera_timeout_sec` (default: `1.0`) -- if no scene frame arrives within this window the GUI's scene panel is outlined in red and marked "STALE/DISCONNECTED", and a throttled warning is logged. Works the same way in both ROS-topic and cv2-capture mode. Added because the Femto Bolt has been observed to drop off the USB bus mid-session (see `compliance-vla/tasks.md`, Day 2) -- this is a liveness check independent of the frame synchronizer, so a disconnected camera doesn't just silently stop appearing in the GUI.

Day 6 (referent/adverb/session metadata):
- `operator_id` (default: `A`).
- `colour` / `manner` -- the ROS-parameter form of the `--colour`/`--manner` flags above (see "Recording a collection"). Both are required, fixed per session, and validated at startup: `colour` must be `red` or `blue` (the only two marks on the board), `manner` must be `gently`/`normally`/`firmly` per proposal §5's target bands.
- `session_id` (default: empty -> auto-generated as the process start time, `YYYYMMDD_HHMMSS`) -- one recorder launch is one session.
- Every successfully saved episode appends one line to `<dataset_root>/session_manifest.jsonl` (`{timestamp, session_id, operator_id, colour, manner, task_family, episode_index, task, frame_count, repo_id}`). `colour`/`manner` are constant within a session by construction, which is the point: the referent of each episode is recoverable directly from the manifest without re-parsing it out of the task string. This is Day 6's "session-level split scaffolding" background task: it doesn't build the Day 10 train/val/test splitter itself, but records the session each episode belongs to while that's still known, so the splitter can group by `session_id` later without reconstructing it from file timestamps.

# Replaying a recorded episode

`replay_lerobot_episode` drives the real follower arm through one recorded episode's joint
trajectory (`observation.state`, not `action` -- the latter is a Cartesian EE pose + gripper
tuple, not a joint target). It reuses the follower's own already-active `follower_controller`
rather than introducing a separate playback controller: that controller is a joint-space
impedance tracker that follows whatever `sensor_msgs/JointState.position` it last received on
its `input_topic` (normally the leader's own `measured_joint_states`, read by index with no name
matching -- see `fr3_bilateral_teleop/src/teleop_follower_controller.cpp`), so this script just
queries that `input_topic` and republishes the episode onto it, interpolated up to
`--publish-rate` (default 200 Hz, well above the dataset's own recording fps) so the impedance
controller sees a smooth trajectory rather than a stair-step of sparse targets. For the
replay's duration this makes the script look like "the leader" to the follower.

Requires the follower's teleop stack already running (`teleop.launch.py`, or at least the
follower half of it) with `follower_controller` active -- this does not launch or activate
anything itself.

```bash
# See what's in a dataset before picking an episode -- dataset-local, compacted indices, NOT
# session_manifest.jsonl's raw episode_index (which includes discarded takes):
ros2 run data_recorder replay_lerobot_episode \
  --dataset-root /home/robot/franka_ros2_ws/data/wiping --list-episodes

ros2 run data_recorder replay_lerobot_episode \
  --dataset-root /home/robot/franka_ros2_ws/data/wiping --episode 3
```

Because this moves a real robot arm through recorded motion with nobody's hand on either side,
several checks run before anything moves, all logged clearly if they fail:
- `follower_controller` must already be loaded **and active** on the follower's
  `controller_manager`.
- Nothing else may already be publishing on its `input_topic` -- a live real leader (or another
  replay) would race with this script for control of the follower. Refuses by default; `--force`
  overrides, with a loud warning.
- The follower's *current* joint position must already be close to the episode's first frame
  (`--max-start-offset-rad`, default `0.5`) -- an impedance controller snapping to a large step
  error is exactly the sudden motion this exists to prevent. Move the follower closer first
  (e.g. `record_lerobot`'s `m` key) if this fails.
- A mandatory `replay` confirmation prompt before any motion starts, showing episode/task/
  duration -- skippable with `--yes`/`-y` (required in a non-interactive session, since there is
  no prompt to answer there).

Other options:
- `--speed` (default `1.0`) -- playback speed multiplier. Values above `1.0` exceed the peak
  joint velocity actually demonstrated in the episode and are logged loudly, not blocked.
- `--replay-gripper` / `--no-replay-gripper` (default: on if the dataset has an `action`
  feature) -- mirrors `fr3_bilateral_teleop`'s `teleop_gripper_node.cpp` bang-bang open/close
  policy exactly (same hysteresis thresholds and `Move`/`Grasp` goal fields), just driven from
  the episode's recorded gripper width instead of a live leader gripper. Uses the episode's own
  observed peak width as the "fully open" target rather than a fixed hardware default, so it
  still works correctly for an episode whose gripper never approaches Franka Hand's 0.07 m
  default. Silently unavailable (not an error) if `franka_gripper`'s action servers aren't up,
  e.g. the follower was launched with `load_gripper:=false`.
- `--namespace` (default: `franka_teleop/follower`) / `--controller-name` (default:
  `follower_controller`) -- match `teleop.launch.py`'s own conventions; override for a
  differently-named pair.
- `--dry-run` -- run every check above and print the plan, without publishing or moving
  anything.

Once replay finishes (or is interrupted), no further commands are published -- exactly as if a
real leader had disconnected, `follower_controller` falls back to gravity compensation
`input_topic_timeout` after the last message unless teleop or another script resumes publishing
to its `input_topic`.