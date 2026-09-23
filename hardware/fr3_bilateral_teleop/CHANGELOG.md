
# Change Log
 
## Forked as fr3_bilateral_teleop

- Renamed from `franka_ros2_teleop` and shipped as an independent package. See `NOTICE` for the
  base commit and the full list of changes. Entries below this one are the original package's
  history, including the changes made here before the rename.

## [Unreleased]

- [Feature] Added timing diagnostics to both bilateral controllers (`~/diagnostics/loop_period_us`
  on both, plus per-channel `~/diagnostics/*_channel_latency_us`) and a
  `verify_bilateral_timing.py` script to check the 4-channel bilateral loop actually runs at
  1 kHz and to plot a channel-latency histogram. See README "Verifying 1 kHz timing and channel
  latency".

- [Feature] Upgraded the bilateral controller to a 4-channel architecture: the leader's own
  sensed external torque is now transmitted to the follower as an explicit force-feedforward
  term (`force_feedforward_gains`), and the existing force reflection from follower to leader now
  has its own explicit gain (`force_reflection_gains`). Position coupling (`k_gains`/`d_gains`)
  can now be tuned purely for tracking, while the two force channels are tuned purely for feel.

## [Unreleased] - 2025-09-15

- [Hotfix] Fixed bug where the Franka Hand of the follower could not be commanded
 
## [0.1.0] - 2025-08-18

Initial release of franka_ros2_teleop package