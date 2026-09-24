# fr3_bilateral_teleop — four-channel bilateral teleoperation and calibration for Franka FR3

> Derived from [franka_ros2_teleop](https://github.com/frankarobotics/franka_ros2_teleop)
> (Apache 2.0, Franka Robotics GmbH) at commit `69ade0e3e3a344c277f0c159a37ff8bd95c30636`,
> renamed so it can be installed alongside the original. See `NOTICE` for what was changed.
> Everything the original package did is still here; the additions are the 4-channel
> bilateral controllers, payload identification, sensorless wrench estimation, compliance
> label extraction, and the calibration/validation tooling described below.

**fr3_bilateral_teleop** is a ROS 2 package that demonstrates the teleoperation of a **Franka FR3** robot with another **Franka FR3** robot as input device.
Due to the robot's force-torque sensors we can send contact forces measured by the follower to the leader robot.
The leader robot will then replay the forces so the teleoperator can feel the contacts as well.

## Bilateral control architecture (4-channel)

The teleoperation controllers implement a **4-channel bilateral control** scheme: position and
force are each transmitted explicitly, in both directions, and each direction has its own
independently tunable gain. This decouples *tracking* from *feel*, so one gain no longer has to
compromise between the two:

| # | Channel | Direction | Signal | Gain(s) | Tunes |
|---|---|---|---|---|---|
| 1 | Position coupling | leader → follower | leader's measured joint position/velocity | `k_gains` / `d_gains` (follower) | Tracking accuracy: how tightly the follower's motion follows the leader |
| 2 | Force reflection | follower → leader | follower's sensed external (contact) torque | `force_reflection_gains` (leader) | Feel: how strongly the operator perceives the follower's contact forces |
| 3 | Force feedforward | leader → follower | leader's own sensed external (operator-applied) torque | `force_feedforward_gains` (follower) | Feel: how directly the operator's applied force acts on the follower, without needing to raise the position gains |

Channel 1 alone would make `k_gains`/`d_gains` the single knob governing both how well the
follower tracks the leader *and*, indirectly, how "stiff" or "soft" the follower feels when it
contacts the environment. Adding channels 2 and 3 as explicit, separately-gained torque terms means
`k_gains`/`d_gains` can be tuned purely for tracking, while `force_reflection_gains` and
`force_feedforward_gains` can be tuned purely for feel.

Both force channels reuse the same `franka_robot_state_broadcaster/external_joint_torques` topic
that each robot (leader and follower) already publishes: the leader controller subscribes to the
follower's external torques (channel 2), and the follower controller subscribes to the leader's own
external torques (channel 3). Setting a `force_*_gains` parameter to all zeros disables that
channel.

To run the teleoperation, you need to have at least two **Franka FR3** robots with the **FCI (Franka Control Interface)** feature.
One will be the leader and the other will be the follower.
The leader can be handguided and the follower will match the leaders position.
The forces encountered by the follower can be felt by the person hand-guiding the leader.

> [!CAUTION] 
> Avoid hand-guiding the follower, as it might lead to abrupt movements of the leader!

### Verifying 1 kHz timing and channel latency

`controller_manager.update_rate` is configured to 1000 Hz ([config/teleop_controllers.yaml](config/teleop_controllers.yaml)),
but the configured rate is only a request: real-time scheduling issues, a non-PREEMPT_RT kernel,
or network jitter between the leader and follower hosts can silently degrade it. Both controllers
publish lightweight timing diagnostics (via `realtime_tools::RealtimePublisher`, so they never
block the control loop) that let you check the *actual* rate and per-channel latency rather than
assuming the configured one:

| Topic (relative to the controller node) | Meaning |
|---|---|
| `~/diagnostics/loop_period_us` | The `update()` period actually measured by `controller_manager`, in microseconds. Published by both `leader_controller` and `follower_controller`. |
| `~/diagnostics/position_channel_latency_us` | Age of the leader's position message (channel 1) when consumed by the follower. Published by `follower_controller`. |
| `~/diagnostics/force_reflection_channel_latency_us` | Age of the follower's contact-torque message (channel 2) when consumed by the leader. Published by `leader_controller`. |
| `~/diagnostics/force_feedforward_channel_latency_us` | Age of the leader's own torque message (channel 3) when consumed by the follower. Published by `follower_controller`. |

The channel-latency numbers are `now - header.stamp`, computed on the *consuming* robot, so they
are only meaningful if the leader and follower hosts' clocks are synchronized (e.g. via `chrony` or
PTP) -- verify and report that separately before trusting them.

`scripts/verify_bilateral_timing.py` subscribes to these topics for a fixed duration, checks the
achieved loop rate and channel latencies against configurable thresholds, and writes a latency
histogram, a CSV of raw samples, and a JSON report:

```bash
ros2 run fr3_bilateral_teleop verify_bilateral_timing.py \
  --leader-namespace leader --follower-namespace follower \
  --duration 30 --output-dir bilateral_timing_report
```

It exits non-zero if the achieved rate falls outside `--rate-tolerance-pct` of `--target-rate-hz`
(default: 1000 Hz +/- 2%) or if p99 channel latency exceeds `--latency-warn-ms` (default: 2 ms).
Run `ros2 run fr3_bilateral_teleop verify_bilateral_timing.py --help` for all options.

### Automatic free-space re-zero on every session

`teleop_coordinator` already moves both arms to the configured
`start_joint_configuration` (`move_to_start_example_controller`, defaults to Franka's standard
"ready" pose -- see `start_joint_configuration` under "Configuration options" below to change it)
before activating the real teleoperation controllers. It now also uses that same, unskippable
startup path to re-zero the
sensorless wrench estimate: once both arms report they've reached the pose, it holds for 1 s to
let residual motion settle, then samples each arm's
`franka_robot_state_broadcaster/external_wrench_in_base_frame` for another 1 s. In free space the
true external wrench should be ~0, so whatever is measured there is bias -- friction, payload
mis-specification, thermal drift -- that would otherwise show up mid-session as a spurious
stiffness trend.

The per-arm bias is written to `<output_dir>/<namespace>_rezero.json` (the canonical file, always
overwritten) and archived at `<output_dir>/history/<namespace>_<unix_time>.json` (kept, so
session-to-session drift stays inspectable). `output_dir` defaults to
`/tmp/franka_teleop_rezero` and can be overridden via the `TELEOP_REZERO_OUTPUT_DIR` environment
variable.

If no wrench messages are received during the sampling window (e.g.
`franka_robot_state_broadcaster` isn't running for an arm), `teleop_coordinator` logs a fatal error
and exits without activating the leader/follower controllers, instead of silently starting a
session with an unmeasured bias.

### Payload calibration (identifying an attached tool without weighing it)

The sensorless wrench estimate is only as good as the mass/COM/inertia the robot thinks it's
carrying (`set_load`) -- a mis-declared payload is a *configuration-dependent* bias (it changes
with the arm's pose, unlike the flat bias the re-zero step above corrects), and the center of
mass in particular is the term most often gotten wrong by hand-measurement. `calibrate_payload.py`
identifies mass, center of mass, and inertia for whatever is rigidly attached to the flange (e.g.
a 3D-printed tool mount screwed onto the Franka Hand) directly from the arm's own joint-torque
sensing, with no scale, calipers, or CAD model required.

**Precondition:** run this *before* ever calling `set_load` for the new attachment (e.g. right
after FCI connects, or after explicitly zeroing any previous custom load). The method works by
comparing measured joint torque against `franka::Model`'s own prediction *for whatever load is
currently configured* -- if that already includes some of the new attachment, the residual this
script attributes to the attachment will be wrong.

**How it works:**

1. **Bring up the follower alone** (no teleop pairing) with `PayloadModelBroadcaster`, a new
   controller that publishes `franka::Model`'s gravity vector, Coriolis vector, mass matrix, and
   flange Jacobian/pose -- all evaluated at the currently-configured load -- as one atomic
   snapshot (`~/payload_model_broadcaster/model_snapshot`):
   ```bash
   ros2 launch fr3_bilateral_teleop calibrate_payload.launch.py namespace:=follower
   ```
2. **Run the calibration**, in a second terminal, once the launch is up:
   ```bash
   ros2 run fr3_bilateral_teleop calibrate_payload.py --namespace follower
   ```
   This sweeps 15 static joint-space waypoints (`config/payload_calibration_waypoints.yaml`; small
   perturbations of the same fixed pose already used for the free-space re-zero, retargeting
   `move_to_start_example_controller` between them without a full reconfigure cycle, at a slower
   `speed_factor` (0.1, vs. the shared 0.2 default) since this repeatedly re-triggers the motion
   rather than moving once per session). Every consecutive waypoint pair differs in exactly one
   joint -- routed back through the base pose between each joint's +/- offset -- because moving
   two joints at once in a single synchronized transit reliably tripped a `joint_velocity_violation`
   reflex on real hardware during testing; single-joint transits did not. Takes ~2-3 minutes. Then:
   - **Mass + center of mass**, from the static holds only. At each pose `dq = ddq = 0`, so the
     residual between measured torque and the model's gravity prediction is explained entirely by
     the attachment's gravity torque, which is linear in `(m, m·cx, m·cy, m·cz)` given the flange
     Jacobian and orientation at that pose -- the same "several poses resolve what one pose can't"
     idea as the paper's own equilibrium/stiffness identifiability argument, applied to a
     different pair of unknowns. This is the well-conditioned, low-risk half of the calibration,
     and the one that matters most (COM is the dominant bias term).
   - **Full inertia tensor**, from the point-to-point transits *between* waypoints (reusing the
     motion that already happens getting from one pose to the next, not a separately-engineered
     excitation trajectory). This stage is meaningfully less certain than the first: acceleration
     is numerically differentiated from logged velocity (noisy), and the attachment's own
     Coriolis/centrifugal coupling is neglected (a reasonable approximation for a small, light,
     near-flange part, but an approximation nonetheless).
3. **Quality gates, not blind trust.** Both fits are checked before being written anywhere: mass
   and COM must be positive/within sane bounds and the regression well-conditioned, or the script
   refuses to write a payload file at all (rather than writing a bad one). The inertia fit is
   additionally checked for physical validity (non-negative eigenvalues) and plausible magnitude;
   if it fails, the script falls back to treating the attachment as a solid sphere of the fitted
   mass (a documented, crude placeholder, not a measurement) rather than propagating a garbage
   inertia into `set_load`.
4. **Output:** `<output_dir>/<namespace>_payload.yaml` (canonical, read by `teleop.launch.py`) plus
   a timestamped copy under `history/`, so re-running calibration after a hardware change doesn't
   erase the ability to compare against the last one. `output_dir` defaults to
   `/tmp/franka_teleop_payload`, overridable via `TELEOP_PAYLOAD_OUTPUT_DIR`.

**Consumption:** `teleop.launch.py` calls `set_load` on the **follower only** (the tool is assumed
to be on the follower's gripper, not the hand-guided leader's) at startup, reading
`<namespace>_payload.yaml`. Unlike the re-zero step, a missing or failed calibration is a loud
warning, not a launch failure -- this is a per-hardware-configuration calibration (redo it when
the attached tool changes), not a per-session one, so gating every teleop launch on it would be
the wrong failure mode. Missing/failed calibration falls back to a zero/no-extra-load `set_load`
call.

**Safety before running on real hardware:** the default waypoints were chosen for joint-limit
margin and single-joint transits only (nearest limit is +/-0.45 rad away), not for clearance
against your specific table/board/fixture layout. Jog through them at low speed (e.g. via Desk) at
least once before letting the script drive through them unattended. `calibrate_payload.launch.py`
applies the same collision-behavior thresholds as normal teleop.

**Real-hardware findings that shaped the above (kept here so they aren't rediscovered the hard
way):** three separate joint reflexes were tripped while developing this against real hardware,
each traced to a specific bug rather than the arm/setup itself: (1) `move_to_start_example_controller`
was calling `set_parameter()` -- not real-time safe -- from inside its control loop on every cycle
once finished, instead of once; harmless for a single-use session-start move, but held for
seconds across many retargeted waypoints it could stall the control loop. (2) its velocity-damping
term (`dq_filtered_`) was only ever zeroed once at controller load time, not on each retarget, so
it fed several-seconds-stale velocity data into the very first control cycle after each
reactivation -- read by the robot as a torque discontinuity right as `franka_hardware` restarts
the torque interface on every controller activation change (confirmed by reading
`franka_hardware_interface.cpp`'s `perform_command_mode_switch`). (3) the original waypoint set
jumped directly between different joints' offsets, requiring two joints to move simultaneously in
one synchronized transit -- fine on paper, but reliably tripped a velocity reflex on this rig,
while every single-joint-only transit did not. All three are fixed as of this writing; if a fourth
distinct fault type shows up, treat it as a new finding rather than assuming it's one of these.

**A fourth, different-in-kind finding (10 Aug): a silent sign bug in the mass/COM regression, not
a reflex.** `GRAVITY_VEC` in `calibrate_payload.py` was `[0, 0, -9.81]` -- the textbook-looking
downward convention -- but the opposite of how `franka::Model::gravity()` actually responds to
`set_load` on this rig. This didn't trip any fault; it silently flipped the sign of the fitted
mass term on every run (the fitted center-of-mass was unaffected, since it's computed as a ratio
`theta[1:4]/mass` and the sign cancels out of it). The telltale symptom: `total_mass` scaling with
whatever `--preload-mass` was chosen instead of converging to a constant value across repeated
runs, no matter how good the fit's own `rank`/`condition_number` looked -- that combination
(reproducible, but preload-dependent) is what pointed at a sign bug rather than noise or a loose
attachment, which would instead show up as *poor* repeatability at a fixed preload. If you ever
see that pattern again (e.g. after a libfranka/SDK upgrade changes this convention back), verify
with `scripts/diagnose_gravity_preload.py` before trusting any calibration number -- it calls
`set_load` with a known mass delta on a **stationary** arm (no waypoint sweep, no reflex risk) and
compares the actual change in the published `gravity` field against the analytically-predicted
one; they should agree to within ~1%, with no motion or regression involved to introduce ambiguity
about where a mismatch is coming from.

**Separately: a robot fault during controller (re)activation used to crash the entire driver, not
just fail the activation.** If a reflex trips (or any other fault `automaticErrorRecovery()`'s one
built-in retry can't clear) right as a controller is being (re)activated -- exactly what repeated
waypoint retargeting does many times per calibration run -- `franka_hardware`'s
`perform_command_mode_switch()` used to let the resulting `franka::ControlException` propagate
uncaught, taking down the entire `ros2_control_node` process (`terminate called after throwing an
instance of 'franka::ControlException'`). Fixed (10 Aug) by catching it there and returning a
failed activation instead, so the driver survives and `controller_manager` reports a normal (if
noisy) rejected-switch error. **This does not mean the fault itself is cleared** -- you still have
to clear it on Desk before retrying, and if an older build had already crashed the process, you
still need to relaunch it.

**Limitations to carry into the paper if this calibration underpins reported numbers:** the
inertia fit is approximate (see above) and has no equivalent of the mass/COM fit's clean
identifiability argument; report it as such rather than as a measurement on the same footing as
the mass/COM fit.

### Residual bias model (free-space sweep -> f_bias(q, q̇))

Payload identification and per-session re-zero (above) remove a *fixed* bias and a *pose-independent
constant*, respectively. What's left is the part that actually varies with configuration and
velocity -- joint friction, small payload-model residuals -- which is expected to be worth
roughly another 2x on the effective noise floor. Two scripts, run in sequence, no ROS
dependency for the second one:

1. **`collect_free_space_sweep.py`** drives the follower through
   `config/free_space_sweep_waypoints.yaml` (a wider, single-joint-transit waypoint set than
   payload calibration's -- chosen for Cartesian workspace coverage, not just wrist-orientation
   diversity) at each of several `speed_factor`s, logging `(q, q̇, external_wrench_in_base_frame)`
   throughout both the transits (`q̇ != 0`) and the static holds (`q̇ ~= 0`). Since nothing is
   touched anywhere in this sweep, every non-zero sample of the already-payload-corrected wrench
   estimate **is** the bias -- no separate ground truth channel needed.
   ```bash
   ros2 run fr3_bilateral_teleop collect_free_space_sweep.py --namespace franka_teleop/follower
   ```
   **Precondition:** run this with the follower's normal payload already configured via
   `set_load` -- this fits the *residual after* payload correction, not the payload itself, and
   running it against an unconfigured (zero) load means the arm is moving with a real,
   uncompensated payload attached, which is a direct route to the same `joint_velocity_violation`
   reflex described above (confirmed on real hardware, 10 Aug: bringing the follower up via
   `calibrate_payload.launch.py` -- which only spawns `move_to_start_example_controller` +
   `payload_model_broadcaster` and never calls `set_load` itself -- then running this script
   directly reproduced exactly that fault on the very first transit). Three ways to satisfy the
   precondition:
   - Bring the follower up via `teleop.launch.py` instead, which calls `set_load` from
     `<namespace>_payload.yaml` automatically at startup; or
   - Run `calibrate_payload.py` in the same session first -- it applies its own preload and then
     the final fitted `set_load` as part of running, so by the time it finishes the follower is
     already correctly configured; or
   - If you already have a good `<namespace>_payload.yaml` from a previous session and don't want
     to redo the sweep, apply it manually before running this script, using the `set_load` block
     from that file:
     ```bash
     ros2 service call /<namespace>/service_server/set_load franka_msgs/srv/SetLoad \
       "{mass: <mass>, center_of_mass: [<cx>, <cy>, <cz>], load_inertia: [<9 values, column-major>]}"
     ```
     ```
     ros2 service call /franka_teleop/follower/service_server/set_load franka_msgs/srv/SetLoad "{mass: 0.26990907489357263, center_of_mass: [-0.037612488722482254, 0.04752582895792784, 0.05584386477830899], load_inertia: [9.716726696168616e-05, 0.0, 0.0, 0.0, 9.716726696168616e-05, 0.0, 0.0, 0.0, 9.716726696168616e-05]}"
    ```

   **Safety:** this sweeps a substantially larger volume than payload calibration; jog through the
   waypoint file at low speed first and confirm the T1/T2 fixtures are either not yet mounted or
   clear of the envelope.
2. **`fit_residual_bias.py`** (offline, numpy-only -- no torch/sklearn in this environment) fits a
   random-Fourier-feature ridge regressor per wrench axis -- the finite-dimensional, closed-form
   approximation of GP kernel-ridge regression, i.e. one of the two model classes considered
   ("small MLP or GP") without a new dependency. Splits are by **session** (one sweep pass),
   never by row, matching this project's session-level-split convention elsewhere.
   ```bash
   python3 fit_residual_bias.py --input sweep1.csv sweep2.csv --output-dir /tmp/foo
   ```
   Writes `residual_bias_model.npz` (weights + normalization stats, reloadable via
   `RFFRidgeBiasModel.load`) and `residual_bias_report.json` (per-axis baseline vs. fitted RMS on
   held-out sessions -- this is the number to check against the ~2x target). Run
   `python3 fit_residual_bias.py --self-test` to sanity-check the fitting/validation pipeline
   itself against synthetic data with a known `f_bias` -- no hardware or logged CSV required; this
   is what to run first if the real fit ever comes back suspicious, to rule out a code bug before
   suspecting the data.

**Status when first written:** both scripts build/run and the self-test passes (mean improvement ratio
~2.8x on synthetic data), but neither has been run against real hardware yet -- there is no real
sweep CSV, no real `residual_bias_model.npz`, and no real noise-floor-improvement number. That is
the next real-hardware step, and it's a precondition for the `sigma_f` measurement being the
*post-bias-correction* number the paper actually wants to report.

**Update (10 Aug, same day): run against real hardware, and the real number misses the ~2x
target.** `collect_free_space_sweep.py` completed cleanly (332,626 samples, all 21 waypoints x 3
`speed_factor`s). Fitting `fit_residual_bias.py` against it at the synthetic-tuned defaults
(`length_scale=3.0`) came back **worse than baseline on all 6 wrench axes** -- the same
underfitting failure mode the self-test exists to catch, showing up on real data because those
hyperparameters were tuned against synthetic data's feature scale, not the real sweep's. Sweeping
`length_scale` (real joint-angle/velocity ranges apparently want a much smoother kernel than the
synthetic self-test did) found a real optimum an order of magnitude higher than the default --
`length_scale=30, ridge_lambda=0.1` -- but even at that best config the mean improvement ratio is
only **~1.10x**, and the shortfall isn't spread evenly: `fy`, `fz`, `tx` reliably land 1.2-1.6x
(genuine improvement) while **`fx`, `ty`, and usually `tz` never beat baseline at all**, at any
hyperparameter setting tried. That per-axis consistency (not just noise on the worst run) points
at something structural to those three axes specifically -- lower intrinsic SNR at this
attachment/speed regime, or a real bug isolated to those columns -- rather than a tuning problem,
but it hasn't been root-caused yet. **Use ~1.1x, not ~2x, as the real noise-floor improvement when
interpreting `sigma_f` and the H1 identifiability thresholds**, and treat the `fx`/`ty`/`tz` gap
as an open risk rather than something this step already solved.

### Workspace and nullspace constraints

`config/workspace_constraints.yaml` holds two things consumed by the future variable-impedance
Cartesian controller (§4.3): a `nullspace_posture` (redundancy-resolution target) and
`workspace_bounds_m` (a Cartesian clamp). The nullspace posture is **not** a placeholder -- it's
the same fixed `start_joint_configuration` already used for the free-space re-zero and payload
calibration, reused here rather than inventing a second "canonical pose." The workspace bounds
**are** a placeholder (`status: placeholder_unmeasured` in the file) -- there is no real fixture to
measure against until T1/T2 are physically mounted and frozen, so the schema
and consuming code can be written and tested now, but the numbers must not be treated as a real
safety bound until replaced.

### Static hanging-mass calibration (sigma_f)

`scripts/static_hanging_mass_calibration.py` is the last step in §4.1b's sensorless
wrench-calibration priority order and the one independent check that everything before it
(payload ID, re-zero, residual bias model, workspace/nullspace constraints) actually produced
an accurate estimate: hang a known, calibrated mass off the tool and compare
`external_wrench_in_base_frame` against the mass's known weight.

**Why this is simple:** a freely-hanging mass exerts a force of `m * g` straight down, in the
**base** frame, regardless of where on the tool it's attached or what pose the arm is in --
unlike torque, which depends on the (unmeasured) lever arm and does rotate with the arm. So
this script validates `fx`/`fy`/`fz` against a clean, pose-independent predicted value at every
pose it visits; `tx`/`ty`/`tz` are logged but reported descriptively only, with no ground truth
in this procedure.

**Semi-automated by necessity:** nothing but the operator can hang a calibrated mass on the
robot, so the script drives the arm to each pose (reusing the same verified single-joint,
routed-through-base-pose waypoint pattern as `calibrate_payload.py`) and then pauses with an
interactive prompt at every `(pose, mass)` combination -- hang the mass, press Enter, it
samples for `--sample-time` seconds and moves on.

**Precondition:** the follower's normal tool payload must already be configured via `set_load`
(`calibrate_payload.py`). The mass you hang for this script must **not** also be
declared to `set_load` -- doing so would make the robot's own model absorb it, and the whole
point is testing the *external* wrench estimate against a load it doesn't know about.

**Safety:** each hold uses the same finished-state, zero-added-torque mechanism as
`calibrate_payload.py`'s and `collect_free_space_sweep.py`'s static holds -- it trusts the
robot's own gravity compensation, which by design does not include the hung mass. Earlier real-hardware sessions found
that holding an undeclared payload this way for several seconds is enough to trip a
`joint_velocity_violation` reflex once the undeclared weight is large enough. Start with the
lightest calibrated mass first and confirm a clean hold before moving to heavier ones.

**sigma_f, computed correctly:** the noise floor is pooled from raw per-sample wrench residuals
across every hold, not from the spread of per-hold *averages* -- averaging `n` samples within a
hold shrinks the spread of the mean by `sqrt(n)` relative to the actual single-sample noise
(that's the standard error, not the noise floor), and the identifiability mask in §4.1
thresholds single-timestep `|f_i|` against `sigma_f`. An early version of this script computed
it from hold-means and its own `--self-test` caught the bug: recovered `sigma_f` came out ~6x
smaller than the synthetic ground truth, matching `sqrt(40)` for 40 samples/hold.

**Output:** `<output_dir>/<namespace>_hanging_mass_calibration.json` (bias, `sigma_f` debiased
and a conservative RMS variant, within-hold noise, per-pose bias spread, and the zero-mass
baseline check), a per-`(pose, mass)` CSV, and a two-panel calibration figure (measured vs.
predicted `fz`, plus a residual histogram) unless `--no-figure`. `output_dir` defaults to
`/tmp/franka_teleop_hanging_mass_calibration`, overridable via
`TELEOP_HANGING_MASS_OUTPUT_DIR`. Default waypoints
(`config/hanging_mass_calibration_waypoints.yaml`) are restricted to the same two directions
verified clear with the eraser mount attached as `payload_calibration_waypoints_verified_subset.yaml`
-- do not extend it until the remaining directions are jog-verified, since this script's holds
are a strictly more demanding test of the undeclared-payload margin than anything already
verified.

```bash
ros2 run fr3_bilateral_teleop static_hanging_mass_calibration.py --namespace follower
python3 scripts/static_hanging_mass_calibration.py --self-test   # no hardware/ROS needed
```

### Variable-impedance controller: building blocks, and where they're wired in

`include/fr3_bilateral_teleop/stiffness_rate_limiter.hpp` and `.../energy_tank.hpp` implement the
two pieces: `|d(log k)/dt| <= gamma` rate limiting
(scale-invariant across the decades stiffness spans, unlike a linear rate limit) and the standard
passivity-via-energy-tank bookkeeping (stiffness *increases* refused once the tank is at its
floor; decreases always permitted). Both are standalone, unit-tested (`test/test_stiffness_rate_limiter.cpp`,
`test/test_energy_tank.cpp`) header-only classes.

**They are not wired into a controller in this package.** `TeleopFollowerController` remains the
original fixed-gain JOINT-space impedance controller used for bilateral teleoperation/data
collection and is not being changed. Cartesian controller is implemented by
the sibling package **`variable_impedance_controllers`** (a fork of
[crisp_controllers](https://github.com/learnsyslab/crisp_controllers), not part of `fr3_bilateral_teleop`)
instead of finishing a from-scratch controller here. The original is a mature, FR3-validated, ros2_control Cartesian-impedance/operational-space controller from
utiasDSL (IEEE RAP 2026), built specifically for deploying VLA policies via Pinocchio-based
dynamics -- a stronger foundation for the rest of this project's rollout use. It smooths via EMA filtering +
torque-rate saturation on its own, not this work's specific log-space-rate-limit +
energy-tank mechanism, so `stiffness_rate_limiter.hpp`/`energy_tank.hpp` are copied (not
depended on cross-package, so each package stays self-contained) into `variable_impedance_controllers/include/variable_impedance_controllers/utils/`, and
`variable_impedance_controllers/src/cartesian_controller.cpp` gained a new `applyStiffnessShaping()` step
(called every control cycle from `update()`) that runs the target stiffness (from either the
`target_stiffness` topic or the static `task.k_*` params) through both, producing the "applied"
stiffness the control law actually uses. New `stiffness_shaping` parameter block in
`variable_impedance_controllers/src/cartesian_controller.yaml` (`enabled`, `gamma`, `min_stiffness_{translational,rotational}`,
`energy_tank.{e0,e_min,e_max}`); `enabled: false` reverts to the original instantaneous
behavior for an A/B comparison. Both the commanded (`stiffness_target_diagonal_`) and realized
(`stiffness_applied_diagonal_`) stiffness, plus the tank's energy, are published every cycle on
`~/diagnostics/{stiffness_target,stiffness_applied,energy_tank_energy}` (always-on plain topics;
ROS2 Control Introspection is also registered but compiles out under this workspace's Humble
`hardware_interface` version, so the topics are what actually works here). Ported test coverage:
`variable_impedance_controllers/tests/test_log_space_stiffness_rate_limiter.cpp` (6 cases),
`.../test_energy_tank.cpp` (9 cases) -- both pass, as does `variable_impedance_controllers`' own pre-existing
suite (31/31 unaffected).


**Validated against fake hardware, same session, 11 Aug -- and a real bug found and fixed by
it.** No robot access this session, but `mock_components/GenericSystem` fake hardware needs none
(no physical robot, no real dynamics either). `launch/validate_variable_impedance.launch.py` +
`config/variable_impedance_validation_controllers.yaml` bring up a single fake FR3 running the
patched `CartesianController` (`fr3_link8` as the end-effector frame, no gripper needed);
`scripts/probe_variable_impedance_sinusoid.py` drives it with a sinusoidal `target_stiffness` and
a fixed, deliberately nonzero `target_pose` offset (zero offset would make every stiffness
increase free and never exercise the tank), records the three diagnostics topics, and renders the
commanded-vs-realized-stiffness + energy-tank figure.

#### Running the validation

```bash
# terminal 1 -- fake hardware by default, no physical robot needed
ros2 launch fr3_bilateral_teleop validate_variable_impedance.launch.py

# terminal 2, once the above logs "Controller activated."
ros2 run fr3_bilateral_teleop probe_variable_impedance_sinusoid.py --duration 20
# writes /tmp/franka_teleop_impedance_probe/sinusoid_probe_<ts>.{csv,png}
```

For a real robot once one is available: add `use_fake_hardware:=false robot_ip:=<ip>` to the
first command. Nothing else changes.

### Impedance label extraction and H1 evaluation

The offline tools that turn recorded demonstrations into labels, and evaluate those
demonstrations, live in `dataset_tools/`, separate from the rig scripts in `scripts/`. None of
them needs ROS, and all are installed for `ros2 run` like the rig scripts:

```
dataset_tools/
  labeling/extract_impedance_labels.py        demo CSV -> per-axis K(t) labels + identifiability mask
  evaluation/evaluate_gate1.py                H1 identifiability conditions over a pilot set of demo CSVs
  evaluation/analyze_adverb_separation.py     per-manner contact-force separation in a LeRobot dataset
  evaluation/plot_demo_force_phase_traces.py  per-axis contact-force traces with phase annotation
```

Three scripts, built together (none of this existed before -- no demo recorder, no
board/contact-frame calibration, no extraction/regression/mask code anywhere in the repo):

1. **`record_demo.py`** -- records one bilateral teleop demonstration to CSV: leader pose
   `x_l(t)`, follower pose `x_f(t)` (both from `franka_robot_state_broadcaster/current_pose`,
   confirmed live at the full 1kHz `convenience_publish_rate` on this rig), and the follower's
   `external_wrench_in_base_frame`. Interactive start (position the arms, press Enter), records
   for `--duration` seconds (default 25s, matching the ~25s/demo data budget).
   ```bash
   ros2 run fr3_bilateral_teleop record_demo.py --task T1_wiping --operator A --manner gently
   ```

2. **`extract_impedance_labels.py`** -- the §4.1 pipeline itself. Reads a demo CSV and produces
   per-axis, per-timestep stiffness `K(t)` with the identifiability mask.
   - **Contact frame is auto-fit from the demo's own in-contact follower positions** (SVD plane
     fit, normal oriented via mean contact-force direction, gated on a planarity-quality check)
     rather than a separate dedicated touch-calibration -- deliberately avoids adding new
     real-hardware motion (an earlier session had a near-miss from an improvised calibration
     workaround). Uses `--frame-fit-force-threshold` (default 8N), a
     **deliberately higher** threshold than the mask's `--contact-force-threshold` (2N) --
     real pilot data showed light/transitional contact near 2N (approach, retreat,
     grazing touches) is genuinely not planar (planarity_ratio 0.15-0.24 on 4/5 real pilots),
     while firm contact above ~6-8N is (0.007-0.13 on the same demos) -- a real geometric
     distinction between two different jobs (plane-fitting precision vs. mask SNR
     eligibility), not a bug fixed by loosening one shared threshold.
   - Pose error `e(t)` and its derivative are computed in that frame (position difference +
     `SO(3)` log-map for orientation), windowed (300ms, trailing/causal) into a regularized,
     log-space, box-constrained regression solved via `scipy.optimize.least_squares` (the
     `||log k - log k_prior||^2` regularizer is genuinely nonlinear, not closed-form).
   - Output is at **30Hz**, matching the policy's action-chunk rate, not the raw 1kHz --
     that's the rate anything downstream actually consumes.
   - The identifiability mask implements all four conditions; excitation (Gram condition
     number) and sustained-contact are window properties, `|e_i|`/`|f_i|` above their noise
     floors are evaluated instantaneously. Coverage is reported both overall and **restricted to
     contact timesteps**, matching H1 condition (i)'s literal wording.
   - Damping `d_i` is fit freely (all the offline analysis figure needs). The POLICY-TARGET
     constrained form `D = 2*zeta*sqrt(K*Mhat)` needs a Cartesian effective-mass estimate this
     demo-CSV pipeline doesn't have -- deferred to training prep, not silently skipped.
   ```bash
   ros2 run fr3_bilateral_teleop extract_impedance_labels.py --input demo1.csv demo2.csv ...
   python3 dataset_tools/labeling/extract_impedance_labels.py --self-test   # synthetic ground truth, no hardware
   ```

3. **`evaluate_gate1.py`** -- runs the extraction pipeline (imported, not reimplemented -- same
   pattern `diagnose_gravity_preload.py` uses against `calibrate_payload.py`) across a pilot set
   and checks all three H1 conditions. The method states condition (i) precisely but
   leaves (ii)/(iii) qualitative -- this script defines and documents concrete computations:
   (ii) compares whole-demo `K` variation against a short-timescale (adjacent 30Hz-step) noise
   estimate, on the reasoning that true stiffness can't swing meaningfully between two mostly-
   overlapping 300ms windows 33ms apart; (iii) is the max/min ratio of the three translational
   axes' median masked `K`. Reports per-demo and pooled, but does **not** auto-decide the outcome --
   whether H1 holds is left as a call for a human to make.
   ```bash
   ros2 run fr3_bilateral_teleop evaluate_gate1.py --input pilot1.csv pilot2.csv pilot3.csv ...
   python3 dataset_tools/evaluation/evaluate_gate1.py --self-test   # synthetic, no hardware
   ```

**Self-test coverage, and a real bug it caught:** `extract_impedance_labels.py --self-test`
recovers known per-axis anisotropic stiffness (300/250/800 N/m, 20/15/40 Nm/rad) to within 0-7%,
correctly suppresses the free-space phase (0% false-identifiable), and clears H1 condition (i)'s 25%
within-contact threshold (85-95%) on data designed to pass it. The first version of the synthetic
ground-truth generator used a different in-plane `(x, y)` basis convention than
`fit_contact_frame` recovers from real data (in-plane axis choice about a normal is inherently
gauge-arbitrary) -- this looked exactly like a stiffness-axis swap between `erx`/`ery` until both
were made to share one `build_inplane_basis` convention. `evaluate_gate1.py --self-test` runs
both an anisotropic case (should pass all three conditions) and an isotropic one (should fail
*only* the anisotropy condition), confirming that check actually discriminates rather than always
passing.

### Live force-band display

`live_force_band_display.py` -- a standalone operator display showing the follower's live
contact-force magnitude against the three adverb target bands
(`gently` 3-6 N, `normally` 8-12 N, `firmly` 15-22 N). Run it alongside teleop and the LeRobot
recorder during collection:
```bash
ros2 run fr3_bilateral_teleop live_force_band_display.py --target-manner gently
```
- Deliberately **not** part of `data_recorder`'s `record_lerobot` node and never writes
  a dataset frame -- "the policy never sees these numbers" holds by construction, not
  by convention, since this process has no path to the dataset at all.
- Displays `||(fx, fy, fz)||` from `external_wrench_in_base_frame`, smoothed over
  `--smoothing-window` samples (default 50, ~50ms at the follower's 1kHz rate) for readability.
  This is a magnitude proxy for the board-normal force, not the real per-axis decomposition --
  that only exists offline, fit from a completed demo's own in-contact positions
  (`extract_impedance_labels.py::fit_contact_frame`). It's an acceptable approximation for a live
  gauge because the real pilot demos already measured a 7-14x contact-frame anisotropy ratio
  once contact is firm, so force magnitude is normal-force-dominated during genuine wiping
  contact. Don't read anything quantitative from this display beyond "in/out of band."
- `--target-manner {gently,normally,firmly}` fills in the band currently being collected so the
  operator has one less thing to track visually.

## Getting started

To run any quickstart example, you must add the correct IP addresses of you robot to the configuration as described in the following subsections. All other parameters have sensible defaults. If you need to change them, they are datailed in the section [Configuration parameters](#configuration-parameters).

You can also find minimal config examples in the config folder: [config/fr3_teleop_config.yaml](config/fr3_teleop_config.yaml) for single arm setups and [config/fr3_duo_teleop_config.yaml](config/fr3_duo_teleop_config.yaml) for custom humanoid setups, **FR3 Duo** or **Mobile FR3 Duo**.

> [!IMPORTANT]
> Before starting, make sure you can reach your robots over the local network and they are in **FCI** Mode.

### ROS 2 Quickstart

If you are familiar with ROS 2 and want to get more into the code, we recommend using any of the following methods.

To add the IP addresses of your robots to the configuration, you will need to edit the file [config/fr3_teleop_config.yaml](config/fr3_teleop_config.yaml) after downloading the repo.
After building and sourcing your workspace (for more detailed instructions see below), you can run the teleoperation example using: `ros2 launch fr3_bilateral_teleop teleop.launch.py`.

Alternativley, you can also create a copy of either config file and supply that file to the launch command: `ros2 launch fr3_bilateral_teleop teleop.launch.py robot_config_file:=/path/to/your/copy/of/fr3_teleop_config.yaml`


#### Integrate it into your own ROS 2 workspace

This is a standard ROS 2 package. Copy it into your workspace's `src` folder as usual, together with its sibling `variable_impedance_controllers` if you want the variable-impedance validation launch. You might need to also download the [franka_ros2](https://github.com/frankarobotics/franka_ros2) dependency manually.

To set up a ROS 2 environment, follow the official ROS 2 Humble installation [instructions](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html).

After creating a [workspace](https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Creating-A-Workspace/Creating-A-Workspace.html), download the repo to the `src` directory and modify the config files as described above.

In a terminal, build the ROS 2 workspace by navigating to `/path/to/your/ros2_workspace/` and executing `colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release`.

After building the workspace, source it using: `source install/setup.bash`

You can now launch the teleoperation.

## Configuration parameters

In this section the parameters of interest to the user are listed and explained here. You can set them in a config file (example: [config/fr3_teleop_config.yaml](config/fr3_teleop_config.yaml)) or in the `x-teleop-config` parameter of the `docker-compose.yml`.

```yaml
# Base parameters:
# These parameters are valid for all started robots
# Except when they overridden by pair or robot-specific parameters.
# If not set there are defaults for most of them.
# Example values shown here are the default values

urdf_file: "fr3/fr3.urdf.xacro" # Which robot model to use.
# Must match your actual robots.
# Possible values are: fr3/fr3.urdf.xacro, fr3v2/fr3v2.urdf or xacro or fp3/fp3.urdf.xacro

load_gripper: false # If 'true' launch additional nodes to allow teleoperation of the Franka Hand.
# Franka Hands must be attached to the robots if set to 'true'.
# If you have a third-party gripper or hand attached you must set this to `false`.

input_topic_timeout: 2500000 # If messages received from other robots are too old the robot goes into zero gravity mode

fake_hardware: false # For testing purposes you can use fake hardware interfaces

base_namespace: null # Which base namespace to use. All nodes will be started in the base namespace

# Collision thresholds will make the robot stop if exceeded
upper_torque_thresholds_acceleration: [85.0, 85.0, 85.0, 85.0, 11.0, 11.0, 11.0]
upper_torque_thresholds_nominal: [85.0, 85.0, 85.0, 85.0, 11.0, 11.0, 11.0]
upper_force_thresholds_acceleration: [85.0, 85.0, 85.0, 85.0, 11.0, 11.0]
upper_force_thresholds_nominal: [85.0, 85.0, 85.0, 85.0, 11.0, 11.0]

# Be careful when changing control-related parameters
k_gains: [600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0] # Stiffness parameters of the follower's joint impedance controller.
d_gains: [30.0, 30.0, 30.0, 30.0, 10.0, 10.0, 5.0] # Damping parameters of the follower's joint impedance controller.

# Force channels of the 4-channel bilateral control scheme (see "Bilateral control architecture"
# above). Independent from k_gains/d_gains, so tracking and feel can each be tuned on their own.
force_reflection_gains: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] # Leader-side gain on the follower's contact torque reflected back to the operator (feel).
force_feedforward_gains: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] # Follower-side gain on the leader's (operator's) own torque, fed forward into the follower (feel).

alpha: [3, 3, 3, 3, 1, 1, 1] # Feedback-avoidance damping on the leader, per joint. Prevents the leader from "jumping away" right after the follower makes contact.

start_joint_configuration: [0.0, -0.7853981633974483, 0.0, -2.356194490192345, 0.0, 1.5707963267948966, 0.7853981633974483] # HOME position (radians, 7 joint values): both move_to_start_example_controller and teleop_coordinator's free-space re-zero drive the arm here before teleop starts.
# Default is Franka's standard "ready" pose. Settable per base/pair/robot like k_gains/d_gains
# above -- e.g. give the leader and follower different homes with leader: / follower: overrides.
# Must stay a joint-limit-safe, collision-free posture for your cell; the value is applied
# as-is with no validation beyond checking there are exactly 7 numbers.
# NOTE: config/workspace_constraints.yaml's nullspace_posture is a separate hardcoded copy of
# this same default and does not automatically follow an override here.

pairs:
    - namespace: pair_one # each pair must have the 'namespace' parameter set
      # parameters set in a pair override base parameters
      load_gripper: true # this pair does not use grippers, so we override the base parameter
      leader:
        # you can also override some parameters for each robot separately
        # some like robot_ip, arm_id and arm_prefix can only be set for one robot specifically
        robot_ip: leader.pair_one.franka.de # IP address or hostname of robot
        arm_prefix: "" # Prefix for arm topics
      follower:
        robot_ip: follower.pair_one.franka.de

    - namespace: pair_two
      urdf_file: "fp3/fp3.urdf.xacro" # this pair uses fp3's and therefore overrides the base parameter
      leader:
        robot_ip: leader.pair_two.franka.de
      follower:
        robot_ip: follower.pair_two.franka.de
        urdf_file: "fr3v2/fr3v2.urdf.xacro" # You can also set the urdf_file as robot-specific parameter

    - namespace: pair_three
      k_gains: [800.0, 800.0, 800.0, 800.0, 350.0, 250.0, 75.0] # this pair has a different task and needs to override some control parameters, so we override the default parameters
      leader:
        robot_ip: leader.pair_three.franka.de
      follower:
        robot_ip: follower.pair_three.franka.de

    - namespace: pair_four
      start_joint_configuration: [0.0, -0.4, 0.0, -2.2, 0.0, 1.8, 0.9] # this pair's task fixture needs a different HOME pose than the default "ready" pose
      leader:
        robot_ip: leader.pair_four.franka.de
      follower:
        robot_ip: follower.pair_four.franka.de
```

## Tests and Linting

In a workspace you can use the following commands to run tests and linters for this package.

```bash
colcon test --packages-select fr3_bilateral_teleop
colcon test-result --verbose
```

Some of the tools can be used independently
```bash
cd path/to/ros2_ws/src/fr3_bilateral_teleop

ament_uncrustify # for c++

ament_flake8 # for python
ament_pep257
```