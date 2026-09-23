# Robot-side code

Four ROS 2 packages, each self-contained:

| Package | Build type | Origin |
| --- | --- | --- |
| `variable_impedance_controllers` | ament_cmake | renamed fork of [crisp_controllers](https://github.com/learnsyslab/crisp_controllers) (MIT) |
| `fr3_bilateral_teleop` | ament_cmake | renamed fork of [franka_ros2_teleop](https://github.com/frankarobotics/franka_ros2_teleop) (Apache 2.0) |
| `deploy_vla` | ament_python | written for this work |
| `data_recorder` | ament_python | written for this work |

```bash
# in a ROS 2 workspace
cp -r hardware/{variable_impedance_controllers,fr3_bilateral_teleop,deploy_vla,data_recorder} src/
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
colcon test --packages-select variable_impedance_controllers fr3_bilateral_teleop
```

`fr3_bilateral_teleop` needs `franka_ros2` (`franka_bringup`, `franka_msgs`,
`franka_semantic_components`), and `variable_impedance_controllers` needs Pinocchio.

The two forks were renamed throughout: package name, C++ namespace, include
directory, plugin library, and plugin class names such as
`variable_impedance_controllers/CartesianController`. They can therefore be
installed next to the originals without clashing. Each fork keeps its upstream
license file, and its `NOTICE` records the upstream commit it was forked from
and lists what was changed.

## variable_impedance_controllers

This is the 1 kHz Cartesian impedance controller. It tracks the policy's
commanded equilibrium pose and stiffness.

This work adds the two mechanisms that make a *policy-commanded*, chunk-rate
stiffness safe to execute, plus their unit tests:

* `utils/log_space_stiffness_rate_limiter.hpp` bounds `|d(log K)/dt| ≤ γ`.
  The bound is imposed in log space because stiffness spans decades, so a single
  γ in linear units would be far too aggressive at low K and far too
  conservative at high K.
* `utils/energy_tank.hpp` provides passivity via an energy tank. Stiffness
  *increases* are refused once the tank is depleted, while decreases are always
  permitted. The controller can always give way but can never stiffen past its
  passivity budget.

Rate limiting alone does not bound cumulative energy injection, and a tank alone
does not bound per-step discontinuity. That is why both are present. The
modifications to `cartesian_controller.{hpp,cpp,yaml}` wire them into the control
loop and expose their parameters under `stiffness_shaping`. The package README
has the details.

## fr3_bilateral_teleop

This is the four-channel bilateral teleoperation stack. It provides the leader
pose that the whole method depends on.

This work upgrades the leader/follower controllers to the four-channel scheme and
adds the entire offline and calibration pipeline. The rig-side scripts, which run
against the robot, plus the calibration fits, are under `scripts/`:

| File | Role |
| --- | --- |
| `fit_residual_bias.py` | RFF-ridge residual wrench bias model |
| `calibrate_payload.py` | Payload mass and centre-of-mass identification |
| `static_hanging_mass_calibration.py` | Ground-truth wrench check |
| `collect_free_space_sweep.py` | Free-space sweep collection for the noise floor |
| `record_demo.py` | Bilateral demonstration recording |
| `probe_variable_impedance_sinusoid.py` | Commanded-vs-realized stiffness validation |
| `run_pilot_rollout.py` | Pilot policy rollout harness |

The offline tools that label and evaluate recorded demonstrations need no ROS. They
are under `dataset_tools/`:

| File | Role |
| --- | --- |
| `labeling/extract_impedance_labels.py` | Compliance label extraction — the method's core |
| `evaluation/evaluate_gate1.py` | Identifiability conditions on real data |
| `evaluation/analyze_adverb_separation.py` | Per-manner contact-force separation in a dataset |
| `evaluation/plot_demo_force_phase_traces.py` | Per-axis force traces with phase annotation |

The package also adds:

* a `payload_model_broadcaster`
* the C++ safety utilities in their original home (`energy_tank.hpp`,
  `stiffness_rate_limiter.hpp`, `cartesian_impedance_math.hpp`), with their tests
* the calibration waypoint and workspace-constraint configs

`validate_variable_impedance.launch.py` brings up
`variable_impedance_controllers` on fake hardware to check the stiffness shaping
without a robot.

`src/compliance_vla/` was lifted from `extract_impedance_labels.py` and
`fit_residual_bias.py`. `tests/test_provenance.py` checks that the two copies
have not diverged.

## deploy_vla and data_recorder

* **`deploy_vla`** handles policy deployment and evaluation. `deploy_smolvla.py`
  is the rollout node. It implements temporal ensembling across overlapping
  action chunks; `src/compliance_vla/ensembling.py` is that mechanism extracted
  so it can be tested without ROS. The package also holds the evaluation and
  scoring harnesses.
* **`data_recorder`** records bilateral demonstrations into the
  LeRobot dataset format, including the leader-pose channel the method needs.

## Running any of this

These packages need a real Franka FR3 setup, with two arms for bilateral
teleoperation. This release's Python test suite does not exercise them. The C++
safety utilities do carry their own unit tests, which run under `colcon test`.
`src/compliance_vla/safety.py` mirrors that C++ in Python, so the same invariants
are checked on any machine.
