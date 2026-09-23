# variable_impedance_controllers

Real-time `ros2_control` controllers for compliant, torque-based control of a manipulator,
with a Cartesian impedance controller that can safely execute a **policy-commanded,
time-varying stiffness**.

> Derived from [crisp_controllers](https://github.com/learnsyslab/crisp_controllers)
> (MIT, Learning Systems and Robotics Lab, TUM) at release 2.4.1, commit
> `ccbead4ba22b1a5b43ac8e81d7137c9ed7b65925`, and renamed so it can be installed alongside
> the original. See `NOTICE` for what was changed. For the upstream controllers' general
> documentation (parameters, demos, robot setups) see the
> [crisp_controllers website](https://utiasdsl.github.io/crisp_controllers/); everything
> documented there still applies here with the package name substituted.

## Controllers

All are exported as `controller_interface::ControllerInterface` plugins:

| Plugin type | What it does |
| --- | --- |
| `variable_impedance_controllers/CartesianController` | Cartesian impedance / operational-space control of the end-effector pose via a Pinocchio model, with stiffness shaping (below) |
| `variable_impedance_controllers/CartesianAdmittanceController` | Cartesian admittance with an impedance outer loop |
| `variable_impedance_controllers/TorqueFeedbackController` | Joint-space PD response to external torques, with friction compensation |
| `variable_impedance_controllers/PoseBroadcaster` | Publishes the end-effector pose |
| `variable_impedance_controllers/TwistBroadcaster` | Publishes the end-effector twist |

## Stiffness shaping

A policy that emits stiffness at chunk rate produces a piecewise-constant `K(t)` with a jump
at every chunk boundary. Applied directly at 1 kHz, each jump injects energy in a single
control step. `CartesianController` therefore runs its target stiffness, whether from the
static `task.k_*` parameters or the `target_stiffness` topic, through two mechanisms every
cycle (`applyStiffnessShaping()` in `src/cartesian_controller.cpp`):

* **Log-space rate limiter**
  (`include/variable_impedance_controllers/utils/log_space_stiffness_rate_limiter.hpp`)
  bounds `|d(log K)/dt| ≤ γ` per axis. The bound is in log space because stiffness spans
  decades, so a single γ in linear units would be far too aggressive at low K and far too
  conservative at high K.
* **Energy tank** (`include/variable_impedance_controllers/utils/energy_tank.hpp`) is a
  passivity guard. Stiffness *increases* are refused once the tank is depleted, while
  decreases are always permitted. The controller can always give way but cannot stiffen
  past its passivity budget.

Rate limiting alone does not bound cumulative energy injection, and a tank alone does not
bound per-step discontinuity. That is why both are present.

When the tank vetoes an axis, the rate limiter's internal state is re-synced to the value
actually applied (`stiffness_rate_limiter_->reset(k_shaped)`), rather than left to keep
chasing the raw target. Otherwise, under sustained zero dissipation, a blocked axis ratchets
toward `k_max` and then stays stuck well beyond what tank depletion alone would explain.

### Parameters

New `stiffness_shaping` block in `src/cartesian_controller.yaml`:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `enabled` | `true` | `false` restores the original instantaneous behaviour, for A/B comparison |
| `gamma` | `[3.0]*6` | max `|d(log K)/dt|` per axis, 1/s |
| `min_stiffness_translational` | `50.0` | lower bound of realizable translational stiffness, N/m |
| `min_stiffness_rotational` | `5.0` | lower bound of realizable rotational stiffness, Nm/rad |
| `energy_tank.e0` / `e_min` / `e_max` | `1.0` / `0.0` / `5.0` | initial energy, floor and ceiling, J |

Upper bounds reuse `variable_max_stiffness.{translational,rotational}`.

### Diagnostics

Published every cycle as plain topics:
`~/diagnostics/{stiffness_target,stiffness_applied,energy_tank_energy}`. These carry the
commanded-versus-realized stiffness signal. The `REGISTER_ROS2_CONTROL_INTROSPECTION` calls
are also present, but they compile out below `hardware_interface` 4.27.0 (ROS 2 Humble).

## Building and testing

```bash
# in a ROS 2 workspace, with this package under src/
colcon build --packages-select variable_impedance_controllers --cmake-args -DCMAKE_BUILD_TYPE=Release
colcon test  --packages-select variable_impedance_controllers
colcon test-result --verbose
```

Unit tests live in `tests/`. `test_log_space_stiffness_rate_limiter.cpp` and
`test_energy_tank.cpp` cover the stiffness-shaping mechanisms. The rest are upstream's.

A fake-hardware validation setup for the stiffness shaping, which needs no robot, lives in
the sibling package: `ros2 launch fr3_bilateral_teleop validate_variable_impedance.launch.py`
followed by `ros2 run fr3_bilateral_teleop probe_variable_impedance_sinusoid.py`.
