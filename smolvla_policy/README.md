# smolvla_policy/

The SmolVLA-based compliance policies (B0/B2/B3/B5) and the from-scratch B4
control. The Python code lives in the `compliance_vla.policy` package
(`../src/compliance_vla/policy/`): the model, its losses and dataset, and the
training, serving, evaluation and diagnostic entrypoints. This directory holds
the one part that cannot live in a Python package, the ROS 2 rollout package
`deploy_vla/`.

| Group | Modules in `compliance_vla.policy` |
| --- | --- |
| Model | `compliance_policy` (B2/B5), `hybrid_policy` (B3), `bi_act_policy` (B4), `losses`, `force_encoder`, `dataset`, `labels` |
| Training | `train_policy`, `train_b5`, `tune_lambda`, `fit_manner_force_calibration` |
| Serving | `serve_policy`, `client_example` |
| Rollout evaluation | `evaluate_gate3`, `score_ink_removal`, `controller_frame_utils` |
| Diagnostics | `diagnose_language_grounding`, `check_input_sensitivity`, `check_state_convention_sensitivity`, `analyze_operatorA_rollout_trajectories` |

Dataset reading, forward kinematics, label extraction and the tool-offset
calibration the policy depends on live in `../data_extraction/`.

## Setup

Every command below runs from the release root (`compliance-vla/`) with the
package installed editable, so the modules can find `data_extraction/` next to
them:

```bash
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -e '.[policy,analysis]' 'lerobot[smolvla]'
```

## Training

One invocation trains one policy/seed:

```bash
python -m compliance_vla.policy.train_policy --policy b5 --seed 0 --steps 30000
```

`--policy` is `b0` (position output, no force input), `b2` (force input,
position output), `b3` (force input, hybrid force-position output), `b4`
(ACT/Bi-ACT trained **from scratch** -- no pretrained VLM, no
ImageNet-pretrained vision backbone, compliance output -- the H4 control), or
`b5` (force input, compliance output -- "ours"). Useful flags: `--lam` (the
compliance loss weight, default 1.0), `--batch-size`, `--lr`,
`--checkpoint-dir`/`--checkpoint-every`, `--wandb`.

Before committing GPU time to a real run, sanity-check the code path with a
short pass (a handful of real steps, checked for NaNs):

```bash
python -m compliance_vla.policy.train_policy --policy b5 --seed 0 --smoke-test
```

`train_b5` is a thin wrapper that forces `--policy b5`, kept for the exact
command line that produced `reports/b5_seed0_smoke_train_log.json`.

B0/B2/B3/B5 fine-tune the ~450M-parameter SmolVLA. B4 has no pretrained VLM
and is ~55M parameters. Run one training job per GPU: concurrent fine-tunes
sharing a card risk OOM or heavy slowdown from compute contention.

### Data split

By default training uses only `demo4` (32 episodes), the frozen train split
from `data_extraction/build_dataset_splits.py` (train=demo4, val=demo1,
test=demo3). `tune_lambda`, `scripts/offline_stiffness_benchmark.py` and
`scripts/fit_b1_oracle_stiffness.py` all assume that split when they read
"val"/"test". `--all-sessions` instead pools demo1+demo3+demo4 (92 episodes),
and `--train-sessions demo1 demo4` picks any subset. A run launched that way has
no held-out val/test, so it is not comparable to the frozen-split baselines.

### Tuning lambda on the validation split

```bash
python -m compliance_vla.policy.tune_lambda --lambdas 0.1 0.3 1.0 3.0 10.0 --tune-steps 2000
```

Trains a short B5 per candidate lambda on demo4, evaluates on demo1, and picks
the lambda minimizing the unweighted sum of val `loss_x_eq + loss_log_k` (see
the module docstring for why not a lambda-weighted number). Pass the winner to
the real runs with `--lam`.

### B1

B1 is not a separate network: it is the trained B0 checkpoint run through the
same controller with stiffness pinned to the constant vector from
`python scripts/fit_b1_oracle_stiffness.py` (CPU-only, under a minute).

### B3

B3 (`hybrid_policy.HybridSmolVLAPolicy`) has the same pretrained backbone and
force injection as B2/B5, but decodes log_k with a separate one-shot regression
head instead of folding it into B5's shared flow-matching target (ForceVLA2 /
Force Policy style). `python -m compliance_vla.policy.hybrid_policy` runs its
synthetic-data self-test (shapes and gradient flow, no GPU needed).

## Serving a checkpoint over HTTP

`serve_policy` loads one checkpoint and exposes it as an HTTP server;
`client_example` is a minimal client. It is meant for the common split where the
GPU lives on one machine and the robot-control loop on another, reachable over
an SSH tunnel -- see the module docstring for the request/response format and
the `ssh -L` command.

```bash
uv pip install fastapi uvicorn json-numpy
python -m compliance_vla.policy.serve_policy --checkpoint checkpoints/b5_seed0/b5_seed0_step30000.pt
```

## In-distribution pilot rollouts (T1)

The pure-Python pieces run without ROS:

```bash
python -m compliance_vla.policy.controller_frame_utils        # contact-frame -> base-frame stiffness rotation, self-tests
python -m compliance_vla.policy.score_ink_removal --self-test # T1 ink-removal metric
python -m compliance_vla.policy.evaluate_gate3 --self-test    # pilot-check decision logic
```

`data_extraction/fit_frozen_contact_frame.py` fits one frozen T1 board-normal
rotation, so a B3/B5 checkpoint's log_k (predicted in the contact frame) can be
rotated into the base-frame diagonal stiffness the controller consumes. The
shipped `data_extraction/contact_frame_t1.npy` was fit on demo4 (planarity
ratio 0.075, 1242 pooled in-contact samples).

The rollouts themselves (`ros2 run fr3_bilateral_teleop run_pilot_rollout.py`,
n=5 per policy, sanity only) need the real rig; see that script's docstring for
usage and the by-hand ROI calibration. It calls `serve_policy` over HTTP and
writes one JSON log per rollout. Copy those logs into `reports/pilot_rollouts/`,
then:

```bash
python -m compliance_vla.policy.evaluate_gate3 \
    --b0-logs 'reports/pilot_rollouts/b0_*.json' \
    --b5-logs 'reports/pilot_rollouts/b5_*.json'
```

This decides PASS/FAIL on "B5 >= B0" in mean targeted ink removal, as a plain
comparison rather than a CI test, since these pilots are not reported. On FAIL
it prints the next diagnostic (re-verify commanded-vs-realized stiffness
tracking with `probe_variable_impedance_sinusoid.py`) instead of touching the
model.

## Robot-side deployment: `deploy_vla/`

An ament_python ROS 2 package. `deploy_smolvla.py` is the rollout node: it
queries `serve_policy`, applies temporal ensembling across overlapping action
chunks (`src/compliance_vla/ensembling.py` is that mechanism extracted so it can
be tested without ROS), and publishes target pose and stiffness to
`variable_impedance_controllers`. The package also holds the evaluation and
scoring harnesses (`run_final_evaluation.py`, `score_wipe.py`).

It reads `tool_offset.npy` from `data_extraction/`, so the release is expected
at `src/compliance-vla/` in the ROS 2 workspace; see `../hardware/README.md` for
the build.
