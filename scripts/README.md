# scripts/

## How to launch training (Day 12: B0, B2, B5; Day 13: B4, B1)

Training code lives in `../training/` (dataset/label bridging, the shared
B2/B5 compliance policy, the B4 ACT/Bi-ACT-from-scratch policy, losses,
force encoder); these scripts are the entrypoints. All of it needs the
project venv:

```bash
cd compliance-vla
source .venv/bin/activate
```

(If `.venv` doesn't exist yet: `uv venv --python 3.11 .venv && source .venv/bin/activate && uv pip install 'lerobot[smolvla]' scipy scikit-learn matplotlib` -- system Python is 3.9, too old for `lerobot`.)

### One run, one policy/seed

```bash
python scripts/train_policy.py --policy b5 --seed 0 --steps 30000
```

`--policy` is `b0` (position output, no force input), `b2` (force input,
position output), `b4` (ACT/Bi-ACT trained **from scratch** -- no pretrained
VLM, no ImageNet-pretrained vision backbone, compliance output -- the H4
control, see `src/compliance_vla/policy/bi_act_policy.py`), or `b5` (force input, compliance
output -- "ours"). Useful flags: `--lam` (B4/B5's compliance loss weight,
default 1.0), `--batch-size`, `--lr`, `--checkpoint-dir`/`--checkpoint-every`.

By default training only uses `demo4` (32 episodes) -- the frozen train split
from `build_dataset_splits.py` (Day 10: train=demo4, val=demo1, test=demo3),
which every val/test-scored script (`tune_lambda.py`,
`offline_stiffness_benchmark.py`, `fit_b1_oracle_stiffness.py`) assumes when
it reads "val"/"test". Pass `--all-sessions` to instead pool every collected
session (`demo1`+`demo3`+`demo4`, 92 episodes total) for training -- e.g. for
the language-grounding unfreeze retraining ablation in
`../language_grounding_issue_handoff.md`, which asks to retrain on "the
existing 92-episode dataset" rather than demo4's 32. A run launched this way
has no held-out val/test left, so don't treat it as comparable to the
frozen-split baselines. `--train-sessions demo1 demo4` (any subset) also
works directly if you want something other than all-or-one.

Before committing GPU time to a real run, sanity-check the code path with a
short pass (a handful of real steps, checked for NaNs -- not a real training
run):

```bash
python scripts/train_policy.py --policy b4 --seed 0 --smoke-test
python scripts/train_policy.py --policy b5 --seed 0 --smoke-test
```

`scripts/train_b5.py` still works as a 3-line wrapper that forces `--policy b5`
(kept for the exact command line Day 11's smoke test used).

### All 9 runs (3 policies x 3 seeds), across your GPUs

```bash
./scripts/launch_day12_runs.sh                 # default 30000 steps each
./scripts/launch_day12_runs.sh --dry-run        # preview the 9 commands + GPU assignment first
./scripts/launch_day12_runs.sh --steps 5000     # any extra flags forward to every one of the 9 runs
```

Jobs are assigned 3 GPUs x one seed's 3 policies each, run sequentially per
GPU (3-way parallel, 3 deep) -- not all 9 truly simultaneous, since three
concurrent 450M-param fine-tunes sharing one 46GB card risks OOM or heavy
slowdown from compute contention.

- Logs: `reports/logs/<policy>_seed<seed>.log` -- e.g. `tail -f reports/logs/b5_seed0.log`
- Job PIDs: `reports/logs/day12_jobs.txt`
- Stop everything: `kill $(awk '{print $NF}' reports/logs/day12_jobs.txt)`

### Day 13: B4 (3 seeds), one dedicated GPU each

```bash
./scripts/launch_day13_b4_runs.sh                 # default 30000 steps each, GPUs 0/1/2
GPUS="1 2 3" ./scripts/launch_day13_b4_runs.sh     # override which GPUs, e.g. if 0 is busy -- check nvidia-smi first
./scripts/launch_day13_b4_runs.sh --dry-run        # preview the 3 commands + GPU assignment first
```

B4 (`src/compliance_vla/policy/bi_act_policy.py`) has no pretrained VLM and is ~55M params
(vs. SmolVLA's ~450M), so unlike the Day-12 3-deep packing this launches all
3 seeds one-job-per-GPU, no sequential queueing needed. Logs:
`reports/logs/b4_seed<seed>.log`, job list `reports/logs/day13_b4_jobs.txt`,
same stop-everything pattern as above.

### Day 13: B1 (oracle constant stiffness, grid-searched on validation)

```bash
python scripts/fit_b1_oracle_stiffness.py
```

CPU-only, no GPU, runs in well under a minute. Not a neural network -- B1's
"policy" is the already-trained B0 checkpoint run through the same
low-level controller as every other baseline, with the controller's
per-axis stiffness pinned to the constant vector this script produces
(`reports/b1_oracle_constant_stiffness.json`, `summary.k_vector_contact_frame`)
instead of B0's own default fixed-high-stiffness. See the script's own
docstring for how the grid search differs from `offline_stiffness_benchmark.py`'s
(Day 10, M8) closed-form train-mean "constant" baseline.

### Tuning lambda (compliance loss weight) on the validation split

```bash
python scripts/tune_lambda.py --lambdas 0.1 0.3 1.0 3.0 10.0 --tune-steps 2000
```

Trains a short B5 per candidate lambda (train=demo4), evaluates on the
frozen val session (demo1), and picks the lambda minimizing the unweighted
sum of val `loss_x_eq + loss_log_k` (see the script's own docstring for why
not a lambda-weighted number). Feed the winner into the real runs:

```bash
./scripts/launch_day12_runs.sh --lam <best_lambda>   # only affects b5; b0/b2 ignore --lam
```

This grid search is itself real (if short) GPU training, one run per
candidate -- run it when you're ready to spend that time, same as the 9-run
campaign above.

## Running on the HPC cluster (offline)

`scripts/slurm_train_hpc.sbatch` submits one policy/seed run to HPC's `acc`
partition (`acc_training` QoS, 48h cap, `--gres=gpu:1` + 20 CPUs/GPU per
the HPC site's rules). **HPC has no internet on login or compute nodes** (only the
the HPC site-staff-only, VPN-gated `glogin4`/`alogin4`), so the venv, all wheels, and
the pretrained VLM weights (`HuggingFaceTB/SmolVLM2-500M-Video-Instruct`,
pulled by `load_vlm_weights=True`) must be staged and copied in ahead of
time -- nothing in the sbatch script may touch the network.

On a machine with internet (matching HPC's `linux/x86_64`, Python 3.12 --
HPC's `module avail python` offers 3.8/3.9/3.10/3.12, no 3.11, so this
project's usual "3.11, system Python is too old" venv step doesn't apply
here; 3.12 is the one to target):

```bash
# 1. wheelhouse -- everything scripts/README's venv step installs, pinned to
#    known-compatible versions (scripts/requirements_hpc.txt) rather than
#    loose names -- `pip download 'lerobot[smolvla]' ...` free-resolves and
#    backtracks forever, since lerobot's dependency ranges conflict across
#    versions when nothing is already installed to anchor the resolver.
#    --platform is intentionally omitted: some deps (e.g. av) only ship
#    manylinux_2_28 wheels, not manylinux2014, and omitting --platform lets
#    pip use the full compatible-tag list for the current host instead of
#    one exact tag.
pip download -r scripts/requirements_hpc.txt --no-deps \
    --dest wheelhouse --python-version 312 --implementation cp --abi cp312 --prefer-binary

# 2. pretrained backbone + processor/tokenizer, into the HF cache dir the
#    sbatch script points HF_HOME at (must be named .hf_cache, not hf_cache_tmp)
python -c "from huggingface_hub import snapshot_download; snapshot_download('HuggingFaceTB/SmolVLM2-500M-Video-Instruct', cache_dir='.hf_cache')"

# 3. the `uv` binary itself -- a single static executable, no Python/network
#    dependency at runtime. Installing via uv instead of plain pip matters on
#    HPC's GPFS-backed home: uv installs packages in parallel and skips
#    bytecode compilation by default (pip compiles every .py to .pyc on
#    install unless told not to -- a large chunk of wall-clock time for
#    source-heavy packages like torch/transformers on slow network storage).
#    musl build chosen over glibc for portability across whatever HPC's
#    actual glibc version turns out to be. Already present at bin/uv in this
#    repo (picked up by the general rsync below) -- rerun this only if it's
#    missing or you want to update it.
mkdir -p bin
curl -sL https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-musl.tar.gz \
    | tar xz --strip-components=1 -C bin --wildcards '*/uv'
chmod +x bin/uv
```

### Copying files to HPC

Find your Host alias first -- open `~/.ssh/config` and look for the block
pointing at a `*.example.invalid` address (e.g. `Host hpc`); the commands below
assume that alias is `hpc`, swap in whatever yours actually is. Confirm it
connects before moving anything:

```bash
ssh hpc echo ok
```

Then, from the `compliance-vla` repo root on your networked machine
(the wheelhouse and `.hf_cache` from steps 1-2 above should already exist
alongside it):

```bash
ssh hpc "mkdir -p compliance-vla"

# project code (skip local-only build artifacts)
rsync -avz --progress \
    --exclude .venv --exclude .git --exclude __pycache__ \
    --exclude wheelhouse --exclude .hf_cache --exclude reports/logs \
    ./ hpc:compliance-vla/

# wheelhouse and HF cache are mostly compressed binaries already -- skip -z
rsync -av --progress wheelhouse/ hpc:compliance-vla/wheelhouse/
rsync -av --progress .hf_cache/  hpc:compliance-vla/.hf_cache/
```

**Also required, and easy to miss**: `scripts/run_extraction_on_dataset.py`
imports `extract_impedance_labels` from `hardware/fr3_bilateral_teleop/dataset_tools/labeling/`.
That directory is inside the release, so the rsync above already carries it. The training
data itself (`dataset/demo4/`, per `TRAIN_SESSIONS`) is not: it is a sibling of
`compliance-vla/`, one level up, so the rsync above never touches it. Copy it separately
from the directory above `compliance-vla/`:

```bash
rsync -av --progress dataset/ hpc:dataset/     # ~3.7G, skip -z (already-compressed data)
```

This lands it as `~/dataset` on HPC, a sibling of `~/compliance-vla` -- same relative
layout as here.

`rsync` resumes cleanly if a large transfer (dataset, HF weights) drops, so
just rerun the same command rather than starting over. Check
`du -sh .hf_cache wheelhouse data/` beforehand and `ssh hpc bsc_quota` (or
whatever HPC's quota command is) if you're unsure the transfer will fit.

Then on HPC (login node, still no network needed since everything's local now):

```bash
module load python/3.12.1            # `module avail python` on HPC: 3.8/3.9/3.10/3.12, no 3.11
bin/uv venv --python "$(command -v python3.12)" .venv
source .venv/bin/activate
bin/uv pip install --no-index --offline --find-links=wheelhouse -r scripts/requirements_hpc.txt
```

`uv venv` builds a fully isolated environment (no leaking of the module's
own `/apps/.../site-packages`, which plain `python -m venv` can do on some
HPC Python modules -- see the `pyvenv.cfg` check below if unsure).
Everything after this point (activation, `python scripts/train_policy.py`,
the sbatch script) is unchanged whether the venv was built by `venv` or
`uv venv` -- `uv` is only faster/more isolated at *building* the
environment, nothing downstream cares which tool did it. Sanity-check
isolation either way with:

```bash
grep include-system-site-packages .venv/pyvenv.cfg   # want: false
```

Check first whether HPC already provides a prebuilt PyTorch module
(`module avail | grep -i pytorch`) matched to its Hopper GPUs/CUDA stack --
if so, use that instead of a generic pip wheel for `torch` and only
offline-install the rest.

### Checkpoints go on `gpfs_projects`, not `gpfs_home`

`gpfs_home` quota is tight (80GB) and the staged wheelhouse/HF cache/venv
already eat a good chunk of it. A single B2/B5 fp32 checkpoint is roughly
1.8GB, and `--checkpoint-every 5000` over a 30000-step run means several per
run -- across the full 9-run campaign (3 policies x 3 seeds) that can exceed
home's quota outright. `scripts/slurm_train_hpc.sbatch` writes checkpoints
to `$CHECKPOINT_ROOT` (default `/gpfs/projects/idri1/checkpoints` --
`gpfs_projects` has 450GB and was essentially empty at last check), not
under the repo. `gpfs_scratch`, despite being larger, is the wrong target
even though it's also empty -- HPC scratch filesystems are typically
purged on a schedule and not meant for things you want to keep.

Submit, monitor, and after the job, sync W&B from a networked machine
(the run is written offline, then pulled back):

```bash
sbatch --export=POLICY=b5,SEED=0,STEPS=30000 scripts/slurm_train_hpc.sbatch
squeue --me
# once done, from a machine with internet:
rsync -av hpc:compliance-vla/wandb ./wandb
wandb sync wandb/offline-run-*
```

### All 9 runs at once, one dedicated GPU each

`scripts/slurm_train_hpc_array.sbatch` submits all 9 (policy, seed)
combinations as a Slurm job array -- each array task is its own job with
its own `--gres=gpu:1`, so this is real parallelism across separate GPUs,
not multiple training runs sharing one card. That distinction matters:
packing concurrent fine-tunes onto a single GPU (even one with room, e.g. an
H100 at 3.6/65GB) risks OOM or serious slowdown from SM contention once
they're actually competing for compute, not just memory -- the same reason
`launch_day12_runs.sh` runs 3-deep sequentially per local GPU rather than
3-way concurrent on it.

```bash
sbatch scripts/slurm_train_hpc_array.sbatch              # all 9
sbatch --array=6-8 scripts/slurm_train_hpc_array.sbatch  # just b5, seeds 0-2
sbatch --array=0-8%3 scripts/slurm_train_hpc_array.sbatch  # cap at 3 running at once
squeue --me
```

Whether all 9 genuinely run simultaneously depends on your account's QOS
limits (concurrent jobs/GPUs per user) -- `acc_debug` and other
debug-tier QOS are often capped low. Check with `bsc_queues` or
`sacctmgr show qos acc_debug` before assuming full parallelism; use the
`%N` suffix above to self-throttle if needed rather than having jobs queue
unpredictably.

### Day 13: B4's 3 runs on HPC

`scripts/slurm_train_b4_hpc_array.sbatch` is the same pattern as the Day-12
array above, kept as its own file so resubmitting it can't accidentally
re-trigger the already-completed Day-12 9-run campaign:

```bash
sbatch scripts/slurm_train_b4_hpc_array.sbatch              # all 3 seeds
sbatch --array=1-1 scripts/slurm_train_b4_hpc_array.sbatch  # just seed 1
```

No extra offline staging needed beyond what Day 12's setup already
transferred -- B4 uses the same `lerobot[smolvla]` install (lerobot's ACT
implementation ships with it) and the same `.hf_cache` snapshot (tokenizer
vocabulary only, no VLM weights loaded -- B4 has no VLM).

### Training on all sessions (not just the frozen demo4 split)

`scripts/slurm_train_hpc_all_sessions.sbatch` is the same single-run pattern
as `slurm_train_hpc.sbatch`, but passes `--all-sessions` so the run pools
`demo1`+`demo3`+`demo4` (92 episodes) instead of just the frozen demo4 train
split (32 episodes) -- see `train_policy.py`'s `--all-sessions` help. Kept as
its own file rather than a flag on the existing scripts, since a run
launched this way has no held-out val/test and isn't comparable to the
frozen-split B0/B2/B5/B4 baselines above.

Motivating use case: the language-grounding retraining ablation in
`../language_grounding_issue_handoff.md`, which asks to retrain on "the
existing 92-episode dataset" and evaluate with
`scripts/diagnose_language_grounding.py` rather than val/test loss.

```bash
sbatch --export=POLICY=b5,SEED=0,STEPS=30000 scripts/slurm_train_hpc_all_sessions.sbatch
squeue --me
# once done, pull the checkpoint back and re-run the diagnostic per
# language_grounding_issue_handoff.md's "How to rerun the diagnostic" section
```

No new offline staging needed beyond what the Day 12 setup already
transferred -- same dataset, same venv, same `.hf_cache`.

**All 4 policies x all 3 seeds at once (12 runs):**
`scripts/slurm_train_hpc_all_sessions_array.sbatch` is the array version --
same idea as `slurm_train_hpc_array.sbatch` (B0/B2/B5, 9 runs) and
`slurm_train_b4_hpc_array.sbatch` (B4, 3 runs) combined into one 12-task
array, each task passing `--all-sessions`. Kept as its own file so
resubmitting it can't re-trigger either already-completed frozen-split
campaign. Checkpoints land under `<policy>_seed<seed>_allsessions/`, distinct
from the frozen-split runs' checkpoint dirs, so the two campaigns can't
collide or be mixed up later.

```bash
sbatch scripts/slurm_train_hpc_all_sessions_array.sbatch              # all 12
sbatch --array=9-11 scripts/slurm_train_hpc_all_sessions_array.sbatch # just b5, seeds 0-2
sbatch --array=0-11%3 scripts/slurm_train_hpc_all_sessions_array.sbatch  # cap at 3 running at once
squeue --me
```

Optional `LR` env var (e.g. `--export=LR=1e-5`) for an unfreeze-style
experiment that needs a smaller learning rate than the config default --
unfreezing itself (`train_expert_only`/`freeze_vision_encoder`, per
`language_grounding_issue_handoff.md`'s ask #1) isn't wired into
`train_policy.py` yet, so this only controls LR, not which layers train.

### Day 14: B3 (hybrid force-position output)

```bash
python scripts/train_policy.py --policy b3 --seed 0 --smoke-test   # correctness check, real data
./scripts/launch_day14_b3_runs.sh                                  # all 3 seeds, one GPU each
./scripts/launch_day14_b3_runs.sh --dry-run
```

B3 is `compliance_vla.policy.hybrid_policy.HybridSmolVLAPolicy` -- same pretrained SmolVLM2
backbone and force-injection machinery as B2/B5 (~450M params, confirmed via
`--smoke-test` against real data, `reports/b3_seed0_smoke_train_log.json`),
but log_k is decoded by a separate, one-shot regression head instead of
being folded into B5's shared flow-matching target -- the proposal's
"closest competing output parameterization" (ForceVLA2/Force Policy style).
See `src/compliance_vla/policy/hybrid_policy.py`'s module docstring for exactly how it
differs from B5, and `python -m compliance_vla.policy.hybrid_policy` for its own
synthetic-data self-test (shape/gradient-flow check, no GPU needed).

### Day 14: Gate 3 -- T1 in-distribution pilot rollouts

Three pure-Python pieces (no ROS2 needed, self-tested in this venv) plus one
real ROS2 node (needs the robot, `external/fr3_bilateral_teleop`):

```bash
python scripts/controller_frame_utils.py                # contact-frame-to-base stiffness rotation, self-tests on import
python scripts/score_ink_removal.py --self-test          # T1's ink-removal metric, §5
python scripts/fit_frozen_contact_frame.py              # real data, run once -- writes contact_frame_t1.npy
python scripts/evaluate_gate3.py --self-test
```

`scripts/fit_frozen_contact_frame.py` fits ONE frozen T1 board-normal
rotation (pooled across the real, already-collected train session) so the
rollout harness can rotate a B3/B5 checkpoint's predicted log_k (fit in the
contact frame, src/compliance_vla/policy/labels.py) into the base-frame diagonal
`target_stiffness` variable_impedance_controllers actually consumes -- see that script's
and `scripts/controller_frame_utils.py`'s docstrings for exactly what this
approximation does and does not preserve (the "Week-4 controller-integration
question" `src/compliance_vla/policy/labels.py` flags as out of scope for training; given a
first, documented pass here since a pilot rollout can't skip it). Run it
once per T1 setup freeze (Day 6); the frozen `contact_frame_t1.npy` and
`reports/contact_frame_t1.json` it writes are real, already run against
demo4 (planarity_ratio 0.075, 1242 pooled in-contact samples).

The actual rollouts (`ros2 run fr3_bilateral_teleop run_pilot_rollout.py`,
tasks.md Day 14: "n=5 per policy, sanity only") need the real rig and are
**not runnable or verified in this environment** (no rclpy install here) --
see that script's own module docstring for the full usage, the ROI-
calibration step it still needs by hand, and exactly which real-hardware
topic/message conventions it reuses (record_demo.py's pose/wrench
subscriptions, probe_variable_impedance_sinusoid.py's target_pose/
target_stiffness publishers). It calls `serve_policy.py` over HTTP (same
split as the "Real-robot evaluation" section below) and writes one JSON log
per rollout to `--output-dir` (default `/tmp/franka_teleop_pilot_rollouts`
on the robot-control machine) -- copy that directory into
`reports/pilot_rollouts/` before running:

```bash
python scripts/evaluate_gate3.py \
    --b0-logs 'reports/pilot_rollouts/b0_*.json' \
    --b5-logs 'reports/pilot_rollouts/b5_*.json'
```

which decides PASS/FAIL per tasks.md Day 14's literal gate ("B5 >= B0" on
mean targeted ink-removal, plain comparison not a CI test -- these are
sanity pilots, n=5, explicitly not meant to be reported) and, on FAIL,
prints the required next diagnostic (re-verify commanded-vs-realized
stiffness tracking, Day 5's `probe_variable_impedance_sinusoid.py`) rather
than touching the model.

## Real-robot evaluation: serving a checkpoint over HTTP

`scripts/serve_policy.py` loads one checkpoint (from `--checkpoint-dir`
above, or pulled back from HPC per the "Copying files to HPC" section) and
exposes it as an HTTP server; `scripts/client_example.py` is a minimal
client for it. Meant for the common split where the checkpoint/GPU live on
one machine and the robot-control loop runs on another, reachable over an
SSH tunnel rather than a shared network -- see `serve_policy.py`'s own
docstring for the request/response format and the `ssh -L` command.

```bash
bin/uv pip install fastapi uvicorn json-numpy   # on top of the usual venv
python scripts/serve_policy.py --checkpoint checkpoints/b5_seed0/b5_seed0_step30000.pt
```
