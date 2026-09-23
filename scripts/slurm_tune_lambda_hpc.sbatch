#!/bin/bash
# Day 12: tune lambda (compliance/log_k loss weight, proposal §4.2) on the
# validation split, on the HPC cluster's ACC (GPU) partition.
# Docs: <hpc-site-documentation>
#
# This runs scripts/tune_lambda.py, which loops over every --lambdas
# candidate *inside one process* (train demo4 for --tune-steps, eval on
# demo1, repeat) -- so this only needs a single GPU/single task, unlike the
# 9-way B0/B2/B5 campaign in slurm_train_hpc_array.sbatch. Same offline
# staging requirement as that script: HPC compute nodes have no internet,
# so the wheelhouse / .hf_cache / .venv must already be on disk (see
# scripts/README.md "Running on the HPC cluster (offline)").
#
# Usage:
#   sbatch scripts/slurm_tune_lambda_hpc.sbatch
#   sbatch --export=LAMBDAS="0.1 0.3 1.0 3.0 10.0",TUNE_STEPS=2000 scripts/slurm_tune_lambda_hpc.sbatch
#   sbatch --export=LAMBDAS="0.1 1.0 10.0",TUNE_STEPS=500,SMOKE_EPISODES=6 scripts/slurm_tune_lambda_hpc.sbatch   # quick check first

#SBATCH --job-name=cvla-tune-lambda
#SBATCH --account=idri1              # project code (not your login!) -- see `sacctmgr show assoc where user=$USER`
#SBATCH --qos=acc_resb              # this project has no acc_training QOS, only acc_debug/acc_interactive/acc_resb -- check `bsc_queues` for its time limit before a real (non-smoke) sweep
#SBATCH --partition=acc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20           # HPC rule: 20 CPU per GPU requested
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00              # tune_lambda.py trains len(LAMBDAS) short models sequentially on one GPU; widen if you pass more candidates or a larger TUNE_STEPS
#SBATCH --output=reports/logs/slurm_%x_%j.out
#SBATCH --error=reports/logs/slurm_%x_%j.err

set -euo pipefail

LAMBDAS="${LAMBDAS:-0.1 0.3 1.0 3.0 10.0}"
TUNE_STEPS="${TUNE_STEPS:-2000}"
SEED="${SEED:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SMOKE_EPISODES="${SMOKE_EPISODES:-}"      # set to e.g. 6 for a quick correctness check before the real sweep; leave unset for the real sweep

cd "$SLURM_SUBMIT_DIR"
mkdir -p reports/logs reports

# --- environment -----------------------------------------------------------
# HPC has NO internet access on login or compute nodes (the HPC site support docs:
# only glogin4/alogin4 are networked, and those are the HPC site-staff-only + VPN).
# Everything below must already be staged on the filesystem before this job
# runs -- see scripts/README.md "Running on the HPC cluster (offline)" for how
# to build the wheelhouse and HF cache on a machine that has internet and
# scp them over. Nothing in this script may reach out to the network.
module load intel
module load impi
module load hdf5
module load mkl
module load python/3.12.1            # matches the wheelhouse built for cp312, see scripts/README.md
# module load cuda/...                 # adjust once you've checked `module avail cuda`
source .venv/bin/activate            # built offline from the transferred wheelhouse
unset PYTHONPATH                     # the python/3.12.1 module otherwise leaks its own older huggingface-hub ahead of the pinned venv copy

# HF_HOME is NOT the cache dir itself -- huggingface_hub resolves the real
# cache path as $HF_HOME/hub unless HF_HUB_CACHE is set explicitly. The
# snapshot_download() that built .hf_cache used cache_dir='.hf_cache'
# directly (flat models--.../snapshots/... layout, no hub/ subdir), so
# HF_HUB_CACHE is the one that actually has to point at it.
export HF_HUB_CACHE="$SLURM_SUBMIT_DIR/.hf_cache"   # populated ahead of time, see README
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "[slurm] node=$SLURMD_NODENAME job=$SLURM_JOB_ID lambdas=($LAMBDAS) tune_steps=$TUNE_STEPS seed=$SEED smoke_episodes=${SMOKE_EPISODES:-<none, real sweep>}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

EXTRA_ARGS=()
if [[ -n "$SMOKE_EPISODES" ]]; then
    EXTRA_ARGS+=(--smoke-episodes "$SMOKE_EPISODES")
fi

python scripts/tune_lambda.py \
    --lambdas $LAMBDAS \
    --tune-steps "$TUNE_STEPS" \
    --seed "$SEED" \
    --batch-size "$BATCH_SIZE" \
    --out-dir reports \
    "${EXTRA_ARGS[@]}"

echo "[slurm] done -- see reports/lambda_tuning.json for the per-lambda val losses and the selected best lambda"
