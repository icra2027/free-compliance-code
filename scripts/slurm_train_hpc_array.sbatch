#!/bin/bash
# All 9 Day-12 runs (3 policies x 3 seeds) on the HPC cluster's ACC partition,
# as a Slurm job array -- each array task is its own job with its own
# dedicated GPU, so this is genuine parallelism across separate H100s
# (unlike packing multiple training runs onto one GPU, which the local
# launcher (launch_day12_runs.sh) deliberately avoids -- see its header for
# why: concurrent 450M-param fine-tunes sharing one card risk OOM or heavy
# slowdown from SM contention).
#
# Submit all 9 at once:
#   sbatch scripts/slurm_train_hpc_array.sbatch
# Only run some of them (e.g. just b5 seeds 6,7,8 -> array indices 6-8):
#   sbatch --array=6-8 scripts/slurm_train_hpc_array.sbatch
# Cap how many run concurrently (e.g. at most 3 at a time) if your QOS
# limits concurrent jobs/GPUs per user -- check with `bsc_queues` or
# `sacctmgr show qos acc_debug` first, debug-tier QOS are often capped low:
#   sbatch --array=0-8%3 scripts/slurm_train_hpc_array.sbatch

#SBATCH --job-name=cvla-train-array
#SBATCH --account=idri1              # project code (not your login!) -- see `sacctmgr show assoc where user=$USER`
#SBATCH --qos=acc_debug              # this project has no acc_training QOS, only acc_debug/acc_interactive/acc_resb -- check `bsc_queues` for its time limit before a real (non-smoke) run
#SBATCH --partition=acc
#SBATCH --array=0-8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20           # HPC rule: 20 CPU per GPU requested
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00              # <= 48:00:00 for acc_training
#SBATCH --output=reports/logs/slurm_%x_%A_%a.out
#SBATCH --error=reports/logs/slurm_%x_%A_%a.err

set -euo pipefail

# index -> (policy, seed), same 3x3 layout as launch_day12_runs.sh
POLICIES=(b0 b0 b0 b2 b2 b2 b5 b5 b5)
SEEDS=(0 1 2 0 1 2 0 1 2)
POLICY="${POLICIES[$SLURM_ARRAY_TASK_ID]}"
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
STEPS="${STEPS:-30000}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/gpfs/projects/idri1/checkpoints}"  # not gpfs_home -- see scripts/README.md quota note

cd "$SLURM_SUBMIT_DIR"
mkdir -p reports/logs
mkdir -p "$CHECKPOINT_ROOT"

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
unset PYTHONPATH

# HF_HOME is NOT the cache dir itself -- huggingface_hub resolves the real
# cache path as $HF_HOME/hub unless HF_HUB_CACHE is set explicitly. The
# snapshot_download() that built .hf_cache used cache_dir='.hf_cache'
# directly (flat models--.../snapshots/... layout, no hub/ subdir), so
# HF_HUB_CACHE is the one that actually has to point at it.
export HF_HUB_CACHE="$SLURM_SUBMIT_DIR/.hf_cache"   # populated ahead of time, see README
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline                       # sync later with `wandb sync` from a networked machine

echo "[slurm] node=$SLURMD_NODENAME job=$SLURM_JOB_ID array_task=$SLURM_ARRAY_TASK_ID policy=$POLICY seed=$SEED steps=$STEPS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python scripts/train_policy.py \
    --policy "$POLICY" --seed "$SEED" --steps "$STEPS" \
    --checkpoint-dir "$CHECKPOINT_ROOT/${POLICY}_seed${SEED}" \
    --wandb --wandb-run-name "${POLICY}_seed${SEED}_hpc_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
