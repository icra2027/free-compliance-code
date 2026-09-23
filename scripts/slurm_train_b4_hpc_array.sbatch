#!/bin/bash
# Day 13: the 3 B4 runs (ACT/Bi-ACT from scratch, seeds 0/1/2) on
# the HPC cluster's ACC partition, as a Slurm job array -- same pattern as
# slurm_train_hpc_array.sbatch (Day 12's B0/B2/B5 array), kept as its own
# file rather than folded into that one so resubmitting Day 13's work can't
# accidentally re-trigger Day 12's already-completed 9 runs (see that
# script's own array, and tasks.md Day 12's "all 9 ... ran the full 30000
# steps" update).
#
# B4 needs no pretrained-VLM staging (src/compliance_vla/policy/bi_act_policy.py has no VLM
# at all -- that's the H4 manipulation) -- the same offline wheelhouse/venv
# already staged for Day 12 covers it (lerobot's ACT implementation is part
# of the same `lerobot[smolvla]` install), and the tokenizer it reuses
# (vocabulary only, not weights, see that module's docstring) comes from the
# same .hf_cache snapshot already transferred for SmolVLA. No new staging
# step vs. scripts/README.md's existing HPC setup.
#
# Submit all 3:
#   sbatch scripts/slurm_train_b4_hpc_array.sbatch
# Just one seed (e.g. seed 1 -> array index 1):
#   sbatch --array=1-1 scripts/slurm_train_b4_hpc_array.sbatch

#SBATCH --job-name=cvla-train-b4
#SBATCH --account=idri1              # project code (not your login!) -- see `sacctmgr show assoc where user=$USER`
#SBATCH --qos=acc_debug              # this project has no acc_training QOS, only acc_debug/acc_interactive/acc_resb -- check `bsc_queues` for its time limit before a real (non-smoke) run
#SBATCH --partition=acc
#SBATCH --array=0-2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20           # HPC rule: 20 CPU per GPU requested
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00              # <= 48:00:00 for acc_training -- B4 is much smaller than B0/B2/B5 (see header), likely finishes well inside this
#SBATCH --output=reports/logs/slurm_%x_%A_%a.out
#SBATCH --error=reports/logs/slurm_%x_%A_%a.err

set -euo pipefail

SEEDS=(0 1 2)
POLICY="b4"
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
STEPS="${STEPS:-30000}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/gpfs/projects/idri1/checkpoints}"  # not gpfs_home -- see scripts/README.md quota note

cd "$SLURM_SUBMIT_DIR"
mkdir -p reports/logs
mkdir -p "$CHECKPOINT_ROOT"

# --- environment -----------------------------------------------------------
# Identical to slurm_train_hpc_array.sbatch -- see that file for the full
# rationale (no internet on HPC compute/login nodes, HF_HUB_CACHE vs
# HF_HOME, etc). Kept in sync deliberately rather than sourced from a shared
# fragment, matching this project's existing sbatch-script convention.
module load intel
module load impi
module load hdf5
module load mkl
module load python/3.12.1            # matches the wheelhouse built for cp312, see scripts/README.md
# module load cuda/...                 # adjust once you've checked `module avail cuda`
source .venv/bin/activate            # built offline from the transferred wheelhouse
unset PYTHONPATH

export HF_HUB_CACHE="$SLURM_SUBMIT_DIR/.hf_cache"   # populated ahead of time, see README -- only used here for the tokenizer, no VLM weights loaded (B4 has no VLM)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline                       # sync later with `wandb sync` from a networked machine

echo "[slurm] node=$SLURMD_NODENAME job=$SLURM_JOB_ID array_task=$SLURM_ARRAY_TASK_ID policy=$POLICY seed=$SEED steps=$STEPS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python scripts/train_policy.py \
    --policy "$POLICY" --seed "$SEED" --steps "$STEPS" \
    --checkpoint-dir "$CHECKPOINT_ROOT/${POLICY}_seed${SEED}" \
    --wandb --wandb-run-name "${POLICY}_seed${SEED}_hpc_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
