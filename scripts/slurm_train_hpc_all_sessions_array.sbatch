#!/bin/bash
# Rerun all 4 policies (B0/B2/B4/B5) x seeds {0,1,2} = 12 runs on
# the HPC cluster's ACC partition, pooling ALL collected sessions
# (demo1+demo3+demo4, 92 episodes) via --all-sessions instead of the frozen
# demo4 train split -- same array-job pattern as slurm_train_hpc_array.sbatch
# (Day 12's B0/B2/B5 9-run campaign) and slurm_train_b4_hpc_array.sbatch
# (Day 13's B4 3-run campaign), combined into one array and re-pointed at
# --all-sessions. Kept as its own file, not folded into either of those, so
# resubmitting this can't accidentally re-trigger the already-completed
# frozen-split campaigns -- same reasoning slurm_train_b4_hpc_array.sbatch's
# header already gives for staying separate from slurm_train_hpc_array.sbatch.
#
# Runs launched this way have no held-out val/test (--all-sessions pools the
# val/test sessions into train too) and are NOT comparable to the
# frozen-split B0/B2/B4/B5 baselines from Day 12/13 -- see train_policy.py's
# --all-sessions help and scripts/slurm_train_hpc_all_sessions.sbatch (the
# single-run version of this same idea). Motivating use case: the
# language-grounding retraining ablation in
# ../language_grounding_issue_handoff.md, which asks to retrain on "the
# existing 92-episode dataset" across seeds, not just demo4's 32 on one seed.
#
# Submit all 12:
#   sbatch scripts/slurm_train_hpc_all_sessions_array.sbatch
# Just B5, seeds 0-2 (array indices 9-11):
#   sbatch --array=9-11 scripts/slurm_train_hpc_all_sessions_array.sbatch
# Cap concurrent jobs (e.g. at most 3 at a time) if your QOS limits it --
# check with `bsc_queues` or `sacctmgr show qos acc_debug` first:
#   sbatch --array=0-11%3 scripts/slurm_train_hpc_all_sessions_array.sbatch

#SBATCH --job-name=cvla-train-allsess-array
#SBATCH --account=idri1              # project code (not your login!) -- see `sacctmgr show assoc where user=$USER`
#SBATCH --qos=acc_debug              # this project has no acc_training QOS, only acc_debug/acc_interactive/acc_resb -- check `bsc_queues` for its time limit before a real (non-smoke) run
#SBATCH --partition=acc
#SBATCH --array=0-11
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20           # HPC rule: 20 CPU per GPU requested
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00              # <= 48:00:00 for acc_training
#SBATCH --output=reports/logs/slurm_%x_%A_%a.out
#SBATCH --error=reports/logs/slurm_%x_%A_%a.err

set -euo pipefail

# index -> (policy, seed), 4 policies x 3 seeds, same seed ordering as the
# two campaigns this combines (slurm_train_hpc_array.sbatch's b0/b2/b5,
# slurm_train_b4_hpc_array.sbatch's b4).
POLICIES=(b0 b0 b0 b2 b2 b2 b4 b4 b4 b5 b5 b5)
SEEDS=(0 1 2 0 1 2 0 1 2 0 1 2)
POLICY="${POLICIES[$SLURM_ARRAY_TASK_ID]}"
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
STEPS="${STEPS:-30000}"
LR="${LR:-}"                          # optional override, e.g. for an unfreeze experiment's smaller LR
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/gpfs/projects/idri1/checkpoints}"  # not gpfs_home -- see scripts/README.md quota note

cd "$SLURM_SUBMIT_DIR"
mkdir -p reports/logs
mkdir -p "$CHECKPOINT_ROOT"

# --- environment -----------------------------------------------------------
# Identical to slurm_train_hpc_array.sbatch -- see that file (and
# scripts/README.md "Running on the HPC cluster (offline)") for the full
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

export HF_HUB_CACHE="$SLURM_SUBMIT_DIR/.hf_cache"   # populated ahead of time, see README -- unused by b4 (no VLM) beyond its tokenizer
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline                       # sync later with `wandb sync` from a networked machine

echo "[slurm] node=$SLURMD_NODENAME job=$SLURM_JOB_ID array_task=$SLURM_ARRAY_TASK_ID policy=$POLICY seed=$SEED steps=$STEPS all_sessions=1 lr=${LR:-<config default>}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

EXTRA_ARGS=()
if [[ -n "$LR" ]]; then
  EXTRA_ARGS+=(--lr "$LR")
fi

python scripts/train_policy.py \
    --policy "$POLICY" --seed "$SEED" --steps "$STEPS" --all-sessions \
    --checkpoint-dir "$CHECKPOINT_ROOT/${POLICY}_seed${SEED}_allsessions" \
    --wandb --wandb-run-name "${POLICY}_seed${SEED}_allsessions_hpc_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" \
    "${EXTRA_ARGS[@]}"
