#!/bin/bash
# One policy/seed training run on the HPC cluster's ACC (GPU) partition,
# pooling ALL collected sessions (demo1+demo3+demo4, 92 episodes) instead of
# just the frozen demo4 train split -- scripts/train_policy.py's
# --all-sessions flag. Same pattern as scripts/slurm_train_hpc.sbatch (that
# one is left untouched, still defaulting to the frozen split, so existing
# baseline-comparison runs aren't affected), kept as its own file since a
# run launched with --all-sessions has no held-out val/test and is not
# comparable to the frozen-split B0/B2/B5/B4 baselines.
#
# Motivating use case: the language-grounding retraining ablation in
# ../language_grounding_issue_handoff.md, which asks to retrain on "the
# existing 92-episode dataset" (not demo4's 32) and evaluate with
# scripts/diagnose_language_grounding.py rather than val/test loss.
# Docs: <hpc-site-documentation>
#
# Fill in --account below (find yours with `bsc_project list`), then:
#   sbatch --export=POLICY=b5,SEED=0,STEPS=30000 scripts/slurm_train_hpc_all_sessions.sbatch
# or edit POLICY/SEED/STEPS defaults and just `sbatch scripts/slurm_train_hpc_all_sessions.sbatch`.

#SBATCH --job-name=cvla-train-allsess
#SBATCH --account=idri1              # project code (not your login!) -- see `sacctmgr show assoc where user=$USER`
#SBATCH --qos=acc_debug              # this project has no acc_training QOS, only acc_debug/acc_interactive/acc_resb -- check `bsc_queues` for its time limit before a real (non-smoke) run
#SBATCH --partition=acc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20           # HPC rule: 20 CPU per GPU requested
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00              # <= 48:00:00 for acc_training
#SBATCH --output=reports/logs/slurm_%x_%j.out
#SBATCH --error=reports/logs/slurm_%x_%j.err

set -euo pipefail

POLICY="${POLICY:-b5}"
SEED="${SEED:-0}"
STEPS="${STEPS:-30000}"
LR="${LR:-}"                          # optional override, e.g. for an unfreeze experiment's smaller LR
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

echo "[slurm] node=$SLURMD_NODENAME job=$SLURM_JOB_ID policy=$POLICY seed=$SEED steps=$STEPS all_sessions=1 lr=${LR:-<config default>}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

EXTRA_ARGS=()
if [[ -n "$LR" ]]; then
  EXTRA_ARGS+=(--lr "$LR")
fi

python scripts/train_policy.py \
    --policy "$POLICY" --seed "$SEED" --steps "$STEPS" --all-sessions \
    --checkpoint-dir "$CHECKPOINT_ROOT/${POLICY}_seed${SEED}_allsessions" \
    --wandb --wandb-run-name "${POLICY}_seed${SEED}_allsessions_hpc_${SLURM_JOB_ID}" \
    "${EXTRA_ARGS[@]}"
