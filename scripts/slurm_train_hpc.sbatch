#!/bin/bash
# One policy/seed training run on the HPC cluster's ACC (GPU) partition.
# Docs: <hpc-site-documentation>
#
# Fill in --account below (find yours with `bsc_project list`), then:
#   sbatch --export=POLICY=b5,SEED=0,STEPS=30000 scripts/slurm_train_hpc.sbatch
# or edit POLICY/SEED/STEPS defaults and just `sbatch scripts/slurm_train_hpc.sbatch`.

#SBATCH --job-name=cvla-train
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

echo "[slurm] node=$SLURMD_NODENAME job=$SLURM_JOB_ID policy=$POLICY seed=$SEED steps=$STEPS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python scripts/train_policy.py \
    --policy "$POLICY" --seed "$SEED" --steps "$STEPS" \
    --checkpoint-dir "$CHECKPOINT_ROOT/${POLICY}_seed${SEED}" \
    --wandb --wandb-run-name "${POLICY}_seed${SEED}_hpc_${SLURM_JOB_ID}"
