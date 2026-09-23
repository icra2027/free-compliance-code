#!/bin/bash
# Fold a --lora checkpoint's adapters into a plain, serve_policy.py-ready
# checkpoint on the HPC cluster, via scripts/merge_lora_checkpoint.py.
# Docs: <hpc-site-documentation>
#
# merge_lora_checkpoint.py never touches a GPU (torch.load(map_location=
# "cpu"), peft's merge_and_unload() runs on whatever device the tensors are
# already on) -- this only needs the acc partition because that's the only
# QOS/account path this project has verified against on HPC, not because
# the work itself needs a GPU. Runs in seconds either way.
#
# Same offline staging requirement as every other HPC script here: HPC
# compute nodes have no internet, so .venv/wheelhouse/.hf_cache must already
# be on disk (see scripts/README.md "Running on the HPC cluster (offline)").
# No new staging needed for this one -- peft==0.20.0 is already pinned in
# scripts/requirements_hpc.txt / staged in wheelhouse/.
#
# Usage -- CHECKPOINT is required, OUT_PATH defaults to
# <checkpoint>_merged.pt next to it if not given:
#   sbatch --export=CHECKPOINT=/gpfs/projects/idri1/checkpoints/b0_seed0_allsessions_lora/b0_seed0_step30000.pt \
#       scripts/slurm_merge_lora_hpc.sbatch
#   sbatch --export=CHECKPOINT=...,OUT_PATH=/gpfs/projects/idri1/checkpoints/b0_merged.pt \
#       scripts/slurm_merge_lora_hpc.sbatch
#
# Then serve the merged checkpoint exactly like any other:
#   python scripts/serve_policy.py --checkpoint <OUT_PATH>

#SBATCH --job-name=cvla-merge-lora
#SBATCH --account=idri1              # project code (not your login!) -- see `sacctmgr show assoc where user=$USER`
#SBATCH --qos=acc_debug              # this project has no acc_training QOS, only acc_debug/acc_interactive/acc_resb -- check `bsc_queues` for its time limit
#SBATCH --partition=acc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20           # HPC rule: 20 CPU per GPU requested (see header -- the GPU itself goes unused)
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00              # a state-dict load + merge_and_unload() takes seconds; generous margin for GPFS cold-cache reads
#SBATCH --output=reports/logs/slurm_%x_%j.out
#SBATCH --error=reports/logs/slurm_%x_%j.err

set -euo pipefail

if [[ -z "${CHECKPOINT:-}" ]]; then
  echo "[slurm] ERROR: CHECKPOINT is required, e.g. --export=CHECKPOINT=/gpfs/projects/idri1/checkpoints/b0_seed0_allsessions_lora/b0_seed0_step30000.pt" >&2
  exit 1
fi
OUT_PATH="${OUT_PATH:-${CHECKPOINT%.pt}_merged.pt}"

cd "$SLURM_SUBMIT_DIR"
mkdir -p reports/logs

# --- environment -----------------------------------------------------------
# Identical to every other HPC script here -- see scripts/README.md "Running
# on the HPC cluster (offline)" for the full rationale (no internet on HPC
# compute/login nodes, the python/3.12.1 module's own PYTHONPATH leaking
# ahead of the pinned venv copy, etc).
module load intel
module load impi
module load hdf5
module load mkl
module load python/3.12.1            # matches the wheelhouse built for cp312, see scripts/README.md
source .venv/bin/activate            # built offline from the transferred wheelhouse
unset PYTHONPATH

export HF_HUB_CACHE="$SLURM_SUBMIT_DIR/.hf_cache"   # populated ahead of time, see README
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "[slurm] node=$SLURMD_NODENAME job=$SLURM_JOB_ID checkpoint=$CHECKPOINT out_path=$OUT_PATH"

python scripts/merge_lora_checkpoint.py "$CHECKPOINT" "$OUT_PATH"

echo "[slurm] done -- serve with: python scripts/serve_policy.py --checkpoint $OUT_PATH"
