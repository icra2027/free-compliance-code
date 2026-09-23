#!/bin/bash
# Local (no Slurm) counterpart to slurm_train_hpc_overfit_check.sbatch -- same
# language-grounding overfit sanity check (language_grounding_issue_handoff.md's
# "Overfit sanity check" section: deliberately overfit on a tiny, colour-balanced
# episode subset to tell "wiring/capacity bug" from "genuinely needs more data"
# before committing to a new collection campaign). For a machine with a GPU
# already attached and normal internet access -- no HPC module loads, offline
# HF cache, or gpfs checkpoint paths needed here.
#
# Usage:
#   scripts/run_overfit_check_local.sh                     # defaults: b0 seed0, full unfreeze, 3 episodes/colour/session, GPU 0
#   GPU=3 scripts/run_overfit_check_local.sh                # pick a specific GPU -- see `nvidia-smi` for a free one
#   N_PER_REFERENT=2 scripts/run_overfit_check_local.sh
#   POLICY=b5 SEED=0 scripts/run_overfit_check_local.sh
#   UNFREEZE_VISION=0 scripts/run_overfit_check_local.sh    # isolate the language-side variable only (matches the
#                                                            # already-run unfreezelm-only checkpoint's config)
#   WANDB=1 scripts/run_overfit_check_local.sh              # also log to Weights & Biases (off by default here --
#                                                            # this machine isn't preconfigured for offline sync like HPC)
#
# How to read the result: train_policy.py prints the exact (session,
# episode_index) subset it picked right after loading the dataset -- also
# saved in this run's log under reports/logs/. Then rerun
# scripts/diagnose_language_grounding.py against the resulting checkpoint,
# restricted to ONLY those episodes (--episode is scoped to one --session at
# a time, so one invocation per session that appears in the picked set):
#   python scripts/serve_policy.py --checkpoint <this run's checkpoint>.pt
#   python3 scripts/diagnose_language_grounding.py --session demo1 --episode <indices from the log> --n-seeds 8
#   (repeat per session -- demo3/demo4 -- for whichever indices were picked from each)
# - Gets those specific training episodes right -> the language pathway does
#   carry colour information end-to-end; the earlier chance-level result is a
#   data-volume/generalization problem (supports the handoff's item 2, new
#   data collection).
# - Still can't get its own training episodes right -> a wiring/capacity bug
#   upstream of data volume, worth chasing down before any new collection
#   campaign (more real-robot data would not fix this).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

GPU="${GPU:-0}"
POLICY="${POLICY:-b0}"
SEED="${SEED:-0}"
N_PER_REFERENT="${N_PER_REFERENT:-3}"     # episodes per colour per session -- see train_policy.py --overfit-n-per-referent
STEPS="${STEPS:-2000}"
LR="${LR:-1e-5}"                          # small on purpose -- single global LR hits the unfrozen pretrained backbone too
BATCH_SIZE="${BATCH_SIZE:-4}"             # a colour-balanced 3/referent/session subset may be well under 8 episodes' worth
                                           # of windows -- bump back to 8 if the printed window count supports it
UNFREEZE_LM="${UNFREEZE_LM:-1}"           # matches both checkpoints already evaluated at chance -- see handoff doc
UNFREEZE_VISION="${UNFREEZE_VISION:-1}"
WANDB="${WANDB:-0}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$REPO_ROOT/checkpoints}"

EXTRA_ARGS=()
RUN_TAG="overfit_n${N_PER_REFERENT}"
if [[ "$UNFREEZE_LM" == "1" ]]; then
  EXTRA_ARGS+=(--unfreeze-lm)
  RUN_TAG="${RUN_TAG}_unfreezelm"
fi
if [[ "$UNFREEZE_VISION" == "1" ]]; then
  EXTRA_ARGS+=(--unfreeze-vision)
  RUN_TAG="${RUN_TAG}_unfreezevision"
fi
if [[ "$WANDB" == "1" ]]; then
  EXTRA_ARGS+=(--wandb --wandb-run-name "${POLICY}_seed${SEED}_${RUN_TAG}_local")
fi

mkdir -p reports/logs "$CHECKPOINT_ROOT"
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES="$GPU"

echo "[run_overfit_check_local] gpu=$GPU policy=$POLICY seed=$SEED steps=$STEPS n_per_referent=$N_PER_REFERENT unfreeze_lm=$UNFREEZE_LM unfreeze_vision=$UNFREEZE_VISION lr=$LR batch_size=$BATCH_SIZE wandb=$WANDB"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader | awk -v g="$GPU" -F',' '$1+0==g'

LOG_PATH="reports/logs/${POLICY}_seed${SEED}_${RUN_TAG}_local.log"
python scripts/train_policy.py --wandb \
    --policy "$POLICY" --seed "$SEED" --steps "$STEPS" --batch-size "$BATCH_SIZE" --device cuda \
    --all-sessions --overfit-n-per-referent "$N_PER_REFERENT" --lr "$LR" \
    --checkpoint-dir "$CHECKPOINT_ROOT/${POLICY}_seed${SEED}_${RUN_TAG}" --checkpoint-every "$STEPS"  \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "$LOG_PATH"

echo "[run_overfit_check_local] done -- log: $LOG_PATH, checkpoint: $CHECKPOINT_ROOT/${POLICY}_seed${SEED}_${RUN_TAG}/"
