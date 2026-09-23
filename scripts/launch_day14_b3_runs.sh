#!/usr/bin/env bash
# Day 14: launch the 3 real B3 training runs -- force input, HYBRID
# force-position output (src/compliance_vla/policy/hybrid_policy.py), seeds {0,1,2} -- via
# scripts/train_policy.py --policy b3. This is the "B3: force input + hybrid
# force-position output (ForceVLA2 / Force Policy parameterization)" task.
# Same spirit as launch_day13_b4_runs.sh -- separate script, not folded into
# launch_day12_runs.sh, since B3 was added two days later and shouldn't risk
# re-triggering the already-completed Day 12 9-run campaign.
#
# B3 has the full pretrained SmolVLM2 backbone (~450M params, same as
# B2/B5 -- see reports/b3_seed0_smoke_train_log.json), unlike B4's from-scratch
# ~55M. Only 3 jobs total though, so -- same reasoning launch_day13_b4_runs.sh
# already used for B4 -- there is nothing to pack: one seed per GPU, one
# dedicated GPU each, no 3-deep sequential queueing needed the way
# launch_day12_runs.sh's 9 (3 policies x 3 seeds) required.
#
# Usage:
#   ./scripts/launch_day14_b3_runs.sh                  # all 3 seeds, default --steps (30000) each
#   ./scripts/launch_day14_b3_runs.sh --steps 5000      # shorter real run
#   ./scripts/launch_day14_b3_runs.sh --dry-run         # print the 3 commands + GPU assignment, launch nothing
#   GPUS="1 2 3" ./scripts/launch_day14_b3_runs.sh      # override which GPUs -- check nvidia-smi first
#
# Any extra flags are forwarded verbatim to every one of the 3
# `train_policy.py --policy b3` invocations (e.g. --lr, --batch-size, --lam).
#
# Logs:      reports/logs/b3_seed<seed>.log
# Job list:  reports/logs/day14_b3_jobs.txt   (gpu seed pid)
#
# Monitor:   tail -f reports/logs/b3_seed0.log
# Stop all:  kill $(awk '{print $NF}' reports/logs/day14_b3_jobs.txt)

set -euo pipefail
cd "$(dirname "$0")/.."   # compliance-vla/

VENV_PYTHON=".venv/bin/python3"
if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "error: $VENV_PYTHON not found -- run this from compliance-vla/, after" >&2
  echo "  uv venv --python 3.11 .venv && source .venv/bin/activate && uv pip install 'lerobot[smolvla]' scipy scikit-learn matplotlib" >&2
  exit 1
fi

LOG_DIR="reports/logs"
mkdir -p "$LOG_DIR"

SEEDS=(0 1 2)
# Overridable via GPUS env var (space-separated), e.g. GPUS="1 2 3" ./scripts/launch_day14_b3_runs.sh
# -- default 0/1/2 assumes an idle 3-GPU box; check `nvidia-smi` first, since Day 12/13's
# runs may still be occupying some of them.
read -r -a GPUS <<< "${GPUS:-0 1 2}"

DRY_RUN=0
EXTRA_ARGS=()
for arg in "$@"; do
  if [[ "$arg" == "--dry-run" ]]; then
    DRY_RUN=1
  else
    EXTRA_ARGS+=("$arg")
  fi
done

: > "$LOG_DIR/day14_b3_jobs.txt"

# Default to a per-seed checkpoint dir (checkpoints/b3_seed<seed>/), matching the
# checkpoints/<policy>_seed<seed>/<policy>_seed<seed>_step<N>.pt convention Day 12/13's
# real runs already established -- unless the caller already passed their own
# --checkpoint-dir, which would otherwise collide across all 3 concurrent seeds
# (train_policy.py writes <checkpoint_dir>/<policy>_seed<seed>_step<N>.pt directly,
# no per-run subdirectory of its own).
user_gave_checkpoint_dir=0
for arg in "${EXTRA_ARGS[@]}"; do
  [[ "$arg" == "--checkpoint-dir" || "$arg" == --checkpoint-dir=* ]] && user_gave_checkpoint_dir=1
done

for i in "${!SEEDS[@]}"; do
  seed="${SEEDS[$i]}"
  gpu="${GPUS[$i]}"
  log_file="$LOG_DIR/b3_seed${seed}.log"
  job_args=("${EXTRA_ARGS[@]}")
  if [[ "$user_gave_checkpoint_dir" -eq 0 ]]; then
    job_args+=(--checkpoint-dir "checkpoints/b3_seed${seed}")
  fi

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "  CUDA_VISIBLE_DEVICES=$gpu $VENV_PYTHON scripts/train_policy.py --policy b3 --seed $seed ${job_args[*]}"
    continue
  fi

  echo "[launch_day14_b3_runs] GPU $gpu starting b3 seed $seed -> $log_file"
  CUDA_VISIBLE_DEVICES="$gpu" "$VENV_PYTHON" scripts/train_policy.py \
    --policy b3 --seed "$seed" "${job_args[@]}" > "$log_file" 2>&1 &
  echo "$gpu $seed $!" >> "$LOG_DIR/day14_b3_jobs.txt"
done

if [[ "$DRY_RUN" -eq 0 ]]; then
  echo "[launch_day14_b3_runs] 3 B3 jobs launched, one per GPU (${GPUS[*]})."
  echo "[launch_day14_b3_runs] job list: $LOG_DIR/day14_b3_jobs.txt"
  echo "[launch_day14_b3_runs] watch one:  tail -f $LOG_DIR/b3_seed0.log"
  echo "[launch_day14_b3_runs] stop all:   kill \$(awk '{print \$NF}' $LOG_DIR/day14_b3_jobs.txt)"
fi
