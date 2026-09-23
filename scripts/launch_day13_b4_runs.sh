#!/usr/bin/env bash
# Day 13: launch the 3 real B4 training runs -- ACT/Bi-ACT from scratch,
# seeds {0,1,2} -- via scripts/train_policy.py --policy b4. This is the
# "B4: ACT/Bi-ACT from scratch, identical data and output space, 3 seeds.
# This is the H4 control and cannot be skipped." task / "Exit: B1 and B4
# running" criterion's B4 half. Same spirit as launch_day12_runs.sh (B0/B2/B5)
# -- separate script, not folded into that one, since B4 was added a day
# later and shouldn't risk re-triggering already-completed Day 12 runs.
#
# B4 has no pretrained VLM (src/compliance_vla/policy/bi_act_policy.py -- that's the point,
# it's the H4 from-scratch control) so it is far cheaper than B0/B2/B5:
# ~55M params total (see the real smoke-test log, reports/b4_seed0_smoke_train_log.json)
# vs. SmolVLA's ~450M. One seed comfortably fits on one GPU alone, so this
# launches 3 seeds on 3 GPUs one-job-per-GPU (no need for launch_day12_runs.sh's
# 3-deep sequential packing -- there's nothing to pack, only 3 jobs total).
#
# Usage:
#   ./scripts/launch_day13_b4_runs.sh                  # all 3 seeds, default --steps (30000) each
#   ./scripts/launch_day13_b4_runs.sh --steps 5000      # shorter real run
#   ./scripts/launch_day13_b4_runs.sh --dry-run         # print the 3 commands + GPU assignment, launch nothing
#   ./scripts/launch_day13_b4_runs.sh --checkpoint-dir reports/checkpoints --checkpoint-every 5000
#   ./scripts/launch_day13_b4_runs.sh --all-sessions    # pool demo1+demo3+demo4 (all 92 episodes) for
#                                                        # training instead of just the frozen demo4
#                                                        # train split -- see train_policy.py's
#                                                        # --all-sessions help. No held-out val/test
#                                                        # left for runs launched this way.
#
# Any extra flags are forwarded verbatim to every one of the 3
# `train_policy.py --policy b4` invocations (e.g. --lr, --batch-size, --lam,
# --all-sessions, --train-sessions demo1 demo3 demo4).
#
# Logs:      reports/logs/b4_seed<seed>.log
# Job list:  reports/logs/day13_b4_jobs.txt   (gpu seed pid)
#
# Monitor:   tail -f reports/logs/b4_seed0.log
# Stop all:  kill $(awk '{print $NF}' reports/logs/day13_b4_jobs.txt)

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
# Overridable via GPUS env var (space-separated), e.g. GPUS="1 2 3" ./scripts/launch_day13_b4_runs.sh
# -- default 0/1/2 assumes an idle 3-GPU box; check `nvidia-smi` first if other jobs may already be running.
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

: > "$LOG_DIR/day13_b4_jobs.txt"

# Default to a per-seed checkpoint dir (checkpoints/b4_seed<seed>/), matching the
# checkpoints/<policy>_seed<seed>/<policy>_seed<seed>_step<N>.pt convention Day 12's
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
  log_file="$LOG_DIR/b4_seed${seed}.log"
  job_args=("${EXTRA_ARGS[@]}")
  if [[ "$user_gave_checkpoint_dir" -eq 0 ]]; then
    job_args+=(--checkpoint-dir "checkpoints/b4_seed${seed}")
  fi

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "  CUDA_VISIBLE_DEVICES=$gpu $VENV_PYTHON scripts/train_policy.py --policy b4 --seed $seed ${job_args[*]}"
    continue
  fi

  echo "[launch_day13_b4_runs] GPU $gpu starting b4 seed $seed -> $log_file"
  CUDA_VISIBLE_DEVICES="$gpu" "$VENV_PYTHON" scripts/train_policy.py \
    --policy b4 --seed "$seed" "${job_args[@]}" > "$log_file" 2>&1 &
  echo "$gpu $seed $!" >> "$LOG_DIR/day13_b4_jobs.txt"
done

if [[ "$DRY_RUN" -eq 0 ]]; then
  echo "[launch_day13_b4_runs] 3 B4 jobs launched, one per GPU (${GPUS[*]})."
  echo "[launch_day13_b4_runs] job list: $LOG_DIR/day13_b4_jobs.txt"
  echo "[launch_day13_b4_runs] watch one:  tail -f $LOG_DIR/b4_seed0.log"
  echo "[launch_day13_b4_runs] stop all:   kill \$(awk '{print \$NF}' $LOG_DIR/day13_b4_jobs.txt)"
fi
