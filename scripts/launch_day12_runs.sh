#!/usr/bin/env bash
# Day 12: launch all 9 real training runs -- B0, B2, B5 x seeds {0,1,2} --
# via scripts/train_policy.py. This is the actual "Launch B0, B2, B5 -- 3
# seeds each" / "Exit: 9 runs in flight" task; nothing else in this repo
# calls it automatically -- when to spend the GPU allocation Day 1 confirmed
# is this project's own call, not something a training script should decide
# on its own.
#
# GPU plan: 3x L40S (confirmed idle on Day 11, `nvidia-smi`). 9 jobs / 3 GPUs
# -> one job per GPU at a time, 3 jobs deep per GPU, run sequentially within
# each GPU's queue. This is 3-way parallelism, not literally "all 9 running
# simultaneously" -- three SmolVLA fine-tunes (450M params, AdamW states,
# activations) sharing one 46GB GPU risks OOM or heavy slowdown from compute
# contention, so this defaults to the safe packing instead. Each GPU's 3
# jobs run back-to-back automatically once launched; "9 runs in flight"
# happens over the course of ~3x one run's wall-clock time, not instantly.
#
# Usage:
#   ./scripts/launch_day12_runs.sh                    # all 9, default --steps (30000) each
#   ./scripts/launch_day12_runs.sh --steps 5000        # shorter real run, e.g. a first checkpoint pass
#   ./scripts/launch_day12_runs.sh --dry-run           # print the 9 commands + GPU assignment, launch nothing
#   ./scripts/launch_day12_runs.sh --checkpoint-dir reports/checkpoints --checkpoint-every 5000
#   ./scripts/launch_day12_runs.sh --all-sessions      # pool demo1+demo3+demo4 (all 92 episodes) for
#                                                       # training instead of just the frozen demo4
#                                                       # train split -- see train_policy.py's
#                                                       # --all-sessions help. No held-out val/test
#                                                       # left for runs launched this way.
#
# Any extra flags are forwarded verbatim to every one of the 9
# `train_policy.py` invocations (e.g. --lr, --batch-size, --all-sessions,
# --train-sessions demo1 demo3 demo4).
#
# Logs:      reports/logs/<policy>_seed<seed>.log   (one per run)
# Job list:  reports/logs/day12_jobs.txt             (gpu policy seed subshell_pid)
#
# Monitor:   tail -f reports/logs/b5_seed0.log
# Stop all:  kill $(cut -d' ' -f4 reports/logs/day12_jobs.txt)   # kills the 3 GPU-queue subshells;
#            the in-progress python process under each will finish its current step then exit.

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

POLICIES=(b0 b2 b5)
SEEDS=(0 1 2)
GPUS=(0 1 2)

DRY_RUN=0
EXTRA_ARGS=()
for arg in "$@"; do
  if [[ "$arg" == "--dry-run" ]]; then
    DRY_RUN=1
  else
    EXTRA_ARGS+=("$arg")
  fi
done

# Build the 9 (policy, seed) jobs, round-robin assigned across the 3 GPUs.
JOBS=()
for policy in "${POLICIES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    JOBS+=("$policy:$seed")
  done
done

: > "$LOG_DIR/day12_jobs.txt"
n_gpus=${#GPUS[@]}

for gpu_i in "${!GPUS[@]}"; do
  gpu="${GPUS[$gpu_i]}"
  gpu_jobs=()
  for j in "${!JOBS[@]}"; do
    if (( j % n_gpus == gpu_i )); then
      gpu_jobs+=("${JOBS[$j]}")
    fi
  done

  echo "[launch_day12_runs] GPU $gpu queue: ${gpu_jobs[*]}"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    for job in "${gpu_jobs[@]}"; do
      policy="${job%%:*}"; seed="${job##*:}"
      echo "  CUDA_VISIBLE_DEVICES=$gpu $VENV_PYTHON scripts/train_policy.py --policy $policy --seed $seed ${EXTRA_ARGS[*]}"
    done
    continue
  fi

  (
    for job in "${gpu_jobs[@]}"; do
      policy="${job%%:*}"; seed="${job##*:}"
      log_file="$LOG_DIR/${policy}_seed${seed}.log"
      echo "[launch_day12_runs] GPU $gpu starting $policy seed $seed -> $log_file"
      CUDA_VISIBLE_DEVICES="$gpu" "$VENV_PYTHON" scripts/train_policy.py \
        --policy "$policy" --seed "$seed" "${EXTRA_ARGS[@]}" > "$log_file" 2>&1
      echo "[launch_day12_runs] GPU $gpu finished $policy seed $seed (exit $?)"
    done
  ) &
  echo "$gpu $(IFS=,; echo "${gpu_jobs[*]}") - $!" >> "$LOG_DIR/day12_jobs.txt"
done

if [[ "$DRY_RUN" -eq 0 ]]; then
  echo "[launch_day12_runs] 3 GPU queues launched (9 jobs total, 3 deep each)."
  echo "[launch_day12_runs] job list: $LOG_DIR/day12_jobs.txt"
  echo "[launch_day12_runs] watch one:  tail -f $LOG_DIR/b5_seed0.log"
  echo "[launch_day12_runs] stop all:   kill \$(awk '{print \$NF}' $LOG_DIR/day12_jobs.txt)"
fi
