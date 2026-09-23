#!/bin/bash
# Sync ONLY what `git status` currently shows as changed in compliance-vla
# (modified + untracked, respecting .gitignore) to HPC -- not the whole tree.
# For the one-time large transfers (wheelhouse/, .hf_cache/, external/,
# dataset/) see scripts/README.md's "Copying files to HPC" section instead;
# this script is for the fast "get my latest code edits onto HPC" loop, e.g.
# after adding scripts/slurm_train_hpc_overfit_check.sbatch or editing
# train_policy.py, right before `sbatch`-ing a job there.
#
# Using `git status` instead of an exclude list means gitignored local
# artifacts (checkpoints/ [~15GB], wandb/, .venv/, wheelhouse/, .hf_cache/,
# __pycache__/) never show up as "changed" in the first place, so they're
# never candidates for transfer -- no exclude list to keep in sync by hand.
#
# Deletions (git status 'D') are reported but NOT applied remotely by
# default -- removing files on HPC is a separate, harder-to-reverse action,
# not implied by "sync my edits". Pass --delete-removed to actually rm them
# on the remote.
#
# Usage:
#   scripts/sync_to_hpc.sh                       # sync current git changes
#   scripts/sync_to_hpc.sh --dry-run              # show what would transfer, change nothing
#   scripts/sync_to_hpc.sh --delete-removed       # also rm files git status shows as deleted, on the remote
#   HPC_HOST=myalias scripts/sync_to_hpc.sh       # different ~/.ssh/config Host alias
#   HPC_DIR=other-name scripts/sync_to_hpc.sh     # different remote directory name
#
# Any other extra args are passed straight through to rsync (e.g. -n).

set -euo pipefail

HPC_HOST="${HPC_HOST:-hpc}"                          # the `Host hpc` alias scripts/README.md assumes
HPC_DIR="${HPC_DIR:-compliance-vla}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

DELETE_REMOVED=0
rsync_args=()
for arg in "$@"; do
  if [[ "$arg" == "--delete-removed" ]]; then
    DELETE_REMOVED=1
  else
    rsync_args+=("$arg")
  fi
done

to_add=()
to_delete=()
while IFS= read -r -d '' entry; do
  status="${entry:0:2}"
  path="${entry:3}"
  if [[ "$status" == *D* ]]; then
    to_delete+=("$path")
  else
    to_add+=("$path")
  fi
done < <(git status --porcelain=v1 -z --no-renames)

if [[ ${#to_add[@]} -eq 0 && ${#to_delete[@]} -eq 0 ]]; then
  echo "[sync_to_hpc] git status is clean -- nothing to sync"
  exit 0
fi

ssh "$HPC_HOST" echo ok >/dev/null || {
  echo "[sync_to_hpc] couldn't reach host '$HPC_HOST' -- check ~/.ssh/config has a Host block for it" \
       "(scripts/README.md's 'Copying files to HPC' section), and that you're on the the HPC site VPN if required." >&2
  exit 1
}
ssh "$HPC_HOST" "mkdir -p '$HPC_DIR'"

if [[ ${#to_add[@]} -gt 0 ]]; then
  echo "[sync_to_hpc] syncing ${#to_add[@]} changed file(s) -> $HPC_HOST:$HPC_DIR/"
  printf '  %s\n' "${to_add[@]}"
  printf '%s\0' "${to_add[@]}" \
    | rsync -avz --progress --files-from=- --from0 "${rsync_args[@]}" ./ "$HPC_HOST:$HPC_DIR/"
fi

if [[ ${#to_delete[@]} -gt 0 ]]; then
  if [[ "$DELETE_REMOVED" == "1" ]]; then
    echo "[sync_to_hpc] removing ${#to_delete[@]} deleted file(s) on $HPC_HOST:$HPC_DIR/"
    printf '  %s\n' "${to_delete[@]}"
    remote_cmd="cd '$HPC_DIR' &&"
    for p in "${to_delete[@]}"; do
      remote_cmd+=" rm -f -- '$p';"
    done
    ssh "$HPC_HOST" "$remote_cmd"
  else
    echo "[sync_to_hpc] NOTE: ${#to_delete[@]} file(s) deleted locally per git status, left untouched on" \
         "$HPC_HOST (rerun with --delete-removed to remove them there too):"
    printf '  %s\n' "${to_delete[@]}"
  fi
fi
