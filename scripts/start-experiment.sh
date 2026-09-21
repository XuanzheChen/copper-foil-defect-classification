#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || "$2" != "--" ]]; then
  echo "Usage: $0 <experiment-name> -- <command> [args...]" >&2
  exit 2
fi

experiment_name="$1"
shift 2

safe_name="$(printf '%s' "$experiment_name" | tr -cs 'A-Za-z0-9._-' '-')"
safe_name="${safe_name#-}"
safe_name="${safe_name%-}"
if [[ -z "$safe_name" ]]; then
  echo "Experiment name must contain at least one letter or digit." >&2
  exit 2
fi

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "ERROR: working tree is dirty. Commit and push changes before starting an experiment." >&2
  git status --short >&2
  exit 2
fi

run_id="$(date -u +'%Y%m%dT%H%M%SZ')-${safe_name}"
run_dir="$repo_root/experiments/$run_id"
mkdir -p "$run_dir"

git rev-parse HEAD > "$run_dir/git_commit.txt"
git remote get-url origin > "$run_dir/git_remote.txt"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$run_dir/started_at.txt"
printf '%q ' "$@" > "$run_dir/command.sh"
printf '\n' >> "$run_dir/command.sh"

nohup bash -c '
  run_dir="$1"
  shift
  set +e
  # The wrapper owns OUTPUT_DIR so checkpoints, tensorboard logs and eval
  # reports land inside this run directory instead of a separate tree.
  export OUTPUT_DIR="$run_dir"
  "$@" >"$run_dir/stdout.log" 2>"$run_dir/stderr.log"
  code=$?
  printf "%s\n" "$code" >"$run_dir/exit_code.txt"
  date -u +"%Y-%m-%dT%H:%M:%SZ" >"$run_dir/finished_at.txt"
  exit "$code"
' _ "$run_dir" "$@" </dev/null >/dev/null 2>&1 &

pid=$!
printf '%s\n' "$pid" > "$run_dir/pid.txt"
ln -sfn "$run_dir" "$repo_root/experiments/latest"

echo "RUN_ID=$run_id"
echo "PID=$pid"
echo "RUN_DIR=$run_dir"
echo "OUTPUT_DIR=$run_dir"
echo "STDOUT=$run_dir/stdout.log"
echo "STDERR=$run_dir/stderr.log"
