#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <experiment-name>" >&2
  exit 2
fi

launch="$(bash "$SCRIPT_DIR/train.sh" "$@")"
printf '%s\n' "$launch"

run_dir="$(printf '%s\n' "$launch" | sed -n 's/^RUN_DIR=//p')"
if [[ -z "$run_dir" || ! -d "$run_dir" ]]; then
  echo "ERROR: could not read RUN_DIR from train.sh output" >&2
  exit 1
fi

# start-experiment.sh already detaches the worker: nohup, stdin from /dev/null,
# stdout and stderr redirected into the run directory. Killing this terminal
# therefore ends only the tail below, never the training.
echo
echo "Watching $run_dir"
echo "Ctrl+C stops watching only. To reattach later:"
echo "  tail -F $run_dir/stdout.log"
echo

exec tail -n +1 -F "$run_dir/stdout.log" "$run_dir/stderr.log"
