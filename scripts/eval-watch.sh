#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <experiment-name> [training-run-id]" >&2
  exit 2
fi

launch="$(bash "$SCRIPT_DIR/eval.sh" "$@")"
printf '%s\n' "$launch"

run_dir="$(printf '%s\n' "$launch" | sed -n 's/^RUN_DIR=//p')"
if [[ -z "$run_dir" || ! -d "$run_dir" ]]; then
  echo "ERROR: could not read RUN_DIR from eval.sh output" >&2
  exit 1
fi

# Same detachment as train-watch.sh: the evaluation outlives this terminal and
# its accuracy report lands in the run directory either way.
echo
echo "Watching $run_dir"
echo "Ctrl+C stops watching only. To reattach later:"
echo "  tail -F $run_dir/stdout.log"
echo

exec tail -n +1 -F "$run_dir/stdout.log" "$run_dir/stderr.log"
