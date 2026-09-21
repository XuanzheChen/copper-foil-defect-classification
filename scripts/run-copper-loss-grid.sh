#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Run the four copper-loss weights sequentially on one GPU. When launched via
# scripts/start-experiment.sh, OUTPUT_DIR is the parent experiment directory and
# each weight gets an isolated child directory beneath it.
GRID_ROOT="${OUTPUT_DIR:-$REPO_ROOT/outputs/copper-loss-grid}"
COPPER_LOSS_WEIGHTS="${COPPER_LOSS_WEIGHTS:-1.0 1.25 1.5 2.0}"

mkdir -p "$GRID_ROOT"
SUMMARY="$GRID_ROOT/copper-loss-grid-summary.tsv"
printf "weight\trun_dir\texit_code\n" > "$SUMMARY"

overall_exit=0

for weight in $COPPER_LOSS_WEIGHTS; do
  tag="${weight//./p}"
  run_dir="$GRID_ROOT/weight-$tag"
  mkdir -p "$run_dir"

  printf "%s\n" "$weight" > "$run_dir/copper_loss_weight.txt"
  printf "COPPER_LOSS_WEIGHT=%q\n" "$weight" > "$run_dir/run_config.env"
  printf "BEST_ACC_TOLERANCE=%q\n" "${BEST_ACC_TOLERANCE:-0.10}" >> "$run_dir/run_config.env"

  echo
  echo "===== copper loss weight $weight ====="
  echo "RUN_DIR=$run_dir"

  set +e
  OUTPUT_DIR="$run_dir" \
  COPPER_LOSS_WEIGHT="$weight" \
  bash "$REPO_ROOT/train_csd.sh" \
    > >(tee "$run_dir/stdout.log") \
    2> >(tee "$run_dir/stderr.log" >&2)
  code=$?
  set -e

  printf "%s\n" "$code" > "$run_dir/exit_code.txt"
  printf "%s\t%s\t%s\n" "$weight" "$run_dir" "$code" >> "$SUMMARY"

  if [[ "$code" -ne 0 ]]; then
    overall_exit=1
    echo "ERROR: copper loss weight $weight failed with exit code $code" >&2
  else
    echo "Completed copper loss weight $weight"
  fi
done

echo
echo "===== copper loss grid summary ====="
cat "$SUMMARY"
exit "$overall_exit"
