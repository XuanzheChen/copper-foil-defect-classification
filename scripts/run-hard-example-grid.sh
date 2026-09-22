#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

HARD_EXAMPLE_MANIFEST="${HARD_EXAMPLE_MANIFEST:?HARD_EXAMPLE_MANIFEST must point at a train-set mining manifest}"
HARD_EXAMPLE_FACTORS="${HARD_EXAMPLE_FACTORS:-1 2 4}"
GRID_ROOT="${OUTPUT_DIR:-$REPO_ROOT/outputs/hard-example-grid}"
EPOCHS="${EPOCHS:-130}"

if [[ ! -f "$HARD_EXAMPLE_MANIFEST" ]]; then
  echo "ERROR: manifest not found: $HARD_EXAMPLE_MANIFEST" >&2
  exit 2
fi

mkdir -p "$GRID_ROOT"
SUMMARY="$GRID_ROOT/hard-example-grid-summary.tsv"
printf "factor\trun_dir\texit_code\n" > "$SUMMARY"
overall_exit=0

for factor in $HARD_EXAMPLE_FACTORS; do
  tag="${factor//./p}"
  run_dir="$GRID_ROOT/factor-$tag"
  mkdir -p "$run_dir"

  printf "%s\n" "$factor" > "$run_dir/hard_example_factor.txt"
  printf "HARD_EXAMPLE_FACTOR=%q\n" "$factor" > "$run_dir/run_config.env"
  printf "HARD_EXAMPLE_MANIFEST=%q\n" "$HARD_EXAMPLE_MANIFEST" >> "$run_dir/run_config.env"
  printf "EPOCHS=%q\n" "$EPOCHS" >> "$run_dir/run_config.env"

  echo
  echo "===== hard-example factor $factor ====="
  echo "RUN_DIR=$run_dir"

  set +e
  OUTPUT_DIR="$run_dir" \
  EPOCHS="$EPOCHS" \
  COPPER_LOSS_WEIGHT=1.0 \
  HARD_EXAMPLE_MANIFEST="$HARD_EXAMPLE_MANIFEST" \
  HARD_EXAMPLE_FACTOR="$factor" \
  bash "$REPO_ROOT/train_csd.sh" \
    > >(tee "$run_dir/stdout.log") \
    2> >(tee "$run_dir/stderr.log" >&2)
  code=$?
  set -e

  printf "%s\n" "$code" > "$run_dir/exit_code.txt"
  printf "%s\t%s\t%s\n" "$factor" "$run_dir" "$code" >> "$SUMMARY"
  if [[ "$code" -ne 0 ]]; then
    overall_exit=1
    echo "ERROR: hard-example factor $factor failed with exit code $code" >&2
  else
    echo "Completed hard-example factor $factor"
  fi
done

echo
echo "===== hard-example grid summary ====="
cat "$SUMMARY"
exit "$overall_exit"
