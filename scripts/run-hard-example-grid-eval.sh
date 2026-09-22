#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=server-env.sh
source "$SCRIPT_DIR/server-env.sh"

# Evaluate every hard-example factor's checkpoint-best on the held-out split.
#
# One invocation produces one experiment directory: start-experiment.sh owns
# OUTPUT_DIR, and each factor gets an isolated test-factor-<n> child beneath it.
# Running one eval.sh per factor instead would create three top-level experiment
# directories, and the second launch would abort anyway -- start-experiment.sh
# refuses to start while the working tree is dirty, and the first run directory
# is untracked until its reports are committed.
#
# The checkpoints live in the training grid, which is a different experiment
# directory, so it is named explicitly rather than derived from OUTPUT_DIR.
if [[ $# -lt 1 ]]; then
  cat >&2 <<'USAGE'
Usage: run-hard-example-grid-eval.sh <training-grid-dir> [factor...]

  training-grid-dir   experiment directory holding factor-<n>/checkpoint-best.pth
  factor              factors to evaluate; defaults to 1 2 4

Environment overrides, all optional:
  EVAL_SPLIT          test (default) or val
  CUDA_VISIBLE_DEVICES
USAGE
  exit 2
fi

TRAINING_GRID_DIR="$1"
shift
HARD_EXAMPLE_FACTORS="${*:-1 2 4}"
GRID_ROOT="${OUTPUT_DIR:-$REPO_ROOT/outputs/hard-example-grid-eval}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"

if [[ ! -d "$TRAINING_GRID_DIR" ]]; then
  echo "ERROR: training grid directory not found: $TRAINING_GRID_DIR" >&2
  exit 2
fi

mkdir -p "$GRID_ROOT"
SUMMARY="$GRID_ROOT/hard-example-grid-eval-summary.tsv"
printf "factor\trun_dir\tcheckpoint\texit_code\n" > "$SUMMARY"
overall_exit=0

for factor in $HARD_EXAMPLE_FACTORS; do
  tag="${factor//./p}"
  run_dir="$GRID_ROOT/test-factor-$tag"
  checkpoint="$TRAINING_GRID_DIR/factor-$tag/checkpoint-best.pth"
  mkdir -p "$run_dir"

  echo
  echo "===== test hard-example factor $factor ====="
  echo "RUN_DIR=$run_dir"
  echo "CKPT=$checkpoint"

  if [[ ! -f "$checkpoint" ]]; then
    echo "ERROR: no checkpoint at $checkpoint" >&2
    printf "%s\n" 2 > "$run_dir/exit_code.txt"
    printf "%s\t%s\t%s\t%s\n" "$factor" "$run_dir" "$checkpoint" 2 >> "$SUMMARY"
    overall_exit=1
    continue
  fi

  # Take the architecture from the checkpoint rather than from eval_csd.sh's
  # defaults. utils.load_state_dict only prints its size-mismatch list instead
  # of raising, so evaluating with the wrong --model would score a half-random
  # network and still print a plausible-looking accuracy.
  read -r model input_size nb_classes < <(
    $COPPER_PYTHON -c '
import sys, torch
stored = torch.load(sys.argv[1], map_location="cpu", weights_only=False)["args"]
print(stored.model, stored.input_size, stored.nb_classes)
' "$checkpoint"
  )

  if [[ -z "${model:-}" || -z "${input_size:-}" || -z "${nb_classes:-}" ]]; then
    echo "ERROR: could not read model/input_size/nb_classes from $checkpoint" >&2
    printf "%s\n" 1 > "$run_dir/exit_code.txt"
    printf "%s\t%s\t%s\t%s\n" "$factor" "$run_dir" "$checkpoint" 1 >> "$SUMMARY"
    overall_exit=1
    continue
  fi

  echo "model:      $model"
  echo "input_size: $input_size"
  echo "nb_classes: $nb_classes"
  echo "split:      $EVAL_SPLIT"

  set +e
  OUTPUT_DIR="$run_dir" \
  MODEL="$model" \
  INPUT_SIZE="$input_size" \
  NB_CLASSES="$nb_classes" \
  EVAL_SPLIT="$EVAL_SPLIT" \
  CKPT="$checkpoint" \
  bash "$REPO_ROOT/eval_csd.sh" \
    > >(tee "$run_dir/stdout.log") \
    2> >(tee "$run_dir/stderr.log" >&2)
  code=$?
  set -e

  printf "%s\n" "$code" > "$run_dir/exit_code.txt"
  printf "%s\t%s\t%s\t%s\n" "$factor" "$run_dir" "$checkpoint" "$code" >> "$SUMMARY"
  if [[ "$code" -ne 0 ]]; then
    overall_exit=1
    echo "ERROR: test hard-example factor $factor failed with exit code $code" >&2
  else
    echo "Completed test hard-example factor $factor"
  fi
done

echo
echo "===== hard-example grid test summary ====="
cat "$SUMMARY"
exit "$overall_exit"
