#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ $# -lt 1 || $# -gt 2 ]]; then
  cat >&2 <<'USAGE'
Usage: eval.sh <experiment-name> <training-run-id>

  experiment-name     name for this evaluation, evaluated in its own run
                      directory so it keeps its own commit and exit code
  training-run-id     training experiment to read checkpoint-best.pth from
                      (required)

Environment overrides, all optional:
  EVAL_CKPT           explicit checkpoint path, overrides training-run-id
  EVAL_SPLIT          test (default) or val
  CUDA_VISIBLE_DEVICES
USAGE
  exit 2
fi

# shellcheck source=server-env.sh
source "$SCRIPT_DIR/server-env.sh"
cd "$REPO_ROOT"

eval_split="${EVAL_SPLIT:-test}"

# A checkpoint must be named explicitly. The old default of experiments/latest
# is deliberately gone: start-experiment.sh repoints that symlink at this run's
# own directory, so a default resolved to it can vanish before the background
# worker reads it (FileNotFoundError). Pass the training run id or EVAL_CKPT.
if [[ -n "${EVAL_CKPT:-}" ]]; then
  checkpoint="$EVAL_CKPT"
elif [[ $# -eq 2 ]]; then
  checkpoint="$REPO_ROOT/experiments/$2/checkpoint-best.pth"
else
  echo "ERROR: no checkpoint specified. Pass the training-run-id as the second argument or set EVAL_CKPT." >&2
  exit 2
fi

if [[ ! -f "$checkpoint" ]]; then
  echo "ERROR: no checkpoint at $checkpoint" >&2
  exit 2
fi

# Take the architecture from the checkpoint rather than from a default.
# utils.load_state_dict only prints its size-mismatch list instead of raising,
# so evaluating with the wrong --model would score a half-random network and
# still print a plausible-looking accuracy.
read -r model input_size nb_classes < <(
  $COPPER_PYTHON -c '
import sys, torch
stored = torch.load(sys.argv[1], map_location="cpu", weights_only=False)["args"]
print(stored.model, stored.input_size, stored.nb_classes)
' "$checkpoint"
)

if [[ -z "${model:-}" || -z "${input_size:-}" || -z "${nb_classes:-}" ]]; then
  echo "ERROR: could not read model/input_size/nb_classes from $checkpoint" >&2
  exit 1
fi

echo "checkpoint: $checkpoint"
echo "model:      $model"
echo "input_size: $input_size"
echo "nb_classes: $nb_classes"
echo "split:      $eval_split"
echo

MODEL="$model" INPUT_SIZE="$input_size" NB_CLASSES="$nb_classes" \
EVAL_SPLIT="$eval_split" CKPT="$checkpoint" \
  ./scripts/start-experiment.sh "$1" -- \
    "$HOME/.local/bin/micromamba" run -n "$COPPER_ENV_NAME" \
    bash eval_csd.sh
