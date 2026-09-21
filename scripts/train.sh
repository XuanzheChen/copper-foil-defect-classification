#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ $# -ne 1 ]]; then
  cat >&2 <<'USAGE'
Usage: train.sh <experiment-name>

Starts a training run through scripts/start-experiment.sh and returns
immediately. Every artifact lands in experiments/<timestamp>-<experiment-name>/.

Environment overrides, all optional and inherited by train_csd.sh:
  MODEL, BATCH_SIZE, UPDATE_FREQ, EPOCHS, CUDA_VISIBLE_DEVICES
USAGE
  exit 2
fi

# server-env.sh only fills in variables that are unset, so anything exported on
# the command line survives.
# shellcheck source=server-env.sh
source "$SCRIPT_DIR/server-env.sh"
cd "$REPO_ROOT"

./scripts/start-experiment.sh "$1" -- \
  "$HOME/.local/bin/micromamba" run -n "$COPPER_ENV_NAME" \
  bash train_csd.sh
