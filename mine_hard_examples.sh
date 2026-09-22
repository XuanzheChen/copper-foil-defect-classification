#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CKPT="${CKPT:?CKPT must point at the control checkpoint used for train-set mining}"
DATA_PATH="${DATA_PATH:-/path/to/dataset_root}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/hard-example-mining}"
COPPER_BOUNDARY_MARGIN_MAX="${COPPER_BOUNDARY_MARGIN_MAX:-0.5}"
FOREIGN_NEGATIVE_MARGIN_MIN="${FOREIGN_NEGATIVE_MARGIN_MIN:--0.5}"

mkdir -p "$OUTPUT_DIR"

EVAL_SPLIT=train \
CKPT="$CKPT" \
DATA_PATH="$DATA_PATH" \
OUTPUT_DIR="$OUTPUT_DIR" \
bash "$SCRIPT_DIR/eval_csd.sh"

python "$SCRIPT_DIR/scripts/build-hard-example-manifest.py" \
  --predictions "$OUTPUT_DIR/eval_predictions.csv" \
  --dataset-root "$DATA_PATH/train" \
  --output "$OUTPUT_DIR/hard_examples.csv" \
  --copper-boundary-margin-max "$COPPER_BOUNDARY_MARGIN_MAX" \
  --foreign-negative-margin-min "$FOREIGN_NEGATIVE_MARGIN_MIN"

echo "HARD_EXAMPLE_MANIFEST=$OUTPUT_DIR/hard_examples.csv"
echo "HARD_EXAMPLE_SUMMARY=$OUTPUT_DIR/hard_examples.summary.json"
