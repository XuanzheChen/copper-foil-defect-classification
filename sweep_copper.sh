#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CKPT="${CKPT:?CKPT must point at the checkpoint to sweep}"
MODEL="${MODEL:-convnextv2_tiny}"
INPUT_SIZE="${INPUT_SIZE:-200}"
NB_CLASSES="${NB_CLASSES:-9}"
EVAL_SPLIT="${EVAL_SPLIT:-val}"
DATA_PATH="${DATA_PATH:-/path/to/dataset_root}"
CLASS_LABELS="${CLASS_LABELS:-$SCRIPT_DIR/configs/class_labels_csd.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/copper-sweep}"

BIAS_MIN="${BIAS_MIN:-0.0}"
BIAS_MAX="${BIAS_MAX:-5.0}"
BIAS_STEPS="${BIAS_STEPS:-101}"
THRESHOLD_MIN="${THRESHOLD_MIN:-0.001}"
THRESHOLD_MAX="${THRESHOLD_MAX:-0.5}"
THRESHOLD_STEPS="${THRESHOLD_STEPS:-100}"
MIN_PRECISION="${MIN_PRECISION:-98.0}"
MAX_ACC_DROP="${MAX_ACC_DROP:-0.10}"

python main_finetune.py \
  --model "$MODEL" \
  --eval true \
  --data_set image_folder \
  --nb_classes "$NB_CLASSES" \
  --input_size "$INPUT_SIZE" \
  --data_path "$DATA_PATH/train" \
  --eval_data_path "$DATA_PATH/$EVAL_SPLIT" \
  --resume "$CKPT" \
  --output_dir "$OUTPUT_DIR" \
  --use_amp true \
  --copper_sweep true \
  --copper_bias_min "$BIAS_MIN" \
  --copper_bias_max "$BIAS_MAX" \
  --copper_bias_steps "$BIAS_STEPS" \
  --copper_threshold_min "$THRESHOLD_MIN" \
  --copper_threshold_max "$THRESHOLD_MAX" \
  --copper_threshold_steps "$THRESHOLD_STEPS" \
  --copper_sweep_min_precision "$MIN_PRECISION" \
  --copper_sweep_max_acc_drop "$MAX_ACC_DROP" \
  "$CLASS_LABELS"
