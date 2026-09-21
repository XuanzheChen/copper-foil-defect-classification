#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The launchers in scripts/ fill these in. CKPT has no default on purpose: an
# evaluation must name the weights it scored.
CKPT="${CKPT:?CKPT must point at the checkpoint to evaluate}"
MODEL="${MODEL:-convnextv2_tiny}"
INPUT_SIZE="${INPUT_SIZE:-200}"
NB_CLASSES="${NB_CLASSES:-9}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
DATA_PATH="${DATA_PATH:-/path/to/dataset_root}"
CLASS_LABELS="${CLASS_LABELS:-$SCRIPT_DIR/configs/class_labels_csd.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/convnextv2-csd}"

# IMNET evaluation always reads <data_path>/val, so it can never reach the test
# split. The image_folder loader is the only one that takes both roots
# explicitly. --nb_classes must appear on the command line because datasets.py
# asserts it against the folder listing, and that assert runs before
# main_finetune.py overwrites nb_classes from the label file.
#
# main_finetune.py builds the training dataset unconditionally, --eval or not,
# so --data_path still has to point at a real ImageFolder root.

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
  "$CLASS_LABELS"
