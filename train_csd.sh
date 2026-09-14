#!/usr/bin/env bash
set -euo pipefail

DATA_PATH="${DATA_PATH:-/path/to/dataset_root}"
CLASS_LABELS="${CLASS_LABELS:-/path/to/class_labels.txt}"
PRETRAINED_WEIGHT="${PRETRAINED_WEIGHT:-/path/to/convnextv2_tiny_1k_224_ema.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/convnextv2-csd}"

python main_finetune.py \
  --model convnextv2_fusion_grn_sa_tiny \
  --batch_size 128 \
  --update_freq 1 \
  --blr 3.2e-3 \
  --epochs 200 \
  --input_size 96 \
  --warmup_epochs 30 \
  --layer_decay_type single \
  --layer_decay 0.9 \
  --weight_decay 0.05 \
  --drop_path 0.2 \
  --mixup 0.8 \
  --cutmix 1.0 \
  --smoothing 0.1 \
  --use_amp true \
  --HSVEnhance true \
  --add_Gauss_Noise false \
  --finetune "$PRETRAINED_WEIGHT" \
  --data_path "$DATA_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --use_grayscale false \
  "$CLASS_LABELS"

