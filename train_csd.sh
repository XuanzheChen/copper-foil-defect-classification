#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Overridable without editing this file, because the training server must not
# be edited directly and every tweak would otherwise cost a commit and a push.
DATA_PATH="${DATA_PATH:-/path/to/dataset_root}"
CLASS_LABELS="${CLASS_LABELS:-$SCRIPT_DIR/configs/class_labels_csd.txt}"
PRETRAINED_WEIGHT="${PRETRAINED_WEIGHT:-/path/to/convnextv2_tiny_1k_224_ema.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/convnextv2-csd}"
MODEL="${MODEL:-convnextv2_tiny}"
BATCH_SIZE="${BATCH_SIZE:-32}"
UPDATE_FREQ="${UPDATE_FREQ:-4}"
EPOCHS="${EPOCHS:-200}"
BEST_ACC_TOLERANCE="${BEST_ACC_TOLERANCE:-0.10}"
# After each save utils.save_model deletes exactly checkpoint-(epoch -
# save_ckpt_num * save_ckpt_freq) and nothing else, so 1 keeps only the newest
# periodic checkpoint beside checkpoint-best.pth. The argparse default of 3
# leaves three ~320 MB files per run, and auto_load_model only ever resumes the
# newest one. Raise this only if an older epoch must stay resumable.
SAVE_CKPT_NUM="${SAVE_CKPT_NUM:-1}"
COPPER_LOSS_WEIGHT="${COPPER_LOSS_WEIGHT:-1.0}"
HARD_EXAMPLE_MANIFEST="${HARD_EXAMPLE_MANIFEST:-}"
HARD_EXAMPLE_FACTOR="${HARD_EXAMPLE_FACTOR:-1.0}"

# --input_size must stay 200. datasets.py raises on any other value, because the
# formal copper-foil pipeline forbids resize and keeps the native 200x200 scale.
#
# Augmentation switches are real controls in datasets.py.build_transform:
#   --HSVEnhance true      gates the grayscale brightness/contrast jitter block
#   --add_Gauss_Noise false  gates SharedChannelGaussianNoise (off; sigma comes
#                          from --GN_factors / 255 when enabled)
#   --use_grayscale        compatibility only: the formal pipeline always decodes
#                          true grayscale and expands to three identical channels
# These used to be inert placeholders -- noise in particular was applied no
# matter what this file declared. This run is therefore the "no Gaussian noise,
# brightness/contrast jitter" configuration that --add_Gauss_Noise false always
# claimed to be.
#
# 200x200 carries 4.3x the pixels of the 96x96 these settings were tuned at, and
# batch 128 exhausted the 16 GB card on the first backward pass. BATCH_SIZE and
# UPDATE_FREQ are coupled and must move together: 32 x 4 holds the effective
# batch at 128 and the actual lr at 1.6e-3, which is what --blr 3.2e-3 yields at
# an effective batch of 128. Changing one alone silently rescales the lr.

python main_finetune.py \
  --model "$MODEL" \
  --batch_size "$BATCH_SIZE" \
  --update_freq "$UPDATE_FREQ" \
  --blr 3.2e-3 \
  --epochs "$EPOCHS" \
  --best_acc_tolerance "$BEST_ACC_TOLERANCE" \
  --save_ckpt_num "$SAVE_CKPT_NUM" \
  --copper_loss_weight "$COPPER_LOSS_WEIGHT" \
  --hard_example_manifest "$HARD_EXAMPLE_MANIFEST" \
  --hard_example_factor "$HARD_EXAMPLE_FACTOR" \
  --input_size 200 \
  --nb_classes "${NB_CLASSES:-9}" \
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
