#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python}" "$SCRIPT_DIR/infer_topk.py" \
  --checkpoint "$SCRIPT_DIR/weights/checkpoint-best.pth" \
  --class-labels "$SCRIPT_DIR/configs/class_labels_csd.txt" \
  --k 3 --conf 0.7 "$@"
