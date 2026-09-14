#!/usr/bin/env bash

# Source this file before running experiments on S635Remote.
export PROJECT_ROOT="${PROJECT_ROOT:-/home/cxz/projects/copper-foil-defect-classification}"
export DATA_PATH="${DATA_PATH:-/share/data/cxz/copper-foil-defect-classification}"
export OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/convnextv2-csd}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/.local/share/mamba}"
export COPPER_ENV_NAME="${COPPER_ENV_NAME:-copper-foil-defect-classification}"
export COPPER_PYTHON="$HOME/.local/bin/micromamba run -n $COPPER_ENV_NAME python"
