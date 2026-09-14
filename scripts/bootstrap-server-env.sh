#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
env_name="copper-foil-defect-classification"
micromamba="$HOME/.local/bin/micromamba"
export MAMBA_ROOT_PREFIX="$HOME/.local/share/mamba"

if [[ ! -x "$micromamba" ]]; then
  temp_dir="$(mktemp -d)"
  trap 'rm -rf "$temp_dir"' EXIT
  archive="$temp_dir/micromamba.tar.bz2"
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest -o "$archive"
  python3 -c 'import sys, tarfile; tarfile.open(sys.argv[1]).extract("bin/micromamba", sys.argv[2])' \
    "$archive" "$temp_dir"
  mkdir -p "$(dirname "$micromamba")"
  install -m 0755 "$temp_dir/bin/micromamba" "$micromamba"
fi

if ! "$micromamba" env list | awk '{print $1}' | grep -Fxq "$env_name"; then
  "$micromamba" create -y -n "$env_name" -c conda-forge python=3.10 pip
else
  "$micromamba" install -y -n "$env_name" -c conda-forge python=3.10 pip
fi

"$micromamba" run -n "$env_name" python -m pip install --upgrade pip
"$micromamba" run -n "$env_name" python -m pip install \
  torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
"$micromamba" run -n "$env_name" python -m pip install -r "$repo_root/requirements.txt"

"$micromamba" run -n "$env_name" python - <<'PY'
import torch
import torchvision
import timm

assert torch.cuda.is_available(), "PyTorch cannot access the NVIDIA GPU"
assert "sm_120" in torch.cuda.get_arch_list(), torch.cuda.get_arch_list()
x = torch.ones(4, device="cuda")
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("timm:", timm.__version__)
print("gpu:", torch.cuda.get_device_name(0))
print("cuda runtime:", torch.version.cuda)
print("gpu smoke test:", x.sum().item())
PY
