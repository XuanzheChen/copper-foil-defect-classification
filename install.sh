#!/usr/bin/env bash
# Copper standalone one-command installer (Linux). Run from any working folder.
# Git access to the private repository is required. No sudo or system Python changes.
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo 'ERROR: install.sh supports Linux only. On Windows run install.ps1.' >&2
  exit 2
fi
if ! command -v git >/dev/null 2>&1; then
  echo 'ERROR: Git is required to clone the private Copper repository.' >&2
  exit 2
fi

if command -v uv >/dev/null 2>&1; then
  uv_cmd="$(command -v uv)"
elif [[ -x "$HOME/.local/bin/uv" ]]; then
  uv_cmd="$HOME/.local/bin/uv"
else
  if [[ "${COPPER_NO_UV_BOOTSTRAP:-0}" == '1' ]]; then
    echo 'ERROR: uv missing; install from https://docs.astral.sh/uv/ or unset COPPER_NO_UV_BOOTSTRAP.' >&2
    exit 2
  fi
  if ! command -v curl >/dev/null 2>&1; then
    echo 'ERROR: curl is needed to bootstrap uv. Install curl or uv manually.' >&2
    exit 2
  fi
  echo 'Installing the official uv launcher into the current user profile...'
  # Official uv installer; no administrator rights and no profile-file edits.
  curl -LsSf https://astral.sh/uv/install.sh | UV_NO_MODIFY_PATH=1 sh
  uv_cmd="$HOME/.local/bin/uv"
  if [[ ! -x "$uv_cmd" ]]; then
    echo 'ERROR: uv installer did not create ~/.local/bin/uv.' >&2
    exit 2
  fi
fi

"$uv_cmd" python install 3.10
managed_python="$("$uv_cmd" python find --managed-python 3.10)"
if [[ ! -f "$managed_python" ]]; then
  echo "ERROR: cannot locate uv-managed Python 3.10: $managed_python" >&2
  exit 2
fi
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export COPPER_UV="$uv_cmd"
exec "$managed_python" "$script_dir/scripts/install/bootstrap.py" "$@"
