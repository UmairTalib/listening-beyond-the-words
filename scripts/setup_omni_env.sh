#!/usr/bin/env bash
# setup_omni_env.sh: create or update the Python environment on OMNI from uv.lock.
#
# Usage:   bash scripts/setup_omni_env.sh        (login node, inside the repository)
#
# Needs   LBW_WORKSPACE, the workspace root (see docs/engineering_conventions.md)
# Writes  $LBW_WORKSPACE/env/venv       the environment (a few GB, rebuilt by this script)
#         $LBW_WORKSPACE/env/uv-cache   the download cache
#         ~/.local/bin/uv               uv itself, only if it is missing
#
# Safe to re-run. Run it again after every pull that changes uv.lock, and after
# moving to a new workspace.
set -euo pipefail
umask 077

UV_VERSION="0.13.7"
: "${LBW_WORKSPACE:?LBW_WORKSPACE is not set; see docs/engineering_conventions.md}"
export UV_PROJECT_ENVIRONMENT="$LBW_WORKSPACE/env/venv"
export UV_CACHE_DIR="$LBW_WORKSPACE/env/uv-cache"

if ! command -v uv > /dev/null 2>&1; then
  echo "installing uv $UV_VERSION into ~/.local/bin ..."
  curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" | env UV_NO_MODIFY_PATH=1 sh
  export PATH="$HOME/.local/bin:$PATH"
fi
echo "using $(uv --version)"

cd "$(git rev-parse --show-toplevel)"
uv python install                       # the version in .python-version
uv sync --locked --extra cu126 --extra pipeline

"$UV_PROJECT_ENVIRONMENT/bin/python" - << 'PY'
import torch

print(f"torch {torch.__version__}, CUDA {torch.version.cuda}, archs {torch.cuda.get_arch_list()}")
assert torch.version.cuda == "12.6", "wrong PyTorch build: sync with --extra cu126"
assert "sm_70" in torch.cuda.get_arch_list(), "this PyTorch build has no V100 (sm_70) kernels"
print("OK: the environment has the PyTorch build with V100 kernels")
PY
