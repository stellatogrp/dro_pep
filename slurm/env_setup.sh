#!/bin/bash
# One-time environment bootstrap ON THE CLUSTER (login node is fine: this is setup, not compute).
# Run it yourself from a laptop terminal after `sync up`, replacing the path with your
# REMOTE_PROJECT_DIR (with $USER spelled out or left for the remote shell to expand):
#   ssh <alias> 'cd /scratch/.../$USER/myproject && bash slurm/env_setup.sh'
# Creates the uv venv inside the project (on scratch) with the uv cache and uv-managed Python
# installs on scratch too, so nothing heavy lands in your small home quota.
set -euo pipefail
cd "$(dirname "$0")/.."
source slurm/cluster.env
PROJECT_DIR="${REMOTE_PROJECT_DIR}"

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
export UV_CACHE_DIR="${UV_CACHE_DIR:-$(dirname "$PROJECT_DIR")/.uv-cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$(dirname "$PROJECT_DIR")/.uv-python}"
mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR"

uv venv --python "${PYTHON_VERSION:-3.12}" .venv
# Baseline-only environment. No SDP solves, training, or GPU dependencies.
# Keep the paper's full training environment in pyproject.toml unchanged.
uv pip install --python .venv/bin/python 'numpy==2.2.6' 'scipy==1.15.3' 'pandas==2.2.3' 'PyYAML==6.0.2' 'jax==0.6.2' 'pytest==8.4.2' 
.venv/bin/python -c "import sys; print('venv OK', sys.version.split()[0])"
