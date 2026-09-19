#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_CACHE_DIR="$PWD/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PWD/.runtime/python"
uv python install 3.12
uv venv --python 3.12 .venv-vllm
uv pip sync --python .venv-vllm/bin/python serving/requirements.lock
