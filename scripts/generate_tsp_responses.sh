#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT/code${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NVIDIA_TF32_OVERRIDE=0
export SCHEMA_HEAD_FV_MODEL=phi4mr
uv sync --locked --extra gpu --extra api
uv run --no-sync python -m dev.schema_llm.explore.tsp_generate --root "$ROOT" "$@"
