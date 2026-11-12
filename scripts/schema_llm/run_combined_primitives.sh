#!/usr/bin/env bash
# Install dependencies inside pack/runtime before running (see pack/README.md).
# bash /path/to/pack/run.sh /path/to/work
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PACK=${PACK:-$SCRIPT_DIR}
WORK=${1:?Usage: bash /path/to/pack/run.sh /path/to/work}
PACK=$(cd "$PACK" && pwd)
if [ ! -f "$PACK/runtime/pyproject.toml" ]; then
  echo "Missing packed runtime; point PACK at the prepared input directory." >&2
  exit 2
fi
mkdir -p "$WORK"
WORK=$(cd "$WORK" && pwd)
cd "$PACK/runtime"
export PYTHONPATH="$PWD/code${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NVIDIA_TF32_OVERRIDE=0
run() { uv run python "$PACK/pipeline.py" "$@" --pack "$PACK"; }
run verify
phi=(--model phi4mr)
qwen=(--model qwen34b)
if [ -n "${PHI_MODEL:-}" ]; then phi+=(--checkpoint "$PHI_MODEL"); fi
if [ -n "${QWEN_MODEL:-}" ]; then qwen+=(--checkpoint "$QWEN_MODEL"); fi
if [ -n "${PHI_REVISION:-}" ]; then phi+=(--revision "$PHI_REVISION"); fi
if [ -n "${QWEN_REVISION:-}" ]; then qwen+=(--revision "$QWEN_REVISION"); fi
run capture --work "$WORK" "${phi[@]}"
run capture --work "$WORK" "${qwen[@]}"
run cluster --work "$WORK" --backend "${ANNOTATE_BACKEND:-trapi}" \
  --annotate-model "${ANNOTATE_MODEL:-gpt-5.6-sol_2026-07-09}"
run merge --work "$WORK" --embed-model "${EMBED_MODEL:-Qwen/Qwen3-Embedding-4B}"
run export --work "$WORK"
