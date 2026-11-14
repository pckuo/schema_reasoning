#!/usr/bin/env bash
# Launch a LOCAL vLLM OpenAI-compatible server in an ISOLATED venv -- the fast alternative to HF
# hosting for LONG response-only runs (reasoning models) when NO latents are needed. Latent capture
# still uses the HF harness (`python -m dev.schema_llm.subjects.capture`); this server only feeds
# behavioural response collection via `collect --backend vllm`.
#
# WHY an isolated venv: vLLM hard-pins `torch==2.11.0`, which conflicts with the repo's torch in
# /opt/venv (the `gpu` extra), so it MUST NOT share it. We build a throwaway venv the first time and
# reuse it thereafter. Its torch (CUDA 12.x bundled) runs fine on a newer CUDA driver via backward
# compatibility.
#
# NOTE: we start the server WITHOUT `--reasoning-parser` on purpose, so a reasoning model's inline
# `<think> ... </think>` stays in the response `content` -- matching the HF-local format so the
# judge's `strip_reasoning()` grades only the Solution, unchanged.
#
# Usage:
#   bash scripts/schema_llm/vllm_serve.sh microsoft/Phi-4-reasoning
#   MODEL=Qwen/Qwen3-14B bash scripts/schema_llm/vllm_serve.sh
# Env overrides:
#   VLLM_VENV       (default /opt/vllm-venv)   isolated env location (reused across runs)
#   VLLM_VERSION    (default 0.25.1)           pinned vLLM release
#   VLLM_PORT       (default 8000)             OpenAI server port
#   VLLM_GPU_UTIL   (default 0.90)             --gpu-memory-utilization
#   VLLM_MAX_LEN    (default 16384)            --max-model-len (prompt + generated)
#   VLLM_EXTRA_ARGS (default empty)            extra `vllm serve` flags (word-split)
set -euo pipefail
cd "$(dirname "$0")/../.." || exit 1

MODEL="${1:-${MODEL:-microsoft/Phi-4-reasoning}}"
VLLM_VENV="${VLLM_VENV:-/opt/vllm-venv}"
VLLM_VERSION="${VLLM_VERSION:-0.25.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.90}"
VLLM_MAX_LEN="${VLLM_MAX_LEN:-16384}"

# 1) build the isolated env once (idempotent). vLLM pulls its own pinned torch==2.11.0.
if [ ! -x "$VLLM_VENV/bin/vllm" ]; then
  echo "=== creating isolated vLLM venv at $VLLM_VENV (vllm==$VLLM_VERSION) $(date -u +%FT%TZ) ==="
  uv venv "$VLLM_VENV"
  uv pip install --python "$VLLM_VENV/bin/python" "vllm==$VLLM_VERSION"
fi

# The base image is nvidia/cuda:*-runtime (NO working nvcc/toolkit), so FlashInfer's JIT-compiled
# sampler kernel cannot build (`ninja` -> `/usr/local/cuda/bin/nvcc` fails, engine core aborts at
# startup). Disable the FlashInfer sampler (vLLM falls back to its native torch top-k/top-p path)
# and pin the prebuilt FLASH_ATTN attention backend (no JIT) instead of FLASHINFER. This is the
# known-good runtime-image config; do not remove without restoring a working nvcc toolchain.
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# 2) serve. --served-model-name = the HF id so `collect --model <id>` matches the OpenAI `model`.
echo "=== serving $MODEL on :$VLLM_PORT (gpu-util=$VLLM_GPU_UTIL max-len=$VLLM_MAX_LEN) $(date -u +%FT%TZ) ==="
# Intentionally word-split: VLLM_EXTRA_ARGS is trusted operator-controlled input for extra flags.
exec "$VLLM_VENV/bin/vllm" serve "$MODEL" \
  --port "$VLLM_PORT" \
  --dtype bfloat16 \
  --gpu-memory-utilization "$VLLM_GPU_UTIL" \
  --max-model-len "$VLLM_MAX_LEN" \
  --served-model-name "$MODEL" \
  ${VLLM_EXTRA_ARGS:-}
