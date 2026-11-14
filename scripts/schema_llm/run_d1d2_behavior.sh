#!/usr/bin/env bash
# Run from the standalone collaborator repository; outputs are resumable per response.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT/code${PYTHONPATH:+:$PYTHONPATH}"
uv sync --locked --extra api --extra vllm --no-install-project
uv run --no-sync python -m dev.schema_llm.explore.d1d2_behavior --root "$ROOT" "$@"
