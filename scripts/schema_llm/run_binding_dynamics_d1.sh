#!/usr/bin/env bash
# BINDING-DYNAMICS **STAGE 1 (DESIGN 1)** END TO END:
#   generation -> vLLM responses -> judge + latents -> merge -> export.
#
# The two designs are deliberately run as SEPARATE STAGES. Design 1 (onset latency, 150 items) is
# taken all the way to an exported bundle before Design 2 (ambiguity/resolution) begins, so the
# first design's numbers are in hand before any of the second design's generation budget is spent.
# Stage 2 is the sibling script `run_binding_dynamics_d2.sh`; it is identical except for DESIGN,
# NAME and STAMP.
#
# Splitting the stages does NOT change the design: `dynamics/design.py` builds each design from its
# own RNG and balances `prop_naming` with a counter keyed on (experiment x schema), so a D1-only
# build is byte-identical to the D1 half of a combined build. That is asserted by
# `tests/test_binding_dynamics.py::test_staging_does_not_change_the_design`.
#
# Stage map (STEP=<gen|collect|latents|merge|export|post|all>, default all):
#   `post` = everything except gen, for when generation was launched separately.
#   gen      150 items (D1a 90 onset-latency / D1b 60 sequential) via the TRAPI pool.
#   collect  BOTH subject models co-tenanted on the one A100, responses only.
#   latents  judge (TRAPI, no GPU) in the BACKGROUND while capture+cluster (GPU) runs in front.
#   merge    source-blind semantic-avglink over both node caches -> the shared alphabet.
#   export   flat tables + sequence-layer activations.
#
# PREREQS: `az login --scope api://trapi/.default` + `uv sync --extra api`.
#
# RESUMABLE AT EVERY STEP -- verified in the code, not assumed, because this box can be interrupted:
#   gen      per-family checkpoints under <RUN>/families/; a family is written only once all its
#            siblings are accepted, so a partial family is regenerated rather than half-used.
#   collect  skips the (scenario_id, task) rows already in responses.<tag>.jsonl; each row is
#            appended and flushed as it is produced, by a single writer thread.
#   capture  writes layer_*.npy + meta.json BEFORE appending the item's response row, so the
#            "done" marker can never point at a half-written activation. Interrupted items are
#            simply redone.
#   judge    --resume reuses cached verdicts that PARSE; unparseable/errored ones are re-called.
#   merge/   deterministic recomputations from the cached clusters.json / merge.json -- cheap, and
#   export   each step overwrites only its own file.
# So the recovery procedure after any interruption is: re-run this exact line. Nothing is lost and
# nothing is double-counted.
#
# Launch detached:
#   nohup bash scripts/schema_llm/run_binding_dynamics_d1.sh \
#       >> results/schema_llm/logs/binding_dyn_d1.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

DESIGN="${DESIGN:-D1}"
NAME="${NAME:-binding_dyn_d1_v1}"
STAMP="${STAMP:-$(date -u +%Y%m%d)}"
STEP="${STEP:-all}"
# Datasets the EXPORT step should pool into one bundle, comma separated. A waved corpus is
# collected and captured per wave but exported as a single bundle, so this is how the second wave
# is merged with the first. Defaults to the single dataset in $NAME.
POOL_NAME="${POOL_NAME:-$NAME}"
# Run dirs the CLUSTER step pools; must match POOL_NAME or tokens.parquet asserts on a token-count
# mismatch between the responses and the clustered stack.
POOL_RUNS="${POOL_RUNS:-results/schema_llm/data/$NAME}"
RUN="results/schema_llm/data/$NAME"
# The bundle is stamped separately from the dataset: the dataset ACCUMULATES (wave 1 + wave 2 = 300
# items in one dir) while each export is a point-in-time artifact. The wave-1 bundle stays frozen at
# its own stamp; a combined run writes a new one rather than overwriting it.
OUT="${OUT:-results/schema_llm/export/${STAMP}-${NAME}}"
LOG="${LOG:-results/schema_llm/logs/${NAME}.log}"

WORKERS="${WORKERS:-12}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-12}"   # PER DEPLOYMENT
TRIES="${TRIES:-3}"
AUDIT_N="${AUDIT_N:-4}"
# POOL=1 round-robins the TRAPI deployment pool (4x the quota). Set POOL=0 to pin one deployment.
# WHY THIS KNOB EXISTS: a deployment that HANGS rather than erroring is invisible to
# binding/llm.pick_healthy, which counts error banners -- a call that never returns produces none.
# In the first D1 run every one of the five stuck families was assigned to the same deployment by
# round-robin (indices 1, 5, 9, 33, 37, all == 1 mod 4), and each parked a worker indefinitely.
# If a run stalls with no new checkpoints and near-zero CPU, re-run with POOL=0.
POOL="${POOL:-1}"
# REPAIR=1 regenerates families that completed but hold a rejected item. The runner keeps the
# least-bad attempt rather than dropping an item (right for lineage), so an ordinary retryable
# prose fault -- "a staged person is missing from segment 4", by far the commonest -- is otherwise
# frozen into the corpus. Pair it with a higher TRIES.
REPAIR="${REPAIR:-0}"
# WAVE=2 is the variability batch: new coordinates (unequal segment lengths so onset segment stops
# being a proxy for absolute position, PATRON staged as the prior-CONGRUENT case, a
# role_naming=named lexical ceiling; and for D2 staff-vs-staff pairs, late resolution and a
# garden-path re-resolution). It writes into the SAME dataset dir as wave 1 -- ids are disjoint --
# so the corpus accumulates to 300 items per design.
WAVE="${WAVE:-1}"
[ "$REPAIR" = 1 ] && REPAIR_ARG="--repair" || REPAIR_ARG=""
[ "$POOL" = 1 ] && POOL_ARG="--pool" || POOL_ARG="--model ${MODEL:-gpt-5.6-sol_2026-07-09}"
PHI_TAG="${PHI_TAG:-phi-4-mini-reasoning}"
QWEN_TAG="${QWEN_TAG:-Qwen3-4B-think}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5.6-terra_2026-07-09}"   # sol is rate-limited on the shared tier
JUDGE_WORKERS="${JUDGE_WORKERS:-12}"
# PARALLEL=0 (SEQUENTIAL) by default, unlike the v2 pipeline. Co-tenanting both 4B servers does not
# fit here: vLLM's --gpu-memory-utilization is a fraction of TOTAL memory, but Phi-4-mini-reasoning
# actually took 42.8 GB against its nominal 0.42 x 79.25 = 33.3 GB, leaving 36.5 GB free when Qwen
# asked for its 0.48 = 38.0 GB, and Qwen's engine refused to start. At 150 items the wall-clock cost
# of running them back to back is small; a failed second arm is not.
PARALLEL="${PARALLEL:-0}"
# Phi's full 8-layer grid is the disk high-water mark. Measured on the binding corpus at
# 22.5 MB/layer/item (Phi) and 7.9 (Qwen), 150 items needs ~27 GB + ~10 GB before pruning.
NEED_GB="${NEED_GB:-60}"

mkdir -p "$(dirname "$LOG")" "$RUN"
log() { echo "=== $(date -u +%FT%TZ) $* ==="; }
free_gb() { df -BG --output=avail /workspace | tail -1 | tr -dc '0-9'; }

log "START binding-dynamics STAGE 1 ($DESIGN): step=$STEP run=$RUN out=$OUT"

# --- 1) generation ------------------------------------------------------------------------------
# Family-serial by construction (a sibling is a continuation of its parent's ACCEPTED segments), so
# WORKERS is how many families run at once; MAX_CONCURRENCY caps requests per deployment and is the
# knob that actually matters on the shared tier.
if [ "$STEP" = all ] || [ "$STEP" = gen ]; then
  log "generation ($DESIGN, 150 items) -> $RUN"
  BINDING_MAX_CONCURRENCY="$MAX_CONCURRENCY" \
  uv run --extra api --directory code python -m dev.schema_llm.binding.dynamics.run \
      --design "$DESIGN" --out "../$RUN" $POOL_ARG \
      --workers "$WORKERS" --concurrency "$MAX_CONCURRENCY" \
      --tries "$TRIES" --audit-n "$AUDIT_N" --wave "$WAVE" $REPAIR_ARG 2>&1 | tee -a "$LOG"
fi

# --- 2) responses from both subject models, co-tenanted on the one GPU ---------------------------
# SKIP_JUDGE=1 on purpose: the judge is deferred into stage 3 so it can overlap capture.
if [ "$STEP" = all ] || [ "$STEP" = post ] || [ "$STEP" = collect ]; then
  log "collect both arms (PARALLEL=$PARALLEL) -> $RUN"
  RUN_DIRS="$RUN" PARALLEL="$PARALLEL" SKIP_JUDGE=1 TASK=identify \
    LOG="results/schema_llm/logs/${NAME}_collect.log" \
    bash scripts/schema_llm/run_scene_order_broaden_rlreasoning.sh 2>&1 | tee -a "$LOG"
fi

# --- 3) judge (TRAPI, background) while capture + cluster (GPU) runs in front --------------------
# Capture/merge/export delegate to run_binding_export.sh, which is parameterised by NAME/STAMP/OUT
# and knows the binding corpus's model revisions, chat-template override and layer budget. Do NOT
# use run_capture_rlsubjects_depth.sh here: it is hardcoded to the core-216 pool and the
# pre-DrGRPO subject models and ignores RUN_DIRS, so it would silently capture the wrong dataset.
if [ "$STEP" = all ] || [ "$STEP" = post ] || [ "$STEP" = latents ]; then
  avail=$(free_gb)
  log "free=${avail}G before capture (need >= ${NEED_GB}G)"
  if [ "$avail" -lt "$NEED_GB" ]; then
    log "!!! insufficient disk for capture; not starting it"; exit 1
  fi
  log "judge both tags in background with trapi:$JUDGE_MODEL"
  judge_one() {
    uv run --extra api python -m dev.schema_llm.subjects.judge --run "$RUN" \
      --judge-backend trapi --model "$JUDGE_MODEL" --tag "$1" \
      --max-workers "$JUDGE_WORKERS" --resume \
      >> "results/schema_llm/logs/${NAME}_judge.$1.log" 2>&1
  }
  judge_one "$PHI_TAG"  & jphi=$!
  judge_one "$QWEN_TAG" & jqwen=$!

  log "capture + cluster + prune both models (GPU) while the judge runs"
  STEP=capture NAME="$NAME" STAMP="$STAMP" OUT="$OUT" \
    bash scripts/schema_llm/run_binding_export.sh || log "!!! capture returned nonzero"

  log "waiting for the judge"
  wait "$jphi"  || log "!!! judge FAILED for $PHI_TAG (re-run STEP=latents to resume)"
  wait "$jqwen" || log "!!! judge FAILED for $QWEN_TAG (re-run STEP=latents to resume)"
  for tag in "$PHI_TAG" "$QWEN_TAG"; do
    log "scores.$tag.jsonl: $(wc -l < "$RUN/scores.$tag.jsonl" 2>/dev/null || echo 0) rows"
  done
fi

# --- 4) source-blind merge into the shared theme alphabet ---------------------------------------
if [ "$STEP" = all ] || [ "$STEP" = post ] || [ "$STEP" = merge ]; then
  STEP=merge NAME="$NAME" POOL_NAME="$POOL_NAME" POOL_RUNS="$POOL_RUNS" STAMP="$STAMP" OUT="$OUT" \
    bash scripts/schema_llm/run_binding_export.sh || { log "!!! merge FAILED"; exit 1; }
fi

# --- 5) export ----------------------------------------------------------------------------------
if [ "$STEP" = all ] || [ "$STEP" = post ] || [ "$STEP" = export ]; then
  log "free=$(free_gb)G before export"
  STEP=export NAME="$NAME" POOL_NAME="$POOL_NAME" POOL_RUNS="$POOL_RUNS" STAMP="$STAMP" OUT="$OUT" \
    bash scripts/schema_llm/run_binding_export.sh || { log "!!! export FAILED"; exit 1; }
  STEP=zip NAME="$NAME" POOL_NAME="$POOL_NAME" POOL_RUNS="$POOL_RUNS" STAMP="$STAMP" OUT="$OUT" \
    bash scripts/schema_llm/run_binding_export.sh || { log "!!! zip FAILED"; exit 1; }
fi

log "DONE binding-dynamics STAGE 1 ($DESIGN). Stage 2 is run_binding_dynamics_d2.sh -- do not"
log "start it until this bundle has been read."
