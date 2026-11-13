#!/usr/bin/env bash
# v3 WAVE 2 generation for both designs, then the confound audit ON THE UNION.
#
# The audit deliberately runs on wave 1 + wave 2 merged, not on wave 2 alone: balance that holds
# within a wave can still drift in the union, and the union is what ships. `--runs` takes both
# directories; `confound_audit` concatenates them and groups CV folds by family_id, which is why
# wave 2 uses its own family-id prefixes (D1aL/D1aP/D2L/D2P).
#
# Resumable: generation checkpoints per family under <RUN>/families/, so re-running this script
# picks up where it stopped rather than regenerating.
set -uo pipefail
cd "$(dirname "$0")/../.."

LOGDIR=results/schema_llm/logs
DATA=results/schema_llm/data
mkdir -p "$LOGDIR"

gen() {  # design
  local d="$1" lower
  lower="$(echo "$d" | tr '[:upper:]' '[:lower:]')"
  local out="$DATA/binding_dyn_${lower}_v3w2"
  local n
  n=$(wc -l < "$out/items.jsonl" 2>/dev/null || echo 0)
  # Completion is marked by a PASSING UNION AUDIT, not by an item count. D2 ships 147 of 150 -- the
  # three missing restaurant `pairs_b` families failed three times for a structural reason (13.11),
  # and a `>= 150` test would send this back to retry them for hours on every resume.
  if grep -q '\*\*PASS\*\*' "$out/confound_audit_union.md" 2>/dev/null; then
    echo "[wave2] $d generation already complete and audit-clean ($n items)"; return 0
  fi
  if [ "$n" -ge 150 ]; then echo "[wave2] $d already complete ($n items)"; return 0; fi
  # Already running (the two designs are independent, so they may be launched in PARALLEL): wait
  # for it rather than starting a second process on the same family checkpoints.
  if pgrep -f "dynamics.run --design $d --wave 4" > /dev/null; then
    echo "[wave2] $d generation already running; waiting for it"
    while pgrep -f "dynamics.run --design $d --wave 4" > /dev/null; do sleep 60; done
    echo "[wave2] $d generation finished ($(wc -l < "$out/items.jsonl" 2>/dev/null || echo 0) items)"
    return 0
  fi
  echo "[wave2] generating $d -> $out"
  uv run --extra api --directory code python -m dev.schema_llm.binding.dynamics.run \
    --design "$d" --wave 4 --tries 3 --audit-n 4 --workers 4 \
    >> "$LOGDIR/w4_${lower}_gen.log" 2>&1
  echo "[wave2] $d generation exit=$? items=$(wc -l < "$out/items.jsonl" 2>/dev/null || echo 0)"
}

# The two designs are INDEPENDENT -- D2 resumes from its own wave-1 corpus (binding_dyn_d2_v3),
# not from D1's wave 2 -- so they may be generated in parallel. `gen` waits for an already-running
# process instead of duplicating it, which is what makes a parallel launch safe here.
gen D1
gen D2

# Each design is audited THREE ways: wave 2 alone, and the union -- plus wave 1's existing report.
# The union is what ships, but a lift there is ambiguous on its own: wave-2 `long`/`late` items run
# 8 segments against wave 1's 6, so A5 (pre-onset text) and A10 (mention profile) see two
# populations with structurally different pre-onset lengths. A channel that lifts in the union but
# sits at the null in EACH wave separately is a passage-length artifact, not a role leak. Having
# all three reports on disk is what makes that distinguishable without a re-run.
echo "[wave2] auditing wave 2 alone, then each design's UNION with wave 1"
for d in d1 d2; do
  uv run python -m dev.schema_llm.binding.dynamics.confound_audit \
    --run "$DATA/binding_dyn_${d}_v3w2" \
    --out "$DATA/binding_dyn_${d}_v3w2/confound_audit" \
    > "$LOGDIR/w4_${d}_audit_wave2.log" 2>&1
  echo "[wave2] $d wave-2 alone: $(grep -oE '\*\*(PASS|FAIL)\*\* . [0-9]+ of [0-9]+' \
      "$DATA/binding_dyn_${d}_v3w2/confound_audit.md" 2>/dev/null || echo '(see log)')"
  uv run python -m dev.schema_llm.binding.dynamics.confound_audit \
    --runs "$DATA/binding_dyn_${d}_v3" "$DATA/binding_dyn_${d}_v3w2" \
    --out "$DATA/binding_dyn_${d}_v3w2/confound_audit_union" \
    > "$LOGDIR/w4_${d}_audit.log" 2>&1
  echo "[wave2] $d UNION:       $(grep -oE '\*\*(PASS|FAIL)\*\* . [0-9]+ of [0-9]+' \
      "$DATA/binding_dyn_${d}_v3w2/confound_audit_union.md" 2>/dev/null || echo '(see log)')"
done

# --- downstream, gated on the UNION audit ---------------------------------------------------------
# Same philosophy as the wave-1 gate: never spend GPU hours on a corpus whose confound audit failed.
# The post chain (collect -> judge + capture -> merge -> export) is the proven one from the stage
# scripts and is resumable at every step, so a retry after an interruption is always safe.
#
# POOL_NAME is what makes this ONE bundle of 300 rather than two of 150: capture runs per wave
# (it reads that wave's own activations), the export pools both.
for d in d1 d2; do
  union="$DATA/binding_dyn_${d}_v3w2/confound_audit_union.md"
  if ! grep -q '\*\*PASS\*\*' "$union" 2>/dev/null; then
    echo "[wave2] $d union audit did not PASS -- NOT collecting. See $union"
    continue
  fi
  # Capture has a hard 60 GB gate and each wave's activations are ~13 GB, so the SECOND design is
  # the one at risk: D1's capture plus the combined 300-item export spends roughly 33 GB. Say so
  # here rather than letting the stage script exit 1 fifty lines deep in its own log.
  free_gb=$(df -BG --output=avail /workspace | tail -1 | tr -dc '0-9')
  echo "[wave2] $d free disk = ${free_gb}G (capture needs >= 60G)"
  if [ "${free_gb:-0}" -lt 60 ]; then
    echo "[wave2] !!! $d capture will be REFUSED for disk. Generation and the audits are complete"
    echo "[wave2] !!! and committed; only the bundle is outstanding. The obvious candidates are the"
    echo "[wave2] !!! SUPERSEDED confounded bundles under export/20260910-binding_dyn_d{1,2}_v2"
    echo "[wave2] !!! (~19G each) -- left in place deliberately: they are not regenerable, since"
    echo "[wave2] !!! the v1 data dirs kept no activations, so removing them is a decision to take"
    echo "[wave2] !!! deliberately rather than to automate."
    continue
  fi
  # GPU pre-flight. `collect` starts a vLLM server per subject model, and two 4B servers do not
  # co-tenant on the one 80 GB card -- so an occupied GPU means OOM, not slowness. This box is
  # shared: the card read 0 MiB earlier in this session and 73 GB later, held by a process outside
  # this namespace. Wait for it rather than crashing into it.
  # Require the card to be free on THREE consecutive reads a minute apart. A single instantaneous
  # check grabs transient dips -- the gap while another job swaps models reads as "free", and this
  # pipeline then claims ~72 GB and starves it. That is exactly the race won once already.
  free_streak=0
  for _try in $(seq 1 600); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [ "${used:-0}" -lt 20000 ]; then
      free_streak=$((free_streak + 1))
      [ "$free_streak" -ge 3 ] && { echo "[wave2] GPU free on 3 consecutive checks (${used} MiB)"; break; }
    else
      [ "$free_streak" -gt 0 ] && echo "[wave2] GPU busy again (${used} MiB); streak reset"
      free_streak=0
      [ "$_try" = 1 ] && echo "[wave2] GPU busy (${used} MiB used by another process); waiting"
    fi
    sleep 60
  done
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  if [ "${used:-0}" -ge 20000 ]; then
    echo "[wave2] !!! $d GPU still busy (${used} MiB) after 8h -- NOT collecting."
    echo "[wave2] !!! Generation and the audits are complete and committed; re-run this script"
    echo "[wave2] !!! when the card is free. Every step is resumable."
    continue
  fi
  # NOTE ON SHARING THE CARD. `collect` serves each subject model at
  # --gpu-memory-utilization 0.88, i.e. ~72 GB of the 80 GB card: this pipeline does not co-tenant,
  # it takes the whole GPU. The pre-flight above only checks the card is free *now*, so on a shared
  # box it wins races against anything else that is also waiting. If another job is queued, either
  # let it go first or lower the claim with VLLM_GPU_UTIL (see scripts/schema_llm/vllm_serve.sh) --
  # but note 0.88 is sized so Phi-4-mini-reasoning fits, so lowering it needs checking, not guessing.
  echo "[wave2] $d union audit passed; running collect -> capture -> merge -> export"
  echo "[wave2] NOTE: this claims ~72 GB of the GPU for the duration"
  STEP=post \
    NAME="binding_dyn_${d}_v3w2" \
    POOL_NAME="binding_dyn_${d}_v3,binding_dyn_${d}_v3w2" \
    STAMP="$(date -u +%Y%m%d)" \
    bash "scripts/schema_llm/run_binding_dynamics_${d}.sh" \
    >> "$LOGDIR/w4_${d}_post.log" 2>&1
  echo "[wave2] $d post exit=$?"
done

echo "[wave2] cross-corpus verifier over all four directories"
uv run python -m dev.schema_llm.binding.dynamics.verify_v3 --runs \
  "$DATA/binding_dyn_d1_v3" "$DATA/binding_dyn_d1_v3w2" \
  "$DATA/binding_dyn_d2_v3" "$DATA/binding_dyn_d2_v3w2" \
  > "$LOGDIR/w4_verify.log" 2>&1
tail -20 "$LOGDIR/w4_verify.log"
echo "[wave2] done"
