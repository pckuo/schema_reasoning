# Handoff: D1/D2 identify + original core spot_error primitives

_2026-11-12. This pack supplies inputs and executable code. Run the expensive stages on
the collaborator's GPU machine; preparing the pack does not run a model or call an annotator._

## Scope

| corpus | frozen source export | task | items | responses |
|---|---|---|---:|---:|
| d1 | `20260915-binding_dyn_d1_v3w2` | identify | 300 | 600 |
| d2 | `20260915-binding_dyn_d2_v3w2` | identify | 297 | 594 |
| core_spot | `20260824-rlreasoning_v1` | spot_error | 216 | 432 |
| total | | | 813 | 1,626 |

Both subject models: Phi-4-mini-reasoning (`phi4mr`) and Qwen3-4B thinking (`qwen34b`).
The D1/D2 bundles each contain **both waves**; do not select only one wave. Original core
`identify` and the older binding-dynamics v1/v2 corpora are excluded.

This is primitive discovery across datasets. Task and dataset are confounded; a joint merge
does not make this a controlled identify-versus-spot_error experiment. Native role inventories,
scene structures, and prose differ. Preserve corpus and family labels in downstream analyses.

## Input pack

Default prepared location: `results/schema_llm/export/20261112-combined-primitives-input/`.
Send the **whole directory**. A minimal source snapshot, project dependency specification and
lockfile are included under `runtime/`; no other checkout or source data directories are needed.
The model weights and third-party dependencies are not bundled; download/install them on the
compute host, or supply local model copies.

- `sequences.parquet`: one row per model response, with exact ordered `token_ids`, token strings,
  segment labels, prompt length, sequence checksum, schema, task, model, and corpus. This replaces
  the input token table's repeated per-token keys; it contains all 4,395,770 selected tokens.
- `tasks.parquet`, `responses.parquet`, `item_lookup.parquet`: all selected source columns, with
  namespaced `item_id` (e.g. `d1__item_0001`), `corpus`, and original `source_item_id`.
- `persons.parquet`, `bindings.parquet`: available D1/D2 annotations, with the same ID mapping.
  Core spot_error has no such exported tables; do not assume these cover every task.
- `provenance/`: source READMEs and D1/D2 combined-wave confound audits.
- `pipeline.py`, `run.sh`, this README, and `manifest.json`: executable stages and checksums.
- `runtime/`: exact algorithm sources, runner source, tests, project metadata and `uv.lock`,
  with Python 3.11 selected. Algorithm and input checksums are verified before each stage.

**Token IDs are authoritative.** Do not generate text, re-tokenize responses, or apply a chat
template. In particular, 72 core spot_error task rows have stale instruction text in
`tasks.prompt_text`; the saved capture token sequences all contain the newer instruction.
Retain the original task text as provenance, but do not use it as a forward input. The token table
also avoids Phi's instruct-versus-reasoning chat-template mismatch.

## Run on the GPU machine

Install the packed environment, then launch from any directory:

```bash
cd /absolute/path/to/pack/runtime
uv sync --locked --extra gpu --extra api
az login --scope api://trapi/.default
bash /absolute/path/to/pack/run.sh /absolute/path/to/new-work-directory
```

Default remote checkpoints/revisions are frozen in the pack manifest. For local copies:

```bash
PHI_MODEL=/models/Phi-4-mini-reasoning \
QWEN_MODEL=/models/Qwen3-4B \
bash /absolute/path/to/pack/run.sh /absolute/path/to/new-work-directory
```

`PHI_REVISION` / `QWEN_REVISION` are optional overrides. A local directory must contain the
same weights and tokenizer; tokenizer identity alone does not prove weight identity. Record
and retain the loaded-checkpoint provenance under `work/capture_config/`.

The shell runner invokes separate processes, freeing each subject model before loading the next
and freeing both before loading the embedding model. The saved sequences contain 2,845,036 Phi
tokens and 1,550,734 Qwen tokens. Eight layers of fp16 residuals require approximately **189.4 GiB**
(Phi 130.2, Qwen 59.2), excluding weights, caches and derived outputs. Full-dimensional clustering
also holds large float32 matrices in host RAM. Check available RAM/storage before launching.

1. **Validate:** verify pack and algorithm checksums. For each model, check every saved ID and
   `convert_ids_to_tokens` result against its tokenizer before loading weights or forwarding anything.
2. **Capture:** forward each saved full sequence once, capture the decoder-block output at
   L0/4/8/12/16/20/24/28, and store finite fp16 `[T, hidden]` arrays. Never truncate. Capture is
   resumable at sequence boundaries; a matching metadata file and all layer files are required.
3. **Cluster:** independently fit each of the six corpus/model sources at each layer using the
   existing full-z-score pipeline (`pca=None`, k=50, meta=8, seed=0), response tokens only.
   Annotate both fine states and meta-clusters, retaining labels, descriptions, and evidence.
   A common neutral annotation context avoids assuming spot_error responses identify roles.
   Missing annotations stop the run and can be retried without reclustering.
4. **Merge:** one source-blind semantic average-linkage merge of 384 nodes (6 sources × 8 layers
   × 8 meta-clusters), using `Qwen/Qwen3-Embedding-4B`. Fine-state labels do not enter the merge.
5. **Export:** write all three levels of assignments and response-level merged sequences.

The annotator defaults to `gpt-5.6-sol_2026-07-09` over TRAPI. Set `ANNOTATE_MODEL` and
`ANNOTATE_BACKEND` (`trapi` or `copilot`) before the first run if necessary. Use one annotator for
all sources. Changing it defines a new run and requires a new work directory. `EMBED_MODEL` can
point at a local copy of Qwen3-Embedding-4B. The existing semantic merge uses that model's default
revision; record/pin the local copy for strict replication.

Individual stages are also available:

```bash
cd /absolute/path/to/pack/runtime
export PYTHONPATH="$PWD/code${PYTHONPATH:+:$PYTHONPATH}"
uv run python /path/to/pack/pipeline.py verify --pack /path/to/pack
uv run python /path/to/pack/pipeline.py capture --pack /path/to/pack \
  --work /path/to/smoke-work --model phi4mr --limit 2
```

`--limit` is a capture smoke check only: all model sequences are still tokenizer-validated first.
Use a separate smoke directory. Clustering refuses incomplete captures. The default full runner
does not set a limit. API transport retries/timeouts follow the existing project implementation.

## Outputs under the work directory

- `captures/<corpus>/activations/<model>/<namespaced-item>__<task>/`: 8 layer arrays and metadata.
- `nodes/<corpus>/<model>/kmeans.npz`: int16 `stack`, `[8, N_response_tokens]` fine-state IDs.
- `nodes/<corpus>/<model>/kmeans.json`: passage offsets, state-to-meta mapping, fine profiles,
  labels and descriptions. Empty states map to -1 and have no fabricated annotation.
- `nodes/<corpus>/<model>/clusters.npz` and `clusters.json`: aligned meta assignments/profiles/labels.
  Layer caches and annotation evidence remain beside these files for inspection/resume.
- `merge/semavglink/merge.json`: the one shared primitive alphabet and complete node mapping.
- `export/tokens.parquet`: per-token `kmeans_L*`, `metacluster_L*`, `merged_primitive_L*` at all
  8 layers, plus exact original token IDs, strings and segments. Prompt assignments are null.
- `export/fine_clusters.parquet`, `clusters.parquet`, `merged_primitive_names.json`: fine states,
  meta-clusters and joint merged names; `response_sequences.parquet` includes run-collapsed
  merged-primitive sequences at every layer, original responses and evaluation fields.
- The original selected task/response/person/binding/lookup tables are also copied to `export/`.
  `export/manifest.json` records counts, skipped responses, and input/merge checksums.

Responses with fewer than 3 response tokens are retained but excluded by the existing clustering
rule; their assignments remain null, sequences empty, and exclusions are listed in the final manifest.
Cluster IDs are source/layer-local. Meta IDs are source/layer-local. Only the new merged IDs share
meaning across this run. None may be compared numerically with old bundle IDs.

## Rebuild the input pack locally

```bash
uv run python -m dev.schema_llm.explore.combined_primitives prepare \
  --pack results/schema_llm/export/20261112-combined-primitives-input
```

Preparing requires only NumPy and PyArrow; the full workflow additionally uses the repository's
SciPy/scikit-learn/matplotlib stack, GPU dependencies and annotation backend. CPU fixture tests
cover packing, tokenizer rejection, real clustering/merge/export, resume and assignment alignment.
GPU forwards and live annotation/embedding calls must still be smoke-tested on the compute host.
