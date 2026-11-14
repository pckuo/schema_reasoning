# Schema reasoning: datasets, behavioral evaluation, primitives and STOP injection

This is a standalone source/data snapshot. No other checkout or original results directory
is needed. Models, Python packages and Azure credentials are not included.

## Start here

Clone/download this repository and open a terminal in its root (the folder containing this README).
Use a Linux NVIDIA/CUDA machine with `uv` and Git installed. For primitive labeling, install
Azure CLI separately and log in with a TRAPI-authorized account as described below.
All `/storage/...` paths in this README are examples: replace them with writable paths on your host.
Scripts create their output folders. Install/download access is needed on the first run.

For D1/D2 behavioral response collection and grading, go directly to [BEHAVIOR.md](BEHAVIOR.md).
The frozen prompts and the existing two subjects' full results are already included.

For the recommended workflow that allows adding TSP later:

```bash
az login --use-device-code --scope api://trapi/.default
bash scripts/run_primitives.sh --check-only
bash scripts/run_primitives.sh --smoke --out /storage/schema-smoke
bash scripts/run_primitives.sh --extract-only --out /storage/schema
bash scripts/merge_primitives.sh --from schema=/storage/schema --out /storage/merge-schema
```

This produces current results now; no future TSP responses are needed. Later, follow
"Add TSP later without repeating existing extraction" below. Keep the completed schema extraction
folder and its input pack for that step.

To reproduce STOP experiments independently, run the documented checks/smoke test first, then:

```bash
bash scripts/run_vstop.sh --out /storage/vstop
```

The top-level shell entrypoints install the locked environment automatically. For the original
D1/D2 scripts under `scripts/schema_llm/`, run the documented `uv sync --locked --extra api` first. Python 3.12 is selected by
`.python-version`; retain it because the STOP shuffled control depends on NumPy's
seed stream. Do not run both jobs concurrently on one GPU without checking capacity.

| Script | Purpose |
| --- | --- |
| `scripts/run_primitives.sh` | Capture, cluster and label; optionally defer merging |
| `scripts/merge_primitives.sh` | Merge completed extractions and export shared assignments |
| `scripts/schema_llm/run_binding_dynamics_d{1,2}.sh` | Original D1/D2 stage scripts; `STEP=gen` runs generation only |
| `scripts/generate_tsp_responses.sh` | Generate TSP responses and an extraction input pack |
| `scripts/schema_llm/run_d1d2_behavior.sh` | Collect and grade the eight additional D1/D2 subject models |
| `scripts/run_vstop.sh` | Reproduce schema/TSP STOP experiments using supplied vectors |

As an alternative, `bash scripts/run_primitives.sh --out /storage/primitives` performs capture,
extraction, merging and export in one command. This also saves reusable extraction results.

## TRAPI login and model downloads

For primitive labeling, install Azure CLI separately and use a TRAPI-authorized account.
On the compute host, as the same OS user who will run the script:

```bash
az login --use-device-code --scope api://trapi/.default
bash scripts/run_primitives.sh --out /storage/primitives
```

The shell script installs the Python API clients, but does not install Azure CLI or log you in.
Authentication uses Azure CLI's cached login with automatically refreshed bearer tokens, or an
authorized managed identity. No static key needs to be pasted into the script; `OPENAI_API_KEY`
is not used by this TRAPI client. In containers/jobs, login must be available inside that runtime.

Defaults are backend `trapi`, deployment `gpt-5.6-sol_2026-07-09`, and endpoint
`https://trapi.research.microsoft.com/redmond/interactive/openai/v1`.
If needed, set `ANNOTATE_MODEL` to an accessible deployment and `OPENAI_BASE_URL` to the endpoint
provided by your TRAPI administrator before running. `ANNOTATE_BACKEND` selects the supported
annotation transport. STOP reproduction needs no annotation API or Azure login.

The primitive runner automatically downloads missing tokenizers and model weights from Hugging
Face and reuses the cache. Phi and Qwen capture checkpoints use pinned revisions:

- `microsoft/Phi-4-mini-reasoning` for the Phi captures.
- `Qwen/Qwen3-4B` for the Qwen captures.
- `Qwen/Qwen3-Embedding-4B` during semantic merging (default embedding model).

No manual weight download is required. First use needs network access and sufficient cache space;
`--check-only` does not download weights, while `--smoke` downloads/loads both capture models if
uncached. Set `HF_HOME=/storage/huggingface` before running to choose the cache location.
An existing cache must contain the pinned revisions. STOP uses only the pinned Phi checkpoint.
The embedding model uses its default Hub revision unless a frozen local copy is selected with
`EMBED_MODEL`; it is not revision-pinned by this release.

## Check first

```bash
bash scripts/run_primitives.sh --check-only
bash scripts/run_vstop.sh --check-only
bash scripts/run_primitives.sh --smoke --out /storage/primitives-smoke
bash scripts/run_vstop.sh --smoke --out /storage/vstop-smoke
```

`--check-only` verifies frozen files and dataset populations without loading weights or using a
GPU/API. Primitive smoke validates ALL token sequences against their model tokenizer, then
captures two sequences/model; it does not annotate, cluster or merge. STOP smoke validates the
full input tokenizer boundaries, captures two reference items and tests TSP cache parity;
it does not build the full vector or run the intervention matrix. Full runs use separate folders.

## Locate the saved D1/D2 prompts

The frozen prompts used in the bundled experiments are in `data/primitives/tasks.parquet`.
Read `prompt_text` and filter `corpus` to `d1` (300 prompts) or `d2` (297 prompts).
`passage_text` contains the passage alone; `gold`, `family_id`, `design` and `wave` retain
answer and design metadata. `data/primitives/manifest.json` records counts and provenance.

## Primitive extraction

`data/primitives/` contains 813 items, 1,626 responses and 4,395,770 exact saved tokens:
D1 identify (600 responses), D2 identify (594), original core spot_error (432).
Both D1/D2 generation waves are retained. All bundle-local IDs are namespaced by corpus.
Gold, family, wave and scoring-exclusion metadata remain in the tables.

The runner forwards saved IDs (never generates/re-tokenizes), captures L0/4/8/12/16/20/24/28,
fits k=50 states then 8 temporal meta-clusters per source/layer, labels both levels and jointly
merges six sources. Annotation context is neutral about task/model/corpus. Original old IDs
are not reused. Results include `tokens.parquet` with `kmeans_L*`, `metacluster_L*`,
`merged_primitive_L*`, labelled fine/meta tables, merged names and collapsed response sequences
under `OUT/export/`. Prompt token assignments remain null.

## Add TSP later without repeating existing extraction

Each dataset pack and extraction folder is immutable with respect to its inputs/settings.
Extract the current data now, then merge it separately:

```bash
bash scripts/run_primitives.sh --extract-only --out /storage/schema
bash scripts/merge_primitives.sh --from schema=/storage/schema --out /storage/merge-schema
```

The original full-run command still captures, clusters, merges and exports in one pass. It also
writes `extraction.json`, so its completed extraction folder can be reused by the merge script.
`--extract-only` stops after capture, clustering and fine/meta labeling; it writes the reusable
node results and `extraction.json`, but final token tables are produced by the merge step.

To generate TSP responses here, use the next section; its output pack can be used directly.
For external responses, prepare a separate normalized pack first (see `INPUT_FORMAT.md`).
An importer for unknown external formats is not included. Once prepared:

```bash
bash scripts/run_primitives.sh --input /storage/tsp-input --check-only
bash scripts/run_primitives.sh --input /storage/tsp-input --smoke --out /storage/tsp-smoke
bash scripts/run_primitives.sh --input /storage/tsp-input --extract-only --out /storage/tsp

bash scripts/merge_primitives.sh \
  --from schema=/storage/schema --from tsp=/storage/tsp \
  --out /storage/merge-schema-tsp --check-only
bash scripts/merge_primitives.sh \
  --from schema=/storage/schema --from tsp=/storage/tsp \
  --out /storage/merge-schema-tsp
```

`--from` is repeatable; names identify export subfolders and must be unique. Inclusion is explicit:
no TSP input is required for the initial run, and incomplete requested sources cause an error.
No try/catch hides failed extraction. Keep input packs and extraction folders at their recorded
paths: `extraction.json` stores absolute paths and hashes. Copying/moving them requires rebuilding
the extraction record against verified input paths, not re-forwarding or re-fitting the data.

The joint merge reads saved source annotations and assignments; it never re-forwards old responses,
re-fits old clusters/meta-clusters, or repeats old annotation calls. It recomputes semantic
embeddings and the joint grouping. Matching layers, k, meta-count, seed, annotation context/model
and capture checkpoints are required. Keep node/assignment files and input packs for merging.
Raw activation dumps are only needed to resume or revise extraction separately.

Results are in `OUT/merge/` and `OUT/export/schema/`, `OUT/export/tsp/`, etc. Each named export
contains the same shared merged-name dictionary plus that input's token and response assignments.
Merged IDs are local to a merge run; adding TSP may change names, IDs, groupings and counts for
existing sources. Fine/meta assignments remain unchanged. Retain the original merge folder and
use a new output folder for every changed source list or merge setting. The merge script's
`--check-only` validates compatibility/hashes without embeddings, GPU forwards or API calls.

For the bundled schema data, raw fp16 latents require about **189.4 GiB**, plus model caches
and output storage. Additional TSP extraction requires extra capacity. Clustering
holds large full-dimensional float32 matrices in host RAM. Task is confounded with corpus:
these inputs support primitive discovery, not a matched identify-versus-spot_error comparison.
Token IDs override stale prompt-text metadata in 72 core spot_error task rows.

Optional `PHI_MODEL`, `QWEN_MODEL`, `PHI_REVISION`, `QWEN_REVISION` and `EMBED_MODEL`
variables select local copies for this workflow. They must be the same weights/tokenizers.

## Generate new TSP responses

The pack includes the previously selected 100 TSP problems (levels 1–8). Generate one sampled
response per problem for each pinned Phi/Qwen model; no TRAPI login is needed for generation:

```bash
bash scripts/generate_tsp_responses.sh --check-only
bash scripts/generate_tsp_responses.sh --smoke --out /storage/tsp-generation-smoke
bash scripts/generate_tsp_responses.sh --out /storage/tsp-generation

bash scripts/run_primitives.sh \
  --input /storage/tsp-generation/pack --extract-only --out /storage/tsp
bash scripts/merge_primitives.sh \
  --from schema=/storage/schema --from tsp=/storage/tsp \
  --out /storage/merge-schema-tsp
```

Generation uses temperature 0.8, top-p 0.95, top-k 50, one draw, no extra system message,
Phi's packed instruct template and Qwen thinking enabled. The default completion cap is 32,768;
`--max-new-tokens 8192` selects the D1/D2 length budget instead. Models download automatically.
The two models run sequentially. No latent capture or labeling happens during generation.
Cluster labeling in the later extraction step requires the TRAPI setup above.

`--check-only` validates task inputs without loading weights. `--smoke` generates up to two
problems/model with a cap of 256 tokens; incomplete answers/thinking are expected at that cap.
Use a separate output folder for the full run. Repeating the full command resumes missing rows.
Changed prompts, budgets, settings or runtime require a new output folder.

`OUT/generation.json` records generation settings, engine, checkpoint revisions, template hash
and runtime. `OUT/shards/` contains exact generated token IDs, prompt boundaries and per-response
seeds. `OUT/pack/` is the ready-to-use normalized input, including text, token IDs, metadata and
flags for hitting the cap, termination and closed thinking. Save these files for later extraction.

To supply different problems, pass `--tasks /path/tasks.jsonl`. Each line needs a unique `uid`
and a nonempty `prompt`; optional problem metadata is retained. The default task list is
`data/tsp_generation/tasks_100.jsonl`. These new generations are separate from the frozen TSP
baselines used by STOP reproduction.

## STOP injection

This workflow uses the ORIGINAL core identify traces and original L16 dictionary, independently
of updated primitive extraction. Do not replace those inputs with freshly fitted assignments.

1. Verify the original 166-item capture frame and the 17 capped/solved/all-right intervention items.
2. Use the bundled `stop`, `shuffled`, `switch`, and `solved` vectors after validating their
   dimensions, finite values, unit norms and release checksum. No vector regeneration is needed.
   The random vector is unused by these experiments and is not required.
3. Run the main schema matrix at L16: own alpha=0 resume plus stop/shuffled/switch/solved at
   doses 0.25, 0.5, 1.0, sustained injection, original remaining 8192-token completion budget.
4. Run TSP transfer on six frozen problems x three stored baselines, at 25/50/75% reasoning,
   original cap 16384, two injection draws, shared inverse-CDF random streams and original arms.
   Baselines are REUSED, not regenerated. Cache parity is a hard prerequisite.
5. Write paired statistics, answer-only accuracy/specificity and figures. TSP outputs report
   all six problems and the original manuscript's three separately, from the same generations.

`--ablations` adds L8/L24, sampling and one-shot schema arms. These reuse the current frozen
17-item population; historical layer figures used an earlier smaller cohort and need not match
numerically. The negative-dose arm and the newer text-forcing controls are outside the default
reproduction scope. No new text-control implementation is implied by this release.

When supplied, recovered vectors are stored in `data/vstop/supplied/directions.npz`, with
`comparison.json` recording their checksum, array checks and reference cosine comparison.
The supplied file contains all four directions used by the experiments. The user approved its
reference cosine differences, including `shuffled|switch` (0.005574) and `shuffled|solved`
(0.003086). These are recorded for transparency and do not block the run. The supplied bytes
are used directly; `OUT/vectors/check_report.json` records their checksum and accepted differences.
The retained reference manifest supplies the original median L16 state norm for dose scaling.

With supplied vectors, the default runner skips recapture and regeneration. If a pack is built
without supplied vectors, the original regeneration/reference gates remain the fallback.
Cross-device bfloat16 arithmetic can change generated results. Stored baseline tails are never
the causal control. Model is pinned Phi-4-mini-reasoning; TSP uses the packed instruct template.
STOP intentionally accepts only that pinned Hub checkpoint/revision (or its existing HF cache),
not arbitrary local model overrides.

Completed schema items, reference capture items and TSP cells are resumable. Use the same command
and output directory. Source/configuration/runtime changes require a new output directory.
STOP logs are in `OUT/logs/`; derived data in `OUT/schema/`, `OUT/tsp/`, `OUT/tsp_original3/`,
`OUT/analysis/` and `OUT/figures/`. Input data is never modified. Repeated stage analysis is cheap.

## Source, validation and GitHub

Everything specific to the current experiments is inside this repository:

| Location | Included contents |
| --- | --- |
| `code/dev/` | D1/D2 generator and YAML assets, validation tools, extraction/STOP modules and CPU tests |
| `scripts/` | Generation, extraction, merging and STOP shell entrypoints |
| `data/primitives/` | Updated schema task/response metadata and exact token sequences |
| `data/vstop/core/` | Original identify tokens/assignments and primitive names |
| `data/vstop/reference/` | Capture frame, head selection, scaling and fingerprints |
| `data/vstop/supplied/` | Supplied four-direction file and accepted comparison report |
| `data/vstop/tsp/` | Six frozen STOP-transfer prompts, 18 baselines and the chat template |
| `data/tsp_generation/` | Selected 100 TSP prompts and the Phi generation template |
| `configs/vstop.json` | STOP experiment protocol |
| `pyproject.toml`, `uv.lock`, `.python-version` | Installable package and locked environment |
| `INPUT_FORMAT.md`, `VALIDATION.md`, `MANIFEST.json` | Input contract, checks and hashes |

The frozen STOP-transfer TSP data is distinct from new responses for primitive extraction.
New TSP responses are not bundled: generate them with the provided script, or import external
responses once their format is known. Extraction and joint merging accept the normalized format.
External requirements are GPU/CUDA, `uv`/Git, Azure CLI/TRAPI authorization for annotation, and
downloadable Python packages/model weights. No other project checkout, reference-code directory,
private dataset path, model weights or credentials need to be copied from the original workspace.

`code/dev/` is the required transitive import closure of the existing research code, plus a small
path/orchestration adapter. It is not a second implementation of the algorithms. Protocol settings
are in `configs/vstop.json`. `MANIFEST.json` verifies data, source, scripts and environment files.
Dependencies are locked in `uv.lock`; model/embedding revisions and remaining limitations are
documented in the input manifest and code. The embedding model uses the existing default Hub
revision unless `EMBED_MODEL` points at a frozen local copy.

CPU tests:

```bash
uv run --with pytest pytest code/dev/schema_llm/tests/ -q
```

GPU capture, generation, live annotation and neural embedding must still be smoke-tested on the
compute host. See `VALIDATION.md` if present for checks actually run while packaging this snapshot.

The repository contains compact Parquet inputs, not weights or activation dumps. `results/`,
`.venv/`, caches and logs are ignored. To publish, create a GitHub repository and push this folder;
upload this folder's contents as the repository root, including `.gitignore` and `.python-version`.
Keep all manifest-listed files and `MANIFEST.json` together. Do not add generated outputs or
credentials. No remote repository was created by packaging.

## Generation code for `20260915-binding_dyn_d{1,2}_v3w2`

This section describes the two generation waves pooled into the frozen source exports
`20260915-binding_dyn_d1_v3w2` (300 items) and `20260915-binding_dyn_d2_v3w2` (297 items).
The original shell scripts are copied **unchanged** under `scripts/schema_llm/`:

- `run_binding_dynamics_d1.sh`: original D1 stage runner.
- `run_binding_dynamics_d2.sh`: original D2 stage runner.
- `run_binding_dyn_wave2.sh`: original second-wave generation/audit/downstream orchestration,
  included to inspect the historical workflow.

Use the D1/D2 runners' existing **`STEP=gen`** option for generation only. Their original
`WAVE` values are preserved:

| Release generation wave | Original shell setting | Implementation | Raw output folder name |
| --- | --- | --- | --- |
| v3 wave 1 | `WAVE=3` | `build_wave3()` | `binding_dyn_d1_v3` / `binding_dyn_d2_v3` |
| v3 wave 2 | `WAVE=4` | `build_wave4()` | `binding_dyn_d1_v3w2` / `binding_dyn_d2_v3w2` |

The values 3/4 are the generator's existing internal revision numbers. The **export** named
`20260915-binding_dyn_d*_v3w2` pools both raw generation folders; the raw folder
`binding_dyn_d*_v3w2` contains only the second wave.

### Where to inspect the generation procedure

- `code/dev/schema_llm/binding/dynamics/design.py`: `build_wave3()` constructs release wave 1
  with balanced identity assignments and neutral opening beats; `build_wave4()` adds the
  long/pressure D1 arms and late/pairs D2 arms, continuing the first wave's assignment history.
- `code/dev/schema_llm/binding/dynamics/prose.py`: prompts used to generate seed passages,
  continuations, ambiguous prefixes and resolving branches.
- `code/dev/schema_llm/binding/dynamics/run.py`: generation order, pinned shared prefixes,
  retries, acceptance checks and resumable family checkpoints.
- `code/dev/schema_llm/binding/dynamics/audit.py` and `verify.py`: per-cut LLM audits and
  deterministic passage checks. `confound_audit.py` and `verify_v3.py` provide corpus checks.
- `code/dev/schema_llm/binding/`: shared cast, task, gold, record and API helpers;
  `binding/pools/`: name/beat/distractor YAML assets;
  `code/dev/schema_llm/core/schemas/`: airport and restaurant YAML specifications.


These are the retained generator implementation and design notes for this dataset family.
The current source includes later fixes. Historical seeds/settings are not fully recovered
(the retained `balance.py` notes that the shipped first-wave name-pool seed was not recorded).
New API generations therefore do not promise byte-identical reproduction of the saved passages;
`data/primitives/tasks.parquet` remains the authoritative frozen prompt set.

### Run the original scripts with generation only

Install the locked API dependencies and authenticate with Azure CLI as described above:

```bash
uv sync --locked --extra api
az login --use-device-code --scope api://trapi/.default
```

Run from the root containing this README. The explicit `STEP`, `WAVE` and `NAME` settings
below select the generation stage and the source datasets' original folder names:

```bash
STEP=gen WAVE=3 NAME=binding_dyn_d1_v3 bash scripts/schema_llm/run_binding_dynamics_d1.sh
STEP=gen WAVE=4 NAME=binding_dyn_d1_v3w2 bash scripts/schema_llm/run_binding_dynamics_d1.sh
STEP=gen WAVE=3 NAME=binding_dyn_d2_v3 bash scripts/schema_llm/run_binding_dynamics_d2.sh
STEP=gen WAVE=4 NAME=binding_dyn_d2_v3w2 bash scripts/schema_llm/run_binding_dynamics_d2.sh
```

Outputs land under `results/schema_llm/data/<NAME>/`; logs under `results/schema_llm/logs/`.
The second-wave generator automatically resumes the same design's first-wave assignment history
from `results/schema_llm/data/binding_dyn_d1_v3/` or `binding_dyn_d2_v3/`. Keep these names and
generate the full first wave before its second wave. The stage runners default to the four-model
TRAPI pool in `binding/config.py`. To use one accessible deployment, prefix a command with
`POOL=0 MODEL=<deployment>`. Other original controls include `WORKERS`, `MAX_CONCURRENCY`,
`TRIES`, `AUDIT_N` and `REPAIR`.

The copied scripts contain the full historical pipeline and default to `STEP=all`; use the
explicit `STEP=gen` commands above for this standalone generation handoff. Other stages, and
the full `run_binding_dyn_wave2.sh`, call collection/capture/export runners from the original
checkout that are outside this package's generation scope.

For a small API smoke test, invoke the same original Python generation driver directly:

```bash
uv run --no-sync python -m dev.schema_llm.binding.dynamics.run --design D1 --wave 3 --smoke --out results/schema_llm/data/binding_dyn_d1_v3_smoke
uv run --no-sync python -m dev.schema_llm.binding.dynamics.run --design D2 --wave 3 --smoke --out results/schema_llm/data/binding_dyn_d2_v3_smoke
```

Generation and per-cut audits use TRAPI; no GPU or local model weights are needed. Smoke runs
use the driver's default deployment `gpt-5.6-sol_2026-07-09`; `--model` overrides it.
Each wave plans 150 items per design. Saved and accepted counts can be lower after retries;
the frozen D2 export contains 297 saved items across both waves. Re-running an identical command
resumes family checkpoints in `OUT/families/`; `REPAIR=1` retries families containing rejected
items. Use a fresh folder when changing settings, and inspect acceptance summaries before
continuing. A full second wave needs the same design's full first-wave name pool.

Outputs are `OUT/items.jsonl` (passages, casts, design metadata, tasks and acceptance flags),
`OUT/probes.jsonl` (`probe_prompt` and `gold`), `OUT/raw/`, family checkpoints and
`OUT/family_lengths.json`. Fresh Phi/Qwen responses, collection and conversion into the
normalized extraction format require a separate step; see `INPUT_FORMAT.md`.
The frozen extraction input remains in `data/primitives/`.

### Validate both waves together

The release's balance is defined on the union: the second wave deliberately corrects the first
wave's identity imbalance. Individual-wave audits are diagnostic and can fail even when the
combined dataset passes. Do not require an isolated-wave PASS as a release gate.

```bash
uv run --no-sync python -m dev.schema_llm.binding.dynamics.confound_audit --run results/schema_llm/data/binding_dyn_d1_v3
uv run --no-sync python -m dev.schema_llm.binding.dynamics.confound_audit --run results/schema_llm/data/binding_dyn_d1_v3w2
uv run --no-sync python -m dev.schema_llm.binding.dynamics.confound_audit --runs results/schema_llm/data/binding_dyn_d1_v3 results/schema_llm/data/binding_dyn_d1_v3w2 --out results/schema_llm/data/binding_dyn_d1_v3w2_audit_union
uv run --no-sync python -m dev.schema_llm.binding.dynamics.confound_audit --run results/schema_llm/data/binding_dyn_d2_v3
uv run --no-sync python -m dev.schema_llm.binding.dynamics.confound_audit --run results/schema_llm/data/binding_dyn_d2_v3w2
uv run --no-sync python -m dev.schema_llm.binding.dynamics.confound_audit --runs results/schema_llm/data/binding_dyn_d2_v3 results/schema_llm/data/binding_dyn_d2_v3w2 --out results/schema_llm/data/binding_dyn_d2_v3w2_audit_union
uv run --no-sync python -m dev.schema_llm.binding.dynamics.verify_v3 --runs results/schema_llm/data/binding_dyn_d1_v3 results/schema_llm/data/binding_dyn_d1_v3w2 results/schema_llm/data/binding_dyn_d2_v3 results/schema_llm/data/binding_dyn_d2_v3w2
```

Corpus checks need full datasets; smoke runs cannot establish balance. Review the combined-wave
checks before collecting model responses.

## Collect D1/D2 responses from eight additional models

The frozen 597 prompts (both waves) are ready to run on six local vLLM models plus GPT-5.5 and
DeepSeek-V4-Pro over TRAPI. The default judge matches the existing D1/D2 judge, GPT-5.6 Terra.
See [BEHAVIOR.md](BEHAVIOR.md) for the model roster, serving requirements, selected-model/stage
commands, resume behavior and output dictionary.

```bash
az login --use-device-code --scope api://trapi/.default
bash scripts/schema_llm/run_d1d2_behavior.sh --check-only
bash scripts/schema_llm/run_d1d2_behavior.sh --smoke --out /storage/d1d2-eightway-smoke
bash scripts/schema_llm/run_d1d2_behavior.sh --include-existing --out /storage/d1d2-eightway
bash scripts/schema_llm/run_d1d2_behavior.sh --stage export --include-existing --out /storage/d1d2-eightway
```

The full job produces 4,776 responses and 3,984 grades. Responses to 99 unresolved D2 controls
are retained with null accuracy. The runner exports stable task IDs, source/wave/family metadata,
judge identity, scores and verbatim texts for later combination with other task families.
The final offline export includes the bundled Phi/Qwen grades, yielding all ten models on D1/D2
(5,970 rows). It saves an explicit task crosswalk and export checksums with the results.
