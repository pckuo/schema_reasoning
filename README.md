# Schema reasoning: primitive extraction and STOP injection

This is a standalone source/data snapshot. No other checkout or original results directory
is needed. Models, Python packages and Azure credentials are not included.

## Start here

Clone/download this repository and open a terminal in its root (the folder containing this README).
Use a Linux NVIDIA/CUDA machine with `uv` and Git installed. For primitive labeling, install
Azure CLI separately and log in with a TRAPI-authorized account as described below.
All `/storage/...` paths in this README are examples: replace them with writable paths on your host.
Scripts create their output folders. Install/download access is needed on the first run.

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

All three shell scripts install the locked environment automatically. Python 3.12 is selected by
`.python-version`; retain it because the STOP shuffled control depends on NumPy's
seed stream. Do not run both jobs concurrently on one GPU without checking capacity.

| Script | Purpose |
| --- | --- |
| `scripts/run_primitives.sh` | Capture, cluster and label; optionally defer merging |
| `scripts/merge_primitives.sh` | Merge completed extractions and export shared assignments |
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

When TSP arrives, convert it to a separate normalized pack first (see `INPUT_FORMAT.md`).
The raw TSP format is not known yet, so a raw-file importer is not included. Once converted:

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
| `code/dev/` | Required source modules, including subprocess entrypoints, plus CPU tests |
| `scripts/` | Three executable shell entrypoints |
| `data/primitives/` | Updated schema task/response metadata and exact token sequences |
| `data/vstop/core/` | Original identify tokens/assignments and primitive names |
| `data/vstop/reference/` | Capture frame, head selection, scaling and fingerprints |
| `data/vstop/supplied/` | Supplied four-direction file and accepted comparison report |
| `data/vstop/tsp/` | Six frozen STOP-transfer prompts, 18 baselines and the chat template |
| `configs/vstop.json` | STOP experiment protocol |
| `pyproject.toml`, `uv.lock`, `.python-version` | Installable package and locked environment |
| `INPUT_FORMAT.md`, `VALIDATION.md`, `MANIFEST.json` | Input contract, checks and hashes |

The frozen STOP-transfer TSP data is distinct from the future dataset for primitive extraction.
That future dataset is not yet available and is not bundled. Its raw-format importer must be added
once a sample is provided; extraction and joint merging already accept the normalized format.
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
