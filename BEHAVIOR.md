# Collect and grade eight more models on D1/D2

Run `scripts/schema_llm/run_d1d2_behavior.sh` from this standalone repository. It installs the
locked API/client environment, serves local subjects one at a time, collects their answers and
grades them with the original binding-aware judge. It reads the frozen D1/D2 prompts; dataset
generation is not needed. The original Phi-4-mini-reasoning and Qwen3-4B responses remain in
`data/primitives/`; their complete response/grade records are also bundled under
`data/d1d2_behavior/existing/` for an offline ten-model export.

## Run

Use Linux, NVIDIA/CUDA, `uv`, Git and Azure CLI. Log in as the OS user running the job:

```bash
az login --use-device-code --scope api://trapi/.default
bash scripts/schema_llm/run_d1d2_behavior.sh --check-only

# First exercise generation AND grading, one prompt per wave per design.
bash scripts/schema_llm/run_d1d2_behavior.sh \
  --smoke --out /storage/d1d2-eightway-smoke

# Then all 597 prompts per new model, followed by the combined ten-model export.
# Use a different folder from smoke.
bash scripts/schema_llm/run_d1d2_behavior.sh --include-existing --out /storage/d1d2-eightway

# After collection/grading, build the combined D1/D2 table for all ten models.
# This step uses saved results and makes no generation or grading API calls.
bash scripts/schema_llm/run_d1d2_behavior.sh --stage export --include-existing \
  --out /storage/d1d2-eightway
```

`--check-only` verifies release hashes, both waves, unique IDs and exact prompt/passage/gold
identity against the bundled extraction pack. It does not call an API or load/download model
weights. It also checks all 1,194 existing Phi/Qwen responses and full grades against that pack.
Installing Python dependencies may need network access. The smoke run is real GPU/API
work: four responses and four grades per selected model. Its budgets match the full run.

The full run produces **4,776 responses** (8 x 597) and **3,984 grades** (8 x 498). D1 has
300 scorable prompts. D2 has 297 prompts, of which 99 unresolved controls carry
`scoring_exclude`: collect their responses, retain them with null accuracy, and omit judge calls
against their non-unique role assignment. Use both waves when estimating accuracy.

The final `--include-existing` export has **5,970 rows = 10 models x 597 prompts**:
4,980 scorable rows and 990 excluded rows. It imports saved Phi/Qwen records without new
generation or grading. The same D2 exclusion is applied to all ten models. It requires the
original Terra judge; a changed judge is rejected rather than mixed with the existing grades.
You can add `--include-existing` to the full `--stage all` command to create this combined table
automatically after the eight new subjects finish.

All response and valid verdict files are reused on rerun. A killed job resumes missing responses
and verdicts. Changed prompts, budgets, judge or collection configuration require a fresh output
folder. Keep the entire output directory when resuming. Do not run two processes in the same
output folder simultaneously.

## Models and decoding

`--models` accepts a comma-separated selection of these exact keys:

| Key | Model/deployment | Serving | Decoding |
| --- | --- | --- | --- |
| `phi-4` | `microsoft/phi-4` | local vLLM | greedy |
| `phi-4-reasoning` | `microsoft/Phi-4-reasoning` | local vLLM | Phi reasoning system; T=.8, top-p=.95, top-k=50 |
| `phi-4-reasoning-plus` | `microsoft/Phi-4-reasoning-plus` | local vLLM | Phi reasoning system; T=.8, top-p=.95, top-k=50 |
| `Qwen3-14B-think` | `Qwen/Qwen3-14B` | local vLLM | thinking on; T=.8, top-p=.95, top-k=50 |
| `deepseek-r1-distill-qwen-14b` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` | local vLLM | no system; T=.6, top-p=.95, top-k=50 |
| `Qwen3.8-27B-think` | `Qwen/Qwen3.8-27B` | local vLLM | thinking on; T=.8, top-p=.95, top-k=50 |
| `deepseek-v4-pro` | `DeepSeek-V4-Pro_2026-04-23` | TRAPI | deployment defaults |
| `gpt-5.5` | `gpt-5.5_2026-04-24` | TRAPI | deployment defaults |

Defaults preserve the historical single-draw panel's decoding: local and hosted completion
caps are 8,192. The local context window is 12,288, matching `run_tenway_binding.sh`'s default.
The later five-draw resampling runner used a 16,384-token hosted cap; this runner follows the
original single-draw cap. Local HTTP requests have no
SDK timeout, avoiding the earlier 600-second truncation artifact. Do not enable a reasoning
parser on the server: the client stores inline reasoning tags and the judge grades the answer
after the final `</think>`. The existing client also folds separate hosted reasoning content
into the saved trace.

The old Distill launcher's comment said "no top-k", but it omitted `--top-k` and the actual
collector CLI default was 50. This runner explicitly uses 50 to match that execution path.

The hosted calls leave `reasoning_effort`, temperature and top-p unset, matching the earlier
deployment-default comparison. The historical audit found DeepSeek-V4-Pro defaulted to no
reasoning while GPT-5.5 used hidden reasoning; this is not equal reasoning compute. Deployment
defaults can change. The exact deployment names and this policy are saved in `experiment.json`.
An unavailable deployment fails the run; it is never replaced silently.

The judge is **`gpt-5.6-terra_2026-07-09` over TRAPI**, matching the existing D1/D2 score files.
`--judge-model` can select another accessible deployment, but use a new output directory and
regrade comparison subjects consistently. The historical core tables use other/mixed judges;
combining them directly would not be judge controlled.

## GPU serving and downloads

The script starts `vllm_serve.sh` in an isolated `.vllm-venv/` (ignored by Git) with vLLM 0.25.1.
It downloads missing weights automatically and reuses the Hugging Face cache. Each model's Hub
commit is resolved on first use, recorded in `revision.<model>.json` and passed to `--revision`
on every resume. An existing server environment with another vLLM version is rejected.
Package versions and server logs are retained. No remote server is assumed for the local models.

The 27B model alone has about 56 GB of bf16 weights; an 80 GB GPU was the original serving
configuration. Leave room for the KV cache and avoid other jobs occupying the card. Set these
variables before running if needed:

```bash
export HF_HOME=/storage/huggingface
export VLLM_VENV=/storage/d1d2-vllm-venv
export VLLM_GPU_UTIL=0.90
```

Models run sequentially: collect one model's D1/D2 responses, grade them, then proceed to the
next model. Collection and grading do not overlap by default. Within a model, collection uses
up to 8 concurrent requests; grading uses up to 4 concurrent judge requests.

`--concurrency` defaults to 8, `--port` to 8400 and `--judge-workers` to 4. Context and completion
budgets have CLI flags shown by `--help`; keep the same collection settings on resume. A fixed
server seed is recorded, but sampled outputs are not promised to be identical after restarting
or changing scheduling. Completed responses are reused rather than regenerated.

## Separate stages or selected models

For a small first API check:

```bash
bash scripts/schema_llm/run_d1d2_behavior.sh --smoke \
  --models gpt-5.5 --out /storage/d1d2-gpt-smoke
```

Stages let the GPU job finish collection before grading. These commands are sequential and
share the same output folder and settings:

```bash
bash scripts/schema_llm/run_d1d2_behavior.sh --stage trapi --out /storage/d1d2-eightway
bash scripts/schema_llm/run_d1d2_behavior.sh --stage local --out /storage/d1d2-eightway
bash scripts/schema_llm/run_d1d2_behavior.sh --stage judge --out /storage/d1d2-eightway
bash scripts/schema_llm/run_d1d2_behavior.sh --stage export --out /storage/d1d2-eightway
```

`--stage prepare` only materializes the immutable inputs and experiment metadata. `local` and
`trapi` collect their respective subjects; `judge` grades the selected models; `export` requires
complete responses and scores for every selected model. `all` (default) does every step.
You can use `--models` to complete models incrementally in the same folder. Afterward run
`--stage export` with all desired model keys to produce their combined table; an export for a
subset describes only that subset.

To rebuild the ten-model D1/D2 CSV later, retain this standalone release and the entire output
folder, then run the final `--stage export --include-existing` command above. Source task IDs and
original scenario IDs are retained explicitly, so rebuilding does not depend on CSV row order.
The three other core task families remain a later addition.

## Outputs to return

Return the entire output folder. It contains:

- `experiment.json`: frozen input/code hashes, roster, decoding, judge and collection settings.
- `input_manifest.json` and `response_outcomes_task_id_map.jsonl`: the frozen source checksums
  and explicit task/source/scenario crosswalk saved with the returned results.
- `revision.*.json`, `runtime.json`, `server_runtime.*.json`, `server.*.log`: model and runtime
  provenance; hosted deployments have no public weight revision.
- `runs/d1/` and `runs/d2/`: exact `items.jsonl`/`probes.jsonl`, `responses.<model>.jsonl`,
  raw responses, scorable response views, `scores.<model>.jsonl`, summaries, cached raw verdicts
  and hashes binding verdicts to the input response file.
- `response_outcomes.csv`: one row per `(model, task_id, sample_id)`, including stable corpus
  IDs, original dataset/scenario IDs, wave/family/design, judge, score fields and exclusion status.
  `model` is the stable roster key; `model_display_name` matches the model labels in the older
  core `response_outcomes.csv`, and `subject_model` preserves the exact HF/deployment identifier.
- `response_outcomes_text.parquet`: the same rows plus verbatim prompt, gold and response.
- `export_manifest.json`: completion marker, per-model response/scored/excluded counts and
  checksums of the CSV, text Parquet and task map. Verify these before downstream analysis.

`task_id` is the namespaced ID from `data/primitives/tasks.parquet` (`d1__item_...` or
`d2__item_...`). It can be joined to the existing two subjects without guessing from names.
The CSV preserves judge/status columns and excludes unresolved controls from accuracy by leaving
outcomes null. Future work can combine these outputs with the existing two subjects and core
tasks after reconciling judges. All role precision/F1 metrics and full parsed per-person verdicts
are retained, including for the original two subjects. Keep the saved texts for later token-length
analysis: the three `*_n_tokens` columns from the older core CSV can be recomputed with its shared
Phi-4 reference tokenizer. This exporter leaves that tokenizer postprocessing to the later merge.
Judge transport errors and incomplete/invalid grading stop the job
and can be retried; they are not published as incorrect answers.

The exact input source datasets and their original file hashes are in
`data/d1d2_behavior/manifest.json`; `task_map.jsonl` maps the original IDs to the stable pack IDs.
