# Normalized primitive input packs and incremental merging

The extraction runner accepts `--input <pack>`. A pack is a directory containing
`manifest.json`, `sequences.parquet` and `responses.parquet`, with optional task/design tables.
Future raw TSP responses need an importer once their actual format is available. This document
specifies the importer's output, not an assumed format for the future raw files.

## Sequence table

Use `dev.schema_llm.explore.combined_primitives.SEQUENCE_SCHEMA` for `sequences.parquet`.
Each row is one exact prompt-plus-response sequence:

| Field | Meaning |
| --- | --- |
| `item_id` | Namespaced ID, unique within a model; include the draw in the ID if there are multiple draws |
| `source_item_id` | Original response/item ID |
| `corpus` | Distinct corpus namespace, e.g. `tsp`; do not reuse `d1`, `d2` or `core_spot` |
| `task_family` | e.g. `tsp` |
| `model` | `phi4mr` or `qwen34b`, using the same pinned checkpoint revisions as the current pack |
| `sample_id` | Integer draw identifier |
| `schema` | Task metadata, e.g. `tsp` |
| `prompt_len` | Number of prompt tokens, including the exact chat framing |
| `n_tokens` | Length of the complete saved token sequence |
| `token_ids` | Exact integer IDs for prompt and response |
| `tokens` | Corresponding tokenizer token strings, not decoded text pieces |
| `segments` | `prompt` for the first `prompt_len` entries, then `think` or `answer` |
| `sequence_sha256` | Computed with `combined_primitives._identity(row)` |

All three lists must have length `n_tokens`. Prompt/response boundaries must be contiguous.
Sequences shorter than three response tokens are retained but excluded from clustering and get
null assignments. `responses.parquet` must contain a matching row for each sequence, keyed by
`model` and `item_id`; retain `task_family`, `sample_id`, `corpus`, response text and any useful
problem/score metadata. Optional `tasks.parquet`, `persons.parquet`, `bindings.parquet` and
`item_lookup.parquet` are copied into exports when present.

Prefer original saved token IDs and generation prompt boundaries. If only text is available,
establish exact chat framing, tokenizer revision, reasoning delimiters and terminal tokens first.
Reconstructed tokenization must be marked as such; it is not guaranteed to reproduce the original
generation sequence. No text-to-token reconstruction is performed silently by the extraction runner.

## Manifest

The manifest must include:

- `layers`: `[0, 4, 8, 12, 16, 20, 24, 28]`.
- `k`: `50`, `meta`: `8`, `seed`: `0`.
- `notes`: a list or object documenting preparation and any token reconstruction.
- `files`: relative data filenames mapped to their SHA-256 hashes; do not include the manifest itself.
- `counts`: one entry per corpus with `tasks`, `responses`, `tokens` totals. `tasks` is the number
  of sequences per model; both models must have that many responses for each corpus. `responses`
  counts both models and `tokens` sums their full prompt-plus-response lengths.
- `sources`: explicit corpus/model/task entries, for example:

```json
[
  {"corpus": "tsp", "model": "phi4mr", "task": "tsp"},
  {"corpus": "tsp", "model": "qwen34b", "task": "tsp"}
]
```

Use one task family per corpus, with no duplicate source entries. Each sequence must belong to
exactly one listed source. Preserve the current pack's `code_sha256` entries to enforce matching
algorithm code. The new pack must have its own manifest; never append TSP to the old pack in place.

## Reusable extraction folders

Run `scripts/run_primitives.sh --input <pack> --extract-only --out <work>` in the standalone
repository. `extraction.json` records the input path/hash, extraction settings, model settings
and hashes of all final node/assignment files. A full run also writes this record before merging.

Use `scripts/merge_primitives.sh --from schema=<old-work> --from tsp=<tsp-work> --out <new-merge>`.
The runner verifies completed inputs and compatibility, invokes the existing semantic merge,
then exports each input under `<new-merge>/export/<name>/` using the same merged dictionary.
Source-local fine and meta assignments are reused; merged assignments are recalculated.
Embedding vectors are currently recomputed for the joint merge. No annotation calls or old latent
captures are repeated by this runner.

Inputs and output must be separate. Changing sources/settings requires a new output directory.
Keep the old merge: adding sources may alter all merged names/IDs/groupings/counts. IDs must be
interpreted with their particular merge dictionary and manifest.

Absolute input/node paths are recorded. To relocate a completed extraction, first verify the
matching pack, then run `incremental_primitives.register(new_pack, new_work)` from Python to
rebuild its path record. This reads/checks input data and node files; it does not rerun models or
clustering. Regenerate the joint merge input record in a new output directory after relocation.
