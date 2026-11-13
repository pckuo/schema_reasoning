# Packaging validation — 2026-11-12

Checks run on the standalone folder, without another checkout on its import path:

- Installed the locked Python 3.12 environment with GPU/API extras, including the pinned baukit dependency.
- Ran both actual shell entrypoints with `--check-only`. Input hashes and populations passed:
  813 primitive items / 1,626 responses; 17 schema intervention items; 166 reference capture items;
  six TSP tasks and 18 frozen baselines.
- Full CPU suite in the relocated archive: 47 passed, one skipped, no deselections.
  The new regression checks four-arm installation without random, byte preservation and malformed-vector rejection.
  This includes the real STOP population test and incremental joint merging. The skipped test
  requires an old results CSV absent from this release.
- After adding TSP generation: 49 passed, one skipped, one deselected. The previously verified
  real STOP population test was deselected on this rerun. Three new CPU tests use a generation
  stand-in to check exact ID/boundary packing, sampling parameters, templates, seeds, cap/closure
  flags, complete/partial resume, and rejection of changed settings or damaged sequences.
- `generate_tsp_responses.sh --check-only` passes on all 100 bundled tasks (200 planned responses),
  without weights, GPU or API calls. The task file is byte-identical to the previously selected
  100-problem sample. All generator module dependencies and its chat template are bundled.
  Both real pinned tokenizers passed synthetic-response packing/boundary checks without weights.
- Validated every saved sequence against its pinned tokenizer: 813 Phi and 813 Qwen primitive
  sequences, plus 216 original Phi identify sequences. All 18 frozen TSP prompt lengths and
  reasoning boundaries passed with the packed chat template.
- Replayed postprocessing on the original saved 221 schema and 432 TSP injection rows, using
  the packed baselines and tokenizer. Scoring, statistics, specificity and all 20 figures completed,
  including both all-six and original-three TSP summaries. These historical generated rows were
  temporary validation inputs and are not bundled as new experiment results.
- New adapter/builder/tests pass Ruff. Shell syntax and release hashes are checked at sealing.
- Incremental CPU fixture runs real clustering and TF-IDF average-linkage merging: six existing
  sources, then two synthetic TSP sources and a new eight-source joint merge. Original input/node
  files and exported fine/meta assignments remain unchanged; both named exports share the new
  merged dictionary. Resume works; changed source lists, duplicate sources and altered assignment
  files are rejected. Annotation is stubbed and TF-IDF replaces neural embeddings for this test.
- The actual primitive shell entrypoint passes `--extract-only --check-only` on the real input
  pack. The actual merge shell entrypoint passes `--check-only` on completed fixture sources.
  The new future-TSP raw-file importer is intentionally deferred until a sample format is known.
- Standalone audit: extracted the release under `/private/tmp`, installed the package from that
  location with the locked GPU/API extras, verified imports resolve inside the extracted folder,
  and repeated primitive/merge shell checks, all tokenizer/boundary checks and supplied-vector
  installation there. All 84 required runtime source files are included; static first-party imports
  resolve, with no symlinks to the original checkout. A temporary Git repository stages every
  release file, including hidden environment files, data and executable scripts. No individual file
  reaches 100 MiB. Only the isolated temporary copy was initialized/staged; no remote was created.

Not run here: GPU TSP response generation, GPU forwards, new injected generations, full vector regeneration/reference gates,
live primitive annotation, or neural embedding. Run both documented `--smoke` commands on the
compute host before full jobs. Smoke tests cover captures/cache parity, not the full annotation
or intervention pipeline. No model weights, credentials, generated latents or experiment outputs
are bundled. No remote GitHub repository was created.

The supplied `data/vstop/supplied/directions.npz` was copied byte-for-byte from the user's file.
It contains all four required finite float32 unit vectors with shape (3072,). Random is unused
by the packaged experiments and is not required. All six available pairwise residual cosines
were compared to the retained reference; the user accepted the differences recorded in
`data/vstop/supplied/comparison.json`. The runner now uses these supplied bytes directly,
retaining the reference median L16 state norm for dose scaling. It validates shapes, finite
values, unit norms and release checksums; vector recapture/regeneration is skipped.
No GPU experiments were run for this update. Cross-device generated text may differ.

## D1/D2 generation addition — 2026-11-13

- Copied 23 generator/helper/validation Python modules and five YAML assets byte-for-byte from
  the active repository, including package initializers. Added offline regression cases for
  both two-wave design grids and the original scripts' generation-only stage.
- In a separate copy under `/private/tmp`, all CPU tests excluding the previously verified
  real-data STOP population case passed: **51 passed, one skipped, one deselected**. No parent
  checkout was on `PYTHONPATH`; the existing locked Python 3.12 validation environment was used.
- The new tests build both designs' complete wave-3 and wave-4 grids, verify 150 planned items
  per wave, distinct family IDs, continued name pools and wave-4 segment counts, and render the
  seed, continuation, ambiguous-prefix and resolving-branch prompts without any API calls.
- The generator CLI imports and displays help from the standalone package. The added test
  passes Ruff. The original shell scripts pass `bash -n`; their `STEP=gen` branches invoke
  the existing generator with the API extra.
- A fresh offline `uv sync` attempt could not complete because this host's cache lacks the locked
  PyArrow wheel. Fresh environment installation was not revalidated in this update; the lockfile
  and dependency declarations are unchanged from the previously validated release.
- The release manifest and `.tar.gz` were refreshed; all archived file hashes and executable bits
  were checked, and existing frozen data/source files were preserved.

Live TRAPI generation and audits were not run for this addition. Run the README's D1 and D2
API smoke commands on the collaborator host. New responses from Phi/Qwen and normalization for
primitive extraction are separate work; `STEP=gen` produces passages, prompts and gold only.

### Original generation scripts restored — 2026-11-13

The handoff now copies `run_binding_dynamics_d1.sh`, `run_binding_dynamics_d2.sh` and
`run_binding_dyn_wave2.sh` byte-for-byte from `scripts/schema_llm/`, preserving that path.
The new translation wrapper has been removed. README commands use the original stage scripts
with `STEP=gen`, `WAVE=3/4` and `NAME=binding_dyn_d{1,2}_v3` / `_v3w2`. The original Python
code resolves second-wave resume paths to the same design's first-wave folder automatically.
The second-wave orchestrator is included for historical inspection; its full downstream stages
call other original-checkout scripts outside this generation handoff.

**Six targeted offline tests passed**, covering both full two-wave grids/prompt rendering and
the D1/D2 original
shell scripts with both WAVE values. Shell tests record the original scripts' `uv` arguments
using a stand-in, check design/wave/output/model values and ensure `STEP=gen` invokes only
one generation command. No live API calls or installations are made by those checks.
Original shell bytes, shell syntax, Ruff, final release hashes and the rebuilt archive are checked.
Frozen data and generator algorithm source remain unchanged.

### Shared documentation cleanup — 2026-11-13

Removed the confound-audit plan Markdown document from the shared package and its README
link. Refreshed the file manifest and archive, verifying that the document is absent and all
remaining release checksums match.
