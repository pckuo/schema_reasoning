"""Freeze existing Phi-mini spot-error tasks and historical baseline references.

No narrative generation, relabeling, or LLM calls. Historical response text is a
reference, not a matched exact-ID control for a newly executed intervention.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from .head_fv_cohort import lineage_hash, read_jsonl, recipe_from_dict
from .head_fv_models import active as active_model
from .head_fv_specificity import MODEL, bind_manifest, digest

SPEC = active_model()

SOURCES = (("spot_36", 36), ("main_72", 72))
SCHEMAS = ("restaurant", "airport")
ROOT = Path("results/schema_llm/data")
OUT = ROOT / SPEC.data_root / "scenarios/existing"


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def indexed(rows: list[dict], *, task: bool = False) -> dict:
    result = {}
    for line, row in enumerate(rows, 1):
        key = (row["scenario_id"], row["task"]) if task else row["scenario_id"]
        if key in result:
            raise ValueError(f"duplicate source key: {key}")
        result[key] = (line, row)
    return result


def join_run(run: Path, dataset: str, schema: str, expected: int) -> tuple[list[dict], dict]:
    paths = {
        "items": run / "items.jsonl",
        "probes": run / "probes.jsonl",
        "responses": run / f"responses.{SPEC.slug}.jsonl",
        "scores": run / f"scores.{SPEC.slug}.jsonl",
    }
    source_rows = {name: read_jsonl(path) for name, path in paths.items()}
    items = indexed(source_rows["items"])
    probes = indexed(source_rows["probes"], task=True)
    responses = indexed(source_rows["responses"], task=True)
    scores = indexed(source_rows["scores"], task=True)
    expected_keys = {(uid, "spot_error") for uid in items}
    if len(items) != expected or any(
        set(rows) != expected_keys for rows in (probes, responses, scores)
    ):
        raise ValueError(f"task/response/score coverage mismatch: {run}")
    rows = []
    for key, (_, probe) in probes.items():
        item = items[key[0]][1]
        recipe = recipe_from_dict(item["gen"]["recipe"])
        (task,) = [task for task in item["tasks"] if task["name"] == "spot_error"]
        if recipe.schema != schema or task["gold"] != probe["gold"]:
            raise ValueError("schema or gold changed between item and probe")
        if not probe["probe_prompt"].startswith(item["prompt"]):
            raise ValueError("stored subject probe does not contain the original passage")
        response_line, response = responses[key]
        if response["model"] != MODEL or not isinstance(response["response"], str):
            raise ValueError(f"baseline is not a {SPEC.key} text response")
        condition = recipe.condition
        rows.append(
            {
                "scenario_uid": f"{dataset}/{schema}_spot/{key[0]}/spot_error",
                "scenario_id": key[0],
                "task": "spot_error",
                "schema": schema,
                "lineage_id": f"existing-spot/lineage/{lineage_hash(recipe)}",
                "condition": condition,
                "condition_group": (
                    "actor_role" if condition in ("wrong_action", "role_swap") else condition
                ),
                "prompt": probe["probe_prompt"],
                "passage": item["prompt"],
                "gold": probe["gold"],
                "prompt_sha256": hashlib.sha256(probe["probe_prompt"].encode()).hexdigest(),
                "recipe": item["gen"]["recipe"],
                "provenance": {
                    "dataset": dataset,
                    "run": str(run),
                    "original_probe_line": probes[key][0],
                    "historical_response_line": response_line,
                    "historical_response_sha256": digest(response),
                    "historical_response_path": str(paths["responses"]),
                    "historical_score_line": scores[key][0],
                    "historical_score_path": str(paths["scores"]),
                    "historical_response_fields": sorted(response),
                    "subject_prompt_source": "probes.jsonl (the capture harness input)",
                    "item_task_prompt_differs": task["probe_prompt"] != probe["probe_prompt"],
                    "primary_control_reusable": False,
                    "control_limitation": "stored text lacks emitted token IDs, stop reasons, and "
                    "the resolved per-response runtime/decoding contract",
                    "prompt_and_gold_unchanged": True,
                },
            }
        )
    return rows, {str(path): file_hash(path) for path in paths.values()}


def freeze_jsonl(path: Path, rows: list[dict]) -> None:
    data = "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows).encode()
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"refusing changed existing-task cohort: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".jsonl.tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def prepare(data: Path = ROOT, out: Path = OUT) -> dict:
    all_rows, pilot_rows, sources = [], [], {}
    for dataset, expected in SOURCES:
        for schema in SCHEMAS:
            rows, hashes = join_run(data / dataset / f"{schema}_spot", dataset, schema, expected)
            sources.update(hashes)
            all_rows.extend(rows)
            if dataset == "spot_36":
                pilot_rows.extend(rows[:6])
    if len(all_rows) != 216 or len(pilot_rows) != 12:
        raise ValueError("existing corpus or historical pilot membership changed")
    if len({row["scenario_uid"] for row in all_rows}) != len(all_rows):
        raise ValueError("dataset-qualified IDs are not unique")
    prompt_groups: dict[str, list[dict]] = {}
    for row in all_rows:
        prompt_groups.setdefault(row["prompt_sha256"], []).append(row)
    if any(len({digest(row["gold"]) for row in group}) != 1 for group in prompt_groups.values()):
        raise ValueError("identical existing prompts have conflicting gold")
    files = {"existing_spot_216.jsonl": all_rows, "legacy_pilot12.jsonl": pilot_rows}
    for name, rows in files.items():
        freeze_jsonl(out / name, rows)
    manifest = {
        "status": "frozen",
        "protocol": "existing-tasks-paired-comparison-v1",
        "authorization": "2026-09-08 user: reuse already-generated tasks and baseline responses",
        "sources": sources,
        "files": {
            name: {"rows": len(rows), "sha256": file_hash(out / name)}
            for name, rows in files.items()
        },
        "n_unique_prompts": len(prompt_groups),
        "n_unique_lineages": len({row["lineage_id"] for row in all_rows}),
        "condition_counts": dict(Counter(row["condition"] for row in all_rows)),
        "historical_pilot_selection": "first six spot_36 probes per schema in stored probe order",
        "historical_baseline_coverage": 216,
        "new_stories": 0,
        "new_llm_calls": 0,
        "gold_relabeling": False,
        "comparison": "same existing prompts across alpha-zero and injection arms; historical "
        "text/scores retained as references, not silently used as matched controls",
        "interpretation": "exploratory reused-instance evidence, not untouched held-out evidence; "
        "keep recipe lineages together in later splits and paired inference",
    }
    bind_manifest(out / "manifest.json", manifest)
    return manifest


def verify_cohort(path: Path) -> list[dict]:
    manifest = json.loads((path.parent / "manifest.json").read_text())
    if (
        manifest["status"] != "frozen"
        or manifest["protocol"] != "existing-tasks-paired-comparison-v1"
    ):
        raise ValueError("unrecognized existing-task manifest")
    for source, expected in manifest["sources"].items():
        if file_hash(Path(source)) != expected:
            raise ValueError(f"historical input changed: {source}")
    expected = manifest["files"][path.name]
    rows = read_jsonl(path)
    if file_hash(path) != expected["sha256"] or len(rows) != expected["rows"]:
        raise ValueError("frozen existing-task cohort changed")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    manifest = prepare(args.data, args.out)
    for name in manifest["files"]:
        verify_cohort(args.out / name)
    print(json.dumps({key: value for key, value in manifest.items() if key != "sources"}, indent=2))


if __name__ == "__main__":
    main()
