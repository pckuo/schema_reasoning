"""Assemble and write the per-scenario data record (A7)."""

from __future__ import annotations

import json
from pathlib import Path

from .sampler import Recipe
from .verifier import VerifyResult


def build_record(
    recipe: Recipe,
    prompt: str,
    labels: dict,
    vr: VerifyResult,
    gen_meta: dict,
    tasks: list[dict] | None = None,
) -> dict:
    c = recipe.controls
    return {
        "scenario_id": recipe.scenario_id,
        "schema": recipe.schema,
        "prop_naming": c["prop_naming"],
        "prop_naming_map": c.get("prop_naming_map"),
        "role_naming": c.get("role_naming", "latent"),
        "condition": recipe.condition,
        "violation": recipe.violation,
        "temporal": recipe.temporal,
        "scene_path": recipe.scene_path,
        "detail_level": c["detail_level"],
        "len_band": c["len_band"],
        "distractor_level": c["distractor_level"],
        "distractors": c.get("distractors", []),
        "prompt": prompt,
        "role_map": vr.role_map,
        "referent_spans": vr.referent_spans,
        "scene_spans": vr.scene_spans,
        "forbidden_ok": not any(r.startswith("forbidden") for r in vr.reasons),
        "verify_ok": vr.ok,
        "verify_reasons": vr.reasons,
        "tasks": tasks or [],
        "gen": gen_meta,
    }


def write_items(records: list[dict], raws: dict[str, str], out_dir: Path) -> None:
    out_dir = Path(out_dir)
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)
    with (out_dir / "items.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    for scenario_id, raw in raws.items():
        (out_dir / "raw" / f"{scenario_id}.txt").write_text(raw, encoding="utf-8")
    _write_probes(records, out_dir)


def _write_probes(records: list[dict], out_dir: Path) -> None:
    """Emit one copy-pasteable probe prompt (stimulus + question) per (item, task): a plain
    .txt under probes/ (open, select-all, copy) and a row in probes.jsonl (with the gold)."""
    (out_dir / "probes").mkdir(parents=True, exist_ok=True)
    with (out_dir / "probes.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            for task in record.get("tasks", []):
                probe = task.get("probe_prompt")
                if not probe:
                    continue
                sid, name = record["scenario_id"], task["name"]
                (out_dir / "probes" / f"{sid}__{name}.txt").write_text(probe, encoding="utf-8")
                row = {
                    "scenario_id": sid,
                    "task": name,
                    "answer_format": task.get("answer_format"),
                    "probe_prompt": probe,
                    "gold": task.get("gold"),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
