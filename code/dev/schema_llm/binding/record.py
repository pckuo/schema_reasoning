"""Assemble and write binding-experiment records.

Logs everything the spec's section 6 asks for, explicitly rather than derivably: B / R / D / E and
the full role-count vector; the omitted role and its tier; merged roles and each person's role
count; distractor flavor and per-distractor licence; per queried binding its role, multiplicity
and tier; family / parent / edit lineage; word, char and scene counts; and both verdicts.

``pipeline/record.py`` logs none of the first six -- B and R are recoverable from ``gen.recipe``,
but tier, multiplicity, merges, flavor and lineage are not recoverable at all.
"""

from __future__ import annotations

import json
from pathlib import Path

from .audit import AuditResult
from .cast import BindingSpec
from .config import ROLE_ALIASES, tier
from .levels import assign_levels
from .verify import VerifyResult


def build_record(
    spec: BindingSpec,
    passage: str,
    labels: dict,
    vr: VerifyResult,
    ar: AuditResult | None,
    task: dict,
    gen_meta: dict,
) -> dict:
    counts = spec.role_counts
    return {
        # `scenario_id` is the join key the whole repo uses (core/schemas.py REQUIRED["item"];
        # subjects.collect / capture / judge and the exporter all key on it). `item_id` is kept as
        # the readable alias because every binding-side module refers to it by that name.
        "scenario_id": spec.item_id,
        "item_id": spec.item_id,
        "schema": spec.schema,
        "experiment": spec.experiment,
        "condition": spec.condition,
        # ---- lineage (spec section 5, rules 1-2)
        "family_id": spec.family_id,
        "parent_id": spec.parent_id,
        "edit_op": spec.edit_op,
        "seed": spec.seed,
        # ---- core quantities (spec section 6)
        "B": spec.B,
        "R": spec.R,
        "D": spec.D,
        "E": spec.E,
        "vector": list(spec.vector),
        "role_counts": {ROLE_ALIASES[spec.schema][rid]: n for rid, n in counts.items()},
        "role_counts_by_id": counts,
        "omitted_roles": spec.omitted_roles,
        "omitted_role_tiers": {rid: tier(spec.schema, rid) for rid in spec.omitted_roles},
        "max_role_multiplicity": spec.max_role_multiplicity,
        "merged_roles": [
            {"pid": p.pid, "name": p.name, "role_ids": list(p.roles), "n_roles": p.n_roles}
            for p in spec.merged_people
        ],
        "n_multiplexed_people": len(spec.merged_people),
        "distractor_flavor": spec.controls.get("distractor_flavor"),
        "distractors": [
            {
                "pid": p.pid,
                "name": p.name,
                "key": p.distractor_key,
                "flavor": p.distractor_flavor,
                "licence": p.licence,
                "scene": p.scene,
            }
            for p in spec.cast
            if p.is_distractor
        ],
        "bindings": spec.bindings(),
        "controls": spec.controls,
        # ---- text + spans
        "prompt": passage,
        "scene_path": list(spec.scene_path),
        "n_scenes": len(spec.scene_path),
        "n_words": vr.n_words,
        "n_chars": vr.n_chars,
        "role_map": vr.role_map,
        "referent_spans": vr.referent_spans,
        "scene_spans": vr.scene_spans,
        # ---- verdicts
        "verify_ok": vr.ok,
        "verify_reasons": vr.reasons,
        "stray_names": vr.stray_names,
        "audit_ok": ar.ok if ar else None,
        "audit_reasons": ar.reasons if ar else [],
        "audit_parts": ar.parts if ar else {},
        "audit_people": ar.people if ar else [],
        "cast": [p.__dict__ for p in spec.cast],
        "labels": labels,
        "tasks": [task],
        "gen": gen_meta,
    }


def write_run(records: list[dict], raws: dict[str, str], out_dir: Path) -> None:
    """Write ``items.jsonl`` + ``probes.jsonl`` + ``raw/``, mirroring the layout
    ``subjects.collect`` / ``subjects.judge`` already consume."""
    out_dir = Path(out_dir)
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)
    # `detail_level`/`distractor_level` are corpus-level derivations (the detail tertile needs every
    # record's length), so they are assigned here, where the full set is in hand, rather than per
    # record at build time.
    assign_levels(records)
    with (out_dir / "items.jsonl").open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    for item_id, raw in raws.items():
        (out_dir / "raw" / f"{item_id}.txt").write_text(raw, encoding="utf-8")
    with (out_dir / "probes.jsonl").open("w", encoding="utf-8") as fh:
        for record in records:
            for task in record.get("tasks", []):
                fh.write(
                    json.dumps(
                        {
                            # `scenario_id` is the join key every downstream consumer uses
                            # (core/schemas.py REQUIRED["probe"]), so it is emitted under that name.
                            "scenario_id": record["item_id"],
                            "task": task["name"],
                            "answer_format": task.get("answer_format"),
                            "probe_prompt": task["probe_prompt"],
                            "gold": task.get("gold"),
                            # self-describing: condition, lineage and B/R/D/E travel with the
                            # probe, so a scoring pass never has to join back to items.jsonl
                            "meta": task.get("meta", {}),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


def family_length_report(records: list[dict]) -> list[dict]:
    """Per family: the word-count spread across siblings.

    Spec section 5 rule 3 wants length matched within a family and verified. Adding people
    lengthens text, so an unchecked D-ladder would confound headcount with passage length.
    """
    by_family: dict[str, list[dict]] = {}
    for record in records:
        by_family.setdefault(record["family_id"], []).append(record)
    out: list[dict] = []
    for family_id, rows in sorted(by_family.items()):
        words = [r["n_words"] for r in rows]
        lo, hi = min(words), max(words)
        out.append(
            {
                "family_id": family_id,
                "n_items": len(rows),
                "min_words": lo,
                "max_words": hi,
                "spread_ratio": round(hi / lo, 3) if lo else None,
            }
        )
    return out
