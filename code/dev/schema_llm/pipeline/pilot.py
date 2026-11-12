"""Micro-pilot driver: sample -> generate -> verify -> record, N times (A4 Phase 4).

Examples
--------
    python -m dev.schema_llm.pipeline.pilot --schema restaurant --n 10 --backend template
    python -m dev.schema_llm.pipeline.pilot --schema restaurant --n 10 --backend copilot
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from ..core.llm import extract_json
from ..core.spec import load_schema
from .generator import (
    DEFAULT_GENERATOR_MODELS,
    Backend,
    CopilotCLIBackend,
    TemplateBackend,
    build_batch_prompt,
    build_generator_prompt,
    build_paraphrase_prompt,
    normalize_item,
    parse_batch,
    verify_violation_present,
)
from .record import build_record, write_items
from .sampler import (
    CONTENT_VIOLATIONS,
    DETAIL_LEVELS,
    DISTRACTOR_LEVELS,
    PROP_NAMING,
    ROLE_NAMING,
    VIOLATION_TYPES,
    sample_recipe,
)
from .tasks import build_tasks
from .verifier import VerifyResult, verify

REPO_ROOT = Path(__file__).resolve().parents[4]


def _run(backend: Backend, schema, recipe, gen_prompt: str) -> tuple[str, dict, VerifyResult, str]:
    raw = backend.generate(gen_prompt, recipe)
    try:
        data = extract_json(raw)
    except ValueError as exc:
        return "", {}, VerifyResult(ok=False, reasons=[f"parse: {exc}"]), raw
    prompt, labels = normalize_item(data.get("prompt", ""), data.get("labels", {}))
    return prompt, labels, verify(prompt, labels, schema, recipe), raw


def _paraphrase_item(
    backend: Backend, schema, recipe, base_prompt: str, tries: int
) -> tuple[str, dict, VerifyResult, str]:
    gen_prompt = build_paraphrase_prompt(schema, recipe, base_prompt)
    prompt, labels, raw = "", {}, ""
    vr = VerifyResult(ok=False, reasons=["not attempted"])
    for _ in range(tries):
        prompt, labels, vr, raw = _run(backend, schema, recipe, gen_prompt)
        if vr.ok:
            break
    return prompt, labels, vr, raw


def _anomaly_survived(schema, recipe, prompt: str, call: Callable[..., str] | None) -> bool:
    """Whether it is safe to adopt a paraphrase (B5). Canonical items and STRUCTURAL violations
    (wrong_order, missing_scene) are already guaranteed by the deterministic verifier (scene
    set/order), so no LLM check is needed; CONTENT violations (wrong_action, role_swap,
    prop_mismatch) are invisible to that verifier, so a strong LLM confirms the paraphrase kept
    the anomaly -- one that smoothed it away is rejected, keeping the violation label true."""
    v = recipe.violation
    if not v or v["type"] not in CONTENT_VIOLATIONS:
        return True
    if call is None:
        return verify_violation_present(schema, recipe, prompt)
    return verify_violation_present(schema, recipe, prompt, call=call)


def _run_batches(
    backend: Backend, schema, recipes: list, batch_size: int, raws: dict[str, str],
    batch_models: dict[str, str | None],
) -> dict[str, tuple[str, dict, VerifyResult]]:
    """Option A: generate items in chunks of one LLM call each so the model self-diversifies
    in-context. Returns scenario_id -> (prompt, labels, verify) for whatever parsed, and fills
    ``batch_models`` with the model that produced each item's chunk."""
    results: dict[str, tuple[str, dict, VerifyResult]] = {}
    for start in range(0, len(recipes), batch_size):
        chunk = recipes[start : start + batch_size]
        raw = backend.generate(build_batch_prompt(schema, chunk), chunk[0])
        used = getattr(backend, "last_model", None)
        raws[f"batch_{start // batch_size:02d}"] = raw
        try:
            parsed = parse_batch(raw, len(chunk))
        except ValueError:
            parsed = [("", {})] * len(chunk)
        for recipe, (prompt, labels) in zip(chunk, parsed, strict=True):
            if prompt:
                prompt, labels = normalize_item(prompt, labels)
                vr = verify(prompt, labels, schema, recipe)
            else:
                vr = VerifyResult(ok=False, reasons=["batch: no output"])
            results[recipe.scenario_id] = (prompt, labels, vr)
            batch_models[recipe.scenario_id] = used
    return results


def _resolve_item(
    backend, schema, i, base_seed, recipe0, batch_res, max_retries, raws,
    overrides=None, condition="canonical", batch_model=None, temporal_form=None,
) -> tuple:
    """Keep a verified batch result; otherwise regenerate this item on its own with fresh
    seeds (also the only path for the template backend). Any pinned ``overrides`` are kept on
    every retry so a controlled item never drifts back to random controls; likewise the
    ``condition`` (canonical or a violation type) is preserved across retries. The last tuple
    element is the model that produced the kept item (``batch_model`` on a batch hit, else the
    backend's last sampled model)."""
    sid = recipe0.scenario_id
    if batch_res is not None and batch_res[2].ok:
        prompt, labels, vr = batch_res
        return recipe0, prompt, labels, vr, 1, "batch", batch_model
    recipe, prompt, labels, vr = recipe0, "", {}, VerifyResult(ok=False, reasons=["none"])
    attempt = 0
    for attempt in range(max_retries):
        seed = base_seed * 1000 + i * 10 + attempt
        recipe = sample_recipe(schema, seed, sid, overrides=overrides, condition=condition,
                               temporal_form=temporal_form)
        gen_prompt = build_generator_prompt(schema, recipe)
        prompt, labels, vr, raw = _run(backend, schema, recipe, gen_prompt)
        raws[sid] = raw
        if vr.ok:
            break
    source = "single" if batch_res is None else "batch->single"
    return recipe, prompt, labels, vr, attempt + 1, source, getattr(backend, "last_model", None)


def run_pilot(
    schema_name: str,
    n: int,
    base_seed: int,
    backend_kind: str,
    model: str | None,
    out_dir: Path,
    enabled_tasks: list[str] | None = None,
    paraphrase: bool = False,
    max_retries: int = 3,
    batch_size: int = 5,
    control_overrides: list[dict] | None = None,
    cut_frac: float = 0.5,
    conditions: list[str] | None = None,
    temporal_forms: list[str] | None = None,
    models: list[str] | None = None,
    anomaly_call: Callable[..., str] | None = None,
) -> dict:
    schema = load_schema(schema_name)
    if control_overrides is not None:
        n = len(control_overrides)
    if conditions is not None:
        n = len(conditions)
    if temporal_forms is not None:
        n = len(temporal_forms)
    is_batch = backend_kind == "copilot"
    # Copilot generator: pin ``model`` if given, else sample per call from ``models`` (default
    # pool) so stimuli aren't locked to one model's house style; unusable picks fall back to auto.
    pool = None if model else (models or list(DEFAULT_GENERATOR_MODELS))
    backend: Backend = (
        TemplateBackend(schema)
        if backend_kind == "template"
        else CopilotCLIBackend(model=model, models=pool, seed=base_seed)
    )
    # Paraphrase only earns its keep on the deterministic template base: an LLM base
    # paraphrasing its own already-natural prose is a no-op (A). Skip it with a warning.
    para_backend: Backend | None = None
    if paraphrase and backend_kind == "template":
        para_backend = CopilotCLIBackend(model=model, models=pool, seed=base_seed + 1)
    elif paraphrase:
        print("[warn] --paraphrase skipped: it only adds value on the deterministic template "
              "base; an LLM base paraphrasing its own prose is a no-op.")

    recipes = [
        sample_recipe(
            schema, base_seed * 1000 + i * 10, f"{schema_name}_{i:04d}",
            overrides=control_overrides[i] if control_overrides else None,
            condition=conditions[i] if conditions else "canonical",
            temporal_form=temporal_forms[i] if temporal_forms else None,
        )
        for i in range(n)
    ]
    raws: dict[str, str] = {}
    batch_models: dict[str, str | None] = {}
    # Option A: one LLM call per chunk; the model sees siblings and diversifies in-context.
    batch_results = (
        _run_batches(backend, schema, recipes, batch_size, raws, batch_models) if is_batch else {}
    )

    records: list[dict] = []
    n_ok = 0
    for i, recipe0 in enumerate(recipes):
        recipe, prompt, labels, vr, attempts, source, model_used = _resolve_item(
            backend, schema, i, base_seed, recipe0,
            batch_results.get(recipe0.scenario_id), max_retries, raws,
            overrides=control_overrides[i] if control_overrides else None,
            condition=conditions[i] if conditions else "canonical",
            batch_model=batch_models.get(recipe0.scenario_id),
            temporal_form=temporal_forms[i] if temporal_forms else None,
        )
        gen_meta = {
            "recipe": recipe.to_dict(),
            "seed": recipe.seed,
            "backend": backend_kind,
            "model": model_used,
            "attempts": attempts,
            "source": source,
            "paraphrased": False,
        }
        if para_backend is not None and vr.ok:
            p_prompt, p_labels, p_vr, p_raw = _paraphrase_item(
                para_backend, schema, recipe, prompt, max_retries
            )
            raws[f"{recipe.scenario_id}__para"] = p_raw
            gen_meta["base_prompt"] = prompt
            if p_vr.ok and _anomaly_survived(schema, recipe, p_prompt, anomaly_call):
                prompt, labels, vr = p_prompt, p_labels, p_vr
                gen_meta["paraphrased"] = True
                gen_meta["paraphrase_model"] = getattr(para_backend, "last_model", None)
            elif p_vr.ok:
                # paraphrase verified but the injected content anomaly was smoothed away; keep the
                # deterministic template base so the violation label stays true (B5).
                gen_meta["paraphrase_dropped_anomaly"] = True
            else:
                gen_meta["paraphrase_failed"] = "; ".join(p_vr.reasons)

        n_ok += int(vr.ok)
        records.append(
            build_record(
                recipe, prompt, labels, vr, gen_meta,
                build_tasks(schema, recipe, vr, prompt, enabled_tasks, cut_frac),
            )
        )
        para = " para" if gen_meta["paraphrased"] else ""
        tag = "" if source == "single" else f" {source}"
        mtag = f" [{model_used}]" if model_used else ""
        cond = "" if recipe.condition == "canonical" else f"/{recipe.condition}"
        flag = "ok " if vr.ok else "FAIL"
        print(f"[{flag}] {recipe.scenario_id} ({recipe.controls['prop_naming']}/"
              f"{recipe.controls['detail_level']}/{recipe.controls['len_band']}{cond}{para}{tag})"
              f"{mtag}{'' if vr.ok else ' -- ' + '; '.join(vr.reasons)}")

    write_items(records, raws, out_dir)
    used = Counter(r["gen"].get("model") for r in records if r["gen"].get("model"))
    if used:
        print("models sampled: " + ", ".join(f"{m}:{c}" for m, c in sorted(used.items())))
    print(f"\n{n_ok}/{n} passed verification -> {out_dir}")
    return {"n_ok": n_ok, "n": n, "out_dir": str(out_dir)}


def _build_control_overrides(
    grid: bool, prop_naming: str | None, n: int, reps: int = 1, role_naming: str | None = None
) -> list[dict] | None:
    """Turn the --grid / --prop-naming / --role-naming flags into a per-item control-override
    list. --grid yields the full detail_level x distractor_level cross (9 items, one per cell),
    repeated ``reps`` times (9*reps items = reps distinct examples per cell, each with its own
    seed); --prop-naming / --role-naming pin those knobs. Returns None when nothing is pinned
    (fully sampled controls)."""
    base: dict = {}
    if prop_naming:
        base["prop_naming"] = prop_naming
    if role_naming:
        base["role_naming"] = role_naming
    if grid:
        cells = [
            {**base, "detail_level": d, "distractor_level": x}
            for d in DETAIL_LEVELS
            for x in DISTRACTOR_LEVELS
        ]
        return [dict(c) for _ in range(reps) for c in cells]
    if base:
        return [dict(base) for _ in range(n)]
    return None


def _build_conditions(n: int) -> list[str]:
    """Balanced canonical + one-of-each-violation cycle across n items (plan A5), so spot_error
    sees a mix of clean (base-rate) and violation trials. Use --n as a multiple of 6 for an even
    split (6 = canonical + the 5 violation types)."""
    cycle = ["canonical", *VIOLATION_TYPES]
    return [cycle[i % len(cycle)] for i in range(n)]


def main() -> None:
    p = argparse.ArgumentParser(description="schema_llm micro-pilot generator")
    p.add_argument("--schema", default="restaurant")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backend", choices=["template", "copilot"], default="template")
    p.add_argument("--model", default=None,
                   help="pin a single copilot model (overrides --models sampling)")
    p.add_argument(
        "--models", default=None,
        help="comma-separated copilot model pool sampled per call when --model is unset "
             "(default: " + ",".join(DEFAULT_GENERATOR_MODELS) + "); unusable picks fall back "
             "to auto",
    )
    p.add_argument(
        "--tasks", default="identify,predict_next_event,predict_correct_action",
        help="comma-separated probe tasks to attach "
             "(identify,predict_next_event,predict_correct_action)",
    )
    p.add_argument(
        "--cut-frac", type=float, default=0.5,
        help="predict_next_event/predict_correct_action truncate the passage after at least this "
             "fraction of the scenes (0-1); the exact interior cut is then sampled (default 0.5)",
    )
    p.add_argument(
        "--paraphrase", action="store_true",
        help="LLM-naturalize each verified item; template base only (needs the Copilot CLI)",
    )
    p.add_argument(
        "--batch-size", type=int, default=5,
        help="items per LLM call on the copilot batch route (Option A)",
    )
    p.add_argument(
        "--prop-naming", choices=PROP_NAMING, default=None,
        help="pin prop_naming for every item (default: sampled per item)",
    )
    p.add_argument(
        "--role-naming", choices=ROLE_NAMING, default=None,
        help="pin role_naming: 'named' = positive control (canonical role nouns allowed in the "
             "passage); default latent (roles conveyed by action only)",
    )
    p.add_argument(
        "--grid", action="store_true",
        help="scan the full detail_level x distractor_level grid (9 items); len_band and "
             "referents stay randomized. Sets --n to 9.",
    )
    p.add_argument(
        "--grid-reps", type=int, default=1,
        help="repeat the --grid cross this many times (9*reps items = reps distinct examples "
             "per detail x distractor cell, each seeded differently)",
    )
    p.add_argument(
        "--violations", action="store_true",
        help="generate a balanced canonical + 5-violation-type mix (plan A5) and attach the "
             "spot_error task. Forces the template backend (it injects the anomaly), then "
             "paraphrases each item into natural prose and LLM-checks that content anomalies "
             "survived (B5) -- so this needs the Copilot CLI.",
    )
    p.add_argument("--out", default=None, help="output dir (default results/schema_llm/<UTC>)")
    args = p.parse_args()

    if args.out:
        out_dir = Path(args.out)
    else:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        out_dir = REPO_ROOT / "results" / "schema_llm" / stamp
    enabled_tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else None
    control_overrides = _build_control_overrides(
        args.grid, args.prop_naming, args.n, args.grid_reps, args.role_naming
    )
    conditions = None
    backend = args.backend
    if args.violations:
        if control_overrides is not None:
            print("[warn] --violations ignores --grid/--prop-naming (controls stay randomized).")
            control_overrides = None
        conditions = _build_conditions(args.n)
        n_cond = 1 + len(VIOLATION_TYPES)
        if args.n % n_cond:
            print(f"[warn] --violations n={args.n} is not a multiple of {n_cond}; the {n_cond} "
                  f"conditions (canonical + {len(VIOLATION_TYPES)} violation types) will be "
                  "unbalanced (some undersampled). Use a multiple of 6 for an even split.")
        if "spot_error" not in enabled_tasks:
            enabled_tasks.append("spot_error")
        if backend != "template":
            print("[warn] --violations needs the template backend (the LLM batch path ignores "
                  "the injected violation); forcing --backend template.")
            backend = "template"
    # --violations naturalizes the stiff template base via a copilot paraphrase (B5), so the
    # spot_error prose is comparable to the copilot-generated identify/predict items.
    paraphrase = args.paraphrase or args.violations
    run_pilot(
        args.schema, args.n, args.seed, backend, args.model, out_dir,
        enabled_tasks, paraphrase, batch_size=args.batch_size,
        control_overrides=control_overrides, cut_frac=args.cut_frac, conditions=conditions,
        models=models,
    )


if __name__ == "__main__":
    main()
