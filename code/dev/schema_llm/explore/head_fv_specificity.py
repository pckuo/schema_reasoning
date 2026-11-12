"""Exact-ID runtime and gate-A calibration for the Phi-mini head-FV experiment.

The generation hook deliberately skips prefill: the first affected prediction is
the second generated token. No reasoning or measurement-window gate is applied.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import importlib.metadata
import json
import math
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from ..core.model import LoadedModel, chat_template_of, load_model, set_seed
from ..primitives.funcvec_heads import build_head_fv, oproj_slices
from ..rlvr.environment.parsing import parse_response
from ..rlvr.environment.verifier import verify_spot_error
from .head_fv_models import active as active_model
from .head_fv_phi_cache import cache_policy, stable_phi_cache
from .primitive_transplant.analyze import _repeat_frac

# Subject-specific constants live in head_fv_models; these names are kept so every existing
# call site and saved contract is unchanged. Default is Phi, so behaviour is identical.
SPEC = active_model()
MODEL = SPEC.model
REVISION = SPEC.revision
INSTRUCT_TEMPLATE = SPEC.chat_template
EOS_IDS = SPEC.eos_ids
SCHEDULE = "skip-prefill-last-position-all-continuation-v1"
STRUCTURAL = SPEC.structural


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def bind_manifest(path: Path, value: dict) -> None:
    # Compare the persisted JSON form: integer keys and tuples change on a round trip.
    value = json.loads(json.dumps(value, allow_nan=False))
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"refusing changed run contract at {path}")
    else:
        write_json(path, value)


def subject_chat_template():
    """The template to load the subject with, or None to use its own.

    Phi-4-mini-reasoning ships no usable chat template, so the pilot borrowed the instruct one.
    Qwen3-4B has a native template whose default already renders thinking mode, matching the
    `--think on` used for the historical captures, so borrowing anything would be wrong.
    """
    return chat_template_of(SPEC.chat_template) if SPEC.chat_template else None


@contextlib.contextmanager
def subject_cache(lm):
    """Apply the Phi3 cache repair only where it is licensed, otherwise use the stock path.

    The repair exists solely because Phi-4-mini-reasoning's short and long LongRoPE factors are
    identical, which makes the preparer's reset at 4,096 cached positions redundant. Qwen3-4B is
    natively 40,960 with no such reset, so there is nothing to bypass and forcing the guard would
    abort. This does NOT mean Qwen needs no long-generation check: Phi's defect only showed past
    4k and every short smoke passed it.
    """
    if SPEC.supports_phi3_cache_fix:
        with stable_phi_cache(lm):
            yield
    else:
        yield


def subject_cache_policy(lm) -> dict:
    if SPEC.supports_phi3_cache_fix:
        return cache_policy(lm)
    return {
        "kind": "stock-native-cache",
        "subject": SPEC.key,
        "reason": "no dual rope factors; the Phi3 preparer reset does not apply",
    }


def certify_geometry(lm: LoadedModel) -> dict:
    cfg = lm.model.config
    actual = (
        cfg.num_hidden_layers,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.hidden_size,
        getattr(cfg, "head_dim", None) or 128,
    )
    blocks, q_heads, kv_heads, hidden, head_dim = SPEC.geometry
    if actual != SPEC.geometry:
        raise ValueError(f"unexpected {SPEC.key} geometry: {actual}")
    if lm.name != MODEL or lm.revision != REVISION:
        raise ValueError(f"runtime is not bound to the pinned {SPEC.key} checkpoint")
    # o_proj maps the concatenated heads back to the residual, so it is (hidden, heads*head_dim).
    # That is square for Phi only by coincidence -- 24*128 == 3072 -- and is 2560x4096 for Qwen.
    # Asserting squareness would reject every model whose head budget differs from its width.
    expected = (hidden, q_heads * head_dim)
    for layer in range(blocks):
        projection = lm.model.get_submodule(f"{lm.layer_prefix}.{layer}.self_attn.o_proj")
        if tuple(projection.weight.shape) != expected or projection.bias is not None:
            raise ValueError(
                f"invalid output projection at layer {layer}: "
                f"{tuple(projection.weight.shape)} != {expected}"
            )
    return {
        "layers": blocks,
        "query_heads": q_heads,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "hidden_size": hidden,
        "projection_bias": False,
    }


def projected_fv(
    lm: LoadedModel, means: np.ndarray, heads: list[tuple[int, int]], *, expected_count: int = 35
) -> tuple[np.ndarray, dict]:
    cfg = lm.model.config
    n_heads = cfg.num_attention_heads
    width = getattr(cfg, "head_dim", None) or cfg.hidden_size // n_heads
    if means.shape != (lm.n_layers, n_heads, width) or not np.isfinite(means).all():
        raise ValueError("head means have invalid shape or values")
    if len(heads) != expected_count or len(set(heads)) != len(heads):
        raise ValueError("wrong number of selected heads or duplicate head")
    if any(not (0 <= layer < lm.n_layers and 0 <= head < n_heads) for layer, head in heads):
        raise ValueError("selected head lies outside the model")
    scored = [(layer, head, 0.0) for layer, head in heads]
    slices = oproj_slices(lm, scored, width)
    if any(key.startswith("bias_") for key in slices):
        raise ValueError("this protocol requires bias-free projections")
    result, used = build_head_fv(
        means[None], list(range(lm.n_layers)), slices, scored, width, overall=None
    )
    if used != heads:
        raise ValueError("construction silently omitted a selected head")
    oracle = np.zeros(cfg.hidden_size, dtype=np.float32)
    for layer in sorted({layer for layer, _ in heads}):
        slots = torch.zeros(n_heads * width, dtype=torch.float32)
        for source_layer, head in heads:
            if source_layer == layer:
                slots[head * width : (head + 1) * width] = torch.from_numpy(
                    means[layer, head].astype(np.float32)
                )
        weight = (
            lm.model.get_submodule(f"{lm.layer_prefix}.{layer}.self_attn.o_proj")
            .weight.detach()
            .float()
            .cpu()
        )
        oracle += torch.nn.functional.linear(slots, weight).numpy()
    np.testing.assert_allclose(result[0], oracle, rtol=1e-4, atol=1e-5)
    if not np.isfinite(result).all() or np.linalg.norm(result[0]) == 0:
        raise ValueError("FV is nonfinite or zero")
    return result[0], {
        "heads": heads,
        "convention": "raw-uncentered-sum",
        "norm": float(np.linalg.norm(result[0])),
        "projection_max_abs_error": float(np.max(np.abs(result[0] - oracle))),
    }


@dataclass
class DecodeState:
    calls: int = 0
    edited_calls: int = 0
    active: torch.Tensor | None = None


@contextmanager
def continuous_injection(
    lm: LoadedModel, layer: int, vector: Any, alpha: float, eos_ids: tuple[int, ...] = EOS_IDS
):
    """Track actual model calls, not sequence length, and suppress finished batch rows."""
    if not 0 <= layer < lm.n_layers or not math.isfinite(alpha):
        raise ValueError("invalid layer or multiplier")
    block = lm.model.get_submodule(f"{lm.layer_prefix}.{layer}")
    parameter = next(lm.model.parameters())
    addition = None
    if alpha != 0:
        vector = torch.as_tensor(vector, dtype=parameter.dtype, device=parameter.device)
        if vector.shape != (lm.hidden_size,) or not torch.isfinite(vector).all():
            raise ValueError("invalid residual-width FV")
        addition = alpha * vector
    state = DecodeState()

    def before(_module, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        if ids is None or ids.ndim != 2:
            raise ValueError("generation schedule requires explicit batched input_ids")
        if state.calls == 0:
            state.active = torch.ones(ids.shape[0], dtype=torch.bool, device=ids.device)
        else:
            if state.active is None or len(state.active) != len(ids):
                raise ValueError("batch membership changed during generation")
            state.active &= ~torch.isin(ids[:, -1], ids.new_tensor(eos_ids))
        state.calls += 1

    def edit(_module, _args, output):
        if state.calls <= 1 or addition is None:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        if state.active is None or hidden.shape[0] != len(state.active):
            raise ValueError("injection batch does not match generation state")
        changed = hidden.clone()
        changed[state.active, -1, :] += addition
        state.edited_calls += 1
        return (changed, *output[1:]) if isinstance(output, tuple) else changed

    pre = lm.model.register_forward_pre_hook(before, with_kwargs=True)
    hook = block.register_forward_hook(edit)
    try:
        yield state
    finally:
        hook.remove()
        pre.remove()


def split_generated(
    sequences: torch.Tensor,
    prompt_mask: torch.Tensor,
    eos_ids: tuple[int, ...],
    max_new_tokens: int,
) -> list[dict]:
    prompt_width = prompt_mask.shape[1]
    if sequences.shape[0] != prompt_mask.shape[0] or sequences.shape[1] < prompt_width:
        raise ValueError("invalid generated batch shape")
    rows = []
    for seq, mask in zip(sequences.cpu(), prompt_mask.cpu(), strict=True):
        if not torch.all((mask == 0) | (mask == 1)) or torch.any(mask[1:] < mask[:-1]):
            raise ValueError("prompt mask must use contiguous left padding")
        prompt = seq[:prompt_width][mask.bool()].tolist()
        generated = seq[prompt_width:].tolist()
        terminal = next((i for i, token in enumerate(generated) if token in eos_ids), None)
        if terminal is not None:
            generated = generated[: terminal + 1]
            reason = "eos"
        elif len(generated) == max_new_tokens:
            reason = "budget"
        else:
            raise ValueError("generation stopped without EOS or its declared budget")
        rows.append(
            {
                "prompt_ids": prompt,
                "generated_ids": generated,
                "full_ids": prompt + generated,
                "n_prompt": len(prompt),
                "stop_reason": reason,
                "terminal_id": generated[-1] if generated else None,
            }
        )
    return rows


def content_mask(tokenizer, ids: list[int]) -> list[bool]:
    mask = [token not in set(tokenizer.all_special_ids) | set(EOS_IDS) for token in ids]
    for marker in STRUCTURAL:
        pattern = tokenizer.encode(marker, add_special_tokens=False)
        if not pattern:
            raise ValueError(f"empty structural tokenization for {marker}")
        for start in range(len(ids) - len(pattern) + 1):
            if ids[start : start + len(pattern)] == pattern:
                mask[start : start + len(pattern)] = [False] * len(pattern)
    return mask


def response_health(text: str, stop_reason: str, gold: dict) -> dict:
    parsed = parse_response(text)
    malformed = text.count("<think>") != text.count("</think>") or text.count("<think>") > 1
    completed = stop_reason == "eos" and parsed.thinking_closed and not malformed
    verdict = verify_spot_error(gold, text, truncated=not completed)
    repetition = _repeat_frac(text)
    return {
        "completed": completed,
        "thinking_closed": parsed.thinking_closed,
        "malformed_reasoning": malformed,
        "valid_final": completed and verdict.format_valid,
        "repetition_4gram": repetition,
        "strong_repetition": repetition > 0.5,
        "strict_rule_success": completed and verdict.diagnostics["strict_correct"],
        "rule_verdict": verdict.as_dict(),
    }


def verifier_gold(scenario: dict) -> dict:
    """Supply verifier context without changing the frozen task's gold labels."""
    schema, gold = scenario["schema"], scenario["gold"]
    if schema not in ("restaurant", "airport"):
        raise ValueError("unsupported schema for spot-error verification")
    if "schema" in gold and gold["schema"] != schema:
        raise ValueError("gold schema conflicts with its scenario")
    return {**gold, "schema": schema}


def generate_exact(
    lm: LoadedModel,
    prompts: list[str],
    *,
    seed: int,
    max_new_tokens: int,
    layer: int = 16,
    vector: Any = None,
    alpha: float = 0,
    diagnostic: bool = False,
    do_sample: bool = True,
) -> tuple[list[dict], dict]:
    if not prompts or max_new_tokens < 1:
        raise ValueError("nonempty prompts and positive generation budget required")
    tok = lm.tokenizer
    rendered = [
        tok.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        for prompt in prompts
    ]
    previous_side, previous_pad = tok.padding_side, tok.pad_token
    try:
        tok.padding_side = "left"
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        enc = tok(rendered, add_special_tokens=False, padding=True, return_tensors="pt")
    finally:
        tok.padding_side, tok.pad_token = previous_side, previous_pad
    cfg = copy.deepcopy(lm.model.generation_config)
    cfg.update(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=0.8,
        top_p=0.95,
        top_k=50,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        eos_token_id=list(EOS_IDS),
        pad_token_id=tok.eos_token_id,
        num_beams=1,
        num_return_sequences=1,
        use_cache=True,
    )
    set_seed(seed)
    started = time.monotonic()
    with (
        torch.inference_mode(),
        subject_cache(lm),
        continuous_injection(lm, layer, vector, alpha) as state,
    ):
        out = lm.model.generate(
            input_ids=enc["input_ids"].to(lm.device),
            attention_mask=enc["attention_mask"].to(lm.device),
            generation_config=cfg,
            return_dict_in_generate=True,
            output_scores=diagnostic,
            output_logits=diagnostic,
        )
    elapsed = time.monotonic() - started
    rows = split_generated(out.sequences, enc["attention_mask"], EOS_IDS, max_new_tokens)
    for row in rows:
        content_ids = row["generated_ids"]
        if row["stop_reason"] == "eos":
            content_ids = content_ids[:-1]
        row["text"] = tok.decode(content_ids, skip_special_tokens=False)
        row["content_mask"] = content_mask(tok, row["generated_ids"])
    info = {
        "generation_config": cfg.to_dict(),
        "seconds": elapsed,
        "seed": seed,
        "batch_size": len(prompts),
        "batch_prompt_mask": enc["attention_mask"].tolist(),
        "rendered_sha256": [digest(text) for text in rendered],
        "forward_calls": state.calls,
        "edited_calls": state.edited_calls,
    }
    if diagnostic:
        for index, row in enumerate(rows):
            generated = row["generated_ids"]
            row["diagnostic_raw_argmax"] = [
                int(logits[index].argmax()) for logits in out.logits[: len(generated)]
            ]
            row["diagnostic_processed_argmax"] = [
                int(scores[index].argmax()) for scores in out.scores[: len(generated)]
            ]
            row["diagnostic_raw_logit_sha256"] = [
                hashlib.sha256(logits[index].float().cpu().numpy().tobytes()).hexdigest()
                for logits in out.logits[: len(generated)]
            ]
    return rows, info


def runtime_contract(lm: LoadedModel) -> dict:
    return {
        "model": lm.name,
        "revision": lm.revision,
        "geometry": certify_geometry(lm),
        "template_sha256": digest(lm.tokenizer.chat_template),
        "schedule": SCHEDULE,
        "cache_policy": subject_cache_policy(lm),
        "eos_ids": list(EOS_IDS),
        "compute_dtype": str(next(lm.model.parameters()).dtype),
        "model_config": lm.model.config.to_dict(),
        "attention_implementation": lm.model.config._attn_implementation,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers")},
        "runtime_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def runtime_certificate(lm: LoadedModel, out_dir: Path) -> dict:
    contract = json.loads(json.dumps(runtime_contract(lm), allow_nan=False))
    prior = out_dir / "runtime_certificate.json"
    if prior.exists() and json.loads(prior.read_text())["contract"] != contract:
        raise ValueError("refusing to overwrite a certificate from another runtime")
    bind_manifest(out_dir / "runtime_manifest.json", contract)
    prompts = ["What is 2 + 3? Answer with the number only."]
    base, baseline = generate_exact(
        lm, prompts, seed=7, max_new_tokens=6, do_sample=False, diagnostic=True
    )
    zero, _ = generate_exact(
        lm,
        prompts,
        seed=7,
        max_new_tokens=6,
        do_sample=False,
        vector=np.ones(SPEC.hidden_size),
        alpha=0,
        diagnostic=True,
    )
    assert base[0]["full_ids"] == zero[0]["full_ids"], "alpha-zero parity failed"
    assert base[0]["diagnostic_raw_logit_sha256"] == zero[0]["diagnostic_raw_logit_sha256"]
    vector = np.random.default_rng(7).normal(size=SPEC.hidden_size).astype(np.float32) * 0.01
    steered, edits = generate_exact(
        lm,
        prompts,
        seed=7,
        max_new_tokens=6,
        do_sample=False,
        vector=vector,
        alpha=1,
        diagnostic=True,
    )
    assert base[0]["generated_ids"][0] == steered[0]["generated_ids"][0]
    assert (
        base[0]["diagnostic_raw_logit_sha256"][0] == (steered[0]["diagnostic_raw_logit_sha256"][0])
    ), "prefill logits were altered"
    assert (
        base[0]["diagnostic_raw_logit_sha256"][1] != (steered[0]["diagnostic_raw_logit_sha256"][1])
    ), "first continuation logits were not affected"
    assert edits["edited_calls"] == edits["forward_calls"] - 1
    assert base[0]["generated_ids"] == base[0]["diagnostic_processed_argmax"]
    ids = torch.tensor([base[0]["full_ids"]], device=lm.device)
    with torch.inference_mode(), subject_cache(lm):
        replay = lm.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    start = base[0]["n_prompt"] - 1
    replay_argmax = replay.logits[0, start:-1].argmax(-1).tolist()
    report = {
        "contract": contract,
        "baseline": base[0],
        "baseline_runtime": baseline,
        "steered": steered[0],
        "steered_runtime": edits,
        "replay_argmax": replay_argmax,
        "replay_generation_argmax_agreement": float(
            np.mean(np.array(replay_argmax) == np.array(base[0]["diagnostic_raw_argmax"]))
        ),
        "alpha_zero_identity": True,
        "prefill_token_identity": True,
        "continuous_schedule": True,
        "scope": "numerical smoke; six-token outputs are not baseline usability trials",
    }
    write_json(out_dir / "runtime_certificate.json", report)
    return report


def read_completed(path: Path, expected: set[tuple[str, int]], protocol_hash: str) -> dict:
    rows = {}
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            key = (row["scenario_uid"], row["repeat"])
            if row["protocol_hash"] != protocol_hash or key not in expected or key in rows:
                raise ValueError("mismatched, duplicate or unexpected calibration record")
            if row["stop_reason"] not in ("eos", "budget"):
                raise ValueError("nonterminal record cannot satisfy resume")
            rows[key] = row
    return rows


def calibrate(
    lm: LoadedModel,
    scenarios_path: Path,
    out_dir: Path,
    *,
    max_new_tokens: int,
    batch_size: int,
    seeds: list[int],
    cohort: Literal["independent24", "existing-pilot12"] = "independent24",
) -> dict:
    if cohort == "existing-pilot12":
        from .head_fv_existing import verify_cohort

        if scenarios_path.name != "legacy_pilot12.jsonl":
            raise ValueError("existing calibration requires the frozen historical pilot subset")
        scenarios = verify_cohort(scenarios_path)
        n_scenarios = 12
        counts = (
            ("canonical", 2),
            ("wrong_order", 1),
            ("wrong_action", 1),
            ("role_swap", 1),
            ("missing_scene", 1),
        )
    elif cohort == "independent24":
        scenarios = [json.loads(line) for line in scenarios_path.read_text().splitlines() if line]
        n_scenarios = 24
        counts = (("canonical", 4), ("wrong_order", 4), ("wrong_action", 2), ("role_swap", 2))
    else:
        raise ValueError(f"unknown calibration cohort: {cohort}")
    if len(scenarios) != n_scenarios or len({r["scenario_uid"] for r in scenarios}) != n_scenarios:
        raise ValueError(f"calibration requires the complete {cohort} cohort")
    quotas = {
        (schema, condition): n for schema in ("restaurant", "airport") for condition, n in counts
    }
    if Counter((row["schema"], row["condition"]) for row in scenarios) != quotas:
        raise ValueError("calibration schema/condition quotas differ from the frozen plan")
    if len({row["lineage_id"] for row in scenarios}) != n_scenarios:
        raise ValueError("calibration must use independent scenario lineages")
    if batch_size < 1 or len(seeds) != 2 or len(set(seeds)) != 2:
        raise ValueError("calibration requires positive batch size and two unique seeds")
    health_golds = {row["scenario_uid"]: verifier_gold(row) for row in scenarios}
    contract = {
        "cohort": cohort,
        "runtime": runtime_contract(lm),
        "scenario_sha256": digest(scenarios),
        "max_new_tokens": max_new_tokens,
        "batch_size": batch_size,
        "seeds": seeds,
        "sampling": {"temperature": 0.8, "top_p": 0.95, "top_k": 50},
        "repetition_rule": "whitespace-case-sensitive-4gram-unique-fraction-v1",
        "verifier_context": "original gold plus row-level schema, without relabeling",
    }
    bind_manifest(out_dir / "calibration_manifest.json", contract)
    protocol_hash = digest(contract)
    summary_path = out_dir / "calibration_summary.json"
    if summary_path.exists():
        saved_summary = json.loads(summary_path.read_text())
        if saved_summary["protocol_hash"] != protocol_hash:
            raise ValueError("saved calibration summary has another protocol")
        for relative, expected_hash in saved_summary["artifacts"].items():
            artifact = out_dir / relative
            if hashlib.sha256(artifact.read_bytes()).hexdigest() != expected_hash:
                raise ValueError(f"saved calibration artifact changed: {relative}")
    expected = {(r["scenario_uid"], repeat) for r in scenarios for repeat in range(2)}
    path = out_dir / "generations.jsonl"
    completed = read_completed(path, expected, protocol_hash)
    for repeat, seed in enumerate(seeds):
        for offset in range(0, len(scenarios), batch_size):
            batch = scenarios[offset : offset + batch_size]
            keys = [(row["scenario_uid"], repeat) for row in batch]
            batch_path = out_dir / "batches" / f"r{repeat}-offset{offset}.json"
            if batch_path.exists():
                committed = json.loads(batch_path.read_text())
                if [(r["scenario_uid"], r["repeat"]) for r in committed] != keys:
                    raise ValueError("saved batch does not match frozen membership")
                for key, row in zip(keys, committed, strict=True):
                    if row["protocol_hash"] != protocol_hash or row["stop_reason"] not in (
                        "eos",
                        "budget",
                    ):
                        raise ValueError("saved batch violates the runtime contract")
                    if key in completed and completed[key] != row:
                        raise ValueError("saved batch conflicts with aggregate generation records")
                    completed[key] = row
            present = [key in completed for key in keys]
            if all(present):
                continue
            if any(present):
                raise ValueError(
                    "partial saved batch; recover atomically, not with a resized batch"
                )
            raw_path = out_dir / "raw_batches" / f"r{repeat}-offset{offset}.json"
            if raw_path.exists():
                raw = json.loads(raw_path.read_text())
                raw_records = raw["records"]
                if (
                    raw["protocol_hash"] != protocol_hash
                    or digest(raw_records) != raw["records_sha256"]
                    or [(r["scenario_uid"], r["repeat"]) for r in raw_records] != keys
                    or any(r["protocol_hash"] != protocol_hash for r in raw_records)
                    or any(r["stop_reason"] not in ("eos", "budget") for r in raw_records)
                ):
                    raise ValueError("saved raw batch violates its identity or content hash")
            else:
                rows, metadata = generate_exact(
                    lm, [r["prompt"] for r in batch],
                    seed=seed + offset, max_new_tokens=max_new_tokens
                )
                raw_records = [
                    {
                        **row, "scenario_uid": key[0], "repeat": repeat,
                        "schema": item["schema"], "condition_group": item["condition_group"],
                        "condition": item["condition"], "protocol_hash": protocol_hash,
                        "runtime": metadata,
                    }
                    for item, row, key in zip(batch, rows, keys, strict=True)
                ]
                write_json(raw_path, {
                    "protocol_hash": protocol_hash, "records": raw_records,
                    "records_sha256": digest(raw_records),
                })
            # Preserve completed model output before any potentially failing health analysis.
            records = [
                {
                    **row,
                    "health": response_health(
                        row["text"], row["stop_reason"], health_golds[row["scenario_uid"]]
                    ),
                }
                for row in raw_records
            ]
            # One atomic file per batch protects original batch membership after interruption.
            write_json(batch_path, records)
            for record, key in zip(records, keys, strict=True):
                completed[key] = record
            temporary = path.with_suffix(".jsonl.tmp")
            temporary.write_text(
                "".join(
                    json.dumps(record, allow_nan=False) + "\n" for record in completed.values()
                ),
                encoding="utf-8",
            )
            temporary.replace(path)
            print(
                f"calibration {len(completed)}/{len(expected)}; "
                f"last batch {raw_records[0]['runtime']['seconds']:.1f}s",
                flush=True,
            )
    records = list(completed.values())
    if set(completed) != expected:
        raise ValueError("incomplete calibration")
    temporary = path.with_suffix(".jsonl.tmp")
    temporary.write_text(
        "".join(json.dumps(row, allow_nan=False) + "\n" for row in records), encoding="utf-8"
    )
    temporary.replace(path)
    valid = sum(row["health"]["valid_final"] for row in records) / len(records)
    bad = sum(
        row["stop_reason"] != "eos"
        or not row["health"]["completed"]
        or row["health"]["strong_repetition"]
        for row in records
    ) / len(records)
    report = {
        "protocol_hash": protocol_hash,
        "n": len(records),
        "valid_final_fraction": valid,
        "exhausted_unclosed_or_repetitive_fraction": bad,
        "passed": valid >= 0.90 and bad <= 0.10,
        "strict_rule_success_fraction": sum(row["health"]["strict_rule_success"] for row in records)
        / len(records),
        "caveat": "rule success is diagnostic, not an independent semantic judge",
        "artifacts": {
            str(artifact.relative_to(out_dir)): hashlib.sha256(artifact.read_bytes()).hexdigest()
            for artifact in (
                path, *sorted((out_dir / "batches").glob("*.json")),
                *sorted((out_dir / "raw_batches").glob("*.json")),
            )
        },
    }
    write_json(summary_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["certificate", "calibrate"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scenarios", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs=2, default=[20260908, 20260909])
    parser.add_argument("--instruct-template", action="store_true")
    parser.add_argument(
        "--cohort", choices=["independent24", "existing-pilot12"], default="existing-pilot12"
    )
    parser.add_argument("--device", default="cuda:0", choices=["cuda:0", "cpu"])
    args = parser.parse_args()
    if args.stage == "calibrate" and args.scenarios is None:
        parser.error("--scenarios is required for calibration")
    lm = load_model(
        MODEL,
        revision=REVISION,
        device_map=args.device,
        chat_template=subject_chat_template() if args.instruct_template else None,
    )
    if args.stage == "certificate":
        report = runtime_certificate(lm, args.out)
        print(
            json.dumps(
                {
                    key: value
                    for key, value in report.items()
                    if isinstance(value, (bool, float, str))
                },
                indent=2,
            )
        )
    else:
        report = calibrate(
            lm,
            args.scenarios,
            args.out,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            seeds=args.seeds,
            cohort=args.cohort,
        )
        print(json.dumps(report, indent=2))
        if not report["passed"]:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
