"""Generate D1/D2-style sampled TSP responses and an exact-token primitive input pack."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from . import combined_primitives as cp


def load_tasks(path: Path) -> list[dict]:
    tasks = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    seen = set()
    for task in tasks:
        uid = str(task["uid"])
        if (
            not uid
            or "/" in uid
            or "\\" in uid
            or uid in seen
            or not isinstance(task["prompt"], str)
            or not task["prompt"].strip()
        ):
            raise ValueError(f"duplicate uid or empty prompt: {uid}")
        seen.add(uid)
    if not tasks:
        raise ValueError("no TSP tasks")
    return tasks


def seed_for(model: str, uid, seed: int) -> int:
    value = f"tsp|{seed}|{model}|{uid}".encode()
    return int.from_bytes(hashlib.blake2b(value, digest_size=4).digest(), "big")


def shard_path(out: Path, model: str, uid) -> Path:
    name = hashlib.sha256(str(uid).encode()).hexdigest()
    return out / "shards" / model / f"{name}.json"


def make_row(
    tokenizer, task: dict, model: str, ids: list[int], prompt_len: int, seed: int, cap: int, eos_ids
) -> dict:
    tokens = tokenizer.convert_ids_to_tokens(ids)
    response_ids = ids[prompt_len:]
    text = tokenizer.decode(response_ids, skip_special_tokens=False)
    # Token strings retain the ASCII marker even when Phi splits it across BPE tokens.
    joined = "".join(tokens[prompt_len:])
    close = joined.find("</think>")
    stop = close + len("</think>") if close >= 0 else len(joined)
    cursor, segments = 0, ["prompt"] * prompt_len
    for token in tokens[prompt_len:]:
        segments.append("think" if cursor < stop else "answer")
        cursor += len(token)
    item = f"tsp__{task['uid']}"
    seq = {
        "item_id": item,
        "source_item_id": str(task["uid"]),
        "corpus": "tsp",
        "task_family": "tsp",
        "model": model,
        "sample_id": 0,
        "schema": "tsp",
        "prompt_len": prompt_len,
        "n_tokens": len(ids),
        "token_ids": ids,
        "tokens": tokens,
        "segments": segments,
    }
    seq["sequence_sha256"] = cp._identity(seq)
    cp._validate_sequence(seq)
    eos = eos_ids
    eos = set(eos if isinstance(eos, list) else [eos])
    ended = bool(response_ids and response_ids[-1] in eos)
    response = {
        "item_id": item,
        "corpus": "tsp",
        "model": model,
        "task_family": "tsp",
        "sample_id": 0,
        "uid": str(task["uid"]),
        "response_text": text,
        "seed": seed,
        "prompt_len": prompt_len,
        "n_generated": len(response_ids),
        "hit_cap": len(response_ids) >= cap,
        "terminated": ended,
        "stop_reason": "eos" if ended else ("length" if len(response_ids) >= cap else "other"),
        "closed_think": "</think>" in text,
        "eos_ids": sorted(eos),
        "model_name": cp.MODELS[model][0],
        "revision": cp.MODELS[model][1],
    }
    return {"sequence": seq, "response": response}


def validate_row(row: dict, task: dict, model: str, config: dict) -> None:
    seq, response = row["sequence"], row["response"]
    cp._validate_sequence(seq)
    if (
        row["generation"] != config
        or seq["model"] != model
        or seq["item_id"] != f"tsp__{task['uid']}"
        or response["item_id"] != seq["item_id"]
        or response["model"] != model
    ):
        raise ValueError("saved TSP row belongs to different inputs/settings")


def write_pack(root: Path, out: Path, tasks: list[dict], rows: list[dict], config: dict) -> Path:
    pack = out / "pack"
    pack.mkdir(exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist([r["sequence"] for r in rows], schema=cp.SEQUENCE_SCHEMA),
        pack / "sequences.parquet",
    )
    pq.write_table(pa.Table.from_pylist([r["response"] for r in rows]), pack / "responses.parquet")
    task_rows = [
        {
            **task,
            "item_id": f"tsp__{task['uid']}",
            "source_item_id": str(task["uid"]),
            "corpus": "tsp",
            "task_family": "tsp",
            "schema": "tsp",
            "prompt_text": task["prompt"],
        }
        for task in tasks
    ]
    pq.write_table(pa.Table.from_pylist(task_rows), pack / "tasks.parquet")
    manifest = {
        "format": "combined-primitives-v1",
        "counts": {
            "tsp": {
                "tasks": len(tasks),
                "responses": len(rows),
                "tokens": sum(r["sequence"]["n_tokens"] for r in rows),
            }
        },
        "sources": [{"corpus": "tsp", "model": model, "task": "tsp"} for model in cp.MODELS],
        "models": cp.MODELS,
        "layers": cp.LAYERS,
        "k": 50,
        "meta": 8,
        "seed": 0,
        "pca": None,
        "annotation_context": cp.CONTEXT,
        "generation": config,
        "code_sha256": {
            f"code/dev/schema_llm/{f}": cp._sha(root / "code/dev/schema_llm" / f)
            for f in cp.CODE_FILES
        },
        "files": {p.name: cp._sha(p) for p in pack.glob("*.parquet")},
        "notes": [
            "Original generated prompt/response IDs; no text re-tokenization.",
            "TSP generation uses sampled decoding with no extra system message.",
            "Generation configuration and per-response seeds are retained.",
        ],
    }
    cp._write_json(pack / "manifest.json", manifest)
    cp.verify(pack)
    cp._write_json(
        out / "complete.json",
        {"pack_sha256": cp._sha(pack / "manifest.json"), "responses": len(rows)},
    )
    return pack


def run(
    root: Path,
    tasks_path: Path,
    out: Path,
    cap: int,
    device: str,
    seed: int = 0,
    smoke: bool = False,
) -> Path:
    from ..core.model import generate_reply, load_model, set_seed
    from .reproduction import _runtime

    tasks = load_tasks(tasks_path)
    if smoke:
        tasks = tasks[:2]
    template_path = root / "data/tsp_generation/phi4_mini_instruct_chat_template.jinja"
    config = {
        "release": cp._sha(root / "MANIFEST.json"),
        "tasks_sha256": cp._sha(tasks_path),
        "models": {k: list(v) for k, v in cp.MODELS.items()},
        "engine": "huggingface-generate",
        "runtime": _runtime(),
        "device": device,
        "dtype": "bfloat16",
        "seed": seed,
        "do_sample": True,
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": 50,
        "max_new_tokens": cap,
        "system_message": None,
        "draws": 1,
        "smoke": smoke,
        "phi_chat_template_sha256": cp._sha(template_path),
        "qwen_enable_thinking": True,
        "token_origin": "original-generation",
    }
    state = out / "generation.json"
    if state.exists() and cp._json(state) != config:
        raise ValueError("generation inputs/settings changed; choose a new output folder")
    out.mkdir(parents=True, exist_ok=True)
    cp._write_json(state, config)
    shutil.copy2(tasks_path, out / "tasks.jsonl")
    rows = []
    for model, (checkpoint, revision) in cp.MODELS.items():
        lm = None
        for task in tasks:
            path = shard_path(out, model, task["uid"])
            if path.exists():
                row = cp._json(path)
                validate_row(row, task, model, config)
            else:
                if lm is None:
                    template = template_path.read_text() if model == "phi4mr" else None
                    lm = load_model(
                        checkpoint, revision=revision, device_map=device, chat_template=template
                    )
                value = seed_for(model, task["uid"], seed)
                set_seed(value)
                full, n_prompt, _ = generate_reply(
                    lm,
                    task["prompt"],
                    system_message=None,
                    max_new_tokens=cap,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.95,
                    top_k=50,
                    enable_thinking=True if model == "qwen34b" else None,
                )
                ids = full.tolist()
                rendered = lm.tokenizer.apply_chat_template(
                    [{"role": "user", "content": task["prompt"]}],
                    add_generation_prompt=True,
                    return_dict=True,
                    **({"enable_thinking": True} if model == "qwen34b" else {}),
                )["input_ids"]
                if n_prompt != len(rendered) or ids[:n_prompt] != rendered:
                    raise ValueError("generated prompt IDs differ from the configured template")
                eos = lm.model.generation_config.eos_token_id
                if eos is None:
                    eos = lm.tokenizer.eos_token_id
                row = make_row(lm.tokenizer, task, model, ids, n_prompt, value, cap, eos)
                row["generation"] = config
                cp._write_json(path, row)
            rows.append(row)
            print(
                f"[{model}] uid={task['uid']} tokens={row['response']['n_generated']}", flush=True
            )
        del lm
        gc.collect()
        import torch

        torch.cuda.empty_cache()
    return write_pack(root, out, tasks, rows, config)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[4])
    p.add_argument(
        "--tasks", type=Path, help="JSONL with unique uid and prompt, plus optional metadata"
    )
    p.add_argument("--out", type=Path)
    p.add_argument("--max-new-tokens", type=int, default=32768)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    root = args.root.resolve()
    os.chdir(root)
    from .reproduction import verify

    verify(root)
    tasks = (args.tasks or root / "data/tsp_generation/tasks_100.jsonl").resolve()
    rows = load_tasks(tasks)
    if args.max_new_tokens <= 0:
        p.error("--max-new-tokens must be positive")
    if args.check_only:
        print(
            f"TSP inputs verified: {len(rows)} tasks, {2 * len(rows)} responses planned; "
            "no weights/GPU/API used."
        )
        return
    import torch

    if not torch.cuda.is_available():
        raise ValueError(
            "TSP generation requires NVIDIA CUDA; use --check-only for input validation"
        )
    out = (
        args.out or root / "results" / ("tsp-generation-smoke" if args.smoke else "tsp-generation")
    ).resolve()
    pack = run(
        root,
        tasks,
        out,
        min(args.max_new_tokens, 256) if args.smoke else args.max_new_tokens,
        args.device,
        args.seed,
        args.smoke,
    )
    print(f"Primitive input pack ready: {pack}")


if __name__ == "__main__":
    main()
