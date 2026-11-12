"""Fresh, model-bound Phi-mini AIE; no borrowed heads or centered vectors.

CPU preparation (no model weights) and a dry-run feasibility budget::

    uv run python -m dev.schema_llm.explore.head_fv_aie prepare
    uv run python -m dev.schema_llm.explore.head_fv_aie feasibility

Model stages require explicit ``--execute --device cpu|cuda:0 --max-forwards N
--max-seconds S``. Run feasibility, means, rank, then validate, in that order.
Never run GPU stages while the shared GPU is occupied. Repeating a command resumes
only exactly bound work; errors are terminal, and exhausted budgets are not success.
The sequential implementation is the oracle. Optional identical-prompt head batching
must pass raw-logit parity on each task before its ranking scan.
``--layer-prefix-reuse`` instead skips unchanged lower decoder blocks at identical
full-prefill shapes. It requires sequential execution and exact full-logit parity;
it is not a KV cache and never reuses outputs at or after the first edited block.
Validation alone accepts ``--rank-source PARENT --root CHILD`` to inherit a completed
rank read-only. Only this module's source hash may differ; the child binds both
the current runtime and the immutable parent rank provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import random
import sqlite3
import time
import unicodedata
import urllib.request
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .head_fv_models import active as active_model

TASKS = (
    "antonym", "capitalize", "country-capital", "english-french",
    "present-past", "singular-plural",
)
TODD_REVISION = "fb9eac7b6dc707ea1475a717379916007fe448d5"
MODEL = active_model().model
MODEL_REVISION = active_model().revision
ROOT = Path("results/schema_llm/data/phi_mini_head_fv/v1/aie")
UPSTREAM = f"https://raw.githubusercontent.com/ericwtodd/function_vectors/{TODD_REVISION}"
SEED = 20260908
N_MEANS, MAX_ATTEMPTS, N_RANK, N_VALIDATION, N_DEMOS = 100, 1000, 50, 50, 10
TOP_K, N_BOOTSTRAP = 35, 1000
ADDITIVE_GRID = tuple((layer, gain) for layer in (8, 16, 24) for gain in (.25, .5, 1.))
FORMAT = {
    "demo": "Q: {input}\nA: {output}\n\n",
    "query": "Q: {input}\nA:",
    "answer": " {output}",
    "add_special_tokens": False,
    "bos": "none",
    "chat_template": "not used: plain-text Todd-style ICL",
}


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()


def object_hash(value: Any) -> str:
    return digest(canonical(value))


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_bytes(path: Path, data: bytes, *, immutable: bool = False) -> None:
    if immutable and path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"immutable artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + ".partial")
    with staging.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    staging.replace(path)


def write_json(path: Path, value: Any, *, immutable: bool = False) -> None:
    write_bytes(path, canonical(value) + b"\n", immutable=immutable)


def read_json(path: Path) -> Any:
    return json.loads(path.read_bytes())


def stable_seed(*parts: Any) -> int:
    return int(object_hash(parts)[:16], 16)


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def split_rows(rows: list[dict], task: str, seed: int = SEED) -> tuple[dict, dict]:
    """Group case/whitespace variants; exclude ALL conflicting input groups prospectively."""
    groups: dict[str, list] = defaultdict(list)
    for index, row in enumerate(rows):
        if not isinstance(row.get("input"), str) or not isinstance(row.get("output"), str):
            raise ValueError(f"{task}: non-string mapping at source row {index}")
        if not normalize(row["input"]) or not normalize(row["output"]):
            raise ValueError(f"{task}: empty mapping at source row {index}")
        groups[normalize(row["input"]).casefold()].append((index, row))
    kept, conflicts, duplicates = [], [], []
    for group, members in sorted(groups.items()):
        outputs = {normalize(row["output"]) for _, row in members}
        if len(outputs) != 1:
            conflicts.append({"group": group, "source_rows": [
                {"index": i, **row} for i, row in members
            ]})
            continue
        _, representative = min(members, key=lambda pair: (
            normalize(pair[1]["input"]), normalize(pair[1]["output"]), pair[0],
        ))
        uid = f"{task}:{digest(group.encode())[:20]}"
        kept.append({
            "uid": uid, "group": group, "input": normalize(representative["input"]),
            "output": normalize(representative["output"]),
            "source_indices": [i for i, _ in members],
        })
        if len(members) > 1:
            duplicates.append({"uid": uid, "source_indices": [i for i, _ in members]})
    random.Random(stable_seed(seed, task, "pools")).shuffle(kept)
    n = len(kept)
    a, b = 2 * n // 5, n // 5
    pools = {
        "demo": kept[:a], "mean": kept[a:a + b],
        "rank": kept[a + b:a + 2 * b], "validation": kept[a + 2 * b:],
    }
    if len(pools["demo"]) < N_DEMOS or any(not pool for pool in pools.values()):
        raise ValueError(f"{task}: insufficient unambiguous mappings for disjoint pools")
    report = {
        "source_rows": len(rows), "input_groups": len(groups), "retained_groups": n,
        "duplicate_groups": duplicates, "conflict_groups": conflicts,
        "excluded_conflict_rows": sum(len(g["source_rows"]) for g in conflicts),
        "removed_duplicate_rows": sum(len(g["source_indices"]) - 1 for g in duplicates),
        "policy": "NFC/whitespace/casefold input groups; exact normalized output; "
                  "exclude whole ambiguous group; lexicographic representative",
        "pool_sizes": {name: len(pool) for name, pool in pools.items()},
        "rounding": "demo=floor(.4N), mean=rank=floor(.2N), validation=remainder",
    }
    assert len({row["uid"] for pool in pools.values() for row in pool}) == n
    return pools, report


def corrupt_outputs(demos: list[dict], rng: random.Random) -> tuple[list[int], int]:
    outputs = [row["output"] for row in demos]
    if len(set(outputs)) < 2:
        raise ValueError("no nonidentity corruption exists for these demo outputs")
    for _ in range(1000):
        permutation = rng.sample(range(len(demos)), len(demos))
        changed = sum(outputs[i] != outputs[j] for i, j in enumerate(permutation))
        if changed:
            return permutation, changed
    raise RuntimeError("could not draw a nonidentity corruption")


def contextual_answer(tokenizer: Any, text: str, answer: str) -> tuple[list[int], list[int]]:
    prefix = list(tokenizer.encode(text, add_special_tokens=False))
    full = list(tokenizer.encode(text + FORMAT["answer"].format(output=answer),
                                 add_special_tokens=False))
    if not prefix or full[:len(prefix)] != prefix or len(full) <= len(prefix):
        raise ValueError("answer tokenization changes prompt prefix; "
                         "amend before measuring effects")
    return prefix, full[len(prefix):]


def realize_prompt(
    task: str, stage: str, index: int, query: dict, demos: list[dict],
    tokenizer: Any, rng: random.Random, block: int | None = None,
) -> dict:
    if len(demos) != N_DEMOS or len({r["uid"] for r in demos}) != N_DEMOS:
        raise ValueError("need exactly ten distinct demonstration groups")
    if query["uid"] in {r["uid"] for r in demos}:
        raise ValueError("query leaks into demonstrations")
    permutation, changed = corrupt_outputs(demos, rng)
    suffix = FORMAT["query"].format(**query)
    clean = "".join(FORMAT["demo"].format(**row) for row in demos) + suffix
    corrupt = "".join(FORMAT["demo"].format(input=row["input"],
                       output=demos[permutation[i]]["output"])
                      for i, row in enumerate(demos)) + suffix
    clean_ids, answer_ids = contextual_answer(tokenizer, clean, query["output"])
    corrupt_ids, corrupt_answer = contextual_answer(tokenizer, corrupt, query["output"])
    zero_ids, zero_answer = contextual_answer(tokenizer, suffix, query["output"])
    if answer_ids != corrupt_answer or answer_ids != zero_answer:
        raise ValueError("answer tokens differ across clean/corrupt/zero-shot contexts")
    return {
        "uid": f"{task}:{stage}:{index:04d}", "task": task, "stage": stage, "index": index,
        "block": block, "query_uid": query["uid"], "query": query,
        "demo_uids": [r["uid"] for r in demos], "demos": demos,
        "permutation": permutation, "changed_pairs": changed,
        "clean_text": clean, "corrupt_text": corrupt, "zero_text": suffix,
        "clean_ids": clean_ids, "corrupt_ids": corrupt_ids, "zero_ids": zero_ids,
        "answer_ids": answer_ids, "target_id": answer_ids[0],
    }


def prompt_records(task: str, stage: str, pools: dict, tokenizer: Any) -> list[dict]:
    size = {"mean": MAX_ATTEMPTS, "rank": N_RANK, "validation": N_VALIDATION}[stage]
    records = []
    for block in range(2 if stage == "rank" else 1):
        rng = random.Random(stable_seed(SEED, task, stage, block))
        queries = []
        count = size // 2 if stage == "rank" else size
        while len(queries) < count:
            queries.extend(rng.sample(pools[stage], len(pools[stage])))
        for query in queries[:count]:
            records.append(realize_prompt(
                task, stage, len(records), query, rng.sample(pools["demo"], N_DEMOS),
                tokenizer, rng, block if stage == "rank" else None,
            ))
    return records


def tokenizer_identity(tokenizer: Any) -> dict:
    # Fast tokenizers lazily change backend post-processing on the first encode.
    # Fingerprint the assay's declared no-special-token mode, not the cold loader state.
    tokenizer.encode("Q: x\nA:", add_special_tokens=False)
    return {
        "model": MODEL, "revision": MODEL_REVISION,
        "class": type(tokenizer).__name__,
        "backend_sha256": digest(tokenizer.backend_tokenizer.to_str().encode()),
        "special_tokens": tokenizer.special_tokens_map,
        "bos_token_id": tokenizer.bos_token_id, "eos_token_id": tokenizer.eos_token_id,
        "template_sha256": object_hash(tokenizer.chat_template),
        "transformers": importlib.metadata.version("transformers"),
        "tokenizers": importlib.metadata.version("tokenizers"),
    }


def prepare(root: Path, *, local_files_only: bool = False) -> dict:
    source_at_start = file_hash(Path(__file__))
    manifest_path = root / "preparation.json"
    if manifest_path.exists():
        return verify_preparation(root)
    sources = {}
    for name, upstream_path in [
        *((f"{task}.json", f"dataset_files/abstractive/{task}.json") for task in TASKS),
        ("LICENSE", "LICENSE"), ("upstream_dataset_README.md", "dataset_files/README.md"),
    ]:
        path = root / "datasets" / name
        url = f"{UPSTREAM}/{upstream_path}"
        if not path.exists():
            with urllib.request.urlopen(url, timeout=90) as response:
                write_bytes(path, response.read(), immutable=True)
        sources[name] = {"url": url, "sha256": file_hash(path), "bytes": path.stat().st_size}
    provenance = {
        "repository": "https://github.com/ericwtodd/function_vectors",
        "revision": TODD_REVISION, "sources": sources,
        "license": "MIT", "license_file": "datasets/LICENSE",
        "license_scope": "Upstream repository MIT license; dataset README declares no separate "
                         "task-specific licenses. This is not independent third-party clearance.",
    }
    write_json(root / "datasets" / "provenance.json", provenance, immutable=True)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, revision=MODEL_REVISION, cache_dir=str(root / "tokenizer_cache"),
        local_files_only=local_files_only,
    )
    reports = {}
    for task in TASKS:
        pools, report = split_rows(read_json(root / "datasets" / f"{task}.json"), task)
        write_json(root / "datasets" / f"{task}_pools.json", pools, immutable=True)
        stage_reports = {}
        for stage in ("mean", "rank", "validation"):
            records = prompt_records(task, stage, pools, tokenizer)
            payload = b"".join(canonical(record) + b"\n" for record in records)
            write_bytes(root / "prompts" / f"{task}_{stage}.jsonl", payload, immutable=True)
            by_token: dict[int, set[str]] = defaultdict(set)
            for record in records:
                by_token[record["target_id"]].add(record["query"]["output"])
            counts = Counter(record["query_uid"] for record in records)
            stage_reports[stage] = {
                "prompts": len(records), "unique_queries": len(counts),
                "query_multiplicities": dict(counts),
                "multi_token_prompts": sum(len(r["answer_ids"]) > 1 for r in records),
                "first_token_collisions": {str(k): sorted(v) for k, v in by_token.items()
                                           if len(v) > 1},
                "prompt_token_range": [
                    min(len(r["clean_ids"]) for r in records),
                    max(max(len(r["clean_ids"]), len(r["corrupt_ids"])) for r in records),
                ],
                "min_changed_pairs": min(r["changed_pairs"] for r in records),
            }
        reports[task] = {**report, "tokenization": stage_reports}
    report_path = root / "datasets" / "audit.json"
    write_json(report_path, reports, immutable=True)
    files = {str(path.relative_to(root)): file_hash(path)
             for folder in ("datasets", "prompts")
             for path in sorted((root / folder).iterdir()) if path.is_file()}
    manifest = {
        "schema": "phi-mini-aie-preparation-v1", "status": "prepared_not_model_evaluated",
        "seed": SEED, "model": MODEL, "model_revision": MODEL_REVISION,
        "todd_revision": TODD_REVISION, "tasks": list(TASKS), "format": FORMAT,
        "tokenizer": tokenizer_identity(tokenizer), "files": files,
        "sampling": {
            "demos": N_DEMOS, "successful_means_per_task": N_MEANS,
            "mean_attempt_limit": MAX_ATTEMPTS, "ranking_blocks": [25, 25],
            "validation": N_VALIDATION, "feasibility": "first rank block, unfiltered",
            "query_repeats": "balanced shuffled cycles within each pool; never across pools",
        },
        "gates": {
            "feasibility": {"clean_accuracy": .60, "probability_drop": .05, "tasks": 4},
            "validation": "pooled selected AIE and selected-minus-random paired-query "
                          "95% lower bounds >0; mean joint >0",
            "additive": "rank-only common layer/gain; held-out corrupt 95% lower bound >0",
        },
        "statistics": {"bootstrap": N_BOOTSTRAP, "seed": SEED,
                       "unit": "query group within task, task-equal aggregate"},
        "additive_grid": ADDITIVE_GRID, "source_sha256_at_preparation": source_at_start,
    }
    write_json(manifest_path, manifest, immutable=True)
    return manifest


def verify_preparation(root: Path) -> dict:
    manifest = read_json(root / "preparation.json")
    if manifest["model"] != MODEL or manifest["model_revision"] != MODEL_REVISION:
        raise ValueError("preparation model identity mismatch")
    if manifest["tasks"] != list(TASKS) or manifest["format"] != FORMAT:
        raise ValueError("preparation protocol mismatch")
    for relative, expected in manifest["files"].items():
        path = root / relative
        if not path.is_file() or file_hash(path) != expected:
            raise ValueError(f"preparation artifact mismatch: {relative}")
    return manifest


def read_prompts(root: Path, task: str, stage: str) -> list[dict]:
    return [json.loads(line) for line in
            (root / "prompts" / f"{task}_{stage}.jsonl").read_text().splitlines()]


@dataclass(frozen=True)
class Geometry:
    # field ORDER is load-bearing: saved contracts and tests construct this positionally
    layers: int = active_model().geometry[0]
    heads: int = active_model().geometry[1]
    kv_heads: int = active_model().geometry[2]
    head_dim: int = active_model().geometry[4]
    hidden: int = active_model().geometry[3]


SUBJECT_GEOMETRY = Geometry()
PHI_GEOMETRY = SUBJECT_GEOMETRY  # retained: the pilot's saved contracts name this


def validate_model(lm: Any, geometry: Geometry = PHI_GEOMETRY) -> list[Any]:
    cfg = lm.model.config
    actual = Geometry(
        cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads,
        getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads,
        cfg.hidden_size,
    )
    if actual != geometry or lm.layer_prefix != "model.layers":
        raise ValueError(f"loaded geometry differs: {actual} != {geometry}")
    modules = [layer.self_attn.o_proj for layer in lm.model.model.layers]
    if len(modules) != geometry.layers:
        raise ValueError("missing decoder layer")
    for module in modules:
        if tuple(module.weight.shape) != (geometry.hidden, geometry.heads * geometry.head_dim):
            raise ValueError("wrong o_proj geometry")
        if module.bias is not None:
            raise ValueError("assay requires bias-free o_proj")
    return modules


def check_heads(heads: list[tuple[int, int]], geometry: Geometry) -> None:
    if len(set(heads)) != len(heads):
        raise ValueError("duplicate heads")
    if any(not (0 <= layer < geometry.layers and 0 <= head < geometry.heads)
           for layer, head in heads):
        raise ValueError("head out of range")


def top_heads(scores: np.ndarray, count: int = TOP_K) -> list[tuple[int, int]]:
    if scores.ndim != 2 or not np.isfinite(scores).all() or not 0 < count <= scores.size:
        raise ValueError("invalid head-score matrix/count")
    return sorted(np.ndindex(scores.shape),
                  key=lambda pair: (-float(scores[pair]), *pair))[:count]


def random_head_sets(
    selected: list[tuple[int, int]], geometry: Geometry, seed: int = SEED,
) -> list[list[tuple[int, int]]]:
    check_heads(selected, geometry)
    counts = Counter(layer for layer, _ in selected)
    sets = []
    for draw in range(3):
        rng = random.Random(stable_seed(seed, "random-head-set", draw))
        sets.append([(layer, head) for layer, count in sorted(counts.items())
                     for head in sorted(rng.sample(range(geometry.heads), count))])
    return sets


@contextmanager
def head_hooks(
    modules: list[Any], geometry: Geometry, means: np.ndarray | None = None,
    interventions: list[list[tuple[int, int]]] | None = None,
    capture: dict[int, np.ndarray] | None = None,
):
    """Per-row query-head replacement at -1 only; actual o_proj always executes."""
    handles = []
    if interventions is not None:
        if means is None or means.shape != (geometry.layers, geometry.heads, geometry.head_dim):
            raise ValueError("mean activation geometry mismatch")
        if not np.isfinite(means).all():
            raise ValueError("nonfinite means")
        for heads in interventions:
            check_heads(heads, geometry)
    try:
        for layer, module in enumerate(modules):
            if capture is None and (interventions is None or not any(
                source_layer == layer for heads in interventions for source_layer, _ in heads
            )):
                continue
            def hook(_module, args, layer=layer):
                import torch

                x = args[0]
                if x.ndim != 3 or x.shape[-1] != geometry.heads * geometry.head_dim:
                    raise ValueError("unexpected pre-o_proj shape")
                if capture is not None:
                    if x.shape[0] != 1:
                        raise ValueError("mean capture requires a single prompt")
                    capture[layer] = x[0, -1].detach().float().cpu().numpy().reshape(
                        geometry.heads, geometry.head_dim,
                    ).copy()
                edits = [] if interventions is None else [
                    (row, head) for row, heads in enumerate(interventions)
                    for source_layer, head in heads if source_layer == layer
                ]
                if not edits:
                    return None
                assert interventions is not None and means is not None
                if x.shape[0] != len(interventions):
                    raise ValueError("intervention roster does not match batch")
                patched = x.clone()
                for row, head in edits:
                    start = head * geometry.head_dim
                    patched[row, -1, start:start + geometry.head_dim] = torch.as_tensor(
                        means[layer, head], dtype=x.dtype, device=x.device,
                    )
                return (patched, *args[1:])
            handles.append(module.register_forward_pre_hook(hook))
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def additive_hook(block: Any, vector: np.ndarray, gain: float):
    """Final predictive prompt position, NOT the later continuation injection schedule."""
    def hook(_module, _args, output):
        import torch

        if gain == 0:
            return None
        x = output[0] if isinstance(output, tuple) else output
        if vector.shape != (x.shape[-1],) or not np.isfinite(vector).all():
            raise ValueError("invalid additive FV")
        edited = x.clone()
        edited[:, -1] += torch.as_tensor(vector * gain, dtype=x.dtype, device=x.device)
        return (edited, *output[1:]) if isinstance(output, tuple) else edited
    handle = block.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def raw_endpoint(logits: Any, target: int) -> tuple[np.ndarray, np.ndarray]:
    if logits.ndim != 3 or not 0 <= target < logits.shape[-1]:
        raise ValueError("invalid final-logit shape or target token")
    values = logits[:, -1].float()
    return (values.softmax(-1)[:, target].detach().cpu().numpy(),
            values.argmax(-1).detach().cpu().numpy())


def clone_layer_output(output: Any) -> Any:
    import torch

    if isinstance(output, torch.Tensor):
        return output.detach().clone()
    if isinstance(output, tuple):
        return tuple(clone_layer_output(item) for item in output)
    if output is None:
        return None
    raise ValueError("layer-prefix reuse requires tensor/tuple decoder outputs")


@contextmanager
def reuse_lower_layers(blocks: Any, outputs: list[Any], stop: int):
    """Keep all tensor shapes and suffix arithmetic identical to full-prefill execution."""
    restored = []
    try:
        for index, (block, output) in enumerate(zip(blocks[:stop], outputs[:stop], strict=True)):
            restored.append((block, "forward" in block.__dict__, block.forward))

            def replay(*_args, output=output, final=index == stop - 1, **_kwargs):
                # Only the final replay feeds live decoder arithmetic; earlier outputs feed replays.
                return clone_layer_output(output) if final else output
            block.forward = replay
        yield
    finally:
        for block, had_override, original in reversed(restored):
            if had_override:
                block.forward = original
            else:
                del block.forward


class BudgetExhausted(RuntimeError):
    pass


@dataclass
class Budget:
    max_forwards: int
    max_seconds: float
    max_forward_tokens: int = 100_000_000
    forwards: int = 0
    forward_tokens: int = 0
    started: float = 0.
    completed_forwards: int = 0
    forward_calls: int = 0
    forward_seconds: float = 0.
    prefix_prefills: int = 0
    reused_decoder_blocks: int = 0

    def __post_init__(self):
        if self.max_forwards <= 0 or self.max_seconds <= 0 or self.max_forward_tokens <= 0:
            raise ValueError("positive explicit budgets are required")
        self.started = time.monotonic()

    def reserve(self, batch: int, length: int) -> None:
        if (self.forwards + batch > self.max_forwards
                or self.forward_tokens + batch * length > self.max_forward_tokens
                or time.monotonic() - self.started >= self.max_seconds):
            raise BudgetExhausted("invocation budget exhausted before next forward")
        self.forwards += batch
        self.forward_tokens += batch * length

    def report(self) -> dict:
        rate = self.completed_forwards / self.forward_seconds if self.forward_seconds else None
        return {
            "forward_examples": self.forwards, "forward_tokens": self.forward_tokens,
            "elapsed_seconds": time.monotonic() - self.started,
            "completed_forward_examples": self.completed_forwards,
            "forward_calls": self.forward_calls, "forward_seconds": self.forward_seconds,
            "prefix_prefills": self.prefix_prefills,
            "reused_decoder_blocks": self.reused_decoder_blocks,
            "completed_examples_per_forward_second": rate,
            "estimated_230400_patched_forward_hours": 230400 / rate / 3600 if rate else None,
            "throughput_caveat": "measured forwards include hook/logit CPU transfers, exclude "
                                 "model loading, disk commits and statistics; profile the "
                                 "chosen head batch before extrapolating",
            "max_forwards": self.max_forwards, "max_seconds": self.max_seconds,
            "max_forward_tokens": self.max_forward_tokens,
            "wall_limit": "checked between forwards; does not interrupt a running forward",
        }


class Ledger:
    """Atomic per-condition records; identity mismatch/error rows cannot masquerade as resume."""

    def __init__(self, path: Path, identity: dict):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS records (stage TEXT, key TEXT, status TEXT, "
            "payload TEXT, array BLOB, PRIMARY KEY(stage,key))"
        )
        encoded = canonical(identity).decode()
        previous = self.db.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()
        if previous is not None and previous[0] != encoded:
            self.db.close()
            raise ValueError("resume identity mismatch (inputs/code/model/runtime/batch)")
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO metadata VALUES ('identity',?)", (encoded,))
        self.identity = object_hash(identity)
        self._grouped = False
        self._pending = 0

    def get(self, stage: str, key: str) -> tuple[dict, np.ndarray | None] | None:
        row = self.db.execute(
            "SELECT status,payload,array FROM records WHERE stage=? AND key=?", (stage, key),
        ).fetchone()
        if row is None:
            return None
        if row[0] != "ok":
            raise RuntimeError(f"terminal {row[0]} record: {stage}/{key}: {row[1]}")
        array = None if row[2] is None else np.load(io.BytesIO(row[2]), allow_pickle=False)
        return json.loads(row[1]), array

    def put(self, stage: str, key: str, payload: dict, array: np.ndarray | None = None,
            *, status: str = "ok") -> None:
        if status not in ("ok", "error"):
            raise ValueError("record must be terminal")
        blob = None
        if array is not None:
            buffer = io.BytesIO()
            np.save(buffer, array, allow_pickle=False)
            blob = buffer.getvalue()
        if self._grouped:
            self.db.execute("INSERT INTO records VALUES (?,?,?,?,?)",
                            (stage, key, status, canonical(payload).decode(), blob))
            self._pending += 1
            if self._pending == 64:
                self.db.commit()
                self._pending = 0
        else:
            with self.db:
                self.db.execute("INSERT INTO records VALUES (?,?,?,?,?)",
                                (stage, key, status, canonical(payload).decode(), blob))

    @contextmanager
    def grouped(self):
        """WAL/FULL commits every 64 records; graceful exits flush completed partial groups."""
        if self._grouped:
            raise ValueError("nested ledger groups are unsupported")
        self._grouped, self._pending = True, 0
        try:
            yield
        finally:
            try:
                self.db.commit()
            finally:
                self._grouped, self._pending = False, 0

    def close(self) -> None:
        self.db.close()

    def progress(self) -> dict:
        records = [{"stage": stage, "status": status, "records": count}
                   for stage, status, count in self.db.execute(
                       "SELECT stage,status,COUNT(*) FROM records GROUP BY stage,status"
                   )]
        coverage = {task: {"attempted": 0, "accepted": 0, "rejected": 0, "errors": 0}
                    for task in TASKS}
        for key, status, payload in self.db.execute(
            "SELECT key,status,payload FROM records WHERE stage='mean_attempt'"
        ):
            task = key.split(":")[0]
            coverage[task]["attempted"] += 1
            outcome = ("errors" if status != "ok"
                       else "accepted" if json.loads(payload)["accepted"] else "rejected")
            coverage[task][outcome] += 1
        for counts in coverage.values():
            counts["acceptance_rate"] = (counts["accepted"] / counts["attempted"]
                                         if counts["attempted"] else None)
        return {"terminal_record_counts": records, "successful_icl_coverage": coverage}


class Assay:
    def __init__(self, lm: Any, ledger: Ledger, budget: Budget,
                 geometry: Geometry = PHI_GEOMETRY, head_batch_size: int = 1,
                 layer_prefix_reuse: bool = False):
        self.lm, self.ledger, self.budget = lm, ledger, budget
        self.geometry, self.head_batch_size = geometry, head_batch_size
        if layer_prefix_reuse and head_batch_size != 1:
            raise ValueError("layer-prefix reuse requires sequential head evaluation")
        self.layer_prefix_reuse = layer_prefix_reuse
        self._prefix_ids: tuple[int, ...] | None = None
        self._prefix_outputs: list[Any] = []
        self.modules = validate_model(lm, geometry)
        self.lm.model.eval()

    def ensure_prefix(self, ids: list[int], target: int) -> None:
        if self._prefix_ids == tuple(ids):
            return
        self._prefix_ids, self._prefix_outputs = None, []
        outputs: dict[int, Any] = {}
        handles = []
        try:
            for index, block in enumerate(self.lm.model.model.layers):
                def capture(_module, _args, output, index=index):
                    outputs[index] = clone_layer_output(output)
                handles.append(block.register_forward_hook(capture))
            self.forward(ids, target, reuse_prefix=False)
            if len(outputs) != self.geometry.layers:
                raise ValueError("missing layer-prefix output")
            self._prefix_outputs = [outputs[i] for i in range(self.geometry.layers)]
            self._prefix_ids = tuple(ids)
            self.budget.prefix_prefills += 1
        finally:
            for handle in handles:
                handle.remove()

    def forward(
        self, ids: list[int], target: int, *, means: np.ndarray | None = None,
        interventions: list[list[tuple[int, int]]] | None = None, capture: bool = False,
        additive: tuple[int, float, np.ndarray] | None = None,
        reuse_prefix: bool | None = None,
    ) -> tuple[dict, np.ndarray | None, np.ndarray]:
        import inspect
        from contextlib import ExitStack

        import torch

        batch = 1 if interventions is None else len(interventions)
        if not ids or batch < 1:
            raise ValueError("empty prompt or intervention batch")
        reuse = self.layer_prefix_reuse if reuse_prefix is None else reuse_prefix
        stop = 0
        if reuse and interventions is not None:
            if batch != 1 or capture or additive is not None:
                raise ValueError("prefix reuse supports sequential replacement without capture")
            check_heads(interventions[0], self.geometry)
            stop = min((layer for layer, _ in interventions[0]), default=0)
            if stop:
                self.ensure_prefix(ids, target)
        self.budget.reserve(batch, len(ids))
        started = time.monotonic()
        tensor = torch.tensor([ids], device=self.lm.device).expand(batch, -1)
        captures: dict[int, np.ndarray] = {}
        with torch.inference_mode(), ExitStack() as stack:
            if stop:
                stack.enter_context(reuse_lower_layers(
                    self.lm.model.model.layers, self._prefix_outputs, stop,
                ))
            stack.enter_context(head_hooks(
                self.modules, self.geometry, means, interventions, captures if capture else None,
            ))
            if additive is not None:
                layer, gain, vector = additive
                stack.enter_context(additive_hook(self.lm.model.model.layers[layer], vector, gain))
            kwargs: dict[str, Any] = {
                "input_ids": tensor, "attention_mask": torch.ones_like(tensor), "use_cache": False,
            }
            if "logits_to_keep" in inspect.signature(self.lm.model.forward).parameters:
                kwargs["logits_to_keep"] = 1
            output = self.lm.model(**kwargs)
            probabilities, predictions = raw_endpoint(output.logits, target)
            logits = output.logits[:, -1].detach().float().cpu().numpy()
        if not np.isfinite(logits).all():
            raise ValueError("nonfinite raw logits")
        self.budget.forward_calls += 1
        self.budget.completed_forwards += batch
        self.budget.forward_seconds += time.monotonic() - started
        self.budget.reused_decoder_blocks += stop
        activation = None
        if capture:
            if set(captures) != set(range(self.geometry.layers)):
                raise ValueError("missing mean-capture layer")
            activation = np.stack([captures[layer] for layer in range(self.geometry.layers)])
        return {"probabilities": probabilities.tolist(), "predictions": predictions.tolist(),
                "target": target}, activation, logits

    def cached(self, stage: str, key: str, operation: Any) -> tuple[dict, np.ndarray | None]:
        previous = self.ledger.get(stage, key)
        if previous is not None:
            return previous
        try:
            payload, array = operation()
        except BudgetExhausted:
            raise
        except Exception as error:
            self.ledger.put(stage, key, {"error": repr(error)}, status="error")
            raise
        self.ledger.put(stage, key, payload, array)
        return payload, array

    def baseline(self, prompt: dict, mode: str) -> dict:
        def operation():
            result, _, _ = self.forward(prompt[f"{mode}_ids"], prompt["target_id"])
            return {**result, "prompt_uid": prompt["uid"], "query_uid": prompt["query_uid"]}, None
        return self.cached("baseline", f"{prompt['uid']}:{mode}", operation)[0]

    def individual(self, prompt: dict, means: np.ndarray,
                   heads: list[tuple[int, int]], stage: str) -> np.ndarray:
        check_heads(heads, self.geometry)
        baseline = self.baseline(prompt, "corrupt")["probabilities"][0]
        effects = []
        with self.ledger.grouped():
            for offset in range(0, len(heads), self.head_batch_size):
                chunk = heads[offset:offset + self.head_batch_size]
                key = f"{prompt['uid']}:{object_hash(chunk)}"

                def operation(chunk=chunk):
                    result, _, _ = self.forward(
                        prompt["corrupt_ids"], prompt["target_id"], means=means,
                        interventions=[[head] for head in chunk],
                    )
                    return {
                        **result, "heads": chunk, "baseline_probability": baseline,
                        "effects": [p - baseline for p in result["probabilities"]],
                        "prompt_uid": prompt["uid"], "query_uid": prompt["query_uid"],
                    }, None
                result, _ = self.cached(stage, key, operation)
                effects.extend(result["effects"])
        return np.asarray(effects, dtype=np.float64)


def save_array(path: Path, array: np.ndarray) -> str:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    write_bytes(path, buffer.getvalue(), immutable=True)
    return file_hash(path)


def bootstrap_task_equal(
    values: list[np.ndarray], query_groups: list[list[str]],
    draws: int = N_BOOTSTRAP, seed: int = SEED,
) -> np.ndarray:
    """Cluster resample queries, retaining every realization; keep tasks equally weighted."""
    if not values or len(values) != len(query_groups):
        raise ValueError("missing task/query groups")
    rng = np.random.default_rng(seed)
    total = None
    for observations, groups in zip(values, query_groups, strict=True):
        observations = np.asarray(observations, dtype=np.float64)
        if observations.ndim == 1:
            observations = observations[:, None]
        if len(observations) != len(groups) or not len(groups):
            raise ValueError("query groups do not align to observations")
        if not np.isfinite(observations).all():
            raise ValueError("nonfinite effects")
        unique = sorted(set(groups))
        sums = np.stack([observations[np.array(groups) == group].sum(0) for group in unique])
        counts = np.array([groups.count(group) for group in unique])
        weights = rng.multinomial(len(unique), np.full(len(unique), 1 / len(unique)), size=draws)
        boot = (weights @ sums) / (weights @ counts)[:, None]
        total = boot if total is None else total + boot
    assert total is not None
    return total / len(values)


def feasibility(assay: Assay, root: Path) -> dict:
    results = {}
    for task in TASKS:
        prompts = read_prompts(root, task, "rank")[:25]
        rows = []
        for prompt in prompts:
            clean, corrupt = (assay.baseline(prompt, mode) for mode in ("clean", "corrupt"))
            rows.append({
                "prompt_uid": prompt["uid"], "query_uid": prompt["query_uid"],
                "clean_probability": clean["probabilities"][0],
                "corrupt_probability": corrupt["probabilities"][0],
                "clean_correct": clean["predictions"][0] == prompt["target_id"],
                "corrupt_correct": corrupt["predictions"][0] == prompt["target_id"],
            })
        accuracy = float(np.mean([row["clean_correct"] for row in rows]))
        drop = float(np.mean([row["clean_probability"] - row["corrupt_probability"]
                              for row in rows]))
        results[task] = {
            "attempted": len(prompts), "evaluated": len(rows), "filtered": 0,
            "clean_accuracy": accuracy,
            "corrupt_accuracy": float(np.mean([row["corrupt_correct"] for row in rows])),
            "clean_probability": float(np.mean([row["clean_probability"] for row in rows])),
            "corrupt_probability": float(np.mean([row["corrupt_probability"] for row in rows])),
            "mean_probability_drop": drop, "passes": accuracy >= .60 and drop >= .05,
            "raw": rows,
        }
    prompt = read_prompts(root, TASKS[0], "rank")[0]

    def certify_identity():
        _, captured, baseline = assay.forward(
            prompt["clean_ids"], prompt["target_id"], capture=True,
        )
        all_heads = list(np.ndindex((assay.geometry.layers, assay.geometry.heads)))
        _, _, replaced = assay.forward(
            prompt["clean_ids"], prompt["target_id"], means=captured,
            interventions=[all_heads],
        )
        if not np.array_equal(baseline, replaced):
            raise ValueError("own-activation replacement is not exact identity")
        return {"prompt_uid": prompt["uid"], "exact_raw_logit_identity": True,
                "replaced_heads": len(all_heads)}, np.stack([baseline, replaced])

    certificate, _ = assay.cached("certificate", "own-activation-identity", certify_identity)
    passed = sum(result["passes"] for result in results.values()) >= 4
    return {"status": "complete" if passed else "failed_gate", "gate_passed": passed,
            "tasks": results, "identity_certificate": certificate,
            "next_stage_permitted": passed, "task_count": len(results)}


def collect_means(assay: Assay, root: Path) -> dict:
    reports = {}
    for task in TASKS:
        accumulated = np.zeros((assay.geometry.layers, assay.geometry.heads,
                                assay.geometry.head_dim), dtype=np.float64)
        accepted, attempts, accepted_ids, accepted_queries = 0, 0, [], []
        for prompt in read_prompts(root, task, "mean"):
            def operation(prompt=prompt):
                result, activations, _ = assay.forward(
                    prompt["clean_ids"], prompt["target_id"], capture=True,
                )
                correct = result["predictions"][0] == prompt["target_id"]
                return {**result, "accepted": correct, "prompt_uid": prompt["uid"],
                        "query_uid": prompt["query_uid"]}, activations if correct else None
            row, activations = assay.cached("mean_attempt", prompt["uid"], operation)
            attempts += 1
            if row["accepted"]:
                if activations is None or activations.shape != accumulated.shape:
                    raise ValueError("missing accepted prompt activations")
                accumulated += activations.astype(np.float64)
                accepted += 1
                accepted_ids.append(prompt["uid"])
                accepted_queries.append(prompt["query_uid"])
            if accepted == N_MEANS:
                break
        report = {
            "attempted": attempts, "accepted": accepted, "rejected": attempts - accepted,
            "acceptance_rate": accepted / attempts, "attempt_limit": MAX_ATTEMPTS,
            "criterion": "argmax of raw final-position logits equals context-stable first token",
            "accepted_prompt_uids": accepted_ids, "passes": accepted == N_MEANS,
            "accepted_query_multiplicities": dict(Counter(accepted_queries)),
            "unique_accepted_queries": len(set(accepted_queries)),
        }
        if report["passes"]:
            mean = (accumulated / accepted).astype(np.float32)
            report["mean_sha256"] = save_array(root / "clean_means" / f"{task}.npy", mean)
        reports[task] = report
    passed = all(report["passes"] for report in reports.values())
    return {"status": "complete" if passed else "failed_gate", "gate_passed": passed,
            "tasks": reports, "next_stage_permitted": passed}


def load_means(root: Path, geometry: Geometry) -> dict[str, np.ndarray]:
    status = require_stage(root, "means")
    means = {}
    for task in TASKS:
        path = root / "clean_means" / f"{task}.npy"
        if file_hash(path) != status["tasks"][task]["mean_sha256"]:
            raise ValueError("clean mean artifact changed")
        means[task] = np.load(path, allow_pickle=False)
        if means[task].shape != (geometry.layers, geometry.heads, geometry.head_dim):
            raise ValueError("clean mean geometry changed")
    return means


def certify_batch(
    assay: Assay, prompt: dict, means: np.ndarray,
    heads: list[tuple[int, int]], label: str,
) -> None:
    if assay.head_batch_size == 1:
        return
    sizes = {min(assay.head_batch_size, len(heads) - offset)
             for offset in range(0, len(heads), assay.head_batch_size)}
    for size in sorted(sizes - {1}):
        sample = [heads[i] for i in np.linspace(0, len(heads) - 1, size, dtype=int)]

        def operation(sample=sample):
            sequential, probabilities = [], []
            for head in sample:
                result, _, logits = assay.forward(
                    prompt["corrupt_ids"], prompt["target_id"], means=means,
                    interventions=[[head]],
                )
                sequential.append(logits[0])
                probabilities.extend(result["probabilities"])
            result, _, batched = assay.forward(
                prompt["corrupt_ids"], prompt["target_id"], means=means,
                interventions=[[head] for head in sample],
            )
            oracle = np.stack(sequential)
            if not np.allclose(oracle, batched, rtol=1e-4, atol=1e-5):
                raise ValueError("head batching failed sequential raw-logit oracle; "
                                 "use a NEW sequential run, not relaxed tolerances")
            return {
                "prompt_uid": prompt["uid"], "heads": sample,
                "max_logit_absolute_error": float(np.max(np.abs(oracle - batched))),
                "rtol": 1e-4, "atol": 1e-5, "passed": True,
                "scope": "one frozen prompt/task/phase for each realized batch size",
            }, np.array([probabilities, result["probabilities"]])
        assay.cached("batch_certificate", f"{label}:{prompt['task']}:{size}", operation)


def certify_prefix(assay: Assay, prompt: dict, means: np.ndarray, label: str) -> None:
    if not assay.layer_prefix_reuse:
        return
    heads = [(layer, (3 * layer) % assay.geometry.heads)
             for layer in range(assay.geometry.layers)]

    def operation():
        probabilities = []
        for head in heads:
            oracle, _, full = assay.forward(
                prompt["corrupt_ids"], prompt["target_id"], means=means,
                interventions=[[head]], reuse_prefix=False,
            )
            candidate, _, reused = assay.forward(
                prompt["corrupt_ids"], prompt["target_id"], means=means,
                interventions=[[head]],
            )
            if not np.array_equal(full, reused):
                raise ValueError(f"layer-prefix reuse failed exact full-logit oracle: {head}")
            probabilities.append([oracle["probabilities"][0], candidate["probabilities"][0]])
        return {
            "prompt_uid": prompt["uid"], "heads": heads, "passed": True,
            "exact_raw_logit_identity": True, "cache": "unchanged lower decoder outputs only",
            "scope": "one frozen prompt/task/phase; one head in every decoder layer",
        }, np.asarray(probabilities)
    assay.cached("prefix_certificate", f"{label}:{prompt['task']}", operation)


def build_task_fvs(
    assay: Assay, root: Path, means: dict[str, np.ndarray],
    selected: list[tuple[int, int]],
) -> dict[str, np.ndarray]:
    from dev.schema_llm.primitives.funcvec_heads import build_head_fv, oproj_slices

    geometry = assay.geometry
    check_heads(selected, geometry)
    triples = [(layer, head, 0.) for layer, head in selected]
    slices = oproj_slices(assay.lm, triples, geometry.head_dim)
    if set(slices) != {f"{layer}_{head}" for layer, head in selected}:
        raise ValueError("missing, extra, or biased projection slices")
    if any(value.shape != (geometry.hidden, geometry.head_dim) for value in slices.values()):
        raise ValueError("projection slice shape mismatch")
    buffer = io.BytesIO()
    np.savez(buffer, **slices)
    write_bytes(root / "effects" / "projection_slices.npz", buffer.getvalue(), immutable=True)
    vectors, certificates = {}, {}
    for task in TASKS:
        vector, used = build_head_fv(
            means[task][None], list(range(geometry.layers)), slices, triples,
            geometry.head_dim, overall=None,
        )
        if used != selected:
            raise ValueError("constructor omitted selected heads")
        vector = vector[0]
        reconstruction = np.zeros(geometry.hidden, dtype=np.float32)
        for layer in sorted({layer for layer, _ in selected}):
            packed = np.zeros(geometry.heads * geometry.head_dim, dtype=np.float32)
            for source, head in selected:
                if source == layer:
                    packed[head * geometry.head_dim:(head + 1) * geometry.head_dim] = (
                        means[task][layer, head]
                    )
            weight = assay.modules[layer].weight.detach().float().cpu().numpy()
            reconstruction += weight @ packed
        if not np.isfinite(vector).all() or not np.linalg.norm(vector):
            raise ValueError("nonfinite or zero task FV")
        if not np.allclose(vector, reconstruction, rtol=1e-4, atol=1e-5):
            raise ValueError("raw task FV failed independent full-projection reconstruction")
        if not np.allclose((vector / len(selected)) * len(selected), vector,
                           rtol=1e-4, atol=1e-5):
            raise ValueError("FV sum/mean conversion failed")
        vectors[task] = vector
        certificates[task] = {
            "sum_sha256": save_array(root / "effects" / f"{task}_fv_sum.npy", vector),
            "mean_sha256": save_array(root / "effects" / f"{task}_fv_mean.npy",
                                      vector / len(selected)),
            "norm": float(np.linalg.norm(vector)),
            "max_reconstruction_error": float(np.max(np.abs(vector - reconstruction))),
        }
    write_json(root / "effects" / "fv_certificate.json", {
        "convention": "uncentered raw head projection sum; no AIE weighting",
        "heads": selected, "alpha_mean_per_alpha_sum": len(selected),
        "projection_slices_sha256": file_hash(root / "effects" / "projection_slices.npz"),
        "tasks": certificates,
    }, immutable=True)
    return vectors


def additive_effect(
    assay: Assay, prompt: dict, vector: np.ndarray, setting: tuple[int, float],
    mode: str, stage: str,
) -> float:
    layer, gain = setting
    baseline = assay.baseline(prompt, mode)["probabilities"][0]

    def operation():
        result, _, _ = assay.forward(prompt[f"{mode}_ids"], prompt["target_id"],
                                     additive=(layer, gain, vector))
        effect = result["probabilities"][0] - baseline
        return {**result, "effect": effect, "baseline_probability": baseline,
                "layer": layer, "gain": gain, "mode": mode, "prompt_uid": prompt["uid"],
                "query_uid": prompt["query_uid"]}, None
    row, _ = assay.cached(stage, f"{prompt['uid']}:{mode}:{layer}:{gain}", operation)
    return row["effect"]


def ranking(assay: Assay, root: Path) -> dict:
    means = load_means(root, assay.geometry)
    all_heads = list(np.ndindex((assay.geometry.layers, assay.geometry.heads)))
    effects, groups = [], []
    for task in TASKS:
        prompts = read_prompts(root, task, "rank")
        certify_prefix(assay, prompts[0], means[task], "rank")
        certify_batch(assay, prompts[0], means[task], all_heads, "rank")
        task_effects = []
        for prompt in prompts:
            assay.baseline(prompt, "clean")
            task_effects.append(assay.individual(prompt, means[task], all_heads, "rank_head"))
        effects.append(np.stack(task_effects))
        groups.append([p["query_uid"] for p in prompts])
    raw = np.stack(effects).reshape(len(TASKS), N_RANK,
                                    assay.geometry.layers, assay.geometry.heads)
    raw_hash = save_array(root / "effects" / "rank_raw.npy", raw)
    per_task = raw.mean(1)
    pooled = per_task.mean(0)
    selected = top_heads(pooled, TOP_K)
    random_sets = random_head_sets(selected, assay.geometry)
    boot = bootstrap_task_equal(effects, groups, draws=N_BOOTSTRAP)
    inclusion = np.zeros_like(pooled)
    for sample in boot:
        for head in top_heads(sample.reshape(pooled.shape), TOP_K):
            inclusion[head] += 1 / N_BOOTSTRAP
    block_lists = [top_heads(raw[:, start:start + 25].mean(1).mean(0), TOP_K)
                   for start in (0, 25)]
    save_array(root / "effects" / "aie_by_task.npy", per_task)
    save_array(root / "effects" / "aie_pooled.npy", pooled)
    save_array(root / "effects" / "bootstrap_inclusion.npy", inclusion)
    selection = {
        "schema": "phi-mini-fresh-aie-heads-v1", "status": "rank_selected_not_validated",
        "model": MODEL, "revision": MODEL_REVISION, "geometry": asdict(assay.geometry),
        "preparation_sha256": object_hash(verify_preparation(root)),
        "runtime_identity": assay.ledger.identity, "raw_effects_sha256": raw_hash,
        "heads": selected, "scores": [float(pooled[head]) for head in selected],
        "top_k": TOP_K, "tie_break": "score descending, layer ascending, head ascending",
        "task_weighting": "equal; all six retained; no signed-effect clipping",
        "random_sets": random_sets,
        "random_overlaps": [len(set(selected) & set(heads)) for heads in random_sets],
        "random_seed": SEED, "layer_counts": dict(Counter(layer for layer, _ in selected)),
        "block_top_heads": block_lists,
        "block_overlap": len(set(block_lists[0]) & set(block_lists[1])),
        "bootstrap_unit": "query groups within each task",
        "n_bootstrap": N_BOOTSTRAP, "bootstrap_seed": SEED,
    }
    write_json(root / "effects" / "selection.json", selection, immutable=True)
    vectors = build_task_fvs(assay, root, means, selected)
    grid_effects = []
    for setting in ADDITIVE_GRID:
        task_values = []
        for task in TASKS:
            task_values.append([
                additive_effect(assay, prompt, vectors[task], setting, "corrupt", "rank_additive")
                for prompt in read_prompts(root, task, "rank")
            ])
        grid_effects.append(task_values)
    grid_effects = np.asarray(grid_effects)
    grid_scores = grid_effects.mean(2).mean(1)
    winner = min(range(len(ADDITIVE_GRID)),
                 key=lambda i: (-float(grid_scores[i]), ADDITIVE_GRID[i][1], ADDITIVE_GRID[i][0]))
    frozen = {
        "selection_sha256": object_hash(selection), "setting": ADDITIVE_GRID[winner],
        "grid": ADDITIVE_GRID, "rank_task_equal_scores": grid_scores.tolist(),
        "raw_sha256": save_array(root / "effects" / "rank_additive_raw.npy", grid_effects),
        "choice": "rank corrupted-ICL probability recovery; ties lower gain, then lower layer",
        "validation_effects_opened": False,
    }
    write_json(root / "effects" / "additive_setting.json", frozen, immutable=True)
    return {"status": "complete", "gate_passed": True, "validated": False,
            "raw_shape": list(raw.shape), "selection_sha256": object_hash(selection),
            "additive_setting_sha256": object_hash(frozen),
            "note": "ranking complete; NOT Gate B evidence until independent validation"}


def read_selection(
    path: Path, expected_sha256: str, geometry: Geometry, *,
    expected_file_sha256: str | None = None,
) -> dict:
    """Restore the v1 integer-key histogram before checking its original object hash."""
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"frozen rank selection duplicate key: {key}")
            result[key] = value
        return result

    data = path.read_bytes()
    if expected_file_sha256 is not None and digest(data) != expected_file_sha256:
        raise ValueError("frozen rank selection file changed")
    selection = json.loads(data, object_pairs_hook=unique_object)
    if (not isinstance(selection, dict)
            or selection.get("schema") != "phi-mini-fresh-aie-heads-v1"
            or not isinstance(selection.get("layer_counts"), dict)):
        raise ValueError("frozen rank selection schema changed")
    counts = {}
    for key, count in selection["layer_counts"].items():
        if (not key.isascii() or not key.isdecimal() or str(int(key)) != key
                or not 0 <= int(key) < geometry.layers):
            raise ValueError("frozen rank selection noncanonical layer_counts key")
        if type(count) is not int or not 0 < count <= geometry.heads:
            raise ValueError("frozen rank selection invalid layer_counts count")
        counts[int(key)] = count
    selection["layer_counts"] = counts
    if object_hash(selection) != expected_sha256:
        raise ValueError("frozen rank selection object changed")
    if selection.get("geometry") != asdict(geometry):
        raise ValueError("frozen rank selection geometry changed")

    def heads_from(value):
        if (not isinstance(value, list) or any(
            not isinstance(pair, list) or len(pair) != 2
            or any(type(index) is not int for index in pair) for pair in value
        )):
            raise ValueError("frozen rank selection invalid heads")
        heads = [tuple(pair) for pair in value]
        check_heads(heads, geometry)
        if len(heads) != TOP_K or Counter(layer for layer, _ in heads) != counts:
            raise ValueError("frozen rank selection inconsistent layer_counts")
        return heads

    selected = heads_from(selection["heads"])
    if type(selection["top_k"]) is not int or selection["top_k"] != TOP_K:
        raise ValueError("frozen rank selection invalid top_k")
    if not isinstance(selection["random_sets"], list) or len(selection["random_sets"]) != 3:
        raise ValueError("frozen rank selection invalid random sets")
    controls = [heads_from(heads) for heads in selection["random_sets"]]
    if selection["random_overlaps"] != [len(set(selected) & set(heads)) for heads in controls]:
        raise ValueError("frozen rank selection inconsistent random overlaps")
    return selection


def validation(assay: Assay, root: Path, *, rank_source: Path | None = None) -> dict:
    input_root = stage_input_root("validate", root, rank_source)
    parent_identity = assay.ledger.identity
    provenance = None
    if rank_source is not None:
        parent, provenance = read_rank_source(input_root)
        stored = json.loads(assay.ledger.db.execute(
            "SELECT value FROM metadata WHERE key='identity'"
        ).fetchone()[0])
        current = {key: value for key, value in stored.items() if key != "rank_source"}
        if object_hash(bind_rank_source(current, parent, provenance)) != assay.ledger.identity:
            raise ValueError("validation ledger does not bind this parent rank")
        parent_identity = provenance["runtime_identity"]
    rank_status = require_stage(input_root, "rank")
    selection = read_selection(
        input_root / "effects" / "selection.json", rank_status["selection_sha256"],
        assay.geometry,
        expected_file_sha256=rank_status.get("artifacts", {}).get("effects/selection.json"),
    )
    setting_artifact = read_json(input_root / "effects" / "additive_setting.json")
    if (object_hash(setting_artifact) != rank_status["additive_setting_sha256"]
            or setting_artifact["selection_sha256"] != rank_status["selection_sha256"]):
        raise ValueError("frozen rank selection/additive setting changed")
    if (selection["runtime_identity"] != parent_identity
            or rank_status.get("runtime_identity", parent_identity) != parent_identity):
        raise ValueError("head list belongs to a different runtime")
    selected = [tuple(pair) for pair in selection["heads"]]
    random_sets = [[tuple(pair) for pair in heads] for heads in selection["random_sets"]]
    union = sorted(set(selected).union(*map(set, random_sets)))
    check_heads(selected, assay.geometry)
    if len(selected) != TOP_K or len(random_sets) != 3:
        raise ValueError("invalid frozen selected/control set counts")
    means = load_means(input_root, assay.geometry)
    certificate = read_json(input_root / "effects" / "fv_certificate.json")
    if certificate["heads"] != selection["heads"]:
        raise ValueError("task FV certificate belongs to different heads")
    setting = tuple(setting_artifact["setting"])
    task_metrics, query_groups, task_reports = [], [], {}
    for task in TASKS:
        path = input_root / "effects" / f"{task}_fv_sum.npy"
        if file_hash(path) != certificate["tasks"][task]["sum_sha256"]:
            raise ValueError("frozen task FV changed")
        vector = np.load(path, allow_pickle=False)
        prompts = read_prompts(input_root, task, "validation")
        certify_prefix(assay, prompts[0], means[task], "validation")
        certify_batch(assay, prompts[0], means[task], union, "validation")
        metrics, raw_rows = [], []
        for prompt in prompts:
            clean, corrupt = (assay.baseline(prompt, mode) for mode in ("clean", "corrupt"))
            individual = assay.individual(prompt, means[task], union, "validation_head")
            lookup = dict(zip(union, individual, strict=True))
            chosen = float(np.mean([lookup[head] for head in selected]))
            random_means = [float(np.mean([lookup[head] for head in heads]))
                            for heads in random_sets]

            def joint_operation(prompt=prompt, corrupt=corrupt, task=task):
                result, _, _ = assay.forward(
                    prompt["corrupt_ids"], prompt["target_id"], means=means[task],
                    interventions=[selected],
                )
                return {**result, "effect": result["probabilities"][0]
                        - corrupt["probabilities"][0], "prompt_uid": prompt["uid"],
                        "query_uid": prompt["query_uid"],
                        "baseline_probability": corrupt["probabilities"][0]}, None
            joint, _ = assay.cached("validation_joint", prompt["uid"], joint_operation)
            added = additive_effect(
                assay, prompt, vector, setting, "corrupt", "validation_additive",
            )
            zero = additive_effect(assay, prompt, vector, setting, "zero", "validation_additive")
            metrics.append([chosen, chosen - np.mean(random_means), joint["effect"], added, zero])
            raw_rows.append({
                "prompt_uid": prompt["uid"], "query_uid": prompt["query_uid"],
                "heads": union, "individual_effects": individual.tolist(),
                "selected_mean": chosen, "random_set_means": random_means,
                "joint_effect": joint["effect"], "additive_corrupt_effect": added,
                "additive_zero_effect": zero,
                "clean": clean, "corrupt": corrupt,
            })
        task_metrics.append(np.asarray(metrics))
        query_groups.append([p["query_uid"] for p in prompts])
        task_reports[task] = {
            "attempted": len(prompts), "evaluated": len(raw_rows), "filtered": 0,
            "unique_queries": len(set(query_groups[-1])),
            "selected_mean_aie": float(np.mean(np.array(metrics)[:, 0])),
            "selected_minus_random_mean": float(np.mean(np.array(metrics)[:, 1])),
            "joint_mean_effect": float(np.mean(np.array(metrics)[:, 2])),
            "additive_corrupt_mean": float(np.mean(np.array(metrics)[:, 3])),
            "additive_zero_mean": float(np.mean(np.array(metrics)[:, 4])),
            "clean_accuracy": float(np.mean([
                row["clean"]["predictions"][0] == row["clean"]["target"] for row in raw_rows
            ])),
        }
        write_json(root / "validation" / f"{task}_raw.json", raw_rows, immutable=True)
    boot = bootstrap_task_equal(task_metrics, query_groups, draws=N_BOOTSTRAP)
    estimate = np.stack(task_metrics).mean(1).mean(0)
    bounds = np.quantile(boot, [.025, .975], axis=0)
    names = ("selected_aie", "selected_minus_random", "joint_replacement",
             "additive_corrupt", "additive_zero")
    summaries = {name: {"mean": float(estimate[i]), "ci95": bounds[:, i].tolist()}
                 for i, name in enumerate(names)}
    gate_b = bool(bounds[0, 0] > 0 and bounds[0, 1] > 0 and estimate[2] > 0)
    additive_passed = bool(bounds[0, 3] > 0)
    result = {
        "status": "complete" if gate_b and additive_passed else "failed_gate",
        "gate_passed": gate_b, "additive_gate_passed": additive_passed,
        "next_stage_permitted": gate_b and additive_passed, "tasks": task_reports,
        "pooled": summaries, "selection_sha256": object_hash(selection),
        "additive_setting_sha256": object_hash(setting_artifact),
        "random_overlaps": selection["random_overlaps"],
        "interval_scope": "paired query-group bootstrap within task; six fixed tasks, "
                          "conditional on the three fixed random sets",
        "successful_icl_coverage": require_stage(input_root, "means")["tasks"],
        "no_reselection_on_validation": True,
    }
    if provenance is not None:
        result["rank_source"] = provenance
    save_array(root / "validation" / "paired_raw.npy", np.stack(task_metrics))
    return result


def require_stage(root: Path, stage: str) -> dict:
    path = root / "runtime" / f"{stage}_status.json"
    if not path.exists():
        raise ValueError(f"required stage not run: {stage}")
    status = read_json(path)
    if status["status"] != "complete" or not status["gate_passed"]:
        raise ValueError(f"required stage did not pass: {stage}")
    for relative, expected in status.get("artifacts", {}).items():
        path = root / relative
        if not path.is_file() or file_hash(path) != expected:
            raise ValueError(f"completed stage artifact changed: {relative}")
    return status


def stage_input_root(stage: str, root: Path, rank_source: Path | None) -> Path:
    if rank_source is None:
        return root
    if stage != "validate":
        raise ValueError("--rank-source is valid only for validate")
    parent, child = rank_source.resolve(), root.resolve()
    if parent == child or parent in child.parents or child in parent.parents:
        raise ValueError("--rank-source and --root must be different, non-overlapping roots")
    return parent


def read_rank_source(root: Path) -> tuple[dict, dict]:
    """Read only checkpointed, completed parent inputs; never open a writable parent ledger."""
    manifest = verify_preparation(root)
    statuses = {stage: require_stage(root, stage) for stage in ("feasibility", "means", "rank")}
    required_artifacts = {
        "means": {f"clean_means/{task}.npy" for task in TASKS},
        "rank": {f"effects/{name}" for name in (
            "rank_raw.npy", "aie_by_task.npy", "aie_pooled.npy", "bootstrap_inclusion.npy",
            "selection.json", "projection_slices.npz", "fv_certificate.json",
            "rank_additive_raw.npy", "additive_setting.json",
            *(f"{task}_fv_{kind}.npy" for task in TASKS for kind in ("sum", "mean")),
        )},
    }
    for stage, required in required_artifacts.items():
        if not required <= statuses[stage].get("artifacts", {}).keys():
            raise ValueError(f"parent {stage} is missing required artifact hashes")
    if not {f"prompts/{task}_validation.jsonl" for task in TASKS} <= manifest["files"].keys():
        raise ValueError("parent preparation is missing validation prompt hashes")
    path = root / "runtime" / "records.sqlite"
    wal = path.with_name(path.name + "-wal")
    if wal.exists() and wal.stat().st_size:
        raise ValueError("parent ledger must be checkpointed before read-only inheritance")
    db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
    try:
        row = db.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()
    finally:
        db.close()
    if row is None:
        raise ValueError("parent ledger is missing its runtime identity")
    identity = json.loads(row[0])
    identity_hash = object_hash(identity)
    if canonical(identity).decode() != row[0] or any(
        status.get("runtime_identity") != identity_hash for status in statuses.values()
    ):
        raise ValueError("parent stage/ledger runtime identities differ")
    if identity.get("preparation_sha256") != object_hash(manifest):
        raise ValueError("parent preparation identity differs")
    return identity, {
        "root": str(root.resolve()), "runtime_identity": identity_hash,
        "rank_status_sha256": file_hash(root / "runtime" / "rank_status.json"),
    }


def bind_rank_source(current: dict, parent: dict, provenance: dict) -> dict:
    source = str(Path(__file__))
    if (source not in parent.get("source_hashes", {})
            or current.get("source_hashes", {}).get(source) != file_hash(Path(__file__))):
        raise ValueError("parent/current AIE source identity is missing or not current")
    expected = {**parent, "source_hashes": {
        **parent["source_hashes"], source: current["source_hashes"][source],
    }}
    if object_hash(current) != object_hash(expected):
        raise ValueError("parent rank runtime differs outside head_fv_aie.py source")
    return {**current, "rank_source": provenance}


def runtime_identity(args: Any, manifest: dict, lm: Any) -> dict:
    import torch

    tokenizer = tokenizer_identity(lm.tokenizer)
    if tokenizer != manifest["tokenizer"]:
        raise ValueError("loaded tokenizer differs from prepared tokenizer")
    parameter_dtypes = sorted({str(parameter.dtype) for parameter in lm.model.parameters()})
    if parameter_dtypes != ["torch.bfloat16"]:
        raise ValueError(f"assay requires BF16 model parameters: {parameter_dtypes}")
    return {
        "preparation_sha256": object_hash(manifest), "model": MODEL,
        "revision": MODEL_REVISION, "geometry": asdict(Geometry()),
        "dtype": "bfloat16", "device": args.device, "head_batch_size": args.head_batch_size,
        "attention_implementation": lm.model.config._attn_implementation,
        "model_config_sha256": object_hash(lm.model.config.to_dict()),
        "numerics": {
            "parameter_dtypes": parameter_dtypes,
            "matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "device_name": (torch.cuda.get_device_name(lm.device)
                            if lm.device.type == "cuda" else "cpu"),
        },
        "source_hashes": {
            str(path): file_hash(path) for path in (
                Path(__file__), Path("code/dev/schema_llm/core/model.py"),
                Path("code/dev/schema_llm/primitives/funcvec_heads.py"), Path("uv.lock"),
            )
        },
        "versions": {name: importlib.metadata.version(name)
                     for name in ("numpy", "torch", "transformers", "tokenizers", "accelerate")},
        "endpoint": "raw FP32 softmax of final predictive-position logits, target first token",
        "cache": False, "padding": "none, identical-ID row replication",
        "batch_oracle": {"rtol": 1e-4, "atol": 1e-5},
        "layer_prefix_reuse": getattr(args, "layer_prefix_reuse", False),
        "prefix_oracle": "exact full-logit equality, every layer, one prompt/task/phase",
        "ledger_commits": "WAL/FULL; individual effects in groups of 64 records; graceful exits "
                          "flush; hard kills may require recomputing fewer than 64 records",
    }


def execute_stage(args: Any, manifest: dict) -> None:
    stage = args.stage
    rank_source = getattr(args, "rank_source", None)
    input_root = stage_input_root(stage, args.root, rank_source)
    if args.head_batch_size < 1 or args.head_batch_size > 768:
        raise ValueError("head batch size must be 1..768")
    if args.layer_prefix_reuse and args.head_batch_size != 1:
        raise ValueError("--layer-prefix-reuse requires --head-batch-size 1")
    budget = Budget(args.max_forwards, args.max_seconds, args.max_forward_tokens)
    if stage != "feasibility":
        require_stage(input_root, "feasibility")
    if stage in ("rank", "validate"):
        require_stage(input_root, "means")
    if stage == "validate":
        require_stage(input_root, "rank")
    parent = read_rank_source(input_root) if rank_source is not None else None
    from dev.schema_llm.core.model import load_model

    lm = load_model(MODEL, dtype="bfloat16", device_map=args.device, revision=MODEL_REVISION)
    validate_model(lm)
    identity = runtime_identity(args, manifest, lm)
    if parent is not None:
        identity = bind_rank_source(identity, *parent)
    ledger = Ledger(args.root / "runtime" / "records.sqlite", identity)
    prerequisites = (["feasibility"] if stage != "feasibility" else [])
    if stage in ("rank", "validate"):
        prerequisites.append("means")
    if stage == "validate":
        prerequisites.append("rank")
    for prerequisite in prerequisites:
        expected_identity = parent[1]["runtime_identity"] if parent is not None else ledger.identity
        if require_stage(input_root, prerequisite)["runtime_identity"] != expected_identity:
            ledger.close()
            raise ValueError(f"{prerequisite} runtime identity differs from the current run")
    assay = Assay(lm, ledger, budget, head_batch_size=args.head_batch_size,
                  layer_prefix_reuse=args.layer_prefix_reuse)
    path = args.root / "runtime" / f"{stage}_status.json"
    running = {"status": "running", "gate_passed": False, "runtime_identity": ledger.identity,
               "stage": stage}
    if parent is not None:
        running["rank_source"] = parent[1]
    write_json(path, running)
    exit_code = 0
    try:
        operation = {"feasibility": feasibility, "means": collect_means,
                     "rank": ranking, "validate": validation}[stage]
        result = (validation(assay, args.root, rank_source=input_root) if parent is not None
                  else operation(assay, args.root))
        if result["status"] != "complete":
            exit_code = 2
    except BudgetExhausted as error:
        result = {"status": "budget_exhausted", "gate_passed": False, "reason": str(error)}
        exit_code = 3
    except KeyboardInterrupt:
        result = {"status": "interrupted", "gate_passed": False, "reason": "operator interruption"}
        exit_code = 130
    except Exception as error:
        result = {"status": "error", "gate_passed": False, "reason": repr(error)}
        exit_code = 1
        raise
    finally:
        if "result" in locals():
            artifact_dir = {"means": "clean_means", "rank": "effects",
                            "validate": "validation"}.get(stage)
            if result["status"] in ("complete", "failed_gate") and artifact_dir is not None:
                result["artifacts"] = {
                    str(item.relative_to(args.root)): file_hash(item)
                    for item in sorted((args.root / artifact_dir).glob("*")) if item.is_file()
                }
            result.update({"stage": stage, "runtime_identity": ledger.identity,
                           "budget": budget.report(), "progress": ledger.progress()})
            if parent is not None:
                result["rank_source"] = parent[1]
            write_json(path, result)
        ledger.close()
    print(json.dumps(result, indent=2))
    if exit_code:
        raise SystemExit(exit_code)


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "feasibility", "means", "rank", "validate"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--rank-source", type=Path,
                        help="Validate a completed read-only parent rank in a separate --root; "
                             "only the head_fv_aie.py source hash may change")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--execute", action="store_true",
                        help="Explicitly authorize weight loading and bounded model forwards")
    parser.add_argument("--device", choices=("cpu", "cuda:0"))
    parser.add_argument("--max-forwards", type=int)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--max-forward-tokens", type=int, default=100_000_000)
    parser.add_argument("--head-batch-size", type=int, default=1)
    parser.add_argument("--layer-prefix-reuse", action="store_true",
                        help="Reuse unchanged lower decoder blocks; exact full-logit oracle "
                             "required, sequential heads only; not a KV cache")
    args = parser.parse_args()
    try:
        input_root = stage_input_root(args.stage, args.root, args.rank_source)
    except ValueError as error:
        parser.error(str(error))
    if args.stage == "prepare":
        result = prepare(args.root, local_files_only=args.local_files_only)
        print(json.dumps({"status": result["status"],
                          "preparation_sha256": object_hash(result), "root": str(args.root)}))
        return
    manifest = verify_preparation(input_root)
    if not args.execute:
        print(json.dumps({
            "status": "dry_run_no_weights_loaded", "stage": args.stage,
            "preparation_sha256": object_hash(manifest),
            "feasibility_forward_examples": 302, "ranking_patched_examples": 230400,
            "mean_forward_examples_range": [600, 6000],
            "ranking_additive_examples": 2700,
            "validation_individual_upper_bound": 42000,
            "prerequisites": [
                "idle authorized device; model weights; explicit forwards/time/token budgets",
                "feasibility passes at least 4/6 tasks before means/rank",
                "100 accepted means per task, maximum 1000 attempts",
                "batch size >1 requires sequential raw-logit oracle parity",
                "frozen all-head ranking and rank-only additive choice before validation",
            ],
            "throughput": "sequential default; identical-prompt intervention batching only "
                          "after parity; no cache/padding/long generations; final logits only "
                          "when supported; budgets count examples, not model calls",
        }, indent=2))
        return
    if args.device is None or args.max_forwards is None or args.max_seconds is None:
        parser.error("--execute needs --device, --max-forwards, and --max-seconds")
    try:
        execute_stage(args, manifest)
    except Exception as error:
        write_json(args.root / "runtime" / f"{args.stage}_invocation_error.json", {
            "status": "error", "stage": args.stage, "reason": repr(error),
            "preparation_sha256": object_hash(manifest),
            "note": "No completed-stage evidence is implied by a failed invocation.",
        })
        raise


if __name__ == "__main__":
    cli()
