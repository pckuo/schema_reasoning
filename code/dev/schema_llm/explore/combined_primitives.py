"""Portable D1/D2-identify + core-spot_error input pack and extraction runner.

The pack contains exact saved token IDs, not re-tokenized text. Heavy capture and
clustering imports are lazy so preparing/checking the handoff needs only Arrow
and NumPy. See main/plans/20261112-COMBINED_PRIMITIVES.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from itertools import groupby
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

LAYERS = [0, 4, 8, 12, 16, 20, 24, 28]
CORPORA = {
    "d1": ("20260915-binding_dyn_d1_v3w2", "identify", 300),
    "d2": ("20260915-binding_dyn_d2_v3w2", "identify", 297),
    "core_spot": ("20260824-rlreasoning_v1", "spot_error", 216),
}
MODELS = {
    "phi4mr": ("microsoft/Phi-4-mini-reasoning", "0e3b1e2d02ee478a3743abe3f629e9c0cb722e0a"),
    "qwen34b": ("Qwen/Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c"),
}
TOKEN_COLUMNS = [
    "item_id",
    "task_family",
    "model",
    "sample_id",
    "position",
    "prompt_len",
    "token_id",
    "token",
    "segment",
]
CONTEXT = (
    "These clusters contain tokens from language-model responses to narrative reasoning "
    "questions. Describe the function supported by the token evidence and excerpts. "
    "Do not assume a particular question, dataset, model, or correct answer."
)
CODE_FILES = [
    "primitives/_cluster.py",
    "primitives/_text.py",
    "primitives/depth_metacluster.py",
    "primitives/crossmodel/merge.py",
    "core/model.py",
    "core/llm.py",
    "core/runio.py",
]
SEQUENCE_SCHEMA = pa.schema(
    [
        ("item_id", pa.string()),
        ("source_item_id", pa.string()),
        ("corpus", pa.string()),
        ("task_family", pa.string()),
        ("model", pa.string()),
        ("sample_id", pa.int64()),
        ("schema", pa.string()),
        ("prompt_len", pa.int64()),
        ("n_tokens", pa.int64()),
        ("token_ids", pa.list_(pa.int32())),
        ("tokens", pa.list_(pa.string())),
        ("segments", pa.list_(pa.string())),
        ("sequence_sha256", pa.string()),
    ]
)


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _write_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _identity(seq: dict) -> str:
    text = json.dumps(
        [
            seq[k]
            for k in (
                "model",
                "item_id",
                "task_family",
                "sample_id",
                "prompt_len",
                "token_ids",
                "tokens",
                "segments",
            )
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(text.encode()).hexdigest()


def _validate_sequence(seq: dict) -> None:
    n, p = seq["n_tokens"], seq["prompt_len"]
    if not 0 < p <= n or any(len(seq[k]) != n for k in ("token_ids", "tokens", "segments")):
        raise ValueError(f"invalid sequence lengths: {seq['item_id']}")
    if seq["segments"][:p] != ["prompt"] * p or any(
        s not in {"think", "answer"} for s in seq["segments"][p:]
    ):
        raise ValueError(f"noncontiguous prompt/response: {seq['item_id']}")
    if seq["sequence_sha256"] != _identity(seq):
        raise ValueError(f"sequence fingerprint changed: {seq['item_id']}")


def sequences(pack: Path, model: str | None = None, corpus: str | None = None):
    for batch in pq.ParquetFile(pack / "sequences.parquet").iter_batches(batch_size=1):
        seq = batch.to_pylist()[0]
        if (model is None or seq["model"] == model) and (corpus is None or seq["corpus"] == corpus):
            _validate_sequence(seq)
            yield seq


def _namespace(table: pa.Table, corpus: str) -> pa.Table:
    old = table["item_id"].to_pylist()
    table = table.append_column("source_item_id", pa.array(old, type=pa.string()))
    table = table.append_column("corpus", pa.array([corpus] * len(old), type=pa.string()))
    return table.set_column(
        table.schema.get_field_index("item_id"),
        "item_id",
        pa.array([f"{corpus}__{i}" for i in old], type=pa.string()),
    )


def prepare(exports: Path, pack: Path, runner: Path) -> None:
    """Stream tokens into one row per response; retain all design/score columns."""
    if pack.exists() and any(pack.iterdir()):
        raise FileExistsError(f"choose a new empty pack directory: {pack}")
    pack.mkdir(parents=True, exist_ok=True)
    tables: dict[str, list[pa.Table]] = {
        k: [] for k in ("tasks", "responses", "item_lookup", "persons", "bindings")
    }
    source_files, counts, totals = {}, {}, {}
    with pq.ParquetWriter(pack / "sequences.parquet", SEQUENCE_SCHEMA) as writer:
        for corpus, (bundle_name, task, expected) in CORPORA.items():
            bundle = exports / bundle_name
            tasks = pq.read_table(bundle / "tasks.parquet", filters=[("task_family", "=", task)])
            task_map = {r["item_id"]: r for r in tasks.to_pylist()}
            if len(task_map) != expected or len(tasks) != expected:
                raise ValueError(f"{corpus}: expected {expected} unique tasks")
            for name in tables:
                path = bundle / f"{name}.parquet"
                if not path.exists():
                    continue
                source_files[str(path)] = _sha(path)
                table = tasks if name == "tasks" else pq.read_table(path)
                table = table.filter(
                    pa.array([i in task_map for i in table["item_id"].to_pylist()])
                )
                if "task_family" in table.column_names:
                    table = table.filter(
                        pa.array([t == task for t in table["task_family"].to_pylist()])
                    )
                tables[name].append(_namespace(table, corpus))
            response_keys = {
                (r["model"], r["item_id"], r["task_family"], r["sample_id"])
                for r in tables["responses"][-1].to_pylist()
            }
            if len(response_keys) != 2 * expected or len(tables["responses"][-1]) != len(
                response_keys
            ):
                raise ValueError(f"{corpus}: missing/duplicate responses")
            token_path = bundle / "tokens.parquet"
            source_files[str(token_path)] = _sha(token_path)
            seen = set()

            def token_rows(path=token_path, selected_task=task):
                for batch in pq.ParquetFile(path).iter_batches(
                    batch_size=16384, columns=TOKEN_COLUMNS
                ):
                    for row in batch.to_pylist():
                        if row["task_family"] == selected_task:
                            yield row

            def key(row: dict) -> tuple:
                return row["model"], row["item_id"], row["task_family"], row["sample_id"]

            for (model, item_id, _, sample), grouped in groupby(token_rows(), key):
                tokens = sorted(grouped, key=lambda r: r["position"])
                full_id = f"{corpus}__{item_id}"
                response_key = model, full_id, task, sample
                if response_key in seen or response_key not in response_keys:
                    raise ValueError(f"noncontiguous or unmatched token sequence: {response_key}")
                seen.add(response_key)
                if model not in MODELS or sample != 0:
                    raise ValueError(f"unexpected model/sample: {response_key}")
                if [r["position"] for r in tokens] != list(range(len(tokens))):
                    raise ValueError(f"missing/duplicate token positions: {response_key}")
                prompt_lengths = {r["prompt_len"] for r in tokens}
                if len(prompt_lengths) != 1:
                    raise ValueError(f"inconsistent prompt length: {response_key}")
                seq = {
                    "item_id": full_id,
                    "source_item_id": item_id,
                    "corpus": corpus,
                    "task_family": task,
                    "model": model,
                    "sample_id": sample,
                    "schema": task_map[item_id]["schema"],
                    "prompt_len": prompt_lengths.pop(),
                    "n_tokens": len(tokens),
                    "token_ids": [r["token_id"] for r in tokens],
                    "tokens": [r["token"] for r in tokens],
                    "segments": [r["segment"] for r in tokens],
                }
                seq["sequence_sha256"] = _identity(seq)
                _validate_sequence(seq)
                writer.write_table(pa.Table.from_pylist([seq], schema=SEQUENCE_SCHEMA))
                totals[corpus] = totals.get(corpus, 0) + len(tokens)
            if seen != response_keys:
                raise ValueError(f"{corpus}: response/token coverage differs")
            counts[corpus] = {"tasks": expected, "responses": len(seen), "tokens": totals[corpus]}
            dest = pack / "provenance" / corpus
            dest.mkdir(parents=True)
            for filename in ("README.md", "confound_audit_union.md", "confound_audit_union.json"):
                if (bundle / filename).exists():
                    shutil.copy2(bundle / filename, dest / filename)
    for name, parts in tables.items():
        if parts:
            pq.write_table(
                pa.concat_tables(parts, promote_options="permissive"), pack / f"{name}.parquet"
            )
    shutil.copy2(Path(__file__), pack / "pipeline.py")
    shutil.copy2(runner, pack / "run.sh")
    shutil.copy2(Path("main/plans/20261112-COMBINED_PRIMITIVES.md"), pack / "README.md")
    # Freeze the small import closure so the collaborator needs no other checkout.
    runtime_files = [
        "pyproject.toml",
        "uv.lock",
        "README.md",
        "LICENSE",
        "code/dev/__init__.py",
        "code/dev/schema_llm/__init__.py",
        "code/dev/schema_llm/core/__init__.py",
        "code/dev/schema_llm/primitives/__init__.py",
        "code/dev/schema_llm/primitives/crossmodel/__init__.py",
        "code/dev/schema_llm/explore/__init__.py",
        "code/dev/schema_llm/explore/combined_primitives.py",
        "code/dev/schema_llm/tests/test_combined_primitives.py",
        "scripts/schema_llm/run_combined_primitives.sh",
        "main/plans/20261112-COMBINED_PRIMITIVES.md",
        *[f"code/dev/schema_llm/{f}" for f in CODE_FILES],
    ]
    for filename in runtime_files:
        target = pack / "runtime" / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(filename, target)
    (pack / "runtime" / ".python-version").write_text("3.11\n")
    _write_json(
        pack / "manifest.json",
        {
            "format": "combined-primitives-v1",
            "counts": counts,
            "models": MODELS,
            "layers": LAYERS,
            "k": 50,
            "meta": 8,
            "seed": 0,
            "pca": None,
            "annotation_context": CONTEXT,
            "source_sha256": source_files,
            "code_sha256": {
                f"code/dev/schema_llm/{f}": _sha(Path("code/dev/schema_llm") / f)
                for f in CODE_FILES
            },
            "files": {
                str(p.relative_to(pack)): _sha(p) for p in sorted(pack.rglob("*")) if p.is_file()
            },
            "notes": [
                "Task is confounded with corpus: discovery/exploratory comparisons only.",
                "Use sequence token IDs, never re-tokenize tasks.prompt_text.",
                "Original merged IDs and assignments are deliberately excluded.",
                "D1/D2 include both waves; retain wave/family/design metadata.",
            ],
        },
    )
    verify(pack)


def verify(pack: Path) -> dict:
    manifest = _json(pack / "manifest.json")
    for name, digest in manifest.get("code_sha256", {}).items():
        if not Path(name).exists() or _sha(Path(name)) != digest:
            raise ValueError(f"use the matching repository code; file differs: {name}")
    for name, digest in manifest["files"].items():
        if _sha(pack / name) != digest:
            raise ValueError(f"input checksum mismatch: {name}")
    counts, seen, per_model, actual_sources = {}, set(), {}, set()
    for seq in sequences(pack):
        key = seq["model"], seq["item_id"]
        if key in seen:
            raise ValueError(f"duplicate sequence: {key}")
        seen.add(key)
        actual_sources.add((seq["corpus"], seq["model"], seq["task_family"]))
        model_key = seq["corpus"], seq["model"]
        per_model[model_key] = per_model.get(model_key, 0) + 1
        c = counts.setdefault(seq["corpus"], {"responses": 0, "tokens": 0})
        c["responses"] += 1
        c["tokens"] += seq["n_tokens"]
    if set(counts) != set(manifest["counts"]):
        raise ValueError("pack corpus coverage differs from manifest")
    declared = [(c, m, t) for c, m, t, _ in _sources(manifest)]
    if set(declared) != actual_sources or len(declared) != len(set(declared)):
        raise ValueError("pack source coverage differs from manifest")
    if len({(c, m) for c, m, _ in declared}) != len(declared):
        raise ValueError("use one task family per corpus/model")
    for corpus, expected in manifest["counts"].items():
        if counts[corpus] != {k: expected[k] for k in ("responses", "tokens")}:
            raise ValueError(f"pack coverage changed: {corpus}")
        if any(per_model.get((corpus, model)) != expected["tasks"] for model in MODELS):
            raise ValueError(f"per-model response count differs: {corpus}")
    print(json.dumps({"verified": str(pack), "counts": counts}, indent=2), flush=True)
    return manifest


def _capture_dir(work: Path, seq: dict) -> Path:
    return (
        work
        / "captures"
        / seq["corpus"]
        / "activations"
        / seq["model"]
        / f"{seq['item_id']}__{seq['task_family']}"
    )


def _complete(path: Path, seq: dict, layers: list[int], capture_key: dict) -> bool:
    if not (path / "meta.json").exists():
        return False
    meta = _json(path / "meta.json")
    if (
        meta.get("sequence_sha256") != seq["sequence_sha256"]
        or meta.get("capture_key") != capture_key
    ):
        raise ValueError(f"capture belongs to different inputs/settings: {path}")
    for layer in layers:
        p = path / f"layer_{layer}.npy"
        if not p.exists():
            return False
        a = np.load(p, mmap_mode="r")
        if a.shape != (seq["n_tokens"], meta["hidden_size"]) or a.dtype != np.float16:
            raise ValueError(f"invalid saved latent: {p}")
    return True


def validate_tokenizer(tokenizer, seqs) -> int:
    vocab = set(tokenizer.get_vocab().values())
    count = 0
    for seq in seqs:
        if not set(seq["token_ids"]) <= vocab:
            raise ValueError(f"saved IDs outside tokenizer vocabulary: {seq['item_id']}")
        if tokenizer.convert_ids_to_tokens(seq["token_ids"]) != seq["tokens"]:
            raise ValueError(f"tokenizer strings differ: {seq['item_id']}")
        count += 1
    return count


def capture(
    pack: Path,
    work: Path,
    model: str,
    checkpoint: str | None,
    revision: str | None,
    limit: int | None,
) -> None:
    manifest = verify(pack)
    from transformers import AutoTokenizer

    checkpoint = checkpoint or MODELS[model][0]
    if revision is None and not Path(checkpoint).exists():
        revision = MODELS[model][1]
    tok = AutoTokenizer.from_pretrained(checkpoint, revision=revision)
    count = validate_tokenizer(tok, sequences(pack, model))
    expected = sum(c["tasks"] for c in manifest["counts"].values())
    if count != expected:
        raise ValueError(f"{model}: expected {expected} validated sequences, got {count}")
    capture_key = {
        "pack": _sha(pack / "manifest.json"),
        "model": model,
        "checkpoint": checkpoint,
        "revision": revision,
        "dtype": "bfloat16",
        "layers": manifest["layers"],
    }
    # Validation of ALL sequences precedes loading the model/any GPU forward.
    import torch
    from baukit import TraceDict

    from dev.schema_llm.core.model import load_model, set_seed

    set_seed(manifest["seed"])
    lm = load_model(checkpoint, revision=revision)
    layers = manifest["layers"]
    if max(layers) >= lm.n_layers:
        raise ValueError("requested layers exceed the model depth")
    names = lm.layer_hook_names(layers)
    provenance = {
        **capture_key,
        "validated_sequences": count,
        "hidden_size": lm.hidden_size,
        "resolved_revision": getattr(lm.model.config, "_commit_hash", None),
    }
    config_path = work / "capture_config" / f"{model}.json"
    if config_path.exists() and _json(config_path) != provenance:
        raise ValueError(f"capture configuration changed: {config_path}")
    _write_json(config_path, provenance)
    for i, seq in enumerate(sequences(pack, model)):
        if limit is not None and i >= limit:
            break
        dest = _capture_dir(work, seq)
        if _complete(dest, seq, layers, capture_key):
            continue
        dest.mkdir(parents=True, exist_ok=True)
        ids = torch.tensor([seq["token_ids"]], dtype=torch.long, device=lm.device)
        with torch.inference_mode(), TraceDict(lm.model, names, retain_output=True) as td:
            # Only the final logit is needed; avoid allocating T x vocab logits.
            lm.model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                use_cache=False,
                logits_to_keep=1,
            )
            for layer, name in zip(layers, names, strict=True):
                h = td[name].output
                h = h[0] if isinstance(h, tuple) else h
                a = h[0].float().cpu().numpy().astype(np.float16)
                if a.shape != (seq["n_tokens"], lm.hidden_size) or not np.isfinite(a).all():
                    raise ValueError(f"invalid residuals: {seq['item_id']} L{layer}")
                temp = dest / f"layer_{layer}.tmp.npy"
                np.save(temp, a)
                temp.replace(dest / f"layer_{layer}.npy")
        _write_json(
            dest / "meta.json",
            {
                "scenario_id": seq["item_id"],
                "dataset": seq["corpus"],
                "task": seq["task_family"],
                "schema": seq["schema"],
                "tokens": seq["tokens"],
                "n_tokens": seq["n_tokens"],
                "response_token_range": [seq["prompt_len"], seq["n_tokens"]],
                "hidden_size": lm.hidden_size,
                "sequence_sha256": seq["sequence_sha256"],
                "capture_key": capture_key,
            },
        )
        print(f"[{model} {i + 1}/{count}] {seq['item_id']}", flush=True)


def _sources(manifest: dict):
    if "sources" in manifest:
        for source in manifest["sources"]:
            corpus, model, task = (source[k] for k in ("corpus", "model", "task"))
            yield corpus, model, task, f"{corpus}/{model}/{task}"
        return
    for corpus in manifest["counts"]:
        for model in MODELS:
            yield corpus, model, CORPORA[corpus][1], f"{corpus}/{model}/{CORPORA[corpus][1]}"


def cluster(pack: Path, work: Path, annotate_model: str, backend: str) -> None:
    manifest = verify(pack)
    from dev.schema_llm.primitives._cluster import (
        _annotate,
        _snippets,
        analyze_scope,
        load_token_sequences,
        scope_refs,
    )

    layers, k, n_meta, seed = (manifest[f] for f in ("layers", "k", "meta", "seed"))
    for corpus, model, task, source in _sources(manifest):
        config = _json(work / "capture_config" / f"{model}.json")
        capture_key = {
            key: config[key]
            for key in ("pack", "model", "checkpoint", "revision", "dtype", "layers")
        }
        seq_map = {s["item_id"]: s for s in sequences(pack, model, corpus)}
        for seq in seq_map.values():
            if not _complete(_capture_dir(work, seq), seq, layers, capture_key):
                raise ValueError(f"missing capture: {seq['item_id']} ({model})")
        dest = work / "nodes" / corpus / model
        dest.mkdir(parents=True, exist_ok=True)
        key = {
            "pack": _sha(pack / "manifest.json"),
            "capture": config,
            "slug": model,
            "tasks": [task],
            "layers": layers,
            "k": k,
            "force_meta": n_meta,
            "seed": seed,
            "pca": None,
            "annotate": True,
            "annotate_model": f"{backend}:{annotate_model}",
            "context": CONTEXT,
        }
        config_path = dest / "config.json"
        if config_path.exists() and _json(config_path) != key:
            raise ValueError(f"clustering inputs/settings changed: {dest}")
        _write_json(config_path, key)
        per_layer, fine_layers, states, metas, passages = [], [], [], [], None
        for layer in layers:
            cache = dest / f"layer_{layer}.json"
            arrays = dest / f"layer_{layer}.npz"
            if not cache.exists() or not arrays.exists():
                run = work / "captures" / corpus
                refs = scope_refs([run], model, layer, [task])
                current, offset = [], 0
                for mp, _ in refs:
                    meta = _json(mp)
                    seq = seq_map[meta["scenario_id"]]
                    n = seq["n_tokens"] - seq["prompt_len"]
                    if n < 3:
                        continue
                    current.append(
                        {
                            "item_id": seq["item_id"],
                            "dataset": corpus,
                            "scenario_id": seq["item_id"],
                            "task": task,
                            "n_tokens": n,
                            "offset": offset,
                        }
                    )
                    offset += n
                seqs = load_token_sequences(
                    [run], model, layer, "response", seed, tasks=[task], pca=None
                )
                if len(seqs) != len(current) or offset < k:
                    raise ValueError(f"clustering passage alignment/count failed: {source}")
                result = analyze_scope(seqs, k, seed, "function", "response", n_meta)
                plot = result["_plot"]["passages"]
                state = np.concatenate([p["state"] for p in plot]).astype(np.int16)
                meta_ids = np.concatenate([p["meta"] for p in plot]).astype(np.int16)
                if state.shape != (offset,) or meta_ids.shape != state.shape:
                    raise ValueError("clustering assignments have wrong lengths")
                rng = np.random.default_rng(seed)
                traces = [{"tokens": p["tokens"]} for p in plot]
                evidence = {}
                for kind, profiles, label_key, group_key in (
                    ("meta", result["meta_profiles"], "meta", "meta"),
                    ("fine", result["fine_states"], "state", "state"),
                ):
                    groups = [p[group_key] for p in plot]
                    evidence[kind] = [
                        {
                            "id": p[label_key],
                            "top": p["top"],
                            "snippets": _snippets(traces, groups, p[label_key], 10, 10, rng),
                        }
                        for p in profiles
                    ]
                mapping = [-1] * k
                for p in result["fine_states"]:
                    mapping[p["state"]] = p["meta"]
                if not np.array_equal(np.asarray(mapping)[state], meta_ids):
                    raise ValueError("fine -> meta mapping failed")
                np.savez_compressed(arrays, state=state, meta=meta_ids)
                _write_json(
                    cache,
                    {
                        "layer": layer,
                        "passages": current,
                        "profiles": result["meta_profiles"],
                        "fine_states": result["fine_states"],
                        "cluster_to_meta": mapping,
                        "evidence": evidence,
                    },
                )
                del result, seqs, plot
            saved = _json(cache)
            current = saved["passages"]
            if passages is not None and passages != current:
                raise ValueError("token order differs across layers")
            passages = current
            with np.load(arrays) as a:
                states.append(a["state"].copy())
                metas.append(a["meta"].copy())
            annotations = {}
            for kind in ("meta", "fine"):
                path = dest / f"layer_{layer}_{kind}_annotations.json"
                evidence = saved["evidence"][kind]
                if path.exists():
                    ann = _json(path)
                else:
                    raw = _annotate(evidence, annotate_model, kind, CONTEXT, backend)
                    ann = {str(i): value for i, value in raw.items()}
                if any(
                    not ann.get(str(e["id"]), {}).get("label", "").strip(" ?")
                    or not ann.get(str(e["id"]), {}).get("description")
                    for e in evidence
                ):
                    raise ValueError(
                        f"incomplete annotation; rerun to retry: {source} L{layer} {kind}"
                    )
                _write_json(path, ann)
                annotations[kind] = ann
            per_layer.append(
                {"layer": layer, "profiles": saved["profiles"], "ann": annotations["meta"]}
            )
            fine_layers.append(
                {
                    "layer": layer,
                    "cluster_to_meta": saved["cluster_to_meta"],
                    "fine_states": saved["fine_states"],
                    "ann": annotations["fine"],
                }
            )
            print(f"[cluster] {source} L{layer} complete", flush=True)
        np.savez_compressed(dest / "kmeans.npz", stack=np.stack(states))
        np.savez_compressed(dest / "clusters.npz", stack=np.stack(metas))
        _write_json(
            dest / "kmeans.json",
            {"key": key, "n_states": k, "passages": passages, "per_layer": fine_layers},
        )
        _write_json(dest / "clusters.json", {"key": key, "per_layer": per_layer})


def merge(pack: Path, work: Path, embed_model: str) -> None:
    manifest = verify(pack)
    inputs = {}
    command = [sys.executable, "-m", "dev.schema_llm.primitives.crossmodel.merge"]
    for corpus, model, _, source in _sources(manifest):
        path = work / "nodes" / corpus / model
        inputs[source] = _sha(path / "clusters.json")
        command.extend(["--source", f"{source}={path}"])
    key = {"pack": _sha(pack / "manifest.json"), "sources": inputs, "embed_model": embed_model}
    path = work / "merge" / "input.json"
    if path.exists() and _json(path) != key:
        raise ValueError("merge inputs changed; choose a new work directory")
    output = work / "merge" / "semavglink" / "merge.json"
    complete = work / "merge" / "complete.json"
    if output.exists() and complete.exists():
        if _json(complete) != {"input": key, "merge_sha256": _sha(output)}:
            raise ValueError("completed merge checksum changed")
        return
    _write_json(path, key)
    command.extend(
        [
            "--methods",
            "semantic-avglink",
            "--embed-model",
            embed_model,
            "--seed",
            str(manifest["seed"]),
            "--title",
            "D1/D2 identify + core spot_error",
            "--out",
            str(work / "merge"),
        ]
    )
    subprocess.run(command, check=True)
    if not output.exists():
        raise ValueError("merge did not produce the expected output")
    _json(output)
    _write_json(complete, {"input": key, "merge_sha256": _sha(output)})


def export(
    pack: Path, work: Path, *, merge_work: Path | None = None, out: Path | None = None
) -> None:
    manifest = verify(pack)
    merge_work = merge_work or work
    merged_path = merge_work / "merge" / "semavglink" / "merge.json"
    merged = _json(merged_path)
    merge_input = _json(merge_work / "merge" / "input.json")
    if _json(merge_work / "merge" / "complete.json") != {
        "input": merge_input,
        "merge_sha256": _sha(merged_path),
    }:
        raise ValueError("merge has no valid completion record")
    for corpus, model, _, source in _sources(manifest):
        if (
            _sha(work / "nodes" / corpus / model / "clusters.json")
            != merge_input["sources"][source]
        ):
            raise ValueError(f"node annotations changed since merge: {source}")
    if [t["theme"] for t in merged["themes"]] != list(range(merged["n_themes"])):
        raise ValueError("merged name order does not match IDs")
    node_map = {}
    for i, theme in enumerate(merged["node_theme"]):
        source = merged["sources"][merged["node_source"][i]]
        key = source, merged["node_layer"][i], merged["node_meta"][i]
        if key in node_map:
            raise ValueError("duplicate merged node")
        node_map[key] = theme
    expected = len(merge_input["sources"]) * len(manifest["layers"]) * manifest["meta"]
    if len(node_map) != expected:
        raise ValueError(f"expected {expected} merged nodes, got {len(node_map)}")
    for _, _, _, source in _sources(manifest):
        for layer in manifest["layers"]:
            for meta in range(manifest["meta"]):
                if (source, layer, meta) not in node_map:
                    raise ValueError(f"missing merged node: {source}, L{layer}, meta {meta}")
    out = out or work / "export"
    out.mkdir(parents=True, exist_ok=True)
    nodes, fine_nodes, assignments, skipped = [], [], {}, []
    for corpus, model, task, source in _sources(manifest):
        path = work / "nodes" / corpus / model
        mc, km = _json(path / "clusters.json"), _json(path / "kmeans.json")
        with np.load(path / "kmeans.npz") as a:
            fine = a["stack"].copy()
        with np.load(path / "clusters.npz") as a:
            meta = a["stack"].copy()
        if fine.shape != meta.shape or fine.shape[0] != len(manifest["layers"]):
            raise ValueError(f"assignment shapes differ: {source}")
        width = sum(p["n_tokens"] for p in km["passages"])
        if width != fine.shape[1] or np.any(fine < 0) or np.any(fine >= manifest["k"]):
            raise ValueError(f"invalid fine assignments: {source}")
        for li, layer in enumerate(manifest["layers"]):
            if np.any(meta[li] < 0) or np.any(meta[li] >= manifest["meta"]):
                raise ValueError(f"invalid meta assignments: {source}")
            mapping = np.asarray(km["per_layer"][li]["cluster_to_meta"])
            if not np.array_equal(mapping[fine[li]], meta[li]):
                raise ValueError(f"fine/meta mapping differs: {source}")
            for pr in mc["per_layer"][li]["profiles"]:
                m = pr["meta"]
                nodes.append(
                    {
                        "source": source,
                        "corpus": corpus,
                        "model": model,
                        "task_family": task,
                        "layer": layer,
                        "metacluster_id": m,
                        "merged_primitive_id": node_map[(source, layer, m)],
                        **pr,
                        **mc["per_layer"][li]["ann"][str(m)],
                    }
                )
            profiles = {p["state"]: p for p in km["per_layer"][li]["fine_states"]}
            for state in range(manifest["k"]):
                pr = profiles.get(state, {})
                fine_nodes.append(
                    {
                        "source": source,
                        "layer": layer,
                        "cluster_id": state,
                        "metacluster_id": int(mapping[state]),
                        "count": pr.get("count", 0),
                        "top": pr.get("top", []),
                        **km["per_layer"][li]["ann"].get(
                            str(state), {"label": None, "description": None}
                        ),
                    }
                )
        offset = 0
        for passage in km["passages"]:
            if passage["offset"] != offset:
                raise ValueError(f"invalid passage offset: {source}")
            n = passage["n_tokens"]
            assignments[(model, passage["item_id"])] = (
                source,
                fine[:, offset : offset + n],
                meta[:, offset : offset + n],
            )
            offset += n
    pq.write_table(pa.Table.from_pylist(nodes), out / "clusters.parquet")
    pq.write_table(pa.Table.from_pylist(fine_nodes), out / "fine_clusters.parquet")
    token_schema = pa.schema(
        [
            ("item_id", pa.string()),
            ("corpus", pa.string()),
            ("model", pa.string()),
            ("task_family", pa.string()),
            ("sample_id", pa.int64()),
            ("position", pa.int64()),
            ("response_position", pa.int64()),
            ("prompt_len", pa.int64()),
            ("token_id", pa.int32()),
            ("token", pa.string()),
            ("segment", pa.string()),
            *[
                (f"{kind}_L{layer}", pa.int16())
                for layer in manifest["layers"]
                for kind in ("kmeans", "metacluster", "merged_primitive")
            ],
        ]
    )
    sequence_rows, n_tokens = [], 0
    with pq.ParquetWriter(out / "tokens.parquet", token_schema) as writer:
        for seq in sequences(pack):
            ids = assignments.pop((seq["model"], seq["item_id"]), None)
            n, p = seq["n_tokens"], seq["prompt_len"]
            if ids is None and n - p >= 3:
                raise ValueError(f"missing token assignments: {seq['item_id']}")
            if ids is None:
                skipped.append(
                    {
                        "item_id": seq["item_id"],
                        "model": seq["model"],
                        "reason": "fewer than 3 response tokens",
                    }
                )
            elif ids[1].shape[1] != n - p:
                raise ValueError("response assignment length differs from saved tokens")
            columns = {
                k: [seq[k]] * n
                for k in ("item_id", "corpus", "model", "task_family", "sample_id", "prompt_len")
            }
            columns.update(
                position=list(range(n)),
                response_position=[None] * p + list(range(n - p)),
                token_id=seq["token_ids"],
                token=seq["tokens"],
                segment=seq["segments"],
            )
            sr = {k: seq[k] for k in ("item_id", "corpus", "model", "task_family", "sample_id")}
            for li, layer in enumerate(manifest["layers"]):
                f = ids[1][li].tolist() if ids else [None] * (n - p)
                m = ids[2][li].tolist() if ids else [None] * (n - p)
                merged_ids = [node_map[(ids[0], layer, i)] for i in m] if ids else m
                for kind, values in (
                    ("kmeans", f),
                    ("metacluster", m),
                    ("merged_primitive", merged_ids),
                ):
                    columns[f"{kind}_L{layer}"] = [None] * p + values
                sr[f"primitive_sequence_L{layer}"] = [
                    v for v, _ in groupby(merged_ids) if v is not None
                ]
            writer.write_table(pa.Table.from_pydict(columns, schema=token_schema))
            sequence_rows.append(sr)
            n_tokens += n
    if assignments:
        raise ValueError("unused assignments remain after export")
    response_table = pq.read_table(pack / "responses.parquet")
    metadata = {(r["model"], r["item_id"]): r for r in response_table.to_pylist()}
    sequence_rows = [{**metadata[(r["model"], r["item_id"])], **r} for r in sequence_rows]
    pq.write_table(pa.Table.from_pylist(sequence_rows), out / "response_sequences.parquet")
    for name in ("tasks", "responses", "persons", "bindings", "item_lookup"):
        if (pack / f"{name}.parquet").exists():
            shutil.copy2(pack / f"{name}.parquet", out / f"{name}.parquet")
    _write_json(out / "merged_primitive_names.json", [t["name"] for t in merged["themes"]])
    _write_json(
        out / "manifest.json",
        {
            "input_pack_sha256": _sha(pack / "manifest.json"),
            "merge_sha256": _sha(merged_path),
            "layers": manifest["layers"],
            "responses": len(sequence_rows),
            "tokens": n_tokens,
            "merged_primitives": merged["n_themes"],
            "skipped": skipped,
            "notes": manifest["notes"],
        },
    )
    print(f"Exported {len(sequence_rows)} responses / {n_tokens} tokens to {out}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--exports", type=Path, default=Path("results/schema_llm/export"))
    prep.add_argument("--pack", type=Path, required=True)
    prep.add_argument(
        "--runner", type=Path, default=Path("scripts/schema_llm/run_combined_primitives.sh")
    )
    for stage in ("verify", "capture", "cluster", "merge", "export"):
        parser = sub.add_parser(stage)
        parser.add_argument("--pack", type=Path, required=True)
        if stage != "verify":
            parser.add_argument("--work", type=Path, required=True)
        if stage == "capture":
            parser.add_argument("--model", choices=list(MODELS), required=True)
            parser.add_argument("--checkpoint")
            parser.add_argument("--revision")
            parser.add_argument("--limit", type=int)
        if stage == "cluster":
            parser.add_argument("--annotate-model", default="gpt-5.6-sol_2026-07-09")
            parser.add_argument("--backend", choices=["trapi", "copilot"], default="trapi")
        if stage == "merge":
            parser.add_argument("--embed-model", default="Qwen/Qwen3-Embedding-4B")
    args = p.parse_args()
    if args.command == "prepare":
        prepare(args.exports, args.pack, args.runner)
    elif args.command == "verify":
        verify(args.pack)
    elif args.command == "capture":
        capture(args.pack, args.work, args.model, args.checkpoint, args.revision, args.limit)
    elif args.command == "cluster":
        cluster(args.pack, args.work, args.annotate_model, args.backend)
    elif args.command == "merge":
        merge(args.pack, args.work, args.embed_model)
    else:
        export(args.pack, args.work)


if __name__ == "__main__":
    main()
