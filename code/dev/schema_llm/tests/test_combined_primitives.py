"""Exercise the handoff without models, GPUs, or annotation services."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dev.schema_llm.explore import combined_primitives as cp


@pytest.fixture
def pack(tmp_path, monkeypatch):
    monkeypatch.setattr(
        cp,
        "CORPORA",
        {
            "d1": ("first", "identify", 2),
            "d2": ("second", "identify", 2),
            "core_spot": ("third", "spot_error", 2),
        },
    )
    exports = tmp_path / "exports"
    for corpus, (name, task, _) in cp.CORPORA.items():
        root = exports / name
        root.mkdir(parents=True)
        tasks = [
            {
                "item_id": f"item_{i}",
                "task_family": task,
                "schema": "restaurant",
                "prompt_text": "stale text is not forwarded",
                "family_id": f"family_{i}",
                "wave": i,
                "gold": "{}",
            }
            for i in range(2)
        ]
        responses, tokens = [], []
        for model in cp.MODELS:
            for i in range(2):
                responses.append(
                    {
                        "item_id": f"item_{i}",
                        "model": model,
                        "task_family": task,
                        "sample_id": 0,
                        "response_text": "saved reply",
                        "correct": True,
                    }
                )
                n_response = 2 if corpus == "d2" and i == 1 else 120
                for position in range(n_response + 4):
                    tokens.append(
                        {
                            "item_id": f"item_{i}",
                            "model": model,
                            "task_family": task,
                            "sample_id": 0,
                            "position": position,
                            "prompt_len": 4,
                            "token_id": position % 6,
                            "token": f"Ġword{position % 6}",
                            "segment": "prompt" if position < 4 else "answer",
                        }
                    )
        # Excluded core identify rows must never sneak into the pack.
        if corpus == "core_spot":
            tasks.append(
                {
                    "item_id": "excluded",
                    "task_family": "identify",
                    "schema": "airport",
                    "prompt_text": "excluded",
                    "gold": "{}",
                }
            )
        for filename, rows in (
            ("tasks", tasks),
            ("responses", responses),
            ("tokens", tokens),
            (
                "item_lookup",
                [
                    {"item_id": t["item_id"], "dataset": name, "scenario_id": t["item_id"]}
                    for t in tasks
                ],
            ),
        ):
            pq.write_table(pa.Table.from_pylist(rows), root / f"{filename}.parquet")
    dest = tmp_path / "pack"
    cp.prepare(exports, dest, Path("scripts/schema_llm/run_combined_primitives.sh"))
    manifest = cp._json(dest / "manifest.json")
    manifest.update(layers=[0, 4], k=6, meta=2)
    cp._write_json(dest / "manifest.json", manifest)
    return dest


def test_pack_preserves_tokens_namespaces_and_design(pack):
    manifest = cp.verify(pack)
    seqs = list(cp.sequences(pack))
    assert len(seqs) == 12
    assert len({s["item_id"] for s in seqs}) == 6
    assert {s["sample_id"] for s in seqs} == {0}
    assert {s["corpus"] for s in seqs} == set(cp.CORPORA)
    assert all(s["token_ids"][:6] == list(range(6)) for s in seqs)
    tasks = pq.read_table(pack / "tasks.parquet").to_pylist()
    assert len(tasks) == 6
    assert {t["source_item_id"] for t in tasks} == {"item_0", "item_1"}
    assert all(t["family_id"] and t["wave"] in (0, 1) for t in tasks)
    assert manifest["counts"]["core_spot"]["tasks"] == 2


def test_tokenizer_rejects_changed_vocabulary_and_strings(pack):
    class Tokenizer:
        def get_vocab(self):
            return {f"Ġword{i}": i for i in range(6)}

        def convert_ids_to_tokens(self, ids):
            return [f"Ġword{i}" for i in ids]

    tok = Tokenizer()
    assert cp.validate_tokenizer(tok, cp.sequences(pack, "phi4mr")) == 6
    seq = next(cp.sequences(pack))
    seq["token_ids"][0] = 999
    with pytest.raises(ValueError, match="outside tokenizer"):
        cp.validate_tokenizer(tok, [seq])
    seq["token_ids"][0] = 0
    seq["tokens"][0] = "wrong tokenizer"
    with pytest.raises(ValueError, match="strings differ"):
        cp.validate_tokenizer(tok, [seq])


def test_pack_tampering_is_rejected(pack):
    with (pack / "responses.parquet").open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        cp.verify(pack)


def _fake_captures(pack: Path, work: Path) -> None:
    manifest = cp._json(pack / "manifest.json")
    rng = np.random.default_rng(4)
    centers = rng.normal(size=(6, 8)) * 8
    for model in cp.MODELS:
        key = {
            "pack": cp._sha(pack / "manifest.json"),
            "model": model,
            "checkpoint": "fixture",
            "revision": "fixture",
            "dtype": "bfloat16",
            "layers": manifest["layers"],
        }
        config = {**key, "hidden_size": 8, "resolved_revision": "fixture"}
        cp._write_json(work / "capture_config" / f"{model}.json", config)
        for seq in cp.sequences(pack, model):
            root = cp._capture_dir(work, seq)
            root.mkdir(parents=True)
            for layer in manifest["layers"]:
                states = np.arange(seq["n_tokens"]) // 10 % 6
                a = centers[states] + rng.normal(size=(seq["n_tokens"], 8)) * 0.1
                np.save(root / f"layer_{layer}.npy", a.astype(np.float16))
            cp._write_json(
                root / "meta.json",
                {
                    "scenario_id": seq["item_id"],
                    "dataset": seq["corpus"],
                    "task": seq["task_family"],
                    "schema": seq["schema"],
                    "tokens": seq["tokens"],
                    "response_token_range": [seq["prompt_len"], seq["n_tokens"]],
                    "hidden_size": 8,
                    "sequence_sha256": seq["sequence_sha256"],
                    "capture_key": key,
                },
            )


def test_cluster_merge_export_alignment_and_resume(pack, tmp_path, monkeypatch):
    from dev.schema_llm.primitives import _cluster

    work = tmp_path / "work"
    _fake_captures(pack, work)
    calls = []

    def annotate(evidence, model, kind, context, backend):
        calls.append(kind)
        # A service stand-in only: real clustering and pooled TF-IDF merge run below.
        words = ["review", "check", "extract", "assign", "compare", "conclude"]
        return {
            e["id"]: {
                "label": f"{words[e['id']]} evidence",
                "description": f"{words[e['id']]} narrative content carefully",
            }
            for e in evidence
        }

    monkeypatch.setattr(_cluster, "_annotate", annotate)
    cp.cluster(pack, work, "fixture", "copilot")
    assert len(calls) == 24  # six sources x two layers x fine/meta
    calls.clear()
    cp.cluster(pack, work, "fixture", "copilot")
    assert calls == []  # same settings reuse saved clustering AND annotations
    cp.merge(pack, work, "none")  # real algorithm, CPU TF-IDF instead of model embeddings
    cp.export(pack, work)
    out = work / "export"
    output = pq.read_table(out / "tokens.parquet").to_pylist()
    assert len(output) == sum(s["n_tokens"] for s in cp.sequences(pack))
    assert all(r["kmeans_L0"] is None for r in output if r["segment"] == "prompt")
    short = {
        (s["model"], s["item_id"])
        for s in cp.sequences(pack)
        if s["n_tokens"] - s["prompt_len"] < 3
    }
    assert len(short) == 2
    clusters = pq.read_table(out / "clusters.parquet").to_pylist()
    mapping = {
        (r["source"], r["layer"], r["metacluster_id"]): r["merged_primitive_id"] for r in clusters
    }
    for r in output:
        if r["segment"] == "prompt" or (r["model"], r["item_id"]) in short:
            assert r["merged_primitive_L0"] is None
            continue
        source = f"{r['corpus']}/{r['model']}/{r['task_family']}"
        for layer in (0, 4):
            assert 0 <= r[f"kmeans_L{layer}"] < 6
            assert (
                mapping[(source, layer, r[f"metacluster_L{layer}"])]
                == r[f"merged_primitive_L{layer}"]
            )
    assert pq.read_table(out / "response_sequences.parquet").num_rows == 12
    assert pq.read_table(out / "fine_clusters.parquet").num_rows == 72
    assert len(cp._json(out / "manifest.json")["skipped"]) == 2
    # A damaged fine/meta mapping must be rejected rather than silently exported.
    path = work / "nodes" / "d1" / "phi4mr" / "kmeans.npz"
    with np.load(path) as a:
        state = a["stack"].copy()
    state[0, 0] = -1
    np.savez_compressed(path, stack=state)
    with pytest.raises(ValueError, match="invalid fine assignments"):
        cp.export(pack, work)


def test_capture_completion_rejects_changed_input(pack, tmp_path):
    work = tmp_path / "work"
    _fake_captures(pack, work)
    seq = next(cp.sequences(pack))
    path = cp._capture_dir(work, seq)
    key = cp._json(path / "meta.json")["capture_key"]
    assert cp._complete(path, seq, [0, 4], key)
    (path / "layer_4.npy").unlink()
    assert not cp._complete(path, seq, [0, 4], key)
    with pytest.raises(ValueError, match="different inputs"):
        cp._complete(path, {**seq, "sequence_sha256": "changed"}, [0, 4], key)


def test_incremental_joint_merge_reuses_sources(pack, tmp_path, monkeypatch):
    from dev.schema_llm.explore import incremental_primitives as inc
    from dev.schema_llm.primitives import _cluster

    monkeypatch.setattr(
        _cluster,
        "_annotate",
        lambda evidence, *a: {
            e["id"]: {
                "label": f"review {e['id']}",
                "description": f"check reasoning evidence {e['id']}",
            }
            for e in evidence
        },
    )
    work = tmp_path / "schema"
    _fake_captures(pack, work)
    cp.cluster(pack, work, "fixture", "copilot")
    inc.register(pack, work)
    snapshot = {str(p): cp._sha(p) for p in work.rglob("*") if p.is_file()}
    inc.merge_extractions([f"schema={work}"], tmp_path / "initial", "none")

    # Emulate the future importer: separate normalized TSP input, same pinned model settings.
    tsp = tmp_path / "tsp-input"
    tsp.mkdir()
    rows = []
    for s in cp.sequences(pack, corpus="d1"):
        s.update(corpus="tsp", task_family="tsp", item_id=s["item_id"].replace("d1__", "tsp__"))
        s["sequence_sha256"] = cp._identity(s)
        rows.append(s)
    pq.write_table(pa.Table.from_pylist(rows, schema=cp.SEQUENCE_SCHEMA), tsp / "sequences.parquet")
    metadata = [
        {k: s[k] for k in ("item_id", "model", "task_family", "sample_id", "corpus")} for s in rows
    ]
    pq.write_table(pa.Table.from_pylist(metadata), tsp / "responses.parquet")
    manifest = cp._json(pack / "manifest.json")
    manifest.update(
        counts={"tsp": {"tasks": 2, "responses": 4, "tokens": sum(s["n_tokens"] for s in rows)}},
        sources=[{"corpus": "tsp", "model": model, "task": "tsp"} for model in cp.MODELS],
        files={p.name: cp._sha(p) for p in tsp.glob("*.parquet")},
    )
    cp._write_json(tsp / "manifest.json", manifest)
    tsp_work = tmp_path / "tsp-work"
    _fake_captures(tsp, tsp_work)
    cp.cluster(tsp, tsp_work, "fixture", "copilot")
    inc.register(tsp, tsp_work)
    joint = tmp_path / "joint"
    inputs = [f"schema={work}", f"tsp={tsp_work}"]
    inc.merge_extractions(inputs, joint, "none", check_only=True)
    assert not joint.exists()
    inc.merge_extractions(inputs, joint, "none")
    assert len(cp._json(joint / "merge/semavglink/merge.json")["sources"]) == 8
    assert pq.read_table(joint / "export/schema/response_sequences.parquet").num_rows == 12
    assert pq.read_table(joint / "export/tsp/response_sequences.parquet").num_rows == 4
    before = pq.read_table(tmp_path / "initial/export/schema/tokens.parquet")
    after = pq.read_table(joint / "export/schema/tokens.parquet")
    stable_columns = [c for c in before.column_names if not c.startswith("merged_primitive_")]
    assert before.select(stable_columns).equals(after.select(stable_columns))
    assert snapshot == {str(p): cp._sha(p) for p in work.rglob("*") if p.is_file()}
    inc.merge_extractions(inputs, joint, "none")  # completed merge is resumable
    with pytest.raises(ValueError, match="duplicate source"):
        inc.merge_extractions([f"first={work}", f"second={work}"], tmp_path / "bad", "none")
    with pytest.raises(ValueError, match="inputs changed"):
        inc.merge_extractions([f"schema={work}"], joint, "none")
    with (tsp_work / "nodes/tsp/phi4mr/clusters.npz").open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="changed since completion"):
        inc.merge_extractions(inputs, tmp_path / "changed", "none", check_only=True)
