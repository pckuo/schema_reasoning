"""Generation settings, exact-token packing, interruption recovery and input validation."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow.parquet as pq
import pytest
import torch

from dev.schema_llm.core import model as core
from dev.schema_llm.explore import combined_primitives as cp
from dev.schema_llm.explore import tsp_generate as gen


class Tokenizer:
    eos_token_id = 9

    def convert_ids_to_tokens(self, ids):
        vocab = {
            0: "prompt",
            1: "<think>",
            2: "reason",
            3: "</",
            4: "think>",
            5: "answer",
            9: "<eos>",
        }
        return [vocab[i] for i in ids]

    def decode(self, ids, **kwargs):
        return "".join(self.convert_ids_to_tokens(ids))

    def apply_chat_template(self, messages, **kwargs):
        assert len(messages) == 1 and messages[0]["role"] == "user"
        return {"input_ids": [0]}


@pytest.fixture
def root(tmp_path, monkeypatch):
    project = Path.cwd()
    root = tmp_path / "repo"
    for name in cp.CODE_FILES:
        path = root / "code/dev/schema_llm" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((project / "code/dev/schema_llm" / name).read_bytes())
    cp._write_json(root / "MANIFEST.json", {"files": {}})
    template = root / "data/tsp_generation/phi4_mini_instruct_chat_template.jinja"
    template.parent.mkdir(parents=True)
    template.write_text("template")
    tasks = root / "tasks.jsonl"
    tasks.write_text("\n".join(json.dumps({"uid": i, "prompt": f"TSP {i}"}) for i in [10, 20]))
    monkeypatch.chdir(root)
    return root


def test_generation_resume_and_direct_extraction_input(root, tmp_path, monkeypatch):
    loads, calls, seeds = [], [], []
    tok = Tokenizer()

    def load(checkpoint, **kw):
        loads.append((checkpoint, kw))
        return SimpleNamespace(
            tokenizer=tok,
            model=SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[8, 9])),
        )

    def generate(lm, prompt, **kw):
        calls.append((prompt, kw))
        return torch.tensor([0, 1, 2, 3, 4, 5, 9]), 1, "unused decoded text"

    monkeypatch.setattr(core, "load_model", load)
    monkeypatch.setattr(core, "generate_reply", generate)
    monkeypatch.setattr(core, "set_seed", seeds.append)
    tasks, out = root / "tasks.jsonl", tmp_path / "generated"
    pack = gen.run(root, tasks, out, 32768, "cuda:0")
    assert len(loads) == 2 and len(calls) == len(seeds) == 4
    assert loads[0][1]["chat_template"] == "template"
    assert loads[1][1]["chat_template"] is None
    for _, kw in calls:
        assert kw["system_message"] is None and kw["do_sample"]
        assert (kw["temperature"], kw["top_p"], kw["top_k"], kw["max_new_tokens"]) == (
            0.8,
            0.95,
            50,
            32768,
        )
    seqs = list(cp.sequences(pack))
    assert len(seqs) == 4
    assert seqs[0]["token_ids"] == [0, 1, 2, 3, 4, 5, 9]
    assert seqs[0]["segments"] == ["prompt", "think", "think", "think", "think", "answer", "answer"]
    assert all(
        r["terminated"] and not r["hit_cap"]
        for r in pq.read_table(pack / "responses.parquet").to_pylist()
    )
    assert len(list(cp._sources(cp.verify(pack)))) == 2
    gen.run(root, tasks, out, 32768, "cuda:0")
    assert len(calls) == 4 and len(loads) == 2  # no model load on complete resume
    gen.shard_path(out, "qwen34b", 20).unlink()
    gen.run(root, tasks, out, 32768, "cuda:0")
    assert len(calls) == 5 and len(loads) == 3  # only the missing response is generated
    with pytest.raises(ValueError, match="settings changed"):
        gen.run(root, tasks, out, 8192, "cuda:0")
    path = gen.shard_path(out, "phi4mr", 10)
    row = cp._json(path)
    row["sequence"]["token_ids"][1] = 9
    cp._write_json(path, row)
    with pytest.raises(ValueError, match="fingerprint changed"):
        gen.run(root, tasks, out, 32768, "cuda:0")


def test_task_validation_and_stable_seeds(tmp_path):
    path = tmp_path / "tasks.jsonl"
    path.write_text('{"uid": 1, "prompt": "problem"}\n{"uid": 1, "prompt": "duplicate"}\n')
    with pytest.raises(ValueError, match="duplicate"):
        gen.load_tasks(path)
    assert gen.seed_for("phi4mr", 1, 0) == gen.seed_for("phi4mr", 1, 0)
    assert gen.seed_for("phi4mr", 1, 0) != gen.seed_for("qwen34b", 1, 0)


def test_smoke_pack_tracks_cap_and_unclosed_thinking(root, tmp_path, monkeypatch):
    tok = Tokenizer()
    monkeypatch.setattr(
        core,
        "load_model",
        lambda *a, **kw: SimpleNamespace(
            tokenizer=tok, model=SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=9))
        ),
    )
    monkeypatch.setattr(core, "generate_reply", lambda *a, **kw: (torch.tensor([0, 1, 2]), 1, ""))
    monkeypatch.setattr(core, "set_seed", lambda seed: None)
    pack = gen.run(root, root / "tasks.jsonl", tmp_path / "smoke", 2, "cuda:0", smoke=True)
    responses = pq.read_table(pack / "responses.parquet").to_pylist()
    assert all(r["hit_cap"] and not r["closed_think"] and not r["terminated"] for r in responses)
    assert all(s["segments"] == ["prompt", "think", "think"] for s in cp.sequences(pack))
