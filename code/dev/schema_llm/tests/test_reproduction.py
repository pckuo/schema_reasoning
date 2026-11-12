"""Release integrity, reference gates and the portable STOP adapter (no weights/API)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dev.schema_llm.explore import reproduction as rp


def test_supplied_vectors_use_four_arms_without_random(tmp_path):
    source = tmp_path / "data/vstop/supplied"
    source.mkdir(parents=True)
    unit = np.zeros(3072, dtype=np.float32)
    unit[0] = 1
    arrays = {name: unit.copy() for name in ("stop", "shuffled", "switch", "solved")}
    np.savez(source / "directions.npz", **arrays)
    rp._write_json(tmp_path / "data/vstop/reference/manifest.json", {"median_l16_state_norm": 51.9})
    report = rp.use_supplied_vectors(tmp_path, tmp_path / "out")
    assert report["passed"] and report["reference_cosine_differences_accepted"]
    assert (tmp_path / "out/vectors/directions.npz").read_bytes() == (
        source / "directions.npz"
    ).read_bytes()
    arrays["stop"][0] = 2
    np.savez(source / "directions.npz", **arrays)
    with pytest.raises(ValueError, match="invalid supplied vector stop"):
        rp.use_supplied_vectors(tmp_path)


def test_release_rejects_modified_input(tmp_path):
    p = tmp_path / "input.txt"
    p.write_text("original")
    rp._write_json(tmp_path / "MANIFEST.json", {"files": {"input.txt": rp._sha(p)}})
    rp.verify(tmp_path)
    p.write_text("modified")
    with pytest.raises(ValueError, match="missing or changed"):
        rp.verify(tmp_path)


def test_vector_gate_requires_reference_and_bitwise_repeat(tmp_path):
    reference, new, repeat = [tmp_path / n for n in ("ref", "new", "repeat")]
    keys = ["stop", "shuffled", "random", "switch", "solved"]
    manifest = {
        "model": "model",
        "revision": "revision",
        "n_heads": 35,
        "source_layers": [0],
        "rows": 100,
        "items": 2,
        "n_stop_positions": 4,
        "n_baseline_positions": 20,
        "seed": 20260919,
        "mask_counts": {"solved": 10},
        "median_l16_state_norm": 51.9,
        "directions": {k: {"raw_norm": 1.0, "head_space_projection_sd": 0.1} for k in keys},
        "residual_cosines": {f"{a}|{b}": 0.0 for i, a in enumerate(keys) for b in keys[i + 1 :]},
        "head_space_cosines": {"stop|random": 0.0},
    }
    arrays = {k: np.eye(5, 3072, dtype=np.float32)[i] for i, k in enumerate(keys)}
    for path in (reference, new, repeat):
        rp._write_json(path / "manifest.json", manifest)
        np.savez(path / "directions.npz", **arrays)
    tolerance = {"cosine_atol": 0.001, "scale_atol": 0.01, "median_atol": 0.05}
    assert rp.check_vectors(reference, new, repeat, tolerance)["passed"]
    changed = json.loads(json.dumps(manifest))
    changed["residual_cosines"]["stop|random"] = 0.01
    rp._write_json(new / "manifest.json", changed)
    with pytest.raises(ValueError, match="reference check failed"):
        rp.check_vectors(reference, new, repeat, tolerance)
    rp._write_json(new / "manifest.json", manifest)
    arrays["random"] = -arrays["random"]
    np.savez(repeat / "directions.npz", **arrays)
    with pytest.raises(ValueError, match="bitwise identical"):
        rp.check_vectors(reference, new, repeat, tolerance)


def test_frame_gate_rejects_different_rows(tmp_path):
    frame = pd.DataFrame(
        {
            "item": ["i"],
            "t": [1],
            "z0": [0],
            "last3": [True],
            "ref_stop": [False],
            "fold": [0],
            "row_end": [10],
        }
    )
    parity = {
        "dropped": 0,
        "total": 1,
        "last3_runs": 1,
        "ref_stop_runs": 0,
        "norm_cut_matches": True,
    }
    for path in (tmp_path / "ref", tmp_path / "new"):
        rp._write_json(path / "parity.json", parity)
        frame.to_parquet(path / "frame.parquet")
    assert rp.check_frame(tmp_path / "ref", tmp_path / "new")["passed"]
    frame["row_end"] = [11]
    frame.to_parquet(tmp_path / "new/frame.parquet")
    with pytest.raises(ValueError, match="frame differs"):
        rp.check_frame(tmp_path / "ref", tmp_path / "new")


def test_selection_adapter_is_not_recursive(tmp_path, monkeypatch):
    from dev.schema_llm.explore.stop_steering import capture, vectors

    # Restore bindings after the adapter test so other tests retain the original defaults.
    monkeypatch.setattr(capture, "selected_heads", capture.selected_heads)
    monkeypatch.setattr(vectors, "selected_heads", vectors.selected_heads)
    selection = {"heads": [[i % 32, i // 32] for i in range(35)]}
    rp._write_json(tmp_path / "data/vstop/reference/selection.json", selection)
    rp.configure_selection(tmp_path)
    assert len(capture.selected_heads()[0]) == 35
    assert vectors.selected_heads()[0] == capture.selected_heads()[0]


def test_schema_item_resume_preserves_plan_and_reuses_model(tmp_path, monkeypatch):
    from dev.schema_llm.explore.stop_steering import pilot, runframe

    targets = pd.DataFrame({"item": ["first", "second"]})
    loads, generated = [], []
    monkeypatch.setattr(runframe, "build", lambda: {})
    monkeypatch.setattr(pilot, "population", lambda *a, **kw: targets)
    monkeypatch.setattr(pilot, "load_model", lambda *a, **kw: loads.append(1) or object())
    monkeypatch.setattr(pilot, "summarize", lambda frame: {"rows": len(frame)})

    def fake_run(vectors, out, **options):
        pilot.load_model("fixture")
        selected = pilot.population({})
        assert len(selected) == 1
        item = selected.item.iloc[0]
        generated.append(item)
        rows = [{"item": item, "arm": "baseline", "dose": 0.0, "draw": 0}]
        rows += [
            {"item": item, "arm": a, "dose": d, "draw": 0}
            for a in options["arms"]
            for d in options["doses"]
        ]
        out.mkdir(parents=True)
        pd.DataFrame(rows).to_parquet(out / "pilot.parquet")

    monkeypatch.setattr(pilot, "run", fake_run)
    config = {"schema": {"name": "main", "arms": ["stop", "shuffled"], "doses": [0.5]}}
    rp.run_schema(tmp_path, tmp_path / "out", config, "cpu")
    assert loads == [1] and generated == ["first", "second"]
    rp.run_schema(tmp_path, tmp_path / "out", config, "cpu")
    assert loads == [1] and generated == ["first", "second"]
    frame = pd.read_parquet(tmp_path / "out/schema/main/pilot.parquet")
    assert len(frame) == 6 and frame.groupby("item").size().eq(3).all()


def test_stop_tokenizer_counts_ids_in_encoding_mapping(tmp_path, monkeypatch):
    from transformers import AutoTokenizer

    from dev.schema_llm.explore.tsp_stop_steering import run

    class Tokenizer:
        def get_vocab(self):
            return {"zero": 0, "eos": 200020, "end": 199999}

        def convert_ids_to_tokens(self, ids):
            return ["zero"] * len(ids)

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["return_dict"] is True
            return {"input_ids": [0, 0, 0], "attention_mask": [1, 1, 1]}

    tokenizer = Tokenizer()
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **kw: tokenizer)
    monkeypatch.setattr(run, "find_think_end", lambda *a: 1)
    core = tmp_path / "data/vstop/core"
    tsp = tmp_path / "data/vstop/tsp"
    core.mkdir(parents=True)
    tsp.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "item_id": str(i),
                    "model": "phi4mr",
                    "position": 0,
                    "token_id": 0,
                    "token": "zero",
                }
                for i in range(216)
            ]
        ),
        core / "tokens.parquet",
    )
    rp._write_json(tsp / "tsp_tasks.json", [{"uid": 70, "prompt": "fixture"}])
    (tsp / "phi4_mini_instruct_chat_template.jinja").write_text("fixture")
    pq.write_table(
        pa.Table.from_pylist([{"uid": 70, "n_prompt": 3, "reasoning_tokens": 1, "token_ids": [0]}]),
        tsp / "baselines.parquet",
    )
    assert rp.validate_stop_tokenizer(tmp_path) is tokenizer


@pytest.mark.realdata
def test_frozen_release_population():
    root = Path(__file__).resolve().parents[4]
    if not (root / "data/vstop/core/tokens.parquet").exists():
        pytest.skip("standalone release data absent")
    rp.configure(root)
    result = rp.check_stop_inputs(root)
    assert result == {
        "schema_intervention_items": 17,
        "reference_capture_items": 166,
        "tsp_tasks": 6,
        "tsp_baselines": 18,
    }
