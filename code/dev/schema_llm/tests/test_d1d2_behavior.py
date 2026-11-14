"""Offline collection, cached grading, exclusions and output identity for the handoff."""

from __future__ import annotations

import csv
import json
import signal
import socket
from contextlib import nullcontext
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from dev.schema_llm.explore import d1d2_behavior as b


def fixture_data(tmp_path):
    data = {}
    for corpus in ("d1", "d2"):
        items, probes, meta = [], [], {}
        for i, wave in enumerate((3, 4)):
            sid = f"{corpus}-{wave}"
            items.append({"scenario_id": sid, "wave": wave, "prompt": "story",
                          "schema": "restaurant", "scoring_exclude": corpus == "d2" and i == 1})
            probes.append({"scenario_id": sid, "task": "identify", "probe_prompt": sid,
                           "gold": {"schema": "restaurant", "roles": [
                               {"pid": "p1", "value": "Ada", "role_ids": ["PATRON"],
                                "roles": ["patron"], "n_roles": 1,
                                "accept_per_role": [["customer"]]}]}})
            meta[sid] = {"task_id": f"{corpus}__item_{i}", "source": f"{corpus}-wave{wave}"}
        data[corpus] = {"items": items, "probes": probes, "meta": meta}
    return b.prepare(tmp_path, tmp_path / "out", data, {"frozen": True}, False)


def fake_judge(prompt, *, model):
    return json.dumps({"schema_granularity": "match", "people": [
        {"id": 0, "parts": ["correct"]}], "extras": []})


def test_interruption_resume_grading_and_export(tmp_path):
    data = fixture_data(tmp_path)
    key = "phi-4"
    calls = []

    def interrupted(prompt, *, model):
        calls.append(prompt)
        if len(calls) == 2:
            raise RuntimeError("interrupted")
        return "restaurant: Ada = customer"

    with pytest.raises(RuntimeError, match="interrupted"):
        b.collect_one(data["d1"], key, b.MODELS[key], interrupted, 1)
    resumed = []

    def response(prompt, *, model):
        resumed.append(prompt)
        return "restaurant: Ada = customer"

    for d in data.values():
        b.collect_one(d, key, b.MODELS[key], response, 1)
        b.grade_one(d, key, b.DEFAULT_JUDGE, 1, fake_judge)
    assert resumed == ["d1-4", "d2-3", "d2-4"]

    def cached_only(prompt, *, model):
        # The original judge always performs a model/auth preflight, even when cached.
        assert "<response>" not in prompt
        return '{"ok": true}'

    for d in data.values():
        b.grade_one(d, key, b.DEFAULT_JUDGE, 1, cached_only)
    out = tmp_path / "out"
    b.export_results(data, [key], out, b.DEFAULT_JUDGE)
    with (out / "response_outcomes.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4
    assert len({(r["model"], r["task_id"], r["sample_id"]) for r in rows}) == 4
    excluded = [r for r in rows if r["status"] == "scoring_excluded"]
    assert len(excluded) == 1 and excluded[0]["correct"] == ""
    assert all(r["correct"] == "True" for r in rows if r not in excluded)
    text = pq.read_table(out / "response_outcomes_text.parquet").to_pylist()
    assert len(text) == 4 and all(r["response"] for r in text)
    assert all(r["think_closed"] is None for r in text)  # plain Phi-4 has no reasoning block
    assert len(b.read_rows(data["d2"]["run"] / f"scores.{key}.jsonl")) == 1


def test_reject_changes_duplicates_and_incomplete_exports(tmp_path):
    data = fixture_data(tmp_path)
    with pytest.raises(ValueError, match="settings changed"):
        b.bind_json(tmp_path / "out/experiment.json", {"frozen": False})
    with pytest.raises(ValueError, match="incomplete responses"):
        b.export_results(data, ["phi-4"], tmp_path / "out", b.DEFAULT_JUDGE)
    d = data["d1"]
    b.collect_one(d, "phi-4", b.MODELS["phi-4"], lambda *a, **kw: "answer", 1)
    b.grade_one(d, "phi-4", b.DEFAULT_JUDGE, 1, fake_judge)
    path = d["run"] / "responses.phi-4.jsonl"
    rows = b.read_rows(path)
    rows[0]["response"] = "changed"
    b.write_rows(path, rows)
    with pytest.raises(ValueError, match="settings changed"):
        b.grade_one(d, "phi-4", b.DEFAULT_JUDGE, 1, fake_judge)
    b.write_rows(path, [*rows, rows[0]])
    with pytest.raises(ValueError, match="duplicate"):
        b.collect_one(d, "phi-4", b.MODELS["phi-4"], lambda *a, **kw: "answer", 1)


def test_transport_failure_is_not_an_incorrect_answer(tmp_path):
    data = fixture_data(tmp_path)
    d = data["d1"]
    b.collect_one(d, "phi-4", b.MODELS["phi-4"], lambda *a, **kw: "answer", 1)
    with pytest.raises(RuntimeError, match="transport failed"):
        b.grade_one(d, "phi-4", b.DEFAULT_JUDGE, 1,
                    lambda *a, **kw: "[trapi RateLimitError after 6 retries]")
    assert not (d["run"] / "scores.phi-4.jsonl").exists()


def test_smoke_keeps_both_waves_and_roster(tmp_path):
    full = fixture_data(tmp_path)
    # Replace the excluded toy wave-4 item so smoke has a scorable item in each wave.
    for d in full.values():
        for item in d["items"]:
            item["scoring_exclude"] = False
    smoke = b.prepare(tmp_path, tmp_path / "smoke", full, {"smoke": True}, True)
    assert all({r["wave"] for r in d["items"]} == {3, 4} for d in smoke.values())
    assert len(b.MODELS) == 8
    assert sum(s["backend"] == "trapi" for s in b.MODELS.values()) == 2
    assert b.MODELS["deepseek-r1-distill-qwen-14b"]["top_k"] == 50
    assert not b.MODELS["phi-4"].get("sample", False)


def test_release_prompt_parity_when_available():
    root = (Path.cwd() if Path("data/d1d2_behavior").is_dir() else
            Path("results/schema_llm/export/20261112-collaborator-repo"))
    if not root.exists():
        pytest.skip("release fixture lives outside source checkout")
    data, mapping = b.load_inputs(root)
    assert [len(data[c]["items"]) for c in ("d1", "d2")] == [300, 297]
    assert len(mapping) == 597


def test_driver_routes_local_and_hosted_and_exports(tmp_path, monkeypatch):
    from dev.schema_llm.subjects import collect

    data = fixture_data(tmp_path)
    for d in data.values():
        for item in d["items"]:
            item["scoring_exclude"] = False
    root = tmp_path / "release"
    files = ["data/d1d2_behavior/manifest.json", "code/dev/schema_llm/explore/d1d2_behavior.py",
             "code/dev/schema_llm/subjects/collect.py", "code/dev/schema_llm/subjects/judge.py",
             "code/dev/schema_llm/core/llm.py", "code/dev/schema_llm/core/schemas/airport.yaml",
             "code/dev/schema_llm/core/schemas/restaurant.yaml"]
    for filename in files:
        path = root / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}" if path.suffix == ".json" else "fixture")
    monkeypatch.setattr(b, "verify_release", lambda root: None)
    monkeypatch.setattr(b, "load_inputs", lambda root: (data, {}))
    monkeypatch.setattr(b.importlib.metadata, "version", lambda name: "fixture")
    monkeypatch.setattr(b, "server", lambda *args: nullcontext("http://fixture/v1"))
    calls = []

    def responder(model, **kwargs):
        calls.append((model, kwargs))
        return lambda *a, **kw: "restaurant: Ada = customer"

    monkeypatch.setattr(collect, "make_vllm_responder", responder)
    monkeypatch.setattr(collect, "make_trapi_responder", responder)
    grade = b.grade_one
    monkeypatch.setattr(b, "grade_one", lambda d, k, j, w: grade(d, k, j, w, fake_judge))
    out = tmp_path / "driver"
    b.main(["--root", str(root), "--out", str(out), "--smoke"])
    assert calls[0][1]["timeout"] is None and calls[0][1]["max_tokens"] == 8192
    assert all(kwargs == {"max_completion_tokens": 8192} for _, kwargs in calls[-2:])
    assert len(calls) == 8
    assert calls[1][1]["system"] == "phi-reasoning"
    assert calls[3][1]["enable_thinking"] is True
    assert calls[4][1]["temperature"] == 0.6 and calls[4][1]["top_k"] == 50
    with (out / "response_outcomes.csv").open() as f:
        assert len(list(csv.DictReader(f))) == 32
    # A complete rerun must not start a server or create a hosted subject client.
    calls.clear()
    b.main(["--root", str(root), "--out", str(out), "--smoke"])
    assert calls == []


def test_existing_records_complete_tenmodel_csv_without_model_calls(tmp_path):
    data = fixture_data(tmp_path)
    for key, spec in b.MODELS.items():
        for d in data.values():
            b.collect_one(d, key, spec, lambda *a, **kw: "answer", 1)
            b.grade_one(d, key, b.DEFAULT_JUDGE, 1, fake_judge)
    root = tmp_path / "release"
    for corpus, d in data.items():
        dest = root / "data/d1d2_behavior/existing" / corpus
        dest.mkdir(parents=True)
        for key, spec in b.EXISTING_MODELS.items():
            responses = [{"scenario_id": p["scenario_id"], "task": p["task"],
                          "model": spec["model"], "response": "<think>trace</think>answer"}
                         for p in d["probes"]]
            # Include a score for the unresolved toy item, as the historical sources did.
            # The importer must discard that score and export a null outcome.
            scores = [{"scenario_id": p["scenario_id"], "task": p["task"],
                       "judge_model": b.DEFAULT_JUDGE, "parse_ok": True, "judge_error": False,
                       "correct": True, "schema_correct": True, "role_f1": 1.0,
                       "role_accuracy": 1.0, "role_precision": 1.0} for p in d["probes"]]
            b.write_rows(dest / f"responses.{key}.jsonl", responses)
            b.write_rows(dest / f"scores.{key}.jsonl", scores)
    b.import_existing(root, data, b.DEFAULT_JUDGE)
    out = tmp_path / "out"
    b.export_results(data, list(b.ALL_MODELS), out, b.DEFAULT_JUDGE)
    with (out / "response_outcomes.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 40 and len({r["model"] for r in rows}) == 10
    assert {r["model_display_name"] for r in rows} == set(b.DISPLAY_NAMES.values())
    assert sum(r["status"] == "scoring_excluded" for r in rows) == 10
    assert all(r["correct"] == "" for r in rows if r["status"] == "scoring_excluded")
    manifest = json.loads((out / "export_manifest.json").read_text())
    assert manifest["complete"] and len(manifest["coverage"]) == 10
    assert all(c == {"responses": 4, "scored": 3, "excluded": 1}
               for c in manifest["coverage"].values())
    assert all(b.sha(out / name) == digest for name, digest in manifest["artifacts"].items())
    with pytest.raises(ValueError, match="original Terra judge"):
        b.import_existing(root, data, "another-judge")
    mapping = b.read_rows(out / "response_outcomes_task_id_map.jsonl")
    assert len(mapping) == 4 and {r["corpus"] for r in mapping} == {"d1", "d2"}


def test_grading_resumes_verdicts_after_midrun_transport_failure(tmp_path):
    d = fixture_data(tmp_path)["d1"]
    b.collect_one(d, "phi-4", b.MODELS["phi-4"], lambda *a, **kw: "answer", 1)
    calls = []

    def interrupted(prompt, *, model):
        if "<response>" in prompt:
            calls.append(prompt)
            if len(calls) == 2:
                return "[trapi RateLimitError after 6 retries]"
        return fake_judge(prompt, model=model)

    with pytest.raises(RuntimeError, match="transport failed"):
        b.grade_one(d, "phi-4", b.DEFAULT_JUDGE, 1, interrupted)
    assert len(list((d["run"] / "judge_raw_phi-4").glob("*.txt"))) == 1
    remaining = []

    def resumed(prompt, *, model):
        if "<response>" in prompt:
            remaining.append(prompt)
        return fake_judge(prompt, model=model)

    b.grade_one(d, "phi-4", b.DEFAULT_JUDGE, 1, resumed)
    assert len(remaining) == 1


def test_changed_judge_is_rejected_before_loading_or_collecting(tmp_path):
    with pytest.raises(SystemExit) as exc:
        b.main(["--root", str(tmp_path), "--include-existing", "--judge-model", "other"])
    assert exc.value.code == 2


@pytest.mark.parametrize("bad_version", [False, True])
def test_owned_server_pins_revision_and_cleans_up(tmp_path, monkeypatch, bad_version):
    key = "phi-4"
    b.write_json(tmp_path / f"revision.{key}.json", {
        "model": b.MODELS[key]["model"], "revision": "a" * 40})
    bound, started, killed = [], [], []

    class Socket:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def bind(self, address):
            bound.append(address)

    class Process:
        pid = 12345

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

    def start(command, **kwargs):
        started.append((command, kwargs))
        return Process()

    monkeypatch.setattr(socket, "socket", lambda *a, **kw: Socket())
    monkeypatch.setattr(b.urllib.request, "urlopen", lambda *a, **kw: nullcontext())
    monkeypatch.setattr(b.subprocess, "Popen", start)
    monkeypatch.setattr(b.subprocess, "check_output", lambda *a, **kw: json.dumps({
        "vllm": "wrong" if bad_version else "0.25.1", "torch": "fixture",
        "transformers": "fixture"}))
    monkeypatch.setattr(b.os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    monkeypatch.delenv("VLLM_VERSION", raising=False)
    if bad_version:
        with pytest.raises(ValueError, match="another vLLM version"):
            with b.server(tmp_path, tmp_path, key, b.MODELS[key], 8400, 12288):
                pytest.fail("a mismatched server must not yield a client endpoint")
    else:
        with b.server(tmp_path, tmp_path, key, b.MODELS[key], 8400, 12288) as url:
            assert url == "http://127.0.0.1:8400/v1"
    assert bound == [("127.0.0.1", 8400)]
    assert started[0][1]["start_new_session"] is True
    env = started[0][1]["env"]
    assert env["VLLM_MAX_LEN"] == "12288" and "--revision " + "a" * 40 in env["VLLM_EXTRA_ARGS"]
    assert killed == [(12345, signal.SIGTERM)]
