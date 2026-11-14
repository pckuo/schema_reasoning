"""Collect and grade the eight additional D1/D2 subjects in the standalone release.

Uses the original collect/judge implementations and frozen user prompts/gold. The two
generation waves are always pooled within each design. No core prompts are selected here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path

# Preserve the original panel's decoding, including greedy Phi-4 and hosted defaults.
MODELS = {
    "phi-4": {"model": "microsoft/phi-4", "backend": "vllm"},
    "phi-4-reasoning": {
        "model": "microsoft/Phi-4-reasoning", "backend": "vllm", "system": "phi-reasoning",
        "sample": True, "temperature": 0.8, "top_p": 0.95, "top_k": 50,
    },
    "phi-4-reasoning-plus": {
        "model": "microsoft/Phi-4-reasoning-plus", "backend": "vllm",
        "system": "phi-reasoning", "sample": True,
        "temperature": 0.8, "top_p": 0.95, "top_k": 50,
    },
    "Qwen3-14B-think": {
        "model": "Qwen/Qwen3-14B", "backend": "vllm", "think": True,
        "sample": True, "temperature": 0.8, "top_p": 0.95, "top_k": 50,
    },
    "deepseek-r1-distill-qwen-14b": {
        "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", "backend": "vllm",
        # The historical launcher omitted --top-k, whose collector CLI default is 50.
        "sample": True, "temperature": 0.6, "top_p": 0.95, "top_k": 50,
    },
    "Qwen3.8-27B-think": {
        "model": "Qwen/Qwen3.8-27B", "backend": "vllm", "think": True,
        "sample": True, "temperature": 0.8, "top_p": 0.95, "top_k": 50,
    },
    "deepseek-v4-pro": {"model": "DeepSeek-V4-Pro_2026-04-23", "backend": "trapi"},
    "gpt-5.5": {"model": "gpt-5.5_2026-04-24", "backend": "trapi"},
}
EXISTING_MODELS = {
    "phi-4-mini-reasoning": {"model": "microsoft/Phi-4-mini-reasoning", "reasoning": True},
    "Qwen3-4B-think": {"model": "Qwen/Qwen3-4B", "reasoning": True},
}
ALL_MODELS = {**MODELS, **EXISTING_MODELS}
DISPLAY_NAMES = {
    "phi-4": "Phi-4", "phi-4-reasoning": "Phi-4-reasoning",
    "phi-4-reasoning-plus": "Phi-4-reasoning+", "Qwen3-14B-think": "Qwen3-14B",
    "deepseek-r1-distill-qwen-14b": "DeepSeek-R1-Distill-Qwen-14B",
    "Qwen3.8-27B-think": "Qwen3.8-27B", "deepseek-v4-pro": "DeepSeek-V4-Pro",
    "gpt-5.5": "GPT-5.5", "phi-4-mini-reasoning": "Phi-4-mini-reasoning",
    "Qwen3-4B-think": "Qwen3-4B",
}
DEFAULT_JUDGE = "gpt-5.6-terra_2026-07-09"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict]:
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def write_json(path: Path, value) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def write_rows(path: Path, rows: list[dict]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    temp.replace(path)


def bind_json(path: Path, value) -> None:
    """Reject resuming into a directory with a different experiment definition."""
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"settings changed; use a new output directory: {path}")
    else:
        write_json(path, value)


def verify_release(root: Path) -> None:
    manifest = json.loads((root / "MANIFEST.json").read_text())
    for name, expected in manifest["files"].items():
        if sha(root / name) != expected:
            raise ValueError(f"release file changed: {name}")


def indexed(rows: list[dict]) -> dict[tuple[str, str], dict]:
    out = {(r["scenario_id"], r["task"]): r for r in rows}
    if len(out) != len(rows):
        raise ValueError("duplicate (scenario_id, task) records")
    return out


def load_inputs(root: Path) -> tuple[dict, dict]:
    """Validate hashes and prompt/gold identity against the existing extraction pack."""
    import pyarrow.parquet as pq

    base = root / "data/d1d2_behavior"
    manifest = json.loads((base / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if sha(base / name) != expected:
            raise ValueError(f"behavior input changed: {name}")
    frozen = {r["item_id"]: r for r in pq.read_table(
        root / "data/primitives/tasks.parquet").to_pylist()}
    frozen_responses = {(r["model"], r["item_id"]): r for r in pq.read_table(
        root / "data/primitives/responses.parquet").to_pylist()}
    mapping = read_rows(base / "task_map.jsonl")
    if len(mapping) != 597 or len({r["task_id"] for r in mapping}) != 597:
        raise ValueError("expected 597 distinct D1/D2 task IDs")
    by_task = {r["task_id"]: r for r in mapping}
    data = {}
    for corpus, count in (("d1", 300), ("d2", 297)):
        items = read_rows(base / corpus / "items.jsonl")
        probes = read_rows(base / corpus / "probes.jsonl")
        by_sid = {r["scenario_id"]: r for r in items}
        probe_by = indexed(probes)
        meta = {r["scenario_id"]: r for r in mapping if r["corpus"] == corpus}
        if (len(items) != count or len(probes) != count or len(by_sid) != count
                or set(by_sid) != set(meta)):
            raise ValueError(f"incomplete {corpus} union")
        for sid, item in by_sid.items():
            p = probe_by[(sid, "identify")]
            old = frozen[meta[sid]["task_id"]]
            if (p["probe_prompt"] != old["prompt_text"]
                    or p["gold"] != json.loads(old["gold"])
                    or item["prompt"] != old["passage_text"]
                    or bool(item.get("scoring_exclude")) != bool(old["scoring_exclude"])
                    or item["wave"] != old["wave"]):
                raise ValueError(f"frozen prompt/gold/design mismatch: {corpus}/{sid}")
        if {r["wave"] for r in items} != {3, 4}:
            raise ValueError(f"both waves required: {corpus}")
        for key, alias in (("phi-4-mini-reasoning", "phi4mr"), ("Qwen3-4B-think", "qwen34b")):
            existing = base / "existing" / corpus
            responses = indexed(read_rows(existing / f"responses.{key}.jsonl"))
            scores = indexed(read_rows(existing / f"scores.{key}.jsonl"))
            if set(responses) != set(probe_by) or set(scores) != set(probe_by):
                raise ValueError(f"incomplete existing records: {corpus}/{key}")
            for pair, resp in responses.items():
                old = frozen_responses[(alias, meta[pair[0]]["task_id"])]
                sc = scores[pair]
                if (resp["response"] != old["response_text"]
                        or resp["model"] != EXISTING_MODELS[key]["model"]
                        or sc["correct"] != old["correct"]
                        or sc["role_accuracy"] != old["judge_role_accuracy"]
                        or sc["judge_model"] != DEFAULT_JUDGE
                        or not sc.get("parse_ok") or sc.get("judge_error")):
                    raise ValueError(f"existing response/score mismatch: {corpus}/{key}/{pair}")
        data[corpus] = {"items": items, "probes": probes, "meta": meta}
    return data, by_task


def prepare(root: Path, out: Path, data: dict, settings: dict, smoke: bool) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    bind_json(out / "experiment.json", settings)
    selected = {}
    for corpus, d in data.items():
        items = d["items"]
        if smoke:
            # One scorable prompt from EACH wave, so smoke exercises both input sources.
            items = [next(r for r in items if r["wave"] == wave
                          and not r.get("scoring_exclude")) for wave in (3, 4)]
        ids = {r["scenario_id"] for r in items}
        probes = [r for r in d["probes"] if r["scenario_id"] in ids]
        run = out / "runs" / corpus
        run.mkdir(parents=True, exist_ok=True)
        for name, rows in (("items", items), ("probes", probes)):
            path = run / f"{name}.jsonl"
            if path.exists() and read_rows(path) != rows:
                raise ValueError(f"prepared input changed: {path}")
            write_rows(path, rows)
        selected[corpus] = {**d, "items": items, "probes": probes, "run": run}
    write_rows(out / "response_outcomes_task_id_map.jsonl", [
        {"task": p["task"], "corpus": corpus, "scenario_id": p["scenario_id"],
         **d["meta"][p["scenario_id"]]} for corpus, d in selected.items() for p in d["probes"]])
    return selected


def import_existing(root: Path, data: dict, judge_model: str) -> None:
    """Import frozen Phi/Qwen records for an offline ten-model export; never call a model."""
    base = root / "data/d1d2_behavior/existing"
    for corpus, d in data.items():
        wanted = set(indexed(d["probes"]))
        excluded = {r["scenario_id"] for r in d["items"] if r.get("scoring_exclude")}
        for key, spec in EXISTING_MODELS.items():
            responses = [r for r in read_rows(base / corpus / f"responses.{key}.jsonl")
                         if (r["scenario_id"], r["task"]) in wanted]
            scores = [r for r in read_rows(base / corpus / f"scores.{key}.jsonl")
                      if (r["scenario_id"], r["task"]) in wanted
                      and r["scenario_id"] not in excluded]
            if set(indexed(responses)) != wanted or any(
                    r["model"] != spec["model"] for r in responses):
                raise ValueError(f"incomplete/mismatched existing responses: {corpus}/{key}")
            if (set(indexed(scores)) != {k for k in wanted if k[0] not in excluded}
                    or any(r["judge_model"] != judge_model or not r.get("parse_ok")
                           or r.get("judge_error") for r in scores)):
                raise ValueError("existing Phi/Qwen grades require their original Terra judge; "
                                 "reconcile judges before combining results")
            for stem, rows in (("responses", responses), ("scores", scores)):
                path = d["run"] / f"{stem}.{key}.jsonl"
                if path.exists() and read_rows(path) != rows:
                    raise ValueError(f"existing record changed: {path}")
                write_rows(path, rows)
            bind_json(d["run"] / f"grading_inputs.{key}.json", {
                "responses_sha256": sha(d["run"] / f"responses.{key}.jsonl"),
                "judge_model": judge_model,
            })


def records(path: Path, probes: list[dict]) -> dict:
    rows = indexed(read_rows(path)) if path.exists() else {}
    if not set(rows) <= set(indexed(probes)):
        raise ValueError(f"unexpected response/score keys in {path}")
    return rows


def collect_one(d: dict, key: str, spec: dict, call, concurrency: int) -> None:
    from ..subjects.collect import run_collect

    path = d["run"] / f"responses.{key}.jsonl"
    old = records(path, d["probes"])
    if any(r.get("model") != spec["model"] for r in old.values()):
        raise ValueError(f"subject model mismatch: {path}")
    run_collect(d["run"], spec["model"], ["identify"], call=call, tag=key,
                concurrency=concurrency)
    if set(records(path, d["probes"])) != set(indexed(d["probes"])):
        raise ValueError(f"incomplete collection: {path}")


def grade_one(d: dict, key: str, judge_model: str, workers: int, call=None) -> None:
    from ..core.llm import _cli_error, run_trapi
    from ..subjects.judge import run_judge

    responses = records(d["run"] / f"responses.{key}.jsonl", d["probes"])
    if set(responses) != set(indexed(d["probes"])):
        raise ValueError(f"collect all responses before grading {key}")
    if any(r.get("model") != MODELS[key]["model"] for r in responses.values()):
        raise ValueError(f"subject model mismatch: {key}")
    bind_json(d["run"] / f"grading_inputs.{key}.json", {
        "responses_sha256": sha(d["run"] / f"responses.{key}.jsonl"),
        "judge_model": judge_model,
    })
    items = {r["scenario_id"]: r for r in d["items"]}
    # D2 unresolved items have no unique gold assignment. Keep their responses, but
    # never spend judge calls or report accuracy against an arbitrary assignment.
    scorable = [r for r in responses.values()
                if not items[r["scenario_id"]].get("scoring_exclude")]
    filename = f"responses.{key}.scorable.jsonl"
    write_rows(d["run"] / filename, scorable)
    score_path = d["run"] / f"scores.{key}.jsonl"
    old = records(score_path, d["probes"])
    for r in old.values():
        if r["judge_model"] != judge_model:
            raise ValueError("judge changed; use a new output directory")
        if not r.get("parse_ok") or r.get("judge_error"):
            raw = d["run"] / f"judge_raw_{key}" / f"{r['scenario_id']}__identify.txt"
            raw.unlink(missing_ok=True)

    def checked_call(prompt, *, model):
        raw = (call or run_trapi)(prompt, model=model)
        if not raw.strip() or _cli_error(raw) or raw.lstrip().startswith("[trapi "):
            raise RuntimeError("judge transport failed; rerun to resume cached verdicts")
        return raw

    run_judge(d["run"], judge_model, call=checked_call, tag=key,
              responses_file=filename, resume=True, max_workers=workers)
    scores = records(score_path, d["probes"])
    if set(scores) != set(indexed(scorable)) or any(
            not r.get("parse_ok") or r.get("judge_error") for r in scores.values()):
        raise ValueError(f"incomplete/invalid grading for {key}; rerun --stage judge")


@contextmanager
def server(root: Path, out: Path, key: str, spec: dict, port: int, context: int):
    """Own one isolated vLLM process group; release all workers on failure/interrupt."""
    revision_path = out / f"revision.{key}.json"
    if revision_path.exists():
        revision = json.loads(revision_path.read_text())["revision"]
    else:
        # Query public Hub metadata before vLLM downloads weights; the client environment
        # needs no torch/transformers/huggingface_hub installation for this one request.
        endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
        req = urllib.request.Request(f"{endpoint}/api/models/{spec['model']}")
        if os.environ.get("HF_TOKEN"):
            req.add_header("Authorization", f"Bearer {os.environ['HF_TOKEN']}")
        with urllib.request.urlopen(req, timeout=60) as response:
            revision = json.load(response)["sha"]
        if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            raise ValueError("Hub returned an invalid checkpoint revision")
        write_json(revision_path, {"model": spec["model"], "revision": revision})
    # Refuse an occupied port instead of accidentally collecting from another server.
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", port))
    env = {**os.environ, "VLLM_PORT": str(port), "VLLM_MAX_LEN": str(context),
           "VLLM_VENV": os.environ.get("VLLM_VENV", str(root / ".vllm-venv")),
           "VLLM_EXTRA_ARGS": f"--revision {revision} --seed 20261113 "
                              "--enable-prefix-caching --max-num-seqs 16"}
    with (out / f"server.{key}.log").open("a") as log:
        proc = subprocess.Popen(["bash", str(root / "scripts/schema_llm/vllm_serve.sh"),
                                 spec["model"]], cwd=root, env=env, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
        try:
            for _ in range(360):
                if proc.poll() is not None:
                    raise RuntimeError(f"vLLM exited; inspect server.{key}.log")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2):
                        break
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(10)
            else:
                raise RuntimeError(f"vLLM startup timed out; inspect server.{key}.log")
            runtime = subprocess.check_output([
                str(Path(env["VLLM_VENV"]) / "bin/python"), "-c",
                "import importlib.metadata as m,json;print(json.dumps({"
                "n:m.version(n) for n in ['vllm','torch','transformers']}))"], text=True)
            versions = json.loads(runtime)
            if versions["vllm"] != os.environ.get("VLLM_VERSION", "0.25.1"):
                raise ValueError("existing VLLM_VENV has another vLLM version; use a fresh venv")
            write_json(out / f"server_runtime.{key}.json", versions)
            yield f"http://127.0.0.1:{port}/v1"
        finally:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            except ProcessLookupError:
                pass


def export_results(data: dict, keys: list[str], out: Path, judge_model: str) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows, texts = [], []
    for corpus, d in data.items():
        items = {r["scenario_id"]: r for r in d["items"]}
        probes = indexed(d["probes"])
        for key in keys:
            responses = records(d["run"] / f"responses.{key}.jsonl", d["probes"])
            scores = records(d["run"] / f"scores.{key}.jsonl", d["probes"])
            if set(responses) != set(probes):
                raise ValueError(f"incomplete responses for {corpus}/{key}")
            marker = json.loads((d["run"] / f"grading_inputs.{key}.json").read_text())
            if marker != {"responses_sha256": sha(d["run"] / f"responses.{key}.jsonl"),
                          "judge_model": judge_model} or any(
                    r.get("model") != ALL_MODELS[key]["model"] for r in responses.values()):
                raise ValueError(f"graded responses changed for {corpus}/{key}")
            expected = {k for k in probes if not items[k[0]].get("scoring_exclude")}
            if set(scores) != expected:
                raise ValueError(f"incomplete scores for {corpus}/{key}")
            for (sid, task), resp in sorted(responses.items()):
                item, score = items[sid], scores.get((sid, task))
                if score and (score.get("judge_error") or not score.get("parse_ok")
                              or score["judge_model"] != judge_model):
                    raise ValueError(f"invalid score for {corpus}/{key}/{sid}")
                meta, probe = d["meta"][sid], probes[(sid, task)]
                row = {"model": key, "model_display_name": DISPLAY_NAMES[key],
                       "subject_model": ALL_MODELS[key]["model"],
                       "task_id": meta["task_id"], "sample_id": "s1", "task": task,
                       "corpus": corpus, "source": meta["source"], "scenario_id": sid,
                       "judge_model": judge_model if score else None,
                       "scoring_exclude": bool(item.get("scoring_exclude")),
                       "status": "ok" if score else "scoring_excluded"}
                rev_path = out / f"revision.{key}.json"
                row["model_revision"] = (json.loads(rev_path.read_text())["revision"]
                                         if rev_path.exists() else None)
                for field in ("wave", "family_id", "design", "arm", "schema",
                              "detail_level", "distractor_level"):
                    row[field] = item.get(field)
                gen = item.get("gen") or {}
                for field in ("prop_naming", "role_naming"):
                    row[field] = item.get(field) or gen.get(field)
                row["temporal_form"] = (item.get("temporal") or {}).get("form")
                for field in ("detected", "localized", "type_correct"):
                    row[field] = None  # identify task; retain the core table's column names
                for field in ("correct", "schema_correct", "role_accuracy", "role_precision",
                              "role_f1", "parse_ok", "judge_error"):
                    row[field] = score.get(field) if score else None
                row["prompt_sha256"] = hashlib.sha256(probe["probe_prompt"].encode()).hexdigest()
                row["response_sha256"] = hashlib.sha256(resp["response"].encode()).hexdigest()
                reasoning = (ALL_MODELS[key].get("sample", False)
                             or ALL_MODELS[key].get("reasoning", False)
                             or "<think>" in resp["response"] or "</think>" in resp["response"])
                row["think_closed"] = "</think>" in resp["response"] if reasoning else None
                rows.append(row)
                texts.append({**row, "prompt": probe["probe_prompt"], "gold": json.dumps(
                    probe["gold"], ensure_ascii=False), "response": resp["response"]})
    if len({(r["model"], r["task_id"], r["sample_id"]) for r in rows}) != len(rows):
        raise ValueError("duplicate output primary keys")
    (out / "export_manifest.json").unlink(missing_ok=True)
    csv_temp = out / "response_outcomes.csv.tmp"
    with csv_temp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    csv_temp.replace(out / "response_outcomes.csv")
    pq_temp = out / "response_outcomes_text.parquet.tmp"
    pq.write_table(pa.Table.from_pylist(texts), pq_temp, compression="zstd")
    pq_temp.replace(out / "response_outcomes_text.parquet")
    write_json(out / "export_manifest.json", {
        "format": "d1d2-behavior-export-v1", "complete": True, "judge_model": judge_model,
        "primary_key": ["model", "task_id", "sample_id"],
        "coverage": {key: {
            "responses": sum(r["model"] == key for r in rows),
            "scored": sum(r["model"] == key and r["status"] == "ok" for r in rows),
            "excluded": sum(r["model"] == key and r["scoring_exclude"] for r in rows),
        } for key in keys},
        "artifacts": {name: sha(out / name) for name in (
            "response_outcomes.csv", "response_outcomes_text.parquet",
            "response_outcomes_task_id_map.jsonl")},
    })
    print(f"[export] {len(rows)} rows, {len(keys)} models -> {out}", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path(
        "results/schema_llm/analysis/behavior/20261113-d1d2-eightway"))
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--models", default=",".join(MODELS), help="comma-separated roster keys")
    p.add_argument("--include-existing", action="store_true",
                   help="include bundled Phi/Qwen records in the export, without new model calls")
    p.add_argument("--stage", choices=["all", "prepare", "local", "trapi", "judge", "export"],
                   default="all")
    p.add_argument("--judge-model", default=DEFAULT_JUDGE)
    p.add_argument("--max-new-tokens", type=int, default=8192)
    p.add_argument("--max-completion-tokens", type=int, default=8192)
    p.add_argument("--context-length", type=int, default=12288)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--judge-workers", type=int, default=4)
    p.add_argument("--port", type=int, default=8400)
    a = p.parse_args(argv)
    keys = a.models.split(",")
    if not keys or len(set(keys)) != len(keys) or any(k not in MODELS for k in keys):
        p.error(f"--models must select unique keys from {', '.join(MODELS)}")
    if a.include_existing and a.stage not in ("all", "export"):
        p.error("--include-existing applies to --stage all or export")
    if a.include_existing and a.judge_model != DEFAULT_JUDGE:
        p.error("--include-existing requires the original Terra judge; reconcile other judges "
                "before combining results")
    if min(a.max_new_tokens, a.max_completion_tokens, a.concurrency, a.judge_workers) < 1:
        p.error("budgets and concurrency must be positive")
    if a.context_length <= a.max_new_tokens:
        p.error("--context-length must leave room for prompts beyond --max-new-tokens")
    root, out = a.root.resolve(), a.out.resolve()
    verify_release(root)
    data, _ = load_inputs(root)
    for corpus, d in data.items():
        n = len(d["items"])
        excluded = sum(bool(r.get("scoring_exclude")) for r in d["items"])
        print(f"[input] {corpus}: {n} prompts, both waves, {excluded} excluded from grading")
    print(f"[models] {', '.join(keys)}; judge={a.judge_model}")
    print(f"[config] one draw; local/hosted caps={a.max_new_tokens}/{a.max_completion_tokens}; "
          f"context={a.context_length}; concurrency={a.concurrency}; "
          f"include_existing={a.include_existing}")
    if a.check_only:
        return
    settings = {"format": "d1d2-behavior-v1", "input_sha256": sha(
        root / "data/d1d2_behavior/manifest.json"), "smoke": a.smoke,
        "roster": MODELS, "judge_model": a.judge_model,
        "max_new_tokens": a.max_new_tokens, "max_completion_tokens": a.max_completion_tokens,
        "context_length": a.context_length, "server_seed": 20261113,
        "concurrency": a.concurrency,
        "code_sha256": {name: sha(root / "code/dev/schema_llm" / name) for name in (
            "explore/d1d2_behavior.py", "subjects/collect.py", "subjects/judge.py",
            "core/llm.py", "core/schemas/airport.yaml", "core/schemas/restaurant.yaml")},
        "hosted_decoding": "deployment defaults; reasoning_effort/temperature/top_p unset",
        "vllm_version": os.environ.get("VLLM_VERSION", "0.25.1"),
        "trapi_endpoint": os.environ.get("OPENAI_BASE_URL", os.environ.get(
            "TRAPI_HOST", "https://trapi.research.microsoft.com")
            + "/redmond/interactive/openai/v1")}
    data = prepare(root, out, data, settings, a.smoke)
    bind_json(out / "input_manifest.json", json.loads(
        (root / "data/d1d2_behavior/manifest.json").read_text()))
    write_json(out / "runtime.json", {"python": sys.version, "packages": {
        name: importlib.metadata.version(name) for name in ("openai", "azure-identity", "pyarrow")
    }})
    if a.stage == "prepare":
        return
    from ..subjects.collect import make_trapi_responder, make_vllm_responder

    # A scheduler's SIGTERM must unwind the server context and terminate GPU workers.
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
    for key in keys:
        spec = MODELS[key]
        if a.stage in ("all", spec["backend"].replace("vllm", "local")):
            complete = all(len(records(d["run"] / f"responses.{key}.jsonl", d["probes"]))
                           == len(d["probes"]) for d in data.values())
            if not complete:
                if spec["backend"] == "trapi":
                    call = make_trapi_responder(spec["model"],
                                               max_completion_tokens=a.max_completion_tokens)
                    for d in data.values():
                        collect_one(d, key, spec, call, a.concurrency)
                else:
                    with server(root, out, key, spec, a.port, a.context_length) as url:
                        call = make_vllm_responder(
                            spec["model"], base_url=url, timeout=None,
                            max_tokens=a.max_new_tokens, system=spec.get("system"),
                            do_sample=spec.get("sample", False),
                            temperature=spec.get("temperature"), top_p=spec.get("top_p"),
                            top_k=spec.get("top_k"), enable_thinking=spec.get("think"))
                        for d in data.values():
                            collect_one(d, key, spec, call, a.concurrency)
        if a.stage in ("all", "judge"):
            for d in data.values():
                grade_one(d, key, a.judge_model, a.judge_workers)
    if a.stage in ("all", "export"):
        if a.include_existing:
            import_existing(root, data, a.judge_model)
        export_results(data, [*keys, *(EXISTING_MODELS if a.include_existing else [])],
                       out, a.judge_model)


if __name__ == "__main__":
    main()
