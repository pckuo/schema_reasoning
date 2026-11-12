"""Entry points for the standalone collaborator reproduction repository.

Algorithms live in the existing primitive/STOP modules. This adapter binds their
data paths to the frozen release, checks inputs and vector fingerprints, and
starts each model stage in a separate process.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .combined_primitives import _json, _sha, _write_json


def verify(root: Path) -> dict:
    manifest = _json(root / "MANIFEST.json")
    for filename, expected in manifest["files"].items():
        path = root / filename
        if not path.is_file() or _sha(path) != expected:
            raise ValueError(f"release file missing or changed: {filename}")
    return manifest


def configure(root: Path) -> None:
    """Bind paths without modifying the original source or its experiment constants."""
    from .stop_steering import runframe
    from .tsp_stop_steering import tasks

    bundle = root / "data/vstop/core"
    runframe.BUNDLE = bundle
    # Default arguments captured the old path at import time; bind them explicitly.
    runframe.build = partial(runframe.build, root=bundle)
    # build() passes root positionally to load_bundle(); binding it there as a
    # keyword too would supply root twice.
    frozen = _json(root / "data/vstop/tsp/tsp_tasks.json")
    by_uid = {int(t["uid"]): t for t in frozen}

    def tasks_by_uid(uids, path=None):
        if not set(map(int, uids)) <= set(by_uid):
            raise ValueError("unknown TSP task requested")
        return {int(uid): by_uid[int(uid)] for uid in uids}

    tasks.tasks_by_uid = tasks_by_uid


def configure_selection(root: Path) -> None:
    from .stop_steering import capture, vectors

    # Retain the ORIGINAL callable: calling capture.selected_heads after replacing
    # it would recurse (the earlier handoff wrapper had that bug).
    original = capture.selected_heads
    bound = partial(original, path=root / "data/vstop/reference/selection.json")
    capture.selected_heads = bound
    vectors.selected_heads = bound


def check_stop_inputs(root: Path) -> dict:
    from .stop_steering import pilot, runframe

    built = runframe.build()
    problems = runframe.verify(runframe.summarize(built))
    if problems:
        raise ValueError(f"original run-frame parity failed: {problems}")
    n = len(pilot.population(built, lexicon_variant="extended", require_all_right=True))
    if n != 17:
        raise ValueError(f"expected 17 capped/solved/all-right items; got {n}")
    tasks = _json(root / "data/vstop/tsp/tsp_tasks.json")
    bases = pq.read_table(root / "data/vstop/tsp/baselines.parquet").to_pylist()
    expected = {(int(t["uid"]), draw) for t in tasks for draw in range(3)}
    if len(bases) != 18 or {(int(b["uid"]), int(b["draw"])) for b in bases} != expected:
        raise ValueError("TSP baselines are not the frozen six-task/three-draw population")
    selection = _json(root / "data/vstop/reference/selection.json")
    if len(selection["heads"]) != 35 or len({tuple(h) for h in selection["heads"]}) != 35:
        raise ValueError("expected 35 unique selected heads")
    return {
        "schema_intervention_items": n,
        "reference_capture_items": len(built["items"]),
        "tsp_tasks": len(tasks),
        "tsp_baselines": len(bases),
    }


def check_frame(reference: Path, capture: Path) -> dict:
    import pandas as pd

    ref, new = _json(reference / "parity.json"), _json(capture / "parity.json")
    errors = [
        f"parity {k}"
        for k in ("dropped", "total", "last3_runs", "ref_stop_runs")
        if ref[k] != new[k]
    ]
    cols = ["item", "t", "z0", "last3", "ref_stop", "fold", "row_end"]
    left = pd.read_parquet(reference / "frame.parquet")[cols].reset_index(drop=True)
    right = pd.read_parquet(capture / "frame.parquet")[cols].reset_index(drop=True)
    if not left.equals(right):
        errors.append("reference capture frame differs")
    if not new.get("norm_cut_matches"):
        errors.append("reference norm cut differs")
    report = {"passed": not errors, "rows": len(right), "errors": errors}
    _write_json(capture / "check_report.json", report)
    if errors:
        raise ValueError(str(report))
    return report


def check_vectors(reference: Path, vectors: Path, repeat: Path, config: dict) -> dict:
    """The original November handoff's checks; do not relax a failed reference gate."""
    ref, new = _json(reference / "manifest.json"), _json(vectors / "manifest.json")
    keys = {"stop", "shuffled", "random", "switch", "solved"}
    errors = []
    with np.load(vectors / "directions.npz") as left, np.load(repeat / "directions.npz") as right:
        same = set(left.files) == set(right.files) == keys and all(
            np.array_equal(left[k], right[k]) for k in keys
        )
        if not same:
            errors.append("repeat build is not bitwise identical")
        for name in keys & set(left.files):
            v = left[name]
            if v.shape != (3072,) or not np.isfinite(v).all() or abs(np.linalg.norm(v) - 1) > 1e-5:
                errors.append(f"invalid vector {name}")
    deltas = {}
    for section in ("residual_cosines", "head_space_cosines"):
        for pair, value in ref[section].items():
            delta = abs(new[section].get(pair, float("inf")) - value)
            deltas[f"{section}/{pair}"] = delta
            if delta > config["cosine_atol"]:
                errors.append(f"{section}/{pair} delta={delta}")
    for name in keys:
        for field in ("raw_norm", "head_space_projection_sd"):
            delta = abs(new["directions"][name][field] - ref["directions"][name][field])
            if delta > config["scale_atol"]:
                errors.append(f"{name}/{field} delta={delta}")
    for field in (
        "model",
        "revision",
        "n_heads",
        "source_layers",
        "rows",
        "items",
        "n_stop_positions",
        "n_baseline_positions",
        "seed",
        "mask_counts",
    ):
        if new[field] != ref[field]:
            errors.append(f"reference {field} differs")
    if abs(new["median_l16_state_norm"] - ref["median_l16_state_norm"]) > config["median_atol"]:
        errors.append("median state norm differs")
    report = {
        "passed": not errors,
        "bitwise_repeat": same,
        "errors": errors,
        "cosine_deltas": deltas,
        "numpy": np.__version__,
        "reference_sha256": _sha(reference / "manifest.json"),
        "directions_sha256": _sha(vectors / "directions.npz"),
    }
    _write_json(vectors / "check_report.json", report)
    if errors:
        raise ValueError(f"STOP vector reference check failed: {errors}")
    return report


def use_supplied_vectors(root: Path, out: Path | None = None) -> dict:
    """Validate the four used directions and install the user-approved supplied file."""
    source = root / "data/vstop/supplied/directions.npz"
    required = {"stop", "shuffled", "switch", "solved"}
    with np.load(source, allow_pickle=False) as vectors:
        if not required <= set(vectors.files):
            raise ValueError("supplied file lacks a required intervention direction")
        for name in required:
            v = vectors[name]
            if v.shape != (3072,) or not np.isfinite(v).all() or abs(np.linalg.norm(v) - 1) > 1e-5:
                raise ValueError(f"invalid supplied vector {name}")
    report = {
        "passed": True,
        "source": "user-supplied",
        "directions_sha256": _sha(source),
        "reference_sha256": _sha(root / "data/vstop/reference/manifest.json"),
        "reference_cosine_differences_accepted": True,
        "required_vectors": sorted(required),
    }
    if out is not None:
        dest = out / "vectors"
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest / "directions.npz")
        shutil.copy2(root / "data/vstop/reference/manifest.json", dest / "manifest.json")
        _write_json(dest / "check_report.json", report)
    return report


def validate_stop_tokenizer(root: Path):
    from transformers import AutoTokenizer

    from .combined_primitives import MODELS, validate_tokenizer
    from .tsp_stop_steering import run, tasks

    model, revision = MODELS["phi4mr"]
    tok = AutoTokenizer.from_pretrained(model, revision=revision)
    tok.chat_template = (root / "data/vstop/tsp/phi4_mini_instruct_chat_template.jinja").read_text()
    table = pq.read_table(
        root / "data/vstop/core/tokens.parquet",
        filters=[("model", "=", "phi4mr")],
        columns=["item_id", "position", "token_id", "token"],
    ).to_pandas()
    seqs = (
        {
            "item_id": item,
            "token_ids": group.sort_values("position").token_id.tolist(),
            "tokens": group.sort_values("position").token.tolist(),
        }
        for item, group in table.groupby("item_id", sort=True)
    )
    if validate_tokenizer(tok, seqs) != 216:
        raise ValueError("expected 216 tokenizer-validated Phi identify sequences")
    if not {200020, 199999} <= set(tok.get_vocab().values()):
        raise ValueError("STOP EOS IDs are absent from the tokenizer")
    selected = _json(root / "data/vstop/tsp/tsp_tasks.json")
    by_uid = {t["uid"]: t for t in selected}
    for b in pq.read_table(root / "data/vstop/tsp/baselines.parquet").to_pylist():
        prompt = tok.apply_chat_template(
            tasks.messages(by_uid[b["uid"]]["prompt"], system=True),
            add_generation_prompt=True,
            return_dict=True,
        )["input_ids"]
        if len(prompt) != b["n_prompt"]:
            raise ValueError(f"TSP prompt boundary differs: {b['uid']}")
        if run.find_think_end(tok, b["token_ids"]) != b["reasoning_tokens"]:
            raise ValueError(f"TSP reasoning boundary differs: {b['uid']}")
    return tok


def run_schema(root: Path, out: Path, config: dict, device: str) -> None:
    """Run the original pilot one item at a time, caching its one model load.

    Item/draw seeds, prefixes, budgets, directions and generation code are unchanged.
    Completed item shards survive an interruption; no comparison is made to stored tails.
    """
    import pandas as pd

    from .head_fv_models import active
    from .stop_steering import pilot, runframe

    built = runframe.build()
    targets = pilot.population(built, lexicon_variant="extended", require_all_right=True)
    spec = active()
    lm = None
    original_population, original_loader, original_build = (
        pilot.population,
        pilot.load_model,
        runframe.build,
    )
    try:

        def load_once(*a, **kw):
            nonlocal lm
            if lm is None:
                lm = original_loader(spec.model, revision=spec.revision, device_map=device)
            return lm

        pilot.load_model = load_once
        runframe.build = lambda *a, **kw: built
        for experiment in [config["schema"], *config.get("active_ablations", [])]:
            dest = out / "schema" / experiment["name"]
            frames = []
            for item in targets.item:
                part = dest / "parts" / item
                path = part / "pilot.parquet"
                options = {k: v for k, v in experiment.items() if k != "name"}
                plan = {
                    (arm, float(dose), draw)
                    for arm in options["arms"]
                    for dose in options["doses"]
                    for draw in range(options.get("draws", 1))
                }
                plan |= {("baseline", 0.0, d) for d in range(options.get("draws", 1))}
                if not path.exists():
                    pilot.population = lambda *a, _item=item, **kw: targets[targets.item == _item]
                    pilot.run(out / "vectors", part, device=device, budget=-1, **options)
                frame = pd.read_parquet(path)
                actual = set(zip(frame.arm, frame.dose, frame.draw, strict=True))
                if len(frame) != len(plan) or actual != plan or set(frame.item) != {item}:
                    raise ValueError(f"schema shard is incomplete or incompatible: {path}")
                frames.append(frame)
            dest.mkdir(parents=True, exist_ok=True)
            table = pd.concat(frames, ignore_index=True)
            table.to_parquet(dest / "pilot.parquet")
            # Original summaries contain NaN for absent optional arms; retain their JSON encoding.
            (dest / "summary.json").write_text(json.dumps(pilot.summarize(table), indent=2))
    finally:
        pilot.population, pilot.load_model, runframe.build = (
            original_population,
            original_loader,
            original_build,
        )


def run_tsp(root: Path, out: Path, device: str, smoke: bool = False) -> None:
    import pandas as pd

    from ..core.model import load_model
    from .head_fv_models import active
    from .tsp_stop_steering import run

    selected = _json(root / "data/vstop/tsp/tsp_tasks.json")
    spec = active()
    template = (root / "data/vstop/tsp/phi4_mini_instruct_chat_template.jinja").read_text()
    lm = load_model(spec.model, revision=spec.revision, device_map=device, chat_template=template)
    dest = out / "tsp"
    dest.mkdir(parents=True, exist_ok=True)
    report = run.run_parity(lm, spec, selected[0], run.prompt_ids_for(lm, selected[0], True), dest)
    if not report["all_exact"]:
        raise ValueError("TSP cache parity failed")
    if smoke:
        return
    base_path = dest / "baselines.parquet"
    if not base_path.exists():
        shutil.copy2(root / "data/vstop/tsp/baselines.parquet", base_path)
    elif _sha(base_path) != _sha(root / "data/vstop/tsp/baselines.parquet"):
        raise ValueError("TSP baseline snapshot changed")
    config = _json(root / "configs/vstop.json")["tsp"]
    baselines = pd.read_parquet(base_path)
    _write_json(
        dest / "manifest.json",
        {
            **config,
            "model": spec.model,
            "revision": spec.revision,
            "tasks": selected,
            "baseline_source": "frozen release",
        },
    )
    run.stage_inject(
        lm,
        spec,
        selected,
        baselines,
        dest,
        cap=config["cap"],
        draws=config["draws"],
        system=True,
        vectors=out / "vectors",
        layer=config["layer"],
        max_batch=config["max_batch"],
    )


def analyse(root: Path, out: Path, config: dict) -> None:
    import pandas as pd

    from .stop_steering import accuracy, figures_panels, specificity, stats
    from .tsp_stop_steering import analyze as tsp_analyze
    from .tsp_stop_steering import figures as tsp_figures

    greedy = [
        out / "schema" / e["name"]
        for e in [config["schema"], *config.get("active_ablations", [])]
        if e.get("temperature", 0) == 0
    ]
    scored = out / "analysis/accuracy"
    accuracy.run([str(p) for p in greedy], scored, scope="answer")
    for path in greedy:
        stats.run(path)
    main, plots = out / "schema/main", out / "figures/schema"
    plots.mkdir(parents=True, exist_ok=True)
    specificity.run(main, scored / "accuracy.parquet", out / "analysis/specificity")
    figures_panels.fig_survival(main, plots)
    figures_panels.fig_dose(main, plots)
    figures_panels.fig_specificity(main, scored / "accuracy.parquet", plots)
    figures_panels.fig_answer_delivery(main, scored / "accuracy.parquet", plots)
    if config.get("active_ablations"):
        figures_panels.fig_damage(
            {8: out / "schema/layer8", 16: main, 24: out / "schema/layer24"}, plots
        )
        figures_panels.fig_sampling(out / "schema/sampling", plots)
        figures_panels.fig_oneshot(out / "schema/oneshot", main, plots)
    tsp_analyze.run(out / "tsp")
    tsp_figures.run(out / "tsp", out / "figures/tsp_all6")
    # Report the original manuscript's three instances separately, from the SAME runs.
    original = out / "tsp_original3"
    original.mkdir(exist_ok=True)
    ids = [
        t["uid"] for t in _json(root / "data/vstop/tsp/tsp_tasks.json") if t["cohort"] == "original"
    ]
    for name in ("baselines", "injections"):
        table = pd.read_parquet(out / "tsp" / f"{name}.parquet")
        table[table.uid.isin(ids)].to_parquet(original / f"{name}.parquet")
    manifest = _json(out / "tsp/manifest.json")
    manifest["tasks"] = [t for t in manifest["tasks"] if t["uid"] in ids]
    _write_json(original / "manifest.json", manifest)
    tsp_analyze.run(original)
    tsp_figures.run(original, out / "figures/tsp_original3")


def _invoke(stage: str, root: Path, out: Path, device: str, ablations: bool) -> None:
    command = [
        sys.executable,
        "-m",
        "dev.schema_llm.explore.reproduction",
        "stop-stage",
        "--root",
        str(root),
        "--out",
        str(out),
        "--device",
        device,
        "--stage",
        stage,
    ]
    if ablations:
        command.append("--ablations")
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    print(f"STOP stage: {stage} (log: {logs / (stage + '.log')})", flush=True)
    with (logs / f"{stage}.log").open("a") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)


def _runtime() -> dict:
    return {
        "python": sys.version,
        **{
            name: importlib.metadata.version(name)
            for name in ("numpy", "torch", "transformers", "pandas")
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("workflow", choices=["primitives", "merge-primitives", "vstop", "stop-stage"])
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[4])
    p.add_argument("--out", type=Path)
    p.add_argument("--input", type=Path, help="Normalized immutable primitive input pack")
    p.add_argument("--extract-only", action="store_true", help="Stop after clustering and labeling")
    p.add_argument("--from", dest="extractions", action="append", default=[], metavar="NAME=FOLDER")
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--ablations", action="store_true")
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--stage",
        choices=[
            "capture",
            "vectors",
            "repeat",
            "schema",
            "tsp",
            "analyse",
            "capture-smoke",
            "tsp-smoke",
        ],
    )
    args = p.parse_args()
    root = args.root.resolve()
    os.chdir(root)
    verify(root)
    if args.workflow == "merge-primitives":
        from .incremental_primitives import merge_extractions

        if args.out is None:
            p.error("merge-primitives requires --out (a separate merge folder)")
        if args.smoke or args.extract_only or args.input is not None:
            p.error("merge-primitives accepts --from, --out and --check-only")
        merge_extractions(
            args.extractions,
            args.out.resolve(),
            os.getenv("EMBED_MODEL", "Qwen/Qwen3-Embedding-4B"),
            args.check_only,
        )
        return
    if args.workflow == "primitives":
        from . import combined_primitives as cp

        pack = (args.input or root / "data/primitives").resolve()
        cp.verify(pack)
    else:
        configure(root)
        config = _json(root / "configs/vstop.json")
        config["active_ablations"] = config["ablations"] if args.ablations else []
        if args.workflow == "vstop":
            if (root / "data/vstop/supplied/directions.npz").exists():
                use_supplied_vectors(root)
            print(
                "Checking the original STOP frame and intervention population (CPU)...", flush=True
            )
            print(json.dumps(check_stop_inputs(root), indent=2), flush=True)
    if args.check_only:
        print("Checks passed; no weights, GPU forwards or API calls were used.")
        return
    if sys.version_info[:2] != (3, 12):
        raise ValueError("Use the release's Python 3.12 environment: uv sync --locked")
    import torch

    if args.workflow != "stop-stage" or args.stage != "analyse":
        if not torch.cuda.is_available():
            raise ValueError(
                "This experiment requires an NVIDIA CUDA device; use --check-only for CPU checks"
            )
    out = (
        args.out or root / "results" / (args.workflow + ("_smoke" if args.smoke else ""))
    ).resolve()
    out.mkdir(parents=True, exist_ok=True)
    key = {
        "release": _sha(root / "MANIFEST.json"),
        "workflow": args.workflow,
        "smoke": args.smoke,
        "ablations": args.ablations,
        "runtime": _runtime(),
        "device": args.device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "cuda_runtime": torch.version.cuda,
    }
    if args.workflow != "stop-stage":
        if args.workflow == "primitives":
            key["input_pack_sha256"] = _sha(pack / "manifest.json")
        state = out / "run_config.json"
        if state.exists() and _json(state) != key:
            raise ValueError("work directory belongs to a different release/configuration/runtime")
        _write_json(state, key)
    if args.workflow == "primitives":
        base = [sys.executable, "-m", "dev.schema_llm.explore.combined_primitives"]
        common = ["--pack", str(pack), "--work", str(out)]
        for model, prefix in (("phi4mr", "PHI"), ("qwen34b", "QWEN")):
            command = [*base, "capture", *common, "--model", model]
            for option in ("MODEL", "REVISION"):
                if os.environ.get(f"{prefix}_{option}"):
                    flag = "--checkpoint" if option == "MODEL" else "--revision"
                    command.extend([flag, os.environ[f"{prefix}_{option}"]])
            if args.smoke:
                command.extend(["--limit", "2"])
            subprocess.run(command, check=True)
        if args.smoke:
            print("Capture smoke passed (2 sequences/model); clustering/API/merge were not run.")
            return
        subprocess.run(
            [
                *base,
                "cluster",
                *common,
                "--backend",
                os.getenv("ANNOTATE_BACKEND", "trapi"),
                "--annotate-model",
                os.getenv("ANNOTATE_MODEL", "gpt-5.6-sol_2026-07-09"),
            ],
            check=True,
        )
        from .incremental_primitives import register

        register(pack, out)
        if args.extract_only:
            print("Extraction complete; extraction.json is ready for a separate joint merge.")
            return
        subprocess.run(
            [
                *base,
                "merge",
                *common,
                "--embed-model",
                os.getenv("EMBED_MODEL", "Qwen/Qwen3-Embedding-4B"),
            ],
            check=True,
        )
        subprocess.run([*base, "export", *common], check=True)
    elif args.workflow == "vstop":
        validate_stop_tokenizer(root)  # ALL inputs, before loading any model weights
        gc.collect()
        if args.smoke:
            for stage in ("capture-smoke", "tsp-smoke"):
                _invoke(stage, root, out, args.device, False)
            print("STOP smoke passed (2 captures and TSP cache parity); full injection not run.")
            return
        reference = root / "data/vstop/reference"
        report = out / "vectors/check_report.json"
        ready = False
        if report.exists():
            r = _json(report)
            ready = (
                r.get("passed")
                and r["reference_sha256"] == _sha(reference / "manifest.json")
                and r["directions_sha256"] == _sha(out / "vectors/directions.npz")
            )
        if (root / "data/vstop/supplied/directions.npz").exists():
            use_supplied_vectors(root, out)
            ready = True
        if not ready:
            for stage in ("capture", "vectors", "repeat"):
                _invoke(stage, root, out, args.device, args.ablations)
            check_vectors(reference, out / "vectors", out / "vectors-repeat", config["vector_gate"])
        for stage in ("schema", "tsp", "analyse"):
            _invoke(stage, root, out, args.device, args.ablations)
        _write_json(
            out / "complete.json", {"passed": True, "configuration": key, "vectors": _json(report)}
        )
    else:
        configure_selection(root)
        from .stop_steering import capture, vectors

        if args.stage in ("capture", "capture-smoke"):
            capture.run(
                out / "capture",
                limit=2 if args.stage.endswith("smoke") else None,
                device=args.device,
                cohort="terminating",
            )
            if args.stage == "capture":
                capture.finalize(out / "capture", cohort="terminating")
                check_frame(root / "data/vstop/reference", out / "capture")
        elif args.stage in ("vectors", "repeat"):
            vectors.build(
                out / "capture",
                out / ("vectors" if args.stage == "vectors" else "vectors-repeat"),
                device=args.device,
            )
        elif args.stage == "schema":
            run_schema(root, out, config, args.device)
        elif args.stage in ("tsp", "tsp-smoke"):
            run_tsp(root, out, args.device, smoke=args.stage.endswith("smoke"))
        elif args.stage == "analyse":
            analyse(root, out, config)


if __name__ == "__main__":
    main()
