"""Reuse immutable per-source extractions in separately versioned joint merges."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from . import combined_primitives as cp

NODE_FILES = ("config.json", "clusters.json", "clusters.npz", "kmeans.json", "kmeans.npz")


def extraction_record(pack: Path, work: Path) -> dict:
    manifest = cp.verify(pack)
    sources, settings = {}, None
    for corpus, model, task, source in cp._sources(manifest):
        path = work / "nodes" / corpus / model
        config = cp._json(path / "config.json")
        shared = {
            k: config[k]
            for k in (
                "layers",
                "k",
                "force_meta",
                "seed",
                "pca",
                "annotate",
                "annotate_model",
                "context",
            )
        }
        if settings is not None and shared != settings:
            raise ValueError("incompatible extraction settings within input")
        settings = shared
        if config["tasks"] != [task] or config["slug"] != model:
            raise ValueError(f"source identity differs: {source}")
        if config["pack"] != cp._sha(pack / "manifest.json"):
            raise ValueError(f"node input pack changed: {source}")
        if cp._json(path / "clusters.json")["key"] != config:
            raise ValueError(f"node completion/configuration differs: {source}")
        if cp._json(path / "kmeans.json")["key"] != config:
            raise ValueError(f"fine node completion/configuration differs: {source}")
        sources[source] = {
            "path": str(path.resolve()),
            "files": {name: cp._sha(path / name) for name in NODE_FILES},
            "capture": config["capture"],
        }
    if not sources:
        raise ValueError("no extraction sources")
    return {
        "pack": str(pack.resolve()),
        "pack_sha256": cp._sha(pack / "manifest.json"),
        "settings": settings,
        "sources": sources,
    }


def register(pack: Path, work: Path) -> None:
    cp._write_json(work / "extraction.json", extraction_record(pack, work))


def merge_extractions(
    inputs: list[str], out: Path, embed_model: str, check_only: bool = False
) -> None:
    entries, sources, settings, captures = {}, {}, None, {}
    for value in inputs:
        name, separator, folder = value.partition("=")
        work = Path(folder if separator else value).resolve()
        name = name if separator else work.name
        if not name or name in {".", ".."} or "/" in name or "\\" in name or name in entries:
            raise ValueError(f"invalid or duplicate input name: {name}")
        if out.resolve() == work or out.resolve().is_relative_to(work):
            raise ValueError("joint merge output must be separate from extraction inputs")
        saved = cp._json(work / "extraction.json")
        pack = Path(saved["pack"])
        current = extraction_record(pack, work)
        if saved != current:
            raise ValueError(f"extraction changed since completion: {name}")
        if settings is not None and current["settings"] != settings:
            raise ValueError("incompatible extraction/annotation settings across inputs")
        settings = current["settings"]
        for source, node in current["sources"].items():
            if source in sources:
                raise ValueError(f"duplicate source: {source}; use distinct corpus namespaces")
            model = source.split("/")[1]
            capture = {
                k: node["capture"][k]
                for k in ("model", "checkpoint", "revision", "dtype", "layers")
            }
            if model in captures and capture != captures[model]:
                raise ValueError(f"capture model/revision/settings differ: {model}")
            captures[model] = capture
            sources[source] = node["files"]["clusters.json"]
        entries[name] = {"work": str(work), **current}
    if not entries:
        raise ValueError("at least one --from extraction is required")
    if check_only:
        print(f"Compatible completed extractions: {len(entries)} inputs, {len(sources)} sources")
        return
    key = {"inputs": entries, "sources": sources, "embed_model": embed_model}
    input_path = out / "merge/input.json"
    if input_path.exists() and cp._json(input_path) != key:
        raise ValueError("merge inputs changed; choose a new output folder")
    cp._write_json(input_path, key)
    merged = out / "merge/semavglink/merge.json"
    completion = out / "merge/complete.json"
    if merged.exists() and completion.exists():
        if cp._json(completion) != {"input": key, "merge_sha256": cp._sha(merged)}:
            raise ValueError("completed merge checksum changed")
    else:
        command = [sys.executable, "-m", "dev.schema_llm.primitives.crossmodel.merge"]
        for entry in entries.values():
            for source, node in entry["sources"].items():
                command.extend(["--source", f"{source}={node['path']}"])
        command.extend(
            [
                "--methods",
                "semantic-avglink",
                "--embed-model",
                embed_model,
                "--seed",
                str(settings["seed"]),
                "--title",
                "Joint primitive merge",
                "--out",
                str(out / "merge"),
            ]
        )
        subprocess.run(command, check=True)
        cp._write_json(completion, {"input": key, "merge_sha256": cp._sha(merged)})
    for name, entry in entries.items():
        cp.export(
            Path(entry["pack"]), Path(entry["work"]), merge_work=out, out=out / "export" / name
        )
    cp._write_json(
        out / "complete.json",
        {
            "inputs": list(entries),
            "sources": list(sources),
            "merge_sha256": cp._sha(merged),
            "exports": {name: str(Path("export") / name) for name in entries},
        },
    )
