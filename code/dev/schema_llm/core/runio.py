"""Per-model output namespacing + crash-safe resumable writes (multi-model comparison).

The collect / capture / judge / analysis pipeline originally wrote a single ``responses.jsonl`` /
``scores.jsonl`` per run dir, so running a second subject model would clobber the first. To compare
several subjects (base Phi-4, Phi-4-reasoning, GPT-5.5, ...) side by side, outputs are namespaced by
a short **tag** (the subject slug): ``responses.<tag>.jsonl`` / ``scores.<tag>.jsonl`` etc. A
``tag`` of ``None`` keeps the legacy un-suffixed name (the existing base-Phi-4 files stay put).

Long collection / judging loops (hundreds of GPU or API calls) must survive a dropped connection or
a container restart, so results are written **incrementally and flushed** per item, and a run can
**resume**: on restart it reads the already-written rows, skips the completed
``(scenario_id, task)`` keys, and appends only the rest.
"""

from __future__ import annotations

import json
from pathlib import Path

DoneKey = tuple


def model_slug(name: str) -> str:
    """Filesystem-safe short id for a model name (drops any ``org/`` prefix). Matches the slug the
    capture harness already uses for ``activations/<slug>/`` (case preserved)."""
    return _safe_tag(name.rstrip("/").split("/")[-1])


def _safe_tag(tag: str) -> str:
    """Reject caller-supplied tags that could escape the run directory namespace."""
    parts = tag.replace("\\", "/").split("/")
    if "/" in tag or "\\" in tag or ".." in parts:
        raise ValueError(f"unsafe run tag: {tag!r}")
    return tag


def _tagged(stem: str, tag: str | None, ext: str) -> str:
    return f"{stem}{ext}" if not tag else f"{stem}.{_safe_tag(tag)}{ext}"


def responses_path(run_dir: Path, tag: str | None = None) -> Path:
    return Path(run_dir) / _tagged("responses", tag, ".jsonl")


def responses_raw_dir(run_dir: Path, tag: str | None = None) -> Path:
    return Path(run_dir) / ("responses_raw" if not tag else f"responses_raw_{_safe_tag(tag)}")


def scores_path(run_dir: Path, tag: str | None = None) -> Path:
    return Path(run_dir) / _tagged("scores", tag, ".jsonl")


def scores_summary_path(run_dir: Path, tag: str | None = None) -> Path:
    return Path(run_dir) / _tagged("scores_summary", tag, ".json")


def judge_raw_dir(run_dir: Path, tag: str | None = None) -> Path:
    return Path(run_dir) / ("judge_raw" if not tag else f"judge_raw_{_safe_tag(tag)}")


def events_path(run_dir: Path, tag: str | None = None, task: str | None = None) -> Path:
    """Parsed reasoning-event rows (behavioral primitive pipeline), paralleling ``scores.<tag>``.
    ``task`` namespaces a multi-task run dir (e.g. a ``*_main`` cell holds identify + predict): with
    it the file is ``events.<tag>.<task>.jsonl``, so different tasks in one dir don't clobber."""
    name = _tagged("events", tag, "")
    if task:
        name = f"{name}.{_safe_tag(task)}"
    return Path(run_dir) / f"{name}.jsonl"


def event_meta_path(run_dir: Path, tag: str | None = None, task: str | None = None) -> Path:
    """Per-trace segmentation QA (fidelity, event count, truncation) for the event pipeline."""
    name = _tagged("event_meta", tag, "")
    if task:
        name = f"{name}.{_safe_tag(task)}"
    return Path(run_dir) / f"{name}.jsonl"


def events_raw_dir(run_dir: Path, tag: str | None = None, task: str | None = None) -> Path:
    """Cached raw LLM segmentation/labeling outputs, paralleling ``responses_raw_<tag>/``."""
    base = "events_raw" if not tag else f"events_raw_{_safe_tag(tag)}"
    return Path(run_dir) / (base if not task else f"{base}_{_safe_tag(task)}")


def read_jsonl(path: Path, required: bool = False) -> list[dict]:
    """Read a jsonl file into a list of dicts. A missing file returns ``[]`` (resume-friendly);
    pass ``required=True`` to instead raise ``FileNotFoundError`` (for inputs a run cannot proceed
    without, e.g. a judge's ``probes.jsonl``)."""
    path = Path(path)
    if not path.exists():
        if required:
            raise FileNotFoundError(f"missing {path.name} in {path.parent}")
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def done_keys(path: Path, keys: tuple[str, ...] = ("scenario_id", "task")) -> set[DoneKey]:
    """Set of ``(scenario_id, task)`` tuples already present in a jsonl file, for skip-on-resume."""
    return {tuple(r.get(k) for k in keys) for r in read_jsonl(path)}


class JsonlAppender:
    """Append-and-flush jsonl writer: each row is written and flushed to the OS immediately so a
    dropped connection / killed process keeps every completed row. Use as a context manager."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.path.open("a", encoding="utf-8")

    def write(self, row: dict) -> None:
        self._f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> JsonlAppender:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
