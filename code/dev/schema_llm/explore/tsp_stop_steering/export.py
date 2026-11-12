"""Flatten a TSP transfer run into one CSV row per generated response.

Every response the run produced -- the 9 sampled baselines and every injected continuation -- with
the task prompt, the injection parameters, the outcome metrics and the text, so the table stands
alone without the parquets or the manifest.

Two conventions worth stating because they shape the columns:

* **Baselines are rows too.** They are what the injected arms are forked from and what the alpha=0
  control re-measures, so dropping them would leave the table unable to explain its own offsets.
  Injection-only fields are empty on those rows and ``kind`` says which is which.
* **``response`` is the whole response**, inherited prefix and injected continuation concatenated,
  as the model would have emitted it. The injection point is therefore not marked in the text; its
  position is recoverable only as a fraction, from ``location``.
* **One outcome column.** ``correct`` is the strict criterion -- a genuine tour at the optimum. The
  diagnostic breakdown (``answered``, ``valid_tour``, ``stated_matches_path``, stated vs recomputed
  distance) and the per-run mechanics (``n_generated``, ``applied``, ``seed``, ...) stay in
  ``injections.parquet``, where the analysis reads them.

Verdicts are recomputed from the stored text via ``analyze.rescore``, so the CSV always reflects
the current scorer rather than whatever was in force when the run executed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from . import analyze, tasks

COLUMNS = [
    # identity
    "response_id", "kind",
    # task
    "level", "uid", "n_cities", "ground_truth", "prompt_hash",
    # which sampled baseline this row is, or was forked from
    "base_draw",
    # injection parameters (empty on baseline rows)
    "arm", "vector", "dose", "layer", "schedule", "location", "injection_draw",
    # generation. prefix + generated split out as well as the total, because three figures need
    # them: the depth panel divides by the control's *remaining* generation, the
    # accuracy-vs-length panel plots tokens generated after injection, and the injection-point
    # panel plots the prefix itself. `budget` stays derivable as `cap - prefix_tokens`.
    "cap", "prefix_tokens", "generated_tokens", "total_tokens", "terminated", "closed_think",
    # verdict. `answered` is kept alongside `correct` because "did it commit to an answer at all"
    # is a separate outcome from "was the answer optimal", and the two move in opposite directions
    # under injection. The finer breakdown (valid_tour, stated vs recomputed distance) stays in
    # the parquets.
    "answered", "correct",
    # text
    "prompt", "response",
]


def build(directory: Path):
    manifest = json.loads((directory / "manifest.json").read_text())
    baselines = pd.read_parquet(directory / "baselines.parquet")
    frame, _ = analyze.load_injections(directory)
    frame = analyze.rescore(directory, frame)

    selected = tasks.tasks_by_uid(set(baselines["uid"]) | set(frame["uid"]))
    tokenizer = AutoTokenizer.from_pretrained(manifest["model"], revision=manifest["revision"])
    ids = {(int(r.uid), int(r.draw)): np.asarray(r.token_ids, dtype=np.int64)
           for r in baselines.itertuples()}
    shared = dict(layer=manifest["layer"], schedule=manifest["schedule"], cap=manifest["cap"])

    rows = []
    for row in baselines.itertuples():
        task = selected[int(row.uid)]
        rows.append(dict(
            response_id=f"baseline-uid{row.uid}-d{row.draw}", kind="baseline",
            level=row.level, uid=int(row.uid), n_cities=task["n_cities"],
            ground_truth=task["ground_truth"], prompt_hash=task["prompt_hash"],
            base_draw=int(row.draw),
            arm="", vector="", dose=None, location=None, injection_draw=None,
            prefix_tokens=0, generated_tokens=int(row.n_generated),
            total_tokens=int(row.n_generated),
            terminated=bool(row.terminated), closed_think=bool(row.closed_think),
            answered=bool(row.answered), correct=bool(row.correct),
            prompt=task["prompt"], response=row.text, **shared))

    prefixes: dict[tuple[int, int, int], str] = {}
    for row in frame.itertuples():
        task = selected[int(row.uid)]
        key = (int(row.uid), int(row.base_draw), int(row.offset))
        if key not in prefixes:
            prefixes[key] = tokenizer.decode(ids[key[:2]][:key[2]], skip_special_tokens=False)
        rows.append(dict(
            response_id=(f"inject-uid{row.uid}-b{row.base_draw}-L{row.location:g}"
                         f"-{row.arm}{row.dose:g}-d{row.draw}"),
            kind="injection",
            level=row.level, uid=int(row.uid), n_cities=task["n_cities"],
            ground_truth=task["ground_truth"], prompt_hash=task["prompt_hash"],
            base_draw=int(row.base_draw),
            arm=row.arm,
            # the control is an alpha=0 fork, so it injects no vector at all
            vector="" if row.arm == "baseline" else f"v_{row.arm}",
            dose=float(row.dose), location=float(row.location),
            injection_draw=int(row.draw),
            prefix_tokens=int(row.offset), generated_tokens=int(row.n_generated),
            total_tokens=int(row.total_tokens), terminated=bool(row.terminated),
            closed_think=bool(row.closed_think), answered=bool(row.answered),
            correct=bool(row.correct),
            prompt=task["prompt"], response=prefixes[key] + row.tail_text, **shared))

    table = pd.DataFrame(rows)
    table["arm"] = table["arm"].replace({"baseline": "control"})
    table["model"] = manifest["model"]
    return table[[*COLUMNS, "model"]]


def run(directory: Path, out: Path | None = None):
    out = out or (directory / "responses.csv")
    table = build(directory)
    table.to_csv(out, index=False)
    size = out.stat().st_size / 1e6
    print(f"wrote {out}  ({len(table)} rows, {len(table.columns)} columns, {size:.1f} MB)")
    print(table.groupby(["kind", "arm"]).size().to_string())
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    run(args.dir, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
