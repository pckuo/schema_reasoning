"""Flatten the corpus-budget STOP-steering runs into one CSV row per generated response.

Every response behind the ``20260922-fullcap-*`` figures -- the unperturbed corpus trace, the
alpha=0 resume that anchors each item, and every injected continuation -- with the task prompt,
the injection parameters, the outcome metrics and the text, so the table stands alone without the
parquets.

Four conventions worth stating because they shape the columns:

* **The corpus trace is a row too** (``kind="corpus"``). It is what the injection prefix is cut
  from, so without it the table cannot explain its own ``prefix_tokens``. It is *not* a control:
  it came from the batched generation path, so a greedy single-sequence resume diverges from it
  within ~37 tokens. The control is the ``alpha=0`` resume, ``kind="control"``.
* **``response`` is the whole completion**, inherited reasoning prefix and injected continuation
  concatenated, as the model would have emitted it. ``generated`` is the continuation *alone* --
  the part the injection actually produced, and the quantity every effect is measured on. Both are
  carried because neither is recoverable from the other without the tokenizer: the split falls at
  ``prefix_tokens`` tokens, not at any marker in the text. ``prompt`` is separate from both.
* **One outcome column.** ``correct`` is the strict criterion -- a final answer was emitted and
  *every* gold person-role pair in it is right. It is scored on the block after ``</think>``, not
  the whole response: injection starts at ``T_solved``, so the shared prefix already contains
  every gold pair and whole-response scoring would largely measure the prefix. ``answered`` is
  kept beside it because "did it commit to an answer at all" is a separate outcome that moves the
  other way under injection.
* **Verdicts are recomputed from the stored text** by the current scorer, so the CSV reflects
  today's definition rather than whatever was in force when the run executed.

Runs from the repo root: the bundle path is relative to the working directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from ..head_fv_models import active as active_model
from . import inject, lexicon, runframe

# the completion budget the corpus itself was generated under; `budget` in the pilot is what was
# left at the injection point, so `budget + prefix_tokens` must come back to this
CORPUS_CAP = 8192

RUNS = [
    "20260921-fullcap-dose-specificity",
    "20260921-fullcap-layer08",
    "20260921-fullcap-layer24",
    "20260921-fullcap-sampling",
    "20260921-fullcap-oneshot",
]

COLUMNS = [
    # identity
    "response_id", "kind", "run",
    # task
    "item", "schema", "cohort", "t_solved", "n_runs",
    # injection parameters (empty on corpus rows). `temperature` is not carried: it is fixed per
    # run, so `run` already determines it.
    "arm", "vector", "dose", "alpha", "layer", "schedule", "draw",
    # generation. prefix and generated are split out as well as the total because they answer
    # different questions: `generated_tokens` is the causal quantity the effect is measured on,
    # `prefix_tokens` is where the injection happened, and `total_tokens` is what a reader waits
    # for. `budget` stays derivable as `cap - prefix_tokens`.
    "cap", "prefix_tokens", "generated_tokens", "total_tokens",
    "terminated", "closed_think", "stop_reason",
    # verdict
    "answered", "correct",
    # text
    "prompt", "generated", "response",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_context():
    """Bundle-derived context: gold roles, prompts, token ids and the prompt/completion split."""
    built = runframe.build()
    responses = built["responses"]
    gold = {(r.item_id, r.model): json.loads(r.gold)["roles"] for r in responses.itertuples()}
    schema = {(r.item_id, r.model): json.loads(r.gold)["schema"]
              for r in responses.itertuples()}
    passages = {(r.item_id, r.model): lexicon.norm(r.passage_text)
                for r in responses.itertuples()}
    prompts = {(r.item_id, r.model): r.prompt_text for r in responses.itertuples()}
    corpus = {(r.item_id, r.model): r.response_text for r in responses.itertuples()}
    _, extended = lexicon.build_lexicons(gold)
    tokens = pd.read_parquet(runframe.BUNDLE / "tokens.parquet",
                             columns=["item_id", "model", "task_family", "position",
                                      "token_id", "segment"])
    tokens = tokens[(tokens.model == "phi4mr") & (tokens.task_family == runframe.TASK)]
    return dict(gold=gold, schema=schema, passages=passages, prompts=prompts, corpus=corpus,
                extended=extended, tokens=tokens)


def score_text(tokenizer, text, roles, passage_norm, extended):
    """Strict verdict on a whole completion, reading only what follows the last ``</think>``."""
    marker = text.rfind("</think>")
    if marker < 0:
        return dict(answered=False, correct=False)
    raw = tokenizer.tokenize(text[marker + len("</think>"):])
    clean = [runframe.clean(t) for t in raw]
    state = lexicon.assignment_state(clean, raw, roles, extended, passage_norm)
    return dict(answered=True, correct=bool(state["all_correct"]))


def build(runs=None, root: Path = Path("results/schema_llm/analysis/stop_steering")):
    runs = list(runs or RUNS)
    ctx = load_context()
    spec = active_model()
    tokenizer = AutoTokenizer.from_pretrained(spec.model, revision=spec.revision)
    model = inject.MODEL

    id_cache: dict[str, np.ndarray] = {}
    prompt_len_cache: dict[str, int] = {}
    prompt_cache: dict[str, str] = {}
    prefix_cache: dict[tuple[str, int], str] = {}

    def context_for(item):
        if item not in id_cache:
            d = ctx["tokens"][ctx["tokens"].item_id == item].sort_values("position")
            require(len(d), f"{item}: no tokens in the bundle")
            id_cache[item] = d["token_id"].values.astype(np.int64)
            prompt_len_cache[item] = int((d["segment"] == "prompt").sum())
            prompt_cache[item] = tokenizer.decode(id_cache[item][:prompt_len_cache[item]],
                                                  skip_special_tokens=False)
        return id_cache[item], prompt_len_cache[item], prompt_cache[item]

    rows = []
    items_seen: set[str] = set()
    for name in runs:
        path = root / name / "pilot.parquet"
        if not path.exists():
            print(f"  skip {name}: no pilot.parquet")
            continue
        table = pd.read_parquet(path)
        require("tail_text" in table, f"{name}: no tail_text; cannot export the response")
        for row in table.itertuples():
            item = row.item
            ids, prompt_len, prompt = context_for(item)
            items_seen.add(item)
            prefix_len = int(row.prefix_len)
            key = (item, prefix_len)
            if key not in prefix_cache:
                prefix_cache[key] = tokenizer.decode(ids[prompt_len:prefix_len],
                                                     skip_special_tokens=False)
            generated = str(row.tail_text)
            response = prefix_cache[key] + generated
            verdict = score_text(tokenizer, response, ctx["gold"][(item, model)],
                                 ctx["passages"][(item, model)], ctx["extended"])
            control = row.arm == "baseline"
            prefix_tokens = prefix_len - prompt_len
            require(int(row.budget) + prefix_tokens == CORPUS_CAP,
                    f"{name}/{item}: budget {row.budget} + prefix {prefix_tokens} "
                    f"!= {CORPUS_CAP}; the run is not at the corpus budget")
            rows.append(dict(
                response_id=(f"{name}-{item}-{'control' if control else row.arm}"
                             f"{'' if control else f'{row.dose:g}'}"
                             f"-L{int(row.layer)}-{row.schedule}-d{int(row.draw)}"),
                kind="control" if control else "injection", run=name,
                item=item, schema=ctx["schema"][(item, model)], cohort=str(row.cohort),
                t_solved=int(row.t_solved), n_runs=int(row.n_runs),
                arm="control" if control else row.arm,
                # the control is an alpha=0 resume, so it injects no vector at all
                vector="" if control else f"v_{row.arm}",
                dose=float(row.dose), alpha=float(row.alpha), layer=int(row.layer),
                schedule=str(row.schedule), draw=int(row.draw),
                cap=CORPUS_CAP, prefix_tokens=prefix_tokens,
                generated_tokens=int(row.n_generated),
                total_tokens=prefix_tokens + int(row.n_generated),
                terminated=bool(row.terminated), closed_think=bool(row.closed_think),
                stop_reason=str(row.stop_reason),
                answered=verdict["answered"], correct=verdict["correct"],
                prompt=prompt, generated=generated, response=response))

    require(rows, "no rows exported")
    # the unperturbed trace each prefix was cut from, so the offsets in the table are explicable
    for item in sorted(items_seen):
        ids, prompt_len, prompt = context_for(item)
        text = ctx["corpus"][(item, model)]
        verdict = score_text(tokenizer, text, ctx["gold"][(item, model)],
                             ctx["passages"][(item, model)], ctx["extended"])
        closed = "</think>" in text
        # nothing precedes the injection point on a corpus row, so the whole completion *is* the
        # generated text and the same prefix/generated/total arithmetic holds with prefix = 0
        completion = len(ids) - prompt_len
        rows.append(dict(
            response_id=f"corpus-{item}", kind="corpus", run="bundle",
            item=item, schema=ctx["schema"][(item, model)], cohort="", t_solved=None,
            n_runs=None,
            arm="", vector="", dose=None, alpha=None, layer=None, schedule="", draw=None,
            cap=CORPUS_CAP, prefix_tokens=0, generated_tokens=completion,
            total_tokens=completion,
            terminated=closed, closed_think=closed, stop_reason="",
            answered=verdict["answered"], correct=verdict["correct"],
            prompt=prompt, generated=text, response=text))

    frame = pd.DataFrame(rows)
    frame["model"] = spec.model
    return frame[[*COLUMNS, "model"]]


def run(out: Path, runs=None, root: Path = Path("results/schema_llm/analysis/stop_steering")):
    out.parent.mkdir(parents=True, exist_ok=True)
    table = build(runs, root)
    table.to_csv(out, index=False)
    size = out.stat().st_size / 1e6
    print(f"wrote {out}  ({len(table)} rows, {len(table.columns)} columns, {size:.1f} MB)")
    print(table.groupby(["kind", "arm"]).size().to_string())
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runs", nargs="+", default=None,
                        help=f"run directory names; default {' '.join(RUNS)}")
    parser.add_argument("--root", type=Path,
                        default=Path("results/schema_llm/analysis/stop_steering"))
    args = parser.parse_args()
    run(args.out, args.runs, args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
