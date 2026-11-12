"""Answer accuracy for intervened continuations -- request section 3, "final-answer accuracy".

The pilot records whether the gold (person, role) pairs are still *present*, which is a content
check, not a correctness check: a continuation can mention every name and role and still assign
them wrongly. This module scores the model's **latest commitment** for each gold name against the
gold role, using the same extended-lexicon assignment the notebook's ``K`` array uses.

Deliberately not an LLM judge. The measure is deterministic, reruns identically, and is the same
instrument already used to define ``T_solved``, so "solved before injection" and "still correct
after injection" are read off one scale rather than two.

Runs on saved output -- it needs no GPU and no regeneration, because ``pilot`` stores the full
continuation text.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from ..head_fv_models import active as active_model
from . import lexicon, runframe


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_context():
    built = runframe.build()
    responses = built["responses"]
    gold = {(r.item_id, r.model): json.loads(r.gold)["roles"] for r in responses.itertuples()}
    passages = {(r.item_id, r.model): lexicon.norm(r.passage_text)
                for r in responses.itertuples()}
    _, extended = lexicon.build_lexicons(gold)
    tokens = pd.read_parquet(runframe.BUNDLE / "tokens.parquet",
                             columns=["item_id", "model", "task_family", "position",
                                      "token_id", "segment"])
    tokens = tokens[(tokens.model == "phi4mr") & (tokens.task_family == runframe.TASK)]
    return built, gold, passages, extended, tokens


def score_row(tokenizer, row, ids, prompt_len, roles, passage_norm, extended,
              scope="answer"):
    """Score the model's role assignments.

    ``scope="answer"`` reads only the text the model emitted **after** ``</think>`` -- the answer
    a reader would actually see. ``scope="response"`` reads the whole thing, prefix included.

    The distinction decides the result here. Injection starts at ``T_solved``, so the shared
    prefix already contains every gold pair; scoring the whole response therefore measures what
    the prefix established and is nearly blind to what the continuation did. An arm that never
    closes ``</think>`` produces no answer at all, which "whole response" scoring silently
    credits and "answer" scoring correctly records as a failure to answer.
    """
    prefix_ids = ids[prompt_len:int(row["prefix_len"])]
    prefix_text = tokenizer.decode(prefix_ids, skip_special_tokens=False)
    tail = str(row["tail_text"])

    if scope == "answer":
        whole = prefix_text + tail
        marker = whole.rfind("</think>")
        if marker < 0:
            return dict(counts=None, n_named=0, n_assigned=0, n_correct=0,
                        accuracy=float("nan"), all_correct=False, answered=False)
        text = whole[marker + len("</think>"):]
    else:
        text = prefix_text + tail

    raw = tokenizer.tokenize(text)
    clean = [runframe.clean(t) for t in raw]
    state = lexicon.assignment_state(clean, raw, roles, extended, passage_norm)
    state["answered"] = True
    return state


def run(pilot_dirs, out: Path, scope: str = "answer"):
    out.mkdir(parents=True, exist_ok=True)
    built, gold, passages, extended, tokens = load_context()
    spec = active_model()
    tokenizer = AutoTokenizer.from_pretrained(spec.model, revision=spec.revision)
    id_cache, prompt_cache = {}, {}
    frames = []
    for directory in pilot_dirs:
        path = Path(directory) / "pilot.parquet"
        if not path.exists():
            print(f"  skip {directory}")
            continue
        table = pd.read_parquet(path)
        require("tail_text" in table, f"{directory}: no tail_text; re-run to score accuracy")
        rows = []
        for _, row in table.iterrows():
            item = row["item"]
            if item not in id_cache:
                d = tokens[tokens.item_id == item].sort_values("position")
                id_cache[item] = d["token_id"].values.astype(np.int64)
                prompt_cache[item] = int((d["segment"] == "prompt").sum())
            state = score_row(tokenizer, row, id_cache[item], prompt_cache[item],
                              gold[(item, "phi4mr")], passages[(item, "phi4mr")], extended,
                              scope=scope)
            whole = score_row(tokenizer, row, id_cache[item], prompt_cache[item],
                              gold[(item, "phi4mr")], passages[(item, "phi4mr")], extended,
                              scope="response")
            rows.append(dict(item=item, arm=row["arm"], dose=float(row["dose"]),
                             answered=bool(state["answered"]),
                             response_accuracy=float(whole["accuracy"]),
                             layer=int(row.get("layer", 16)),
                             schedule=str(row.get("schedule", "sustained")),
                             terminated=bool(row["terminated"]),
                             n_generated=int(row["n_generated"]),
                             accuracy=float(state["accuracy"]),
                             all_correct=bool(state["all_correct"]),
                             n_assigned=int(state["n_assigned"]),
                             n_named=int(state["n_named"])))
        frame = pd.DataFrame(rows)
        frame["run"] = Path(directory).name
        frames.append(frame)
    require(frames, "no scorable runs")
    table = pd.concat(frames, ignore_index=True)
    table.to_parquet(out / "accuracy.parquet")

    summary = (table.groupby(["run", "schedule", "layer", "arm", "dose"])
               .agg(n=("item", "size"), accuracy=("accuracy", "mean"),
                    all_correct=("all_correct", "mean"), terminated=("terminated", "mean"))
               .reset_index())
    summary.to_csv(out / "accuracy_by_arm.csv", index=False)
    print(summary.to_string(index=False))

    base = table[table["arm"] == "baseline"][["item", "run", "accuracy", "all_correct"]]
    base = base.rename(columns={"accuracy": "baseline_accuracy",
                                "all_correct": "baseline_all_correct"})
    merged = table.merge(base, on=["item", "run"], how="left")
    delta = (merged.groupby(["run", "arm", "dose"])
             .apply(lambda g: pd.Series({
                 "delta_accuracy": float((g["accuracy"] - g["baseline_accuracy"]).mean()),
                 "lost_all_correct": int((g["baseline_all_correct"] & ~g["all_correct"]).sum()),
                 "gained_all_correct": int((~g["baseline_all_correct"] & g["all_correct"]).sum()),
             }), include_groups=False)
             .reset_index())
    delta.to_csv(out / "accuracy_delta.csv", index=False)
    print("\npaired change against the alpha=0 resume:")
    print(delta.to_string(index=False))
    return summary


def load_accuracy(path: Path, run: str | None = None) -> pd.DataFrame:
    """Read a scored table and make the answered/unanswered distinction explicit.

    Under ``scope="answer"`` an arm that never closes ``</think>`` has ``accuracy = NaN``. Taking
    a plain mean of that column silently conditions on having answered, which flatters exactly the
    arms that answer least: an arm that answers once and gets it right scores 1.00. Two columns are
    therefore returned, and figures must say which they use.

    ``accuracy_cond``  NaN where unanswered -- "when it answers, how good is the answer"
    ``accuracy_unc``   0 where unanswered   -- "per item, does a correct answer come out at all"
    """
    table = pd.read_parquet(path)
    if run is not None and "run" in table:
        table = table[table["run"] == run]
    require(len(table), f"{path}: no rows" + (f" for run {run}" if run else ""))
    table = table.assign(
        accuracy_cond=table["accuracy"].astype(float),
        accuracy_unc=table["accuracy"].astype(float).fillna(0.0),
        answered=table["answered"].astype(float),
        all_correct=table["all_correct"].astype(float),
    )
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dirs", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scope", default="answer", choices=("answer", "response"),
                        help="score the final answer block, or the whole response")
    args = parser.parse_args()
    run(args.dirs, args.out, scope=args.scope)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
