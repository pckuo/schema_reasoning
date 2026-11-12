"""Stage 3 pilot: inject the STOP direction after the task is already solved.

Population: phi traces that are judge-all-right, reach ``T_solved``, and then **do not stop** --
the case the whole experiment is aimed at. Injection begins at the run end where the last gold
(person, role) pair is completed, so nothing is truncated before the answer exists.

Arms are norm-matched in residual space by construction (every direction is unit-norm), so the
dose is expressed as a fraction of the median layer-16 run-end state norm and is identical across
arms. The shuffled-label direction is the primary null: it is built through the same 35 heads, so
it is matched in subspace as well as in norm.

The control arm is the alpha=0 resume, never the stored trace -- see ``inject`` for why the corpus
cannot be reproduced token for token.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ...core.model import load_model
from ..head_fv_models import active as active_model
from . import events, inject, lexicon, runframe

DOSES = (0.25, 0.5, 1.0)
ARMS = ("stop", "shuffled")
BUDGET = 512
CORPUS_CAP = 8192   # the completion-token budget the corpus itself was generated under
LAYER = 16
SCHEDULE = "sustained"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def population(built, cohort="capped", limit=None, seed=0, lexicon_variant="extended",
               require_all_right=True):
    """Judge-all-right and solved; ``capped`` still reasoning at the cap, ``terminating`` not.

    The two cohorts answer different questions. Only ``capped`` has room to *gain* a termination;
    only ``terminating`` has room to lose or delay one, which is where a negative dose can be
    falsified at all.
    """
    if lexicon_variant == "extended":
        table, _ = lexicon.solved_table(built)
    else:
        table, _ = events.build(built)
    # The injection point is defined by the *behavioural* solved marker -- every gold
    # (person, role) pair stated. `all_right` is the judge's separate verdict on the final
    # answer; requiring it is extra conservatism that costs half the capped population.
    solved = table[table["solved"] & table["all_right"]] if require_all_right \
        else table[table["solved"]]
    keep = ~solved["terminating"] if cohort == "capped" else solved["terminating"]
    out = solved[keep].sort_values("item").reset_index(drop=True)
    if limit and len(out) > limit:
        out = out.sample(limit, random_state=seed).sort_values("item").reset_index(drop=True)
    return out


def injection_point(trace, t_solved):
    """Token index just past the run end at which the last gold pair completed."""
    _, starts, durations = runframe.runs_of(trace.z)
    ends = starts + durations - 1
    require(0 <= t_solved < len(ends), "t_solved outside the run list")
    return int(trace.pos[ends[t_solved]]) + 1


def switch_profile(tail_tokens, roles):
    """Role-changing person switches in the continuation.

    Carry-forward the most recently named gold person, as the screen's ``schema_state`` does, and
    count transitions to a person holding a *different* role -- the same event its ``is_new`` mask
    marks. This is the positive control the switch arm needs: a direction that fails to cause STOP
    is only informative if it demonstrably does its own job.
    """
    named = [(q["value"], q["role_id"]) for q in roles
             if q["value"][:1].isupper() and " " not in q["value"]]
    hits = []
    for value, role in named:
        for end in events.find_spans(tail_tokens, value):
            hits.append((end, role))
    hits.sort()
    switches = [pos for i, (pos, role) in enumerate(hits)
                if i > 0 and role != hits[i - 1][1]]
    per_100 = 100.0 * len(switches) / max(len(tail_tokens), 1)
    return dict(n_switches=len(switches), switches_per_100_tokens=float(per_100),
                first_switch_token=int(switches[0]) if switches else -1,
                n_person_mentions=len(hits))


def tail_stats(tokenizer, generated, prefix_text, roles):
    text = tokenizer.decode(generated, skip_special_tokens=False)
    whole = prefix_text + text
    _, pairs, n_named = events.progress(
        [runframe.clean(t) for t in tokenizer.tokenize(whole)], roles)
    tail_tokens = [runframe.clean(t) for t in tokenizer.tokenize(text)]
    # switching outcome: how much of the cast the continuation revisits, and how soon
    named = [q["value"] for q in roles
             if q["value"][:1].isupper() and " " not in q["value"]]
    first = []
    for value in named:
        spans = events.find_spans(tail_tokens, value)
        if spans:
            first.append(spans[0])
    distinct = len(set(generated)) / max(len(generated), 1)
    return dict(closed_think="</think>" in text, tail_chars=len(text),
                distinct_token_ratio=float(distinct),
                pairs_complete=bool(n_named > 0 and pairs[-1] >= n_named),
                names_in_tail=int(len(first)),
                first_name_token=int(min(first)) if first else -1,
                **switch_profile(tail_tokens, roles),
                tail_text=text, tail_preview=text[:240].replace("\n", " "))


def run(vectors_dir: Path, out: Path, device="cuda:0", budget=BUDGET, limit=None,
        arms=ARMS, doses=DOSES, layer=LAYER, cohort="capped", lexicon_variant="extended",
        schedule=SCHEDULE, require_all_right=True, draws=1, temperature=0.0, top_p=0.95):
    out.mkdir(parents=True, exist_ok=True)
    directions = dict(np.load(vectors_dir / "directions.npz"))
    manifest = json.loads((vectors_dir / "manifest.json").read_text())
    state_norm = float(manifest["median_l16_state_norm"])

    built = runframe.build()
    targets = population(built, cohort=cohort, limit=limit, lexicon_variant=lexicon_variant,
                         require_all_right=require_all_right)
    require(len(targets), "empty intervention population")
    responses = built["responses"]
    gold = {(r.item_id, r.model): json.loads(r.gold)["roles"] for r in responses.itertuples()}

    tokens = pd.read_parquet(runframe.BUNDLE / "tokens.parquet",
                             columns=["item_id", "model", "task_family", "position",
                                      "token_id", "segment"])
    tokens = tokens[(tokens.model == inject.MODEL) & (tokens.task_family == runframe.TASK)]
    prompt_lens = inject.prompt_lengths(tokens)

    spec = active_model()
    lm = load_model(spec.model, revision=spec.revision, device_map=device)
    eos = set(spec.eos_ids)
    print(f"population {len(targets)} {cohort}+solved items ({lexicon_variant} lexicon) | "
          f"budget {budget} tokens | layer {layer} | schedule {schedule} | "
          f"state norm {state_norm:.1f}", flush=True)

    plan = [("baseline", 0.0)]
    plan += [(arm, dose) for arm in arms for dose in doses]
    rows = []
    for _, target in targets.iterrows():
        item = target["item"]
        trace = built["traces"][(inject.MODEL, item)]
        ids = inject.load_ids(tokens, item)
        prefix_len = injection_point(trace, int(target["t_solved"]))
        prefix_text = lm.tokenizer.decode(ids[:prefix_len], skip_special_tokens=False)
        prompt_len = int(prompt_lens[item])
        # budget < 0 means "match the corpus cap": whatever completion budget the original
        # generation had left at this point, so `terminated` means *before the real cap*.
        limit_tokens = (budget if budget > 0
                        else max(CORPUS_CAP - (prefix_len - prompt_len), 1))
        for draw in range(draws):
          # one seed per (item, draw), shared by every arm -> common random numbers
          seed = (abs(hash((item, draw))) % (2 ** 31)) if temperature > 0 else None
          for arm, dose in plan:
            vector = None if arm == "baseline" else directions[arm]
            alpha = dose * state_norm
            result = inject.resume_generate(
                lm, ids, prefix_len, vector=vector, alpha=alpha, layer=layer,
                schedule=schedule, budget=limit_tokens, eos_ids=eos,
                temperature=temperature, top_p=top_p, seed=seed)
            stats = tail_stats(lm.tokenizer, result["generated"], prefix_text,
                              gold[(item, inject.MODEL)])
            rows.append(dict(item=item, arm=arm, dose=dose, alpha=alpha, layer=layer,
                             draw=draw, temperature=temperature,
                             cohort=cohort, lexicon=lexicon_variant, schedule=schedule,
                             all_right=bool(target["all_right"]),
                             t_solved=int(target["t_solved"]), n_runs=int(target["n_runs"]),
                             prefix_len=prefix_len, budget=limit_tokens,
                             stop_reason=result["stop_reason"],
                             n_generated=len(result["generated"]),
                             # exact ids: re-tokenising `tail_text` does not always round-trip
                             # (decode -> encode can split differently), which leaves the
                             # downstream primitive labelling reading a slightly different token
                             # grid than the model actually produced
                             generated_ids=list(result["generated"]),
                             terminated=result["stop_reason"] == "eos",
                             applied=result["applied"],
                             mean_arrived_norm=result["mean_arrived_norm"], **stats))
            print(f"  {item} d{draw} {arm:9s} dose {dose:>4} -> "
                  f"{result['stop_reason']:6s} {len(result['generated']):4d} tok  "
                  f"think_closed={stats['closed_think']}", flush=True)
    table = pd.DataFrame(rows)
    table.to_parquet(out / "pilot.parquet")
    summary = summarize(table)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def summarize(table):
    out = {"n_items": int(table["item"].nunique()),
           "max_budget": int(table["budget"].max()) if "budget" in table else None,
           "median_budget": float(table["budget"].median()) if "budget" in table else None}
    grouped = []
    for (arm, dose), group in table.groupby(["arm", "dose"]):
        grouped.append(dict(
            arm=arm, dose=float(dose), n=int(len(group)),
            terminated=float(group["terminated"].mean()),
            median_tokens=float(group["n_generated"].median()),
            closed_think=float(group["closed_think"].mean()),
            pairs_complete=float(group["pairs_complete"].mean()),
            distinct_token_ratio=float(group["distinct_token_ratio"].mean()),
        ))
    out["by_arm"] = sorted(grouped, key=lambda r: (r["arm"], r["dose"]))
    base = table[table["arm"] == "baseline"]
    out["baseline_terminated"] = float(base["terminated"].mean())
    paired = []
    for dose in sorted(table["dose"].unique()):
        if dose == 0.0:
            continue
        for arm in ARMS:
            sub = table[(table["arm"] == arm) & (table["dose"] == dose)]
            merged = sub.merge(base[["item", "n_generated", "terminated"]], on="item",
                               suffixes=("", "_base"))
            paired.append(dict(
                arm=arm, dose=float(dose),
                delta_tokens_median=float((merged["n_generated"]
                                           - merged["n_generated_base"]).median()),
                extra_terminations=int((merged["terminated"]
                                        & ~merged["terminated_base"]).sum()),
                lost_terminations=int((~merged["terminated"]
                                       & merged["terminated_base"]).sum()),
                n_pairs=int(len(merged))))
    out["paired_vs_baseline"] = paired
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--budget", type=int, default=BUDGET)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--doses", nargs="+", type=float, default=list(DOSES))
    parser.add_argument("--layer", type=int, default=LAYER)
    parser.add_argument("--cohort", default="capped", choices=("capped", "terminating"))
    parser.add_argument("--lexicon", default="extended", choices=("extended", "canonical"))
    parser.add_argument("--schedule", default=SCHEDULE, choices=("sustained", "once"))
    parser.add_argument("--draws", type=int, default=1,
                        help="stochastic replicates per item (needs --temperature > 0)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--all-solved", action="store_true",
                        help="drop the judge-all-right filter: every capped+solved trace")
    args = parser.parse_args()
    run(args.vectors, args.out, device=args.device, budget=args.budget, limit=args.limit,
        arms=args.arms, doses=args.doses, layer=args.layer, cohort=args.cohort,
        lexicon_variant=args.lexicon, schedule=args.schedule,
        require_all_right=not args.all_solved, draws=args.draws,
        temperature=args.temperature, top_p=args.top_p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
