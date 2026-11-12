"""Analysis for the TSP transfer test.

Two things this enforces that a naive table would get wrong:

**Draws are clustered, not independent.** A cell is 3 nucleus-sampling draws from one
``(task, baseline trace, offset)``; they share a prefix and a problem, so 27 units x 3 draws are
not 81 data points. Inference runs on per-unit means over draws (n = 27) and the draw-level table
is reported descriptively alongside the intraclass correlation.

**The comparison is paired at the draw, through common random numbers.** Every arm at a given
``(uid, baseline draw, offset, draw)`` was generated from the same seed and the same forked cache,
so the only difference is ``alpha * v``. Deltas are therefore formed per draw and then averaged,
not formed between arm means.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import tasks as tsp

UNIT = ["uid", "base_draw", "location"]
CELL = [*UNIT, "draw"]
OUTCOMES = ("total_tokens", "terminated", "correct", "answered", "closed_think")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def paired(frame):
    """One row per (cell, treatment arm) carrying the control's outcomes alongside."""
    control = frame[frame["arm"] == "baseline"]
    require(len(control), "no alpha=0 control arm in the table")
    require(not control.duplicated(CELL).any(), "control is not unique per cell")
    keep = [*CELL, *OUTCOMES]
    treated = frame[frame["arm"] != "baseline"]
    merged = treated.merge(control[keep], on=CELL, suffixes=("", "_ctl"))
    require(len(merged) == len(treated), "every treated run must find its control")
    for outcome in OUTCOMES:
        delta = merged[outcome].astype(float) - merged[f"{outcome}_ctl"].astype(float)
        merged[f"d_{outcome}"] = delta
    merged["ratio_tokens"] = merged["total_tokens"] / merged["total_tokens_ctl"].clip(lower=1)
    # The prefix is shared by both arms, so it cancels in the paired difference but NOT in a
    # ratio of totals. The depth question -- does injecting later do more or less? -- is only
    # meaningful on what was still to be generated, hence the control's post-injection length.
    generated_ctl = (merged["total_tokens_ctl"] - merged["offset"]).clip(lower=1)
    merged["gen_ctl"] = generated_ctl
    merged["remaining_ratio"] = merged["n_generated"] / generated_ctl
    return merged


def depth_trend(collapsed, arm="stop", dose=0.25):
    """Does the effect strengthen with injection depth? Tested on two denominators.

    Absolute token savings shrink with depth for a near-trivial reason -- less remains to cut --
    so the claim "largest earliest" is an artefact of the denominator. The informative measure is
    the share of the *remaining* generation removed.
    """
    group = collapsed[(collapsed["arm"] == arm) & (collapsed["dose"] == dose)]
    if len(group) < 4:
        return {}
    out = {"arm": arm, "dose": float(dose), "n_units": int(len(group))}
    for field, label in [("remaining_ratio", "remaining"), ("ratio_tokens", "total")]:
        rho, pvalue = stats.spearmanr(group["location"], group[field])
        out[f"spearman_{label}_rho"] = float(rho)
        out[f"spearman_{label}_p"] = float(pvalue)
    by_depth = (group.groupby("location")
                .agg(remaining_ratio=("remaining_ratio", "mean"),
                     total_ratio=("ratio_tokens", "mean"),
                     n_generated=("n_generated", "mean"),
                     gen_ctl=("gen_ctl", "mean")).round(4))
    out["by_depth"] = by_depth.reset_index().to_dict("records")
    return out


def intraclass_correlation(frame, value="total_tokens"):
    """Share of variance between units rather than between draws within a unit."""
    groups = [g[value].astype(float).values for _, g in frame.groupby(UNIT) if len(g) > 1]
    if len(groups) < 2:
        return float("nan")
    within = float(np.mean([g.var(ddof=1) for g in groups]))
    between = float(np.var([g.mean() for g in groups], ddof=1))
    total = within + between
    return float(between / total) if total > 0 else float("nan")


def per_unit(merged):
    """Collapse draws: one row per (unit, arm, dose)."""
    columns = {f"d_{o}": (f"d_{o}", "mean") for o in OUTCOMES}
    columns |= {o: (o, "mean") for o in OUTCOMES}
    columns |= {f"{o}_ctl": (f"{o}_ctl", "mean") for o in OUTCOMES}
    columns["ratio_tokens"] = ("ratio_tokens", "mean")
    columns["remaining_ratio"] = ("remaining_ratio", "mean")
    columns["gen_ctl"] = ("gen_ctl", "mean")
    # tokens generated *after* the injection point, as distinct from the whole response: this is
    # the axis the `stop_steering` accuracy-vs-length panel uses
    columns["n_generated"] = ("n_generated", "mean")
    columns["n_draws"] = ("draw", "nunique")
    return merged.groupby([*UNIT, "arm", "dose", "level"], as_index=False).agg(**columns)


def tests(collapsed, by_location=False):
    keys = ["arm", "dose"] + (["location"] if by_location else [])
    rows = []
    for values, group in collapsed.groupby(keys):
        values = values if isinstance(values, tuple) else (values,)
        delta = group["d_total_tokens"].values
        row = dict(zip(keys, values, strict=True))
        row |= dict(
            n_units=int(len(group)),
            median_tokens=float(group["total_tokens"].median()),
            median_tokens_ctl=float(group["total_tokens_ctl"].median()),
            delta_tokens_median=float(np.median(delta)),
            ratio_tokens_median=float(group["ratio_tokens"].median()),
            terminated=float(group["terminated"].mean()),
            terminated_ctl=float(group["terminated_ctl"].mean()),
            gained_termination=int((group["d_terminated"] > 0).sum()),
            lost_termination=int((group["d_terminated"] < 0).sum()),
            correct=float(group["correct"].mean()),
            correct_ctl=float(group["correct_ctl"].mean()),
            answered=float(group["answered"].mean()),
            answered_ctl=float(group["answered_ctl"].mean()),
            shorter=int((delta < 0).sum()), longer=int((delta > 0).sum()))
        if len(delta) >= 3 and np.any(delta != 0):
            statistic, pvalue = stats.wilcoxon(delta)
            row["wilcoxon_stat"] = float(statistic)
            row["p_tokens"] = float(pvalue)
        else:
            row["wilcoxon_stat"] = float("nan")
            row["p_tokens"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def load_injections(directory: Path):
    """The per-trace shards, which are the source of truth, with the combined table as fallback.

    ``stage_inject`` writes ``injections.parquet`` by concatenating the shards, so it is a derived
    artifact and goes stale the moment a later run adds traces. Preferring it would silently hide
    new data -- which it did: a replication run's first shard was invisible to the analysis until
    this was inverted. Shards are read whenever they exist; the combined file is only used if the
    shard directory is missing.
    """
    shard_dir = directory / "injections"
    shards = sorted(shard_dir.glob("*.parquet")) if shard_dir.exists() else []
    combined = directory / "injections.parquet"
    if not shards:
        require(combined.exists(), f"no injection shards or combined table under {directory}")
        return pd.read_parquet(combined), True
    frame = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)
    fresh = combined.exists() and len(pd.read_parquet(combined, columns=["uid"])) == len(frame)
    return frame, fresh


VERDICT = ("answered", "valid_tour", "stated_matches_path", "correct")
# the numeric companions must be refreshed too, or a rescored row can end up `correct = True`
# while still carrying the previous parser's `tour_distance = NaN`
VERDICT_NUMERIC = ("stated_distance", "tour_distance")


def rescore(directory: Path, frame):
    """Recompute TSP verdicts from the saved continuation text.

    Scoring is cheap and the text is stored, so a scoring fix must never cost a regeneration. This
    matters in practice: the first run of this experiment scored only ``<final_answer>`` and missed
    every answer the model committed in its native ``\\boxed{...}``.

    Scoring the **tail alone** is equivalent to scoring prefix+tail here. The prefix always ends
    strictly inside the first reasoning block (the offsets are at most 75% of the distance to
    ``</think>``), so it cannot contain the model's final commitment; the only answer-shaped string
    it can carry is the prompt's format example, whose ``'INT_TOTAL_DISTANCE'`` placeholder fails
    the integer parse and is rejected.
    """
    selected = tsp.tasks_by_uid(frame["uid"].unique())
    verdicts = [tsp.score(row.tail_text, selected[row.uid]["weight_matrix"],
                          selected[row.uid]["ground_truth"])
                for row in frame.itertuples()]
    out = frame.copy()
    for field in VERDICT:
        out[field] = [bool(v[field]) for v in verdicts]
    for field in VERDICT_NUMERIC:
        out[field] = [v[field] for v in verdicts]
    return out


def run(directory: Path):
    frame, complete = load_injections(directory)
    baselines = pd.read_parquet(directory / "baselines.parquet")
    if not complete:
        done = frame.groupby(["uid", "base_draw"]).ngroups
        print(f"PARTIAL: {done}/{len(baselines)} baseline traces injected so far", flush=True)
    before = float(frame["correct"].mean())
    frame = rescore(directory, frame)
    print(f"rescored from saved text: correct {before:.3f} -> {frame['correct'].mean():.3f}",
          flush=True)
    merged = paired(frame)
    collapsed = per_unit(merged)
    overall = tests(collapsed)
    located = tests(collapsed, by_location=True)

    summary = dict(
        n_runs=int(len(frame)), n_units=int(frame.groupby(UNIT).ngroups), complete=bool(complete),
        n_draws_per_cell=int(frame.groupby([*CELL, "arm", "dose"]).size().max()),
        icc_total_tokens=intraclass_correlation(frame[frame["arm"] == "baseline"]),
        baseline_traces=[dict(level=r.level, uid=int(r.uid), draw=int(r.draw),
                              n_generated=int(r.n_generated), reasoning=int(r.reasoning_tokens),
                              stop_reason=r.stop_reason, correct=bool(r.correct))
                         for r in baselines.itertuples()],
        overall=overall.to_dict("records"), by_location=located.to_dict("records"),
        depth_trend=depth_trend(collapsed))
    (directory / "stats.json").write_text(json.dumps(summary, indent=2, default=float))
    collapsed.to_parquet(directory / "per_unit.parquet")
    overall.to_csv(directory / "overall.csv", index=False)
    located.to_csv(directory / "by_location.csv", index=False)

    pd.set_option("display.width", 200)
    print(f"\n== overall (n units = {summary['n_units']}, draws collapsed) ==")
    print(overall[["arm", "dose", "n_units", "median_tokens_ctl", "median_tokens",
                   "delta_tokens_median", "ratio_tokens_median", "p_tokens", "terminated_ctl",
                   "terminated", "correct_ctl", "correct"]].to_string(index=False))
    print("\n== by injection location ==")
    print(located[["arm", "dose", "location", "n_units", "median_tokens_ctl", "median_tokens",
                   "delta_tokens_median", "p_tokens", "terminated", "correct"]]
          .to_string(index=False))
    print(f"\nICC(total_tokens, control arm) = {summary['icc_total_tokens']:.3f}")
    trend = summary.get("depth_trend") or {}
    if trend:
        print("\n== depth trend, stop@0.25 (share of REMAINING generation removed) ==")
        for row in trend["by_depth"]:
            print(f"  {row['location']:.0%}: removed {1 - row['remaining_ratio']:.1%} of the "
                  f"{row['gen_ctl']:.0f} tokens the control still had "
                  f"(total-length ratio {row['total_ratio']:.3f})")
        print(f"  Spearman(depth, remaining ratio) rho={trend['spearman_remaining_rho']:+.3f} "
              f"p={trend['spearman_remaining_p']:.4f}  |  "
              f"on total ratio rho={trend['spearman_total_rho']:+.3f} "
              f"p={trend['spearman_total_p']:.4f}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
