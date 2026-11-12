"""Analysis for the replicated (stochastic-draw) design.

Greedy decoding gives one path per item, so an item's effect estimate carries unmeasurable luck.
Re-running under nucleus sampling with **common random numbers** -- one seed per (item, draw),
shared by every arm -- gives several paths per item and makes that luck measurable.

The statistical point this module enforces: **k draws from n items are not n*k independent data
points.** They are clustered, and the paired tests operate on items. So inference is run on
*per-item means over draws* (n items, less noise per item), while the draw-level table is reported
descriptively alongside the within-item spread and the intraclass correlation, which says how much
of the total variance is between items rather than between draws.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def require(condition, message):
    if not condition:
        raise ValueError(message)


def intraclass_correlation(frame, value="n_generated"):
    """Share of variance that is between items rather than between draws within an item."""
    groups = [g[value].astype(float).values for _, g in frame.groupby("item") if len(g) > 1]
    if len(groups) < 2:
        return float("nan")
    within = np.mean([g.var(ddof=1) for g in groups])
    between = np.var([g.mean() for g in groups], ddof=1)
    total = within + between
    return float(between / total) if total > 0 else float("nan")


def per_item(frame):
    """Collapse draws to one row per (item, arm, dose)."""
    return (frame.groupby(["item", "arm", "dose"], as_index=False)
            .agg(n_generated=("n_generated", "mean"),
                 terminated=("terminated", "mean"),
                 n_draws=("draw", "nunique"),
                 sd_tokens=("n_generated", lambda s: float(np.std(s, ddof=1)) if len(s) > 1
                            else 0.0)))


def paired_tests(collapsed):
    base = collapsed[collapsed["arm"] == "baseline"].set_index("item")
    rows = []
    for (arm, dose), group in collapsed[collapsed["arm"] != "baseline"].groupby(["arm", "dose"]):
        group = group.set_index("item")
        common = group.index.intersection(base.index)
        if len(common) < 3:
            continue
        delta = (group.loc[common, "n_generated"] - base.loc[common, "n_generated"])
        rate = (group.loc[common, "terminated"] - base.loc[common, "terminated"])
        p = float(stats.wilcoxon(delta).pvalue) if (delta != 0).any() else 1.0
        p_rate = float(stats.wilcoxon(rate).pvalue) if (rate != 0).any() else 1.0
        dz = float(delta.mean() / delta.std(ddof=1)) if delta.std(ddof=1) > 0 else np.nan
        rows.append(dict(arm=arm, dose=float(dose), n_items=int(len(common)),
                         mean_tokens=float(group.loc[common, "n_generated"].mean()),
                         baseline_tokens=float(base.loc[common, "n_generated"].mean()),
                         median_delta_tokens=float(delta.median()),
                         dz=dz, wilcoxon_p=p,
                         termination_rate=float(group.loc[common, "terminated"].mean()),
                         baseline_termination_rate=float(base.loc[common, "terminated"].mean()),
                         termination_p=p_rate))
    return pd.DataFrame(rows)


def draw_level(frame):
    """Descriptive only -- these rows are clustered and must not be tested as independent."""
    return (frame.groupby(["arm", "dose"], as_index=False)
            .agg(draws=("n_generated", "size"),
                 items=("item", "nunique"),
                 terminated=("terminated", "mean"),
                 median_tokens=("n_generated", "median"),
                 mean_tokens=("n_generated", "mean")))


def run(sampling: Path, greedy: Path | None, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(sampling / "pilot.parquet")
    require("draw" in frame, "this run has no draw column")
    collapsed = per_item(frame)
    tests = paired_tests(collapsed)
    draws = draw_level(frame)
    icc = {arm: intraclass_correlation(frame[frame["arm"] == arm])
           for arm in frame["arm"].unique()}

    report = {
        "n_items": int(frame["item"].nunique()),
        "draws_per_item": int(frame["draw"].nunique()),
        "total_draw_rows": int(len(frame)),
        "temperature": float(frame["temperature"].iloc[0]),
        "inference_n": int(frame["item"].nunique()),
        "intraclass_correlation": icc,
    }

    if greedy is not None and (greedy / "pilot.parquet").exists():
        g = pd.read_parquet(greedy / "pilot.parquet")
        gbase = g[g["arm"] == "baseline"].set_index("item")
        comparison = []
        for (arm, dose), grp in g[g["arm"] != "baseline"].groupby(["arm", "dose"]):
            grp = grp.set_index("item")
            common = grp.index.intersection(gbase.index)
            if len(common) < 3:
                continue
            delta = grp.loc[common, "n_generated"] - gbase.loc[common, "n_generated"]
            comparison.append(dict(arm=arm, dose=float(dose),
                                   greedy_median_delta=float(delta.median())))
        report["greedy_reference"] = comparison

    tests.to_csv(out / "paired_by_item.csv", index=False)
    draws.to_csv(out / "draw_level_descriptive.csv", index=False)
    collapsed.to_parquet(out / "per_item.parquet")
    (out / "sampling_report.json").write_text(json.dumps(report, indent=2))

    print(f"{report['total_draw_rows']} draw rows = {report['n_items']} items "
          f"x {report['draws_per_item']} draws at T={report['temperature']}")
    print(f"inference n = {report['inference_n']} (items), NOT {report['total_draw_rows']}\n")
    print("per-item means, paired against each item's own baseline:")
    print(tests.round(4).to_string(index=False))
    print("\ndraw-level (descriptive only, rows are clustered):")
    print(draws.round(3).to_string(index=False))
    print("\nintraclass correlation (share of variance between items):")
    print(json.dumps({k: round(v, 3) for k, v in icc.items() if v == v}, indent=2))
    return tests, draws, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sampling", type=Path, required=True)
    parser.add_argument("--greedy", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.sampling, args.greedy, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
