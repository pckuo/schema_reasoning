"""Censored time-to-STOP -- the plan's pre-registered primary outcome.

The 512-token pilot asked a binary question ("did it stop?") that turned out to be partly a
question about the window: with the corpus budget restored, the alpha=0 resume *does* stop, just
late. The right outcome is therefore the one the plan named and ``COMPLETION_DEFINITION`` already
uses -- **time to STOP with the budget right-censored** -- rather than a rate at an arbitrary cap.

Kaplan-Meier per arm, plus a paired per-item comparison against that item's own alpha=0 resume,
which is exact here because greedy decoding makes every arm deterministic.
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


def kaplan_meier(times, events):
    """Survival of 'still reasoning' against generated tokens; events are terminations."""
    order = np.argsort(times)
    times, events = np.asarray(times)[order], np.asarray(events)[order]
    at_risk = len(times)
    survival, curve = 1.0, []
    for t in np.unique(times):
        mask = times == t
        n_event = int(events[mask].sum())
        if at_risk > 0 and n_event:
            survival *= 1.0 - n_event / at_risk
        curve.append((float(t), float(survival), int(at_risk), n_event))
        at_risk -= int(mask.sum())
    return curve


def median_survival(curve):
    for t, s, _, _ in curve:
        if s <= 0.5:
            return float(t)
    return float("nan")


def summarize(table):
    rows = []
    for (run, schedule, arm, dose), group in table.groupby(
            ["run", "schedule", "arm", "dose"], dropna=False):
        curve = kaplan_meier(group["n_generated"].values, group["terminated"].values)
        rows.append(dict(run=run, schedule=schedule, arm=arm, dose=float(dose),
                         n=int(len(group)),
                         events=int(group["terminated"].sum()),
                         censored=int((~group["terminated"].astype(bool)).sum()),
                         km_median_tokens=median_survival(curve),
                         mean_tokens_observed=float(group["n_generated"].mean()),
                         median_tokens_observed=float(group["n_generated"].median())))
    return pd.DataFrame(rows)


def paired(table):
    """Per-item change in generated tokens against that item's own alpha=0 resume."""
    rows = []
    for run, block in table.groupby("run"):
        base = block[block["arm"] == "baseline"].set_index("item")
        for (schedule, arm, dose), group in block[block["arm"] != "baseline"].groupby(
                ["schedule", "arm", "dose"], dropna=False):
            group = group.set_index("item")
            common = group.index.intersection(base.index)
            if len(common) < 3:
                continue
            delta = (group.loc[common, "n_generated"].astype(float)
                     - base.loc[common, "n_generated"].astype(float))
            ratio = (group.loc[common, "n_generated"].astype(float)
                     / base.loc[common, "n_generated"].astype(float).clip(lower=1))
            p = stats.wilcoxon(delta).pvalue if (delta != 0).any() else 1.0
            gained = int((group.loc[common, "terminated"].astype(bool)
                          & ~base.loc[common, "terminated"].astype(bool)).sum())
            lost = int((~group.loc[common, "terminated"].astype(bool)
                        & base.loc[common, "terminated"].astype(bool)).sum())
            rows.append(dict(run=run, schedule=schedule, arm=arm, dose=float(dose),
                             n=int(len(common)),
                             median_delta_tokens=float(delta.median()),
                             mean_ratio_to_baseline=float(ratio.mean()),
                             wilcoxon_p=float(p),
                             terminations_gained=gained, terminations_lost=lost))
    return pd.DataFrame(rows)


def run(dirs, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    frames = []
    for directory in dirs:
        path = Path(directory) / "pilot.parquet"
        if not path.exists():
            print(f"  skip {directory}")
            continue
        frame = pd.read_parquet(path)
        frame["run"] = Path(directory).name
        if "schedule" not in frame:
            frame["schedule"] = "sustained"
        frames.append(frame)
    require(frames, "no runs found")
    table = pd.concat(frames, ignore_index=True)

    curves = summarize(table)
    changes = paired(table)
    curves.to_csv(out / "survival_by_arm.csv", index=False)
    changes.to_csv(out / "survival_paired.csv", index=False)
    print(curves.to_string(index=False))
    print("\npaired against each item's own alpha=0 resume:")
    print(changes.to_string(index=False))
    (out / "survival.json").write_text(json.dumps(
        {"by_arm": curves.to_dict("records"), "paired": changes.to_dict("records")},
        indent=2, default=str))
    return curves, changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dirs", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.dirs, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
