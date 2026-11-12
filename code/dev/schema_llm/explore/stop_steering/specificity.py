"""Causal specificity matrix: does each direction move its *own* target?

The dissociation reported so far is one-sided. ``v_solved`` and ``v_switch`` fail to cause STOP,
which is only strong evidence if each demonstrably does the job it is named for -- otherwise
"it did not cause STOP" is indistinguishable from "it is an ineffective vector".

``COUNTERFACTUAL_SCREEN`` built exactly this matrix observationally, scoring each direction's
*predicted* effect against event profiles. This is the causal counterpart: three directions against
three outcomes, each measured as a paired change from the item's own alpha = 0 resume.

| direction | its own target |
|---|---|
| ``stop`` | tokens until termination (lower = sooner) |
| ``switch`` | person switches per 100 generated tokens |
| ``solved`` | answer accuracy |

A direction passes if it moves its own target and the off-diagonal directions do not. The switch
rate is length-normalised on purpose: arms that terminate early produce short tails, and a raw
switch count would reward whichever arm rambles longest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

OUTCOMES = {
    "stop_tokens": ("n_generated", -1),          # lower is a stronger STOP effect
    "switch_rate": ("switches_per_100_tokens", 1),
    "accuracy": ("accuracy", 1),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def merge_accuracy(pilot: pd.DataFrame, accuracy: pd.DataFrame | None):
    if accuracy is None:
        pilot = pilot.copy()
        pilot["accuracy"] = np.nan
        return pilot
    keys = ["item", "arm", "dose"]
    slim = accuracy[[*keys, "accuracy", "all_correct"]].drop_duplicates(keys)
    return pilot.merge(slim, on=keys, how="left")


def paired_effect(arm_rows, base_rows, column):
    common = arm_rows.index.intersection(base_rows.index)
    if len(common) < 3:
        return None
    treated = arm_rows.loc[common, column].astype(float)
    control = base_rows.loc[common, column].astype(float)
    ok = treated.notna() & control.notna()
    treated, control = treated[ok], control[ok]
    if len(treated) < 3:
        return None
    delta = treated - control
    p = float(stats.wilcoxon(delta).pvalue) if (delta != 0).any() else 1.0
    return dict(n=int(len(delta)), mean_delta=float(delta.mean()),
                median_delta=float(delta.median()), p=p,
                treated_mean=float(treated.mean()), baseline_mean=float(control.mean()))


def matrix(table):
    rows = []
    base = table[table["arm"] == "baseline"].set_index("item")
    for (arm, dose), group in table[table["arm"] != "baseline"].groupby(["arm", "dose"]):
        group = group.set_index("item")
        entry = dict(arm=arm, dose=float(dose))
        for name, (column, sign) in OUTCOMES.items():
            if column not in group:
                continue
            effect = paired_effect(group, base, column)
            if effect is None:
                continue
            entry[f"{name}_delta"] = sign * effect["mean_delta"]
            entry[f"{name}_p"] = effect["p"]
            entry[f"{name}_treated"] = effect["treated_mean"]
            entry[f"{name}_baseline"] = effect["baseline_mean"]
            entry["n"] = effect["n"]
        rows.append(entry)
    return pd.DataFrame(rows)


def verdict(frame, alpha=0.05):
    """Did each direction move its own target, and leave the others' targets alone?"""
    own = {"stop": "stop_tokens", "switch": "switch_rate", "solved": "accuracy"}
    out = {}
    for arm, target in own.items():
        sub = frame[frame["arm"] == arm]
        if not len(sub) or f"{target}_p" not in sub:
            continue
        hits = sub[(sub[f"{target}_p"] < alpha) & (sub[f"{target}_delta"] > 0)]
        best = (hits.loc[hits[f"{target}_delta"].idxmax()] if len(hits) else None)
        out[arm] = dict(
            own_target=target,
            moved_own_target=bool(len(hits)),
            best_dose=float(best["dose"]) if best is not None else None,
            best_delta=float(best[f"{target}_delta"]) if best is not None else None,
            best_p=float(best[f"{target}_p"]) if best is not None else None,
            moved_stop=bool(((sub["stop_tokens_p"] < alpha)
                             & (sub["stop_tokens_delta"] > 0)).any())
            if "stop_tokens_p" in sub else None,
        )
    return out


def run(pilot_dir: Path, accuracy_path: Path | None, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    pilot = pd.read_parquet(pilot_dir / "pilot.parquet")
    accuracy = None
    if accuracy_path is not None and accuracy_path.exists():
        acc = pd.read_parquet(accuracy_path)
        accuracy = acc[acc["run"] == pilot_dir.name] if "run" in acc else acc
    table = merge_accuracy(pilot, accuracy)
    frame = matrix(table)
    frame.to_csv(out / "specificity_matrix.csv", index=False)

    columns = ["arm", "dose", "n",
               "stop_tokens_delta", "stop_tokens_p",
               "switch_rate_delta", "switch_rate_p",
               "accuracy_delta", "accuracy_p"]
    shown = [c for c in columns if c in frame]
    print("paired change from each item's own alpha=0 resume "
          "(stop_tokens_delta > 0 means it stopped SOONER)\n")
    print(frame[shown].round(4).to_string(index=False))

    result = verdict(frame)
    (out / "specificity_verdict.json").write_text(json.dumps(result, indent=2))
    print("\ndid each direction move its own target?")
    print(json.dumps(result, indent=2))
    return frame, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--accuracy", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.pilot, args.accuracy, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
