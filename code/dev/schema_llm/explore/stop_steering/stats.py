"""Paired statistics for the STOP-steering pilot.

Greedy decoding makes every arm deterministic, so baseline and treatment differ only by the
injection and the item-level comparison is exactly paired: McNemar's exact test on discordant
pairs is the right test, and no seed variance enters.

The damage axis is reported alongside the termination axis on purpose. A perturbation that merely
breaks the model can also end a generation, so "terminates more" is only interesting if it holds
at a dose where the *norm- and subspace-matched* shuffled direction does not, and without the
degeneracy that the shuffled direction shows at the same dose.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from scipy import stats


def mcnemar(treatment, control):
    """Exact binomial test on discordant pairs; returns (p, n_gained, n_lost)."""
    gained = int((treatment & ~control).sum())
    lost = int((~treatment & control).sum())
    if gained + lost == 0:
        return 1.0, gained, lost
    return float(stats.binomtest(gained, gained + lost, 0.5).pvalue), gained, lost


def analyse(table: pd.DataFrame):
    base = table[table["arm"] == "baseline"].set_index("item").sort_index()
    names = [a for a in table["arm"].unique() if a != "baseline"]
    rows = []
    for dose in sorted(d for d in table["dose"].unique() if d != 0):
        present = [a for a in names
                   if len(table[(table["arm"] == a) & (table["dose"] == dose)])]
        arms = {a: table[(table["arm"] == a) & (table["dose"] == dose)]
                .set_index("item").sort_index() for a in present}
        for name, arm in arms.items():
            p_base, gained, lost = mcnemar(arm["terminated"], base["terminated"])
            row = dict(arm=name, dose=float(dose), n=int(len(arm)),
                       terminated=int(arm["terminated"].sum()),
                       baseline_terminated=int(base["terminated"].sum()),
                       gained_vs_baseline=gained, lost_vs_baseline=lost,
                       p_vs_baseline=p_base,
                       median_tokens=float(arm["n_generated"].median()),
                       baseline_median_tokens=float(base["n_generated"].median()),
                       layer=int(arm["layer"].iloc[0]) if "layer" in arm else 16,
                       cohort=str(arm["cohort"].iloc[0]) if "cohort" in arm else "capped",
                       closed_think=float(arm["closed_think"].mean()),
                       pairs_complete=float(arm["pairs_complete"].mean()),
                       distinct_token_ratio=float(arm["distinct_token_ratio"].mean()),
                       baseline_distinct_token_ratio=float(
                           base["distinct_token_ratio"].mean()))
            if "names_in_tail" in arm:
                row.update(names_in_tail=float(arm["names_in_tail"].mean()),
                           baseline_names_in_tail=float(base["names_in_tail"].mean()))
            if name != "shuffled" and "shuffled" in arms:
                p_sh, g_sh, l_sh = mcnemar(arm["terminated"], arms["shuffled"]["terminated"])
                row.update(p_vs_shuffled=p_sh, gained_vs_shuffled=g_sh,
                           lost_vs_shuffled=l_sh)
            rows.append(row)
    return pd.DataFrame(rows)


def run(pilot: Path):
    table = pd.read_parquet(pilot / "pilot.parquet")
    result = analyse(table)
    result.to_csv(pilot / "stats.csv", index=False)
    (pilot / "stats.json").write_text(json.dumps(result.to_dict("records"), indent=2))
    print(result.to_string(index=False))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    args = parser.parse_args()
    run(args.pilot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
