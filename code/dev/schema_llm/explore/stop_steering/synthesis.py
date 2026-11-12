"""Combine every STOP-steering intervention run into one table and one verdict.

Each run directory holds a ``pilot.parquet`` for one (cohort, layer) cell; this concatenates them,
recomputes the paired statistics per (cohort, layer, arm, dose), and writes the synthesis that the
findings prose quotes. Keeping the combination in code rather than in prose means the numbers in
the report cannot drift from the numbers on disk.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .stats import analyse


def require(condition, message):
    if not condition:
        raise ValueError(message)


def collect(dirs):
    frames = []
    for directory in dirs:
        path = Path(directory) / "pilot.parquet"
        if not path.exists():
            print(f"  skip (no pilot.parquet): {directory}")
            continue
        frame = pd.read_parquet(path)
        frame["run"] = Path(directory).name
        # runs predating the layer/cohort columns are the original L16 capped pilot; without
        # this backfill they carry NaN keys and groupby drops them silently.
        if "layer" not in frame:
            frame["layer"] = 16
        if "cohort" not in frame:
            frame["cohort"] = "capped"
        require(frame[["layer", "cohort"]].notna().all().all(), f"{directory}: null group keys")
        frames.append(frame)
    require(frames, "no runs found")
    table = pd.concat(frames, ignore_index=True)
    # greedy decoding makes cells deterministic, so a cell repeated across runs is identical:
    # dedupe rather than double-count it.
    keys = ["item", "arm", "dose", "layer", "cohort"]
    before = len(table)
    table = table.drop_duplicates(subset=keys).reset_index(drop=True)
    if before != len(table):
        print(f"  deduped {before - len(table)} repeated cells")
    return table


def per_cell(table):
    rows = []
    for (cohort, layer), group in table.groupby(["cohort", "layer"]):
        result = analyse(group)
        result["cohort"] = cohort
        result["layer"] = layer
        rows.append(result)
    return pd.concat(rows, ignore_index=True)


def run(dirs, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    table = collect(dirs)
    table.to_parquet(out / "all_runs.parquet")
    summary = per_cell(table)
    summary.to_csv(out / "all_stats.csv", index=False)

    columns = ["cohort", "layer", "arm", "dose", "n", "terminated", "baseline_terminated",
               "p_vs_baseline", "p_vs_shuffled", "median_tokens", "closed_think",
               "pairs_complete", "distinct_token_ratio", "names_in_tail"]
    shown = [c for c in columns if c in summary]
    print(summary[shown].sort_values(["cohort", "layer", "arm", "dose"]).to_string(index=False))

    verdict = {}
    stop = summary[(summary["arm"] == "stop") & (summary["cohort"] == "capped")]
    if len(stop):
        best = stop.loc[stop["terminated"].idxmax()]
        verdict["best_stop_cell"] = {k: (float(best[k]) if isinstance(best[k], float) else
                                         (int(best[k]) if isinstance(best[k], int | bool)
                                          else str(best[k])))
                                     for k in shown if k in best}
    for arm in ("solved", "switch", "shuffled"):
        sub = summary[(summary["arm"] == arm) & (summary["cohort"] == "capped")]
        if len(sub):
            verdict[f"{arm}_max_terminated"] = int(sub["terminated"].max())
            verdict[f"{arm}_min_p_vs_baseline"] = float(sub["p_vs_baseline"].min())
    (out / "verdict.json").write_text(json.dumps(verdict, indent=2, default=str))
    print("\n" + json.dumps(verdict, indent=2, default=str))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dirs", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.dirs, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
