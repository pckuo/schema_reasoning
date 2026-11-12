"""Reconstruct injection measurements from the run log when the shards are gone.

The run prints one line per generation before the shard is written, so a matrix interrupted
mid-trace -- or whose shards were deleted after a plan change -- still leaves a complete record of
the outcomes that do not need the response text: token count, stop reason, arm, dose, offset and
draw. That is enough for the primary endpoints (length and termination) and not enough for the
answer-health endpoints (``correct``, ``answered``), which are therefore absent here by
construction rather than silently defaulted.

Cells are deduplicated **keeping the last occurrence**, because the log accumulates across
restarts and the most recent measurement is the one taken under the current configuration.

This is a salvage path, not the analysis path: numbers from here are preliminary and are
superseded by ``injections.parquet`` as soon as the matrix completes.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

LOG = Path("results/schema_llm/logs/tsp_stop_steering.log")
RUN = Path("results/schema_llm/analysis/tsp_stop_steering/20260921-transfer")
LINE = re.compile(
    r"uid=(\d+) b(\d+) @(0\.\d+) d(\d+) (\w+)\s+dose ([\d.]+)\s+-> (\w+)\s+(\d+) tok")
COLOR = {"baseline": "#8A8A95", "stop": "#E76F51", "shuffled": "#2A9D8F"}

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "axes.titlesize": 11, "axes.titleweight": "bold",
                     "axes.titlelocation": "left"})


def require(condition, message):
    if not condition:
        raise ValueError(message)


def parse(log: Path = LOG, run: Path = RUN):
    rows = []
    for line in log.read_text(errors="ignore").splitlines():
        match = LINE.search(line)
        if match:
            rows.append(dict(
                uid=int(match.group(1)), base_draw=int(match.group(2)),
                location=float(match.group(3)), draw=int(match.group(4)),
                arm=match.group(5), dose=float(match.group(6)),
                stop_reason=match.group(7), n_generated=int(match.group(8))))
    require(rows, f"no injection lines in {log}")
    frame = pd.DataFrame(rows)
    # the log spans restarts; the last measurement of a cell is the current-configuration one
    frame = frame.drop_duplicates(["uid", "base_draw", "location", "draw", "arm", "dose"],
                                  keep="last").reset_index(drop=True)
    baselines = pd.read_parquet(run / "baselines.parquet",
                                columns=["uid", "draw", "reasoning_tokens", "n_generated"])
    reasoning = {(int(r.uid), int(r.draw)): int(r.reasoning_tokens)
                 for r in baselines.itertuples()}
    frame["reasoning"] = [reasoning[(r.uid, r.base_draw)] for r in frame.itertuples()]
    frame["offset"] = (frame["location"] * frame["reasoning"]).round().astype(int)
    frame["total_tokens"] = frame["offset"] + frame["n_generated"]
    frame["terminated"] = frame["stop_reason"] == "eos"
    return frame


def paired(frame):
    keys = ["uid", "base_draw", "location", "draw"]
    control = frame[frame["arm"] == "baseline"][keys + ["total_tokens", "terminated"]]
    treated = frame[frame["arm"] != "baseline"]
    merged = treated.merge(control, on=keys, suffixes=("", "_ctl"))
    merged["d_total_tokens"] = merged["total_tokens"] - merged["total_tokens_ctl"]
    return merged


def sem(values):
    values = np.asarray(values, dtype=float)
    return float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0


def plot(frame, merged, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    arms = sorted({(r.arm, float(r.dose)) for r in merged.itertuples()})
    locations = sorted(frame["location"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    ax = axes[0]
    for index, (arm, dose) in enumerate(arms):
        group = merged[(merged["arm"] == arm) & (merged["dose"] == dose)]
        values = group["d_total_tokens"].values
        ax.bar(index, np.mean(values), yerr=sem(values), color=COLOR.get(arm, "#999"),
               width=0.6, capsize=4, edgecolor="none")
        jitter = np.random.default_rng(0).uniform(-0.12, 0.12, len(values))
        ax.scatter(np.full(len(values), index) + jitter, values, s=14, color="black", alpha=0.4,
                   zorder=3)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([f"{a}\n{d:g}" for a, d in arms])
    ax.set_ylabel("change in total response tokens\nvs the α=0 fork")
    ax.set_title("Paired effect on response length")

    ax = axes[1]
    for arm, dose in arms:
        xs, ys, es = [], [], []
        for location in locations:
            group = merged[(merged["arm"] == arm) & (merged["dose"] == dose)
                           & (merged["location"] == location)]
            if not len(group):
                continue
            xs.append(location)
            ys.append(float(group["d_total_tokens"].mean()))
            es.append(sem(group["d_total_tokens"].values))
        ax.errorbar(xs, ys, yerr=es, marker="o" if dose == 0.25 else "s",
                    linestyle="-" if dose == 0.25 else "--", capsize=3,
                    color=COLOR.get(arm, "#999"), label=f"{arm} {dose:g}")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(locations)
    ax.set_xticklabels([f"{int(v * 100)}%" for v in locations])
    ax.set_xlabel("injection start, as a fraction of the reasoning trace")
    ax.set_ylabel("change in total response tokens")
    ax.set_title("Effect against where injection begins")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "prelim_tokens.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    all_arms = [("baseline", 0.0), *arms]
    width = 0.8 / len(all_arms)
    for index, (arm, dose) in enumerate(all_arms):
        xs, ys, es = [], [], []
        for position, location in enumerate(locations):
            group = frame[(frame["arm"] == arm) & (frame["dose"] == dose)
                          & (frame["location"] == location)]
            if not len(group):
                continue
            xs.append(position + index * width - 0.4 + width / 2)
            ys.append(float(group["terminated"].mean()))
            es.append(sem(group["terminated"].values.astype(float)))
        shade = 1.0 if dose <= 0.25 else 0.6
        ax.bar(xs, ys, yerr=es, width=width * 0.9, capsize=3, color=COLOR.get(arm, "#999"),
               alpha=shade, edgecolor="none", label=f"{arm} {dose:g}")
    ax.set_xticks(range(len(locations)))
    ax.set_xticklabels([f"{int(v * 100)}%" for v in locations])
    ax.set_xlabel("injection start")
    ax.set_ylabel("terminated before the budget")
    ax.set_ylim(0, 1.08)
    ax.set_title("Does the direction end the trace?")
    ax.legend(frameon=False, ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    fig.tight_layout()
    fig.savefig(out / "prelim_termination.png")
    plt.close(fig)
    return ["prelim_tokens.png", "prelim_termination.png"]


def run(log: Path = LOG, run_dir: Path = RUN, out: Path | None = None):
    out = out or (run_dir.parent / "20260922-preliminary-from-log")
    frame = parse(log, run_dir)
    merged = paired(frame)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "runs.csv", index=False)
    merged.to_csv(out / "paired.csv", index=False)

    summary = (merged.groupby(["arm", "dose"])
               .agg(n=("d_total_tokens", "size"),
                    delta_median=("d_total_tokens", "median"),
                    delta_mean=("d_total_tokens", "mean"),
                    terminated=("terminated", "mean"),
                    terminated_ctl=("terminated_ctl", "mean"))
               .reset_index())
    by_loc = (merged.groupby(["arm", "dose", "location"])
              .agg(n=("d_total_tokens", "size"),
                   delta_median=("d_total_tokens", "median"),
                   terminated=("terminated", "mean"))
              .reset_index())
    print(f"reconstructed {len(frame)} runs over "
          f"{frame.groupby(['uid', 'base_draw']).ngroups} baseline traces\n")
    print(summary.to_string(index=False))
    print()
    print(by_loc.to_string(index=False))
    made = plot(frame, merged, out)
    print("\nwrote:", ", ".join(made))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=LOG)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    run(args.log, args.run, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
