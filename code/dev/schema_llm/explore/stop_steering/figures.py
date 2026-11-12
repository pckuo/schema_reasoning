"""Figures for the STOP-steering result.

Four panels, one per claim the report makes:

1. survival -- censored time-to-STOP by arm, the primary outcome;
2. dose -- the paired token change against dose, showing the therapeutic window;
3. specificity -- each direction against each outcome, the causal counterpart of the screen's
   observational specificity matrix;
4. damage -- termination against output health across every cell, the panel that rules out
   "a big enough perturbation ends a generation".
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "axes.titlesize": 11, "axes.titleweight": "bold",
                     "axes.titlelocation": "left"})
COLOR = {"baseline": "#8A8A95", "stop": "#E76F51", "shuffled": "#2A9D8F",
         "switch": "#6A4CC0", "solved": "#E9C46A"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def km_curve(times, events):
    order = np.argsort(times)
    times, events = np.asarray(times)[order], np.asarray(events)[order].astype(bool)
    at_risk, survival = len(times), 1.0
    xs, ys = [0.0], [1.0]
    for t in np.unique(times):
        mask = times == t
        n_event = int(events[mask].sum())
        if at_risk > 0 and n_event:
            survival *= 1.0 - n_event / at_risk
        xs.append(float(t))
        ys.append(survival)
        at_risk -= int(mask.sum())
    return xs, ys


def panel_survival(ax, table, dose=0.5):
    for arm in ("baseline", "stop", "shuffled"):
        sub = table[(table["arm"] == arm) & ((table["dose"] == dose) | (arm == "baseline"))]
        if not len(sub):
            continue
        xs, ys = km_curve(sub["n_generated"].values, sub["terminated"].values)
        ax.step(xs, ys, where="post", color=COLOR[arm], lw=2,
                label=f"{arm}" + ("" if arm == "baseline" else f" (dose {dose})"))
    ax.set_xlabel("generated tokens")
    ax.set_ylabel("P(still reasoning)")
    ax.set_title("a  censored time-to-STOP")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, fontsize=8)


def panel_dose(ax, frame):
    for arm in ("stop", "shuffled", "switch", "solved"):
        sub = frame[frame["arm"] == arm].sort_values("dose")
        if not len(sub):
            continue
        ax.plot(sub["dose"], sub["stop_tokens_delta"], "o-", color=COLOR[arm], lw=2,
                ms=5, label=arm)
        for _, row in sub.iterrows():
            if row["stop_tokens_p"] < 0.05:
                ax.plot(row["dose"], row["stop_tokens_delta"], "o", ms=11, mfc="none",
                        mec=COLOR[arm], mew=1.6)
    ax.axhline(0, color="#8A8A95", lw=1, ls="--")
    ax.set_xlabel("dose (fraction of median state norm)")
    ax.set_ylabel("tokens saved vs own baseline")
    ax.set_title("b  dose–response (ringed = p < .05)")
    ax.legend(frameon=False, fontsize=8)


def panel_specificity(ax, frame, doses=None):
    """Each direction at *its own* effective dose -- a single shared dose would put one arm in
    its damage regime and misrepresent it."""
    doses = doses or {"stop": 0.5, "switch": 0.25, "solved": 0.25}
    arms = ["stop", "switch", "solved"]
    outcomes = [("stop_tokens", "STOP\n(sooner)"), ("switch_rate", "switch rate"),
                ("accuracy", "accuracy")]
    grid = np.full((len(arms), len(outcomes)), np.nan)
    stars = np.empty((len(arms), len(outcomes)), dtype=object)
    for i, arm in enumerate(arms):
        sub = frame[(frame["arm"] == arm) & (frame["dose"] == doses[arm])]
        if not len(sub):
            continue
        row = sub.iloc[0]
        for j, (key, _) in enumerate(outcomes):
            value, p = row.get(f"{key}_delta"), row.get(f"{key}_p", 1.0)
            if value is None or not np.isfinite(value):
                continue
            scale = {"stop_tokens": 1000.0, "switch_rate": 1.0, "accuracy": 0.1}[key]
            grid[i, j] = value / scale
            stars[i, j] = "*" if p < 0.05 else ""
    limit = np.nanmax(np.abs(grid)) or 1.0
    ax.imshow(grid, cmap="RdBu_r", vmin=-limit, vmax=limit)
    # a null result must not read as a strong one: wash out every non-significant cell
    for i in range(len(arms)):
        for j in range(len(outcomes)):
            if not stars[i, j]:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="white",
                                           alpha=0.72, edgecolor="none", zorder=2))
    ax.set_xticks(range(len(outcomes)), [o[1] for o in outcomes], fontsize=8)
    ax.set_yticks(range(len(arms)), [f"v_{a}\n(dose {doses[a]})" for a in arms], fontsize=8.5)
    for i in range(len(arms)):
        for j in range(len(outcomes)):
            if np.isfinite(grid[i, j]):
                strong = stars[i, j] and abs(grid[i, j]) > 0.55 * limit
                ax.text(j, i, f"{grid[i, j]:+.2f}{stars[i, j] or ''}", ha="center",
                        va="center", fontsize=8.5, zorder=3,
                        color=("white" if strong else ("#222" if stars[i, j] else "#888")))
    ax.set_title("c  specificity, each at its own dose\n(* p<.05; pale = not significant)",
                 fontsize=10)


def panel_oneshot(ax, oneshot, sustained):
    """Section 5: a single injection does nothing; the state has to be held."""
    for label, frame, colour in (("sustained", sustained, COLOR["stop"]),
                                 ("one-shot", oneshot, "#6A4CC0")):
        if frame is None or not len(frame):
            continue
        base = frame[frame["arm"] == "baseline"].set_index("item")["n_generated"]
        sub = frame[frame["arm"] == "stop"]
        doses, deltas = [], []
        for dose, grp in sub.groupby("dose"):
            grp = grp.set_index("item")
            common = grp.index.intersection(base.index)
            if len(common) < 3:
                continue
            doses.append(float(dose))
            deltas.append(float((base.loc[common] - grp.loc[common, "n_generated"]).median()))
        order = np.argsort(doses)
        ax.plot(np.array(doses)[order], np.array(deltas)[order], "o-", color=colour,
                lw=2, ms=5, label=label)
    ax.axhline(0, color="#8A8A95", lw=1, ls="--")
    ax.set_xlabel("dose")
    ax.set_ylabel("median tokens saved")
    ax.set_title("f  one-shot vs sustained injection", fontsize=10)
    ax.legend(frameon=False, fontsize=8)


def panel_damage(ax, cells):
    control = cells[cells["arm"] != "stop"]
    treat = cells[cells["arm"] == "stop"]
    ax.scatter(control["distinct_token_ratio"], control["delta_rate"], s=34,
               c="#8A8A95", label="controls", zorder=2)
    ax.scatter(treat["distinct_token_ratio"], treat["delta_rate"], s=52,
               c=COLOR["stop"], label="v_stop", zorder=3)
    ax.axhline(0, color="#8A8A95", lw=1, ls="--")
    ax.set_xlabel("output health (distinct-token ratio)")
    ax.set_ylabel("termination rate − baseline")
    ax.set_title("d  damage does not buy termination\n"
                 "(28-cell grid, n = 11/20 — pre-lexicon cohort)", fontsize=10)
    ax.legend(frameon=False, fontsize=8)


def panel_draws(ax, sampling):
    """Per-item effect under stochastic decoding, with the within-item draw spread.

    One point per item, so the eye is not invited to read clustered draws as independent
    observations; the vertical bar is the spread across that item's draws.
    """
    base = (sampling[sampling["arm"] == "baseline"]
            .groupby("item")["n_generated"].mean())
    arms = [("stop", 0.5), ("shuffled", 0.5)]
    for offset, (arm, dose) in enumerate(arms):
        sub = sampling[(sampling["arm"] == arm) & (sampling["dose"] == dose)]
        if not len(sub):
            continue
        per = sub.groupby("item")["n_generated"]
        items = sorted(per.groups)
        means = np.array([per.get_group(i).mean() for i in items])
        sds = np.array([per.get_group(i).std(ddof=1) for i in items])
        deltas = np.array([base.get(i, np.nan) for i in items]) - means
        xs = np.arange(len(items)) + (offset - 0.5) * 0.28
        ax.errorbar(xs, deltas, yerr=sds, fmt="o", ms=4.5, lw=1,
                    color=COLOR[arm], capsize=2, label=f"{arm} (dose {dose})")
    ax.axhline(0, color="#8A8A95", lw=1, ls="--")
    ax.set_xlabel("item")
    ax.set_ylabel("tokens saved vs own baseline")
    ax.set_title("e  stochastic decoding: per-item effect\n(bars = spread across draws)",
                 fontsize=10)
    ax.set_xticks([])
    ax.legend(frameon=False, fontsize=8, loc="lower left", ncol=2,
              bbox_to_anchor=(0.0, -0.02))


def run(fullcap: Path, dose_dir: Path, specificity: Path, grid: Path, out: Path,
        sampling: Path | None = None, oneshot: Path | None = None):
    out.mkdir(parents=True, exist_ok=True)
    table = pd.read_parquet(fullcap / "pilot.parquet")
    frame = pd.read_csv(specificity / "specificity_matrix.csv")
    cells = pd.read_csv(grid / "all_stats.csv")
    cells = cells.assign(delta_rate=cells["terminated"] / cells["n"]
                         - cells["baseline_terminated"] / cells["n"])

    draws = None
    if sampling is not None and (sampling / "pilot.parquet").exists():
        draws = pd.read_parquet(sampling / "pilot.parquet")

    ncols = 3 if draws is not None else 2
    fig, axes = plt.subplots(2, ncols, figsize=(5 * ncols, 7.5))
    panel_survival(axes[0, 0], table)
    panel_dose(axes[0, 1], frame)
    panel_specificity(axes[1, 0], frame)
    panel_damage(axes[1, 1], cells)
    if draws is not None:
        panel_draws(axes[0, 2], draws)
        shot = (pd.read_parquet(oneshot / "pilot.parquet")
                if oneshot is not None and (oneshot / "pilot.parquet").exists() else None)
        if shot is not None:
            panel_oneshot(axes[1, 2], shot, pd.read_parquet(dose_dir / "pilot.parquet"))
        else:
            axes[1, 2].axis("off")
    fig.suptitle("STOP control-state steering — phi4mr, layer 16 "
                 "(a,b,c,f: n=17 greedy · e: n=17 sampled T=0.7 · d: 28-cell grid, n=11/20)",
                 fontsize=12, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = out / "stop_steering_summary.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")

    # standalone survival panel, large, for the report header
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    panel_survival(ax, table)
    ax.set_title("Censored time-to-STOP at the corpus budget (n = 17)")
    fig.tight_layout()
    path = out / "survival_fullcap.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fullcap", type=Path, required=True)
    parser.add_argument("--dose-dir", type=Path, required=True)
    parser.add_argument("--specificity", type=Path, required=True)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sampling", type=Path, default=None)
    parser.add_argument("--oneshot", type=Path, default=None)
    args = parser.parse_args()
    run(args.fullcap, args.dose_dir, args.specificity, args.grid, args.out,
        sampling=args.sampling, oneshot=args.oneshot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
