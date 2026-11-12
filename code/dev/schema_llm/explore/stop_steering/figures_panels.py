"""One figure per claim, each with its own error bars.

Replaces the single multi-panel overview: each result is written as a standalone PNG so it can be
shown on its own. Error bars are **SEM across items** everywhere, because item is the unit of
inference; where an item carries several stochastic draws they are averaged first, since pooling
draws would divide by the wrong n and understate the error.

The survival curve carries no error band: the step function already encodes the censoring
structure, and a band there adds clutter without adding information.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from .accuracy import load_accuracy  # noqa: E402

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "axes.titlesize": 11, "axes.titleweight": "bold",
                     "axes.titlelocation": "left"})
COLOR = {"baseline": "#8A8A95", "stop": "#E76F51", "shuffled": "#2A9D8F",
         "switch": "#6A4CC0", "solved": "#E9C46A"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sem(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


def collapse_draws(frame):
    keys = ["item", "arm", "dose"]
    numeric = [c for c in ("n_generated", "terminated", "accuracy", "distinct_token_ratio",
                           "switches_per_100_tokens", "answered", "all_correct",
                           "accuracy_cond", "accuracy_unc", "tokens_saved") if c in frame]
    return frame.groupby(keys, as_index=False)[numeric].mean()


def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s."


def km_curve(times, events):
    """Kaplan-Meier survival. No error band: the step function already carries the censoring."""
    order = np.argsort(times)
    times, events = np.asarray(times)[order], np.asarray(events)[order].astype(bool)
    at_risk, survival = len(times), 1.0
    xs, ys = [0.0], [1.0]
    for t in np.unique(times):
        mask = times == t
        d = int(events[mask].sum())
        if at_risk > 0 and d:
            survival *= 1.0 - d / at_risk
        xs.append(float(t))
        ys.append(survival)
        at_risk -= int(mask.sum())
    return np.array(xs), np.array(ys)


def fig_survival(fullcap: Path, out: Path, dose=0.5):
    table = pd.read_parquet(fullcap / "pilot.parquet")
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    for arm in ("baseline", "stop", "shuffled"):
        sub = table[(table["arm"] == arm) & ((table["dose"] == dose) | (arm == "baseline"))]
        if not len(sub):
            continue
        xs, ys = km_curve(sub["n_generated"].values, sub["terminated"].values)
        label = arm if arm == "baseline" else f"{arm} (dose {dose})"
        ax.step(xs, ys, where="post", color=COLOR[arm], lw=2, label=label)
    ax.set_xlabel("generated tokens")
    ax.set_ylabel("P(still reasoning)")
    ax.set_title("Censored time-to-STOP")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    path = out / "fig_a_survival.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def _paired(frame, column="n_generated"):
    """Per-arm/dose paired **change** against each item's own baseline, with SEM.

    Signed as treatment minus baseline, so a shortened trace reads as a negative number.
    """
    frame = collapse_draws(frame)
    base = frame[frame["arm"] == "baseline"].set_index("item")
    rows = []
    for (arm, dose), group in frame[frame["arm"] != "baseline"].groupby(["arm", "dose"]):
        group = group.set_index("item")
        common = group.index.intersection(base.index)
        if len(common) < 3:
            continue
        delta = (group.loc[common, column].astype(float)
                 - base.loc[common, column].astype(float))
        p = float(stats.wilcoxon(delta).pvalue) if (delta != 0).any() else 1.0
        rows.append(dict(arm=arm, dose=float(dose), mean=float(delta.mean()),
                         sem=sem(delta.values), p=p, n=int(len(common))))
    return pd.DataFrame(rows)


def fig_dose(dose_dir: Path, out: Path):
    table = pd.read_parquet(dose_dir / "pilot.parquet")
    effects = _paired(table)
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    for arm in ("stop", "shuffled", "switch", "solved"):
        sub = effects[effects["arm"] == arm].sort_values("dose")
        if not len(sub):
            continue
        ax.errorbar(sub["dose"], sub["mean"], yerr=sub["sem"], fmt="o-", color=COLOR[arm],
                    lw=2, ms=6, capsize=3, label=arm)
        for _, row in sub.iterrows():
            if row["p"] < 0.05:
                sign = 1 if row["mean"] >= 0 else -1
                ax.annotate(stars(row["p"]),
                            (row["dose"], row["mean"] + sign * row["sem"]),
                            textcoords="offset points", xytext=(0, 6 * sign), ha="center",
                            fontsize=8.5, color=COLOR[arm])
    ax.axhline(0, color="#8A8A95", lw=1, ls="--")
    ax.set_xlabel("dose (fraction of median state norm)")
    ax.set_ylabel("mean token change vs own baseline\n(negative = shorter)")
    ax.set_title("Dose-response")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    path = out / "fig_b_dose_response.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def fig_specificity(dose_dir: Path, accuracy_path: Path, out: Path, dose: float = 0.25):
    """Each direction against each outcome at one **common** dose, value +/- SEM.

    All vectors are unit-norm, so holding the dose fixed is what makes the comparison a
    norm-matched one -- reading each arm at its own most flattering dose would not be.

    ``answered`` is carried as an outcome in its own right: under final-answer scoring the control
    directions mostly fail to emit an answer at all, and an accuracy row alone would hide that.
    """
    table = pd.read_parquet(dose_dir / "pilot.parquet")
    accuracy = load_accuracy(accuracy_path, run=dose_dir.name)
    table = table.merge(accuracy[["item", "arm", "dose", "answered", "all_correct"]],
                        on=["item", "arm", "dose"], how="left")
    # every column is oriented "+ = more of the named thing", so one colour scale means one thing
    # across the matrix; length is therefore carried as tokens *saved* rather than tokens emitted
    table = table.assign(tokens_saved=-table["n_generated"].astype(float))
    outcomes = [("tokens_saved", "tokens saved\n(+ = stops sooner)", 1000.0),
                ("switches_per_100_tokens", "switch rate\n(+ = switches more)", 1.0),
                ("answered", "answers at all\n(+ = more often)", 0.1),
                ("all_correct", "all pairs correct\n(+ = more often)", 0.1)]
    arms = ["stop", "switch", "solved"]
    grid = np.full((len(arms), len(outcomes)), np.nan)
    text = np.empty((len(arms), len(outcomes)), dtype=object)
    for i, arm in enumerate(arms):
        for j, (column, _, scale) in enumerate(outcomes):
            effects = _paired(table, column)
            row = effects[(effects["arm"] == arm) & (effects["dose"] == dose)]
            if not len(row):
                continue
            row = row.iloc[0]
            grid[i, j] = row["mean"] / scale
            text[i, j] = (f"{row['mean'] / scale:+.2f}\n±{row['sem'] / abs(scale):.2f}"
                          f" {stars(row['p'])}")
    limit = np.nanmax(np.abs(grid)) or 1.0
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.imshow(grid, cmap="RdBu_r", vmin=-limit, vmax=limit)
    for i in range(len(arms)):
        for j in range(len(outcomes)):
            if text[i, j] is None:
                continue
            significant = text[i, j].endswith(("*",))
            if not significant:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="white",
                                           alpha=0.72, edgecolor="none", zorder=2))
            strong = significant and abs(grid[i, j]) > 0.55 * limit
            ax.text(j, i, text[i, j], ha="center", va="center", fontsize=8, zorder=3,
                    color=("white" if strong else ("#222" if significant else "#888")))
    ax.set_xticks(range(len(outcomes)), [o[1] for o in outcomes], fontsize=8)
    ax.set_yticks(range(len(arms)), [f"v_{a}" for a in arms], fontsize=9)
    ax.set_title(f"Specificity: all directions norm-matched at dose {dose}")
    ax.set_xlabel("colour = sign of the paired change against each item's own baseline",
                  fontsize=8, color="#555")
    fig.tight_layout()
    path = out / "fig_c_specificity.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def fig_answer_delivery(dose_dir: Path, accuracy_path: Path, out: Path, dose: float = 0.25):
    """Why final-answer scoring changes the ranking: delivery, then quality given delivery.

    Whole-response scoring rewarded arms for the content of their rambling; scoring the block after
    ``</think>`` shows most control arms never produce an answer. The middle panel is deliberately
    plotted only where an arm answered at all -- it is a conditional quantity, and its n is printed
    so a 1.00 from a single answer cannot be mistaken for a result.
    """
    accuracy = load_accuracy(accuracy_path, run=Path(dose_dir).name)
    arms = ["baseline", "stop", "switch", "solved", "shuffled"]
    rows = []
    for arm in arms:
        sub = accuracy[(accuracy["arm"] == arm)
                       & (accuracy["dose"] == (0.0 if arm == "baseline" else dose))]
        if not len(sub):
            continue
        answered = sub["answered"].astype(float)
        cond = sub["accuracy_cond"].dropna().astype(float)
        rows.append(dict(arm=arm, n=len(sub), answered=float(answered.mean()),
                         answered_sem=sem(answered.values), n_answered=int(answered.sum()),
                         cond=float(cond.mean()) if len(cond) else np.nan,
                         cond_sem=sem(cond.values) if len(cond) else 0.0,
                         strict=float(sub["all_correct"].astype(float).mean()),
                         strict_sem=sem(sub["all_correct"].astype(float).values)))
    frame = pd.DataFrame(rows)
    require(len(frame), "no arms to plot")

    panels = [("answered", "answered_sem", "a  produces a final answer",
               "fraction of items", None),
              ("cond", "cond_sem", "b  accuracy *given* an answer",
               "gold pairs correct", "n_answered"),
              ("strict", "strict_sem", "c  all pairs correct, per item",
               "fraction of items", None)]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.0))
    x = np.arange(len(frame))
    for ax, (column, err, title, ylabel, annotate) in zip(axes, panels, strict=True):
        values = frame[column].values.astype(float)
        ax.bar(x, np.nan_to_num(values), yerr=frame[err].values, capsize=4,
               color=[COLOR[a] for a in frame["arm"]], error_kw=dict(ecolor="#333", lw=1.1))
        ax.set_xticks(x, frame["arm"], fontsize=8.5, rotation=20)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_ylim(0, 1.18)
        ax.set_title(title)
        for xi, value, e in zip(x, values, frame[err].values, strict=True):
            if not np.isfinite(value):
                ax.text(xi, 0.04, "no\nanswers", ha="center", fontsize=7.5, color="#B03030")
                continue
            tag = f"{value:.2f}"
            if annotate:
                tag += f"\n(n={int(frame.iloc[xi][annotate])})"
            ax.text(xi, value + e + 0.03, tag, ha="center", fontsize=8)
    fig.suptitle(f"Final-answer scoring, dose {dose} (n = {int(frame['n'].iloc[0])} items)",
                 fontsize=11.5, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = out / "fig_g_answer_delivery.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def fig_damage(layer_dirs, out: Path):
    """Termination gain against output health, per cell, with SEM on both axes."""
    frames = []
    for layer, directory in layer_dirs.items():
        path = Path(directory) / "pilot.parquet"
        if not path.exists():
            continue
        frame = collapse_draws(pd.read_parquet(path))
        frame["layer"] = layer
        frames.append(frame)
    require(frames, "no runs for the damage panel")
    table = pd.concat(frames, ignore_index=True)
    base_rate = float(table[table["arm"] == "baseline"]["terminated"].mean())

    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    for arm in sorted(table["arm"].unique()):
        if arm == "baseline":
            continue
        colour = COLOR["stop"] if arm == "stop" else "#8A8A95"
        xs, ys, xe, ye = [], [], [], []
        for (_, _), cell in table[table["arm"] == arm].groupby(["layer", "dose"]):
            xs.append(float(cell["distinct_token_ratio"].mean()))
            xe.append(sem(cell["distinct_token_ratio"].values))
            ys.append(float(cell["terminated"].mean()) - base_rate)
            ye.append(sem(cell["terminated"].values))
        ax.errorbar(xs, ys, xerr=xe, yerr=ye, fmt="o", ms=6, color=colour,
                    ecolor="#bbb", elinewidth=0.9, capsize=2,
                    label="v_stop" if arm == "stop" else "controls",
                    zorder=3 if arm == "stop" else 2)
    ax.axhline(0, color="#8A8A95", lw=1, ls="--")
    ax.set_xlabel("output health (distinct-token ratio)")
    ax.set_ylabel("termination rate − baseline")
    ax.set_title("Damage does not buy termination")
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=True))
    ax.legend(unique.values(), unique.keys(), frameon=False, fontsize=9)
    fig.tight_layout()
    path = out / "fig_d_damage.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def fig_sampling(sampling: Path, out: Path, dose=0.5):
    """Stochastic decoding: arm means with SEM across items, plus the per-item effects."""
    table = pd.read_parquet(sampling / "pilot.parquet")
    per_item = collapse_draws(table)
    base = per_item[per_item["arm"] == "baseline"].set_index("item")

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    arms = ["stop", "shuffled"]
    means, errs, ps = [], [], []
    for arm in arms:
        sub = per_item[(per_item["arm"] == arm) & (per_item["dose"] == dose)].set_index("item")
        common = sub.index.intersection(base.index)
        delta = sub.loc[common, "n_generated"] - base.loc[common, "n_generated"]
        means.append(float(delta.mean()))
        errs.append(sem(delta.values))
        ps.append(float(stats.wilcoxon(delta).pvalue) if (delta != 0).any() else 1.0)
        axes[1].scatter(np.full(len(delta), arms.index(arm)) + np.random.default_rng(0)
                        .uniform(-0.09, 0.09, len(delta)), delta.values,
                        s=26, color=COLOR[arm], alpha=0.75, zorder=3)
    axes[0].bar(np.arange(len(arms)), means, yerr=errs, capsize=4,
                color=[COLOR[a] for a in arms], error_kw=dict(ecolor="#333", lw=1.2))
    axes[0].axhline(0, color="#555", lw=1)
    axes[0].set_xticks(range(len(arms)), arms, fontsize=9)
    axes[0].set_ylabel("mean token change vs own baseline\n(negative = shorter)")
    axes[0].set_title("a  effect under sampling")
    for xi, value, err, p in zip(range(len(arms)), means, errs, ps, strict=True):
        sign = 1 if value >= 0 else -1
        axes[0].text(xi, value + sign * (err + 90), stars(p), ha="center", fontsize=9)

    axes[1].axhline(0, color="#8A8A95", lw=1, ls="--")
    axes[1].set_xticks(range(len(arms)), arms, fontsize=9)
    axes[1].set_ylabel("token change (one point per item)")
    axes[1].set_title("b  per-item effects behind the mean")

    fig.suptitle(f"Stochastic decoding, dose {dose}",
                 fontsize=11.5, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = out / "fig_e_sampling.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def fig_oneshot(oneshot: Path, sustained: Path, out: Path, *,
                series=("sustained", "one-shot"), filename="fig_f_oneshot.png",
                title="One-shot vs sustained injection"):
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    for label, directory, colour in (("sustained", sustained, COLOR["stop"]),
                                     ("one-shot", oneshot, "#6A4CC0")):
        if label not in series:
            continue
        effects = _paired(pd.read_parquet(Path(directory) / "pilot.parquet"))
        sub = effects[effects["arm"] == "stop"].sort_values("dose")
        if not len(sub):
            continue
        ax.errorbar(sub["dose"], sub["mean"], yerr=sub["sem"], fmt="o-", color=colour,
                    lw=2, ms=6, capsize=3, label=label)
        for _, row in sub.iterrows():
            sign = 1 if row["mean"] >= 0 else -1
            ax.annotate(stars(row["p"]), (row["dose"], row["mean"] + sign * row["sem"]),
                        textcoords="offset points", xytext=(0, 6 * sign), ha="center",
                        fontsize=8, color=colour)
    ax.axhline(0, color="#8A8A95", lw=1, ls="--")
    ax.set_xlabel("injection strength", fontsize=14)
    ax.set_ylabel(r"$\Delta$ token length", fontsize=14)
    ax.set_title(title)
    if len(series) > 1:
        ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    path = out / filename
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fullcap", type=Path, default=None,
                        help="survival source; omit to skip the survival figure")
    parser.add_argument("--dose-dir", type=Path, required=True)
    parser.add_argument("--accuracy", type=Path, required=True)
    parser.add_argument("--sampling", type=Path, required=True)
    parser.add_argument("--oneshot", type=Path, default=None)
    parser.add_argument("--l8", type=Path, required=True)
    parser.add_argument("--l16", type=Path, required=True)
    parser.add_argument("--l24", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dose", type=float, default=0.25,
                        help="headline dose for the single-dose figures")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.fullcap is not None:
        fig_survival(args.fullcap, args.out, dose=args.dose)
    fig_dose(args.dose_dir, args.out)
    fig_specificity(args.dose_dir, args.accuracy, args.out, dose=args.dose)
    fig_answer_delivery(args.dose_dir, args.accuracy, args.out, dose=args.dose)
    fig_damage({8: args.l8, 16: args.l16, 24: args.l24}, args.out)
    fig_sampling(args.sampling, args.out, dose=args.dose)
    if args.oneshot is not None and (args.oneshot / "pilot.parquet").exists():
        fig_oneshot(args.oneshot, args.dose_dir, args.out)
        fig_oneshot(args.oneshot, args.dose_dir, args.out, series=("one-shot",),
                    filename="fig_f_oneshot_only.png", title="One-shot injection")
    else:
        print("skipping fig_f_oneshot: no one-shot pilot yet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
