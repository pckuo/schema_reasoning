"""Standalone result figures: the dose-0.5 bar summary, the accuracy/efficiency trade, and the
layer sweep.

Kept separate from ``figures.py`` (the multi-panel overview) because each of these answers one
question and is meant to be shown on its own.

Every panel labels the cohort it was computed on. That matters here: the layer sweep predates the
extended-lexicon population fix and is n = 11, while the dose-0.5 and accuracy panels are the
n = 17 cohort, so a single "n" in a shared title would be wrong.
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

from . import runframe  # noqa: E402
from .accuracy import load_accuracy  # noqa: E402

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "axes.titlesize": 11, "axes.titleweight": "bold",
                     "axes.titlelocation": "left"})
COLOR = {"baseline": "#8A8A95", "stop": "#E76F51", "shuffled": "#2A9D8F",
         "switch": "#6A4CC0", "solved": "#E9C46A"}
ORDER = ["baseline", "stop", "shuffled", "switch", "solved"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s."


def sem(values):
    """Standard error **across items**.

    Item is the unit of inference, so this is the error bar the tests correspond to. Where an
    item has several stochastic draws they are averaged first: pooling draws would divide by
    sqrt(n_items * n_draws) and understate the error by about sqrt(n_draws).
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


def collapse_draws(frame):
    """One row per (item, arm, dose); a no-op on greedy runs, which have a single draw."""
    keys = ["item", "arm", "dose"]
    numeric = [c for c in ("n_generated", "terminated", "accuracy", "distinct_token_ratio",
                           "accuracy_cond", "accuracy_unc", "answered", "all_correct")
               if c in frame]
    return frame.groupby(keys, as_index=False)[numeric].mean()


def dose_bars(pilot: pd.DataFrame, dose: float, out: Path):
    """Termination rate and tokens saved, per arm, at one dose."""
    pilot = collapse_draws(pilot)
    base = pilot[pilot["arm"] == "baseline"].set_index("item")
    arms = [a for a in ORDER if a != "baseline"
            and len(pilot[(pilot["arm"] == a) & (pilot["dose"] == dose)])]
    rates, rate_err, saved, saved_err, pvals, labels = [], [], [], [], [], []
    base_rate = float(base["terminated"].mean())
    base_rate_err = sem(base["terminated"].values)
    for arm in arms:
        sub = pilot[(pilot["arm"] == arm) & (pilot["dose"] == dose)].set_index("item")
        common = sub.index.intersection(base.index)
        rates.append(float(sub.loc[common, "terminated"].mean()))
        rate_err.append(sem(sub.loc[common, "terminated"].values))
        # token CHANGE, so a shortened trace reads as negative
        delta = (sub.loc[common, "n_generated"].astype(float)
                 - base.loc[common, "n_generated"].astype(float))
        saved.append(float(delta.mean()))
        saved_err.append(sem(delta.values))
        pvals.append(float(stats.wilcoxon(delta).pvalue) if (delta != 0).any() else 1.0)
        labels.append(arm)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.9))
    x = np.arange(len(labels) + 1)
    colours = [COLOR["baseline"]] + [COLOR[a] for a in labels]

    axes[0].bar(x, [base_rate, *rates], color=colours,
                yerr=[base_rate_err, *rate_err], capsize=4,
                error_kw=dict(ecolor="#333", lw=1.2))
    axes[0].set_xticks(x, ["baseline", *labels], fontsize=9)
    axes[0].set_ylabel("fraction of items terminating")
    axes[0].set_title("a  termination rate")
    axes[0].set_ylim(0, 1.15)
    for xi, value, err in zip(x, [base_rate, *rates], [base_rate_err, *rate_err], strict=True):
        axes[0].text(xi, value + err + 0.04, f"{value:.2f}", ha="center", fontsize=8.5)

    axes[1].bar(x, [0.0, *saved], color=colours, yerr=[0.0, *saved_err], capsize=4,
                error_kw=dict(ecolor="#333", lw=1.2))
    axes[1].axhline(0, color="#555", lw=1)
    axes[1].set_xticks(x, ["baseline", *labels], fontsize=9)
    axes[1].set_title("b  change in reasoning length")
    span = max(abs(min(saved + [0])), abs(max(saved + [0]))) or 1.0
    for xi, value, err, p in zip(x[1:], saved, saved_err, pvals, strict=True):
        sign = 1 if value >= 0 else -1
        axes[1].text(xi, value + sign * (err + 0.07 * span), stars(p), ha="center", fontsize=9)
    # keep the significance labels inside the axes
    low = min(saved + [0]) - max(saved_err + [0]) - 0.22 * span
    high = max(saved + [0]) + max(saved_err + [0]) + 0.22 * span
    axes[1].set_ylim(low, high)
    axes[1].set_ylabel("mean token change vs own baseline\n(negative = shorter)")

    fig.suptitle(f"STOP steering at dose {dose}",
                 fontsize=11.5, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = out / f"dose{dose}_bars.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def accuracy_efficiency(pilot: pd.DataFrame, accuracy: pd.DataFrame, dose: float, out: Path):
    """Two standalone figures: accuracy per arm, and the accuracy/length trade-off.

    Accuracy is **strict and unconditional**: an item counts only if a final answer is emitted and
    every gold person-role pair in it is right. Conditioning on having answered would rank the arms
    by how selectively they answer instead of by how often they are right, which at this dose
    reverses the ordering. The graded per-pair score is kept for the decomposition figure.
    """
    merged = pilot.merge(accuracy[["item", "arm", "dose", "accuracy_cond", "accuracy_unc",
                                   "answered", "all_correct"]],
                         on=["item", "arm", "dose"], how="left")
    merged = collapse_draws(merged)
    base = merged[merged["arm"] == "baseline"].set_index("item")
    arms = [a for a in ORDER if a != "baseline"
            and len(merged[(merged["arm"] == a) & (merged["dose"] == dose)])]
    names = ["baseline", *arms]

    acc, acc_err, tok, tok_err, acc_p, answered = [], [], [], [], [], []
    acc.append(float(base["all_correct"].mean()))
    acc_err.append(sem(base["all_correct"].values))
    tok.append(float(base["n_generated"].mean()))
    tok_err.append(sem(base["n_generated"].values))
    acc_p.append(None)
    answered.append(float(base["answered"].mean()))
    for arm in arms:
        sub = merged[(merged["arm"] == arm) & (merged["dose"] == dose)].set_index("item")
        common = sub.index.intersection(base.index)
        acc.append(float(sub.loc[common, "all_correct"].mean()))
        acc_err.append(sem(sub.loc[common, "all_correct"].values))
        tok.append(float(sub.loc[common, "n_generated"].mean()))
        tok_err.append(sem(sub.loc[common, "n_generated"].values))
        answered.append(float(sub.loc[common, "answered"].mean()))
        delta = (sub.loc[common, "all_correct"].astype(float)
                 - base.loc[common, "all_correct"].astype(float)).dropna()
        acc_p.append(float(stats.wilcoxon(delta).pvalue) if (delta != 0).any() else 1.0)

    # --- figure 1: accuracy per arm ---
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    x = np.arange(len(names))
    ax.bar(x, acc, color=[COLOR[n] for n in names], yerr=acc_err, capsize=4,
           error_kw=dict(ecolor="#333", lw=1.2))
    ax.axhline(acc[0], color="#555", lw=1, ls="--")
    ax.set_xticks(x, [f"{n}\nanswers {a:.0%}" for n, a in zip(names, answered, strict=True)],
                  fontsize=8)
    ax.set_ylabel("all gold pairs correct\n(per item; no answer = 0)")
    ax.set_ylim(0, 1.15)
    ax.set_title(f"Final-answer accuracy per item (dose {dose})")
    for xi, value, err, p in zip(x, acc, acc_err, acc_p, strict=True):
        tag = f"{value:.2f}" + ("" if p is None else f"\n{stars(p)}")
        ax.text(xi, value + err + 0.04, tag, ha="center", fontsize=8.5)
    fig.tight_layout()
    path = out / f"accuracy_dose{dose}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")

    # --- figure 2: accuracy against length, legend instead of inline labels ---
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for name, a, ae, t, te in zip(names, acc, acc_err, tok, tok_err, strict=True):
        ax.errorbar(t, a, yerr=ae, xerr=te, fmt="o", ms=11, color=COLOR[name],
                    ecolor="#333", elinewidth=1.1, capsize=3, zorder=3,
                    label="baseline (no injection)" if name == "baseline" else f"v_{name}")
    ax.set_xlabel("mean tokens generated")
    ax.set_ylabel("all gold pairs correct (per item)")
    ax.set_ylim(0, 1.0)
    # up and to the left is better: more correct answers from fewer tokens
    ax.set_title(f"Accuracy against length (dose {dose})")
    ax.legend(frameon=False, fontsize=8.5, title="injected vector", title_fontsize=8.5,
              loc="lower left")
    fig.tight_layout()
    path = out / f"accuracy_vs_length_dose{dose}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def prompt_lengths() -> dict[str, int]:
    """Prompt token count per item, read from the bundle rather than assumed.

    The pilot stores ``prefix_len`` (prompt + reasoning so far) and ``budget``, but not the prompt
    length, so "how much *reasoning* happened before injection" cannot be read off it directly.
    """
    tokens = pd.read_parquet(runframe.BUNDLE / "tokens.parquet",
                             columns=["item_id", "model", "task_family", "segment"])
    tokens = tokens[(tokens.model == "phi4mr") & (tokens.task_family == runframe.TASK)]
    counts = tokens[tokens.segment == "prompt"].groupby("item_id").size()
    require(len(counts), "no prompt segments found in the bundle")
    return {str(k): int(v) for k, v in counts.items()}


def add_token_totals(frame: pd.DataFrame, prompts: dict[str, int]) -> pd.DataFrame:
    """Split the trace into reasoning-before-injection and reasoning-after, in completion tokens.

    The pilot's ``budget`` is what the corpus had *left* at the injection point, i.e.
    ``cap - completion_before``. That makes ``budget + completion_before`` a constant, and the
    constant is the corpus's own completion cap -- so deriving the prompt length from the bundle
    and then checking that identity validates the whole decomposition instead of hard-coding 8192.
    """
    frame = frame.copy()
    missing = set(frame["item"]) - set(prompts)
    require(not missing, f"no prompt length for items: {sorted(missing)[:5]}")
    frame["prompt_len"] = frame["item"].map(prompts).astype(int)
    frame["completion_before"] = frame["prefix_len"].astype(int) - frame["prompt_len"]
    frame["total_completion"] = frame["completion_before"] + frame["n_generated"].astype(int)

    caps = (frame["budget"].astype(int) + frame["completion_before"]).unique()
    require(len(caps) == 1,
            f"budget + completion_before is not constant ({sorted(caps)[:5]}); the prompt "
            "lengths do not match the run, so the decomposition would be wrong")
    require((frame["completion_before"] >= 0).all(), "negative reasoning-before-injection")
    require((frame["total_completion"] <= caps[0]).all(), "total exceeds the corpus cap")
    return frame.attrs.update(corpus_cap=int(caps[0])) or frame


def _panel_distribution(axes, frame, arms, column, dose, cap, xlabel):
    """Shared two-panel body: histogram over generations, then every generation individually."""
    bins = np.linspace(0, cap or frame[column].max(), 28)
    for arm in arms:
        sub = (frame[frame["arm"] == "baseline"] if arm == "baseline"
               else frame[(frame["arm"] == arm) & (frame["dose"] == dose)])
        axes[0].hist(sub[column], bins=bins, histtype="step", lw=2,
                     color=COLOR[arm], label=f"{arm} (n={len(sub)})")
        axes[0].hist(sub[column], bins=bins, color=COLOR[arm], alpha=0.18)
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel("number of generations")
    axes[0].set_title("a  distribution over all generations")
    axes[0].legend(frameon=False, fontsize=8.5)

    rng = np.random.default_rng(0)
    for i, arm in enumerate(arms):
        sub = (frame[frame["arm"] == "baseline"] if arm == "baseline"
               else frame[(frame["arm"] == arm) & (frame["dose"] == dose)])
        jitter = rng.uniform(-0.11, 0.11, len(sub))
        axes[1].scatter(np.full(len(sub), i) + jitter, sub[column], s=26,
                        color=COLOR[arm], alpha=0.75, zorder=3)
        axes[1].hlines(sub[column].median(), i - 0.25, i + 0.25, color="#222", lw=2, zorder=4)
    if cap:
        axes[1].axhline(cap, color="#555", lw=1.2, ls=":")
    axes[1].set_xticks(range(len(arms)), arms, fontsize=9)
    axes[1].set_ylabel(xlabel)
    axes[1].set_title("b  every generation (bar = median)")


def total_token_distribution(frame: pd.DataFrame, prompts: dict[str, int], out: Path,
                             dose: float = 0.25, suffix: str = ""):
    """Whole-trace completion length: reasoning before injection plus everything generated after.

    The companion figure counts only what follows the injection, which is the causal quantity.
    This one is what a reader actually waits for, and it is deliberately *less* separated between
    arms: the reasoning before injection is identical across arms, so it dilutes the difference.
    Both are needed -- the after-injection view alone overstates the end-to-end saving.
    """
    frame = add_token_totals(frame, prompts)
    cap = frame.attrs["corpus_cap"]
    arms = [a for a in ("baseline", "stop", "shuffled", "switch", "solved")
            if a in set(frame["arm"])]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
    _panel_distribution(axes, frame, arms, "total_completion", dose, cap,
                        "total completion tokens (before + after injection)")
    axes[0].axvline(cap, color="#555", lw=1.2, ls=":")
    fig.suptitle(f"Total tokens per generation, dose {dose} — cap {cap}",
                 fontsize=11.5, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = out / f"total_token_distribution_dose{dose}{suffix}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def injection_point_figure(frame: pd.DataFrame, prompts: dict[str, int], out: Path,
                           dose: float = 0.25):
    """Where the injection happens, in completion tokens.

    Not split by arm on purpose: the injection point is fixed per item by ``T_solved`` on the
    unperturbed trace, so every arm shares it. Drawing it per arm would show identical
    distributions and invite the reader to compare noise. The function asserts that invariance
    rather than trusting it.
    """
    frame = add_token_totals(frame, prompts)
    cap = frame.attrs["corpus_cap"]
    per_item = frame.groupby("item")
    require((per_item["completion_before"].nunique() == 1).all(),
            "injection point differs between arms of the same item")
    points = per_item["completion_before"].first().sort_values()
    solved = per_item["t_solved"].first()
    base = frame[frame["arm"] == "baseline"].set_index("item")["total_completion"]

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))
    axes[0].hist(points.values, bins=12, color=COLOR["baseline"], alpha=0.85,
                 edgecolor="white")
    # rug: with n = 17 and two far outliers the bars alone suggest empty regions rather than
    # single items, so every item is also drawn as a tick
    axes[0].plot(points.values, np.full(len(points), -0.22), "|", color="#333", ms=8,
                 clip_on=False)
    axes[0].set_xlabel("reasoning tokens before injection")
    axes[0].set_ylabel("number of items")
    axes[0].set_title(f"a  injection point (n = {len(points)} items)")
    median = float(np.median(points.values))
    axes[0].axvline(median, color="#E76F51", lw=1.8, ls="--")
    axes[0].text(median, axes[0].get_ylim()[1] * 0.94, f" median {median:.0f}",
                 color="#E76F51", fontsize=8.5, va="top")

    y = np.arange(len(points))
    axes[1].barh(y, [base.get(i, np.nan) for i in points.index], color="#D8D8DE",
                 label="baseline total")
    axes[1].barh(y, points.values, color=COLOR["stop"], label="before injection")
    axes[1].axvline(cap, color="#555", lw=1.2, ls=":")
    axes[1].set_yticks(y, [f"{i.replace('item_', '')} (T={solved[i]})" for i in points.index],
                       fontsize=6.5)
    axes[1].set_xlabel("completion tokens")
    axes[1].set_title("b  injection point within each item's baseline trace")
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")

    fig.suptitle("Where the STOP vector is injected",
                 fontsize=11.5, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = out / "injection_point.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def token_counts(pilot: pd.DataFrame, dose: float, out: Path, suffix: str = ""):
    """Absolute tokens generated after the injection point, per arm.

    The change-from-baseline plots answer "how much shorter", this answers "how long is it".
    Both matter: a large reduction from a capped baseline and a small one from a short baseline
    can look identical in the difference.
    """
    frame = collapse_draws(pilot)
    arms = ["baseline"] + [a for a in ORDER if a != "baseline"
                           and len(frame[(frame["arm"] == a) & (frame["dose"] == dose)])]
    means, errs = [], []
    for arm in arms:
        sub = (frame[frame["arm"] == "baseline"] if arm == "baseline"
               else frame[(frame["arm"] == arm) & (frame["dose"] == dose)])
        means.append(float(sub["n_generated"].mean()))
        errs.append(sem(sub["n_generated"].values))

    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    x = np.arange(len(arms))
    ax.bar(x, means, color=[COLOR[a] for a in arms], yerr=errs, capsize=4,
           error_kw=dict(ecolor="#333", lw=1.2))
    budget = int(pilot["budget"].median()) if "budget" in pilot else None
    if budget:
        ax.axhline(budget, color="#555", lw=1.2, ls=":")
        ax.text(-0.45, budget * 1.015, "generation budget", fontsize=8,
                color="#555", ha="left", va="bottom")
        ax.set_ylim(0, budget * 1.14)
    ax.set_xticks(x, arms, fontsize=9)
    ax.set_ylabel("tokens generated after injection")
    ax.set_title(f"Tokens used after injection (dose {dose})"
                 + (" — corpus budget" if suffix else ""))
    for xi, value, err in zip(x, means, errs, strict=True):
        ax.text(xi, value + err + (0.025 * max(means)), f"{value:.0f}", ha="center",
                fontsize=8.5)
    fig.tight_layout()
    path = out / f"token_counts_dose{dose}{suffix}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def token_distribution(sampling: pd.DataFrame, out: Path, dose: float = 0.5, suffix: str = ""):
    """Every individual generation, not just its arm mean.

    Plotted at the draw level on purpose: this figure is descriptive, and the point is the shape
    of the distribution -- how bimodal it is, how many runs sit against the budget -- which an
    item-averaged summary hides. No inference is drawn from these rows.
    """
    arms = [a for a in ("baseline", "stop", "shuffled") if a in set(sampling["arm"])]
    budget = int(sampling["budget"].median()) if "budget" in sampling else None
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))

    bins = np.linspace(0, budget or sampling["n_generated"].max(), 28)
    for arm in arms:
        sub = (sampling[sampling["arm"] == "baseline"] if arm == "baseline"
               else sampling[(sampling["arm"] == arm) & (sampling["dose"] == dose)])
        # step outlines: three filled histograms piled at the budget are unreadable
        axes[0].hist(sub["n_generated"], bins=bins, histtype="step", lw=2,
                     color=COLOR[arm], label=f"{arm} (n={len(sub)})")
        axes[0].hist(sub["n_generated"], bins=bins, color=COLOR[arm], alpha=0.18)
    axes[0].set_xlabel("tokens generated after injection")
    axes[0].set_ylabel("number of generations")
    axes[0].set_title("a  distribution over all generations")
    axes[0].legend(frameon=False, fontsize=8.5)

    rng = np.random.default_rng(0)
    for i, arm in enumerate(arms):
        sub = (sampling[sampling["arm"] == "baseline"] if arm == "baseline"
               else sampling[(sampling["arm"] == arm) & (sampling["dose"] == dose)])
        jitter = rng.uniform(-0.11, 0.11, len(sub))
        axes[1].scatter(np.full(len(sub), i) + jitter, sub["n_generated"], s=26,
                        color=COLOR[arm], alpha=0.75, zorder=3)
        axes[1].hlines(sub["n_generated"].median(), i - 0.25, i + 0.25, color="#222",
                       lw=2, zorder=4)
    if budget:
        axes[1].axhline(budget, color="#555", lw=1.2, ls=":")
    axes[1].set_xticks(range(len(arms)), arms, fontsize=9)
    axes[1].set_ylabel("tokens generated after injection")
    axes[1].set_title("b  every generation (bar = median)")

    fig.suptitle(f"Token use per generation, dose {dose}"
                 + (" — corpus budget" if suffix else ""),
                 fontsize=11.5, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = out / f"token_distribution_dose{dose}{suffix}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def layer_sweep(layer_dirs, out: Path):
    """Two standalone figures over injection depth: termination rate, and token change.

    Depth on the x-axis, one curve per dose, colour-grouped by injected vector. Recomputed from
    per-item rows so the error bars are real SEMs -- the pre-aggregated grid holds only cell
    means, which cannot yield a standard error.
    """
    frames = []
    for layer, directory in layer_dirs.items():
        path = Path(directory) / "pilot.parquet"
        if not path.exists():
            continue
        frame = collapse_draws(pd.read_parquet(path))
        frame["layer"] = layer
        frames.append(frame)
    require(frames, "no layer runs found")
    table = pd.concat(frames, ignore_index=True)
    layers = sorted(table["layer"].unique())
    doses = sorted(d for d in table["dose"].unique() if d > 0)

    # one colour family per condition, darker with dose
    families = {"stop": plt.get_cmap("Oranges"), "shuffled": plt.get_cmap("BuGn")}
    shades = np.linspace(0.45, 0.9, max(len(doses), 2))

    def series(arm, dose, value):
        """(layers, mean, sem) for one arm/dose; token change is paired per item."""
        xs, means, errs = [], [], []
        for layer in layers:
            cell = table[(table["layer"] == layer) & (table["arm"] == arm)
                         & (table["dose"] == dose)].set_index("item")
            if not len(cell):
                continue
            if value == "terminated":
                values = cell["terminated"].astype(float)
            else:
                base = table[(table["layer"] == layer)
                             & (table["arm"] == "baseline")].set_index("item")
                common = cell.index.intersection(base.index)
                values = (cell.loc[common, "n_generated"].astype(float)
                          - base.loc[common, "n_generated"].astype(float))
            xs.append(layer)
            means.append(float(values.mean()))
            errs.append(sem(values.values))
        return xs, means, errs

    specs = [
        ("terminated", "termination rate", "Injection depth: termination rate",
         "layer_sweep_termination.png", True),
        ("token_change", "mean token change vs own baseline\n(negative = shorter)",
         "Injection depth: change in reasoning length",
         "layer_sweep_tokens.png", False),
    ]
    for value, ylabel, title, filename, show_baseline in specs:
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        for arm, cmap in families.items():
            for di, dose in enumerate(doses):
                xs, means, errs = series(arm, dose, value)
                if not xs:
                    continue
                ax.errorbar(xs, means, yerr=errs, color=cmap(shades[di]), marker="o",
                            ms=6, capsize=3, lw=2.2 if arm == "stop" else 1.4,
                            ls="-" if arm == "stop" else "--",
                            label=f"v_{arm} · dose {dose}")
        if show_baseline:
            base = table[table["arm"] == "baseline"]
            ax.axhline(float(base["terminated"].mean()), color="#555", lw=1.2, ls=":",
                       label="baseline")
        else:
            ax.axhline(0, color="#555", lw=1.2, ls=":")
        ax.set_xlabel("injection layer")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(layers, [f"L{i}" for i in layers])
        # outside the axes: the L16 peak and its error bar own the top of the plot
        ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
        fig.tight_layout()
        path = out / filename
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {path}")


def run(dose_dir: Path, accuracy_path: Path, layer_dirs: dict, out: Path, dose: float = 0.5,
        sampling: Path | None = None, fullcap: Path | None = None):
    out.mkdir(parents=True, exist_ok=True)
    pilot = pd.read_parquet(dose_dir / "pilot.parquet")
    dose_bars(pilot, dose, out)
    token_counts(pilot, dose, out)

    accuracy = load_accuracy(accuracy_path, run=dose_dir.name)
    accuracy_efficiency(pilot, accuracy, dose, out)

    layer_sweep(layer_dirs, out)
    prompts = prompt_lengths()
    # the injection point is a property of the item cohort, so it is read off the greedy run
    injection_point_figure(pilot, prompts, out, dose=dose)
    total_token_distribution(pilot, prompts, out, dose=dose, suffix="_greedy")
    if sampling is not None and (sampling / "pilot.parquet").exists():
        draws = pd.read_parquet(sampling / "pilot.parquet")
        token_distribution(draws, out, dose=dose)
        total_token_distribution(draws, prompts, out, dose=dose)
    if fullcap is not None and (fullcap / "pilot.parquet").exists():
        # the corpus's own budget: per item it is 8192 minus what was already generated, so
        # the cap line is drawn at the median rather than a single shared value
        table = pd.read_parquet(fullcap / "pilot.parquet")
        token_counts(table, dose, out, suffix="_fullcap")
        token_distribution(table, out, dose=dose, suffix="_fullcap")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dose-dir", type=Path, required=True)
    parser.add_argument("--accuracy", type=Path, required=True)
    parser.add_argument("--l8", type=Path, required=True)
    parser.add_argument("--l16", type=Path, required=True)
    parser.add_argument("--l24", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dose", type=float, default=0.5)
    parser.add_argument("--sampling", type=Path, default=None)
    parser.add_argument("--fullcap", type=Path, default=None)
    args = parser.parse_args()
    run(args.dose_dir, args.accuracy, {8: args.l8, 16: args.l16, 24: args.l24},
        args.out, dose=args.dose, sampling=args.sampling, fullcap=args.fullcap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
