"""Figures for the TSP transfer test.

Five panels, one per claim the findings can make:

1. ``fig_a_tokens.png``     -- paired token change by arm and dose, the primary outcome;
2. ``fig_b_location.png``   -- the same effect against *where* injection started, which is the
   question this experiment adds over the ``identify`` work;
3. ``fig_c_termination.png``-- termination-before-cap by arm and location;
4. ``fig_d_accuracy.png``   -- answer health: still commits to an answer, still gets it right;
5. ``fig_e_survival.png``   -- censored total-length curves, the honest view when a control is
   still reasoning at the cap.

Error bars are SEM across **units** (a unit is one ``task x baseline trace x offset``), with the
draws inside a unit averaged first -- draws are clustered, so treating them as independent points
would understate the error. Figure text carries no sample sizes, matching the `stop_steering`
convention: the cohort travels in the theme README, not on the slide.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import analyze  # noqa: E402

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "axes.titlesize": 11, "axes.titleweight": "bold",
                     "axes.titlelocation": "left"})
COLOR = {"baseline": "#8A8A95", "stop": "#E76F51", "shuffled": "#2A9D8F"}
LEVEL_MARKER = {"level 1": "o", "level 2": "s", "level 3": "^"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sem(values):
    values = np.asarray(values, dtype=float)
    return float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0


def label(arm, dose):
    return "control\n(α=0)" if arm == "baseline" else f"{arm}\n{dose:g}"


def arm_order(collapsed):
    pairs = collapsed[["arm", "dose"]].drop_duplicates()
    return sorted((r.arm, float(r.dose)) for r in pairs.itertuples())


def km_curve(times, events):
    """Kaplan-Meier survival, where an 'event' is terminating before the cap."""
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
    return np.asarray(xs), np.asarray(ys)


def fig_tokens(collapsed, out: Path):
    """Paired change in total response length, control included as the zero reference."""
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    order = arm_order(collapsed)
    for index, (arm, dose) in enumerate(order):
        group = collapsed[(collapsed["arm"] == arm) & (collapsed["dose"] == dose)]
        values = group["d_total_tokens"].values
        ax.bar(index, np.mean(values), yerr=sem(values), color=COLOR.get(arm, "#999"),
               width=0.62, capsize=4, edgecolor="none")
        ax.scatter(np.full(len(values), index) + np.random.default_rng(0).uniform(
            -0.13, 0.13, len(values)), values, s=11, color="black", alpha=0.35, zorder=3)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([label(a, d) for a, d in order])
    ax.set_ylabel("change in total response tokens\nvs the α=0 fork")
    ax.set_title("Injecting the frozen STOP direction into TSP reasoning")
    fig.tight_layout()
    fig.savefig(out / "fig_a_tokens.png")
    plt.close(fig)


def fig_location(collapsed, out: Path):
    """Does it matter where in the trace injection starts? Shown on both denominators.

    The left panel is the paired token saving, which *shrinks* with depth for a near-trivial
    reason: less remains to cut. The right panel is the change relative to the control's remaining
    generation, which is the denominator the depth question actually needs -- and it moves the
    other way. Plotting only the left panel invites the wrong reading.

    Both panels are signed the same way (negative = shorter), so a curve does not flip sense
    between them.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    treated = [p for p in arm_order(collapsed) if p[0] != "baseline"]
    locations = sorted(collapsed["location"].unique())
    doses = sorted({d for _, d in treated})

    for ax, field, ylabel, title, ref in [
            (axes[0], "d_total_tokens", "change in total response tokens",
             "Absolute saving (shrinks: less left to cut)", 0.0),
            (axes[1], "change_share", "fractional change vs the control's\nremaining generation",
             "Relative to what remained (the fair denominator)", 0.0)]:
        for arm, dose in treated:
            xs, ys, es = [], [], []
            for location in locations:
                group = collapsed[(collapsed["arm"] == arm) & (collapsed["dose"] == dose)
                                  & (collapsed["location"] == location)]
                if not len(group):
                    continue
                # signed like the left panel: negative = shorter, so both curves keep one sense
                values = (group["remaining_ratio"].values - 1.0 if field == "change_share"
                          else group[field].values)
                xs.append(location)
                ys.append(float(np.mean(values)))
                es.append(sem(values))
            style = ["-", "--", ":"][doses.index(dose) % 3]
            ax.errorbar(xs, ys, yerr=es, marker="o" if dose == doses[0] else "s", linestyle=style,
                        capsize=3, color=COLOR.get(arm, "#999"), label=f"{arm} {dose:g}")
        ax.axhline(ref, color="black", linewidth=0.8)
        ax.set_xticks(locations)
        ax.set_xticklabels([f"{int(v * 100)}%" for v in locations])
        ax.set_xlabel("injection start, as a fraction of the reasoning trace")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    axes[0].legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out / "fig_b_location.png")
    plt.close(fig)


def fig_termination(collapsed, out: Path):
    """Termination before the cap, control against each treated arm, split by location."""
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    locations = sorted(collapsed["location"].unique())
    order = arm_order(collapsed)
    doses = sorted({d for a, d in order if a != "baseline"})
    width = 0.8 / max(len(order), 1)
    for index, (arm, dose) in enumerate(order):
        xs, ys, es = [], [], []
        for position, location in enumerate(locations):
            group = collapsed[(collapsed["arm"] == arm) & (collapsed["dose"] == dose)
                              & (collapsed["location"] == location)]
            if not len(group):
                continue
            xs.append(position + index * width - 0.4 + width / 2)
            ys.append(float(group["terminated"].mean()))
            es.append(sem(group["terminated"].values))
        # within an arm the dose is carried by shade, so same-colour bars stay distinguishable
        shade = 1.0 if arm == "baseline" else 1.0 - 0.45 * doses.index(dose) / max(len(doses), 1)
        ax.bar(xs, ys, yerr=es, width=width * 0.9, capsize=3, color=COLOR.get(arm, "#999"),
               alpha=shade, edgecolor="none", label=label(arm, dose).replace("\n", " "))
    ax.set_xticks(range(len(locations)))
    ax.set_xticklabels([f"{int(v * 100)}%" for v in locations])
    ax.set_xlabel("injection start")
    ax.set_ylabel("terminated before the cap")
    ax.set_ylim(0, 1.05)
    ax.set_title("Does the direction end the trace?")
    # below the axes: at high termination rates the bars fill the panel, and anchoring above the
    # axes collides with the left-aligned title
    ax.legend(frameon=False, ncol=4, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.16))
    fig.tight_layout()
    fig.savefig(out / "fig_c_termination.png")
    plt.close(fig)


def fig_accuracy(collapsed, out: Path):
    """Damage check: a shorter trace is only interesting if the answer survives."""
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    order = arm_order(collapsed)
    xs = np.arange(len(order))
    for offset, (field, name, colour) in enumerate(
            [("answered", "commits to an answer", "#4C72B0"),
             ("correct", "answer is optimal", "#DD8452")]):
        ys = [float(collapsed[(collapsed["arm"] == a) & (collapsed["dose"] == d)][field].mean())
              for a, d in order]
        es = [sem(collapsed[(collapsed["arm"] == a) & (collapsed["dose"] == d)][field].values)
              for a, d in order]
        ax.bar(xs + (offset - 0.5) * 0.38, ys, yerr=es, width=0.34, capsize=3, color=colour,
               edgecolor="none", label=name)
    ax.set_xticks(xs)
    ax.set_xticklabels([label(a, d) for a, d in order])
    ax.set_ylabel("rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Answer health under injection")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig_d_accuracy.png")
    plt.close(fig)


def fig_survival(frame, out: Path):
    """Censored view: many controls are still reasoning at the cap, so medians alone mislead."""
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for (arm, dose), group in frame.groupby(["arm", "dose"]):
        xs, ys = km_curve(group["total_tokens"].values, group["terminated"].values)
        ax.step(xs, ys, where="post", color=COLOR.get(arm, "#999"),
                linestyle="-" if arm != "shuffled" else "--",
                alpha=1.0 if arm != "stop" or dose == 0.25 else 0.55,
                label=label(arm, dose).replace("\n", " "))
    ax.set_xlabel("total response tokens")
    ax.set_ylabel("still reasoning")
    ax.set_ylim(0, 1.02)
    ax.set_title("Time to stop, censored at the cap")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_e_survival.png")
    plt.close(fig)


def fig_accuracy_vs_length(collapsed, out: Path):
    """Answer health against length, the TSP analogue of the `stop_steering` dose-0.25 panel.

    Up-and-to-the-left is better: more answers from fewer tokens. Two panels rather than the one
    the `identify` version uses, because TSP accuracy is low (0.04-0.17) and a single 0-1 axis
    would flatten every arm onto the floor -- `answered` carries most of the signal, `correct` is
    the strict criterion, and collapsing them would hide which one moves.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    order = [("baseline", 0.0), *[p for p in arm_order(collapsed) if p[0] != "baseline"]]
    doses = sorted({d for a, d in order if a != "baseline"})
    markers = ["o", "s", "D"]
    handles = []
    for ax, field, title in [(axes[0], "answered", "Commits to a parseable answer"),
                             (axes[1], "correct", "Answer is a valid optimal tour")]:
        for arm, dose in order:
            group = collapsed[(collapsed["arm"] == arm) & (collapsed["dose"] == dose)]
            if not len(group):
                continue
            tokens = group["n_generated"].values.astype(float)
            value = group[field].values.astype(float)
            # arm sets colour, dose sets marker and shade, so same-colour arms stay separable
            index = 0 if arm == "baseline" else doses.index(dose)
            bar = ax.errorbar(
                np.mean(tokens), np.mean(value), yerr=sem(value), xerr=sem(tokens),
                fmt=markers[index % len(markers)], ms=11, color=COLOR.get(arm, "#999"),
                alpha=1.0 if arm == "baseline" else 1.0 - 0.35 * index,
                ecolor="#333", elinewidth=1.1, capsize=3, zorder=3,
                label="control (no injection)" if arm == "baseline" else f"v_{arm} @ {dose:g}")
            if ax is axes[0]:
                handles.append(bar)
        ax.set_xlabel("mean tokens generated after injection")
        ax.set_ylabel(field)
        ax.set_ylim(bottom=0)
        ax.margins(x=0.18, y=0.22)
        ax.set_title(title)
    # legend below the axes: the control sits top-right in both panels, where an inset would land
    fig.legend(handles=handles, frameon=False, fontsize=9, ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, -0.04), title="injected vector", title_fontsize=9)
    fig.suptitle("Answer health against length — up and to the left is better", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(out / "fig_f_accuracy_vs_length.png", bbox_inches="tight")
    plt.close(fig)


def lighten(colour, amount):
    """Blend toward white, so two doses of one arm stay the same hue but separate visually."""
    rgb = np.array(matplotlib.colors.to_rgb(colour))
    return tuple(rgb + (1.0 - rgb) * float(amount))


def _strip(ax, groups, values, colors, budget=None, budget_label=None):
    """Every observation as a jittered point, with a median bar -- the reference's panel b."""
    rng = np.random.default_rng(0)
    for index, (series, color) in enumerate(zip(values, colors, strict=True)):
        jitter = rng.uniform(-0.11, 0.11, len(series))
        ax.scatter(np.full(len(series), index) + jitter, series, s=26, color=color,
                   alpha=0.75, zorder=3)
        ax.hlines(np.median(series), index - 0.25, index + 0.25, color="#222", lw=2, zorder=4)
    if budget is not None:
        ax.axhline(budget, color="#555", lw=1.2, ls=":")
        if budget_label:
            ax.text(len(groups) - 0.5, budget, budget_label, fontsize=8, color="#555",
                    va="bottom", ha="right")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, fontsize=9)


def fig_token_distribution(frame, out: Path):
    """Total response length per generation, every run shown rather than an arm mean.

    Descriptive on purpose and plotted at the **draw** level: the point is the shape -- how bimodal
    it is, how many runs pile against the cap -- which a per-unit average hides. No inference is
    drawn from these rows; the tests in `analyze` operate on per-unit means.

    Total length, not tokens-after-injection, because the arms at one injection point share a
    prefix: the total is what a reader of the response actually sees.
    """
    order = [("baseline", 0.0), *[p for p in arm_order(frame) if p[0] != "baseline"]]
    doses = sorted({d for a, d in order if a != "baseline"})
    cap = int(frame["budget"].max() + frame["offset"].min())
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    groups, values, colors = [], [], []
    bins = np.linspace(0, frame["total_tokens"].max(), 28)
    for arm, dose in order:
        sub = frame[(frame["arm"] == arm) & (frame["dose"] == dose)]
        if not len(sub):
            continue
        name = "control" if arm == "baseline" else f"{arm} {dose:g}"
        # same hue per arm, lightened by dose, so stop@0.25 and stop@0.5 stay distinguishable
        step = 0.0 if arm == "baseline" else 0.42 * doses.index(dose)
        colour = lighten(COLOR.get(arm, "#999"), step)
        # step outlines: filled histograms piled at the cap are unreadable
        axes[0].hist(sub["total_tokens"], bins=bins, histtype="step", lw=2, color=colour,
                     label=f"{name} (n={len(sub)})")
        axes[0].hist(sub["total_tokens"], bins=bins, color=colour, alpha=0.18)
        groups.append(name)
        values.append(sub["total_tokens"].values)
        colors.append(colour)
    axes[0].set_xlabel("total response tokens")
    axes[0].set_ylabel("number of generations")
    axes[0].set_title("a  distribution over all generations")
    axes[0].legend(frameon=False, fontsize=8.5)

    _strip(axes[1], groups, values, colors, budget=cap, budget_label="cap")
    axes[1].set_ylabel("total response tokens")
    axes[1].set_title("b  every generation (bar = median)")

    fig.suptitle("Total response length per generation", fontsize=11.5, fontweight="bold",
                 x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "fig_g_token_distribution.png", bbox_inches="tight")
    plt.close(fig)


def fig_injection_point(frame, out: Path):
    """How many tokens of reasoning were kept before injection, in absolute terms.

    Split by **offset fraction, not by arm**: within a cell every arm forks from the identical
    prefix, so an arm-wise split would draw the same distribution three times. One point per
    ``(task, baseline trace, offset)`` for the same reason -- the 216 runs contain only 27 distinct
    injection points, and plotting all of them would replicate each eight times and imply an n the
    design does not have.
    """
    unique = frame.drop_duplicates(["uid", "base_draw", "location"])
    locations = sorted(unique["location"].unique())
    shades = ["#9CC3D5", "#4A90A4", "#1F5673"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    groups, values, colors = [], [], []
    bins = np.linspace(0, unique["offset"].max() * 1.05, 24)
    for index, location in enumerate(locations):
        sub = unique[unique["location"] == location]
        colour = shades[index % len(shades)]
        axes[0].hist(sub["offset"], bins=bins, histtype="step", lw=2, color=colour,
                     label=f"{location:.0%} of reasoning (n={len(sub)})")
        axes[0].hist(sub["offset"], bins=bins, color=colour, alpha=0.18)
        groups.append(f"{location:.0%}")
        values.append(sub["offset"].values)
        colors.append(colour)
    axes[0].set_xlabel("tokens before the injection point")
    axes[0].set_ylabel("number of injection points")
    axes[0].set_title("a  distribution over injection points")
    axes[0].legend(frameon=False, fontsize=8.5)

    _strip(axes[1], groups, values, colors)
    axes[1].set_xlabel("injection start, as a fraction of the reasoning trace")
    axes[1].set_ylabel("tokens before the injection point")
    axes[1].set_title("b  every injection point (bar = median)")

    fig.suptitle("Where injection begins, in absolute tokens", fontsize=11.5,
                 fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "fig_h_injection_point.png", bbox_inches="tight")
    plt.close(fig)


def run(directory: Path, out: Path | None = None):
    out = out or directory
    out.mkdir(parents=True, exist_ok=True)
    frame, _ = analyze.load_injections(directory)
    frame = analyze.rescore(directory, frame)
    collapsed = analyze.per_unit(analyze.paired(frame))
    # the control carries no delta of its own; add it as an explicit zero row so every panel
    # shows the reference rather than implying it
    control = frame[frame["arm"] == "baseline"]
    control_unit = (control.groupby([*analyze.UNIT, "arm", "dose", "level"], as_index=False)
                    .agg(total_tokens=("total_tokens", "mean"),
                         n_generated=("n_generated", "mean"),
                         terminated=("terminated", "mean"), correct=("correct", "mean"),
                         answered=("answered", "mean")))
    control_unit["d_total_tokens"] = 0.0
    panels = pd.concat([collapsed, control_unit], ignore_index=True)

    require(len(collapsed), "no treated rows to plot")
    fig_tokens(panels, out)
    fig_location(collapsed, out)
    fig_termination(panels, out)
    fig_accuracy(panels, out)
    fig_survival(frame, out)
    fig_accuracy_vs_length(panels, out)
    fig_token_distribution(frame, out)
    fig_injection_point(frame, out)
    made = sorted(p.name for p in out.glob("fig_*.png"))
    print("wrote: " + ", ".join(made))
    return made


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    run(args.dir, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
