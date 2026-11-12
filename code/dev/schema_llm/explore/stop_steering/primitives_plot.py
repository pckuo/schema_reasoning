"""Plot the primitive mix of generated text, by injected vector.

Two views of the same counts:

* **composition** -- what fraction of each arm's generated tokens falls in each primitive. Answers
  "does the STOP vector change *what the model does*, or only how long it does it for".
* **rate** -- primitive tokens per continuation. The composition view normalises length away, which
  is the right control for a mix question but hides that the STOP arm produces far fewer tokens
  overall; the rate view puts that back.

Both are needed: an arm can hold its composition constant while collapsing its token count, and
only the pair distinguishes "stops sooner" from "reasons differently".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ORDER = ["baseline", "stop", "shuffled", "switch", "solved"]
COLOR = {"baseline": "#8A8A95", "stop": "#E76F51", "shuffled": "#2A9D8F",
         "switch": "#6A4CC0", "solved": "#E9C46A"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sem(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0


def match_length(frame, reference="stop", dose=0.5):
    """Re-cut every arm to the reference arm's token count, per item.

    Primitives have characteristic positions in a continuation, so an arm that runs to the budget
    visits late-phase work that a short arm never reaches. Comparing their compositions therefore
    mixes "what work does it do" with "how far through the work did it get". Truncating all arms
    to the reference arm's length holds the second constant and isolates the first.

    Arms shorter than the reference are kept whole and counted in the returned report -- padding
    them would invent tokens, and dropping them would bias the comparison toward long arms.
    """
    require("labels" in frame.columns,
            "per-token `labels` missing; re-run the labeller to enable length matching")
    ref = frame[(frame["arm"] == reference) & (frame["dose"] == dose)]
    length_of = dict(zip(ref["item"], ref["labels"].apply(len), strict=True))
    require(length_of, f"no rows for reference arm {reference} at dose {dose}")

    rows, shorter = [], 0
    n_meta = 8
    for _, row in frame.iterrows():
        limit = length_of.get(row["item"])
        if limit is None:
            continue
        labels = np.asarray(row["labels"])[:limit]
        shorter += int(len(row["labels"]) < limit)
        counts = np.bincount(labels, minlength=n_meta)
        rows.append({**{k: row[k] for k in ("item", "arm", "dose")},
                     "n_tokens": int(len(labels)),
                     **{f"meta_{i}": int(counts[i]) for i in range(n_meta)}})
    out = pd.DataFrame(rows)
    return out, dict(reference=reference, dose=float(dose),
                     items=int(len(length_of)),
                     median_reference_tokens=float(np.median(list(length_of.values()))),
                     rows_shorter_than_reference=int(shorter), rows=int(len(out)))


def run(counts_path: Path, names_path: Path, out: Path, dose: float = 0.5,
        alphabet: str = "merged", match: str | None = None):
    """``alphabet``: ``merged`` = the shared cross-model primitive ids (what the bundle's
    ``merged_primitive_L16`` column holds); ``meta`` = this cell's 8 raw metaclusters."""
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(counts_path)
    suffix = ""
    if match:
        frame, report = match_length(frame, reference=match, dose=dose)
        suffix = f"_matched-{match}"
        print("length-matched:", json.dumps(report))
        (out / f"length_match_report_dose{dose}.json").write_text(json.dumps(report, indent=2))
        # the meta->merged mapping is an artifact of the labelling step, so it sits beside the
        # counts; reading it from `out` only worked while figures were written into that same dir
        mapping = json.loads((counts_path.parent / "meta_to_merged.json").read_text())
        for meta, info in mapping.items():
            col = f"merged_{info['merged_id']}"
            frame[col] = frame.get(col, 0) + frame[f"meta_{meta}"]
    prefix = "merged_" if alphabet == "merged" else "meta_"
    metas = sorted((c for c in frame.columns if c.startswith(prefix)),
                   key=lambda c: int(c.split("_")[1]))
    require(metas, f"no {prefix}* columns; re-run the labeller")
    names = {int(k): v for k, v in json.loads(names_path.read_text()).items()}
    ids = [int(c.split("_")[1]) for c in metas]
    n_meta = len(metas)
    labels = [f"{i}  {names.get(i, f'primitive {i}')}" for i in ids]

    arms = []
    for a in ORDER:
        sub = frame[frame["arm"] == "baseline"] if a == "baseline" else \
            frame[(frame["arm"] == a) & (frame["dose"] == dose)]
        if len(sub):
            arms.append((a, sub))
    require(arms, "no arms to plot")

    # ---- composition: share of each arm's generated tokens, averaged over items ----
    comp, comp_err, rate, rate_err = {}, {}, {}, {}
    for a, sub in arms:
        shares = sub[metas].to_numpy(float)
        totals = shares.sum(1, keepdims=True)
        shares = np.divide(shares, np.where(totals == 0, 1, totals))
        comp[a] = shares.mean(0)
        comp_err[a] = np.array([sem(shares[:, i]) for i in range(n_meta)])
        rate[a] = sub[metas].to_numpy(float).mean(0)
        rate_err[a] = np.array([sem(sub[metas].to_numpy(float)[:, i]) for i in range(n_meta)])

    x = np.arange(n_meta)
    width = 0.8 / len(arms)
    for kind, values, errors, ylabel, title in (
        ("composition", comp, comp_err,
         "fraction of that arm's generated tokens\n(each arm sums to 1)",
         f"Primitive composition after injection (dose {dose})"
         + (f" — all arms cut to v_{match}'s length" if match else "")),
        ("rate", rate, rate_err,
         "mean tokens per continuation\n(absolute, not normalised)",
         f"Primitive token counts after injection (dose {dose})"
         + (f" — all arms cut to v_{match}'s length" if match else "")),
    ):
        fig, ax = plt.subplots(figsize=(11.5, 5.2))
        for i, (a, _) in enumerate(arms):
            ax.bar(x + (i - (len(arms) - 1) / 2) * width, values[a], width,
                   yerr=errors[a], capsize=2, color=COLOR[a], label=a,
                   error_kw=dict(ecolor="#333", lw=0.9))
        ax.set_xticks(x, labels, rotation=30, ha="right", fontsize=8.5)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=9, ncol=len(arms))
        fig.tight_layout()
        path = out / f"primitive_{kind}_dose{dose}{suffix}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {path}")

    # ---- difference from baseline, composition only ----
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    base = comp["baseline"]
    treat = [(a, s) for a, s in arms if a != "baseline"]
    width = 0.8 / max(len(treat), 1)
    for i, (a, _) in enumerate(treat):
        ax.bar(x + (i - (len(treat) - 1) / 2) * width, comp[a] - base, width,
               yerr=comp_err[a], capsize=2, color=COLOR[a], label=a,
               error_kw=dict(ecolor="#333", lw=0.9))
    ax.axhline(0, color="#555", lw=1)
    ax.set_xticks(x, labels, rotation=30, ha="right", fontsize=8.5)
    ax.set_ylabel("arm's fraction − baseline's fraction\n(positive = more of this primitive)")
    ax.set_title(f"Change in primitive mix vs baseline (dose {dose})"
                 + (f" — all arms cut to v_{match}'s length" if match else ""))
    ax.legend(frameon=False, fontsize=9, ncol=max(len(treat), 1))
    fig.tight_layout()
    path = out / f"primitive_delta_dose{dose}{suffix}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")

    summary = pd.DataFrame({a: comp[a] for a, _ in arms}, index=labels).round(4)
    summary.to_csv(out / f"primitive_composition_dose{dose}{suffix}.csv")
    print(summary.to_string())
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", type=Path, required=True)
    parser.add_argument("--names", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dose", type=float, default=0.5)
    parser.add_argument("--alphabet", default="merged", choices=("merged", "meta"))
    parser.add_argument("--match", default=None,
                        help="truncate every arm to this arm's token count, per item")
    args = parser.parse_args()
    run(args.counts, args.names, args.out, dose=args.dose, alphabet=args.alphabet,
        match=args.match)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
