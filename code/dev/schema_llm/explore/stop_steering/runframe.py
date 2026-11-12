"""Run frame for the STOP-steering experiment: a port of ``COUNTERFACTUAL_SCREEN`` §0-1.

Rebuilds the collaborator's run segmentation and event masks from the exported bundle so the
head-space STOP vector is extracted over *exactly* the positions its 48-D screening counterpart
used. Nothing here touches a GPU; ``verify()`` checks the port against the screen's printed
counts and refuses to publish a frame that does not reproduce them.

A *run* (the screen calls it a segment) is a maximal stretch of one ``metacluster_L16`` label.
Positions are run **ends**, one row per run, and the final run is dropped because it has no
successor -- both inherited from ``seg_frame``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

BUNDLE = Path("results/schema_llm/export/20260824-rlreasoning_v1")
TASK = "identify"
MODELS = ("phi4mr", "qwen34b")
SEED = 0
N_FOLDS = 5
MIN_RUNS = 20

# '</think>' is stored inside the think segment and shares one metacluster; left in, these
# tokens become "the last run before STOP" and leak into the stop-hazard target.
MARK = frozenset({"</", "think", ">", "</think>", "<", "/think"})

# The screen's stored cell-1/cell-9 outputs. The port is only trusted if it hits these.
EXPECTED = {
    "regimes": {"phi terminating": 166, "phi non-terminating": 50, "qwen terminating": 216},
    "stripped": {0: 50, 3: 166, 1: 216},
    "common_items": 166,
    "segments": {"phi4mr": 92_006, "qwen34b": 63_317},
    "median_tokens_per_segment": {"phi4mr": 2.0, "qwen34b": 2.0},
    "share_1_token": {"phi4mr": 0.43, "qwen34b": 0.45},
    "share_le3_tokens": {"phi4mr": 0.70, "qwen34b": 0.75},
    "last3_runs_uncut": {"phi4mr": 498, "qwen34b": 498},
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def clean(token):
    return str(token).replace("Ġ", "").replace("Ċ", "").strip()


def find_spans_compat(tokens, value, maxlen=4):
    """Name-span end indices; re-exported so the lexicon port does not import ``events``."""
    from .events import find_spans
    return find_spans(tokens, value, maxlen)


def runs_of(z):
    """Maximal same-label stretches: labels, start indices, durations."""
    change = np.r_[True, z[1:] != z[:-1]]
    idx = np.where(change)[0]
    return z[change], idx, np.diff(np.r_[idx, len(z)])


@dataclass(frozen=True)
class Trace:
    model: str
    item: str
    z: np.ndarray          # metacluster label per response token
    pos: np.ndarray        # row index into the item's capture, per response token
    tokens: list[str]      # cleaned token text, aligned with z
    raw: list[str]         # original token text, retaining the word-boundary markers
    stripped: int          # trailing marker tokens removed
    terminating: bool


def load_bundle(root: Path = BUNDLE):
    columns = ["item_id", "model", "task_family", "segment", "position", "token",
               "metacluster_L16", "merged_primitive_L16"]
    tokens = pd.read_parquet(root / "tokens.parquet", columns=columns)
    tokens = tokens[tokens.task_family == TASK]
    responses = pd.read_parquet(root / "response_sequences.parquet")
    responses = responses[responses.task_family == TASK]
    names = json.loads((root / "merged_primitive_names.json").read_text())
    require(len(tokens) and len(responses), "empty bundle slice")
    return tokens, responses, names


def nonterminating(responses, model):
    """Items whose response never closed ``</think>`` -- the screen's termination test."""
    d = responses[responses.model == model]
    closed = d.response_text.str.contains("</think>", regex=False).values
    return set(d.item_id.values[~closed])


def build_traces(tokens, responses):
    traces: dict[tuple[str, str], Trace] = {}
    for model in MODELS:
        nonterm = nonterminating(responses, model)
        sub = tokens[tokens.model == model]
        for item, group in sub.groupby("item_id", sort=True):
            # A trace that never closed </think> has no think segment, so its reasoning is
            # carried by the answer segment instead.
            segments = ["think", "answer"] if item in nonterm else ["think"]
            d = group[group.segment.isin(segments)].sort_values("position")
            ok = ~pd.isna(d["metacluster_L16"].values)
            z = d["metacluster_L16"].values[ok].astype(int)
            pos = d["position"].values[ok].astype(int)
            raw = [str(x) for x in d["token"].values[ok]]
            text = [clean(x) for x in raw]
            n = len(text)
            while n > 0 and text[n - 1] in MARK:
                n -= 1
            if n < MIN_RUNS:
                continue
            traces[(model, item)] = Trace(
                model=model, item=item, z=z[:n], pos=pos[:n], tokens=text[:n],
                raw=raw[:n], stripped=len(text) - n, terminating=item not in nonterm,
            )
    require(traces, "no traces survived the run-length filter")
    return traces


def regime(trace: Trace):
    if trace.model == "phi4mr":
        return "phi terminating" if trace.terminating else "phi non-terminating"
    return "qwen terminating" if trace.terminating else "qwen non-terminating"


def common_items(traces):
    """Items terminating in *both* models -- the screen's shared fitting cohort."""
    per_model = {m: {k[1] for k, t in traces.items() if k[0] == m and t.terminating}
                 for m in MODELS}
    return sorted(set.intersection(*per_model.values()))


def folds(items, n_folds=N_FOLDS, seed=SEED):
    order = np.random.default_rng(seed).permutation(list(items))
    return {item: i % n_folds for i, item in enumerate(order)}


def seg_frame(trace: Trace, fold: int):
    """One row per run, dropping the last (no successor), with run-end token positions."""
    labels, starts, durations = runs_of(trace.z)
    ends = starts + durations - 1
    n = len(labels)
    frame = pd.DataFrame(dict(
        model=trace.model, item=trace.item, t=np.arange(n), z0=labels.astype(int),
        y=np.r_[labels[1:], -1].astype(int), dur=durations,
        row_end=trace.pos[ends], row_first=trace.pos[starts],
        row_shift=trace.pos[np.maximum(ends - 3, 0)],
        shift_prev=(ends - 3) < starts, fold=fold, n=n,
    ))
    return frame.iloc[:-1]


def add_stop_masks(frame):
    """``to_end`` in runs, and the screen's two STOP-approach windows."""
    tend = frame["n"].values - 2
    to_end = tend - frame["t"].values
    require((to_end >= 0).all(), "negative runs-to-end")
    out = frame.assign(to_end=to_end)
    out["last3"] = to_end <= 2
    out["ref_stop"] = (to_end >= 20) & (to_end <= 40)
    return out


def build(root: Path = BUNDLE):
    tokens, responses, names = load_bundle(root)
    traces = build_traces(tokens, responses)
    items = common_items(traces)
    assignment = folds(items)
    frames = {}
    for model in MODELS:
        parts = [seg_frame(traces[(model, item)], assignment[item]) for item in items]
        frames[model] = add_stop_masks(pd.concat(parts, ignore_index=True))
    return dict(traces=traces, items=items, folds=assignment, seg=frames,
                responses=responses, names=names)


def build_nonterminating(built, model="phi4mr"):
    """Run frame for the capped traces. They have no STOP, so ``to_end`` is not defined."""
    traces = built["traces"]
    items = sorted(item for (m, item), t in traces.items() if m == model and not t.terminating)
    require(items, "no non-terminating traces")
    parts = [seg_frame(traces[(model, item)], -1) for item in items]
    frame = pd.concat(parts, ignore_index=True)
    frame["to_end"] = -1
    frame["last3"] = False
    frame["ref_stop"] = False
    return items, frame


def summarize(built):
    traces, seg = built["traces"], built["seg"]
    counts: dict[str, int] = {}
    stripped: dict[int, int] = {}
    for trace in traces.values():
        counts[regime(trace)] = counts.get(regime(trace), 0) + 1
        stripped[trace.stripped] = stripped.get(trace.stripped, 0) + 1
    rows = []
    for model, frame in seg.items():
        rows.append(dict(
            model=model, items=int(frame["item"].nunique()), segments=int(len(frame)),
            median_tokens_per_segment=float(np.median(frame["dur"])),
            share_1_token=float(np.mean(frame["dur"] == 1)),
            share_le3_tokens=float(np.mean(frame["dur"] <= 3)),
            share_shift_in_earlier_segment=float(np.mean(frame["shift_prev"])),
            last3_runs=int(frame["last3"].sum()), ref_stop_runs=int(frame["ref_stop"].sum()),
        ))
    return dict(regimes=counts, stripped=stripped, common_items=len(built["items"]),
                table0=pd.DataFrame(rows))


def verify(summary):
    """Refuse the port unless it reproduces the screen's printed counts."""
    problems = []
    if summary["regimes"] != EXPECTED["regimes"]:
        problems.append(f"regimes {summary['regimes']} != {EXPECTED['regimes']}")
    if summary["stripped"] != EXPECTED["stripped"]:
        problems.append(f"stripped {summary['stripped']} != {EXPECTED['stripped']}")
    if summary["common_items"] != EXPECTED["common_items"]:
        problems.append(f"common items {summary['common_items']}")
    for row in summary["table0"].to_dict("records"):
        model = row["model"]
        if row["segments"] != EXPECTED["segments"][model]:
            problems.append(f"{model} segments {row['segments']} != {EXPECTED['segments'][model]}")
        if row["median_tokens_per_segment"] != EXPECTED["median_tokens_per_segment"][model]:
            problems.append(f"{model} median tokens {row['median_tokens_per_segment']}")
        for key in ("share_1_token", "share_le3_tokens"):
            if abs(row[key] - EXPECTED[key][model]) > 0.005:
                problems.append(f"{model} {key} {row[key]:.3f} != {EXPECTED[key][model]}")
        if row["last3_runs"] != EXPECTED["last3_runs_uncut"][model]:
            problems.append(f"{model} last3 {row['last3_runs']} "
                            f"!= {EXPECTED['last3_runs_uncut'][model]} (pre norm-cut)")
    return problems


def main():
    built = build()
    summary = summarize(built)
    print("regimes:", summary["regimes"])
    print("marker tokens stripped per trace:", summary["stripped"])
    print("common terminating items:", summary["common_items"])
    print(summary["table0"].to_string(index=False))
    problems = verify(summary)
    print("\nPARITY:", "OK -- reproduces the screen" if not problems else "FAILED")
    for problem in problems:
        print("  -", problem)
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
