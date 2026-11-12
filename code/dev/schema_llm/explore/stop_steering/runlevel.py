"""Run-level outcomes for intervened continuations -- request section 3.

The request asks for "probability of STOP within the next 1, 2, 4 and 8 **runs**" and "number of
**runs** from injection to STOP". Everything measured so far is in tokens, because a run is a
stretch of one primitive label and the generated text carries no labels.

This labels it. The frozen ``phi4mr_spot_error`` L16 assigner (standardise, nearest of 50
centroids, cluster -> metacluster) is applied to layer-16 residuals captured during generation, and
consecutive identical labels are segmented into runs exactly as ``runframe`` does for the corpus.

``calibrate()` comes first and is not optional: the assigner is the *head-FV reader*, trained on
``spot_error``, whereas the corpus column ``metacluster_L16`` came from the cross-model merge. They
are different label spaces, so the agreement between them on the corpus's own run ends bounds how
literally these run counts can be compared with the screen's.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

from .gate import load

ASSIGNER = Path("results/schema_llm/analysis/primitive_transplant/20260828-single_primitive"
                "/assigner/assigner.npz")
SOURCE = "phi4mr_spot_error"
LAYER = 16


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_assigner(path: Path = ASSIGNER, source=SOURCE, layer=LAYER):
    payload = np.load(path)
    prefix = f"{source}__L{layer}__"
    parts = {key: payload[prefix + key]
             for key in ("mean", "scale", "centers", "cluster_to_meta")}
    require(parts["centers"].ndim == 2, "malformed centroid table")
    require(len(parts["cluster_to_meta"]) == len(parts["centers"]), "cluster/meta mismatch")
    return parts


def assign(residual, parts, batch=8192):
    """Standardise, nearest centroid, then cluster -> metacluster."""
    mean, scale = parts["mean"], np.where(parts["scale"] == 0, 1.0, parts["scale"])
    centers = (parts["centers"] - mean) / scale
    norms = (centers ** 2).sum(1)
    out = np.empty(len(residual), np.int32)
    for start in range(0, len(residual), batch):
        block = (residual[start:start + batch].astype(np.float32) - mean) / scale
        distance = norms[None, :] - 2.0 * block @ centers.T
        out[start:start + batch] = parts["cluster_to_meta"][distance.argmin(1)]
    return out


def best_match_accuracy(predicted, actual):
    """Agreement after optimally matching the two label vocabularies."""
    pred_ids = np.unique(predicted)
    act_ids = np.unique(actual)
    table = np.zeros((len(pred_ids), len(act_ids)), np.int64)
    for i, p in enumerate(pred_ids):
        mask = predicted == p
        for j, a in enumerate(act_ids):
            table[i, j] = int((actual[mask] == a).sum())
    rows, cols = linear_sum_assignment(-table)
    return float(table[rows, cols].sum() / len(actual))


def calibrate(capture: Path, out: Path):
    """How closely does the head-FV reader reproduce the corpus's own run labels?"""
    _, residual, frame = load(capture)
    parts = load_assigner()
    predicted = assign(residual, parts)
    actual = frame["z0"].values.astype(int)
    report = dict(
        n=int(len(actual)),
        n_labels_assigner=int(len(np.unique(predicted))),
        n_labels_corpus=int(len(np.unique(actual))),
        best_match_accuracy=best_match_accuracy(predicted, actual),
        adjusted_rand=float(adjusted_rand_score(actual, predicted)),
    )
    # run segmentation is the quantity that matters, not the label names
    pred_runs = int((predicted[1:] != predicted[:-1]).sum()) + 1
    act_runs = int((actual[1:] != actual[:-1]).sum()) + 1
    report["run_boundaries_assigner"] = pred_runs
    report["run_boundaries_corpus"] = act_runs
    report["boundary_ratio"] = pred_runs / act_runs
    out.mkdir(parents=True, exist_ok=True)
    (out / "assigner_calibration.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


def run_outcomes(labels, terminated):
    """Runs in the continuation, and whether STOP fell within the first k of them."""
    if len(labels) == 0:
        return dict(runs_generated=0, runs_to_stop=-1,
                    **{f"stop_within_{k}_runs": False for k in (1, 2, 4, 8)})
    boundaries = int((labels[1:] != labels[:-1]).sum()) + 1
    runs_to_stop = boundaries if terminated else -1
    out = dict(runs_generated=boundaries, runs_to_stop=runs_to_stop)
    for k in (1, 2, 4, 8):
        out[f"stop_within_{k}_runs"] = bool(terminated and boundaries <= k)
    return out


def tokens_per_run(capture: Path):
    """Measured tokens per run, per item, from the corpus frame."""
    _, _, frame = load(capture)
    per_item = frame.groupby("item")["dur"].mean()
    return per_item, float(frame["dur"].mean())


def horizons_from_tokens(pilot_dirs, capture: Path, out: Path):
    """Section-3 run horizons, converted from tokens at each item's own measured run length.

    Labelling the generated text in the corpus's own label space is not possible here -- see
    ``calibrate`` -- so the run axis is reconstructed rather than measured. The conversion is
    per item, using that trace's mean tokens per run, and the resulting horizons are reported
    as approximate.
    """
    per_item, overall = tokens_per_run(capture)
    frames = []
    for directory in pilot_dirs:
        path = Path(directory) / "pilot.parquet"
        if not path.exists():
            continue
        table = pd.read_parquet(path)
        scale = table["item"].map(per_item).fillna(overall)
        table = table.assign(tokens_per_run=scale,
                             runs_generated=table["n_generated"] / scale)
        for k in (1, 2, 4, 8):
            table[f"stop_within_{k}_runs"] = (table["terminated"]
                                              & (table["runs_generated"] <= k))
        table["runs_to_stop"] = np.where(table["terminated"], table["runs_generated"], np.nan)
        table["run"] = Path(directory).name
        frames.append(table)
    require(frames, "no runs to convert")
    table = pd.concat(frames, ignore_index=True)
    columns = ["terminated", "runs_generated", "runs_to_stop",
               *[f"stop_within_{k}_runs" for k in (1, 2, 4, 8)]]
    summary = (table.groupby(["run", "schedule", "arm", "dose"])[columns]
               .mean().reset_index())
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "run_horizons.csv", index=False)
    table[["item", "run", "arm", "dose", "terminated", "n_generated", "tokens_per_run",
           "runs_generated", "runs_to_stop"]].to_parquet(out / "run_horizons.parquet")
    print(f"tokens per run: mean {overall:.2f} over the corpus\n")
    print(summary.to_string(index=False))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path,
                        default=Path("results/schema_llm/analysis/stop_steering"
                                     "/20260919-phi-headcapture"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--dirs", nargs="*", default=[])
    args = parser.parse_args()
    if args.calibrate:
        calibrate(args.capture, args.out)
    if args.dirs:
        horizons_from_tokens(args.dirs, args.capture, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
