"""Stage 1 of the STOP-steering plan: build the head-space directions and run the kill gate.

The gate asks the question the plan put in front of everything else: **does the STOP-approach
contrast survive projection into the 35 AIE-selected heads at all?** Those heads were ranked by
average indirect effect on six synthetic in-context-learning tasks, not on this corpus and not on
termination, so their sufficiency for a control state is an assumption, not a result.

Five checks, in the order they can kill the experiment:

a. **decodability** -- held-out balanced accuracy of last-3-runs vs 20-40-runs-before-STOP from
   ``Z_top35`` against the same probe on the full layer-16 residual, and against permuted labels.
b. **ramp** -- the cross-fitted projection as a function of runs-to-STOP.
c. **natural validation** -- the screen's Table 2 in head space: inside (primitive, trace-third)
   cells, do run ends high on the direction end sooner?
d. **stability** -- item bootstrap of the direction against its shuffled-label null.
e. **in-distribution** -- do the capped traces, where we would inject, occupy the same projection
   range as the terminating traces they were never estimated from?

Everything is within-primitive centered, as the screen centers, so "which primitive is running"
cannot masquerade as the contrast.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

SEED = 20260919
N_BOOTSTRAP = 200
MIN_CELL = 40
HORIZON = 8


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def load(directory: Path):
    payload = np.load(directory / "capture.npz")
    frame = pd.read_parquet(directory / "frame.parquet")
    heads = payload["heads"]
    require(len(heads) == len(frame), "capture/frame row mismatch")
    return heads.reshape(len(heads), -1), payload["residual"].astype(np.float32), frame


def centered(features, primitives, rows=None):
    """Subtract the per-primitive centroid, as ``Hc = H - MU[z]`` in the screen."""
    fit = np.arange(len(features)) if rows is None else rows
    out = features.copy()
    for z in np.unique(primitives):
        mask = primitives == z
        source = mask.copy()
        source[~np.isin(np.arange(len(features)), fit)] = False
        if source.sum() == 0:
            continue
        out[mask] -= features[source].mean(0)
    return out


def contrast(features, positive, negative):
    require(positive.sum() and negative.sum(), "empty contrast side")
    return unit(features[positive].mean(0) - features[negative].mean(0))


def shuffled_labels(frame, positive, negative, rng):
    """Permute STOP-approach vs baseline *within item*, preserving each item's counts."""
    out = np.zeros(len(frame), bool)
    items = frame["item"].values
    pool = positive | negative
    for item in np.unique(items[pool]):
        idx = np.where(pool & (items == item))[0]
        k = int(positive[idx].sum())
        out[rng.choice(idx, size=k, replace=False)] = True
    return out, pool & ~out


def probe(features, labels, folds, seed=SEED):
    """Item-grouped cross-validated balanced accuracy, centering fitted on train rows only."""
    scores = []
    for fold in sorted(set(folds)):
        train, test = folds != fold, folds == fold
        if not (labels[train].any() and (~labels[train]).any() and test.sum()):
            continue
        scaler = StandardScaler().fit(features[train])
        model = LogisticRegression(max_iter=2000, C=0.01, class_weight="balanced",
                                   random_state=seed)
        model.fit(scaler.transform(features[train]), labels[train])
        scores.append(balanced_accuracy_score(
            labels[test], model.predict(scaler.transform(features[test]))))
    require(scores, "no usable folds")
    return float(np.mean(scores)), [float(s) for s in scores]


def check_decodability(heads, residual, frame, rng):
    positive = frame["last3"].values
    negative = frame["ref_stop"].values
    rows = np.where(positive | negative)[0]
    primitives = frame["z0"].values
    folds = frame["fold"].values[rows]
    labels = positive[rows]
    out = {"n_positive": int(labels.sum()), "n_negative": int((~labels).sum())}
    for name, block in (("heads_top35", heads), ("residual_l16", residual)):
        features = centered(block, primitives)[rows]
        mean, per_fold = probe(features, labels, folds)
        out[name] = {"balanced_accuracy": mean, "per_fold": per_fold, "dim": int(block.shape[1])}
    permuted = rng.permutation(labels)
    features = centered(heads, primitives)[rows]
    mean, _ = probe(features, permuted, folds)
    out["heads_top35_permuted_labels"] = {"balanced_accuracy": mean}
    return out


def crossfit_projection(features, frame, positive, negative):
    """Project each fold on a direction built only from the other folds."""
    folds = frame["fold"].values
    primitives = frame["z0"].values
    projection = np.full(len(frame), np.nan, np.float32)
    for fold in sorted(set(folds)):
        train = folds != fold
        block = centered(features, primitives, rows=np.where(train)[0])
        direction = contrast(block, positive & train, negative & train)
        projection[folds == fold] = block[folds == fold] @ direction
    require(np.isfinite(projection).all(), "unprojected rows")
    return projection


def check_ramp(projection, frame, bins=(0, 3, 6, 10, 20, 40, 80, 160, 320, 10**6)):
    to_end = frame["to_end"].values
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:], strict=False):
        mask = (to_end >= lo) & (to_end < hi)
        if mask.sum() < MIN_CELL:
            continue
        rows.append(dict(runs_to_end_lo=int(lo), runs_to_end_hi=int(hi), n=int(mask.sum()),
                         mean_projection=float(projection[mask].mean()),
                         sd=float(projection[mask].std())))
    return rows


def check_natural_validation(projection, frame, horizon=HORIZON):
    """Screen Table 2 in head space: top vs bottom quintile inside (primitive, third) cells."""
    third = np.minimum((frame["t"].values / np.maximum(frame["n"].values, 1) * 3).astype(int), 2)
    cells = pd.DataFrame(dict(z0=frame["z0"].values, third=third,
                              projection=projection, to_end=frame["to_end"].values))
    high, low = [], []
    for _, group in cells.groupby(["z0", "third"]):
        if len(group) < MIN_CELL:
            continue
        lo_cut, hi_cut = np.percentile(group["projection"], [20, 80])
        high.append(group[group["projection"] >= hi_cut])
        low.append(group[group["projection"] <= lo_cut])
    require(high and low, "no populated cells")
    high, low = pd.concat(high), pd.concat(low)
    return dict(
        n_cells_used=int(len(cells.groupby(["z0", "third"]).size()
                             [cells.groupby(["z0", "third"]).size() >= MIN_CELL])),
        n_high=int(len(high)), n_low=int(len(low)),
        p_end_within_8_high=float(np.mean(high["to_end"] <= horizon)),
        p_end_within_8_low=float(np.mean(low["to_end"] <= horizon)),
        mean_runs_to_end_high=float(np.mean(np.minimum(high["to_end"], 200))),
        mean_runs_to_end_low=float(np.mean(np.minimum(low["to_end"], 200))),
        projection_gap=float(high["projection"].mean() - low["projection"].mean()),
    )


def check_stability(heads, frame, rng, n_bootstrap=N_BOOTSTRAP):
    primitives = frame["z0"].values
    block = centered(heads, primitives)
    positive, negative = frame["last3"].values, frame["ref_stop"].values
    reference = contrast(block, positive, negative)
    items = frame["item"].values
    unique = np.unique(items)
    cosines = []
    for _ in range(n_bootstrap):
        picked = rng.choice(unique, size=len(unique), replace=True)
        mask = np.isin(items, picked)
        try:
            cosines.append(float(reference @ contrast(block, positive & mask, negative & mask)))
        except ValueError:
            continue
    shuffle_cos = []
    for _ in range(min(n_bootstrap, 50)):
        pos, neg = shuffled_labels(frame, positive, negative, rng)
        shuffle_cos.append(float(reference @ contrast(block, pos, neg)))
    return dict(n_bootstrap=len(cosines),
                bootstrap_cos_mean=float(np.mean(cosines)),
                bootstrap_cos_p05=float(np.percentile(cosines, 5)),
                shuffled_cos_mean=float(np.mean(shuffle_cos)),
                shuffled_cos_p95=float(np.percentile(shuffle_cos, 95)),
                reference_norm=float(np.linalg.norm(reference)))


def check_in_distribution(heads, frame, capped_heads, capped_frame):
    """Do the capped traces occupy the projection range the direction was estimated over?"""
    primitives = frame["z0"].values
    block = centered(heads, primitives)
    direction = contrast(block, frame["last3"].values, frame["ref_stop"].values)
    means = {z: heads[primitives == z].mean(0) for z in np.unique(primitives)}
    fallback = heads.mean(0)
    capped_primitives = capped_frame["z0"].values
    capped_centered = capped_heads - np.stack(
        [means.get(z, fallback) for z in capped_primitives])
    capped_projection = capped_centered @ direction
    terminating = block @ direction
    baseline = terminating[frame["ref_stop"].values]
    stop = terminating[frame["last3"].values]
    quantiles = [5, 25, 50, 75, 95]
    return dict(
        sd_baseline=float(baseline.std()),
        terminating_baseline_mean=float(baseline.mean()),
        terminating_last3_mean=float(stop.mean()),
        separation_in_sd=float((stop.mean() - baseline.mean()) / (baseline.std() + 1e-12)),
        capped_mean=float(capped_projection.mean()),
        capped_mean_in_sd=float((capped_projection.mean() - baseline.mean())
                                / (baseline.std() + 1e-12)),
        capped_quantiles=dict(zip(map(str, quantiles),
                                  [float(x) for x in np.percentile(capped_projection, quantiles)],
                                  strict=True)),
        baseline_quantiles=dict(zip(map(str, quantiles),
                                    [float(x) for x in np.percentile(baseline, quantiles)],
                                    strict=True)),
        share_capped_above_last3_median=float(
            np.mean(capped_projection > np.median(stop))),
    )


def run(capture: Path, capped: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    heads, residual, frame = load(capture)
    capped_heads, _, capped_frame = load(capped)
    print(f"terminating rows {len(frame):,} over {frame['item'].nunique()} items | "
          f"capped rows {len(capped_frame):,} over {capped_frame['item'].nunique()} items",
          flush=True)

    report = {"rows": int(len(frame)), "items": int(frame["item"].nunique()),
              "capped_rows": int(len(capped_frame)),
              "capped_items": int(capped_frame["item"].nunique())}

    print("(a) decodability ...", flush=True)
    report["a_decodability"] = check_decodability(heads, residual, frame, rng)

    print("(b) cross-fitted ramp ...", flush=True)
    projection = crossfit_projection(heads, frame, frame["last3"].values,
                                     frame["ref_stop"].values)
    report["b_ramp"] = check_ramp(projection, frame)

    print("(c) natural validation ...", flush=True)
    report["c_natural_validation"] = check_natural_validation(projection, frame)

    print("(d) stability ...", flush=True)
    report["d_stability"] = check_stability(heads, frame, rng)

    print("(e) in-distribution ...", flush=True)
    report["e_in_distribution"] = check_in_distribution(heads, frame, capped_heads, capped_frame)

    verdict = []
    a = report["a_decodability"]
    if a["heads_top35"]["balanced_accuracy"] < 0.60:
        verdict.append("FAIL(a): STOP-approach is not decodable from the 35 heads")
    if a["heads_top35"]["balanced_accuracy"] < a["heads_top35_permuted_labels"][
            "balanced_accuracy"] + 0.05:
        verdict.append("FAIL(a): head-space probe does not beat permuted labels")
    c = report["c_natural_validation"]
    if c["p_end_within_8_high"] <= c["p_end_within_8_low"]:
        verdict.append("FAIL(c): high-projection run ends do not end sooner")
    d = report["d_stability"]
    if d["bootstrap_cos_p05"] <= d["shuffled_cos_p95"]:
        verdict.append("FAIL(d): direction is not distinguishable from its shuffled null")
    report["verdict"] = verdict or ["PASS"]

    (out / "gate.json").write_text(json.dumps(report, indent=2))
    np.save(out / "crossfit_projection.npy", projection)
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--capped", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.capture, args.capped, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
