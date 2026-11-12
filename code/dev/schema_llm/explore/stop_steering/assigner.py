"""Freeze an identify-native primitive assigner, and prove it reproduces the corpus labels.

The bundle's ``metacluster_L16`` column is scoped to a ``(model, task, layer)`` cell, and the only
assigners saved on disk are ``phi4mr_spot_error`` / ``qwen34b_spot_error``. Labelling generated
``identify`` text with the spot_error assigner would put it in a *different task's* primitive
vocabulary -- which is why that assigner scores near-zero agreement against the identify column.
That is a cell mismatch, not a broken instrument.

This module refits the identify cell from the surviving source residuals under the config recorded
in the node manifest (k=50, force_meta=8, pca=None, seed=0), then **validates** the refit against
the stored per-token labels in ``clusters.npz``. Only a refit that reproduces those labels is
written out, so the instrument is proven before it is pointed at generated text.

The frozen artifact holds the scaler statistics, the 50 k-means centroids and the
cluster -> metacluster map, which is everything needed to label new residuals.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

from ...primitives._cluster import (
    KMeansTemporalMetaClustering,
    MetaConfig,
    fit_token_transform,
    load_token_sequences,
    scope_refs,
)

NODE = Path("results/schema_llm/analysis/depth/20260824-rlreasoning_nodes/phi4mr_identify")
# EXACTLY the order `export_bundle.POOL["identify"]` uses. `scope_refs` walks run dirs in the
# order given, so any other order would align the refit labels against the wrong tokens and make
# the validation meaningless rather than merely weaker.
DATA = Path("results/schema_llm/data")
RUNS = [DATA / "main_36/restaurant_main", DATA / "main_36/airport_main",
        DATA / "main_72/restaurant_main", DATA / "main_72/airport_main"]
SLUG = "phi-4-mini-reasoning"
LAYER = 16
K = 50
FORCE_META = 8
SEED = 0


def require(condition, message):
    if not condition:
        raise ValueError(message)


def stored_labels(node: Path = NODE, layer: int = LAYER):
    """The per-token metacluster ids the bundle was built from."""
    stack = np.load(node / "clusters.npz")["stack"]
    manifest = json.loads((node / "clusters.json").read_text())
    layers = manifest["key"]["layers"]
    require(layer in layers, f"layer {layer} not in {layers}")
    return stack[layers.index(layer)].astype(int), manifest["key"]


def refit(runs=None, slug=SLUG, layer=LAYER, k=K, force_meta=FORCE_META, seed=SEED):
    """Refit under the node manifest's config; also return the scaler so it can be frozen.

    ``load_token_sequences`` fits the scaler internally and does not hand it back, so it is fit
    again here with the identical arguments -- same refs, scope, seed and cap -- which makes it
    the same transform rather than a similar one.
    """
    runs = runs or [r for r in RUNS if r.exists()]
    require(runs, "no source residual roots found")
    refs = scope_refs(runs, slug, layer, ["identify"])
    require(refs, "no capture refs found")
    scaler, reducer = fit_token_transform(refs, "response", seed, 30000, None)
    require(reducer is None or not hasattr(reducer, "components_"),
            "manifest says pca=None; a reducer would change the feature space")
    seqs = load_token_sequences(runs, slug, layer, "response", seed,
                                tasks=["identify"], pca=None)
    require(seqs, "no response tokens loaded")
    model = KMeansTemporalMetaClustering(
        MetaConfig(n_clusters=k, random_state=seed, force_metaclusters=force_meta)
    ).fit([s["std"] for s in seqs], [s["positions"] for s in seqs])
    labels = np.concatenate([model.meta_labels(s["std"]) for s in seqs])
    return model, seqs, labels, scaler


def validate(refit_labels, stored):
    """Agreement with the stored labels; ARI is invariant to any id permutation."""
    n = min(len(refit_labels), len(stored))
    ari = float(adjusted_rand_score(stored[:n], refit_labels[:n]))
    exact = float(np.mean(stored[:n] == refit_labels[:n]))
    return dict(n_compared=int(n), n_stored=int(len(stored)), n_refit=int(len(refit_labels)),
                adjusted_rand=ari, exact_match=exact)


def run(out: Path, min_ari: float = 0.90):
    out.mkdir(parents=True, exist_ok=True)
    stored, key = stored_labels()
    print(f"stored labels: {len(stored):,} tokens | node config {key}")
    model, seqs, labels, scaler = refit()
    print(f"refit: {len(labels):,} tokens over {len(seqs)} traces")

    report = validate(labels, stored)
    print(json.dumps(report, indent=2))
    if report["adjusted_rand"] < min_ari:
        (out / "assigner_validation.json").write_text(json.dumps(
            {**report, "status": "REJECTED", "min_ari": min_ari}, indent=2))
        raise SystemExit(
            f"refit does not reproduce the stored labels (ARI {report['adjusted_rand']:.3f} "
            f"< {min_ari}); refusing to freeze an unvalidated assigner")

    payload = dict(centers=model.kmeans.cluster_centers_.astype(np.float32),
                   cluster_to_meta=model.cluster_to_meta_.astype(np.int32),
                   mean=scaler.mean_.astype(np.float32),
                   scale=scaler.scale_.astype(np.float32))
    np.savez(out / "assigner_identify_L16.npz", **payload)
    (out / "assigner_validation.json").write_text(json.dumps(
        {**report, "status": "ACCEPTED", "config": key}, indent=2))
    print(f"wrote {out}/assigner_identify_L16.npz")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-ari", type=float, default=0.90)
    args = parser.parse_args()
    run(args.out, min_ari=args.min_ari)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
