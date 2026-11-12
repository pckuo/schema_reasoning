#!/usr/bin/env python3
"""Depth-resolved clustering of Phi-4-reasoning RESPONSE latents across all 40 layers.

The residual stream is *additive*, so raw latent norm grows ~240x from layer 0 to 39 (verified on
our captures) and a few massive-activation dims dominate the top layers -- any cross-layer pooling
in raw space is swamped by the last layers. Every analysis here therefore standardizes **per layer**
(z-score each of the 5120 dims within a layer, exactly as the single-layer ``metacluster`` pipeline
does), which removes both the depth scale and the rogue dims.

Modes:

- ``diagnostic``: per-layer geometry of the response tokens -- mean L2 norm, participation ratio
  (effective dimensionality), massive-activation energy share, and adjacent-layer linear CKA (all on
  RAW residuals, so the growth is visible) -- plus a cross-layer **depth manifold** (per-layer-
  standardized tokens pooled, PCA-2D, colored by depth). Writes diagnostic.png + depth_manifold.png
  + diagnostic.json.

- ``sweep``: run the existing response-scope temporal meta-clustering
  (:func:`primitives.metacluster.analyze_scope`) at each layer independently (self-standardized) and
  profile structure-quality vs depth -- eigengap phase count, temporal coherence ratio, segment
  monotonicity, schema-generality (AMI meta<->schema), and k-means silhouette -- to locate the
  depth(s) where the reasoning response's latent trajectory is most structured. Writes sweep.png +
  sweep.json + findings.md.
- ``merge``: cluster + LLM-annotate each layer's meta-clusters INDEPENDENTLY, then merge them across
  layers into cross-layer THEMES only where functionally the same. Default **``semantic-avglink``**:
  average-linkage on ``1 - semantic cosine`` of each node's ``label + description``, cut at the
  silhouette-optimal height; the semantic backend is **instruction embeddings** (Qwen3-Embedding-4B)
  by default (``--merge-embed`` on; ``--no-merge-embed`` falls back to the CPU-only TF-IDF cosine).
  The go-forward alternatives are ``semantic-cpm`` (the same embeddings through a k-NN Leiden-CPM
  consensus -- coarser, better worst-case balance) and ``labels`` (one live LLM grouping call); all
  three need ``--annotate``. The token-graph methods ``avglink`` / ``leiden-cpm`` / ``tokens`` are
  **RETIRED** (superseded by the semantic merges -- token-Jaccard cannot separate co-occurring
  reasoning functions) but kept runnable for comparison. Emits a per-theme arc-span (member mean_pos
  spread) as a chaining health-check, and profiles how each layer's token mass splits over themes
  (inference- vs formatting-heavy).
  Per-layer clustering is cached (clusters.npz/json) so re-merging with another method is instant.
  Writes layer_themes.png + theme_membership.png + ami_matrix.png + merge.json + merge_findings.md.
Usage::

    # default merge = semantic-avglink on Qwen3-Embedding-4B (needs the `gpu` extra + --annotate;
    # --no-merge-embed = CPU TF-IDF fallback; --merge semantic-cpm for coarser themes).
    python -m dev.schema_llm.primitives.depth_metacluster --mode merge --annotate \
        --run <dirs> --slug Phi-4-reasoning --layers 0 4 8 12 16 20 24 28 32 36 39 \
        --k 50 --meta 8 --out results/schema_llm/analysis/depth/<stamp>_Phi-4-reasoning
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_mutual_info_score, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from ._cluster import (
    _SCOPE_CTX,
    _annotate_schema,
    _clean,
    _distinctive_tokens,
    _phase_flow,
    _scope_token_indices,
    _short_med_long,
    analyze_scope,
    load_token_sequences,
)

DEFAULT_LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 39]
DEFAULT_ANNOTATE_MODEL = "claude-opus-4.8"


def _parse_pca(s: str | None) -> float | int | None:
    """CLI '--pca': 'none' -> None (full z-scored); '0.9' -> variance fraction; '128' -> #comps."""
    if not s or s.lower() == "none":
        return None
    val = float(s)
    return val if 0.0 < val < 1.0 else int(val)


def _response_dirs(run_dirs: list[Path], slug: str, tasks: list[str]) -> list[Path]:
    dirs: list[Path] = []
    for run in run_dirs:
        adir = run / "activations" / slug
        if not adir.exists():
            print(f"[warn] {adir} missing; skipping")
            continue
        for tsk in tasks:
            dirs += sorted(adir.glob(f"*__{tsk}"))
    return dirs


def _lin_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear CKA between two token x feature matrices sharing the SAME rows (tokens)."""
    x = x - x.mean(0)
    y = y - y.mean(0)
    hsic = np.linalg.norm(x.T @ y) ** 2
    return float(hsic / (np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y) + 1e-12))


def _response_subsample(dirs: list[Path], seed: int, sub_per: int) -> list[tuple]:
    """Fixed per-scenario (dir, lo, hi, token-indices) so the same response tokens are read at every
    layer (needed for a meaningful cross-layer CKA)."""
    rng = np.random.default_rng(seed)
    scn = []
    for d in dirs:
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        rr = meta.get("response_token_range")
        if not rr:
            continue
        lo, hi = int(rr[0]), int(rr[1])
        if meta.get("n_tokens"):
            hi = min(hi, int(meta["n_tokens"]))
        n = hi - lo
        if n < 3:
            continue
        take = np.sort(rng.permutation(n)[: min(sub_per, n)]) + lo
        scn.append((d, lo, hi, take))
    return scn


def diagnostic(run_dirs: list[Path], slug: str, layers: list[int], out_dir: Path,
               tasks: list[str], seed: int = 0, max_scenarios: int = 40, sub_per: int = 200,
               manifold_per: int = 300) -> dict:
    """Per-layer raw geometry (norm / PR / rogue share / adjacent CKA) + a per-layer-standardized
    cross-layer depth-manifold scatter."""
    dirs = _response_dirs(run_dirs, slug, tasks)
    if not dirs:
        raise SystemExit(f"no response captures for slug={slug} under {run_dirs}")
    rng = np.random.default_rng(seed)
    if len(dirs) > max_scenarios:
        dirs = [dirs[i] for i in sorted(rng.permutation(len(dirs))[:max_scenarios])]
    scn = _response_subsample(dirs, seed, sub_per)
    if not scn:
        raise SystemExit("no response tokens to diagnose")

    rows: list[dict] = []
    manifold_x: list[np.ndarray] = []
    manifold_layer: list[np.ndarray] = []
    prev: np.ndarray | None = None
    for lay in layers:
        norms, subs = [], []
        for d, lo, hi, take in scn:
            arr = np.load(d / f"layer_{lay}.npy", mmap_mode="r")
            norms.append(np.linalg.norm(np.asarray(arr[lo:hi], np.float32), axis=1))
            subs.append(np.asarray(arr[take], np.float32))
        norms = np.concatenate(norms)
        x = np.concatenate(subs)
        xc = x - x.mean(0)
        ev = np.clip(np.linalg.eigvalsh(xc.T @ xc / len(xc)), 0, None)
        pr = float((ev.sum() ** 2) / ((ev**2).sum() + 1e-12))
        energy = (x**2).mean(0)
        order = np.argsort(energy)[::-1]
        tot = float(energy.sum()) + 1e-12
        rows.append({
            "layer": int(lay),
            "mean_norm": round(float(norms.mean()), 3),
            "participation_ratio": round(pr, 2),
            "rogue_top1_frac": round(float(energy[order[0]]) / tot, 4),
            "rogue_top5_frac": round(float(energy[order[:5]].sum()) / tot, 4),
            "cka_prev": round(_lin_cka(prev, x), 4) if prev is not None else None,
        })
        prev = x
        # per-layer-standardized sample for the pooled depth manifold (norm removed)
        std = StandardScaler().fit_transform(x)
        sel = rng.permutation(len(std))[:manifold_per]
        manifold_x.append(std[sel])
        manifold_layer.append(np.full(len(sel), lay))
        print(f"[diag] L{lay:2d} |h|={rows[-1]['mean_norm']:.1f} PR={pr:.0f} "
              f"top5={rows[-1]['rogue_top5_frac']:.2f} cka={rows[-1]['cka_prev']}")

    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {"slug": slug, "tasks": tasks, "layers": list(layers), "n_scenarios": len(scn),
             "sub_per": sub_per, "per_layer": rows}
    (out_dir / "diagnostic.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    _diagnostic_figure(rows, out_dir / "diagnostic.png", slug)
    _manifold_figure(np.concatenate(manifold_x), np.concatenate(manifold_layer), seed,
                     out_dir / "depth_manifold.png", slug)
    print(f"-> {out_dir}")
    return stats


def sweep(run_dirs: list[Path], slug: str, layers: list[int], k: int, out_dir: Path,
          tasks: list[str], seed: int = 0, force_meta: int = 8, sil_sample: int = 3000,
          pca: float | int | None = None) -> dict:
    """Response-scope temporal meta-clustering at each layer (self-standardized); profile
    structure-quality metrics vs depth. Schemas are pooled (shared primitives) so each layer yields
    one row; ``ami_meta_vs_schema`` reports how schema-general the phase structure is. ``pca`` picks
    the clustering feature space (None = full z-scored; float in (0,1) = variance fraction)."""
    rows: list[dict] = []
    for lay in layers:
        seqs = load_token_sequences(run_dirs, slug, lay, "response", seed, tasks=tasks, pca=pca)
        if not seqs:
            print(f"[skip] layer {lay}: no response tokens")
            continue
        res = analyze_scope(seqs, k, seed, "function", "response", force_meta)
        emb = res["_plot"]["embed"]
        xe, me = np.asarray(emb["X"]), np.asarray(emb["meta"])
        sil = None
        if len(set(me.tolist())) > 1 and len(xe) > 3:
            m = min(sil_sample, len(xe))
            sil = round(float(silhouette_score(xe[:m], me[:m])), 4)
        flow = res["flow"]
        rows.append({
            "layer": int(lay),
            "n_tokens": res["n_tokens"],
            "eigengap_n_meta": int(res["n_metaclusters_auto"]),
            "coherence_ratio": res["coherence"]["ratio"],
            "monotonicity": flow["monotonicity"],
            "monotonicity_shuffled": flow["monotonicity_shuffled"],
            "forward_frac": flow["forward_frac"],
            "ami_meta_vs_schema": res["ami_meta_vs_schema"],
            "silhouette_meta": sil,
        })
        print(f"[sweep] L{lay:2d} n_meta_auto={rows[-1]['eigengap_n_meta']} "
              f"coh={rows[-1]['coherence_ratio']} mono={rows[-1]['monotonicity']} sil={sil}")
    if not rows:
        raise SystemExit(f"no response tokens for slug={slug} under {run_dirs}")

    def _valid(key: str) -> list[dict]:
        return [r for r in rows if isinstance(r.get(key), int | float)]

    coh_rows = _valid("coherence_ratio")
    best = max(coh_rows, key=lambda r: r["coherence_ratio"]) if coh_rows else rows[0]
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {"slug": slug, "tasks": tasks, "k": k, "force_meta": force_meta, "pca": pca,
             "layers": [r["layer"] for r in rows], "best_layer_by_coherence": best["layer"],
             "per_layer": rows}
    (out_dir / "sweep.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    _sweep_figure(rows, out_dir / "sweep.png", slug)
    _write_findings(stats, out_dir / "findings.md")
    print(f"-> {out_dir} (best depth by coherence: L{best['layer']})")
    return stats


# --------------------------------------------------------------------------- figures


def _diagnostic_figure(rows: list[dict], path: Path, slug: str) -> None:
    lay = [r["layer"] for r in rows]
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    ax[0, 0].plot(lay, [r["mean_norm"] for r in rows], "o-")
    ax[0, 0].set_yscale("log")
    ax[0, 0].set_title("mean residual norm  ||h||  (log)")
    ax[0, 1].plot(lay, [r["participation_ratio"] for r in rows], "o-", color="tab:green")
    ax[0, 1].set_title("participation ratio (effective dim / 5120)")
    ax[1, 0].plot(lay, [r["rogue_top1_frac"] for r in rows], "o-", label="top-1 dim")
    ax[1, 0].plot(lay, [r["rogue_top5_frac"] for r in rows], "s-", label="top-5 dims")
    ax[1, 0].set_title("massive-activation energy share")
    ax[1, 0].legend()
    ck = [(r["layer"], r["cka_prev"]) for r in rows if r["cka_prev"] is not None]
    ax[1, 1].plot([c[0] for c in ck], [c[1] for c in ck], "o-", color="tab:red")
    ax[1, 1].set_title("linear CKA vs previous sampled layer")
    for a in ax.ravel():
        a.set_xlabel("layer")
        a.grid(alpha=0.3)
    fig.suptitle(f"{slug}: response-latent geometry vs depth (raw)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _manifold_figure(x: np.ndarray, layer: np.ndarray, seed: int, path: Path, slug: str) -> None:
    coords = PCA(n_components=2, random_state=seed).fit_transform(x)
    fig, axis = plt.subplots(figsize=(7, 6))
    sc = axis.scatter(coords[:, 0], coords[:, 1], c=layer, cmap="viridis", s=6, alpha=0.6)
    fig.colorbar(sc, ax=axis, label="layer")
    axis.set_title(f"{slug}: cross-layer depth manifold\n(per-layer z-scored tokens, PCA-2D)")
    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _sweep_figure(rows: list[dict], path: Path, slug: str) -> None:
    lay = [r["layer"] for r in rows]

    def _y(key: str) -> list:
        return [r.get(key) for r in rows]

    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    ax[0, 0].plot(lay, _y("eigengap_n_meta"), "o-")
    ax[0, 0].set_title("eigengap phase count")
    ax[0, 1].plot(lay, _y("coherence_ratio"), "o-", color="tab:green")
    ax[0, 1].axhline(1.0, ls="--", c="gray", lw=1)
    ax[0, 1].set_title("temporal coherence ratio (vs shuffled)")
    ax[0, 2].plot(lay, _y("monotonicity"), "o-", label="meta")
    ax[0, 2].plot(lay, _y("monotonicity_shuffled"), "s--", color="gray", label="shuffled")
    ax[0, 2].set_title("segment monotonicity")
    ax[0, 2].legend()
    ax[1, 0].plot(lay, _y("forward_frac"), "o-", color="tab:purple")
    ax[1, 0].axhline(0.5, ls="--", c="gray", lw=1)
    ax[1, 0].set_title("forward transition fraction")
    ax[1, 1].plot(lay, _y("silhouette_meta"), "o-", color="tab:orange")
    ax[1, 1].set_title("k-means silhouette (meta)")
    ax[1, 2].plot(lay, _y("ami_meta_vs_schema"), "o-", color="tab:brown")
    ax[1, 2].set_title("AMI meta vs schema (schema-specificity)")
    for a in ax.ravel():
        a.set_xlabel("layer")
        a.grid(alpha=0.3)
    fig.suptitle(f"{slug}: response meta-cluster structure vs depth")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _write_findings(stats: dict, path: Path) -> None:
    rows = stats["per_layer"]
    best = stats["best_layer_by_coherence"]
    lines = [f"# Response meta-cluster depth sweep — {stats['slug']}", "",
             f"Tasks: {', '.join(stats['tasks'])} · k={stats['k']} · meta={stats['force_meta']}",
             f"Layers: {stats['layers']}", "",
             f"**Best depth by temporal coherence: L{best}.**", "",
             "| layer | n_meta | coherence | monotonicity | forward | silhouette | AMI-schema |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['layer']} | {r['eigengap_n_meta']} | {r['coherence_ratio']} | "
            f"{r['monotonicity']} | {r['forward_frac']} | {r['silhouette_meta']} | "
            f"{r['ami_meta_vs_schema']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- cross-layer align


def _contingency(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    c = np.zeros((n, n))
    np.add.at(c, (a, b), 1.0)
    return c


def _layer_clusters(
    run_dirs: list[Path],
    slug: str,
    layer: int,
    k: int,
    seed: int,
    force_meta: int,
    tasks: list[str],
    pca: float | int | None,
    annotate: bool,
    model: str,
    distinctive: bool = False,
    backend: str = "copilot",
) -> dict | None:
    """Cluster ONE layer's response tokens and (optionally) LLM-annotate its meta-clusters
    INDEPENDENTLY. Returns {layer, labels (N,), profiles [{meta, top, mean_pos, n_tokens}],
    ann {meta: {label, description}}}; ``labels`` are per-token meta ids in the fixed shared token
    order (identical across layers) so nodes can be matched by token overlap later."""
    seqs = load_token_sequences(run_dirs, slug, layer, "response", seed, tasks=tasks, pca=pca)
    if not seqs:
        return None
    res = analyze_scope(seqs, k, seed, "function", "response", force_meta)
    labels = np.concatenate([p["meta"] for p in res["_plot"]["passages"]])
    raw = (
        _annotate_schema(
            res, model, seed, _SCOPE_CTX["response"], distinctive=distinctive, backend=backend
        ).get("meta", {})
        if annotate
        else {}
    )
    ann = {m: (raw.get(m) or raw.get(str(m)) or {}) for m in range(force_meta)}
    return {"layer": layer, "labels": labels, "profiles": res["meta_profiles"], "ann": ann}


def _load_or_cluster(
    run_dirs: list[Path],
    slug: str,
    layers: list[int],
    k: int,
    seed: int,
    force_meta: int,
    tasks: list[str],
    pca: float | int | None,
    annotate: bool,
    model: str,
    out_dir: Path,
    distinctive: bool = False,
    backend: str = "copilot",
) -> list[dict]:
    """Cluster + annotate each layer INDEPENDENTLY, caching the per-token labels (clusters.npz) +
    profiles/annotations (clusters.json) so a re-merge at another threshold/method skips the
    expensive per-layer clustering. Returns the per-layer dicts (labels/profiles/ann)."""
    key = {"slug": slug, "layers": list(layers), "k": k, "force_meta": force_meta, "pca": pca,
           "tasks": list(tasks), "annotate": annotate, "seed": seed}
    if distinctive:  # distinctive tokens change the annotation -> separate cache
        key["distinctive"] = True
    if annotate and (backend, model) != ("copilot", DEFAULT_ANNOTATE_MODEL):
        # the annotator IS the cross-source common currency, so a non-default one must not silently
        # reuse (or be reused by) the default's cache; omitted for the default to keep older caches
        key["annotate_model"] = f"{backend}:{model}"
    meta_p, npz_p = out_dir / "clusters.json", out_dir / "clusters.npz"
    if meta_p.exists() and npz_p.exists():
        cached = json.loads(meta_p.read_text(encoding="utf-8"))
        if cached.get("key") == key:
            stack = np.load(npz_p)["stack"]
            print(f"[merge] reusing cached clustering ({len(cached['per_layer'])} layers)")
            return [{"layer": pl["layer"], "labels": stack[i], "profiles": pl["profiles"],
                     "ann": {int(mm): vv for mm, vv in pl["ann"].items()}}
                    for i, pl in enumerate(cached["per_layer"])]
    per_layer = []
    for lay in layers:
        pl = _layer_clusters(
            run_dirs,
            slug,
            lay,
            k,
            seed,
            force_meta,
            tasks,
            pca,
            annotate,
            model,
            distinctive,
            backend,
        )
        if pl is None:
            print(f"[skip] layer {lay}: no response tokens")
            continue
        per_layer.append(pl)
        print(f"[merge] L{lay:2d} clustered ({len(pl['labels'])} tokens)")
    if len(per_layer) < 2:
        raise SystemExit("need >= 2 layers with response tokens to merge")
    if len({len(pl["labels"]) for pl in per_layer}) != 1:
        raise SystemExit("token counts differ across layers; merging needs the shared token set")
    out_dir.mkdir(parents=True, exist_ok=True)
    stack = np.stack([pl["labels"] for pl in per_layer]).astype(np.int16)
    np.savez_compressed(npz_p, stack=stack)
    meta_p.write_text(json.dumps({"key": key, "per_layer": [
        {"layer": pl["layer"], "profiles": pl["profiles"],
         "ann": {str(m): pl["ann"].get(m, {}) for m in range(force_meta)}} for pl in per_layer]},
        indent=2), encoding="utf-8")
    return per_layer


def merge(
    run_dirs: list[Path],
    slug: str,
    layers: list[int],
    k: int,
    out_dir: Path,
    tasks: list[str],
    seed: int = 0,
    force_meta: int = 8,
    pca: float | int | None = None,
    threshold: float = 0.2,
    method: str = "semantic-avglink",
    annotate: bool = True,
    annotate_model: str = DEFAULT_ANNOTATE_MODEL,
    resolution: float | None = None,
    merge_k: int | None = None,
    consensus_runs: int = 3,
    stability_floor: float = 0.9,
    merge_knn: int = 6,
    merge_seed: str = "avglink",
    name_themes: bool = False,
    judge_coherence: bool = False,
    distinctive: bool = False,
    embed_model: str | None = None,
    annotate_backend: str = "copilot",
) -> dict:
    """Cluster + annotate each layer's meta-clusters INDEPENDENTLY, then MERGE across layers into
    cross-layer THEMES **only when functionally the same**, so the theme count is DATA-DRIVEN
    (distinct clusters stay separate, even as singletons): by shared-token Jaccard overlap >=
    ``threshold`` (``tokens``, connected components) or by semantic label (``labels``: one LLM call
    grouping only functionally-equivalent clusters, count flexible). Per-layer clustering is cached
    (clusters.npz/json) so re-merging at another threshold/method is instant."""
    per_layer = _load_or_cluster(
        run_dirs,
        slug,
        layers,
        k,
        seed,
        force_meta,
        tasks,
        pca,
        annotate,
        annotate_model,
        out_dir,
        distinctive,
        annotate_backend,
    )
    used = [pl["layer"] for pl in per_layer]
    n_layers = len(per_layer)
    stack = np.stack([pl["labels"] for pl in per_layer])
    sizes = np.array([[int((stack[i] == m).sum()) for m in range(force_meta)]
                      for i in range(n_layers)])
    ami = np.eye(n_layers)
    for a in range(n_layers):
        for b in range(a + 1, n_layers):
            ami[a, b] = ami[b, a] = adjusted_mutual_info_score(stack[a], stack[b])
    if method in ("labels", "semantic-cpm", "semantic-avglink") and not annotate:
        raise SystemExit(
            f"--merge {method} needs --annotate (per-layer LLM annotations of the meta-clusters). "
            "Add --annotate, or use a retired token method (avglink / leiden-cpm / tokens) "
            "for an annotation-free structural merge.")
    if method == "labels" and annotate:
        node_theme, theme_names, info = _merge_by_labels(per_layer, force_meta, annotate_model,
                                                         consensus_runs, distinctive)
    elif method == "tokens":
        node_theme, theme_names, info = _merge_by_tokens(stack, sizes, force_meta, threshold)
    elif method == "leiden-cpm":
        node_theme, theme_names, info = _merge_by_leiden_cpm(stack, sizes, force_meta, resolution,
                                                             merge_k, seed,
                                                             stability_floor=stability_floor)
    elif method == "semantic-cpm" and annotate:
        node_theme, theme_names, info = _merge_by_semantic_cpm(per_layer, force_meta, resolution,
                                                               merge_k, seed,
                                                               stability_floor=stability_floor,
                                                               knn=merge_knn,
                                                               embed_model=embed_model)
    elif method == "semantic-avglink" and annotate:
        node_theme, theme_names, info = _merge_by_semantic_avglink(per_layer, force_meta, merge_k,
                                                                   embed_model=embed_model)
    elif method == "llm-refine":
        node_theme, theme_names, info = _merge_by_llm_refine(per_layer, stack, sizes, force_meta,
                                                             annotate_model, merge_seed, resolution,
                                                             merge_k, seed)
    else:
        method = "avglink"  # default; also the fallback for labels-without-annotate
        node_theme, theme_names, info = _merge_by_avglink(stack, sizes, force_meta, merge_k)
    grid = node_theme.reshape(n_layers, force_meta)
    n_final = int(node_theme.max()) + 1
    n_singleton = sum(int((node_theme == t).sum()) == 1 for t in range(n_final))
    layer_theme = np.zeros((n_layers, n_final))
    for i in range(n_layers):
        for m in range(force_meta):
            layer_theme[i, grid[i, m]] += sizes[i, m]
    layer_theme = layer_theme / layer_theme.sum(1, keepdims=True)
    themes = _theme_profiles(per_layer, grid, sizes, force_meta, n_final, theme_names)
    if name_themes and annotate:
        _name_themes(themes, annotate_model, annotate_backend)
        if judge_coherence:
            _judge_theme_coherence(themes, annotate_model)
    spans = _arc_spans(per_layer, grid, force_meta, n_final)
    max_arc = round(max(spans), 3) if spans else 0.0
    n_arc = int(sum(s > 0.4 for s in spans))
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {"slug": slug, "tasks": tasks, "k": k, "force_meta": force_meta, "pca": pca,
             "method": method, "threshold": threshold if method == "tokens" else None,
             "merge_info": info, "max_arc_span": max_arc, "n_arc_spanning": n_arc,
             "n_themes": n_final, "n_singletons": n_singleton, "n_nodes": n_layers * force_meta,
             "layers": used, "jaccard_pctl": info.get("jaccard_pctl"),
             "pairwise_ami": ami.round(4).tolist(), "layer_theme": layer_theme.round(4).tolist(),
             "node_theme": grid.tolist(), "themes": themes,
             "per_layer_clusters": [
                 {"layer": pl["layer"],
                  "clusters": [{"meta": p["meta"],
                                "label": pl["ann"].get(p["meta"], {}).get("label", ""),
                                "mean_pos": p["mean_pos"], "top": p["top"][:12]}
                               for p in pl["profiles"]]}
                 for pl in per_layer]}
    (out_dir / "merge.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    _layer_theme_figure(layer_theme, used, themes, out_dir / "layer_themes.png", slug)
    _theme_membership_figure(grid, used, force_meta, themes,
                             out_dir / "theme_membership.png", slug)
    _merge_map_figure(per_layer, grid, themes, force_meta, used, out_dir / "merge_map.png", slug)
    _ami_figure(ami, used, out_dir / "ami_matrix.png", slug)
    _write_merge_findings(out_dir / "merge_findings.md", used, layer_theme, themes, per_layer,
                          method)
    _theme_depth_profile_figure(layer_theme, themes, used, out_dir / "theme_depth_profile.png",
                                slug)
    _theme_nodes_figure(per_layer, grid, sizes, themes, out_dir / "theme_nodes.png", slug)
    if method == "tokens":
        _theme_selection_figure(stack, sizes, force_meta, threshold,
                                out_dir / "theme_selection.png", slug)
    if method in ("leiden-cpm", "semantic-cpm") and info.get("gamma_sweep"):
        _cpm_selection_figure(info["gamma_sweep"], info.get("resolution"),
                              info.get("stability_floor", 0.9), out_dir / "cpm_selection.png", slug,
                              info.get("objective", "modularity"))
    passages = _passage_tokens(run_dirs, slug, tasks, used[0])
    if sum(len(t) for t in passages) == stack.shape[1]:
        plens = [len(t) for t in passages]
        _theme_transition_figure(stack, grid, themes, plens, out_dir / "theme_transition.png", slug)
        _theme_flow_figure(stack, grid, themes, plens, seed, out_dir / "theme_flow.png", slug)
        _theme_graph_figure(stack, grid, themes, plens, seed, out_dir / "theme_graph.png", slug)
        _theme_trace_depth_figure(stack, grid, themes, plens, used,
                                  out_dir / "theme_trace_depth.png", slug)
        _trace_html(stack, grid, themes, _trace_sample(passages), used, out_dir / "traces.html",
                    slug, method)
    else:
        print(f"[warn] token reconstruction {sum(len(t) for t in passages)} != {stack.shape[1]} "
              "cols; skipping trace figures")
    if method == "tokens":
        extra = f" thr={threshold}"
    elif method in ("leiden-cpm", "semantic-cpm"):
        extra = (f" res={info.get('resolution')} sel={info.get('selection')}"
                 f"/{info.get('objective', 'modularity')}"
                 f" stability={info.get('stability')} modularity={info.get('modularity')}")
        if info.get("silhouette") is not None:
            extra += f" silhouette={info.get('silhouette')}"
        if info.get("knn") is not None:
            extra += f" knn={info.get('knn')}"
    elif info.get("gap_cut") is not None:
        extra = f" gap_cut={info.get('gap_cut')}"
    elif info.get("silhouette_cut_k") is not None:
        extra = f" silhouette_cut_k={info.get('silhouette_cut_k')}"
    elif merge_k:
        extra = f" k={merge_k}"
    elif method == "labels":
        extra = f" runs={info.get('n_runs')} stability={info.get('stability')}"
    elif method == "llm-refine":
        extra = f" seed={info.get('seed')} refined={info.get('refined')}"
    else:
        extra = ""
    if info.get("embed_model"):
        extra += f" embed={info['embed_model'].split('/')[-1]}"
    warn = f" [WARN {n_arc} arc-spanning >0.4]" if n_arc else ""
    print(f"-> {out_dir} ({n_final} themes = {n_final - n_singleton} merged + {n_singleton} "
          f"singletons via {method}{extra}; max_arc_span={max_arc}{warn})")
    return stats


def _jaccard_matrix(stack: np.ndarray, sizes: np.ndarray, force_meta: int) -> np.ndarray:
    """(n_layers*force_meta)^2 SHARED-TOKEN Jaccard between (layer, meta) nodes -- cross-layer
    token-instance co-membership (0 within a layer, which partitions the same tokens)."""
    n_layers = stack.shape[0]
    n_nodes = n_layers * force_meta
    jac = np.zeros((n_nodes, n_nodes))
    for i in range(n_layers):
        for j in range(i + 1, n_layers):
            cont = _contingency(stack[i], stack[j], force_meta)
            for a in range(force_meta):
                for b in range(force_meta):
                    union = sizes[i, a] + sizes[j, b] - cont[a, b]
                    if union > 0:
                        v = cont[a, b] / union
                        jac[i * force_meta + a, j * force_meta + b] = v
                        jac[j * force_meta + b, i * force_meta + a] = v
    return jac


def _consecutive(labels: np.ndarray) -> np.ndarray:
    """0-based consecutive theme ids (merge downstream assumes range(max+1) is dense)."""
    _, inv = np.unique(labels, return_inverse=True)
    return inv.astype(int)


def _merge_by_tokens(stack: np.ndarray, sizes: np.ndarray, force_meta: int,
                     threshold: float) -> tuple[np.ndarray, dict | None, dict]:
    """RETIRED single-linkage merge (kept runnable; superseded by the semantic merges -- token
    overlap can't separate co-occurring functions): edge two nodes whose Jaccard >= ``threshold``,
    take connected components. Fragile in ``threshold`` (chains via one bridge edge)."""
    from scipy.sparse.csgraph import connected_components
    jac = _jaccard_matrix(stack, sizes, force_meta)
    n_nodes = jac.shape[0]
    _, labels = connected_components(jac >= threshold, directed=False)
    pos = jac[np.triu_indices(n_nodes, 1)]
    pos = pos[pos > 0]
    pctl = ({str(p): round(float(np.percentile(pos, p)), 4) for p in (50, 75, 90, 95, 99)}
            if len(pos) else {})
    return labels, None, {"jaccard_pctl": pctl}


def _gap_cut(z: np.ndarray, k_min: int = 4, k_max: int = 40) -> tuple[np.ndarray, float]:
    """Cut an agglomerative tree at the largest gap between successive merge heights whose cut
    yields k in [k_min, k_max] -- a data-driven theme count (no similarity threshold)."""
    from scipy.cluster.hierarchy import fcluster
    heights = z[:, 2]
    best = (-1.0, float(np.median(heights)))
    for t in range(1, len(heights)):
        cut = float((heights[t] + heights[t - 1]) / 2)
        kk = int(fcluster(z, t=cut, criterion="distance").max())
        if k_min <= kk <= k_max and heights[t] - heights[t - 1] > best[0]:
            best = (float(heights[t] - heights[t - 1]), cut)
    return _consecutive(fcluster(z, t=best[1], criterion="distance")), round(best[1], 4)


def _silhouette_cut(z: np.ndarray, dist: np.ndarray, k_min: int = 4,
                    k_max: int = 40) -> tuple[np.ndarray, int]:
    """Cut an agglomerative tree at the k in [k_min, k_max] whose flat clustering MAXIMISES the
    silhouette in the precomputed distance space -- a geometry-aware theme count that is more robust
    than the dendrogram gap (:func:`_gap_cut`) when the metric space is meaningful (instruction
    embeddings): gap-cut over-merges the ``spot`` runs, silhouette tracks the LLM-labels count.
    Only sound on a dense/real metric -- prefer :func:`_gap_cut` for near-orthogonal TF-IDF."""
    from scipy.cluster.hierarchy import fcluster
    n = dist.shape[0]
    best = (-2.0, min(k_min, n - 1))
    for k in range(k_min, min(k_max, n - 1) + 1):
        lab = fcluster(z, t=k, criterion="maxclust")
        if 1 < len(np.unique(lab)) < n:
            s = float(silhouette_score(dist, lab, metric="precomputed"))
            if s > best[0]:
                best = (s, k)
    return _consecutive(fcluster(z, t=best[1], criterion="maxclust")), best[1]


def _merge_by_avglink(stack: np.ndarray, sizes: np.ndarray, force_meta: int,
                      merge_k: int | None) -> tuple[np.ndarray, dict | None, dict]:
    """RETIRED token-cohesion merge (kept runnable; superseded by ``semantic-avglink`` -- token
    Jaccard can't separate co-occurring functions, the dense semantic cosine can): average-linkage
    over 1 - Jaccard, cut at ``merge_k`` (if given) else the dendrogram gap. Merges on group MEAN
    (not one bridge edge), so it resists single-linkage chaining. Deterministic."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    d = 1.0 - _jaccard_matrix(stack, sizes, force_meta)
    d = (d + d.T) / 2
    np.fill_diagonal(d, 0.0)
    z = linkage(squareform(d, checks=False), method="average")
    if merge_k is not None:
        lab = fcluster(z, t=merge_k, criterion="maxclust")
        return _consecutive(lab), None, {"merge_k": merge_k}
    labels, cut = _gap_cut(z)
    return labels, None, {"gap_cut": cut}


def _leiden_cpm(jac: np.ndarray, resolution: float, seed: int) -> np.ndarray:
    try:
        import igraph as ig
        import leidenalg
    except ImportError as e:  # optional dep: the leiden-cpm / semantic-cpm methods need it
        raise SystemExit("--merge leiden-cpm/semantic-cpm needs leidenalg + igraph "
                         "(`uv sync --extra merge`)") from e
    n = jac.shape[0]
    edges, weights = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if jac[i, j] > 0:
                edges.append((i, j))
                weights.append(float(jac[i, j]))
    part = leidenalg.find_partition(ig.Graph(n=n, edges=edges), leidenalg.CPMVertexPartition,
                                    weights=weights, resolution_parameter=resolution, seed=seed)
    return np.asarray(part.membership)


def _graph_modularity(jac: np.ndarray, labels: np.ndarray) -> float:
    """Newman weighted modularity of ``labels`` on the Jaccard graph (zero diagonal). This scores
    how cleanly the partition separates dense communities from sparse background -- it is the AUTO
    operating-point selector, replacing "coarsest reproducible gamma" (which sought maximal merging
    and thus biased toward a single giant blob) with "best-structured reproducible gamma"."""
    a = jac.copy()
    np.fill_diagonal(a, 0.0)
    m2 = float(a.sum())  # 2m
    if m2 <= 0:
        return 0.0
    deg = a.sum(axis=1)
    q = 0.0
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        q += a[np.ix_(idx, idx)].sum() / m2 - (deg[idx].sum() / m2) ** 2
    return float(q)


def _cpm_consensus(jac: np.ndarray, gamma: float, seed: int,
                   n_seeds: int) -> tuple[np.ndarray, float]:
    """Multi-seed CPM at a fixed ``gamma``: deterministic CONSENSUS partition (median co-assignment
    over ``n_seeds`` seeds) + a stability score (mean pairwise AMI across seeds). Tames Leiden's
    seed stochasticity so the reported partition/count is reproducible, not a single-seed draw."""
    parts = [_leiden_cpm(jac, gamma, seed + s) for s in range(n_seeds)]
    return _consensus_partition(parts)


def _merge_by_leiden_cpm(stack: np.ndarray, sizes: np.ndarray, force_meta: int,
                         resolution: float | None, merge_k: int | None, seed: int,
                         n_seeds: int = 11, stability_floor: float = 0.9,
                         ) -> tuple[np.ndarray, dict | None, dict]:
    """RETIRED token-graph merge (kept runnable; superseded by ``semantic-cpm`` -- the SAME CPM
    consensus on a semantic graph, not token-Jaccard). Leiden with the CPM objective (no modularity
    resolution limit) on the weighted Jaccard graph, made reproducible by a MULTI-SEED CONSENSUS (B)
    with a
    reported stability (mean pairwise AMI). The resolution ``gamma`` IS the density threshold (nodes
    group when internal Jaccard density > gamma); the operating point is chosen PRINCIPALLY (A),
    not left as a magic constant:

    - ``merge_k`` given    -> gamma-scan, pick consensus closest to that theme count (tie: coarser)
    - ``resolution`` given -> that exact gamma (explicit override)
    - else (AUTO, default) -> among the *reproducible* gammas (consensus stability >=
      ``stability_floor``) the one whose partition has the HIGHEST modularity, i.e. the cleanest
      community structure (tie: coarser). Falls back to the max-stability gamma when none clear the
      floor. Picking the best-structured -- not merely the coarsest -- scale avoids the over-merge
      blob the coarsest reproducible gamma tends to sit on.

    A gamma-sweep (theme count / singletons / largest / stability / modularity vs gamma) is always
    emitted so the operating point is inspectable. Deterministic given ``seed``. Needs leidenalg +
    igraph."""
    jac = _jaccard_matrix(stack, sizes, force_meta)
    grid = [round(float(g), 3) for g in np.arange(0.02, 0.301, 0.02)]
    labels, info = _cpm_select_from_graph(jac, grid, resolution, merge_k, seed, n_seeds,
                                          stability_floor)
    return labels, None, info


def _cpm_select_from_graph(graph: np.ndarray, grid: list[float], resolution: float | None,
                           merge_k: int | None, seed: int, n_seeds: int,
                           stability_floor: float,
                           select_sim: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Shared CPM operating-point search for any weighted node graph (token-Jaccard for
    ``leiden-cpm`` or semantic cosine for ``semantic-cpm``): sweep ``gamma`` over ``grid`` with a
    multi-seed consensus, then pick the operating gamma by target-k / explicit resolution / AUTO
    (coarser breaks ties; max-stability fallback). Returns (consecutive consensus labels, info).

    AUTO objective: by default the highest-**modularity** gamma among those whose consensus
    stability >= ``stability_floor``. When a dense ``select_sim`` (NxN cosine) is supplied -- the
    embedding backend -- AUTO instead maximises the **silhouette** of the consensus in that metric
    space. Newman modularity has a resolution limit that biases the pick coarse (it re-introduces
    the over-merge CPM avoids); silhouette, sound only when the geometry is real (instruction
    embeddings, NOT ~90%-zero TF-IDF), tracks the LLM-labels granularity far better on the harder
    runs and is a no-op where the two agree."""
    dist = None
    if select_sim is not None:
        dist = 1.0 - select_sim
        dist = (dist + dist.T) / 2
        np.fill_diagonal(dist, 0.0)
        np.clip(dist, 0.0, None, out=dist)
    sweep, cache = [], {}
    for g in grid:
        cons, stab = _cpm_consensus(graph, g, seed, n_seeds)
        cache[g] = (cons, stab)
        bc = np.bincount(cons)
        row = {"gamma": g, "n_themes": int(len(bc)),
               "n_singletons": int((bc == 1).sum()), "largest": int(bc.max()),
               "stability": round(float(stab), 3),
               "modularity": round(_graph_modularity(graph, cons), 3)}
        if dist is not None:
            row["silhouette"] = (round(float(silhouette_score(dist, cons, metric="precomputed")), 3)
                                 if 1 < len(bc) < len(cons) else -1.0)
        sweep.append(row)
    objective = "silhouette" if dist is not None else "modularity"
    if merge_k is not None:
        gamma = min(grid, key=lambda g: (abs(len(np.unique(cache[g][0])) - merge_k), g))
        labels, stability = cache[gamma]
        selection = "target-k"
    elif resolution is not None:
        gamma = round(float(resolution), 4)
        labels, stability = _cpm_consensus(graph, gamma, seed, n_seeds)
        selection = "fixed"
    else:
        eligible = [s for s in sweep if s["stability"] >= stability_floor]
        if eligible:
            gamma = max(eligible, key=lambda s: (s[objective], -s["gamma"]))["gamma"]
            selection = "auto"
        else:
            gamma = max(sweep, key=lambda s: s["stability"])["gamma"]
            selection = "auto-fallback"
        labels, stability = cache[gamma]
    chosen = next((s for s in sweep if s["gamma"] == gamma), {})
    info = {"resolution": round(float(gamma), 4), "stability": round(float(stability), 3),
            "modularity": chosen.get("modularity"), "objective": objective,
            "n_seeds": n_seeds, "selection": selection, "stability_floor": stability_floor,
            "gamma_sweep": sweep}
    if dist is not None:
        info["silhouette"] = chosen.get("silhouette")
    return _consecutive(labels), info


# Qwen3-Embedding instruction (the model is instruction-tuned; the card recommends prepending a
# one-line task so "same function" texts embed closer -- +1-5% and, here, removes the k-fragility /
# avglink chaining that un-instructed embeddings suffer from anisotropy).
_EMBED_TASK = ("Identify the functional role of this language-model reasoning feature, so that "
               "features implementing the same reasoning function are grouped into the same "
               "cluster")
_EMBED_CACHE: dict = {}  # model_name -> (tokenizer, model); load the weights at most once/process


def _embed_texts(texts: list[str], model_name: str, batch_size: int = 16,
                 max_length: int = 512) -> np.ndarray:
    """L2-normalised instruction embeddings (Qwen3-Embedding style: prepend ``_EMBED_TASK``, then
    LAST-token pooling with LEFT padding). Lazy-imports torch/transformers and caches the loaded
    model per process. No sampling -- reproducible up to bf16 GPU arithmetic (far more so than the
    LLM annotation pass it builds on). Opt-in similarity backend for the semantic merges."""
    try:
        import torch
        import torch.nn.functional as F
        from transformers import AutoModel, AutoTokenizer
    except ImportError as e:  # opt-in: only --merge-embed needs the neural backend
        raise SystemExit("--merge-embed needs torch + transformers "
                         "(`uv sync --extra gpu`)") from e
    if model_name not in _EMBED_CACHE:
        tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        model = AutoModel.from_pretrained(
            model_name, dtype=torch.bfloat16,
            device_map="cuda" if torch.cuda.is_available() else None).eval()
        _EMBED_CACHE[model_name] = (tok, model)
    tok, model = _EMBED_CACHE[model_name]
    device = next(model.parameters()).device
    prompt = [f"Instruct: {_EMBED_TASK}\nQuery:{t}" for t in texts]
    out = []
    with torch.no_grad():
        for i in range(0, len(prompt), batch_size):
            enc = tok(prompt[i:i + batch_size], padding=True, truncation=True,
                      max_length=max_length, return_tensors="pt").to(device)
            h = model(**enc).last_hidden_state
            mask = enc["attention_mask"]
            if bool(mask[:, -1].sum() == mask.shape[0]):  # left-padded -> real last col
                pooled = h[:, -1]
            else:
                pooled = h[torch.arange(h.shape[0]), mask.sum(dim=1) - 1]
            emb = F.normalize(pooled.float(), p=2, dim=1)
            out.append(emb.cpu().numpy())
    return np.concatenate(out, 0)


def _node_annotation_texts(per_layer: list[dict], force_meta: int,
                           include_top: bool) -> tuple[list[str], np.ndarray]:
    """(text, present) per (layer, meta) node from the cached annotations. ``include_top`` appends
    the top-15 raw cluster tokens: TF-IDF needs that lexical mass, but the neural backend does
    BETTER on ``label`` + ``description`` alone -- the raw reasoning tokens (wait/okay/the/...) are
    shared across functions and bridge otherwise-distinct nodes."""
    texts, present = [], []
    for pl in per_layer:
        prof = {p["meta"]: p for p in pl["profiles"]}
        for m in range(force_meta):
            ann = pl["ann"].get(m, {})
            top = " ".join((prof.get(m, {}).get("top") or [])[:15]) if include_top else ""
            txt = f"{ann.get('label', '')} {ann.get('description', '')} {top}".strip()
            texts.append(txt or "empty")
            present.append(m in prof)
    return texts, np.array(present)


def _semantic_similarity(per_layer: list[dict], force_meta: int,
                         embed_model: str | None = None) -> np.ndarray:
    """DENSE NxN semantic cosine similarity between nodes from the cached annotations, zero
    diagonal, absent (layer, meta) nodes disconnected. Backend: TF-IDF cosine over
    ``label`` + ``description`` + top-tokens (default, CPU-only, deterministic) or -- when
    ``embed_model`` is given -- cosine of instruction embeddings over ``label`` + ``description``
    (:func:`_embed_texts`). Both carry the functional signal the token-instance Jaccard graph lacks
    (Jaccard is 0 within a layer and cannot separate functions that co-occur in token space)."""
    texts, present = _node_annotation_texts(per_layer, force_meta, include_top=embed_model is None)
    if embed_model:
        emb = _embed_texts(texts, embed_model)
        sim = (emb @ emb.T).astype(float)
    else:
        tfidf = TfidfVectorizer(stop_words="english", min_df=1).fit_transform(texts)
        sim = cosine_similarity(tfidf).astype(float)
    np.fill_diagonal(sim, 0.0)
    sim[~present, :] = 0.0
    sim[:, ~present] = 0.0
    return sim


def _knn_sparsify(sim: np.ndarray, knn: int) -> np.ndarray:
    """Union/OR k-NN sparsification of a dense similarity: keep each node's ``knn`` strongest
    neighbours, symmetrise by ``max`` so an edge survives if EITHER endpoint ranks the other in its
    top-k. Zero diagonal. ``knn`` falsy/>=N returns the dense matrix unchanged."""
    if not knn or knn >= sim.shape[0]:
        return sim
    sparse = np.zeros_like(sim)
    for i in range(sim.shape[0]):
        nbr = np.argsort(-sim[i])[:knn]
        sparse[i, nbr] = sim[i, nbr]
    out = np.maximum(sparse, sparse.T)
    np.fill_diagonal(out, 0.0)
    return out


def _merge_by_semantic_cpm(per_layer: list[dict], force_meta: int, resolution: float | None,
                           merge_k: int | None, seed: int, n_seeds: int = 11,
                           stability_floor: float = 0.9, knn: int = 6,
                           embed_model: str | None = None,
                           ) -> tuple[np.ndarray, dict | None, dict]:
    """GO-FORWARD semantic merge (recommended with ``--merge-embed``): build a semantic-cosine k-NN
    graph over the cached per-node annotations (dense cosine :func:`_semantic_similarity` -- TF-IDF
    by default, or instruction
    embeddings when ``embed_model`` is set -- union/OR-sparsified by :func:`_knn_sparsify`) and run
    the SAME multi-seed Leiden-CPM consensus + AUTO selection as :func:`_merge_by_leiden_cpm`.
    Token-instance Jaccard over-concentrates the largest theme (a functional mixture glued by shared
    reasoning vocabulary); the semantic graph separates those functions deterministically, reaching
    LLM-labels-like balance without a live LLM grouping call. With ``embed_model`` the dense cosine
    is also handed to the selector as ``select_sim`` so AUTO picks the gamma by silhouette (removes
    modularity's coarsening bias); TF-IDF keeps modularity. Needs annotated clusters + leidenalg."""
    sim = _semantic_similarity(per_layer, force_meta, embed_model)
    graph = _knn_sparsify(sim, knn)
    grid = [round(float(g), 3) for g in np.arange(0.05, 0.75, 0.05)]
    labels, info = _cpm_select_from_graph(graph, grid, resolution, merge_k, seed, n_seeds,
                                          stability_floor,
                                          select_sim=sim if embed_model else None)
    info["knn"] = knn
    info["embed_model"] = embed_model
    return labels, None, info


def _merge_by_semantic_avglink(per_layer: list[dict], force_meta: int, merge_k: int | None,
                               embed_model: str | None = None,
                               ) -> tuple[np.ndarray, dict | None, dict]:
    """DEFAULT semantic merge (recommended with ``--merge-embed``): average-linkage over
    ``1 - dense semantic cosine``
    (:func:`_semantic_similarity`), cut at ``merge_k`` else -- for the embedding backend -- the
    SILHOUETTE-optimal k (:func:`_silhouette_cut`), else the dendrogram gap (:func:`_gap_cut`).
    No k-NN, no gamma, no leidenalg. Unlike token avglink (which chains/over-splits on ~90%-zero
    Jaccard) this needs a DENSE similarity: it shines with ``embed_model`` (instruction embeddings
    give real distances everywhere), whereas TF-IDF's near-orthogonal annotation vectors make
    linkage over-split into singletons. Gap-cut over-merges the embedding ``spot`` runs, so the
    embed path uses the silhouette cut (gap-cut stays the default for the sparse TF-IDF geometry).
    Deterministic; needs annotated clusters."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    sim = _semantic_similarity(per_layer, force_meta, embed_model)
    d = 1.0 - sim
    d = (d + d.T) / 2
    np.fill_diagonal(d, 0.0)
    np.clip(d, 0.0, None, out=d)  # bf16 cosine can round fractionally > 1 -> guard scipy's check
    z = linkage(squareform(d, checks=False), method="average")
    if merge_k is not None:
        return (_consecutive(fcluster(z, t=merge_k, criterion="maxclust")), None,
                {"merge_k": merge_k, "embed_model": embed_model})
    if embed_model:
        labels, k = _silhouette_cut(z, d)
        return labels, None, {"silhouette_cut_k": k, "embed_model": embed_model}
    labels, cut = _gap_cut(z)
    return labels, None, {"gap_cut": cut, "embed_model": embed_model}


def _arc_spans(per_layer: list[dict], grid: np.ndarray, force_meta: int,
               n_themes: int) -> list[float]:
    """Per-theme spread of member ``mean_pos`` (max - min): the chaining diagnostic -- a theme
    spanning the whole read -> deliberate -> verdict arc is likely an over-merge."""
    spans = []
    for t in range(n_themes):
        pos = []
        for i, pl in enumerate(per_layer):
            by_meta = {p["meta"]: p for p in pl["profiles"]}
            for m in range(force_meta):
                if grid[i, m] == t and by_meta.get(m, {}).get("mean_pos") is not None:
                    pos.append(by_meta[m]["mean_pos"])
        if len(pos) >= 2:
            spans.append(max(pos) - min(pos))
    return spans


_MERGE_PROMPT = (
    "Below are meta-clusters of a reasoning model's response tokens, discovered SEPARATELY at "
    "several network layers. Each line has: id, layer + local cluster, depth position (pos, "
    "0=start..1=end of the response), a short label -- description, "
    "its top tokens, and 'near:' the ids of its most token-similar clusters at OTHER layers. Merge "
    "ONLY clusters doing the SAME function into a shared THEME; keep a functionally DISTINCT "
    "cluster as its OWN theme. Use as many themes as the data warrants -- do NOT force a small "
    "number. Return JSON {{\"themes\": [{{\"theme\": 0, \"name\": \"...\"}}], "
    "\"assignments\": [{{\"id\": 0, \"theme\": 0}}]}} covering every id.\n\n{items}"
)


def _parse_assignments(data: dict, n_nodes: int) -> tuple[np.ndarray, dict, dict]:
    """Validate an LLM ``{themes, assignments}`` merge response into (raw_labels, names, report).
    Enforces COVERAGE: every node 0..n_nodes-1 gets a theme; an id the LLM OMITS becomes its OWN
    fresh singleton (never silently theme 0). Drops out-of-range / negative / non-int / DUPLICATE
    ids (first assignment wins). Requires an ``assignments`` list, else raises LLMParseError."""
    from ..core.llm import LLMParseError
    if not isinstance(data.get("assignments"), list):
        raise LLMParseError("LLM merge response missing an 'assignments' list")
    labels = np.full(n_nodes, -1, dtype=int)  # -1 = not-yet-assigned sentinel
    report = {"unassigned": 0, "dropped": 0, "duplicates": 0}
    for a in data["assignments"]:
        try:
            nid, tid = int(a["id"]), int(a["theme"])
        except (KeyError, TypeError, ValueError):
            report["dropped"] += 1
            continue
        if not (0 <= nid < n_nodes) or tid < 0:
            report["dropped"] += 1
            continue
        if labels[nid] != -1:
            report["duplicates"] += 1  # first assignment wins
            continue
        labels[nid] = tid
    nxt = int(labels.max()) + 1 if (labels >= 0).any() else 0
    for nid in np.where(labels < 0)[0]:  # omitted ids -> their own singleton, NOT theme 0
        labels[nid] = nxt
        nxt += 1
        report["unassigned"] += 1
    names: dict = {}
    for th in data.get("themes") or []:
        try:
            names[int(th["theme"])] = str(th.get("name", "")).strip()
        except (KeyError, TypeError, ValueError):
            continue
    return labels, names, report


def _consensus_partition(parts: list[np.ndarray]) -> tuple[np.ndarray, float]:
    """Consensus of repeated partitions: average-linkage on ``1 - co-assignment frequency`` (cut at
    the median theme count) + mean pairwise AMI. Mirrors explore.merge_eval.stability (offline)."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    amis = [adjusted_mutual_info_score(parts[a], parts[b])
            for a in range(len(parts)) for b in range(a + 1, len(parts))]
    n = len(parts[0])
    co = np.zeros((n, n))
    for p in parts:
        co += (p[:, None] == p[None, :]).astype(float)
    co /= len(parts)
    d = 1.0 - co
    d = (d + d.T) / 2
    np.fill_diagonal(d, 0.0)
    z = linkage(squareform(d, checks=False), method="average")
    k = max(int(np.median([len(np.unique(p)) for p in parts])), 1)
    _, cons = np.unique(fcluster(z, t=k, criterion="maxclust"), return_inverse=True)
    return cons, round(float(np.mean(amis)), 3) if amis else 1.0


def _labels_once(prompt: str, n_nodes: int, model: str) -> tuple[np.ndarray | None, dict, dict]:
    """One LLM merge call -> coverage-validated raw labels (None on failure) + names + repair."""
    from ..core.llm import extract_json, run_copilot_cli
    try:
        data = extract_json(run_copilot_cli(prompt, model=model), require_key="assignments")
        return _parse_assignments(data, n_nodes)
    except (ValueError, KeyError, TypeError, RuntimeError) as e:
        print(f"[merge labels run failed: {e}]")
        return None, {}, {"unassigned": 0, "dropped": 0, "duplicates": 0}


def _merge_item_lines(per_layer: list[dict], force_meta: int, neighbors: bool = True,
                      distinctive: bool = False) -> tuple[list[str], list, np.ndarray]:
    """Enriched per-node lines for the merge/refine prompt: depth (mean_pos),
    ``label -- description``, top tokens, optional ``distinctive`` tf-idf
    tokens (characteristic vs all other nodes), and (if ``neighbors``) the top-3 cross-layer
    Jaccard-nearest node ids (a geometry hint). Returns (lines, nodes, jaccard)."""
    stack = np.stack([pl["labels"] for pl in per_layer])
    sizes = np.array([[int((stack[i] == m).sum()) for m in range(force_meta)]
                      for i in range(len(per_layer))])
    jac = _jaccard_matrix(stack, sizes, force_meta)
    meta, node_counts = [], []
    for pl in per_layer:
        by_meta = {p["meta"]: p for p in pl["profiles"]}
        for m in range(force_meta):
            ann, pr = pl["ann"].get(m, {}), by_meta.get(m, {})
            meta.append((pl["layer"], m, ann.get("label", "?"), ann.get("description", ""),
                         pr.get("mean_pos"), ", ".join(pr.get("top", [])[:8])))
            node_counts.append([(t, c) for t, c in (pr.get("tok_counts") or [])])
    disc = _distinctive_tokens(node_counts, 8) if distinctive else None
    lines, nodes = [], []
    for nid, (layer, m, lab, desc, pos, top) in enumerate(meta):
        nbr = [int(j) for j in np.argsort(-jac[nid]) if j != nid and jac[nid, j] > 0][:3]
        pos_s = f" pos={pos:.2f}" if pos is not None else ""
        desc_s = f" -- {desc}" if desc else ""
        near_s = f" | near: {', '.join(map(str, nbr))}" if (neighbors and nbr) else ""
        disc_s = f" | distinctive: {', '.join(disc[nid])}" if (disc and disc[nid]) else ""
        lines.append(
            f"- id {nid} (L{layer} m{m}){pos_s}: {lab}{desc_s} | top: {top}{disc_s}{near_s}")
        nodes.append((layer, m))
    return lines, nodes, jac


def _merge_by_labels(per_layer: list[dict], force_meta: int, model: str,
                     consensus_runs: int = 1,
                     distinctive: bool = False) -> tuple[np.ndarray, dict | None, dict]:
    """Group the per-layer labelled meta-clusters into cross-layer themes via the LLM. Output is
    COVERAGE-validated (omitted ids -> singletons, never a silent theme-0 collapse). With
    ``consensus_runs > 1`` the call is repeated and a CONSENSUS partition (co-assignment
    average-linkage) is returned with a stability score (mean pairwise AMI); consensus themes are
    named downstream from member labels. Falls back to token overlap only if EVERY run fails."""
    lines, nodes, _jac = _merge_item_lines(per_layer, force_meta, distinctive=distinctive)
    prompt = _MERGE_PROMPT.format(items="\n".join(lines))
    if distinctive:
        prompt = ("Note: each line also has 'distinctive:' tokens characteristic of that cluster "
                  "vs others. Use them to keep genuinely different functions SEPARATE even when "
                  "their generic reasoning vocabulary overlaps.\n\n" + prompt)
    runs = [_labels_once(prompt, len(nodes), model) for _ in range(max(1, consensus_runs))]
    good = [(raw, nm, rep) for raw, nm, rep in runs if raw is not None]
    if not good:
        print("[merge labels: all runs failed; falling back to token overlap]")
        stack = np.stack([pl["labels"] for pl in per_layer])
        sizes = np.array([[int((stack[i] == m).sum()) for m in range(force_meta)]
                          for i in range(len(per_layer))])
        return _merge_by_tokens(stack, sizes, force_meta, 0.2)
    rep = {kk: sum(g[2].get(kk, 0) for g in good) for kk in ("unassigned", "dropped", "duplicates")}
    if len(good) == 1:
        raw, raw_names = good[0][0], good[0][1]
        info = {"n_runs": 1, "stability": 1.0, "repairs": rep}
    else:
        raw, stab = _consensus_partition([g[0] for g in good])
        raw_names = {}  # consensus themes named downstream (_theme_profiles) from member labels
        info = {"n_runs": len(good), "stability": stab, "repairs": rep}
    uniq, theme = np.unique(raw, return_inverse=True)
    names = {new: raw_names.get(int(old), f"theme {new}") for new, old in enumerate(uniq)}
    if any(rep.values()):
        print(f"[merge labels repairs: {rep['unassigned']} unassigned->singleton, "
              f"{rep['dropped']} dropped, {rep['duplicates']} dup]")
    return theme, names, info


_REFINE_PROMPT = (
    "A geometric clustering has grouped meta-clusters of a reasoning model's response tokens "
    "(discovered SEPARATELY at several network layers) into CANDIDATE cross-layer themes. Each "
    "candidate lists its member nodes: id, (layer, meta), depth pos, label -- "
    "description, top tokens. Your job: (1) NAME each final theme; (2) if a candidate MIXES "
    "functionally distinct steps, SPLIT it by listing sub-groups of its member ids; (3) if two "
    "candidates do the SAME function, MERGE them. Return JSON {{\"themes\": [{{\"theme\": <id>, "
    "\"name\": \"...\"}}], \"merges\": [[<id>, <id>]], \"splits\": {{\"<id>\": [[<node_id>, ...], "
    "[<node_id>, ...]]}}}}. Omit any merges/splits you don't need.\n\n{candidates}"
)


def _refine_candidates(seed_labels: np.ndarray, node_lines: list[str]) -> str:
    """Render the geometric seed partition as candidate-theme blocks for the refine prompt."""
    blocks = []
    for t in np.unique(seed_labels):
        ids = np.where(seed_labels == t)[0]
        body = "\n".join(f"  {node_lines[i]}" for i in ids)
        blocks.append(f"Candidate theme {int(t)}:\n{body}")
    return "\n\n".join(blocks)


def _apply_refine_ops(seed, merges, splits, n_nodes: int) -> np.ndarray:
    """Apply LLM refine ops to a geometric seed partition: MERGE seed-theme ids (union) and SPLIT a
    theme's members into sub-groups (fresh ids). Preserves full node coverage; ignores invalid
    theme/node refs; returns a dense 0-based partition."""
    labels = np.asarray(seed).copy()
    parent = {int(t): int(t) for t in np.unique(labels)}

    def find(x: int) -> int:
        x = int(x)
        while parent.get(x, x) != x:
            parent[x] = parent[parent.get(x, x)]
            x = parent[x]
        return x

    for grp in merges or []:
        ids = [int(g) for g in grp if int(g) in parent]
        for g in ids[1:]:
            parent[find(g)] = find(ids[0])
    labels = np.array([find(t) for t in labels])
    nxt = int(labels.max()) + 1 if len(labels) else 0
    for tid_s, subgroups in (splits or {}).items():
        try:
            tid = find(int(tid_s))
        except (ValueError, TypeError):
            continue
        members = set(np.where(labels == tid)[0].tolist())
        for gi, grp in enumerate(subgroups or []):
            ids = [int(x) for x in grp if 0 <= int(x) < n_nodes and int(x) in members]
            if not ids:
                continue
            new_id = tid if gi == 0 else nxt
            nxt += 0 if gi == 0 else 1
            for x in ids:
                labels[x] = new_id
    _, dense = np.unique(labels, return_inverse=True)
    return dense


def _merge_by_llm_refine(per_layer: list[dict], stack: np.ndarray, sizes: np.ndarray,
                         force_meta: int, model: str, seed_method: str = "avglink",
                         resolution: float | None = None, merge_k: int | None = None,
                         seed: int = 0) -> tuple[np.ndarray, dict | None, dict]:
    """Geometry-seeded LLM merge: a deterministic geometric partition (avglink / leiden-cpm)
    proposes CANDIDATE themes, then one LLM call SPLITS incoherent candidates, MERGES same-function
    ones, and names them -- a ~15-30 item task, validated for full coverage. Falls back to the seed
    on failure. Final themes are named downstream from member labels (Phase 4 adds naming)."""
    node_lines, nodes, _jac = _merge_item_lines(per_layer, force_meta, neighbors=False)
    if seed_method == "leiden-cpm":
        seed_labels = _merge_by_leiden_cpm(stack, sizes, force_meta, resolution, merge_k, seed)[0]
    else:
        seed_labels = _merge_by_avglink(stack, sizes, force_meta, merge_k)[0]
    prompt = _REFINE_PROMPT.format(candidates=_refine_candidates(seed_labels, node_lines))
    from ..core.llm import extract_json, run_copilot_cli
    try:
        data = extract_json(run_copilot_cli(prompt, model=model), require_key="themes")
        merges = data.get("merges") or []
        splits = {str(kk): vv for kk, vv in (data.get("splits") or {}).items()}
        labels = _apply_refine_ops(seed_labels, merges, splits, len(nodes))
    except (ValueError, KeyError, TypeError, RuntimeError) as e:
        print(f"[merge llm-refine failed: {e}; using geometric seed]")
        _, seed_dense = np.unique(seed_labels, return_inverse=True)
        return seed_dense, None, {"seed": seed_method, "refined": False}
    return labels, None, {"seed": seed_method, "refined": True,
                          "n_seed_themes": int(len(np.unique(seed_labels)))}


def _theme_profiles(per_layer: list[dict], grid: np.ndarray, sizes: np.ndarray, force_meta: int,
                    n_themes: int, theme_names: dict | None) -> list[dict]:
    """Aggregate each theme's member (layer, meta) nodes -> size-weighted UNIQUE name, top tokens,
    mean token position, and member list."""
    out = []
    for t in range(n_themes):
        members, label_w, tops = [], Counter(), Counter()
        pos_num = pos_den = 0.0
        for i, pl in enumerate(per_layer):
            by_meta = {p["meta"]: p for p in pl["profiles"]}
            for m in range(force_meta):
                if grid[i, m] != t:
                    continue
                sz = int(sizes[i, m])
                members.append(f"L{pl['layer']}m{m}")
                ann = pl["ann"].get(m, {})
                if ann.get("label"):
                    label_w[ann["label"]] += sz
                pr = by_meta.get(m, {})
                if pr.get("mean_pos") is not None:
                    pos_num += pr["mean_pos"] * sz
                    pos_den += sz
                for w in pr.get("top", [])[:10]:
                    tops[w] += sz
        name = (theme_names or {}).get(t) or (label_w.most_common(1)[0][0] if label_w
                                              else f"theme {t}")
        out.append({"theme": t, "name": name, "n_members": len(members), "members": members,
                    "member_labels": [lb for lb, _ in label_w.most_common(6)],
                    "mean_pos": round(pos_num / pos_den, 3) if pos_den else None,
                    "top": [w for w, _ in tops.most_common(12)]})
    _dedup_theme_names(out)
    return out


def _dedup_theme_names(themes: list[dict]) -> None:
    """Make theme display names UNIQUE in place: a repeated name gets a ' (2)', ' (3)', ... suffix,
    so a theme is never ambiguous downstream (label-quality: uniqueness)."""
    seen: dict[str, int] = {}
    for th in themes:
        key = str(th["name"]).strip().lower()
        seen[key] = seen.get(key, 0) + 1
        if seen[key] > 1:
            th["name"] = f"{th['name']} ({seen[key]})"


_NAME_PROMPT = (
    "Below are cross-layer THEMES in a reasoning model's response tokens. Each lists its member "
    "per-layer cluster labels, top tokens, and depth range (pos 0=start..1=end). For EACH theme "
    "give a concise UNIQUE name (2-5 words) for the reasoning step and a one-sentence description "
    "of what it does. Respond ONLY as JSON: {{\"themes\": [{{\"theme\": <id>, \"name\": \"...\", "
    "\"description\": \"...\"}}]}}.\n\n{blocks}"
)

_COHERENCE_PROMPT = (
    "For each THEME below, judge how well its name + description match its member labels and top "
    "tokens, as a coherence score from 0.0 (mismatch) to 1.0 (perfect). Respond ONLY as JSON: "
    "{{\"scores\": [{{\"theme\": <id>, \"coherence\": <float>}}]}}.\n\n{blocks}"
)


def _theme_blocks(themes: list[dict]) -> str:
    lines = []
    for th in themes:
        pos = f"pos {th['mean_pos']}" if th.get("mean_pos") is not None else "pos ?"
        labs = "; ".join(th.get("member_labels", [])[:6])
        top = ", ".join(th.get("top", [])[:10])
        lines.append(f"Theme {th['theme']} ({pos}, n={th['n_members']}): "
                     f"labels: {labs} | top: {top}")
    return "\n".join(lines)


def _name_themes(themes: list[dict], model: str, backend: str = "copilot") -> None:
    """Dedicated LLM naming pass over the FINAL themes (in place): a concise unique name + a
    one-line description per theme. Keeps member-label names on failure."""
    from ..core.llm import extract_json, run_copilot_cli, run_trapi

    prompt = _NAME_PROMPT.format(blocks=_theme_blocks(themes))
    try:
        raw = (
            run_trapi(prompt, model=model)
            if backend == "trapi"
            else run_copilot_cli(prompt, model=model)
        )
        data = extract_json(raw, require_key="themes")
    except (ValueError, KeyError, TypeError, RuntimeError) as e:
        print(f"[name-themes failed: {e}; keeping member-label names]")
        return
    by_id: dict[int, dict] = {}
    for t in data.get("themes", []):
        try:
            by_id[int(t["theme"])] = t
        except (KeyError, TypeError, ValueError):
            continue
    for th in themes:
        rec = by_id.get(th["theme"])
        if not rec:
            continue
        if str(rec.get("name", "")).strip():
            th["name"] = str(rec["name"]).strip()
        if str(rec.get("description", "")).strip():
            th["description"] = str(rec["description"]).strip()
    _dedup_theme_names(themes)


def _judge_theme_coherence(themes: list[dict], model: str) -> None:
    """LLM-judge each theme's name<->members coherence in place (0..1); no-op on failure."""
    from ..core.llm import extract_json, run_copilot_cli
    blocks = [f"Theme {th['theme']}: name='{th['name']}' desc='{th.get('description', '')}' | "
              f"labels: {'; '.join(th.get('member_labels', [])[:6])} | "
              f"top: {', '.join(th.get('top', [])[:8])}" for th in themes]
    try:
        data = extract_json(run_copilot_cli(_COHERENCE_PROMPT.format(blocks="\n".join(blocks)),
                                            model=model), require_key="scores")
    except (ValueError, KeyError, TypeError, RuntimeError) as e:
        print(f"[coherence judge failed: {e}]")
        return
    by_id: dict[int, float] = {}
    for s in data.get("scores", []):
        try:
            by_id[int(s["theme"])] = float(s["coherence"])
        except (KeyError, TypeError, ValueError):
            continue
    for th in themes:
        if th["theme"] in by_id:
            th["coherence"] = round(max(0.0, min(1.0, by_id[th["theme"]])), 3)


def _ami_figure(ami: np.ndarray, used: list[int], path: Path, slug: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(ami, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(used)))
    ax.set_xticklabels(used)
    ax.set_yticks(range(len(used)))
    ax.set_yticklabels(used)
    ax.set_xlabel("layer")
    ax.set_ylabel("layer")
    for i in range(len(used)):
        for j in range(len(used)):
            ax.text(j, i, f"{ami[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if ami[i, j] < 0.6 else "black")
    fig.colorbar(im, ax=ax, label="AMI between layers' meta labelings")
    ax.set_title(f"{slug}: cross-layer meta-cluster agreement")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _distinct_colors(n: int) -> list:
    """Up to 60 qualitative colors (tab20 + tab20b + tab20c), cycled beyond that."""
    palette: list = []
    for nm in ("tab20", "tab20b", "tab20c"):
        palette += list(plt.get_cmap(nm).colors)
    return [palette[i % len(palette)] for i in range(max(n, 0))]


_SINGLETON_GREY = (0.72, 0.72, 0.72, 1.0)


def _theme_colors(themes: list[dict]) -> tuple[list, list[int]]:
    """Per-theme RGBA: each MULTI-member (cross-layer) theme gets a distinct color, every SINGLETON
    (n_members==1, an unmerged layer-local primitive) shares one grey -- singletons stay drawn but
    read as a single visually-distinct class. Returns (colors_by_theme_id, merged_theme_ids)."""
    merged = [t for t, th in enumerate(themes) if th["n_members"] > 1]
    pal = _distinct_colors(len(merged))
    rank = {t: i for i, t in enumerate(merged)}
    colors = [tuple(pal[rank[t]]) if th["n_members"] > 1 else _SINGLETON_GREY
              for t, th in enumerate(themes)]
    return colors, merged


def _layer_theme_figure(layer_theme: np.ndarray, used: list[int], themes: list[dict], path: Path,
                        slug: str) -> None:
    n_th = len(themes)
    sing = [th["n_members"] == 1 for th in themes]
    fig, ax = plt.subplots(figsize=(min(1.3 * n_th + 3, 46), 0.62 * len(used) + 3))
    im = ax.imshow(layer_theme, cmap="magma", aspect="auto", vmin=0)
    ax.set_yticks(range(len(used)))
    ax.set_yticklabels([f"L{lay}" for lay in used], fontsize=11)
    ax.set_xticks(range(n_th))
    labels = [f"{t['theme']}: {t['name']}" + (" \u00b0" if s else "")
              for t, s in zip(themes, sing, strict=True)]
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=11)
    for lbl, s in zip(ax.get_xticklabels(), sing, strict=True):
        if s:
            lbl.set_color("0.45")
            lbl.set_style("italic")
    hi = layer_theme.max() * 0.6
    for i in range(len(used)):
        for j in range(n_th):
            ax.text(j, i, f"{layer_theme[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if layer_theme[i, j] < hi else "black")
    ax.set_ylabel("layer", fontsize=12)
    cb = fig.colorbar(im, ax=ax, label="fraction of layer's response tokens")
    cb.ax.tick_params(labelsize=10)
    ax.set_title(f"{slug}: how each layer's tokens split across cross-layer themes "
                 "(grey italic \u00b0 label = singleton, present in one layer)", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _theme_membership_figure(grid: np.ndarray, used: list[int], force_meta: int,
                             themes: list[dict], path: Path, slug: str) -> None:
    from matplotlib.colors import BoundaryNorm, ListedColormap
    n_themes = len(themes)
    colors, _merged = _theme_colors(themes)
    cmap = ListedColormap([colors[t] for t in range(n_themes)])
    norm = BoundaryNorm(np.arange(n_themes + 1) - 0.5, n_themes)
    fig, ax = plt.subplots(figsize=(0.7 * force_meta + 3, 0.5 * len(used) + 2))
    ax.imshow(grid, cmap=cmap, norm=norm, aspect="auto")
    ax.set_yticks(range(len(used)))
    ax.set_yticklabels([f"L{lay}" for lay in used])
    ax.set_xticks(range(force_meta))
    ax.set_xticklabels([f"m{m}" for m in range(force_meta)])
    for i in range(len(used)):
        for m in range(force_meta):
            t = int(grid[i, m])
            c = colors[t]
            lum = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
            ax.text(m, i, t, ha="center", va="center", fontsize=7,
                    color="white" if lum < 0.5 else "black")
    ax.set_xlabel("per-layer meta-cluster id")
    ax.set_ylabel("layer")
    ax.set_title(f"{slug}: which theme each layer's meta-cluster merged into (grey = singleton)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _merge_map_figure(per_layer: list[dict], grid: np.ndarray, themes: list[dict], force_meta: int,
                      used: list[int], path: Path, slug: str) -> None:
    """Crosswalk: every per-layer meta-cluster's ORIGINAL (independently-annotated) label + top
    tokens laid out at (layer, meta) and background-colored by the THEME it merged into (grey =
    singleton) -- shows all labels across layers and how the merge groups them into themes."""
    import textwrap

    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.lines import Line2D
    n_th = len(themes)
    n_lay = len(used)
    colors, merged = _theme_colors(themes)
    n_sing = n_th - len(merged)
    cmap = ListedColormap([colors[t] for t in range(n_th)])
    norm = BoundaryNorm(np.arange(n_th + 1) - 0.5, n_th)
    fig, ax = plt.subplots(figsize=(max(13.0, 3.3 * force_meta + 2), max(9.0, 1.75 * n_lay + 2)))
    ax.imshow(grid, cmap=cmap, norm=norm, aspect="auto")
    for i, pl in enumerate(per_layer):
        by_meta = {p["meta"]: p for p in pl["profiles"]}
        for m in range(force_meta):
            t = int(grid[i, m])
            lab = pl["ann"].get(m, {}).get("label", "") or f"m{m}"
            c = colors[t]
            tcol = "white" if 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2] < 0.5 else "black"
            lab_txt = "\n".join(textwrap.wrap(f"[T{t}] {lab}", 18)[:3])
            top_txt = " ".join(by_meta.get(m, {}).get("top", [])[:5])
            top_txt = "\n".join(textwrap.wrap(top_txt, 22)[:2])
            ax.text(m, i - 0.42, lab_txt, ha="center", va="top", fontsize=14, linespacing=0.95,
                    color=tcol)
            ax.text(m, i + 0.42, top_txt, ha="center", va="bottom", fontsize=11,
                    linespacing=0.95, color=tcol, style="italic", alpha=0.9)
    ax.set_xticks(range(force_meta))
    ax.set_xticklabels([f"m{m}" for m in range(force_meta)], fontsize=13)
    ax.set_yticks(range(n_lay))
    ax.set_yticklabels([f"L{lay}" for lay in used], fontsize=13)
    ax.set_xticks(np.arange(-0.5, force_meta, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_lay, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.0)
    ax.tick_params(which="minor", length=0)
    ax.set_xlabel("per-layer meta-cluster id", fontsize=13)
    ax.set_ylabel("layer", fontsize=13)
    handles = [Line2D([0], [0], marker="s", ls="", markersize=8, markerfacecolor=colors[t],
                      markeredgecolor="none", label=f"T{t}: {themes[t]['name'][:26]}")
               for t in merged]
    handles.append(Line2D([0], [0], marker="s", ls="", markersize=8,
                          markerfacecolor=_SINGLETON_GREY, markeredgecolor="none",
                          label=f"singleton x{n_sing} (1 layer)"))
    ax.legend(handles=handles, fontsize=12, loc="center left", bbox_to_anchor=(1.01, 0.5),
              frameon=False, title="merged theme", title_fontsize=13)
    cnt = f"{n_th} themes = {len(merged)} merged + {n_sing} singletons"
    ax.set_title(f"{slug}: original meta-cluster labels by (layer, meta), colored by merged "
                 f"theme \u2014 {cnt} (grey)", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_merge_findings(path: Path, used: list[int], layer_theme: np.ndarray, themes: list[dict],
                          per_layer: list[dict], method: str) -> None:
    n_th = len(themes)
    n_sing = sum(1 for th in themes if th["n_members"] == 1)
    n_merged = n_th - n_sing
    lines = [f"# Cross-layer meta-cluster merge (method={method})", "",
             f"Sampled layers: {used} \u00b7 {n_th} themes = {n_merged} merged + {n_sing} "
             "singletons", "",
             "## Themes (merged across layers)", ""]
    has_coh = any("coherence" in th for th in themes)
    cols = ["theme", "name", "#members", "mean_pos"]
    if has_coh:
        cols.append("coherence")
    cols += ["member labels", "top tokens"]
    lines += ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for th in themes:
        cells = [str(th["theme"]), th["name"], str(th["n_members"]), str(th["mean_pos"])]
        if has_coh:
            cells.append(str(th.get("coherence", "")))
        cells += ["; ".join(th["member_labels"][:4]), " ".join(th["top"][:10])]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "## Layer character: fraction of each layer's tokens per theme", "",
              "| layer | " + " | ".join(f"{th['theme']}:{th['name'][:16]}" for th in themes) + " |",
              "|" + "---|" * (n_th + 1)]
    for i, lay in enumerate(used):
        lines.append(f"| L{lay} | "
                     + " | ".join(f"{layer_theme[i, j]:.2f}" for j in range(n_th)) + " |")
    lines += ["", "## Figures", "",
              "Singletons (themes with one member -- a layer-local primitive that did not merge) "
              "stay plotted but are drawn in a shared grey (and flagged with a grey italic \u00b0 "
              "label in `layer_themes`); multi-layer themes get distinct colors.", "",
              "- `layer_themes.png` -- fraction of each layer's tokens per theme (layer x theme); "
              "singleton columns carry a grey italic \u00b0 label.",
              "- `theme_membership.png` -- which theme each (layer, meta) node merged into; "
              "multi-layer themes colored, singletons grey.",
              "- `merge_map.png` -- every per-layer meta-cluster's ORIGINAL label + top tokens, "
              "colored by the theme it merged into ([T#] tag; grey = singleton).",
              "- `ami_matrix.png` -- pairwise adjusted-MI between layers' cluster partitions.",
              "- `theme_depth_profile.png` -- each theme's token share vs depth (rise/fall lines).",
              "- `theme_nodes.png` -- every (layer, meta) node at position x layer, colored by "
              "merged theme (singletons grey), sized by tokens, with a theme legend "
              "(cross-layer analog of single-layer fine-states scatter).",
              "- `theme_transition.png` -- P(next-token theme | this theme) pooled over layers "
              "(temporal theme flow; cross-layer analog of the single-layer transition matrix).",
              "- `theme_flow.png` -- segment-level theme->theme flow with forward-fraction and "
              "monotonicity (cross-layer analog of the single-layer phase-flow figure).",
              "- `theme_graph.png` -- directed theme-transition GRAPH (network view): "
              "nodes = themes (grey = singleton) sized by token share, arrows = P(next theme).",
              "- `theme_trace_depth.png` -- 3 example traces as layer x token theme heatmaps "
              "(singletons grey, with a theme legend; cross-layer analog of trace-states).",
              "- `traces.html` -- interactive per-token theme coloring with layer/response/theme "
              "controls."]
    if method == "tokens":
        lines.append("- `theme_selection.png` -- theme count / singletons / largest-node-component "
                     "vs Jaccard merge threshold (cross-layer analog of the eigengap meta-count "
                     "heuristic; tokens method only).")
    lines += ["", "## Per-layer meta-clusters (annotated independently)", ""]
    for pl in per_layer:
        lines.append(f"### L{pl['layer']}")
        for p in pl["profiles"]:
            ann = pl["ann"].get(p["meta"], {})
            lines.append(f"- m{p['meta']} **{ann.get('label', '')}** (pos {p['mean_pos']}): "
                         f"{' '.join(p['top'][:10])}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _theme_depth_profile_figure(layer_theme: np.ndarray, themes: list[dict], used: list[int],
                                path: Path, slug: str) -> None:
    """Line plot of each prominent theme's token-fraction vs depth -- a clearer trend view than the
    layer x theme heatmap (which theme grows/shrinks/peaks with depth)."""
    order = sorted(range(len(themes)), key=lambda t: -layer_theme[:, t].max())
    shown = [t for t in order if layer_theme[:, t].max() >= 0.05][:14]
    fig, ax = plt.subplots(figsize=(11, 6))
    for t in shown:
        ax.plot(used, layer_theme[:, t], marker="o", markersize=4,
                label=f"{themes[t]['theme']}: {themes[t]['name'][:30]}")
    ax.set_xlabel("layer")
    ax.set_ylabel("fraction of layer's response tokens")
    ax.set_title(f"{slug}: theme prevalence vs depth")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _passage_tokens(run_dirs: list[Path], slug: str, tasks: list[str],
                    ref_layer: int) -> list[list]:
    """Per-passage response token strings in the SAME order/filtering as the clustering (so they
    align with the cached token stack columns), read from meta.json (no model / npy load)."""
    out: list[list] = []
    for run in run_dirs:
        adir = run / "activations" / slug
        if not adir.exists():
            continue
        for tsk in tasks:
            for pdir in sorted(adir.glob(f"*__{tsk}")):
                if not ((pdir / "meta.json").exists()
                        and (pdir / f"layer_{ref_layer}.npy").exists()):
                    continue
                meta = json.loads((pdir / "meta.json").read_text(encoding="utf-8"))
                toks = meta.get("tokens") or []
                sel = _scope_token_indices(meta, "response", toks)
                if len(sel) < 3:
                    continue
                idx = [j for j, _ in sorted(sel, key=lambda t: t[0])]
                out.append([_clean(toks[i]) if i < len(toks) else "" for i in idx])
    return out


def _passage_ids(run_dirs: list[Path], slug: str, tasks: list[str], ref_layer: int) -> list[dict]:
    """Provenance of each passage in :func:`_passage_tokens` order (same filtering, so index i
    here is index i there): the run dir, scenario id, task and schema behind every passage."""
    out: list[dict] = []
    for run in run_dirs:
        adir = run / "activations" / slug
        if not adir.exists():
            continue
        for tsk in tasks:
            for pdir in sorted(adir.glob(f"*__{tsk}")):
                if not (
                    (pdir / "meta.json").exists() and (pdir / f"layer_{ref_layer}.npy").exists()
                ):
                    continue
                meta = json.loads((pdir / "meta.json").read_text(encoding="utf-8"))
                if len(_scope_token_indices(meta, "response", meta.get("tokens") or [])) < 3:
                    continue
                out.append(
                    {
                        "run": run.as_posix(),
                        "scenario_id": meta.get("scenario_id"),
                        "task": meta.get("task"),
                        "schema": meta.get("schema"),
                    }
                )
    return out


def _trace_sample(passages: list[list], n_show: int = 12, max_tok: int = 2000) -> dict:
    """Length-stratified sample of passages, each with its start offset into the token stack."""
    lens = [len(t) for t in passages]
    offs = np.concatenate([[0], np.cumsum(lens)])
    order = list(np.argsort(lens))
    if len(order) <= n_show:
        pick = order
    else:
        sel = sorted(set(np.linspace(0, len(order) - 1, n_show).round().astype(int).tolist()))
        pick = [int(order[s]) for s in sel]
    return {"passages": [{"start": int(offs[i]), "tokens": passages[i][:max_tok]} for i in pick]}


_TRACE_JS = (
    "<script>"
    "function render(){var p=DATA[P],h='';for(var i=0;i<p.tok.length;i++){var t=p.tok[i],"
    "th=p.th[i][Lx];if(t.trim()===''){h+='<br>';continue;}var d=(H!==null&&th!==H)?' dim':'';"
    "h+=\"<span class='t\"+d+\"' style='background:\"+COLORS[th]+\"' title='\"+"
    "NAMES[th].replace(/'/g,'')+\"'>\"+t+\"</span>\";}"
    "document.getElementById('resp').innerHTML=h;mk('lb',Lx);mk('rb',P);}"
    "function selL(i){Lx=i;render();}function selR(i){P=i;render();}"
    "function selH(h){H=h;mk('hb',h===null?'all':h);document.getElementById('info').textContent="
    "(h===null)?'':('theme '+h+': '+NAMES[h]);render();}"
    "function mk(c,v){var b=document.querySelectorAll('.'+c);for(var k=0;k<b.length;k++)"
    "b[k].classList.remove('active');var e=document.getElementById(c+v);"
    "if(e)e.classList.add('active');}"
    "</script>"
)
_TRACE_CSS = (
    "<style>body{font-family:system-ui,sans-serif;max-width:1150px;margin:16px auto;color:#222}"
    ".ctl{position:sticky;top:0;background:#fff;padding:8px 0;border-bottom:1px solid #eee;"
    "z-index:9}"
    ".row{margin:5px 0;display:flex;gap:4px;flex-wrap:wrap;align-items:center}"
    ".row .lab{font-weight:bold;color:#555;font-size:13px;margin-right:4px}"
    "button{cursor:pointer;border-radius:5px;border:1px solid #bbb;padding:3px 8px;font-size:12px;"
    "background:#f5f5f5;color:#222}button.active{outline:2px solid #333}"
    ".resp{white-space:pre-wrap;line-height:2.0;font-size:14px;border:1px solid #e0e0e0;"
    "padding:12px;"
    "border-radius:6px;margin-top:8px}.t{border-radius:3px;padding:0 1px}.t.dim{opacity:.12}"
    "#info{font-size:13px;color:#555;margin:6px 0;min-height:16px}"
    ".hb{max-width:240px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}</style>"
)


def _trace_html(stack: np.ndarray, grid: np.ndarray, themes: list[dict], sample: dict,
                used: list[int], path: Path, slug: str, method: str) -> None:
    """Interactive response-trace viewer: sampled CoT responses with each token background-colored
    by its cross-layer THEME, plus a LAYER selector to watch the theme assignment change across
    depth, a response selector, per-theme highlight, and hover tooltips. Self-contained."""
    import html as _html

    import matplotlib.colors as mcolors
    n_th = len(themes)
    colors_rgba, merged = _theme_colors(themes)
    n_sing = n_th - len(merged)
    colors = [mcolors.to_hex(c) for c in colors_rgba]
    txt_colors = []
    for c in colors:
        r, g, b = mcolors.to_rgb(c)
        txt_colors.append("#000" if 0.299 * r + 0.587 * g + 0.114 * b > 0.6 else "#fff")
    names = [f"{t['theme']}: {t['name']}" + ("" if t["n_members"] > 1 else " (singleton)")
             for t in themes]
    n_layers = len(used)
    data = []
    for p in sample["passages"]:
        lo, toks = p["start"], p["tokens"]
        metas = stack[:, lo:lo + len(toks)]
        th = grid[np.arange(n_layers)[:, None], metas].T
        data.append({"tok": [_html.escape(t) for t in toks], "th": th.astype(int).tolist()})
    payload = ("<script>const COLORS=" + json.dumps(colors) + ",NAMES=" + json.dumps(names)
               + ",LAYERS=" + json.dumps(list(used)) + ",DATA=" + json.dumps(data)
               + ";let P=0,Lx=" + str(n_layers // 2) + ",H=null;</script>")
    layer_row = ("<div class='row'><span class='lab'>Layer:</span>"
                 + "".join(f"<button id='lb{i}' class='lb' onclick='selL({i})'>L{lay}</button>"
                           for i, lay in enumerate(used)) + "</div>")
    resp_row = ("<div class='row'><span class='lab'>Response:</span>"
                + "".join(f"<button id='rb{i}' class='rb' onclick='selR({i})'>{i + 1}</button>"
                          for i in range(len(data))) + "</div>")
    theme_row = ("<div class='row'><span class='lab'>Highlight theme:</span>"
                 "<button id='hball' class='hb active' onclick='selH(null)'>All</button>"
                 + "".join(f"<button id='hb{t}' class='hb' onclick='selH({t})' "
                           f"style='background:{colors[t]};color:{txt_colors[t]};"
                           f"border-color:{colors[t]}'>"
                           f"{_html.escape(names[t])}</button>" for t in range(n_th)) + "</div>")
    header = (f"<h2>{_html.escape(slug)} \u2014 response-trace theme viewer ({method}, {n_th} "
              f"themes = {len(merged)} merged + {n_sing} singletons)</h2>"
              "<p>Each token is colored by its cross-layer <b>theme</b> (grey = "
              "singleton: a layer-local meta-cluster that did not merge). Pick a <b>layer</b> to "
              "see the theme assignment at that depth, switch <b>response</b>, or click a theme to "
              "highlight it; hover a token for its theme.</p>")
    parts = ["<!doctype html><html><head><meta charset='utf-8'><title>",
             _html.escape(slug), " traces</title>", _TRACE_CSS, payload, _TRACE_JS,
             "</head><body>", header, "<div class='ctl'>", layer_row, resp_row, theme_row,
             "</div><div id='info'></div><div class='resp' id='resp'></div>",
             "<script>render();</script></body></html>"]
    path.write_text("".join(parts), encoding="utf-8")


def _theme_order(themes: list[dict]) -> tuple[list[int], np.ndarray]:
    """Themes sorted by mean token position -> (position order, per-theme rank) so transition/flow
    matrices read as a temporal progression and 'forward' means advancing in position."""
    pos = [t["mean_pos"] if t["mean_pos"] is not None else 1.0 for t in themes]
    order = sorted(range(len(themes)), key=lambda t: pos[t])
    rank = np.zeros(len(themes), dtype=int)
    for r, t in enumerate(order):
        rank[t] = r
    return order, rank


def _theme_axis_labels(themes: list[dict], order: list[int]) -> list[str]:
    return [f"{themes[o]['theme']}: {themes[o]['name'][:20]}" for o in order]


def _theme_transition_figure(stack: np.ndarray, grid: np.ndarray, themes: list[dict],
                             plens: list[int], path: Path, slug: str) -> None:
    """Temporal theme-transition matrix P(next-token theme | this theme), pooled over all layers'
    per-token theme sequences (within passages), themes position-ordered. Cross-layer analog of the
    single-layer s3_transition."""
    n_th = len(themes)
    order, rank = _theme_order(themes)
    bounds = np.cumsum([0] + plens)
    valid = np.ones(int(bounds[-1]), dtype=bool)
    valid[bounds[1:] - 1] = False
    idx = np.where(valid)[0]
    trans = np.zeros((n_th, n_th))
    for lay in range(grid.shape[0]):
        th = rank[grid[lay, stack[lay]]]
        np.add.at(trans, (th[idx], th[idx + 1]), 1)
    trans = trans / (trans.sum(1, keepdims=True) + 1e-9)
    names = _theme_axis_labels(themes, order)
    fig, ax = plt.subplots(figsize=(max(8.0, 0.62 * n_th + 4), max(7.0, 0.62 * n_th + 3)))
    im = ax.imshow(trans, cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(n_th), names, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(n_th), names, fontsize=10)
    ax.set_xlabel("to theme (next token)", fontsize=12)
    ax.set_ylabel("from theme", fontsize=12)
    cb = fig.colorbar(im, ax=ax, label="P(next-token theme | this theme), pooled over layers")
    cb.ax.tick_params(labelsize=10)
    ax.set_title(f"{slug}: temporal theme transition (position-ordered)", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _theme_flow_figure(stack: np.ndarray, grid: np.ndarray, themes: list[dict], plens: list[int],
                       seed: int, path: Path, slug: str) -> None:
    """Segment-level theme flow (collapsing same-theme runs) + order-stereotypy (forward-frac,
    monotonicity vs shuffled), over all passage x layer theme sequences. Cross-layer analog of the
    single-layer s3_flow."""
    n_th = len(themes)
    order, rank = _theme_order(themes)
    bounds = np.cumsum([0] + plens)
    seqs = []
    for lay in range(grid.shape[0]):
        th = rank[grid[lay, stack[lay]]]
        for p in range(len(plens)):
            seg = th[bounds[p]:bounds[p + 1]]
            if len(seg) >= 2:
                seqs.append(seg)
    flow = _phase_flow(seqs, n_th, seed)
    trans = np.asarray(flow["seg_transition"])
    names = _theme_axis_labels(themes, order)
    fig, ax = plt.subplots(figsize=(max(8.0, 0.62 * n_th + 4), max(7.0, 0.62 * n_th + 3)))
    im = ax.imshow(trans, cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(n_th), names, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(n_th), names, fontsize=10)
    ax.set_xlabel("to theme (next segment)", fontsize=12)
    ax.set_ylabel("from theme", fontsize=12)
    cb = fig.colorbar(im, ax=ax, label="P(next-segment theme | current)")
    cb.ax.tick_params(labelsize=10)
    ax.set_title(f"{slug}: theme segment flow (position-ordered) \u2014 forward-frac "
                 f"{flow['forward_frac']} | monotonicity {flow['monotonicity']} "
                 f"(shuffled {flow['monotonicity_shuffled']})", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _theme_graph_figure(stack: np.ndarray, grid: np.ndarray, themes: list[dict], plens: list[int],
                        seed: int, path: Path, slug: str) -> None:
    """Directed cross-layer theme-transition GRAPH (network form of theme_flow): nodes = themes
    (colored by _theme_colors, grey = singleton), sized by total token share; curved weighted arrows
    = segment-level P(next theme | current) pooled over layers, colored by source. Each node keeps
    its strongest outgoing edge. Analog of the single-layer primitive-transition graph."""
    import networkx as nx
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch
    n_th = len(themes)
    colors, merged = _theme_colors(themes)
    n_sing = n_th - len(merged)
    bounds = np.cumsum([0] + plens)
    seqs, tok = [], np.zeros(n_th)
    for lay in range(grid.shape[0]):
        th = grid[lay, stack[lay]]
        np.add.at(tok, th, 1)
        for p in range(len(plens)):
            seg = th[bounds[p]:bounds[p + 1]]
            if len(seg) >= 2:
                seqs.append(seg)
    flow = _phase_flow(seqs, n_th, seed)
    trans = np.asarray(flow["seg_transition"])
    share = tok / max(tok.sum(), 1)
    thr = 0.08
    edges: dict[tuple[int, int], float] = {}
    for i in range(n_th):
        for j in range(n_th):
            if i != j and trans[i, j] >= thr:
                edges[(i, j)] = float(trans[i, j])
        row = [(j, trans[i, j]) for j in range(n_th) if j != i]
        if row:
            j, w = max(row, key=lambda t: t[1])
            if w > 0:
                edges[(i, j)] = float(w)
    g = nx.DiGraph()
    g.add_nodes_from(range(n_th))
    for (i, j), w in edges.items():
        g.add_edge(i, j, weight=w)
    pos = nx.spring_layout(g, seed=seed, weight="weight", k=2.8 / np.sqrt(max(n_th, 1)))
    fig, ax = plt.subplots(figsize=(15, 12))
    for (i, j), w in edges.items():
        ax.add_patch(FancyArrowPatch(
            pos[i], pos[j], connectionstyle="arc3,rad=0.16", arrowstyle="-|>",
            mutation_scale=12 + 26 * w, lw=0.8 + 5.5 * w, color=colors[i],
            alpha=float(min(0.9, 0.3 + w)), shrinkA=16, shrinkB=18, zorder=2))
    for t in range(n_th):
        ax.scatter(*pos[t], s=350 + 8000 * share[t], color=colors[t], edgecolors="k",
                   linewidths=1.3, zorder=3)
        ax.text(pos[t][0], pos[t][1], str(t), ha="center", va="center", fontsize=11,
                weight="bold", zorder=4)
    handles = [Line2D([0], [0], marker="o", ls="", markersize=10, color=colors[t],
                      label=f"T{t}: {themes[t]['name'][:26]} ({share[t] * 100:.0f}%)")
               for t in merged]
    handles.append(Line2D([0], [0], marker="o", ls="", markersize=10, color=_SINGLETON_GREY,
                          label=f"singletons x{n_sing} (grey)"))
    ax.legend(handles=handles, fontsize=9, loc="center left", bbox_to_anchor=(1.0, 0.5),
              frameon=False, title="theme")
    ax.set_axis_off()
    ff, mo, ms = flow["forward_frac"], flow["monotonicity"], flow["monotonicity_shuffled"]
    ax.set_title(f"{slug}: cross-layer theme transition graph \u2014 {n_th} themes = "
                 f"{len(merged)} merged + {n_sing} singletons\n"
                 f"forward-frac {ff} | monotonicity {mo} (shuffled {ms})", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _theme_nodes_figure(per_layer: list[dict], grid: np.ndarray, sizes: np.ndarray,
                        themes: list[dict], path: Path, slug: str) -> None:
    """Every per-layer meta-cluster NODE plotted at (mean position, layer), colored by its merged
    theme (singletons share a grey), sized by #tokens, annotated with its top tokens. Cross-layer
    analog of s2_fine_states (nodes are the 'fine states'; themes are the 'meta-clusters')."""
    from matplotlib.lines import Line2D
    n_th = len(themes)
    colors, merged = _theme_colors(themes)
    n_sing = n_th - len(merged)
    fig, ax = plt.subplots(figsize=(18, max(7.0, 0.75 * len(per_layer) + 2.5)))
    for i, pl in enumerate(per_layer):
        for p in pl["profiles"]:
            t = int(grid[i, p["meta"]])
            single = themes[t]["n_members"] == 1
            ax.scatter(p["mean_pos"], pl["layer"], s=25 + np.sqrt(sizes[i, p["meta"]]) * 2.0,
                       color=colors[t], alpha=0.5 if single else 0.85, edgecolors="none")
            ax.text(p["mean_pos"] + 0.004, pl["layer"], " ".join(p["top"][:2]), fontsize=8,
                    va="center", color="0.4" if single else "black")
    handles = [Line2D([0], [0], marker="o", ls="", markersize=9, markerfacecolor=colors[t],
                      markeredgecolor="none", label=f"{t}: {themes[t]['name'][:22]}")
               for t in merged]
    handles.append(Line2D([0], [0], marker="o", ls="", markersize=9,
                          markerfacecolor=_SINGLETON_GREY, markeredgecolor="none",
                          label=f"singleton x{n_sing} (1 layer)"))
    ax.legend(handles=handles, fontsize=9, loc="center left", bbox_to_anchor=(1.01, 0.5),
              frameon=False, title="cross-layer theme", title_fontsize=10)
    ax.set_xlabel("mean token position", fontsize=12)
    ax.set_ylabel("layer", fontsize=12)
    ax.tick_params(labelsize=11)
    ax.grid(alpha=0.2)
    ax.set_title(f"{slug}: meta-cluster nodes by position x depth \u2014 color = merged theme, "
                 f"grey = singleton ({n_th} themes = {len(merged)} merged + {n_sing} singletons)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _theme_selection_figure(stack: np.ndarray, sizes: np.ndarray, force_meta: int,
                            threshold: float, path: Path, slug: str) -> None:
    """Theme count / singletons / largest-component vs the Jaccard merge threshold -- the
    data-driven granularity selector; cross-layer analog of the single-layer eigengap heuristic."""
    from scipy.sparse.csgraph import connected_components
    n_layers = stack.shape[0]
    n_nodes = n_layers * force_meta
    jac = np.zeros((n_nodes, n_nodes))
    for i in range(n_layers):
        for j in range(i + 1, n_layers):
            cont = _contingency(stack[i], stack[j], force_meta)
            for a in range(force_meta):
                for b in range(force_meta):
                    union = sizes[i, a] + sizes[j, b] - cont[a, b]
                    if union > 0:
                        v = cont[a, b] / union
                        jac[i * force_meta + a, j * force_meta + b] = v
                        jac[j * force_meta + b, i * force_meta + a] = v
    thrs = np.linspace(0.1, 0.7, 25)
    n_theme, n_sing, n_merged, largest = [], [], [], []
    for thr in thrs:
        _, lab = connected_components(jac >= thr, directed=False)
        bc = np.bincount(lab)
        n_theme.append(len(bc))
        n_sing.append(int((bc == 1).sum()))
        n_merged.append(int((bc >= 2).sum()))
        largest.append(int(bc.max()))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thrs, n_theme, "o-", color="black", label="# themes total (= merged + singletons)")
    ax.plot(thrs, n_merged, "D-", color="tab:blue", alpha=0.7,
            label="# merged themes (\u22652 nodes)")
    ax.plot(thrs, n_sing, "s-", color="tab:orange", alpha=0.8, label="# singletons (size-1)")
    ax.plot(thrs, largest, "^-", color="tab:green", alpha=0.8, label="largest theme (# nodes)")
    ax.axvline(threshold, ls="--", color="red", lw=1.3, label=f"used thr={threshold}")
    ax.set_xlabel("Jaccard merge threshold")
    ax.set_ylabel("count / size (nodes)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(f"{slug}: theme granularity vs merge threshold\n(cross-layer analog of the "
                 "single-layer eigengap meta-count heuristic)")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _cpm_selection_figure(sweep: list[dict], chosen_gamma: float, stability_floor: float,
                          path: Path, slug: str, objective: str = "modularity") -> None:
    """leiden-cpm analog of ``theme_selection``: theme count / singletons / largest theme plus the
    selection scores -- multi-seed consensus stability, partition modularity, and (embedding
    backend) silhouette -- vs the CPM resolution gamma, with the chosen operating point marked.
    Turns the (previously fixed, magic) gamma into an inspectable choice: the auto rule takes the
    gamma with the best ``objective`` (modularity, or silhouette for embeddings) among those whose
    stability clears ``stability_floor`` (best-structured reproducible scale, not coarsest)."""
    gammas = [s["gamma"] for s in sweep]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(gammas, [s["n_themes"] for s in sweep], "o-", color="black",
            label="# themes total (merged + singletons)")
    ax.plot(gammas, [s["n_singletons"] for s in sweep], "s-", color="tab:orange", alpha=0.8,
            label="# singletons (size-1)")
    ax.plot(gammas, [s["largest"] for s in sweep], "^-", color="tab:green", alpha=0.8,
            label="largest theme (# nodes)")
    ax.axvline(chosen_gamma, ls="--", color="red", lw=1.3, label=f"used \u03b3={chosen_gamma}")
    ax.set_xlabel("CPM resolution \u03b3 (internal-density threshold; higher = finer)")
    ax.set_ylabel("count / size (nodes)")
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(gammas, [s["stability"] for s in sweep], "d:", color="tab:purple", alpha=0.85,
             label="multi-seed stability (mean AMI)")
    ax2.axhline(stability_floor, ls=":", color="tab:purple", alpha=0.5, lw=1.0)
    mod_lbl = "modularity" + (" (selection score)" if objective == "modularity" else "")
    ax2.plot(gammas, [s.get("modularity", 0.0) for s in sweep], "v-", color="tab:blue", alpha=0.85,
             label=mod_lbl)
    ylab = "stability (mean AMI)  /  modularity"
    if any("silhouette" in s for s in sweep):
        ax2.plot(gammas, [s.get("silhouette", 0.0) for s in sweep], "*-", color="tab:red",
                 alpha=0.85, label="silhouette (selection score)")
        ylab += "  /  silhouette"
    ax2.set_ylabel(ylab)
    ax2.set_ylim(-0.1, 1.02)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")
    ax.set_title(f"{slug}: CPM theme granularity + reproducibility vs resolution \u03b3\n"
                 f"(operating \u03b3 = best {objective} among \u03b3 with stability \u2265 floor)")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _theme_trace_depth_figure(stack: np.ndarray, grid: np.ndarray, themes: list[dict],
                              plens: list[int], used: list[int], path: Path, slug: str) -> None:
    """Three representative traces (shortest/median/longest) as layer x token heatmaps, each token
    colored by its theme (singletons share a grey). Cross-layer analog of the single-layer
    trace_states rasters."""
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch
    n_th = len(themes)
    colors, merged = _theme_colors(themes)
    n_sing = n_th - len(merged)
    tcmap = ListedColormap([colors[t] for t in range(max(n_th, 1))])
    norm = BoundaryNorm(np.arange(max(n_th, 1) + 1) - 0.5, max(n_th, 1))
    bounds = np.cumsum([0] + plens)
    chosen = _short_med_long(plens)
    names = (["shortest", "median", "longest"][:len(chosen)] if len(chosen) == 3
             else [f"trace {i + 1}" for i in range(len(chosen))])
    fig, axes = plt.subplots(len(chosen), 1, figsize=(13, 2.2 * len(chosen) + 1.2), squeeze=False)
    for ax, pi, nm in zip(axes[:, 0], chosen, names, strict=False):
        s, e = int(bounds[pi]), int(bounds[pi + 1])
        cols = stack[:, s:e][:, :2500]
        block = grid[np.arange(grid.shape[0])[:, None], cols]
        ax.imshow(block, aspect="auto", cmap=tcmap, norm=norm, interpolation="nearest")
        ax.set_yticks(range(len(used)))
        ax.set_yticklabels([f"L{lay}" for lay in used], fontsize=6)
        ax.set_ylabel("layer")
        ax.set_title(f"{nm} trace ({e - s} tokens)", fontsize=9, loc="left")
    axes[-1, 0].set_xlabel("token index")
    handles = [Patch(facecolor=colors[t], label=f"{t}: {themes[t]['name'][:22]}") for t in merged]
    handles.append(Patch(facecolor=_SINGLETON_GREY, label=f"singleton x{n_sing} (1 layer)"))
    fig.legend(handles=handles, fontsize=6, loc="center left", bbox_to_anchor=(1.0, 0.5),
               frameon=False, title="theme")
    fig.suptitle(f"{slug}: theme assignment across depth along example traces "
                 "(color = merged theme, grey = singleton)")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def run(
    run_dirs: list[Path],
    slug: str,
    layers: list[int],
    k: int,
    out_dir: Path,
    mode: str,
    tasks: list[str],
    seed: int = 0,
    force_meta: int = 8,
    pca: float | int | None = None,
    threshold: float = 0.2,
    method: str = "semantic-avglink",
    annotate: bool = False,
    annotate_model: str = DEFAULT_ANNOTATE_MODEL,
    resolution: float | None = None,
    merge_k: int | None = None,
    consensus_runs: int = 3,
    stability_floor: float = 0.9,
    merge_knn: int = 6,
    merge_seed: str = "avglink",
    name_themes: bool = False,
    judge_coherence: bool = False,
    distinctive: bool = False,
    embed_model: str | None = None,
    annotate_backend: str = "copilot",
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict = {}
    if mode in ("diagnostic", "both", "all"):
        result["diagnostic"] = diagnostic(run_dirs, slug, layers, out_dir, tasks, seed)
    if mode in ("sweep", "both", "all"):
        result["sweep"] = sweep(run_dirs, slug, layers, k, out_dir, tasks, seed, force_meta,
                                pca=pca)
    if mode in ("merge", "all"):
        result["merge"] = merge(
            run_dirs,
            slug,
            layers,
            k,
            out_dir,
            tasks,
            seed,
            force_meta,
            pca=pca,
            threshold=threshold,
            method=method,
            annotate=annotate,
            annotate_model=annotate_model,
            resolution=resolution,
            merge_k=merge_k,
            consensus_runs=consensus_runs,
            stability_floor=stability_floor,
            merge_knn=merge_knn,
            merge_seed=merge_seed,
            name_themes=name_themes,
            judge_coherence=judge_coherence,
            distinctive=distinctive,
            embed_model=embed_model,
            annotate_backend=annotate_backend,
        )
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="depth-resolved response-latent clustering")
    p.add_argument("--run", required=True, nargs="+", help="run dir(s) with activations/<slug>/")
    p.add_argument("--slug", required=True, help="model activation slug (e.g. Phi-4-reasoning)")
    p.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS)
    p.add_argument("--k", type=int, default=30, help="number of k-means token states")
    p.add_argument("--meta", type=int, default=8, help="force this many meta-clusters (primitives)")
    p.add_argument("--tasks", nargs="+", default=["identify"], help="capture task suffix(es)")
    p.add_argument("--mode", default="both",
                   choices=["diagnostic", "sweep", "merge", "both", "all"])
    p.add_argument("--pca", default="none",
                   help="clustering feature space: 'none' (full z-scored, default), a variance "
                        "fraction in (0,1) e.g. 0.9, or an int #components")
    p.add_argument("--merge-threshold", type=float, default=0.2,
                   help="merge mode (tokens): merge nodes whose Jaccard token-overlap >= this "
                        "(data-driven # themes; distinct clusters stay separate)")
    p.add_argument("--merge", default="semantic-avglink",
                   choices=["semantic-avglink", "semantic-cpm", "labels", "avglink", "leiden-cpm",
                            "tokens", "llm-refine"],
                   help="merge. GO-FORWARD (need --annotate): semantic-avglink (DEFAULT "
                        "-- average-linkage on the DENSE semantic cosine of label+description, "
                        "silhouette-cut, no k-NN/gamma) | semantic-cpm (same cosine through a k-NN "
                        "Leiden-CPM consensus -- coarser, best worst-case balance) | labels (one "
                        "live LLM grouping call). Semantic backend = instruction EMBEDDINGS by "
                        "default (--merge-embed; --no-merge-embed = CPU TF-IDF fallback). RETIRED "
                        "token-graph methods (kept for comparison): avglink | leiden-cpm | "
                        "tokens (single-linkage @ --merge-threshold). Also llm-refine "
                        "(geometry-seeded LLM split/merge).")
    p.add_argument("--merge-resolution", type=float, default=None,
                   help="leiden-cpm CPM resolution gamma (density threshold); OMIT to auto-select "
                        "the coarsest reproducible scale (multi-seed consensus stability >= "
                        "floor). Ignored when --merge-k is given.")
    p.add_argument("--merge-stability-floor", type=float, default=0.9,
                   help="leiden-cpm auto-gamma: min multi-seed consensus stability (mean pairwise "
                        "AMI) the chosen resolution must clear; the coarsest such gamma is used")
    p.add_argument("--merge-knn", type=int, default=6,
                   help="semantic-cpm: keep this many nearest semantic-cosine neighbours per node "
                        "when sparsifying the annotation graph (6 = de-concentrates best; higher "
                        "merges more)")
    p.add_argument("--merge-embed", action=argparse.BooleanOptionalAction, default=True,
                   help="semantic-cpm / semantic-avglink: build the semantic graph from neural "
                        "instruction EMBEDDINGS (--merge-embed-model) of label+description "
                        "(DEFAULT). --no-merge-embed falls back to the CPU-only TF-IDF cosine (the "
                        "dependency-free path; sound with semantic-cpm, but TF-IDF makes "
                        "semantic-avglink over-split). Embeddings need the `gpu` extra "
                        "(torch+transformers) + the model.")
    p.add_argument("--merge-embed-model", default="Qwen/Qwen3-Embedding-4B",
                   help="embedding model for --merge-embed (default Qwen/Qwen3-Embedding-4B)")
    p.add_argument("--merge-k", type=int, default=None,
                   help="target theme count: avglink cuts the dendrogram here, leiden-cpm scans "
                        "resolution (multi-seed consensus) to hit it (default: data-driven "
                        "gap-cut / auto-gamma)")
    p.add_argument("--merge-consensus", type=int, default=3,
                   help="labels merge: repeat the LLM call this many times and return the "
                        "consensus partition + a stability score (1 = single call, no consensus)")
    p.add_argument("--merge-seed", default="avglink", choices=["avglink", "leiden-cpm"],
                   help="llm-refine: the geometric method that proposes the candidate themes")
    p.add_argument("--name-themes", action="store_true",
                   help="dedicated LLM naming pass: unique name + description per final theme")
    p.add_argument("--judge-coherence", action="store_true",
                   help="with --name-themes: LLM-judge each theme's name/members coherence (0..1)")
    p.add_argument("--distinctive", action="store_true",
                   help="add tf-idf 'distinctive' tokens (characteristic vs other clusters) to the "
                        "annotation + labels-merge prompts; separate distinctive-keyed cache")
    p.add_argument("--annotate", action="store_true",
                   help="merge mode: LLM-annotate each layer's meta-clusters (to name themes)")
    p.add_argument("--annotate-model", default=DEFAULT_ANNOTATE_MODEL, help="annotation model")
    p.add_argument(
        "--annotate-backend",
        default="copilot",
        choices=["copilot", "trapi"],
        help="annotation transport: copilot = a Copilot CLI agent session (minutes per "
        "layer); trapi = a direct Azure OpenAI chat-completion (seconds per layer; "
        "needs `az login --scope api://trapi/.default` + the `api` extra, and "
        "--annotate-model is then the TRAPI deployment, e.g. "
        "gpt-5.6-sol_2026-07-09). The annotator is the cross-source common "
        "currency, so a non-default backend/model gets its own clusters cache.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, help="output dir for figures / json / findings")
    args = p.parse_args()
    run(
        [Path(r) for r in args.run],
        args.slug,
        args.layers,
        args.k,
        Path(args.out),
        args.mode,
        args.tasks,
        seed=args.seed,
        force_meta=args.meta,
        pca=_parse_pca(args.pca),
        threshold=args.merge_threshold,
        method=args.merge,
        annotate=args.annotate,
        annotate_model=args.annotate_model,
        resolution=args.merge_resolution,
        merge_k=args.merge_k,
        consensus_runs=args.merge_consensus,
        stability_floor=args.merge_stability_floor,
        merge_knn=args.merge_knn,
        merge_seed=args.merge_seed,
        name_themes=args.name_themes,
        judge_coherence=args.judge_coherence,
        distinctive=args.distinctive,
        embed_model=(args.merge_embed_model if args.merge_embed else None),
        annotate_backend=args.annotate_backend,
    )


if __name__ == "__main__":
    main()
