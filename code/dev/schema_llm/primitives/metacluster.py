"""Temporal meta-clustering of scene-token trajectories (SCHEMA_LATENT_GEOMETRY_PLAN S2/S3).

Our own re-implementation of the algotrace ``KMeansTemporalMetaClustering`` idea (the reference in
``code/reference/`` is read-only spec, never imported): a two-level clustering of per-token residual
trajectories -- k-means into a token-state vocabulary, then group states into **meta-clusters** by
their *temporal* transition structure (spectral clustering on the mean consecutive-token transition
matrix; meta-count via an eigengap heuristic; meta-clusters ordered by mean position).

Applied to our identify **stimulus scene-token** sequences to test:
- **S2**: do recovered meta-clusters align with the native scenes? (AMI / purity vs a token-identity
  baseline).
- **S3**: does the meta-cluster transition graph recover the schema's scene ORDER?

Example::

    python -m dev.schema_llm.primitives.metacluster --run <dirs> --slug phi-4 --layer 17 --k 30 \
        --out results/schema_llm/analysis/identify/20260715-metacluster_phi-4_216
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.preprocessing import StandardScaler

from ._cluster import (
    _ANNOT_PROMPT as _ANNOT_PROMPT,
)
from ._cluster import (
    _DELIB as _DELIB,
)
from ._cluster import (
    _INSTR as _INSTR,
)
from ._cluster import (
    _SCOPE_CTX as _SCOPE_CTX,
)
from ._cluster import (
    _STIM_CTX as _STIM_CTX,
)
from ._cluster import (
    META_TRANSITION_SUBSTOCHASTIC_NOTE as META_TRANSITION_SUBSTOCHASTIC_NOTE,
)
from ._cluster import (
    KMeansTemporalMetaClustering as KMeansTemporalMetaClustering,
)
from ._cluster import (
    MetaConfig as MetaConfig,
)
from ._cluster import (
    _annotate as _annotate,
)
from ._cluster import (
    _annotate_schema as _annotate_schema,
)
from ._cluster import (
    _clean as _clean,
)
from ._cluster import (
    _distinctive_tokens as _distinctive_tokens,
)
from ._cluster import (
    _fit_reducer as _fit_reducer,
)
from ._cluster import (
    _func_label as _func_label,
)
from ._cluster import (
    _mean_run_len as _mean_run_len,
)
from ._cluster import (
    _phase_flow as _phase_flow,
)
from ._cluster import (
    _purity as _purity,
)
from ._cluster import (
    _reconstruct_words as _reconstruct_words,
)
from ._cluster import (
    _scope_token_indices as _scope_token_indices,
)
from ._cluster import (
    _segments as _segments,
)
from ._cluster import (
    _short_med_long as _short_med_long,
)
from ._cluster import (
    _snippets as _snippets,
)
from ._cluster import (
    _top_content as _top_content,
)
from ._cluster import (
    _top_content_counts as _top_content_counts,
)
from ._cluster import (
    analyze_scope as analyze_scope,
)
from ._cluster import (
    load_token_sequences as load_token_sequences,
)
from ._cluster import (
    mean_meta_adjacency as mean_meta_adjacency,
)


def load_scene_sequences(run_dirs: list[Path], slug: str, layer: int) -> list[dict]:
    """Per passage: the ordered (by token index) scene-labeled stimulus token residuals + scene
    labels + relative positions."""
    seqs: list[dict] = []
    for run in run_dirs:
        adir = run / "activations" / slug
        if not adir.exists():
            continue
        for pdir in sorted(adir.glob("*__identify")):
            meta_path, npy_path = pdir / "meta.json", pdir / f"layer_{layer}.npy"
            if not (meta_path.exists() and npy_path.exists()):
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            resid = np.load(npy_path, mmap_mode="r")
            meta_tokens = meta.get("tokens") or []
            n_prompt = int(meta.get("n_prompt_tokens") or resid.shape[0])
            toks: list[tuple[int, str]] = []
            for sp in meta.get("scene_token_spans", []):
                ts, te = sp.get("tok_start"), sp.get("tok_end")
                if ts is None or te is None:
                    continue
                toks.extend((i, sp["scene"]) for i in range(ts, te))
            toks.sort(key=lambda t: t[0])
            if len(toks) < 3:
                continue
            idx = [i for i, _ in toks]
            seqs.append({
                "schema": meta.get("schema"),
                "vectors": np.asarray(resid[idx], np.float32),
                "scenes": [s for _, s in toks],
                "positions": np.array([i / max(n_prompt - 1, 1) for i, _ in toks]),
                "tokens": [_clean(meta_tokens[i]) if i < len(meta_tokens) else "" for i in idx],
                "npy_path": npy_path,
                "full_span": (idx[0], idx[-1] + 1),
                "n_prompt": n_prompt,
            })
    return seqs


def _full_trajectory(seq: dict, scaler, pca, model: KMeansTemporalMetaClustering) -> dict:
    """Contiguous narrative span (first->last scene token) with EVERY token assigned to its nearest
    meta-cluster, for a smooth per-token trajectory instead of the sparse scene-token subset."""
    lo, hi = seq["full_span"]
    resid = np.load(seq["npy_path"], mmap_mode="r")
    proj = pca.transform(scaler.transform(np.asarray(resid[lo:hi], np.float32)))
    return {"coords": proj, "pos": np.arange(lo, hi) / max(seq["n_prompt"] - 1, 1),
            "meta": model.cluster_to_meta_[model.kmeans.predict(proj)]}




def analyze_schema(seqs: list[dict], k: int, seed: int, force_meta: int | None = None) -> dict:
    """Fit meta-clustering on one schema's scene sequences; score meta vs scene + transitions and
    (CoT-style) characterize the meta-clusters and their fine k-means sub-states.

    The meta-count defaults to the schema's scene count so meta<->scene AMI is granularity-matched
    to the position baseline (equal-width position bins with the same count) -- pass ``force_meta``
    to override it (e.g. for a richer primitive graph); the eigengap's own suggestion is reported
    separately as ``n_metaclusters_auto``.

    Adds, beyond the raw transition matrix: the meta x scene composition P(scene | meta), a temporal
    coherence ratio (mean meta-run length vs a position-shuffled control), the segment-level flow /
    order-stereotypy metrics, per-meta and per-fine-state top-token profiles, and in-memory
    per-passage plot artifacts under the ``_plot`` key (stripped from stats.json by ``_public``).
    """
    all_scene = [sc for s in seqs for sc in s["scenes"]]
    n_scenes = len(set(all_scene))
    raw = np.concatenate([s["vectors"] for s in seqs])
    scaler = StandardScaler().fit(raw)
    n_comp = min(50, raw.shape[1], raw.shape[0])
    sub = np.random.default_rng(seed).permutation(len(raw))[:20000]
    pca = PCA(n_components=n_comp, random_state=seed).fit(scaler.transform(raw[sub]))
    std = [pca.transform(scaler.transform(s["vectors"])) for s in seqs]
    model = KMeansTemporalMetaClustering(
        MetaConfig(n_clusters=k, random_state=seed, force_metaclusters=force_meta or n_scenes)
    ).fit(std, [s["positions"] for s in seqs])
    states = [model.kmeans.predict(v) for v in std]
    metas = [model.cluster_to_meta_[s] for s in states]
    all_state = np.concatenate(states)
    all_meta = np.concatenate(metas)
    all_tok_pos = np.concatenate([s["positions"] for s in seqs])
    all_words = [w for s in seqs for w in _reconstruct_words(s["tokens"])]
    n_meta = model.n_metaclusters_
    # scenes ordered by mean token position (narrative arrival -> leave) -> composition columns
    scene_pos = {sc: float(np.mean([all_tok_pos[i] for i in range(len(all_scene))
                                    if all_scene[i] == sc])) for sc in set(all_scene)}
    scene_order = sorted(scene_pos, key=scene_pos.get)
    scene_col = {sc: j for j, sc in enumerate(scene_order)}
    # position baseline at MATCHED granularity: equal-width position bins, one per scene
    tok_bucket = np.floor(np.clip(all_tok_pos, 0, 0.999) * n_scenes).astype(int)
    # per-token meta transition matrix (ordered)
    trans = np.zeros((n_meta, n_meta))
    for m in metas:
        for a, b in zip(m[:-1], m[1:], strict=False):
            trans[a, b] += 1
    trans = trans / (trans.sum(axis=1, keepdims=True) + 1e-9)
    # meta x scene composition P(scene | meta) + per-meta profile (dominant scene + top tokens)
    comp = np.zeros((n_meta, n_scenes))
    dom, profiles = {}, []
    for mm in range(n_meta):
        sel = all_meta == mm
        wi = np.where(sel)[0]
        labs = [all_scene[i] for i in wi]
        for sc in labs:
            comp[mm, scene_col[sc]] += 1
        dom[mm] = Counter(labs).most_common(1)[0][0] if labs else None
        profiles.append({
            "meta": mm, "n_tokens": int(sel.sum()), "dom_scene": dom[mm],
            "mean_pos": round(float(all_tok_pos[sel].mean()), 3) if sel.any() else None,
            "top": _top_content([all_words[i] for i in wi], 25),
        })
    comp = comp / (comp.sum(axis=1, keepdims=True) + 1e-9)
    # fine k-means states (ordered by mean position), characterized by their top stimulus tokens
    fine = []
    for c in range(k):
        sel = all_state == c
        if not sel.any():
            continue
        wi = np.where(sel)[0]
        labs = [all_scene[i] for i in wi]
        fine.append({
            "state": int(c), "meta": int(model.cluster_to_meta_[c]), "count": int(sel.sum()),
            "mean_pos": round(float(all_tok_pos[sel].mean()), 3),
            "dom_scene": Counter(labs).most_common(1)[0][0] if labs else None,
            "top": _top_content([all_words[i] for i in wi], 20),
        })
    fine.sort(key=lambda d: d["mean_pos"])
    # temporal coherence: mean contiguous meta-run length vs a position-shuffled control
    rng = np.random.default_rng(seed)
    coh = float(np.mean([_mean_run_len(m) for m in metas]))
    shuf = float(np.mean([_mean_run_len(rng.permutation(m)) for m in metas]))
    # per-passage plotting artifacts (kept in-memory; stripped from stats.json by _public)
    passages = [{"tokens": s["tokens"], "positions": s["positions"], "meta": metas[i],
                 "state": states[i],
                 "scene_idx": np.array([scene_col[sc] for sc in s["scenes"]])}
                for i, s in enumerate(seqs)]
    # embedding artifacts for optional PCA/t-SNE cluster viz (subsample + representative
    # short/median/long trajectories)
    rng2 = np.random.default_rng(seed + 1)
    x_all = np.concatenate(std)
    scene_idx_all = np.array([scene_col[sc] for s in seqs for sc in s["scenes"]])
    idx_sub = rng2.permutation(len(x_all))[: min(4000, len(x_all))]
    chosen = _short_med_long([len(s["positions"]) for s in seqs])
    embed = {
        "X": x_all[idx_sub], "meta": all_meta[idx_sub], "pos": all_tok_pos[idx_sub],
        "scene_idx": scene_idx_all[idx_sub],
        "traj": [{"coords": std[i], "pos": seqs[i]["positions"], "meta": metas[i],
                  "tokens": seqs[i]["tokens"]} for i in chosen],
        "traj_full": [_full_trajectory(seqs[i], scaler, pca, model) for i in chosen],
    }
    return {
        "n_tokens": len(all_scene),
        "scored": True,
        "n_scenes": n_scenes,
        "n_metaclusters": int(n_meta),
        "n_metaclusters_auto": int(model.n_metaclusters_auto_),
        "laplacian_eigenvalues": model.eigenvalues_.round(5).tolist(),
        "ami_meta_vs_scene": round(float(adjusted_mutual_info_score(all_scene, all_meta)), 4),
        "ami_positionbin_vs_scene": round(
            float(adjusted_mutual_info_score(all_scene, tok_bucket)), 4),
        "purity_meta_vs_scene": _purity(all_meta, all_scene),
        "coherence": {"mean_run": round(coh, 2), "shuffled": round(shuf, 2),
                      "ratio": round(coh / shuf, 2) if shuf else float("nan")},
        "dominant_scene_per_meta": dom,
        "composition": {"scenes": scene_order, "matrix": comp.round(4).tolist()},
        "meta_profiles": profiles,
        "fine_states": fine,
        "flow": _phase_flow(metas, n_meta, seed),
        "transition_matrix": trans.round(3).tolist(),
        "meta_transition_matrix": mean_meta_adjacency(metas, n_meta).round(4).tolist(),
        "_plot": {"passages": passages, "scene_order": scene_order, "embed": embed},
    }




# --------------------------------------------------------------------------- figures


def _transition_figure(res: dict, schema: str, path: Path, slug: str, layer: int,
                       annot: dict | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trans = np.asarray(res["transition_matrix"])
    lw = res.get("label_word", "scene")
    labs = _meta_labels(res, annot)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(trans, cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(len(labs)), labs, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(labs)), labs, fontsize=7)
    ax.set_xlabel(f"to meta-cluster (dominant {lw})")
    ax.set_ylabel("from meta-cluster")
    fig.colorbar(im, ax=ax, label="transition prob (row-normalized)")
    sub = (f"AMI(meta,{lw})={res['ami_meta_vs_scene']} vs pos-bin "
           f"{res['ami_positionbin_vs_scene']}") if res.get("ami_meta_vs_scene") is not None else (
        f"unsupervised; no ground-truth {lw} labels")
    ax.set_title(f"meta-cluster transition graph ({lw}) -- {schema} -- {slug} L{layer}\n{sub}")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _meta_labels(res: dict, annot: dict | None = None) -> list[str]:
    """Axis labels for meta-clusters: ``<id>: <LLM label>`` when annotated (else the
    top content tokens). The dominant reference label is prefixed only when it *varies* across
    clusters -- for the response scope every cluster is dominated by the same generic ``reasoning``
    bucket, so it is dropped in favor of the descriptive name; for the scene scope it is the
    (varied, meaningful) scene name and is kept."""
    dom = res["dominant_scene_per_meta"]
    profiles = {p["meta"]: p for p in res.get("meta_profiles", [])}
    doms = [dom.get(m, dom.get(str(m))) for m in range(res["n_metaclusters"])]
    show_dom = len(set(doms)) > 1
    labs = []
    for m in range(res["n_metaclusters"]):
        a = (annot or {}).get(m) or (annot or {}).get(str(m))
        head = f"{m}:{doms[m]}" if show_dom else f"{m}"
        if a and a.get("label"):
            labs.append(f"{head} {a['label']}")
        else:
            top = profiles.get(m, {}).get("top", [])[:3]
            labs.append(head if show_dom else f"{m}: " + (" ".join(top) if top else "?"))
    return labs


def _composition_figure(res: dict, schema: str, path: Path, slug: str, layer: int,
                        annot: dict | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    comp = np.asarray(res["composition"]["matrix"])
    scenes = res["composition"]["scenes"]
    n_meta = comp.shape[0]
    lw = res.get("label_word", "scene")
    labs = _meta_labels(res, annot)
    fig, ax = plt.subplots(figsize=(max(5.0, 1.1 * len(scenes) + 2), max(4.0, 0.5 * n_meta + 2)))
    im = ax.imshow(comp, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(scenes)), scenes, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(n_meta), labs, fontsize=7)
    for i in range(n_meta):
        for j in range(len(scenes)):
            ax.text(j, i, f"{comp[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="w" if comp[i, j] < 0.6 else "k")
    fig.colorbar(im, ax=ax, label=f"P({lw} | meta-cluster)")
    ax.set_xlabel(f"{lw} (order)")
    ax.set_ylabel("meta-cluster (position-ordered)")
    ax.set_title(f"meta-cluster x {lw} composition -- {schema} -- {slug} L{layer}\n"
                 f"AMI={res['ami_meta_vs_scene']} vs pos-bin {res['ami_positionbin_vs_scene']} | "
                 f"purity={res['purity_meta_vs_scene']}")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _raster_figure(plot: dict, res: dict, schema: str, path: Path, slug: str, layer: int,
                   n_show: int = 16) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    passages = plot["passages"][:n_show]
    n_meta = res["n_metaclusters"]
    scenes = plot["scene_order"]
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for y, p in enumerate(passages):
        pos = p["positions"]
        axes[0].scatter(pos, np.full(len(pos), y, float),
                        c=[cmap(int(v) % 10) for v in p["meta"]], s=7, marker="s")
        axes[1].scatter(pos, np.full(len(pos), y, float),
                        c=[cmap(int(v) % 10) for v in p["scene_idx"]], s=7, marker="s")
    axes[0].set_ylabel("passage")
    axes[0].set_title(f"colored by unsupervised meta-cluster (n={n_meta})")
    axes[1].set_ylabel("passage")
    axes[1].set_title(f"colored by true scene (n={len(scenes)}): {', '.join(scenes)}")
    axes[1].set_xlabel("normalized stimulus position")
    fig.suptitle(f"S2 scene-token raster -- {schema} -- {slug} L{layer} "
                 f"(both position-ordered; aligned colors = recovered scenes)")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _flow_figure(res: dict, schema: str, path: Path, slug: str, layer: int,
                 annot: dict | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    flow = res["flow"]
    trans = np.asarray(flow["seg_transition"])
    n_meta = res["n_metaclusters"]
    labs = _meta_labels(res, annot)
    fig, ax = plt.subplots(figsize=(max(6.0, 0.8 * n_meta + 3), max(5.0, 0.7 * n_meta + 2)))
    im = ax.imshow(trans, cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(n_meta), labs, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(n_meta), labs, fontsize=7)
    for i in range(n_meta):
        for j in range(n_meta):
            if trans[i, j] >= 0.08:
                ax.text(j, i, f"{trans[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="w" if trans[i, j] < 0.6 else "k")
    fig.colorbar(im, ax=ax, label="P(next segment meta | current)")
    ax.set_xlabel("to meta-cluster (next segment)")
    ax.set_ylabel("from meta-cluster")
    ax.set_title(f"S3 meta-cluster segment flow -- {schema} -- {slug} L{layer}\n"
                 f"forward-frac {flow['forward_frac']} | monotonicity {flow['monotonicity']} "
                 f"(shuffled {flow['monotonicity_shuffled']})")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _meta_graph_labels(res: dict, annot: dict | None = None) -> list[str]:
    """Short per-primitive labels for the graph/raster legends: the LLM label when annotated,
    else the top-3 content tokens."""
    profiles = {p["meta"]: p for p in res["meta_profiles"]}
    out = []
    for m in range(res["n_metaclusters"]):
        a = (annot or {}).get(m) or (annot or {}).get(str(m))
        if a and a.get("label"):
            out.append(f"{a['label']}")
        else:
            top = profiles.get(m, {}).get("top", [])[:3]
            out.append(" ".join(top) if top else "?")
    return out


def _primitive_graph_figure(res: dict, schema: str, path: Path, slug: str, layer: int,
                            annot: dict | None = None, seed: int = 0) -> None:
    """Directed primitive-transition graph (algotrace panel-d style). Nodes = position-ordered
    meta-clusters (primitives), sized by token share; directed weighted edges = segment-level
    P(next primitive | current) (self-transitions excluded by construction), drawn as curved
    arrows colored by their source, width/opacity ~ probability. Each node keeps at least its
    single strongest outgoing edge so the reasoning flow stays connected."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx
    from matplotlib.patches import FancyArrowPatch

    flow = np.asarray(res["flow"]["seg_transition"])
    n_meta = res["n_metaclusters"]
    lw = res.get("label_word", "scene")
    profiles = {p["meta"]: p for p in res["meta_profiles"]}
    counts = np.array([profiles.get(m, {}).get("n_tokens", 1) or 1 for m in range(n_meta)], float)
    share = counts / counts.sum()
    cmap = plt.get_cmap("tab10" if n_meta <= 10 else "tab20")
    node_color = [cmap(m % cmap.N) for m in range(n_meta)]
    labs = _meta_graph_labels(res, annot)
    thr = 0.08
    edges: dict[tuple[int, int], float] = {}
    for i in range(n_meta):
        for j in range(n_meta):
            if i != j and flow[i, j] >= thr:
                edges[(i, j)] = float(flow[i, j])
        row = [(j, flow[i, j]) for j in range(n_meta) if j != i]
        if row:
            j, w = max(row, key=lambda t: t[1])
            if w > 0:
                edges[(i, j)] = float(w)
    g = nx.DiGraph()
    g.add_nodes_from(range(n_meta))
    for (i, j), w in edges.items():
        g.add_edge(i, j, weight=w)
    pos = nx.spring_layout(g, seed=seed, weight="weight", k=2.2 / np.sqrt(max(n_meta, 1)))
    fig, ax = plt.subplots(figsize=(11, 8))
    for (i, j), w in edges.items():
        ax.add_patch(FancyArrowPatch(
            pos[i], pos[j], connectionstyle="arc3,rad=0.16", arrowstyle="-|>",
            mutation_scale=10 + 24 * w, lw=0.7 + 5.0 * w, color=node_color[i],
            alpha=float(min(0.9, 0.3 + w)), shrinkA=16, shrinkB=18, zorder=2))
    for m in range(n_meta):
        ax.scatter(*pos[m], s=500 + 7000 * share[m], color=node_color[m], edgecolors="k",
                   linewidths=1.3, zorder=3)
        ax.text(pos[m][0], pos[m][1], str(m), ha="center", va="center", fontsize=11,
                weight="bold", zorder=4)
    handles = [plt.Line2D([0], [0], marker="o", ls="", color=node_color[m],
                          label=f"{m}: {labs[m]}  ({share[m] * 100:.0f}%)")
               for m in range(n_meta)]
    ax.legend(handles=handles, fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5),
              frameon=False)
    ax.set_axis_off()
    fl = res["flow"]
    ax.set_title(f"primitive transition graph ({lw}) -- {schema} -- {slug} L{layer}\n"
                 f"{n_meta} primitives | forward-frac {fl['forward_frac']} | "
                 f"monotonicity {fl['monotonicity']} (shuffled {fl['monotonicity_shuffled']})")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _meta_mean_transition(passages: list[dict], n_meta: int) -> np.ndarray:
    """AlgoTrace Figure-2b meta-cluster transition matrix (``n_meta x n_meta``), computed from the
    per-token meta sequences cached in ``res`` (``_plot.passages[*].meta``). Thin wrapper over
    :func:`mean_meta_adjacency` (the reference ``meta_adjacency_``): per-response token-level meta
    transitions with self-transitions INCLUDED, row-normalized, then simple-mean across responses.
    ``run()`` also saves the result as the ``meta_transition_matrix`` field; this recomputes it for
    callers (e.g. the figure / tests) that only have the passages."""
    return mean_meta_adjacency([p["meta"] for p in passages], n_meta)


def _meta_transition_graph_figure(res: dict, schema: str, path: Path, slug: str, layer: int,
                                  annot: dict | None = None, seed: int = 0,
                                  edge_cutoff: float = 0.08, self_cutoff: float = 0.1,
                                  draw_self_loops: bool = False) -> None:
    """AlgoTrace Figure-2b meta-cluster transition graph. Nodes = the ``n_meta`` spectral
    meta-clusters (the paper's ``MC1..MCn``), coloured per meta + sized by token share. Directed
    curved edges are the off-diagonal TOKEN-level P(next MC | cur MC) from
    :func:`_meta_mean_transition` above ``edge_cutoff`` (each node also keeps its strongest outgoing
    edge for connectivity), width/opacity ~ probability, coloured by the source MC. The transition
    counts are taken directly on each response trace, per the algotrace method, NOT globally pooled
    (``res['transition_matrix']``) or segment-collapsed (``flow``); the off-diagonal flow is ~0.95
    correlated with the segment-level :func:`_primitive_graph_figure` (same reasoning-order
    backbone). Self-transitions are counted into the matrix but, like Figure 2b, are NOT drawn by
    default; set ``draw_self_loops=True`` to overlay them as rings sized/labelled by the diagonal
    self-transition probability (token dwell -- the one thing the segment graph discards)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx
    from matplotlib.lines import Line2D
    from matplotlib.patches import Arc, FancyArrowPatch

    passages = res.get("_plot", {}).get("passages", [])
    n_meta = res["n_metaclusters"]
    if not passages or n_meta < 1:
        return
    stored = res.get("meta_transition_matrix")
    trans = np.asarray(stored, float) if stored is not None else _meta_mean_transition(passages,
                                                                                       n_meta)
    lw = res.get("label_word", "scene")
    profiles = {p["meta"]: p for p in res["meta_profiles"]}
    counts = np.array([profiles.get(m, {}).get("n_tokens", 1) or 1 for m in range(n_meta)], float)
    share = counts / counts.sum()
    cmap = plt.get_cmap("tab10" if n_meta <= 10 else "tab20")
    node_color = [cmap(m % cmap.N) for m in range(n_meta)]
    labs = _meta_graph_labels(res, annot)

    edges: dict[tuple[int, int], float] = {}
    for i in range(n_meta):
        for j in range(n_meta):
            if i != j and trans[i, j] >= edge_cutoff:
                edges[(i, j)] = float(trans[i, j])
        row = [(j, trans[i, j]) for j in range(n_meta) if j != i]
        if row:
            j, w = max(row, key=lambda t: t[1])
            if w > 0:
                edges[(i, j)] = float(w)

    g = nx.DiGraph()
    g.add_nodes_from(range(n_meta))
    for (i, j), w in edges.items():
        g.add_edge(i, j, weight=w)
    pos = nx.spring_layout(g, seed=seed, weight="weight", k=2.2 / np.sqrt(max(n_meta, 1)))
    fig, ax = plt.subplots(figsize=(11, 8))
    wmax = max(edges.values()) if edges else 1.0
    for (i, j), w in edges.items():
        frac = w / wmax  # width/opacity proportional to P within the drawn edge set
        ax.add_patch(FancyArrowPatch(
            pos[i], pos[j], connectionstyle="arc3,rad=0.16", arrowstyle="-|>",
            mutation_scale=8 + 22 * frac, lw=0.6 + 6.0 * frac, color=node_color[i],
            alpha=float(0.30 + 0.60 * frac), shrinkA=16, shrinkB=18, zorder=2))
    if draw_self_loops:
        for m in range(n_meta):
            s = float(trans[m, m])
            if s >= self_cutoff:
                x, y = pos[m]
                r = 0.07 + 0.10 * s
                ax.add_patch(Arc((x, y + r), 2 * r, 2 * r, angle=0.0, theta1=-60.0, theta2=240.0,
                                 color=node_color[m], lw=1.0 + 5.0 * s,
                                 alpha=float(min(0.9, 0.35 + s)), zorder=1))
                ax.text(x, y + 2 * r + 0.02, f"{s:.2f}", ha="center", va="bottom", fontsize=7,
                        color=node_color[m], zorder=4)
    for m in range(n_meta):
        ax.scatter(*pos[m], s=500 + 7000 * share[m], color=node_color[m], edgecolors="k",
                   linewidths=1.3, zorder=3)
        ax.text(pos[m][0], pos[m][1], str(m), ha="center", va="center", fontsize=11,
                weight="bold", zorder=4)
    handles = [Line2D([0], [0], marker="o", ls="", color=node_color[m],
                      label=f"M{m}: {labs[m]}  ({share[m] * 100:.0f}%)") for m in range(n_meta)]
    ax.legend(handles=handles, fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5),
              frameon=False, title="meta-cluster (MC)")
    ax.set_axis_off()
    loops = " | self-loop = token dwell" if draw_self_loops else ""
    ax.set_title(f"meta-cluster transition graph (algotrace fig-2b) ({lw}) -- {schema} -- {slug} "
                 f"L{layer}\n{n_meta} meta-clusters | token-level P(next MC | cur MC) on response "
                 f"traces (edge width/opacity ~ P){loops}")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _trace_states_figure(plot: dict, res: dict, schema: str, path: Path, slug: str, layer: int,
                         annot: dict | None = None) -> None:
    """Single-trace cluster rasters (algotrace panel-c style): for three representative traces
    (shortest / median / longest) the k-means state id (y, states grouped/ordered by their
    meta-cluster so each meta occupies a contiguous band) vs token index (x); horizontal bands +
    point colors mark the meta-cluster (primitive)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_meta = res["n_metaclusters"]
    fine = sorted(res["fine_states"], key=lambda d: (d["meta"], d["mean_pos"]))
    rank = {d["state"]: r for r, d in enumerate(fine)}
    st_meta = {d["state"]: d["meta"] for d in fine}
    cmap = plt.get_cmap("tab10" if n_meta <= 10 else "tab20")
    labs = _meta_graph_labels(res, annot)
    passages = plot["passages"]
    chosen = _short_med_long([len(q["state"]) for q in passages])
    names = (["shortest", "median", "longest"][:len(chosen)] if len(chosen) == 3
             else [f"trace {i + 1}" for i in range(len(chosen))])
    fig, axes = plt.subplots(len(chosen), 1, figsize=(12, 2.4 * len(chosen) + 1.4), squeeze=False)
    for ax, pi, name in zip(axes[:, 0], chosen, names, strict=True):
        states = [int(s) for s in passages[pi]["state"]]
        y = [rank.get(s, 0) for s in states]
        for m in range(n_meta):
            ranks = [rank[d["state"]] for d in fine if d["meta"] == m]
            if ranks:
                ax.axhspan(min(ranks) - 0.5, max(ranks) + 0.5, color=cmap(m % cmap.N), alpha=0.22,
                           zorder=0)
        ax.plot(range(len(y)), y, color="0.5", lw=0.35, alpha=0.35, zorder=2)
        ax.scatter(range(len(y)), y, c=[cmap(st_meta.get(s, 0) % cmap.N) for s in states], s=7,
                   zorder=3)
        ax.set_ylabel("k-means state")
        ax.set_title(f"{name} trace ({len(y)} tokens)", fontsize=9, loc="left")
    axes[-1, 0].set_xlabel("token index")
    handles = [plt.Line2D([0], [0], marker="s", ls="", color=cmap(m % cmap.N),
                          label=f"{m}: {labs[m]}") for m in range(n_meta)]
    fig.legend(handles=handles, fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5),
               frameon=False)
    fig.suptitle(f"single-trace primitive rasters -- {schema} -- {slug} L{layer} "
                 f"(shortest / median / longest of {len(passages)} traces)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _fine_states_figure(res: dict, schema: str, path: Path, slug: str, layer: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fine = res["fine_states"]
    n_meta = res["n_metaclusters"]
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(9, max(4.0, 0.28 * len(fine) + 2)))
    for d in fine:
        ax.scatter(d["mean_pos"], d["state"], s=20 + np.sqrt(d["count"]) * 3,
                   c=[cmap(d["meta"] % 10)])
        ax.text(d["mean_pos"] + 0.004, d["state"], " ".join(d["top"][:4]), fontsize=6, va="center")
    handles = [plt.Line2D([0], [0], marker="o", ls="", color=cmap(m % 10), label=f"meta {m}")
               for m in range(n_meta)]
    ax.legend(handles=handles, fontsize=7, loc="best")
    ax.set_xlabel("mean stimulus position")
    ax.set_ylabel("k-means state id")
    ax.set_title(f"S2 fine k-means states (color = meta-cluster, size ~ count) -- {schema} -- "
                 f"{slug} L{layer}")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _eigengap_figure(res: dict, schema: str, path: Path, slug: str, layer: int) -> None:
    """The normalized-Laplacian eigenvalue spectrum of the mean temporal-adjacency graph (the
    'eigenmap' the meta-count heuristic reads) + the consecutive eigengaps. The largest gap in the
    search window sets the auto-suggested count; the forced ``--meta`` count is marked too, so
    over-/under-segmentation vs the natural granularity is visible."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eig = np.asarray(res.get("laplacian_eigenvalues") or [], dtype=float)
    if eig.size == 0:
        return
    k_show = int(min(len(eig), 20))
    ev = eig[:k_show]
    gaps = np.diff(ev)
    auto, forced = res["n_metaclusters_auto"], res["n_metaclusters"]
    lw = res.get("label_word", "scene")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(range(1, k_show + 1), ev, "o-", color="#4C72B0", ms=4)
    ax1.axvline(auto, ls="--", color="green", lw=1.3, label=f"eigengap suggests {auto}")
    ax1.axvline(forced, ls=":", color="red", lw=1.3, label=f"used {forced}")
    ax1.set_xlabel("k (candidate meta-count)")
    ax1.set_ylabel("normalized-Laplacian eigenvalue")
    ax1.set_title("eigenvalue spectrum ('eigenmap')")
    ax1.legend(fontsize=8)
    if gaps.size:
        ax2.bar(range(1, len(gaps) + 1), gaps, color="#DD8452")
        ax2.axvline(auto, ls="--", color="green", lw=1.3)
        ax2.axvline(forced, ls=":", color="red", lw=1.3)
    ax2.set_xlabel("k (gap after the k-th eigenvalue)")
    ax2.set_ylabel("eigengap Δλ")
    ax2.set_title("consecutive eigengaps (largest in window -> suggested count)")
    fig.suptitle(f"meta-count eigengap -- {schema} -- {slug} L{layer} ({lw} scope)")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _answer_start(tokens: list[str]) -> int | None:
    """Token index just after the ``</think>`` marker (= start of the final answer) for a reasoning
    trace, else None (no explicit think block)."""
    full = "".join(tokens)
    end = full.rfind("</think>")
    if end < 0:
        return None
    end += len("</think>")
    acc = 0
    for i, t in enumerate(tokens):
        acc += len(t)
        if acc >= end:
            return i + 1
    return None


def _html_figure(res: dict, schema: str, path: Path, slug: str, layer: int,
                 annot: dict | None = None, n_show: int = 12) -> None:
    """Interactive token map (algotrace ``clustering_visualization.print_metacluster_tabs`` analog).
    Shows the **full** text (CoT + final answer) of a length-stratified sample of responses -- so
    both long CoT-heavy traces and short answer-visible ones appear -- every token
    background-colored by its unsupervised meta-cluster. Client-side JS (no server): buttons to
    switch the displayed response, buttons to **highlight** one primitive (dimming the rest) or
    show All, and a custom **hover tooltip** giving each token's meta-cluster + fine k-means state.
    The ``</think>`` boundary is flagged so the final answer is easy to find; whitespace/newline
    tokens become line breaks."""
    import html as _html

    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    n_meta = res["n_metaclusters"]
    cmap = plt.get_cmap("tab10" if n_meta <= 10 else "tab20")
    light, solid = [], []
    for m in range(n_meta):
        r, g, b, _ = mcolors.to_rgba(cmap(m % cmap.N))
        light.append(f"rgba({int(r * 255)},{int(g * 255)},{int(b * 255)},0.6)")
        solid.append(mcolors.to_hex(cmap(m % cmap.N)))
    labs = _meta_graph_labels(res, annot)
    lw = res.get("label_word", "scene")
    # length-stratified sample (shortest first) so short, answer-visible responses appear too
    srt = sorted(res["_plot"]["passages"], key=lambda p: len(p["tokens"]))
    if len(srt) <= n_show:
        passages = srt
    else:
        idx = sorted(set(np.linspace(0, len(srt) - 1, n_show).round().astype(int).tolist()))
        passages = [srt[i] for i in idx]
    css = (
        "<style>body{font-family:system-ui,sans-serif;max-width:1150px;margin:18px auto;"
        "color:#222;}.ctl{background:#fff;padding:8px 0 10px;margin-bottom:6px;"
        "border-bottom:1px solid #eee;}"
        ".row{margin:5px 0;display:flex;gap:5px;flex-wrap:wrap;align-items:center;}"
        ".row .lab{font-weight:bold;color:#555;font-size:13px;margin-right:4px;}"
        "button{cursor:pointer;border-radius:5px;border:1px solid #bbb;padding:4px 9px;"
        "font-size:12px;background:#f5f5f5;color:#222;}button.active{outline:2px solid #333;}"
        ".resp{white-space:pre-wrap;line-height:2.0;font-size:14px;border:1px solid #e0e0e0;"
        "padding:12px;border-radius:6px;margin-top:8px;}.t{border-radius:3px;}"
        ".t.dim{opacity:0.1;}#info{font-size:13px;color:#555;margin:6px 0;min-height:17px;}"
        ".ans{display:block;margin:10px 0 4px;padding:4px 8px;background:#fffbe6;"
        "border-left:4px solid #f0ad4e;color:#8a6d3b;font-weight:bold;font-size:12px;"
        "border-radius:4px;}#tip{position:fixed;pointer-events:none;background:#222;color:#fff;"
        "padding:5px 8px;border-radius:5px;font-size:12px;max-width:340px;display:none;"
        "z-index:100;line-height:1.4;}</style>"
    )
    js = (
        "<script>let R=0,M='all';"
        "function selR(i){R=i;var d=document.querySelectorAll('.resp');"
        "for(var k=0;k<d.length;k++)d[k].style.display=(k==i)?'block':'none';mk('r',i);}"
        "function selM(m){M=''+m;mk('m',m);var t=document.querySelectorAll('.t');"
        "for(var k=0;k<t.length;k++){var on=(m==='all'||t[k].dataset.m===''+m);"
        "t[k].classList.toggle('dim',!on);}"
        "document.getElementById('info').textContent="
        "(m==='all')?'':('highlighting primitive '+m+': '+LAB[m]);}"
        "function mk(p,v){var b=document.querySelectorAll('.'+p+'btn');"
        "for(var k=0;k<b.length;k++)b[k].classList.remove('active');"
        "var e=document.getElementById(p+'btn'+v);if(e)e.classList.add('active');}</script>"
    )
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>meta-cluster token map -- {_html.escape(schema)} -- {slug} L{layer}</title>",
        css, f"<script>const LAB={json.dumps(labs)};</script>", js, "</head><body>",
        f"<h2>meta-cluster token map \u2014 {_html.escape(schema)} \u2014 {slug} L{layer} "
        f"({lw} scope, {n_meta} primitives)</h2>",
        "<p>Full response text (CoT + final answer) colored by unsupervised meta-cluster. "
        "Responses span the length range (shortest first); the final answer follows the \u25bc "
        "marker. Pick a response, click a primitive to highlight it (or <b>All</b>), and hover a "
        "token for its meta-cluster + fine state.</p>",
        "<div class='ctl'><div class='row'><span class='lab'>Response:</span>",
    ]
    for pi, p in enumerate(passages):
        a = " active" if pi == 0 else ""
        parts.append(f"<button id='rbtn{pi}' class='rbtn{a}' onclick='selR({pi})'>"
                     f"{pi + 1} ({len(p['tokens'])}t)</button>")
    parts.append("</div><div class='row'><span class='lab'>Highlight primitive:</span>"
                 "<button id='mbtnall' class='mbtn active' onclick=\"selM('all')\">All</button>")
    for m in range(n_meta):
        parts.append(f"<button id='mbtn{m}' class='mbtn' onclick='selM({m})' "
                     f"style='background:{solid[m]};color:#fff;border-color:{solid[m]}'>"
                     f"{m}: {_html.escape(labs[m])}</button>")
    parts.append("</div></div><div id='info'></div><div class='stage'>")
    for pi, p in enumerate(passages):
        disp = "block" if pi == 0 else "none"
        metas = [int(x) for x in p["meta"]]
        states = [int(x) for x in p["state"]]
        ans = _answer_start(p["tokens"])
        parts.append(f"<div class='resp' id='resp{pi}' style='display:{disp}'>")
        for ti, (tok, m, st) in enumerate(zip(p["tokens"], metas, states, strict=False)):
            if ans is not None and ti == ans:
                parts.append("<div class='ans'>\u25bc final answer (after &lt;/think&gt;)</div>")
            if tok.strip() == "":
                parts.append("<br>")
                continue
            ttl = f"meta {m}: {_html.escape(labs[m], quote=True)} \u00b7 state {st}"
            parts.append(
                f"<span class='t' data-m='{m}' data-s='{st}' style='background:{light[m]}' "
                f"title='{ttl}'>{_html.escape(tok)}</span>")
        parts.append("</div>")
    parts.append("</div><div id='tip'></div>")
    parts.append(
        "<script>(function(){var tip=document.getElementById('tip');"
        "var st=document.querySelector('.stage');"
        "st.addEventListener('mouseover',function(e){var t=e.target;"
        "if(t.classList&&t.classList.contains('t')){tip.textContent='\\u201c'+t.textContent+"
        "'\\u201d \\u2014 meta '+t.dataset.m+': '+LAB[t.dataset.m]+' \\u00b7 state '+t.dataset.s;"
        "tip.style.display='block';}});"
        "st.addEventListener('mousemove',function(e){tip.style.left=(e.clientX+12)+'px';"
        "tip.style.top=(e.clientY+14)+'px';});"
        "st.addEventListener('mouseout',function(e){if(e.target.classList&&"
        "e.target.classList.contains('t'))tip.style.display='none';});})();</script>")
    parts.append("</body></html>")
    path.write_text("".join(parts), encoding="utf-8")


def _scatter_2panel(coords: np.ndarray, meta: np.ndarray, scene_idx: np.ndarray | None,
                    meta_labs: list, scenes: list | None, title: str, path: Path, axis: str,
                    label_word: str = "scene") -> None:
    """2-D scatter of the projection colored by meta-cluster; when ground-truth labels exist
    (``scenes`` given) a second panel colored by that label is added. For scopes without ground
    truth (the response scope) only the meta-cluster panel is drawn."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab10")
    two = scenes is not None
    fig, axes = plt.subplots(1, 2 if two else 1, figsize=(13 if two else 6.8, 5.5), squeeze=False)
    axes = axes[0]
    axes[0].scatter(coords[:, 0], coords[:, 1], s=5, alpha=0.5,
                    c=[cmap(int(m) % 10) for m in meta])
    axes[0].set_title("colored by meta-cluster")
    axes[0].legend(handles=[plt.Line2D([0], [0], marker="o", ls="", color=cmap(i % 10), label=lab)
                            for i, lab in enumerate(meta_labs)], fontsize=6, ncol=2, loc="best")
    if two:
        axes[1].scatter(coords[:, 0], coords[:, 1], s=5, alpha=0.5,
                        c=[cmap(int(s) % 10) for s in scene_idx])
        axes[1].set_title(f"colored by true {label_word}")
        axes[1].legend(handles=[plt.Line2D([0], [0], marker="o", ls="", color=cmap(i % 10),
                                           label=sc) for i, sc in enumerate(scenes)],
                       fontsize=6, ncol=2, loc="best")
    for ax in axes:
        ax.set_xlabel(f"{axis}1")
        ax.set_ylabel(f"{axis}2")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _draw_traj3d(path: Path, pos_bg: np.ndarray, bg: np.ndarray, meta_bg: np.ndarray,
                 paths: list, meta_labs: list, title: str, names: list[str] | None = None) -> None:
    """3-D trajectory plots, one panel per selected trace (side by side): x = token index,
    (y, z) = t-SNE; each panel draws the faint meta-colored background cloud + a single overlaid
    path (neutral line = temporal order, marker color = meta-cluster, green start / red end)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not paths:
        return
    cmap = plt.get_cmap("tab10")
    if names is None:
        names = (["shortest", "median", "longest"][:len(paths)] if len(paths) == 3
                 else [f"trace {i + 1}" for i in range(len(paths))])
    fig = plt.figure(figsize=(7.2 * len(paths), 7.0))
    for j, (tp, yt, tm) in enumerate(paths):
        ax = fig.add_subplot(1, len(paths), j + 1, projection="3d")
        ax.view_init(elev=18, azim=-62)
        ax.scatter(pos_bg, bg[:, 0], bg[:, 1], s=3, alpha=0.06,
                   c=[cmap(int(m) % 10) for m in meta_bg])
        ax.plot(tp, yt[:, 0], yt[:, 1], color="0.4", lw=0.8, alpha=0.6, zorder=3)
        ax.scatter(tp, yt[:, 0], yt[:, 1], s=16, edgecolors="k", linewidths=0.2, zorder=4,
                   c=[cmap(int(m) % 10) for m in tm])
        ax.scatter(tp[:1], yt[:1, 0], yt[:1, 1], s=110, c="lime", edgecolors="k", zorder=6)
        ax.scatter(tp[-1:], yt[-1:, 0], yt[-1:, 1], s=110, c="red", edgecolors="k", zorder=6)
        ax.text(tp[0], yt[0, 0], yt[0, 1], "  start", fontsize=8, weight="bold")
        ax.text(tp[-1], yt[-1, 0], yt[-1, 1], "  end", fontsize=8, weight="bold")
        ax.set_xlabel("token index (normalized)")
        ax.set_ylabel("tSNE1")
        ax.set_zlabel("tSNE2")
        ax.set_title(f"{names[j]} ({len(tp)} tokens)", fontsize=10)
        if j == 0:
            ax.legend(handles=[plt.Line2D([0], [0], marker="o", ls="", color=cmap(i % 10),
                                          label=lab) for i, lab in enumerate(meta_labs)],
                      fontsize=6, loc="upper left")
    fig.suptitle(title, fontsize=11)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.05, wspace=0.06)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _embedding_figures(res: dict, schema: str, out_dir: Path, slug: str, layer: int, seed: int,
                       n_traj: int = 3, annot: dict | None = None) -> None:
    """PCA(2) + t-SNE cluster scatters (raw scene tokens), plus 3-D token-index x t-SNE trajectory
    plots: ``trajectory`` (sparse scene tokens) and ``trajectory_full`` (every token of the
    contiguous narrative span assigned to its nearest meta-cluster -> a smoother path)."""
    from sklearn.manifold import TSNE

    embed = res["_plot"]["embed"]
    x = np.asarray(embed["X"])
    meta = np.asarray(embed["meta"])
    pos = np.asarray(embed["pos"])
    scored = res.get("scored", True)
    scene_idx = np.asarray(embed["scene_idx"]) if scored else None
    scenes = res["composition"]["scenes"] if scored and res.get("composition") else None
    meta_labs = _meta_labels(res, annot)
    lw = res.get("label_word", "scene")
    tag = f"-- {schema} -- {slug} L{layer}"
    # (1) PCA(2): the top two components are already the first two PCA-50 columns
    _scatter_2panel(x[:, :2], meta, scene_idx, meta_labs, scenes,
                    f"PCA(2) of {lw}-token residuals {tag}",
                    out_dir / f"pca_scatter_{schema}.png", "PC", lw)
    # (2)+(3) one t-SNE over background + sparse-traj + full-traj tokens, then slice each block
    traj = embed["traj"][:n_traj]
    trajf = embed.get("traj_full", [])[:n_traj]  # smooth full-narrative paths for the same traces
    blocks = ([x] + [np.asarray(t["coords"]) for t in traj]
              + [np.asarray(t["coords"]) for t in trajf])
    perp = int(min(30, max(5, (sum(len(b) for b in blocks) - 1) // 3)))
    y = TSNE(n_components=2, random_state=seed, perplexity=perp,
             init="pca").fit_transform(np.vstack(blocks))
    bg = y[: len(x)]
    _scatter_2panel(bg, meta, scene_idx, meta_labs, scenes,
                    f"t-SNE of {lw}-token residuals {tag}",
                    out_dir / f"tsne_scatter_{schema}.png", "tSNE", lw)

    def _slice(items: list, start: int):
        off, out = start, []
        for t in items:
            yt = y[off: off + len(t["coords"])]
            off += len(t["coords"])
            order = np.argsort(np.asarray(t["pos"]))
            out.append((np.asarray(t["pos"])[order], yt[order], np.asarray(t["meta"])[order]))
        return out, off

    raw_paths, off1 = _slice(traj, len(x))
    full_paths, _ = _slice(trajf, off1)
    raw_desc = "scene tokens" if lw == "scene" else f"{lw}-labeled"
    _draw_traj3d(out_dir / f"trajectory_{schema}.png", pos, bg, meta, raw_paths, meta_labs,
                 f"Cluster latent trajectories ({raw_desc}; shortest / median / longest) {tag}")
    if full_paths:
        _draw_traj3d(out_dir / f"trajectory_full_{schema}.png", pos, bg, meta, full_paths,
                     meta_labs, f"Cluster latent trajectories (full narrative; shortest / median /"
                                f" longest) {tag}")


def _public(obj):
    """Deep copy dropping in-memory-only keys (``_``-prefixed) so stats.json stays serializable."""
    if isinstance(obj, dict):
        return {k: _public(v) for k, v in obj.items()
                if not (isinstance(k, str) and k.startswith("_"))}
    if isinstance(obj, list):
        return [_public(v) for v in obj]
    return obj


def write_report(stats: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stats.json").write_text(json.dumps(_public(stats), indent=2), encoding="utf-8")
    scope = next(iter(stats["by_schema"].values()), {}).get("scope")
    head = f"# Temporal meta-clustering -- {stats['slug']} (layer {stats['layer']}"
    head += f", scope={scope})" if scope else ")"
    lines = [
        head,
        "",
        f"Runs: {', '.join(stats['runs'])} | k-means states: {stats['k']}",
        "",
    ]
    if any("meta_transition_matrix" in r for r in stats["by_schema"].values()):
        lines += [
            f"> NOTE: {META_TRANSITION_SUBSTOCHASTIC_NOTE}",
            "",
        ]
    for schema, r in stats["by_schema"].items():
        lw = r.get("label_word", "scene")
        scored = r.get("scored", True)
        coh, fl = r["coherence"], r["flow"]
        if scored:
            lines.append(f"## {schema} ({r['n_scenes']} {lw}s -> {r['n_metaclusters']} "
                         f"meta-clusters; eigengap auto-suggests {r['n_metaclusters_auto']})")
            lines += [
                f"- AMI(meta, {lw}) = {r['ami_meta_vs_scene']} "
                f"(matched position-bin baseline {r['ami_positionbin_vs_scene']})",
                f"- purity(meta -> {lw}) = {r['purity_meta_vs_scene']}",
            ]
        else:
            lines.append(f"## {schema} ({r['n_metaclusters']} meta-clusters/primitives; eigengap "
                         f"auto-suggests {r['n_metaclusters_auto']}; unsupervised -- no "
                         f"ground-truth {lw} labels, so AMI/composition are omitted)")
        lines += [
            f"- coherence: mean meta-run {coh['mean_run']} vs shuffled {coh['shuffled']} "
            f"({coh['ratio']}x)",
            f"- order: forward-fraction {fl['forward_frac']}, monotonicity {fl['monotonicity']} "
            f"(shuffled {fl['monotonicity_shuffled']})",
        ]
        if r.get("ami_meta_vs_schema") is not None:
            lines.append(f"- AMI(meta, schema) = {r['ami_meta_vs_schema']} (0 = schema-general)")
        if r.get("ami_meta_vs_task") is not None:
            lines.append(f"- AMI(meta, task) = {r['ami_meta_vs_task']} (0 = task-general "
                         f"primitives; higher = task-specialized)")
        order = None
        if scored:
            order = [r["dominant_scene_per_meta"].get(str(m), r["dominant_scene_per_meta"].get(m))
                     for m in range(r["n_metaclusters"])]
            lines.append(f"- meta-cluster order (dominant {lw}): {order}")
        lines.append("")
        annot = (r.get("annotations") or {}).get("meta") or {}
        if annot:
            for m in range(r["n_metaclusters"]):
                a = annot.get(m) or annot.get(str(m))
                if a:
                    dom_tag = f" [{order[m]}]" if order else ""
                    lines.append(
                        f"  - meta {m}{dom_tag}: {a['label']} -- {a['description']}")
            lines.append("")
    (out_dir / "findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")




_RESPONSE_CTX_MULTITASK = (
    "You are analyzing clusters of tokens from a language model's RESPONSE, POOLED ACROSS SEVERAL "
    "DIFFERENT TASKS on the same kind of short narrative (a restaurant visit or an airport trip "
    "about several people and their roles): identifying the scenario and each person's role, "
    "predicting the next event or a person's correct action, and spotting whether some event is "
    "out of order / missing / mis-assigned. Clusters group response tokens by an unsupervised "
    "temporal clustering of the residual stream, so a primitive may be SHARED across tasks or "
    "SPECIFIC to one -- take that into account when labeling."
)

_LABEL_WORD = {"response": "function", "prompt_response": "segment"}




def _write_csvs(r: dict, schema: str, out_dir: Path, annot: dict | None = None) -> None:
    annot = annot or {}

    def _meta_annot(m: int) -> str:
        a = annot.get(m) or annot.get(str(m)) or {}
        return a.get("label", "")

    # composition P(reference label | meta) -- only when the reference labels are ground truth
    if r.get("scored", True) and r.get("composition"):
        scenes = r["composition"]["scenes"]
        with (out_dir / f"composition_{schema}.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["meta", "label", "dom_scene", "n_tokens", "mean_pos",
                        *[f"P({s})" for s in scenes], "top_words"])
            for pr, row in zip(r["meta_profiles"], r["composition"]["matrix"], strict=False):
                w.writerow([pr["meta"], _meta_annot(pr["meta"]), pr["dom_scene"], pr["n_tokens"],
                            pr["mean_pos"], *[round(x, 4) for x in row], " ".join(pr["top"])])
    # per meta-cluster (primitive) summary: which fine states it groups + label + whole-word profile
    fine_by_meta: dict[int, list[int]] = {}
    for d in r["fine_states"]:
        fine_by_meta.setdefault(d["meta"], []).append(d["state"])
    with (out_dir / f"meta_clusters_{schema}.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["meta", "label", "n_fine_states", "fine_state_ids",
                    "n_tokens", "mean_pos", "top_words"])
        for pr in r["meta_profiles"]:
            ids = fine_by_meta.get(pr["meta"], [])
            w.writerow([pr["meta"], _meta_annot(pr["meta"]), len(ids), " ".join(map(str, ids)),
                        pr["n_tokens"], pr["mean_pos"], " ".join(pr["top"])])
    # every fine k-means micro-state + its meta-cluster grouping + whole-word profile (ALL states)
    with (out_dir / f"fine_states_{schema}.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["state", "meta", "meta_label", "count", "mean_pos", "top_words"])
        for d in r["fine_states"]:
            w.writerow([d["state"], d["meta"], _meta_annot(d["meta"]), d["count"], d["mean_pos"],
                        " ".join(d["top"])])
    # algotrace fig-2b meta-transition matrix P(next MC | cur MC): token-level, per-response mean,
    # self-transitions on the diagonal. Exact matrix behind meta_transition_graph_<schema>.png.
    # CAVEAT: row-sub-stochastic (rows sum <= 1) -- see META_TRANSITION_SUBSTOCHASTIC_NOTE; the
    # findings.md report restates this for downstream graph analysis.
    mt = r.get("meta_transition_matrix")
    if mt is not None:
        n = len(mt)
        labs = {pr["meta"]: _meta_annot(pr["meta"]) for pr in r["meta_profiles"]}
        with (out_dir / f"meta_transition_{schema}.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["from_meta", "from_label", *[f"->{j}" for j in range(n)]])
            for i, row in enumerate(mt):
                w.writerow([i, labs.get(i, ""), *[round(float(x), 4) for x in row]])


def _schema_composition_figure(res: dict, schema: str, path: Path, slug: str, layer: int) -> None:
    """Stacked bar: per pooled primitive, the fraction of its tokens from each schema (even split =
    schema-general primitive; skewed = schema-specific). Title carries AMI(meta, schema)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sc = res.get("schema_composition")
    if not sc:
        return
    mat = np.asarray(sc["matrix"])  # (n_meta, n_schema)
    schemas = sc["schemas"]
    nm = mat.shape[0]
    fig, ax = plt.subplots(figsize=(max(6.0, 0.6 * nm + 3), 4.5))
    bottom = np.zeros(nm)
    for j, s in enumerate(schemas):
        ax.bar(range(nm), mat[:, j], bottom=bottom, label=s)
        bottom += mat[:, j]
    ax.axhline(1.0 / len(schemas), ls="--", color="k", lw=0.8, label="even split")
    ax.set_xticks(range(nm), [f"m{m}" for m in range(nm)])
    ax.set_ylim(0, 1)
    ax.set_ylabel("fraction of primitive's tokens")
    ax.set_title(f"pooled primitive schema composition -- {slug} L{layer}\n"
                 f"AMI(meta, schema) = {res.get('ami_meta_vs_schema')} (0 = schema-general)")
    ax.legend(fontsize=8, ncol=len(schemas) + 1)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _task_composition_figure(res: dict, schema: str, path: Path, slug: str, layer: int) -> None:
    """Stacked bar: per pooled primitive, the fraction of its tokens from each task (even split =
    task-general primitive; skewed = task-specific). Title carries AMI(meta, task)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tc = res.get("task_composition")
    if not tc:
        return
    mat = np.asarray(tc["matrix"])  # (n_meta, n_task)
    tasks = tc["tasks"]
    nm = mat.shape[0]
    fig, ax = plt.subplots(figsize=(max(6.0, 0.6 * nm + 3), 4.5))
    bottom = np.zeros(nm)
    for j, t in enumerate(tasks):
        ax.bar(range(nm), mat[:, j], bottom=bottom, label=t)
        bottom += mat[:, j]
    ax.axhline(1.0 / len(tasks), ls="--", color="k", lw=0.8, label="even split")
    ax.set_xticks(range(nm), [f"m{m}" for m in range(nm)])
    ax.set_ylim(0, 1)
    ax.set_ylabel("fraction of primitive's tokens")
    ax.set_title(f"pooled primitive task composition -- {slug} L{layer}\n"
                 f"AMI(meta, task) = {res.get('ami_meta_vs_task')} (0 = task-general)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def run(run_dirs: list[Path], slug: str, layer: int, k: int, out_dir: Path, seed: int = 0,
        annotate: bool = False, annotate_model: str = "claude-opus-4.8", embed: bool = False,
        n_traj: int = 3, scope: str = "scene", force_meta: int | None = None,
        annotations: str | None = None, tasks: list[str] | None = None,
        pooled: bool = False) -> dict:
    tasks = tasks or ["identify"]
    if scope == "scene":
        seqs = load_scene_sequences(run_dirs, slug, layer)
        if not seqs:
            raise SystemExit(f"no scene sequences for slug={slug} layer={layer} under {run_dirs}")
        schemas = sorted({s["schema"] for s in seqs})
        by_schema = {sch: analyze_schema([s for s in seqs if s["schema"] == sch], k, seed,
                                         force_meta)
                     for sch in schemas}
    else:
        seqs = load_token_sequences(run_dirs, slug, layer, scope, seed, tasks=tasks)
        if not seqs:
            raise SystemExit(f"no {scope} tokens for slug={slug} layer={layer} under {run_dirs}")
        lw = _LABEL_WORD[scope]
        if pooled:
            # cluster ALL schemas' tokens together -> shared primitives; schema_composition +
            # ami_meta_vs_schema reveal whether the primitives are schema-general or -specific.
            schemas = ["combined"]
            by_schema = {"combined": analyze_scope(seqs, k, seed, lw, scope, force_meta or 8)}
        else:
            schemas = sorted({s["schema"] for s in seqs})
            by_schema = {sch: analyze_scope([s for s in seqs if s["schema"] == sch], k, seed, lw,
                                            scope, force_meta)
                         for sch in schemas}
    stats = {
        "slug": slug,
        "layer": layer,
        "k": k,
        "scope": scope,
        "tasks": tasks,
        "runs": [f"{d.parent.name}/{d.name}" for d in run_dirs],
        "by_schema": by_schema,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    reuse = json.loads(Path(annotations).read_text(encoding="utf-8")) if annotations else None
    if reuse is not None:
        for sch in schemas:
            stats["by_schema"][sch]["annotations"] = reuse.get(sch, {})
    elif annotate:
        ctx = _SCOPE_CTX.get(scope, _STIM_CTX)
        if scope == "response" and len(tasks) > 1:
            ctx = _RESPONSE_CTX_MULTITASK
        for sch in schemas:
            stats["by_schema"][sch]["annotations"] = _annotate_schema(
                stats["by_schema"][sch], annotate_model, seed, ctx)
    write_report(stats, out_dir)
    for sch in schemas:
        r = stats["by_schema"][sch]
        annot = (r.get("annotations") or {}).get("meta")
        scored = r.get("scored", True)
        _transition_figure(r, sch, out_dir / f"s3_transition_{sch}.png", slug, layer, annot)
        if scored:
            _composition_figure(r, sch, out_dir / f"s2_composition_{sch}.png", slug, layer, annot)
        _primitive_graph_figure(r, sch, out_dir / f"primitive_graph_{sch}.png", slug, layer,
                                annot, seed)
        _meta_transition_graph_figure(r, sch, out_dir / f"meta_transition_graph_{sch}.png",
                                      slug, layer, annot, seed)
        _trace_states_figure(r["_plot"], r, sch, out_dir / f"trace_states_{sch}.png", slug, layer,
                             annot)
        if scope == "scene":
            _raster_figure(r["_plot"], r, sch, out_dir / f"s2_raster_{sch}.png", slug, layer)
        _flow_figure(r, sch, out_dir / f"s3_flow_{sch}.png", slug, layer, annot)
        _fine_states_figure(r, sch, out_dir / f"s2_fine_states_{sch}.png", slug, layer)
        _eigengap_figure(r, sch, out_dir / f"eigengap_{sch}.png", slug, layer)
        if r.get("schema_composition"):
            _schema_composition_figure(r, sch, out_dir / f"schema_composition_{sch}.png",
                                       slug, layer)
        if r.get("task_composition"):
            _task_composition_figure(r, sch, out_dir / f"task_composition_{sch}.png", slug, layer)
        _html_figure(r, sch, out_dir / f"clusters_{sch}.html", slug, layer, annot)
        _write_csvs(r, sch, out_dir, annot)
        if embed:
            _embedding_figures(r, sch, out_dir, slug, layer, seed, n_traj, annot)
    if annotate or reuse is not None:
        ann = {sch: stats["by_schema"][sch].get("annotations", {}) for sch in schemas}
        (out_dir / "annotations.json").write_text(json.dumps(_public(ann), indent=2),
                                                  encoding="utf-8")
    print(f"-> {out_dir}")
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description="scene temporal meta-clustering (S2/S3)")
    p.add_argument("--run", required=True, nargs="+", help="run dir(s) with activations/<slug>/")
    p.add_argument("--slug", required=True, help="model activation slug (e.g. phi-4)")
    p.add_argument("--layer", type=int, default=17)
    p.add_argument("--k", type=int, default=30, help="number of k-means token states")
    p.add_argument("--meta", type=int, default=None,
                   help="force this many meta-clusters/primitives (default = #reference labels; "
                        "set e.g. 8 for a richer response-scope primitive graph)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--annotate", action="store_true", help="LLM-annotate meta-clusters + states")
    p.add_argument("--annotate-model", default="claude-opus-4.8", help="annotation model")
    p.add_argument("--annotations", default=None,
                   help="reuse an existing annotations.json (skip the LLM call) to just "
                        "re-render figures/labels")
    p.add_argument("--embed", action="store_true", help="PCA + t-SNE cluster scatter + trajectory")
    p.add_argument("--traj", type=int, default=3, help="# passages overlaid on the trajectory plot")
    p.add_argument("--scope", default="scene", choices=["scene", "response", "prompt_response"],
                   help="token scope to cluster (scene=prompt scene-tokens; else response)")
    p.add_argument("--task", "--tasks", dest="tasks", nargs="+", default=["identify"],
                   help="capture task suffix(es) (<sid>__<task>); pass several to pool tasks "
                        "(e.g. --tasks identify predict_next_event predict_correct_action "
                        "spot_error)")
    p.add_argument("--pooled", action="store_true",
                   help="cluster ALL schemas together (one shared primitive set; response scope)")
    p.add_argument("--out", required=True, help="output dir for stats.json / findings.md / figures")
    args = p.parse_args()
    run([Path(r) for r in args.run], args.slug, args.layer, args.k, Path(args.out),
        seed=args.seed, annotate=args.annotate, annotate_model=args.annotate_model,
        embed=args.embed, n_traj=args.traj, scope=args.scope, force_meta=args.meta,
        annotations=args.annotations, tasks=args.tasks, pooled=args.pooled)


if __name__ == "__main__":
    main()
