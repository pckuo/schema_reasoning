"""Shared clustering, loading, and annotation core for reasoning primitives.

This module holds the shared clustering/loader/annotation core used by
``metacluster``, ``depth_metacluster``, and ``cot_trajectory``.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import eigvalsh
from scipy.sparse.csgraph import laplacian as csgraph_laplacian
from scipy.stats import spearmanr
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.preprocessing import StandardScaler

from ._text import (
    _clean,
    _distinctive_tokens,
    _mean_run_len,
    _segments,
    _top_content,
    _top_content_counts,
)

# light functional lexicons (crude keyword sets) for response-token labeling / phase validation
_DELIB = {"but", "so", "wait", "actually", "hmm", "maybe", "perhaps", "possibly", "think", "let",
          "seems", "seem", "might", "could", "however", "because", "thus", "therefore", "probably",
          "guess", "assume", "likely", "since", "though", "unless", "if"}
_INSTR = {"instructions", "instruction", "list", "format", "separately", "answer", "schema",
          "scenario", "line", "lines", "person", "people", "role", "roles", "state", "specific",
          "exact", "refer", "final", "output", "markdown"}

_ANNOT_PROMPT = (
    "{context} Below are {n} token clusters ({kind}); each has its most frequent tokens and a few "
    "example snippets (windows of text where a cluster token occurred). For EACH cluster give a "
    "concise label (2-5 words) for the functional or cognitive step, plus a one-sentence "
    'description. Respond ONLY as JSON: {{"annotations": [{{"id": <id>, "label": "...", '
    '"description": "..."}}]}}.\n\n{blocks}'
)


def _reconstruct_words(tokens: list[str]) -> list[str]:
    """Map each (cleaned, byte-level-BPE) token to the whole word that contains it, so per-token
    cluster members can be summarized by real words instead of subword fragments. A word starts at
    a space-prefixed token (``_clean`` turned the ``Ġ`` boundary into a leading space) and absorbs
    the following non-space alphabetic continuation tokens; the same word string is written back to
    every token position it spans (so the returned list is index-aligned with ``tokens``)."""
    out = [""] * len(tokens)
    i = 0
    while i < len(tokens):
        raw = tokens[i]
        j = i + 1
        while j < len(tokens) and tokens[j][:1].isalpha() and not tokens[j][:1].isspace():
            raw += tokens[j]
            j += 1
        word = raw.strip()
        for p in range(i, j):
            out[p] = word
        i = j
    return out


def _func_label(tok: str) -> str:
    """Coarse response-token function: deliberation vs instruction/format vs substantive
    reasoning."""
    w = tok.strip().lower()
    if w in _DELIB:
        return "deliberate"
    if w in _INSTR:
        return "instruction"
    return "reasoning"


def _phase_flow(metas: list[np.ndarray], n_meta: int, seed: int) -> dict:
    """Segment-level meta transition matrix + order-stereotypy metrics: does the sequence advance
    through meta-clusters in a consistent forward order?"""
    seg_trans = np.zeros((n_meta, n_meta))
    fwd = back = 0
    steps: list[int] = []
    for m in metas:
        seg = [s[0] for s in _segments(m)]  # meta id per contiguous segment
        for a, b in zip(seg[:-1], seg[1:], strict=False):
            seg_trans[a, b] += 1
            steps.append(b - a)
            fwd += int(b > a)
            back += int(b < a)
    row = seg_trans / (seg_trans.sum(1, keepdims=True) + 1e-9)

    def _mono(seqs: list[np.ndarray]) -> float:
        vals = [spearmanr(np.arange(len(m)), m)[0] for m in seqs if len(set(m.tolist())) > 1]
        vals = [v for v in vals if not np.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")

    rng = np.random.default_rng(seed)
    total = fwd + back
    return {
        "seg_transition": row.round(3).tolist(),
        "forward_frac": round(fwd / total, 4) if total else float("nan"),
        "mean_step": round(float(np.mean(steps)), 4) if steps else float("nan"),
        "monotonicity": round(_mono(metas), 4),
        "monotonicity_shuffled": round(_mono([rng.permutation(m) for m in metas]), 4),
        "modal_next": [int(np.argmax(row[i])) if seg_trans[i].sum() else i for i in range(n_meta)],
    }


def _snippets(traces: list[dict], per_trace: list[np.ndarray], target: int, n: int,
              win: int, rng) -> list[str]:
    """A few readable +/-win-token windows around tokens assigned to `target` (meta or state)."""
    hits = [(i, int(j)) for i, lab in enumerate(per_trace) for j in np.where(lab == target)[0]]
    if not hits:
        return []
    out = []
    for k in rng.permutation(len(hits))[:n]:
        i, j = hits[k]
        toks = traces[i]["tokens"]
        out.append("".join(toks[max(0, j - win): j + win]).strip().replace("\n", " ")[:150])
    return out


def _annotate(
    clusters: list[dict], model: str, kind: str, context: str, backend: str = "copilot"
) -> dict:
    """One batched judge-LLM call: label + describe each cluster from its top tokens + snippets.
    ``backend`` picks the transport: ``copilot`` (a Copilot CLI agent session, minutes per call) or
    ``trapi`` (a direct Azure OpenAI chat-completion, seconds per call)."""
    from ..core.llm import extract_json, run_copilot_cli, run_trapi

    blocks = []
    for c in clusters:
        snips = "\n".join(f"   - {s}" for s in c["snippets"]) or "   (none)"
        head = f"Cluster {c['id']}:\n  top tokens: {', '.join(c['top'])}"
        if c.get("distinctive"):
            head += f"\n  distinctive tokens: {', '.join(c['distinctive'])}"
        blocks.append(f"{head}\n  snippets:\n{snips}")
    disc = (" Some clusters also list 'distinctive tokens' -- ones characteristic of that cluster "
            "relative to the others; use them to tell apart clusters that share generic vocabulary."
            ) if any(c.get("distinctive") for c in clusters) else ""
    prompt = _ANNOT_PROMPT.format(context=context + disc, kind=kind, n=len(clusters),
                                  blocks="\n\n".join(blocks))
    try:
        raw = (
            run_trapi(prompt, model=model)
            if backend == "trapi"
            else run_copilot_cli(prompt, model=model)
        )
        data = extract_json(raw)
    except (ValueError, KeyError, RuntimeError, TypeError) as e:
        print(f"   [annotate {kind} failed: {e}]")
        return {}
    out = {}
    for a in data.get("annotations", []) if isinstance(data, dict) else []:
        try:
            out[int(a["id"])] = {"label": str(a.get("label", "?")).strip(),
                                 "description": str(a.get("description", "")).strip()}
        except (ValueError, KeyError, TypeError):
            continue
    return out


@dataclass
class MetaConfig:
    n_clusters: int = 30
    random_state: int = 0
    n_init: int = 4
    max_fit_tokens: int = 15000
    eigengap_start: int = 2
    eigengap_end_fraction: int = 3
    min_metaclusters: int = 2
    max_metaclusters: int | None = None
    force_metaclusters: int | None = None  # if set, use this meta-count instead of the eigengap


class KMeansTemporalMetaClustering:
    """k-means token states -> temporal-adjacency spectral meta-clusters (ordered by position)."""

    def __init__(self, config: MetaConfig):
        self.cfg = config
        self.kmeans: KMeans | None = None
        self.mean_adjacency_: np.ndarray | None = None
        self.n_metaclusters_: int | None = None
        self.n_metaclusters_auto_: int | None = None  # eigengap suggestion (before any force)
        self.cluster_to_meta_: np.ndarray | None = None  # ordered meta id per k-means cluster
        self.eigenvalues_: np.ndarray | None = None  # sorted normalized-Laplacian spectrum

    @staticmethod
    def _temporal_adjacency(labels: np.ndarray, n: int) -> np.ndarray:
        onehot = np.zeros((labels.shape[0], n), dtype=np.float32)
        onehot[np.arange(labels.shape[0]), labels] = 1.0
        adj = onehot[:-1].T @ onehot[1:]
        return adj / (adj.sum(axis=1, keepdims=True) + 1e-8)

    def _n_meta(self, eigvals: np.ndarray) -> int:
        cfg = self.cfg
        start = cfg.eigengap_start
        end = max(start + 1, int(cfg.n_clusters / cfg.eigengap_end_fraction))
        window = np.diff(eigvals)[start:end]
        tentative = int(np.argmax(window) + 1 + start) if window.size else cfg.min_metaclusters
        return max(cfg.min_metaclusters, min(tentative, cfg.max_metaclusters or cfg.n_clusters))

    def fit(self, sequences: list[np.ndarray], positions: list[np.ndarray]):
        n = self.cfg.n_clusters
        x = np.concatenate(sequences, axis=0)
        rng = np.random.default_rng(self.cfg.random_state)
        fit_x = x if len(x) <= self.cfg.max_fit_tokens else x[
            rng.permutation(len(x))[: self.cfg.max_fit_tokens]]
        self.kmeans = KMeans(n_clusters=n, random_state=self.cfg.random_state,
                             n_init=self.cfg.n_init).fit(fit_x)
        adjs, weights = [], []
        for seq in sequences:
            lab = self.kmeans.predict(seq)
            adjs.append(self._temporal_adjacency(lab, n))
            weights.append(len(seq))
        mean_adj = sum(w * a for a, w in zip(adjs, weights, strict=False)) / sum(weights)
        self.mean_adjacency_ = mean_adj
        sym = (mean_adj + mean_adj.T) / 2.0
        eig = np.sort(eigvalsh(csgraph_laplacian(sym, normed=True)))
        self.eigenvalues_ = eig
        self.n_metaclusters_auto_ = self._n_meta(eig)
        n_meta = self.cfg.force_metaclusters or self.n_metaclusters_auto_
        raw = SpectralClustering(n_meta, affinity="precomputed",
                                 random_state=self.cfg.random_state).fit_predict(sym)
        # order meta-clusters by the mean sequence position of their k-means clusters' tokens
        all_lab = self.kmeans.predict(x)
        all_pos = np.concatenate(positions)
        cluster_pos = np.array([all_pos[all_lab == c].mean() if (all_lab == c).any() else 0.0
                                for c in range(n)])
        meta_pos = np.array([cluster_pos[raw == m].mean() if (raw == m).any() else 0.0
                             for m in range(n_meta)])
        order = np.argsort(meta_pos)
        remap = np.empty(n_meta, dtype=int)
        remap[order] = np.arange(n_meta)  # raw meta id -> position-ordered id
        self.n_metaclusters_ = n_meta
        self.cluster_to_meta_ = remap[raw]  # (n_clusters,) ordered meta per cluster
        return self

    def meta_labels(self, seq: np.ndarray) -> np.ndarray:
        return self.cluster_to_meta_[self.kmeans.predict(seq)]


META_TRANSITION_SUBSTOCHASTIC_NOTE = (
    "meta_transition_matrix (algotrace fig-2b `meta_adjacency_`) is ROW-SUB-STOCHASTIC: it is a "
    "simple mean of per-response row-normalized matrices, so a meta-cluster that is absent as a "
    "source in some responses gets zero rows there and its row sums to the FRACTION of responses "
    "in which it appears as a source (<= 1), NOT 1. Before feeding it to downstream graph analysis "
    "that assumes a proper transition matrix (e.g. stationary-distribution / flux measures, or "
    "graphstats.from_transition), row-renormalize the non-zero rows. Its diagonal carries token "
    "dwell (self-transition prob); analyses that drop the diagonal discard that signal."
)


def mean_meta_adjacency(metas: list[np.ndarray], n_meta: int) -> np.ndarray:
    """AlgoTrace Figure-2b meta-cluster transition matrix (the reference ``meta_adjacency_``): on
    EACH response's per-token meta sequence, count adjacent token-level meta transitions
    (``onehot[:-1].T @ onehot[1:]``, self-transitions INCLUDED), row-normalize that response to
    P(next MC | cur MC), then take the SIMPLE mean across responses. Distinct from the sibling
    ``transition_matrix`` (pool ALL token pairs globally, normalize once) and the segment-level
    ``flow['seg_transition']`` (self-loops collapsed out). Passages shorter than 2 tokens are
    skipped.

    WARNING -- the result is ROW-SUB-STOCHASTIC, not a proper transition matrix; see
    :data:`META_TRANSITION_SUBSTOCHASTIC_NOTE` and row-renormalize before downstream graph analysis
    that assumes row-stochastic P."""
    adjs = [KMeansTemporalMetaClustering._temporal_adjacency(np.asarray(m), n_meta)
            for m in metas if len(m) >= 2]
    return np.mean(np.stack(adjs), axis=0) if adjs else np.zeros((n_meta, n_meta))


def _scope_token_indices(meta: dict, scope: str, tokens: list) -> list[tuple[int, str]]:
    """Token indices + reference label for a scope. response: functional label; prompt_response:
    scene name for prompt scene-tokens, 'prompt_other' for the rest of the prompt, 'response'."""
    n_total = len(tokens)
    rng = meta.get("response_token_range")
    lo_r, hi_r = (int(rng[0]), int(min(rng[1], n_total))) if rng else (n_total, n_total)
    if scope == "response":
        return [(i, _func_label(tokens[i])) for i in range(lo_r, hi_r)]
    if scope == "prompt_response":
        scene_of: dict[int, str] = {}
        for sp in meta.get("scene_token_spans", []):
            ts, te = sp.get("tok_start"), sp.get("tok_end")
            if ts is None or te is None:
                continue
            for i in range(ts, te):
                scene_of[i] = sp["scene"]
        return [(i, "response" if i >= lo_r else scene_of.get(i, "prompt_other"))
                for i in range(hi_r)]
    raise ValueError(f"unknown scope {scope!r}")


def _fit_reducer(fit_std: np.ndarray, pca: float | int | None, seed: int):
    """Optional dimensionality reducer for the clustering features, fit on the standardized fit
    subsample. ``pca=None`` -> cluster in the FULL z-scored space (no PCA); a float in (0, 1) ->
    keep the smallest #components explaining that fraction of variance; an int -> that many PCs."""
    if pca is None:
        return None
    if isinstance(pca, float) and 0.0 < pca < 1.0:
        return PCA(n_components=pca, svd_solver="full", random_state=seed).fit(fit_std)
    n = min(int(pca), fit_std.shape[1], fit_std.shape[0])
    return PCA(n_components=n, random_state=seed).fit(fit_std)


def scope_refs(
    run_dirs: list[Path], slug: str, layer: int, tasks: list[str] | None = None
) -> list[tuple[Path, Path]]:
    """The ``(meta.json, layer_<L>.npy)`` pairs the clustering consumes, in the fixed order that
    fixes the shared token order (run dirs as given, tasks as given, scenario dirs sorted)."""
    tasks = tasks or ["identify"]
    refs = []
    for run in run_dirs:
        adir = run / "activations" / slug
        if not adir.exists():
            continue
        for tsk in tasks:
            for pdir in sorted(adir.glob(f"*__{tsk}")):
                mp, npy = pdir / "meta.json", pdir / f"layer_{layer}.npy"
                if mp.exists() and npy.exists():
                    refs.append((mp, npy))
    return refs


def fit_token_transform(
    refs: list[tuple[Path, Path]],
    scope: str,
    seed: int = 0,
    cap_fit: int = 30000,
    pca: float | int | None = 50,
):
    """The ``StandardScaler`` (+ optional PCA reducer) that :func:`load_token_sequences` fits on a
    bounded, seeded token subsample. Exposed separately so a **frozen assigner** can standardize
    NEW residuals with the exact statistics the cached clustering was fit under."""
    rng = np.random.default_rng(seed)
    quota = max(1, cap_fit // len(refs))
    pool = []
    for mp, npy in refs:
        meta = json.loads(mp.read_text(encoding="utf-8"))
        sel = _scope_token_indices(meta, scope, meta.get("tokens") or [])
        if len(sel) < 3:
            continue
        idx = [i for i, _ in sel]
        resid = np.load(npy, mmap_mode="r")
        take = rng.permutation(len(idx))[:quota]
        pool.append(np.asarray(resid[[idx[j] for j in take]], np.float32))
    fit = np.concatenate(pool)
    scaler = StandardScaler().fit(fit)
    reducer = _fit_reducer(scaler.transform(fit), pca, seed)
    return scaler, reducer


def load_token_sequences(
    run_dirs: list[Path],
    slug: str,
    layer: int,
    scope: str,
    seed: int = 0,
    cap_fit: int = 30000,
    tasks: list[str] | None = None,
    pca: float | int | None = 50,
) -> list[dict]:
    """Memory-safe loader for the response / prompt_response scopes: fit a shared StandardScaler
    (+ optional PCA) on a bounded token subsample, then stream each passage through it (raw
    residuals are never all held at once -- reasoning responses are ~3k tokens x100s of passages).
    ``pca`` controls the clustering feature space: ``None`` = full z-scored dims, a float in (0, 1)
    = variance-threshold PCA, an int = fixed #components (default 50 keeps existing callers stable).
    Returns per-passage dicts with pre-projected ``std`` + per-token reference ``labels``. Pass
    several ``tasks`` to pool response tokens across tasks (each passage tagged with its task)."""
    refs = scope_refs(run_dirs, slug, layer, tasks)
    if not refs:
        return []
    scaler, reducer = fit_token_transform(refs, scope, seed, cap_fit, pca)
    seqs = []
    for mp, npy in refs:
        meta = json.loads(mp.read_text(encoding="utf-8"))
        toks = meta.get("tokens") or []
        sel = _scope_token_indices(meta, scope, toks)
        if len(sel) < 3:
            continue
        sel.sort(key=lambda t: t[0])
        idx = [i for i, _ in sel]
        lo, hi = idx[0], idx[-1]
        resid = np.load(npy, mmap_mode="r")
        z = scaler.transform(np.asarray(resid[idx], np.float32))
        std = z if reducer is None else reducer.transform(z)
        seqs.append({
            "schema": meta.get("schema"),
            "task": meta.get("task"),
            "std": std,
            "labels": [lb for _, lb in sel],
            "positions": np.array([(i - lo) / max(hi - lo, 1) for i in idx]),
            "tokens": [_clean(toks[i]) if i < len(toks) else "" for i in idx],
        })
    return seqs


def _purity(meta: np.ndarray, scenes: list) -> float:
    total = 0
    for m in set(meta.tolist()):
        labs = [scenes[i] for i in range(len(scenes)) if meta[i] == m]
        total += Counter(labs).most_common(1)[0][1]
    return round(total / len(scenes), 4)


def analyze_scope(seqs: list[dict], k: int, seed: int, label_word: str, scope: str,
                  force_meta: int | None = None) -> dict:
    """Meta-clustering + metrics for a pre-projected token scope (response / prompt_response),
    generic over the per-token reference ``label``. Mirrors ``analyze_schema``'s output keys so the
    same figures / annotation / embedding code applies (``scene`` <-> generic reference label).
    ``force_meta`` overrides the default meta-count (= #reference labels) for a richer primitive
    set (the response reference labels are coarse, so the default 3 is too few for a graph)."""
    all_lab = [lb for s in seqs for lb in s["labels"]]
    n_lab = len(set(all_lab))
    std = [s["std"] for s in seqs]
    model = KMeansTemporalMetaClustering(
        MetaConfig(n_clusters=k, random_state=seed, force_metaclusters=force_meta or n_lab)
    ).fit(std, [s["positions"] for s in seqs])
    states = [model.kmeans.predict(v) for v in std]
    metas = [model.cluster_to_meta_[s] for s in states]
    all_state = np.concatenate(states)
    all_meta = np.concatenate(metas)
    all_pos = np.concatenate([s["positions"] for s in seqs])
    all_words = [w for s in seqs for w in _reconstruct_words(s["tokens"])]
    scored = scope != "response"  # response reference labels are degenerate -> no ground truth
    n_meta = model.n_metaclusters_
    lab_pos = {lb: float(np.mean([all_pos[i] for i in range(len(all_lab)) if all_lab[i] == lb]))
               for lb in set(all_lab)}
    lab_order = sorted(lab_pos, key=lab_pos.get)
    lab_col = {lb: j for j, lb in enumerate(lab_order)}
    tok_bucket = np.floor(np.clip(all_pos, 0, 0.999) * n_lab).astype(int)
    trans = np.zeros((n_meta, n_meta))
    for m in metas:
        for a, b in zip(m[:-1], m[1:], strict=False):
            trans[a, b] += 1
    trans = trans / (trans.sum(axis=1, keepdims=True) + 1e-9)
    comp = np.zeros((n_meta, n_lab))
    dom, profiles = {}, []
    for mm in range(n_meta):
        sel = all_meta == mm
        wi = np.where(sel)[0]
        labs = [all_lab[i] for i in wi]
        for lb in labs:
            comp[mm, lab_col[lb]] += 1
        dom[mm] = Counter(labs).most_common(1)[0][0] if labs else None
        profiles.append({"meta": mm, "n_tokens": int(sel.sum()), "dom_scene": dom[mm],
                         "mean_pos": round(float(all_pos[sel].mean()), 3) if sel.any() else None,
                         "top": _top_content([all_words[i] for i in wi], 25),
                         "tok_counts": _top_content_counts([all_words[i] for i in wi], 40)})
    comp = comp / (comp.sum(axis=1, keepdims=True) + 1e-9)
    fine = []
    for c in range(k):
        sel = all_state == c
        if not sel.any():
            continue
        wi = np.where(sel)[0]
        labs = [all_lab[i] for i in wi]
        fine.append({"state": int(c), "meta": int(model.cluster_to_meta_[c]),
                     "count": int(sel.sum()),
                     "mean_pos": round(float(all_pos[sel].mean()), 3),
                     "dom_scene": Counter(labs).most_common(1)[0][0] if labs else None,
                     "top": _top_content([all_words[i] for i in wi], 20)})
    fine.sort(key=lambda d: d["mean_pos"])
    rng = np.random.default_rng(seed)
    coh = float(np.mean([_mean_run_len(m) for m in metas]))
    shuf = float(np.mean([_mean_run_len(rng.permutation(m)) for m in metas]))
    rng2 = np.random.default_rng(seed + 1)
    x_all = np.concatenate(std)
    lab_idx_all = np.array([lab_col[lb] for lb in all_lab])
    idx_sub = rng2.permutation(len(x_all))[: min(4000, len(x_all))]
    chosen = _short_med_long([len(s["positions"]) for s in seqs])
    embed = {
        "X": x_all[idx_sub], "meta": all_meta[idx_sub], "pos": all_pos[idx_sub],
        "scene_idx": lab_idx_all[idx_sub],
        "traj": [{"coords": std[i], "pos": seqs[i]["positions"], "meta": metas[i],
                  "tokens": seqs[i]["tokens"]} for i in chosen],
        "traj_full": [],  # scope tokens are already contiguous -> no separate full path
    }
    passages = [{"tokens": s["tokens"], "positions": s["positions"], "meta": metas[i],
                 "state": states[i],
                 "scene_idx": np.array([lab_col[lb] for lb in s["labels"]])}
                for i, s in enumerate(seqs)]
    all_group = [s["schema"] for s in seqs for _ in s["labels"]]
    groups = sorted({g for g in all_group if g is not None})
    schema_comp = ami_schema = None
    if len(groups) > 1:
        gcol = {g: j for j, g in enumerate(groups)}
        gidx = np.array([gcol[g] for g in all_group])
        gmat = np.zeros((n_meta, len(groups)))
        np.add.at(gmat, (all_meta, gidx), 1.0)
        ami_schema = round(float(adjusted_mutual_info_score(all_group, all_meta)), 4)
        gmat = gmat / (gmat.sum(axis=1, keepdims=True) + 1e-9)
        schema_comp = {"schemas": groups, "matrix": gmat.round(4).tolist()}
    all_task = [s.get("task") for s in seqs for _ in s["labels"]]
    tasks_u = sorted({t for t in all_task if t is not None})
    task_comp = ami_task = None
    if len(tasks_u) > 1:
        tcol = {t: j for j, t in enumerate(tasks_u)}
        tidx = np.array([tcol[t] for t in all_task])
        tmat = np.zeros((n_meta, len(tasks_u)))
        np.add.at(tmat, (all_meta, tidx), 1.0)
        ami_task = round(float(adjusted_mutual_info_score(all_task, all_meta)), 4)
        tmat = tmat / (tmat.sum(axis=1, keepdims=True) + 1e-9)
        task_comp = {"tasks": tasks_u, "matrix": tmat.round(4).tolist()}
    return {
        "n_tokens": len(all_lab), "scored": scored, "n_scenes": n_lab,
        "n_metaclusters": int(n_meta),
        "n_metaclusters_auto": int(model.n_metaclusters_auto_),
        "laplacian_eigenvalues": model.eigenvalues_.round(5).tolist(),
        "ami_meta_vs_scene": round(float(adjusted_mutual_info_score(all_lab, all_meta)), 4)
        if scored else None,
        "ami_positionbin_vs_scene": round(
            float(adjusted_mutual_info_score(all_lab, tok_bucket)), 4) if scored else None,
        "purity_meta_vs_scene": _purity(all_meta, all_lab) if scored else None,
        "coherence": {"mean_run": round(coh, 2), "shuffled": round(shuf, 2),
                      "ratio": round(coh / shuf, 2) if shuf else float("nan")},
        "dominant_scene_per_meta": dom,
        "composition": {"scenes": lab_order, "matrix": comp.round(4).tolist()} if scored else None,
        "ami_meta_vs_schema": ami_schema, "schema_composition": schema_comp,
        "ami_meta_vs_task": ami_task, "task_composition": task_comp,
        "meta_profiles": profiles, "fine_states": fine,
        "flow": _phase_flow(metas, n_meta, seed),
        "transition_matrix": trans.round(3).tolist(),
        "meta_transition_matrix": mean_meta_adjacency(metas, n_meta).round(4).tolist(),
        "label_word": label_word, "scope": scope,
        "_plot": {"passages": passages, "scene_order": lab_order, "embed": embed},
    }


def _short_med_long(sizes: list[int]) -> list[int]:
    """Indices of the shortest, median, and longest sequences (deduped, in that order); used to
    pick representative example traces for the raster/trajectory figures instead of only the
    longest one."""
    order = np.argsort(sizes)
    return list(dict.fromkeys(int(order[i]) for i in (0, len(order) // 2, -1)))


_STIM_CTX = (
    "You are analyzing clusters of stimulus passage tokens grouped by an unsupervised temporal "
    "clustering of a language model's residual stream as it reads a short narrative (e.g. a "
    "restaurant visit or an airport trip) describing several people and their roles across ordered "
    "scenes."
)

_SCOPE_CTX = {
    "response": (
        "You are analyzing clusters of tokens from a language model's RESPONSE as it answers a "
        "task: after reading a short narrative (a restaurant visit or an airport trip) it names "
        "the scenario and lists each person with their specific role. Clusters group response "
        "tokens by an unsupervised temporal clustering of the model's residual stream."
    ),
    "prompt_response": (
        "You are analyzing clusters of tokens spanning a language model's full PROMPT + RESPONSE: "
        "it reads a short narrative (restaurant visit or airport trip) about several people and "
        "their roles, then answers. Clusters group tokens by an unsupervised temporal clustering "
        "of the residual stream (so a cluster may capture a reading phase or a generating phase)."
    ),
}


def _annotate_schema(
    r: dict,
    model: str,
    seed: int,
    context: str = _STIM_CTX,
    distinctive: bool = False,
    backend: str = "copilot",
) -> dict:
    """LLM-label each meta-cluster and fine k-means state from its top tokens + example snippets.
    When ``distinctive`` is set, each meta-cluster also gets its tf-idf ``distinctive`` tokens
    (characteristic vs the other metas of this layer) to sharpen the label."""
    rng = np.random.default_rng(seed)
    passages = r["_plot"]["passages"]
    traces = [{"tokens": p["tokens"]} for p in passages]
    metas = [p["meta"] for p in passages]
    states = [p["state"] for p in passages]
    profs = r["meta_profiles"]
    disc = (_distinctive_tokens([p.get("tok_counts", []) for p in profs], 10)
            if distinctive else [None] * len(profs))
    mcl = [{"id": pr["meta"], "top": pr["top"], "distinctive": disc[i],
            "snippets": _snippets(traces, metas, pr["meta"], 10, 10, rng)}
           for i, pr in enumerate(profs)]
    fcl = [{"id": d["state"], "top": d["top"],
            "snippets": _snippets(traces, states, d["state"], 10, 10, rng)}
           for d in r["fine_states"]]
    return {
        "meta": _annotate(mcl, model, "meta-clusters", context, backend),
        "fine_states": _annotate(fcl, model, "fine sub-states", context, backend),
    }
