"""Pool per-(model, task) meta-cluster nodes into ONE cross-source theme merge.

``primitives.depth_metacluster --mode merge`` clusters + LLM-annotates each layer's meta-clusters
and then merges them *within a single (model, task)* into cross-layer THEMES (80 nodes = 10 layers x
8 meta). Good as a per-run inventory, but each run's themes live in their OWN space, so "does model
A use primitive X more than model B" or "is theme Y an identify- or spot_error-thing" cannot be read
off directly.

This exploration runs the SAME semantic merge over the POOLED node set of several runs at once, so
the discovered themes are a **shared vocabulary** across (model, task). Each node keeps its source
tag; after the merge we read a **theme x source** token-mass matrix -- the direct comparison of what
reasoning primitives each model uses on each task.

Design (reuses the shipped merge core, nothing in ``depth_metacluster`` is modified):

* Load each run's cached ``clusters.json`` (per-layer profiles + LLM annotations; produced by the
  ``--annotate`` merge). No GPU/latents needed for pooling -- only the node annotations.
* Concatenate the per-(source, layer) entries into one ``per_layer`` list (force_meta=8 nodes each),
  tracking each node's source. All sources must share ``force_meta`` and the layer set.
* Merge with the three go-forward methods on the pooled nodes:
    - ``semantic-avglink`` -- average-linkage on ``1 - instruction-embedding cosine`` of each node's
      ``label + description``, cut at the silhouette-optimal k (generous ``k_max`` so the pooled
      theme count is data-driven, never forced to the single-run 9-15 band);
    - ``semantic-cpm`` -- the same embeddings through the k-NN Leiden-CPM consensus (silhouette
      AUTO);
    - ``labels`` -- one LLM grouping call over the pooled nodes (consensus over repeats).
  The merges are **blind to the source tag** (embeddings see only label+description; the labels
  prompt omits the model/task) so the grouping is a pure functional grouping and any per-source
  imbalance we read out afterwards is real, not planted.
* Aggregate a theme x source **token-mass fraction** (each source normalised to its own 10*T tokens
  = mean-over-depth share of reasoning tokens on that theme) + node counts, name themes from
  size-weighted member labels, and emit heatmap / bars / depth figures + findings + merge.json.
* Validate that pooling PRESERVES within-source structure: AMI(new pooled node->theme restricted to
  a source, old per-source node->theme) per source and method.

Run (one invocation embeds once and runs every requested method for that version)::

    python -m dev.schema_llm.primitives.crossmodel.merge \
        --source phi4R/identify=.../20260729-cpm_auto_merge/phi4R_identify \
        --source Qwen3/identify=.../20260729-cpm_auto_merge/Qwen3_identify \
        --methods semantic-avglink semantic-cpm labels \
        --old semantic-avglink=_semavg_embed --old semantic-cpm=_semcpm_embed \
        --out results/schema_llm/analysis/crossmodel_merge/20260730-pooled/identify
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
from sklearn.metrics import adjusted_mutual_info_score

from ..depth_metacluster import (
    _consecutive,
    _consensus_partition,
    _cpm_select_from_graph,
    _dedup_theme_names,
    _knn_sparsify,
    _semantic_similarity,
    _silhouette_cut,
)

FORCE_META = 8  # rebound from the sources by _set_force_meta(); all sources must agree
_CPM_GRID = [round(float(g), 3) for g in np.arange(0.05, 0.75, 0.05)]


def _set_force_meta(n: int) -> None:
    """Bind the metaclusters-per-layer the loaded sources were built with.

    Was pinned to the literal 8 that every shipped run used. It is a genuine knob, though
    (``depth_metacluster --meta``), and it is the CAP ON GRANULARITY the pooled merge can ever
    reach: the merge only ever groups whole metaclusters, so if one metacluster already holds a
    third of a layer's tokens, no cut-k can split it and the biggest theme stays huge. Raising
    ``--meta`` upstream is the lever for a finer primitive vocabulary; this makes the pooled merge
    able to consume the result."""
    global FORCE_META  # noqa: PLW0603 - module-level run config, matching the exporter's pattern
    FORCE_META = int(n)


# --------------------------------------------------------------------------- load / pool


def _load_source(name: str, run_dir: Path) -> dict:
    """Load one run's cached per-layer annotations. Returns
    ``{name, slug, task, layers, per_layer}`` where ``per_layer`` is the list the merge core wants
    ({layer, profiles, ann}) with ``ann`` re-keyed to int metas."""
    cj = json.loads((run_dir / "clusters.json").read_text(encoding="utf-8"))
    key = cj["key"]
    per_layer = [
        {"layer": pl["layer"], "profiles": pl["profiles"],
         "ann": {int(mm): vv for mm, vv in pl["ann"].items()}}
        for pl in cj["per_layer"]
    ]
    assert key["force_meta"] == FORCE_META, (
        f"{name}: force_meta {key['force_meta']} != {FORCE_META} (all sources must agree)"
    )
    return {"name": name, "slug": key["slug"], "task": key["tasks"][0], "k": key["k"],
            "layers": [pl["layer"] for pl in per_layer], "per_layer": per_layer}


def _pool(sources: list[dict]) -> dict:
    """Concatenate the sources' per-(layer) entries into one node set. Returns pooled arrays aligned
    to the merge-core flatten order (``for entry in per_layer: for m in range(FORCE_META)``)."""
    layers0 = sources[0]["layers"]
    for s in sources:
        assert s["layers"] == layers0, f"{s['name']} layer set {s['layers']} != {layers0}"
    pooled_per_layer: list[dict] = []
    entry_src: list[int] = []
    for si, s in enumerate(sources):
        for pl in s["per_layer"]:
            pooled_per_layer.append(pl)
            entry_src.append(si)
    node_src, node_layer, node_meta, node_tokens, node_pos = [], [], [], [], []
    node_label, node_desc = [], []
    for ei, pl in enumerate(pooled_per_layer):
        by_meta = {p["meta"]: p for p in pl["profiles"]}
        for m in range(FORCE_META):
            pr, ann = by_meta.get(m, {}), pl["ann"].get(m, {})
            node_src.append(entry_src[ei])
            node_layer.append(pl["layer"])
            node_meta.append(m)
            node_tokens.append(int(pr.get("n_tokens", 0)))
            node_pos.append(pr.get("mean_pos"))
            node_label.append(ann.get("label", ""))
            node_desc.append(ann.get("description", ""))
    n_nodes = len(pooled_per_layer) * FORCE_META
    assert len(node_src) == n_nodes
    return {"per_layer": pooled_per_layer, "src": np.array(node_src), "layer": np.array(node_layer),
            "meta": np.array(node_meta), "tokens": np.array(node_tokens, dtype=float),
            "pos": node_pos, "label": node_label, "desc": node_desc, "n_nodes": n_nodes}


# --------------------------------------------------------------------------- merge methods


def _avglink_k_sweep(z: np.ndarray, d: np.ndarray, k_min: int, k_max: int) -> list[dict]:
    """Silhouette-vs-k sweep behind the ``_silhouette_cut`` choice: for each candidate cut k, the
    selection score (precomputed-distance silhouette) plus theme granularity (count / singletons /
    largest). Makes the operating k an inspectable, robustness-checkable choice (the CPM analog is
    ``gamma_sweep``)."""
    from scipy.cluster.hierarchy import fcluster
    from sklearn.metrics import silhouette_score
    n = d.shape[0]
    sweep = []
    for k in range(k_min, min(k_max, n - 1) + 1):
        lab = fcluster(z, t=k, criterion="maxclust")
        bc = np.bincount(lab)[1:]
        sil = (round(float(silhouette_score(d, lab, metric="precomputed")), 3)
               if 1 < len(np.unique(lab)) < n else None)
        sweep.append({"k": int(k), "n_themes": int((bc > 0).sum()),
                      "n_singletons": int((bc == 1).sum()), "largest": int(bc.max()),
                      "silhouette": sil})
    return sweep


def _merge_semantic_avglink(
    pooled: dict, sim: np.ndarray, k_max: int, merge_k: int | None = None
) -> tuple[np.ndarray, dict]:
    """Average-linkage on ``1 - sim`` cut at the silhouette-optimal k in [4, k_max]; also emits the
    full silhouette-vs-k ``k_sweep`` so the chosen operating k can be checked for robustness.

    ``merge_k`` forces the cut instead of taking the silhouette argmax. That is not a cosmetic
    override: the silhouette curve over a pooled instruction-embedding space is often nearly FLAT
    (on the gameof24 pool it moves only 0.282 -> 0.254 across k = 4..42), so the argmax is
    dominated by noise and lands on ``k_min`` = 4, collapsing a genuinely fine-grained node
    vocabulary into a handful of super-themes. When the sweep is flat, read it and pick a local
    maximum at the granularity the analysis needs -- the sweep is written to ``merge.json`` either
    way, so a forced cut stays auditable."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    d = 1.0 - sim
    d = (d + d.T) / 2
    np.fill_diagonal(d, 0.0)
    np.clip(d, 0.0, None, out=d)
    z = linkage(squareform(d, checks=False), method="average")
    hi = min(k_max, d.shape[0] - 1)
    if merge_k is None:
        labels, k = _silhouette_cut(z, d, k_min=4, k_max=hi)
    else:
        k = int(min(merge_k, d.shape[0] - 1))
        labels = _consecutive(fcluster(z, t=k, criterion="maxclust"))
    return labels, {
        "method": "semantic-avglink",
        "silhouette_cut_k": int(k),
        "k_max": int(k_max),
        "cut": "forced" if merge_k is not None else "silhouette",
        "k_sweep": _avglink_k_sweep(z, d, 4, hi),
    }


def _merge_semantic_cpm(sim: np.ndarray, knn: int, seed: int,
                        stability_floor: float) -> tuple[np.ndarray, dict]:
    """k-NN Leiden-CPM consensus over the embedding cosine, AUTO gamma by silhouette."""
    graph = _knn_sparsify(sim, knn)
    labels, info = _cpm_select_from_graph(graph, _CPM_GRID, None, None, seed, 11, stability_floor,
                                          select_sim=sim)
    info["method"] = "semantic-cpm"
    info["knn"] = knn
    return labels, info


def _pooled_node_lines(pooled: dict) -> list[str]:
    """Per-node lines for the labels prompt -- SOURCE-BLIND (no model/task tag) so the LLM groups
    purely by function, exactly like the embeddings. id, depth pos, label -- description, top."""
    lines = []
    for nid in range(pooled["n_nodes"]):
        pos = pooled["pos"][nid]
        pos_s = f" pos={pos:.2f}" if pos is not None else ""
        lab = pooled["label"][nid] or "?"
        desc = pooled["desc"][nid]
        desc_s = f" -- {desc}" if desc else ""
        ei, m = nid // FORCE_META, pooled["meta"][nid]
        top = ", ".join((pooled["per_layer"][ei]["profiles"][m].get("top", [])
                         if m < len(pooled["per_layer"][ei]["profiles"]) else [])[:8])
        lines.append(f"- id {nid}{pos_s}: {lab}{desc_s} | top: {top}")
    return lines


# Compact member-list output (not the per-node ``assignments`` of the single-run pipeline): with
# 160-320 pooled nodes an id-per-line assignments dump is too long to generate and times out; a
# themes->member-ids response is a few KB and returns reliably. Source-blind (no model/task tag) so
# the grouping is purely functional.
_POOL_LABELS_PROMPT = (
    "Below are meta-clusters of a reasoning model's response tokens, each discovered SEPARATELY "
    "(at some network layer). Each line: numeric id, depth position (pos 0=start..1=end of the "
    "response), a short label -- description, and its top tokens. Group ONLY clusters doing the "
    "SAME reasoning function into a shared THEME; keep a functionally DISTINCT cluster as its OWN "
    "theme. Use AS MANY themes as the data warrants -- do NOT force a small number, but do merge "
    "genuine duplicates. Return ONLY JSON: "
    '{{"themes": [{{"name": "<2-5 word function>", "members": [<id>, <id>, ...]}}]}} '
    "-- every id from 0 to {last} must appear in exactly one theme's members.\n\n{items}"
)


def _parse_members(data: dict, n_nodes: int) -> tuple[np.ndarray, dict, dict]:
    """Validate a compact ``{themes:[{name, members:[ids]}]}`` response -> (labels, names, report).
    Themes are re-indexed to consecutive ids in listing order; coverage-enforced (omitted id -> its
    own singleton, duplicate id -> first theme wins, out-of-range dropped)."""
    from ...core.llm import LLMParseError
    themes = data.get("themes")
    if not isinstance(themes, list) or not themes:
        raise LLMParseError("pooled labels response missing a non-empty 'themes' list")
    labels = np.full(n_nodes, -1, dtype=int)
    names: dict = {}
    report = {"unassigned": 0, "dropped": 0, "duplicates": 0}
    for t, th in enumerate(themes):
        names[t] = str(th.get("name", f"theme {t}")).strip()
        for raw in th.get("members") or []:
            try:
                nid = int(raw)
            except (TypeError, ValueError):
                report["dropped"] += 1
                continue
            if not (0 <= nid < n_nodes):
                report["dropped"] += 1
            elif labels[nid] != -1:
                report["duplicates"] += 1
            else:
                labels[nid] = t
    nxt = int(labels.max()) + 1 if (labels >= 0).any() else 0
    for nid in np.where(labels < 0)[0]:
        labels[nid] = nxt
        names[nxt] = f"ungrouped node {nid}"
        nxt += 1
        report["unassigned"] += 1
    return labels, names, report


def _labels_once_pooled(prompt: str, n_nodes: int, model: str,
                        timeout: int) -> tuple[np.ndarray | None, dict, dict]:
    """One compact-format LLM merge call -> coverage-validated labels (None on failure) + names."""
    from ...core.llm import extract_json, run_copilot_cli
    try:
        data = extract_json(run_copilot_cli(prompt, model=model, timeout=timeout),
                            require_key="themes")
        return _parse_members(data, n_nodes)
    except (ValueError, KeyError, TypeError, RuntimeError) as e:
        print(f"[pooled labels run failed: {e}]")
        return None, {}, {"unassigned": 0, "dropped": 0, "duplicates": 0}


def _merge_labels(pooled: dict, model: str, consensus_runs: int,
                  timeout: int) -> tuple[np.ndarray, dict]:
    """Consensus of source-blind LLM grouping call(s) over the pooled nodes (compact member-list
    output). Returns (consecutive node->theme, info) with per-run raw names carried when a single
    run is used (consensus themes are named downstream from member labels)."""
    lines = _pooled_node_lines(pooled)
    n = pooled["n_nodes"]
    prompt = _POOL_LABELS_PROMPT.format(last=n - 1, items="\n".join(lines))
    runs = [_labels_once_pooled(prompt, n, model, timeout) for _ in range(max(1, consensus_runs))]
    good = [r for r in runs if r[0] is not None]
    if not good:
        raise SystemExit("all labels merge runs failed")
    rep = {k: sum(g[2].get(k, 0) for g in good) for k in ("unassigned", "dropped", "duplicates")}
    if len(good) == 1:
        raw, names = good[0][0], good[0][1]
        info = {"method": "labels", "n_runs": 1, "stability": 1.0, "repairs": rep}
    else:
        raw, stab = _consensus_partition([g[0] for g in good])
        names = {}
        info = {"method": "labels", "n_runs": len(good), "stability": stab, "repairs": rep}
    uniq, theme = np.unique(raw, return_inverse=True)
    remap = {new: names.get(int(old), "") for new, old in enumerate(uniq)}
    info["llm_names"] = remap
    return _consecutive(theme), info


# --------------------------------------------------------------------------- aggregation


def _theme_source_matrix(pooled: dict, node_theme: np.ndarray, n_src: int,
                         n_themes: int) -> dict:
    """theme x source token mass + node counts, plus per-source-normalised fraction."""
    mass = np.zeros((n_themes, n_src))
    counts = np.zeros((n_themes, n_src), dtype=int)
    for nid in range(pooled["n_nodes"]):
        t, s = int(node_theme[nid]), int(pooled["src"][nid])
        mass[t, s] += pooled["tokens"][nid]
        counts[t, s] += 1
    src_total = mass.sum(0)
    frac = mass / np.where(src_total > 0, src_total, 1.0)
    assert np.allclose(frac.sum(0), 1.0, atol=1e-6) or n_themes == 0
    return {"mass": mass, "counts": counts, "frac": frac, "src_total": src_total}


def _theme_profiles(pooled: dict, node_theme: np.ndarray, n_themes: int,
                    names: dict | None = None) -> list[dict]:
    """Aggregate each theme: size-weighted unique name, top tokens, mean depth, source-tagged
    members, per-source token fraction."""
    src_names = pooled["source_names"]
    out = []
    for t in range(n_themes):
        members, label_w, tops = [], Counter(), Counter()
        pos_num = pos_den = 0.0
        for nid in np.where(node_theme == t)[0]:
            nid = int(nid)
            s, layer, m = pooled["src"][nid], pooled["layer"][nid], pooled["meta"][nid]
            sz = pooled["tokens"][nid]
            members.append(f"{src_names[s]}:L{layer}m{m}")
            if pooled["label"][nid]:
                label_w[pooled["label"][nid]] += sz
            if pooled["pos"][nid] is not None:
                pos_num += pooled["pos"][nid] * sz
                pos_den += sz
            ei = nid // FORCE_META
            for w in (pooled["per_layer"][ei]["profiles"][m].get("top", [])
                      if m < len(pooled["per_layer"][ei]["profiles"]) else [])[:10]:
                tops[w] += sz
        name = (names or {}).get(t) or (label_w.most_common(1)[0][0] if label_w else f"theme {t}")
        out.append({"theme": t, "name": name, "n_members": len(members), "members": members,
                    "member_labels": [lb for lb, _ in label_w.most_common(6)],
                    "mean_pos": round(pos_num / pos_den, 3) if pos_den else None,
                    "top": [w for w, _ in tops.most_common(12)]})
    _dedup_theme_names(out)
    return out


def _old_comparison(sources: list[dict], pooled: dict, node_theme: np.ndarray,
                    old_suffix: str | None, method: str) -> dict:
    """AMI between the pooled node->theme (restricted to each source's nodes) and that source's OLD
    per-run node->theme (from its own merge.json), to show pooling preserves within-source
    structure. ``old_suffix`` locates the per-source dir (run_dir.name + suffix); None -> skip."""
    if old_suffix is None:
        return {}
    per_src = {}
    for si, s in enumerate(sources):
        old_dir = s["run_dir"].parent / (s["run_dir"].name + old_suffix)
        mj = old_dir / "merge.json"
        if not mj.exists():
            per_src[s["name"]] = {"ami": None, "note": f"missing {mj}"}
            continue
        old = json.loads(mj.read_text(encoding="utf-8"))
        old_flat = np.array(old["node_theme"]).reshape(-1)  # layer-major, same order as pooled
        new_flat = node_theme[pooled["src"] == si]
        assert old_flat.shape == new_flat.shape, f"{s['name']} {old_flat.shape}!={new_flat.shape}"
        ami = round(float(adjusted_mutual_info_score(old_flat, new_flat)), 3)
        per_src[s["name"]] = {"ami": ami, "old_method": old.get("method"),
                              "old_n_themes": old.get("n_themes")}
    return per_src


# --------------------------------------------------------------------------- figures


def _src_colors(n: int) -> list:
    base = plt.get_cmap("tab10").colors
    return [base[i % len(base)] for i in range(n)]


def _heatmap_figure(themes: list[dict], tsm: dict, src_names: list[str], path: Path,
                    title: str) -> None:
    frac = tsm["frac"]
    order = sorted(range(len(themes)), key=lambda t: (themes[t]["mean_pos"] is None,
                                                      themes[t]["mean_pos"] or 0))
    frac = frac[order]
    ylab = [f"{themes[t]['name'][:34]}  (p{themes[t]['mean_pos']})" for t in order]
    fig, ax = plt.subplots(figsize=(1.6 + 1.5 * len(src_names), 0.5 + 0.42 * len(order)))
    im = ax.imshow(frac, aspect="auto", cmap="magma", vmin=0)
    ax.set_xticks(range(len(src_names)))
    ax.set_xticklabels(src_names, rotation=25, ha="right", fontsize=8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(ylab, fontsize=7)
    for i in range(frac.shape[0]):
        for j in range(frac.shape[1]):
            v = frac[i, j]
            if v >= 0.005:
                ax.text(j, i, f"{v*100:.0f}", ha="center", va="center", fontsize=6,
                        color="white" if v < frac.max() * 0.6 else "black")
    ax.set_title(f"{title}\ntheme token-mass share per source (%, columns sum to 100)", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _bars_figure(themes: list[dict], tsm: dict, src_names: list[str], path: Path,
                 title: str) -> None:
    frac = tsm["frac"]
    order = sorted(range(len(themes)), key=lambda t: -tsm["mass"].sum(1)[t])
    frac = frac[order]
    names = [themes[t]["name"][:30] for t in order]
    x = np.arange(len(order))
    w = 0.8 / len(src_names)
    cols = _src_colors(len(src_names))
    fig, ax = plt.subplots(figsize=(2 + 0.55 * len(order), 4.2))
    for j, sn in enumerate(src_names):
        ax.bar(x + j * w, frac[:, j] * 100, w, label=sn, color=cols[j])
    ax.set_xticks(x + 0.4 - w / 2)
    ax.set_xticklabels(names, rotation=55, ha="right", fontsize=7)
    ax.set_ylabel("token-mass share of source (%)")
    ax.set_title(f"{title}: theme usage by source", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- findings


def _specificity(tsm: dict, src_names: list[str], themes: list[dict],
                 min_frac: float = 0.03) -> dict:
    """Classify each theme by which sources USE it (>= ``min_frac`` of THEIR OWN tokens -- a
    length-normalised share, so a long-CoT model does not look dominant just by mass). A theme is
    'universal' if all sources use it, 'shared' if >=2, else source-specific. ``emphasis`` = the
    source with the largest own-token fraction (which model/task leans on it most)."""
    frac = tsm["frac"]
    rows = []
    for t in range(len(themes)):
        fr = frac[t]
        users = [src_names[j] for j in range(len(src_names)) if fr[j] >= min_frac]
        top = int(np.argmax(fr))
        rows.append({"theme": t, "name": themes[t]["name"], "mean_pos": themes[t]["mean_pos"],
                     "users": users, "n_users": len(users),
                     "emphasis_src": src_names[top], "emphasis_frac": round(float(fr[top]), 3),
                     "src_frac": {src_names[j]: round(float(fr[j]), 3)
                                  for j in range(len(src_names))}})
    return {"rows": rows, "n_universal": sum(r["n_users"] == len(src_names) for r in rows),
            "n_shared": sum(r["n_users"] >= 2 for r in rows),
            "n_specific": sum(r["n_users"] == 1 for r in rows)}


def _write_findings(path: Path, method: str, pooled: dict, themes: list[dict], tsm: dict,
                    spec: dict, info: dict, old_ami: dict, title: str) -> None:
    src_names = pooled["source_names"]
    L = [f"# Pooled cross-source merge — {title} — `{method}`", ""]
    L.append(f"- Sources ({len(src_names)}): {', '.join(src_names)}")
    L.append(f"- Pooled nodes: {pooled['n_nodes']} ({len(src_names)} sources x "
             f"{pooled['n_nodes'] // len(src_names)} each) → **{len(themes)} shared themes** "
             f"({info.get('silhouette_cut_k') or info.get('resolution') or ''})")
    L.append(f"- Theme specificity: {spec['n_universal']} universal (all sources ≥3%), "
             f"{spec['n_shared']} shared (≥2), {spec['n_specific']} source-specific (1)")
    if old_ami:
        amis = [v["ami"] for v in old_ami.values() if v.get("ami") is not None]
        if amis:
            per = ", ".join(f"{k}={v['ami']}" for k, v in old_ami.items()
                            if v.get("ami") is not None)
            L.append(f"- Within-source AMI vs old per-run themes: mean {np.mean(amis):.3f} ({per})")
    L.append("")
    L.append("## Themes (depth-ordered) — token-mass share per source (%)")
    L.append("")
    hdr = "| theme | pos | n | " + " | ".join(src_names) + " | top tokens |"
    L.append(hdr)
    L.append("|" + "---|" * (4 + len(src_names)))
    order = sorted(range(len(themes)), key=lambda t: (themes[t]["mean_pos"] is None,
                                                      themes[t]["mean_pos"] or 0))
    for t in order:
        fr = " | ".join(f"{tsm['frac'][t, j]*100:.0f}" for j in range(len(src_names)))
        top = ", ".join(themes[t]["top"][:7])
        L.append(f"| {themes[t]['name'][:40]} | {themes[t]['mean_pos']} | {themes[t]['n_members']} "
                 f"| {fr} | {top} |")
    L.append("")
    L.append("## Source-specific themes (used by only ONE source, >=3% of its own tokens)")
    L.append("")
    specific = [r for r in spec["rows"] if r["n_users"] == 1]
    for r in sorted(specific, key=lambda r: -r["emphasis_frac"]):
        fr = ", ".join(f"{k} {v*100:.0f}%" for k, v in r["src_frac"].items())
        L.append(f"- **{r['name']}** (pos {r['mean_pos']}): only {r['emphasis_src']} "
                 f"({r['emphasis_frac']*100:.0f}% of its tokens) — [{fr}]")
    if not specific:
        L.append("- (none — every theme is used by >=2 sources)")
    L.append("")
    L.append("## Most source-skewed SHARED themes (own-token-fraction ratio, length-normalised)")
    L.append("")
    def _skew(r):
        vals = [v for v in r["src_frac"].values() if v > 0]
        return (max(r["src_frac"].values()) / min(vals)) if len(vals) >= 2 and r["n_users"] >= 2 \
            else 0.0
    shared = [r for r in spec["rows"] if r["n_users"] >= 2]
    for r in sorted(shared, key=_skew, reverse=True)[:8]:
        fr = ", ".join(f"{k} {v*100:.0f}%" for k, v in r["src_frac"].items())
        L.append(f"- **{r['name']}** (pos {r['mean_pos']}): {_skew(r):.1f}x — [{fr}]")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- driver


def run_method(
    method: str,
    pooled: dict,
    sources: list[dict],
    sim: np.ndarray | None,
    out_dir: Path,
    title: str,
    args: argparse.Namespace,
    old_suffix: dict,
) -> dict:
    """Run one merge method on the pooled nodes; write per-method outputs; return summary dict."""
    if method == "semantic-avglink":
        node_theme, info = _merge_semantic_avglink(pooled, sim, args.k_max, args.merge_k)
    elif method == "semantic-cpm":
        node_theme, info = _merge_semantic_cpm(sim, args.merge_knn, args.seed, args.stability_floor)
    elif method == "labels":
        node_theme, info = _merge_labels(
            pooled, args.annotate_model, args.consensus, args.labels_timeout
        )
    else:
        raise SystemExit(f"unknown method {method}")
    n_themes = int(node_theme.max()) + 1
    n_src = len(sources)
    tsm = _theme_source_matrix(pooled, node_theme, n_src, n_themes)
    llm_names = {int(k): v for k, v in info.get("llm_names", {}).items() if v} or None
    themes = _theme_profiles(pooled, node_theme, n_themes, llm_names)
    spec = _specificity(tsm, pooled["source_names"], themes)
    old_ami = _old_comparison(sources, pooled, node_theme, old_suffix.get(method), method)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_singleton = sum(int((node_theme == t).sum()) == 1 for t in range(n_themes))
    stats = {"title": title, "method": method, "sources": pooled["source_names"],
             "n_nodes": pooled["n_nodes"], "n_themes": n_themes, "n_singletons": n_singleton,
             "merge_info": info, "specificity": spec,
             "theme_source_frac": tsm["frac"].round(4).tolist(),
             "theme_source_mass": tsm["mass"].round(1).tolist(),
             "theme_source_counts": tsm["counts"].tolist(),
             "source_totals": tsm["src_total"].round(1).tolist(),
             "old_within_source_ami": old_ami,
             "node_theme": node_theme.tolist(),
             "node_source": pooled["src"].tolist(), "node_layer": pooled["layer"].tolist(),
             "node_meta": pooled["meta"].tolist(),
             "node_pos": [None if p is None else round(float(p), 4) for p in pooled["pos"]],
             "node_tokens": pooled["tokens"].tolist(), "themes": themes}
    (out_dir / "merge.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    _heatmap_figure(themes, tsm, pooled["source_names"], out_dir / "theme_by_source.png", title)
    _bars_figure(themes, tsm, pooled["source_names"], out_dir / "theme_source_bars.png", title)
    _write_findings(out_dir / "findings.md", method, pooled, themes, tsm, spec, info, old_ami,
                    title)
    amis = [v["ami"] for v in old_ami.values() if v.get("ami") is not None]
    print(f"  [{method}] {n_themes} themes ({n_singleton} singletons); "
          f"universal={spec['n_universal']} shared={spec['n_shared']} specific={spec['n_specific']}"
          + (f"; mean within-src AMI={np.mean(amis):.3f}" if amis else "")
          + f" -> {out_dir}")
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", action="append", required=True,
                   help="name=run_dir (run_dir holds clusters.json); repeatable")
    p.add_argument("--methods", nargs="+",
                   default=["semantic-avglink", "semantic-cpm", "labels"])
    p.add_argument("--old", action="append", default=[],
                   help="method=suffix -> compare vs old per-run dir (run_dir.name+suffix)")
    p.add_argument("--k-max", type=int, default=60,
                   help="avglink silhouette cut ceiling (generous: data-driven theme count)")
    p.add_argument(
        "--merge-k",
        type=int,
        default=None,
        help="force the avglink cut at exactly this many themes instead of the "
        "silhouette argmax. Use when the k_sweep in merge.json is flat (the argmax "
        "is then noise and pins k to the k_min=4 floor, over-merging a "
        "fine-grained node vocabulary); pick a local maximum of the sweep.",
    )
    p.add_argument("--merge-knn", type=int, default=6)
    p.add_argument("--stability-floor", type=float, default=0.9)
    p.add_argument("--consensus", type=int, default=3, help="labels consensus runs")
    p.add_argument("--labels-timeout", type=int, default=600,
                   help="per-call Copilot CLI timeout for the labels merge (large pooled prompt)")
    p.add_argument("--embed-model", default="Qwen/Qwen3-Embedding-4B")
    p.add_argument("--annotate-model", default="claude-opus-4.8")
    p.add_argument("--title", default="pooled")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    sources = []
    for spec in args.source:
        name, _, d = spec.partition("=")
        if not sources:  # the first source defines it; _load_source asserts the rest agree
            _set_force_meta(
                json.loads((Path(d) / "clusters.json").read_text(encoding="utf-8"))["key"][
                    "force_meta"
                ]
            )
        s = _load_source(name, Path(d))
        s["run_dir"] = Path(d)
        sources.append(s)
    print(
        f"[pool] {len(sources)} sources: "
        + ", ".join(f"{s['name']}(k={s['k']})" for s in sources)
        + f"; meta={FORCE_META}/layer"
    )
    pooled = _pool(sources)
    pooled["source_names"] = [s["name"] for s in sources]
    old_suffix = dict(o.partition("=")[::2] for o in args.old)

    sim = None
    embed_model = None if args.embed_model.lower() == "none" else args.embed_model
    if any(m in ("semantic-avglink", "semantic-cpm") for m in args.methods):
        backend = embed_model or "TF-IDF (CPU)"
        print(f"[embed] {pooled['n_nodes']} node annotations via {backend} ...")
        sim = _semantic_similarity(pooled["per_layer"], FORCE_META, embed_model)
        assert sim.shape == (pooled["n_nodes"], pooled["n_nodes"])

    out_root = Path(args.out)
    summary = {}
    for method in args.methods:
        sub = out_root / method.replace("semantic-", "sem")
        summary[method] = run_method(method, pooled, sources, sim, sub, args.title, args,
                                     old_suffix)
    out_root.mkdir(parents=True, exist_ok=True)
    spath = out_root / "summary.json"
    # merge into any existing summary so a partial (e.g. semantic-only) re-run does not drop the
    # methods it did not re-run (e.g. the slow labels entry).
    prev = json.loads(spath.read_text(encoding="utf-8")) if spath.exists() else {}
    methods_summary = prev.get("methods", {})
    methods_summary.update({m: {"n_themes": summary[m]["n_themes"],
                                "n_singletons": summary[m]["n_singletons"],
                                "specificity": {k: summary[m]["specificity"][k]
                                                for k in ("n_universal", "n_shared", "n_specific")}}
                            for m in args.methods})
    spath.write_text(json.dumps(
        {"title": args.title, "sources": pooled["source_names"], "n_nodes": pooled["n_nodes"],
         "methods": methods_summary}, indent=2), encoding="utf-8")
    print(f"[done] {args.title}: "
          + "; ".join(f"{m}={summary[m]['n_themes']}th" for m in args.methods))


if __name__ == "__main__":
    main()
