"""Mean-difference steering vectors for reasoning primitives (SCHEMA_LATENT_GEOMETRY_PLAN D2,
extraction half).

For each **meta-cluster (primitive)** recovered by the response-scope temporal meta-clustering
(:mod:`metacluster` ``--scope response``), extract a residual-stream **steering vector** (the
ActAdd / contrastive-activation-addition family): the mean layer-``L`` residual over that
primitive's response tokens, and -- the injectable form -- its **contrast** against the overall
response mean (``v_m = mean_m - mean_all``), which cancels the shared residual substrate (massive-
activation dims, the BOS sink, the generic "reasoning text" direction) that dominates an absolute
mean. The vector is directly **injectable** at layer ``L`` (add ``alpha * v`` to the residual
during generation, the :mod:`intervene` step).

NB naming: this is deliberately *not* the Todd et al. / algotrace **function vector**, which sums
the residual writes of a few causally-selected attention heads (``sum_(L,H) W^O_L a_(L,H)``). That
head-projection construction lives in :mod:`funcvec_heads`. Keep the two distinct: "steering
vector" (this module, residual mean-difference) vs "function vector" (head projection).

Beyond extracting the vectors it characterizes their geometry, entirely from the saved residuals
(CPU, no re-capture):

- **Distinctness** -- pairwise cosine between the (standardized) primitive vectors: do the
  primitives occupy *different* residual directions, or collapse onto one axis?
- **Function-code alignment** -- what fraction of each primitive vector lies in the comprehension
  **role** subspace vs the **scene** subspace (both built from the *stimulus* captures via
  :func:`latents.load_captures`), against the random-direction chance ``sqrt(r/D)``. A primitive
  whose vector loads on the role subspace is doing *role-binding*-like work; one loading on the
  scene subspace is doing *scenario / latent-inference*-like work -- a quantitative complement to
  the LLM labels.
- **Nearest role** -- the role centroid each primitive vector points most toward.

Saves the raw + standardized + **contrast** vectors to ``.npy`` (the contrast file
``function_vectors_contrast_<schema>.npy`` is the one to inject via :mod:`intervene`), a
per-primitive ``funcvec_<schema>.json``, ``stats.json``, ``findings.md``, and two figures per
schema.

Example::

    python -m dev.schema_llm.primitives.funcvec \
        --run results/schema_llm/data/main_36/restaurant_main \
              results/schema_llm/data/main_36/airport_main \
              results/schema_llm/data/main_72/restaurant_main \
              results/schema_llm/data/main_72/airport_main \
        --slug Phi-4-reasoning --layer 17 --k 30 --meta 8 \
        --annotations <metacluster-out>/annotations.json \
        --out results/schema_llm/analysis/identify/20260717-funcvec_Phi-4-reasoning_response_216
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ..geometry.latents import load_captures
from ._text import _clean, _top_content
from .metacluster import (
    KMeansTemporalMetaClustering,
    MetaConfig,
    _scope_token_indices,
)

_MAX_TOK_BUF = 6000  # cap per-primitive token strings kept for the top-token profile


def _response_refs(run_dirs: list[Path], slug: str, layer: int,
                   task: str = "identify") -> list[tuple[Path, Path, str]]:
    """``(meta_path, npy_path, schema)`` per ``task`` probe with a captured layer + response."""
    out: list[tuple[Path, Path, str]] = []
    for run in run_dirs:
        adir = run / "activations" / slug
        if not adir.exists():
            continue
        for pdir in sorted(adir.glob(f"*__{task}")):
            mp, npy = pdir / "meta.json", pdir / f"layer_{layer}.npy"
            if mp.exists() and npy.exists():
                schema = json.loads(mp.read_text(encoding="utf-8")).get("schema")
                out.append((mp, npy, schema))
    return out


def fit_primitives(refs: list[tuple[Path, Path, str]], meta: int, k: int, seed: int,
                   cap_fit: int = 30000) -> tuple:
    """Fit the same StandardScaler + PCA(50) + temporal meta-clustering as ``metacluster
    --scope response`` on one schema's response tokens (memory-safe: scaler/PCA on a bounded
    subsample, then stream-project each passage). Returns ``(scaler, pca, model, passages)`` where
    each passage carries its response-token ``idx`` / ``positions`` / ``tokens`` / projected
    ``std``."""
    rng = np.random.default_rng(seed)
    quota = max(1, cap_fit // max(len(refs), 1))
    passages: list[dict] = []
    pool: list[np.ndarray] = []
    for mp, npy, _schema in refs:
        m = json.loads(mp.read_text(encoding="utf-8"))
        toks = m.get("tokens") or []
        sel = _scope_token_indices(m, "response", toks)
        if len(sel) < 3:
            continue
        sel.sort(key=lambda t: t[0])
        idx = [i for i, _ in sel]
        lo, hi = idx[0], idx[-1]
        resid = np.load(npy, mmap_mode="r")
        take = rng.permutation(len(idx))[:quota]
        pool.append(np.asarray(resid[[idx[j] for j in take]], np.float32))
        passages.append({
            "npy": npy, "idx": idx,
            "positions": np.array([(i - lo) / max(hi - lo, 1) for i in idx]),
            "tokens": [_clean(toks[i]) if i < len(toks) else "" for i in idx],
        })
    if not passages:
        raise SystemExit("no response tokens to fit")
    fit = np.concatenate(pool)
    scaler = StandardScaler().fit(fit)
    pca = PCA(n_components=min(50, fit.shape[1], fit.shape[0]),
              random_state=seed).fit(scaler.transform(fit))
    del pool, fit
    stds = []
    for p in passages:
        resid = np.load(p["npy"], mmap_mode="r")
        p["std"] = pca.transform(scaler.transform(np.asarray(resid[p["idx"]], np.float32)))
        stds.append(p["std"])
    model = KMeansTemporalMetaClustering(
        MetaConfig(n_clusters=k, random_state=seed, force_metaclusters=meta)
    ).fit(stds, [p["positions"] for p in passages])
    return scaler, pca, model, passages


def extract_vectors(passages: list[dict], scaler: StandardScaler,
                    model: KMeansTemporalMetaClustering, n_meta: int) -> tuple:
    """Per-primitive mean residual (raw) + mean standardized residual (for geometry) + the
    injectable mean-difference contrast vectors (global = primitive-minus-overall, loo =
    primitive-minus-other-primitives) + overall baseline + token counts + top content tokens,
    streamed over the passages."""
    dim = int(np.load(passages[0]["npy"], mmap_mode="r").shape[1])
    raw_sum = np.zeros((n_meta, dim))
    std_sum = np.zeros((n_meta, dim))
    cnt = np.zeros(n_meta)
    tokbuf: list[list[str]] = [[] for _ in range(n_meta)]
    for p in passages:
        resid = np.asarray(np.load(p["npy"], mmap_mode="r")[p["idx"]], np.float32)
        stdz = scaler.transform(resid)
        metas = model.meta_labels(p["std"])
        for m in range(n_meta):
            selm = metas == m
            if selm.any():
                raw_sum[m] += resid[selm].sum(0)
                std_sum[m] += stdz[selm].sum(0)
                cnt[m] += int(selm.sum())
        for tok, mm in zip(p["tokens"], metas, strict=False):
            if len(tokbuf[mm]) < _MAX_TOK_BUF:
                tokbuf[mm].append(tok)
    denom = np.maximum(cnt[:, None], 1.0)
    fv_raw = raw_sum / denom
    fv_std = std_sum / denom
    # overall response mean (baseline) + mean-difference contrast vectors (the injectable form):
    # global = primitive - overall; loo = primitive - mean of the OTHER primitives' tokens. The
    # contrast cancels the shared residual substrate an absolute mean is dominated by.
    total = max(float(cnt.sum()), 1.0)
    fv_overall = raw_sum.sum(0) / total
    fv_contrast = fv_raw - fv_overall
    oth_den = np.maximum(total - cnt, 1.0)[:, None]
    fv_contrast_loo = fv_raw - (raw_sum.sum(0)[None, :] - raw_sum) / oth_den
    top = [_top_content(tokbuf[m], 10) for m in range(n_meta)]
    return fv_raw, fv_std, fv_contrast, fv_contrast_loo, fv_overall, cnt.astype(int), top


def _subspace(vec_by_label: dict[str, list[np.ndarray]], scaler: StandardScaler) -> tuple:
    """Orthonormal basis (rows) of the standardized between-label centroid deviations, plus those
    centroid deviations. ``r`` = number of independent centroid directions (<= #labels - 1)."""
    labs = sorted(vec_by_label)
    if len(labs) < 2:
        dim = scaler.mean_.shape[0]
        return np.zeros((0, dim)), np.zeros((0, dim)), labs
    cents = np.stack([scaler.transform(np.stack(vec_by_label[lb])).mean(0) for lb in labs])
    cents = cents - cents.mean(0, keepdims=True)
    _u, s, vt = np.linalg.svd(cents, full_matrices=False)
    tol = 1e-8 * (s[0] if s.size and s[0] > 0 else 1.0)
    return vt[s > tol], cents, labs


def _proj_frac(v: np.ndarray, basis: np.ndarray) -> float:
    nv = float(np.linalg.norm(v))
    if nv < 1e-12 or basis.shape[0] == 0:
        return 0.0
    return float(np.linalg.norm(basis @ v) / nv)


def _cos_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return an @ bn.T


def analyze_schema(run_dirs: list[Path], slug: str, layer: int, schema: str,
                   refs: list[tuple[Path, Path, str]], referents, scene_tokens, meta: int, k: int,
                   seed: int, annot: dict | None) -> dict:
    """Extract + characterize the steering vectors for one schema."""
    scaler, _pca, model, passages = fit_primitives(refs, meta, k, seed)
    n_meta = model.n_metaclusters_
    fv_raw, fv_std, fv_contrast, fv_contrast_loo, fv_overall, cnt, top = extract_vectors(
        passages, scaler, model, n_meta)
    role_by: dict[str, list[np.ndarray]] = defaultdict(list)
    for r in referents:
        role_by[r.role_id].append(r.vector)
    scene_by: dict[str, list[np.ndarray]] = defaultdict(list)
    for st in scene_tokens:
        scene_by[st.scene].append(st.vector)
    role_basis, role_cents, role_labs = _subspace(role_by, scaler)
    scene_basis, _, _ = _subspace(scene_by, scaler)
    dim = fv_std.shape[1]
    role_chance = float(np.sqrt(role_basis.shape[0] / dim)) if dim else 0.0
    scene_chance = float(np.sqrt(scene_basis.shape[0] / dim)) if dim else 0.0
    cos = _cos_rows(fv_std, fv_std)
    off = cos[~np.eye(n_meta, dtype=bool)]
    cos_contrast = _cos_rows(fv_contrast, fv_contrast)
    off_c = cos_contrast[~np.eye(n_meta, dtype=bool)]
    labels = _fv_labels(annot, top, n_meta)
    role_cos = _cos_rows(fv_std, role_cents) if role_cents.shape[0] else np.zeros((n_meta, 0))
    records = []
    for m in range(n_meta):
        rc = role_cos[m]
        nearest = (role_labs[int(np.argmax(rc))], round(float(rc.max()), 3)) if rc.size else None
        records.append({
            "meta": m, "label": labels[m], "n_tokens": int(cnt[m]),
            "norm_std": round(float(np.linalg.norm(fv_std[m])), 3),
            "norm_raw": round(float(np.linalg.norm(fv_raw[m])), 3),
            "norm_contrast": round(float(np.linalg.norm(fv_contrast[m])), 3),
            "cos_raw_contrast": round(float(_cos_rows(fv_raw[m:m + 1], fv_contrast[m:m + 1])[0, 0]),
                                     3),
            "role_alignment": round(_proj_frac(fv_std[m], role_basis), 3),
            "scene_alignment": round(_proj_frac(fv_std[m], scene_basis), 3),
            "nearest_role": nearest, "top": top[m],
        })
    return {
        "schema": schema, "n_metaclusters": int(n_meta),
        "n_metaclusters_auto": int(model.n_metaclusters_auto_),
        "role_subspace_rank": int(role_basis.shape[0]),
        "scene_subspace_rank": int(scene_basis.shape[0]),
        "role_alignment_chance": round(role_chance, 3),
        "scene_alignment_chance": round(scene_chance, 3),
        "mean_abs_offdiag_cosine": round(float(np.mean(np.abs(off))), 3) if off.size else 0.0,
        "mean_abs_offdiag_cosine_contrast": round(float(np.mean(np.abs(off_c))), 3)
        if off_c.size else 0.0,
        "cosine_matrix": cos.round(3).tolist(),
        "primitives": records,
        "_fv_raw": fv_raw, "_fv_std": fv_std, "_fv_contrast": fv_contrast,
        "_fv_contrast_loo": fv_contrast_loo, "_fv_overall": fv_overall, "_labels": labels,
    }


def _fv_labels(annot: dict | None, top: list[list[str]], n_meta: int) -> list[str]:
    out = []
    for m in range(n_meta):
        a = (annot or {}).get(str(m)) or (annot or {}).get(m)
        if a and a.get("label"):
            out.append(f"{a['label']}")
        else:
            out.append(" ".join(top[m][:3]) if top[m] else "?")
    return out


def _cosine_figure(res: dict, path: Path, slug: str, layer: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cos = np.asarray(res["cosine_matrix"])
    labs = res["_labels"]
    n = len(cos)
    fig, ax = plt.subplots(figsize=(max(6.0, 0.7 * n + 3), max(5.0, 0.7 * n + 2)))
    im = ax.imshow(cos, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(n), [f"{i}" for i in range(n)], fontsize=8)
    ax.set_yticks(range(n), [f"{i}: {labs[i]}" for i in range(n)], fontsize=7)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cos[i, j]:.2f}", ha="center", va="center", fontsize=6,
                    color="k" if abs(cos[i, j]) < 0.6 else "w")
    fig.colorbar(im, ax=ax, label="cosine (standardized function vectors)")
    ax.set_title(f"function-vector similarity -- {res['schema']} -- {slug} L{layer}\n"
                 f"mean |off-diagonal cosine| = {res['mean_abs_offdiag_cosine']}")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _alignment_figure(res: dict, path: Path, slug: str, layer: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prims = res["primitives"]
    n = len(prims)
    labs = res["_labels"]
    role = [p["role_alignment"] for p in prims]
    scene = [p["scene_alignment"] for p in prims]
    x = np.arange(n)
    fig, ax = plt.subplots(figsize=(max(8.0, 0.9 * n + 2), 6))
    ax.bar(x - 0.2, role, 0.4, label="role-subspace", color="#4C72B0")
    ax.bar(x + 0.2, scene, 0.4, label="scene-subspace", color="#DD8452")
    ax.axhline(res["role_alignment_chance"], ls="--", color="#4C72B0", lw=1,
               label=f"role chance {res['role_alignment_chance']}")
    ax.axhline(res["scene_alignment_chance"], ls="--", color="#DD8452", lw=1,
               label=f"scene chance {res['scene_alignment_chance']}")
    ax.set_xticks(x, [f"{i}: {labs[i]}" for i in range(n)], rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("fraction of function-vector norm in subspace")
    ax.legend(fontsize=8)
    ax.set_title(f"primitive alignment to comprehension role / scene codes -- {res['schema']} -- "
                 f"{slug} L{layer}")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _public(obj):
    if isinstance(obj, dict):
        return {k: _public(v) for k, v in obj.items()
                if not (isinstance(k, str) and k.startswith("_"))}
    if isinstance(obj, list):
        return [_public(v) for v in obj]
    return obj


def write_report(stats: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stats.json").write_text(json.dumps(_public(stats), indent=2), encoding="utf-8")
    lines = [f"# Reasoning-primitive function vectors -- {stats['slug']} (layer {stats['layer']}, "
             f"scope=response)", "",
             f"Runs: {', '.join(stats['runs'])} | k-means states: {stats['k']} | "
             f"primitives: {stats['meta']}", ""]
    for schema, r in stats["by_schema"].items():
        lines += [
            f"## {schema} ({r['n_metaclusters']} primitives; eigengap {r['n_metaclusters_auto']})",
            f"- distinctness: mean |off-diagonal cosine| = {r['mean_abs_offdiag_cosine']} "
            "(lower = more distinct directions)",
            f"- contrast distinctness: mean |off-diagonal cosine| = "
            f"{r['mean_abs_offdiag_cosine_contrast']} (mean-difference vectors -- the injectable "
            "form; expected < raw, which shares the residual substrate)",
            f"- role subspace rank {r['role_subspace_rank']} (chance align "
            f"{r['role_alignment_chance']}); scene subspace rank {r['scene_subspace_rank']} "
            f"(chance {r['scene_alignment_chance']})", "",
            "| # | label | tokens | role-align | scene-align | nearest role |",
            "|---|---|---|---|---|---|",
        ]
        for p in r["primitives"]:
            nr = f"{p['nearest_role'][0]} ({p['nearest_role'][1]})" if p["nearest_role"] else "-"
            lines.append(
                f"| {p['meta']} | {p['label']} | "
                f"{p['n_tokens']} | {p['role_alignment']} | {p['scene_alignment']} | {nr} |")
        lines.append("")
    (out_dir / "findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(run_dirs: list[Path], slug: str, layer: int, k: int, meta: int, out_dir: Path,
        seed: int = 0, annotations: Path | None = None, task: str = "identify",
        pooled: bool = False) -> dict:
    refs = _response_refs(run_dirs, slug, layer, task=task)
    if not refs:
        raise SystemExit(f"no {task} captures for slug={slug} layer={layer} under {run_dirs}")
    referents, scene_tokens = load_captures(run_dirs, slug, layer, task=task)
    annot_all = json.loads(annotations.read_text(encoding="utf-8")) if annotations else {}
    schemas = sorted({s for _, _, s in refs if s})
    out_dir.mkdir(parents=True, exist_ok=True)
    if pooled:
        # one shared primitive set + function vectors fit across ALL schemas' response tokens.
        groups = [("combined", refs, referents, scene_tokens)]
    else:
        groups = [(sch, [r for r in refs if r[2] == sch],
                   [r for r in referents if r.schema == sch],
                   [t for t in scene_tokens if t.schema == sch]) for sch in schemas]
    by_schema = {}
    for gname, srefs, sref, sscene in groups:
        annot = (annot_all.get(gname, {}) or {}).get("meta") if annot_all else None
        res = analyze_schema(run_dirs, slug, layer, gname, srefs, sref, sscene, meta, k, seed,
                             annot)
        np.save(out_dir / f"function_vectors_raw_{gname}.npy", res["_fv_raw"])
        np.save(out_dir / f"function_vectors_std_{gname}.npy", res["_fv_std"])
        np.save(out_dir / f"function_vectors_contrast_{gname}.npy", res["_fv_contrast"])
        np.save(out_dir / f"function_vectors_contrast_loo_{gname}.npy", res["_fv_contrast_loo"])
        np.save(out_dir / f"function_vectors_overall_{gname}.npy", res["_fv_overall"])
        _cosine_figure(res, out_dir / f"funcvec_cosine_{gname}.png", slug, layer)
        _alignment_figure(res, out_dir / f"funcvec_alignment_{gname}.png", slug, layer)
        (out_dir / f"funcvec_{gname}.json").write_text(
            json.dumps(_public(res), indent=2), encoding="utf-8")
        by_schema[gname] = res
    stats = {"slug": slug, "layer": layer, "k": k, "meta": meta,
             "method": "mean-difference steering vector (ActAdd/CAA); residual layer-L mean; "
                       "inject the contrast (primitive-minus-overall) file",
             "runs": [f"{d.parent.name}/{d.name}" for d in run_dirs], "by_schema": by_schema}
    write_report(stats, out_dir)
    print(f"-> {out_dir}")
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description="reasoning-primitive function vectors (D2 extraction)")
    p.add_argument("--run", required=True, nargs="+", help="run dir(s) with activations/<slug>/")
    p.add_argument("--slug", required=True)
    p.add_argument("--layer", type=int, default=17)
    p.add_argument("--k", type=int, default=30, help="k-means token states (match metacluster)")
    p.add_argument("--meta", type=int, default=8, help="number of primitives (match metacluster)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--annotations", default=None,
                   help="metacluster annotations.json to reuse primitive labels")
    p.add_argument("--task", default="identify", help="capture task suffix (<sid>__<task>)")
    p.add_argument("--pooled", action="store_true",
                   help="one shared primitive set + function vectors across ALL schemas")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    run([Path(r) for r in args.run], args.slug, args.layer, args.k, args.meta, Path(args.out),
        seed=args.seed, annotations=Path(args.annotations) if args.annotations else None,
        task=args.task, pooled=args.pooled)


if __name__ == "__main__":
    main()
