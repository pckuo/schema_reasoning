"""Algotrace / Todd et al. head-projection **function vectors** for reasoning primitives
(SCHEMA_LATENT_GEOMETRY_PLAN D2). This is the faithful attention-head construction, complementing
the residual mean-difference *steering* vector in :mod:`funcvec`::

    FV_primitive = sum_(L,H in top heads) W^O_L ( a_(L,H) placed in head H's slot )

where ``a_(L,H)`` is the mean, over the primitive's response tokens, of the **input** to
``model.layers.L.self_attn.o_proj`` (the concatenated per-head attention output) reshaped into
``(n_heads, head_dim)``, and ``W^O_L`` is that layer's real output projection. The top heads are
the model's universal high-AIE set copied verbatim from algotrace
``extract_utils.compute_universal_function_vector`` (``phi-4`` / ``phi-4-reasoning``); they span
several layers, so -- unlike the single-layer residual capture -- this needs o_proj inputs at every
layer that owns a selected head.

The cost is split so the GPU pass is never repeated:

- primitive labels come from the ALREADY-saved layer-``L`` residuals via
  :func:`funcvec.fit_primitives` (CPU, no re-capture);
- the GPU pass is **forward-only** over the saved responses (no re-generation) and saves the
  expensive intermediate ``mean_head_activations_<schema>.npy`` (per-primitive mean per-head
  activations) plus ``oproj_slices.npz`` (the o_proj weight columns for every head in the set), so
  the function vector can be **rebuilt for any ``n_top_heads`` on CPU** (``--rebuild``) without ever
  touching the GPU again.

Example (GPU)::

    python -m dev.schema_llm.primitives.funcvec_heads \
        --run results/schema_llm/data/main_36/restaurant_main \
              results/schema_llm/data/main_36/airport_main \
              results/schema_llm/data/main_72/restaurant_main \
              results/schema_llm/data/main_72/airport_main \
        --slug Phi-4-reasoning --model microsoft/Phi-4-reasoning --layer 17 --k 30 --meta 8 \
        --n-top-heads 10 --annotations <metacluster-out>/annotations.json \
        --out results/schema_llm/analysis/identify/20260718-funcvec_head_Phi-4-reasoning_216

Rebuild for a different head count (CPU, no GPU, reuses the saved means + slices)::

    python -m dev.schema_llm.primitives.funcvec_heads --rebuild \
        --out results/schema_llm/analysis/identify/20260718-funcvec_head_Phi-4-reasoning_216 \
        --slug Phi-4-reasoning --n-top-heads 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ._text import _clean, _top_content
from .funcvec import _response_refs, fit_primitives

# Universal high-AIE head sets, copied verbatim from algotrace
# code/reference/algotrace/src/utils/extract_utils.py (compute_universal_function_vector). Each
# entry is (layer, head, avg-indirect-effect). Order = descending AIE; we take the first
# ``n_top_heads``. These were computed by AIE on Todd et al.'s ICL tasks -- reusing them is the
# algotrace-faithful choice; a schema-task-specific AIE would be a separate, larger effort.
UNIVERSAL_HEADS: dict[str, list[tuple[int, int, float]]] = {
    "phi-4-reasoning": [
        (17, 22, 0.0418), (17, 23, 0.0238), (25, 18, 0.0228), (22, 5, 0.0221), (18, 11, 0.0216),
        (18, 12, 0.02), (18, 14, 0.02), (18, 8, 0.0133), (24, 32, 0.0125), (24, 28, 0.012),
        (21, 16, 0.011), (19, 21, 0.0105), (24, 31, 0.0103), (17, 28, 0.0095), (20, 13, 0.0094),
        (18, 9, 0.0092), (22, 33, 0.0086), (25, 16, 0.0071), (18, 26, 0.0069), (23, 5, 0.0067),
        (22, 7, 0.0067), (26, 22, 0.0063), (24, 3, 0.0062), (21, 35, 0.006), (23, 30, 0.006),
        (18, 16, 0.006), (26, 13, 0.0059), (17, 27, 0.0058), (22, 12, 0.0055), (19, 7, 0.0053),
        (19, 20, 0.0052), (32, 11, 0.0052), (22, 34, 0.0051), (31, 4, 0.0049), (19, 32, 0.0049),
    ],
    "phi-4": [
        (17, 22, 0.0374), (17, 23, 0.0347), (22, 5, 0.0278), (25, 18, 0.0275), (18, 12, 0.0245),
        (18, 11, 0.0204), (22, 33, 0.0186), (21, 16, 0.0153), (21, 35, 0.0146), (18, 8, 0.0144),
        (24, 32, 0.0142), (18, 9, 0.0128), (18, 14, 0.0125), (23, 5, 0.0121), (19, 20, 0.0115),
        (25, 16, 0.0103), (26, 13, 0.0101), (24, 28, 0.01), (19, 21, 0.0099), (17, 28, 0.0097),
        (24, 3, 0.0089), (24, 31, 0.0089), (18, 16, 0.0087), (26, 22, 0.0083), (22, 12, 0.0082),
        (18, 26, 0.0082), (17, 27, 0.0076), (21, 11, 0.0073), (20, 13, 0.0073), (32, 11, 0.0072),
        (21, 38, 0.007), (19, 32, 0.0067), (17, 5, 0.0066), (31, 4, 0.0064), (22, 34, 0.0062),
    ],
}


def heads_for(slug: str) -> list[tuple[int, int, float]]:
    """The universal head set for ``slug`` (``phi-4-reasoning`` checked before ``phi-4`` so the
    reasoning variant is not shadowed by the base-model substring)."""
    key = slug.lower()
    for name, heads in UNIVERSAL_HEADS.items():
        if name in key:
            return heads
    raise SystemExit(f"no universal head set for slug={slug!r}; known: {list(UNIVERSAL_HEADS)}")


def capture_layers(heads: list[tuple[int, int, float]]) -> list[int]:
    """Sorted union of the layers owning any head in the set (what the GPU pass must hook)."""
    return sorted({L for L, _H, _s in heads})


def validate_heads(heads: list[tuple[int, int, float]], n_layers: int, n_heads: int,
                   slug: str) -> None:
    """Refuse a head set that does not fit the loaded model's attention geometry. ``heads_for``
    matches on substring, so ``Phi-4-mini-reasoning`` silently draws ``phi-4``'s 40-head/40-layer
    list even though the mini is 32 x 24 -- heads 26-38 do not exist there and would slice o_proj
    out of bounds (numpy returns an EMPTY column block rather than raising). No AIE has been
    computed for the mini, so the honest failure is to stop, not to substitute."""
    bad_L = sorted({L for L, _H, _s in heads if L >= n_layers})
    bad_H = sorted({H for _L, H, _s in heads if H >= n_heads})
    if bad_L or bad_H:
        raise SystemExit(
            f"universal head set for slug={slug!r} does not fit this model "
            f"({n_layers} layers x {n_heads} heads): out-of-range layers {bad_L}, heads {bad_H}. "
            "The set is copied from algotrace's AIE on Phi-4/Phi-4-reasoning; run AIE for this "
            "architecture before using head function vectors on it.")


def accumulate(raw_sum: np.ndarray, overall_sum: np.ndarray, cnt: np.ndarray,
               sels: list[np.ndarray], metas: np.ndarray, n_meta: int) -> None:
    """Fold one passage's per-layer head activations (``sels[li]`` = ``(n_resp, n_heads, head_dim)``
    for captured layer ``li``, aligned to ``metas`` = per-response-token primitive labels) into the
    running per-primitive and overall sums (in place). Pure numpy -- the unit-tested core."""
    for li, sel in enumerate(sels):
        overall_sum[li] += sel.sum(0)
    for m in range(n_meta):
        mask = metas == m
        c = int(mask.sum())
        if c:
            cnt[m] += c
            for li, sel in enumerate(sels):
                raw_sum[m, li] += sel[mask].sum(0)


def build_head_fv(mean_head: np.ndarray, captured_layers: list[int], slices: dict,
                  top_heads: list[tuple[int, int, float]], head_dim: int,
                  overall: np.ndarray | None = None) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Todd/algotrace function vector per primitive: for each selected head place its mean
    activation in that head's slot and push it through the real ``W^O_L``, then sum. When
    ``overall`` is given, use the head-level contrast ``a - a_overall`` (cancels the shared
    substrate, the head-space analog of the residual mean-difference). Returns ``(fv, used_heads)``
    with ``fv`` of shape ``(n_meta, resid_dim)``. Pure numpy -- unit-tested on a stub."""
    row = {L: i for i, L in enumerate(captured_layers)}
    n_meta = mean_head.shape[0]
    resid = next(slices[k].shape[0] for k in slices if not k.startswith("bias_"))
    fv = np.zeros((n_meta, resid), np.float32)
    used: list[tuple[int, int]] = []
    for L, H, _s in top_heads:
        if L not in row:
            continue
        a = mean_head[:, row[L], H, :]  # (n_meta, head_dim)
        if overall is not None:
            a = a - overall[row[L], H, :]
        w = slices[f"{L}_{H}"]  # (resid, head_dim)
        fv = fv + a.astype(np.float32) @ w.T
        bias = slices.get(f"bias_{L}")
        if bias is not None:
            fv = fv + bias  # per-head, faithful to compute_function_vector's out_proj(x)
        used.append((int(L), int(H)))
    _ = head_dim  # kept for signature symmetry / callers that pass it explicitly
    return fv, used


def _head_dims(lm) -> tuple[int, int, int]:
    """``(n_heads, head_dim, hidden)``. ``head_dim`` is read from the config when present and only
    falls back to ``hidden // n_heads``: Qwen3-4B sets ``head_dim = 128`` explicitly while
    ``2560 / 32 = 80``, so the fallback mis-slices o_proj (whose real input width is 4096)."""
    cfg = lm.model.config
    inner = getattr(cfg, "text_config", None)
    if inner is not None and hasattr(inner, "num_attention_heads"):
        cfg = inner
    n_heads = int(cfg.num_attention_heads)
    hidden = int(cfg.hidden_size)
    head_dim = int(getattr(cfg, "head_dim", None) or hidden // n_heads)
    return n_heads, head_dim, hidden


def _module(root, dotted: str):
    obj = root
    for attr in dotted.split("."):
        obj = obj[int(attr)] if attr.isdigit() else getattr(obj, attr)
    return obj


def oproj_slices(lm, heads: list[tuple[int, int, float]], head_dim: int) -> dict:
    """The ``W^O`` columns (and bias, if any) for every head in the set -- saved once so the FV can
    be rebuilt on CPU. ``slices["L_H"]`` = ``(resid_dim, head_dim)`` and ``slices["bias_L"]`` =
    bias."""
    out: dict[str, np.ndarray] = {}
    for L, H, _s in heads:
        mod = _module(lm.model, f"{lm.layer_prefix}.{L}.self_attn.o_proj")
        w = mod.weight.detach().float().cpu().numpy()  # (resid_out, resid_in)
        out[f"{L}_{H}"] = np.ascontiguousarray(w[:, H * head_dim:(H + 1) * head_dim])
        if getattr(mod, "bias", None) is not None and f"bias_{L}" not in out:
            out[f"bias_{L}"] = mod.bias.detach().float().cpu().numpy()
    return out


def extract_head_means(lm, passages: list[dict], model, layers: list[int], n_meta: int,
                       n_heads: int, head_dim: int) -> tuple:
    """GPU forward-only pass: for each passage rebuild the exact ``[prompt+response]`` ids from the
    saved ``meta.json`` tokens, hook the o_proj inputs at ``layers``, and accumulate per-primitive
    (and overall) mean per-head activations over the primitive's response tokens. Returns
    ``(mean_head (n_meta,L,n_heads,head_dim), overall (L,n_heads,head_dim), cnt, top_tokens)``."""
    import gc

    import torch
    from baukit import TraceDict

    tok = lm.tokenizer
    hook_names = [f"{lm.layer_prefix}.{L}.self_attn.o_proj" for L in layers]
    n_lay = len(layers)
    raw_sum = np.zeros((n_meta, n_lay, n_heads, head_dim), np.float64)
    overall_sum = np.zeros((n_lay, n_heads, head_dim), np.float64)
    cnt = np.zeros(n_meta)
    tokbuf: list[list[str]] = [[] for _ in range(n_meta)]
    for pi, p in enumerate(passages):
        meta = json.loads((Path(p["npy"]).parent / "meta.json").read_text(encoding="utf-8"))
        full_ids = tok.convert_tokens_to_ids(meta["tokens"])
        idx = p["idx"]
        metas = model.meta_labels(p["std"])
        with torch.no_grad(), TraceDict(
            lm.model, layers=hook_names, retain_input=True, retain_output=False
        ) as td:
            lm.model(input_ids=torch.tensor([full_ids], device=lm.device))
        sels = []
        for name in hook_names:
            act = td[name].input
            act = act[0] if isinstance(act, tuple) else act
            heads = act.view(act.size(0), act.size(1), n_heads, head_dim)[0]  # (n_tok,H,hd)
            sels.append(heads[idx].float().cpu().numpy())
        accumulate(raw_sum, overall_sum, cnt, sels, metas, n_meta)
        for t, mm in zip(p["tokens"], metas, strict=False):
            if len(tokbuf[mm]) < 6000:
                tokbuf[mm].append(_clean(t))
        del td
        if pi % 16 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    den = np.maximum(cnt[:, None, None, None], 1.0)
    mean_head = (raw_sum / den).astype(np.float32)
    overall = (overall_sum / max(float(cnt.sum()), 1.0)).astype(np.float32)
    top = [_top_content(tokbuf[m], 10) for m in range(n_meta)]
    return mean_head, overall, cnt.astype(int), top


def _labels(annot: dict | None, top: list[list[str]], n_meta: int) -> list[str]:
    out = []
    for m in range(n_meta):
        a = (annot or {}).get(str(m)) or (annot or {}).get(m)
        if a and a.get("label"):
            out.append(f"{a['label']}")
        else:
            out.append(" ".join(top[m][:3]) if top[m] else "?")
    return out


def _write_schema(out_dir: Path, schema: str, mean_head: np.ndarray, overall: np.ndarray,
                  captured_layers: list[int], cnt: np.ndarray, top: list[list[str]],
                  slices: dict, heads: list[tuple[int, int, float]], n_top_heads: int,
                  head_dim: int, annot: dict | None, contrast: bool) -> dict:
    """Save the reusable intermediate (mean head activations) + build & save the function vector."""
    np.save(out_dir / f"mean_head_activations_{schema}.npy", mean_head)
    np.save(out_dir / f"mean_head_activations_overall_{schema}.npy", overall)
    n_meta = mean_head.shape[0]
    labels = _labels(annot, top, n_meta)
    fv, used = build_head_fv(mean_head, captured_layers, slices, heads[:n_top_heads], head_dim)
    np.save(out_dir / f"funcvec_head_{schema}.npy", fv)
    if contrast:
        fvc, _ = build_head_fv(mean_head, captured_layers, slices, heads[:n_top_heads], head_dim,
                               overall=overall)
        np.save(out_dir / f"funcvec_head_contrast_{schema}.npy", fvc)
    prims = [{
        "meta": m, "label": labels[m], "n_tokens": int(cnt[m]),
        "norm": round(float(np.linalg.norm(fv[m])), 3), "top": top[m],
    } for m in range(n_meta)]
    doc = {
        "schema": schema, "method": "head-projection function vector (Todd/algotrace)",
        "n_top_heads": n_top_heads, "top_heads": [[int(L), int(H)] for L, H in used],
        "captured_layers": captured_layers, "n_metaclusters": int(n_meta),
        "contrast": contrast, "primitives": prims,
    }
    (out_dir / f"funcvec_head_{schema}.json").write_text(json.dumps(doc, indent=2),
                                                         encoding="utf-8")
    return doc


def run(
    run_dirs: list[Path],
    slug: str,
    model_name: str,
    layer: int,
    k: int,
    meta: int,
    out_dir: Path,
    n_top_heads: int = 10,
    seed: int = 0,
    annotations: Path | None = None,
    contrast: bool = True,
    task: str = "identify",
    pooled: bool = False,
    revision: str | None = None,
    chat_template_from: str | None = None,
    dtype: str = "bfloat16",
) -> dict:
    """GPU pass: per schema, fit primitives (CPU, reuse saved residuals), extract mean head
    activations (GPU forward-only), and build the head-projection function vectors. Saves all
    reusable intermediates so :func:`rebuild` can re-derive FVs on CPU for any head count.

    ``revision`` / ``chat_template_from`` / ``dtype`` must match the capture that produced the
    latents, or the re-forwarded prompt differs from the one they were recorded on."""
    from ..core.model import chat_template_of, load_model, set_seed

    refs = _response_refs(run_dirs, slug, layer, task=task)
    if not refs:
        raise SystemExit(f"no {task} captures for slug={slug} layer={layer} under {run_dirs}")
    heads = heads_for(slug)
    layers = capture_layers(heads)
    annot_all = json.loads(annotations.read_text(encoding="utf-8")) if annotations else {}
    schemas = sorted({s for _, _, s in refs if s})
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    template = chat_template_of(chat_template_from) if chat_template_from else None
    lm = load_model(model_name, dtype=dtype, revision=revision, chat_template=template)
    n_heads, head_dim, resid = _head_dims(lm)
    validate_heads(heads, lm.n_layers, n_heads, slug)
    slices = oproj_slices(lm, heads, head_dim)
    np.savez(out_dir / "oproj_slices.npz", **slices)
    if pooled:
        groups = [("combined", refs)]
    else:
        groups = [(sch, [(mp, npy, s) for mp, npy, s in refs if s == sch]) for sch in schemas]
    by_schema = {}
    for gname, srefs in groups:
        _scaler, _pca, model, passages = fit_primitives(srefs, meta, k, seed)
        n_meta = model.n_metaclusters_
        mean_head, overall, cnt, top = extract_head_means(
            lm, passages, model, layers, n_meta, n_heads, head_dim)
        annot = (annot_all.get(gname, {}) or {}).get("meta") if annot_all else None
        doc = _write_schema(out_dir, gname, mean_head, overall, layers, cnt, top, slices, heads,
                            n_top_heads, head_dim, annot, contrast)
        by_schema[gname] = doc
    stats = {
        "slug": slug,
        "model": model_name,
        "revision": revision,
        "chat_template_from": chat_template_from,
        "dtype": dtype,
        "layer": layer,
        "k": k,
        "meta": meta,
        "n_top_heads": n_top_heads,
        "captured_layers": layers,
        "n_heads": n_heads,
        "head_dim": head_dim,
        "resid_dim": resid,
        "contrast": contrast,
        "method": "head-projection function vector (Todd/algotrace); universal high-AIE heads",
        "runs": [f"{d.parent.name}/{d.name}" for d in run_dirs],
        "schemas": list(by_schema),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    _write_findings(out_dir, stats, by_schema)
    print(f"-> {out_dir}")
    return stats


def rebuild(out_dir: Path, slug: str, n_top_heads: int, contrast: bool = True) -> None:
    """CPU: rebuild function vectors for a different ``n_top_heads`` from the saved mean head
    activations + o_proj slices (no GPU, no re-forward)."""
    heads = heads_for(slug)
    npz = np.load(out_dir / "oproj_slices.npz")
    slices = {kk: npz[kk] for kk in npz.files}
    head_dim = next(slices[kk].shape[1] for kk in slices if not kk.startswith("bias_"))
    stats = json.loads((out_dir / "stats.json").read_text(encoding="utf-8"))
    captured_layers = stats["captured_layers"]
    for sch in stats["schemas"]:
        mean_head = np.load(out_dir / f"mean_head_activations_{sch}.npy")
        fv, used = build_head_fv(mean_head, captured_layers, slices, heads[:n_top_heads], head_dim)
        np.save(out_dir / f"funcvec_head_top{n_top_heads}_{sch}.npy", fv)
        if contrast:
            overall = np.load(out_dir / f"mean_head_activations_overall_{sch}.npy")
            fvc, _ = build_head_fv(mean_head, captured_layers, slices, heads[:n_top_heads],
                                   head_dim, overall=overall)
            np.save(out_dir / f"funcvec_head_contrast_top{n_top_heads}_{sch}.npy", fvc)
        print(f"{sch}: rebuilt {len(used)}-head FV -> funcvec_head_top{n_top_heads}_{sch}.npy")


def _write_findings(out_dir: Path, stats: dict, by_schema: dict) -> None:
    lines = [f"# Head-projection function vectors -- {stats['slug']} (layer {stats['layer']})", "",
             f"Universal high-AIE heads (algotrace), top {stats['n_top_heads']} of "
             f"{len(heads_for(stats['slug']))}; captured layers {stats['captured_layers']}.",
             "Reusable intermediate: mean_head_activations_<schema>.npy (+ oproj_slices.npz) -> "
             "rebuild any head count on CPU via --rebuild.", ""]
    for sch, doc in by_schema.items():
        lines += [f"## {sch} ({doc['n_metaclusters']} primitives)",
                  f"- heads used: {doc['top_heads']}", "",
                  "| # | label | tokens | fv-norm |",
                  "|---|---|---|---|"]
        for p in doc["primitives"]:
            lines.append(f"| {p['meta']} | {p['label']} | "
                         f"{p['n_tokens']} | {p['norm']} |")
        lines.append("")
    (out_dir / "findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    from ..core.model import CAPTURE_GRID, validate_layer

    p = argparse.ArgumentParser(description="head-projection function vectors (D2; algotrace/Todd)")
    p.add_argument("--run", nargs="+", help="run dir(s) with activations/<slug>/ (GPU pass)")
    p.add_argument("--slug", required=True)
    p.add_argument("--model", required=True, help="HF model id; MUST match the captured latents")
    p.add_argument("--revision", default=None, help="pinned checkpoint revision, as at capture")
    p.add_argument(
        "--chat-template-from",
        default=None,
        metavar="MODEL[@REV]",
        help="render prompts with ANOTHER checkpoint's chat template; MUST match the "
        "capture that produced the latents",
    )
    p.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "auto"],
        help="COMPUTE dtype; capture used bfloat16 (latents are only stored as fp16)",
    )
    p.add_argument("--layer", type=int, required=True)
    p.add_argument(
        "--capture-layers",
        default=",".join(str(x) for x in CAPTURE_GRID),
        help="grid --layer is validated against (the depths latents exist at)",
    )
    p.add_argument("--k", type=int, default=30)
    p.add_argument("--meta", type=int, default=8)
    p.add_argument("--n-top-heads", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--annotations", default=None)
    p.add_argument("--no-contrast", action="store_true", help="skip the head-level contrast FV")
    p.add_argument("--rebuild", action="store_true",
                   help="CPU: rebuild FVs for --n-top-heads from saved intermediates (no GPU)")
    p.add_argument("--task", default="identify", help="capture task suffix (<sid>__<task>)")
    p.add_argument("--pooled", action="store_true",
                   help="one shared primitive set across ALL schemas (combined)")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = Path(args.out)
    if args.rebuild:
        rebuild(out, args.slug, args.n_top_heads, contrast=not args.no_contrast)
        return
    if not args.run:
        raise SystemExit("--run is required for the GPU extraction pass")
    grid = tuple(int(x) for x in args.capture_layers.split(",") if x.strip())
    validate_layer(args.layer, grid)
    run(
        [Path(r) for r in args.run],
        args.slug,
        args.model,
        args.layer,
        args.k,
        args.meta,
        out,
        n_top_heads=args.n_top_heads,
        seed=args.seed,
        annotations=Path(args.annotations) if args.annotations else None,
        contrast=not args.no_contrast,
        task=args.task,
        pooled=args.pooled,
        revision=args.revision,
        chat_template_from=args.chat_template_from,
        dtype=args.dtype,
    )


if __name__ == "__main__":
    main()
