"""Label generated text with the identify primitive alphabet, and compare arms.

What primitives does the model *run* after the injection? The pilot records only how long each
continuation was and whether it stopped; this re-forwards each continuation, reads the layer-16
residual at every generated position, and assigns it a metacluster with the frozen identify
assigner -- the same instrument, validated at ARI 1.0 against the corpus labels, that produced the
``metacluster_L16`` column the STOP masks were built from.

That makes the comparison like-for-like: the distribution shown here is in the same alphabet as the
observational analysis, not a neighbouring task's vocabulary.

Cost note: this is one forward pass per continuation, not autoregressive decoding, so it is far
cheaper than generating the text was.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ...core.model import load_model
from ..head_fv_capture import _forward_options, _unhooked
from ..head_fv_models import active as active_model
from . import runframe

ASSIGNER = Path("results/schema_llm/analysis/stop_steering/20260921-assigner-identify"
                "/assigner_identify_L16.npz")
NODE = Path("results/schema_llm/analysis/depth/20260824-rlreasoning_nodes/phi4mr_identify")
LAYER = 16


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_assigner(path: Path = ASSIGNER):
    d = np.load(path)
    return {k: d[k] for k in ("mean", "scale", "centers", "cluster_to_meta")}


def cluster_names(node: Path = NODE, layer: int = LAYER):
    """Per-cell metacluster labels (the LLM annotations on this (model, task, layer) cell)."""
    manifest = json.loads((node / "clusters.json").read_text())
    entry = [p for p in manifest["per_layer"] if p["layer"] == layer][0]
    ann = entry.get("ann") or {}
    return {int(k): v["label"] for k, v in ann.items()}


def merged_mapping(layer: int = LAYER, model: str = "phi4mr", task: str = "identify"):
    """metacluster -> (merged primitive id, name) in the pooled cross-model alphabet.

    ``metacluster_L<layer>`` is scoped to one (model, task, layer) cell;
    ``merged_primitive_L<layer>`` is the id in the shared 35-primitive alphabet. The map is read
    from the bundle rather than recomputed, and asserted to be 1:1 -- two metaclusters may share a
    merged primitive (here 3 and 5 both mean "Person entity identification"), but one metacluster
    never splits across two.
    """
    tokens = pd.read_parquet(runframe.BUNDLE / "tokens.parquet",
                             columns=["model", "task_family", "segment",
                                      f"metacluster_L{layer}", f"merged_primitive_L{layer}"])
    sub = tokens[(tokens.model == model) & (tokens.task_family == task)
                 & (tokens.segment != "prompt")].dropna(subset=[f"metacluster_L{layer}"])
    pairs = sub.groupby(f"metacluster_L{layer}")[f"merged_primitive_L{layer}"]
    require((pairs.nunique() == 1).all(), "metacluster -> merged primitive is not 1:1")
    names = json.loads((runframe.BUNDLE / "merged_primitive_names.json").read_text())
    return {int(m): (int(v.iloc[0]), names[int(v.iloc[0])]) for m, v in pairs}


def assign(residual, parts, batch=16384):
    """Standardise with the frozen statistics, nearest centroid, then cluster -> metacluster."""
    mean = parts["mean"]
    scale = np.where(parts["scale"] == 0, 1.0, parts["scale"])
    centers = parts["centers"]
    sq = (centers ** 2).sum(1)
    out = np.empty(len(residual), np.int32)
    for i in range(0, len(residual), batch):
        block = (residual[i:i + batch].astype(np.float32) - mean) / scale
        out[i:i + batch] = parts["cluster_to_meta"][
            (sq[None, :] - 2.0 * block @ centers.T).argmin(1)]
    return out


def residuals_for(lm, ids, n_prefix, hook_name, hidden):
    """Layer-16 residuals at the generated positions of one continuation."""
    _unhooked(lm.model)
    lm.model.eval()
    size = len(ids)
    captured = []

    def hook(_module, _args, output):
        h = output[0] if isinstance(output, tuple) else output
        require(tuple(h.shape) == (1, size, hidden), "unexpected residual shape")
        captured.append(h[0, n_prefix:].detach().float().cpu().numpy())

    handle = lm.model.get_submodule(hook_name).register_forward_hook(hook)
    try:
        tensor = torch.from_numpy(np.ascontiguousarray(ids))[None].to(lm.device)
        with torch.inference_mode():
            lm.model(input_ids=tensor, attention_mask=torch.ones_like(tensor),
                     position_ids=torch.arange(size, device=lm.device)[None],
                     **_forward_options(lm.model))
    finally:
        handle.remove()
    require(len(captured) == 1, "residual hook did not fire exactly once")
    return captured[0]


def run(pilot_dir: Path, out: Path, device="cuda:0", limit=None):
    out.mkdir(parents=True, exist_ok=True)
    table = pd.read_parquet(pilot_dir / "pilot.parquet")
    require("tail_text" in table, f"{pilot_dir}: no tail_text saved")
    parts = load_assigner()
    names = cluster_names()

    tokens = pd.read_parquet(runframe.BUNDLE / "tokens.parquet",
                             columns=["item_id", "model", "task_family", "position", "token_id"])
    tokens = tokens[(tokens.model == "phi4mr") & (tokens.task_family == runframe.TASK)]
    ids_of = {i: g.sort_values("position")["token_id"].values.astype(np.int64)
              for i, g in tokens.groupby("item_id")}

    spec = active_model()
    lm = load_model(spec.model, revision=spec.revision, device_map=device)
    hook_name = lm.layer_hook_names([LAYER])[0]
    hidden = lm.hidden_size

    rows = []
    subset = table if limit is None else table.head(limit)
    print(f"labelling {len(subset)} continuations from {pilot_dir.name}", flush=True)
    exact = "generated_ids" in subset.columns
    print(f"  token source: {'stored ids (exact)' if exact else 're-tokenised text'}", flush=True)
    for n, (_, row) in enumerate(subset.iterrows()):
        # prefer the ids the model actually emitted; fall back to re-tokenising the saved text
        # for runs produced before `generated_ids` was recorded
        generated = (list(row["generated_ids"]) if exact
                     else lm.tokenizer(row["tail_text"], add_special_tokens=False)["input_ids"])
        if len(generated) == 0:
            continue
        prefix = ids_of[row["item"]][: int(row["prefix_len"])]
        ids = np.concatenate([prefix, np.asarray(generated, np.int64)])
        resid = residuals_for(lm, ids, len(prefix), hook_name, hidden)
        metas = assign(resid, parts)
        counts = np.bincount(metas, minlength=8)
        rows.append(dict(item=row["item"], arm=row["arm"], dose=float(row["dose"]),
                         n_tokens=int(len(metas)),
                         # the per-token sequence, so the arms can be re-cut to a common length
                         # offline: primitives have characteristic positions, so comparing a
                         # 470-token arm against a 2048-token one confounds "what work" with
                         # "how far through the work"
                         labels=metas.astype(np.int16).tolist(),
                         **{f"meta_{i}": int(counts[i]) for i in range(8)}))
        if (n + 1) % 25 == 0:
            print(f"  {n + 1}/{len(subset)}", flush=True)

    frame = pd.DataFrame(rows)
    # fold the 8 per-cell metaclusters into the shared merged-primitive alphabet
    mapping = merged_mapping()
    merged_names = {}
    for meta, (mid, mname) in mapping.items():
        merged_names[mid] = mname
        col = f"merged_{mid}"
        frame[col] = frame.get(col, 0) + frame[f"meta_{meta}"]
    frame.to_parquet(out / "primitive_counts.parquet")
    (out / "cluster_names.json").write_text(json.dumps(names, indent=2))
    (out / "merged_names.json").write_text(json.dumps(
        {str(k): v for k, v in sorted(merged_names.items())}, indent=2))
    (out / "meta_to_merged.json").write_text(json.dumps(
        {str(k): {"merged_id": v[0], "name": v[1]} for k, v in sorted(mapping.items())}, indent=2))
    print(f"wrote {out}/primitive_counts.parquet  ({len(frame)} rows)")
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run(args.pilot, args.out, device=args.device, limit=args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
