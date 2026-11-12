"""Capture top-35-head activations at run-end positions -- Stage 1 of the STOP-steering plan.

One forward per item over the stored prompt+response IDs, hooking the ``o_proj`` *inputs* of the
16 AIE source layers (the per-head ``z``, before the output projection) and the layer-16 block
output. Exactly the quantity ``head_fv_existing_vectors.replay`` accumulates, but retained
per position instead of summed into theme groups, so any number of contrasts can be built
offline without another GPU pass.

Parity handle: the 99.5th percentile of run-end L16 norms must land on the screen's 62.8, which
is what pins the capture to the same activation site the 48-D screen read.
"""

from __future__ import annotations

import argparse
import json
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ...core.model import load_model
from .. import head_fv_aie as aie
from .. import head_fv_specificity as runtime
from ..head_fv_capture import _forward_options, _unhooked
from ..head_fv_models import active as active_model
from . import runframe

SELECTION = Path("results/schema_llm/data/phi_mini_head_fv/v1/aie/runs/layer_prefix_v2"
                 "/effects/selection.json")
MODEL = "phi4mr"
LAYER = 16
VARIANTS = ("row_end", "row_first", "row_shift")
EXPECTED_NORM_CUT = 62.8       # screen cell 10, phi4mr
NORM_PERCENTILE = 99.5


def require(condition, message):
    if not condition:
        raise ValueError(message)


def selected_heads(path: Path = SELECTION):
    payload = json.loads(path.read_text())
    heads = [(int(layer), int(head)) for layer, head in payload["heads"]]
    require(len(heads) == 35 and len(set(heads)) == 35, "expected 35 distinct heads")
    layers = tuple(sorted({layer for layer, _ in heads}))
    return heads, layers, payload


def full_ids(tokens, item):
    """Stored prompt+response IDs; ``position`` is asserted to be the row index, not assumed."""
    d = tokens[tokens.item_id == item].sort_values("position")
    positions = d["position"].values
    require(np.array_equal(positions, np.arange(len(positions))),
            f"{item}: positions are not a contiguous 0..N-1 index")
    return d["token_id"].values.astype(np.int64)


def capture_item(lm, ids, rows, heads, layers, geometry, hook_name):
    """Return (heads at ``rows``, L16 residual at ``rows``) for one item."""
    modules = aie.validate_model(lm, geometry)
    _unhooked(lm.model)
    size = len(ids)
    width = geometry.head_dim
    index = {layer: i for i, layer in enumerate(layers)}
    per_layer = np.zeros((len(layers), len(rows), geometry.heads, width), np.float32)
    residual: list[np.ndarray] = []
    seen: set[int] = set()
    handles = []
    select = torch.tensor(rows, dtype=torch.long, device=lm.device)

    def head_hook(li, _module, args):
        require(li not in seen and len(args) == 1
                and tuple(args[0].shape) == (1, size, geometry.heads * width),
                "duplicate/malformed o_proj input")
        taken = args[0][0].index_select(0, select)
        per_layer[li] = taken.reshape(-1, geometry.heads, width).detach().float().cpu().numpy()
        seen.add(li)

    def residual_hook(_module, _args, output):
        hidden = output[0] if isinstance(output, tuple) and output else output
        require(tuple(hidden.shape) == (1, size, geometry.hidden), "malformed L16 output")
        require(not residual, "L16 hook ran more than once")
        residual.append(hidden[0].index_select(0, select).detach().float().cpu().numpy())

    lm.model.eval()
    try:
        for layer in layers:
            handles.append(modules[layer].register_forward_pre_hook(
                partial(head_hook, index[layer])))
        handles.append(lm.model.get_submodule(hook_name).register_forward_hook(residual_hook))
        tensor = torch.from_numpy(np.ascontiguousarray(ids))[None].to(lm.device)
        with torch.inference_mode(), runtime.subject_cache(lm):
            lm.model(input_ids=tensor, attention_mask=torch.ones_like(tensor),
                     position_ids=torch.arange(size, device=lm.device)[None],
                     **_forward_options(lm.model))
        require(seen == set(range(len(layers))) and len(residual) == 1,
                "missing a selected source contribution")
    finally:
        for handle in handles:
            handle.remove()

    stacked = np.stack([per_layer[index[layer], :, head] for layer, head in heads], axis=1)
    require(stacked.shape == (len(rows), len(heads), width), "bad head stack")
    require(np.isfinite(stacked).all() and np.isfinite(residual[0]).all(), "nonfinite capture")
    return stacked, residual[0]


def run(out: Path, limit: int | None = None, device: str = "cuda:0", cohort: str = "terminating"):
    out.mkdir(parents=True, exist_ok=True)
    (out / "parts").mkdir(exist_ok=True)
    built = runframe.build()
    if cohort == "terminating":
        frame, items = built["seg"][MODEL], built["items"]
    else:
        items, frame = runframe.build_nonterminating(built, MODEL)
    tokens = pd.read_parquet(runframe.BUNDLE / "tokens.parquet",
                             columns=["item_id", "model", "task_family", "position", "token_id"])
    tokens = tokens[(tokens.model == MODEL) & (tokens.task_family == runframe.TASK)]

    heads, layers, payload = selected_heads()
    geometry = aie.SUBJECT_GEOMETRY
    spec = active_model()
    require(payload["model"] == spec.model, "selection/model spec disagree")
    lm = load_model(spec.model, revision=spec.revision, device_map=device)
    hook_name = lm.layer_hook_names([LAYER])[0]
    print(f"model {spec.model} | hook {hook_name} | {len(heads)} heads "
          f"over layers {layers}", flush=True)

    items = items if limit is None else items[:limit]
    start = time.time()
    for i, item in enumerate(items):
        target = out / "parts" / f"{item}.npz"
        if target.exists():
            continue
        group = frame[frame.item == item]
        require(len(group), f"{item}: no segments")
        rows = np.unique(np.concatenate([group[v].values for v in VARIANTS])).astype(np.int64)
        ids = full_ids(tokens, item)
        require(rows.max() < len(ids), f"{item}: run-end row beyond the stored sequence")
        stacked, residual = capture_item(lm, ids, rows, heads, layers, geometry, hook_name)
        lookup = {row: j for j, row in enumerate(rows)}
        end_index = np.array([lookup[r] for r in group["row_end"].values])
        norms = {v: np.linalg.norm(residual[[lookup[r] for r in group[v].values]], axis=1)
                 for v in VARIANTS}
        np.savez(target, heads=stacked[end_index].astype(np.float32),
                 residual=residual[end_index].astype(np.float16),
                 t=group["t"].values.astype(np.int32),
                 **{f"norm_{v}": norms[v].astype(np.float32) for v in VARIANTS})
        if (i + 1) % 10 == 0 or i == 0:
            rate = (time.time() - start) / (i + 1)
            print(f"  {i + 1}/{len(items)} {item} seq={len(ids)} rows={len(group)} "
                  f"{rate:.1f}s/item", flush=True)
    print(f"captured {len(items)} items in {time.time() - start:.0f}s", flush=True)
    return out


def finalize(out: Path, cohort: str = "terminating"):
    """Concatenate parts in frame order, apply the screen's norm cut, and record parity."""
    built = runframe.build()
    if cohort == "terminating":
        frame, items = built["seg"][MODEL], built["items"]
    else:
        items, frame = runframe.build_nonterminating(built, MODEL)
    heads, residual, norms = [], [], {v: [] for v in VARIANTS}
    for item in items:
        part = np.load(out / "parts" / f"{item}.npz")
        group = frame[frame.item == item]
        require(np.array_equal(part["t"], group["t"].values), f"{item}: part/frame misalignment")
        heads.append(part["heads"])
        residual.append(part["residual"])
        for v in VARIANTS:
            norms[v].append(part[f"norm_{v}"])
    heads = np.concatenate(heads)
    residual = np.concatenate(residual)
    norms = {v: np.concatenate(norms[v]) for v in VARIANTS}
    require(len(heads) == len(frame), "row count differs from the run frame")

    cut = (float(np.percentile(norms["row_end"], NORM_PERCENTILE))
           if cohort == "terminating" else EXPECTED_NORM_CUT)
    keep = np.all([norms[v] <= cut for v in VARIANTS], axis=0)
    parity = dict(norm_cut=cut, expected_norm_cut=EXPECTED_NORM_CUT,
                  norm_cut_matches=bool(abs(cut - EXPECTED_NORM_CUT) <= 0.05),
                  dropped=int((~keep).sum()), total=int(len(keep)),
                  dropped_share=float((~keep).mean()), expected_dropped=1263,
                  max_norm=float(max(norms[v].max() for v in VARIANTS)))
    kept = frame[keep].reset_index(drop=True)
    parity["last3_runs"] = int(kept["last3"].sum())
    parity["ref_stop_runs"] = int(kept["ref_stop"].sum())
    parity["expected_last3_runs"] = 496

    np.savez(out / "capture.npz", heads=heads[keep].astype(np.float32),
             residual=residual[keep], **{f"norm_{v}": norms[v][keep] for v in VARIANTS})
    kept.to_parquet(out / "frame.parquet")
    (out / "parity.json").write_text(json.dumps(parity, indent=2))
    print(json.dumps(parity, indent=2))
    return parity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--cohort", default="terminating",
                        choices=("terminating", "nonterminating"))
    args = parser.parse_args()
    if args.finalize:
        finalize(args.out, cohort=args.cohort)
    else:
        run(args.out, limit=args.limit, device=args.device, cohort=args.cohort)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
