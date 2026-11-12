"""Build the steering directions: head-space contrast, then ``o_proj`` back to the residual.

Same construction as the existing primitive vectors -- per-head means over selected positions,
scattered into their concatenated-head slots and pushed through each source layer's ``o_proj`` --
but the positions come from the screen's STOP-approach masks instead of a primitive's reader
tokens, and the statistic is a *contrast* rather than a mean.

Every direction is unit-norm in residual space, so arms are norm-matched at the injection site by
construction and the dose is the only thing that varies. The shuffled-label direction is built by
the identical path, which makes it the only null that is matched in subspace as well as in norm --
an isotropic random residual direction is not, because block 16 destroys out-of-subspace
perturbations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ...core.model import load_model
from .. import head_fv_specificity as runtime
from ..head_fv_models import active as active_model
from . import events, runframe
from .capture import selected_heads
from .gate import centered, contrast, load, shuffled_labels, unit

SEED = 20260919


def require(condition, message):
    if not condition:
        raise ValueError(message)


def head_space_directions(heads, frame, rng, masks=None):
    """Unit-norm head-space contrasts: STOP approach, its nulls, and the specificity arms.

    ``masks`` carries the per-run ``pre_switch`` / ``away_from_switch`` / ``solved`` flags from
    ``events.build_masks``; when absent only the STOP direction and its nulls are built.
    """
    primitives = frame["z0"].values
    block = centered(heads, primitives)
    positive, negative = frame["last3"].values, frame["ref_stop"].values
    out = {"stop": contrast(block, positive, negative)}
    shuffled_pos, shuffled_neg = shuffled_labels(frame, positive, negative, rng)
    out["shuffled"] = contrast(block, shuffled_pos, shuffled_neg)
    out["random"] = unit(rng.standard_normal(heads.shape[1]).astype(np.float32))
    if masks is not None:
        require(len(masks) == len(frame), "mask/frame row mismatch")
        out["switch"] = contrast(block, masks["pre_switch"].values,
                                 masks["away_from_switch"].values)
        out["solved"] = contrast(block, masks["solved"].values, ~masks["solved"].values)
    projection_sd = {name: float(np.std(block @ v)) for name, v in out.items()}
    return out, projection_sd


def to_residual(lm, vector, heads_index, layers, geometry):
    """Scatter the 35 head slices into their layers and project with ``o_proj``."""
    per_head = np.asarray(vector, dtype=np.float32).reshape(len(heads_index), geometry.head_dim)
    dense = np.zeros((geometry.layers, geometry.heads, geometry.head_dim), np.float32)
    for (layer, head), slice_ in zip(heads_index, per_head, strict=True):
        dense[layer, head] = slice_
    projected, report = runtime.projected_fv(lm, dense, heads_index,
                                             expected_count=len(heads_index))
    require(np.isfinite(projected).all() and np.linalg.norm(projected) > 0,
            "degenerate projected direction")
    return np.asarray(projected, dtype=np.float32), report


def build(capture: Path, out: Path, device="cuda:0"):
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    heads, residual, frame = load(capture)
    built = runframe.build()
    masks = events.build_masks(built)
    # the capture drops norm-cut rows, so realign the masks on (item, t)
    key = frame[["item", "t"]].merge(masks, on=["item", "t"], how="left", validate="one_to_one")
    require(len(key) == len(frame) and not key["solved"].isna().any(), "mask realignment failed")
    directions, projection_sd = head_space_directions(heads, frame, rng, masks=key)

    heads_index, layers, payload = selected_heads()
    spec = active_model()
    require(payload["model"] == spec.model, "selection/model spec disagree")
    lm = load_model(spec.model, revision=spec.revision, device_map=device)
    from .. import head_fv_aie as aie
    geometry = aie.SUBJECT_GEOMETRY

    state_norm = float(np.median(np.linalg.norm(residual.astype(np.float32), axis=1)))
    vectors, reports = {}, {}
    for name, vector in directions.items():
        projected, report = to_residual(lm, vector, heads_index, layers, geometry)
        vectors[name] = unit(projected)
        reports[name] = {"raw_norm": float(np.linalg.norm(projected)),
                         "head_space_projection_sd": projection_sd[name],
                         "oproj_heads_used": report.get("heads", len(heads_index))}
    cosines = {f"{a}|{b}": float(vectors[a] @ vectors[b])
               for i, a in enumerate(vectors) for b in list(vectors)[i + 1:]}
    head_cosines = {f"{a}|{b}": float(directions[a] @ directions[b])
                    for i, a in enumerate(directions) for b in list(directions)[i + 1:]}
    manifest = dict(model=spec.model, revision=spec.revision, n_heads=len(heads_index),
                    source_layers=list(layers), rows=int(len(frame)),
                    items=int(frame["item"].nunique()),
                    n_stop_positions=int(frame["last3"].sum()),
                    n_baseline_positions=int(frame["ref_stop"].sum()),
                    median_l16_state_norm=state_norm,
                    directions=reports, residual_cosines=cosines,
                    head_space_cosines=head_cosines,
                    screen_48d_reference={"stop|solved": -0.00, "stop|switch": -0.46},
                    mask_counts={"pre_switch": int(key["pre_switch"].sum()),
                                 "away_from_switch": int(key["away_from_switch"].sum()),
                                 "solved": int(key["solved"].sum())},
                    seed=SEED)
    np.savez(out / "directions.npz", **vectors)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    del lm
    torch.cuda.empty_cache()
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    build(args.capture, args.out, device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
