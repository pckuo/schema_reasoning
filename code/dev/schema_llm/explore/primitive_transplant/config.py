"""Shared constants + small readers for the primitive-transplant exploration (steps A-D).

The subject is **Phi-4-mini-reasoning** on ``spot_error``, pinned exactly as the capture that
produced the latents and the primitive dictionary
(``scripts/schema_llm/run_phi4mini_reasoning_arm.sh``): a fixed ``revision`` plus
Phi-4-mini-INSTRUCT's chat template, because the stock reasoning template
hard-codes a *"Your name is Phi, an AI math expert"* system turn.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[5]
DATA = REPO / "results/schema_llm/data"

MODEL = "microsoft/Phi-4-mini-reasoning"
REVISION = "0e3b1e2d02ee478a3743abe3f629e9c0cb722e0a"
CHAT_TEMPLATE_FROM = "microsoft/Phi-4-mini-instruct@cfbefacb99257ffa30c83adab238a50856ac3083"
SLUG = "phi-4-mini-reasoning"
DTYPE = "bfloat16"  # capture's COMPUTE dtype; latents are merely STORED as fp16
MAX_NEW_TOKENS = 8192  # capture's budget; the mean spot_error trace is ~3.2k tokens
LAYERS = (0, 4, 8, 12, 16, 20, 24, 28)
TASK = "spot_error"

POOL_SPOT = [
    DATA / "spot_36/restaurant_spot",
    DATA / "spot_36/airport_spot",
    DATA / "main_72/restaurant_spot",
    DATA / "main_72/airport_spot",
]

NODES = REPO / "results/schema_llm/analysis/depth/20260824-rlreasoning_nodes"
MERGE = REPO / "results/schema_llm/analysis/crossmodel_merge/20260824-rlreasoning/all/semavglink"
# clustering hyper-parameters, read back from each node cache's `key` and asserted against these
K = 50
FORCE_META = 8
SEED = 0
PCA = None

SOURCE_DIRS = {
    "phi4mr/spot_error": NODES / "phi4mr_spot",
    "phi4mr/identify": NODES / "phi4mr_identify",
    "qwen34b/spot_error": NODES / "qwen34b_spot",
    "qwen34b/identify": NODES / "qwen34b_identify",
}


def merge_doc(path: Path = MERGE) -> dict:
    return json.loads((path / "merge.json").read_text(encoding="utf-8"))


def node_theme_map(doc: dict | None = None) -> dict[tuple[str, int, int], int]:
    """``(source, layer, meta) -> merged theme id`` -- the only cross-model link in the design."""
    doc = doc or merge_doc()
    srcs = doc["sources"]
    return {
        (srcs[s], int(lay), int(m)): int(t)
        for s, lay, m, t in zip(
            doc["node_source"], doc["node_layer"], doc["node_meta"], doc["node_theme"], strict=True
        )
    }


def theme_names(doc: dict | None = None) -> dict[int, str]:
    doc = doc or merge_doc()
    return {int(t["theme"]): t["name"] for t in doc["themes"]}


def cached_key(source: str) -> dict:
    return json.loads((SOURCE_DIRS[source] / "clusters.json").read_text(encoding="utf-8"))["key"]


def cached_stack(source: str) -> np.ndarray:
    """The cached ``(n_layers, n_tokens)`` int16 per-token meta labels for a source."""
    return np.load(SOURCE_DIRS[source] / "clusters.npz")["stack"]
