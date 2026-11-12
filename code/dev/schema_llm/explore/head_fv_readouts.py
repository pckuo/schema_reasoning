"""Token-association arithmetic for exact-ID head-FV responses, not semantic scoring.

Assignments must come from uninjected replay at consumed generated-token positions.
The default is L16 fine clusters; an existing cross-layer theme reader requires
an explicit layer/label-space/count. This module checks contracts, not replay.
The first512 denominator is generated positions, including structural positions,
not the first 512 content tokens. Structural tokens break associated-token runs.
"""

from __future__ import annotations

from itertools import groupby

from .head_fv_specificity import EOS_IDS, digest

WINDOW = 512
N_CLUSTERS = 50
ALIGNMENT = "consumed-generated-token"


def _label_id(value: object, count: int) -> bool:
    return type(value) is int and 0 <= value < count


def summarize_response(
    generation: dict,
    assignment: dict | None,
    targets: dict[str, int],
    *,
    readout_error: str | None = None,
    reader_layer: int = 16,
    label_count: int = N_CLUSTERS,
    label_space: str = "fine_cluster",
) -> dict:
    """Return a result even for empty/truncated output or an explicitly failed readout.

    Malformed contracts raise instead of masquerading as measured zero effects.
    Readout errors retain known lengths but have null association metrics.
    """
    if (
        type(reader_layer) is not int or not 0 <= reader_layer < 32
        or type(label_count) is not int or label_count < 1
        or label_space not in ("fine_cluster", "cross_layer_theme")
    ):
        raise ValueError("invalid explicit reader/label-space specification")
    prompt, generated, full = (
        generation["prompt_ids"], generation["generated_ids"], generation["full_ids"]
    )
    mask = generation["content_mask"]
    if full != prompt + generated or generation["n_prompt"] != len(prompt):
        raise ValueError("inconsistent exact-ID response")
    if any(type(token) is not int or token < 0 for token in full):
        raise ValueError("token IDs must be nonnegative integers")
    if len(mask) != len(generated) or any(type(value) is not bool for value in mask):
        raise ValueError("content mask must align with generated token positions")
    reason = generation["stop_reason"]
    if reason not in ("eos", "budget", "error"):
        raise ValueError("unknown generation stop reason")
    if reason == "error" and not generation.get("error"):
        raise ValueError("generation errors must retain their error description")
    terminals = [i for i, token in enumerate(generated) if token in EOS_IDS]
    if terminals != ([len(generated) - 1] if reason == "eos" else []):
        raise ValueError("terminal IDs disagree with stop reason or contain post-EOS tokens")
    if terminals and mask[terminals[0]]:
        raise ValueError("EOS cannot count as content")
    if (
        not targets
        or any(not isinstance(name, str) or not name.strip() or not _label_id(cid, label_count)
               for name, cid in targets.items())
        or len(set(targets.values())) != len(targets)
    ):
        raise ValueError("targets require distinct valid label IDs and nonempty names")
    full_hash = digest(full)
    labels = None
    if assignment is None:
        if not isinstance(readout_error, str) or not readout_error.strip():
            raise ValueError("missing assignments require an explicit readout error")
    else:
        if readout_error is not None:
            raise ValueError("cannot report both measured assignments and a readout error")
        if (
            assignment["full_ids_sha256"] != full_hash
            or assignment["layer"] != reader_layer
            or assignment.get("label_space", "fine_cluster") != label_space
            or assignment.get("n_clusters" if label_space == "fine_cluster" else "n_labels")
            != label_count
            or assignment["alignment"] != ALIGNMENT
        ):
            raise ValueError("assignment does not match the frozen exact-ID readout contract")
        dictionary_hash = assignment["dictionary_sha256"]
        if (
            not isinstance(dictionary_hash, str) or len(dictionary_hash) != 64
            or any(c not in "0123456789abcdef" for c in dictionary_hash)
        ):
            raise ValueError("assignment must identify its frozen dictionary")
        labels = assignment["labels"]
        if len(labels) != len(generated):
            raise ValueError("assignments must align with every generated token position")
        if any(
            not _label_id(label, label_count) if content else label is not None
            for label, content in zip(labels, mask, strict=True)
        ):
            raise ValueError("content requires valid labels; masked positions require null")
    n_content, n_early = sum(mask), sum(mask[:WINDOW])
    primitives = {}
    for name, target in targets.items():
        metrics = dict.fromkeys((
            "associated_token_count", "associated_token_fraction",
            "first512_associated_token_count", "first512_fixed_fraction",
            "first512_content_fraction", "associated_run_count", "longest_associated_run",
        ))
        if labels is not None:
            hits = [content and label == target
                    for label, content in zip(labels, mask, strict=True)]
            count, early = sum(hits), sum(hits[:WINDOW])
            runs = [sum(1 for _ in group) for hit, group in groupby(hits) if hit]
            metrics.update(
                associated_token_count=count,
                associated_token_fraction=count / n_content if n_content else None,
                first512_associated_token_count=early,
                first512_fixed_fraction=early / WINDOW,
                first512_content_fraction=early / n_early if n_early else None,
                associated_run_count=len(runs),
                longest_associated_run=max(runs, default=0),
            )
        id_field = "cluster_id" if label_space == "fine_cluster" else "theme_id"
        primitives[name] = {id_field: target, **metrics}
    return {
        **{key: generation[key] for key in ("scenario_uid", "repeat", "protocol_hash", "health")
           if key in generation},
        "full_ids_sha256": full_hash,
        "dictionary_sha256": assignment["dictionary_sha256"] if assignment is not None else None,
        "reader_layer": reader_layer,
        "label_space": label_space,
        "label_count": label_count,
        "readout_status": "ok" if assignment is not None else "error",
        "readout_error": readout_error,
        "stop_reason": reason,
        "generation_error": generation.get("error"),
        "generated_token_count": len(generated),
        "content_token_count": n_content,
        "first512_content_token_count": n_early,
        "first512_position_denominator": WINDOW,
        "primitives": primitives,
    }
