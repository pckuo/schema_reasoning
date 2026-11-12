"""Deterministic task-root identity, ordering, and split-isolation contracts."""

from __future__ import annotations

import hashlib
import json
import random
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping
from copy import deepcopy
from typing import Any

EVAL_MATCHED_CONDITIONS = (
    "canonical",
    "wrong_order",
    "missing_scene",
    "wrong_action",
    "role_swap",
    "prop_mismatch",
)
EVAL_MATCHED_SCHEMAS = ("restaurant", "airport")
EVAL_MATCHED_SUPERBLOCK_SPOTS = (
    (("canonical", "restaurant"), ("wrong_order", "airport")),
    (("canonical", "airport"), ("wrong_order", "restaurant")),
    (("missing_scene", "restaurant"), ("wrong_action", "airport")),
    (("missing_scene", "airport"), ("wrong_action", "restaurant")),
    (("role_swap", "restaurant"), ("prop_mismatch", "airport")),
    (("role_swap", "airport"), ("prop_mismatch", "restaurant")),
)


def normalize_prompt(prompt: str) -> str:
    """Return the canonical text used for exact and near-exact leakage checks."""
    text = unicodedata.normalize("NFKC", prompt).casefold()
    return " ".join(text.split())


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(normalize_prompt(prompt).encode("utf-8")).hexdigest()


def _sha256_parts(
    contract_seed: int,
    split: str,
    block: int | str,
    semantic_slot: str,
    replacement_index: int,
) -> bytes:
    payload = json.dumps(
        [contract_seed, split, block, semantic_slot, replacement_index],
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).digest()


def derive_seed(
    contract_seed: int,
    split: str,
    block: int | str,
    semantic_slot: str,
    replacement_index: int = 0,
) -> int:
    """Return the deterministic 64-bit seed for one root or replacement."""
    return int.from_bytes(
        _sha256_parts(contract_seed, split, block, semantic_slot, replacement_index)[:8],
        "big",
    )


def sha256_order_key(
    contract_seed: int,
    split: str,
    block: int | str,
    semantic_slot: str,
    replacement_index: int = 0,
) -> str:
    return _sha256_parts(
        contract_seed,
        split,
        block,
        semantic_slot,
        replacement_index,
    ).hex()


def _assert_unique(rows: Iterable[Mapping[str, Any]], field: str, split: str) -> None:
    values = [row[field] for row in rows]
    duplicates = [value for value, count in Counter(values).items() if count > 1]
    if duplicates:
        raise AssertionError(f"{split} has duplicate {field}: {duplicates[:3]}")


def _provenance_keys(rows: Iterable[Mapping[str, Any]]) -> set[tuple[str, int]]:
    return {
        (str(row["schema"]), int(row["generator_seed"]))
        for row in rows
        if row.get("generator_seed") is not None
    }


def assert_isolation(
    train: list[Mapping[str, Any]],
    dev: list[Mapping[str, Any]],
    core: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Fail closed on item, prompt, and generator-provenance leakage."""
    for name, rows in (("train", train), ("dev", dev)):
        _assert_unique(rows, "item_id", name)
        _assert_unique(rows, "normalized_prompt_hash", name)

    def disjoint(
        left: list[Mapping[str, Any]],
        right: list[Mapping[str, Any]],
        field: str,
        names: str,
    ) -> None:
        overlap = {row[field] for row in left} & {row[field] for row in right}
        if overlap:
            raise AssertionError(
                f"{names} overlap on {field}: {sorted(overlap)[:3]}"
            )

    disjoint(train, dev, "item_id", "train/dev")
    disjoint(train, core, "item_id", "train/core")
    disjoint(dev, core, "item_id", "dev/core")
    disjoint(train, dev, "normalized_prompt_hash", "train/dev")
    disjoint(train, core, "normalized_prompt_hash", "train/core")
    disjoint(dev, core, "normalized_prompt_hash", "dev/core")

    train_provenance = _provenance_keys(train)
    dev_provenance = _provenance_keys(dev)
    core_provenance = _provenance_keys(core)
    if not train_provenance.isdisjoint(dev_provenance):
        raise AssertionError("train/dev reuse generator provenance")
    if not train_provenance.isdisjoint(core_provenance):
        raise AssertionError("train provenance can regenerate a core recipe")
    if not dev_provenance.isdisjoint(core_provenance):
        raise AssertionError("dev provenance can regenerate a core recipe")

    return {
        "status": "pass",
        "counts": {
            "train": len(train),
            "dev": len(dev),
            "core": len(core),
            "train_by_task": dict(Counter(row["task"] for row in train)),
            "dev_by_task": dict(Counter(row["task"] for row in dev)),
            "core_by_task": dict(Counter(row["task"] for row in core)),
        },
        "checks": [
            "pairwise item-id disjointness",
            "pairwise normalized-prompt-hash disjointness",
            "internal train/dev uniqueness",
            "pairwise generator-provenance disjointness",
        ],
    }


def order_for_training(
    rows: list[Mapping[str, Any]],
    seed: int = 0,
) -> list[Mapping[str, Any]]:
    """Deterministically alternate tasks while balancing schemas and conditions."""
    rng = random.Random(seed)
    task_streams: dict[str, list[Mapping[str, Any]]] = {}
    for task in sorted({str(row["task"]) for row in rows}):
        buckets: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
        for row in rows:
            if row["task"] != task:
                continue
            key = (
                str(row["schema"]),
                str(row["control_factors"]["condition"]),
            )
            buckets.setdefault(key, []).append(row)
        for bucket in buckets.values():
            rng.shuffle(bucket)
        stream: list[Mapping[str, Any]] = []
        keys = sorted(buckets)
        while any(buckets.values()):
            for key in keys:
                if buckets[key]:
                    stream.append(buckets[key].pop())
        task_streams[task] = stream

    ordered: list[Mapping[str, Any]] = []
    task_names = sorted(task_streams)
    while any(task_streams.values()):
        for task in task_names:
            if task_streams[task]:
                ordered.append(task_streams[task].pop(0))
    return ordered


def _eval_matched_order_key(
    seed: int,
    purpose: str,
    *parts: object,
) -> str:
    payload = json.dumps(
        ["schema-rlvr-eval-matched-sampling-v1", seed, purpose, *parts],
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _eval_matched_bucket(
    rows: list[Mapping[str, Any]],
    *,
    task: str,
    condition: str,
    schema: str,
    seed: int,
    count: int,
) -> list[Mapping[str, Any]]:
    bucket = [
        row
        for row in rows
        if row.get("task") == task
        and row.get("condition") == condition
        and row.get("schema") == schema
    ]
    bucket.sort(
        key=lambda row: (
            _eval_matched_order_key(
                seed,
                "stratum",
                task,
                condition,
                schema,
                row.get("item_id"),
            ),
            str(row.get("item_id")),
        )
    )
    if len(bucket) < count:
        raise ValueError(
            "frozen root pool cannot satisfy eval-matched stratum "
            f"{task}/{condition}/{schema}: need {count}, found {len(bucket)}"
        )
    return bucket[:count]


def select_eval_matched_training_roots(
    rows: list[Mapping[str, Any]],
    *,
    seed: int,
    updates: int = 252,
) -> list[dict[str, Any]]:
    """Select and schedule an exact 1/6 spot-condition training prior."""
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("eval-matched seed must be an unsigned 64-bit integer")
    if type(updates) is not int or updates < 6 or updates % 6:
        raise ValueError("eval-matched updates must be a positive multiple of six")
    if not rows or any(not isinstance(row, Mapping) for row in rows):
        raise ValueError("eval-matched source rows must be nonempty mappings")
    item_ids = [row.get("item_id") for row in rows]
    if any(not isinstance(item_id, str) or not item_id for item_id in item_ids):
        raise ValueError("eval-matched source rows require nonempty item IDs")
    duplicates = [item_id for item_id, count in Counter(item_ids).items() if count > 1]
    if duplicates:
        raise ValueError(f"eval-matched source rows contain duplicate item IDs: {duplicates[:3]}")

    per_spot_cell = updates // 6
    buckets: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for schema in EVAL_MATCHED_SCHEMAS:
        buckets[("identify", "canonical", schema)] = _eval_matched_bucket(
            rows,
            task="identify",
            condition="canonical",
            schema=schema,
            seed=seed,
            count=updates,
        )
        for condition in EVAL_MATCHED_CONDITIONS:
            buckets[("spot_error", condition, schema)] = _eval_matched_bucket(
                rows,
                task="spot_error",
                condition=condition,
                schema=schema,
                seed=seed,
                count=per_spot_cell,
            )

    consumed = Counter()

    def take(task: str, condition: str, schema: str) -> Mapping[str, Any]:
        key = (task, condition, schema)
        index = consumed[key]
        consumed[key] += 1
        return buckets[key][index]

    selected: list[dict[str, Any]] = []
    for step in range(updates):
        spot_cells = EVAL_MATCHED_SUPERBLOCK_SPOTS[step % 6]
        candidates = [
            take("identify", "canonical", "restaurant"),
            take("identify", "canonical", "airport"),
            *(take("spot_error", condition, schema) for condition, schema in spot_cells),
        ]
        candidates.sort(
            key=lambda row: (
                _eval_matched_order_key(
                    seed,
                    "physical-slot",
                    step,
                    row["item_id"],
                ),
                str(row["item_id"]),
            )
        )
        for root_slot, source in enumerate(candidates):
            row = deepcopy(dict(source))
            row["source_root_block"] = row["root_block"]
            row["source_root_slot"] = row["root_slot"]
            row["root_block"] = step
            row["root_slot"] = root_slot
            row["eval_matched_superblock"] = step // 6
            row["eval_matched_superblock_offset"] = step % 6
            selected.append(row)

    validate_eval_matched_training_roots(selected, updates=updates)
    return selected


def validate_eval_matched_training_roots(
    rows: list[Mapping[str, Any]],
    *,
    updates: int,
) -> dict[str, Any]:
    """Validate exact six-update scheduling and return its composition summary."""
    if type(updates) is not int or updates < 6 or updates % 6:
        raise ValueError("eval-matched updates must be a positive multiple of six")
    if len(rows) != updates * 4:
        raise ValueError(f"eval-matched root count is {len(rows)}, expected {updates * 4}")
    item_ids = [row.get("item_id") for row in rows]
    if len(set(item_ids)) != len(item_ids):
        raise ValueError("eval-matched training roots reuse item IDs")

    by_block: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        block = row.get("root_block")
        slot = row.get("root_slot")
        if type(block) is not int or type(slot) is not int:
            raise ValueError("eval-matched roots require integer block and slot")
        by_block.setdefault(block, []).append(row)
    if sorted(by_block) != list(range(updates)):
        raise ValueError("eval-matched root blocks must be contiguous from zero")

    for step in range(updates):
        block = sorted(by_block[step], key=lambda row: row["root_slot"])
        if [row["root_slot"] for row in block] != list(range(4)):
            raise ValueError(f"eval-matched block {step} lacks root slots 0..3")
        identify = [row for row in block if row.get("task") == "identify"]
        spot = [row for row in block if row.get("task") == "spot_error"]
        if Counter(row.get("schema") for row in identify) != Counter(EVAL_MATCHED_SCHEMAS):
            raise ValueError(f"eval-matched block {step} lacks one identify root per schema")
        actual_spot = Counter((row.get("condition"), row.get("schema")) for row in spot)
        expected_spot = Counter(EVAL_MATCHED_SUPERBLOCK_SPOTS[step % 6])
        if actual_spot != expected_spot:
            raise ValueError(
                f"eval-matched block {step} spot schedule is "
                f"{dict(actual_spot)}, expected {dict(expected_spot)}"
            )
        if any(
            row.get("eval_matched_superblock") != step // 6
            or row.get("eval_matched_superblock_offset") != step % 6
            for row in block
        ):
            raise ValueError(f"eval-matched block {step} has invalid superblock metadata")

    cell_counts = Counter(
        (str(row["task"]), str(row["condition"]), str(row["schema"])) for row in rows
    )
    expected_cell_counts = {
        ("identify", "canonical", schema): updates for schema in EVAL_MATCHED_SCHEMAS
    }
    expected_cell_counts.update(
        {
            ("spot_error", condition, schema): updates // 6
            for condition in EVAL_MATCHED_CONDITIONS
            for schema in EVAL_MATCHED_SCHEMAS
        }
    )
    if cell_counts != Counter(expected_cell_counts):
        raise ValueError("eval-matched aggregate stratum counts are invalid")
    return {
        "updates": updates,
        "roots": len(rows),
        "superblocks": updates // 6,
        "by_task": dict(Counter(str(row["task"]) for row in rows)),
        "spot_by_condition": dict(
            Counter(str(row["condition"]) for row in rows if row["task"] == "spot_error")
        ),
        "by_task_condition_schema": {
            "|".join(key): count for key, count in sorted(cell_counts.items())
        },
    }
