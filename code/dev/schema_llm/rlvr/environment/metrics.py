"""Deterministic rollout and evaluation metric aggregation."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping
from statistics import mean
from typing import Any

EVAL_MATCHED_ADVANTAGE_TARGETS = {
    "identify": 0.5,
    "spot_canonical": 1.0 / 12.0,
    "spot_violation": 5.0 / 12.0,
}


def _mean(values: list[Any]) -> float | None:
    usable = [float(value) for value in values if value is not None]
    return mean(usable) if usable else None


def _uniform_value(values: list[Any], label: str) -> Any:
    usable = [value for value in values if value is not None]
    if not usable:
        return None
    encoded = {json.dumps(value, sort_keys=True) for value in usable}
    if len(encoded) != 1:
        raise ValueError(f"inconsistent {label} metadata across evaluated responses")
    return usable[0]


def require_exact_keys(expected: set[Any], actual: list[Any], label: str) -> None:
    counts = Counter(actual)
    duplicates = sorted(key for key, count in counts.items() if count > 1)
    missing = sorted(expected - set(actual))
    unknown = sorted(set(actual) - expected)
    if duplicates or missing or unknown:
        raise ValueError(
            f"{label} coverage mismatch: missing={missing[:3]} "
            f"unknown={unknown[:3]} duplicates={duplicates[:3]}"
        )


def summarize_graded_items(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize strict and graded verdicts for selection and diagnostics."""
    identify = [row for row in rows if row["task"] == "identify"]
    spot = [row for row in rows if row["task"] == "spot_error"]
    clean = [row for row in spot if row["condition"] == "canonical"]
    violation = [row for row in spot if row["condition"] != "canonical"]
    violation_types = sorted({str(row["condition"]) for row in violation})
    violation_by_type = {
        violation_type: _mean(
            [
                row["strict_reward"]
                for row in violation
                if row["condition"] == violation_type
            ]
        )
        for violation_type in violation_types
    }
    return {
        "n": len(rows),
        "counts": dict(Counter(row["task"] for row in rows)),
        "graded_reward": _mean([row["graded_reward"] for row in rows]),
        "strict_accuracy": _mean([row["strict_reward"] for row in rows]),
        "identify_strict_accuracy": _mean(
            [row["strict_reward"] for row in identify]
        ),
        "identify_role_f1": _mean(
            [
                row["reward_components"].get("role_f1")
                for row in identify
                if isinstance(row.get("reward_components"), dict)
            ]
        ),
        "clean_specificity": _mean([row["strict_reward"] for row in clean]),
        "violation_strict_accuracy": _mean(
            [row["strict_reward"] for row in violation]
        ),
        "violation_strict_by_type": violation_by_type,
        "violation_strict_macro": _mean(list(violation_by_type.values())),
        "decision_accuracy": _mean(
            [
                row["normalized_verdict"].get("decision_correct")
                for row in spot
                if isinstance(row.get("normalized_verdict"), dict)
            ]
        ),
        "localization_accuracy": _mean(
            [
                row["normalized_verdict"].get("localization_correct")
                for row in violation
                if isinstance(row.get("normalized_verdict"), dict)
            ]
        ),
        "type_accuracy": _mean(
            [
                row["normalized_verdict"].get("type_correct")
                for row in violation
                if isinstance(row.get("normalized_verdict"), dict)
            ]
        ),
        "deterministic_accuracy": _mean(
            [row.get("deterministic_success") for row in rows]
        ),
        "judge_deterministic_agreement": _mean(
            [row.get("judge_deterministic_agreement") for row in rows]
        ),
    }


def summarize_graded_rollouts(
    rows: list[dict[str, Any]],
    *,
    group_size: int,
    roots_per_update: int,
) -> dict[str, Any]:
    """Validate complete updates and summarize graded reward by frozen strata."""
    for index, row in enumerate(rows):
        invalid = row.get("model_outcome_invalid")
        graded = row.get("graded_reward")
        strict = row.get("strict_reward")
        if type(invalid) is not bool:
            raise ValueError(f"graded rollout row {index} has invalid model_outcome_invalid")
        if (
            type(graded) not in (int, float)
            or not math.isfinite(float(graded))
            or type(strict) not in (int, float)
            or not math.isfinite(float(strict))
            or not 0.0 <= float(graded) <= 1.0
            or float(strict) not in (0.0, 1.0)
        ):
            raise ValueError(f"graded rollout row {index} has invalid rewards")
        components = row.get("reward_components")
        if invalid:
            if float(graded) != 0.0 or float(strict) != 0.0:
                raise ValueError(
                    f"graded rollout row {index} gives reward to an invalid model outcome"
                )
            if components is not None:
                raise ValueError(
                    f"graded rollout row {index} gives components to an invalid model outcome"
                )
        elif not isinstance(components, Mapping):
            raise ValueError(
                f"graded rollout row {index} lacks valid reward components"
            )
    groups: dict[tuple[int, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            int(row["policy_global_step"]),
            int(row["root_slot"]),
            str(row["item_id"]),
        )
        groups.setdefault(key, []).append(row)
    bad_groups = {
        str(key): len(group)
        for key, group in groups.items()
        if len(group) != group_size
    }
    if bad_groups:
        raise ValueError(f"incomplete or duplicate graded root groups: {bad_groups}")
    bad_samples = {}
    for key, group in groups.items():
        try:
            sample_ids = sorted(
                int(row["sample_index"])
                for row in group
                if type(row["sample_index"]) is int
            )
        except KeyError as exc:
            raise ValueError(f"graded root group {key} is missing sample_index") from exc
        if len(sample_ids) != len(group) or sample_ids != list(range(group_size)):
            bad_samples[str(key)] = sample_ids
    if bad_samples:
        raise ValueError(f"invalid graded root sample coverage: {bad_samples}")
    by_step = Counter(key[0] for key in groups)
    bad_steps = {
        step: count
        for step, count in by_step.items()
        if count != roots_per_update
    }
    if bad_steps:
        raise ValueError(f"incomplete graded optimizer updates: {bad_steps}")
    bad_slots = {
        step: sorted(key[1] for key in groups if key[0] == step)
        for step in by_step
        if sorted(key[1] for key in groups if key[0] == step)
        != list(range(roots_per_update))
    }
    if bad_slots:
        raise ValueError(f"invalid graded root-slot coverage: {bad_slots}")
    if sorted(by_step) != list(range(len(by_step))):
        raise ValueError(
            f"graded optimizer steps are not contiguous from zero: {sorted(by_step)}"
        )

    root_summaries = []
    for key, group in sorted(groups.items()):
        values = [float(row["graded_reward"]) for row in group]
        average = sum(values) / len(values)
        variance = sum((value - average) ** 2 for value in values) / len(values)
        root_summaries.append(
            {
                "policy_global_step": key[0],
                "root_slot": key[1],
                "item_id": key[2],
                "task": group[0]["task"],
                "schema": group[0]["schema"],
                "condition": group[0]["condition"],
                "graded_reward_mean": average,
                "graded_reward_variance": variance,
                "strict_reward_mean": sum(
                    float(row["strict_reward"]) for row in group
                )
                / len(group),
            }
        )

    def average(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    strata = {}
    for task_name, schema_name, condition_name in sorted(
        {
            (row["task"], row["schema"], row["condition"])
            for row in rows
        }
    ):
        subset = [
            row
            for row in rows
            if (row["task"], row["schema"], row["condition"])
            == (task_name, schema_name, condition_name)
        ]
        root_subset = [
            root
            for root in root_summaries
            if (root["task"], root["schema"], root["condition"])
            == (task_name, schema_name, condition_name)
        ]
        component_names = sorted(
            {
                name
                for row in subset
                if isinstance(row.get("reward_components"), dict)
                for name, value in row["reward_components"].items()
                if type(value) in (bool, int, float)
            }
        )
        strata[f"{task_name}|{schema_name}|{condition_name}"] = {
            "responses": len(subset),
            "roots": len(root_subset),
            "graded_reward_mean": average(
                [float(row["graded_reward"]) for row in subset]
            ),
            "strict_reward_mean": average(
                [float(row["strict_reward"]) for row in subset]
            ),
            "zero_variance_root_rate": average(
                [
                    float(root["graded_reward_variance"] == 0.0)
                    for root in root_subset
                ]
            ),
            "cache_hit_rate": average(
                [float(row["judge"]["cache_hit"]) for row in subset]
            ),
            "judge_latency_seconds": average(
                [
                    float(row["judge"]["latency_seconds"])
                    for row in subset
                    if row["judge"]["latency_seconds"] is not None
                ]
            ),
            "component_means": {
                name: average(
                    [
                        float(row["reward_components"][name])
                        for row in subset
                        if isinstance(row.get("reward_components"), dict)
                        and type(row["reward_components"].get(name))
                        in (bool, int, float)
                    ]
                )
                for name in component_names
            },
            "deterministic_agreement": average(
                [float(row["judge_deterministic_agreement"]) for row in subset]
            ),
        }
    return {
        "judge_calls": len(rows),
        "unique_roots": len(groups),
        "optimizer_updates": len(by_step),
        "generated_completion_tokens": sum(
            int(row["completion_tokens"]) for row in rows
        ),
        "graded_reward_mean": average(
            [float(row["graded_reward"]) for row in rows]
        ),
        "strict_reward_mean": average(
            [float(row["strict_reward"]) for row in rows]
        ),
        "invalid_model_outcomes": sum(
            bool(row["model_outcome_invalid"]) for row in rows
        ),
        "truncations": sum(bool(row["truncated"]) for row in rows),
        "root_summaries": root_summaries,
        "strata": strata,
    }


def summarize_advantage_diagnostics(
    rows: list[dict[str, Any]],
    *,
    group_size: int,
    maximum_target_ratio: float = 1.5,
) -> dict[str, Any]:
    """Summarize Dr.GRPO advantage mass and spot positive rates by task stratum."""
    if type(group_size) is not int or group_size < 2:
        raise ValueError("advantage diagnostics require group_size >= 2")
    if (
        type(maximum_target_ratio) not in (int, float)
        or not math.isfinite(float(maximum_target_ratio))
        or float(maximum_target_ratio) <= 1.0
    ):
        raise ValueError("maximum_target_ratio must be finite and greater than one")

    groups: dict[tuple[int, int, str], list[dict[str, Any]]] = {}
    for index, row in enumerate(rows):
        try:
            key = (
                int(row["policy_global_step"]),
                int(row["root_slot"]),
                str(row["item_id"]),
            )
            sample_index = row["sample_index"]
            reward = row["graded_reward"]
            task = row["task"]
            condition = row["condition"]
        except KeyError as exc:
            raise ValueError(f"advantage diagnostic row {index} lacks {exc.args[0]}") from exc
        if (
            type(sample_index) is not int
            or type(reward) not in (int, float)
            or not math.isfinite(float(reward))
            or not 0.0 <= float(reward) <= 1.0
            or task not in {"identify", "spot_error"}
            or not isinstance(condition, str)
            or not condition
        ):
            raise ValueError(f"advantage diagnostic row {index} is invalid")
        groups.setdefault(key, []).append(row)

    root_diagnostics = []
    for key, group in sorted(groups.items()):
        if len(group) != group_size:
            raise ValueError(f"advantage diagnostic root {key} has {len(group)} responses")
        if sorted(row["sample_index"] for row in group) != list(range(group_size)):
            raise ValueError(f"advantage diagnostic root {key} has invalid sample coverage")
        identities = {(row["task"], row["schema"], row["condition"]) for row in group}
        if len(identities) != 1:
            raise ValueError(f"advantage diagnostic root {key} mixes strata")
        task, schema, condition = identities.pop()
        rewards = [float(row["graded_reward"]) for row in group]
        group_mean = sum(rewards) / group_size
        absolute_advantages = [abs(reward - group_mean) for reward in rewards]
        stratum = (
            "identify"
            if task == "identify"
            else "spot_canonical"
            if condition == "canonical"
            else "spot_violation"
        )
        positives: list[bool] = []
        if task == "spot_error":
            for row in group:
                verdict = row.get("normalized_verdict")
                if row.get("model_outcome_invalid") or not isinstance(verdict, Mapping):
                    continue
                decision_correct = verdict.get("decision_correct")
                if type(decision_correct) is not bool:
                    raise ValueError(f"advantage diagnostic root {key} has invalid spot verdict")
                positives.append(
                    not decision_correct if condition == "canonical" else decision_correct
                )
        root_diagnostics.append(
            {
                "policy_global_step": key[0],
                "root_slot": key[1],
                "item_id": key[2],
                "task": task,
                "schema": schema,
                "condition": condition,
                "stratum": stratum,
                "mean_absolute_advantage": sum(absolute_advantages) / group_size,
                "absolute_advantage_mass": sum(absolute_advantages),
                "positive_count": sum(positives),
                "positive_denominator": len(positives),
            }
        )

    total_mass = sum(root["absolute_advantage_mass"] for root in root_diagnostics)
    strata = {}
    warnings = []
    lower_ratio = 1.0 / float(maximum_target_ratio)
    for stratum, target in EVAL_MATCHED_ADVANTAGE_TARGETS.items():
        subset = [root for root in root_diagnostics if root["stratum"] == stratum]
        mass = sum(root["absolute_advantage_mass"] for root in subset)
        share = mass / total_mass if total_mass else 0.0
        ratio = share / target
        warning = ratio < lower_ratio or ratio > float(maximum_target_ratio)
        if warning:
            warnings.append(stratum)
        positive_count = sum(root["positive_count"] for root in subset)
        positive_denominator = sum(root["positive_denominator"] for root in subset)
        strata[stratum] = {
            "roots": len(subset),
            "responses": len(subset) * group_size,
            "mean_absolute_advantage": (
                sum(root["mean_absolute_advantage"] for root in subset) / len(subset)
                if subset
                else None
            ),
            "absolute_advantage_mass": mass,
            "advantage_mass_share": share,
            "advantage_mass_share_target": target,
            "advantage_mass_target_ratio": ratio,
            "prior_warning": warning,
            "positive_rate": (
                positive_count / positive_denominator if positive_denominator else None
            ),
            "positive_rate_denominator": positive_denominator,
        }
    clean_mass = strata["spot_canonical"]["absolute_advantage_mass"]
    violation_mass = strata["spot_violation"]["absolute_advantage_mass"]
    return {
        "roots": len(root_diagnostics),
        "responses": len(rows),
        "absolute_advantage_mass": total_mass,
        "strata": strata,
        "prior_warning_strata": warnings,
        "prior_balance_within_threshold": not warnings,
        "maximum_target_ratio": float(maximum_target_ratio),
        "clean_to_violation_advantage_mass_ratio": (
            clean_mass / violation_mass if violation_mass else None
        ),
    }
