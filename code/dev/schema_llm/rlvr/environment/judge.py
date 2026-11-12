"""Pure judge-result records and all-or-nothing root semantics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


class JudgeUnavailable(RuntimeError):
    """No valid judge verdict was obtained within the declared retry contract."""


@dataclass(frozen=True, slots=True)
class JudgeScore:
    sample_id: int
    task: str
    item_metadata: Mapping[str, Any]
    model_outcome_invalid: bool
    final_answer: str | None
    final_answer_hash: str | None
    cache_key: str | None
    normalized_verdict: Mapping[str, Any] | None
    graded_reward: float
    strict_reward: float
    cache_hit: bool
    latency_seconds: float | None
    attempts: int | None
    rate_limit_incidents: int | None
    model_failure_reason: str | None
    source: str = "trapi_judge"

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "task": self.task,
            "item_metadata": dict(self.item_metadata),
            "model_outcome_invalid": self.model_outcome_invalid,
            "final_answer": self.final_answer,
            "final_answer_hash": self.final_answer_hash,
            "cache_key": self.cache_key,
            "normalized_verdict": (
                dict(self.normalized_verdict)
                if self.normalized_verdict is not None
                else None
            ),
            "graded_reward": self.graded_reward,
            "strict_reward": self.strict_reward,
            "cache_hit": self.cache_hit,
            "latency_seconds": self.latency_seconds,
            "attempts": self.attempts,
            "rate_limit_incidents": self.rate_limit_incidents,
            "model_failure_reason": self.model_failure_reason,
            "source": self.source,
        }


def _validate_score(score: JudgeScore) -> None:
    if score.source not in ("trapi_judge", "deterministic_verifier"):
        raise ValueError(f"unsupported judge score source {score.source!r}")
    if score.sample_id < 0:
        raise ValueError("judge sample_id must be nonnegative")
    if not 0.0 <= score.graded_reward <= 1.0 or score.strict_reward not in (0.0, 1.0):
        raise ValueError("judge reward is outside the contract range")
    if score.model_outcome_invalid:
        if score.graded_reward != 0.0 or score.strict_reward != 0.0:
            raise ValueError("invalid model outcomes must receive zero reward")
        if any(
            value is not None
            for value in (
                score.final_answer,
                score.final_answer_hash,
                score.cache_key,
                score.normalized_verdict,
            )
        ):
            raise ValueError("invalid model outcomes cannot carry judge verdict metadata")
        if not score.model_failure_reason:
            raise ValueError("invalid model outcomes require a failure reason")
        return
    if score.model_failure_reason is not None:
        raise ValueError("valid model outcomes cannot carry a failure reason")
    if any(
        value is None
        for value in (
            score.final_answer,
            score.final_answer_hash,
            score.cache_key,
            score.normalized_verdict,
        )
    ):
        raise ValueError("valid model outcomes require complete judge verdict metadata")


def resolve_judge_root(
    results: Sequence[JudgeScore | JudgeUnavailable],
    *,
    group_size: int,
) -> tuple[JudgeScore, ...]:
    """Return one complete ordered root or propagate a judge failure without partial output."""
    if len(results) != group_size:
        raise ValueError(f"judge root must contain exactly {group_size} results")
    scores: list[JudgeScore] = []
    for result in results:
        if isinstance(result, JudgeUnavailable):
            raise result
        _validate_score(result)
        scores.append(result)
    if [score.sample_id for score in scores] != list(range(group_size)):
        raise ValueError(f"judge scorer did not return sample IDs 0..{group_size - 1}")
    if len({score.task for score in scores}) != 1:
        raise ValueError("judge root contains mixed tasks")
    return tuple(scores)
