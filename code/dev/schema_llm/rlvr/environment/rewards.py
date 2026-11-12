"""Deterministic graded and strict reward formulas."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import product
from typing import Any, Literal

from dev.schema_llm.subjects.judge import score_item

REWARD_CONTRACT_ID = "schema-graded-v1"
REWARD_SCHEMA_VERSION = "schema-graded-reward-v1"
ROLE_VERDICTS = frozenset(("correct", "wrong", "vague", "missing"))


class RewardContractError(ValueError):
    """A normalized verdict violates the frozen reward contract."""


@dataclass(frozen=True, slots=True)
class IdentifyRoleVerdict:
    id: int
    verdict: Literal["correct", "wrong", "vague", "missing"]
    role_as_stated: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class IdentifyVerdict:
    schema_correct: bool
    roles: tuple[IdentifyRoleVerdict, ...]
    extras: tuple[str, ...]
    parse_ok: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_correct": self.schema_correct,
            "roles": [role.as_dict() for role in self.roles],
            "extras": list(self.extras),
            "parse_ok": self.parse_ok,
        }


@dataclass(frozen=True, slots=True)
class SpotVerdict:
    decision_correct: bool
    localization_correct: bool | None
    type_correct: bool | None
    parse_ok: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision_correct": self.decision_correct,
            "localization_correct": self.localization_correct,
            "type_correct": self.type_correct,
            "parse_ok": self.parse_ok,
        }


@dataclass(frozen=True, slots=True)
class RewardResult:
    task: str
    graded_reward: float
    strict_reward: float
    strict_success: bool
    components: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_bool(value: object, field: str) -> bool:
    if type(value) is not bool:
        raise RewardContractError(f"{field} must be a JSON boolean")
    return value


def identify_verdict_from_dict(value: Mapping[str, Any]) -> IdentifyVerdict:
    """Construct an identify verdict without guessing missing fields."""
    try:
        schema_correct = _require_bool(value["schema_correct"], "schema_correct")
        raw_roles = value["roles"]
        raw_extras = value["extras"]
    except KeyError as exc:
        raise RewardContractError(f"missing identify field {exc.args[0]!r}") from exc
    if not isinstance(raw_roles, list):
        raise RewardContractError("roles must be a list")
    if not isinstance(raw_extras, list) or not all(
        isinstance(extra, str) for extra in raw_extras
    ):
        raise RewardContractError("extras must be a list of strings")
    parse_ok = _require_bool(value.get("parse_ok", True), "parse_ok")
    roles: list[IdentifyRoleVerdict] = []
    seen: set[int] = set()
    for raw_role in raw_roles:
        if not isinstance(raw_role, Mapping):
            raise RewardContractError("each role must be an object")
        role_id = raw_role.get("id")
        verdict = raw_role.get("verdict")
        if type(role_id) is not int:
            raise RewardContractError("role id must be an integer")
        if role_id in seen:
            raise RewardContractError(f"duplicate role id {role_id}")
        if verdict not in ROLE_VERDICTS:
            raise RewardContractError(f"invalid role verdict {verdict!r}")
        role_as_stated = raw_role.get("role_as_stated")
        if role_as_stated is not None and not isinstance(role_as_stated, str):
            raise RewardContractError("role_as_stated must be a string or null")
        seen.add(role_id)
        roles.append(IdentifyRoleVerdict(role_id, verdict, role_as_stated))
    return IdentifyVerdict(schema_correct, tuple(roles), tuple(raw_extras), parse_ok)


def spot_verdict_from_dict(
    value: Mapping[str, Any],
    *,
    has_violation: bool,
) -> SpotVerdict:
    """Construct a spot verdict and enforce condition-aware nullability."""
    try:
        decision = _require_bool(value["decision_correct"], "decision_correct")
        localization = value["localization_correct"]
        kind = value["type_correct"]
    except KeyError as exc:
        raise RewardContractError(f"missing spot field {exc.args[0]!r}") from exc
    if has_violation:
        if type(localization) is not bool or type(kind) is not bool:
            raise RewardContractError(
                "violation localization_correct and type_correct must be booleans"
            )
    elif localization is not None or kind is not None:
        raise RewardContractError(
            "canonical spot localization_correct and type_correct must be null"
        )
    parse_ok = _require_bool(value.get("parse_ok", True), "parse_ok")
    return SpotVerdict(decision, localization, kind, parse_ok)


def _roster_by_id(
    verdict: IdentifyVerdict,
    roster: Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    if roster is None or not roster:
        raise RewardContractError("identify reward requires a non-empty gold roster")
    expected: dict[int, Mapping[str, Any]] = {}
    for entry in roster:
        role_id = entry.get("id")
        if type(role_id) is not int or role_id in expected:
            raise RewardContractError("roster ids must be unique integers")
        expected[role_id] = entry
    actual = {role.id for role in verdict.roles}
    missing = sorted(set(expected) - actual)
    unknown = sorted(actual - set(expected))
    if missing or unknown or len(actual) != len(verdict.roles):
        raise RewardContractError(
            f"role coverage mismatch: missing={missing}, unknown={unknown}"
        )
    return [
        {
            "id": role.id,
            "value": (
                entry.get("name")
                or (entry.get("descriptors") or [None])[0]
                or str(role.id)
            ),
            "role_id": entry.get("role_id"),
            "distractor": bool(entry.get("distractor")),
            "accept_roles": list(entry.get("accept_roles") or []),
            "verdict": role.verdict,
            "role_as_stated": role.role_as_stated,
        }
        for role in verdict.roles
        for entry in (expected[role.id],)
    ]


def score_identify(
    verdict: IdentifyVerdict,
    *,
    roster: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if not verdict.parse_ok:
        raise RewardContractError("unparseable verdict cannot receive a reward")
    legacy_verdict = {
        "schema_granularity": "match" if verdict.schema_correct else "unrelated",
        "roles": _roster_by_id(verdict, roster),
        "extras": list(verdict.extras),
    }
    return score_item("identify", legacy_verdict)


def identify_reward(
    verdict: IdentifyVerdict,
    *,
    roster: Sequence[Mapping[str, Any]] | None = None,
) -> RewardResult:
    score = score_identify(verdict, roster=roster)
    schema = float(bool(score["schema_correct"]))
    role_f1 = float(score["role_f1"])
    strict = bool(score["correct"])
    graded = 0.15 * schema + 0.45 * role_f1 + 0.40 * float(strict)
    return RewardResult(
        task="identify",
        graded_reward=graded,
        strict_reward=float(strict),
        strict_success=strict,
        components={
            "schema_correct": bool(score["schema_correct"]),
            "role_f1": role_f1,
            "tp": int(score["n_correct"]) + int(score["n_recovered"]),
            "fn": int(score["n_roles"])
            - int(score["n_correct"])
            - int(score["n_recovered"]),
            "fp": int(score["n_wrong"])
            + int(score["n_extras"])
            + int(score["n_distractor_intrusions"]),
            "n_roles": int(score["n_roles"]),
            "n_correct": int(score["n_correct"]),
            "n_recovered": int(score["n_recovered"]),
            "n_wrong": int(score["n_wrong"]),
            "n_vague": int(score["n_vague"]),
            "n_missing": int(score["n_missing"]),
            "n_extras": int(score["n_extras"]),
            "n_distractor_intrusions": int(score["n_distractor_intrusions"]),
        },
    )


def spot_reward(verdict: SpotVerdict, *, has_violation: bool) -> RewardResult:
    if not verdict.parse_ok:
        raise RewardContractError("unparseable verdict cannot receive a reward")
    decision = float(verdict.decision_correct)
    if not has_violation:
        if verdict.localization_correct is not None or verdict.type_correct is not None:
            raise RewardContractError("canonical spot components must be null")
        return RewardResult(
            task="spot_error",
            graded_reward=decision,
            strict_reward=decision,
            strict_success=bool(decision),
            components={
                "has_violation": False,
                "decision_correct": bool(decision),
                "localization_correct": None,
                "type_correct": None,
            },
        )
    if type(verdict.localization_correct) is not bool or type(verdict.type_correct) is not bool:
        raise RewardContractError("violation spot components must be booleans")
    localization = float(verdict.localization_correct)
    kind = float(verdict.type_correct)
    strict = bool(decision and localization and kind)
    graded = 0.35 * decision + 0.15 * decision * localization + 0.10 * decision * kind
    graded += 0.40 * float(strict)
    return RewardResult(
        task="spot_error",
        graded_reward=graded,
        strict_reward=float(strict),
        strict_success=strict,
        components={
            "has_violation": True,
            "decision_correct": bool(decision),
            "localization_correct": bool(localization),
            "type_correct": bool(kind),
        },
    )


def compute_reward(
    task: str,
    verdict: IdentifyVerdict | SpotVerdict,
    *,
    roster: Sequence[Mapping[str, Any]] | None = None,
    has_violation: bool | None = None,
) -> RewardResult:
    if task == "identify":
        if not isinstance(verdict, IdentifyVerdict):
            raise RewardContractError("identify requires an IdentifyVerdict")
        return identify_reward(verdict, roster=roster)
    if task == "spot_error":
        if not isinstance(verdict, SpotVerdict) or has_violation is None:
            raise RewardContractError("spot_error requires SpotVerdict and has_violation")
        return spot_reward(verdict, has_violation=has_violation)
    raise RewardContractError(f"unsupported reward task {task!r}")


def _sha256_json(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def reward_contract_hash() -> str:
    contract = {
        "id": REWARD_CONTRACT_ID,
        "schema": REWARD_SCHEMA_VERSION,
        "identify": [0.15, 0.45, 0.40],
        "violation": [0.35, 0.15, 0.10, 0.40],
        "clean": "decision",
        "role_verdicts": sorted(ROLE_VERDICTS),
    }
    return _sha256_json(contract)


def _assert_strict_dominance(result: RewardResult) -> None:
    if result.strict_success:
        if result.graded_reward != 1.0 or result.strict_reward != 1.0:
            raise AssertionError(f"strict reward contract violated: {result}")
    elif not 0.0 <= result.graded_reward < 1.0:
        raise AssertionError(f"non-strict reward contract violated: {result}")


def assert_legal_reward_combinations(max_roles: int = 6) -> None:
    if max_roles < 1:
        raise ValueError("max_roles must be positive")
    statuses = tuple(sorted(ROLE_VERDICTS))
    for n_roles in range(1, max_roles + 1):
        roster = [{"id": index, "distractor": False} for index in range(n_roles)]
        roster.append({"id": n_roles, "distractor": True})
        for schema, person_states, distractor_state, has_extra in product(
            (False, True),
            product(statuses, repeat=n_roles),
            ("correct", "wrong"),
            (False, True),
        ):
            roles = tuple(
                [
                    *(
                        IdentifyRoleVerdict(index, state)
                        for index, state in enumerate(person_states)
                    ),
                    IdentifyRoleVerdict(n_roles, distractor_state),
                ]
            )
            result = identify_reward(
                IdentifyVerdict(
                    schema,
                    roles,
                    ("invented participant",) if has_extra else (),
                ),
                roster=roster,
            )
            _assert_strict_dominance(result)
    for decision, localization, kind in product((False, True), repeat=3):
        _assert_strict_dominance(
            spot_reward(
                SpotVerdict(decision, localization, kind),
                has_violation=True,
            )
        )
    for decision in (False, True):
        _assert_strict_dominance(
            spot_reward(
                SpotVerdict(decision, None, None),
                has_violation=False,
            )
        )
