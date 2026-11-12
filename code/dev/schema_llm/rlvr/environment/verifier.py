"""Deterministic rule-based final-answer verifiers for identify and spot_error.

Torch-free reward source graduated from the flat Dr.GRPO baseline; a modular alternative to the
live LLM judge, parity-checked against the original in tests/test_rlvr_verifier.py.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from dev.schema_llm.core.llm import strip_reasoning
from dev.schema_llm.core.spec import load_schema

_PAIR_SEPARATORS = (":", "->", "=>", "—", "–")
_ERROR_RE = re.compile(r"(?im)^\s*(?:[-*]\s*)?(?:\*\*)?error(?:\*\*)?\s*:\s*(yes|no)\b")
_WHERE_RE = re.compile(r"(?im)^\s*(?:[-*]\s*)?(?:\*\*)?where(?:\*\*)?\s*:\s*(.+?)\s*$")
_WHAT_RE = re.compile(r"(?im)^\s*(?:[-*]\s*)?(?:\*\*)?what(?:\*\*)?\s*:\s*(.+?)\s*$")
_SCHEMA_RE = re.compile(
    r"(?im)^\s*(?:[-*]\s*)?(?:\*\*)?"
    r"(?:schema(?:\s*/\s*scenario)?|scenario|script)(?:\*\*)?\s*:\s*(.+?)\s*$"
)
_POST_ANSWER_HEADING_RE = re.compile(
    r"(?i)^\s*(?:#+\s*)?(?:\*\*)?(?:reasoning|explanation|rationale|analysis)"
    r"(?:\*\*)?\s*:"
)
_TYPE_ALIASES = {
    "wrong_order": (
        "wrong order",
        "out of order",
        "reversed order",
        "before it should",
        "sequence",
    ),
    "missing_scene": ("missing step", "missing event", "skipped step", "omitted", "never happens"),
    "wrong_action": (
        "wrong action",
        "not their action",
        "isn't theirs to do",
        "does someone else's",
    ),
    "role_swap": (
        "role swap",
        "roles swapped",
        "parts swapped",
        "parts are swapped",
        "swapped roles",
        "each other's role",
    ),
    "prop_mismatch": ("wrong object", "incorrect object", "object mismatch", "instead of the"),
}
_SCENE_ALIASES = {
    "restaurant": {
        "arrive": ("arrive", "arrival", "enter", "come in", "came in", "through the door"),
        "seated": ("seat", "seated", "shown to", "led to", "place to sit", "table"),
        "order": ("order", "menu", "request", "choice", "asked for"),
        "eat": ("eat", "ate", "meal", "dish", "food", "brought out", "served"),
        "pay": ("pay", "paid", "payment", "bill", "check", "settle", "amount owed"),
        "leave": ("leave", "left", "exit", "headed out", "stepped outside"),
    },
    "airport": {
        "arrive": ("arrive", "arrival", "enter", "came in", "building"),
        "check_in": ("check in", "check-in", "front counter", "booking", "details", "tagged"),
        "security": ("security", "checkpoint", "scanner", "screening", "inspected"),
        "walk_gate": ("walk to", "waiting area", "departure point", "headed for", "made their way"),
        "board": ("board", "boarding", "get on", "went down the ramp", "pass is checked"),
        "fly": ("fly", "flight", "takeoff", "in the air", "controls", "ride takes place"),
        "deplane": ("deplane", "disembark", "get off", "stepped off", "far end", "jet bridge"),
        "baggage": ("baggage", "luggage", "carousel", "collect", "bag", "belt"),
    },
}
_ROLE_ACTION_ALIASES = {
    "restaurant": {
        "PATRON": ("ordered", "ate", "paid"),
        "FRONT_AGENT": ("greeted", "guided", "led to", "showed to", "seated"),
        "SERVICE_AGENT": (
            "brought the menu",
            "took the order",
            "delivered the meal",
            "presented the bill",
            "served",
        ),
        "PRODUCER": ("prepared the meal", "worked in back", "kitchen", "prepared food"),
    },
    "airport": {
        "PATRON": (),
        "FRONT_AGENT": ("taking details", "check in", "check-in", "ticket counter"),
        "GATEKEEPER": ("checking belongings", "security screening", "checkpoint"),
        "SERVICE_AGENT": ("boarding", "verifying the pass", "granting access", "gate counter"),
        "PRODUCER": ("operating the aircraft", "operates the aircraft", "flying"),
    },
}
_HARMLESS_DISTRACTOR_TERMS = (
    "background",
    "bystander",
    "passerby",
    "spectator",
    "not involved",
    "not part of",
    "unrelated",
    "not a person",
)


@dataclass(frozen=True)
class RewardResult:
    task: str
    reward: float
    correct: bool
    format_valid: bool
    parsed: dict[str, Any]
    diagnostics: dict[str, Any]
    failure_reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _norm(text: object) -> str:
    value = str(text).casefold().replace("_", " ")
    value = re.sub(r"[`*_#\"'()\[\]{}]", " ", value)
    value = re.sub(r"[^a-z0-9\s/-]", " ", value)
    return " ".join(value.split())


def _entity_norm(text: object) -> str:
    value = _norm(text)
    for article in ("the ", "a ", "an "):
        if value.startswith(article):
            return value[len(article) :]
    return value


def _contains_phrase(text: str, phrase: str) -> bool:
    return bool(re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text))


def _response_final(response: str, truncated: bool) -> tuple[str, bool, list[str]]:
    reasons: list[str] = []
    has_open = "<think>" in response
    has_close = "</think>" in response
    if truncated:
        reasons.append("truncated")
    if has_open and not has_close:
        reasons.append("unclosed reasoning block")
        return "", False, reasons
    final = strip_reasoning(response).strip()
    if not final:
        reasons.append("empty final answer")
    return final, bool(final) and not reasons, reasons


def _role_aliases(gold: dict) -> dict[str, tuple[str, ...]]:
    schema = load_schema(gold["schema"])
    aliases: dict[str, tuple[str, ...]] = {}
    for role_id, role in schema.roles.items():
        if role.kind != "person":
            continue
        values = [
            role_id.replace("_", " "),
            *role.forbidden,
            *role.accept,
            *_ROLE_ACTION_ALIASES[schema.name].get(role_id, ()),
        ]
        aliases[role_id] = tuple(dict.fromkeys(_norm(value) for value in values if value))
    return aliases


def _matched_roles(text: str, aliases: dict[str, tuple[str, ...]]) -> set[str]:
    normalized = _norm(text)
    hits: list[tuple[int, str]] = []
    for role_id, terms in aliases.items():
        for term in terms:
            if term and _contains_phrase(normalized, term):
                hits.append((len(term.split()), role_id))
    if not hits:
        return set()
    longest = max(length for length, _ in hits)
    return {role_id for length, role_id in hits if length == longest}


def _line_pair(line: str) -> tuple[str, str] | None:
    stripped = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line.strip())
    if stripped.startswith("|") and stripped.endswith("|"):
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) >= 2 and not set(_norm(cells[0])) <= {"-", " "}:
            return cells[0], cells[1]
    for separator in _PAIR_SEPARATORS:
        if separator in stripped:
            left, right = stripped.split(separator, 1)
            if left.strip() and right.strip():
                return left.strip(), right.strip()
    match = re.match(r"^(.+?)\s+\(([^()]+)\)\s*$", stripped)
    return (match.group(1), match.group(2)) if match else None


def _schema_matches(stated: str | None, schema_name: str) -> bool:
    if not stated:
        return False
    schema = load_schema(schema_name)
    accepted = [schema.name, *schema.forbidden_labels, *schema.schema_accept]
    if schema_name == "airport":
        accepted.extend(
            ("airplane boarding", "plane boarding", "airport travel", "airport journey")
        )
    elif schema_name == "restaurant":
        accepted.append("meal service")
    normalized = _norm(stated)
    if normalized == "airport security checkpoint":
        return False
    return any(_contains_phrase(normalized, _norm(term)) for term in accepted)


def _identify_answer_block(final: str) -> str:
    """Ignore an explanatory section that a model appends after its formatted answer."""
    lines = final.splitlines()
    schema_line = next((i for i, line in enumerate(lines) if _SCHEMA_RE.match(line)), None)
    if schema_line is None:
        return final
    kept: list[str] = []
    for line in lines[schema_line:]:
        if kept and _POST_ANSWER_HEADING_RE.match(line):
            break
        kept.append(line)
    return "\n".join(kept)


def _lookup_entity(normalized: str, entities: dict[str, str]) -> str | None:
    exact = entities.get(normalized)
    if exact is not None:
        return exact
    candidates = [
        (len(key.split()), value)
        for key, value in entities.items()
        if normalized.startswith(f"{key} ")
        or (len(key.split()) > 1 and f" {key} " in f" {normalized} ")
    ]
    return max(candidates, default=(0, None))[1]


def _lookup_distractor(normalized: str, distractors: dict[str, str]) -> str | None:
    exact = _lookup_entity(normalized, distractors)
    if exact is not None:
        return exact
    ignored = {
        "a",
        "an",
        "the",
        "person",
        "man",
        "woman",
        "old",
        "young",
        "two",
        "with",
        "who",
        "near",
    }
    words = {word for word in normalized.split() if word not in ignored and len(word) >= 4}
    candidates = []
    for key, value in distractors.items():
        key_words = {word for word in key.split() if word not in ignored and len(word) >= 4}
        overlap = len(words & key_words)
        if overlap:
            candidates.append((overlap, len(key_words), value))
    return max(candidates, default=(0, 0, None))[2]


def _entities_in_text(text: str, entities: dict[str, str]) -> set[str]:
    normalized = _entity_norm(text)
    return {
        value
        for key, value in entities.items()
        if key and _contains_phrase(normalized, key)
    }


def _harmless_distractor_label(left: str, right: str, role_hits: set[str]) -> bool:
    if role_hits == {"PATRON"}:
        return True
    if not role_hits:
        normalized_right = _norm(right)
        if "staff" in normalized_right:
            return False
        if any(term in normalized_right for term in _HARMLESS_DISTRACTOR_TERMS):
            return True
        return True
    return False


def verify_identify(gold: dict, response: str, *, truncated: bool = False) -> RewardResult:
    final, structure_ok, reasons = _response_final(response, truncated)
    answer_block = _identify_answer_block(final)
    schema_match = _SCHEMA_RE.findall(answer_block)
    schema_as_stated = schema_match[-1].strip() if schema_match else None
    aliases = _role_aliases(gold)
    people = {role["value"]: role for role in gold.get("roles", [])}
    entity_keys = {_entity_norm(value): value for value in people}
    distractor_keys = {
        _entity_norm(value): value
        for value in gold.get("distractors", [])
        if isinstance(value, str)
    }
    assignments: dict[str, list[str]] = {value: [] for value in people}
    unbound_roles: list[str] = []
    extras: list[str] = []
    distractor_intrusions: list[str] = []
    prose_contradictions: list[str] = []

    for line in answer_block.splitlines():
        pair = _line_pair(line)
        if pair is None:
            prose_roles = _matched_roles(line, aliases)
            for person in _entities_in_text(line, entity_keys):
                if prose_roles and prose_roles != {people[person]["role_id"]}:
                    prose_contradictions.append(line.strip())
            continue
        left, right = pair
        if _norm(left) in {"schema", "schema/scenario", "schema scenario", "scenario", "script"}:
            continue
        if _norm(left) in {"person", "name", "entity"} and _norm(right) in {"role", "function"}:
            continue
        if "name or description from the passage" in _norm(left):
            continue
        left_key = _entity_norm(left)
        right_key = _entity_norm(right)
        role_hits = _matched_roles(right, aliases)
        person = _lookup_entity(left_key, entity_keys)
        if person is None:
            reverse_person = _lookup_entity(right_key, entity_keys)
            reverse_roles = _matched_roles(left, aliases)
            if reverse_person is not None and reverse_roles:
                assignments[reverse_person].extend(sorted(reverse_roles))
                continue
        if person is not None:
            assignments[person].extend(sorted(role_hits))
            continue
        distractor = _lookup_distractor(left_key, distractor_keys)
        if distractor is not None:
            if not _harmless_distractor_label(left, right, role_hits):
                distractor_intrusions.append(line.strip())
            continue
        left_roles = _matched_roles(left, aliases)
        if len(left_roles) == 1 and role_hits == left_roles and "PATRON" not in left_roles:
            unbound_roles.extend(left_roles)
            continue
        if not any(term in _norm(right) for term in _HARMLESS_DISTRACTOR_TERMS):
            extras.append(line.strip())

    for role_id in unbound_roles:
        missing = [
            value
            for value, role in people.items()
            if role["role_id"] == role_id and not assignments[value]
        ]
        if len(missing) == 1:
            assignments[missing[0]].append(role_id)

    role_verdicts: list[dict[str, Any]] = []
    n_correct = 0
    for value, role in people.items():
        found = set(assignments[value])
        correct = found == {role["role_id"]}
        n_correct += int(correct)
        role_verdicts.append(
            {
                "value": value,
                "gold_role": role["role_id"],
                "parsed_roles": sorted(found),
                "correct": correct,
                "missing": not found,
                "contradictory": len(found) > 1,
            }
        )
    n_roles = len(role_verdicts)
    role_accuracy = n_correct / n_roles if n_roles else 0.0
    schema_correct = _schema_matches(schema_as_stated, gold["schema"])
    complete_mapping = n_roles > 0 and n_correct == n_roles
    format_valid = structure_ok and schema_as_stated is not None and all(
        assignments[value] for value in people
    )
    correct = (
        format_valid
        and schema_correct
        and complete_mapping
        and not extras
        and not distractor_intrusions
        and not prose_contradictions
    )
    if not schema_correct:
        reasons.append("schema mismatch or missing")
    if not complete_mapping:
        reasons.append("incomplete or incorrect role map")
    if extras:
        reasons.append("unsupported role assignments")
    if distractor_intrusions:
        reasons.append("distractor assigned a schema role")
    if prose_contradictions:
        reasons.append("contradictory prose in final answer")
    return RewardResult(
        task="identify",
        reward=float(correct),
        correct=correct,
        format_valid=format_valid,
        parsed={"schema_as_stated": schema_as_stated, "assignments": assignments},
        diagnostics={
            "schema_correct": schema_correct,
            "role_accuracy": role_accuracy,
            "roles": role_verdicts,
            "extras": extras,
            "distractor_intrusions": distractor_intrusions,
            "prose_contradictions": prose_contradictions,
        },
        failure_reasons=tuple(dict.fromkeys(reasons)),
    )


def _scene_matches(schema: str, scene: str, text: str) -> bool:
    normalized = _norm(text)
    aliases = _SCENE_ALIASES[schema].get(scene, (scene.replace("_", " "),))
    return any(_contains_phrase(normalized, _norm(alias)) for alias in aliases)


def _type_diagnostics(expected: str | None, text: str) -> tuple[bool | None, list[str]]:
    if expected is None:
        return None, []
    normalized = _norm(text)
    found = [
        kind
        for kind, terms in _TYPE_ALIASES.items()
        if any(_contains_phrase(normalized, _norm(term)) for term in terms)
    ]
    return expected in found, [kind for kind in found if kind != expected]


def _empty_spot_placeholder(text: str | None) -> bool:
    if text is None:
        return True
    normalized = _norm(text)
    return normalized in {
        "",
        "none",
        "n a",
        "na",
        "no error",
        "no error detected",
        "not applicable",
    }


def verify_spot_error(gold: dict, response: str, *, truncated: bool = False) -> RewardResult:
    final, structure_ok, reasons = _response_final(response, truncated)
    error_matches = _ERROR_RE.findall(final)
    where_matches = _WHERE_RE.findall(final)
    what_matches = _WHAT_RE.findall(final)
    predicted_error = error_matches[-1].casefold() == "yes" if error_matches else None
    where = where_matches[-1].strip() if where_matches else None
    what = what_matches[-1].strip() if what_matches else None
    contradictory_calls = len({value.casefold() for value in error_matches}) > 1
    has_violation = bool(gold.get("has_violation"))
    decision_correct = predicted_error is not None and predicted_error == has_violation

    if predicted_error is True:
        format_valid = structure_ok and bool(where) and bool(what) and not contradictory_calls
    else:
        format_valid = structure_ok and predicted_error is False and not contradictory_calls
        if not _empty_spot_placeholder(where) or not _empty_spot_placeholder(what):
            format_valid = False

    if has_violation and predicted_error:
        location_text = " ".join(part for part in (where, what) if part)
        scenes = list(gold.get("scenes") or [])
        localization_correct: bool | None = bool(scenes) and all(
            _scene_matches(gold["schema"], scene, location_text) for scene in scenes
        )
        type_correct, unsupported_types = _type_diagnostics(gold.get("type"), what or "")
    elif has_violation:
        localization_correct = False
        type_correct = False
        unsupported_types = []
    else:
        localization_correct = None
        type_correct = None
        unsupported_types = []

    correct = format_valid and decision_correct
    strict_correct = correct and (
        not has_violation
        or (bool(localization_correct) and bool(type_correct) and not unsupported_types)
    )
    if not format_valid:
        reasons.append("malformed or contradictory final answer")
    if not decision_correct:
        reasons.append("incorrect clean/error decision")
    if unsupported_types:
        reasons.append("multiple unsupported anomaly types")
    return RewardResult(
        task="spot_error",
        reward=float(correct),
        correct=correct,
        format_valid=format_valid,
        parsed={"error": predicted_error, "where": where, "what": what},
        diagnostics={
            "has_violation": has_violation,
            "decision_correct": decision_correct,
            "localization_correct": localization_correct,
            "type_correct": type_correct,
            "unsupported_types": unsupported_types,
            "strict_correct": strict_correct,
        },
        failure_reasons=tuple(dict.fromkeys(reasons)),
    )


def verify_response(
    task: str,
    gold: dict,
    response: str,
    *,
    truncated: bool = False,
) -> RewardResult:
    if task == "identify":
        return verify_identify(gold, response, truncated=truncated)
    if task == "spot_error":
        return verify_spot_error(gold, response, truncated=truncated)
    raise ValueError(f"unsupported RLVR task: {task!r}")
