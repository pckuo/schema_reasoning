"""Load a per-schema YAML spec and compute its forbidden lexicon (A4.2 / A2)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

SCHEMAS_DIR = Path(__file__).parent / "schemas"


@dataclass
class Role:
    id: str
    kind: str  # person | object | place
    tier: str  # core | periph
    desc: str
    forbidden: list[str] = field(default_factory=list)
    accept: list[str] = field(default_factory=list)  # judge-only: extra correct role words
    # judge-only: forbidden nouns that must NOT be credited as this role's answer. A word stays
    # banned from the prose but leaves the accept vocab -- for bare heads like "agent" that also
    # head another role's accept term ("gate agent"), where crediting them makes the two roles
    # indistinguishable to the judge. See `judge_accept_words`.
    accept_exclude: list[str] = field(default_factory=list)
    max_count: int = 1  # a person role may appear as a small party (e.g. two PATRONs)
    adult_only: bool = False  # exclude child descriptions (e.g. working staff roles)


@dataclass
class Schema:
    name: str
    forbidden_labels: list[str]
    scenes: list[str]
    branches: dict
    roles: dict[str, Role]
    props: dict[str, dict[str, list[str]]]
    adjuncts: dict[str, list[str]]
    actions: dict[str, dict[str, list[str]]]
    # Judge-only glosses (never shown to the probed model): scene_gloss = the overall event at
    # each scene (predict_next_event gold); patron_gloss = what the protagonist should do at each
    # scene they act in (predict_correct_action gold).
    scene_gloss: dict[str, str] = field(default_factory=dict)
    patron_gloss: dict[str, str] = field(default_factory=dict)
    # Sibling scripts at the SAME basic level (judge-only, for schema-granularity grading A3.1):
    # a too-general label ("travel", "service encounter") would fit these too -> superordinate.
    siblings: list[str] = field(default_factory=list)
    # Judge-only accept list: activity-level synonyms of the script (e.g. "dining", "air travel")
    # that count as a MATCH, beyond the forbidden nouns and the schema name.
    schema_accept: list[str] = field(default_factory=list)
    # Object nouns that separate this script from its `siblings`. A passage may name any licensed
    # prop, but only these make the script identifiable: airport `seat` fits a train or bus too,
    # whereas `plane` does not. Generation requires at least one so that a role-latent passage
    # does not also become an unidentifiable one.
    diagnostic_props: list[str] = field(default_factory=list)

    def person_roles(self) -> list[str]:
        return [rid for rid, role in self.roles.items() if role.kind == "person"]

    def object_roles(self) -> list[str]:
        return [rid for rid, role in self.roles.items() if role.kind != "person"]


def load_schema(name: str, schemas_dir: Path = SCHEMAS_DIR) -> Schema:
    data = yaml.safe_load((schemas_dir / f"{name}.yaml").read_text(encoding="utf-8"))
    roles = {
        rid: Role(
            id=rid,
            kind=r["kind"],
            tier=r.get("tier", "periph"),
            desc=r["desc"],
            forbidden=[w.lower() for w in r.get("forbidden", [])],
            accept=[w.lower() for w in r.get("accept", [])],
            accept_exclude=[w.lower() for w in r.get("accept_exclude", [])],
            max_count=r.get("max_count", 1),
            adult_only=r.get("adult_only", False),
        )
        for rid, r in data["roles"].items()
    }
    return Schema(
        name=data["schema"],
        forbidden_labels=[w.lower() for w in data.get("forbidden_labels", [])],
        scenes=list(data["scenes"]),
        branches=data.get("branches") or {},
        roles=roles,
        props=data.get("props", {}),
        adjuncts=data.get("adjuncts", {}),
        actions=data.get("actions", {}),
        scene_gloss=data.get("scene_gloss", {}),
        patron_gloss=data.get("patron_gloss", {}),
        siblings=list(data.get("siblings", [])),
        schema_accept=list(data.get("schema_accept", [])),
        diagnostic_props=list(data.get("diagnostic_props", [])),
    )


def judge_accept_words(role: Role) -> list[str]:
    """Words the judge credits as a correct answer for ``role``, in stable order.

    A role's ``forbidden`` nouns are exactly what a correct answer should supply (the passage was
    banned from saying "waiter", so "waiter" is the right answer), widened by the judge-only
    ``accept`` list and narrowed by ``accept_exclude``. The exclusion is what keeps a bare head
    noun from silently subsuming a sibling role's multiword term.
    """
    drop = {w.lower() for w in role.accept_exclude}
    return [w for w in dict.fromkeys([*role.forbidden, *role.accept]) if w.lower() not in drop]


def named_diagnostic_props(text: str, schema: Schema) -> list[str]:
    """Which of ``schema.diagnostic_props`` the passage actually names.

    The enforceable half of the prose contract. "Never name the script" is checkable by
    :func:`forbidden_set`, but "stay recognisable as the script" was only ever a hope; this makes
    it a gate the generator can fail and retry on.
    """
    low = text.lower()
    return [w for w in schema.diagnostic_props if re.search(rf"\b{re.escape(w.lower())}\b", low)]


def licensed_props(schema: Schema, prop_naming: str | dict[str, str] = "named") -> list[str]:
    """Object/place nouns the narrative is explicitly ALLOWED to use, given ``prop_naming``.

    The exact complement of what :func:`forbidden_set` withholds for non-person roles. Generators
    must state these positively: told only "never name the script", an LLM treats a schema's
    diagnostic props as part of the script and paraphrases them away, which makes the passage
    unidentifiable rather than merely role-latent.
    """
    words: list[str] = []
    for rid, role in schema.roles.items():
        if role.kind == "person":
            continue
        mode = prop_naming.get(rid, "named") if isinstance(prop_naming, dict) else prop_naming
        if mode != "abstract":
            words.extend(role.forbidden)
    return [w for w in dict.fromkeys(words) if w.lower() not in schema.forbidden_labels]


def forbidden_set(
    schema: Schema, prop_naming: str | dict[str, str], role_naming: str = "latent"
) -> set[str]:
    """Lower-cased nouns the generated narrative must never contain.

    Always forbids schema-label nouns. Person-role nouns are forbidden when ``role_naming`` is
    ``"latent"`` (the default, fully-latent design) and ALLOWED when ``"named"`` (the positive
    control -- the passage may say 'waiter'/'cook'/'security officer'); the schema label stays
    forbidden either way, so only roles are surfaced. An object/place-role's nouns are forbidden
    when that role's naming mode is ``"abstract"``. ``prop_naming`` may be a global mode
    (``"named"``/``"abstract"``) or a per-role map ``{role_id: mode}``.
    """
    words: set[str] = set(schema.forbidden_labels)
    for rid, role in schema.roles.items():
        if role.kind == "person":
            if role_naming != "named":
                words.update(role.forbidden)
            continue
        mode = prop_naming.get(rid, "named") if isinstance(prop_naming, dict) else prop_naming
        if mode == "abstract":
            words.update(role.forbidden)
    return words
