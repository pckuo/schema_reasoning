"""Static configuration for the binding experiments: role caps, tiers, merge ladders, TRAPI model.

Deliberately NOT in ``core/schemas/*.yaml``. ``Role.max_count`` there is read by
``pipeline.sampler`` to drive ``rng.randint(1, max_count)``, so raising it would silently change
the sampling distribution of every existing corpus on regeneration. The binding experiments carry
their own caps and leave the shared schema spec frozen.
"""

from __future__ import annotations

from pathlib import Path

POOLS_DIR = Path(__file__).parent / "pools"

# All LLM prosing and parsing for these experiments goes through TRAPI.
TRAPI_MODEL = "gpt-5.6-sol_2026-07-09"

# Several deployments can be driven at once: each carries its own quota, so a pool multiplies
# throughput on the shared `redmond/interactive` tier rather than sharing one budget.
#
# A model is assigned PER FAMILY and is constant across that family's siblings. That constraint is
# not negotiable: siblings are edited from their parent's accepted text, so switching model inside
# a family would reintroduce exactly the prose-style confound the seed-family design exists to
# remove (spec section 5, rule 1). Across families the pool is round-robined, so generator model is
# balanced over conditions and enters analysis as a between-family nuisance factor, absorbed by the
# `(1|family)` random effect the spec already plans for.
#
# `pipeline` sets the same precedent for a generator pool (pilot.py:167: "sample per call from
# `models` so stimuli aren't locked to one model's house style").
TRAPI_MODEL_POOL = (
    "gpt-5.6-sol_2026-07-09",
    "gpt-5.6-luna_2026-07-09",
    "gpt-5.6-terra_2026-07-09",
    "gpt-5.5_2026-04-24",
)

SCHEMAS = ("restaurant", "airport")

# Spec role names -> our opaque schema-YAML role ids. The YAML ids are deliberately generic
# (FRONT_AGENT is the host in a restaurant and the check-in agent at an airport), so this table is
# the contract between the spec's prose and the code.
ROLE_ALIASES: dict[str, dict[str, str]] = {
    "restaurant": {
        "PATRON": "patron",
        "FRONT_AGENT": "host",
        "SERVICE_AGENT": "waiter",
        "PRODUCER": "cook",
    },
    "airport": {
        "PATRON": "patron",
        "FRONT_AGENT": "check-in",
        "GATEKEEPER": "security",
        "SERVICE_AGENT": "gate",
        "PRODUCER": "pilot",
    },
}

# Max people per role (spec section "Max per-role counts"). Order is significant: it defines the
# canonical role-count VECTOR order used everywhere (logging, design, analysis).
ROLE_CAPS: dict[str, list[tuple[str, int]]] = {
    "restaurant": [("PATRON", 4), ("FRONT_AGENT", 3), ("SERVICE_AGENT", 3), ("PRODUCER", 3)],
    "airport": [
        ("PATRON", 4),
        ("FRONT_AGENT", 3),
        ("GATEKEEPER", 3),
        ("SERVICE_AGENT", 2),
        ("PRODUCER", 2),
    ],
}

# Occupancy rules. restaurant: every role >= 1, so R = 4 always. airport: patron >= 1 and at most
# ONE of the four staff roles may be empty, so R = 5, or R = 4 with one role omitted.
MAX_OMITTED: dict[str, int] = {"restaurant": 0, "airport": 1}

# Roles that are narrated out of sight, so they can never be merged onto an on-scene person (a
# waiter who is also the pilot is incoherent) and are logged with tier="out_of_sight".
OUT_OF_SIGHT: dict[str, frozenset[str]] = {
    "restaurant": frozenset({"PRODUCER"}),
    "airport": frozenset({"PRODUCER"}),
}

# The 1C staff ladder: roles that may be multiplexed onto one person. PATRON is excluded (many
# patrons is a multiplicity manipulation, already covered by 1B) and the out-of-sight role is
# excluded unless the schema is reframed.
#
# ``open_kitchen`` reframes the restaurant PRODUCER as visible so the two schemas' ladders match at
# three roles. It is applied to the WHOLE 1C restaurant arm (including the unmerged P3 baseline),
# never to a subset -- otherwise the framing would vary within a triple, which is the one thing
# that must not move. See the plan, section 8.
MERGE_LADDER: dict[str, list[str]] = {
    "restaurant": ["FRONT_AGENT", "SERVICE_AGENT", "PRODUCER"],
    "airport": ["FRONT_AGENT", "GATEKEEPER", "SERVICE_AGENT"],
}
MERGE_LADDER_CLOSED_KITCHEN: dict[str, list[str]] = {
    "restaurant": ["FRONT_AGENT", "SERVICE_AGENT"],
    "airport": ["FRONT_AGENT", "GATEKEEPER", "SERVICE_AGENT"],
}
# Replaces PRODUCER.desc for restaurant 1C items when open_kitchen is on.
OPEN_KITCHEN_DESC = (
    "works at the open counter in full view of the room, preparing what the visitor will eat"
)

DISTRACTOR_FLAVORS = ("outside_inventory", "inside_unqueried")

# Word-count targets. Length is confounded with headcount (more people = more text), so the target
# scales with E and every sibling in a family is generated against the SAME target -- the family's
# largest -- with the shorter ones padded by role-irrelevant description (spec rule 3).
WORDS_PER_PERSON = 26
WORDS_BASE = 90


def length_target(headcount: int) -> tuple[int, int]:
    """(low, high) word band for a cast of ``headcount``. Used as the family-wide target so
    siblings are length-matched rather than each drifting to its own natural length."""
    mid = WORDS_BASE + WORDS_PER_PERSON * headcount
    return int(mid * 0.8), int(mid * 1.2)


def role_order(schema: str) -> list[str]:
    """Canonical role order for ``schema`` -- the role-count vector's index order."""
    return [rid for rid, _ in ROLE_CAPS[schema]]


def role_cap(schema: str, role_id: str) -> int:
    return dict(ROLE_CAPS[schema])[role_id]


def tier(schema: str, role_id: str) -> str:
    """``"out_of_sight"`` | ``"on_scene"`` -- logged per binding (spec section 6)."""
    return "out_of_sight" if role_id in OUT_OF_SIGHT[schema] else "on_scene"


def merge_ladder(schema: str, open_kitchen: bool = True) -> list[str]:
    table = MERGE_LADDER if open_kitchen else MERGE_LADDER_CLOSED_KITCHEN
    return list(table[schema])
