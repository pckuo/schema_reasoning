"""The ``identify`` probe for binding stimuli, made multi-role aware.

The question is the production one (``pipeline.tasks._IDENTIFY_Q``, imported rather than copied so
the two corpora stay comparable), with one addition: a line telling the reader a person may hold
more than one part. Without it, 1C measures the model's willingness to repeat a name rather than
its ability to bind -- and the spec is right that the one-name-per-answer prior is a real effect,
so it must be given an explicit escape hatch and then measured, not left to chance.

The gold differs structurally from ``pipeline``: one entry PER PERSON carrying a list of roles,
rather than one entry per (role, referent) pair. ``pipeline/tasks.py:198-203`` emits the latter, so
a merged person yields two gold entries with an identical handle, which the judge roster
(``subjects/judge.py:201``: "one roles entry per roster id") cannot disambiguate.
"""

from __future__ import annotations

from ..core.spec import Schema, judge_accept_words
from ..pipeline.tasks import _IDENTIFY_Q
from .cast import BindingSpec
from .config import ROLE_ALIASES, role_order, tier

MULTIROLE_NOTE = (
    "Note: in this passage a single person may play more than one part, and some people may play "
    "no part at all. If one person plays two parts, name both on their line; do not invent a "
    "second person to fill the second part.\n\n"
)


def identify_question(spec: BindingSpec) -> str:
    """The identify prompt for this item. The multi-role note is added only when the item actually
    contains a multiplexed person, so 1A / 1B prompts stay byte-identical to the production one."""
    return (MULTIROLE_NOTE if spec.merged_people else "") + _IDENTIFY_Q


def build_identify(
    schema: Schema, spec: BindingSpec, passage: str, generator_model: str | None = None
) -> dict:
    """The identify task: question, gold, and a ready-to-use probe prompt.

    Gold ``roles`` is one entry per person. ``accept`` reuses each role's canonical forbidden nouns
    (the passage is forbidden from saying "waiter", so those nouns are exactly what a correct
    answer should supply) widened by the schema's judge-only ``accept`` list.

    The gold is deliberately **self-describing**: it carries the person->role map, the inverse
    role->person map, the per-binding rows and the structural quantities, so ``probes.jsonl`` can
    be scored and sliced without joining back to ``items.jsonl``.
    """
    counts = spec.role_counts
    people = [
        {
            "pid": person.pid,
            "value": person.name,
            "gender": person.gender,
            "role_ids": list(person.roles),
            "roles": [ROLE_ALIASES[spec.schema][rid] for rid in person.roles],
            "n_roles": person.n_roles,
            "accept": list(
                dict.fromkeys(
                    word for rid in person.roles for word in judge_accept_words(schema.roles[rid])
                )
            ),
            # the same words split PER ROLE, in role order: a person holding two roles must be
            # graded part by part, and a flat union would credit naming either one as both
            "accept_per_role": [judge_accept_words(schema.roles[rid]) for rid in person.roles],
            "tiers": [tier(spec.schema, rid) for rid in person.roles],
            "role_multiplicity": [counts[rid] for rid in person.roles],
            "distractor": person.is_distractor,
            "distractor_flavor": person.distractor_flavor,
            "distractor_key": person.distractor_key,
            "licence": person.licence,
        }
        for person in spec.cast
    ]
    role_to_persons = {
        ROLE_ALIASES[spec.schema][rid]: [p.name for p in spec.holders(rid)]
        for rid in role_order(spec.schema)
    }
    question = identify_question(spec)
    return {
        "name": "identify",
        "question": question,
        "answer_format": "open",
        "probe_prompt": f"{passage}\n\n{question}",
        "gold": {
            "schema": spec.schema,
            "roles": people,
            "role_to_persons": role_to_persons,
            "role_to_person_ids": {
                ROLE_ALIASES[spec.schema][rid]: [p.pid for p in spec.holders(rid)]
                for rid in role_order(spec.schema)
            },
            "bindings": spec.bindings(),
            "distractor_names": [p.name for p in spec.cast if p.is_distractor],
            "multiplexed": [
                {"value": p.name, "roles": [ROLE_ALIASES[spec.schema][r] for r in p.roles]}
                for p in spec.merged_people
            ],
            "B": spec.B,
            "R": spec.R,
            "D": spec.D,
            "E": spec.E,
            "vector": list(spec.vector),
            "role_counts": {ROLE_ALIASES[spec.schema][r]: n for r, n in counts.items()},
            "omitted_roles": spec.omitted_roles,
            "max_role_multiplicity": spec.max_role_multiplicity,
        },
        "meta": {
            "item_id": spec.item_id,
            "experiment": spec.experiment,
            "condition": spec.condition,
            "family_id": spec.family_id,
            "parent_id": spec.parent_id,
            "edit_op": spec.edit_op,
            "seed": spec.seed,
            "schema": spec.schema,
            "B": spec.B,
            "R": spec.R,
            "D": spec.D,
            "E": spec.E,
            "vector": list(spec.vector),
            "max_role_multiplicity": spec.max_role_multiplicity,
            "n_multiplexed_people": len(spec.merged_people),
            "omitted_roles": spec.omitted_roles,
            "distractor_flavor": spec.controls.get("distractor_flavor"),
            "open_kitchen": spec.controls.get("open_kitchen"),
            "merge_pair": spec.controls.get("merge_pair"),
            "merge_adjacent": spec.controls.get("merge_adjacent"),
            # constant within a family, round-robined across families -- a between-family nuisance
            # factor, so it must be visible to the analysis
            "generator_model": generator_model,
        },
    }
