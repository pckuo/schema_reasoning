"""Seed-cast construction and the three scripted cast edits (spec section 5, rule 1).

Each experiment moves exactly one quantity, and each does it with one edit operation:

===== ================== ============================================ ==================
 Set   Edit                Moves                                        Holds fixed
===== ================== ============================================ ==================
 1A    insert_bystander    D up, E up                                   vector, B, R
 1B    promote             B up, D down                                 E, R, cast identity
 1C    merge_roles         people-per-binding down, D up                B, R, E
===== ================== ============================================ ==================

All three are **pure cast transforms**: they return a new :class:`BindingSpec` and never touch
prose. Prose follows the cast, not the other way round, so "the same names appear in every sibling"
is a structural guarantee rather than a matter of LLM compliance.
"""

from __future__ import annotations

import random

from .cast import BindingSpec, Person
from .config import OUT_OF_SIGHT, role_order
from .pools import DistractorAllocator, NameAllocator
from .vectors import Vector, vector_to_counts


def build_seed_cast(
    schema: str,
    vector: Vector,
    names: NameAllocator,
    *,
    item_id: str,
    experiment: str,
    condition: str,
    family_id: str,
    seed: int,
    scene_path: list[str],
    controls: dict | None = None,
) -> BindingSpec:
    """A seed spec: one person per binding, nobody multiplexed, no distractors.

    ``vector`` fixes the role counts, so ``B == sum(vector)`` and ``E == B`` here; distractors and
    merges are then applied by the edit operations below.
    """
    counts = vector_to_counts(schema, vector)
    cast: list[Person] = []
    for rid in role_order(schema):
        for _ in range(counts[rid]):
            name, gender = names.take()
            cast.append(Person(pid=f"p{len(cast) + 1}", name=name, gender=gender, roles=[rid]))
    return BindingSpec(
        item_id=item_id,
        schema=schema,
        experiment=experiment,
        condition=condition,
        family_id=family_id,
        parent_id=None,
        edit_op="seed",
        seed=seed,
        cast=cast,
        scene_path=list(scene_path),
        controls=dict(controls or {}),
    )


def insert_bystander(
    parent: BindingSpec,
    names: NameAllocator,
    distractors: DistractorAllocator,
    *,
    item_id: str,
    condition: str,
    n: int = 1,
) -> BindingSpec:
    """1A: add ``n`` licensed, named, role-less participants. B, R and the vector are untouched.

    Each bystander carries the licence + concrete beat that stop them reading as an unexplained
    extra person -- which, per the spec, is the failure mode that would silently bind them to a
    role and move B.
    """
    child = parent.copy(
        item_id=item_id,
        condition=condition,
        parent_id=parent.item_id,
        edit_op="insert_bystander",
    )
    for _ in range(n):
        entry = distractors.take()
        name, gender = names.take()
        child.cast.append(
            Person(
                pid=f"p{len(child.cast) + 1}",
                name=name,
                gender=gender,
                roles=[],
                distractor_flavor=distractors.flavor,
                distractor_key=entry["key"],
                licence=entry["licence"],
                beat=entry["beat"],
                scene=entry["scene"],
            )
        )
    assert child.B == parent.B and child.vector == parent.vector, "insert must not move B"
    assert child.E == parent.E + n, "insert must add exactly n people"
    return child


def promote(
    parent: BindingSpec,
    assignments: list[tuple[str, str]],
    *,
    item_id: str,
    condition: str,
) -> BindingSpec:
    """1B: give already-present distractors a role. Nobody is added or removed.

    ``assignments`` is ``[(pid, role_id), ...]``. Because the cast list is untouched, E is fixed by
    construction and every sibling in the family shares the same names -- which is what makes the
    iso-E comparison a comparison of structure rather than of prose.
    """
    child = parent.copy(
        item_id=item_id,
        condition=condition,
        parent_id=parent.item_id,
        edit_op="promote",
    )
    for pid, role_id in assignments:
        person = child.person(pid)
        if person.roles:
            raise ValueError(f"{pid!r} already holds {person.roles}; promote targets distractors")
        person.roles = [role_id]
        person.distractor_flavor = person.distractor_key = None
        person.licence = person.beat = person.scene = None
    assert child.E == parent.E, "promote must not change headcount"
    assert child.B == parent.B + len(assignments), "promote must add exactly len(assignments)"
    return child


def merge_roles(
    parent: BindingSpec,
    keeper_pid: str,
    donor_pids: list[str],
    distractors: DistractorAllocator,
    names: NameAllocator,
    *,
    item_id: str,
    condition: str,
    ladder: list[str],
    backfill: bool = True,
) -> BindingSpec:
    """1C: move the donors' roles onto ``keeper_pid``, then pad headcount back with bystanders.

    B, R and E are all held fixed; only the person->role mapping changes. The donors leave the cast
    entirely (they are not kept as role-less people, or the manipulation would be "who acts"
    rather than "how many people carry the bindings"), and one licensed bystander is added per
    donor so E never moves -- which is what makes the contrast airtight.

    That backfill is exactly what makes D rise with multiplexing, because E = (B - M) + D and 1C
    pins B and E: with two of the three fixed, D has to absorb every merge. ``backfill=False``
    (used by 1D) drops the padding and pins B and D instead, letting E fall as M rises. Only two
    of {B, D, E} can ever be held at once, so the two experiments are complements: 1C reads D-free
    at constant headcount, 1D reads headcount-free at constant distractor count, and a decrement
    in both can be attributed to multiplexing itself rather than to either covariate.

    ``ladder`` is the schema's mergeable staff ladder (:func:`config.merge_ladder`). Every role
    involved must be on it, which is what enforces "never merge the out-of-sight role" -- a waiter
    who is also the pilot is incoherent -- and also keeps PATRON out (many patrons is a
    multiplicity manipulation, already covered by 1B).
    """
    keeper = parent.person(keeper_pid)
    moving = [r for pid in donor_pids for r in parent.person(pid).roles]
    for rid in [*keeper.roles, *moving]:
        if rid not in ladder:
            reason = (
                "is out of sight and must never be merged"
                if rid in OUT_OF_SIGHT[parent.schema]
                else "is not on the mergeable staff ladder"
            )
            raise ValueError(f"{rid!r} {reason} in {parent.schema} (ladder={ladder})")

    child = parent.copy(
        item_id=item_id,
        condition=condition,
        parent_id=parent.item_id,
        edit_op="merge_roles",
    )
    order = role_order(parent.schema)
    new_keeper = child.person(keeper_pid)
    new_keeper.roles = sorted({*new_keeper.roles, *moving}, key=order.index)
    child.cast = [p for p in child.cast if p.pid not in set(donor_pids)]
    # Recorded, not inferred: an unpadded merge that wrongly gained a bystander is structurally
    # identical to a legal padded one, so verify cannot tell the arms apart from the cast delta.
    child.controls["merge_backfill"] = backfill
    if backfill:
        for _ in donor_pids:
            entry = distractors.take()
            name, gender = names.take()
            child.cast.append(
                Person(
                    pid=f"d{len(child.cast) + 1}",
                    name=name,
                    gender=gender,
                    roles=[],
                    distractor_flavor=distractors.flavor,
                    distractor_key=entry["key"],
                    licence=entry["licence"],
                    beat=entry["beat"],
                    scene=entry["scene"],
                )
            )
    assert child.B == parent.B, "merge must not change the number of bindings"
    assert child.R == parent.R, "merge must not change the number of occupied roles"
    if backfill:
        assert child.E == parent.E, "merge must not change headcount"
        assert child.D == parent.D + len(donor_pids), "padded merge must add one D per donor"
    else:
        assert child.D == parent.D, "unpadded merge must never change the distractor count"
        assert child.E == parent.E - len(donor_pids), "unpadded merge must drop one E per donor"
    return child


def choose_promotions(
    parent: BindingSpec, target_counts: dict[str, int], rng: random.Random
) -> list[tuple[str, str]]:
    """Which distractors to promote, and to which roles, to reach ``target_counts``.

    Picks distractors in cast order (stable) and fills role deficits in canonical role order, so a
    family's promote ladder is deterministic given its seed.
    """
    current = parent.role_counts
    deficits = [
        (rid, target_counts.get(rid, 0) - current.get(rid, 0)) for rid in role_order(parent.schema)
    ]
    wanted = [rid for rid, gap in deficits for _ in range(max(0, gap))]
    free = [p.pid for p in parent.cast if p.is_distractor]
    if len(free) < len(wanted):
        raise ValueError(
            f"need {len(wanted)} promotions but only {len(free)} distractors in {parent.item_id!r}"
        )
    rng.shuffle(free)
    return list(zip(free[: len(wanted)], wanted, strict=True))
