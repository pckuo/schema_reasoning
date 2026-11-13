"""Build the 1A / 1B / 1C condition grids as seed FAMILIES.

The unit of generation is a family, not a condition cell, because the spec's section 5 rule 1
("one seed, minimal edits; never regenerate from scratch") is the only thing that makes these
comparisons about structure rather than prose style.

1B's family structure is the load-bearing one. Because ``promote`` adds and removes nobody, a
family is exactly an **iso-E stratum**: one cast of 8 people yields (B=8,D=0), (B=6,D=2) and
(B=4,D=4) as three siblings sharing every name. The spec's "priority cells" are therefore the
natural output of the generator, not a special case needing extra care.

Totals reproduce the spec: 1A 240, 1B 516, 1C 240 = 996, split 498 restaurant / 498 airport.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..core.spec import load_schema
from .cast import BindingSpec
from .config import ROLE_CAPS, merge_ladder, role_order
from .edits import build_seed_cast, choose_promotions, insert_bystander, merge_roles, promote
from .pools import DistractorAllocator, NameAllocator
from .vectors import Vector, counts_to_vector, enumerate_vectors, omitted_role, pick_contrasting

D_LEVELS_1A = (0, 1, 2, 3, 4, 5)
B_LEVELS_1B = (4, 5, 6, 7, 8, 9, 10)
D_LEVELS_1B = (0, 2, 4)
PRIORITY_E = (8, 10)  # the two iso-E strata read as direct pairwise comparisons
N_1A, N_1B, N_1B_PRIORITY, N_1C = 10, 10, 18, 20
N_1D, PATRONS_1D = 10, 2

# 1A arms, in condition-id order: (schema, B, R, n_vectors, omission_balance)
ARMS_1A = (
    ("restaurant", 4, 4, 1, False),
    ("restaurant", 7, 4, 4, False),
    ("airport", 4, 4, 4, True),
    ("airport", 7, 4, 4, True),
)
# 1C arms: (schema, n_patrons). Restaurant R=4 / airport R=5 throughout; staff ladder is 3 roles.
ARMS_1C = (("restaurant", 1), ("restaurant", 2), ("airport", 1), ("airport", 2))
# 1D arms: (schema, D). Patrons are pinned at 2 -- with 1, an unpadded restaurant triple bottoms
# out at a 2-person cast, which stops being a story and starts being a riddle.
ARMS_1D = (("restaurant", 0), ("restaurant", 3), ("airport", 0), ("airport", 3))


@dataclass
class Family:
    """One seed story plus the siblings derived from it by single scripted cast edits."""

    family_id: str
    experiment: str
    schema: str
    items: list[BindingSpec] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


PROP_NAMING_LEVELS = ("named", "abstract")


def assign_naming(families: list[Family]) -> None:
    """Fix both naming factors on every family, in place.

    **prop_naming** is balanced. Whether the passage may use its script's diagnostic object nouns
    decides whether the subject can tell what the situation *is*, which is upstream of -- and
    separable from -- how hard the role bindings are. v1 fixed this to ``named`` for every item and
    never said so in the prompt, so the generator quietly reverted to paraphrase and schema
    recognition became an uncontrolled property of the passage. Making it a declared factor is what
    lets a later analysis condition on schema recognition instead of confounding it with binding
    difficulty.

    Round-robin **within each (experiment, schema) arm**, for the same reason ``assign_models``
    does: the assignment must be orthogonal to condition, and it must not move when a run is
    scoped with ``--experiment`` / ``--schema`` / ``--limit``. A family is the unit because
    siblings are copy-and-patch edits of their parent's accepted text -- switching prop vocabulary
    mid-family would be a rewrite, not an edit, and would fail preservation.

    **role_naming** is NOT balanced and is not crossed with anything. Naming the role in the
    passage does not make binding harder or easier -- it removes the task, turning ``identify``
    into extraction -- so crossing it with the full grid would buy ~1000 ceiling items with no
    binding signal. Its one job was to separate inference failure from judge unfairness.

    No builder currently sets it: the calibration arm that would have (1E) was cut as a diagnostic
    rather than a condition, so every item in the corpus ships ``latent`` and the column is a
    logged constant. The mechanism is kept, and stays under test, because re-enabling it is a
    one-line ``family.meta["role_naming"] = "named"`` in whichever builder wants it; the prose
    contract and the positive gate in :mod:`verify` are already wired for it.

    Role-named families are pinned to ``prop_naming="named"``. The abstract cross is incoherent:
    forbidding "menu" while allowing "waiter" hands over the script anyway, so the item is neither
    a clean disguise nor a clean control.

    Both arms are mirrored onto every item's ``controls``. ``judge._CONTROL_KEYS`` already names
    them, and it reads the item's top-level ``controls``, so this is what carries the factors all
    the way into the score rows -- without it every ``prop_naming`` column would come out null and
    no analysis could condition on the very thing this control exists to expose.
    """
    counters: dict[tuple[str, str], int] = {}
    for family in families:
        family.meta.setdefault("role_naming", "latent")
        if family.meta["role_naming"] == "named":
            family.meta["prop_naming"] = "named"
        else:
            key = (family.experiment, family.schema)
            i = counters.get(key, 0)
            counters[key] = i + 1
            family.meta["prop_naming"] = PROP_NAMING_LEVELS[i % len(PROP_NAMING_LEVELS)]
        for spec in family.items:
            spec.controls["prop_naming"] = family.meta["prop_naming"]
            spec.controls["role_naming"] = family.meta["role_naming"]


def _scenes(schema: str) -> list[str]:
    return list(load_schema(schema).scenes)


def _role_scenes(schema: str, role_ids: list[str]) -> frozenset[str]:
    """The beats those roles act at -- used to keep bystanders away from an omitted role's beat."""
    actions = load_schema(schema).actions
    return frozenset(s for rid in role_ids for s in actions.get(rid, {}))


def _n_items(experiment: str, e_level: int | None) -> int:
    if experiment == "1A":
        return N_1A
    if experiment == "1C":
        return N_1C
    return N_1B_PRIORITY if e_level in PRIORITY_E else N_1B


# --------------------------------------------------------------------------- 1A


def _conditions_1a() -> dict[tuple[str, int], list[str]]:
    """(schema, B) -> the six condition ids for D = 0..5, in spec order A01..A24."""
    out: dict[tuple[str, int], list[str]] = {}
    for arm, (schema, b, _r, _nv, _bal) in enumerate(ARMS_1A):
        out[(schema, b)] = [f"A{arm * 6 + d + 1:02d}" for d in D_LEVELS_1A]
    return out


def build_1a(rng: random.Random, flavor: str = "outside_inventory") -> list[Family]:
    """Entity load: hold the vector / B / R fixed, walk D from 0 to 5 by inserting bystanders."""
    conditions = _conditions_1a()
    families: list[Family] = []
    for schema, b, r, n_vec, balance in ARMS_1A:
        pool = enumerate_vectors(schema, b, r)
        vectors = pick_contrasting(pool, n_vec, rng, omission_balance=balance)
        scenes = _scenes(schema)
        cond_ids = conditions[(schema, b)]
        for i in range(N_1A):
            vector = vectors[i % len(vectors)]
            seed = rng.randrange(2**31)
            fid = f"1A-{schema}-B{b}-{i:02d}"
            names = NameAllocator(rng=random.Random(seed))
            omitted = omitted_role(schema, vector)
            distractors = DistractorAllocator(
                schema,
                flavor,
                random.Random(seed + 1),
                avoid_scenes=_role_scenes(schema, [omitted] if omitted else []),
            )
            controls = {
                "distractor_flavor": flavor,
                "omitted_role": omitted,
                "open_kitchen": False,
            }
            item = build_seed_cast(
                schema,
                vector,
                names,
                item_id=f"{fid}-D0",
                experiment="1A",
                condition=cond_ids[0],
                family_id=fid,
                seed=seed,
                scene_path=scenes,
                controls=controls,
            )
            family = Family(fid, "1A", schema, [item], {"vector": list(vector), "B": b, "R": r})
            for d in D_LEVELS_1A[1:]:
                item = insert_bystander(
                    item,
                    names,
                    distractors,
                    item_id=f"{fid}-D{d}",
                    condition=cond_ids[d],
                    n=1,
                )
                family.items.append(item)
            families.append(family)
    return families


# --------------------------------------------------------------------------- 1B


def _conditions_1b() -> dict[tuple[str, int, int], str]:
    """(schema, B, D) -> condition id, in spec order B01..B42."""
    out: dict[tuple[str, int, int], str] = {}
    n = 0
    for schema in ("restaurant", "airport"):
        for b in B_LEVELS_1B:
            for d in D_LEVELS_1B:
                n += 1
                out[(schema, b, d)] = f"B{n:02d}"
    return out


def _nested_chain(schema: str, b_levels: list[int], rng: random.Random) -> list[Vector]:
    """Vectors for an ascending ``b_levels`` chain such that each is contained in the next.

    Nesting is required: ``promote`` only ever ADDS a role to someone, so a sibling's vector must
    be reachable from its parent's by increments. Picks a contrasting shape at the top B, then
    walks down by decrementing the currently-largest role (never below 1, never un-omitting a role
    -- R must stay 4).
    """
    top = pick_contrasting(enumerate_vectors(schema, b_levels[-1], 4), 4, rng)
    if not top:
        raise ValueError(f"no vector for {schema} B={b_levels[-1]} R=4")
    counts = dict(zip(role_order(schema), rng.choice(top), strict=True))
    chain = {b_levels[-1]: counts_to_vector(schema, counts)}
    for b in reversed(b_levels[:-1]):
        while sum(counts.values()) > b:
            # decrement the most-occupied role that can still lose a person (floor of 1 keeps R)
            rid = max(
                (r for r in role_order(schema) if counts[r] > 1),
                key=lambda r: (counts[r], -role_order(schema).index(r)),
            )
            counts[rid] -= 1
        chain[b] = counts_to_vector(schema, counts)
    return [chain[b] for b in b_levels]


def build_1b(rng: random.Random, flavor: str = "outside_inventory") -> list[Family]:
    """Binding load: hold E and R fixed, walk B by promoting already-present people."""
    conditions = _conditions_1b()
    by_e: dict[tuple[str, int], list[tuple[int, int]]] = {}
    for schema in ("restaurant", "airport"):
        for b in B_LEVELS_1B:
            for d in D_LEVELS_1B:
                by_e.setdefault((schema, b + d), []).append((b, d))
    families: list[Family] = []
    for (schema, e_level), cells in sorted(by_e.items()):
        cells.sort()  # ascending B, so the seed is the lowest-B sibling
        scenes = _scenes(schema)
        b_levels = [b for b, _ in cells]
        for i in range(_n_items("1B", e_level)):
            seed = rng.randrange(2**31)
            fid = f"1B-{schema}-E{e_level:02d}-{i:02d}"
            names = NameAllocator(rng=random.Random(seed))
            chain = _nested_chain(schema, b_levels, random.Random(seed + 2))
            b0, d0 = cells[0]
            omitted = omitted_role(schema, chain[0])
            distractors = DistractorAllocator(
                schema,
                flavor,
                random.Random(seed + 1),
                avoid_scenes=_role_scenes(schema, [omitted] if omitted else []),
            )
            controls = {
                "distractor_flavor": flavor,
                "omitted_role": omitted,
                "open_kitchen": False,
                "E": e_level,
            }
            item = build_seed_cast(
                schema,
                chain[0],
                names,
                item_id=f"{fid}-B{b0:02d}",
                experiment="1B",
                condition=conditions[(schema, b0, d0)],
                family_id=fid,
                seed=seed,
                scene_path=scenes,
                controls=controls,
            )
            if d0:  # pad the seed up to the family's fixed headcount
                item = insert_bystander(
                    item,
                    names,
                    distractors,
                    item_id=item.item_id,
                    condition=item.condition,
                    n=d0,
                )
                item.edit_op = "seed"
                item.parent_id = None
            family = Family(fid, "1B", schema, [item], {"E": e_level, "R": 4})
            for (b, d), vector in list(zip(cells, chain, strict=True))[1:]:
                targets = dict(zip(role_order(schema), vector, strict=True))
                moves = choose_promotions(item, targets, random.Random(seed + 3 + b))
                item = promote(
                    item,
                    moves,
                    item_id=f"{fid}-B{b:02d}",
                    condition=conditions[(schema, b, d)],
                )
                family.items.append(item)
            families.append(family)
    return families


# --------------------------------------------------------------------------- 1C


def _merge_pairs(schema: str, open_kitchen: bool) -> list[tuple[str, str]]:
    ladder = merge_ladder(schema, open_kitchen)
    return [(a, b) for i, a in enumerate(ladder) for b in ladder[i + 1 :]]


def _adjacent_in_script(schema: str, roles: list[str]) -> bool:
    """Whether the merged roles act at adjacent script beats -- the spec asks for this to be
    logged, since an adjacent merge ("greeted them, then took their order") may be easier."""
    sch = load_schema(schema)
    idx = {s: i for i, s in enumerate(sch.scenes)}
    positions = [sorted(idx[s] for s in sch.actions.get(rid, {}) if s in idx) for rid in roles]
    if any(not p for p in positions):
        return False
    return any(
        abs(a - b) == 1
        for i, pa in enumerate(positions)
        for pb in positions[i + 1 :]
        for a in pa
        for b in pb
    )


def build_1c(rng: random.Random, open_kitchen: bool = True) -> list[Family]:
    """Filler reuse: hold B, R and E fixed; change only how many people carry the bindings.

    ``open_kitchen`` reframes the restaurant cook as visible so both schemas have a three-role
    staff ladder. It is applied to the WHOLE restaurant arm, including the unmerged P3 baseline --
    the framing must never vary *within* a triple, which is the one thing that would break the
    contrast (see the plan, section 8).
    """
    families: list[Family] = []
    flavor = "outside_inventory"
    for arm, (schema, n_patrons) in enumerate(ARMS_1C):
        ladder = merge_ladder(schema, open_kitchen)
        caps = dict(ROLE_CAPS[schema])
        counts = {rid: (n_patrons if rid == "PATRON" else 1) for rid in role_order(schema)}
        for rid in role_order(schema):
            if rid not in ladder and rid != "PATRON":
                counts[rid] = 1 if caps[rid] else 0
        vector = counts_to_vector(schema, counts)
        scenes = _scenes(schema)
        pairs = _merge_pairs(schema, open_kitchen)
        cond_ids = [f"C{arm * 3 + k + 1:02d}" for k in range(3)]
        for i in range(N_1C):
            seed = rng.randrange(2**31)
            fid = f"1C-{schema}-P{n_patrons}-{i:02d}"
            names = NameAllocator(rng=random.Random(seed))
            distractors = DistractorAllocator(schema, flavor, random.Random(seed + 1))
            pair = pairs[i % len(pairs)]  # counterbalance which pair merges
            controls = {
                "distractor_flavor": flavor,
                "omitted_role": None,
                "open_kitchen": open_kitchen and schema == "restaurant",
                "merge_pair": list(pair),
                "merge_adjacent": _adjacent_in_script(schema, list(pair)),
            }
            p3 = build_seed_cast(
                schema,
                vector,
                names,
                item_id=f"{fid}-P3",
                experiment="1C",
                condition=cond_ids[0],
                family_id=fid,
                seed=seed,
                scene_path=scenes,
                controls=controls,
            )
            keeper = p3.holders(pair[0])[0].pid
            donor = p3.holders(pair[1])[0].pid
            p2 = merge_roles(
                p3,
                keeper,
                [donor],
                distractors,
                names,
                item_id=f"{fid}-P2",
                condition=cond_ids[1],
                ladder=ladder,
            )
            third = next(r for r in ladder if r not in pair)
            p1 = merge_roles(
                p2,
                keeper,
                [p2.holders(third)[0].pid],
                distractors,
                names,
                item_id=f"{fid}-P1",
                condition=cond_ids[2],
                ladder=ladder,
            )
            p1.controls = {
                **controls,
                "merge_all": list(ladder),
                "merge_adjacent": _adjacent_in_script(schema, list(ladder)),
            }
            families.append(
                Family(
                    fid, "1C", schema, [p3, p2, p1], {"vector": list(vector), "patrons": n_patrons}
                )
            )
    return families


# --------------------------------------------------------------------------- 1D


def build_1d(rng: random.Random, open_kitchen: bool = True) -> list[Family]:
    """Multiplexing at a *fixed* distractor count: hold B and D; let E fall as M rises.

    1C cannot answer this. E = (B - M) + D is an identity, so pinning B and E forces D to absorb
    every merge -- in 1C the distractor count rises in lockstep with roles-per-person, and no
    post-hoc subselection can undo it because the two never vary independently. This arm pins the
    other pair: D is fixed within a family (at 0 or 3) and E is allowed to fall.

    Neither arm alone is clean; together they bracket the effect. A decrement that shows up in both
    can be neither D (constant here, rising there) nor E (constant there, falling here). And 1D's
    residual confound runs *against* the hypothesis: E falls as M rises, and 1B says smaller casts
    are easier, so any decrement observed here is a lower bound on the true multiplexing cost.
    """
    families: list[Family] = []
    flavor = "outside_inventory"
    for arm, (schema, n_distractors) in enumerate(ARMS_1D):
        ladder = merge_ladder(schema, open_kitchen)
        caps = dict(ROLE_CAPS[schema])
        counts = {rid: (PATRONS_1D if rid == "PATRON" else 1) for rid in role_order(schema)}
        for rid in role_order(schema):
            if rid not in ladder and rid != "PATRON":
                counts[rid] = 1 if caps[rid] else 0
        vector = counts_to_vector(schema, counts)
        scenes = _scenes(schema)
        pairs = _merge_pairs(schema, open_kitchen)
        cond_ids = [f"D{arm * 3 + k + 1:02d}" for k in range(3)]
        for i in range(N_1D):
            seed = rng.randrange(2**31)
            fid = f"1D-{schema}-D{n_distractors}-{i:02d}"
            names = NameAllocator(rng=random.Random(seed))
            distractors = DistractorAllocator(schema, flavor, random.Random(seed + 1))
            pair = pairs[i % len(pairs)]  # counterbalance which pair merges, as in 1C
            controls = {
                "distractor_flavor": flavor,
                "omitted_role": None,
                "open_kitchen": open_kitchen and schema == "restaurant",
                "merge_pair": list(pair),
                "merge_adjacent": _adjacent_in_script(schema, list(pair)),
                "fixed_distractors": n_distractors,
            }
            base = build_seed_cast(
                schema,
                vector,
                names,
                item_id=f"{fid}-M0",
                experiment="1D",
                condition=cond_ids[0],
                family_id=fid,
                seed=seed,
                scene_path=scenes,
                controls=controls,
            )
            if n_distractors:
                # The distractors belong to the baseline, not to a later rung: D must already be
                # at its final value before the first merge, or it would vary within the family.
                base = insert_bystander(
                    base,
                    names,
                    distractors,
                    item_id=f"{fid}-M0",
                    condition=cond_ids[0],
                    n=n_distractors,
                )
                base.parent_id = None
            keeper = base.holders(pair[0])[0].pid
            donor = base.holders(pair[1])[0].pid
            m1 = merge_roles(
                base,
                keeper,
                [donor],
                distractors,
                names,
                item_id=f"{fid}-M1",
                condition=cond_ids[1],
                ladder=ladder,
                backfill=False,
            )
            third = next(r for r in ladder if r not in pair)
            m2 = merge_roles(
                m1,
                keeper,
                [m1.holders(third)[0].pid],
                distractors,
                names,
                item_id=f"{fid}-M2",
                condition=cond_ids[2],
                ladder=ladder,
                backfill=False,
            )
            # Update rather than replace: rebuilding from the family-level `controls` would
            # discard whatever the edit recorded on the child (e.g. `merge_backfill`).
            m2.controls = {
                **m2.controls,
                "merge_all": list(ladder),
                "merge_adjacent": _adjacent_in_script(schema, list(ladder)),
            }
            families.append(
                Family(
                    fid,
                    "1D",
                    schema,
                    [base, m1, m2],
                    {"vector": list(vector), "patrons": PATRONS_1D, "D": n_distractors},
                )
            )
    return families


def build_all(seed: int = 20260826, open_kitchen: bool = True) -> list[Family]:
    rng = random.Random(seed)
    return [
        *build_1a(rng),
        *build_1b(rng),
        *build_1c(rng, open_kitchen),
        *build_1d(rng, open_kitchen),
    ]
