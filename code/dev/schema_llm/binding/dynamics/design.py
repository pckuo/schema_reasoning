"""D1 / D2 family grids for the binding-dynamics starter batch (`binding_dyn_v1`).

Implements [`main/plans/20260909-BINDING_DYNAMICS_STORY_DESIGNS.md`] section 8. The unit of
generation is again a **family**: one cast, one seed passage, and siblings that differ only in
*when* information about the focal person arrives.

Two things distinguish this grid from `binding/design.py`:

* the manipulated variable is **narration order**, not cast structure, so every sibling in a family
  shares an identical cast and differs only in its ``schedule``;
* every role is filled **exactly once**, so ``max_role_multiplicity == 1`` corpus-wide. That is not
  tidiness -- role sharing leaks the focal's role into its own pre-onset window (a second unbound
  person is likelier to hold a role someone else already holds), which would break the design's
  central control.

The dynamics fields live in ``BindingSpec.controls`` rather than as new dataclass fields, so
``cast.py`` stays byte-identical and every existing corpus under ``results/schema_llm/data/``
remains reproducible.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..cast import BindingSpec, Person
from ..config import role_order
from ..design import Family
from ..edits import build_seed_cast, insert_bystander
from ..pools import GENDERS, DistractorAllocator, NameAllocator, neutral_beats
from ..vectors import counts_to_vector
from .balance import BalancedAssigner

# ----------------------------------------------------------------------------- design constants

N_SEGMENTS = 6
SEG_WORDS = (55, 75)
SCHEMA_ONSET = 2  # the situation is disclosed here in every D1 arm; only the role onset moves

# D1a: one staged person, role onset at segment 2 / 3 / 4 -> 1 / 2 / 3 pre-onset mentions and
# 4 / 3 / 2 post-onset ones. Never later than 4: the bound state has to be readable afterwards.
D1A_ARMS = (("onset2", 2), ("onset3", 3), ("onset4", 4))
# D1b: two staged people. The first is pinned at segment 2, the second moves -- so the arm is the
# GAP between an established binding and a new one. The latest gap is 2 rather than 3 because every
# onset must leave >= 2 later segments to read the bound state from (a causal model cannot write the
# binding back to an earlier position), and `gap0` -- both people disclosed together -- is the
# simultaneous baseline the gap is measured against.
D1B_FIRST_ONSET = 2
D1B_ARMS = (("gap0", 2), ("gap1", 3), ("gap2", 4))

# D2: segments 2-3 are the ambiguous window (two readout positions in the ambiguous state, so a
# stable mixture is distinguishable from a transient one); segment 4 resolves.
D2_AMBIGUOUS = (2, 3)
D2_RESOLVE = 4
D2_ARMS = ("resolve_a", "resolve_b", "unresolved")

# Balanced 50/50 across families within every (design x schema) cell, exactly as v2 balances it.
#
# `abstract` is NOT merely a cosmetic variant here -- it changes what the design measures, and in a
# useful way. Under `named` the situation is disclosed at segment 2 and the role later, so the
# schema always precedes the role. Under `abstract` the script's objects are written around, so
# there IS no schema onset: the situation stays unnamed until the role disclosure gives it away,
# because a role is defined by its script (plan 7.6). The two levels are therefore the *order*
# contrast that the failed reverse-order arm could not deliver -- schema-early versus schema-late --
# obtained legitimately, by withholding the situation rather than by trying to hide it after the
# fact.
#
# They must NOT be pooled (v2 section 5.3: subject models recognise the airport schema in 6.5% of
# abstract items). `prop_naming` is a declared factor and every analysis conditions on it.
PROP_NAMING_LEVELS = ("named", "abstract")

FAMILIES_PER_SCHEMA = 25
D1A_FAMILIES = 15
D1B_FAMILIES = 10
# D >= 1 everywhere in D1: at D = 0 the cast is exactly one person per role, so once the unstaged
# role-bearers are disclosed the staged person's part is fixed by elimination. A licensed bystander
# gives an unbound person an alternative explanation. D1b caps at 2 because it also stages two
# people, and E = B + D has to stay inside the S = 2 prose ceiling (plan 7.2).
D_LEVELS_D1A = (1, 2, 3)
D_LEVELS_D1B = (1, 2)
D_LEVELS_D2 = (0, 1)

# The focal's role. PATRON is deliberately EXCLUDED here: an unbound person is read as the patron
# by default (plan 7.5), so a patron-staged item's pre-onset answer would already BE the target
# role and the latency manipulation would have nothing to move. The prior-congruence question is
# measured instead in D2, where two of the three pairs are patron-vs-staff and the model's
# commitment direction at the ambiguous cut is the measurement.
STAGED_ROLES: dict[str, tuple[str, ...]] = {
    "restaurant": ("FRONT_AGENT", "SERVICE_AGENT", "PRODUCER"),
    "airport": ("FRONT_AGENT", "GATEKEEPER", "SERVICE_AGENT", "PRODUCER"),
}

# Candidate pairs for D2. Each must be structurally OPPOSED (served vs serving, or two roles whose
# accept vocabularies are disjoint), so the resolution can be decisive. Two of the three per schema
# are patron-vs-staff -- maximally decisive -- and one is staff-vs-staff, which is the hard case.
D2_PAIRS: dict[str, tuple[tuple[str, str], ...]] = {
    "restaurant": (
        ("PATRON", "SERVICE_AGENT"),
        ("PATRON", "FRONT_AGENT"),
        ("SERVICE_AGENT", "PRODUCER"),
    ),
    "airport": (
        ("PATRON", "FRONT_AGENT"),
        ("PATRON", "GATEKEEPER"),
        ("FRONT_AGENT", "SERVICE_AGENT"),
    ),
}


@dataclass
class Schedule:
    """When each kind of information arrives, in segment indices (1-based).

    ``role_onsets`` maps pid -> segment for the STAGED people only. Everyone else is disclosed
    wherever the prose naturally puts them; their realized onset is measured by the per-cut audit
    rather than scheduled, which is what turns them into a measured covariate instead of an
    uncontrolled nuisance.
    """

    design: str  # D1a | D1b | D2
    arm: str
    schema_onset: int
    # Passage length is a per-schedule property, not a corpus constant: the v3 wave-2 `long` and
    # `late_resolve` arms run at 8 so the pre-onset run-up can be 4 segments while still leaving
    # 4 post-onset. Everything downstream reads controls["n_segments"], never the module default.
    n_segments: int = N_SEGMENTS
    role_onsets: dict[str, int] = field(default_factory=dict)
    ambiguous: tuple[int, ...] = ()
    resolve_segment: int | None = None
    truth_role: str | None = None
    foil_role: str | None = None

    @property
    def staged_pids(self) -> list[str]:
        return sorted(self.role_onsets)

    def to_dict(self) -> dict:
        return {
            "design": self.design,
            "arm": self.arm,
            "n_segments": self.n_segments,
            "schema_onset": self.schema_onset,
            "role_onsets": dict(self.role_onsets),
            "ambiguous_segments": list(self.ambiguous),
            "resolve_segment": self.resolve_segment,
            "truth_role": self.truth_role,
            "foil_role": self.foil_role,
            "n_staged": len(self.role_onsets),
        }


def is_unresolved(ctrl: dict) -> bool:
    """The D2 sibling with no determinate gold, identified semantically rather than by arm name.

    Falls back to the wave-1 literal so corpora already on disk (which predate ``d2_branch``) still
    read correctly.
    """
    branch = ctrl.get("d2_branch")
    if branch is not None:
        return branch == "un"
    return ctrl.get("arm") == "unresolved"


def _full_vector(schema: str) -> tuple[int, ...]:
    """Every role occupied exactly once -- the shared cast shape for the whole batch."""
    return counts_to_vector(schema, dict.fromkeys(role_order(schema), 1))


#: Set by ``build_*`` when ``wave=3``. A corpus-level object: the (role x gender) balance is a
#: running property across items, so a per-item instance would achieve nothing.
_ASSIGNER: BalancedAssigner | None = None


def _base_cast(
    schema: str,
    n_distractors: int,
    *,
    family_id: str,
    item_id: str,
    experiment: str,
    condition: str,
    seed: int,
    rng: random.Random,
) -> BindingSpec:
    """One person per role plus ``n_distractors`` licensed bystanders.

    Under v3 (``_ASSIGNER`` set) the role-bearers come from :class:`BalancedAssigner`, which makes
    gender, name and roster order independent of role. The v1/v2 path is left untouched so those
    corpora stay reproducible.
    """
    if _ASSIGNER is not None:
        return _v3_base_cast(
            schema, n_distractors, family_id=family_id, item_id=item_id, experiment=experiment,
            condition=condition, seed=seed, rng=rng,
        )
    names = NameAllocator(rng=rng)
    scene_path = list(_scenes(schema))
    spec = build_seed_cast(
        schema,
        _full_vector(schema),
        names,
        item_id=item_id,
        experiment=experiment,
        condition=condition,
        family_id=family_id,
        seed=seed,
        scene_path=scene_path,
        controls={},
    )
    if n_distractors:
        distractors = DistractorAllocator(schema=schema, flavor="outside_inventory", rng=rng)
        spec = insert_bystander(
            spec,
            names,
            distractors,
            item_id=item_id,
            condition=condition,
            n=n_distractors,
        )
        spec.edit_op = "seed"
        spec.parent_id = None
    return spec


def _v3_base_cast(
    schema: str,
    n_distractors: int,
    *,
    family_id: str,
    item_id: str,
    experiment: str,
    condition: str,
    seed: int,
    rng: random.Random,
) -> BindingSpec:
    """v3 cast: balanced genders, shared name pool, randomised introduction order.

    Bystanders are drawn from the same name pool and shuffled in with everyone else, so "is this
    person a bystander" is not recoverable from position either -- v1 appended them last, and the
    pre-onset text identified them at .959.
    """
    assert _ASSIGNER is not None
    cast = _ASSIGNER.build_cast(schema)
    used = {p.name for p in cast}
    if n_distractors:
        distractors = DistractorAllocator(schema=schema, flavor="outside_inventory", rng=rng)
        for _ in range(n_distractors):
            entry = distractors.take()
            gender = _ASSIGNER.rng.choice(list(GENDERS))
            name = _ASSIGNER.take_name(gender, "BYSTANDER", schema, used=used)
            used.add(name)
            cast.append(
                Person(pid="", name=name, gender=gender, roles=[],
                       distractor_flavor=distractors.flavor, distractor_key=entry["key"],
                       licence=entry["licence"], beat=entry["beat"], scene=entry["scene"])
            )
        _ASSIGNER.rng.shuffle(cast)
    for i, person in enumerate(cast, start=1):
        person.pid = f"p{i}"
    # a role-independent opening beat per person, so segment 1 cannot stage people by role
    beats = neutral_beats()
    # How often a person is NAMED was the last identity channel: v1 named the patron 12.4 times
    # against a bystander's 1.9, because the patron is the protagonist and appears in every scene.
    # The fix is about naming, not acting -- a patron can still act throughout while being referred
    # to by pronoun -- so each person gets a target drawn independently of their role.
    targets = {p.pid: _ASSIGNER.rng.choice(MENTION_TARGETS) for p in cast}
    controls = {
        "neutral_beats": _ASSIGNER.assign_beats(cast, beats),
        "mention_targets": targets,
    }
    return BindingSpec(
        item_id=item_id, schema=schema, experiment=experiment, condition=condition,
        family_id=family_id, parent_id=None, edit_op="seed", seed=seed, cast=cast,
        scene_path=list(_scenes(schema)), controls=controls,
    )


def _scenes(schema: str) -> list[str]:
    from ...core.spec import load_schema

    return list(load_schema(schema).scenes)


def _holder_pid(spec: BindingSpec, role_id: str) -> str:
    holders = spec.holders(role_id)
    assert holders, f"{role_id} unoccupied in {spec.item_id}"
    return holders[0].pid


def build_d1(seed: int = 20260909) -> list[Family]:
    """D1a (S=1, 15 families/schema) + D1b (S=2, 10 families/schema), 3 siblings each = 150."""
    families: list[Family] = []
    rng = random.Random(seed)
    for schema in ("restaurant", "airport"):
        roles = STAGED_ROLES[schema]
        for i in range(D1A_FAMILIES):
            staged = roles[i % len(roles)]
            n_d = D_LEVELS_D1A[i % len(D_LEVELS_D1A)]
            families.append(_d1a_family(schema, i, staged, n_d, rng))
        for i in range(D1B_FAMILIES):
            # counterbalanced: the two staged roles swap early/late slots on alternate families,
            # so "late" is never confounded with a particular role.
            pair = _d1b_pair(schema, i)
            n_d = D_LEVELS_D1B[i % len(D_LEVELS_D1B)]
            families.append(_d1b_family(schema, i, pair, n_d, rng))
    return families


def _d1b_pair(schema: str, i: int) -> tuple[str, str]:
    """(early role, late role), alternating direction so each role sees both slots."""
    staff = list(STAGED_ROLES[schema])
    a = staff[i % len(staff)]
    b = staff[(i + 1 + i // len(staff)) % len(staff)]
    if b == a:
        b = staff[(i + 2) % len(staff)]
    return (a, b) if i % 2 == 0 else (b, a)


def _d1a_family(
    schema: str, i: int, staged_role: str, n_d: int, rng: random.Random,
    *, arms: tuple = D1A_ARMS, n_segments: int = N_SEGMENTS, prefix: str = "D1a",
) -> Family:
    # `prefix` keeps family ids unique across WAVES. Wave 1 and wave 2 are merged into one bundle
    # and the audit groups CV folds by family_id, so a collision would silently pool two different
    # casts into one fold.
    family_id = f"{prefix}-{schema}-{i:02d}"
    base = _base_cast(
        schema,
        n_d,
        family_id=family_id,
        item_id=f"{family_id}-{arms[0][0]}",
        experiment="D1a",
        condition=f"{staged_role}-D{n_d}",
        seed=rng.randrange(1 << 30),
        rng=rng,
    )
    focal = _holder_pid(base, staged_role)
    items: list[BindingSpec] = []
    for j, (arm, onset) in enumerate(arms):
        schedule = Schedule(
            design="D1a", arm=arm, schema_onset=SCHEMA_ONSET, role_onsets={focal: onset},
            n_segments=n_segments,
        )
        items.append(_sibling(base, family_id, arm, j, schedule, staged_role, n_d))
    return Family(family_id=family_id, experiment="D1a", schema=schema, items=items,
                  meta={"staged_role": staged_role, "D": n_d, "focal_pid": focal})


def _d1b_family(
    schema: str, i: int, pair: tuple[str, str], n_d: int, rng: random.Random
) -> Family:
    family_id = f"D1b-{schema}-{i:02d}"
    early_role, late_role = pair
    base = _base_cast(
        schema,
        n_d,
        family_id=family_id,
        item_id=f"{family_id}-{D1B_ARMS[0][0]}",
        experiment="D1b",
        condition=f"{early_role}+{late_role}-D{n_d}",
        seed=rng.randrange(1 << 30),
        rng=rng,
    )
    early_pid, late_pid = _holder_pid(base, early_role), _holder_pid(base, late_role)
    items: list[BindingSpec] = []
    for j, (arm, late_onset) in enumerate(D1B_ARMS):
        schedule = Schedule(
            design="D1b",
            arm=arm,
            schema_onset=SCHEMA_ONSET,
            role_onsets={early_pid: D1B_FIRST_ONSET, late_pid: late_onset},
        )
        items.append(
            _sibling(base, family_id, arm, j, schedule, f"{early_role}+{late_role}", n_d)
        )
    return Family(family_id=family_id, experiment="D1b", schema=schema, items=items,
                  meta={"early_role": early_role, "late_role": late_role, "D": n_d,
                        "early_pid": early_pid, "late_pid": late_pid})


def build_d2(seed: int = 20260910) -> list[Family]:
    """25 ambiguous-prefix families per schema x 3 siblings = 150."""
    families: list[Family] = []
    rng = random.Random(seed)
    for schema in ("restaurant", "airport"):
        pairs = D2_PAIRS[schema]
        for i in range(FAMILIES_PER_SCHEMA):
            pair = pairs[i % len(pairs)]
            n_d = D_LEVELS_D2[i % len(D_LEVELS_D2)]
            families.append(_d2_family(schema, i, pair, n_d, rng))
    return families


def _d2_family(
    schema: str, i: int, pair: tuple[str, str], n_d: int, rng: random.Random,
    *, arms: tuple = D2_ARMS, ambiguous: tuple = D2_AMBIGUOUS, resolve: int = D2_RESOLVE,
    n_segments: int = N_SEGMENTS, prefix: str = "D2",
) -> Family:
    family_id = f"{prefix}-{schema}-{i:02d}"
    role_a, role_b = pair
    base = _base_cast(
        schema,
        n_d,
        family_id=family_id,
        item_id=f"{family_id}-{arms[0]}",
        experiment="D2",
        condition=f"{role_a}vs{role_b}-D{n_d}",
        seed=rng.randrange(1 << 30),
        rng=rng,
    )
    # Both candidate roles are occupied, by two named people, and the AMBIGUITY IS WHICH IS WHICH.
    # `resolve_b` swaps their role lists, so the two branches carry genuinely different gold off an
    # identical prefix -- the load-bearing contrast. Keeping one cast and only relabelling
    # `focal_pid` (the first attempt) gave both branches the SAME gold, so the contrast was prose
    # deep only. B, R, D and E are untouched by the swap.
    focal_a, focal_b = _holder_pid(base, role_a), _holder_pid(base, role_b)
    # `unresolved` has no determinate gold, so if it always inherited resolve_a's assignment a
    # candidate would hold role A in two of three siblings and role B in one. That 2:1 is a
    # position->role association built into the design, and it is what failed audit A3 on the first
    # D2 v3 run (+0.101 against a random band whose 95th percentile is +0.081).
    unresolved_swaps = bool(rng.getrandbits(1))
    items: list[BindingSpec] = []
    for j, arm in enumerate(arms):
        # arm ROLE is positional, so the late-resolve arms reuse the same a/b/unresolved semantics
        is_b, is_un = j == 1, j == 2
        truth = role_b if is_b else role_a
        foil = role_a if is_b else role_b
        schedule = Schedule(
            design="D2",
            arm=arm,
            schema_onset=SCHEMA_ONSET,
            ambiguous=ambiguous,
            resolve_segment=None if is_un else resolve,
            n_segments=n_segments,
            truth_role=truth,
            foil_role=foil,
        )
        spec = _sibling(base, family_id, arm, j, schedule, f"{role_a}vs{role_b}", n_d)
        swap = is_b or (is_un and unresolved_swaps)
        if swap:
            # swap the two candidates' roles in this sibling's own cast
            spec.person(focal_a).roles = [role_b]
            spec.person(focal_b).roles = [role_a]
        spec.controls["unresolved_swapped"] = swap
        # the tracked person is the SAME individual in every arm -- only their role moves, which is
        # what makes the branches comparable at the same name token.
        spec.controls["focal_pid"] = focal_a
        spec.controls["partner_pid"] = focal_b
        spec.controls["candidate_pids"] = [focal_a, focal_b]
        spec.controls["candidate_roles"] = [role_a, role_b]
        # the unresolved sibling has no determinate gold role for either candidate: it is dropped
        # from role_accuracy / role_precision and read only through the judge's `role_as_stated`.
        spec.controls["scoring_exclude"] = is_un
        # SEMANTIC branch label. Consumers used to compare `arm` against the literals
        # "resolve_a"/"resolve_b"/"unresolved"; wave 4 renames the arms (`late_*`, `pairs_*`), and
        # every such comparison silently took the wrong branch -- `resolve_segment` stayed set on
        # an unresolved sibling, which surfaced as `int(None)` deep inside the prefix audit.
        spec.controls["d2_branch"] = ("a", "b", "un")[j]
        items.append(spec)
    return Family(family_id=family_id, experiment="D2", schema=schema, items=items,
                  meta={"pair": [role_a, role_b], "D": n_d,
                        "focal_pids": {role_a: focal_a, role_b: focal_b}})


def _sibling(
    base: BindingSpec,
    family_id: str,
    arm: str,
    index: int,
    schedule: Schedule,
    staged: str,
    n_d: int,
) -> BindingSpec:
    """One arm of a family: the SAME cast, a different schedule.

    ``edit_op`` is ``seed`` for the first arm and ``reschedule`` / ``resolve`` for the rest, so the
    lineage reads the same way 1A/1B/1C do -- and ``verify._cast_identity`` leaves unknown ops
    alone, which is correct here because the cast is identical by construction.
    """
    seed_arm = (
        D2_ARMS[0]
        if schedule.design == "D2"
        else (D1A_ARMS if schedule.design == "D1a" else D1B_ARMS)[0][0]
    )
    spec = base.copy(
        item_id=f"{family_id}-{arm}",
        condition=f"{staged}-D{n_d}-{arm}",
        parent_id=None if index == 0 else f"{family_id}-{seed_arm}",
    )
    if index:
        spec.edit_op = "resolve" if schedule.design == "D2" else "reschedule"
    else:
        spec.edit_op = "seed"
    spec.controls = {
        **base.controls,
        **schedule.to_dict(),
        "staged": staged,
        "D": n_d,
        "prop_naming": "named",
        "role_naming": "latent",
    }
    if schedule.role_onsets:
        spec.controls["focal_pid"] = schedule.staged_pids[0]
    return spec


def build_all(seed: int = 20260909) -> list[Family]:
    families = build_d1(seed) + build_d2(seed + 1)
    assign_prop_naming(families)
    return families


def assign_prop_naming(families: list[Family]) -> None:
    """Fix ``prop_naming`` on every family, in place, balanced 50/50 within each
    (experiment x schema) arm.

    Round-robin rather than random, for the reason ``binding/design.assign_naming`` gives: the
    assignment has to be orthogonal to condition and must not move when a run is resumed. Under
    ``abstract`` the schema onset is dropped -- there is nothing to disclose at segment 2, because
    the situation is never named -- so ``schema_onset`` becomes ``None`` and the diagnostic-prop
    gate inverts (see ``dynamics/verify.py``).
    """
    counters: dict[tuple[str, str], int] = {}
    for family in families:
        key = (family.experiment, family.schema)
        i = counters.get(key, 0)
        counters[key] = i + 1
        level = PROP_NAMING_LEVELS[i % len(PROP_NAMING_LEVELS)]
        family.meta["prop_naming"] = level
        for item in family.items:
            item.controls["prop_naming"] = level
            if level == "abstract":
                item.controls["schema_onset"] = None


# ============================================================================ wave 2
# A second batch of 150 items per design. The point is NOT more n at the same coordinates -- v1
# already has 50 items per D1a arm -- but coordinates v1 cannot reach, chosen from what the v1 data
# actually showed:
#
#   * onset segment is PERFECTLY confounded with absolute position in v1 (the onset segment begins
#     at 0.17 / 0.33 / 0.49 of the passage, sd ~ .015, because every segment is the same length).
#     So "the decode rises at mention 3" cannot be separated from "the decode rises at 49% of the
#     way through". `LENGTH_PROFILES` breaks that.
#   * D2's ambiguity gate passes 0.69 for staff-vs-staff pairs and 0.25 for patron-vs-staff: the
#     default-role prior eats the ambiguity. Wave 2 is weighted to where usable ambiguity lives.
#   * PATRON is on no D1 ladder and `role_naming="named"` has never been generated for any binding
#     corpus, so the prior-congruent case and the lexical ceiling are both unmeasured.

# Per-segment word bands. `uniform` reproduces wave 1. `front` makes the opening segments long and
# the rest short, `back` the reverse, which moves the character position of a given onset SEGMENT
# by ~0.25 of the passage in either direction -- enough that onset3 spans the range wave 1 gave to
# onset2 and onset4, and segment index is no longer a proxy for position.
LENGTH_PROFILES: dict[str, tuple[tuple[int, int], ...]] = {
    "uniform": ((55, 75),) * N_SEGMENTS,
    "front": ((90, 110), (90, 110), (40, 55), (40, 55), (40, 55), (40, 55)),
    "back": ((40, 55), (40, 55), (75, 95), (75, 95), (75, 95), (75, 95)),
}

# D1 wave-2 arms.
W2_D1_ARMS = ("position", "congruent", "lexical")
# D2 wave-2 arms.
W2_D2_ARMS = ("staff", "late", "garden")

D2_RESOLVE_LATE = 5
# Garden path: the ambiguous window LEANS toward the foil at segment 3, and segment 5 resolves the
# other way. Not a contradiction -- the lean is suggestive evidence, never a statement -- so the
# opening stays true, which the consistency audit checks. This is the only arm in either design
# where a binding has to be REVISED rather than merely formed.
#: Per-person name-mention targets, drawn independently of role. The floor is 6 because a staged
#: person must be named at least once in each of the six segments; the spread keeps the prose from
#: reading like a roll-call while leaving count decorrelated from role.
MENTION_TARGETS = (6, 7, 8, 9)

D2_GARDEN_LEAN = 3
D2_GARDEN_RESOLVE = 5

# Pairs for the BALANCED-ambiguity arms, chosen from wave 1's measured pass rates rather than from
# taste. Per pair, `audit_ok` was:
#
#   restaurant (SERVICE_AGENT, PRODUCER)   0.88     airport (PATRON, GATEKEEPER)        0.62
#   airport    (FRONT_AGENT, SERVICE_AGENT) 0.50    restaurant (PATRON, SERVICE_AGENT)  0.22
#   airport    (PATRON, FRONT_AGENT)       0.19     restaurant (PATRON, FRONT_AGENT)    0.00
#
# Two readings of that table drive wave 2. First, every PATRON pair fails: an under-determined
# person is read as the patron, so "balanced" is not reachable there. Second, restaurant's
# FRONT_AGENT has no workable partner at all -- a host's actions either read as host or, once
# generalised enough to be ambiguous, collapse into "server" (8/8 in the wave-2 smoke).
D2_STAFF_PAIRS: dict[str, tuple[tuple[str, str], ...]] = {
    "restaurant": (("SERVICE_AGENT", "PRODUCER"),),
    "airport": (
        ("FRONT_AGENT", "SERVICE_AGENT"),
        ("GATEKEEPER", "SERVICE_AGENT"),
        ("FRONT_AGENT", "GATEKEEPER"),
    ),
}

# The garden path wants the OPPOSITE property: a pair whose prior is lopsided, so segment 3 can
# lean on the reader's own default and segment 5 can overturn it. The pairs that fail balanced
# ambiguity are therefore the BEST garden-path pairs -- PATRON is the attractor, so it is always
# the FOIL and the staff role is always the truth ("you assumed customer; they are staff").
D2_GARDEN_PAIRS: dict[str, tuple[tuple[str, str], ...]] = {
    "restaurant": (("SERVICE_AGENT", "PATRON"), ("FRONT_AGENT", "PATRON")),
    "airport": (("FRONT_AGENT", "PATRON"), ("GATEKEEPER", "PATRON")),
}


# Wave-1 D1a staged-role counts, in ITEMS (3 per family), measured from
# results/schema_llm/data/binding_dyn_d1_v1. Wave 2 allocates its staff families to whichever role
# is thinnest so far, so the COMBINED corpus is balanced rather than each wave being balanced
# separately -- round-robin within wave 2 alone left airport PRODUCER at 18 items against
# FRONT_AGENT's 27, and role is a factor we want to compare across, not merely log.
WAVE1_D1A_ROLE_ITEMS: dict[str, dict[str, int]] = {
    "restaurant": {"FRONT_AGENT": 15, "SERVICE_AGENT": 15, "PRODUCER": 15, "PATRON": 0},
    "airport": {
        "FRONT_AGENT": 12, "GATEKEEPER": 12, "SERVICE_AGENT": 12, "PRODUCER": 9, "PATRON": 0
    },
}


def _thinnest(counts: dict[str, int], candidates: tuple[str, ...]) -> str:
    """The candidate role with the fewest items so far; ties break in declared order."""
    return min(candidates, key=lambda r: (counts.get(r, 0), candidates.index(r)))


def build_d1_wave2(seed: int = 20260910) -> list[Family]:
    """150 D1 items: 60 position-decorrelated + 45 prior-congruent + 45 lexical-ceiling."""
    families: list[Family] = []
    rng = random.Random(seed)
    for schema in ("restaurant", "airport"):
        staff = STAGED_ROLES[schema]
        counts = dict(WAVE1_D1A_ROLE_ITEMS[schema])
        n_side = 8 if schema == "restaurant" else 7
        # --- position (10 families/schema x 3 onsets = 60): same schedule, unequal segments ------
        for i in range(10):
            role = _thinnest(counts, staff)
            counts[role] = counts.get(role, 0) + len(D1A_ARMS)
            families.append(
                _w2_d1a_family(
                    schema, f"W2p-{schema}-{i:02d}", role,
                    D_LEVELS_D1A[i % 3], rng, arm="position",
                    length_profile=("front", "back")[i % 2],
                )
            )
        # --- lexical (7-8 families/schema x 3 = 45): role_naming="named", the ceiling control ----
        for i in range(n_side):
            role = _thinnest(counts, staff)
            counts[role] = counts.get(role, 0) + len(D1A_ARMS)
            families.append(
                _w2_d1a_family(
                    schema, f"W2l-{schema}-{i:02d}", role,
                    D_LEVELS_D1A[i % 3], rng, arm="lexical", length_profile="uniform",
                    role_naming="named",
                )
            )
        # --- congruent (7-8 families/schema x 3 = 45): PATRON staged, the prior-CONGRUENT case ---
        # PATRON is on no wave-1 ladder at all (it is the DEFAULT reading of an unbound person, so
        # wave 1 excluded it), which is exactly why it needs its own arm rather than a share of the
        # staff rotation: without it the corpus can only say what happens when disclosure has to
        # MOVE the representation, never what happens when it merely confirms it.
        for i in range(n_side):
            families.append(
                _w2_d1a_family(
                    schema, f"W2c-{schema}-{i:02d}", "PATRON",
                    D_LEVELS_D1A[i % 3], rng, arm="congruent", length_profile="uniform",
                )
            )
    return families


def _w2_d1a_family(
    schema: str,
    family_id: str,
    staged_role: str,
    n_d: int,
    rng: random.Random,
    *,
    arm: str,
    length_profile: str,
    role_naming: str = "latent",
) -> Family:
    """A wave-2 D1a family: the wave-1 onset ladder under one new coordinate."""
    base = _base_cast(
        schema, n_d, family_id=family_id, item_id=f"{family_id}-{D1A_ARMS[0][0]}",
        experiment="D1a-w2", condition=f"{arm}-{staged_role}-D{n_d}",
        seed=rng.randrange(1 << 30), rng=rng,
    )
    focal = _holder_pid(base, staged_role)
    items: list[BindingSpec] = []
    for j, (onset_arm, onset) in enumerate(D1A_ARMS):
        schedule = Schedule(
            design="D1a", arm=onset_arm, schema_onset=SCHEMA_ONSET, role_onsets={focal: onset}
        )
        spec = _sibling(base, family_id, onset_arm, j, schedule, staged_role, n_d)
        spec.controls.update(
            wave=2, w2_arm=arm, length_profile=length_profile, role_naming=role_naming
        )
        items.append(spec)
    return Family(
        family_id=family_id, experiment="D1a-w2", schema=schema, items=items,
        meta={"staged_role": staged_role, "D": n_d, "focal_pid": focal,
              "w2_arm": arm, "length_profile": length_profile, "role_naming": role_naming},
    )


def build_d2_wave2(seed: int = 20260911) -> list[Family]:
    """150 D2 items: 60 staff-vs-staff + 45 late-resolution + 45 garden-path."""
    families: list[Family] = []
    rng = random.Random(seed)
    for schema in ("restaurant", "airport"):
        staff = D2_STAFF_PAIRS[schema]
        for i in range(10):  # staff: 10 x 3 x 2 = 60
            families.append(
                _w2_d2_family(schema, f"W2s-{schema}-{i:02d}", staff[i % len(staff)],
                              D_LEVELS_D2[i % 2], rng, arm="staff")
            )
        for i in range(8 if schema == "restaurant" else 7):  # late: 45
            families.append(
                _w2_d2_family(schema, f"W2t-{schema}-{i:02d}", staff[i % len(staff)],
                              D_LEVELS_D2[i % 2], rng, arm="late")
            )
        garden = D2_GARDEN_PAIRS[schema]
        for i in range(8 if schema == "restaurant" else 7):  # garden: 45
            families.append(
                _w2_d2_family(schema, f"W2g-{schema}-{i:02d}", garden[i % len(garden)],
                              D_LEVELS_D2[i % 2], rng, arm="garden")
            )
    return families


def _w2_d2_family(
    schema: str, family_id: str, pair: tuple[str, str], n_d: int,
    rng: random.Random, *, arm: str,
) -> Family:
    role_a, role_b = pair
    base = _base_cast(
        schema, n_d, family_id=family_id, item_id=f"{family_id}-{D2_ARMS[0]}",
        experiment="D2-w2", condition=f"{arm}-{role_a}vs{role_b}-D{n_d}",
        seed=rng.randrange(1 << 30), rng=rng,
    )
    focal_a, focal_b = _holder_pid(base, role_a), _holder_pid(base, role_b)
    unresolved_swaps = bool(rng.getrandbits(1))
    resolve = {"staff": D2_RESOLVE, "late": D2_RESOLVE_LATE, "garden": D2_GARDEN_RESOLVE}[arm]
    items: list[BindingSpec] = []
    for j, d2_arm in enumerate(D2_ARMS):
        truth = role_b if d2_arm == "resolve_b" else role_a
        foil = role_a if d2_arm == "resolve_b" else role_b
        schedule = Schedule(
            design="D2", arm=d2_arm, schema_onset=SCHEMA_ONSET, ambiguous=D2_AMBIGUOUS,
            resolve_segment=None if d2_arm == "unresolved" else resolve,
            truth_role=truth, foil_role=foil,
        )
        spec = _sibling(base, family_id, d2_arm, j, schedule, f"{role_a}vs{role_b}", n_d)
        # `resolve_b` swaps the candidates' roles; `unresolved` has no determinate gold, so left
        # alone it would inherit resolve_a's assignment and a candidate would hold role A in two of
        # three siblings and role B in one. That 2:1 is a position->role association built into the
        # design, and it is what failed A3 on the first D2 v3 run (+0.101 against a random band
        # whose 95th percentile is +0.081). Alternating the unresolved arm balances it.
        swap = d2_arm == "resolve_b" or (d2_arm == "unresolved" and unresolved_swaps)
        if swap:
            spec.person(focal_a).roles = [role_b]
            spec.person(focal_b).roles = [role_a]
        spec.controls.update(
            focal_pid=focal_a, partner_pid=focal_b, unresolved_swapped=swap,
            candidate_pids=[focal_a, focal_b], candidate_roles=[role_a, role_b],
            scoring_exclude=d2_arm == "unresolved",
            d2_branch=("a", "b", "un")[j],
            wave=2, w2_arm=arm,
            # The garden path leans the SAME way in every arm -- toward `role_b`, which for a
            # garden pair is always the attractor (PATRON) -- and only the resolution differs:
            # `resolve_a` overturns the lean (REVISE), `resolve_b` confirms it (CONFIRM). Holding
            # the lean fixed is what makes the pair a clean confirm-vs-revise contrast off an
            # identical prefix; leaning toward whichever role happened to be the foil would have
            # made the two arms differ in two things at once, and would also have asked the
            # generator to lean AGAINST the reader's prior in half the items.
            lean_segment=D2_GARDEN_LEAN if arm == "garden" else None,
            lean_role=role_b if arm == "garden" and d2_arm != "unresolved" else None,
            revises=arm == "garden" and d2_arm == "resolve_a",
        )
        items.append(spec)
    return Family(
        family_id=family_id, experiment="D2-w2", schema=schema, items=items,
        meta={"pair": [role_a, role_b], "D": n_d, "w2_arm": arm,
              "focal_pids": {role_a: focal_a, role_b: focal_b}},
    )


def build_wave2(seed: int = 20260910) -> list[Family]:
    families = build_d1_wave2(seed) + build_d2_wave2(seed + 1)
    assign_prop_naming(families)
    # the lexical-ceiling arm's whole point is that the ROLE word is present; leaving prop_naming
    # free there would confound the two lexical manipulations, so it is pinned to `named`.
    for family in families:
        if family.meta.get("w2_arm") == "lexical":
            family.meta["prop_naming"] = "named"
            for item in family.items:
                item.controls["prop_naming"] = "named"
                item.controls["schema_onset"] = SCHEMA_ONSET
    return families


# ======================================================================= wave 3 (de-confounded)
# Same timing design as waves 1-2 -- onsets were independently gated and the external review
# confirmed them sound -- with the identity and staging confounds removed. See
# main/plans/20260913-BINDING_DYNAMICS_CONFOUND_AUDIT_PLAN.md.


def build_wave3(design: str, seed: int = 20260914) -> list[Family]:
    """The v3 grid for one design. ``design`` is ``"D1"`` or ``"D2"``.

    Installs the corpus-level :class:`BalancedAssigner` before building, so every cast drawn during
    this call shares one running balance. Reuses the wave-1 grid shape (arms, n, prop_naming
    balancing) -- only the cast assignment and the segment-1 staging change.
    """
    global _ASSIGNER
    # distinct seed per design: the assigner samples its name subset at construction, so a shared
    # seed would give D1 and D2 the SAME names and halve the corpus-wide variety
    _ASSIGNER = BalancedAssigner(seed=seed + (0 if design == "D1" else 977))
    try:
        families = build_d1(seed) if design == "D1" else build_d2(seed + 1)
        assign_prop_naming(families)
        for family in families:
            family.meta["wave"] = 3
            for item in family.items:
                item.controls["wave"] = 3
        return families
    finally:
        _ASSIGNER = None


def assigner_report() -> dict | None:
    """Balance diagnostics from the last :func:`build_wave3` call, for tests and logging."""
    return _ASSIGNER.balance_report() if _ASSIGNER is not None else None


# ------------------------------------------------------------------- v3 wave 2 (the second wave)
#
# Coverage measured on the shipped v3 wave 1 (300 items, both designs) before choosing these:
# every passage is 6 segments; distractors are never flavoured; D1 runs 120 low / 30 high; and
# D2's candidate pairs are skewed because (PATRON, FRONT_AGENT) is the only pair listed under BOTH
# schemas, so it draws twice as often and takes 23% of items against 5-7% for the rest.
#
# Two arms per design, chosen to vary TIMING and PRESSURE inside the existing label space:
#
#   D1 `long`      -- 8 segments, onsets 3/4/5. The pre-onset flat null is the load-bearing claim
#                     of the whole design and wave 1 tests it over a 1-3 segment run-up; this gives
#                     2-4, while still leaving >= 3 post-onset segments to read the bound state.
#   D1 `pressure`  -- D = 3 (high distractors) at the wave-1 onsets, which also flattens 120/30.
#   D2 `late`      -- 8 segments, ambiguous 2-5, resolution at 6: a four-segment ambiguous window
#                     against wave 1's two, so a stable mixture is distinguishable from a decaying
#                     one over a longer hold.
#   D2 `pairs`     -- drops (PATRON, FRONT_AGENT) and cycles the remaining pairs, which flattens
#                     the pair distribution of the UNION rather than of this wave alone.
#
# Deliberately NOT here: two people sharing a role, and one person holding two roles. Both are real
# gaps, but every consumer assumes one role per person -- `confound_audit.person_table` takes
# `roles[0]`, the role classifier is single-label, A9 counts (schema, role, gender) cells, and the
# per-mention decode has no defined target when two names share a role. That is a data-model change
# plus an audit rewrite, so it belongs in its own wave.

W4_LONG_SEGMENTS = 8
W4_LONG_ARMS = (("long3", 3), ("long4", 4), ("long5", 5))
W4_PRESSURE_ARMS = (("press2", 2), ("press3", 3), ("press4", 4))
W4_PRESSURE_D = 3

W4_LATE_SEGMENTS = 8
# The ambiguous WINDOW stays at wave 1's proven two segments; what the arm extends is what comes
# AFTER the resolution -- five segments against wave 1's two.
#
# Stretching the hold was tried first and does not work. At (2,3,4,5)/resolve 6 every pilot sibling
# failed identically: "1.00 of readings commit to PATRON at the ambiguous cut", 0.00 holding both.
# That is plan section 7.5 -- an unbound person reads as the diner by DEFAULT -- so a longer hold
# does not preserve ambiguity, it gives the reader more time to settle into the prior. Shortening
# to three and dropping PATRON pairs did not rescue it either: the surviving pair
# (SERVICE_AGENT, PRODUCER) shares a SUPERORDINATE, so readers answered "restaurant staff" and
# scored 0.38 on holding both, while the other family leaked outright.
#
# The shipped wave-1 corpus settles it. It runs at 50% audit_ok overall (75 / 63 / 12), and by pair:
# (SERVICE_AGENT, PRODUCER) 21 ok / 3 flagged, (FRONT_AGENT, SERVICE_AGENT) 15/9,
# (PATRON, GATEKEEPER) 12/12, (PATRON, SERVICE_AGENT) 9/18, (PATRON, FRONT_AGENT) 18/33 -- exactly
# the ordering the default-prior account predicts. Two segments is the window that holds, so the
# arm changes the one thing that is safe to change and keeps the vetted pair list.
#
# This also serves the decode directly: the post-onset trajectory is the measurement, and wave 1
# gives it two segments. Five is what makes "does the binding keep strengthening?" answerable.
W4_LATE_AMBIGUOUS = D2_AMBIGUOUS
W4_LATE_RESOLVE = D2_RESOLVE
W4_LATE_ARMS = ("late_a", "late_b", "late_un")
W4_PAIR_ARMS = ("pairs_a", "pairs_b", "pairs_un")
# Pair allocation is set by two MEASURED facts about wave 1, not by taste.
#
# 1. Union balance. Wave 1 ran (PATRON, FRONT_AGENT) 51 and every other pair 24-27, because it is
#    the only pair listed under BOTH schemas and so drew twice. Flattening the union at 300 needs
#    roughly +36 / +36 / +33 / +36 for the other four and +9 for that one, so wave 2 gives it none.
# 2. Which pairs actually hold their ambiguity. Wave 1 audit_ok by pair:
#    (SERVICE_AGENT, PRODUCER) 21 ok / 3 flagged, (FRONT_AGENT, SERVICE_AGENT) 15/9,
#    (PATRON, GATEKEEPER) 12/12, (PATRON, SERVICE_AGENT) 9/18, (PATRON, FRONT_AGENT) 18/33.
#    Every PATRON pair is at or below chance of holding, for the plan-7.5 reason.
#
# So the EIGHT-segment arm -- the one whose extra length is the point -- is spent on the two pairs
# that reliably hold, and the six-segment arm carries the PATRON pairs that repair the balance and
# that wave 1 already shows are usable at ~50%.
#: 8-segment arm: the two pairs that hold their ambiguity.
W4_LATE_PAIRS: dict[str, tuple[tuple[str, str], ...]] = {
    "restaurant": (("SERVICE_AGENT", "PRODUCER"),),
    "airport": (("FRONT_AGENT", "SERVICE_AGENT"),),
}
#: 6-segment arm: the pairs the union is short of. (PATRON, FRONT_AGENT) is already over.
W4_PAIRS: dict[str, tuple[tuple[str, str], ...]] = {
    "restaurant": (("PATRON", "SERVICE_AGENT"),),
    "airport": (("PATRON", "GATEKEEPER"),),
}
W4_LONG_FAMILIES = 13
W4_OTHER_FAMILIES = 12
# D2 leans away from `late`: its pair list is down to one per schema, so 13 families would put 13
# identical pairs in a row. `pairs` carries the surplus and is where the union's pair balance is
# repaired anyway.
W4_LATE_FAMILIES = 13
W4_PAIRS_FAMILIES = 12


def build_d1_wave4(seed: int = 20260915) -> list[Family]:
    """150 D1 items: 13 `long` + 12 `pressure` families per schema, 3 siblings each."""
    families: list[Family] = []
    rng = random.Random(seed)
    for schema in ("restaurant", "airport"):
        roles = STAGED_ROLES[schema]
        for i in range(W4_LONG_FAMILIES):
            families.append(_d1a_family(
                schema, i, roles[i % len(roles)], D_LEVELS_D1A[i % len(D_LEVELS_D1A)], rng,
                arms=W4_LONG_ARMS, n_segments=W4_LONG_SEGMENTS, prefix="D1aL",
            ))
        for i in range(W4_OTHER_FAMILIES):
            families.append(_d1a_family(
                schema, i, roles[i % len(roles)], W4_PRESSURE_D, rng,
                arms=W4_PRESSURE_ARMS, prefix="D1aP",
            ))
    return families


def build_d2_wave4(seed: int = 20260916) -> list[Family]:
    """150 D2 items: 13 `late` + 12 `pairs` families per schema, 3 siblings each."""
    families: list[Family] = []
    rng = random.Random(seed)
    for schema in ("restaurant", "airport"):
        late, wide = W4_LATE_PAIRS[schema], W4_PAIRS[schema]
        for i in range(W4_LATE_FAMILIES):
            families.append(_d2_family(
                schema, i, late[i % len(late)], D_LEVELS_D2[i % len(D_LEVELS_D2)], rng,
                arms=W4_LATE_ARMS, ambiguous=W4_LATE_AMBIGUOUS, resolve=W4_LATE_RESOLVE,
                n_segments=W4_LATE_SEGMENTS, prefix="D2L",
            ))
        for i in range(W4_PAIRS_FAMILIES):
            families.append(_d2_family(
                schema, i, wide[i % len(wide)], D_LEVELS_D2[i % len(D_LEVELS_D2)], rng,
                arms=W4_PAIR_ARMS, prefix="D2P",
            ))
    return families


#: Wave 5 = the LEXICAL CEILING, built as matched TWINS of wave 4 rather than as an independent
#: corpus. `build_wave4` is re-run with the same seed, so the assigner replays identically and every
#: cast, schedule, onset and prop_naming is **the same object it was**; only `role_naming` flips to
#: "named", which is read at prose time (`prose._role_rule`) and nowhere else.
#:
#: Why twins and not a fresh grid. A ceiling is only a reference if it differs from the latent arm
#: in exactly one thing. Sharing the cast gives that, and it buys two more properties that matter at
#: small N: the comparison is PAIRED, so it has power a 45-item independent sample would not; and
#: any residual identity signal is IDENTICAL in both arms, so it cancels in the contrast instead of
#: inflating the ceiling. That is what makes a small named set scientifically worth generating.
WAVE5_FAMILIES = 15


def build_named_ceiling(design: str, resume_from, seed: int = 20260915,
                        n_families: int = WAVE5_FAMILIES, *,
                        role_naming: str = "named", suffix: str = "NAMED") -> list[Family]:
    """Matched twins of the first ``n_families`` wave-4 families, with ``role_naming`` flipped.

    ``role_naming="named"`` builds the lexical ceiling itself. ``role_naming="latent"`` builds the
    GENERATOR CONTROL: the ceiling was generated on a different TRAPI deployment than the wave-4
    latent corpus (the default one was saturated), which confounds "role named" with "different
    writer". Generating a latent arm on the *same* deployment as the named arm turns the ceiling
    back into a clean within-generator contrast, and costs one run because the casts are shared.
    """
    families = build_wave4(design, resume_from, seed)[:n_families]
    for family in families:
        twin_id = family.family_id
        family.family_id = f"{twin_id}-{suffix}"
        family.meta["wave"] = 5
        family.meta["twin_of"] = twin_id
        for item in family.items:
            item.family_id = family.family_id
            item.item_id = f"{item.item_id}-{suffix}"
            item.controls["role_naming"] = role_naming
            item.controls["wave"] = 5
            # the latent item this one is paired with -- the analysis joins on it
            item.controls["twin_of"] = item.item_id[: -(len(suffix) + 1)]
    return families


def build_wave4(design: str, resume_from, seed: int = 20260915) -> list[Family]:
    """The v3 wave-2 grid for one design, CONTINUING the wave-1 corpus at ``resume_from``.

    The assigner is resumed rather than rebuilt so the combined corpus reuses wave 1's exact name
    subset and fills each name's missing roles: projected over the union, every name lands in >= 2
    roles at ~4.0 roles/name, against 91.7% and 2.32 for a fresh pool (see
    :meth:`BalancedAssigner.resume_from`).
    """
    global _ASSIGNER
    _ASSIGNER = BalancedAssigner.resume_from(
        resume_from, seed=seed + (0 if design == "D1" else 977)
    )
    try:
        families = build_d1_wave4(seed) if design == "D1" else build_d2_wave4(seed + 1)
        assign_prop_naming(families)
        for family in families:
            family.meta["wave"] = 4
            for item in family.items:
                item.controls["wave"] = 4
        return families
    finally:
        _ASSIGNER = None
