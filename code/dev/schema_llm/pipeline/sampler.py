"""Sample a per-scenario recipe (canonical) from a schema spec (A4.3 step 1)."""

from __future__ import annotations

import random
import re
from dataclasses import asdict, dataclass

import yaml

from ..core.spec import SCHEMAS_DIR, Schema

SHARED = SCHEMAS_DIR / "_shared"

PROP_NAMING = ["named", "abstract", "mixed"]
ROLE_NAMING = ["latent", "named"]  # latent = action-only; named = positive control
DETAIL_LEVELS = ["terse", "medium", "rich"]
LEN_BANDS = ["short", "medium", "long"]
DISTRACTOR_LEVELS = ["none", "low", "high"]
# Violation conditions (plan A5). Structural ones edit scene_path/roles (verified as-is);
# content ones record a spec that generator.realize_template injects into the text.
PROTAGONIST = "PATRON"  # the shared core anchor both schemas follow end-to-end
VIOLATION_TYPES = ["wrong_order", "missing_scene", "wrong_action", "role_swap", "prop_mismatch"]
# Structural violations edit scene_path/roles, so the deterministic verifier already checks them
# (scene set/order); content violations are injected into the TEXT by generator.realize_template
# and are invisible to that verifier -> after a paraphrase they must be LLM-checked to confirm the
# anomaly survived (B5).
CONTENT_VIOLATIONS = ["wrong_action", "role_swap", "prop_mismatch"]
CONDITIONS = ["canonical", *VIOLATION_TYPES]
# Temporal-order forms (SCHEMA_SCENE_ORDER_CONTROL_PLAN): each maps the canonical story order
# `schema.scenes` onto a textual beat order `scene_path` (a permutation, or repeat-augmented) so
# STORY order and TEXTUAL position can be decorrelated. These are DECLARED, canonical-content
# reorderings -- ORTHOGONAL to violations (`condition` stays "canonical"; identify gold is
# unchanged; no roles are dropped). `canonical` keeps the natural order but still records
# `recipe.temporal` (the scene-order corpus's matched-note control); a plain item that never
# passes through `apply_temporal_form` keeps `temporal=None` and is byte-identical to legacy
# generation.
TEMPORAL_FORMS = [
    "canonical", "reversed", "shuffled", "blockswap",
    "flashback", "forgetful", "repeat",
]
_SLOT = re.compile(r"\{([A-Z_]+)\}")
# Detail and length share one lever (adjuncts on a fixed scene skeleton), so length is sampled
# conditioned on detail: more detail biases toward (allows) a longer passage. This forbids the
# contradictory corners -- terse+long, rich+short -- while keeping genuine length spread.
_LEN_BY_DETAIL = {
    "terse": ["short", "medium"],
    "medium": ["short", "medium", "long"],
    "rich": ["medium", "long"],
}
_DISTRACTOR_COUNTS = {"none": (0, 0), "low": (1, 2), "high": (3, 4)}
_GENDER_KEY = {"m": "male", "f": "female", "n": "neutral"}


@dataclass
class Referent:
    ref: str
    form: str  # name | description
    value: str
    gender: str  # m | f | n


@dataclass
class Recipe:
    scenario_id: str
    schema: str
    seed: int
    condition: str
    scene_path: list[str]
    roles: dict[str, list[Referent]]
    controls: dict
    violation: dict | None = None
    temporal: dict | None = None  # temporal-order form metadata (None = natural order, legacy)

    def to_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "schema": self.schema,
            "seed": self.seed,
            "condition": self.condition,
            "scene_path": self.scene_path,
            "roles": {rid: [asdict(r) for r in refs] for rid, refs in self.roles.items()},
            "controls": self.controls,
            "violation": self.violation,
            "temporal": self.temporal,
        }


def load_names() -> dict[str, list[str]]:
    return yaml.safe_load((SHARED / "names.yaml").read_text(encoding="utf-8"))


def load_descriptions() -> dict:
    return yaml.safe_load((SHARED / "descriptions.yaml").read_text(encoding="utf-8"))


def load_distractors() -> dict:
    return yaml.safe_load((SHARED / "distractors.yaml").read_text(encoding="utf-8"))


def _pick_name(rng: random.Random, names: dict, gender: str, used: set[str]) -> str | None:
    pool = [n for n in names.get(_GENDER_KEY[gender], []) if n.lower() not in used]
    if not pool:  # fall back to any gender
        pool = [n for grp in names.values() for n in grp if n.lower() not in used]
    return rng.choice(pool) if pool else None


def _pick_description(
    rng: random.Random,
    descriptions: dict,
    gender: str,
    used: set[str],
    adult_only: bool = False,
) -> tuple[str, str]:
    pool = descriptions["templates"]
    if adult_only:  # working staff roles are never a child (A4.1)
        pool = [t for t in pool if not t.get("child")]
    templates = [t for t in pool if t["gender"] == gender] or pool
    for _ in range(8):  # try a few times to avoid an exact duplicate within the prompt
        t = rng.choice(templates)
        text = t["text"]
        if "{color}" in text:
            text = text.replace("{color}", rng.choice(descriptions["colors"]))
        if text.lower() not in used:
            return text, t["gender"]
    return text, t["gender"]


def _prop_map(schema: Schema, mode: str, rng: random.Random) -> dict[str, str]:
    """Per-object naming: ``named``/``abstract`` apply to every object; ``mixed`` samples
    each object/place role independently (more diverse prompts)."""
    if mode == "mixed":
        return {rid: rng.choice(["named", "abstract"]) for rid in schema.object_roles()}
    return {rid: mode for rid in schema.object_roles()}


def _sample_distractors(distractors: dict, level: str, rng: random.Random) -> list[str]:
    """Pick k off-schema background items (entities + events) for the given level."""
    lo, hi = _DISTRACTOR_COUNTS[level]
    k = rng.randint(lo, hi)
    if not k:
        return []
    pool = list(distractors.get("entities", [])) + list(distractors.get("events", []))
    return rng.sample(pool, min(k, len(pool)))


def sample_recipe(
    schema: Schema,
    seed: int,
    scenario_id: str,
    names: dict | None = None,
    descriptions: dict | None = None,
    distractors: dict | None = None,
    overrides: dict | None = None,
    condition: str = "canonical",
    temporal_form: str | None = None,
) -> Recipe:
    """Canonical recipe: full linear scene path, one referent per person role
    (PATRON occasionally two), controls sampled. ``len_band`` is sampled *conditioned on*
    ``detail_level`` (soft coupling, see ``_LEN_BY_DETAIL``): more detail allows a longer
    passage, so the contradictory corners (terse+long, rich+short) never occur.

    ``overrides`` pins specific control knobs (any of ``prop_naming`` / ``detail_level`` /
    ``len_band`` / ``distractor_level``) instead of sampling them; dependent fields are
    recomputed to stay consistent -- ``prop_naming_map`` from ``prop_naming``, ``distractors``
    from ``distractor_level``, and ``len_band`` from a pinned ``detail_level`` (unless
    ``len_band`` is itself pinned). Everything else (referents, un-pinned knobs) stays
    randomized. Used to build balanced control grids -- e.g. fixed prop_naming while scanning
    detail x distractor."""
    rng = random.Random(seed)
    names = names or load_names()
    descriptions = descriptions or load_descriptions()
    distractors = distractors or load_distractors()

    prop_naming = rng.choice(PROP_NAMING)
    distractor_level = rng.choice(DISTRACTOR_LEVELS)
    detail_level = rng.choice(DETAIL_LEVELS)
    controls = {
        "prop_naming": prop_naming,
        "prop_naming_map": _prop_map(schema, prop_naming, rng),
        "role_naming": "latent",  # default; --role-naming named = positive control
        "detail_level": detail_level,
        "len_band": rng.choice(_LEN_BY_DETAIL[detail_level]),
        "distractor_level": distractor_level,
        "distractors": _sample_distractors(distractors, distractor_level, rng),
    }
    if overrides:
        for key in ("prop_naming", "role_naming", "detail_level", "len_band", "distractor_level"):
            if key in overrides:
                controls[key] = overrides[key]
        if "prop_naming" in overrides:
            controls["prop_naming_map"] = _prop_map(schema, controls["prop_naming"], rng)
        if "detail_level" in overrides and "len_band" not in overrides:
            controls["len_band"] = rng.choice(_LEN_BY_DETAIL[controls["detail_level"]])
        if "distractor_level" in overrides:
            controls["distractors"] = _sample_distractors(
                distractors, controls["distractor_level"], rng
            )

    gender_cycle = ["m", "f", "n"]
    rng.shuffle(gender_cycle)
    used: set[str] = set()
    roles: dict[str, list[Referent]] = {}
    counter = 0
    g = 0
    for role_id in schema.person_roles():
        # A role may appear as a small party (e.g. two PATRONs) when its max_count > 1.
        max_count = schema.roles[role_id].max_count
        count = rng.randint(1, max_count) if max_count > 1 else 1
        refs: list[Referent] = []
        for _ in range(count):
            counter += 1
            gender = gender_cycle[g % len(gender_cycle)]
            g += 1
            value = _pick_name(rng, names, gender, used) if rng.random() < 0.7 else None
            if value is None:
                value, gender = _pick_description(
                    rng, descriptions, gender, used,
                    adult_only=schema.roles[role_id].adult_only,
                )
                form = "description"
            else:
                form = "name"
            used.add(value.lower())
            refs.append(Referent(ref=f"e{counter}", form=form, value=value, gender=gender))
        roles[role_id] = refs

    recipe = Recipe(
        scenario_id=scenario_id,
        schema=schema.name,
        seed=seed,
        condition=condition,
        scene_path=list(schema.scenes),
        roles=roles,
        controls=controls,
    )
    if condition != "canonical":
        apply_violation(schema, recipe, condition, rng)
    if temporal_form is not None:
        apply_temporal_form(schema, recipe, temporal_form, rng)
    return recipe


def _derangement(order: list[str], rng: random.Random, tries: int = 20) -> list[str]:
    """A shuffled order with NO beat left in its canonical slot (a full derangement). Falls back
    to a single rotation (also fixed-point-free) if the random tries don't land one."""
    for _ in range(tries):
        shuffled = list(order)
        rng.shuffle(shuffled)
        if all(a != b for a, b in zip(order, shuffled, strict=True)):
            return shuffled
    return order[1:] + order[:1]


def _flashback_path(order: list[str], rng: random.Random) -> tuple[list[str], str]:
    """Open on a later beat, flash back through the beginning, then resume:
    ``[order[p]] + order[:p] + order[p+1:]``. Returns the path + the pivot scene (told first)."""
    p = rng.randint(2, len(order) - 2)
    return [order[p], *order[:p], *order[p + 1 :]], order[p]


def _displace_one(order: list[str], rng: random.Random) -> tuple[list[str], str]:
    """Forgetful narrator: pull one interior beat out of its slot and re-insert it LATER, so it
    reads late but happens early (nothing is missing). Returns the path + the displaced scene."""
    j = rng.randint(1, len(order) - 2)
    target = order[j]
    rest = order[:j] + order[j + 1 :]
    pos = rng.randint(j + 1, len(rest))  # strictly after its original slot
    return [*rest[:pos], target, *rest[pos:]], target


def _duplicate_one(order: list[str], rng: random.Random) -> tuple[list[str], str]:
    """Repeat events: render one interior beat a SECOND time later in the text (same story
    position, two textual positions). Returns the path + the repeated scene."""
    j = rng.randint(1, len(order) - 2)
    target = order[j]
    pos = rng.randint(j + 2, len(order))  # after the original occurrence
    path = list(order)
    path.insert(pos, target)
    return path, target


def apply_temporal_form(
    schema: Schema, recipe: Recipe, form: str, rng: random.Random
) -> Recipe:
    """Reorder (or repeat) a canonical recipe's ``scene_path`` per a temporal FORM and record
    ``recipe.temporal`` (form + canonical order + per-beat story_idx/text_ordinal + target scene).
    Story identity travels with the scene NAME; the new ``scene_path`` is the textual beat order,
    so story order and textual position are decorrelated. Controls / roles / ``condition`` are left
    untouched (identify gold is intact). ``canonical`` / unknown forms keep the natural order (the
    matched-note control still records ``temporal``)."""
    canonical = list(schema.scenes)
    story_idx = {s: i for i, s in enumerate(canonical)}
    target: str | None = None
    if form == "reversed":
        path = canonical[::-1]
    elif form == "shuffled":
        path = _derangement(canonical, rng)
    elif form == "blockswap":
        h = len(canonical) // 2
        path = canonical[h:] + canonical[:h]
    elif form == "flashback":
        path, target = _flashback_path(canonical, rng)
    elif form == "forgetful":
        path, target = _displace_one(canonical, rng)
    elif form == "repeat":
        path, target = _duplicate_one(canonical, rng)
    else:  # canonical / unknown -> natural order (matched-note control)
        path = list(canonical)
    recipe.scene_path = list(path)
    recipe.temporal = {
        "form": form,
        "canonical_order": canonical,
        "beats": [
            {"scene": s, "story_idx": story_idx[s], "text_ordinal": t}
            for t, s in enumerate(path)
        ],
        "target_scene": target,
    }
    return recipe


def _object_slots(schema: Schema, rid: str, scene: str) -> set[str]:
    """Object/place roles referenced by any of role ``rid``'s action templates at ``scene``."""
    obj = set(schema.object_roles())
    return {
        m
        for tmpl in schema.actions.get(rid, {}).get(scene, [])
        for m in _SLOT.findall(tmpl)
        if m in obj
    }


def apply_violation(schema: Schema, recipe: Recipe, condition: str, rng: random.Random) -> Recipe:
    """Turn a canonical recipe into a violation item in place (plan A5). Structural violations
    (wrong_order, missing_scene) edit ``scene_path`` / ``roles`` so the existing verifier checks
    them as-is; content violations (wrong_action, role_swap, prop_mismatch) record a spec that
    ``generator.realize_template`` injects into the text. ``recipe.violation`` records the anomaly
    (type + location) for the spot_error gold; ``recipe.condition`` is synced to what was actually
    applied. Degrades to wrong_order if the schema is too small for the requested type."""
    canonical = list(schema.scenes)
    present = [rid for rid in schema.person_roles() if rid in recipe.roles]
    staff = [rid for rid in present if rid != PROTAGONIST]
    path = list(recipe.scene_path)

    def commit(vio: dict) -> Recipe:
        recipe.violation = vio
        recipe.condition = vio["type"]
        return recipe

    def swap_adjacent() -> Recipe:
        i = rng.randint(0, len(path) - 2)
        path[i], path[i + 1] = path[i + 1], path[i]
        recipe.scene_path = path
        return commit({"type": "wrong_order", "scenes": [path[i], path[i + 1]],
                       "canonical_order": canonical})

    if condition == "wrong_order" and len(path) >= 2:
        return swap_adjacent()

    if condition == "missing_scene" and len(path) >= 3:
        drop = rng.choice(path[1:-1])  # an interior step
        path.remove(drop)
        orphaned = [
            rid for rid in staff
            if set(schema.actions.get(rid, {})) and not set(schema.actions[rid]) & set(path)
        ]
        for rid in orphaned:
            del recipe.roles[rid]
        recipe.scene_path = path
        return commit({"type": "missing_scene", "scene": drop, "dropped_roles": orphaned,
                       "canonical_order": canonical})

    if condition == "wrong_action" and staff:
        # Borrow a protagonist action at an interior scene (skip the weak arrive/leave edges), so
        # a staff member is seen doing the patron's core action (e.g. the waiter eats).
        patron_scenes = [s for s in path[1:-1] if s in schema.actions.get(PROTAGONIST, {})]
        if not patron_scenes:
            patron_scenes = [s for s in path if s in schema.actions.get(PROTAGONIST, {})]
        if patron_scenes:
            scene = rng.choice(patron_scenes)
            return commit({"type": "wrong_action", "scene": scene, "actor_role": rng.choice(staff),
                           "borrowed_role": PROTAGONIST, "borrowed_scene": scene})

    if condition == "role_swap" and staff:
        # Swap two roles' actors from some scene onward. Require BOTH roles to act before that
        # scene (so each referent still appears under its own name, keeping the item verifiable)
        # and at least one to act at/after it (so the swap is actually visible in the text).
        a_scenes = set(schema.actions.get(PROTAGONIST, {}))
        cands = [
            (b, path[k])
            for b in staff
            for k in range(1, len(path))
            if (a_scenes & set(path[:k]))
            and (set(schema.actions.get(b, {})) & set(path[:k]))
            and ((a_scenes | set(schema.actions.get(b, {}))) & set(path[k:]))
        ]
        if cands:
            role_b, from_scene = rng.choice(cands)
            return commit({"type": "role_swap", "roles": [PROTAGONIST, role_b],
                           "from_scene": from_scene})

    if condition == "prop_mismatch":
        obj_props = [o for o in schema.object_roles() if schema.props.get(o)]
        cands = [
            (s, rid, p)
            for s in path for rid in present if s in schema.actions.get(rid, {})
            for p in _object_slots(schema, rid, s) if p in obj_props
        ]
        if cands and len(obj_props) >= 2:
            scene, rid, prop = rng.choice(cands)
            wrong = rng.choice([o for o in obj_props if o != prop])
            return commit({"type": "prop_mismatch", "scene": scene, "role": rid, "prop": prop,
                           "wrong_prop": wrong})

    return swap_adjacent() if len(path) >= 2 else recipe
