"""Cast-first stimulus model: people are primitive, roles are assigned onto them.

``pipeline.sampler`` is role-first -- it walks ``schema.person_roles()`` and draws a fresh referent
per role slot (``sampler.py:206-224``). That is why B, D, E and role multiplicity are inseparable
there, why nobody can hold two roles, and why no role can be empty.

Inverting it makes every quantity the spec needs independent:

===========  =========================================================
 Quantity     Definition here
===========  =========================================================
 ``E``        ``len(cast)`` -- headcount, primitive
 ``B``        ``sum(len(p.roles))`` -- bindings, NOT headcount
 ``R``        ``len({r for p in cast for r in p.roles})``
 ``D``        number of people with no roles
===========  =========================================================

so ``E == B + D`` holds only when nobody is multiplexed, exactly as the spec requires.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .config import ROLE_ALIASES, role_order, tier


@dataclass
class Person:
    """One named participant. ``roles`` empty == a distractor."""

    pid: str
    name: str
    gender: str  # m | f | n
    roles: list[str] = field(default_factory=list)  # role_ids, in schema order
    distractor_flavor: str | None = None  # outside_inventory | inside_unqueried
    distractor_key: str | None = None  # pool entry key, e.g. "courier"
    licence: str | None = None  # explicit non-role reason to be present
    beat: str | None = None  # concrete action, so a distractor is a participant
    scene: str | None = None  # where a distractor's beat sits

    @property
    def is_distractor(self) -> bool:
        return not self.roles

    @property
    def n_roles(self) -> int:
        return len(self.roles)


@dataclass
class BindingSpec:
    """One planned stimulus: a cast, a scene path, and its lineage.

    ``family_id`` / ``parent_id`` / ``edit_op`` carry the seed-family structure the spec's section 5
    rules 1-2 require -- every sibling is derived from one seed by a single scripted cast edit, so
    ``(1|family)`` is available downstream and the lineage is auditable.
    """

    item_id: str
    schema: str
    experiment: str  # 1A | 1B | 1C
    condition: str  # A07 | B13 | C02 ...
    family_id: str
    parent_id: str | None
    edit_op: str  # seed | insert_bystander | promote | merge_roles
    seed: int
    cast: list[Person]
    scene_path: list[str]
    controls: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- derived quantities

    @property
    def E(self) -> int:
        """Total cast (headcount). Primitive -- never derived from B + D."""
        return len(self.cast)

    @property
    def B(self) -> int:
        """Bindings: role assignments that will be queried. A person with 2 roles contributes 2."""
        return sum(p.n_roles for p in self.cast)

    @property
    def R(self) -> int:
        """Distinct occupied roles."""
        return len(self.occupied_roles)

    @property
    def D(self) -> int:
        """Distractors: participants holding no queried role."""
        return sum(1 for p in self.cast if p.is_distractor)

    @property
    def occupied_roles(self) -> list[str]:
        seen = {r for p in self.cast for r in p.roles}
        return [rid for rid in role_order(self.schema) if rid in seen]

    @property
    def role_counts(self) -> dict[str, int]:
        """People per role, including explicit zeros -- the role-count vector as a mapping."""
        counts = dict.fromkeys(role_order(self.schema), 0)
        for person in self.cast:
            for rid in person.roles:
                counts[rid] += 1
        return counts

    @property
    def vector(self) -> tuple[int, ...]:
        """The role-count vector in canonical schema order, e.g. restaurant ``(2, 1, 2, 1)``."""
        counts = self.role_counts
        return tuple(counts[rid] for rid in role_order(self.schema))

    @property
    def omitted_roles(self) -> list[str]:
        return [rid for rid, n in self.role_counts.items() if n == 0]

    @property
    def merged_people(self) -> list[Person]:
        return [p for p in self.cast if p.n_roles > 1]

    @property
    def max_role_multiplicity(self) -> int:
        """The largest number of people sharing one role.

        Logged as a first-class factor because at fixed R, raising B necessarily raises it -- so
        1B's B partly indexes "how many people share a role" rather than binding load alone.
        """
        counts = [n for n in self.role_counts.values() if n]
        return max(counts) if counts else 0

    @property
    def multiplex_load(self) -> int:
        """M: extra roles beyond the first, summed over the cast. The dual of the above.

        E = (B - M) + D is an identity, so at most two of {B, D, E} can be pinned at once. 1C pins
        B and E and lets D float; 1D pins B and D and lets E float. M is what both are actually
        manipulating, which is why it is logged rather than reconstructed downstream.
        """
        return sum(p.n_roles - 1 for p in self.cast if p.roles)

    def person(self, pid: str) -> Person:
        for p in self.cast:
            if p.pid == pid:
                return p
        raise KeyError(f"{pid!r} not in cast of {self.item_id!r}")

    def holders(self, role_id: str) -> list[Person]:
        return [p for p in self.cast if role_id in p.roles]

    def bindings(self) -> list[dict]:
        """One row per queried binding (spec section 6: role, multiplicity, tier)."""
        counts = self.role_counts
        return [
            {
                "pid": p.pid,
                "name": p.name,
                "role_id": rid,
                "role": ROLE_ALIASES[self.schema][rid],
                "role_multiplicity": counts[rid],
                "tier": tier(self.schema, rid),
                "n_roles_this_person": p.n_roles,
            }
            for p in self.cast
            for rid in p.roles
        ]

    def names(self) -> list[str]:
        return [p.name for p in self.cast]

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "schema": self.schema,
            "experiment": self.experiment,
            "condition": self.condition,
            "family_id": self.family_id,
            "parent_id": self.parent_id,
            "edit_op": self.edit_op,
            "seed": self.seed,
            "cast": [asdict(p) for p in self.cast],
            "scene_path": list(self.scene_path),
            "controls": dict(self.controls),
        }

    @classmethod
    def from_dict(cls, data: dict) -> BindingSpec:
        return cls(
            item_id=data["item_id"],
            schema=data["schema"],
            experiment=data["experiment"],
            condition=data["condition"],
            family_id=data["family_id"],
            parent_id=data.get("parent_id"),
            edit_op=data["edit_op"],
            seed=data["seed"],
            cast=[Person(**p) for p in data["cast"]],
            scene_path=list(data["scene_path"]),
            controls=dict(data.get("controls", {})),
        )

    def copy(self, **changes) -> BindingSpec:
        """A deep-enough copy (cast people are cloned) with field overrides."""
        base = {
            "item_id": self.item_id,
            "schema": self.schema,
            "experiment": self.experiment,
            "condition": self.condition,
            "family_id": self.family_id,
            "parent_id": self.parent_id,
            "edit_op": self.edit_op,
            "seed": self.seed,
            "cast": [Person(**asdict(p)) for p in self.cast],
            "scene_path": list(self.scene_path),
            "controls": dict(self.controls),
        }
        base.update(changes)
        return BindingSpec(**base)
