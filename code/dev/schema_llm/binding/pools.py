"""Load the binding-experiment pools and allocate names without collisions."""

from __future__ import annotations

import random
from dataclasses import dataclass
from functools import lru_cache

import yaml

from .config import POOLS_DIR

GENDERS = ("m", "f", "n")
_GENDER_KEY = {"m": "male", "f": "female", "n": "neutral"}


@lru_cache(maxsize=1)
def load_names() -> dict[str, list[str]]:
    return yaml.safe_load((POOLS_DIR / "names.yaml").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def neutral_beats() -> list[str]:
    """Role-independent opening beats for v3 segment 1 (see pools/neutral_beats.yaml).

    v1 left segment 1's content to the generator, which staged people by role anyway and leaked
    role at .730. Assigning the beat removes the channel by construction.
    """
    data = yaml.safe_load((POOLS_DIR / "neutral_beats.yaml").read_text(encoding="utf-8"))
    return list(data["beats"])


@lru_cache(maxsize=1)
def load_distractor_people() -> dict[str, dict[str, list[dict]]]:
    return yaml.safe_load((POOLS_DIR / "distractor_people.yaml").read_text(encoding="utf-8"))


def all_names() -> list[str]:
    return [n for group in load_names().values() for n in group]


def distractor_pool(schema: str, flavor: str) -> list[dict]:
    return list(load_distractor_people()[schema][flavor])


@dataclass
class NameAllocator:
    """Hands out given names that are unique within a family.

    Uniqueness must hold across the WHOLE family, not just one item: 1A inserts a new bystander
    into an existing cast, so a name already spent on the seed must not come back for the
    bystander. Genders are cycled so co-present people stay pronoun-distinguishable.
    """

    rng: random.Random
    used: set[str] = None  # type: ignore[assignment]
    _gender_i: int = 0

    def __post_init__(self) -> None:
        if self.used is None:
            self.used = set()

    def take(self, gender: str | None = None) -> tuple[str, str]:
        """One unused ``(name, gender)``. ``gender`` None cycles m/f/n."""
        if gender is None:
            gender = GENDERS[self._gender_i % len(GENDERS)]
            self._gender_i += 1
        names = load_names()
        pool = [n for n in names[_GENDER_KEY[gender]] if n not in self.used]
        if not pool:  # exhausted that bucket -> fall back to any unused name
            pool = [n for n in all_names() if n not in self.used]
            if not pool:
                raise RuntimeError("name pool exhausted; enlarge binding/pools/names.yaml")
        name = self.rng.choice(sorted(pool))
        self.used.add(name)
        return name, gender

    def reserve(self, names: list[str]) -> None:
        self.used.update(names)


@dataclass
class DistractorAllocator:
    """Hands out distractor-people entries, unique within a family.

    1A's D ladder inserts up to 5 bystanders into one seed story, so the same courier must not be
    inserted twice; and identities are re-drawn per family so distractor identity varies across
    items (spec section 5 rule 7).

    ``avoid_scenes`` keeps bystanders away from the beat of an OMITTED role. A bystander standing
    at the very beat where nobody is attending to the script is the maximal temptation to bind
    them into it -- which would silently restore R and move B, the exact failure the omission
    licensing exists to prevent.
    """

    schema: str
    flavor: str
    rng: random.Random
    used: set[str] = None  # type: ignore[assignment]
    avoid_scenes: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if self.used is None:
            self.used = set()

    def take(self) -> dict:
        pool = [
            d
            for d in distractor_pool(self.schema, self.flavor)
            if d["key"] not in self.used and d["scene"] not in self.avoid_scenes
        ]
        if not pool:  # only avoided scenes left -- uniqueness matters more than placement
            pool = [
                d for d in distractor_pool(self.schema, self.flavor) if d["key"] not in self.used
            ]
        if not pool:
            raise RuntimeError(
                f"distractor pool exhausted for {self.schema}/{self.flavor}; "
                "enlarge binding/pools/distractor_people.yaml"
            )
        entry = self.rng.choice(pool)
        self.used.add(entry["key"])
        return dict(entry)
