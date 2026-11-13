"""Role-count vector enumeration and contrasting-shape sampling.

A vector is the per-role headcount in canonical schema order, e.g. restaurant ``(2, 1, 2, 1)``
means 2 patrons, 1 host, 2 waiters, 1 cook -> B = 6, R = 4.

Occupancy rules (spec "Constraints"):
  * restaurant -- every role >= 1, so R = 4 always;
  * airport    -- patron >= 1 and at most ONE of check-in/security/gate/pilot may be 0,
                  so R = 5, or R = 4 with one role omitted.

Enumerating under :data:`config.ROLE_CAPS` reproduces the spec's availability table exactly
(restaurant R=4: 1/4/10/17/22/22/17 for B=4..10; airport R=4: 4/16/34/50/54/44/26), which is the
check that the cap interpretation is right.
"""

from __future__ import annotations

import random
from itertools import product

from .config import MAX_OMITTED, ROLE_CAPS, role_order

Vector = tuple[int, ...]


def enumerate_vectors(schema: str, b: int, r: int) -> list[Vector]:
    """Every legal vector for ``schema`` with exactly ``b`` bindings across exactly ``r`` roles.

    Deterministic order (lexicographic by construction), so downstream sampling is reproducible.
    """
    caps = ROLE_CAPS[schema]
    max_omitted = MAX_OMITTED[schema]
    out: list[Vector] = []
    for combo in product(*[range(cap + 1) for _, cap in caps]):
        if sum(combo) != b:
            continue
        if combo[0] < 1:  # the protagonist role is never empty
            continue
        n_zero = sum(1 for x in combo if x == 0)
        if n_zero > max_omitted:
            continue
        if len(caps) - n_zero != r:
            continue
        out.append(combo)
    return out


def omitted_role(schema: str, vector: Vector) -> str | None:
    """The role id left empty by ``vector`` (airport R=4 conditions), else ``None``."""
    return next((rid for rid, n in zip(role_order(schema), vector, strict=True) if n == 0), None)


def spread(vector: Vector) -> float:
    """How evenly the bindings are spread across occupied roles: 0 = perfectly even, higher =
    more concentrated. Used to pick CONTRASTING shapes so ``vector`` does not alias ``condition``
    (spec section 3): concentrated ``(4,1,2,1)`` vs spread ``(2,2,2,2)``."""
    occupied = [n for n in vector if n]
    if not occupied:
        return 0.0
    mean = sum(occupied) / len(occupied)
    return (sum((n - mean) ** 2 for n in occupied) / len(occupied)) ** 0.5


def pick_contrasting(
    vectors: list[Vector], k: int, rng: random.Random, omission_balance: bool = False
) -> list[Vector]:
    """Up to ``k`` vectors chosen to span the shape range rather than cluster.

    Takes the most concentrated and the most spread shapes first, then fills from the middle --
    so a condition's items deliberately differ in shape and ``vector`` cannot alias ``condition``.
    With ``omission_balance`` (airport R=4), one vector per omitted role is taken first, because
    the grid naturally skews toward gate/pilot omissions -- their caps are lower, so omitting them
    leaves fewer ways to reach a given B (spec section 2, "balance the four omission types").
    """
    if not vectors:
        return []
    pool = sorted(vectors, key=lambda v: (spread(v), v))
    chosen: list[Vector] = []
    if omission_balance:
        by_omission: dict[str | None, list[Vector]] = {}
        for vec in pool:
            by_omission.setdefault(_omitted_index(vec), []).append(vec)
        for key in sorted(by_omission, key=lambda x: (x is None, x)):
            chosen.append(rng.choice(by_omission[key]))
            if len(chosen) == k:
                return chosen
    remaining = [v for v in pool if v not in chosen]
    while remaining and len(chosen) < k:
        chosen.append(remaining.pop(-1) if len(chosen) % 2 else remaining.pop(0))
    return chosen[:k]


def _omitted_index(vector: Vector) -> int | None:
    return next((i for i, n in enumerate(vector) if n == 0), None)


def vector_to_counts(schema: str, vector: Vector) -> dict[str, int]:
    return dict(zip(role_order(schema), vector, strict=True))


def counts_to_vector(schema: str, counts: dict[str, int]) -> Vector:
    return tuple(counts.get(rid, 0) for rid in role_order(schema))
