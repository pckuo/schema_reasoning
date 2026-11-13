"""Confound-free cast assignment for binding-dynamics v3.

v1 built casts like this::

    for rid in role_order(schema):      # fixed canonical order
        name, gender = names.take()     # gender = GENDERS[i % 3], cycled by call index

which made a role's *index* determine its gender, hence its name bucket, and made the cast list the
role order. Gender, name and introduction order were therefore all deterministic functions of role:
pooled they predicted role at .976, and at **1.000** inside the restaurant schema.

This module replaces that with three properties, each of which the audit checks:

1. **(role x gender) balanced across the corpus**, while keeping a spread of genders *within* each
   item. Both halves matter: corpus balance is what kills the confound, and within-item spread is
   what keeps pronouns disambiguating. v1 got the second for free and the first not at all.
2. **One shared name pool.** Names are drawn without regard to role, and a name that has already
   been used for a role is actively deprioritised for it, so names recur across roles rather than
   becoming role-locked (audit A12).
3. **Randomised introduction order**, so position in the passage carries no role information.

The assignment is *greedy-balanced rather than random*: random assignment removes the bias but
leaves sampling noise in the cells, and at n=300 that noise is large enough to show up as residual
leakage. Greedy assignment against a running count drives the cells to near-exact balance, which is
both cleaner and easier to assert in a test.

Simulated before implementation (plan section 4.5): gender+order leakage falls from .778 (+.556 over
the null) to .203 (-.019), and to -.092 / +.003 within schema.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from pathlib import Path

from ..cast import Person
from ..config import role_order
from ..pools import _GENDER_KEY, GENDERS, load_names  # noqa: PLC2701


class BalancedAssigner:
    """Hands out (gender, name) for whole casts, balanced over the corpus.

    One instance per corpus build -- the balancing is a running property across items, so a
    per-item instance would achieve nothing.
    """

    #: permutations sampled per item when searching for the least-imbalanced assignment
    _TRIES = 32
    #: Names kept per gender bucket, per design. A design has only ~50 distinct CASTS (siblings
    #: share one), and the audit groups by family, so a name must appear in two different families
    #: to be testable at all. Drawing freely from all 362 names would spend most of them once and
    #: make audit test A2 vacuous -- not confounded (assignment is role-independent either way),
    #: just unauditable. The cap forces recurrence.
    #:
    #: The subset is SAMPLED per assigner rather than taken alphabetically, so D1 and D2 draw
    #: different names and the corpus as a whole still spans the full pool.
    POOL_CAP = 36

    def __init__(self, seed: int = 0, pool_cap: int | None = None) -> None:
        self.rng = random.Random(seed)
        self.pool_cap = self.POOL_CAP if pool_cap is None else pool_cap
        self._cell: Counter[tuple[str, str, str]] = Counter()  # (schema, role, gender) -> n
        self._name_roles: dict[str, set[str]] = defaultdict(set)  # name -> roles seen with
        self._name_use: Counter[str] = Counter()
        self._beat_cycle: dict[str, list] = {}  # role -> [shuffled beat order, cursor]
        # Running (role, roster position) counts and per-role name-length totals. Gender was the
        # only channel balanced against a running count; order was merely SHUFFLED and name length
        # was not controlled at all, and at n=600 rows a classifier finds what randomisation leaves
        # (union A3 +0.099, A8 +0.068 within restaurant). Both are now balanced the same way, and
        # `resume_from` carries them so a later wave CORRECTS an earlier one.
        self._pos: Counter[tuple[str, str, int]] = Counter()   # (schema, role, position) -> n
        self._len: dict[tuple[str, str], list[int]] = defaultdict(list)  # (schema, role) -> lengths
        self._items = 0
        self._buckets = {
            g: sorted(self.rng.sample(sorted(names), min(self.pool_cap, len(names))))
            for g, names in (
                (g, load_names()[_GENDER_KEY[g]]) for g in GENDERS
            )
        }

    @classmethod
    def resume_from(cls, run_dir, seed: int = 0) -> BalancedAssigner:
        """An assigner that CONTINUES an existing corpus instead of starting fresh.

        A second wave that draws a fresh name subset spends the combined corpus across ~180 names
        used twice each; continuing wave 1's subset spends it across the SAME ~108 used four times
        each. Measured on the shipped v3 D1 corpus, projecting the combined 300:

            fresh pool, no history : 181 names,  91.7% in >=2 roles, 2.32 roles/name
            reuse pool, no history : 108 names, 100.0% in >=2 roles, 3.22 roles/name
            reuse pool + history   : 108 names, 100.0% in >=2 roles, 4.04 roles/name

        The history matters because ``take_name`` deprioritises names already seen with the role
        being filled -- carrying it forward makes wave 2 actively fill each name's MISSING roles.
        Carrying ``_cell`` likewise balances gender over the union rather than within each wave.

        The pool is read from the corpus rather than reproduced from a seed on purpose: the shipped
        v3 corpora were built with a seed that is not recorded in their metadata, and
        ``BalancedAssigner(seed=20260914)`` overlaps them by only 36 of 108 names. Reading what was
        actually used is exact and cannot drift.
        """
        import json
        from collections import defaultdict

        self = cls(seed=seed)
        buckets: dict[str, set[str]] = defaultdict(set)
        for line in Path(run_dir).joinpath("items.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            for pos, person in enumerate(item.get("cast") or []):
                buckets[person["gender"]].add(person["name"])
                self._name_use[person["name"]] += 1
                roles = person.get("roles") or []
                if roles:
                    self._name_roles[person["name"]].add(roles[0])
                    self._cell[(item["schema"], roles[0], person["gender"])] += 1
                    # position and length too, so a later wave CORRECTS the earlier one's residue
                    # rather than merely not adding to it
                    self._pos[(item["schema"], roles[0], pos)] += 1
                    self._len[(item["schema"], roles[0])].append(len(person["name"]))
        if not buckets:
            raise ValueError(f"no cast found in {run_dir}/items.jsonl")
        self._buckets = {g: sorted(names) for g, names in buckets.items()}
        return self

    # ---------------------------------------------------------------- gender

    def assign_genders(self, schema: str, roles: list[str]) -> dict[str, str]:
        """Map each role to a gender for one item.

        The *multiset* of genders is fixed by cycling m/f/n, which guarantees within-item spread;
        only the assignment of that multiset **to roles** is free, and it is chosen to reduce the
        running (schema, role, gender) imbalance. So pronoun disambiguation is preserved exactly
        while the corpus-level confound is removed.
        """
        # Rotate WHERE the m/f/n cycle starts, per item. Without this the multiset for 4 roles is
        # always (m, f, n, m) -- male doubled -- so role->gender would be flat while the corpus
        # MARGINAL sat at 50/25/25. Rotating the start spreads the doubling evenly over genders, so
        # the marginal is balanced too, which is what a released benchmark needs.
        start = self._items % len(GENDERS)
        self._items += 1
        multiset = [GENDERS[(start + i) % len(GENDERS)] for i in range(len(roles))]
        best: tuple[int, list[str]] | None = None
        for _ in range(self._TRIES):
            perm = self.rng.sample(multiset, len(multiset))
            cost = sum(self._cell[(schema, r, g)] for r, g in zip(roles, perm, strict=True))
            if best is None or cost < best[0]:
                best = (cost, perm)
        assert best is not None
        chosen = dict(zip(roles, best[1], strict=True))
        for role, gender in chosen.items():
            self._cell[(schema, role, gender)] += 1
        return chosen

    # ------------------------------------------------------------------ name

    def _length_target(self, schema: str) -> float:
        """Corpus-wide mean name length so far, or the pool's mean before anything is assigned."""
        seen = [x for (s_, _r), v in self._len.items() if s_ == schema for x in v]
        if seen:
            return sum(seen) / len(seen)
        names = [n for b in self._buckets.values() for n in b]
        return sum(len(n) for n in names) / max(len(names), 1)

    def take_name(
        self, gender: str, role: str, schema: str = "", *, used: set[str] | None = None
    ) -> str:
        """A name for ``role``, from the shared pool, avoiding ``used`` (the family's names).

        Ordering deprioritises names already seen with this role, which is what stops the pools
        drifting back into role-specific sets; ties break on how often a name has been used at all,
        so the pool is consumed evenly.
        """
        used = used or set()
        pool = [n for n in self._buckets[gender] if n not in used]
        if not pool:  # bucket exhausted within this family -- fall back to any unused name
            pool = [n for group in load_names().values() for n in group if n not in used]
        if not pool:
            raise RuntimeError("name pool exhausted; enlarge binding/pools/names.yaml")
        # Ordered by: never-seen-with-this-role first (this is what drives name sharing to 100%),
        # then least-used, then whichever length pulls this role's running mean toward the corpus
        # mean, then alphabetical. Length is a TIE-BREAK, deliberately: it must never outrank the
        # role-novelty criterion, which is the one the name channel depends on.
        target = self._length_target(schema)
        cur = self._len[(schema, role)]
        mean_now = (sum(cur) / len(cur)) if cur else target

        def pull(n: str) -> float:
            after = (sum(cur) + len(n)) / (len(cur) + 1)
            return abs(after - target) - abs(mean_now - target)

        pool.sort(key=lambda n: (role in self._name_roles[n], self._name_use[n], pull(n), n))
        # sample among the equally-good front of the pool so the choice is not fully deterministic
        head = pool[: max(1, min(8, len(pool)))]
        name = self.rng.choice(head)
        self._name_roles[name].add(role)
        self._name_use[name] += 1
        self._len[(schema, role)].append(len(name))
        return name

    # ------------------------------------------------------------------ cast

    def build_cast(self, schema: str, counts: dict[str, int] | None = None) -> list[Person]:
        """One person per role slot, genders balanced, names shared, **order randomised**.

        The returned list order is the roster order handed to the generator, and the generator is
        separately told to introduce people in that order (see ``prose``), so introduction order in
        the passage carries no role information.
        """
        roles = role_order(schema)
        counts = counts or dict.fromkeys(roles, 1)
        genders = self.assign_genders(schema, roles)
        used: set[str] = set()
        people: list[Person] = []
        for role in roles:
            for _ in range(counts.get(role, 0)):
                gender = genders[role]
                name = self.take_name(gender, role, schema, used=used)
                used.add(name)
                people.append(Person(pid="", name=name, gender=gender, roles=[role]))
        # Roster position was previously a plain shuffle. Randomised is not the same as balanced:
        # over 600 restaurant rows the residue was detectable (A3 +0.099). Pick, among candidate
        # shuffles, the one that best flattens the running (schema, role, position) counts -- the
        # same greedy treatment gender already gets. Bystanders are included, since they occupy
        # roster positions too and so displace role-bearers.
        best = None
        for _ in range(self._TRIES):
            cand = people[:]
            self.rng.shuffle(cand)
            cost = sum(
                self._pos[(schema, p.roles[0], i)] for i, p in enumerate(cand) if p.roles
            )
            if best is None or cost < best[0]:
                best = (cost, cand)
        people = best[1]
        for i, person in enumerate(people):
            if person.roles:
                self._pos[(schema, person.roles[0], i)] += 1
        for i, person in enumerate(people, start=1):
            person.pid = f"p{i}"  # pid follows roster position, so it too is role-independent
        return people

    def assign_beats(self, people: list[Person], beats: list[str]) -> dict[str, str]:
        """One opening beat per person, EXACTLY balanced over (role, beat).

        Each role walks its own shuffled cycle through the whole beat pool, so over the corpus a
        role meets every beat equally often. Greedy sampling was not enough: with 30 beats over
        ~250 assignments the (role, beat) cells are thin, a beat lands with one role a little too
        often, and a bag-of-words classifier on segment 1 picks the coincidence up -- that put A5
        at +0.082, learning "sleeve"/"wrist"/"fingers", i.e. fragments of the beats themselves,
        rather than v1's real staging leak ("arrived"/"rain" vs "partition"/"beyond").

        Cycling is exact where sampling is merely unbiased, which is the same reason the gender
        assignment is greedy-balanced rather than random.
        """
        out: dict[str, str] = {}
        used: set[str] = set()
        for person in people:
            role = person.roles[0] if person.roles else "BYSTANDER"
            if role not in self._beat_cycle:
                order = list(beats)
                self.rng.shuffle(order)
                self._beat_cycle[role] = [order, 0]
            order, idx = self._beat_cycle[role]
            beat = order[idx % len(order)]
            for _ in range(len(order)):  # skip beats already spent in THIS item
                beat = order[idx % len(order)]
                idx += 1
                if beat not in used:
                    break
            self._beat_cycle[role][1] = idx
            used.add(beat)
            out[person.pid] = beat
        return out

    # ----------------------------------------------------------------- audit

    def balance_report(self) -> dict:
        """Realised (schema, role, gender) counts, for the unit test and the audit's A9."""
        by_schema: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(dict))
        for (schema, role, gender), n in self._cell.items():
            by_schema[schema][role][gender] = n
        worst = 1.0
        for roles in by_schema.values():
            for genders in roles.values():
                total = sum(genders.values())
                if not total:
                    continue
                uniform = total / len(GENDERS)
                worst = min(worst, min(genders.get(g, 0) for g in GENDERS) / uniform)
        locked = sum(1 for roles in self._name_roles.values() if len(roles) < 2)
        marginal: Counter[str] = Counter()
        for (_s, _r, gender), n in self._cell.items():
            marginal[gender] += n
        total = sum(marginal.values()) or 1
        return {
            "gender_marginal": {g: round(marginal[g] / total, 4) for g in GENDERS},
            "cells": {s: dict(r) for s, r in by_schema.items()},
            "worst_gender_cell_vs_uniform": round(worst, 4),
            "names_used": len(self._name_roles),
            "names_locked_to_one_role": locked,
        }
