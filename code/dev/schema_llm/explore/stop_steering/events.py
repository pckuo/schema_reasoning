"""Solved-state marker for the STOP-steering experiment -- a port of ``COUNTERFACTUAL_SCREEN``
cell 3 ``progress()`` and the ``COMPLETION_DEFINITION`` solved rule.

``T_solved`` is the first run by which **every named gold (person, role) pair has been stated**:
the person's name appears, a canonical word for their role appears within ``W`` tokens of it, and
the pair is credited at the *later* of the two -- so the marker is causal, never crediting a pair
before both halves exist.

Only single-word capitalised gold values are counted, as the screen does; descriptive referents
("a woman with a long braid") are not names and are excluded from the denominator.

This uses the gold canonical lists only. ``COMPLETION_DEFINITION``'s headline 95/78/17 uses an
*extended* lexicon that also admits role words the models invent, so our solved counts are a
lower bound on its; ``main()`` reports both our number and the notebook's for comparison.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import runframe

WINDOW = 15
MAX_SPAN = 4


def require(condition, message):
    if not condition:
        raise ValueError(message)


def find_spans(tokens, value, maxlen=MAX_SPAN):
    """End indices of every non-overlapping token run whose text concatenates to ``value``."""
    target = value.lower()
    ends = []
    for i, first in enumerate(tokens):
        if not first or not target.startswith(first.lower()):
            continue
        acc = ""
        for offset in range(maxlen):
            if i + offset >= len(tokens):
                break
            acc += tokens[i + offset].lower()
            if acc == target:
                ends.append(i + offset)
                break
            if not target.startswith(acc):
                break
    out, last = [], -99
    for end in ends:
        if end > last:
            out.append(end)
            last = end
    return out


def progress(tokens, roles):
    """Cumulative named-person and settled-pair counts per token."""
    lowered = [t.lower() for t in tokens]
    named = [q for q in roles if q["value"][:1].isupper() and " " not in q["value"]]
    first_person, first_pair = [], []
    for q in named:
        spans = find_spans(tokens, q["value"])
        if not spans:
            continue
        first_person.append(spans[0])
        canonical = {c.lower() for c in q.get("canonical", []) if c}
        role_positions = {i for i, x in enumerate(lowered) if x in canonical}
        for c in canonical:
            if " " in c:
                role_positions.update(find_spans(tokens, c))
        hits = [e for e in spans
                if any(abs(e - r) <= WINDOW for r in role_positions)]
        if hits:
            first_pair.append(min(
                max(e, min(r for r in role_positions if abs(e - r) <= WINDOW)) for e in hits))
    persons = np.zeros(len(tokens), int)
    pairs = np.zeros(len(tokens), int)
    for e in first_person:
        persons[e:] += 1
    for e in first_pair:
        pairs[e:] += 1
    return persons, pairs, len(named)


def solved_runs(trace, roles):
    """Per-run solved flag and ``T_solved`` (run index), or ``None`` if never solved."""
    _, pairs, n_named = progress(trace.tokens, roles)
    _, starts, durations = runframe.runs_of(trace.z)
    ends = starts + durations - 1
    at_run = pairs[ends]
    if n_named == 0:
        return np.zeros(len(at_run), bool), None, 0
    flag = at_run >= n_named
    hit = np.where(flag)[0]
    return flag, (int(hit[0]) if len(hit) else None), n_named


def build(built=None, model="phi4mr"):
    built = built or runframe.build()
    responses = built["responses"]
    gold = {(r.item_id, r.model): json.loads(r.gold)["roles"]
            for r in responses.itertuples()}
    judge = {(r.item_id, r.model): r.judge_role_accuracy for r in responses.itertuples()}
    rows = []
    marks = {}
    for (m, item), trace in built["traces"].items():
        if m != model:
            continue
        flag, t_solved, n_named = solved_runs(trace, gold[(item, m)])
        marks[(m, item)] = flag
        accuracy = pd.to_numeric(pd.Series([judge[(item, m)]]), errors="coerce").iloc[0]
        rows.append(dict(item=item, terminating=trace.terminating, n_runs=len(flag),
                         n_named=n_named, t_solved=t_solved, solved=t_solved is not None,
                         judge_role_accuracy=float(accuracy) if accuracy == accuracy else np.nan,
                         all_right=bool(accuracy == accuracy and accuracy >= 0.999)))
    return pd.DataFrame(rows), marks


def per_person(trace, roles):
    """Mention runs and role for each gold named person, keyed by name."""
    _, starts, durations = runframe.runs_of(trace.z)
    out = {}
    for q in roles:
        value = q["value"]
        if not (value[:1].isupper() and " " not in value):
            continue
        spans = find_spans(trace.tokens, value)
        if not spans:
            continue
        out[value] = dict(role=q["role_id"],
                          mention_runs=np.unique(
                              np.searchsorted(starts, np.array(spans), side="right") - 1))
    return out, len(starts), durations


def schema_state(trace, roles):
    """Carry-forward active person and role per run, plus the enumeration/assignment phase."""
    people, n, durations = per_person(trace, roles)
    active = np.full(n, "", dtype=object)
    role = np.full(n, "", dtype=object)
    events = sorted((int(r), name) for name, d in people.items() for r in d["mention_runs"])
    current, j = None, 0
    for t in range(n):
        while j < len(events) and events[j][0] <= t:
            current = events[j][1]
            j += 1
        if current is not None:
            active[t] = current
            role[t] = people[current]["role"]

    _, starts, _ = runframe.runs_of(trace.z)
    ends = starts + durations - 1
    persons, pairs, n_named = progress(trace.tokens, roles)
    denominator = max(n_named, 1)
    cp = persons[ends] / denominator
    cq = pairs[ends] / denominator
    phase = np.where(cp < 1, 0, np.where(cq < 1, 1, 2))
    return dict(active=active, role=role, phase=phase, n=n)


def switch_masks(state):
    """Pre-switch runs: the *next* run begins a first-ever visit to a different role.

    Mirrors the screen's ``is_new = (event[t+1] == NEW) & (role[t+1] != role[t])``. The negative
    set excludes anything within three runs of a switch, so the contrast is not contaminated by
    the switch's own neighbourhood.
    """
    active, role, n = state["active"], state["role"], state["n"]
    is_new_event = np.zeros(n, bool)
    visited: set[str] = set()
    previous = None
    for t in range(n):
        entity = active[t]
        if entity and entity != previous:
            is_new_event[t] = entity not in visited
            visited.add(entity)
        previous = entity if entity else previous
    pre_switch = np.zeros(n, bool)
    for t in range(n - 1):
        if is_new_event[t + 1] and role[t + 1] != role[t]:
            pre_switch[t] = True
    near = np.zeros(n, bool)
    for t in np.where(pre_switch)[0]:
        near[max(0, t - 3):t + 4] = True
    return pre_switch, ~near


def build_masks(built, model="phi4mr"):
    """Per-run ``pre_switch`` / ``solved`` / ``phase`` masks, aligned with the seg frame."""
    responses = built["responses"]
    gold = {(r.item_id, r.model): json.loads(r.gold)["roles"] for r in responses.itertuples()}
    frames = []
    for item in built["items"]:
        trace = built["traces"][(model, item)]
        roles = gold[(item, model)]
        state = schema_state(trace, roles)
        pre_switch, away = switch_masks(state)
        solved, _, _ = solved_runs(trace, roles)
        n = state["n"]
        require(len(solved) == n and len(pre_switch) == n, f"{item}: mask length mismatch")
        # the seg frame drops the final run, which has no successor
        frames.append(pd.DataFrame(dict(
            item=item, t=np.arange(n - 1), pre_switch=pre_switch[:-1],
            away_from_switch=away[:-1], solved=solved[:-1], phase=state["phase"][:-1])))
    return pd.concat(frames, ignore_index=True)


def main():
    built = runframe.build()
    table, _ = build(built)
    solved = table[table["solved"] & table["all_right"]]
    print(f"phi identify traces: {len(table)}")
    print(f"  judge-all-right: {int(table['all_right'].sum())}")
    print(f"  solved (gold-canonical lexicon) AND judge-all-right: {len(solved)}")
    print(f"    terminating: {int(solved['terminating'].sum())} | "
          f"capped: {int((~solved['terminating']).sum())}")
    print("  COMPLETION_DEFINITION (extended lexicon): 95 solved, 78 STOP, 17 capped")
    capped = solved[~solved["terminating"]]
    if len(capped):
        print(f"  capped solved traces: T_solved median {capped['t_solved'].median():.0f}, "
              f"runs after solving median "
              f"{(capped['n_runs'] - capped['t_solved']).median():.0f}")
    out = Path("results/schema_llm/analysis/stop_steering/20260919-stage1-gate")
    if out.exists():
        table.to_parquet(out / "solved.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
