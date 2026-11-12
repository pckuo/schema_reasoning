"""The extended solved lexicon -- a port of ``COUNTERFACTUAL_SCREEN`` cell 7.

``events.progress`` credits a (person, role) pair using only that person's own gold canonical
words, which is strict enough to lose about a third of the traces the notebook counts as solved.
This module ports the notebook's ``K-lexicon (extended, quote-free)`` definition instead:

* role words are **pooled across the corpus per role id**, so a word used for a PATRON anywhere
  counts as a PATRON word everywhere, plus a hand-list of terms the models invent (``EXT``);
* matches are **whole-word over subword tokens**, phrases before single words;
* a pair is credited when a role word falls within ``W`` tokens of the name, preferring a word
  *after* the name ("X is the Y") over one before ("the Y, X"), and only when the role matches
  the gold role id;
* name mentions that sit inside a **verbatim quotation of the passage** are skipped, so copying
  the scenario text does not count as solving it.

This lifts the intervention population from 11 to the notebook's own count.
"""

from __future__ import annotations

import collections
import json
import re

import numpy as np

from . import runframe

WINDOW = 15
QUOTE_WINDOW = 8
MAX_PHRASE = 6
BOUNDARY = ("\u0120", "\u010a", " ", "\n")

# terms the models use in their answers that the gold canonical lists miss
EXT = {
    "SERVICE_AGENT": ["gate agent", "gate agents", "wait staff", "waitstaff", "boarding agent",
                      "gate staff", "boarding staff", "cabin crew", "cabin staff",
                      "flight attendant", "flight attendants", "crew member"],
    "PATRON": ["visitor", "visitors", "client", "clients", "tourist", "tourists"],
    "FRONT_AGENT": ["greeter", "greeters", "receptionist", "receptionists", "front counter",
                    "front desk", "check-in agent", "check in agent", "check-in staff",
                    "check in staff", "counter staff", "desk staff", "ticket agent",
                    "ticketing agent", "concierge"],
    "GATEKEEPER": ["security", "inspector", "inspectors", "checkpoint", "screening",
                   "security staff", "security officer", "security guard", "security personnel"],
    "PRODUCER": ["kitchen staff", "line cook", "head chef", "sous chef", "baker", "bakers"],
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def norm(text):
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def build_lexicons(gold):
    """Pooled canonical role words, and the extended variant."""
    canonical = collections.defaultdict(set)
    for roles in gold.values():
        for q in roles:
            canonical[q["role_id"]].update(c.lower() for c in q.get("canonical", []) if c)
    plain = {r: set(words) for r, words in canonical.items()}
    extended = {r: set(words) | set(EXT.get(r, [])) for r, words in canonical.items()}
    return plain, extended


def word_spans(raw, low, word, maxlen=MAX_PHRASE):
    """End indices of whole-word matches of ``word`` over subword tokens."""
    target = norm(word)
    n = len(low)
    phrase = " " in word or "-" in word
    out = []

    def starts(i):
        return (i == 0 or raw[i].startswith(BOUNDARY) or not raw[i][:1].isalnum()
                or not raw[i - 1][-1:].isalnum())

    def ends(e):
        return e + 1 >= n or raw[e + 1].startswith(BOUNDARY) or not raw[e + 1][:1].isalnum()

    for i in range(n):
        if not starts(i):
            continue
        acc = ""
        for offset in range(maxlen):
            if i + offset >= n:
                break
            if offset > 0 and not phrase and raw[i + offset].startswith(BOUNDARY):
                break
            acc += norm(low[i + offset])
            if acc == target:
                if ends(i + offset):
                    out.append(i + offset)
                break
            if not target.startswith(acc):
                break
    return out


def role_hits(low, raw, lexicon):
    """Token index -> (role_id, is_phrase); phrases win over single words they contain."""
    out, covered = {}, set()
    for role, words in lexicon.items():
        for word in words:
            if " " in word or "-" in word:
                for end in word_spans(raw, low, word):
                    out[end] = (role, True)
                    covered.update(range(max(0, end - 5), end + 1))
    for role, words in lexicon.items():
        for word in words:
            if " " in word or "-" in word:
                continue
            for end in word_spans(raw, low, word):
                if end not in covered and end not in out:
                    out[end] = (role, False)
    return out


def is_quote(tokens, end, passage_norm, window=QUOTE_WINDOW):
    """True when the neighbourhood of ``end`` is a verbatim stretch of the source passage."""
    stretch = norm("".join(tokens[max(0, end - window):end + window + 1]))
    return len(stretch) >= (20 if window >= 8 else 12) and stretch in passage_norm


def assignment_state(tokens, raw, roles, lexicon, passage_norm, quote_free=True):
    """Latest-commitment role assignment per gold name, plus the monotone confirmed-pair count.

    Ports the notebook's ``C`` (pairs confirmed) and ``K`` (every gold name currently assigned to
    its correct role). ``K`` is an offline, reproducible answer-accuracy measure: it reads the
    model's *final* commitment rather than asking a judge, so it cannot drift between runs.
    """
    low = [t.lower() for t in tokens]
    n = len(tokens)
    named = [q for q in roles if q["value"][:1].isupper() and " " not in q["value"]]
    hits = role_hits(low, raw, lexicon)
    confirmed, assigned = {}, []
    for q in named:
        for end in runframe.find_spans_compat(tokens, q["value"]):
            if quote_free and is_quote(tokens, end, passage_norm):
                continue
            after = sorted((i - end, i) for i in hits if 0 < i - end <= WINDOW)
            before = sorted((end - i, i) for i in hits if 0 < end - i <= WINDOW)
            near = after or before
            if not near:
                continue
            index = near[0][1]
            role = hits[index][0]
            assigned.append((max(end, index), q["value"], role))
            if role == q["role_id"] and q["value"] not in confirmed:
                confirmed[q["value"]] = max(end, index)
    assigned.sort()
    need = {q["value"]: q["role_id"] for q in named}
    current = {}
    for _, name, role in assigned:
        current[name] = role
    counts = np.zeros(n, int)
    for end in confirmed.values():
        counts[end:] += 1
    correct = sum(1 for name, role in need.items() if current.get(name) == role)
    return dict(counts=counts, n_named=len(named), n_assigned=len(current),
                n_correct=correct,
                accuracy=(correct / len(need)) if need else float("nan"),
                all_correct=bool(need and correct == len(need)))


def confirmed_pairs(trace, roles, lexicon, passage_norm, quote_free=True):
    """Monotone count of gold (person, role) pairs stated, per token."""
    tokens, raw = trace.tokens, trace.raw
    low = [t.lower() for t in tokens]
    n = len(tokens)
    named = [q for q in roles if q["value"][:1].isupper() and " " not in q["value"]]
    hits = role_hits(low, raw, lexicon)
    confirmed = {}
    for q in named:
        for end in runframe.find_spans_compat(tokens, q["value"]):
            if quote_free and is_quote(tokens, end, passage_norm):
                continue
            after = sorted((i - end, i) for i in hits if 0 < i - end <= WINDOW)
            before = sorted((end - i, i) for i in hits if 0 < end - i <= WINDOW)
            near = after or before
            if not near:
                continue
            index = near[0][1]
            role = hits[index][0]
            if role == q["role_id"] and q["value"] not in confirmed:
                confirmed[q["value"]] = max(end, index)
    counts = np.zeros(n, int)
    for end in confirmed.values():
        counts[end:] += 1
    return counts, len(named)


def solved_table(built, model="phi4mr"):
    """Per-trace solved flag and ``T_solved`` under the extended quote-free lexicon."""
    responses = built["responses"]
    gold = {(r.item_id, r.model): json.loads(r.gold)["roles"] for r in responses.itertuples()}
    passages = {(r.item_id, r.model): norm(r.passage_text) for r in responses.itertuples()}
    judge = {(r.item_id, r.model): r.judge_role_accuracy for r in responses.itertuples()}
    _, extended = build_lexicons(gold)
    rows, marks = [], {}
    for (m, item), trace in built["traces"].items():
        if m != model:
            continue
        counts, n_named = confirmed_pairs(trace, gold[(item, m)], extended, passages[(item, m)])
        _, starts, durations = runframe.runs_of(trace.z)
        at_run = counts[starts + durations - 1]
        flag = at_run >= n_named if n_named else np.zeros(len(at_run), bool)
        hit = np.where(flag)[0]
        marks[(m, item)] = flag
        accuracy = float(judge[(item, m)]) if judge[(item, m)] == judge[(item, m)] else np.nan
        rows.append(dict(item=item, terminating=trace.terminating, n_runs=len(flag),
                         n_named=n_named, t_solved=int(hit[0]) if len(hit) else None,
                         solved=bool(len(hit)), judge_role_accuracy=accuracy,
                         all_right=bool(accuracy == accuracy and accuracy >= 0.999)))
    import pandas as pd
    return pd.DataFrame(rows), marks


def main():
    built = runframe.build()
    table, _ = solved_table(built)
    solved = table[table["solved"] & table["all_right"]]
    capped = solved[~solved["terminating"]]
    print(f"phi identify traces: {len(table)} | judge-all-right {int(table['all_right'].sum())}")
    print(f"  solved (extended, quote-free) AND judge-all-right: {len(solved)}")
    print(f"    terminating {int(solved['terminating'].sum())} | capped {len(capped)}")
    print("  COMPLETION_DEFINITION reference: 95 solved, 78 STOP, 17 capped")
    if len(capped):
        print(f"  capped: T_solved median {capped['t_solved'].median():.0f} "
              f"(notebook 63), runs after solving median "
              f"{(capped['n_runs'] - capped['t_solved']).median():.0f} (notebook 2650)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
