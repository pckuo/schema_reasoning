"""Deterministic verification of a generated binding stimulus.

Covers what ``pipeline.verifier`` covers (forbidden lexicon, referent anchors, scene set and
order) plus the three things it structurally cannot:

* ``role_map`` is ``name -> list[role_id]``. ``pipeline/verifier.py:84`` writes
  ``role_map[ref.value] = role_id``, keyed on the referent string, so a person holding two roles
  silently loses the first one -- which is exactly the 1C case.
* **cast identity against the parent**, so an edit that quietly added or dropped somebody is
  caught rather than logged as a clean sibling.
* **stray capitalised names**, the only automatic guard against invented people. The generator is
  told "introduce NO other named people"; nothing in ``pipeline`` ever checks it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..core.spec import Schema, forbidden_set, named_diagnostic_props
from .cast import BindingSpec
from .prose import role_word

# Capitalised tokens that are ordinary words even MID-sentence (months, weekdays, titles). The
# sentence-initial case is handled positionally instead of by a wordlist -- see `_stray_names`.
_STOPWORDS = frozenset(
    {
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
        "Mr",
        "Mrs",
        "Ms",
        "Dr",
        "Sir",
        "Madam",
        "God",
        "English",
        "French",
        "Spanish",
        "Italian",
        "German",
        "Chinese",
        "Japanese",
        "Greek",
    }
)
_CAP_TOKEN = re.compile(r"\b[A-Z][a-z]{2,}\b")
_SENTENCE_END = frozenset(".!?:;\"'”’)]")


def _sentence_initial(text: str, start: int) -> bool:
    """Whether the token at ``start`` opens a sentence.

    A wordlist cannot enumerate every capitalised common noun ("Steam clouded his glasses"), but
    position can: an invented person name is used mid-sentence at least once, whereas a
    sentence-initial common noun never is.
    """
    i = start - 1
    while i >= 0 and text[i] in " \t":
        i -= 1
    return i < 0 or text[i] in _SENTENCE_END or text[i] == "\n"


def _stray_names(passage: str, known: set[str]) -> list[str]:
    """Capitalised tokens that are not cast members and appear at least once mid-sentence.

    The only automatic guard against invented people. The generator is told "introduce NO other
    named people"; nothing in ``pipeline`` ever checks it.
    """
    seen: dict[str, bool] = {}
    for match in _CAP_TOKEN.finditer(passage):
        token = match.group(0)
        if token in known or token in _STOPWORDS:
            continue
        mid = not _sentence_initial(passage, match.start())
        seen[token] = seen.get(token, False) or mid
    return sorted(tok for tok, mid in seen.items() if mid)


@dataclass
class VerifyResult:
    ok: bool
    reasons: list[str] = field(default_factory=list)
    role_map: dict[str, list[str]] = field(default_factory=dict)
    referent_spans: dict[str, list[dict]] = field(default_factory=dict)
    scene_spans: list[dict] = field(default_factory=list)
    stray_names: list[str] = field(default_factory=list)
    n_words: int = 0
    n_chars: int = 0
    # Realized values of the two naming factors, so a silent revert shows up in the DATA rather
    # than having to be inferred from the passage after the fact.
    diagnostic_props: list[str] = field(default_factory=list)
    labelled_roles: dict[str, list[str]] = field(default_factory=dict)


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _labelled_roles(passage: str, schema: Schema, spec: BindingSpec) -> dict[str, list[str]]:
    """For each role-bearer, which of their roles are named IN THE SAME SENTENCE as their name.

    The enforceable half of the ``role_naming="named"`` contract. Merely finding "waiter" somewhere
    in the passage does not make the item a positive control -- the word has to be attached to the
    right person, which is the whole thing the probed model is asked to report. Same-sentence
    co-occurrence is a deliberately loose proxy for that attachment: it will accept an appositive,
    a copula or a trailing clause, and it is only ever used to catch a generator that ignored the
    instruction outright.
    """
    sentences = _SENTENCE_SPLIT.split(passage)
    out: dict[str, list[str]] = {}
    for person in spec.cast:
        if person.is_distractor:
            continue
        hit: list[str] = []
        for rid in person.roles:
            word = role_word(schema, rid)
            pattern = rf"\b{re.escape(word)}s?\b"
            if any(
                re.search(rf"\b{re.escape(person.name)}\b", s) and re.search(pattern, s, re.I)
                for s in sentences
            ):
                hit.append(rid)
        out[person.name] = hit
    return out


def _find(text: str, name: str) -> list[dict]:
    """Every span where a given name appears, as a whole word, case-sensitively.

    Case matters here in a way it does not in ``pipeline``: role-bearers are plain given names, so
    a case-insensitive match would let a sentence-initial common word masquerade as a person.
    """
    return [
        {"char_start": m.start(), "char_end": m.end(), "text": m.group(0)}
        for m in re.finditer(rf"\b{re.escape(name)}\b", text)
    ]


def _quoted(text: str, span: dict) -> bool:
    i, j = span["char_start"], span["char_end"]
    return i > 0 and j < len(text) and text[i - 1] == '"' and text[j] == '"'


def verify(
    passage: str,
    labels: dict,
    schema: Schema,
    spec: BindingSpec,
    prop_naming: str,
    role_naming: str,
    parent: BindingSpec | None = None,
) -> VerifyResult:
    reasons: list[str] = []

    forbidden = forbidden_set(schema, prop_naming, role_naming)
    hits = sorted(w for w in forbidden if re.search(rf"\b{re.escape(w)}\b", passage, re.I))
    if hits:
        reasons.append("forbidden words: " + ", ".join(hits))

    # Both naming factors are enforced HERE, in the same place the forbidden lexicon is, because
    # the two arms fail in opposite directions and only one of them was ever caught before. An
    # `abstract` violation is a word that got USED, which the forbidden sweep above already
    # catches. A `named` violation is a word that was OMITTED -- invisible to any ban-list, and
    # precisely how v1 reverted to paraphrase without a single item failing verification.
    diagnostic = named_diagnostic_props(passage, schema)
    if prop_naming == "named" and not diagnostic:
        reasons.append(
            "no diagnostic prop: passage names none of " + ", ".join(schema.diagnostic_props)
        )

    labelled = _labelled_roles(passage, schema, spec) if role_naming == "named" else {}
    if role_naming == "named":
        missing = [
            f"{person.name}:{rid}"
            for person in spec.cast
            if not person.is_distractor
            for rid in person.roles
            if rid not in labelled.get(person.name, [])
        ]
        if missing:
            reasons.append("role not named beside its bearer: " + ", ".join(missing))

    role_map: dict[str, list[str]] = {}
    spans: dict[str, list[dict]] = {}
    for person in spec.cast:
        found = _find(passage, person.name)
        if not found:
            reasons.append(f"missing person: {person.name!r}")
            continue
        spans[person.name] = found
        role_map[person.name] = list(person.roles)
        if any(_quoted(passage, s) for s in found):
            reasons.append(f"quoted name: {person.name!r}")

    scene_spans: list[dict] = []
    cursor = 0
    for entry in labels.get("scenes", []) if isinstance(labels, dict) else []:
        scene, text = entry.get("scene"), entry.get("text", "")
        idx = passage.find(text, cursor) if text else -1
        if idx == -1 and text:
            idx = passage.find(text)
        if idx == -1:
            reasons.append(f"scene text not found: {scene!r}")
            continue
        # Scene labels are an internal key, not prose: models sometimes title-case them
        # ("Arrive"), and rejecting a perfectly good passage over label casing is a verifier bug,
        # not a stimulus fault. Normalise before comparing; keep the raw label for the record.
        scene_spans.append(
            {
                "scene": str(scene).strip().lower().replace(" ", "_"),
                "scene_as_labelled": scene,
                "char_start": idx,
                "char_end": idx + len(text),
            }
        )
        cursor = idx + len(text)
    ordered = [d["scene"] for d in sorted(scene_spans, key=lambda d: d["char_start"])]
    if ordered != [s.lower() for s in spec.scene_path]:
        reasons.append(f"scene set/order mismatch: {ordered} != {spec.scene_path}")

    known = {p.name for p in spec.cast}
    stray = _stray_names(passage, known)
    if stray:
        reasons.append("possible invented people: " + ", ".join(stray))

    if parent is not None:
        reasons += _cast_identity(spec, parent)

    return VerifyResult(
        ok=not reasons,
        reasons=reasons,
        role_map=role_map,
        referent_spans=spans,
        scene_spans=scene_spans,
        stray_names=stray,
        n_words=len(passage.split()),
        n_chars=len(passage),
        diagnostic_props=diagnostic,
        labelled_roles=labelled,
    )


def _cast_identity(child: BindingSpec, parent: BindingSpec) -> list[str]:
    """Check the cast delta matches the declared edit. Structural, not textual -- this catches a
    planning bug (or a bad retry) before any prose is judged."""
    before, after = set(parent.names()), set(child.names())
    op = child.edit_op
    if op == "insert_bystander":
        if not before <= after:
            return [f"insert dropped: {sorted(before - after)}"]
        if child.B != parent.B or child.vector != parent.vector:
            return [f"insert moved B: {parent.vector} -> {child.vector}"]
    elif op == "promote":
        if before != after:
            return [
                f"promote changed the cast: +{sorted(after - before)} -{sorted(before - after)}"
            ]
        if child.E != parent.E:
            return [f"promote changed E: {parent.E} -> {child.E}"]
    elif op == "merge_roles":
        donors = before - after
        if (child.B, child.R) != (parent.B, parent.R):
            return [f"merge moved (B,R): {(parent.B, parent.R)} -> {(child.B, child.R)}"]
        # Only two of {B, D, E} can be pinned at once, so the two merge arms differ in which one
        # absorbs the merge: 1C backfills a bystander per donor (E pinned, D rises), 1D does not
        # (D pinned, E falls). Asserting 1C's invariant on a 1D child flags every correctly-built
        # item. The arm is read from the recorded flag rather than the cast delta, because an
        # unpadded merge that wrongly gained a bystander looks exactly like a legal padded one.
        if child.controls.get("merge_backfill", True):
            if (child.E, child.D) != (parent.E, parent.D + len(donors)):
                return [
                    f"padded merge moved (E,D): {(parent.E, parent.D)} -> "
                    f"{(child.E, child.D)}, expected {(parent.E, parent.D + len(donors))}"
                ]
        elif (child.E, child.D) != (parent.E - len(donors), parent.D):
            return [
                f"unpadded merge moved (E,D): {(parent.E, parent.D)} -> "
                f"{(child.E, child.D)}, expected {(parent.E - len(donors), parent.D)}"
            ]
    return []


def length_ok(n_words: int, band: tuple[int, int], slack: float = 0.35) -> bool:
    """Whether a passage is close enough to the family's shared word band.

    Generous on purpose: the point is to catch drift (a D=5 sibling twice the length of its D=0
    seed), not to police style."""
    lo, hi = band
    return lo * (1 - slack) <= n_words <= hi * (1 + slack)


_SENTENCE = re.compile(r"[^.!?]+[.!?]+")

# Preservation is MEASURED and LOGGED on every sibling, but is not an acceptance gate by default.
# Siblings are allowed to differ in wording; what has to hold is the STRUCTURE (cast identity, B,
# R, E), and that is checked separately and exactly by `_cast_identity`. Keeping the number around
# still earns its place: it is the QA signal for "did this edit turn into a fresh retelling", and
# it is available downstream as a per-item covariate.
#
# Pass `--min-preservation` to `run.py` to turn it back into a floor. These are the per-edit
# weights that floor is scaled by: a merge legitimately disturbs more text than an insert, because
# it removes a person outright and re-attributes their actions, whereas an insert only adds.
PRESERVATION_WEIGHT = {"insert_bystander": 1.0, "promote": 0.8, "merge_roles": 0.6}


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.findall(text.replace("\n", " ")) if s.strip()]


def preservation(parent_passage: str, child_passage: str) -> float:
    """Fraction of the parent's sentences reproduced verbatim in the child.

    A more honest measure of "is this an edit or a retelling" than a character-level diff ratio,
    which stays high for a fluent paraphrase that shares no actual sentence.
    """
    parent = sentences(parent_passage)
    if not parent:
        return 1.0
    child = set(sentences(child_passage))
    return sum(1 for s in parent if s in child) / len(parent)


def preservation_ok(edit_op: str, score: float, floor: float | None) -> bool:
    """Whether ``score`` clears the (optional) floor for ``edit_op``. ``floor=None`` disables the
    check entirely, which is the default -- see :data:`PRESERVATION_WEIGHT`."""
    if floor is None:
        return True
    return score >= floor * PRESERVATION_WEIGHT.get(edit_op, 1.0)
