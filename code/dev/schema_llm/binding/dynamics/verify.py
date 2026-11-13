"""Deterministic gates for the binding-dynamics batch (plan section 8.3, checks 1-7).

Everything here is what a lexicon or a regex CAN decide. The half that it cannot -- "is the part
inferable yet?" -- is the per-cut audit in :mod:`audit`, and neither is sufficient alone: the
role-before-schema pilot passed every check in this file and still failed its premise.

``binding.verify.verify`` is reused unchanged for the shared checks (forbidden lexicon, every name
present and unquoted, scene set and order, stray invented names). Its ``_cast_identity`` ignores
edit ops it does not know, which is correct for ``reschedule`` / ``resolve``: the cast is identical
across siblings by construction, so there is no delta to check.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ...core.spec import Schema, judge_accept_words, named_diagnostic_props
from ..cast import BindingSpec
from ..verify import VerifyResult
from ..verify import verify as base_verify
from .design import N_SEGMENTS

# A segment number that leaked out of the JSON list into the prose shifts every character offset
# downstream, so it is stripped before assembly rather than merely flagged.
_SEG_NUM = re.compile(r"^\s*\d+\s*[.)]\s*")

# Anonymised stand-ins for a cast member. The D2 replication bought its ambiguity by writing "the
# standing figure" / "the seated figure" instead of names -- which is ambiguity by NON-ATTACHMENT:
# there is no candidate binding left to be ambiguous about, and the measurement disappears.
#
# The head noun is required. A bare "the other" is ordinary prose ("the other side of the room")
# and flagging it rejected good passages in the first smoke run.
_ANON = re.compile(
    r"\bthe (?:standing|seated|other|first|second|older|younger|taller|shorter)\s+"
    r"(?:figure|one|person|man|woman|guest|visitor|customer|worker)\b"
    r"|\bone of them\b",
    re.I,
)

MIN_POST_ONSET_MENTIONS = 2
MIN_AMBIGUOUS_MENTIONS = 2


def strip_segment_numbers(segments: list[str]) -> list[str]:
    return [_SEG_NUM.sub("", s).strip() for s in segments]


def assemble(segments: list[str]) -> tuple[str, list[tuple[int, int]]]:
    """Join segments with single spaces and return ``(passage, [(char_start, char_end), ...])``.

    The offsets are the whole point: a mention's segment index -- and therefore its position
    relative to an onset -- is then exact rather than inferred from a sentence heuristic.
    """
    passage, spans, cursor = "", [], 0
    for i, seg in enumerate(segments):
        if i:
            passage += " "
            cursor += 1
        spans.append((cursor, cursor + len(seg)))
        passage += seg
        cursor += len(seg)
    return passage, spans


def segment_of(char_start: int, spans: list[tuple[int, int]]) -> int:
    """1-based segment index containing ``char_start`` (the last segment for a trailing offset)."""
    for i, (lo, hi) in enumerate(spans, start=1):
        if lo <= char_start < hi:
            return i
    return len(spans)


def mentions_by_segment(
    passage: str, name: str, spans: list[tuple[int, int]]
) -> list[int]:
    """Per-segment count of plain-name mentions of ``name``."""
    counts = [0] * len(spans)
    for match in re.finditer(rf"\b{re.escape(name)}\b", passage):
        counts[segment_of(match.start(), spans) - 1] += 1
    return counts


@dataclass
class DynamicsVerifyResult:
    ok: bool
    reasons: list[str] = field(default_factory=list)
    base: VerifyResult | None = None
    segment_spans: list[tuple[int, int]] = field(default_factory=list)
    mentions: dict[str, list[int]] = field(default_factory=dict)
    diagnostic_by_segment: list[list[str]] = field(default_factory=list)
    anon_hits: list[str] = field(default_factory=list)


def verify_dynamics(
    segments: list[str],
    labels: dict,
    schema: Schema,
    spec: BindingSpec,
    *,
    parent_segments: list[str] | None = None,
    n_pinned: int = 0,
) -> tuple[str, DynamicsVerifyResult]:
    """Run every deterministic gate. Returns ``(assembled_passage, result)``."""
    ctrl = spec.controls
    reasons: list[str] = []
    segments = strip_segment_numbers(segments)
    n_expected = int(ctrl.get("n_segments") or N_SEGMENTS)
    if len(segments) != n_expected:
        # `base` is populated even on this early exit. The caller hands it straight to
        # ``binding.record.build_record``, which reads ``vr.base.n_words`` -- a None here crashed
        # the first full run at 44/50 families, losing nothing but costing a restart.
        passage, spans = assemble(segments)
        return passage, DynamicsVerifyResult(
            False,
            [f"expected {n_expected} segments, got {len(segments)}"],
            base=base_verify(
                passage, labels, schema, spec, ctrl["prop_naming"], ctrl["role_naming"], None
            ),
            segment_spans=spans,
        )

    passage, spans = assemble(segments)

    # ---- 7. prefix byte-identity: the pinned opening must survive verbatim ------------------
    if parent_segments is not None and n_pinned:
        want = " ".join(strip_segment_numbers(parent_segments[:n_pinned]))
        got = " ".join(segments[:n_pinned])
        if want != got:
            reasons.append(f"pinned prefix drifted over segments 1-{n_pinned}")

    # ---- 2/3. mention schedule ---------------------------------------------------------------
    mentions = {p.name: mentions_by_segment(passage, p.name, spans) for p in spec.cast}
    staged_pids = list(ctrl.get("role_onsets", {}))
    if ctrl["design"] == "D2":
        staged_pids = [ctrl["focal_pid"]]
    for pid in staged_pids:
        name = spec.person(pid).name
        counts = mentions[name]
        if any(c < 1 for c in counts):
            missing = [i + 1 for i, c in enumerate(counts) if c < 1]
            reasons.append(f"{name}: not named in segment(s) {missing}")
        onset = int(ctrl.get("role_onsets", {}).get(pid, ctrl.get("resolve_segment") or 0))
        if onset:
            post = sum(counts[onset:])
            if post < MIN_POST_ONSET_MENTIONS:
                reasons.append(
                    f"{name}: {post} post-onset mentions after segment {onset}, "
                    f"need {MIN_POST_ONSET_MENTIONS}"
                )

    # ---- 3 (D2). the focal must be the named agent inside the ambiguous window ---------------
    if ctrl["design"] == "D2":
        focal = spec.person(ctrl["focal_pid"]).name
        for seg in ctrl["ambiguous_segments"]:
            got = mentions[focal][seg - 1]
            if got < MIN_AMBIGUOUS_MENTIONS:
                reasons.append(
                    f"{focal}: {got} mentions in ambiguous segment {seg}, "
                    f"need {MIN_AMBIGUOUS_MENTIONS}"
                )
    anon = sorted({m.group(0).lower() for m in _ANON.finditer(passage)})
    if anon:
        reasons.append("anonymised cast descriptor(s): " + ", ".join(anon))

    # ---- 5. diagnostic-prop onset -------------------------------------------------------------
    # Under `named` the situation must arrive at the schema onset and never before it. Under
    # `abstract` there is no onset at all: the situation is never nameable, so a diagnostic prop
    # ANYWHERE is a violation (the forbidden-lexicon sweep below catches the same words, but this
    # says which segment, which is what the analysis needs).
    diagnostic = [named_diagnostic_props(s, schema) for s in segments]
    schema_onset = ctrl.get("schema_onset")
    if ctrl["prop_naming"] == "named":
        assert schema_onset, "a `named` item must declare a schema onset"
        schema_onset = int(schema_onset)
        if any(diagnostic[: schema_onset - 1]):
            early = [i + 1 for i, d in enumerate(diagnostic[: schema_onset - 1]) if d]
            reasons.append(f"diagnostic prop before the schema onset, in segment(s) {early}")
        if not any(diagnostic[schema_onset - 1 :]):
            reasons.append(
                "no diagnostic prop at or after the schema onset: names none of "
                + ", ".join(schema.diagnostic_props)
            )
    else:
        if schema_onset is not None:
            reasons.append("an `abstract` item must not declare a schema onset")
        if any(diagnostic):
            where = [i + 1 for i, d in enumerate(diagnostic) if d]
            reasons.append(f"diagnostic prop in an abstract item, in segment(s) {where}")

    # ---- 5b. the lexical-ceiling arm: the role word must arrive AT the onset, never before ----
    # `role_naming="named"` is a positive requirement, so a ban-list cannot see a violation in
    # either direction: the word being absent is a silent failure, and the word appearing early
    # destroys the manipulation. Both are checked per segment.
    if ctrl["role_naming"] == "named":
        for pid, onset in ctrl.get("role_onsets", {}).items():
            person = spec.person(pid)
            onset = int(onset)
            words = [w for rid in person.roles for w in judge_accept_words(schema.roles[rid])]
            seen = [
                i + 1
                for i, seg in enumerate(segments)
                if any(re.search(rf"\b{re.escape(w)}\b", seg, re.I) for w in words)
            ]
            early = [i for i in seen if i < onset]
            if early:
                reasons.append(f"{person.name}: role word before the onset, in segment(s) {early}")
            if not any(i >= onset for i in seen):
                reasons.append(
                    f"{person.name}: role_naming=named but no role word at or after segment {onset}"
                )

    # ---- 4/6. the shared checks -------------------------------------------------------------
    # The lexical-ceiling arm names ONE person's role, not everyone's -- that is what makes it a
    # per-person ceiling with the rest of the passage held identical to the latent arms. The
    # inherited `role_naming="named"` mode is all-or-nothing (it bans no role noun and requires
    # every one), so it is run in `latent` mode here -- keeping the ban on every OTHER role -- and
    # the staged person's own words are then subtracted from its complaint. Their arrival at the
    # onset, and absence before it, is checked per segment above.
    staged_words: set[str] = set()
    if ctrl["role_naming"] == "named":
        staged_words = {
            w.lower()
            for pid in ctrl.get("role_onsets", {})
            for rid in spec.person(pid).roles
            for w in judge_accept_words(schema.roles[rid])
        }
    base = base_verify(
        passage,
        labels,
        schema,
        spec,
        ctrl["prop_naming"],
        "latent" if ctrl["role_naming"] == "named" else ctrl["role_naming"],
        None,
    )
    if staged_words:
        kept = []
        for reason in base.reasons:
            if reason.startswith("forbidden words: "):
                left = [
                    w
                    for w in reason.removeprefix("forbidden words: ").split(", ")
                    if w.lower() not in staged_words
                ]
                if left:
                    kept.append("forbidden words: " + ", ".join(left))
            else:
                kept.append(reason)
        base.reasons = kept
        base.ok = not kept
    # `base` re-runs its own diagnostic check on the whole passage; the segment-resolved one above
    # is strictly stronger, so a duplicate complaint is dropped rather than double-counted.
    reasons += [r for r in base.reasons if not r.startswith("no diagnostic prop")]

    return passage, DynamicsVerifyResult(
        ok=not reasons,
        reasons=reasons,
        base=base,
        segment_spans=spans,
        mentions=mentions,
        diagnostic_by_segment=diagnostic,
        anon_hits=anon,
    )
