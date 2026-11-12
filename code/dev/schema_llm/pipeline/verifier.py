"""Verify a generated item: forbidden-lexicon check + span alignment (A4.3 step 3).

Pronoun mentions are NOT verified (by design); only explicit name/description
anchors and scene substrings are aligned to char offsets. Anchor matching is
article-/whitespace-tolerant (natural 'a' <-> 'the' drift is allowed), so an anchor need
not be byte-for-byte verbatim; the recorded span holds the ACTUAL matched text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..core.spec import Schema, forbidden_set
from .sampler import Recipe


@dataclass
class VerifyResult:
    ok: bool
    reasons: list[str] = field(default_factory=list)
    role_map: dict[str, str] = field(default_factory=dict)
    scene_spans: list[dict] = field(default_factory=list)
    referent_spans: dict[str, list[dict]] = field(default_factory=dict)


_ARTICLES = {"a", "an", "the"}


def _anchor_pattern(value: str) -> re.Pattern[str] | None:
    """Regex for a referent anchor: content words matched in order, but any a/an/the is
    interchangeable and whitespace runs are flexible. So a description like 'a person in a
    navy cap' also matches natural article drift ('the person in the navy cap'), while a bare
    name (no articles) stays an effectively exact, case-insensitive match. Function-word drift
    preserves the two invariants we need -- the referent stays locatable and, with content
    words kept in order, unambiguous -- so it should not be rejected (plan A4)."""
    tokens = value.split()
    if not tokens:
        return None
    parts = ["(?:a|an|the)" if t.lower() in _ARTICLES else re.escape(t) for t in tokens]
    return re.compile(r"\s+".join(parts), re.IGNORECASE)


def _find_anchors(text: str, value: str) -> list[dict]:
    """All spans where the anchor appears (article-/whitespace-tolerant). Each span carries the
    ACTUAL matched substring so char offsets and text reflect the prompt, not the recipe's
    idealized string (keeps B2 activation extraction aligned to the real tokens)."""
    pattern = _anchor_pattern(value)
    if pattern is None:
        return []
    return [
        {"char_start": m.start(), "char_end": m.end(), "text": m.group(0)}
        for m in pattern.finditer(text)
    ]


def _is_quoted(text: str, span: dict) -> bool:
    """True if the matched anchor is wrapped in literal double quotes (the artifact where the
    model writes e.g. \"Theo\" instead of Theo)."""
    i, j = span["char_start"], span["char_end"]
    return i > 0 and j < len(text) and text[i - 1] == '"' and text[j] == '"'


def verify(prompt: str, labels: dict, schema: Schema, recipe: Recipe) -> VerifyResult:
    reasons: list[str] = []

    # 1. forbidden-lexicon check (word-boundary, case-insensitive).
    forbidden = forbidden_set(
        schema, recipe.controls["prop_naming_map"], recipe.controls.get("role_naming", "latent")
    )
    hits = sorted(w for w in forbidden if re.search(rf"\b{re.escape(w)}\b", prompt, re.IGNORECASE))
    if hits:
        reasons.append("forbidden words: " + ", ".join(hits))

    # 2. every referent anchor (name / description) must appear -- article-/whitespace-tolerant
    #    (natural 'a' <-> 'the' drift is allowed), content words in order. role_map keeps the
    #    stable recipe value as the label; referent_spans carry the ACTUAL matched text.
    role_map: dict[str, str] = {}
    referent_spans: dict[str, list[dict]] = {}
    for role_id, refs in recipe.roles.items():
        for ref in refs:
            spans = _find_anchors(prompt, ref.value)
            if spans:
                role_map[ref.value] = role_id
                referent_spans[ref.value] = spans
            else:
                reasons.append(f"missing referent: {ref.value!r}")
            # catch the artifact where the model wraps the anchor in literal quotes
            if any(_is_quoted(prompt, s) for s in spans):
                reasons.append(f"quoted anchor: {ref.value!r}")

    # 3. scene labels: each text must be a substring; the recovered sequence (WITH duplicates) must
    #    equal scene_path. Scan with a cursor so a scene rendered twice (the `repeat` temporal form)
    #    gets distinct, ordered spans instead of both collapsing onto the first occurrence; for the
    #    usual unique-scene case this is identical to a plain first-occurrence find.
    scene_spans: list[dict] = []
    cursor = 0
    for item in labels.get("scenes", []) if isinstance(labels, dict) else []:
        scene, text = item.get("scene"), item.get("text", "")
        idx = prompt.find(text, cursor) if text else -1
        if idx == -1 and text:  # nothing after the cursor -> labels slightly out of order; retry
            idx = prompt.find(text)
        if idx == -1:
            reasons.append(f"scene text not found: {scene!r}")
            continue
        scene_spans.append({"scene": scene, "char_start": idx, "char_end": idx + len(text)})
        cursor = idx + len(text)
    ordered = [d["scene"] for d in sorted(scene_spans, key=lambda d: d["char_start"])]
    if ordered != recipe.scene_path:
        reasons.append(f"scene set/order mismatch: {ordered} != {recipe.scene_path}")

    return VerifyResult(
        ok=not reasons,
        reasons=reasons,
        role_map=role_map,
        scene_spans=scene_spans,
        referent_spans=referent_spans,
    )
