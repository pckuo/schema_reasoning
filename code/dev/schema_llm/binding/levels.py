"""Derive the core set's two descriptive controls -- ``detail_level`` and ``distractor_level`` --
for a binding corpus, using the core set's own vocabulary so the columns pool cleanly.

In the core set both are *generation* controls: the generator was told to write a terse/medium/rich
passage with none/low/high distractors, so each is crossed with -- and therefore orthogonal to --
every other factor by randomisation. The binding generator manipulates cast structure instead
(B/R/D/E), so nothing sets these directly and they exported as NULL. Deriving them is fine, but the
derivation has to preserve the property that made the core columns usable.

``distractor_level`` is a straight coarsening of D, which *is* a manipulated variable here:

    none = 0 distractors · low = 1-2 · high = 3 or more

``detail_level`` cannot be a raw length tertile. Passage length in this corpus is mostly cast size:
``corr(n_words, E) = 0.76``, and B/D/E/experiment together predict length with ``R^2 = 0.74``. A raw
tertile therefore puts *zero* 1C items in ``rich`` and 201/240 of them in ``terse`` -- the column
would silently re-encode the manipulation, and anyone "controlling for detail_level" would be
controlling away 1A/1B. Raw length is already exported (``n_words``/``n_chars``), so a raw tertile
would also add no information. Instead the tertile is taken over length *residualised on the
design* -- how wordy a passage is **for its cast size and arm** -- which is the non-redundant part
and restores the orthogonality the core column had by construction.
"""

from __future__ import annotations

import numpy as np

# The core set's realised vocabulary (results/schema_llm/data/identify_720, main_72): detail is
# terse/medium/rich and distractor is none/low/high -- NOT low/medium/high.
DETAIL_LEVELS = ("terse", "medium", "rich")
DISTRACTOR_LEVELS = ("none", "low", "high")

# columns residualised out of length before the tertile. B/D/E are the manipulated cast counts;
# experiment/schema/prop_naming are the strata the core column was crossed with.
_NUMERIC = ("B", "D", "E")
_CATEGORICAL = ("experiment", "schema")


def distractor_level(n_distractors: int) -> str:
    """none / low / high for 0 / 1-2 / 3+ distractor people."""
    if n_distractors <= 0:
        return DISTRACTOR_LEVELS[0]
    return DISTRACTOR_LEVELS[1] if n_distractors <= 2 else DISTRACTOR_LEVELS[2]


def _words(record: dict) -> float:
    n = record.get("n_words")
    return float(n if n is not None else len(str(record.get("prompt") or "").split()))


def _design(records: list[dict]) -> np.ndarray:
    """Intercept + cast counts + one-hot strata. Rank deficiency is fine: ``lstsq`` takes the
    minimum-norm solution, so a corpus with a single arm or a constant column still residualises."""
    cols = [np.ones(len(records))]
    for key in _NUMERIC:
        cols.append(np.array([float(r.get(key) or 0) for r in records]))
    for key in (*_CATEGORICAL, "prop_naming"):
        vals = [
            (r.get(key) if r.get(key) is not None else (r.get("gen") or {}).get(key))
            for r in records
        ]
        for lvl in sorted({v for v in vals if v is not None})[1:]:  # drop-first coding
            cols.append(np.array([1.0 if v == lvl else 0.0 for v in vals]))
    return np.column_stack(cols)


def detail_levels(records: list[dict]) -> list[str]:
    """Tertile of design-residualised passage length, in core-set labels.

    Returns one label per record, in input order. Residualising first is what keeps this from
    aliasing the manipulation -- see the module docstring.
    """
    y = np.array([_words(r) for r in records])
    resid = y - _design(records) @ np.linalg.lstsq(_design(records), y, rcond=None)[0]
    # a degenerate corpus (all one length, or fewer items than levels) has no meaningful spread;
    # calling everything `medium` is honest, where forcing a 3-way split would invent one.
    lo, hi = np.quantile(resid, [1 / 3, 2 / 3])
    if not lo < hi:
        return [DETAIL_LEVELS[1]] * len(records)
    return [
        DETAIL_LEVELS[0] if v <= lo else (DETAIL_LEVELS[2] if v > hi else DETAIL_LEVELS[1])
        for v in resid
    ]


def assign_levels(records: list[dict]) -> list[dict]:
    """Fill ``detail_level`` / ``distractor_level`` in place and return the records.

    Written at top level (not under ``gen``) because that is where the core corpus keeps them and
    therefore where every reader -- ``explore.export_bundle``, ``subjects.judge``,
    ``geometry.latents``, ``subjects.capture`` -- already looks.
    """
    if not records:
        return records
    for record, detail in zip(records, detail_levels(records), strict=True):
        record["detail_level"] = detail
        record["distractor_level"] = distractor_level(int(record.get("D") or 0))
    return records
