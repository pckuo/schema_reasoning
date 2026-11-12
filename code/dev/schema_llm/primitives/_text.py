"""Shared token / meta-label text utilities for the geometry suite.

Byte-level-BPE token cleaning, content-word ranking, SEM, and contiguous-run segmentation used by
the temporal-clustering modules (``metacluster``, ``cot_trajectory``) and the function-vector
modules (``funcvec``, ``funcvec_heads``). Extracted here so there is a single copy — these were
byte-identical across those modules.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

_STOP = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "is", "are", "was", "were",
         "they", "them", "he", "she", "it", "that", "this", "then", "for", "with", "as", "by",
         "their", "her", "his", "who", "s", "i", "we", "you", "not", "be", "have", "has", "one",
         "two", "there", "here", "which", "what", "into", "out", "up", "down"}


def _clean(tok: str) -> str:
    """Render a byte-level-BPE token readable (Ġ=space, Ċ=newline)."""
    return tok.replace("\u0120", " ").replace("\u010a", " ").replace("\u0109", " ")


def _top_content_counts(words: list[str], n: int) -> list[tuple[str, int]]:
    """Top-n content words with their counts (freq-ranked; stopwords / non-alpha dropped)."""
    good = (w for w in words if len(w) >= 2 and w not in _STOP and w.isalpha())
    return Counter(good).most_common(n)


def _top_content(words: list[str], n: int) -> list[str]:
    return [w for w, _ in _top_content_counts(words, n)]


def _distinctive_tokens(docs: list[list[tuple[str, int]]], top_m: int = 8) -> list[list[str]]:
    """Per-doc *distinctive* tokens via tf-idf across the given docs (each a token->count list):
    tokens shared by many docs (boilerplate) are down-weighted and a doc's signature tokens surface.
    Tokens present in EVERY doc (idf=0) are dropped. Returns per-doc top-``top_m`` token lists."""
    n = len(docs)
    if n == 0:
        return []
    df: Counter = Counter()
    for d in docs:
        df.update({t for t, _ in d})
    out = []
    for d in docs:
        tot = sum(c for _, c in d) or 1
        scored = [(t, (c / tot) * math.log(n / df[t])) for t, c in d if 0 < df[t] < n]
        scored.sort(key=lambda tc: -tc[1])
        out.append([t for t, _ in scored[:top_m]])
    return out


def _sem(a: np.ndarray) -> np.ndarray:
    n = len(a)
    return a.std(0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros(a.shape[1])


def _segments(labels: np.ndarray) -> list[tuple[int, int, int]]:
    """Contiguous runs -> (meta_id, start, length)."""
    segs, start = [], 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            segs.append((int(labels[start]), start, i - start))
            start = i
    return segs


def _mean_run_len(labels: np.ndarray) -> float:
    return float(np.mean([s[2] for s in _segments(labels)]))
