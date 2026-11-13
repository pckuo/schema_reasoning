"""Throttle-tolerant TRAPI calls for binding generation.

``core.llm.run_trapi`` is the right primitive but its defaults are tuned for short interactive
batches: six retries with ``1.5 * 2**attempt`` backoff totals only ~95 s, after which it returns a
banner string rather than raising. Under sustained throttling on the shared endpoint every call in
a concurrent batch exhausts that budget at once and comes back as a banner, which reads downstream
as a generation failure -- 16/16 audits failed that way at 4 workers while the same call succeeded
sequentially.

``run_trapi`` itself is left alone deliberately: it is hashed into the immutable RLVR V13 contract
(see ``main/STATUS.md``), so it must stay byte-identical.

Two things are added here:

* a **process-wide concurrency cap**, so widening the worker pool cannot widen the request rate
  past what the endpoint tolerates; and
* **banner-aware outer retries** with long, jittered backoff, performed OUTSIDE the semaphore so a
  waiting call does not hold a slot.

Point ``OPENAI_BASE_URL`` at a provisioned endpoint to raise the ceiling; then
``BINDING_MAX_CONCURRENCY`` is the only knob that needs changing.
"""

from __future__ import annotations

import os
import random
import threading
import time

from ..core.llm import run_trapi

# PER-DEPLOYMENT, not global: each TRAPI deployment carries its own quota, so running several in
# parallel multiplies throughput rather than sharing one pool. On the shared `redmond/interactive`
# tier a single deployment tolerates only a handful of simultaneous requests.
DEFAULT_CONCURRENCY = int(os.environ.get("BINDING_MAX_CONCURRENCY", "3"))
GEN_TIMEOUT = 900
MAX_TOKENS = 32768
OUTER_ATTEMPTS = 4
# A deployment that has just refused several calls in a row is quota-starved; keep sending to it
# and every worker holding one of its families stalls in backoff. Park it briefly instead and let
# healthy deployments carry the load.
FAIL_THRESHOLD = 2
COOLDOWN_S = 180.0
_INNER_RETRIES = 2  # keep the fast path short; the patient waiting happens out here

_CONCURRENCY = DEFAULT_CONCURRENCY
_LIMITS: dict[str, threading.Semaphore] = {}
_LOCK = threading.Lock()
_STATS: dict[str, dict] = {}
_HEALTH: dict[str, dict] = {}


def set_concurrency(n: int) -> None:
    """Set the PER-DEPLOYMENT cap. Called once from the CLI before any worker starts."""
    global _CONCURRENCY
    with _LOCK:
        _CONCURRENCY = max(1, n)
        _LIMITS.clear()


def _limiter(model: str) -> threading.Semaphore:
    with _LOCK:
        if model not in _LIMITS:
            _LIMITS[model] = threading.Semaphore(_CONCURRENCY)
        return _LIMITS[model]


def _bump(model: str, key: str, amount: float = 1) -> None:
    with _LOCK:
        _STATS.setdefault(model, {"calls": 0, "banners": 0, "waited_s": 0.0})[key] += amount


def _mark(model: str, ok: bool) -> None:
    """Track consecutive refusals so a starved deployment can be parked."""
    with _LOCK:
        state = _HEALTH.setdefault(model, {"fails": 0, "cool_until": 0.0})
        if ok:
            state["fails"] = 0
            state["cool_until"] = 0.0
        else:
            state["fails"] += 1
            if state["fails"] >= FAIL_THRESHOLD:
                state["cool_until"] = time.time() + COOLDOWN_S


def is_cooling(model: str) -> bool:
    with _LOCK:
        return _HEALTH.get(model, {}).get("cool_until", 0.0) > time.time()


def pick_healthy(preferred: str, pool: list[str]) -> str:
    """``preferred`` if it is answering, else the least-recently-failed alternative.

    Called ONCE per family, before any of its items exist, so the whole family still shares one
    deployment -- switching model mid-family would reintroduce the prose-style confound the
    seed-family design exists to remove.
    """
    if not is_cooling(preferred):
        return preferred
    alternatives = [m for m in pool if m != preferred and not is_cooling(m)]
    if not alternatives:
        return preferred
    with _LOCK:
        return min(alternatives, key=lambda m: _HEALTH.get(m, {}).get("fails", 0))


def stats() -> dict[str, dict]:
    """Per-deployment call / throttle counts, so a starved deployment is visible."""
    with _LOCK:
        return {m: dict(v) for m, v in _STATS.items()}


def is_banner(raw: str) -> bool:
    """Whether ``raw`` is ``run_trapi``'s give-up banner rather than a model answer."""
    text = raw.strip()
    return text.startswith("[trapi ") or text.startswith("[copilot ")


def call(
    prompt: str,
    *,
    model: str,
    timeout: int = GEN_TIMEOUT,
    max_completion_tokens: int = MAX_TOKENS,
    attempts: int = OUTER_ATTEMPTS,
) -> str:
    """One TRAPI call, capped and patient. Returns the reply text, or the last banner."""
    raw = ""
    limiter = _limiter(model)
    for attempt in range(attempts):
        with limiter:
            raw = run_trapi(
                prompt,
                model=model,
                timeout=timeout,
                max_retries=_INNER_RETRIES,
                max_completion_tokens=max_completion_tokens,
            )
        _bump(model, "calls")
        if not is_banner(raw):
            _mark(model, True)
            return raw
        _bump(model, "banners")
        _mark(model, False)
        if attempt == attempts - 1:
            break
        # Long, jittered, and outside the semaphore: throttling clears on the order of minutes, and
        # holding a slot while sleeping would just starve the other workers of this deployment.
        delay = min(240.0, 20.0 * 2**attempt) + random.uniform(0.0, 15.0)
        _bump(model, "waited_s", delay)
        time.sleep(delay)
    return raw
