"""Schema preamble: an opt-in in-context script prepended to a probe at collection time.

See ``main/plans/SCHEMA_INCONTEXT_PROMPT_PLAN.md``. The preamble is the schema's typical scene
sequence + the person-roles named by their canonical noun (the "named" register), bare (no header
line). It is prepended to a probe's ``probe_prompt`` just before the model call; the frozen
``probes.jsonl`` corpus is never modified, and responses land under a distinct ``-ic`` tag. Kept in
sync with ``core/schemas/*.yaml`` by ``tests/test_schema_incontext.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

# Register modes: "none" = baseline (no preamble); "named" = canonical-noun preamble. The leak-free
# "latent" register is deferred (plan §9).
MODES = ("none", "named")

# The bare preambles (named register; plan §1). Sequence = each schema's `scenes` list, one word
# per scene (compound tokens spelled out); roles = the person roles in schema order, each by its
# canonical noun (airport's colliding "agent" roles disambiguated to their accept forms). A drift
# test ties both lines back to the YAML spec.
PREAMBLES: dict[str, str] = {
    "restaurant": (
        "typical sequence: arrive -> seated -> order -> eat -> pay -> leave\n"
        "typical roles: customer, host, waiter, cook"
    ),
    "airport": (
        "typical sequence: arrive -> check in -> security -> walk to gate -> board -> fly -> "
        "deplane -> baggage\n"
        "typical roles: passenger, check-in agent, security officer, gate agent, pilot"
    ),
}


def preamble_text(schema: str) -> str:
    """The in-context preamble for a schema name (KeyError if unknown)."""
    return PREAMBLES[schema]


def schema_of_run(run_dir: Path) -> str:
    """The schema name of a run dir, read from its ``items.jsonl`` (each run dir is single-schema).
    Raises ``ValueError`` if the dir mixes schemas or has no items."""
    path = Path(run_dir) / "items.jsonl"
    schemas = {
        json.loads(line)["schema"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if len(schemas) != 1:
        raise ValueError(f"{path} is not single-schema: {sorted(schemas)}")
    return schemas.pop()


def apply_incontext(probes: list[dict], schema: str, mode: str) -> list[dict]:
    """Prepend the schema preamble to each probe's ``probe_prompt`` (mode ``"named"``), or return
    the probes unchanged (mode ``"none"``). Returns NEW probe dicts (the input is not mutated) and
    stamps ``schema_incontext=<mode>`` on each, for response-row provenance."""
    if mode not in MODES:
        raise ValueError(f"unknown schema-incontext mode {mode!r} (expected one of {MODES})")
    if mode == "none":
        return probes
    pre = preamble_text(schema)
    return [
        {**p, "probe_prompt": f"{pre}\n\n{p['probe_prompt']}", "schema_incontext": mode}
        for p in probes
    ]
