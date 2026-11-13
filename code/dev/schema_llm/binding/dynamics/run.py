"""Generation driver for the binding-dynamics batch (`binding_dyn_v1`).

Per family, in order:

1. **seed** -- for D1 the family's LATEST-onset arm (the only one whose early segments carry no role
   information); for D2 the shared 3-segment ambiguous prefix;
2. **siblings** -- each generated as a continuation with as many leading segments of the seed
   **pinned verbatim** as its own schedule allows, so a family shares a byte-identical prefix and
   the pre-onset comparison is the same forward pass rather than a matched one;
3. **verify** (deterministic) and **audit** (per cut, via TRAPI) with bounded retries.

Audits of a shared prefix are computed once and reused across the siblings that share it -- the
text is identical, so a second sample would only add noise and cost.

Examples
--------
    # 4 families per design, both schemas -- the pre-flight pilot
    uv run --extra api python -m dev.schema_llm.binding.dynamics.run --smoke \
        --out results/schema_llm/data/binding_dyn_smoke

    # the full batch
    uv run --extra api python -m dev.schema_llm.binding.dynamics.run \
        --out results/schema_llm/data/binding_dyn_v1
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ...core.spec import Schema, load_schema
from ..cast import BindingSpec
from ..config import TRAPI_MODEL, TRAPI_MODEL_POOL
from ..design import Family
from ..llm import DEFAULT_CONCURRENCY, pick_healthy, set_concurrency, stats
from ..record import build_record, family_length_report, write_run
from ..tasks import build_identify
from ..verify import length_ok
from .audit import CutAudit, DynamicsAuditResult, audit_item
from .design import (
    LENGTH_PROFILES,
    N_SEGMENTS,
    build_all,
    build_named_ceiling,
    build_wave2,
    build_wave3,
    build_wave4,
)
from .prose import (
    build_continue_prompt,
    build_d2_branch_prompt,
    build_d2_prefix_prompt,
    build_seed_prompt,
    generate,
)
from .verify import (
    MIN_AMBIGUOUS_MENTIONS,
    DynamicsVerifyResult,
    strip_segment_numbers,
    verify_dynamics,
)

REPO_ROOT = Path(__file__).resolve().parents[5]
DATA_ROOT = REPO_ROOT / "results" / "schema_llm" / "data"
# One dataset per design, on purpose. The two designs are run as SEPARATE STAGES -- D1 generated,
# collected, judged, clustered and exported end to end before D2 begins -- so they must not share a
# run directory: `subjects.collect` / `judge` / `capture` all key on the run dir, and a half-written
# D2 sitting next to a finished D1 would make every downstream "did this finish?" check ambiguous.
DEFAULT_OUT = {
    "D1-w3": DATA_ROOT / "binding_dyn_d1_v3",
    "D2-w3": DATA_ROOT / "binding_dyn_d2_v3",
    # wave 4 lands in its own directory; the two waves are MERGED at export, and keeping them
    # separate on disk is what lets the audit run on wave 4 alone or on the union at will.
    "D1-w4": DATA_ROOT / "binding_dyn_d1_v3w2",
    "D2-w4": DATA_ROOT / "binding_dyn_d2_v3w2",
    # the ceiling is its own dataset: it is a REFERENCE for the latent corpus, not part of it
    "D1-w5": DATA_ROOT / "binding_dyn_d1_named",
    "D1-w5-latent": DATA_ROOT / "binding_dyn_d1_namedctrl",
    "D2-w5": DATA_ROOT / "binding_dyn_d2_named",
    "D1": DATA_ROOT / "binding_dyn_d1_v1",
    "D1a": DATA_ROOT / "binding_dyn_d1_v1",
    "D1b": DATA_ROOT / "binding_dyn_d1_v1",
    "D2": DATA_ROOT / "binding_dyn_d2_v1",
}


def is_d2(family: Family) -> bool:
    """Wave 2 tags its experiments ``D1a-w2`` / ``D2-w2``, so every dispatch here must match on the
    PREFIX. Comparing for equality silently sent wave-2 D1a families down the D1b branch, which a
    prompt-render test caught before any generation ran."""
    return family.experiment.startswith("D2")


def seed_arm(family: Family) -> str:
    """The arm generated first: latest role onset (D1) or the prefix owner (D2).

    Derived from the family's OWN items rather than from the wave-1 arm constants. Wave 4 renames
    the arms (`long5`, `press4`, `late_a`, ...), and returning a name the family does not contain
    made the seed lookup raise a bare StopIteration that surfaced as "FAILED (StopIteration: )" on
    every family, with no indication of which lookup had failed.
    """
    if is_d2(family):
        return family.items[0].controls["arm"]
    return max(
        family.items,
        key=lambda i: max((int(s) for s in i.controls["role_onsets"].values()), default=0),
    ).controls["arm"]


def pinned_depth(spec: BindingSpec, seed_spec: BindingSpec) -> int:
    """How many leading segments this sibling inherits verbatim from the seed.

    Everything up to the first segment where the sibling's schedule differs from the seed's. For
    D2 that is the whole ambiguous prefix; for D1 it is one before the onset that moved.
    """
    if spec.controls["design"] == "D2":
        return max(spec.controls["ambiguous_segments"])
    n_total = int(spec.controls.get("n_segments") or N_SEGMENTS)
    seed_onsets = seed_spec.controls["role_onsets"]
    moved = [
        int(seg)
        for pid, seg in spec.controls["role_onsets"].items()
        if int(seg) < int(seed_onsets.get(pid, n_total + 1))
    ]
    return (min(moved) - 1) if moved else n_total


def _band(family: Family) -> tuple[int, int]:
    """Word band from the FORMAT, not the cast size -- and from THIS family's length profile, since
    wave 2 deliberately makes segments unequal to decorrelate onset segment from absolute position.

    ``binding/config.length_target`` scales with headcount because the free-form generator's length
    does. This batch is a fixed six-segment format, so its length is set by the format and the band
    has to follow it -- and a constant band removes the length-vs-D confound that 1A carries
    (``main/TODO.md``: mean words rise ~19 per bystander), because every item targets the same
    total regardless of E.
    """
    ctrl = family.items[0].controls
    profile = LENGTH_PROFILES[ctrl.get("length_profile") or "uniform"]
    # v3 wave 2 runs `long`/`late` at 8 segments, so the band is no longer a corpus constant: it
    # follows THIS item's segment count. Profiles are written for the 6-segment default and repeat
    # their final band when a passage is longer. Without this an 8-segment item was measured
    # against a 6-segment band -- ~560 words against (330, 450) -- and only escaped the drift gate
    # because that gate is deliberately loose.
    n_seg = int(ctrl.get("n_segments") or N_SEGMENTS)
    if len(profile) < n_seg:
        profile = profile + (profile[-1],) * (n_seg - len(profile))
    profile = profile[:n_seg]
    return (sum(lo for lo, _ in profile), sum(hi for _, hi in profile))


# Length is deliberately a LOGGED COVARIATE here, not a tight gate. Staging costs words -- the S=3
# pilot ran +51% over the corpus mean for its cast size, because a pre-onset window is filler that
# still has to be concrete -- and a fixed six-segment format already removes the length-vs-D
# confound that a per-cast band exists to control. So the check only catches runaway drift (a
# sibling half or double its family's target), and `n_words` carries the rest into the analysis.
LENGTH_SLACK = 0.5


def _merge_labels(seed_labels: dict, new_labels: dict, pinned_text: str) -> dict:
    """Scene labels for an assembled sibling.

    A continuation only labels the scenes it wrote, so the assembled passage would appear to be
    missing every scene in the pinned opening. The opening is byte-identical to the seed's, so the
    seed's labels for it are still exactly true and are carried across.

    Duplicates are dropped, keeping the earlier entry: a continuation that re-narrates the last
    pinned beat would otherwise produce ``[arrive, arrive, seated, ...]`` and fail the scene-order
    check for a labelling artefact rather than a real fault (observed on every sibling in the first
    smoke run).
    """
    kept = [
        entry
        for entry in (seed_labels or {}).get("scenes", [])
        if entry.get("text") and entry["text"] in pinned_text
    ]
    merged, seen = [], set()
    for entry in kept + list((new_labels or {}).get("scenes", [])):
        scene = str(entry.get("scene", "")).strip().lower().replace(" ", "_")
        if scene in seen:
            continue
        seen.add(scene)
        merged.append(entry)
    return {"scenes": merged}


def covered_scenes(seed_labels: dict, pinned_text: str) -> list[str]:
    """Scene names already narrated inside the pinned opening.

    Computed from the seed's own labels rather than apportioned by segment count. The heuristic
    version dropped ``check_in`` from an airport continuation's instructions, so the assembled
    passage was genuinely missing a scene -- a generation fault caused by the generator's own
    prompt.
    """
    return [
        str(entry["scene"]).strip().lower().replace(" ", "_")
        for entry in (seed_labels or {}).get("scenes", [])
        if entry.get("text") and entry["text"] in pinned_text
    ]


def _attempt(
    schema: Schema,
    spec: BindingSpec,
    prompt: str,
    *,
    seed_segments: list[str] | None,
    seed_labels: dict | None,
    n_pinned: int,
    band: tuple[int, int],
    model: str,
    audit_n: int,
    do_audit: bool,
    prefix_cache: dict[str, CutAudit],
) -> tuple[list[str], dict, str, DynamicsVerifyResult, DynamicsAuditResult | None]:
    new, labels, raw = generate(prompt, model=model)
    if not new:
        return [], {}, raw, DynamicsVerifyResult(False, ["generate: unparseable"]), None
    pinned = strip_segment_numbers(seed_segments[:n_pinned]) if n_pinned else []
    segments = pinned + new
    if n_pinned:
        labels = _merge_labels(seed_labels or {}, labels, " ".join(pinned))
    passage, vr = verify_dynamics(
        segments, labels, schema, spec, parent_segments=seed_segments, n_pinned=n_pinned
    )
    n_words = len(passage.split())
    if vr.ok and not length_ok(n_words, band, slack=LENGTH_SLACK):
        vr.reasons.append(f"length {n_words} far outside {band}")
        vr.ok = False
    ar = None
    if vr.ok and do_audit:
        ar = audit_item(
            schema, spec, segments, model=model, n=audit_n, prefix_cache=prefix_cache
        )
    return segments, labels, raw, vr, ar


def generate_item(
    schema: Schema,
    spec: BindingSpec,
    prompt: str,
    *,
    seed_segments: list[str] | None,
    seed_labels: dict | None,
    n_pinned: int,
    band: tuple[int, int],
    model: str,
    tries: int,
    audit_n: int,
    do_audit: bool,
    prefix_cache: dict[str, CutAudit],
) -> tuple[dict, list[str]] | None:
    """Best of ``tries`` attempts. The least-bad attempt is kept, not merely the first -- a later
    try with one complaint is a better parent for the rest of the family than an earlier one with
    five (the rule ``binding/run.py`` already follows)."""
    best = None
    started = time.time()
    for attempt in range(1, tries + 1):
        segments, labels, raw, vr, ar = _attempt(
            schema,
            spec,
            prompt,
            seed_segments=seed_segments,
            seed_labels=seed_labels,
            n_pinned=n_pinned,
            band=band,
            model=model,
            audit_n=audit_n,
            do_audit=do_audit,
            prefix_cache=prefix_cache,
        )
        ok = vr.ok and (ar is None or ar.ok)
        score = (ok, -len(vr.reasons) - len(ar.reasons if ar else []), bool(segments))
        if best is None or score > best[0]:
            best = (score, segments, labels, raw, vr, ar, attempt)
        if ok:
            break
    if best is None or not best[1]:
        return None
    _s, segments, labels, raw, vr, ar, attempts = best
    passage = " ".join(segments)
    task = build_identify(schema, spec, passage, generator_model=model)
    meta = {
        "model": model,
        "attempts": attempts,
        "seconds": round(time.time() - started, 2),
        "generator_prompt": prompt,
        "band": list(band),
        "accepted": vr.ok and (ar is None or ar.ok),
        "prop_naming": spec.controls["prop_naming"],
        "role_naming": spec.controls["role_naming"],
        "n_pinned_segments": n_pinned,
        "spec": spec.to_dict(),
    }
    record = build_record(spec, passage, labels, vr.base, None, task, meta)
    # dynamics-specific logging: the segment structure, the declared schedule, and the REALIZED
    # onsets the audit measured. The last of these is the analysis variable.
    record.update(
        {
            "segments": segments,
            "segment_spans": [list(s) for s in vr.segment_spans],
            "n_segments": len(segments),
            "mentions_by_segment": vr.mentions,
            "diagnostic_by_segment": vr.diagnostic_by_segment,
            "verify_ok": vr.ok,
            "verify_reasons": vr.reasons,
            "anon_hits": vr.anon_hits,
            **{k: v for k, v in spec.controls.items() if k not in ("prop_naming", "role_naming")},
            **(ar.to_dict() if ar else {"audit_ok": None, "audit_reasons": []}),
        }
    )
    return record, segments


def generate_family(
    family: Family,
    *,
    tries: int,
    audit_n: int,
    do_audit: bool,
    model: str,
    out_dir: Path | None,
    pool: list[str] | None,
    repair: bool = False,
) -> tuple[list[dict], dict[str, str], list[str]]:
    if out_dir is not None:
        cached = _load_checkpoint(out_dir, family, repair=repair)
        if cached is not None:
            return cached[0], cached[1], [f"  == {family.family_id}: resumed =="]
    if pool:
        model = pick_healthy(model, pool)
    schema = load_schema(family.schema)
    band = _band(family)
    arm0 = seed_arm(family)
    seed_spec = next(i for i in family.items if i.controls["arm"] == arm0)
    log = [
        f"  == {family.family_id} ({family.experiment}, {family.meta['prop_naming']}, "
        f"seed={arm0}, band={band}) =="
    ]
    prefix_cache: dict[str, CutAudit] = {}

    prompt = (
        build_d2_prefix_prompt(schema, seed_spec)
        if is_d2(family)
        else build_seed_prompt(schema, seed_spec)
    )
    records: list[dict] = []
    raws: dict[str, str] = {}

    seed_labels: dict = {}
    if is_d2(family):
        prefix = _generate_prefix(schema, seed_spec, prompt, model=model, tries=tries)
        if prefix is None:
            return [], {}, log + ["    [drop] ambiguous prefix never accepted"]
        seed_segments, seed_labels = prefix
    else:
        out = generate_item(
            schema,
            seed_spec,
            prompt,
            seed_segments=None,
            seed_labels=None,
            n_pinned=0,
            band=band,
            model=model,
            tries=tries,
            audit_n=audit_n,
            do_audit=do_audit,
            prefix_cache=prefix_cache,
        )
        if out is None:
            return [], {}, log + [f"    [drop] {seed_spec.item_id}: no usable generation"]
        record, seed_segments = out
        seed_labels = record.get("labels") or {}
        records.append(record)
        raws[seed_spec.item_id] = record["gen"].pop("generator_prompt")
        log.append(_line(record))

    for spec in family.items:
        if spec.controls["arm"] == arm0 and not is_d2(family):
            continue
        n_pinned = pinned_depth(spec, seed_spec)
        pinned_text = " ".join(seed_segments[:n_pinned])
        done = set(covered_scenes(seed_labels, pinned_text))
        remaining = [s for s in spec.scene_path if s not in done]
        prompt = (
            build_d2_branch_prompt(
                schema, spec, seed_segments[:n_pinned], remaining_scenes=remaining
            )
            if is_d2(family)
            else build_continue_prompt(
                schema, spec, seed_segments[:n_pinned], remaining_scenes=remaining
            )
        )
        out = generate_item(
            schema,
            spec,
            prompt,
            seed_segments=seed_segments,
            seed_labels=seed_labels,
            n_pinned=n_pinned,
            band=band,
            model=model,
            tries=tries,
            audit_n=audit_n,
            do_audit=do_audit,
            prefix_cache=prefix_cache,
        )
        if out is None:
            log.append(f"    [drop] {spec.item_id}: no usable generation")
            continue
        record, _segs = out
        records.append(record)
        raws[spec.item_id] = record["gen"].pop("generator_prompt")
        log.append(_line(record))

    if out_dir is not None and len(records) == len(family.items):
        _save_checkpoint(out_dir, family.family_id, records, raws)
    return records, raws, log


def _generate_prefix(
    schema: Schema, spec: BindingSpec, prompt: str, *, model: str, tries: int
) -> tuple[list[str], dict] | None:
    """The shared ambiguous opening. Only the segment count and the name-attachment rule are
    checkable here -- the ambiguity itself is scored on the assembled siblings, where the audit has
    a whole passage to read."""
    want = max(spec.controls["ambiguous_segments"])
    focal = spec.person(spec.controls["focal_pid"]).name
    for _ in range(tries):
        segments, labels, _raw = generate(prompt, model=model)
        segments = strip_segment_numbers(segments)
        if len(segments) != want:
            continue
        # the focal must be the NAMED agent inside the ambiguous window, not an anonymous
        # descriptor -- ambiguity by non-attachment leaves no binding to be ambiguous about
        if all(seg.count(focal) >= MIN_AMBIGUOUS_MENTIONS for seg in segments[1:]):
            return segments, labels
    return None


def _line(record: dict) -> str:
    flag = "ok " if record["verify_ok"] and record.get("audit_ok") is not False else "FLAG"
    onsets = record.get("realized_onsets", {})
    return (
        f"    [{flag}] {record['item_id']}  E={record['E']} D={record['D']} "
        f"{record['n_words']}w  pin={record['gen']['n_pinned_segments']}  "
        f"onsets={onsets}  x{record['gen']['attempts']}"
    )


def _checkpoint_path(out_dir: Path, family_id: str) -> Path:
    return out_dir / "families" / f"{family_id}.json"


def _load_checkpoint(
    out_dir: Path, family: Family, repair: bool = False
) -> tuple[list[dict], dict[str, str]] | None:
    """A completed family's saved output, if the run already produced it.

    With ``repair``, a family that completed but contains a REJECTED item is treated as absent, so
    it is generated again with a fresh budget. The runner keeps the least-bad attempt rather than
    dropping an item, which is right for keeping a family's lineage intact but means an ordinary
    retryable prose fault ("a staged person is missing from segment 4") is otherwise frozen in.
    """
    path = _checkpoint_path(out_dir, family.family_id)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data.get("records", [])
    if len(records) != len(family.items):
        return None
    if repair and any(
        not r.get("verify_ok") or r.get("audit_ok") is False for r in records
    ):
        return None
    return records, data.get("prompts", {})


def _save_checkpoint(out_dir: Path, family_id: str, records: list[dict], prompts: dict) -> None:
    path = _checkpoint_path(out_dir, family_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"records": records, "prompts": prompts}, ensure_ascii=False), encoding="utf-8"
    )


def select_families(args) -> list[Family]:
    if args.wave == 5:
        # wave 5 = the LEXICAL CEILING: matched named twins of wave 4, same casts and schedules.
        families = build_named_ceiling(
            args.design[:2], args.resume_from, args.seed,
            role_naming=args.ceiling_role_naming,
            suffix="NAMED" if args.ceiling_role_naming == "named" else "CTRL",
        )
    elif args.wave == 4:
        # wave 4 CONTINUES wave 3's corpus: it reuses that corpus's realized name pool and
        # assignment history, so the union spends 108 names four times each instead of 180 twice.
        families = build_wave4(args.design[:2], args.resume_from, args.seed)
    elif args.wave == 3:
        families = build_wave3(args.design[:2], args.seed)
    elif args.wave == 2:
        families = build_wave2(args.seed)
    else:
        families = build_all(args.seed)
    if args.design:
        families = [f for f in families if f.experiment.startswith(args.design)]
    if args.schema:
        families = [f for f in families if f.schema == args.schema]
    if args.smoke:
        # One family per distinct CELL. In wave 2 the cell that matters is the new coordinate
        # (`w2_arm`), not prop_naming: each arm exercises a different prompt path, and a smoke that
        # misses one is how an unexercised path reaches a 150-item batch.
        picked: list[Family] = []
        seen: set = set()
        for family in families:
            key = (
                family.experiment,
                family.schema,
                family.meta.get("w2_arm") or family.meta["prop_naming"],
            )
            if key in seen:
                continue
            seen.add(key)
            picked.append(family)
        families = picked
    if args.limit:
        families = families[: args.limit]
    return families


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    # `--design` is REQUIRED: the two designs are separate stages, and defaulting to "both" would
    # let one command start work the stage plan says must wait.
    ap.add_argument("--design", required=True, choices=["D1", "D1a", "D1b", "D2"])
    ap.add_argument("--out", type=Path, help="defaults to the design's own dataset directory")
    ap.add_argument("--schema", choices=["restaurant", "airport"])
    ap.add_argument("--smoke", action="store_true", help="one family per design x schema x arm")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--seed", type=int, default=20260909)
    ap.add_argument(
        "--wave",
        type=int,
        default=1,
        choices=(1, 2, 3, 4, 5),
        help="1 = original grid; 2 = variability batch; 3 = de-confounded regeneration; "
             "4 = de-confounded second wave (requires --resume-from); "
             "5 = lexical-ceiling named twins of wave 4 (requires --resume-from)",
    )
    ap.add_argument(
        "--resume-from",
        type=Path,
        help="wave 4 only: the wave-3 corpus whose name pool and assignment history to continue",
    )
    ap.add_argument(
        "--ceiling-role-naming", choices=("named", "latent"), default="named",
        help="wave 5 only: `named` is the lexical ceiling; `latent` is its generator control, "
             "the same casts written latently on the SAME deployment as the ceiling",
    )
    ap.add_argument("--tries", type=int, default=3)
    ap.add_argument("--audit-n", type=int, default=4)
    ap.add_argument("--no-audit", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    ap.add_argument("--model", default=TRAPI_MODEL)
    ap.add_argument("--pool", action="store_true", help="round-robin the TRAPI deployment pool")
    ap.add_argument(
        "--repair",
        action="store_true",
        help="regenerate completed families holding a rejected item (pair with a higher --tries)",
    )
    args = ap.parse_args()

    if args.wave in (4, 5):
        if args.resume_from is None:
            args.resume_from = DEFAULT_OUT[f"{args.design[:2]}-w3"]
        if not Path(args.resume_from, "items.jsonl").exists():
            ap.error(f"--resume-from {args.resume_from} has no items.jsonl to continue")
    if args.out is None:
        key = f"{args.design}-w{args.wave}" if args.wave >= 3 else args.design
        if args.wave == 5 and args.ceiling_role_naming == "latent":
            key += "-latent"
        args.out = DEFAULT_OUT[key]
    set_concurrency(args.concurrency)
    families = select_families(args)
    pool = list(TRAPI_MODEL_POOL) if args.pool else None
    models = {f.family_id: (pool[i % len(pool)] if pool else args.model)
              for i, f in enumerate(families)}
    print(
        f"[dynamics] {args.design}: {len(families)} families / "
        f"{sum(len(f.items) for f in families)} items -> {args.out}"
    )
    args.out.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    raws: dict[str, str] = {}
    started = time.time()

    def run_one(family: Family):
        # A family is isolated: an unexpected error in one must not take down a multi-hour run
        # through `ex.map`, which propagates the first exception and discards the rest. Completed
        # families are already checkpointed, so the cost of a failure is that family alone.
        try:
            return generate_family(
                family,
                tries=args.tries,
                audit_n=args.audit_n,
                do_audit=not args.no_audit,
                model=models[family.family_id],
                out_dir=args.out,
                pool=pool,
                repair=args.repair,
            )
        except Exception as exc:  # noqa: BLE001 -- deliberately broad; the run continues
            return [], {}, [f"  == {family.family_id}: FAILED ({type(exc).__name__}: {exc}) =="]

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for recs, rws, log in ex.map(run_one, families):
            records += recs
            raws.update(rws)
            print("\n".join(log), flush=True)

    write_run(records, raws, args.out)
    (args.out / "family_lengths.json").write_text(
        json.dumps(family_length_report(records), indent=1), encoding="utf-8"
    )
    accepted = sum(1 for r in records if r["gen"]["accepted"])
    print(
        f"[dynamics] wrote {len(records)} items ({accepted} accepted) to {args.out} "
        f"in {(time.time() - started) / 60:.1f} min; llm={stats()}"
    )


if __name__ == "__main__":
    main()
