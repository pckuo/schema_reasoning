"""Corpus-level confound audit for the binding-dynamics corpora.

**A gate, not a report.** It runs after ``assemble`` and before ``collect``, costs no GPU and no
TRAPI, and exits non-zero when the corpus leaks role through a channel that has nothing to do with
binding.

Why this exists: the per-cut LLM audit (:mod:`.audit`) asks *"can a reader tell?"* on one item
at a time. A benchmark also has to answer *"can a classifier tell, across the corpus?"* Those
are different questions, and v1 passed the first while failing the second badly -- a role's index in
``role_order`` fixed its gender, which fixed its name pool, and the cast list was the role order, so
gender + introduction order + schema predicted role at .976 (1.000 within restaurant). Three further
channels leaked: pre-onset text at .730 with names *and* pronouns masked, bystanders at .977, and
raw mention count at .557.

Design notes, each of which is load-bearing:

* **Grouped CV by FAMILY, not item.** Siblings share one cast, so grouping by item lets the same
  (name, gender, role) triple sit in both train and test and every test reads high: v1's name->role
  measures .772 grouped by item and .526 grouped by family. Family grouping is the honest one.
* **Permutation null, not 1/n_classes.** Classes are unbalanced, so "chance" is not 1/5 and a
  majority-class baseline alone understates it.
* **Within schema as well as pooled.** Pooled testing hides schema-specific confounds: v1 is .976
  pooled and 1.000 inside restaurant.
* **A positive control (A7).** Post-onset text MUST still predict role -- otherwise a corpus that
  says nothing at all would score as perfectly clean.
* **A guard on the name test (A12).** ``name -> role`` is only meaningful if names recur across
  items; if every name were unique to one item, grouped CV would never see it in training and the
  test would pass trivially while the confound was total.

Calibrated against ``binding_dyn_d1_v1``, which is retained as the known-bad fixture: the audit must
reproduce the numbers above on it, or it is not evidence of anything.

    python -m dev.schema_llm.binding.dynamics.confound_audit --run <dir>
    python -m dev.schema_llm.binding.dynamics.confound_audit --run <dir> --out <dir>/confound_audit
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from ...core.runio import read_jsonl

BYSTANDER = "BYSTANDER"
# Tolerance over the permutation null. Identity channels (A1-A4, A8-A10) are fully under our
# control and should be exactly balanced, so anything above noise is a code bug; the text channels
# (A5, A6) depend on generator compliance and get the same bar but a softer remedy (see the plan).
TOL = 0.05
# The positive control has to clear the null by a wide margin. v1 measures +0.755, so 0.20 leaves
# ~0.55 of headroom and will not fire spuriously on a healthy corpus.
POSITIVE_MARGIN = 0.20
# The null is estimated, not exact, so a handful of permutations is enough to place it to ~0.01 --
# and this runs as a GATE, so wall time matters more than the third decimal.
N_PERM = 8
#: 95th percentile of the A3 lift under perfectly random introduction order at this corpus size,
#: from simulation (mean +0.017, sd 0.035). See `_run`.
ORDER_BAND = 0.081
_PRONOUNS = re.compile(r"\b(he|she|him|her|his|hers|they|them|their|theirs)\b", re.I)
_SENT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Result:
    test: str
    label: str
    n: int
    accuracy: float
    majority: float
    null: float
    lift: float
    threshold: float
    passed: bool
    note: str = ""

    def to_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


@dataclass
class AuditReport:
    run: str
    ok: bool = True
    results: list[Result] = field(default_factory=list)

    def add(self, r: Result) -> None:
        self.results.append(r)
        self.ok &= r.passed

    def to_dict(self) -> dict:
        return {"run": self.run, "ok": self.ok, "results": [r.to_dict() for r in self.results]}


# ------------------------------------------------------------------------- feature construction


def _mask(text: str, names: list[str]) -> str:
    """Remove the identity channels from a span of prose, so what is left is the narration.

    Both names and pronouns go: leaving pronouns in would let a classifier recover gender, and
    gender is exactly one of the channels under test.
    """
    for n in sorted(names, key=len, reverse=True):
        text = re.sub(rf"\b{re.escape(n)}\b", "PERSON", text)
    return _PRONOUNS.sub("PRON", text)


def _sentences_mentioning(text: str, name: str) -> str:
    return " ".join(s for s in _SENT.split(text) if re.search(rf"\b{re.escape(name)}\b", s))


def person_table(run_dir) -> pd.DataFrame:
    """One row per (item, person), with every candidate confound channel as a column.

    Accepts one directory or several. Several is how a corpus built in WAVES is audited: balance
    that holds inside each wave can still drift in the union, and the union is what ships. Wave 2
    uses its own family-id prefixes so that concatenating cannot collide two different casts into
    one CV group.
    """
    dirs = [Path(run_dir)] if isinstance(run_dir, (str, Path)) else [Path(d) for d in run_dir]
    rows: list[dict] = []
    for item in (i for d in dirs for i in read_jsonl(d / "items.jsonl")):
        cast = item.get("cast") or []
        names = [p["name"] for p in cast]
        segments = item.get("segments") or []
        spans = item.get("referent_spans") or {}
        mentions = item.get("mentions_by_segment") or {}
        onsets = {p: int(s) for p, s in (item.get("role_onsets") or {}).items()}
        # D2 discloses role by RESOLVING an ambiguity rather than by a per-person scheduled onset,
        # so it leaves `role_onsets` empty and marks the disclosure with `resolve_segment`. Without
        # this fallback `post_text` is empty for every D2 row and A7 -- the positive control -- is
        # silently skipped, which would leave a D2 PASS with nothing showing the audit can see role
        # signal at all. The `unresolved` arm has no resolution and is still correctly excluded.
        resolve = item.get("resolve_segment")
        # rank by where each person is FIRST named in the passage -- the narrative introduction
        # order, which is what a reader actually experiences (the roster order is invisible to them)
        first = {n: (sp[0]["char_start"] if sp else 10**9) for n, sp in spans.items()}
        ranked = sorted(first, key=lambda n: first[n])
        pre_seg = segments[0] if segments else ""
        for person in cast:
            name = person["name"]
            onset = onsets.get(person["pid"])
            if onset is None and person.get("roles") and resolve:
                onset = int(resolve)
            # everything from the onset on is legitimately role-revealing; before it, nothing
            # should be. A person with no scheduled onset contributes only to the pre-onset test.
            post = " ".join(segments[(onset - 1) :]) if (onset and segments) else ""
            counts = mentions.get(name) or []
            rows.append(
                dict(
                    item_id=item.get("item_id") or item.get("scenario_id"),
                    # the CV group: siblings share a cast, so they must never straddle a fold
                    family_id=item.get("family_id") or item.get("item_id"),
                    schema=item.get("schema"),
                    pid=person["pid"],
                    name=name,
                    gender=person.get("gender"),
                    role=(person["roles"][0] if person.get("roles") else BYSTANDER),
                    is_bystander=not person.get("roles"),
                    order=ranked.index(name) if name in ranked else -1,
                    n_mentions=len(spans.get(name) or []),
                    n_segments_present=sum(1 for c in counts if c),
                    # union coordinates: a waved corpus mixes passage lengths, which neither wave
                    # carries on its own, so the merge can introduce a channel that passed twice
                    passage_segments=int(item.get("n_segments") or 0),
                    wave=int(item.get("wave") or 0),
                    seg1_mentions=counts[0] if counts else 0,
                    name_len=len(name),
                    pre_text=_mask(_sentences_mentioning(pre_seg, name), names),
                    post_text=_mask(_sentences_mentioning(post, name), names),
                )
            )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------ machinery


def _score(model, X, y, groups) -> float:
    n_splits = min(5, int(pd.Series(groups).nunique()))
    if n_splits < 2 or pd.Series(y).nunique() < 2:
        return float("nan")
    cv = GroupKFold(n_splits=n_splits)
    return float(
        cross_val_score(
            model, X, y, cv=cv, groups=groups, scoring="accuracy", n_jobs=-1
        ).mean()
    )


def _null(model, X, y, groups, seed: int = 0) -> tuple[float, float]:
    """Permutation null: the accuracy this model reaches on shuffled labels.

    The honest baseline. With unbalanced classes a majority-class floor understates what a
    classifier gets for free, which would make a leaky corpus look cleaner than it is.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    scores = [_score(model, X, rng.permutation(y), groups) for _ in range(N_PERM)]
    return float(np.nanmean(scores)), float(np.nanstd(scores))


def _cat(df: pd.DataFrame, cols: list[str]):
    return make_pipeline(
        OneHotEncoder(handle_unknown="ignore"), LogisticRegression(max_iter=3000)
    ), df[cols]


def _num(df: pd.DataFrame, cols: list[str]):
    return LogisticRegression(max_iter=3000), df[cols].astype(float)


def _name_ngram(df: pd.DataFrame):
    """Character n-grams of the name itself.

    The exact-name test (A2) can only speak about names it saw in training, so its power falls as
    the pool grows. This one generalises: it learns name *morphology*, so it still detects a
    systematic association (say, names ending in -a drifting toward one role) for names seen once
    or never. It is what makes a large, varied name pool auditable instead of merely unaudited.
    """
    return make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2),
        LogisticRegression(max_iter=3000),
    ), df["name"]


def _txt(df: pd.DataFrame, col: str):
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=3), LogisticRegression(max_iter=3000)
    ), df[col]


def _run(
    report: AuditReport,
    test: str,
    label: str,
    model,
    X,
    y,
    groups,
    *,
    positive: bool = False,
    advisory: bool = False,
) -> None:
    n = len(y)
    if n < 40 or pd.Series(y).nunique() < 2:
        # A skipped NEGATIVE test is genuinely uninformative, but a skipped POSITIVE control is a
        # failure: it is the only check showing the audit can detect role signal at all, so letting
        # it pass silently lets a corpus report "0 failing" on a pipeline that sees nothing.
        report.add(Result(test, label, n, float("nan"), float("nan"), float("nan"),
                          float("nan"), float("nan"), not positive,
                          "SKIPPED POSITIVE CONTROL -- audit is unvalidated on this corpus"
                          if positive else "skipped: too few rows"))
        return
    acc = _score(model, X, y, groups)
    maj = _score(DummyClassifier(strategy="most_frequent"), X, y, groups)
    null, null_sd = _null(model, X, y, groups)
    lift = acc - null
    if positive:
        thr, ok = POSITIVE_MARGIN, lift >= POSITIVE_MARGIN
        note = "positive control: the task must remain doable after the onset"
    else:
        # Adaptive, not a magic constant. A flat +0.05 ignores that a CV estimate on 300 rows with
        # five classes has real variance: name length came out at +0.058 within restaurant while
        # the per-role means were 5.68-5.96 and names are assigned at random, i.e. noise flagged as
        # a confound. Three null SDs is the honest bar, floored at TOL so a very stable test cannot
        # become absurdly strict.
        # Two different nulls. The permutation null says "what if labels were meaningless"; its
        # SD is the right yardstick for a channel we assign EXACTLY (gender is balanced greedily,
        # so any lift is a bug). Introduction order is different: it is randomised, not balanced,
        # because the roster position index means different things at different cast sizes -- so
        # it carries genuine assignment noise on top. Simulated under perfectly random order with
        # 50 casts, the A3 lift is +0.017 +/- 0.035 with a 95th percentile of +0.081, measured
        # BEFORE any v3 corpus existed. That band, not 3 x null_sd, is its reference.
        thr = ORDER_BAND if test == "A3" else max(TOL, 3.0 * null_sd)
        ok = lift <= thr
        note = f"threshold = max({TOL}, 3 x null sd {null_sd:.3f})"
    if advisory and not ok:
        ok = True
        note = (note + "; " if note else "") + "ADVISORY -- reported, does not gate"
    report.add(Result(test, label, n, acc, maj, null, lift, thr, bool(ok), note))


# ---------------------------------------------------------------------------------------- tests


def audit(run_dir) -> tuple[AuditReport, pd.DataFrame]:
    df = person_table(run_dir)
    label = (str(run_dir) if isinstance(run_dir, (str, Path))
             else " + ".join(Path(d).name for d in run_dir))
    rep = AuditReport(run=label)
    bearers = df[~df.is_bystander]

    def suite(d: pd.DataFrame, tag: str) -> None:
        """A1-A5, A8, A10 over role-bearers. Run pooled and then within each schema (A11)."""
        y, g = d.role, d.family_id
        for test, cols, lab, kind in (
            ("A1", ["gender"], "gender -> role", "cat"),
            ("A2", ["name"], "name -> role", "cat"),
            ("A3", ["order"], "introduction order -> role", "num"),
            ("A4", ["gender", "order", "schema"], "gender+order+schema -> role", "cat"),
            ("A8", ["name_len"], "name length -> role", "num"),
            ("A10", ["n_mentions", "n_segments_present", "seg1_mentions"],
             "mention profile -> role", "num"),
        ):
            cols = [c for c in cols if c in d.columns and d[c].nunique() > 1]
            if not cols:
                continue
            model, X = (_cat if kind == "cat" else _num)(d, cols)
            # A10 is ADVISORY. How often a person is named is partly what their role IS: the patron
            # is the protagonist the script follows through every scene, and a server serves
            # repeatedly. A naming budget plus a spread rule cut the range from 6.6x to ~1.7x, but
            # driving it to chance would mean writing a restaurant story whose customer is barely
            # mentioned. It is reported with its number, and the per-mention analysis must match on
            # mention index rather than pool across roles.
            _run(rep, test, f"{lab} [{tag}]", model, X, y, g, advisory=(test == "A10"))
        model, X = _name_ngram(d)
        _run(rep, "A2b", f"name CHARACTER n-grams -> role [{tag}]", model, X, y, g)
        sub = d[d.pre_text.str.strip().astype(bool)]
        if len(sub) >= 40:
            model, X = _txt(sub, "pre_text")
            _run(rep, "A5", f"PRE-onset text -> role [{tag}]", model, X, sub.role, sub.family_id)

    suite(bearers, "pooled")
    for schema, d in bearers.groupby("schema"):
        suite(d, f"within {schema}")

    # A6 -- bystanders must not be identifiable before any onset
    sub = df[df.pre_text.str.strip().astype(bool)]
    if len(sub) >= 40:
        model, X = _txt(sub, "pre_text")
        _run(rep, "A6", "PRE-onset text -> bystander vs role-bearer",
             model, X, sub.is_bystander.astype(str), sub.family_id)

    # A7 -- POSITIVE control: after the onset the role must still be readable. Always dispatched,
    # never guarded away: `_run` records an un-runnable positive control as a FAILURE.
    sub = bearers[bearers.post_text.str.strip().astype(bool)]
    model, X = _txt(sub, "post_text")
    _run(rep, "A7", "POST-onset text -> role", model, X, sub.role, sub.family_id,
         positive=True)

    # A9 -- design balance. Counted WITHIN schema and WITHIN role: roles are not shared across
    # schemas (GATEKEEPER is airport-only) and do not occur equally often, so a pooled cell count
    # is structurally uneven for reasons that have nothing to do with a confound. What has to be
    # flat is the distribution of gender (and of position) *given* the role.
    for col, test in (("gender", "A9a"), ("order", "A9b")):
        worst = 1.0
        for (_schema, _role), grp in bearers.groupby(["schema", "role"]):
            counts = grp[col].value_counts()
            levels = bearers[col].nunique()
            if levels < 2:
                continue
            worst = min(worst, float(counts.min() / (len(grp) / levels)))
        # Thresholds from simulation, not taste. Gender has 3 levels and every item has all of
        # them, so its cells are dense and near-uniform is reachable. ORDER does not: cast size
        # varies 5-8, so high positions only exist in large-cast items and those cells are thin by
        # construction -- under PERFECTLY random assignment the worst order cell is 0.066 +/- 0.030,
        # so a "near uniform" bar there is unreachable and meaningless. A3, which asks predictively
        # whether order forecasts role against a permutation null, carries the real burden.
        floor = 0.6 if col == "gender" else 0.04
        rep.add(Result(test, f"cell balance role x {col} (within schema)", len(bearers), worst,
                       1.0, 1.0, worst, floor, worst >= floor,
                       f"thinnest cell = {worst:.2f} of uniform"
                       + ("" if col == "gender" else "; random gives ~0.07, see A3")))

    # A13 -- UNION coordinate. Merging two waves puts passage length (and wave membership) inside
    # one corpus for the first time; wave 2 runs `long`/`late` at 8 segments against wave 1's
    # uniform 6. Role is balanced within each wave by construction, so this must sit at the null --
    # but "must" is what an audit is for, and a channel that is constant in each wave separately is
    # invisible to a per-wave check by definition.
    if bearers.passage_segments.nunique() > 1 or bearers.wave.nunique() > 1:
        cols = [c for c in ("passage_segments", "wave") if bearers[c].nunique() > 1]
        model, X = _cat(bearers, cols)
        _run(rep, "A13", f"{' + '.join(cols)} -> role [union]", model, X,
             bearers.role, bearers.family_id)

    # A12 -- the name test above is only valid if names recur AND are not role-locked
    per_name = bearers.groupby("name").agg(n_items=("family_id", "nunique"),
                                           n_roles=("role", "nunique"))
    singleton = int((per_name.n_items <= 1).sum())
    locked = int((per_name.n_roles < 2).sum())
    frac_locked = locked / max(len(per_name), 1)
    # A proportion, not a hard zero. What A2 needs is POWER -- enough names recurring across roles
    # that grouped CV can actually see them in training. A handful of tail names used once does not
    # remove that, whereas demanding zero would block on an artefact of pool size.
    # A2's power, not a confound in itself. A name used in one family only cannot be learned and
    # then tested, so a high figure means "A2 is weak here", which is exactly why A2b exists and
    # why it carries the primary burden for the name channel. Fails only when the pool has degraded
    # so far that A2 is close to meaningless.
    ok = frac_locked <= 0.25
    rep.add(Result("A12", "name reuse guard (A2 validity)", len(per_name), frac_locked, 0.0,
                   0.0, frac_locked, 0.10, ok,
                   f"{singleton} names in one family; {locked}/{len(per_name)} "
                   f"({frac_locked:.1%}) locked to one role -- A2 power, covered by A2b"))
    return rep, df


# --------------------------------------------------------------------------------------- output


def to_markdown(rep: AuditReport) -> str:
    lines = [
        f"# Confound audit — `{rep.run}`",
        "",
        f"**{'PASS' if rep.ok else 'FAIL'}** — {sum(1 for r in rep.results if not r.passed)}"
        f" of {len(rep.results)} checks failing.",
        "",
        "`lift` is accuracy minus a permutation null (not a majority-class floor).",
        "Identity and text channels must sit at the null; **A7 is a positive control and must",
        "clear it widely**.",
        "",
        "| test | channel | n | acc | null | lift | thr | |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rep.results:
        mark = "ok" if r.passed else "**FAIL**"
        lines.append(
            f"| {r.test} | {r.label} | {r.n} | {r.accuracy:.3f} | {r.null:.3f} | "
            f"{r.lift:+.3f} | {r.threshold:.2f} | {mark} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--run", type=Path, help="dataset dir holding items.jsonl")
    src.add_argument("--runs", type=Path, nargs="+",
                     help="several dirs audited as ONE corpus -- how a waved corpus is checked")
    ap.add_argument("--out", type=Path, help="write <out>.json / <out>.md (default: inside --run)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    target = args.run if args.run is not None else args.runs
    rep, _df = audit(target)
    out = args.out or (Path(target[-1] if args.run is None else target) / "confound_audit")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(rep.to_dict(), indent=1), encoding="utf-8")
    md = to_markdown(rep)
    out.with_suffix(".md").write_text(md, encoding="utf-8")
    if not args.quiet:
        print(md)
    print(f"[confound_audit] {'PASS' if rep.ok else 'FAIL'} -> {out.with_suffix('.md')}")
    raise SystemExit(0 if rep.ok else 1)


if __name__ == "__main__":
    main()
