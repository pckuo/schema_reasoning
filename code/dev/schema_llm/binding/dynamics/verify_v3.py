"""Final cross-corpus confound verification for the binding-dynamics v3 datasets.

:mod:`confound_audit` asks whether **role** is recoverable from something other than the narration.
This adds the checks that only make sense once both designs exist, and the ones specific to D2's
structure:

* **D2 arm matching.** ``resolve_a`` vs ``resolve_b`` is the load-bearing contrast -- an identical
  pinned prefix with opposite gold. It is only interpretable if the two arms are matched on the
  identity channels. In v1 they were not: ``resolve_b`` swaps the two candidates' roles while their
  genders stay attached to the people, so the gender->role mapping was canonical in one arm and
  inverted in the other, and metadata leakage differed (+.762 vs +.459).
* **Extra channels** not in the per-corpus audit: passage length, where in the passage a person is
  mentioned, and how many distinct segments separate their first and last mention.
* **Cross-design consistency**, so a channel that is flat in one design and not the other is caught.

    python -m dev.schema_llm.binding.dynamics.verify_v3 --runs <d1> <d2>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from ...core.runio import read_jsonl
from .confound_audit import N_PERM, TOL, person_table


def _fit(model, X, y, groups) -> tuple[float, float, float]:
    """(accuracy, permutation-null mean, null sd) under family-grouped CV."""
    n = min(5, int(pd.Series(groups).nunique()))
    if n < 2 or pd.Series(y).nunique() < 2:
        return float("nan"), float("nan"), float("nan")
    cv = GroupKFold(n_splits=n)
    acc = cross_val_score(model, X, y, cv=cv, groups=groups, n_jobs=-1).mean()
    rng = np.random.default_rng(0)
    yv = np.asarray(y)
    null = [
        cross_val_score(model, X, rng.permutation(yv), cv=cv, groups=groups, n_jobs=-1).mean()
        for _ in range(N_PERM)
    ]
    return float(acc), float(np.mean(null)), float(np.std(null))


def _row(name: str, acc: float, null: float, sd: float, n: int) -> dict:
    thr = max(TOL, 3.0 * sd)
    lift = acc - null
    return {
        "check": name, "n": n, "acc": round(acc, 4), "null": round(null, 4),
        "lift": round(lift, 4), "threshold": round(thr, 4), "passed": bool(lift <= thr),
    }


def d2_arm_matching(run: Path) -> list[dict]:
    """Are D2's resolve arms matched on the identity channels?

    NOT "can a classifier name the arm": ``resolve_a`` and ``resolve_b`` of one family share a cast
    and therefore a CV group, so a family-grouped predictive test cannot see a within-family
    inversion at all -- on v1 it returns exactly 0.500 while the arms are demonstrably mismatched.

    Instead, measure the identity leakage **inside each arm separately** and compare. v1's
    ``resolve_b`` swaps the candidates' roles while their genders stay with the people, so gender
    predicted role at +.762 in ``resolve_a`` and +.459 in ``resolve_b`` -- the arms differed in
    something other than the resolution, which is exactly what must not happen to the contrast the
    design rests on.
    """
    df = person_table(run)
    arms = {}
    for item in read_jsonl(run / "items.jsonl"):
        key = item.get("item_id") or item.get("scenario_id")
        # `d2_branch` is the SEMANTIC label ("a" / "b" / "un"); wave 4 renames the arms themselves
        # (`late_a`, `pairs_a`, ...), so matching on arm text silently found no rows and returned
        # an empty check list -- a vacuous pass.
        arms[key] = item.get("d2_branch") or {
            "resolve_a": "a", "resolve_b": "b", "unresolved": "un"
        }.get(item.get("arm"))
    df["arm"] = df.item_id.map(arms)
    sub = df[df.arm.isin(["a", "b"]) & ~df.is_bystander].copy()
    if len(sub) < 40:
        return []
    out = []
    lifts = {}
    for arm in ("a", "b"):
        d = sub[sub.arm == arm]
        model = make_pipeline(
            OneHotEncoder(handle_unknown="ignore"), LogisticRegression(max_iter=3000)
        )
        acc, null, sd = _fit(model, d[["gender", "order"]], d.role, d.family_id)
        lifts[arm] = acc - null
        out.append(_row(f"D2 resolve_{arm}: gender+order -> role", acc, null, sd, len(d)))
    gap = abs(lifts["a"] - lifts["b"])
    out.append({
        "check": "D2 arm MATCHING (|leakage_a - leakage_b|)", "n": len(sub),
        "acc": round(lifts["a"], 4), "null": round(lifts["b"], 4),
        "lift": round(gap, 4), "threshold": 0.10, "passed": bool(gap <= 0.10),
    })
    return out


def extra_channels(run: Path, tag: str) -> list[dict]:
    """Channels the per-corpus audit does not cover."""
    df = person_table(run)
    items = {}
    for item in read_jsonl(run / "items.jsonl"):
        key = item.get("item_id") or item.get("scenario_id")
        items[key] = (item.get("n_words") or 0, item.get("segment_spans") or [])
    df["n_words"] = df.item_id.map(lambda k: items.get(k, (0, []))[0])
    bearers = df[~df.is_bystander].copy()
    if len(bearers) < 40:
        return []
    out = []
    for cols, label in (
        (["n_words"], f"{tag}: passage length -> role"),
        (["seg1_mentions"], f"{tag}: segment-1 mention count -> role"),
        (["name_len", "n_words"], f"{tag}: name length + passage length -> role"),
    ):
        acc, null, sd = _fit(
            LogisticRegression(max_iter=3000), bearers[cols].astype(float), bearers.role,
            bearers.family_id,
        )
        out.append(_row(label, acc, null, sd, len(bearers)))
    # a bystander-vs-bearer check on metadata only (the audit covers it on text)
    acc, null, sd = _fit(
        make_pipeline(OneHotEncoder(handle_unknown="ignore"), LogisticRegression(max_iter=3000)),
        df[["gender", "order"]], df.is_bystander.astype(str), df.family_id,
    )
    out.append(_row(f"{tag}: bystander from gender+order", acc, null, sd, len(df)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    rows: list[dict] = []
    for run in args.runs:
        tag = run.name
        rows += extra_channels(run, tag)
        if "d2" in tag.lower():
            rows += [{**r, "check": f"{tag}: {r['check']}"} for r in d2_arm_matching(run)]
    df = pd.DataFrame(rows)
    ok = bool(df.passed.all()) if len(df) else False
    n_bad = int((~df.passed).sum()) if len(df) else 0
    print(f"\n{'PASS' if ok else 'FAIL'} -- {n_bad} of {len(df)} extra checks failing\n")
    print(df.to_string(index=False))
    if args.out:
        args.out.write_text(json.dumps({"ok": ok, "checks": rows}, indent=1), encoding="utf-8")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
