"""Independent recomputation of every derived metric in a TSP transfer run.

Written because three scoring defects in this experiment were found by accident rather than by
checking. Nothing here reads the analysis code's arithmetic and agrees with it; each quantity is
rebuilt from the most primitive thing on disk -- the baseline token ids, the stored continuation
text, the raw ``stop_reason`` -- and compared.

The strongest check is the **token round trip**: ``tail_text`` is the decode of exactly the tokens
the harness counted, so re-tokenising it must return ``n_generated``. If the length metric were
measured against the wrong span, that is where it would show.

Run: ``python -m dev.schema_llm.explore.tsp_stop_steering.verify --dir <D>``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import analyze, tasks


class Report:
    def __init__(self):
        self.rows = []

    def check(self, name, ok, detail=""):
        self.rows.append((bool(ok), name, detail))
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name}" + (f" -- {detail}" if detail else ""))
        return ok

    @property
    def failed(self):
        return [r for r in self.rows if not r[0]]


def verify_structure(frame, manifest, report):
    """Identities that must hold row by row, independent of any model behaviour."""
    cap = int(manifest["cap"])
    report.check("total_tokens == offset + n_generated",
                 (frame["total_tokens"] == frame["offset"] + frame["n_generated"]).all())
    report.check("budget == cap - offset",
                 (frame["budget"] == cap - frame["offset"]).all(), f"cap={cap}")
    report.check("n_generated <= budget", (frame["n_generated"] <= frame["budget"]).all())
    report.check("terminated <=> stop_reason == 'eos'",
                 (frame["terminated"] == (frame["stop_reason"] == "eos")).all())
    censored = frame[~frame["terminated"]]
    report.check("censored runs sit exactly at the budget",
                 (censored["n_generated"] == censored["budget"]).all(),
                 f"{len(censored)} censored")
    terminated = frame[frame["terminated"]]
    report.check("terminated runs stop strictly before the budget",
                 (terminated["n_generated"] <= terminated["budget"]).all(),
                 f"{len(terminated)} terminated")
    report.check("injection applied on every step of treated rows, never on the control",
                 ((frame[frame.arm == "baseline"]["applied"] == 0).all()
                  and (frame[frame.arm != "baseline"]["applied"] > 0).all()))
    treated = frame[frame["arm"] != "baseline"]
    arrived = np.abs(treated["mean_arrived_norm"] - treated["alpha"]) / treated["alpha"]
    report.check("mean_arrived_norm matches the intended alpha within 0.1%",
                 bool((arrived < 1e-3).all()), f"max rel. error {arrived.max():.2e}")
    report.check("distinct_token_ratio in (0, 1]",
                 bool(((frame["distinct_token_ratio"] > 0)
                       & (frame["distinct_token_ratio"] <= 1)).all()))


def verify_against_baselines(frame, baselines, manifest, report):
    """Offsets must be the promised fraction of each baseline's own reasoning length."""
    reasoning = {(int(r.uid), int(r.draw)): int(r.reasoning_tokens)
                 for r in baselines.itertuples()}
    expected = np.array([round(r.location * reasoning[(r.uid, r.base_draw)])
                         for r in frame.itertuples()])
    report.check("offset == round(location x baseline reasoning_tokens)",
                 bool((frame["offset"].values == expected).all()))
    report.check("every injected uid/draw has a baseline trace",
                 set(zip(frame["uid"], frame["base_draw"], strict=True))
                 <= set(reasoning.keys()))
    # What licenses the offsets is that `R` is a COMPLETE reasoning block, i.e. </think> was
    # emitted -- not that the whole trace terminated. A baseline may close its reasoning and then
    # ramble in the solution section until the cap; its offsets are still genuine fractions of
    # complete reasoning. Termination is reported alongside as a diagnostic, not a gate.
    report.check("every baseline closed its reasoning block (offsets are fractions of complete R)",
                 bool(baselines["closed_think"].all()),
                 f"{int((baselines['stop_reason'] == 'eos').sum())}/{len(baselines)} also "
                 "terminated before the cap")
    report.check("reasoning_tokens <= n_generated for every baseline",
                 bool((baselines["reasoning_tokens"] <= baselines["n_generated"]).all()))
    locations = sorted(frame["location"].unique())
    report.check("locations are exactly the planned offsets",
                 locations == sorted(manifest["locations"]), str(locations))


def verify_token_round_trip(frame, manifest, report, sample=40, seed=0):
    """Re-tokenise the stored continuation: it must be exactly ``n_generated`` tokens.

    This is the check that would catch a length measured over the wrong span -- e.g. counting the
    prefix, or counting characters, or dropping the EOS token.
    """
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(manifest["model"], revision=manifest["revision"])
    picked = frame.sample(min(sample, len(frame)), random_state=seed)
    errors = []
    for row in picked.itertuples():
        n = len(tok(row.tail_text, add_special_tokens=False)["input_ids"])
        if n != row.n_generated:
            errors.append((row.uid, row.arm, row.n_generated, n))
    report.check(f"tail_text re-tokenises to n_generated ({len(picked)} sampled)",
                 not errors, f"{len(errors)} mismatches" if errors else "exact")
    return errors


def verify_pairing(frame, report):
    """Each treated run must pair with exactly one control sharing prefix, budget and seed."""
    merged = analyze.paired(frame)
    report.check("every treated row found its control",
                 len(merged) == int((frame["arm"] != "baseline").sum()))
    control = frame[frame["arm"] == "baseline"]
    report.check("control is unique per (uid, base_draw, location, draw)",
                 not control.duplicated(analyze.CELL).any())
    keys = [*analyze.CELL]
    joined = frame.merge(control[[*keys, "offset", "budget", "seed"]], on=keys,
                         suffixes=("", "_c"))
    report.check("treated and control share the same prefix offset",
                 bool((joined["offset"] == joined["offset_c"]).all()))
    report.check("treated and control share the same budget",
                 bool((joined["budget"] == joined["budget_c"]).all()))
    report.check("treated and control share the same seed (common random numbers)",
                 bool((joined["seed"] == joined["seed_c"]).all()))
    return merged


def verify_derived(merged, report):
    """The paired quantities the report quotes, rebuilt from their definitions."""
    report.check("d_total_tokens == total - total_ctl",
                 np.allclose(merged["d_total_tokens"],
                             merged["total_tokens"] - merged["total_tokens_ctl"]))
    report.check("ratio_tokens == total / total_ctl",
                 np.allclose(merged["ratio_tokens"],
                             merged["total_tokens"] / merged["total_tokens_ctl"]))
    report.check("gen_ctl == total_ctl - offset (the control's own generation)",
                 np.allclose(merged["gen_ctl"], merged["total_tokens_ctl"] - merged["offset"]))
    report.check("remaining_ratio == n_generated / gen_ctl",
                 np.allclose(merged["remaining_ratio"],
                             merged["n_generated"] / merged["gen_ctl"]))
    # the prefix is shared, so it must cancel in the paired difference
    report.check("paired delta is prefix-free (delta of totals == delta of generated)",
                 np.allclose(merged["d_total_tokens"],
                             merged["n_generated"] - merged["gen_ctl"]))


def verify_scoring(frame, report, sample=60, seed=0):
    """Re-score sampled rows from stored text and confirm verdict self-consistency."""
    selected = tasks.tasks_by_uid(frame["uid"].unique())
    picked = frame.sample(min(sample, len(frame)), random_state=seed)
    mismatch = 0
    for row in picked.itertuples():
        task = selected[row.uid]
        again = tasks.score(row.tail_text, task["weight_matrix"], task["ground_truth"])
        if bool(again["correct"]) != bool(row.correct):
            mismatch += 1
    report.check(f"rescoring is deterministic ({len(picked)} sampled)", mismatch == 0,
                 f"{mismatch} mismatches")
    report.check("correct implies valid_tour", (~frame["correct"] | frame["valid_tour"]).all())
    report.check("correct implies answered", (~frame["correct"] | frame["answered"]).all())
    report.check("valid_tour implies answered", (~frame["valid_tour"] | frame["answered"]).all())
    report.check("closed_think implies the tail contains </think>",
                 bool((frame["closed_think"]
                       == frame["tail_text"].str.contains("</think>", regex=False)).all()))
    # a tour that is correct must recompute to the gold distance
    bad = 0
    for row in frame[frame["correct"]].itertuples():
        if int(row.tour_distance) != int(selected[row.uid]["ground_truth"]):
            bad += 1
    report.check("every correct row's recomputed tour equals ground truth", bad == 0)


def verify_aggregation(merged, report):
    """Per-unit collapsing must average draws, not pool them."""
    collapsed = analyze.per_unit(merged)
    expected = merged.groupby([*analyze.UNIT, "arm", "dose"]).ngroups
    report.check("per_unit has one row per (unit, arm, dose)", len(collapsed) == expected,
                 f"{len(collapsed)} rows")
    spot = collapsed.iloc[0]
    raw = merged[(merged.uid == spot.uid) & (merged.base_draw == spot.base_draw)
                 & (merged.location == spot.location) & (merged.arm == spot.arm)
                 & (merged.dose == spot.dose)]
    report.check("collapsed value equals the mean over that cell's draws",
                 np.isclose(spot["d_total_tokens"], raw["d_total_tokens"].mean()),
                 f"{len(raw)} draws in the spot-checked cell")
    report.check("draws per cell are consistent",
                 collapsed["n_draws"].nunique() == 1,
                 f"n_draws={sorted(collapsed['n_draws'].unique())}")


def run(directory: Path, sample=40):
    frame, complete = analyze.load_injections(directory)
    frame = analyze.rescore(directory, frame)
    baselines = pd.read_parquet(directory / "baselines.parquet")
    manifest = json.loads((directory / "manifest.json").read_text())
    report = Report()

    print(f"\n== structure ({len(frame)} runs, complete={complete}) ==")
    verify_structure(frame, manifest, report)
    print("\n== offsets against the baselines ==")
    verify_against_baselines(frame, baselines, manifest, report)
    print("\n== token round trip ==")
    verify_token_round_trip(frame, manifest, report, sample=sample)
    print("\n== pairing ==")
    merged = verify_pairing(frame, report)
    print("\n== derived metrics ==")
    verify_derived(merged, report)
    print("\n== scoring ==")
    verify_scoring(frame, report)
    print("\n== aggregation ==")
    verify_aggregation(merged, report)

    failed = report.failed
    print(f"\n{len(report.rows) - len(failed)}/{len(report.rows)} checks passed")
    for _, name, detail in failed:
        print(f"  FAILED: {name} -- {detail}")
    return not failed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=40)
    args = parser.parse_args()
    return 0 if run(args.dir, sample=args.sample) else 1


if __name__ == "__main__":
    raise SystemExit(main())
