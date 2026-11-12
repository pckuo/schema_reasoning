"""Pure-logic tests for the TSP stop-steering transfer experiment.

Everything here is CPU-only and deterministic -- no model, no GPU. The target is the logic whose
failures are **silent**, because those are the ones that corrupt an overnight run without raising:

* answer parsing, where matching the wrong ``<final_answer>`` block turned 94.8% corpus accuracy
  into 14.6% and marked correct answers as unanswered;
* ``missing_cells``, where an off-by-one in the required set would either regenerate a finished
  matrix or silently leave holes in it;
* ``seed_of``, which must be stable **across processes** -- the builtin ``hash`` is salted per
  interpreter, so common random numbers would quietly stop being common;
* ``_pick_batch``, the inverse-CDF sampler that carries common random numbers across arms.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest
import torch

from dev.schema_llm.explore.tsp_stop_steering import resume, run, tasks

# a 4-city instance whose optimal tour is 0-1-2-3-0 = 10
MATRIX = [
    [0, 1, 5, 4],
    [1, 0, 2, 6],
    [5, 2, 0, 3],
    [4, 6, 3, 0],
]
GOLD = 10


def answer(path, distance):
    return f"<final_answer>{{'Path': '{path}', 'TotalDistance': '{distance}'}}</final_answer>"


def test_parse_answer_takes_the_last_block_not_the_first():
    """The prompt's format example is echoed inside <think>; the first match is the template."""
    template = ("<final_answer>{'Path': '0->1->2->...->N->0', "
                "'TotalDistance': 'INT_TOTAL_DISTANCE'}</final_answer>")
    text = f"<think>the format is {template} so I will use it</think>" + answer("0->1->2->3->0", 10)
    parsed = tasks.parse_answer(text)
    assert parsed == ([0, 1, 2, 3, 0], 10)


def test_parse_answer_rejects_the_unparseable_template_alone():
    template = ("<final_answer>{'Path': '0->1->2->...->N->0', "
                "'TotalDistance': 'INT_TOTAL_DISTANCE'}</final_answer>")
    assert tasks.parse_answer(f"<think>{template}</think>") is None


def test_parse_answer_prefers_a_later_revision():
    text = answer("0->3->2->1->0", 10) + " on reflection " + answer("0->1->2->3->0", 10)
    assert tasks.parse_answer(text)[0] == [0, 1, 2, 3, 0]


def test_parse_answer_accepts_phi_native_boxed_form():
    """Phi is a math checkpoint and often commits in \\boxed{...} rather than the asked-for tag."""
    text = "so the answer is \\boxed{{'Path': '0->1->2->3->0', 'TotalDistance': 10}}"
    assert tasks.parse_answer(text) == ([0, 1, 2, 3, 0], 10)


def test_parse_answer_accepts_prose_with_bare_boxed_distance():
    """The commonest real form: a bare \\boxed{N} with the tour written out separately."""
    text = ("<think>lots of searching</think>\n\n**Final Answer**\n\n\\boxed{10}\n\n"
            "**Path:** 0->1->2->3->0  \n**Total Distance:** 10")
    assert tasks.parse_answer(text) == ([0, 1, 2, 3, 0], 10)


def test_parse_answer_accepts_unicode_arrows():
    """The model writes tours with → as often as with ->."""
    text = ("<think>x</think>\nThe shortest path is 0 → 1 → 2 → 3 → 0 with a total distance "
            "of 10.\n\n\\boxed{10}")
    assert tasks.parse_answer(text) == ([0, 1, 2, 3, 0], 10)


def test_parse_answer_prose_needs_the_solution_section():
    """No closed </think> means still reasoning, so mid-trace routes are not answers."""
    text = "<think>maybe 0->1->2->3->0 giving a total distance of 10, or maybe not"
    assert tasks.parse_answer(text) is None


def test_parse_answer_prose_ignores_routes_inside_the_reasoning():
    """Dozens of candidate routes are enumerated while thinking; only the final one counts."""
    text = ("<think>try 0->3->2->1->0 total distance 99, try 0->2->1->3->0 total distance 50"
            "</think>\n**Path:** 0->1->2->3->0\n**Total Distance:** 10\n\\boxed{10}")
    assert tasks.parse_answer(text) == ([0, 1, 2, 3, 0], 10)


def test_structured_form_wins_over_prose():
    text = ("<think>x</think>\n**Path:** 0->3->2->1->0\n\\boxed{99}\n"
            + answer("0->1->2->3->0", 10))
    assert tasks.parse_answer(text)[0] == [0, 1, 2, 3, 0]


def test_prose_scoring_still_enforces_a_valid_tour():
    """The looser parser must not loosen the verdict: a bad tour is still not correct."""
    text = "<think>x</think>\n**Path:** 0->1->1->3->0\n**Total Distance:** 10\n\\boxed{10}"
    verdict = tasks.score(text, MATRIX, GOLD)
    assert verdict["answered"] and not verdict["valid_tour"] and not verdict["correct"]


def test_parse_answer_prefers_the_tag_over_a_boxed_aside():
    text = "\\boxed{{'Path': '0->3->2->1->0', 'TotalDistance': 10}}" + answer("0->1->2->3->0", 10)
    assert tasks.parse_answer(text)[0] == [0, 1, 2, 3, 0]


def test_score_accepts_the_optimal_tour():
    verdict = tasks.score(answer("0->1->2->3->0", 10), MATRIX, GOLD)
    assert verdict["answered"] and verdict["valid_tour"] and verdict["correct"]
    assert verdict["tour_distance"] == 10 and verdict["stated_matches_path"]


def test_score_rejects_a_tour_that_revisits_a_city():
    verdict = tasks.score(answer("0->1->1->3->0", 10), MATRIX, GOLD)
    assert verdict["answered"] and not verdict["valid_tour"] and not verdict["correct"]


def test_score_rejects_an_unclosed_path():
    verdict = tasks.score(answer("0->1->2->3", 10), MATRIX, GOLD)
    assert not verdict["valid_tour"] and not verdict["correct"]


def test_score_is_strict_about_optimality_not_just_the_stated_number():
    """A correct-looking number attached to a suboptimal tour must not count as correct."""
    verdict = tasks.score(answer("0->1->3->2->0", 10), MATRIX, GOLD)
    assert verdict["valid_tour"]
    assert verdict["tour_distance"] == 1 + 6 + 3 + 5
    assert not verdict["stated_matches_path"]
    assert not verdict["correct"]


def test_score_reports_no_answer_when_none_is_given():
    verdict = tasks.score("<think>I am still thinking</think>", MATRIX, GOLD)
    assert not verdict["answered"] and not verdict["correct"]


def test_missing_cells_requires_the_whole_plan_when_no_shard_exists(tmp_path):
    wanted, existing = run.missing_cells(tmp_path / "absent.parquet", draws=2)
    assert existing is None
    assert len(wanted) == len(run.LOCATIONS) * len(run.PLAN) * 2


def test_missing_cells_finds_exactly_the_widened_arm(tmp_path):
    """Widening `shuffled` from one offset to three must regenerate 6 cells, not the whole trace."""
    import pandas as pd

    old_plan = [(arm, key, dose, (0.50,) if arm == "shuffled" else where)
                for arm, key, dose, where in run.PLAN]
    rows = [dict(location=loc, arm=arm, dose=dose, draw=draw)
            for loc in run.LOCATIONS for draw in range(2)
            for arm, _key, dose, where in old_plan if loc in where]
    shard = tmp_path / "uid1_d0.parquet"
    pd.DataFrame(rows).to_parquet(shard)

    wanted, existing = run.missing_cells(shard, draws=2)
    assert len(existing) == len(rows)
    assert wanted == {(loc, "shuffled", 0.25, draw)
                      for loc in (0.25, 0.75) for draw in range(2)}


def test_missing_cells_is_empty_for_a_complete_shard(tmp_path):
    import pandas as pd

    rows = [dict(location=loc, arm=arm, dose=dose, draw=draw)
            for loc in run.LOCATIONS for draw in range(2)
            for arm, _key, dose, where in run.PLAN if loc in where]
    shard = tmp_path / "uid1_d0.parquet"
    pd.DataFrame(rows).to_parquet(shard)
    wanted, _ = run.missing_cells(shard, draws=2)
    assert wanted == set()


def test_seed_of_is_stable_across_processes():
    """``hash`` on strings is PYTHONHASHSEED-salted; common random numbers must not be."""
    expected = run.seed_of("inject", 70, 0, 0.25, 1)
    code = ("from dev.schema_llm.explore.tsp_stop_steering.run import seed_of;"
            "print(seed_of('inject', 70, 0, 0.25, 1))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={"PYTHONHASHSEED": "1", "PATH": "/usr/bin:/bin",
                             "PYTHONPATH": "code"}, cwd=".")
    assert out.returncode == 0, out.stderr
    assert int(out.stdout.strip()) == expected


def test_seed_of_separates_locations_and_draws():
    seeds = {run.seed_of("inject", 70, 0, loc, draw)
             for loc in (0.25, 0.5, 0.75) for draw in range(3)}
    assert len(seeds) == 9


def test_pick_batch_is_greedy_at_zero_temperature():
    logits = torch.tensor([[0.0, 5.0, 1.0], [7.0, 0.0, 2.0]])
    picked = resume._pick_batch(logits, temperature=0.0, top_p=1.0,
                                uniforms=torch.tensor([0.9, 0.9]))
    assert picked.tolist() == [1, 0]


def test_pick_batch_inverse_cdf_selects_by_uniform():
    """A tiny uniform takes the mode; a large one reaches into the tail."""
    logits = torch.log(torch.tensor([[0.6, 0.3, 0.1]]))
    low = resume._pick_batch(logits, 1.0, 1.0, torch.tensor([0.01]))
    high = resume._pick_batch(logits, 1.0, 1.0, torch.tensor([0.99]))
    assert low.tolist() == [0]
    assert high.tolist() == [2]


def test_pick_batch_top_p_excludes_the_tail():
    """With top_p below the tail mass, even a uniform near 1 cannot select the rare token."""
    logits = torch.log(torch.tensor([[0.9, 0.09, 0.01]]))
    picked = resume._pick_batch(logits, 1.0, 0.9, torch.tensor([0.999]))
    assert picked.item() != 2


def test_pick_batch_shares_one_uniform_across_rows_with_equal_logits():
    """Common random numbers: identical rows under the same uniform must pick identically."""
    logits = torch.log(torch.tensor([[0.5, 0.3, 0.2], [0.5, 0.3, 0.2]]))
    picked = resume._pick_batch(logits, 1.0, 1.0, torch.tensor([0.7, 0.7]))
    assert picked[0].item() == picked[1].item()


@pytest.mark.parametrize("uniform,expected", [(0.1, 0), (0.6, 1), (0.95, 2)])
def test_pick_batch_boundaries(uniform, expected):
    logits = torch.log(torch.tensor([[0.5, 0.3, 0.2]]))
    assert resume._pick_batch(logits, 1.0, 1.0, torch.tensor([uniform])).item() == expected


def test_export_columns_cover_identity_task_injection_outcome_and_text():
    """The CSV must stand alone: prompt, injection parameters, verdict and text all present."""
    from dev.schema_llm.explore.tsp_stop_steering import export

    needed = {"response_id", "kind", "uid", "ground_truth", "prompt",
              "arm", "vector", "dose", "location",
              "total_tokens", "terminated", "correct", "response"}
    assert needed <= set(export.COLUMNS)


def test_export_carries_one_whole_response_and_a_minimal_outcome_set():
    """Whole response in one column, and only the outcomes the figures actually need.

    ``answered`` and ``correct`` are both kept: they are different questions ("did it commit at
    all" vs "was it optimal") that move in opposite directions under injection. What stays out is
    the *finer* verdict breakdown (``valid_tour``, ``stated_matches_path``, stated vs recomputed
    distance) and the per-run mechanics (``seed``, ``alpha``, ``applied``, ``batch_rows``), which
    live in the parquets and would invite quoting a looser criterion than ``correct``.
    """
    from dev.schema_llm.explore.tsp_stop_steering import export

    assert "response" in export.COLUMNS
    for dropped in ("prefix_text", "generated_text", "valid_tour", "stated_matches_path",
                    "stated_distance", "tour_distance", "seed", "alpha", "applied",
                    "batch_rows", "mean_arrived_norm"):
        assert dropped not in export.COLUMNS, f"{dropped} belongs in the parquets, not the CSV"


def test_lighten_keeps_hue_and_moves_toward_white():
    """Two doses of one arm share a hue, so the figures separate them by lightness instead."""
    import matplotlib.colors

    from dev.schema_llm.explore.tsp_stop_steering import figures

    base = matplotlib.colors.to_rgb(figures.COLOR["stop"])
    assert figures.lighten(figures.COLOR["stop"], 0.0) == pytest.approx(base)
    light = figures.lighten(figures.COLOR["stop"], 0.42)
    assert all(light[i] > base[i] for i in range(3)), "lightening must move toward white"
    assert all(0.0 <= c <= 1.0 for c in light)


def test_load_injections_prefers_shards_over_a_stale_combined_table(tmp_path):
    """Shards are the source of truth; ``injections.parquet`` is derived and goes stale.

    A replication run appends new shards but does not rewrite the combined table until it
    finishes, so preferring the combined file silently hides the new traces -- which it did once.
    """
    import pandas as pd

    from dev.schema_llm.explore.tsp_stop_steering import analyze

    (tmp_path / "injections").mkdir()
    old = pd.DataFrame({"uid": [1, 1], "arm": ["stop", "stop"]})
    old.to_parquet(tmp_path / "injections" / "uid1_d0.parquet")
    # the stale combined table knows only about uid 1
    old.to_parquet(tmp_path / "injections.parquet")
    # ...but a second trace has since landed as a shard
    pd.DataFrame({"uid": [2, 2], "arm": ["stop", "stop"]}).to_parquet(
        tmp_path / "injections" / "uid2_d0.parquet")

    frame, fresh = analyze.load_injections(tmp_path)
    assert sorted(frame["uid"].unique()) == [1, 2], "new shard must be visible"
    assert not fresh, "combined table is stale and must be reported as such"


def test_export_is_self_sufficient_for_every_figure():
    """The CSV must stand alone for figure regeneration, not just for reading.

    Four of the eight panels need quantities the headline table does not: the depth panel divides
    by the control's *remaining* generation, the accuracy-vs-length panel plots tokens generated
    after injection, the injection-point panel plots the prefix, and the answer-health panel needs
    `answered` as well as `correct`. `budget` is intentionally absent -- it is `cap -
    prefix_tokens` -- but its inputs must be present.
    """
    from dev.schema_llm.explore.tsp_stop_steering import export

    columns = set(export.COLUMNS)
    # pairing keys: a treated run must find its own alpha=0 control
    assert {"uid", "base_draw", "location", "injection_draw", "arm", "dose"} <= columns
    # the three token quantities, so prefix/generated/total are all recoverable
    assert {"prefix_tokens", "generated_tokens", "total_tokens", "cap"} <= columns
    # both outcome axes, which move in opposite directions under injection
    assert {"answered", "correct", "terminated", "closed_think"} <= columns


def test_export_token_columns_are_additive():
    """total = prefix + generated must hold, or the derived budget and ratios are wrong."""
    import pandas as pd

    path = pathlib.Path("results/schema_llm/analysis/tsp_stop_steering/20260921-transfer"
                        "/responses.csv")
    if not path.exists():  # the run directory is gitignored; skip where it is absent
        pytest.skip("run directory not present")
    frame = pd.read_csv(path, usecols=["prefix_tokens", "generated_tokens", "total_tokens"])
    assert (frame["total_tokens"]
            == frame["prefix_tokens"] + frame["generated_tokens"]).all()
