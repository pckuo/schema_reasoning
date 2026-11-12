"""TSP task selection, prompt framing and deterministic answer scoring.

The uploaded corpus (``results/schema_llm/upload/tsp/inference_result.jsonl``) carries no model
field, but it is Phi-4-mini-reasoning: its ``model_output`` re-tokenises to within 0.4% of the
reported ``n_output_tokens`` under the pinned checkpoint's tokenizer, and the reported
``prompt_tokens`` (594) is reproduced to one token (595) by the borrowed instruct template plus
the model card's 208-token reasoning system message. That framing is reused here so our baselines
sit on the same length scale as the corpus -- see the plan for the caveat this buys.

Scoring is arithmetic, not an LLM judge: a TSP answer is checkable. ``answered`` and ``correct``
are kept separate because Phi is mostly *wrong* on TSP (14.6% at level 1, 6.6% at level 3), so the
informative damage measure under intervention is whether the model still commits to a parseable
answer at all.
"""

from __future__ import annotations

import ast
import json
import re
import statistics
from pathlib import Path

CORPUS = Path("results/schema_llm/upload/tsp/inference_result.jsonl")
LEVELS = ("level 1", "level 2", "level 3")

# Phi-4-mini-reasoning's model-card system prompt. Verified against the corpus: with the borrowed
# instruct template this renders the level-1 prompt at 595 tokens against the corpus's reported
# 594, while no system message gives 385 and the stock reasoning template gives 608.
SYSTEM_PROMPT = (
    "Your role as an assistant involves thoroughly exploring questions through a systematic "
    "thinking process before providing the final precise and accurate solutions. This requires "
    "engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, "
    "reflection, backtracing, and iteration to develop well-considered thinking process. Please "
    "structure your response into two main sections: Thought and Solution using the specified "
    "format: <think> {Thought section} </think> {Solution section}. In the Thought section, "
    "detail your reasoning process in steps. Each step should include detailed considerations "
    "such as analysing questions, summarizing relevant findings, brainstorming new ideas, "
    "verifying the accuracy of the current steps, refining any errors, and revisiting previous "
    "steps. In the Solution section, based on various attempts, explorations, and reflections "
    "from the Thought section, systematically present the final solution that you deem correct. "
    "The Solution section should be a logical, accurate, concise expression style and detail "
    "necessary steps needed to reach the conclusion, formatted as: <think> {Thought section} "
    "</think> {Solution section}"
)

ANSWER_RE = re.compile(r"<final_answer>(.*?)</final_answer>", re.S)
# Phi-4-mini-reasoning is a math checkpoint and often commits in its native \boxed{...} instead of
# the tag the prompt asks for, e.g. \boxed{{'Path': '0->4->3->2->1->5->0', 'TotalDistance': 141}}.
# The corpus almost always uses the tag (99.6%), but our sampled traces frequently do not, and
# scoring only the tag marks a correct answer as no answer at all.
BOXED_RE = re.compile(r"\\boxed\{\s*(\{.*?\})\s*\}", re.S)
# ...and it *also* commits in prose, with the distance in a bare \boxed{195} and the tour written
# out separately ("**Path:** 0->3->5->4->6->1->2->0"), often with unicode arrows. Both arms do
# this, in different layouts, so ignoring it biases the arms against each other rather than just
# lowering every number equally.
BOXED_INT_RE = re.compile(r"\\boxed\{\s*(-?\d+)\s*\}")
DISTANCE_RE = re.compile(
    r"(?:total\s*distance|totaldistance)\W{0,12}?(-?\d+)", re.I)
ARROW_PATH_RE = re.compile(r"\d+(?:\s*(?:->|→|—>|-->)\s*\d+)+")
ARROW_SPLIT_RE = re.compile(r"\s*(?:->|→|—>|-->)\s*")
THINK_CLOSE = "</think>"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_corpus(path: Path = CORPUS, levels=LEVELS):
    rows = []
    with open(path) as handle:
        for line in handle:
            row = json.loads(line)
            if row["category"] in levels:
                rows.append(row)
    require(rows, f"no rows for {levels} in {path}")
    return rows


def select_tasks(path: Path = CORPUS, levels=LEVELS, rank: int = 0):
    """One problem per level, ordered by its own repeats' median length.

    ``rank`` indexes outward from the rank-median instance: 0 is the median, 1 the next longest,
    and so on. Deterministic and seedless. Selecting a *typical* instance matters because trace
    length is the outcome variable, and picking by rank rather than by distance-to-the-median
    avoids the exact ties a median-of-medians produces by construction (with an even problem count
    -- 100 per level -- two instances always sit equidistant from it, and the winner then depends
    on dict order).

    ``rank > 0`` is how a second instance per level is drawn for replication: adjacent ranks are
    comparably typical, so the added tasks probe generalisation across *instances* without also
    changing the length regime.
    """
    rows = load_corpus(path, levels)
    by_problem: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        by_problem.setdefault((row["category"], row["prompt_hash"]), []).append(row)

    chosen = []
    for level in levels:
        problems = [(statistics.median(r["n_output_tokens"] for r in group), key[1], group)
                    for key, group in by_problem.items() if key[0] == level]
        require(problems, f"no problems at {level}")
        problems.sort(key=lambda p: (p[0], p[1]))  # total order on (median tokens, hash)
        index = len(problems) // 2 + rank
        require(0 <= index < len(problems), f"rank {rank} outside the {level} problem list")
        repeat_median, prompt_hash, group = problems[index]
        head = group[0]
        chosen.append(dict(
            level=level, prompt_hash=prompt_hash, uid=int(head["uid"]),
            n_cities=len(head["weight_matrix"]), prompt=head["prompt"],
            weight_matrix=head["weight_matrix"], ground_truth=int(head["ground_truth"]),
            corpus_repeat_median_tokens=float(repeat_median), n_corpus_repeats=len(group),
            rank=int(rank)))
    return chosen


def tasks_by_uid(uids, path: Path = CORPUS):
    """Look up arbitrary tasks by uid, independent of which rank selected them.

    Downstream code (scoring, export, verification) must resolve whatever uids appear in the data,
    which after a replication run is a union of several ranks. Keying off ``select_tasks()`` alone
    would raise a ``KeyError`` the moment a second instance per level is added.
    """
    wanted = {int(u) for u in uids}
    found: dict[int, dict] = {}
    lengths: dict[int, list[int]] = {}
    with open(path) as handle:
        for line in handle:
            row = json.loads(line)
            uid = int(row["uid"])
            # repeats of one problem share a prompt_hash but carry distinct uids, so match the
            # hash of any requested uid to collect that problem's full repeat set
            if uid in wanted and uid not in found:
                found[uid] = dict(
                    level=row["category"], prompt_hash=row["prompt_hash"], uid=uid,
                    n_cities=len(row["weight_matrix"]), prompt=row["prompt"],
                    weight_matrix=row["weight_matrix"],
                    ground_truth=int(row["ground_truth"]))
            lengths.setdefault(uid, []).append(int(row["n_output_tokens"]))
    missing = wanted - set(found)
    require(not missing, f"uids not found in the corpus: {sorted(missing)}")
    return found


def messages(prompt: str, system: bool = True):
    """Chat turns for one task. ``system=False`` is the extraction-regime framing arm."""
    turns = [{"role": "system", "content": SYSTEM_PROMPT}] if system else []
    return turns + [{"role": "user", "content": prompt}]


def _structured(text: str):
    """A ``{'Path': ..., 'TotalDistance': ...}`` literal, from the tag or from ``\\boxed{}``."""
    blocks = ANSWER_RE.findall(text) or BOXED_RE.findall(text)
    for block in reversed(blocks):
        try:
            payload = ast.literal_eval(block.strip())
            path = [int(c) for c in ARROW_SPLIT_RE.split(str(payload["Path"])) if c.strip() != ""]
            stated = int(str(payload["TotalDistance"]).strip())
        except (ValueError, SyntaxError, KeyError, TypeError):
            continue
        return path, stated
    return None


def _prose(text: str):
    """Last-resort: a tour and a distance written out in the solution section.

    Restricted to the text after the **last** ``</think>``, which is where the model's final
    commitment lives. Searching the whole response would pick up one of the dozens of candidate
    routes it enumerates while reasoning and score that as the answer.
    """
    cut = text.rfind(THINK_CLOSE)
    if cut < 0:
        return None
    tail = text[cut + len(THINK_CLOSE):]
    paths = ARROW_PATH_RE.findall(tail)
    numbers = BOXED_INT_RE.findall(tail) or DISTANCE_RE.findall(tail)
    if not paths or not numbers:
        return None
    try:
        path = [int(c) for c in ARROW_SPLIT_RE.split(paths[-1]) if c.strip() != ""]
        return path, int(numbers[-1])
    except ValueError:
        return None


def parse_answer(text: str):
    """The model's last answer commitment, as ``(path, stated_distance)`` or ``None``.

    The **last** block is used rather than the first for two reasons: the prompt carries a format
    example (``{'Path': '0->1->2->...->N->0', ...}``) that the model quotes back inside
    ``<think>``, and an intervened continuation may state a provisional answer and then revise it.
    The model's commitment is its final one.

    Three forms are accepted, in decreasing order of explicitness: the ``<final_answer>`` tag, a
    ``\\boxed{}`` wrapping the same dict, and -- only within the solution section -- a tour and a
    distance written out in prose. The third exists because the arms use *different* prose layouts,
    so ignoring it would bias them against each other rather than lowering every number equally.
    """
    return _structured(text) or _prose(text)


def score(text: str, weight_matrix, ground_truth: int):
    """Deterministic TSP verdict for a whole response.

    ``correct`` is the strict definition -- a genuine tour whose recomputed length equals the
    optimum -- so a lucky number with an invalid path does not count. ``stated_matches_path``
    separates arithmetic slips from search failures.
    """
    n = len(weight_matrix)
    parsed = parse_answer(text)
    out = dict(answered=parsed is not None, valid_tour=False, stated_distance=None,
               tour_distance=None, stated_matches_path=False, correct=False)
    if parsed is None:
        return out
    path, stated = parsed
    out["stated_distance"] = stated
    # a tour is n+1 entries, returns to the origin, and visits every city exactly once
    closed = len(path) == n + 1 and path[0] == path[-1]
    visited = path[:-1] if closed else path
    if not (closed and sorted(visited) == list(range(n))):
        return out
    out["valid_tour"] = True
    total = sum(weight_matrix[path[i]][path[i + 1]] for i in range(n))
    out["tour_distance"] = int(total)
    out["stated_matches_path"] = stated == total
    out["correct"] = total == int(ground_truth)
    return out


def main():
    """``python -m dev.schema_llm.explore.tsp_stop_steering.tasks`` -- show the selection."""
    for task in select_tasks():
        print(f"{task['level']}  uid={task['uid']:<5} hash={task['prompt_hash'][:8]}  "
              f"cities={task['n_cities']}  gt={task['ground_truth']:<5} "
              f"corpus_median_tokens={task['corpus_repeat_median_tokens']:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
