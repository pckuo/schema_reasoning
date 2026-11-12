"""Probe tasks attached to each stimulus (A3): a question + gold.

The question is what we later pose to the *probed* model (scenario + question); it is never
shown to the generator. Gold is derived from the hidden labels -- deterministically for
``identify``, and from the schema's scene glosses for the prediction tasks.

The prediction tasks run on a *truncated* stimulus: the passage is cut at a scene boundary
sampled AFTER a controllable fraction of the scenes (``cut_frac``), so there is a genuine
"next" and enough context precedes the cut for prediction to be meaningful. The two prediction
tasks are deliberately different (plan A3) and get INDEPENDENT cuts: ``predict_next_event`` asks
what the reader should *expect* to happen next (descriptive, cut at any scene boundary), while
``predict_correct_action`` asks what the *protagonist* (the PATRON the passage follows) should
themselves *do* next (normative, agent-centric) and is therefore cut right before a scene the
protagonist acts in, so there is always a well-defined patron action to predict.
"""

from __future__ import annotations

import math
import random

from ..core.spec import Schema
from .sampler import Recipe
from .verifier import VerifyResult

DEFAULT_TASKS = ["identify", "predict_next_event", "predict_correct_action"]
# The two temporal-order prediction arms (20260824-SCENE_ORDER_TASK_FAMILIES_PLAN). They share one
# cut, so a single item yields a PAIRED story-vs-narration contrast. Opt-in only -- they never
# appear in DEFAULT_TASKS, so legacy runs are untouched.
STORY_GAP_TASK = "predict_story_gap"
TEXT_NEXT_TASK = "predict_text_next"
TEMPORAL_PREDICT_TASKS = [STORY_GAP_TASK, TEXT_NEXT_TASK]

# The character both schemas follow end-to-end (a shared core anchor, always present): the
# "protagonist" whose own next action predict_correct_action asks about.
PROTAGONIST_ROLE = "PATRON"

_IDENTIFY_Q = (
    "Read the passage. In one or two words, what is the schema, script, or scenario "
    "that best explains the sequence of events that happened in the passage? Then list "
    "EVERY person in the passage. Refer to each person by the exact name or description "
    'used in the passage (e.g. "Ana" or "the man in a flat cap") -- do NOT replace a person '
    'with their role -- then state the specific role they play (not just "staff"). Give one '
    "line per person, listing people who share a role separately. Use this format:\n"
    "Schema/scenario: {one or two words}\n"
    "<name or description from the passage>: <role>\n"
    "<name or description from the passage>: <role>\n"
    "(If there was reasoning involved, provide your reasoning before your final answer.)"
)
_PREDICT_NEXT_Q = (
    "The passage above describes a kind of everyday situation, schema, or script but stops "
    "partway through, before the sequence of events typically should have finished. Based only "
    "on what you have read, what is the single most likely event to happen next according to "
    "the schema? Describe just the immediate next event in one sentence, not the rest of the "
    "story. (If reasoning helps, give it first, then put your final answer on a line beginning "
    "'Most likely next event:'.)"
)
_PREDICT_CORRECT_Q = (
    "The passage above describes a kind of everyday situation, schema, or script but stops "
    "partway through, before the sequence of events typically should have finished. Focus on "
    "the protagonist. What is the single most appropriate, specific action they should do next "
    "for upcoming events to unfold the way they normally should according to the schema? "
    "Describe their most appropriate next action in one sentence. (If reasoning helps, give it "
    "first, then put your final answer on a line beginning 'Appropriate next action:'.)"
)
_SPOT_ERROR_Q = (
    "Read the passage. It depicts an everyday situation, schema, or script (e.g. dining out, air "
    "travel). Either the events all unfold correctly, OR exactly one thing is genuinely wrong -- "
    "an event out of its normal order, a missing or skipped event, a person doing something that "
    "isn't theirs to do, two people's parts swapped, or the wrong object used. Decide which. Do "
    "NOT assume there must be an error, and do NOT assume everything must be fine -- judge from "
    "the events themselves, and check whether they happen in their usual order.\n"
    "The following are NORMAL and are never errors: a worker acting out of sight (someone "
    "preparing the food in back, or operating the plane up front); one person handling several "
    "staff jobs in a small or quiet place; and an unnamed or briefly-mentioned background person.\n"
    "Treat the passage as complete (do not assume missing narrative). Answer in this format -- "
    "'Error: yes' or 'Error: no'; if yes, add 'Where: <the step or moment it happens>' and 'What: "
    "<what is wrong>'. (If reasoning was involved, provide it before your final answer.)"
)

_SENT_END = ".!?"

# --- the two temporal-order prediction arms -------------------------------------------------
# Both are posed on the SAME cut, so a passage yields a paired contrast between the STORY timeline
# and the NARRATION timeline. The story arm anchors on a VERBATIM QUOTE from the passage rather
# than on ``schema.scene_gloss``: the airport glosses contain forbidden role nouns ("traveler"),
# and a gloss has so far only ever reached the JUDGE, never the probed model.
_PREDICT_STORY_GAP_Q = (
    "The passage above stops partway through, and it does not describe every step of what "
    "happened. Work out the REAL chronological sequence of events -- the order things actually "
    "happened in -- regardless of the order the passage happens to recount them in.\n"
    "In that real sequence, what happened immediately {direction} this moment?\n"
    '  "{anchor}"\n'
    "That step is NOT described anywhere in the passage; work it out from the situation. "
    "Describe just that one step in one sentence. (If reasoning helps, give it first, then put "
    "your final answer on a line beginning 'Missing step:'.)"
)
_PREDICT_TEXT_NEXT_Q = (
    "The passage above stops partway through. Do NOT tell us what happens next in the real "
    "sequence of events -- tell us what the NARRATOR will recount next, given the way this "
    "passage is being told. Describe just the next thing the passage itself will describe, in "
    "one sentence. (If reasoning helps, give it first, then put your final answer on a line "
    "beginning 'Narrated next:'.)"
)
_ANCHOR_MAX_WORDS = 28

# One-line identify preamble per temporal-order form: it tells the probed model the ordering is
# INTENTIONAL (so identify stays comprehension, not error-spotting). The `canonical` form carries a
# matched neutral note so "a note exists at all" is controlled across the scene-order corpus.
_TEMPORAL_NOTES = {
    "canonical": "the passage above recounts the events in their natural chronological order",
    "reversed": "the passage above deliberately recounts the events in REVERSE chronological "
                "order (the last event first, the first event last)",
    "shuffled": "the passage above deliberately recounts the events OUT OF their natural "
                "chronological order",
    "blockswap": "the passage above tells the later part of the story first and then goes back "
                 "to the beginning",
    "flashback": "the passage above is told with flashbacks -- it jumps between earlier and "
                 "later moments on purpose",
    "forgetful": "the narrator of the passage above skips one step at first and returns to fill "
                 "it in later (nothing is actually missing)",
    "repeat": "the passage above retells one part of the story a second time on purpose",
}


def _temporal_note(recipe: Recipe) -> str:
    """A one-line preamble telling the probed model the passage's ordering is INTENTIONAL (naming
    the temporal form), so identify stays a comprehension task rather than error-spotting. Empty
    when the item has no temporal form -- legacy identify prompts are unchanged."""
    temporal = getattr(recipe, "temporal", None)
    if not temporal:
        return ""
    body = _TEMPORAL_NOTES.get(temporal.get("form"))
    if not body:
        return ""
    return (
        f"Note: {body}. This is intentional and not a mistake -- do not treat the ordering itself "
        "as an error. Work out what actually happens and in what real order, then answer.\n\n"
    )


def build_tasks(
    schema: Schema,
    recipe: Recipe,
    vr: VerifyResult,
    prompt: str,
    enabled: list[str] | None = None,
    cut_frac: float = 0.5,
) -> list[dict]:
    """Return the enabled probe tasks for one stimulus. Each task carries a ready-to-use
    ``probe_prompt`` (stimulus + question) to paste straight into the model under test.

    ``identify`` sees the full passage. The two prediction tasks get INDEPENDENT truncations
    (both after ``cut_frac`` of the verified scenes): ``predict_next_event`` cuts at any scene
    boundary, while ``predict_correct_action`` cuts right before a scene the protagonist acts in
    (see ``_sample_cut``'s ``only_scenes``), so its gold is always a well-defined patron action.
    Spot-the-error (violation items) comes later (plan A5).

    The two opt-in temporal arms (``predict_story_gap`` / ``predict_text_next``) share ONE further
    cut, so their golds are paired within an item: the story arm names an un-narrated beat by its
    chronological neighbour, the text arm names the beat the narrator will recount next. Under a
    canonical telling the two coincide by construction; under a reordering they need not."""
    enabled = enabled or DEFAULT_TASKS
    tasks: list[dict] = []
    if "identify" in enabled:
        tasks.append(_identify(schema, recipe))
    if "predict_next_event" in enabled:
        cut = _sample_cut(prompt, vr.scene_spans, recipe.seed, cut_frac)
        if cut:
            tasks.append(_predict_next_event(schema, cut))
    if "predict_correct_action" in enabled:
        # anchor the cut just before a scene the protagonist acts in; its own seed keeps the two
        # prediction tasks from collapsing onto the same boundary.
        patron_scenes = set(schema.actions.get(PROTAGONIST_ROLE, {}))
        pcut = _sample_cut(prompt, vr.scene_spans, recipe.seed + 1, cut_frac, patron_scenes)
        if pcut:
            tasks.append(_predict_correct_action(schema, recipe, pcut))
    if any(t in enabled for t in TEMPORAL_PREDICT_TASKS):
        # its own seed again, so the temporal arms never collapse onto either legacy cut
        tcut = _sample_temporal_cut(prompt, vr.scene_spans, list(schema.scenes), recipe.seed + 2)
        if tcut:
            if STORY_GAP_TASK in enabled:
                tasks.append(_predict_story_gap(schema, recipe, tcut))
            if TEXT_NEXT_TASK in enabled:
                tasks.append(_predict_text_next(schema, recipe, tcut))
    if "spot_error" in enabled:
        tasks.append(_spot_error(recipe))
    for task in tasks:
        # identify / spot_error fall back to the full passage; prediction tasks already carry
        # a truncated one.
        task.setdefault("probe_prompt", f"{prompt}\n\n{task['question']}")
    return tasks


def _identify(schema: Schema, recipe: Recipe) -> dict:
    # The canonical role nouns (which the stimulus forbids) double as judge-friendly gold
    # role labels: the model is expected to infer e.g. "waiter" even though it never appears.
    roles = [
        {"value": ref.value, "role_id": role_id, "canonical": schema.roles[role_id].forbidden}
        for role_id, refs in recipe.roles.items()
        for ref in refs
    ]
    gold = {"schema": recipe.schema, "roles": roles}
    if getattr(recipe, "temporal", None):
        # carry the temporal form + canonical/textual order so the analysis can key on both axes
        gold["temporal"] = recipe.temporal
    return {
        "name": "identify",
        "question": _temporal_note(recipe) + _IDENTIFY_Q,
        "answer_format": "open",
        "gold": gold,
    }


def _predict_next_event(schema: Schema, cut: dict) -> dict:
    """Descriptive forward model: what happens right after the cut. Gold = the next scene's
    overall-event gloss (schema.scene_gloss)."""
    scene = cut["scenes"][cut["next_idx"]]
    return {
        "name": "predict_next_event",
        "question": _PREDICT_NEXT_Q,
        "answer_format": "open",
        "probe_prompt": f"{cut['stimulus']}\n\n{_PREDICT_NEXT_Q}",
        "gold": {
            "kind": "next_event",
            "next_scene": scene,
            "answer": schema.scene_gloss.get(scene, scene.replace("_", " ")),
            "cut_char": cut["cut_char"],
            "shown_scenes": cut["scenes"][: cut["next_idx"]],
        },
    }


def _predict_correct_action(schema: Schema, recipe: Recipe, cut: dict) -> dict:
    """Normative, agent-centric: what the protagonist (PATRON) should DO next. The cut is anchored
    just before a scene the protagonist acts in (``build_tasks``), so the immediate next scene is
    that action; gold = its patron-action gloss (schema.patron_gloss). The forward search is kept
    as a safety net in case the cut ever lands before a non-protagonist beat."""
    protagonists = [r.value for r in recipe.roles.get(PROTAGONIST_ROLE, [])]
    patron_scenes = set(schema.actions.get(PROTAGONIST_ROLE, {}))
    scenes = cut["scenes"]
    idx = next(
        (i for i in range(cut["next_idx"], len(scenes)) if scenes[i] in patron_scenes),
        cut["next_idx"],
    )
    scene = scenes[idx]
    return {
        "name": "predict_correct_action",
        "question": _PREDICT_CORRECT_Q,
        "answer_format": "open",
        "probe_prompt": f"{cut['stimulus']}\n\n{_PREDICT_CORRECT_Q}",
        "gold": {
            "kind": "protagonist_action",
            "protagonist": protagonists,
            "correct_scene": scene,
            "answer": schema.patron_gloss.get(scene, scene.replace("_", " ")),
            "cut_char": cut["cut_char"],
        },
    }


def _sample_cut(
    prompt: str,
    scene_spans: list[dict],
    seed: int,
    cut_frac: float,
    only_scenes: set[str] | None = None,
) -> dict | None:
    """Pick a truncation boundary AFTER at least ``cut_frac`` of the verified scenes, always
    leaving >=1 hidden scene. Boundary ``k`` shows scenes ``[0, k)`` and makes scene ``k`` the
    "next". The char cut is backed up to the previous sentence end so the shown passage ends on
    a whole sentence. Returns None if there are fewer than two scenes (no genuine "next").
    Deterministic in ``seed``.

    ``only_scenes`` restricts the boundary to those whose NEXT scene is one of them, so the shown
    passage stops right before that kind of scene (predict_correct_action anchors on a scene the
    protagonist acts in). The ``cut_frac`` floor is relaxed if no such scene lands after it, and
    None is returned if there is none at all."""
    spans = sorted(scene_spans, key=lambda d: d["char_start"])
    n = len(spans)
    if n < 2:
        return None
    min_k = min(max(1, math.ceil(cut_frac * n)), n - 1)
    if only_scenes is None:
        k = random.Random(seed).randint(min_k, n - 1)
    else:
        cand = [j for j in range(min_k, n) if spans[j]["scene"] in only_scenes]
        if not cand:  # relax the cut_frac floor if no matching scene lands after it
            cand = [j for j in range(1, n) if spans[j]["scene"] in only_scenes]
        if not cand:
            return None
        k = random.Random(seed).choice(cand)
    cut_char = _trim_to_sentence(prompt, spans[k]["char_start"])
    return {
        "next_idx": k,
        "cut_char": cut_char,
        "scenes": [s["scene"] for s in spans],
        "stimulus": prompt[:cut_char].rstrip(),
    }


def _trim_to_sentence(text: str, cut_char: int) -> int:
    """Back the cut up to just after the previous sentence-ending punctuation, so the shown
    passage ends on a complete sentence; fall back to the raw boundary if none is found."""
    best = max((text.rfind(c, 0, cut_char) for c in _SENT_END), default=-1)
    return best + 1 if best != -1 else cut_char


# --------------------------------------------------------------- temporal prediction arms


def _anchor_quote(prompt: str, span: dict) -> str:
    """A short VERBATIM quote of the beat the story arm anchors on. Verbatim keeps the question
    free of the forbidden role lexicon (the airport ``scene_gloss`` leaks "traveler"), and it is
    unambiguous even when a beat is narrated twice (the ``repeat`` form)."""
    text = " ".join(prompt[span["char_start"] : span["char_end"]].split())
    words = text.split()
    return text if len(words) <= _ANCHOR_MAX_WORDS else " ".join(words[:_ANCHOR_MAX_WORDS]) + " ..."


def _gap_candidates(path: list[str], canonical: list[str], k: int) -> list[tuple[int, str, int]]:
    """``(target_story_idx, direction, anchor_story_idx)`` for every beat that is still hidden
    after text boundary ``k`` yet is chronologically ADJACENT to a beat already narrated -- the
    only gaps whose position the reader can pin down. ``direction`` is how the target sits
    relative to its anchor."""
    story_idx = {s: i for i, s in enumerate(canonical)}
    shown = {story_idx[s] for s in path[:k] if s in story_idx}
    hidden = [i for i in range(len(canonical)) if i not in shown]
    if not hidden or not shown:
        return []
    return [
        (u, direction, u + delta)
        for u in hidden
        for direction, delta in (("after", -1), ("before", 1))
        if u + delta in shown
    ]


def _sample_temporal_cut(
    prompt: str,
    scene_spans: list[dict],
    canonical: list[str],
    seed: int,
    only_scenes: set[str] | None = None,
) -> dict | None:
    """Cut + gap-fill target for the two temporal arms, sampled TARGET-FIRST so the gold beat is
    balanced rather than dictated by the telling.

    A purely text-driven cut makes the gold degenerate on the strong reorderings -- under
    ``reversed`` the earliest un-narrated beat is *always* story beat 0 (zero entropy, majority
    baseline 1.0). So we instead enumerate every ``(boundary, target, direction)`` that leaves the
    target hidden but chronologically adjacent to something narrated, pick the TARGET uniformly
    (preferring interior beats, which every temporal form can reach), and only then pick a
    boundary that realizes it. Returns ``None`` when no boundary hides an adjacent beat -- e.g. a
    ``repeat`` item whose every distinct beat has already been narrated.

    ``only_scenes`` restricts the target to those scenes (the protagonist-action variant).
    Deterministic in ``seed``."""
    spans = sorted(scene_spans, key=lambda d: d["char_start"])
    path = [s["scene"] for s in spans]
    n = len(canonical)
    if len(path) < 2 or n < 3:
        return None
    options: list[tuple[int, int, str, int]] = [
        (k, u, direction, anchor)
        for k in range(1, len(path))
        for u, direction, anchor in _gap_candidates(path, canonical, k)
        if only_scenes is None or canonical[u] in only_scenes
    ]
    if not options:
        return None
    rng = random.Random(seed)
    targets = sorted({u for _, u, _, _ in options})
    interior = [u for u in targets if 0 < u < n - 1]
    target = rng.choice(interior or targets)
    # Prefer a boundary where the target is NOT also the next beat narrated: that is the only
    # configuration in which the two arms can disagree, so it carries all the paired power. A
    # MONOTONE telling (canonical forward, reversed backward) admits exactly one boundary per
    # target and it always agrees -- those forms are the structural agreement controls.
    for_target = [o for o in options if o[1] == target]
    split = [o for o in for_target if path[o[0]] != canonical[target]]
    k, target, direction, anchor = rng.choice(split or for_target)

    story_idx = {s: i for i, s in enumerate(canonical)}
    shown_idx = sorted({story_idx[s] for s in path[:k] if s in story_idx})
    cut_char = _trim_to_sentence(prompt, spans[k]["char_start"])
    anchor_span = next(sp for sp in spans[:k] if sp["scene"] == canonical[anchor])
    return {
        "next_idx": k,
        "cut_char": cut_char,
        "stimulus": prompt[:cut_char].rstrip(),
        "scenes": path,
        "shown_scenes": path[:k],
        "shown_story_idx": shown_idx,
        "target_scene": canonical[target],
        "target_story_idx": target,
        "direction": direction,
        "anchor_scene": canonical[anchor],
        "anchor_story_idx": anchor,
        "anchor_quote": _anchor_quote(prompt, anchor_span),
        "text_next_scene": path[k],
        "text_next_story_idx": story_idx.get(path[k]),
        # how much of the story-FUTURE relative to the target has already been narrated: the
        # covariate that separates "inferred it" from "read it off the page".
        "n_seen_future": sum(1 for i in shown_idx if i > target),
        "dissociates": canonical[target] != path[k],
    }


def _story_gap_question(cut: dict) -> str:
    return _PREDICT_STORY_GAP_Q.format(direction=cut["direction"], anchor=cut["anchor_quote"])


def _temporal_gold(recipe: Recipe, cut: dict, extra: dict) -> dict:
    """Shared gold fields for both temporal arms: the cut, what was narrated, the paired other
    arm's answer, and the leak covariate -- so every analysis can stratify without recomputing."""
    return {
        "temporal_form": (getattr(recipe, "temporal", None) or {}).get("form", "canonical"),
        "cut_char": cut["cut_char"],
        "shown_scenes": cut["shown_scenes"],
        "shown_story_idx": cut["shown_story_idx"],
        "n_seen_future": cut["n_seen_future"],
        "dissociates": cut["dissociates"],
        **extra,
    }


def _predict_story_gap(schema: Schema, recipe: Recipe, cut: dict) -> dict:
    """STORY timeline: name the un-narrated beat that sits immediately before/after a quoted
    anchor beat in real time. Gold = the target scene's overall-event gloss."""
    question = _story_gap_question(cut)
    target = cut["target_scene"]
    return {
        "name": STORY_GAP_TASK,
        "question": question,
        "answer_format": "open",
        "probe_prompt": f"{cut['stimulus']}\n\n{_temporal_note(recipe)}{question}",
        "gold": _temporal_gold(
            recipe,
            cut,
            {
                "kind": "story_gap",
                "target_scene": target,
                "target_story_idx": cut["target_story_idx"],
                "direction": cut["direction"],
                "anchor_scene": cut["anchor_scene"],
                "anchor_quote": cut["anchor_quote"],
                "answer": schema.scene_gloss.get(target, target.replace("_", " ")),
                "text_next_scene": cut["text_next_scene"],
            },
        ),
    }


def _predict_text_next(schema: Schema, recipe: Recipe, cut: dict) -> dict:
    """NARRATION timeline: name the beat the narrator will recount next, given the declared
    telling. Gold = the next beat in ``scene_path``, by its overall-event gloss."""
    nxt = cut["text_next_scene"]
    return {
        "name": TEXT_NEXT_TASK,
        "question": _PREDICT_TEXT_NEXT_Q,
        "answer_format": "open",
        "probe_prompt": f"{cut['stimulus']}\n\n{_temporal_note(recipe)}{_PREDICT_TEXT_NEXT_Q}",
        "gold": _temporal_gold(
            recipe,
            cut,
            {
                "kind": "text_next",
                "next_scene": nxt,
                "next_story_idx": cut["text_next_story_idx"],
                "answer": schema.scene_gloss.get(nxt, nxt.replace("_", " ")),
                "story_gap_target": cut["target_scene"],
            },
        ),
    }


def _spot_error(recipe: Recipe) -> dict:
    """Violation-detection task (plan A5), posed on the full passage of BOTH canonical items
    (gold = no error, the base-rate control) and violation items (gold = the injected type + the
    scene(s) a correct 'where' answer should point at). Gold comes from ``recipe.violation``."""
    v = recipe.violation
    if not v:
        gold = {"has_violation": False, "type": None, "scenes": []}
    else:
        gold = {"has_violation": True, "type": v["type"], "scenes": _violation_scenes(v)}
    return {
        "name": "spot_error",
        "question": _SPOT_ERROR_Q,
        "answer_format": "open",
        "gold": gold,
    }


def _violation_scenes(v: dict) -> list[str]:
    """The scene name(s) a correct 'where is the error' answer should localize to, per type."""
    if v["type"] == "wrong_order":
        return list(v["scenes"])
    if v["type"] == "role_swap":
        return [v["from_scene"]]
    return [v["scene"]]  # missing_scene / wrong_action / prop_mismatch
