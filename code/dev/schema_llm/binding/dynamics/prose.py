"""Segmented prose generation for the binding-dynamics batch, on TRAPI.

Three prompt builders, all returning the same JSON contract:

    {"segments": ["<seg 1>", ...], "labels": {"scenes": [{"scene": ..., "text": ...}]}}

``segments`` is what makes the design possible: the passage is assembled deterministically as
``" ".join(segments)``, so an information onset lands at a known character offset and a sibling can
inherit a byte-identical prefix. ``labels`` is the production contract from ``binding/prose.py``,
kept unchanged so ``verify.verify`` still checks scene set and order the same way.

The **seed of a D1 family is its LATEST-onset arm**, not its earliest. Only that arm's early
segments are free of role information, so only it can be pinned by the others. Generating from the
earliest-onset arm would be incoherent: its segment 2 already discloses the role, and no later-onset
sibling could inherit it.
"""

from __future__ import annotations

import os

from ...core.llm import extract_json
from ...core.spec import Schema, forbidden_set, judge_accept_words, licensed_props
from ..cast import BindingSpec, Person
from ..config import ROLE_ALIASES
from ..llm import MAX_TOKENS, call
from ..prose import oblique_examples
from .design import LENGTH_PROFILES, N_SEGMENTS, is_unresolved

_ORDINALS = ("ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT")

# `binding/llm.GEN_TIMEOUT` is 900 s, sized for the v1/v2 sibling prompt that asks a reasoning model
# to reproduce a ~330-word passage verbatim while patching it. This batch's prompts are much
# smaller -- a six-segment passage generates in 30-60 s -- so a 900 s ceiling does not protect
# anything here, it only means a hung connection parks a worker for a quarter of an hour before the
# outer retries even begin. One stall of exactly that kind froze the first run for 35 minutes with
# five families left. Fail fast and let the retry machinery do its job.
DYN_GEN_TIMEOUT = int(os.environ.get("BINDING_DYN_GEN_TIMEOUT", "300"))


def _shape(lo: int, hi: int) -> str:
    return ", ".join(f'"<segment {i}>"' for i in range(lo, hi + 1))


def _roster(
    schema: Schema, spec: BindingSpec, staged: dict[str, int], hide: set[str] | None = None
) -> str:
    """One line per person. Hidden people are listed WITHOUT their role -- their part is introduced
    by the schedule / ambiguity block instead, so the model cannot leak it into segment 1.

    For D2 BOTH candidates must be hidden. Listing only the focal left the other candidate's role
    description in the roster, which let the model deduce the focal by elimination: the first smoke
    read the focal as `customer` 8/8, and the ambiguity never existed."""
    hide = hide or set(staged)
    lines = []
    for person in spec.cast:
        if person.pid in hide:
            lines.append(f"    - {person.name}: see the DISCLOSURE SCHEDULE below")
        elif person.is_distractor:
            lines.append(
                f"    - {person.name}: takes NO part in the main sequence of events. "
                f"From SEGMENT 2 ONWARD (never in segment 1) show why they are present: "
                f"they {person.licence}. Show them: {person.beat}"
            )
        else:
            descs = "; ".join(schema.roles[rid].desc for rid in person.roles)
            lines.append(f"    - {person.name}: {descs}")
    return "\n".join(lines)


def _schedule_block(
    schema: Schema, spec: BindingSpec, staged: dict[str, int], role_naming: str = "latent"
) -> str:
    """The per-person onset block.

    Under ``role_naming="named"`` the onset ALSO hands over the role noun. That arm is the lexical
    ceiling: it says what a per-mention decode looks like when the role is literally in the text at
    the onset, which is the upper bound the latent arms are measured against. The word appears only
    AT the onset, never before it, so the schedule is still the manipulation."""
    lines = []
    for pid, onset in sorted(staged.items(), key=lambda kv: kv[1]):
        person = spec.person(pid)
        desc = "; ".join(schema.roles[rid].desc for rid in person.roles)
        line = (
            f"    - {person.name}: their part must become clear in SEGMENT {onset}, and NOT ONE "
            f"SEGMENT EARLIER. Their part is: {desc}\n"
            f"      Before segment {onset}, everything {person.name} does must be something ANY "
            f"person present could be doing -- a reader must not be able to guess their part. "
            f"From segment {onset} on, show the part in concrete action and keep them in it."
        )
        if role_naming == "named":
            words = "/".join(
                judge_accept_words(schema.roles[rid])[0] for rid in person.roles
            )
            line += (
                f"\n      IN SEGMENT {onset}, and only from there on, also SAY WHAT "
                f"{person.name} IS, using this exact word: \"{words}\". Attach it to their name "
                f"the first time the part is shown. Never use it before segment {onset}."
            )
        lines.append(line)
    return "\n".join(lines)


_SITUATION_RULE_NAMED = """- SEGMENT <<SCHEMAONSET>> is where the SITUATION becomes clear, and from
  there it stays clear.
- Refer to everyday objects by their plain, common names. Naming an OBJECT is not naming the
  script, so do not avoid an object word just because it hints at the setting. These are ALLOWED
  from segment <<SCHEMAONSET>> onward: <<PROPS>>. At least one of these must appear from segment
  <<SCHEMAONSET>> onward, and NONE may appear in segment 1: <<DIAGNOSTIC>>."""

_SITUATION_RULE_ABSTRACT = """- The SITUATION must never become nameable. The objects of this
  script must be written AROUND, never named: refer to each one only by what it looks like or what
  it does -- for example: <<OBLIQUE>>.
- This is a positive requirement, not merely the word ban below: every beat that would normally
  turn on one of those objects must still happen, rendered obliquely but concretely. Do not drop a
  beat in order to avoid a word, and do not retreat into vagueness.
- A reader must be able to follow every action without being able to say what kind of place it
  was."""

_SEED_TEMPLATE = """You write a short narrative that follows a fixed script.
NEVER name the script itself (the kind of place or situation). Return JSON only.

The narrative is written as <<NSEG>> segments, <<SEGWORDS>>. They are concatenated in
order into one passage, so they must read as continuous prose.

PEOPLE -- every person below must appear, referred to by exactly the given name shown, written
plainly (no quotation marks). Introduce NO other named or nameable people at all:
<<ROSTER>>

DISCLOSURE SCHEDULE -- this is the point of the task and the hard requirement:
<<SCHEDULE>>

- SEGMENT 1 must be UNINFORMATIVE FOR EVERYONE. After reading it alone a reader must not be able to
  say what kind of place or situation this is, nor what part anyone plays. Nobody may do anything
  specific to this script, and no object of this script may appear.
<<SEG1>>
- <<NAMEDPEOPLE>> must be named plainly (not only by pronoun) in EVERY segment.

SCENES -- cover ALL of these across the segments, in this exact order: <<SCENES>>

RULES
<<ROLERULE>>
- NEVER write any of these words, or their plural / inflected forms: <<FORBIDDEN>>.
<<SITUATIONRULE>>
- Use NO proper noun other than the given names above -- no other personal names, and no place,
  street, city, business or brand names.
- Do NOT write the segment number into the prose.

Return JSON only, in exactly this shape. Each scene "text" must be an exact substring of the
segments joined together with single spaces:
{"segments": [<<SHAPE>>],
 "labels": {"scenes": [{"scene": "<scene name>", "text": "<substring for that scene>"}]}}
"""

_CONTINUE_TEMPLATE = """Below is the OPENING of a narrative that follows a fixed script. Continue
it. NEVER name the script itself (the kind of place or situation). Return JSON only.

THE OPENING IS FIXED. Do not repeat it, do not alter it, and do not contradict anything in it --
everything it says must remain true of your continuation.

OPENING (segments 1 to <<NPINNED>>):
<<PREFIX>>

Write segments <<FIRSTNEW>> to <<NSEG>>, <<SEGWORDS>>, continuing directly from the
opening.

PEOPLE already in the story, and everyone who must appear:
<<ROSTER>>
Introduce NO other named or nameable people at all.

DISCLOSURE SCHEDULE -- this is the point of the task and the hard requirement:
<<SCHEDULE>>
<<EXTRA>>
- <<NAMEDPEOPLE>> must be named plainly (not only by pronoun) in EVERY segment you write.

SCENES still to be covered, in this exact order: <<SCENES>>

RULES
<<ROLERULE>>
- NEVER write any of these words, or their plural / inflected forms: <<FORBIDDEN>>.
<<SITUATIONRULE>>
- Use NO proper noun other than the names above.
- Do NOT write the segment number into the prose.

Return JSON only, in exactly this shape. Each scene "text" must be an exact substring of YOUR OWN
segments joined together with single spaces (not of the opening):
{"segments": [<<SHAPE>>],
 "labels": {"scenes": [{"scene": "<scene name>", "text": "<substring for that scene>"}]}}
"""

_AMBIGUITY_RULE = """- <<FOCAL>> and <<PARTNER>> hold these two parts BETWEEN THEM, one each:
      (A) <<ROLEA>>
      (B) <<ROLEB>>
- SEGMENTS <<AMBIG>> must leave it GENUINELY UNDECIDABLE **which of the two holds which**, with
  the evidence balanced. Everything each of them does must be equally consistent with that person
  being (A) and with them being (B). Do not hedge with vagueness and do NOT hide either of them
  behind an unnamed description ("the standing figure", "the other one") -- write concrete,
  specific action, attached to the names <<FOCAL>> and <<PARTNER>> at least twice each per segment,
  that genuinely fits both assignments. Useful devices: actions that read either way (passing an
  object BETWEEN them, either could be giving or receiving), symmetric phrasing, and withholding
  who sits and who stands.
- A careful reader must end segment <<LASTAMBIG>> unable to say which of <<FOCAL>> and <<PARTNER>>
  is which, and must feel that BOTH assignments are live rather than that the text was simply
  uninformative. Do NOT resolve it and do not hint which way it goes."""

_GARDEN_RULE = """- SEGMENT <<LEAN>> must LEAN toward <<FOCAL>> being the one who <<FOIL>> --
  suggestively, never stated. Give the reader concrete grounds to start believing it, while leaving
  every action still literally compatible with the other assignment. Do NOT assert it, do not have
  anyone say it, and do not make it certain.
- SEGMENT <<RESOLVE>> must then RESOLVE the other way, decisively and only through action:
  <<FOCAL>> is in fact the one who <<TRUTH>>, and <<PARTNER>> is the one who <<FOIL>>. The reader
  must have to REVISE the belief segment <<LEAN>> encouraged. Nothing earlier may be contradicted --
  the lean was suggestive evidence, not a statement, so every earlier sentence must remain true.
- The remaining segments carry the script to its end with each of them in that settled part."""

_RESOLVE_RULE = """- SEGMENT <<RESOLVE>> must RESOLVE the ambiguity, decisively and only through
  action: <<FOCAL>> is in fact the one who <<TRUTH>>, and <<PARTNER>> is therefore the one who
  <<FOIL>>. After that segment no careful reader can still hold the other assignment. Nothing in
  the opening may be contradicted -- it was compatible with both, and it must stay true.
- The remaining segments carry the script to its end with each of them acting in that same part."""

_UNRESOLVED_RULE = """- The ambiguity must NEVER be resolved. Carry the script to its end while
  every action of <<FOCAL>>'s and <<PARTNER>>'s stays equally consistent with either of them being
      (A) <<ROLEA>>
      (B) <<ROLEB>>
  Do not settle it in the last segment, and do not retreat into vagueness -- keep the action
  concrete and attached to the names <<FOCAL>> and <<PARTNER>>."""


def _lex(schema: Schema, prop_naming: str, role_naming: str) -> dict[str, str]:
    return {
        "<<FORBIDDEN>>": ", ".join(sorted(forbidden_set(schema, prop_naming, role_naming))),
        "<<PROPS>>": ", ".join(licensed_props(schema, prop_naming)),
        "<<DIAGNOSTIC>>": ", ".join(schema.diagnostic_props),
    }


def _continue_situation_rule(schema: Schema, ctrl: dict) -> str:
    """The props rule for a continuation. The opening has already established (or withheld) the
    situation, so ``named`` only needs the allow-list, while ``abstract`` must repeat its positive
    requirement -- otherwise the continuation quietly starts naming things the opening avoided."""
    if ctrl["prop_naming"] == "named":
        props = _lex(schema, ctrl["prop_naming"], ctrl["role_naming"])["<<PROPS>>"]
        return f"- Objects are ALLOWED and should be used where the scenes call for them: {props}."
    return _SITUATION_RULE_ABSTRACT.replace("<<OBLIQUE>>", oblique_examples(schema))


_SEG1_RULE = """- IN SEGMENT 1, introduce the people in EXACTLY this order, and give each one the
  action listed beside their name -- these are assigned, not suggestions:
<<BEATS>>
- Those actions are the ONLY thing anyone does in segment 1. In particular, in segment 1 NOBODY may
  arrive from outside, leave, stand at a post or counter, be behind a door or partition, be out of
  sight, be seated while others stand (or the reverse), wait to be attended to, or attend to anyone
  else. Everyone is simply present in the same ordinary space, doing the small thing assigned to
  them. A reader must not be able to tell from segment 1 who will turn out to matter."""


_MENTION_RULE = """- NAMING BUDGET, across the whole passage. Use each person's plain name about
  this many times in total, and refer to them by pronoun the rest of the time:
<<TARGETS>>
  This is about how often each name is WRITTEN, not about how much each person does -- someone can
  act in every scene while being named only a few times. Do not let any one person's name crowd
  out the others.
- SPREAD THOSE MENTIONS OUT: every person listed above must be named by name at least once in
  EVERY segment. Nobody may drop out of the passage and reappear, and nobody may be concentrated
  into two or three segments. If someone has little to do in a segment, name them doing something
  small and incidental rather than leaving them out."""


_ROLE_RULE_LATENT = (
    "- Convey each person's part ONLY through what they DO. Never state a role, job, or category."
)
# The lexical-ceiling arm names the role AT the onset, so the blanket ban would contradict the
# schedule. Everyone else stays latent, and the word is still banned before the onset -- which
# `verify` checks per segment.
_ROLE_RULE_NAMED = """- Everyone EXCEPT the person named in the schedule above is shown ONLY
  through what they do -- never state their role, job, or category.
- For the scheduled person, use the exact word the schedule gives, and only from their onset
  segment onward. Never use it earlier."""


def _mention_rule(spec: BindingSpec) -> str:
    """The per-person naming budget, or nothing for waves 1-2.

    v1 had no budget and the patron -- the protagonist, present in every scene -- was named 12.4
    times against a bystander's 1.9, so raw name count alone predicted role at .557.
    """
    targets = (spec.controls.get("mention_targets") or {}) if spec.controls else {}
    if not targets:
        return ""
    lines = [
        f"    - {spec.person(pid).name}: about {n} times" for pid, n in targets.items()
    ]
    return _MENTION_RULE.replace("<<TARGETS>>", "\n".join(lines))


def _seg1_rule(spec: BindingSpec) -> str:
    """The assigned-beat block, or nothing for waves 1-2.

    v1 left segment 1's content to the generator and it staged people by role regardless -- the
    patron arrived, the producer was already out back -- which a classifier recovered at .730.
    Assigning each person a role-independent beat, and banning the staging outright, removes the
    channel by construction rather than by instruction.
    """
    beats = (spec.controls.get("neutral_beats") or {}) if spec.controls else {}
    if not beats:
        return ""
    lines = [
        f"    {i}. {spec.person(pid).name} -- {beat}"
        for i, (pid, beat) in enumerate(beats.items(), start=1)
    ]
    return _SEG1_RULE.replace("<<BEATS>>", "\n".join(lines))


def _role_rule(ctrl: dict) -> str:
    return _ROLE_RULE_NAMED if ctrl["role_naming"] == "named" else _ROLE_RULE_LATENT


def _situation_rule(schema: Schema, ctrl: dict) -> str:
    """The named / abstract branch. Under ``abstract`` there is no schema onset to announce.

    Its own placeholders are resolved here rather than left to the caller's ``_fill``: the rule is
    substituted *into* the template, so anything still unfilled inside it would survive a
    single-pass replacement (and did, until a test caught it).
    """
    lex = _lex(schema, ctrl["prop_naming"], ctrl["role_naming"])
    if ctrl["prop_naming"] == "named":
        return _fill(
            _SITUATION_RULE_NAMED,
            **{
                "<<SCHEMAONSET>>": str(ctrl["schema_onset"]),
                "<<PROPS>>": lex["<<PROPS>>"],
                "<<DIAGNOSTIC>>": lex["<<DIAGNOSTIC>>"],
            },
        )
    return _SITUATION_RULE_ABSTRACT.replace("<<OBLIQUE>>", oblique_examples(schema))


def _fill(template: str, **fields: str) -> str:
    out = template
    for key, value in fields.items():
        out = out.replace(key, value)
    assert "<<" not in out, f"unfilled placeholder: {out[out.index('<<'):][:48]!r}"
    return out


def _nseg(ctrl: dict) -> int:
    """Segments for THIS item. Wave-2 `long`/`late_resolve` run at 8; everything else at 6."""
    return int(ctrl.get("n_segments") or N_SEGMENTS)


def _seg_words(ctrl: dict, lo: int = 1, hi: int | None = None) -> str:
    """The per-segment word target for segments ``lo..hi``.

    A uniform band is stated once; a non-uniform profile has to be stated PER SEGMENT, because that
    is the manipulation: wave 1 made every segment the same length, which locked the onset SEGMENT
    to a fixed fraction of the passage (0.17 / 0.33 / 0.49, sd ~.015) and left "the decode rises at
    mention 3" indistinguishable from "the decode rises at 49% of the way through".
    """
    hi = _nseg(ctrl) if hi is None else hi
    profile = LENGTH_PROFILES.get(ctrl.get("length_profile") or "uniform")
    # profiles are written for the 6-segment default; a longer passage repeats the final band
    if len(profile) < hi:
        profile = profile + (profile[-1],) * (hi - len(profile))
    bands = profile[lo - 1 : hi]
    if len(set(bands)) == 1:
        return f"{bands[0][0]}-{bands[0][1]} words each"
    parts = ", ".join(f"segment {i}: {b[0]}-{b[1]}" for i, b in enumerate(bands, start=lo))
    return f"of the following lengths -- {parts} -- and these targets matter"


def build_seed_prompt(
    schema: Schema,
    spec: BindingSpec,
    *,
    n_segments: int | None = None,
    schedule_text: str | None = None,
    heading: str = "DISCLOSURE SCHEDULE",
    hide_pids: list[str] | None = None,
) -> str:
    """The family's LATEST-onset arm (D1) or its ambiguous prefix owner (D2, segments 1-3).

    ``schedule_text`` overrides the per-person onset block -- D2's prefix substitutes the ambiguity
    requirement there, because its focal person's part is described by BOTH candidates rather than
    scheduled.
    """
    ctrl = spec.controls
    staged = {pid: int(seg) for pid, seg in ctrl.get("role_onsets", {}).items()}
    n_seg = n_segments or _nseg(ctrl)
    body = _fill(
        _SEED_TEMPLATE,
        **_lex(schema, ctrl["prop_naming"], ctrl["role_naming"]),
        **{
            "<<NSEG>>": _ORDINALS[n_seg - 1],
            "<<SEGWORDS>>": _seg_words(ctrl, 1, n_seg),
            "<<ROSTER>>": _roster(
                schema,
                spec,
                staged or {ctrl.get("focal_pid", ""): 0},
                hide=set(hide_pids) if hide_pids else None,
            ),
            "<<SCHEDULE>>": schedule_text
            or _schedule_block(schema, spec, staged, ctrl["role_naming"])
            or "    - (none)",
            "<<ROLERULE>>": _role_rule(ctrl),
            "<<SEG1>>": _seg1_rule(spec) + _mention_rule(spec),
            "<<SITUATIONRULE>>": _situation_rule(schema, ctrl),
            "<<NAMEDPEOPLE>>": " and ".join(
                spec.person(p).name for p in (hide_pids or sorted(staged) or [ctrl["focal_pid"]])
            ),
            "<<SCENES>>": ", ".join(_scenes_for(spec, n_seg)),
            "<<SHAPE>>": _shape(1, n_seg),
        },
    )
    return body.replace("DISCLOSURE SCHEDULE", heading)


def build_d2_prefix_prompt(schema: Schema, spec: BindingSpec) -> str:
    """Segments 1-3: situation clear from segment 2, focal's part undecidable across 2-3."""
    ctrl = spec.controls
    focal = spec.person(ctrl["focal_pid"]).name
    partner = spec.person(ctrl["partner_pid"]).name
    role_a, role_b = ctrl["candidate_roles"]
    ambig = ctrl["ambiguous_segments"]
    rule = _fill(
        _AMBIGUITY_RULE,
        **{
            "<<AMBIG>>": " and ".join(str(s) for s in ambig),
            "<<LASTAMBIG>>": str(max(ambig)),
            "<<FOCAL>>": focal,
            "<<PARTNER>>": partner,
            "<<ROLEA>>": schema.roles[role_a].desc,
            "<<ROLEB>>": schema.roles[role_b].desc,
        },
    )
    return build_seed_prompt(
        schema,
        spec,
        n_segments=max(ambig),
        schedule_text=rule,
        heading="AMBIGUITY REQUIREMENT",
        hide_pids=[ctrl["focal_pid"], ctrl["partner_pid"]],
    )


def build_continue_prompt(
    schema: Schema,
    spec: BindingSpec,
    pinned: list[str],
    extra_rule: str = "",
    remaining_scenes: list[str] | None = None,
    hide_pids: list[str] | None = None,
) -> str:
    """Segments ``len(pinned)+1 .. N``, with the opening pinned verbatim in the prompt.

    ``remaining_scenes`` should be the scenes the pinned opening has NOT already narrated, computed
    from the seed's own labels. Apportioning them by segment count instead once dropped a scene
    from the instructions, and the assembled passage was then genuinely missing it.
    """
    ctrl = spec.controls
    staged = {
        pid: int(seg)
        for pid, seg in ctrl.get("role_onsets", {}).items()
        if int(seg) > len(pinned)
    }
    covered = remaining_scenes if remaining_scenes is not None else _scenes_covered(
        spec, len(pinned)
    )
    return _fill(
        _CONTINUE_TEMPLATE,
        **_lex(schema, ctrl["prop_naming"], ctrl["role_naming"]),
        **{
            "<<NPINNED>>": str(len(pinned)),
            "<<FIRSTNEW>>": str(len(pinned) + 1),
            "<<NSEG>>": str(_nseg(ctrl)),
            "<<SEGWORDS>>": _seg_words(ctrl, len(pinned) + 1, _nseg(ctrl)),
            "<<PREFIX>>": " ".join(pinned),
            "<<ROSTER>>": _roster(
                schema, spec, staged, hide=set(hide_pids) if hide_pids else None
            ),
            "<<SCHEDULE>>": _schedule_block(schema, spec, staged, ctrl["role_naming"])
            or "    - (none)",
            "<<ROLERULE>>": _role_rule(ctrl),
            "<<SITUATIONRULE>>": _continue_situation_rule(schema, ctrl),
            "<<EXTRA>>": ("\n" + extra_rule) if extra_rule else "",
            "<<NAMEDPEOPLE>>": " and ".join(
                spec.person(p).name for p in (hide_pids or sorted(staged) or [spec.cast[0].pid])
            ),
            "<<SCENES>>": ", ".join(covered),
            "<<SHAPE>>": _shape(len(pinned) + 1, _nseg(ctrl)),
        },
    )


def build_d2_branch_prompt(
    schema: Schema,
    spec: BindingSpec,
    prefix: list[str],
    remaining_scenes: list[str] | None = None,
) -> str:
    ctrl = spec.controls
    focal = spec.person(ctrl["focal_pid"]).name
    partner = spec.person(ctrl["partner_pid"]).name
    role_a, role_b = ctrl["candidate_roles"]
    if ctrl.get("lean_segment") and not is_unresolved(ctrl):
        rule = _fill(
            _GARDEN_RULE,
            **{
                "<<LEAN>>": str(ctrl["lean_segment"]),
                "<<RESOLVE>>": str(ctrl["resolve_segment"]),
                "<<FOCAL>>": focal,
                "<<PARTNER>>": partner,
                "<<TRUTH>>": schema.roles[ctrl["truth_role"]].desc,
                "<<FOIL>>": schema.roles[ctrl["foil_role"]].desc,
            },
        )
    elif is_unresolved(ctrl):
        rule = _fill(
            _UNRESOLVED_RULE,
            **{
                "<<FOCAL>>": focal,
                "<<PARTNER>>": partner,
                "<<ROLEA>>": schema.roles[role_a].desc,
                "<<ROLEB>>": schema.roles[role_b].desc,
            },
        )
    else:
        rule = _fill(
            _RESOLVE_RULE,
            **{
                "<<RESOLVE>>": str(ctrl["resolve_segment"]),
                "<<FOCAL>>": focal,
                "<<PARTNER>>": partner,
                "<<TRUTH>>": schema.roles[ctrl["truth_role"]].desc,
                "<<FOIL>>": schema.roles[ctrl["foil_role"]].desc,
            },
        )
    return build_continue_prompt(
        schema,
        spec,
        prefix,
        extra_rule=rule,
        remaining_scenes=remaining_scenes,
        hide_pids=[ctrl["focal_pid"], ctrl["partner_pid"]],
    )


def _scenes_for(spec: BindingSpec, n_seg: int) -> list[str]:
    """Scenes to cover in a passage of ``n_seg`` segments. A partial passage (D2's 3-segment
    prefix) covers only its share, so the generator is not asked to finish a script it has no
    room for; ``verify`` checks the realized set and order on the ASSEMBLED passage."""
    total = _nseg(spec.controls)
    if n_seg >= total:
        return list(spec.scene_path)
    share = max(1, int(round(len(spec.scene_path) * n_seg / total)))
    return list(spec.scene_path[:share])


def _scenes_covered(spec: BindingSpec, n_pinned: int) -> list[str]:
    """Scenes still to come, apportioned by segment. Coarse on purpose: the exact split is the
    generator's business, and ``verify`` checks the realized set and order against the schema."""
    if not n_pinned:
        return list(spec.scene_path)
    per = len(spec.scene_path) / _nseg(spec.controls)
    start = int(round(per * n_pinned))
    return list(spec.scene_path[max(0, start - 1) :])


def generate(prompt: str, *, model: str) -> tuple[list[str], dict, str]:
    """Call TRAPI and parse. Returns ``(segments, labels, raw)``; ``([], {}, raw)`` on a bad parse
    so the caller retries rather than crashing a batch -- the contract ``binding/prose`` uses."""
    raw = call(prompt, model=model, timeout=DYN_GEN_TIMEOUT, max_completion_tokens=MAX_TOKENS)
    try:
        data = extract_json(raw, require_key="segments")
    except ValueError:
        return [], {}, raw
    segments = [str(s).strip() for s in data.get("segments") or []]
    if not all(segments):
        return [], {}, raw
    labels = data.get("labels") or {}
    return segments, labels if isinstance(labels, dict) else {}, raw


def role_line(schema: Schema, person: Person, schema_name: str) -> str:
    """Readable role summary for logging."""
    return ", ".join(ROLE_ALIASES[schema_name][rid] for rid in person.roles) or "bystander"
