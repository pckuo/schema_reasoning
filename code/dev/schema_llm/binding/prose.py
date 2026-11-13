"""Prose generation for binding stimuli, on ``gpt-5.6-sol`` via TRAPI.

Two prompts:

* :func:`build_seed_prompt` -- write a family's seed passage from its cast.
* :func:`build_sibling_prompt` -- rewrite the PARENT passage under exactly one scripted cast edit.
  The parent text is pinned in the prompt and the model is told to keep every sentence it can, so
  siblings differ by structure rather than by prose style (spec section 5, rule 1).

Length is the confound that comes free with headcount, so every sibling in a family is generated
against the SAME word band -- the family's largest cast -- with shorter ones padded by
role-irrelevant description rather than the longer ones trimmed (spec section 5, rule 3).
"""

from __future__ import annotations

from ..core.llm import extract_json
from ..core.spec import Schema, forbidden_set, judge_accept_words, licensed_props
from .cast import BindingSpec, Person
from .config import OPEN_KITCHEN_DESC, ROLE_ALIASES, TRAPI_MODEL
from .llm import GEN_TIMEOUT, MAX_TOKENS, call

# Both naming modes are FACTORS, assigned per family in `design`, and both must be passed
# explicitly to every builder below. v1 kept them as module constants that no prompt ever
# mentioned, so the generator quietly reverted to paraphrase and schema recognition drifted as an
# uncontrolled property of the passage. Making them required arguments is what stops that
# recurring: there is no default left to silently fall back to.
#
# `role_naming="named"` is the positive control (`spec.forbidden_set`): the passage hands over the
# role word, so `identify` reduces to extraction. Its purpose is to separate inference failure
# from judge unfairness -- if a passage says "waiter" and the answer is still marked wrong, that
# is the judge, not the model.

_PROP_RULE_NAMED = """- Refer to everyday objects by their plain, common names.
- Naming an OBJECT is not naming the script, so do not avoid an object word just because it
  hints at the setting. These are all explicitly ALLOWED and you should use the ones the
  scenes call for: <<PROPS>>. Use at least one of: <<DIAGNOSTIC>>. A reader who finishes the
  passage must be able to say what kind of situation it was; only the ROLES stay implicit."""

_PROP_RULE_ABSTRACT = """- The objects of this script must be written AROUND, never named. Refer to
  each one only by what it looks like or what it does -- for example: <<OBLIQUE>>.
- This is a positive requirement, not merely the word ban above: every beat that would normally
  turn on one of those objects must still happen, rendered obliquely but concretely. Do not drop a
  beat in order to avoid a word, and do not retreat into vagueness.
- A reader who finishes the passage should be able to follow every action without being able to
  say what kind of place it was. Here BOTH the roles and the setting stay implicit."""

_ROLE_RULE_LATENT = """- Convey each person's part ONLY through what they DO. Never state a role,
  job, or category."""

_ROLE_RULE_NAMED = """- Say outright what each role-bearing person IS. The roster below gives the
  exact word to use for each of them: attach it to that person by name the first time they appear,
  then go on showing what they do. Do not substitute a synonym.
- Bystanders get NO job word at all -- they are described only by what they are doing."""

_SEED_TEMPLATE = """You write a short narrative that follows a fixed script.
NEVER name the script itself (the kind of place or situation). Return JSON only.

PEOPLE -- every person below must appear, referred to by exactly the given name shown, written
plainly (no quotation marks). After introducing someone you may use natural pronouns when the
reference is unambiguous. Introduce NO other named or nameable people at all:
<<ROSTER>>

SCENES -- cover ALL of these, in this exact order, one short beat each: <<SCENES>>

RULES
<<ROLERULE>>
- NEVER write any of these words, or their plural / inflected forms: <<FORBIDDEN>>.
<<PROPRULE>>
- Choose a specific, concrete world (kind of place, time, weather, mood); keep details specific.
- Use NO proper noun other than the given names above -- no other personal names, and
  no place, street, city, business or brand names. Make the world specific through
  concrete detail, not through naming it.
- Target overall length: <<LENGTH>> words. If the cast is small, reach that length with
  scene-setting and sensory description -- never by inventing extra people.<<EXTRA>>

Return JSON only, in exactly this shape. Each scene "text" must be an exact substring of "prompt":
{"prompt": "<the full narrative as one string>",
 "labels": {"scenes": [{"scene": "<scene name>", "text": "<substring for that scene>"}]}}
"""

_SIBLING_TEMPLATE = """Below is a PASSAGE. Reproduce it with ONE structural change, described
under EDIT. Return JSON only.

THIS IS A COPY-AND-PATCH, NOT A REWRITE. Start from the passage text itself. Any sentence the EDIT
does not force you to touch must come through WORD FOR WORD -- same words, same punctuation, same
order. Do not improve, re-pace, re-word or re-imagine anything. Change the smallest number of
sentences that can carry the edit, and leave the rest alone. A version that reads as a fresh
retelling of the same events is WRONG, even if it is better prose.

EDIT
<<EDIT>>

THE FULL CAST OF THE RESULT -- every person below must appear, referred to by exactly the given
name shown, written plainly. Introduce NO other named or nameable people at all:
<<ROSTER>>

SCENES -- the result must still cover ALL of these, in this exact order: <<SCENES>>

RULES
<<ROLERULE>>
- NEVER write any of these words, or their plural / inflected forms: <<FORBIDDEN>>.
<<PROPRULE>>
- Use NO proper noun other than the given names above -- no other personal names, and
  no place, street, city, business or brand names. Make the world specific through
  concrete detail, not through naming it.
- Target overall length: <<LENGTH>> words -- the same band the passage was written to. Absorb the
  edit without letting the length drift.<<EXTRA>>

PASSAGE:
<<BASE>>

Return JSON only, in exactly this shape. Each scene "text" must be an exact substring of your OWN
rewritten "prompt" (not of the passage above):
{"prompt": "<the rewritten narrative>",
 "labels": {"scenes": [{"scene": "<scene name>", "text": "<substring for that scene>"}]}}
"""


def role_desc(schema: Schema, role_id: str, controls: dict) -> str:
    """The role's action description, with the 1C open-kitchen override applied when active."""
    if controls.get("open_kitchen") and schema.name == "restaurant" and role_id == "PRODUCER":
        return OPEN_KITCHEN_DESC
    return schema.roles[role_id].desc


def role_word(schema: Schema, role_id: str) -> str:
    """The canonical noun for a person role, used only when ``role_naming == "named"``.

    Drawn from ``role.forbidden`` -- the same vocabulary the latent design bans -- but filtered
    through :func:`judge_accept_words`, and that filter is load-bearing rather than defensive.
    The point of the named arm is that a wrong mark cannot be an inference failure, so the word
    handed to the generator MUST be one the judge will credit. Airport ``FRONT_AGENT`` is the case
    that proves it: its first forbidden noun is the bare "agent", which ``accept_exclude`` drops
    because it also heads ``GATEKEEPER``'s "gate agent". Naming that person "agent" in the passage
    would have produced a control item the judge marks wrong by construction.
    """
    creditable = {w.lower() for w in judge_accept_words(schema.roles[role_id])}
    for word in schema.roles[role_id].forbidden:
        if word.lower() in creditable:
            return word
    raise ValueError(
        f"{schema.name}.{role_id}: no forbidden noun survives accept_exclude, so the role cannot "
        f"be named in a way the judge would credit"
    )


def _role_phrase(schema: Schema, role_id: str, controls: dict, role_naming: str) -> str:
    desc = role_desc(schema, role_id, controls)
    if role_naming != "named":
        return desc
    return f'{desc} -- say plainly that they are the "{role_word(schema, role_id)}"'


def _roster_line(schema: Schema, person: Person, controls: dict, role_naming: str) -> str:
    if person.is_distractor:
        return (
            f"- {person.name} -- {person.licence}. Show them at the '{person.scene}' beat: "
            f"{person.beat}. {person.name} plays NO part in the sequence of events and never "
            f"takes on anyone else's job."
        )
    descs = [_role_phrase(schema, rid, controls, role_naming) for rid in person.roles]
    if len(descs) == 1:
        return f"- {person.name} -- {descs[0]}"
    joined = "; and ALSO ".join(descs)
    return (
        f"- {person.name} -- this one person does BOTH of these, and the passage must show each "
        f"of them explicitly and separately: {joined}"
    )


def roster(schema: Schema, spec: BindingSpec, role_naming: str) -> str:
    return "\n".join(_roster_line(schema, p, spec.controls, role_naming) for p in spec.cast)


def oblique_examples(schema: Schema) -> str:
    """Worked oblique phrasings for the ``abstract`` prop arm, from the schema's own variants.

    The abstract arm needs a POSITIVE instruction, not only the word ban: told merely what it may
    not write, the generator drops the beat or goes vague. These give it a concrete way to keep
    the beat while losing the noun.
    """
    parts: list[str] = []
    for rid, role in schema.roles.items():
        if role.kind == "person" or not role.forbidden:
            continue
        variants = (schema.props.get(rid) or {}).get("abstract") or []
        if variants:
            parts.append(f'{role.forbidden[0]} -> "{variants[0]}"')
    return "; ".join(parts)


def _extra_rules(schema: Schema, spec: BindingSpec) -> str:
    """Per-item rules that only apply sometimes: omission licensing and multi-role narration."""
    parts: list[str] = []
    for rid in spec.omitted_roles:
        alias = ROLE_ALIASES[spec.schema][rid]
        parts.append(
            f"- Nobody attends to the '{alias}' part of this script. Keep that beat, but let it "
            f"happen with no other person present -- show it being got through alone, or already "
            f"done. Do NOT introduce anyone, named or unnamed, to carry it out."
        )
    for person in spec.merged_people:
        parts.append(
            f"- {person.name} carries {person.n_roles} separate parts of the script. Narrate each "
            f"one as its own explicit action at its own beat (for example: greeted them at the "
            f"door, and later took what they asked for). Never leave the second part implied."
        )
    if spec.D:
        names = ", ".join(p.name for p in spec.cast if p.is_distractor)
        parts.append(
            f"- {names} are present and doing their own thing, but take NO part in the sequence "
            f"of events. Never let them serve, attend to, or stand in for anyone."
        )
    return ("\n" + "\n".join(parts)) if parts else ""


def _fill(
    template: str,
    schema: Schema,
    spec: BindingSpec,
    words: tuple[int, int],
    prop_naming: str,
    role_naming: str,
) -> str:
    if prop_naming not in ("named", "abstract"):
        raise ValueError(f"unknown prop_naming {prop_naming!r}")
    if role_naming not in ("latent", "named"):
        raise ValueError(f"unknown role_naming {role_naming!r}")
    prop_rule = _PROP_RULE_NAMED if prop_naming == "named" else _PROP_RULE_ABSTRACT
    role_rule = _ROLE_RULE_NAMED if role_naming == "named" else _ROLE_RULE_LATENT
    forbidden = sorted(forbidden_set(schema, prop_naming, role_naming))
    return (
        template.replace("<<PROPRULE>>", prop_rule)
        .replace("<<ROLERULE>>", role_rule)
        .replace("<<ROSTER>>", roster(schema, spec, role_naming))
        .replace("<<SCENES>>", " -> ".join(spec.scene_path))
        .replace("<<FORBIDDEN>>", ", ".join(forbidden))
        .replace("<<PROPS>>", ", ".join(licensed_props(schema, prop_naming)))
        .replace("<<DIAGNOSTIC>>", ", ".join(schema.diagnostic_props))
        .replace("<<OBLIQUE>>", oblique_examples(schema))
        .replace("<<LENGTH>>", f"{words[0]}-{words[1]}")
        .replace("<<EXTRA>>", _extra_rules(schema, spec))
    )


def build_seed_prompt(
    schema: Schema,
    spec: BindingSpec,
    words: tuple[int, int],
    prop_naming: str,
    role_naming: str,
) -> str:
    return _fill(_SEED_TEMPLATE, schema, spec, words, prop_naming, role_naming)


def describe_edit(schema: Schema, parent: BindingSpec, child: BindingSpec, role_naming: str) -> str:
    """The single scripted change from ``parent`` to ``child``, in plain instructions.

    Computed from the two casts rather than declared, so the instruction can never drift out of
    step with the structure the item is actually logged as having.
    """
    before = {p.pid: p for p in parent.cast}
    after = {p.pid: p for p in child.cast}
    lines: list[str] = []

    if child.edit_op == "insert_bystander":
        for pid in after.keys() - before.keys():
            p = after[pid]
            lines.append(
                f"Add exactly ONE new person, {p.name}, who {p.licence}. Show {p.name} at the "
                f"'{p.scene}' beat: {p.beat}. {p.name} takes NO part in the sequence of events, "
                f"is never served or attended to as though they were part of it, and never takes "
                f"on anyone else's job. Change nothing else -- everyone already in the passage "
                f"keeps doing exactly what they were doing, in the same words wherever possible."
            )
    elif child.edit_op == "promote":
        for pid, p in after.items():
            was, now = before[pid], p
            if was.is_distractor and not now.is_distractor:
                descs = "; and ALSO ".join(
                    _role_phrase(schema, rid, child.controls, role_naming) for rid in now.roles
                )
                lines.append(
                    f"{now.name} is already in the passage as someone who {was.licence}. In the "
                    f"rewrite {now.name} is no longer a bystander: replace that background moment "
                    f"with {now.name} taking a real part in the sequence of events -- {descs}."
                )
        lines.append(
            "Add nobody and remove nobody. Exactly the same people appear, with exactly the same "
            "names, as in the passage above. Everyone not named in this EDIT keeps doing exactly "
            "what they were doing, in the same words wherever possible."
        )
    elif child.edit_op == "merge_roles":
        gone = [before[pid] for pid in before.keys() - after.keys()]
        for pid, p in after.items():
            if pid in before and p.n_roles > before[pid].n_roles:
                descs = "; and ALSO ".join(
                    _role_phrase(schema, rid, child.controls, role_naming) for rid in p.roles
                )
                lines.append(
                    f"{p.name} now personally does ALL of the following, and each must be shown "
                    f"as its own explicit action at its own beat: {descs}."
                )
        if gone:
            names = ", ".join(g.name for g in gone)
            lines.append(
                f"Remove {names} from the story completely. {names} must not appear at all, not "
                f"even in passing, and nobody else takes over what they used to do -- the person "
                f"named above now does it."
            )
        for pid in after.keys() - before.keys():
            p = after[pid]
            lines.append(
                f"Add ONE new person, {p.name}, who {p.licence}. Show {p.name} at the "
                f"'{p.scene}' beat: {p.beat}. {p.name} takes NO part in the sequence of events."
            )
    else:
        raise ValueError(f"no edit description for {child.edit_op!r}")
    return "\n".join(f"- {line}" for line in lines)


def build_sibling_prompt(
    schema: Schema,
    parent: BindingSpec,
    child: BindingSpec,
    parent_passage: str,
    words: tuple[int, int],
    prop_naming: str,
    role_naming: str,
) -> str:
    return (
        _fill(_SIBLING_TEMPLATE, schema, child, words, prop_naming, role_naming)
        .replace("<<EDIT>>", describe_edit(schema, parent, child, role_naming))
        .replace("<<BASE>>", parent_passage)
    )


def generate(prompt: str, *, model: str = TRAPI_MODEL) -> tuple[str, dict, str]:
    """Call TRAPI and parse the item. Returns ``(passage, labels, raw)``.

    A failed parse yields ``("", {}, raw)`` so the caller can retry rather than crash a batch --
    the same contract ``pipeline`` uses.

    The timeout is far above ``run_trapi``'s 240 s default on purpose. A sibling prompt asks a
    reasoning model to reproduce a ~330-word passage verbatim inside JSON while patching it; that
    has been measured at 13 minutes on a call whose median is 24 s. At the default the call times
    out, burns all six retries, and returns an error banner that reads as a generation failure --
    which is what dropped a third of the first smoke batch.
    """
    raw = call(prompt, model=model, timeout=GEN_TIMEOUT, max_completion_tokens=MAX_TOKENS)
    try:
        data = extract_json(raw, require_key="prompt")
    except ValueError:
        return "", {}, raw
    passage = str(data.get("prompt", "")).strip()
    labels = data.get("labels") or {}
    return passage, labels if isinstance(labels, dict) else {}, raw
