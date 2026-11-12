"""First-pass latent geometry of the identify task (SCHEMA_PROBING_PLAN Part B, H1/H2).

Reads the residual-stream captures written by :mod:`capture`
(``<run>/activations/<slug>/<sid>__identify/{layer_L.npy, meta.json}``) and asks two questions
about the geometry of the identify representation:

- **H2 (role geometry).** Does a per-referent activation vector -- the residual mean-pooled over
  that referent's aligned stimulus tokens -- DECODE the latent role better than a bag-of-words
  lexical baseline over the referent's surface mentions? A logistic-regression probe (grouped CV,
  grouped by scenario) is compared against a word-count probe and a most-frequent-class chance
  floor; the activation probe must beat the lexical baseline to support a *non-lexical* role
  representation (the plan's core control: "beat a lexical baseline").
- **H1 (state structure).** Do per-token residuals cluster into the schema's scene structure beyond
  surface token identity? KMeans over a sample of scene-labeled stimulus-token residuals is scored
  by ARI / NMI against the native scene labels, next to a token-identity partition baseline.

Both are broken down by the ``correct`` / ``error`` split (joined from the judged
``scores.<tag>.jsonl`` when available), the ``role_naming`` difficulty axis (``latent`` vs the
``named`` positive control), and schema. Controls / ``role_naming`` come from ``items.jsonl``
(always present); correctness comes from the judge, so ``--scores-tag`` is optional -- without it
the geometry still runs, just without the correct/error contrast.

Examples
--------
    python -m dev.schema_llm.geometry.latents \
        --run results/schema_llm/data/identify_720/restaurant_main \
              results/schema_llm/data/identify_720/airport_main \
        --slug phi-4 --layer 17 --scores-tag phi-4 \
        --out results/schema_llm/data/identify_720/analysis/phi-4
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    f1_score,
    normalized_mutual_info_score,
)
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..core.runio import read_jsonl

REPO_ROOT = Path(__file__).resolve().parents[4]

# Minimum sample / class sizes for a decoding cell to be trustworthy (else it is reported as
# skipped rather than as a noisy accuracy).
_MIN_ROWS = 40
_MIN_PER_CLASS = 5
_CV_SPLITS = 5

# Lexical baselines: n-gram range for the BoW probes; context window = tokens each side of a mention
# (prompt tokens only) for the stronger context_ngram baseline; min doc-freq drops hapax context
# terms (further curbs scenario-unique leakage on top of the grouped CV). context_ngram_left is the
# same over a preceding-only window (mention + left tokens), matched to causal info the residual
# at the mention had -- the symmetric window can peek at future tokens the model could not.
_LEX_NGRAM = (1, 2)
_CTX_WINDOW = 12
_CTX_MIN_DF = 2


# --------------------------------------------------------------------------- data


@dataclass
class Referent:
    """One mention-group of an entity: its role label + the residual mean-pooled over its aligned
    stimulus tokens, plus the metadata the breakdowns split on. ``group`` is the run-qualified id
    (``run:scenario_id``) used for CV grouping, so pooling runs whose scenario ids overlap
    (systematic_v2 / _x2 / the new reps all restart ids at 0000) never share a CV group."""

    scenario_id: str
    group: str
    schema: str
    role_id: str
    text: str  # surface mentions, for the lexical baseline
    context: str  # +/- windowed surrounding prompt tokens, for the context n-gram baseline
    context_left: str  # left-only (causal) window: preceding tokens + mention, no forward peek
    vector: np.ndarray  # (hidden,)
    position: float  # mean relative token position of the referent's mentions (0..1)
    role_naming: str
    prop_naming: str
    detail_level: str | None
    distractor_level: str | None
    correct: bool | None
    schema_correct: bool | None
    role_correct: bool | None  # was THIS entity's role identified correctly (per-entity verdict)
    role_verdict: str | None = None  # fine verdict: correct/wrong/vague/missing/recovered
    role_stated: str | None = None  # the model's stated role string (for the wrong-role analysis)


@dataclass
class SceneToken:
    """One scene-labeled stimulus token: its residual + native scene + surface token string, plus
    its relative position in the prompt (token index / n_prompt) for the S5 position control.
    ``group`` is the run-qualified scenario id used for CV grouping (mirrors ``Referent.group``).
    ``correct`` is the judged outcome of the ITEM the token came from -- scene tokens are stimulus
    tokens, so there is no per-token verdict the way a referent has its own role verdict."""

    scenario_id: str
    group: str
    schema: str
    scene: str
    token: str  # surface token string, for the token-identity baseline
    vector: np.ndarray  # (hidden,)
    position: float  # relative token position in the prompt (0..1)
    story_index: int | None = None  # scene's canonical index in schema.scenes (STORY order)
    text_ordinal: int | None = None  # scene beat's ordinal in TEXTUAL order (char_start rank)
    correct: bool | None = None  # scenario-level judged outcome; None when unjudged


def _read_jsonl(path: Path) -> list[dict]:
    return read_jsonl(path)


def _item_meta(run: Path) -> dict[str, dict]:
    """scenario_id -> generation controls (role_naming/prop_naming/...) from the run's items.jsonl.
    role_naming defaults to ``latent`` when absent (the generator default). The binding generator
    nests the controls under an ``items.jsonl`` ``controls`` block (top level is null there), so
    each control falls back to that block before defaulting."""
    out: dict[str, dict] = {}
    for it in _read_jsonl(run / "items.jsonl"):
        ctrl = it.get("controls") or {}

        def _c(key: str, item: dict = it, nested: dict = ctrl):
            v = item.get(key)
            return nested.get(key) if v is None else v

        out[it["scenario_id"]] = {
            "role_naming": _c("role_naming") or "latent",
            "prop_naming": _c("prop_naming"),
            "detail_level": _c("detail_level"),
            "distractor_level": _c("distractor_level"),
            "schema": it.get("schema"),
        }
    return out


def _role_label(entry: str | list | None) -> str | None:
    """The single role class for one ``meta.json`` ``role_map`` entry, or None when the referent has
    no well-defined one. Most corpora map ``value -> "ROLE"``; the binding corpora map
    ``value -> [roles]`` because a person may hold several roles at once (multiplexing). An empty
    list is a DISTRACTOR (present, named, holds no role) and a list of 2+ is a MULTIPLEXED person,
    which has no single label -- both are dropped from the role decode rather than given an
    invented class."""
    if isinstance(entry, list):
        return str(entry[0]) if len(entry) == 1 else None
    return entry or None


def _scores(run: Path, tag: str | None, task: str = "identify") -> dict[str, dict]:
    """scenario_id -> judged outcome for the correct/error split of ``task``; empty when the run is
    unjudged for this tag. identify carries per-entity role verdicts; the prediction tasks a plain
    binary ``correct``; spot_error exposes detected/localized/type_correct and defines the effective
    ``correct`` as ``detected`` -- which is correctness-aligned in BOTH conditions
    (judge._spot_error_prompt: on a violation item detected=True == "said there IS an error"; on a
    canonical item detected=True == "said everything is fine"), i.e. the model made the right
    presence call. (Repo lesson: do NOT read detected as single-polarity "flagged an error".)
    The binding rubric emits ``answers.people[]`` (per-person, one verdict per role part) instead of
    ``answers.roles[]``; both are read into the same ``role_verdicts`` / ``role_stated`` maps."""
    if tag is None:
        return {}
    from ..core.runio import scores_path

    out: dict[str, dict] = {}
    for r in _read_jsonl(scores_path(run, tag)):
        if r.get("task") != task:
            continue
        if task == "spot_error":
            out[r["scenario_id"]] = {
                "correct": r.get("detected"),
                "schema_correct": None,
                "detected": r.get("detected"),
                "localized": r.get("localized"),
                "type_correct": r.get("type_correct"),
                "role_verdicts": {},
                "role_stated": {},
            }
            continue
        roles = (r.get("answers") or {}).get("roles", [])
        people = (r.get("answers") or {}).get("people", [])
        if people and not roles:
            # Binding rubric (answers.people[]): person-grained, with one verdict per role PART
            # because a multiplexed person holds several. Only a single-part person has a
            # well-defined per-referent verdict, which is exactly the set _role_label admits.
            verdicts = {p.get("value"): (p.get("parts") or [None])[0]
                        for p in people
                        if p.get("value") is not None and len(p.get("parts") or []) == 1}
            stated = {p.get("value"): (p.get("parts_as_stated") or [None])[0]
                      for p in people
                      if p.get("value") is not None and len(p.get("parts_as_stated") or []) == 1}
        else:
            verdicts = {role.get("value"): role.get("verdict")
                        for role in roles if role.get("value") is not None}
            stated = {role.get("value"): role.get("role_as_stated")
                      for role in roles if role.get("value") is not None}
        out[r["scenario_id"]] = {
            "correct": r.get("correct"),
            "schema_correct": r.get("schema_correct"),
            "role_verdicts": verdicts,
            "role_stated": stated,
        }
    return out


def load_captures(
    run_dirs: list[Path],
    slug: str,
    layer: int,
    scores_tag: str | None = None,
    task: str = "identify",
    max_scene_tokens_per_probe: int = 24,
    seed: int = 0,
) -> tuple[list[Referent], list[SceneToken]]:
    """Load per-referent role vectors and a per-probe sample of scene-labeled token vectors across
    the given run dirs, joining generation controls (items.jsonl) and, when ``scores_tag`` is set,
    the judged correct/error label. ``task`` selects which captures to read
    (``<sid>__<task>``) and how the correct/error label is derived (see :func:`_scores`). The
    ``layer_{layer}.npy`` files are memory-mapped and only the needed token rows are copied out, so
    peak memory stays small."""
    rng = np.random.default_rng(seed)
    referents: list[Referent] = []
    scene_tokens: list[SceneToken] = []
    for run in run_dirs:
        adir = run / "activations" / slug
        if not adir.exists():
            print(f"[warn] {adir} missing; skipping")
            continue
        items = _item_meta(run)
        scores = _scores(run, scores_tag, task)
        for pdir in sorted(adir.glob(f"*__{task}")):
            meta_path = pdir / "meta.json"
            npy_path = pdir / f"layer_{layer}.npy"
            if not (meta_path.exists() and npy_path.exists()):
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            resid = np.load(npy_path, mmap_mode="r")
            sid = meta["scenario_id"]
            group = f"{run.parent.name}/{run.name}:{sid}"
            schema = meta.get("schema") or items.get(sid, {}).get("schema")
            ctrl = items.get(sid, {})
            sc = scores.get(sid, {})
            verdicts = sc.get("role_verdicts", {})
            role_map = meta.get("role_map", {})
            tokens = meta.get("tokens") or []
            n_prompt = int(meta.get("n_prompt_tokens") or len(tokens))
            denom = max(n_prompt - 1, 1)

            for value, spans in meta.get("referent_token_spans", {}).items():
                role_id = _role_label(role_map.get(value))
                if not role_id:
                    continue
                rows, texts, ctx, ctxl, poss = [], [], [], [], []
                for sp in spans:
                    if sp.get("aligned") and sp.get("tok_start") is not None:
                        rows.append(np.asarray(resid[sp["tok_start"] : sp["tok_end"]], np.float32))
                        texts.append(str(sp.get("text", value)))
                        lo = max(0, sp["tok_start"] - _CTX_WINDOW)
                        hi = min(n_prompt or sp["tok_end"], sp["tok_end"] + _CTX_WINDOW)
                        ctx.append(" ".join(tokens[lo:hi]))
                        ctxl.append(" ".join(tokens[lo : sp["tok_end"]]))
                        poss.extend(i / denom for i in range(sp["tok_start"], sp["tok_end"]))
                if not rows:
                    continue
                vd = verdicts.get(value)
                role_ok = vd in ("correct", "recovered") if vd is not None else None
                referents.append(
                    Referent(
                        scenario_id=sid,
                        group=group,
                        schema=schema,
                        role_id=role_id,
                        text=f"{value} " + " ".join(texts),
                        context=" ".join(ctx),
                        context_left=" ".join(ctxl),
                        vector=np.concatenate(rows, axis=0).mean(axis=0),
                        position=float(np.mean(poss)),
                        role_naming=ctrl.get("role_naming", "latent"),
                        prop_naming=ctrl.get("prop_naming"),
                        detail_level=ctrl.get("detail_level"),
                        distractor_level=ctrl.get("distractor_level"),
                        correct=sc.get("correct"),
                        schema_correct=sc.get("schema_correct"),
                        role_correct=role_ok,
                        role_verdict=vd,
                        role_stated=sc.get("role_stated", {}).get(value),
                    )
                )

            scene_tokens.extend(
                _sample_scene_tokens(meta, resid, schema, sid, group, rng,
                                     max_scene_tokens_per_probe, sc.get("correct"))
            )
    return referents, scene_tokens


@cache
def _scene_story_order(schema_name: str) -> tuple[str, ...]:
    """The schema's canonical scene order (``schema.scenes``), cached; () for an unknown schema
    (e.g. a synthetic-fixture name), so ``story_index`` falls back to None."""
    from ..core.spec import load_schema

    try:
        return tuple(load_schema(schema_name).scenes)
    except (FileNotFoundError, KeyError):
        return ()


def _sample_scene_tokens(
    meta: dict, resid: np.ndarray, schema: str, sid: str, group: str,
    rng: np.random.Generator, cap: int, correct: bool | None = None
) -> list[SceneToken]:
    """Collect up to ``cap`` scene-labeled stimulus tokens for one probe (sampled without
    replacement across the probe's aligned scene spans, so long passages don't dominate). Each
    token carries its scene's canonical STORY index (``schema.scenes`` order) and its beat's
    TEXTUAL ordinal (rank of the span's char_start) -- decorrelated only under a temporal-order
    form, identical for a canonical passage."""
    tokens = meta.get("tokens") or []
    n_prompt = int(meta.get("n_prompt_tokens") or resid.shape[0])
    denom = max(n_prompt - 1, 1)
    story_order = _scene_story_order(schema)
    spans = meta.get("scene_token_spans", [])
    starts = sorted(sp["char_start"] for sp in spans if sp.get("char_start") is not None)
    pool: list[tuple[int, str, int | None, int | None]] = []  # (tok idx, scene, story, text_ord)
    for sp in spans:
        ts, te = sp.get("tok_start"), sp.get("tok_end")
        if ts is None or te is None:
            continue
        scene = sp["scene"]
        sidx = story_order.index(scene) if scene in story_order else None
        cs = sp.get("char_start")
        tord = starts.index(cs) if cs in starts else None
        for i in range(ts, te):
            pool.append((i, scene, sidx, tord))
    if not pool:
        return []
    if len(pool) > cap:
        pool = [pool[k] for k in rng.choice(len(pool), size=cap, replace=False)]
    out: list[SceneToken] = []
    for i, scene, sidx, tord in pool:
        tok = tokens[i] if i < len(tokens) else ""
        out.append(
            SceneToken(sid, group, schema, scene, tok, np.asarray(resid[i], np.float32), i / denom,
                       story_index=sidx, text_ordinal=tord, correct=correct)
        )
    return out


# --------------------------------------------------------------------------- decoding


def _decode_cell(
    vectors: np.ndarray, texts: list[str], contexts: list[str], contexts_left: list[str],
    positions: list[float], roles: list[str], groups: list[str], with_context: bool = True,
) -> dict | None:
    """Activation vs lexical / positional baselines vs chance role decoding for one subset, via
    grouped out-of-fold prediction (group = scenario, so a scenario's referents never straddle a
    fold). Baselines: unigram BoW + (1,2)-gram BoW over the entity's own mention strings
    (``lexical`` / ``lexical_ngram``), a (1,2)-gram BoW over the +/-window surrounding prompt
    tokens (``context_ngram``), the same over a left-only (preceding + mention) window
    (``context_ngram_left``) -- a causal-matched surface upper bound that cannot see future tokens
    -- and the mention's scalar relative token position alone (``positional``, the position
    confound's strength, mirroring the scene S5 control). Every probe also carries the SEM of its
    per-fold accuracy. Returns None-valued baseline scores when a probe's vocabulary is degenerate,
    or ``{skipped: ...}`` when the subset is too small / degenerate overall."""
    n = len(roles)
    counts = Counter(roles)
    n_groups = len(set(groups))
    if n < _MIN_ROWS or len(counts) < 2:
        return {"skipped": f"n={n}, classes={len(counts)}"}
    if min(counts.values()) < _MIN_PER_CLASS or n_groups < _CV_SPLITS:
        return {"skipped": f"n={n}, min_class={min(counts.values())}, groups={n_groups}"}
    y = np.asarray(roles)
    grp = np.asarray(groups)
    cv = GroupKFold(n_splits=_CV_SPLITS)
    folds = list(cv.split(np.zeros((n, 1)), y, grp))

    def _scored(pred: np.ndarray | None) -> dict:
        if pred is None:
            return {"accuracy": None, "macro_f1": None, "accuracy_sem": None}
        fold_acc = [accuracy_score(y[te], pred[te]) for _, te in folds]
        sem = float(np.std(fold_acc, ddof=1) / np.sqrt(len(fold_acc))) if len(fold_acc) > 1 else 0.0
        return {
            "accuracy": round(float(accuracy_score(y, pred)), 4),
            "macro_f1": round(float(f1_score(y, pred, average="macro", zero_division=0)), 4),
            "accuracy_sem": round(sem, 4),
        }

    def _cvp(pipe, x) -> np.ndarray | None:
        try:
            return cross_val_predict(pipe, x, y, cv=cv, groups=grp)
        except ValueError:  # e.g. empty vocabulary after min_df on a degenerate context subset
            return None

    def _bow(ngram: tuple[int, int], min_df: int = 1):
        return make_pipeline(
            CountVectorizer(lowercase=True, ngram_range=ngram, min_df=min_df,
                            token_pattern=r"(?u)\b\w+\b"),
            LogisticRegression(max_iter=2000))

    act = _scored(_cvp(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), vectors))
    lx = _scored(_cvp(_bow((1, 1)), texts))
    lx_ng = _scored(_cvp(_bow(_LEX_NGRAM), texts))
    ctx = _scored(_cvp(_bow(_LEX_NGRAM, _CTX_MIN_DF), contexts)) if with_context else _scored(None)
    ctxl = (_scored(_cvp(_bow(_LEX_NGRAM, _CTX_MIN_DF), contexts_left))
            if with_context else _scored(None))
    pos = _scored(_cvp(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
                       np.asarray(positions, dtype=float).reshape(-1, 1)))
    ch = _scored(cross_val_predict(DummyClassifier(strategy="most_frequent"),
                                   np.zeros((n, 1)), y, cv=cv, groups=grp))

    def _delta(b: dict) -> float | None:
        return None if b["accuracy"] is None else round(act["accuracy"] - b["accuracy"], 4)

    return {
        "n": n,
        "n_scenarios": n_groups,
        "classes": dict(sorted(counts.items())),
        "activation": act,
        "lexical": lx,
        "lexical_ngram": lx_ng,
        "context_ngram": ctx,
        "context_ngram_left": ctxl,
        "positional": pos,
        "chance": ch,
        "activation_minus_lexical": _delta(lx),
        "activation_minus_context": _delta(ctx),
        "activation_minus_context_left": _delta(ctxl),
    }


def role_decoding(referents: list[Referent], with_split: bool, with_context: bool = True) -> dict:
    """Role decoding pooled and split by schema, role_naming (latent vs named control), and -- when
    judged scores are present -- the identify correct/error split."""
    if not referents:
        return {}
    vecs = np.stack([r.vector for r in referents])
    texts = [r.text for r in referents]
    contexts = [r.context for r in referents]
    contexts_left = [r.context_left for r in referents]
    positions = [r.position for r in referents]
    roles = [r.role_id for r in referents]
    groups = [r.group for r in referents]

    def cell(mask: list[bool]) -> dict | None:
        idx = [i for i, m in enumerate(mask) if m]
        if not idx:
            return None
        return _decode_cell(vecs[idx], [texts[i] for i in idx], [contexts[i] for i in idx],
                            [contexts_left[i] for i in idx], [positions[i] for i in idx],
                            [roles[i] for i in idx], [groups[i] for i in idx],
                            with_context=with_context)

    out: dict = {"pooled": _decode_cell(vecs, texts, contexts, contexts_left, positions,
                                        roles, groups, with_context=with_context)}
    by_schema = {}
    for schema in sorted({r.schema for r in referents}):
        by_schema[schema] = cell([r.schema == schema for r in referents])
    out["by_schema"] = by_schema
    by_role_naming = {}
    for rn in sorted({r.role_naming for r in referents}):
        by_role_naming[rn] = cell([r.role_naming == rn for r in referents])
    out["by_role_naming"] = by_role_naming
    if with_split and any(r.correct is not None for r in referents):
        out["by_outcome"] = {
            "correct": cell([r.correct is True for r in referents]),
            "error": cell([r.correct is False for r in referents]),
        }
    # per-PERSON outcome (each referent = one entity): split by whether THIS entity's role was
    # judged correct (verdict correct/recovered) vs not (wrong/vague/missing) -- unlike by_outcome,
    # which splits on the whole scenario's correctness.
    if with_split and any(r.role_correct is not None for r in referents):
        out["by_role_outcome"] = {
            "role_correct": cell([r.role_correct is True for r in referents]),
            "role_error": cell([r.role_correct is False for r in referents]),
        }
    return out


# --------------------------------------------------------------------------- clustering


def scene_clustering(
    scene_tokens: list[SceneToken], max_tokens: int = 20000, seed: int = 0
) -> dict:
    """Descriptive H1 readout per schema: KMeans (k = #scenes) over scene-labeled token residuals,
    scored by ARI / NMI against the native scene labels, next to a token-identity partition
    baseline (how much scene structure raw surface token identity already carries)."""
    rng = np.random.default_rng(seed)
    out: dict = {}
    by_schema: dict[str, list[SceneToken]] = {}
    for t in scene_tokens:
        by_schema.setdefault(t.schema, []).append(t)
    for schema, toks in sorted(by_schema.items()):
        if len(toks) > max_tokens:
            toks = [toks[i] for i in rng.choice(len(toks), size=max_tokens, replace=False)]
        scenes = [t.scene for t in toks]
        k = len(set(scenes))
        if k < 2 or len(toks) < k * 10:
            out[schema] = {"skipped": f"n={len(toks)}, scenes={k}"}
            continue
        x = StandardScaler().fit_transform(np.stack([t.vector for t in toks]))
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(x)
        tokid = [t.token for t in toks]
        out[schema] = {
            "n_tokens": len(toks),
            "n_scenes": k,
            "activation_kmeans": {
                "ari": round(float(adjusted_rand_score(scenes, km)), 4),
                "nmi": round(float(normalized_mutual_info_score(scenes, km)), 4),
            },
            "token_identity_baseline": {
                "ari": round(float(adjusted_rand_score(scenes, tokid)), 4),
                "nmi": round(float(normalized_mutual_info_score(scenes, tokid)), 4),
            },
        }
    return out


# --------------------------------------------------------------------------- report


def _acc(b: dict | None) -> str:
    return "n/a" if not b or b.get("accuracy") is None else f"{b['accuracy']:.3f}"


def _fmt_cell(cell: dict | None) -> str:
    if not cell or "skipped" in (cell or {}):
        return f"skipped ({cell.get('skipped') if cell else 'no data'})"
    a, ch = cell["activation"], cell["chance"]
    d = cell.get("activation_minus_context")
    return (
        f"act {a['accuracy']:.3f} (F1 {a['macro_f1']:.3f}) | "
        f"lex {_acc(cell['lexical'])} | lex-ngram {_acc(cell.get('lexical_ngram'))} | "
        f"ctx-ngram {_acc(cell.get('context_ngram'))} | "
        f"ctx-L {_acc(cell.get('context_ngram_left'))} | pos {_acc(cell.get('positional'))} | "
        f"chance {ch['accuracy']:.3f} | "
        f"act-ctx {('n/a' if d is None else f'{d:+.3f}')} | n={cell['n']}"
    )


def write_report(stats: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    lines = [
        f"# Identify-task latent geometry -- {stats['slug']} (layer {stats['layer']})",
        "",
        f"Runs: {', '.join(stats['runs'])}  |  referents: {stats['n_referents']}  |  "
        f"scene tokens: {stats['n_scene_tokens']}  |  judged: {stats['judged']}",
        "",
        "## H2 -- role decoding (activation vs lexical baseline vs chance)",
        "",
        f"- **pooled**: {_fmt_cell(stats['role_decoding'].get('pooled'))}",
    ]
    for group in ("by_schema", "by_role_naming", "by_outcome", "by_role_outcome"):
        block = stats["role_decoding"].get(group)
        if not block:
            continue
        lines.append(f"- **{group}**:")
        for key, cell in block.items():
            lines.append(f"    - {key}: {_fmt_cell(cell)}")
    lines += ["", "## H1 -- scene clustering (KMeans ARI/NMI vs token-identity)", ""]
    for schema, c in stats["scene_clustering"].items():
        if "skipped" in c:
            lines.append(f"- {schema}: skipped ({c['skipped']})")
        else:
            km, base = c["activation_kmeans"], c["token_identity_baseline"]
            lines.append(
                f"- {schema}: activation ARI {km['ari']:.3f} / NMI {km['nmi']:.3f} vs "
                f"token-identity ARI {base['ari']:.3f} / NMI {base['nmi']:.3f} "
                f"(n={c['n_tokens']}, scenes={c['n_scenes']})"
            )
    (out_dir / "findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _role_cells(rd: dict) -> list[tuple[str, dict]]:
    """The (label, cell) groups for a role-decode plot: the pooled cell plus the per-PERSON
    correct/error split. The role_naming split is intentionally omitted here -- it is redundant
    with the by-control figure."""
    cells: list[tuple[str, dict]] = []
    if rd.get("pooled") and "skipped" not in rd["pooled"]:
        cells.append(("pooled", rd["pooled"]))
    olabel = {"role_correct": "role-correct", "role_error": "role-error"}
    for key, cell in (rd.get("by_role_outcome") or {}).items():
        if cell and "skipped" not in cell:
            cells.append((olabel.get(key, key), cell))
    return cells


def _plot_decode_cells(
    cells: list[tuple[str, dict]], out_path: Path, title: str, ylabel: str,
    lexical_label: str = "lexical BoW", show_context: bool = False, dpi: int = 130,
) -> None:
    """Shared grouped-bar renderer for the decode plots (role + scene). One bar group per cell;
    bars = residual activity / lexical / positional / most frequent (majority-class baseline), plus
    the two context n-grams when ``show_context``; each bar carries per-fold SEM error bars, legend
    outside to the right. A dashed line marks uniform chance (1 / #classes) -- one spanning line
    when every cell shares the same class count, else a per-group segment."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = [(lab, c) for lab, c in cells if c and "skipped" not in c]
    if not cells:
        return
    labels = [c[0] for c in cells]
    series = [("residual activity", "activation"), (lexical_label, "lexical")]
    if show_context:
        series += [("context n-gram", "context_ngram"),
                   ("context n-gram (left)", "context_ngram_left")]
    series += [("positional", "positional"), ("most frequent", "chance")]

    def _val_sem(cell: dict, key: str) -> tuple[float, float]:
        b = cell.get(key) or {}
        return (b.get("accuracy") or 0.0, b.get("accuracy_sem") or 0.0)

    x = np.arange(len(labels))
    w = 0.8 / len(series)
    offsets = (np.arange(len(series)) - (len(series) - 1) / 2) * w
    fig, ax = plt.subplots(figsize=(max(6, 1.9 * len(labels)), 4))
    for (label, key), off in zip(series, offsets, strict=True):
        vals = [_val_sem(c, key)[0] for _, c in cells]
        sems = [_val_sem(c, key)[1] for _, c in cells]
        ax.bar(x + off, vals, w, yerr=sems, capsize=3, label=label)
    # uniform chance = 1 / #classes (true random-guess floor; the "most frequent" bars are the
    # majority-class baseline). One spanning line when all cells share a class count, else segments.
    unif = [1.0 / len(c["classes"]) for _, c in cells]
    if len({round(u, 6) for u in unif}) == 1:
        ax.axhline(unif[0], color="black", ls="--", lw=1.0, label="chance")
    else:
        half = w * len(series) / 2
        for i, (xi, u) in enumerate(zip(x, unif, strict=True)):
            ax.plot([xi - half, xi + half], [u, u], color="black", ls="--", lw=1.0,
                    label="chance" if i == 0 else None)
    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_role_decoding(stats: dict, out_dir: Path, show_context: bool = False) -> None:
    """Grouped bar chart of role-decode accuracy (mean +/- per-fold SEM): the pooled cell plus the
    per-person correct/error split. Bars: residual activity, lexical BoW, positional, most frequent
    (majority-class baseline), plus a uniform-chance dashed line; the context n-gram / context
    n-gram (left) baselines appear only when ``show_context`` is set."""
    _plot_decode_cells(
        _role_cells(stats["role_decoding"]), out_dir / "role_decoding.png",
        title=f"Identify role geometry -- {stats['slug']} L{stats['layer']}",
        ylabel="role-decode accuracy (grouped CV, mean +/- SEM)", show_context=show_context)


# --------------------------------------------------------------------------- cli


def run(
    run_dirs: list[Path], slug: str, layer: int, scores_tag: str | None, out_dir: Path,
    show_context: bool = False, task: str = "identify",
) -> dict:
    referents, scene_tokens = load_captures(run_dirs, slug, layer, scores_tag, task=task)
    judged = scores_tag is not None and any(r.correct is not None for r in referents)
    stats = {
        "slug": slug,
        "layer": layer,
        "task": task,
        "runs": [f"{d.parent.name}/{d.name}" for d in run_dirs],
        "scores_tag": scores_tag,
        "judged": judged,
        "n_referents": len(referents),
        "n_scene_tokens": len(scene_tokens),
        "role_decoding": role_decoding(referents, with_split=judged),
        "scene_clustering": scene_clustering(scene_tokens),
    }
    write_report(stats, out_dir)
    plot_role_decoding(stats, out_dir, show_context=show_context)
    print(f"pooled role decode: {_fmt_cell(stats['role_decoding'].get('pooled'))}")
    print(f"-> {out_dir}")
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description="first-pass identify-task latent geometry (H1/H2)")
    p.add_argument("--run", required=True, nargs="+", help="run dir(s) with activations/<slug>/")
    p.add_argument("--slug", required=True, help="model activation slug (e.g. phi-4)")
    p.add_argument("--layer", type=int, default=17)
    p.add_argument("--task", default="identify",
                   help="capture task to load (identify/predict_next_event/predict_correct_action/"
                        "spot_error)")
    p.add_argument("--scores-tag", default=None,
                   help="judged scores namespace for the correct/error split (default: none)")
    p.add_argument("--out", required=True, help="output dir for stats.json / findings.md / figure")
    p.add_argument("--show-context", action="store_true",
                   help="include the context n-gram surface baselines in the plot (default: off)")
    args = p.parse_args()
    run([Path(r) for r in args.run], args.slug, args.layer, args.scores_tag, Path(args.out),
        show_context=args.show_context, task=args.task)


if __name__ == "__main__":
    main()
