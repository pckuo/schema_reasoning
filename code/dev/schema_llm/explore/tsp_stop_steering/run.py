"""Run the TSP transfer test for the frozen STOP direction.

Three stages, all resumable because the GPU is shared and a full matrix is hours long:

``parity``   greedy gate -- a forked cache must reproduce the trace it was forked from, exactly.
``baseline`` 3 sampled traces per task, generated through the same single-sequence path the
             injected arms resume into.
``inject``   for each baseline trace, fork at 25/50/75% of its reasoning and run the arm matrix.

The control is the ``alpha = 0`` fork at the same offset under the same seed, never the stored
baseline tail: under sampling the tail is one draw from the same distribution the control samples
from, so comparing against it would charge sampling noise to the intervention.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ...core.model import chat_template_of, load_model
from ..head_fv_models import active as active_model
from . import resume, tasks

VECTORS = Path("results/schema_llm/analysis/stop_steering/20260919-vectors-full")
CAP = 16384
# Baselines use their own, larger cap: the injection budget was lowered later for wall-clock
# reasons, and regenerating baselines under it would censor traces the original nine did not hit.
BASELINE_CAP = 24576
LAYER = 16
SCHEDULE = "sustained"
TEMPERATURE = 0.7
TOP_P = 0.95
DRAWS = 2
# Baselines carry their own draw count: the first run generated 3 baseline traces per task and
# later reduced only the *injection* draws, so a replication must match 3 or the instances would
# enter the design with unequal numbers of forked traces.
BASELINE_DRAWS = 3
LOCATIONS = (0.25, 0.50, 0.75)
# One batch covers a whole injection point (4 arms x 3 draws). Rows share the prefix and step in
# lockstep, so there is no padding; peak KV is bounded by the injection budget.
MAX_BATCH = 12
# (arm, direction, dose, locations). The alpha=0 fork is the control; `shuffled` is the matched
# null -- same 35-head subspace, same norm, permuted labels. Running it at all three offsets costs
# essentially nothing in wall-clock, because `stop@0.5` already never terminates and so pins every
# batch at the full budget regardless.
PLAN = (
    ("baseline", None, 0.0, LOCATIONS),
    ("stop", "stop", 0.25, LOCATIONS),
    ("stop", "stop", 0.50, LOCATIONS),
    ("shuffled", "shuffled", 0.25, LOCATIONS),
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def seed_of(*parts) -> int:
    """Stable across processes, unlike ``hash`` on strings (which is PYTHONHASHSEED-salted)."""
    key = "|".join(str(p) for p in parts).encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=4).digest(), "big")


def find_think_end(tokenizer, ids) -> int:
    """Token index at which ``</think>`` starts, or ``len(ids)`` if the trace never closed.

    Phi's think markers are not single vocabulary tokens, so the marker is located in decoded text
    and mapped back through per-token decodes. The reconstruction is asserted against a whole-
    sequence decode rather than trusted, because a byte-level BPE can split a multi-byte character
    across tokens and silently shift every offset after it.
    """
    pieces = [tokenizer.decode([int(t)]) for t in ids]
    text = "".join(pieces)
    require(text == tokenizer.decode(list(map(int, ids))),
            "per-token decode does not reconstruct the sequence; offsets would be wrong")
    char = text.find("</think>")
    if char < 0:
        return len(ids)
    cursor = 0
    for index, piece in enumerate(pieces):
        if cursor + len(piece) > char:
            return index
        cursor += len(piece)
    return len(ids)


def prompt_ids_for(lm, task, system: bool):
    enc = lm.tokenizer.apply_chat_template(
        tasks.messages(task["prompt"], system=system), add_generation_prompt=True,
        return_tensors="pt", return_dict=True)
    return enc["input_ids"][0].numpy().astype(np.int64)


def load_lm(device):
    spec = active_model()
    lm = load_model(spec.model, revision=spec.revision, device_map=device,
                    chat_template=chat_template_of(spec.chat_template))
    return lm, spec


def run_parity(lm, spec, task, prompt_ids, out: Path, budget=256):
    """Greedy gate. A fork that cannot reproduce its own trace invalidates every later number.

    Two invariants, both cheap and both load-bearing:

    1. an ``alpha = 0`` fork reproduces the trace it was forked from, token for token;
    2. the **batched** path at ``batch = 1`` is token-identical to the single-sequence path.

    The second is what separates "batching changed the schedule" from "batching is broken". At
    batch > 1 the rows legitimately drift from a single-sequence run after a few dozen tokens --
    bf16 matmul numerics, the same effect ``core/model.py`` documents for its batched generator --
    but at batch 1 the shapes and kernels are identical, so any difference is a real misalignment
    of positions or cache rows. Validity does not rest on matching the single-sequence path: every
    arm at an injection point shares one batch with its own control, so the paired comparison is
    numerically exact regardless.
    """
    base = resume.baseline(lm, prompt_ids, budget=budget * 4, eos_ids=set(spec.eos_ids),
                           temperature=0.0, top_p=1.0, seed=None)
    ids = base["generated"]
    require(len(ids) > budget, "greedy probe too short to fork inside")
    checks = []
    for offset in (len(ids) // 4, len(ids) // 2):
        check = resume.parity(lm, prompt_ids, ids, offset=offset, eos_ids=set(spec.eos_ids),
                              budget=min(budget, len(ids) - offset))
        check["offset"] = offset
        checks.append(check)
        print(f"  parity @{offset}: compared {check['compared']} exact={check['exact']} "
              f"first_divergence={check['first_divergence']}", flush=True)
    batch_ok = batch_parity(lm, spec, prompt_ids, ids, offset=len(ids) // 4,
                            budget=min(budget, len(ids) - len(ids) // 4))
    print(f"  batch parity @batch=1: identical={batch_ok}", flush=True)
    report = dict(uid=task["uid"], probe_tokens=len(ids), checks=checks,
                  batch1_matches_single=bool(batch_ok),
                  all_exact=all(c["exact"] for c in checks) and bool(batch_ok))
    (out / "parity.json").write_text(json.dumps(report, indent=2))
    return report


def batch_parity(lm, spec, prompt_ids, response_ids, *, offset, budget):
    """Is ``generate_batch`` at batch=1 token-identical to ``generate``?"""
    width = int(lm.model.config.hidden_size)
    resume._unhooked(lm.model)
    lm.model.eval()
    with torch.inference_mode(), resume.runtime.subject_cache(lm):
        replay = resume.Replay(lm, prompt_ids, response_ids).advance_to(offset)
        cache, logits, position = replay.fork()
        single = resume.generate(lm, cache, logits, position, budget=budget,
                                 eos_ids=set(spec.eos_ids), temperature=0.0)["generated"]
        cache, logits, position = replay.fork(batch=1)
        batched = resume.generate_batch(lm, cache, logits, position,
                                        deltas=torch.zeros(1, width, device=lm.device),
                                        budget=budget, eos_ids=set(spec.eos_ids),
                                        temperature=0.0)[0]["generated"]
    return single == batched


def stage_baseline(lm, spec, selected, out: Path, cap, draws, system):
    """Generate the missing baselines, batching a task's draws together.

    **Incremental**, like the injection stage: existing ``(uid, draw)`` rows are kept and only the
    gaps are generated, so a replication run adds instances without regenerating -- or perturbing
    -- what is already on disk.

    **Batched over draws**, because a task's draws share an identical prompt and differ only in
    their random stream: exactly the no-padding, lockstep case the injection batcher already
    handles. The draws consume per-draw uniforms through the same inverse-CDF sampler, so they stay
    independent samples; batching changes the schedule, not the distribution.
    """
    path = out / "baselines.parquet"
    existing = pd.read_parquet(path) if path.exists() else None
    have = set() if existing is None else {
        (int(r.uid), int(r.draw)) for r in existing.itertuples()}
    eos = set(spec.eos_ids)
    width = int(lm.model.config.hidden_size)
    rows = []
    for task in selected:
        wanted = [d for d in range(draws) if (int(task["uid"]), d) not in have]
        if not wanted:
            print(f"  skip uid={task['uid']} (all {draws} draws present)", flush=True)
            continue
        prompt_ids = prompt_ids_for(lm, task, system)
        seeds = {d: seed_of("baseline", task["uid"], d) for d in wanted}
        cells = [dict(draw=d) for d in wanted]
        resume._unhooked(lm.model)
        lm.model.eval()
        with torch.inference_mode(), resume.runtime.subject_cache(lm):
            cache, logits = resume.prefill(lm, prompt_ids)
            if len(wanted) > 1:
                cache = resume.clone_cache(cache, batch=len(wanted))
                logits = logits.expand(len(wanted), -1).contiguous()
            results = resume.generate_batch(
                lm, cache, logits, len(prompt_ids),
                deltas=torch.zeros(len(wanted), width, device=lm.device),
                budget=cap, eos_ids=eos, temperature=TEMPERATURE, top_p=TOP_P,
                uniform_source=uniform_source_for(cells, seeds, lm.device))
        for draw, result in zip(wanted, results, strict=True):
            ids = result["generated"]
            text = lm.tokenizer.decode(ids, skip_special_tokens=False)
            think_end = find_think_end(lm.tokenizer, ids)
            verdict = tasks.score(text, task["weight_matrix"], task["ground_truth"])
            rows.append(dict(
                uid=task["uid"], level=task["level"], n_cities=task["n_cities"],
                ground_truth=task["ground_truth"], draw=draw, seed=seeds[draw],
                n_prompt=len(prompt_ids), n_generated=len(ids),
                stop_reason=result["stop_reason"], terminated=result["stop_reason"] == "eos",
                closed_think=think_end < len(ids), reasoning_tokens=think_end,
                token_ids=np.asarray(ids, dtype=np.int32), text=text, **verdict))
            print(f"  baseline {task['level']} uid={task['uid']} d{draw}: {len(ids)} tok "
                  f"({result['stop_reason']}) reasoning={think_end} "
                  f"correct={verdict['correct']}", flush=True)
        torch.cuda.empty_cache()
    if not rows:
        print(f"baselines: all present in {path}", flush=True)
        return existing
    table = pd.DataFrame(rows)
    if existing is not None:
        table = pd.concat([existing, table], ignore_index=True)
    table.to_parquet(path)
    return table


def uniform_source_for(cells, seeds, device):
    """Common random numbers: one uniform per *draw* per step, shared by every arm in that draw.

    Streams are advanced for every draw at every step regardless of which rows are still alive, so
    compaction cannot shift a stream, and each chunk reseeds from the same ``seeds`` so splitting a
    location across several batches leaves the numbers identical.
    """
    draws = sorted({cell["draw"] for cell in cells})
    generators = {d: torch.Generator(device="cpu").manual_seed(int(seeds[d])) for d in draws}

    def source(rows, _step):
        values = {d: float(torch.rand(1, generator=generators[d]).item()) for d in draws}
        return torch.tensor([values[cells[r]["draw"]] for r in rows], device=device,
                            dtype=torch.float32)

    return source


def missing_cells(shard: Path, draws, locations=LOCATIONS, plan=PLAN):
    """Cells the plan requires that this shard does not already contain.

    Makes a plan change incremental instead of destructive: widening the arm matrix regenerates
    only the new cells. The trade-off is recorded per row in ``batch_rows`` -- a backfilled cell is
    generated in a smaller batch than the cells it will be compared against, and batch size shifts
    bf16 matmul numerics, so its token stream is not the one it would have had in the full batch.
    That is acceptable for a large qualitative contrast (``shuffled`` fails to terminate at all)
    and not acceptable for a fine paired estimate, which is why the column exists rather than the
    caveat living only in prose.
    """
    required = {(float(location), arm, float(dose), draw)
                for location in locations
                for draw in range(draws)
                for arm, _key, dose, where in plan if location in where}
    if not shard.exists():
        return required, None
    existing_frame = pd.read_parquet(shard)
    existing = {(float(r.location), r.arm, float(r.dose), int(r.draw))
                for r in existing_frame.itertuples()}
    return required - existing, existing_frame


def stage_inject(lm, spec, selected, baselines, out: Path, cap, draws, system, vectors, layer,
                 max_batch=MAX_BATCH):
    directions = dict(np.load(vectors / "directions.npz"))
    manifest = json.loads((vectors / "manifest.json").read_text())
    state_norm = float(manifest["median_l16_state_norm"])
    eos = set(spec.eos_ids)
    by_uid = {t["uid"]: t for t in selected}
    shard_dir = out / "injections"
    shard_dir.mkdir(parents=True, exist_ok=True)
    width = int(lm.model.config.hidden_size)

    # Draw-major, not task-major: the matrix takes ~24 h, so the order decides what a *partial*
    # result looks like. Task-major would finish all three level-1 traces first, making any early
    # read a level-1 result; draw-major completes one trace per level first, so the partial set
    # spans the difficulty range and can speak to transfer rather than to one task.
    for base in baselines.sort_values(["draw", "uid"]).itertuples():
        shard = shard_dir / f"uid{base.uid}_d{base.draw}.parquet"
        wanted, existing_frame = missing_cells(shard, draws)
        if not wanted:
            print(f"  skip {shard.name} (complete)", flush=True)
            continue
        if existing_frame is not None:
            print(f"  backfill {shard.name}: {len(wanted)} missing cells", flush=True)
        task = by_uid[base.uid]
        prompt_ids = prompt_ids_for(lm, task, system)
        response_ids = np.asarray(base.token_ids, dtype=np.int64)
        reasoning = int(base.reasoning_tokens)
        require(reasoning > 8, f"uid {base.uid} draw {base.draw}: reasoning trace too short")
        rows = []
        resume._unhooked(lm.model)
        lm.model.eval()
        with torch.inference_mode(), resume.runtime.subject_cache(lm):
            replay = resume.Replay(lm, prompt_ids, response_ids)
            for location in LOCATIONS:
                cells = [dict(arm=arm, key=key, dose=dose, draw=draw)
                         for draw in range(draws)
                         for arm, key, dose, where in PLAN
                         if location in where
                         and (location, arm, float(dose), draw) in wanted]
                if not cells:
                    continue
                offset = int(round(location * reasoning))
                replay.advance_to(offset)
                budget = cap - offset
                require(budget > 0, "no budget left at this offset")
                seeds = {d: seed_of("inject", base.uid, base.draw, location, d)
                         for d in range(draws)}
                for start in range(0, len(cells), max_batch):
                    chunk = cells[start:start + max_batch]
                    deltas = torch.zeros(len(chunk), width, device=lm.device)
                    for index, cell in enumerate(chunk):
                        if cell["key"] is not None:
                            vector = np.asarray(directions[cell["key"]], dtype=np.float32)
                            deltas[index] = torch.tensor(vector * (cell["dose"] * state_norm),
                                                         device=lm.device)
                    cache, logits, position = replay.fork(batch=len(chunk))
                    results = resume.generate_batch(
                        lm, cache, logits, position, deltas=deltas, layer=layer, budget=budget,
                        eos_ids=eos, temperature=TEMPERATURE, top_p=TOP_P,
                        uniform_source=uniform_source_for(chunk, seeds, lm.device))
                    del cache
                    for cell, result in zip(chunk, results, strict=True):
                        rows.append(_row(lm, task, base, location, offset, budget, cell["arm"],
                                         cell["dose"], state_norm, layer, cell["draw"],
                                         int(seeds[cell["draw"]]), result, response_ids,
                                         len(chunk)))
                        print(f"  uid={base.uid} b{base.draw} @{location:.2f} d{cell['draw']} "
                              f"{cell['arm']:9s} dose {cell['dose']:<4} -> "
                              f"{result['stop_reason']:6s} {len(result['generated']):5d} tok",
                              flush=True)
                    torch.cuda.empty_cache()
            del replay
        torch.cuda.empty_cache()
        table = pd.DataFrame(rows)
        if existing_frame is not None:
            table = pd.concat([existing_frame, table], ignore_index=True)
        table.to_parquet(shard)
    shards = sorted(shard_dir.glob("*.parquet"))
    table = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)
    table.to_parquet(out / "injections.parquet")
    return table


def _row(lm, task, base, location, offset, budget, arm, dose, state_norm, layer, draw, seed,
         result, response_ids, batch_rows):
    generated = result["generated"]
    kept = lm.tokenizer.decode(response_ids[:offset], skip_special_tokens=False)
    tail = lm.tokenizer.decode(generated, skip_special_tokens=False)
    verdict = tasks.score(kept + tail, task["weight_matrix"], task["ground_truth"])
    distinct = len(set(generated)) / max(len(generated), 1)
    return dict(
        uid=base.uid, level=base.level, base_draw=base.draw, location=location, offset=offset,
        arm=arm, dose=dose, alpha=dose * state_norm, layer=layer, draw=draw, seed=seed,
        budget=budget, batch_rows=int(batch_rows), baseline_tokens=int(base.n_generated),
        baseline_reasoning=int(base.reasoning_tokens),
        n_generated=len(generated), total_tokens=offset + len(generated),
        stop_reason=result["stop_reason"], terminated=result["stop_reason"] == "eos",
        closed_think="</think>" in tail, applied=result["applied"],
        mean_arrived_norm=result["mean_arrived_norm"], distinct_token_ratio=float(distinct),
        tail_preview=tail[:200].replace("\n", " "), tail_text=tail, **verdict)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--vectors", type=Path, default=VECTORS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cap", type=int, default=CAP)
    parser.add_argument("--draws", type=int, default=DRAWS)
    parser.add_argument("--layer", type=int, default=LAYER)
    parser.add_argument("--max-batch", type=int, default=MAX_BATCH,
                        help="rows generated together at one injection point")
    parser.add_argument("--baseline-cap", type=int, default=BASELINE_CAP,
                        help="completion cap for baselines (kept above --cap on purpose)")
    parser.add_argument("--baseline-draws", type=int, default=BASELINE_DRAWS,
                        help="sampled traces per task (forked by every injection cell)")
    parser.add_argument("--task-rank", type=int, default=0,
                        help="0 = rank-median instance per level; 1 = the next, for replication")
    parser.add_argument("--stage", default="all",
                        choices=("all", "parity", "baseline", "inject"))
    parser.add_argument("--no-system", action="store_true",
                        help="extraction-regime framing arm: drop the card system message")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    selected = tasks.select_tasks(rank=args.task_rank)
    system = not args.no_system
    lm, spec = load_lm(args.device)
    (args.out / "manifest.json").write_text(json.dumps(dict(
        model=spec.model, revision=spec.revision, vectors=str(args.vectors), cap=args.cap,
        layer=args.layer, schedule=SCHEDULE, temperature=TEMPERATURE, top_p=TOP_P,
        draws=args.draws, locations=list(LOCATIONS), system_prompt=system,
        max_batch=args.max_batch, baseline_cap=args.baseline_cap, task_rank=args.task_rank,
        baseline_draws=args.baseline_draws,
        plan=[(a, k, d, list(w)) for a, k, d, w in PLAN],
        tasks=[{k: v for k, v in t.items() if k != "prompt"} for t in selected]), indent=2))

    if args.stage in ("all", "parity"):
        print("stage: parity", flush=True)
        report = run_parity(lm, spec, selected[0],
                            prompt_ids_for(lm, selected[0], system), args.out)
        require(report["all_exact"], "forked cache does not reproduce its own trace; aborting")
    if args.stage == "parity":
        return 0

    print("stage: baseline", flush=True)
    baselines = stage_baseline(lm, spec, selected, args.out, args.baseline_cap,
                               args.baseline_draws, system)
    print(baselines[["level", "uid", "draw", "n_generated", "stop_reason",
                     "reasoning_tokens", "correct"]].to_string(index=False), flush=True)
    if args.stage == "baseline":
        return 0

    print("stage: inject", flush=True)
    stage_inject(lm, spec, selected, baselines, args.out, args.cap, args.draws, system,
                 args.vectors, args.layer, max_batch=args.max_batch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
