"""Mid-trace injection harness -- Stage 2 of the STOP-steering plan.

Neither existing harness does what the experiment needs. ``head_fv_*`` generates from a prompt
under a fixed ``skip-prefill-last-position-all-continuation`` schedule and cannot start at
generated token *N*; ``bigloop`` resumes mid-trace from a KV cache but injects once, regenerates a
fixed 40 tokens, and never reads a stop reason. This module is the missing composition: resume an
existing trace from an arbitrary token, add ``alpha * v`` to a block-output residual either once or
for the whole continuation, run to EOS or a budget, and report **how the generation ended**.

The corpus was generated greedily but through the **batched** path, which ``core.model`` documents
as "NOT byte-identical to generate_reply (batching changes padding, numerics ...)" -- so a
single-sequence resume cannot reproduce the stored tail exactly, and measurement confirms it:
greedy continuations track the corpus for a few dozen tokens and then diverge. That is a property
of the corpus, not of this harness, and the same limitation is recorded for the transplant parity
arm. ``parity()`` therefore tests what *is* verifiable -- the harness is bit-deterministic, and
every arm shares one resume path, so a paired difference is attributable to the injection -- and
reports corpus tracking as a diagnostic rather than a pass/fail.

The consequence for the design is explicit: the control arm is the alpha=0 **resume**, never the
stored trace, so baseline termination must be re-measured here rather than inherited from the
corpus.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ...core.model import load_model
from .. import head_fv_specificity as runtime
from ..head_fv_capture import _unhooked
from ..head_fv_models import active as active_model
from . import runframe

MODEL = "phi4mr"
DEFAULT_LAYER = 16
TOKEN_BUDGET = 8192
SCHEDULES = ("once", "sustained")


def require(condition, message):
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class Arm:
    name: str
    alpha: float = 0.0
    layer: int = DEFAULT_LAYER
    schedule: str = "sustained"

    def __post_init__(self):
        require(self.schedule in SCHEDULES, f"unknown schedule {self.schedule}")


@dataclass
class Outcome:
    item: str
    arm: str
    alpha: float
    layer: int
    schedule: str
    prefix_len: int
    generated: list[int]
    stop_reason: str

    @property
    def n_generated(self):
        return len(self.generated)


class Injector:
    """Adds ``alpha * v`` to the block-output residual at the decoding position."""

    def __init__(self, lm, layer, vector, alpha, schedule):
        self.module = lm.model.get_submodule(lm.layer_hook_names([layer])[0])
        self.delta = None if vector is None or alpha == 0.0 else (
            torch.tensor(np.asarray(vector, dtype=np.float32) * alpha, device=lm.device))
        self.schedule = schedule
        self.steps = 0
        self.applied = 0
        self.handle = None
        self.arrived = []

    def _hook(self, _module, _args, output):
        if self.delta is None:
            return output
        if self.schedule == "once" and self.steps > 0:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        require(hidden.shape[1] == 1, "injection expects a single decoding position")
        before = hidden[:, -1, :].detach().float()
        hidden[:, -1, :] = hidden[:, -1, :] + self.delta.to(hidden.dtype)
        self.arrived.append(float(torch.linalg.vector_norm(
            hidden[:, -1, :].detach().float() - before)))
        self.applied += 1
        return output

    def __enter__(self):
        self.handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, *_):
        if self.handle is not None:
            self.handle.remove()


def resume_generate(lm, ids, prefix_len, *, vector=None, alpha=0.0, layer=DEFAULT_LAYER,
                    schedule="sustained", budget=TOKEN_BUDGET, eos_ids=(), prompt_len=None,
                    temperature=0.0, top_p=1.0, seed=None):
    """Greedy continuation from ``ids[:prefix_len]``; injection never touches the prefix.

    ``prompt_len`` replays the original decode path: prefill the prompt in one chunk, then
    teacher-force the stored response tokens one at a time, exactly as the corpus was produced.
    Bulk-prefilling the whole prefix instead is faster but numerically different, and greedy
    argmax turns that difference into divergent tokens within a few dozen steps.
    """
    require(0 < prefix_len <= len(ids), "prefix out of range")
    boundary = prefix_len if prompt_len is None else int(prompt_len)
    require(0 < boundary <= prefix_len, "prompt boundary out of range")
    head = torch.from_numpy(np.ascontiguousarray(ids[:boundary]))[None].to(lm.device)
    _unhooked(lm.model)
    lm.model.eval()
    generated: list[int] = []
    stop_reason = "budget"
    # Common random numbers: the generator is seeded per (item, draw), never per arm, so
    # baseline and treatment consume the *same* random stream and stay paired. Without this,
    # sampling noise would be confounded with the intervention.
    gen = None
    if temperature > 0:
        gen = torch.Generator(device=lm.device)
        gen.manual_seed(int(seed if seed is not None else 0))
    with torch.inference_mode(), runtime.subject_cache(lm):
        out = lm.model(input_ids=head, attention_mask=torch.ones_like(head),
                       position_ids=torch.arange(boundary, device=lm.device)[None],
                       use_cache=True)
        cache, logits = out.past_key_values, out.logits[:, -1, :].float()
        for pos in range(boundary, prefix_len):
            step = torch.tensor([[int(ids[pos])]], device=lm.device)
            position = torch.tensor([[pos]], device=lm.device)
            out = lm.model(input_ids=step, past_key_values=cache, use_cache=True,
                           position_ids=position)
            cache, logits = out.past_key_values, out.logits[:, -1, :].float()
        with Injector(lm, layer, vector, alpha, schedule) as injector:
            while len(generated) < budget:
                token = int(_pick(logits, temperature, top_p, gen))
                generated.append(token)
                if token in eos_ids:
                    stop_reason = "eos"
                    break
                step = torch.tensor([[token]], device=lm.device)
                position = torch.tensor([[prefix_len + len(generated) - 1]], device=lm.device)
                out = lm.model(input_ids=step, past_key_values=cache, use_cache=True,
                               position_ids=position)
                cache, logits = out.past_key_values, out.logits[:, -1, :].float()
                injector.steps += 1
            applied, arrived = injector.applied, injector.arrived
    return dict(generated=generated, stop_reason=stop_reason, applied=applied,
                mean_arrived_norm=float(np.mean(arrived)) if arrived else 0.0)


def prompt_lengths(tokens):
    prompt = tokens[tokens.segment == "prompt"] if "segment" in tokens else None
    require(prompt is not None, "tokens frame must carry the segment column")
    return prompt.groupby("item_id").size().to_dict()


def _pick(logits, temperature, top_p, gen):
    """Greedy when temperature is 0; otherwise nucleus sampling from the shared stream."""
    if temperature <= 0:
        return torch.argmax(logits, dim=-1).item()
    scaled = logits / temperature
    probs = torch.softmax(scaled, dim=-1)
    if top_p < 1.0:
        ordered, index = torch.sort(probs, descending=True, dim=-1)
        cumulative = ordered.cumsum(-1)
        drop = cumulative - ordered > top_p
        ordered[drop] = 0.0
        ordered = ordered / ordered.sum(-1, keepdim=True)
        choice = torch.multinomial(ordered, 1, generator=gen)
        return index.gather(-1, choice).item()
    return torch.multinomial(probs, 1, generator=gen).item()


def load_ids(tokens, item):
    d = tokens[tokens.item_id == item].sort_values("position")
    positions = d["position"].values
    require(np.array_equal(positions, np.arange(len(positions))), f"{item}: non-contiguous")
    return d["token_id"].values.astype(np.int64)


def parity(lm, tokens, items, frame, eos_ids, prompt_lens, n_items=5, budget=64,
           teacher_force=True):
    """How far an alpha=0 resume tracks the stored tail, plus a bit-determinism check."""
    rows = []
    for item in items[:n_items]:
        ids = load_ids(tokens, item)
        group = frame[frame.item == item]
        # resume a quarter of the way through the trace, at a run boundary
        row_end = int(group["row_end"].values[len(group) // 4])
        prefix_len = row_end + 1
        remaining = ids[prefix_len:]
        if len(remaining) < 8:
            continue
        limit = int(min(budget, len(remaining)))
        result = resume_generate(
            lm, ids, prefix_len, budget=limit, eos_ids=eos_ids,
            prompt_len=prompt_lens[item] if teacher_force else None)
        produced = np.array(result["generated"], dtype=np.int64)
        target = remaining[:len(produced)]
        match = int(np.argmax(produced != target)) if (produced != target).any() else len(produced)
        repeat = resume_generate(
            lm, ids, prefix_len, budget=limit, eos_ids=eos_ids,
            prompt_len=prompt_lens[item] if teacher_force else None)
        rows.append(dict(item=item, prefix_len=prefix_len, compared=len(produced),
                         exact=bool(np.array_equal(produced, target)),
                         first_divergence=match,
                         deterministic=bool(repeat["generated"] == result["generated"])))
    return pd.DataFrame(rows)


def run_parity(out: Path, device="cuda:0", n_items=5):
    out.mkdir(parents=True, exist_ok=True)
    built = runframe.build()
    frame = built["seg"][MODEL]
    tokens = pd.read_parquet(runframe.BUNDLE / "tokens.parquet",
                             columns=["item_id", "model", "task_family", "position",
                                      "token_id", "segment"])
    tokens = tokens[(tokens.model == MODEL) & (tokens.task_family == runframe.TASK)]
    prompt_lens = prompt_lengths(tokens)
    spec = active_model()
    lm = load_model(spec.model, revision=spec.revision, device_map=device)
    summary = {}
    for mode in (True, False):
        table = parity(lm, tokens, built["items"], frame, set(spec.eos_ids), prompt_lens,
                       n_items=n_items, teacher_force=mode)
        label = "teacher_forced" if mode else "bulk_prefill"
        print(f"\n-- {label} --")
        print(table.to_string(index=False))
        summary[label] = dict(n_items=int(len(table)), exact=int(table["exact"].sum()),
                              tokens_compared=int(table["compared"].sum()),
                              all_exact=bool(table["exact"].all()),
                              deterministic=bool(table["deterministic"].all()),
                              median_first_divergence=float(table["first_divergence"].median()))
    (out / "parity.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-items", type=int, default=5)
    parser.add_argument("--parity", action="store_true")
    args = parser.parse_args()
    require(args.parity, "only --parity is implemented as a direct entry point")
    run_parity(args.out, device=args.device, n_items=args.n_items)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
