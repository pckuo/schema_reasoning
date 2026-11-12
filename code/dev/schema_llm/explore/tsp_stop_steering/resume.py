"""Prefix-cache replay with snapshots, so many injected arms share one teacher-forcing pass.

``stop_steering.inject.resume_generate`` replays the prefix **one token per forward** to reproduce
the decode numerics exactly, which is right but costs ``prefix_len`` forwards per run. This
experiment resumes the *same* 9 baseline traces from 3 offsets under 12 arm/draw combinations
each, so replaying per run would spend more compute on replay than on generation.

Here the replay runs once per trace, in a single increasing pass, and the KV cache is cloned at
each offset. Cost falls from ``1.5 * R`` per arm-draw to ``0.75 * R`` per trace -- about 12x.

The equivalence this relies on is checked, not assumed: :func:`parity` asserts that an ``alpha=0``
continuation from a clone reproduces the baseline tail **token for token** under the baseline's own
seed. That check was impossible in the ``identify`` work, whose corpus came from the batched
generation path; here the baseline is produced by this very function, so exact parity is the
expected result and anything else is a bug.
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import DynamicCache

from .. import head_fv_specificity as runtime
from ..head_fv_capture import _unhooked
from ..stop_steering.inject import Injector, _pick


def require(condition, message):
    if not condition:
        raise ValueError(message)


def clone_cache(cache: DynamicCache, batch: int = 1) -> DynamicCache:
    """An independent copy, optionally replicated to ``batch`` identical rows.

    ``DynamicLayer.crop`` slices, which would alias storage; the tensors are cloned instead.
    Replication is what makes batching possible here: every run at one injection point resumes
    from the *same* prefix, so the rows are identical at step zero and stay positionally aligned
    for the whole continuation -- no padding and no per-row attention mask.
    """
    out = DynamicCache()
    for index, layer in enumerate(cache.layers):
        keys, values = layer.keys, layer.values
        if batch > 1:
            keys = keys.expand(batch, -1, -1, -1).contiguous()
            values = values.expand(batch, -1, -1, -1).contiguous()
        else:
            keys, values = keys.clone(), values.clone()
        out.update(keys, values, index)
    require(out.get_seq_length() == cache.get_seq_length(), "clone lost cache positions")
    return out


def select_cache_rows(cache: DynamicCache, keep: torch.Tensor) -> None:
    """Drop finished rows in place, so a long control does not drag finished stop arms along."""
    for layer in cache.layers:
        layer.keys = layer.keys[keep].contiguous()
        layer.values = layer.values[keep].contiguous()


class BatchInjector:
    """Adds a per-row ``alpha * v`` to the block-output residual at the decoding position.

    The batch-1 :class:`stop_steering.inject.Injector` broadcasts one delta over every row, which
    is exactly wrong here: the whole point of the batch is that its rows are *different arms*.
    """

    def __init__(self, lm, layer, deltas):
        self.module = lm.model.get_submodule(lm.layer_hook_names([layer])[0])
        self.deltas = deltas  # (B, hidden), zero rows for the alpha=0 control
        self.handle = None
        self.applied = None
        self.arrived = None

    def reset(self, batch):
        self.applied = torch.zeros(batch, dtype=torch.int64, device=self.deltas.device)
        self.arrived = torch.zeros(batch, dtype=torch.float32, device=self.deltas.device)

    def _hook(self, _module, _args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        require(hidden.shape[1] == 1, "injection expects a single decoding position")
        require(hidden.shape[0] == self.deltas.shape[0], "delta rows must match the batch")
        hidden[:, -1, :] = hidden[:, -1, :] + self.deltas.to(hidden.dtype)
        moved = torch.linalg.vector_norm(self.deltas.float(), dim=-1)
        self.arrived += moved
        self.applied += (moved > 0).to(torch.int64)
        return output

    def __enter__(self):
        self.handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, *_):
        if self.handle is not None:
            self.handle.remove()


def _pick_batch(logits, temperature, top_p, uniforms):
    """Nucleus sampling by inverse CDF, one shared uniform per row.

    Deliberately not ``torch.multinomial``: that consumes a different amount of randomness per
    row, so two arms driven by "the same seed" would still diverge for reasons unrelated to the
    intervention. Feeding one uniform per *draw* to every arm in that draw makes common random
    numbers exact.
    """
    if temperature <= 0:
        return torch.argmax(logits, dim=-1)
    probs = torch.softmax(logits / temperature, dim=-1)
    ordered, index = torch.sort(probs, descending=True, dim=-1)
    if top_p < 1.0:
        cumulative = ordered.cumsum(-1)
        ordered = ordered.masked_fill(cumulative - ordered > top_p, 0.0)
    ordered = ordered / ordered.sum(-1, keepdim=True)
    cdf = ordered.cumsum(-1)
    choice = torch.searchsorted(cdf, uniforms.unsqueeze(-1).to(cdf.dtype))
    choice = choice.clamp(max=ordered.shape[-1] - 1)
    return index.gather(-1, choice).squeeze(-1)


def generate_batch(lm, cache, logits, start_position, *, deltas, layer=16, budget=512,
                   eos_ids=(), temperature=0.0, top_p=1.0, uniform_source=None):
    """Continue ``B`` rows in lockstep from one shared prefix. ``cache`` is consumed.

    ``deltas`` is ``(B, hidden)``; a zero row is the ``alpha = 0`` control. ``uniform_source``
    yields a ``(B,)`` tensor of uniforms per step and carries the common-random-number policy.

    Rows are positionally identical throughout -- they start at the same offset and advance one
    token per step -- so no attention mask or position bookkeeping is needed. Finished rows are
    compacted out of the batch as they hit EOS.
    """
    batch = deltas.shape[0]
    require(cache.get_seq_length() == start_position, "cache length must match the start position")
    alive = torch.arange(batch, device=lm.device)
    generated: list[list[int]] = [[] for _ in range(batch)]
    stop_reason = ["budget"] * batch
    injector = BatchInjector(lm, layer, deltas)
    injector.reset(batch)
    applied = np.zeros(batch, dtype=np.int64)
    arrived = np.zeros(batch, dtype=np.float64)
    with injector:
        for step in range(budget):
            rows = alive.tolist()
            # the uniform is keyed by the ROW, not its position in a shrinking batch -- otherwise
            # compaction would silently re-assign random streams between arms
            uniforms = (uniform_source(rows, step) if uniform_source is not None
                        else torch.zeros(len(rows), device=lm.device))
            tokens = _pick_batch(logits, temperature, top_p, uniforms)
            finished = []
            for position, (row, token) in enumerate(zip(rows, tokens.tolist(), strict=True)):
                generated[row].append(int(token))
                if int(token) in eos_ids:
                    stop_reason[row] = "eos"
                    finished.append(position)
            if finished:
                counts = injector.applied.detach().cpu().numpy()
                norms = injector.arrived.detach().cpu().numpy()
                for position in finished:
                    row = rows[position]
                    applied[row], arrived[row] = counts[position], norms[position]
                done = set(finished)
                keep = torch.tensor([i for i in range(len(rows)) if i not in done],
                                    device=lm.device, dtype=torch.long)
                if keep.numel() == 0:
                    alive = keep
                    break
                select_cache_rows(cache, keep)
                injector.deltas = injector.deltas[keep].contiguous()
                injector.applied = injector.applied[keep].contiguous()
                injector.arrived = injector.arrived[keep].contiguous()
                tokens = tokens[keep]
                alive = alive[keep]
            if step == budget - 1:
                break
            step_ids = tokens.reshape(-1, 1)
            positions = torch.full((int(alive.numel()), 1), start_position + step,
                                   device=lm.device, dtype=torch.long)
            out = lm.model(input_ids=step_ids, past_key_values=cache, use_cache=True,
                           position_ids=positions)
            cache, logits = out.past_key_values, out.logits[:, -1, :].float()
        if alive.numel():
            counts = injector.applied.detach().cpu().numpy()
            norms = injector.arrived.detach().cpu().numpy()
            for position, row in enumerate(alive.tolist()):
                applied[row], arrived[row] = counts[position], norms[position]
    return [dict(generated=generated[row], stop_reason=stop_reason[row],
                 applied=int(applied[row]),
                 mean_arrived_norm=float(arrived[row] / max(applied[row], 1)))
            for row in range(batch)]


def _forward(lm, token_id: int, position: int, cache):
    step = torch.tensor([[int(token_id)]], device=lm.device)
    pos = torch.tensor([[int(position)]], device=lm.device)
    out = lm.model(input_ids=step, past_key_values=cache, use_cache=True, position_ids=pos)
    return out.past_key_values, out.logits[:, -1, :].float()


def prefill(lm, prompt_ids):
    """Prefill the prompt in one chunk and return ``(cache, next-token logits)``."""
    head = torch.from_numpy(np.ascontiguousarray(prompt_ids))[None].to(lm.device)
    out = lm.model(input_ids=head, attention_mask=torch.ones_like(head),
                   position_ids=torch.arange(len(prompt_ids), device=lm.device)[None],
                   use_cache=True)
    return out.past_key_values, out.logits[:, -1, :].float()


def generate(lm, cache, logits, start_position, *, vector=None, alpha=0.0, layer=16,
             schedule="sustained", budget=512, eos_ids=(), temperature=0.0, top_p=1.0,
             seed=None):
    """Continue from a prepared cache. ``cache`` is consumed and must already be a clone.

    Mirrors ``inject.resume_generate``'s loop exactly, including the detail that the *first*
    sampled token comes from the un-injected prefix logits: the hook perturbs the residual at each
    decoding step, which moves the token after it.
    """
    gen = None
    if temperature > 0:
        gen = torch.Generator(device=lm.device)
        gen.manual_seed(int(seed if seed is not None else 0))
    generated: list[int] = []
    stop_reason = "budget"
    with Injector(lm, layer, vector, alpha, schedule) as injector:
        while len(generated) < budget:
            token = int(_pick(logits, temperature, top_p, gen))
            generated.append(token)
            if token in eos_ids:
                stop_reason = "eos"
                break
            cache, logits = _forward(lm, token, start_position + len(generated) - 1, cache)
            injector.steps += 1
        applied, arrived = injector.applied, injector.arrived
    return dict(generated=generated, stop_reason=stop_reason, applied=applied,
                mean_arrived_norm=float(np.mean(arrived)) if arrived else 0.0)


def baseline(lm, prompt_ids, *, budget, eos_ids, temperature, top_p, seed):
    """Generate a fresh trace through the same single-sequence path the arms resume into."""
    _unhooked(lm.model)
    lm.model.eval()
    with torch.inference_mode(), runtime.subject_cache(lm):
        cache, logits = prefill(lm, prompt_ids)
        result = generate(lm, cache, logits, len(prompt_ids), budget=budget, eos_ids=eos_ids,
                          temperature=temperature, top_p=top_p, seed=seed)
    return result


class Replay:
    """Teacher-force a stored response once, exposing a clone of the cache at any offset.

    Offsets must be requested in increasing order -- the point of the class is that the master
    cache is *extended*, never rebuilt.
    """

    def __init__(self, lm, prompt_ids, response_ids):
        self.lm = lm
        self.prompt_ids = np.asarray(prompt_ids, dtype=np.int64)
        self.response_ids = np.asarray(response_ids, dtype=np.int64)
        self.cache, self.logits = prefill(lm, self.prompt_ids)
        self.offset = 0
        self.forwards = 0

    @property
    def position(self):
        return len(self.prompt_ids) + self.offset

    def advance_to(self, offset: int):
        """Teacher-force response tokens until ``offset`` of them are in the cache."""
        require(offset >= self.offset, "Replay offsets must be requested in increasing order")
        require(offset <= len(self.response_ids), "offset past the end of the response")
        while self.offset < offset:
            token = int(self.response_ids[self.offset])
            self.cache, self.logits = _forward(self.lm, token, self.position, self.cache)
            self.offset += 1
            self.forwards += 1
        return self

    def fork(self, batch: int = 1):
        """A private ``(cache, logits, position)`` for one continuation, or ``batch`` of them."""
        logits = self.logits.clone()
        if batch > 1:
            logits = logits.expand(batch, -1).contiguous()
        return clone_cache(self.cache, batch), logits, self.position


def parity(lm, prompt_ids, response_ids, *, offset, eos_ids, budget):
    """Does a greedy alpha=0 fork reproduce the stored tail? Returns the first divergence index.

    Greedy on purpose. Under sampling a fork cannot match: the baseline's generator is seeded at
    the start of generation, so by ``offset`` its stream has advanced, and a freshly seeded fork
    draws different numbers. Determinism is therefore the only form in which this check is
    meaningful, and it isolates exactly what is being tested -- that a cloned cache is numerically
    the same object as the one the baseline decoded through.
    """
    _unhooked(lm.model)
    lm.model.eval()
    with torch.inference_mode(), runtime.subject_cache(lm):
        replay = Replay(lm, prompt_ids, response_ids).advance_to(offset)
        cache, logits, position = replay.fork()
        result = generate(lm, cache, logits, position, budget=budget, eos_ids=eos_ids)
    produced = np.asarray(result["generated"], dtype=np.int64)
    target = np.asarray(response_ids[offset:offset + len(produced)], dtype=np.int64)
    compared = int(min(len(produced), len(target)))
    differ = produced[:compared] != target[:compared]
    return dict(compared=compared, exact=bool(compared and not differ.any()),
                first_divergence=int(np.argmax(differ)) if differ.any() else compared,
                forwards=replay.forwards)
