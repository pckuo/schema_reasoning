"""Load Phi-4 (or another HF causal LM) for local latent capture (B0).

Ports the ``"phi"`` branch of the algotrace reference loader
(``code/reference/algotrace/src/utils/model_utils.py``): half-precision weights,
``device_map='auto'``, and per-layer residual hook names ``model.layers.{i}``.

Smoke test (also triggers the weight download into ``HF_HOME``)::

    uv run python -m dev.schema_llm.core.model --model microsoft/phi-4
    uv run python -m dev.schema_llm.core.model --model Qwen/Qwen3-14B --think on
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "microsoft/phi-4"

# Depths the schema_llm captures actually recorded (every 4th block, L0-L28). Interventions must
# be validated against this: the primitive dictionary only exists at these layers, so injecting at
# e.g. L17 has no vector to inject and no layer to read out at.
CAPTURE_GRID = (0, 4, 8, 12, 16, 20, 24, 28)


def validate_layer(layer: int, grid: tuple[int, ...] = CAPTURE_GRID) -> int:
    """Reject a layer the captured latents (and hence the primitive dictionary) do not exist at."""
    if layer not in grid:
        raise SystemExit(
            f"--layer {layer} is not in the captured grid {list(grid)}: no primitive "
            "vectors exist at that depth (pass --capture-layers for another grid)"
        )
    return layer


_DTYPES: dict[str, Any] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "auto": "auto",
}


@dataclass
class LoadedModel:
    """A loaded causal LM plus the config the capture harness needs."""

    model: Any
    tokenizer: Any
    name: str
    n_layers: int
    hidden_size: int
    revision: str | None = None
    layer_prefix: str = "model.layers"

    @property
    def device(self) -> torch.device:
        """Device of the first parameter (cuda:0 for a single-GPU ``device_map='auto'``)."""
        return next(self.model.parameters()).device

    def layer_hook_names(self, layers: list[int]) -> list[str]:
        """baukit ``TraceDict`` names for the residual stream out of each decoder block.
        ``layer_prefix`` is the model-family module path (``model.layers`` for Phi/Qwen; gemma-3
        multimodal nests the text stack at ``model.language_model.layers``)."""
        return [f"{self.layer_prefix}.{i}" for i in layers]


def chat_template_kwargs(enable_thinking: bool | None) -> dict[str, Any]:
    """Extra ``apply_chat_template`` kwargs. ``enable_thinking`` toggles Qwen3's hybrid ``<think>``
    mode and is forwarded ONLY when set (``None`` keeps the template default), so tokenizers whose
    template does not accept the kwarg are unaffected. Shared with capture as the single source."""
    return {} if enable_thinking is None else {"enable_thinking": enable_thinking}


def _from_pretrained_cls(
    cls: Any,
    name: str,
    torch_dtype: Any,
    device_map: str,
    revision: str | None,
) -> Any:
    """``cls.from_pretrained`` with the transformers>=5 ``dtype`` kwarg, falling back to the older
    ``torch_dtype`` name."""
    kwargs = {"device_map": device_map}
    if revision is not None:
        kwargs["revision"] = revision
    try:
        return cls.from_pretrained(name, dtype=torch_dtype, **kwargs)
    except TypeError:
        return cls.from_pretrained(name, torch_dtype=torch_dtype, **kwargs)


def _load_hf_model(
    name: str,
    torch_dtype: Any,
    device_map: str,
    revision: str | None,
) -> Any:
    """Load the right class for ``name``. Plain causal LMs (Phi, Qwen2.5, Qwen3, ...) use
    ``AutoModelForCausalLM``; multimodal gemma-3 ``-it`` checkpoints ship as
    ``Gemma3ForConditionalGeneration`` (which ``AutoModelForCausalLM`` does not map), so they load
    via ``AutoModelForImageTextToText`` and are fed text only -- we hook the text backbone."""
    config_kwargs = {"revision": revision} if revision is not None else {}
    cfg = AutoConfig.from_pretrained(name, **config_kwargs)
    if getattr(cfg, "model_type", "") == "gemma3" and getattr(cfg, "text_config", None) is not None:
        from transformers import AutoModelForImageTextToText

        return _from_pretrained_cls(
            AutoModelForImageTextToText,
            name,
            torch_dtype,
            device_map,
            revision,
        )
    return _from_pretrained_cls(
        AutoModelForCausalLM,
        name,
        torch_dtype,
        device_map,
        revision,
    )


def _text_config(cfg: Any) -> Any:
    """The sub-config carrying the decoder dims. Multimodal models (gemma-3-it) nest the
    language-model dims under ``cfg.text_config``; plain causal LMs expose them directly."""
    inner = getattr(cfg, "text_config", None)
    if inner is not None and hasattr(inner, "num_hidden_layers"):
        return inner
    return cfg


def _decoder_layer_prefix(model: Any) -> str:
    """baukit dotted path (relative to the loaded model) to the decoder-block ``ModuleList``. Plain
    causal LMs put it at ``model.layers``; multimodal gemma-3 nests the text stack at
    ``model.language_model.layers``. Returns the first candidate that resolves to a sized module."""
    candidates = (
        "model.layers",
        "model.language_model.layers",
        "language_model.model.layers",
        "language_model.layers",
    )
    for path in candidates:
        obj: Any = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        if hasattr(obj, "__len__"):
            return path
    raise RuntimeError(f"cannot locate decoder layers on {type(model).__name__}")


def chat_template_of(spec: str) -> str:
    """The chat template of another checkpoint, as ``model_id`` or ``model_id@revision``. Used to
    render prompts for a model under a DIFFERENT model's template -- e.g. Phi-4-mini-reasoning's
    stock template hard-codes a ``"Your name is Phi, an AI math expert"`` system turn that no other
    model in a comparison gets, and borrowing Phi-4-mini-instruct's (byte-identical Phi format minus
    that line) removes the asymmetry."""
    name, _, rev = spec.partition("@")
    tok = AutoTokenizer.from_pretrained(name, **({"revision": rev} if rev else {}))
    template = tok.chat_template
    if not template:
        raise ValueError(f"{name} has no chat_template to borrow")
    return template


def load_model(
    name: str = DEFAULT_MODEL,
    dtype: str = "bfloat16",
    device_map: str = "auto",
    revision: str | None = None,
    chat_template: str | None = None,
) -> LoadedModel:
    """Load a HF causal LM in half precision, sharded by ``device_map`` (mirrors the reference
    ``"phi"`` branch). Handles plain causal LMs (Phi-4, Qwen2.5, Qwen3) and multimodal gemma-3
    ``-it`` checkpoints (loaded text-only via their text backbone).

    ``dtype`` is one of ``bfloat16`` (Phi-4's native type), ``float16``, or ``auto``.
    """
    if dtype not in _DTYPES:
        raise ValueError(f"dtype must be one of {sorted(_DTYPES)}, got {dtype!r}")
    tokenizer_kwargs = {"revision": revision} if revision is not None else {}
    tokenizer = AutoTokenizer.from_pretrained(name, **tokenizer_kwargs)
    if chat_template is not None:  # prompts are rendered with this instead of the stock template
        tokenizer.chat_template = chat_template
    model = _load_hf_model(name, _DTYPES[dtype], device_map, revision)
    model.eval()
    text_cfg = _text_config(model.config)
    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        name=name,
        n_layers=text_cfg.num_hidden_layers,
        hidden_size=text_cfg.hidden_size,
        revision=revision,
        layer_prefix=_decoder_layer_prefix(model),
    )


def set_seed(seed: int) -> None:
    """Seed python / numpy / torch for reproducible generation (ported from the reference)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def generate_reply(
    lm: LoadedModel,
    prompt: str,
    *,
    system_message: str | None = None,
    max_new_tokens: int = 1024,
    do_sample: bool = False,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    enable_thinking: bool | None = None,
) -> tuple[torch.Tensor, int, str]:
    """Chat-template a single user turn (optionally with a system message), generate one reply,
    and return ``(full_ids, n_prompt_tokens, reply)``. ``full_ids`` is the 1-D CPU tensor of the
    whole ``[prompt + reply]`` sequence (the capture harness re-forwards it to grab residuals);
    ``reply`` is decoded from the tokens after the prompt. Shared by :mod:`collect` and
    :mod:`capture` so both produce byte-identical Phi-4 responses for the same probe (B0)."""
    tok = lm.tokenizer
    messages: list[dict] = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": prompt})
    enc = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True,
        **chat_template_kwargs(enable_thinking),
    )
    input_ids = enc["input_ids"].to(lm.device)
    attn = enc.get("attention_mask")
    attn = attn.to(lm.device) if attn is not None else torch.ones_like(input_ids)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": pad_id,
    }
    if do_sample:
        for key, val in (("temperature", temperature), ("top_p", top_p), ("top_k", top_k)):
            if val is not None:
                gen_kwargs[key] = val
    n_prompt_tokens = int(input_ids.shape[1])
    with torch.no_grad():
        out = lm.model.generate(input_ids=input_ids, attention_mask=attn, **gen_kwargs)
    full_ids = out[0].detach().cpu()
    reply = tok.decode(full_ids[n_prompt_tokens:], skip_special_tokens=True).strip()
    return full_ids, n_prompt_tokens, reply


def _split_batched_generation(
    out: torch.Tensor, prompt_attn: torch.Tensor, prompt_len: int, eos_id: int | None
) -> list[tuple[torch.Tensor, int]]:
    """Split a LEFT-padded batched ``generate`` output into per-sequence ``(full_ids, n_prompt)``.

    ``out`` is ``(B, prompt_len + gen_len)`` (the batched ``[prompt + generated]``); ``prompt_attn``
    is the ``(B, prompt_len)`` prompt mask (1 on real tokens, 0 on the left pad). For each row we
    strip the left pad off the prompt and truncate the generated part at (and including) its first
    ``eos`` -- dropping the trailing pad a batched ``generate`` appends once a shorter sequence has
    finished -- so ``full_ids`` matches the ``[prompt + response]`` that a one-at-a-time
    ``generate`` would have produced for that prompt (and that the capture harness re-forwards)."""
    results: list[tuple[torch.Tensor, int]] = []
    for i in range(out.shape[0]):
        n_prompt = int(prompt_attn[i].sum())
        prompt_ids = out[i, prompt_len - n_prompt : prompt_len]
        gen = out[i, prompt_len:]
        if eos_id is not None:
            eos_hits = (gen == eos_id).nonzero(as_tuple=False)
            end = int(eos_hits[0]) + 1 if eos_hits.numel() else gen.shape[0]
        else:
            end = gen.shape[0]
        full_ids = torch.cat([prompt_ids, gen[:end]]).detach().cpu()
        results.append((full_ids, n_prompt))
    return results


def generate_batched(
    lm: LoadedModel,
    prompts: list[str],
    *,
    system_message: str | None = None,
    max_new_tokens: int = 1024,
    do_sample: bool = False,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    enable_thinking: bool | None = None,
) -> list[tuple[torch.Tensor, int, str]]:
    """Generate replies for a batch of prompts in a single ``generate`` call; returns, per prompt
    (input order preserved), ``(full_ids, n_prompt_tokens, reply)`` -- the same triple as
    :func:`generate_reply`, where ``full_ids`` is the 1-D CPU ``[prompt + response]`` sequence with
    padding stripped and the response truncated at eos, so the capture harness can re-forward it for
    residuals exactly as in the one-at-a-time path.

    Left-pads the batch (required for correct decoder-only generation). The autoregressive decode is
    what dominates wall-clock for reasoning models, so batching it here is the throughput win.
    Outputs are NOT byte-identical to :func:`generate_reply` (batching changes padding, numerics and
    RNG draws), but each returned sequence is internally consistent -- a per-sequence residual
    re-forward of its ``full_ids`` is valid."""
    if not prompts:
        return []
    tok = lm.tokenizer
    prev_side = tok.padding_side
    tok.padding_side = "left"
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    try:
        rendered: list[str] = []
        for prompt in prompts:
            messages: list[dict] = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})
            rendered.append(
                tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    **chat_template_kwargs(enable_thinking),
                )
            )
        enc = tok(rendered, return_tensors="pt", padding=True, add_special_tokens=False)
        input_ids = enc["input_ids"].to(lm.device)
        attn = enc["attention_mask"].to(lm.device)
        prompt_len = input_ids.shape[1]
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": pad_id,
        }
        if do_sample:
            for key, val in (("temperature", temperature), ("top_p", top_p), ("top_k", top_k)):
                if val is not None:
                    gen_kwargs[key] = val
        with torch.no_grad():
            out = lm.model.generate(input_ids=input_ids, attention_mask=attn, **gen_kwargs)
        splits = _split_batched_generation(out.cpu(), attn.cpu(), prompt_len, tok.eos_token_id)
    finally:
        tok.padding_side = prev_side
    results: list[tuple[torch.Tensor, int, str]] = []
    for full_ids, n_prompt in splits:
        reply = tok.decode(full_ids[n_prompt:], skip_special_tokens=True).strip()
        results.append((full_ids, n_prompt, reply))
    return results


def build_full_ids(
    lm: LoadedModel,
    prompt: str,
    *,
    system_message: str | None = None,
    response_text: str | None = None,
    response_token_ids: list[int] | None = None,
    enable_thinking: bool | None = None,
) -> tuple[torch.Tensor, int]:
    """Reconstruct ``(full_ids, n_prompt_tokens)`` for an ALREADY-produced response so the capture
    harness can re-forward the exact ``[prompt + response]`` without regenerating.

    The prompt half is re-rendered deterministically (same chat template as :func:`generate_reply`,
    so the ids match a fresh generation's prompt). The response half is used **verbatim and in
    full** -- the entire generated text, INCLUDING any ``<think> ... </think>`` chain of thought;
    it is never reasoning-stripped or truncated (``strip_reasoning`` is a judge-only concern). Pass
    ``response_token_ids`` for a token-exact rebuild, else ``response_text`` is re-tokenized (same
    text, but the token split may differ from the original by a boundary)."""
    if response_text is None and response_token_ids is None:
        raise ValueError("build_full_ids needs response_text or response_token_ids")
    tok = lm.tokenizer
    messages: list[dict] = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": prompt})
    enc = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True,
        **chat_template_kwargs(enable_thinking),
    )
    prompt_ids = enc["input_ids"][0]
    n_prompt_tokens = int(prompt_ids.shape[0])
    if response_token_ids is not None:
        resp_ids = torch.tensor(response_token_ids, dtype=prompt_ids.dtype)
    else:
        resp_ids = torch.tensor(
            tok(response_text, add_special_tokens=False)["input_ids"], dtype=prompt_ids.dtype
        )
    full_ids = torch.cat([prompt_ids, resp_ids])
    return full_ids, n_prompt_tokens


def _smoke(
    name: str = DEFAULT_MODEL, dtype: str = "bfloat16", enable_thinking: bool | None = None
) -> None:
    lm = load_model(name, dtype=dtype)
    print(f"loaded {lm.name}: {lm.n_layers} layers, hidden {lm.hidden_size}, "
          f"prefix {lm.layer_prefix!r}, device {lm.device}")
    messages = [{"role": "user", "content": "In one short sentence, what happens at a restaurant?"}]
    enc = lm.tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True,
        **chat_template_kwargs(enable_thinking),
    )
    enc = {k: v.to(lm.device) for k, v in enc.items()}
    with torch.no_grad():
        out = lm.model.generate(
            **enc, max_new_tokens=40, do_sample=False, pad_token_id=lm.tokenizer.eos_token_id
        )
    n_prompt = enc["input_ids"].shape[1]
    print("generation:", lm.tokenizer.decode(out[0, n_prompt:], skip_special_tokens=True))


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="smoke-test the loader (loads + generates once)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--dtype", default="bfloat16", choices=sorted(_DTYPES))
    p.add_argument("--think", default="auto", choices=["auto", "on", "off"],
                   help="Qwen3 hybrid thinking toggle (auto keeps the template default)")
    args = p.parse_args()
    _smoke(args.model, args.dtype, {"auto": None, "on": True, "off": False}[args.think])


if __name__ == "__main__":
    main()
