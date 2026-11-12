"""Guarded Phi3 cache compatibility for checkpoints with identical LongRoPE factors.

The pinned preparer clears its cache at the original context boundary, but forwards
``next_sequence_length=1`` to the newer generic preparer. The full prefix is then
lost. When short/long factors are identical, that reset is unnecessary: use the
same generic preparation without clearing the cache, and retain the initialized
FP32 rotary basis throughout. This changes neither weights nor config values.
"""

from __future__ import annotations

import hashlib
import inspect
import math
from contextlib import contextmanager
from pathlib import Path
from types import MethodType

import torch
from transformers.generation.utils import GenerationMixin
from transformers.models.phi3.modeling_phi3 import Phi3ForCausalLM

from .head_fv_models import active as active_model

POLICY = "phi3-equal-factor-continuous-cache-v1"
NATIVE_SHA = "ea0114c4430aa08623686f43560dcd340b874cc85d2518d7a2c54d7afb5c5a90"
GENERIC_SHA = "dfe543260af0c8fcae16536e7320ba6708d5b280e99510fb5c49f411f9d67907"
_MISSING = object()


def _prepare_without_reset(
    self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None,
    position_ids=None, use_cache=True, logits_to_keep=None, **kwargs,
):
    return GenerationMixin.prepare_inputs_for_generation(
        self, input_ids=input_ids, past_key_values=past_key_values,
        attention_mask=attention_mask, inputs_embeds=inputs_embeds,
        position_ids=position_ids, use_cache=use_cache, logits_to_keep=logits_to_keep, **kwargs,
    )


def _checked(lm):
    model = lm.model
    spec = active_model()
    if not spec.supports_phi3_cache_fix:
        raise ValueError(
            f"subject {spec.key!r} does not use the Phi3 cache fix: it exists only because that "
            "checkpoint's short and long LongRoPE factors are identical, making the preparer's "
            "4k history reset redundant. Applying it elsewhere is unsupported, and silently "
            "doing nothing would hide that the guard was reached at all."
        )
    if not isinstance(model, Phi3ForCausalLM):
        raise ValueError("cache compatibility is limited to the inspected Phi3 implementation")
    for fn, expected in (
        (Phi3ForCausalLM.prepare_inputs_for_generation, NATIVE_SHA),
        (GenerationMixin.prepare_inputs_for_generation, GENERIC_SHA),
    ):
        if hashlib.sha256(inspect.getsource(fn).encode()).hexdigest() != expected:
            raise ValueError("generation preparation source changed; recertification required")
    if model.prepare_inputs_for_generation.__func__ not in (
        Phi3ForCausalLM.prepare_inputs_for_generation, _prepare_without_reset
    ):
        raise ValueError("unrecognized existing generation preparation override")
    params = model.config.rope_parameters
    if (
        params.get("rope_type") != "longrope" or not params.get("short_factor")
        or params["short_factor"] != params.get("long_factor")
        or any(not math.isfinite(x) or x <= 0 for x in params["short_factor"])
    ):
        raise ValueError("cache compatibility requires identical positive short/long factors")
    rotary = model.model.rotary_emb
    original = rotary.original_inv_freq
    if (
        original.dtype != torch.float32 or original.ndim != 1
        or original.numel() != len(params["short_factor"]) or not torch.isfinite(original).all()
    ):
        raise ValueError("unexpected initialized rotary basis")
    return model, rotary


def cache_policy(lm) -> dict:
    """Describe the policy generate_exact/clean replay will apply, without changing the model."""
    _, rotary = _checked(lm)
    return {
        "kind": POLICY,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "native_prepare_sha256": NATIVE_SHA,
        "generic_prepare_sha256": GENERIC_SHA,
        "short_long_factors_identical": True,
        "original_inv_freq_sha256": hashlib.sha256(
            rotary.original_inv_freq.detach().cpu().numpy().tobytes()
        ).hexdigest(),
        "attention_scaling": float(rotary.attention_scaling),
        "model_config_changed": False,
    }


@contextmanager
def stable_phi_cache(lm):
    """Apply only to this model instance; restore methods and buffers even on failure."""
    model, rotary = _checked(lm)
    old_method = vars(model).get("prepare_inputs_for_generation", _MISSING)
    old_inv = rotary.inv_freq
    old_long = getattr(rotary, "long_inv_freq", _MISSING)
    model.prepare_inputs_for_generation = MethodType(_prepare_without_reset, model)
    rotary.inv_freq = rotary.original_inv_freq
    rotary.long_inv_freq = rotary.original_inv_freq
    try:
        yield
    finally:
        rotary.inv_freq = old_inv
        if old_long is _MISSING:
            delattr(rotary, "long_inv_freq")
        else:
            rotary.long_inv_freq = old_long
        if old_method is _MISSING:
            delattr(model, "prepare_inputs_for_generation")
        else:
            model.prepare_inputs_for_generation = old_method
