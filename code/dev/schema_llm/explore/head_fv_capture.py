"""Immutable L16 residual capture from an ALREADY loaded, unhooked Phi-mini model.

Phase 1 only: no generation, tokenizer calls, model loading, dictionary fitting,
head means, or gate/cohort certification. Call ``capture_runtime(lm)`` first.
The source manifest must contain dataset_id, cohort, data_sha256,
partition_sha256, generation_file_sha256, generator_runtime_sha256, attempt_ids,
and runtime_contract_sha256 = digest(capture_runtime(lm)). Generation rows retain
the exact-ID readout fields plus their unique, ordered ``attempt_id``.

runtime_contract_sha256 names the separate CLEAN MEASUREMENT runtime, not the
generator's cached/batched runtime. It excludes the source/condition/roster, so
discovery and treatment can share a dictionary under identical replay settings.
Batch-1 full-forward measurements are not claimed identical to cached generation.
Full replay enters the generator's shared ``stable_phi_cache`` compatibility
context, with its exact ``cache_policy`` and initialized FP32 basis hash frozen.
This is not a new long/short rotary regime. Early-window labels remain slices of
full-replay assignments, not a separately replayed prefix or a certificate of
length-independent residuals or fine-cluster assignments.

``capture_roster`` writes contract.json, responses/<digest(attempt_id)>.{npy,json},
and a complete receipt.json. Keep receipt_sha256 externally, then ``load_captures``
returns dictionary Capture records and its source manifest without a model.
Complete resume requires this trusted receipt hash and never forwards/rewrites.
Interrupted prefixes resume only under the exact same full roster/contract.
Explicit replay failures and orphan files fail closed; use a new caller-provenance-
bound namespace after diagnosis, not silent regeneration of failed attempts.
"""

from __future__ import annotations

import importlib.metadata
import inspect
import json
import platform
from pathlib import Path

import numpy as np
import torch
from transformers import modeling_rope_utils

from ..core.model import LoadedModel
from .head_fv_dictionary import LAYER, WIDTH, Capture, _capture_record, _hash
from .head_fv_existing import file_hash
from .head_fv_phi_cache import cache_policy, stable_phi_cache
from .head_fv_readouts import ALIGNMENT, summarize_response
from .head_fv_specificity import MODEL, REVISION, bind_manifest, certify_geometry, digest

FORMAT = "head-fv-residual-capture-v1"
REPLAY_POLICY = {
    "sequence": "full-prompt-plus-generated-ids",
    "rotary": "shared-stable-phi-cache-policy",
    "early_window": "slice-full-replay-assignments",
    "prefix_only_replay": False,
    "future_length_independence_certified": False,
}


def _json(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _unhooked(model) -> None:
    global_hooks = torch.nn.modules.module
    if global_hooks._global_forward_hooks or global_hooks._global_forward_pre_hooks:
        raise ValueError("clean replay refuses existing global forward hooks")
    for name, module in model.named_modules():
        if module._forward_hooks or module._forward_pre_hooks or "forward" in module.__dict__:
            raise ValueError(f"clean replay refuses existing hooks/patched forwards: {name}")


def _forward_options(model) -> dict:
    parameters = inspect.signature(model.forward).parameters
    variadic = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values())
    options = {
        "use_cache": False, "output_hidden_states": False,
        "output_attentions": False, "return_dict": True,
    }
    required = ("input_ids", "attention_mask", "position_ids", *options)
    if not variadic and any(name not in parameters for name in required):
        raise ValueError("forward cannot accept the explicit clean replay kwargs")
    for name in ("logits_to_keep", "num_logits_to_keep"):
        if name in parameters:
            options[name] = 1
            break
    return options


def capture_runtime(lm: LoadedModel) -> dict:
    """Describe the fixed measurement contract without forwarding or loading weights.

    Registered hooks and instance-patched forwards are rejected, not removed.
    The caller remains responsible for supplying the actual unmodified pinned
    checkpoint; metadata inspection is not a byte-level weight certificate.
    """
    _unhooked(lm.model)
    geometry = certify_geometry(lm)
    if (lm.n_layers, lm.hidden_size) != (32, WIDTH):
        raise ValueError("LoadedModel geometry disagrees with Phi-mini")
    parameters = list(lm.model.parameters())
    dtypes = sorted({str(p.dtype) for p in parameters if p.is_floating_point()})
    devices = sorted({str(p.device) for p in parameters})
    if (
        dtypes != ["torch.bfloat16"] or devices != [str(lm.device)]
        or lm.device.type not in ("cpu", "cuda")
    ):
        raise ValueError("capture requires a single-device BF16 model, without offload")
    attention = getattr(lm.model.config, "_attn_implementation", None)
    if not isinstance(attention, str) or not attention:
        raise ValueError("capture requires an explicit attention implementation")
    here = Path(__file__).parent
    return _json({
        "format": FORMAT, "model": lm.name, "revision": lm.revision,
        "geometry": geometry, "model_config": lm.model.config.to_dict(),
        "model_class": f"{type(lm.model).__module__}.{type(lm.model).__qualname__}",
        "hook_name": lm.layer_hook_names([LAYER])[0],
        "layer": LAYER, "activation_site": "block-output-residual",
        "alignment": ALIGNMENT, "clean_replay": True,
        "replay_policy": REPLAY_POLICY,
        "cache_policy": cache_policy(lm),
        "batch_size": 1, "eval_mode": True, "inference_mode": True,
        "attention_mask": "all-ones-unpadded", "position_ids": "arange-full-ids-from-zero",
        "past_key_values": None, "forward_options": _forward_options(lm.model),
        "parameter_dtypes": dtypes, "input_device": str(lm.device),
        "attention_implementation": attention, "saved_dtype": "<f4",
        "conversion": "select-content; BF16-to-CPU; lossless-FP32",
        "empty_full_ids": "zero-row-array-without-forward",
        "versions": {
            "python": platform.python_version(),
            **{name: importlib.metadata.version(name)
               for name in ("torch", "transformers", "numpy")},
        },
        "numerics": {
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
        },
        "code_sha256": {
            name: file_hash(here / name) for name in (
                "head_fv_capture.py", "head_fv_dictionary.py", "head_fv_specificity.py",
                "head_fv_readouts.py", "head_fv_existing.py", "head_fv_phi_cache.py",
                "../core/model.py",
            )
        },
        "native_code_sha256": {
            "model": file_hash(Path(inspect.getfile(type(lm.model)))),
            "rope": file_hash(Path(inspect.getfile(modeling_rope_utils))),
        },
    })


def _generation(row: dict) -> None:
    if (
        not isinstance(row, dict)
        or not isinstance(row.get("attempt_id"), str) or not row["attempt_id"].strip()
        or any(not isinstance(row.get(key), list)
               for key in ("prompt_ids", "generated_ids", "full_ids", "content_mask"))
        or type(row.get("n_prompt")) is not int or "stop_reason" not in row
    ):
        raise ValueError("generation requires attempt_id and complete exact-ID fields")
    summarize_response(row, None, {"validation": 0}, readout_error="capture pending")


def _contract(lm: LoadedModel, generations: list[dict], source: dict, runtime: dict) -> dict:
    for key in (
        "data_sha256", "partition_sha256", "generation_file_sha256",
        "generator_runtime_sha256", "runtime_contract_sha256",
    ):
        _hash(source.get(key), key)
    for row in generations:
        _generation(row)
        if row["full_ids"] and max(row["full_ids"]) >= lm.model.config.vocab_size:
            raise ValueError("exact token ID exceeds the loaded model vocabulary")
        if len(row["full_ids"]) > lm.model.config.max_position_embeddings:
            raise ValueError("full IDs exceed the declared model context; never truncate replay")
    ids = [row["attempt_id"] for row in generations]
    if (
        not ids or len(ids) != len(set(ids)) or source.get("attempt_ids") != ids
        or not isinstance(source.get("dataset_id"), str) or not source["dataset_id"].strip()
        or source.get("cohort") not in ("discovery", "development", "evaluation")
    ):
        raise ValueError("source must bind the complete ordered generation/partition roster")
    if source["runtime_contract_sha256"] != digest(runtime):
        raise ValueError("runtime_contract_sha256 must bind the CLEAN capture runtime")
    return {
        "format": FORMAT, "source_manifest": source, "capture_runtime": runtime,
        "generations": generations, "generations_sha256": digest(generations),
    }


def _positions(row: dict) -> list[int]:
    return [row["n_prompt"] + i for i, content in enumerate(row["content_mask"]) if content]


def _input_metadata(row: dict) -> dict:
    size = len(row["full_ids"])
    return {
        "input_shape": [1, size], "input_ids_sha256": digest([row["full_ids"]]),
        "attention_mask_sha256": digest([[1] * size]),
        "position_ids_sha256": digest([list(range(size))]),
        "hook_output_shape": [1, size, WIDTH] if size else None,
        "hook_output_dtype": "torch.bfloat16" if size else None,
        "forward_calls": int(bool(size)),
    }


def _replay(lm: LoadedModel, row: dict, runtime: dict) -> tuple[np.ndarray, dict]:
    """Hook only L16 output; discard model outputs and always restore training flags."""
    metadata = _input_metadata(row)
    if not row["full_ids"]:
        return np.empty((0, WIDTH), dtype="<f4"), {**metadata, "logits_shape": None}
    _unhooked(lm.model)
    size = len(row["full_ids"])
    captured = []
    training = [(module, module.training) for module in lm.model.modules()]
    handle = None

    def retain(_module, _args, output):
        hidden = output[0] if isinstance(output, tuple) and output else output
        if (
            not isinstance(hidden, torch.Tensor) or tuple(hidden.shape) != (1, size, WIDTH)
            or hidden.dtype != torch.bfloat16
        ):
            raise ValueError("L16 output must be batch-1 full-ID BF16 residuals")
        if captured:
            raise ValueError("L16 output hook ran more than once")
        positions = torch.tensor(_positions(row), dtype=torch.long, device=hidden.device)
        array = hidden[0].index_select(0, positions).detach().cpu().float().numpy()
        if not np.isfinite(array).all():
            raise ValueError("nonfinite captured content residuals")
        captured.append(array)

    try:
        lm.model.eval()
        block = lm.model.get_submodule(runtime["hook_name"])
        handle = block.register_forward_hook(retain)
        with torch.inference_mode(), stable_phi_cache(lm):
            ids = torch.tensor([row["full_ids"]], dtype=torch.long, device=lm.device)
            output = lm.model(
                input_ids=ids, attention_mask=torch.ones_like(ids),
                position_ids=torch.arange(size, device=lm.device).unsqueeze(0),
                **runtime["forward_options"],
            )
            if len(captured) != 1 or getattr(output, "past_key_values", None) is not None:
                raise ValueError("replay must capture L16 exactly once and return no KV cache")
            shape = list(output.logits.shape)
            limited = any(key in runtime["forward_options"]
                          for key in ("logits_to_keep", "num_logits_to_keep"))
            if len(shape) != 3 or shape[:2] != [1, 1 if limited else size]:
                raise ValueError("forward did not honor the declared logits shape")
            metadata["logits_shape"] = shape
            del output
        return captured[0], metadata
    finally:
        if handle is not None:
            handle.remove()
        for module, flag in training:
            module.training = flag


def _base_evidence(row: dict, contract: dict) -> dict:
    runtime, source = contract["capture_runtime"], contract["source_manifest"]
    return {
        "attempt_id": row["attempt_id"], "generation": row,
        "run_contract_sha256": digest(contract),
        "source_data_sha256": source["data_sha256"],
        "capture_code_sha256": runtime["code_sha256"]["head_fv_capture.py"],
        "model": MODEL, "revision": REVISION, "layer": LAYER,
        "activation_site": "block-output-residual", "alignment": ALIGNMENT, "clean_replay": True,
        "replay_policy": runtime["replay_policy"],
        "cache_policy": runtime["cache_policy"],
        "runtime_contract_sha256": source["runtime_contract_sha256"],
        "full_ids_sha256": digest(row["full_ids"]),
        "content_mask_sha256": digest(row["content_mask"]), "row_full_positions": _positions(row),
        "shape": [sum(row["content_mask"]), WIDTH], "dtype": "<f4",
    }


def _paths(directory: Path, row: dict) -> tuple[Path, Path, Path]:
    stem = directory / "responses" / digest(row["attempt_id"])
    return stem.with_suffix(".npy"), stem.with_suffix(".json"), stem.with_suffix(".error.json")


def _read_capture(directory: Path, row: dict, contract: dict) -> tuple[Capture, dict]:
    array_path, manifest_path, error_path = _paths(directory, row)
    if error_path.exists() or error_path.is_symlink():
        raise ValueError(f"recorded replay failure; refuse regeneration: {error_path}")
    if any(path.is_symlink() or not path.is_file() for path in (array_path, manifest_path)):
        raise ValueError("incomplete/orphan capture artifacts; refuse regeneration")
    wrapped = json.loads(manifest_path.read_text())
    evidence = wrapped["manifest"]
    manifest_hash = _hash(wrapped.get("manifest_sha256"), "capture manifest hash")
    if digest(evidence) != manifest_hash:
        raise ValueError("capture manifest hash mismatch")
    base = _base_evidence(row, contract)
    if digest({key: evidence.get(key) for key in base}) != digest(base):
        raise ValueError("capture source/generation/runtime contract mismatch")
    metadata = evidence["forward"]
    expected = _input_metadata(row)
    if {key: metadata.get(key) for key in expected} != expected:
        raise ValueError("capture input/consumed-position metadata mismatch")
    size = len(row["full_ids"])
    limited = any(key in contract["capture_runtime"]["forward_options"]
                  for key in ("logits_to_keep", "num_logits_to_keep"))
    logits = metadata.get("logits_shape")
    if (
        (not size and logits is not None)
        or (size and (not isinstance(logits, list) or len(logits) != 3
                      or logits[:2] != [1, 1 if limited else size]
                      or type(logits[2]) is not int or logits[2] < 1))
    ):
        raise ValueError("capture logits metadata mismatch")
    capture = Capture(row["attempt_id"], row, array_path, evidence, manifest_hash)
    source = contract["source_manifest"]
    _capture_record(capture, source["runtime_contract_sha256"], source["data_sha256"])
    entry = {
        "attempt_id": row["attempt_id"], "content_rows": sum(row["content_mask"]),
        "manifest_file": str(manifest_path.relative_to(directory)),
        "manifest_file_sha256": file_hash(manifest_path), "manifest_sha256": manifest_hash,
        "activation_file": str(array_path.relative_to(directory)),
        "activations_sha256": evidence["activations_sha256"],
    }
    return capture, entry


def _receipt(directory: Path, contract: dict, entries: list[dict]) -> dict:
    generations = contract["generations"]
    if len(entries) != len(generations):
        raise ValueError("cannot publish an incomplete capture receipt")
    receipt = {
        "format": FORMAT, "contract_sha256": digest(contract),
        "contract_file_sha256": file_hash(directory / "contract.json"),
        "runtime_contract_sha256": contract["source_manifest"]["runtime_contract_sha256"],
        "replay_policy": contract["capture_runtime"]["replay_policy"],
        "cache_policy": contract["capture_runtime"]["cache_policy"],
        "responses": entries,
        "accounting": {
            "attempts": len(generations), "captured_attempts": len(entries),
            "content_rows": sum(entry["content_rows"] for entry in entries),
            "empty_content_responses": sum(not any(row["content_mask"]) for row in generations),
            "generation_errors": sum(row["stop_reason"] == "error" for row in generations),
            "capture_errors": 0, "capture_complete": True,
        },
    }
    return {**receipt, "receipt_sha256": digest(receipt)}


def load_captures(
    directory: Path, *, expected_receipt_sha256: str
) -> tuple[list[Capture], dict]:
    """Verify a COMPLETE externally sealed receipt; return captures and dictionary source."""
    directory = Path(directory)
    expected = _hash(expected_receipt_sha256, "expected receipt hash")
    if (directory / "receipt.json").is_symlink():
        raise ValueError("aggregate receipt must not be a symlink")
    receipt = json.loads((directory / "receipt.json").read_text())
    payload = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if receipt.get("receipt_sha256") != expected or digest(payload) != expected:
        raise ValueError("aggregate receipt hash mismatch")
    contract_path = directory / "contract.json"
    if contract_path.is_symlink() or file_hash(contract_path) != receipt["contract_file_sha256"]:
        raise ValueError("capture contract file hash mismatch")
    contract = json.loads(contract_path.read_text())
    if contract.get("format") != FORMAT or digest(contract) != receipt["contract_sha256"]:
        raise ValueError("capture contract hash mismatch")
    if digest(contract["capture_runtime"]) != receipt["runtime_contract_sha256"]:
        raise ValueError("capture runtime hash mismatch")
    captures, entries = [], []
    for row in contract["generations"]:
        capture, entry = _read_capture(directory, row, contract)
        captures.append(capture)
        entries.append(entry)
    if _receipt(directory, contract, entries) != receipt:
        raise ValueError("aggregate receipt does not match every capture component")
    return captures, contract["source_manifest"]


def capture_roster(
    lm: LoadedModel, directory: Path, generations: list[dict], *, source_manifest: dict,
    expected_receipt_sha256: str | None = None,
) -> dict:
    """Capture a frozen ordered roster, or verify identical completed/prefix resume.

    Runtime/Value/Type replay exceptions get immutable .error.json records and
    propagate; no successful aggregate is returned for partial captures. No
    retries, health exclusions, batch resizing, or context truncation occur.
    """
    directory = Path(directory)
    runtime = capture_runtime(lm)
    contract = _contract(lm, _json(generations), _json(source_manifest), runtime)
    contract_path, receipt_path = directory / "contract.json", directory / "receipt.json"
    if contract_path.is_symlink() or receipt_path.is_symlink():
        raise ValueError("capture contract and receipt must not be symlinks")
    if receipt_path.exists():
        if expected_receipt_sha256 is None:
            raise ValueError("complete resume requires its trusted expected_receipt_sha256")
        if contract_path.is_symlink() or not contract_path.is_file():
            raise ValueError("complete receipt lacks its original immutable contract")
        bind_manifest(contract_path, contract)
        load_captures(directory, expected_receipt_sha256=expected_receipt_sha256)
        return json.loads(receipt_path.read_text())
    if expected_receipt_sha256 is not None:
        raise ValueError("expected complete receipt is missing")
    if directory.exists() and any(directory.iterdir()) and not contract_path.is_file():
        raise ValueError("nonempty capture directory lacks its immutable contract")
    bind_manifest(contract_path, contract)
    ready = {}
    for row in contract["generations"]:
        paths = _paths(directory, row)
        if any(path.exists() or path.is_symlink() for path in paths):
            _, ready[row["attempt_id"]] = _read_capture(directory, row, contract)
    entries = []
    for row in contract["generations"]:
        if row["attempt_id"] not in ready:
            array_path, manifest_path, error_path = _paths(directory, row)
            try:
                array, metadata = _replay(lm, row, runtime)
            except (RuntimeError, ValueError, TypeError) as error:
                bind_manifest(error_path, {
                    "attempt_id": row["attempt_id"], "generation": row,
                    "run_contract_sha256": digest(contract), "status": "capture_error",
                    "error_type": type(error).__name__, "error": str(error),
                })
                raise
            array_path.parent.mkdir(parents=True, exist_ok=True)
            with array_path.open("xb") as handle:
                np.save(handle, array, allow_pickle=False)
            evidence = {
                **_base_evidence(row, contract), "activations_sha256": file_hash(array_path),
                "forward": metadata,
            }
            bind_manifest(
                manifest_path, {"manifest": evidence, "manifest_sha256": digest(evidence)}
            )
            del array
            _, ready[row["attempt_id"]] = _read_capture(directory, row, contract)
        entries.append(ready[row["attempt_id"]])
    if capture_runtime(lm) != runtime:
        raise ValueError("capture runtime changed during replay; no aggregate receipt published")
    verified = [_read_capture(directory, row, contract)[1] for row in contract["generations"]]
    if verified != entries:
        raise ValueError("capture components changed during replay; no aggregate receipt published")
    receipt = _receipt(directory, contract, entries)
    bind_manifest(receipt_path, receipt)
    load_captures(directory, expected_receipt_sha256=receipt["receipt_sha256"])
    return receipt
