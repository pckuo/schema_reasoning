"""Collect model responses to the generated probes (the input the judge scores, A3.1).

Reads a run's ``probes.jsonl`` (stimulus + question per (item, task)), runs each ``probe_prompt``
through a model, and writes ``responses.jsonl`` + raw stdout under ``responses_raw/``. Kept
separate from judging so re-scoring never re-spends generation credits.

Examples
--------
    python -m dev.schema_llm.subjects.collect --run results/schema_llm/<stamp>
    python -m dev.schema_llm.subjects.collect --run results/schema_llm/<stamp> --model gpt-5
    # locally loaded Phi-4 (same model + chat prompting as the capture harness, B0):
    python -m dev.schema_llm.subjects.collect --run results/schema_llm/<stamp> --backend local \
        --model microsoft/Phi-4-reasoning --system phi-reasoning --sample --tasks identify
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ..core.llm import run_copilot_cli
from ..core.runio import (
    JsonlAppender,
    done_keys,
    model_slug,
    read_jsonl,
    responses_path,
    responses_raw_dir,
)
from .schema_incontext import apply_incontext, schema_of_run

Responder = Callable[..., str]

# "argument not supplied" sentinel: distinguishes "leave the client's own default alone" from an
# explicit ``None`` (which the OpenAI SDK reads as "no timeout at all").
_UNSET = object()

# Phi-4-reasoning REQUIRES this ChatML system prompt for correct behavior (per its model card):
# it elicits the <think> Thought </think> Solution format the model was trained on. Matches the
# reference PHI_SYSTEM_MESSAGE (code/reference/algotrace/src/data_loading.py). Defined here (a
# torch-free module) rather than in the GPU-only capture harness so the pure-HTTP vLLM client path
# (make_vllm_responder, `--extra vllm` = just the openai SDK) does not transitively import torch.
PHI_REASONING_SYSTEM = (
    "You are Phi, a language model trained by Microsoft to help users. Your role as an "
    "assistant involves thoroughly exploring questions through a systematic thinking "
    "process before providing the final precise and accurate solutions. This requires "
    "engaging in a comprehensive cycle of analysis, summarizing, exploration, "
    "reassessment, reflection, backtracing, and iteration to develop well-considered "
    "thinking process. Please structure your response into two main sections: Thought "
    "and Solution using the specified format: <think> {Thought section} </think> "
    "{Solution section}. In the Thought section, detail your reasoning process in "
    "steps. Each step should include detailed considerations such as analysing "
    "questions, summarizing relevant findings, brainstorming new ideas, verifying the "
    "accuracy of the current steps, refining any errors, and revisiting previous "
    "steps. In the Solution section, based on various attempts, explorations, and "
    "reflections from the Thought section, systematically present the final solution "
    "that you deem correct. The Solution section should be logical, accurate, and "
    "concise and detail necessary steps needed to reach the conclusion. Now, try to "
    "solve the following question through the above guidelines:"
)


def _reopen_think(text: str) -> str:
    """Restore the opening ``<think>`` when the chat template emitted it in the PROMPT.

    DeepSeek-R1-Distill's template ends the generation prompt with ``<|Assistant|><think>\\n``, so
    the completion closes the block (``… </think> {Solution}``) but never opens it. Every other
    reasoning subject stores ``<think> … </think> {Solution}``, and the downstream trace splitters
    key on the tag pair, so prepend the tag the template already consumed. A no-op for a response
    that already opens the block or has no ``</think>`` at all.
    """
    if "</think>" in text and "<think>" not in text:
        return f"<think>\n{text}"
    return text


def _read_probes(run_dir: Path) -> list[dict]:
    path = run_dir / "probes.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"no probes.jsonl in {run_dir} (run the pilot first)")
    return read_jsonl(path)


def run_collect(
    run_dir: Path,
    model: str | None = None,
    tasks: list[str] | None = None,
    call: Responder = run_copilot_cli,
    limit: int | None = None,
    tag: str | None = None,
    resume: bool = True,
    schema_incontext: str = "none",
    concurrency: int = 1,
    scenario_ids: set[str] | None = None,
) -> dict:
    """Run every probe_prompt through ``call`` and record the responses. ``call`` is injectable
    (defaults to the Copilot CLI) so tests can pass a deterministic fake.

    Outputs are namespaced by ``tag`` (the subject slug) so several models can be collected into
    the same run dir without clobbering: ``responses.<tag>.jsonl`` (``tag=None`` keeps the legacy
    ``responses.jsonl``). Each response is appended + flushed as it is produced and, when ``resume``
    is set, a re-run skips the ``(scenario_id, task)`` already present -- so a dropped connection
    or a killed process only re-does the unfinished probes. ``concurrency`` > 1 fans the
    (network-bound) responder calls out over a thread pool -- for the vllm/trapi servers, which then
    batch -- while results are still written on ONE thread, so the skip/resume guarantees hold."""
    if schema_incontext != "none" and not tag:
        raise ValueError(
            "schema_incontext requires an explicit tag (e.g. '<slug>-ic') so preamble responses "
            "do not overwrite a baseline responses file"
        )
    run_dir = Path(run_dir)
    probes = _read_probes(run_dir)
    if tasks:
        probes = [p for p in probes if p.get("task") in tasks]
    if scenario_ids is not None:
        # Collect only part of a corpus. The binding battery draws a stratified 72-item subsample
        # of binding_v2's 1,116, so re-sampling the whole corpus would be ~15x the work for rows
        # the panel never reads.
        probes = [p for p in probes if p.get("scenario_id") in scenario_ids]
    if limit is not None:
        probes = probes[:limit]
    if schema_incontext != "none":
        probes = apply_incontext(probes, schema_of_run(run_dir), schema_incontext)

    out = responses_path(run_dir, tag)
    raw_dir = responses_raw_dir(run_dir, tag)
    raw_dir.mkdir(parents=True, exist_ok=True)
    done = done_keys(out) if resume else set()
    pending = [p for p in probes if (p["scenario_id"], p["task"]) not in done]
    n_skipped = len(probes) - len(pending)
    n_new = 0

    def _fetch(probe: dict) -> tuple[dict, str]:
        return probe, call(probe["probe_prompt"], model=model)

    with JsonlAppender(out) as sink:
        def _write(probe: dict, raw: str) -> None:
            nonlocal n_new
            sid, task = probe["scenario_id"], probe["task"]
            (raw_dir / f"{sid}__{task}.txt").write_text(raw, encoding="utf-8")
            row = {"scenario_id": sid, "task": task, "model": model, "response": raw.strip()}
            if probe.get("schema_incontext"):
                row["schema_incontext"] = probe["schema_incontext"]
            sink.write(row)
            n_new += 1
            print(f"[ok] {sid}__{task} ({len(raw.strip())} chars)")

        if concurrency > 1:
            # Fan the (network-bound) responder calls out over a thread pool so a vllm/trapi server
            # actually batches; write each result on THIS thread as it lands, keeping the sink a
            # single writer (a killed run keeps every flushed row; --resume redoes only the rest).
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                for fut in as_completed([pool.submit(_fetch, p) for p in pending]):
                    _write(*fut.result())
        else:
            for probe in pending:
                _write(*_fetch(probe))

    total = len(done) + n_new
    tail = f" ({n_skipped} already done, skipped)" if n_skipped else ""
    print(f"\ncollected {n_new} responses{tail} -> {out} [{total} total]")
    return {"n": total, "n_new": n_new, "n_skipped": n_skipped, "out": str(out)}


# --------------------------------------------------------------------------- local HF backend


def _generate_local(
    lm,
    prompt: str,
    *,
    system_message: str | None,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    enable_thinking: bool | None = None,
) -> str:
    """One chat-templated generation on a loaded HF model; returns only the decoded reply.
    Delegates to the shared ``model.generate_reply`` (also used by ``capture.capture_probe``) so a
    collected response is byte-identical to what the capture harness produces for the same probe
    (B0)."""
    from ..core.model import generate_reply

    _, _, reply = generate_reply(
        lm, prompt, system_message=system_message, max_new_tokens=max_new_tokens,
        do_sample=do_sample, temperature=temperature, top_p=top_p, top_k=top_k,
        enable_thinking=enable_thinking,
    )
    return reply


def make_local_responder(
    model_name: str | None = None,
    *,
    dtype: str = "bfloat16",
    system: str | None = None,
    max_new_tokens: int = 1024,
    do_sample: bool = False,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    enable_thinking: bool | None = None,
) -> Responder:
    """Load a local HF causal LM (Phi-4 by default) ONCE and return a ``Responder`` closure that
    generates a reply per probe -- the same model + chat prompting as the capture harness (B0),
    so behavioral responses and captured activations describe comparable forward passes. torch
    and the weights are imported lazily so the default Copilot-CLI backend never needs the gpu
    extra. ``system="phi-reasoning"`` injects the Phi-4-reasoning system prompt (shared with
    ``capture`` as the single source of truth); ``enable_thinking`` toggles Qwen3's hybrid
    ``<think>`` mode (``None`` keeps the model's template default)."""
    from ..core.model import DEFAULT_MODEL, load_model

    lm = load_model(model_name or DEFAULT_MODEL, dtype=dtype)
    system_message: str | None = None
    if system == "phi-reasoning":
        system_message = PHI_REASONING_SYSTEM

    def _respond(prompt: str, model: str | None = None) -> str:  # noqa: ARG001 (model preloaded)
        return _generate_local(
            lm, prompt, system_message=system_message, max_new_tokens=max_new_tokens,
            do_sample=do_sample, temperature=temperature, top_p=top_p, top_k=top_k,
            enable_thinking=enable_thinking,
        )

    return _respond


# --------------------------------------------------------------------------- TRAPI (Azure OpenAI)


def make_trapi_responder(
    deployment: str,
    *,
    base_url: str | None = None,
    scope: str = "api://trapi/.default",
    max_completion_tokens: int | None = 8192,
    max_retries: int = 8,
    timeout: float | None = None,
) -> Responder:
    """Load an Azure OpenAI (TRAPI) chat-completions client ONCE and return a ``Responder`` closure
    that answers one probe per call. Auth is Entra ID via ``az login`` / managed identity -- a
    bearer-token provider that auto-refreshes per request, matching the TRAPI reference script.
    ``openai`` + ``azure-identity`` are imported lazily (the ``api`` extra) so the default backends
    never need them.

    The shared TRAPI endpoint throttles aggressively, so throttling / transient server errors are
    retried here with exponential backoff + jitter (honoring the server's ``Retry-After``) rather
    than failing the whole batch; only a sustained outage past ``max_retries`` propagates (and
    ``--resume`` then re-does just the unfinished probes).

    ``deployment`` is the TRAPI model deployment name (e.g. ``gpt-5.5_2026-04-24`` or
    ``DeepSeek-R1_1``). ``max_completion_tokens`` budgets the reply; reasoning models (GPT-5-family,
    DeepSeek-R1) spend reasoning tokens from this budget, so keep it generous or the visible answer
    can be empty. DeepSeek-R1's chain of thought is returned in a separate ``reasoning_content``
    field and folded back into an inline ``<think>...</think>`` block (see ``_respond``)."""
    import random
    import sys
    import time

    import openai
    from azure.identity import (
        AzureCliCredential,
        ChainedTokenCredential,
        ManagedIdentityCredential,
        get_bearer_token_provider,
    )
    from openai import OpenAI

    host = os.environ.get("TRAPI_HOST", "https://trapi.research.microsoft.com")
    endpoint = base_url or os.environ.get(
        "OPENAI_BASE_URL", f"{host}/redmond/interactive/openai/v1"
    )
    token_provider = get_bearer_token_provider(
        ChainedTokenCredential(AzureCliCredential(), ManagedIdentityCredential()), scope
    )
    # drive retries ourselves (below) so we back off patiently and honor Retry-After
    client = OpenAI(base_url=endpoint, api_key=token_provider, max_retries=0)
    retryable = (
        openai.RateLimitError,
        openai.APITimeoutError,
        openai.APIConnectionError,
        openai.InternalServerError,
    )

    def _retry_delay(exc: Exception, attempt: int, base: float = 1.5, cap: float = 60.0) -> float:
        """Server Retry-After (if present) or exponential backoff, whichever is larger, + jitter."""
        server = 0.0
        resp = getattr(exc, "response", None)
        if resp is not None:
            try:
                hdr = resp.headers.get("retry-after")
                server = float(hdr) if hdr is not None else 0.0
            except (TypeError, ValueError, AttributeError):
                server = 0.0
        return max(server, min(cap, base * (2**attempt))) + random.uniform(0.0, 1.0)

    def _respond(prompt: str, model: str | None = None) -> str:  # noqa: ARG001 (deployment pinned)
        kwargs: dict = {"model": deployment, "messages": [{"role": "user", "content": prompt}]}
        if max_completion_tokens is not None:
            kwargs["max_completion_tokens"] = max_completion_tokens
        if timeout is not None:
            kwargs["timeout"] = timeout
        for attempt in range(max_retries + 1):
            try:
                msg = client.chat.completions.create(**kwargs).choices[0].message
                content = (msg.content or "").strip()
                # DeepSeek-R1 returns its chain of thought in a separate ``reasoning_content``
                # field (not inline in ``content``). Fold it back into an inline
                # ``<think>...</think>`` block so a stored TRAPI reasoning response matches the
                # Phi-4-reasoning format (CoT + Solution) and the judge's ``strip_reasoning()``
                # grades only the final answer. OpenAI GPT-5-family models keep reasoning hidden
                # (no ``reasoning_content``), so this is a no-op for them.
                reasoning = (getattr(msg, "reasoning_content", None) or "").strip()
                return f"<think>\n{reasoning}\n</think>\n{content}" if reasoning else content
            except retryable as exc:
                if attempt == max_retries:
                    raise
                delay = _retry_delay(exc, attempt)
                print(f"[trapi] {type(exc).__name__} (attempt {attempt + 1}/{max_retries}); "
                      f"retry in {delay:.1f}s", file=sys.stderr, flush=True)
                time.sleep(delay)
        raise RuntimeError("unreachable")  # loop always returns or raises above

    return _respond


# --------------------------------------------------------------------------- vLLM (local server)


def make_vllm_responder(
    model: str,
    *,
    base_url: str | None = None,
    api_key: str = "EMPTY",
    system: str | None = None,
    max_tokens: int | None = 1024,
    do_sample: bool = False,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    enable_thinking: bool | None = None,
    max_retries: int = 6,
    timeout: float | None | object = _UNSET,
) -> Responder:
    """Answer probes with a LOCALLY hosted vLLM OpenAI-compatible server -- the fast alternative to
    HF hosting for LONG response-only runs (reasoning models: Phi-4-reasoning[-plus], Qwen3-14B
    thinking) when NO latents are needed. Latent capture still goes through the HF harness
    (:mod:`capture`); vLLM outputs are NOT byte-identical to HF (different kernels / RNG), so use
    this only for behavioural responses (the judge grades the final answer), never as the source of
    a residual re-forward.

    vLLM's continuous batching removes the head-of-line blocking of HF batched decode (a batch runs
    until the LONGEST trace in it finishes), which is the throughput win for variable-length
    reasoning traces. The server runs in its OWN isolated venv (vLLM hard-pins ``torch==2.11.0``,
    conflicting with the repo's torch), started by ``scripts/schema_llm/vllm_serve.sh``; only the
    thin ``openai`` client is needed here (the ``vllm`` extra), imported lazily so the default
    backends never require it.

    ``system="phi-reasoning"`` injects the Phi-4-reasoning ChatML system prompt (shared with
    ``capture`` as the single source of truth); ``enable_thinking`` toggles Qwen3's hybrid
    ``<think>`` mode (forwarded to the server via ``chat_template_kwargs``). Started WITHOUT
    ``--reasoning-parser`` the model's inline ``<think>...</think>`` stays in ``content`` and
    matches the HF-local format (the judge's ``strip_reasoning`` grades only the Solution); if a
    parser IS enabled the CoT comes back in ``reasoning_content`` and is folded back inline, exactly
    like the DeepSeek-R1 TRAPI path. Templates that OPEN the think block in the generation prompt
    (DeepSeek-R1-Distill appends ``<|Assistant|><think>\\n``) return a completion that closes but
    never opens the block; :func:`_reopen_think` restores the leading tag so every stored reasoning
    response carries the same ``<think>…</think>`` shape."""
    import random
    import sys
    import time

    import openai
    from openai import OpenAI

    system_message: str | None = None
    if system == "phi-reasoning":
        system_message = PHI_REASONING_SYSTEM

    endpoint = base_url or os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
    # a local single-tenant server: drive our own light retries so a still-warming-up / restarting
    # server rides through instead of failing the batch (--resume then re-does only the rest).
    # ``timeout`` matters for UNCAPPED generation: the SDK's 600s default aborts a long reasoning
    # trace mid-flight and the retry then re-decodes it from scratch, so pass ``None`` (no timeout)
    # when the token budget is unbounded.
    client_kwargs: dict = {"base_url": endpoint, "api_key": api_key, "max_retries": 0}
    if timeout is not _UNSET:
        client_kwargs["timeout"] = timeout
    client = OpenAI(**client_kwargs)
    retryable = (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError)

    extra_body: dict = {}
    if do_sample and top_k is not None:
        extra_body["top_k"] = top_k  # top-k is a vLLM sampling extension (not native OpenAI)
    if enable_thinking is not None:
        extra_body["chat_template_kwargs"] = {"enable_thinking": enable_thinking}

    def _retry_delay(attempt: int, base: float = 1.0, cap: float = 30.0) -> float:
        return min(cap, base * (2**attempt)) + random.uniform(0.0, 0.5)

    served_model = model  # requests stay pinned to the served model; caller's model arg is ignored

    def _respond(prompt: str, model: str | None = None) -> str:  # noqa: ARG001 (server model pinned)
        messages: list[dict] = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        kwargs: dict = {"model": served_model, "messages": messages}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        # greedy = temperature 0; sampling forwards the algotrace temp / top-p (+ top-k via body)
        kwargs["temperature"] = temperature if (do_sample and temperature is not None) else 0.0
        if do_sample and top_p is not None:
            kwargs["top_p"] = top_p
        if extra_body:
            kwargs["extra_body"] = extra_body
        for attempt in range(max_retries + 1):
            try:
                msg = client.chat.completions.create(**kwargs).choices[0].message
                content = (msg.content or "").strip()
                # fold a reasoning-parser server's separate CoT back inline (no-op without a parser)
                reasoning = (getattr(msg, "reasoning_content", None) or "").strip()
                if reasoning:
                    return f"<think>\n{reasoning}\n</think>\n{content}"
                return _reopen_think(content)
            except retryable as exc:
                if attempt == max_retries:
                    raise
                delay = _retry_delay(attempt)
                print(
                    f"[vllm] {type(exc).__name__} (attempt {attempt + 1}/{max_retries}); "
                    f"retry in {delay:.1f}s",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(delay)
        raise RuntimeError("unreachable")  # loop always returns or raises above

    return _respond


def main() -> None:
    p = argparse.ArgumentParser(description="collect model responses to schema_llm probes")
    p.add_argument("--run", required=True, help="run dir containing probes.jsonl")
    p.add_argument("--backend", choices=["copilot", "local", "trapi", "vllm"], default="copilot",
                   help="copilot: GitHub Copilot CLI; local: a locally loaded HF model (Phi-4); "
                        "trapi: Azure OpenAI (TRAPI) chat-completions; vllm: a LOCAL vLLM "
                        "OpenAI-compatible server (fast alternative to local for long "
                        "response-only reasoning runs; needs the vllm extra + a running "
                        "vllm_serve.sh)")
    p.add_argument("--model", default=None,
                   help="copilot model id, local/vllm HF name (default microsoft/phi-4), or TRAPI "
                        "deployment name (default gpt-5.5_2026-04-24; e.g. DeepSeek-R1_1)")
    p.add_argument("--tag", default=None,
                   help="output namespace (subject slug) -> responses.<tag>.jsonl; defaults to the "
                        "model slug for local, 'gpt-5.5' for trapi, and none (legacy "
                        "responses.jsonl) for copilot")
    p.add_argument("--no-resume", dest="resume", action="store_false",
                   help="re-collect every probe instead of skipping those already in the output")
    p.add_argument("--tasks", default=None, help="comma-separated task filter (e.g. identify)")
    p.add_argument("--limit", type=int, default=None, help="cap number of probes (dry runs)")
    p.add_argument("--scenario-ids-file", default=None,
                   help="restrict to the scenario_ids listed in this file (one per line); used to "
                        "re-sample only the subsample of a corpus that a panel actually reads")
    # local-backend generation knobs (mirror capture.py so behavior matches the capture run)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "auto"])
    p.add_argument("--system", default="none", choices=["none", "phi-reasoning"],
                   help="system-prompt preset (use phi-reasoning for microsoft/Phi-4-reasoning)")
    p.add_argument("--think", default="auto", choices=["auto", "on", "off"],
                   help="Qwen3 hybrid thinking: on/off set apply_chat_template(enable_thinking); "
                        "auto keeps the template default (no-op for non-Qwen3 models)")
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="reply token budget; 0 = UNCAPPED for the vllm/trapi backends (no limit "
        "is sent, so a vLLM server generates until EOS or --max-model-len). "
        "Reasoning traces on hard tasks routinely exceed any fixed cap, and a "
        "truncated trace has no final answer to grade.",
    )
    p.add_argument("--sample", action="store_true", help="sample instead of greedy decoding")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--concurrency", type=int, default=1,
                   help="vllm/trapi: number of probes in flight at once (thread pool -> the server "
                        "actually batches; 1 = sequential). Not supported for local/copilot.")
    # trapi-backend knob
    p.add_argument(
        "--max-completion-tokens",
        type=int,
        default=8192,
        help="trapi: reply token budget (reasoning models spend some on hidden "
        "reasoning, so budget generously); 0 = send no limit",
    )
    # vllm-backend knob (local + sampling knobs above are shared)
    p.add_argument("--vllm-base-url", default=None,
                   help="vllm: base URL of the local vLLM OpenAI server "
                        "(default $VLLM_BASE_URL or http://localhost:8000/v1)")
    p.add_argument(
        "--request-timeout",
        type=float,
        default=None,
        help="vllm: per-request HTTP timeout in seconds; 0 = no timeout. Long "
        "reasoning traces exceed the OpenAI SDK's 600s default, which aborts them "
        "mid-decode and then re-runs them from scratch, so pair 0 with "
        "--max-new-tokens 0. Unset keeps the SDK default.",
    )
    p.add_argument("--schema-incontext", choices=["none", "named"], default="none",
                   help="prepend the schema's in-context preamble (named register) to each probe; "
                        "requires an explicit --tag (e.g. <slug>-ic) so it never clobbers a "
                        "baseline responses file")
    args = p.parse_args()
    if args.schema_incontext != "none" and not args.tag:
        p.error("--schema-incontext requires an explicit --tag (e.g. phi-4-ic)")
    if args.concurrency > 1 and args.backend in ("local", "copilot"):
        p.error("--concurrency > 1 is only for the vllm/trapi backends (local is a single GPU "
                "model; copilot is one CLI process)")
    if args.max_new_tokens <= 0 and args.backend in ("local", "copilot"):
        p.error(
            "--max-new-tokens 0 (uncapped) is only for the vllm/trapi backends; the local HF "
            "generate() call needs an explicit budget"
        )
    # None = send no token limit at all, letting the server run to EOS / its context window
    max_new_tokens = args.max_new_tokens if args.max_new_tokens > 0 else None
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()] if args.tasks else None
    ids = None
    if args.scenario_ids_file:
        ids = {ln.strip() for ln in Path(args.scenario_ids_file).read_text(
            encoding="utf-8").splitlines() if ln.strip()}
        print(f"[ids] restricting to {len(ids)} scenario_ids from {args.scenario_ids_file}")

    if args.backend == "local":
        from ..core.model import DEFAULT_MODEL

        model_name = args.model or DEFAULT_MODEL
        responder = make_local_responder(
            model_name,
            dtype=args.dtype,
            system=args.system,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.sample,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            enable_thinking={"auto": None, "on": True, "off": False}[args.think],
        )
        run_collect(
            Path(args.run),
            model_name,
            tasks,
            call=responder,
            limit=args.limit,
            tag=args.tag or model_slug(model_name),
            resume=args.resume,
            schema_incontext=args.schema_incontext,
        )
    elif args.backend == "trapi":
        deployment = args.model or os.environ.get("OPENAI_MODEL", "gpt-5.5_2026-04-24")
        responder = make_trapi_responder(
            deployment,
            max_completion_tokens=args.max_completion_tokens
            if args.max_completion_tokens > 0
            else None,
        )
        run_collect(
            Path(args.run),
            deployment,
            tasks,
            call=responder,
            limit=args.limit,
            tag=args.tag or "gpt-5.5",
            resume=args.resume,
            schema_incontext=args.schema_incontext,
            concurrency=args.concurrency,
            scenario_ids=ids,
        )
    elif args.backend == "vllm":
        from ..core.model import DEFAULT_MODEL

        model_name = args.model or DEFAULT_MODEL
        vllm_kwargs: dict = {}
        if args.request_timeout is not None:
            # 0 = "no timeout"; the SDK reads an explicit None that way
            vllm_kwargs["timeout"] = args.request_timeout or None
        responder = make_vllm_responder(
            model_name,
            base_url=args.vllm_base_url,
            system=args.system,
            max_tokens=max_new_tokens,
            do_sample=args.sample,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            enable_thinking={"auto": None, "on": True, "off": False}[args.think],
            **vllm_kwargs,
        )
        run_collect(
            Path(args.run),
            model_name,
            tasks,
            call=responder,
            limit=args.limit,
            tag=args.tag or model_slug(model_name),
            resume=args.resume,
            schema_incontext=args.schema_incontext,
            concurrency=args.concurrency,
            scenario_ids=ids,
        )
    else:
        run_collect(
            Path(args.run),
            args.model,
            tasks,
            limit=args.limit,
            tag=args.tag,
            resume=args.resume,
            schema_incontext=args.schema_incontext,
            scenario_ids=ids,
        )


if __name__ == "__main__":
    main()
