"""Shared LLM-CLI + response-parsing utilities.

Extracted from ``generator.py`` / ``judge.py`` so the Copilot-CLI wrapper, JSON extraction, and
reasoning-trace stripping are usable across ``pipeline`` (generation), ``subjects`` (collection +
judging), and ``geometry`` (LLM cluster annotation) without any of them importing another cluster.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile


class LLMParseError(ValueError):
    """No parseable JSON object could be extracted from an LLM response. Subclasses ``ValueError``
    so existing callers that catch ``ValueError`` keep working."""


def _iter_json_objects(text: str):
    """Yield each balanced top-level ``{...}`` substring, tracking string literals + escapes so a
    brace INSIDE a JSON string value does not fool the depth counter (the old extract_json bug)."""
    i, n = 0, len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth = 0
        in_str = esc = False
        j = i
        while j < n:
            c = text[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    yield text[i : j + 1]
                    break
            j += 1
        i = j + 1


def extract_json(text: str, require_key: str | None = None) -> dict:
    """Pull a JSON object out of a model's stdout, robustly. Prefers a ```json fenced block; tracks
    string literals so braces inside strings don't truncate; among balanced ``{...}`` candidates
    returns the LARGEST that parses to a dict (skipping a decoy ``{}`` preamble). If ``require_key``
    is given, only objects containing that key qualify. Raises ``LLMParseError`` on no match."""
    def _best(cands: list[str]) -> tuple[int, dict] | None:
        best: tuple[int, dict] | None = None
        for cand in cands:
            try:
                obj = json.loads(cand)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict) or (require_key is not None and require_key not in obj):
                continue
            if best is None or len(cand) > best[0]:
                best = (len(cand), obj)
        return best

    regions = re.findall(r"```(?:json)?\s*(\{.*)", text, re.DOTALL)
    fenced = [c for region in regions for c in _iter_json_objects(region)]
    best = _best(fenced) or _best(list(_iter_json_objects(text)))
    if best is None:
        suffix = f" with key {require_key!r}" if require_key else ""
        raise LLMParseError(f"no parseable JSON object{suffix} in output")
    return best[1]


def _iter_json_arrays(text: str):
    """Yield each balanced top-level ``[...]`` substring, tracking string literals + escapes so a
    bracket INSIDE a JSON string value does not fool the depth counter (the array analog of
    :func:`_iter_json_objects`)."""
    i, n = 0, len(text)
    while i < n:
        if text[i] != "[":
            i += 1
            continue
        depth = 0
        in_str = esc = False
        j = i
        while j < n:
            c = text[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    yield text[i : j + 1]
                    break
            j += 1
        i = j + 1


def _strip_trailing_commas(s: str) -> str:
    """Remove JSON trailing commas (``,`` right before ``}`` / ``]``), string-aware so a comma
    inside a string is never touched. LLMs emit these often; strict ``json`` rejects them."""
    out: list[str] = []
    in_str = esc = False
    n = len(s)
    for i, c in enumerate(s):
        if in_str:
            out.append(c)
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
            out.append(c)
            continue
        if c == ",":
            j = i + 1
            while j < n and s[j] in " \t\r\n":
                j += 1
            if j < n and s[j] in "}]":
                continue  # drop this trailing comma
        out.append(c)
    return "".join(out)


def _lenient_loads(cand: str):
    """``json.loads`` with a trailing-comma-repair retry (the one LLM JSON quirk worth handling)."""
    try:
        return json.loads(cand)
    except json.JSONDecodeError:
        return json.loads(_strip_trailing_commas(cand))


def extract_json_array(text: str) -> list:
    """Pull a JSON ARRAY out of a model's stdout, robustly -- the sibling of :func:`extract_json`
    for prompts whose schema is a top-level ``[...]`` (e.g. the reasoning-event labeler). Prefers a
    ```json fenced block; tracks string literals so brackets inside strings don't truncate;
    tolerates trailing commas; among balanced ``[...]`` candidates returns the LARGEST list parse
    (so an array wrapped in an object ``{"events": [...]}`` is recovered too). Raises
    ``LLMParseError`` on no match."""
    def _best(cands: list[str]) -> tuple[int, list] | None:
        best: tuple[int, list] | None = None
        for cand in cands:
            try:
                obj = _lenient_loads(cand)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, list):
                continue
            if best is None or len(cand) > best[0]:
                best = (len(cand), obj)
        return best

    regions = re.findall(r"```(?:json)?\s*(\[.*)", text, re.DOTALL)
    fenced = [c for region in regions for c in _iter_json_arrays(region)]
    best = _best(fenced) or _best(list(_iter_json_arrays(text)))
    if best is None:
        raise LLMParseError("no parseable JSON array in output")
    return best[1]


# stdout/stderr signatures of a Copilot CLI that ran but returned no usable answer (unavailable
# model / no auth); these exit 0, so callers must detect them by text, not by exit code.
_CLI_ERROR_SIGNS = ("not available", "no authentication", "cannot find github copilot")


def _cli_error(raw: str) -> bool:
    """True if the CLI returned an error banner instead of an answer (bad model / no auth)."""
    low = raw.lower()
    return any(sign in low for sign in _CLI_ERROR_SIGNS)


def run_copilot_cli(
    prompt: str,
    *,
    model: str | None = None,
    timeout: int = 240,
    workdir: str | None = None,
    bin_name: str = "copilot",
) -> str:
    """Call the GitHub Copilot CLI (``copilot -p``) on one prompt; return stdout+stderr.
    Runs from a clean dir so repo AGENTS.md / copilot-instructions do not bleed in, and denies
    the shell/write tools. Handles the Windows case where ``copilot`` is a PowerShell shim.
    Shared by the generator, the response collector, and the judge (all one-shot CLI calls)."""
    workdir = workdir or os.path.join(tempfile.gettempdir(), "schema_llm_gen")
    os.makedirs(workdir, exist_ok=True)
    exe = os.environ.get("COPILOT_BIN") or shutil.which(bin_name)
    tmp_file: str | None = None
    try:
        # On Windows ``copilot`` is a .bat/.ps1 shim that mangles a multi-line argv,
        # so always route through PowerShell + Get-Content there. Only a real binary
        # (Linux/macOS) gets the prompt as a direct argument.
        if os.name != "nt" and exe and not exe.lower().endswith(".ps1"):
            cmd = [exe, "-p", prompt, "--deny-tool", "shell", "--deny-tool", "write"]
            if model:
                cmd += ["--model", model]
        else:
            cmd, tmp_file = _powershell_command(prompt, workdir, model)
        proc = subprocess.run(
            cmd,
            cwd=workdir,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        # don't crash a long batch run on one slow call: return a benign non-JSON banner so the
        # caller treats it as a failed generation (parse fails -> single-item regen / retry).
        return f"[copilot CLI timed out after {timeout}s]"
    finally:
        if tmp_file and os.path.exists(tmp_file):
            os.unlink(tmp_file)
    if proc.returncode != 0 and not proc.stdout.strip():
        raise RuntimeError(f"copilot CLI exited {proc.returncode}: {proc.stderr[:500]}")
    # the model answer is on stdout; a credits/tokens footer (no braces) may follow.
    return f"{proc.stdout}\n{proc.stderr}"


TRAPI_ENDPOINT = "https://trapi.research.microsoft.com/redmond/interactive/openai/v1"
TRAPI_SCOPE = "api://trapi/.default"
_TRAPI_CLIENT: dict[str, object] = {}


def _trapi_client():
    """One process-wide Azure OpenAI client with an auto-refreshing Entra bearer token (``az
    login`` / managed identity). ``openai`` + ``azure-identity`` are imported lazily (the ``api``
    extra) so the Copilot-CLI path never needs them."""
    if "c" not in _TRAPI_CLIENT:
        from azure.identity import (
            AzureCliCredential,
            ChainedTokenCredential,
            ManagedIdentityCredential,
            get_bearer_token_provider,
        )
        from openai import OpenAI

        endpoint = os.environ.get("OPENAI_BASE_URL", TRAPI_ENDPOINT)
        provider = get_bearer_token_provider(
            ChainedTokenCredential(AzureCliCredential(), ManagedIdentityCredential()), TRAPI_SCOPE
        )
        _TRAPI_CLIENT["c"] = OpenAI(base_url=endpoint, api_key=provider, max_retries=0)
    return _TRAPI_CLIENT["c"]


def run_trapi(
    prompt: str,
    *,
    model: str,
    timeout: int = 240,
    max_retries: int = 6,
    max_completion_tokens: int = 16384,
) -> str:
    """Answer one prompt with a TRAPI (Azure OpenAI) deployment; return the reply text.

    Same contract as :func:`run_copilot_cli` -- one prompt in, raw text out, and a sustained
    failure returns a benign non-JSON banner instead of raising, so a long batch treats it as one
    failed generation rather than dying. A direct chat-completion, so it avoids the whole Copilot
    CLI agent session (minutes -> seconds) for bulk labelling work.

    ``model`` is the TRAPI deployment name (e.g. ``gpt-5.6-sol_2026-07-09``). Reasoning models
    spend hidden reasoning tokens from ``max_completion_tokens``, so keep it generous or the
    visible answer comes back empty.
    """
    import random
    import time

    import openai

    retryable = (
        openai.RateLimitError,
        openai.APITimeoutError,
        openai.APIConnectionError,
        openai.InternalServerError,
    )
    client = _trapi_client()
    for attempt in range(max_retries + 1):
        try:
            return (
                client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=max_completion_tokens,
                    timeout=timeout,
                )
                .choices[0]
                .message.content
                or ""
            ).strip()
        except retryable as exc:
            if attempt == max_retries:
                return f"[trapi {type(exc).__name__} after {max_retries} retries]"
            time.sleep(min(60.0, 1.5 * 2**attempt) + random.uniform(0.0, 1.0))
    return "[trapi exhausted retries]"


def _powershell_command(prompt: str, workdir: str, model: str | None) -> tuple[list[str], str]:
    """On Windows ``copilot`` is a .ps1 that mis-parses a multi-line argv. Pass the prompt
    via a file read with ``Get-Content -Raw`` inside a ``pwsh -Command`` (needs PowerShell 7;
    Windows PowerShell 5.1 splits the argument)."""
    psh = shutil.which("pwsh") or shutil.which("powershell")
    if psh is None:
        raise RuntimeError("the Copilot CLI needs PowerShell (pwsh) here")
    fd, tmp_file = tempfile.mkstemp(suffix=".txt", dir=workdir)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(prompt)
    inner = (
        "& { param([string]$PromptPath, [string]$Model) "
        "$CopilotArgs = @('-p', (Get-Content -Raw -LiteralPath $PromptPath), "
        "'--deny-tool', 'shell', '--deny-tool', 'write'); "
        "if ($Model) { $CopilotArgs += @('--model', $Model) }; "
        "& copilot @CopilotArgs }"
    )
    return [psh, "-NoProfile", "-Command", inner, tmp_file, model or ""], tmp_file


def strip_reasoning(text: str) -> str:
    """Return only the final answer of a reasoning subject: the text after the last ``</think>``.
    Phi-4-reasoning emits ``<think> ... </think> {Solution}``; the judge should grade the Solution,
    not the chain of thought. Non-reasoning responses (no ``</think>``) pass through unchanged."""
    marker = "</think>"
    idx = text.rfind(marker)
    return text[idx + len(marker):].strip() if idx != -1 else text


# The Copilot CLI appends a stats footer to stdout (blank lines then a contiguous block of
# `Changes` / `AI Credits` / `Tokens ↑…` / `Resume copilot --resume=…` lines). JSON callers ignore
# it (the extractor keys on braces/brackets), but the VERBATIM segmentation stage must remove it so
# it does not contaminate the trace / depress the fidelity check.
_FOOTER_STRONG = re.compile(r"^(AI Credits\b|Tokens\s+[↑↓]|Resume\s+copilot\b|Changes\s+[+-]\d)")
_FOOTER_LINE = re.compile(
    r"^(Changes|AI Credits|Tokens|Resume|Total|Elapsed|Duration|Model|Session|Cost)\b|--resume=")


def strip_cli_footer(raw: str) -> str:
    """Drop the trailing Copilot-CLI stats footer from an otherwise-verbatim response. Removes the
    contiguous trailing block of blank + stat lines ONLY when it contains a strong CLI marker
    (``AI Credits`` / ``Tokens`` / ``Resume copilot`` / ``Changes +N``), so ordinary trace text is
    never truncated. Idempotent; footer-free input is returned unchanged (bar trailing blanks)."""
    lines = raw.splitlines()
    i = len(lines)
    while i > 0 and (not lines[i - 1].strip() or _FOOTER_LINE.match(lines[i - 1].strip())):
        i -= 1
    if any(_FOOTER_STRONG.match(ln.strip()) for ln in lines[i:]):
        lines = lines[:i]
    return "\n".join(lines).rstrip()
