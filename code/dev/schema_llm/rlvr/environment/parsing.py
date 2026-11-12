"""Final-response parsing shared by training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass


def split_reasoning(response: str) -> tuple[str, str, bool]:
    """Return reasoning, final answer, and whether an opened think block closed."""
    if "<think>" not in response:
        return "", response.strip(), True
    start = response.find("<think>") + len("<think>")
    end = response.rfind("</think>")
    if end < start:
        return response[start:].strip(), "", False
    return response[start:end].strip(), response[end + len("</think>") :].strip(), True


@dataclass(frozen=True, slots=True)
class ParsedResponse:
    reasoning: str
    final_answer: str
    thinking_closed: bool


def parse_response(response: str) -> ParsedResponse:
    reasoning, final_answer, thinking_closed = split_reasoning(response)
    return ParsedResponse(reasoning, final_answer, thinking_closed)
