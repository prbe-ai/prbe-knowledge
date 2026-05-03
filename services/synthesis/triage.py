"""Triage stage — Haiku call + token-budget batching.

Inputs are full document bodies, never chunks: triage decides whether a doc
is wiki-worthy, and that judgment requires reading the whole document, not
retrieval-chunked windows. See plan §3.

Public surface:
- `pack_into_batches(events, budget)` — pure function, used by the cron and
  by tests.
- `call_triage(client, events, *, now)` — fires one Haiku tool-use call,
  validates the response against `TriageOutput`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from anthropic import AsyncAnthropic

from services.synthesis.models import TriageInput, TriageOutput
from services.synthesis.prompts import build_triage_prompt, triage_tool_name
from shared.constants import HAIKU_MODEL, WIKI_TRIAGE_TOKEN_BUDGET
from shared.logging import get_logger

log = get_logger(__name__)


class TriageParseError(RuntimeError):
    """Haiku returned a tool_use block we couldn't parse into TriageOutput."""


def pack_into_batches(
    events: list[TriageInput],
    *,
    budget: int = WIKI_TRIAGE_TOKEN_BUDGET,
) -> list[list[TriageInput]]:
    """Greedy bin-pack by `body_token_count`.

    Tiny events accumulate into big batches; a single oversized event becomes
    its own single-row batch (Haiku will still handle it up to its 200K
    context). Order is preserved per FIFO.
    """
    batches: list[list[TriageInput]] = []
    current: list[TriageInput] = []
    current_tokens = 0
    for event in events:
        # Defensive lower bound: very short docs still cost some tokens
        # for framing. Charge at least 50 per event.
        cost = max(event.body_token_count, 50)
        if current and current_tokens + cost > budget:
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(event)
        current_tokens += cost
    if current:
        batches.append(current)
    return batches


async def call_triage(
    client: AsyncAnthropic,
    events: list[TriageInput],
    *,
    now: datetime,
) -> TriageOutput:
    """Fire one Haiku call for one batch and return the validated output.

    Raises `TriageParseError` if the model didn't return the expected
    tool_use block or the input dict failed Pydantic validation.
    """
    if not events:
        return TriageOutput(verdicts={})

    kwargs = build_triage_prompt(events, now=now)
    resp = await client.messages.create(model=HAIKU_MODEL, **kwargs)
    payload = _extract_tool_input(resp.content, expected_name=triage_tool_name())
    try:
        return TriageOutput(**payload)
    except Exception as exc:
        raise TriageParseError(f"triage tool input failed validation: {exc}") from exc


def _extract_tool_input(blocks: list[Any], *, expected_name: str) -> dict[str, Any]:
    for block in blocks:
        if getattr(block, "type", "") == "tool_use" and getattr(block, "name", "") == expected_name:
            payload = getattr(block, "input", None)
            if isinstance(payload, dict):
                return payload
            raise TriageParseError(f"tool_use input was not a dict: {type(payload).__name__}")
    raise TriageParseError(f"haiku response had no {expected_name} tool_use block")
