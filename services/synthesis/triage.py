"""Triage stage — token-budget batching + provider-dispatched call.

Inputs are full document bodies, never chunks: triage decides whether a doc
is wiki-worthy, and that judgment requires reading the whole document, not
retrieval-chunked windows.

Public surface:
- `pack_into_batches(events, budget)` — pure function, used by the worker
  and by tests.
- `call_triage(client, events, *, now)` — fires one batch via the
  configured provider (Anthropic Haiku by default, Gemini Flash Lite if
  `WIKI_TRIAGE_MODEL` env var flips it). The function signature stays
  Anthropic-shaped (takes `client`) for call-site compatibility — when
  the provider is Gemini, `client` is unused.
- `TriageParseError` — re-exported from `providers` so existing call sites
  importing it from this module still work.
"""

from __future__ import annotations

from datetime import datetime

from anthropic import AsyncAnthropic

from services.synthesis.models import TriageInput, TriageOutput
from services.synthesis.providers import (
    TriageParseError,
    get_triage_provider,
)
from shared.constants import WIKI_TRIAGE_TOKEN_BUDGET
from shared.logging import get_logger

__all__ = [
    "TriageParseError",
    "call_triage",
    "pack_into_batches",
]

log = get_logger(__name__)


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
    """Fire one triage call for one batch and return the validated output.

    Dispatches to the configured provider (Anthropic Haiku or Gemini
    Flash Lite). The Anthropic `client` is only used when the configured
    provider is Anthropic; passed through for caller compatibility.

    Raises `TriageParseError` if the model didn't return the expected
    structured output.
    """
    provider = get_triage_provider(client)
    return await provider.triage(events, now=now)
