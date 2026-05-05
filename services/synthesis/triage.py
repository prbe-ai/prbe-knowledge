"""Triage stage — token-budget batching + provider-dispatched call.

Inputs are full document bodies, never chunks: triage decides whether a doc
is wiki-worthy, and that judgment requires reading the whole document, not
retrieval-chunked windows.

Public surface:
- `pack_into_batches(events, budget)` — pure function, used by the worker
  and by tests. Returns `(batches, oversized)` where `oversized` is the
  list of events whose own body exceeds `OVERSIZED_EVENT_TOKENS` (these
  cannot be triaged in one call regardless of batching).
- `call_triage(client, events, *, now)` — fires one batch via the
  configured provider (Anthropic Haiku by default, Gemini Flash Lite if
  `WIKI_TRIAGE_MODEL` env var flips it). The function signature stays
  Anthropic-shaped (takes `client`) for call-site compatibility — when
  the provider is Gemini, `client` is unused.
- `TriageParseError` — re-exported from `providers` so existing call sites
  importing it from this module still work.

Token accounting (why this is non-trivial):

`documents.body_token_count` is computed at enqueue time using
`tiktoken.cl100k_base` (the OpenAI tokenizer used elsewhere for chunking).
Anthropic's tokenizer disagrees with cl100k_base — Anthropic typically
counts ~10-30% MORE tokens for the same text, especially on code and
markup. The packer must account for this drift OR batches that look like
they fit in 200K end up being rejected at the wire as 208K-211K. Plus the
wire request includes a system prompt, a tool-input JSON Schema, and
per-event framing in the user message that aren't part of any per-event
body count. We:

1. Multiply each event's `body_token_count` by `ANTHROPIC_TOKEN_MULTIPLIER`
   to get a conservative Anthropic-token estimate.
2. Add `EVENT_FRAMING_TOKENS` per event for the user-message wrapper.
3. Subtract `PROMPT_OVERHEAD_TOKENS` from the budget for the system
   prompt + tool-schema + envelope.
4. Drop any event whose own estimated cost exceeds
   `OVERSIZED_EVENT_TOKENS` — these can never fit even alone, so the
   caller DLQ's them rather than letting them poison a batch.
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
    "ANTHROPIC_TOKEN_MULTIPLIER",
    "EVENT_FRAMING_TOKENS",
    "OVERSIZED_EVENT_TOKENS",
    "PROMPT_OVERHEAD_TOKENS",
    "TriageParseError",
    "call_triage",
    "estimate_event_cost",
    "pack_into_batches",
]

log = get_logger(__name__)


# Conservative scaling factor from cl100k_base counts to Anthropic-tokenizer
# counts. Empirically Anthropic counts ~10-25% more for English/code; we
# round up to 1.30 so packed batches don't peek over the model's hard
# 200K context limit. Don't lower this without measuring — production was
# DLQ'ing entire customers at the implicit 1.0x multiplier.
ANTHROPIC_TOKEN_MULTIPLIER = 1.30

# Per-event overhead added by `_format_triage_user_message` — the
# `queue_id:`, `doc_id:`, `<body>` framing plus title/author lines.
# Measured at ~50 cl100k tokens; bumped to 80 so realistically-long titles
# don't slip through underestimated.
EVENT_FRAMING_TOKENS = 80

# Fixed overhead per request: system prompt (rubric + threshold copy),
# tool input_schema, message envelope, plus the model's own output.
# Measured at ~700 cl100k tokens for system+tool; we budget 2_000 to
# absorb tokenizer drift and the response slice.
PROMPT_OVERHEAD_TOKENS = 2_000

# A single event whose estimated Anthropic cost exceeds this threshold
# can't be triaged at all — even alone in a batch it would exceed Haiku's
# 200K context. Drop it (caller DLQ's) rather than letting it poison the
# pipeline.
OVERSIZED_EVENT_TOKENS = 150_000

# Floor charge per event. Even a one-line body costs framing + body tokens
# > zero; cap the lower bound so 1000 zero-byte events don't pretend to
# fit in one call.
_MIN_EVENT_COST = 50


def estimate_event_cost(event: TriageInput) -> int:
    """Conservative Anthropic-token estimate for one event in the user message.

    Combines the cl100k body count (scaled to Anthropic) with the
    per-event framing overhead. Floored at `_MIN_EVENT_COST` so very
    short events still consume real budget.
    """
    body_anthropic = int(event.body_token_count * ANTHROPIC_TOKEN_MULTIPLIER)
    cost = body_anthropic + EVENT_FRAMING_TOKENS
    return max(cost, _MIN_EVENT_COST)


def pack_into_batches(
    events: list[TriageInput],
    *,
    budget: int = WIKI_TRIAGE_TOKEN_BUDGET,
) -> tuple[list[list[TriageInput]], list[TriageInput]]:
    """Greedy bin-pack by estimated Anthropic token cost.

    Returns `(batches, oversized)`:

    - `batches` is the list of batches, each guaranteed to fit in
      `budget - PROMPT_OVERHEAD_TOKENS` Anthropic tokens of user content
      (so the full wire request stays under `budget` plus envelope).
    - `oversized` is the list of events whose own body alone exceeds
      `OVERSIZED_EVENT_TOKENS`. These cannot be triaged regardless of
      batching; the worker should DLQ them with a logged reason rather
      than letting them blow up a real batch.

    Order is preserved per FIFO within `batches`. Tiny events accumulate
    into big batches; medium-but-large events get their own single-row
    batch (Haiku still handles them up to its 200K context).
    """
    available = max(budget - PROMPT_OVERHEAD_TOKENS, 1)
    batches: list[list[TriageInput]] = []
    oversized: list[TriageInput] = []
    current: list[TriageInput] = []
    current_tokens = 0
    for event in events:
        cost = estimate_event_cost(event)
        if cost > OVERSIZED_EVENT_TOKENS:
            log.warning(
                "triage.pack.oversized_event_dropped",
                queue_id=event.queue_id,
                doc_id=event.doc_id,
                doc_type=event.doc_type,
                body_token_count=event.body_token_count,
                estimated_anthropic_tokens=cost,
                threshold=OVERSIZED_EVENT_TOKENS,
            )
            oversized.append(event)
            continue
        if current and current_tokens + cost > available:
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(event)
        current_tokens += cost
    if current:
        batches.append(current)
    return batches, oversized


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
