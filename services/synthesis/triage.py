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
- `call_triage_with_split_retry(client, events, *, now)` — defense in
  depth on top of the upfront packer. If the wire request still trips
  Anthropic's 200K hard limit (because of tokenizer drift, prompt
  changes, or an unusually large doc that snuck past the packer), this
  wrapper recursively splits the batch in half and retries instead of
  failing the whole batch. A single-event batch that overflows is
  marked rejected with reason
  `OVERSIZED_AT_CALL_TIME_REASON` so the worker can DLQ-route it.
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

import re
from datetime import datetime

from anthropic import AsyncAnthropic, BadRequestError

from services.synthesis.models import TriageInput, TriageOutput, TriageVerdict
from services.synthesis.providers import (
    TriageParseError,
    get_triage_provider,
)
from shared.constants import (
    WIKI_TRIAGE_MAX_EVENTS_PER_BATCH,
    WIKI_TRIAGE_TOKEN_BUDGET,
)
from shared.logging import get_logger

__all__ = [
    "ANTHROPIC_TOKEN_MULTIPLIER",
    "EVENT_FRAMING_TOKENS",
    "OVERSIZED_AT_CALL_TIME_REASON",
    "OVERSIZED_EVENT_TOKENS",
    "PROMPT_OVERHEAD_TOKENS",
    "TriageParseError",
    "call_triage",
    "call_triage_with_split_retry",
    "estimate_event_cost",
    "is_anthropic_oversize_error",
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
# can't be triaged at all — even alone in a batch the event + prompt
# overhead + tokenizer drift would land near or over Haiku's 200K
# context. Drop it (caller DLQ's) rather than letting it poison the
# pipeline.
#
# Lowered from 150_000 to 100_000 after probe-founders' first run
# exposed the failure mode: a single event of ~150K Anthropic tokens
# on its own pushed a 1-event batch to 214K wire tokens because the
# 1.30x ANTHROPIC_TOKEN_MULTIPLIER undershot. 100K leaves real room
# for prompt overhead + tokenizer drift even when the offender is
# alone in its batch.
OVERSIZED_EVENT_TOKENS = 100_000

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
    max_events: int = WIKI_TRIAGE_MAX_EVENTS_PER_BATCH,
) -> tuple[list[list[TriageInput]], list[TriageInput]]:
    """Greedy bin-pack by estimated Anthropic token cost AND event count.

    Returns `(batches, oversized)`:

    - `batches` is the list of batches, each guaranteed to fit BOTH:
        (a) `budget - PROMPT_OVERHEAD_TOKENS` Anthropic tokens of user
            content (input ceiling), AND
        (b) at most `max_events` rows (output ceiling — see below).
    - `oversized` is the list of events whose own body alone exceeds
      `OVERSIZED_EVENT_TOKENS`. These cannot be triaged regardless of
      batching; the worker should DLQ them with a logged reason rather
      than letting them blow up a real batch.

    Why a max-events cap (not just a token budget):

      Haiku's `max_tokens` ceiling is 8192 (Anthropic's wire limit for
      its 4.5 family). The triage tool returns one TriageVerdict per
      event packed into the request — so a batch of N events produces
      ~N x WIKI_TRIAGE_VERDICT_TOKENS (~150) of structured output.
      Without an event cap, a batch of 100 events would generate ~15K
      output tokens, hit max_tokens before finishing the tool_use,
      return a partially-built `{}` payload, and the Pydantic parser
      would crash on the missing `verdicts` field — DLQ-ing the entire
      batch. Production drains saw exactly this on May 5: ~50% of
      batches DLQ'd because Haiku stopped at max_tokens, NOT because
      the input was too long.

    Order is preserved per FIFO within `batches`. Tiny events accumulate
    until either the input budget OR the event cap binds; medium-but-
    large events get their own single-row batch.
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
        # Close current batch if EITHER constraint would be violated by
        # appending this event: input-token budget OR output-side event cap.
        would_exceed_tokens = current_tokens + cost > available
        would_exceed_events = len(current) >= max_events
        if current and (would_exceed_tokens or would_exceed_events):
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


# ---------------------------------------------------------------------------
# Split-on-overflow retry — defense in depth over the upfront packer.
# ---------------------------------------------------------------------------


# Reason tag for events that the packer thought were under-budget but
# Anthropic still rejected at the wire as too long. Surfaced on the
# rejected verdict so the dashboard / DLQ surface can distinguish them
# from regular score-rejected rows.
OVERSIZED_AT_CALL_TIME_REASON = "triage.oversized_event_at_call_time"


# Anthropic's BadRequestError message for an oversized prompt looks like:
#   "prompt is too long: 211739 tokens > 200000 maximum"
# Match either the literal "prompt is too long" prefix OR the more
# generic "<N> tokens > <M> maximum" suffix; the SDK has occasionally
# shifted wording across versions.
_OVERSIZE_REGEXES = (
    re.compile(r"prompt is too long", re.IGNORECASE),
    re.compile(r"\btokens?\s*>\s*\d+\s*maximum\b", re.IGNORECASE),
)


def _bad_request_message(exc: BadRequestError) -> str:
    """Pull the human-readable message out of a BadRequestError.

    Tries `.message` (set by APIStatusError.__init__) first, then falls
    back to `.body['error']['message']` for SDK versions that didn't
    populate `.message`, then `str(exc)`.
    """
    msg = getattr(exc, "message", None)
    if isinstance(msg, str) and msg:
        return msg
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            inner = err.get("message")
            if isinstance(inner, str) and inner:
                return inner
    return str(exc)


def is_anthropic_oversize_error(exc: BaseException) -> bool:
    """True iff `exc` is the Anthropic 400 for an oversized prompt.

    Other 400s (bad API key, bad model name, malformed schema) MUST
    return False so the caller propagates them instead of split-retrying.
    """
    if not isinstance(exc, BadRequestError):
        return False
    msg = _bad_request_message(exc)
    return any(rx.search(msg) for rx in _OVERSIZE_REGEXES)


def _rejected_output_for_single_oversize(event: TriageInput) -> TriageOutput:
    """Synthesize a rejected verdict for a single event that the wire
    rejected as too long. The worker's `_apply_verdicts` will route this
    into `mark_rejected` via the score-below-threshold path, with the
    reason string preserved for the audit log.
    """
    verdict = TriageVerdict(
        important=False,
        score=0.0,
        reason=OVERSIZED_AT_CALL_TIME_REASON,
    )
    return TriageOutput(verdicts={str(event.queue_id): verdict})


def _merge_outputs(a: TriageOutput, b: TriageOutput) -> TriageOutput:
    merged = dict(a.verdicts)
    merged.update(b.verdicts)
    return TriageOutput(verdicts=merged)


async def call_triage_with_split_retry(
    client: AsyncAnthropic,
    events: list[TriageInput],
    *,
    now: datetime,
) -> TriageOutput:
    """Call triage; on overflow-shaped failure, recursively split + retry.

    Two failure modes both signal "the batch was too big":

      A) Input-side overflow — Anthropic 400 with "prompt is too long".
         Caught via `BadRequestError` matching the oversize signature.

      B) Output-side overflow — Haiku stops at `max_tokens` before
         finishing the tool_use. The SDK returns a successful 200 with
         either no tool_use block or a partial `{}` payload, and the
         downstream parser raises `TriageParseError`. We use this as a
         signal that the batch produced more verdicts than fit in the
         output budget — same shape as input overflow, same recovery.

    The upfront packer caps both axes (input tokens via
    `WIKI_TRIAGE_TOKEN_BUDGET`, output verdicts via
    `WIKI_TRIAGE_MAX_EVENTS_PER_BATCH`). This wrapper is the second
    line of defense for tokenizer drift, prompt changes, or a freakish
    doc that slipped through the upfront packer.

    Algorithm:

    - Try `call_triage(client, events, now=now)`.
    - If it raises an overflow-shaped error:
      - `len(events) > 1`: split the batch into two halves, recurse on
        each half, merge the resulting dicts, return.
      - `len(events) == 1`: the single event is the offender — return
        a rejected verdict tagged with `OVERSIZED_AT_CALL_TIME_REASON`
        so the worker DLQ-routes it. Do NOT recurse further.
    - On any other exception (non-oversize 400, 401, 5xx, network,
      a TriageParseError from a malformed-but-not-truncated response):
      re-raise unchanged.

    The merged return shape is identical to a single `call_triage`
    success — callers don't see the recursion.
    """
    try:
        return await call_triage(client, events, now=now)
    except (BadRequestError, TriageParseError) as exc:
        if not _is_overflow_shaped(exc):
            raise
        if len(events) <= 1:
            # Single event is itself oversized; mark rejected so the
            # worker's apply-verdicts path durably moves it out of the
            # pending pool. No further recursion.
            log.warning(
                "triage.split_retry.single_event_oversize",
                queue_id=events[0].queue_id if events else None,
                doc_id=events[0].doc_id if events else None,
                cause=type(exc).__name__,
                error=_overflow_error_message(exc),
            )
            if not events:
                # Defensive: shouldn't happen — call_triage short-circuits
                # on empty input — but if it does, surface it.
                raise
            return _rejected_output_for_single_oversize(events[0])
        midpoint = len(events) // 2
        left = events[:midpoint]
        right = events[midpoint:]
        log.warning(
            "triage.split_retry.splitting",
            batch_size=len(events),
            left_size=len(left),
            right_size=len(right),
            cause=type(exc).__name__,
            error=_overflow_error_message(exc),
        )
        left_out = await call_triage_with_split_retry(client, left, now=now)
        right_out = await call_triage_with_split_retry(client, right, now=now)
        return _merge_outputs(left_out, right_out)


# Output-side overflow signature on TriageParseError: the parser's
# error message contains either "no <name> tool_use block" (Anthropic
# returned no tool_use blocks because Haiku stopped at max_tokens) OR
# Pydantic's "Field required" (Anthropic returned a tool_use block but
# its input is `{}` because Haiku stopped mid-output). Both indicate
# the model ran out of output budget for the batch size.
_PARSE_OVERFLOW_REGEXES = (
    re.compile(r"no\s+\w+\s+tool_use\s+block", re.IGNORECASE),
    re.compile(r"Field required", re.IGNORECASE),
    re.compile(r"max_tokens", re.IGNORECASE),
)


def _is_parse_overflow_error(exc: TriageParseError) -> bool:
    """True iff a TriageParseError indicates an output-truncated response.

    Other parse errors (e.g. wrong tool name, malformed schema) MUST
    return False so the caller propagates them instead of split-retrying.
    """
    msg = str(exc)
    return any(rx.search(msg) for rx in _PARSE_OVERFLOW_REGEXES)


def _is_overflow_shaped(exc: BaseException) -> bool:
    """Combined predicate: input-side BadRequestError OR output-side parse-fail."""
    if isinstance(exc, BadRequestError):
        return is_anthropic_oversize_error(exc)
    if isinstance(exc, TriageParseError):
        return _is_parse_overflow_error(exc)
    return False


def _overflow_error_message(exc: BaseException) -> str:
    """Best-effort human-readable message for either overflow shape."""
    if isinstance(exc, BadRequestError):
        return _bad_request_message(exc)
    return str(exc)
