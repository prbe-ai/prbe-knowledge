"""Regression tests for `pack_into_batches` token accounting.

Pinned to the 2026-05 production incident where the packer's batches
blew past Anthropic Haiku's 200K context limit on probe-founders. Root
cause: the packer counted only `documents.body_token_count` (cl100k_base
from the chunker), but the wire request to Anthropic also includes:

  - The system prompt (rubric + threshold copy, ~500 tokens)
  - The triage tool's input_schema (~250 tokens)
  - Per-event user-message framing (queue_id/doc_id/<body> tags, ~50/ev)
  - Tokenizer drift — Anthropic counts ~10-30% more tokens than cl100k

A 66-event drain measuring as 84K cl100k tokens shipped as 208K
Anthropic tokens. Anthropic rejected the request; the worker DLQ'd the
whole customer slice.

These tests pin the fix:

  1. Realistic 200-event batches stay under 200K Anthropic tokens with
     headroom on every batch.
  2. A single oversized event (>150K Anthropic tokens) is surfaced via
     the `oversized` return list, never packed into a batch.
"""

from __future__ import annotations

from services.synthesis.models import TriageInput
from services.synthesis.triage import (
    ANTHROPIC_TOKEN_MULTIPLIER,
    EVENT_FRAMING_TOKENS,
    OVERSIZED_EVENT_TOKENS,
    PROMPT_OVERHEAD_TOKENS,
    estimate_event_cost,
    pack_into_batches,
)
from shared.constants import (
    WIKI_TRIAGE_MAX_EVENTS_PER_BATCH,
    WIKI_TRIAGE_MAX_OUTPUT_TOKENS,
    WIKI_TRIAGE_VERDICT_TOKENS,
)

ANTHROPIC_HARD_LIMIT = 200_000


def _ev(qid: int, body_tokens: int) -> TriageInput:
    return TriageInput(
        queue_id=qid,
        doc_id=f"doc:{qid}",
        doc_type="github.commit",
        source_system="github",
        title=f"Doc {qid}",
        author_id="alice",
        body="x" * max(body_tokens * 4, 1),
        body_token_count=body_tokens,
    )


def _wire_token_estimate(batch: list[TriageInput]) -> int:
    """Conservative estimate of the wire-shape token cost of a batch.

    Mirrors what the Anthropic API will actually count: prompt overhead +
    sum of (body * multiplier) + per-event framing.
    """
    body_cost = sum(estimate_event_cost(ev) for ev in batch)
    return PROMPT_OVERHEAD_TOKENS + body_cost


# ---------------------------------------------------------------------------
# Regression: realistic batch shapes must fit under 200K
# ---------------------------------------------------------------------------


def test_pack_200_realistic_events_stays_under_anthropic_limit() -> None:
    """Repro: 200 events with realistic per-event sizes, packed under default budget.

    Mix mirrors a typical wiki drain: short Slack acks (~50 tokens),
    medium Linear comments / GitHub PRs (~500-2K tokens), occasional long
    claude_code transcripts (~10-30K tokens). Every batch must fit under
    Anthropic Haiku's 200K hard limit with margin.
    """
    events: list[TriageInput] = []
    for i in range(120):
        # short messages
        events.append(_ev(i, 80))
    for i in range(120, 180):
        # medium-length authored content
        events.append(_ev(i, 1_500))
    for i in range(180, 200):
        # long transcripts
        events.append(_ev(i, 25_000))

    batches, oversized = pack_into_batches(events)

    # Nothing should drop — 25K cl100k * 1.30 = 32.5K, well under
    # the 150K oversized cap.
    assert oversized == []

    # Every event must be present exactly once in the batches.
    packed_ids = [ev.queue_id for batch in batches for ev in batch]
    assert sorted(packed_ids) == list(range(200))
    assert len(packed_ids) == 200

    # Every batch's wire-shape estimate must fit under Anthropic's limit
    # with headroom.
    for batch in batches:
        wire = _wire_token_estimate(batch)
        assert wire < ANTHROPIC_HARD_LIMIT, (
            f"batch with {len(batch)} events estimated at {wire} tokens "
            f"exceeds Anthropic's {ANTHROPIC_HARD_LIMIT} limit"
        )


def test_pack_high_density_batch_fits_under_limit() -> None:
    """The 2026-05 incident shape: many medium events, packed densely.

    Production saw batch_size=66 producing 208K Anthropic tokens. That
    pattern matches ~2K-token events in a 120K cl100k budget. Re-pack
    that same shape under the new budget and verify the wire estimate
    stays under 200K.
    """
    # 100 events at 2K cl100k each = 200K cl100k. The old budget (120K
    # cl100k) would have packed ~60 of these per batch. Each at 2.6K
    # Anthropic = 156K; plus framing + overhead pushed past 200K.
    events = [_ev(i, 2_000) for i in range(100)]
    batches, oversized = pack_into_batches(events)
    assert oversized == []
    for batch in batches:
        wire = _wire_token_estimate(batch)
        assert wire < ANTHROPIC_HARD_LIMIT, (
            f"high-density batch ({len(batch)} events) estimated at "
            f"{wire} > 200K"
        )


# ---------------------------------------------------------------------------
# Regression: oversized single events are dropped, not packed
# ---------------------------------------------------------------------------


def test_pack_drops_oversized_single_event() -> None:
    """A single event whose body alone exceeds 150K Anthropic tokens
    cannot fit in any batch. It must be returned via `oversized`, not
    packed (otherwise the batch will be > 200K when fired)."""
    # 130K cl100k * 1.30 = 169K Anthropic — over the 150K cap.
    big = _ev(42, 130_000)
    small_a = _ev(1, 100)
    small_b = _ev(2, 100)
    batches, oversized = pack_into_batches([small_a, big, small_b])

    assert len(oversized) == 1
    assert oversized[0].queue_id == 42

    # The two small events still pack normally.
    packed_ids = [ev.queue_id for batch in batches for ev in batch]
    assert sorted(packed_ids) == [1, 2]

    # Sanity: no batch contains the oversized event.
    for batch in batches:
        assert all(ev.queue_id != 42 for ev in batch)


def test_pack_event_at_oversized_boundary_is_kept() -> None:
    """Events right at the OVERSIZED_EVENT_TOKENS boundary are still
    packable (single-row batch). Only events exceeding it are dropped."""
    # cost = body * 1.30 + 80; we want cost just <= 150_000.
    # Solve: body = (150_000 - 80) / 1.30 = ~115_323
    body_tokens = int((OVERSIZED_EVENT_TOKENS - EVENT_FRAMING_TOKENS) / ANTHROPIC_TOKEN_MULTIPLIER)
    near_max = _ev(7, body_tokens)
    assert estimate_event_cost(near_max) <= OVERSIZED_EVENT_TOKENS

    batches, oversized = pack_into_batches([near_max])
    assert oversized == []
    assert len(batches) == 1
    assert batches[0][0].queue_id == 7


# ---------------------------------------------------------------------------
# Tokenizer multiplier is conservative (>= 1.0)
# ---------------------------------------------------------------------------


def test_anthropic_token_multiplier_is_conservative() -> None:
    """Guards against accidentally lowering the multiplier below 1.0,
    which would re-introduce the 2026-05 incident: cl100k underestimates
    Anthropic's tokenizer, so we must scale UP, never down."""
    assert ANTHROPIC_TOKEN_MULTIPLIER >= 1.20, (
        "Anthropic counts more tokens than cl100k_base for the same text; "
        "lowering the multiplier risks 200K-context overflows."
    )


# ---------------------------------------------------------------------------
# Output-side cap: WIKI_TRIAGE_MAX_EVENTS_PER_BATCH
# ---------------------------------------------------------------------------
#
# Even if events are tiny enough to fit dozens-to-hundreds in the input
# token budget, the OUTPUT side bounds us: Haiku 4.5's max_tokens is
# 8192, and each TriageVerdict occupies ~150 Anthropic tokens. The
# packer must cap at floor(WIKI_TRIAGE_MAX_OUTPUT_TOKENS /
# WIKI_TRIAGE_VERDICT_TOKENS) events so the response can finish before
# hitting max_tokens. Without this cap, a 200-event batch of tiny docs
# would overflow output and crash Pydantic on a missing `verdicts`
# field — which is the failure mode the second drain hit.


def test_pack_caps_at_max_events_when_input_budget_underbinds() -> None:
    """200 tiny events would all fit in the 150K input budget, but the
    output side caps each batch at WIKI_TRIAGE_MAX_EVENTS_PER_BATCH.
    The packer must close the batch when the event-count cap binds."""
    # Each event ~150 cl100k tokens (~195 Anthropic), well under any
    # single-event cap. Total cl100k for 200 events = ~30K, way under
    # the 150K input budget — so without the event cap, this packs
    # into ONE batch of 200.
    events = [_ev(i, 150) for i in range(200)]
    batches, oversized = pack_into_batches(events)

    assert oversized == []
    for batch in batches:
        assert len(batch) <= WIKI_TRIAGE_MAX_EVENTS_PER_BATCH, (
            f"batch of {len(batch)} events exceeds output cap "
            f"{WIKI_TRIAGE_MAX_EVENTS_PER_BATCH}"
        )

    # Sanity: 200 events at cap=50 must produce >= 4 batches.
    assert len(batches) >= 4, (
        f"expected >= 4 batches at cap 50; got {len(batches)}"
    )

    # Order preserved: flatten batches and check FIFO.
    flat = [ev.queue_id for batch in batches for ev in batch]
    assert flat == list(range(200))


def test_max_events_cap_keeps_output_under_max_tokens() -> None:
    """Sanity-check the constant arithmetic: a full-cap batch's expected
    output should fit under WIKI_TRIAGE_MAX_OUTPUT_TOKENS with margin.
    If a future tuning lowers max_tokens or raises the per-verdict
    estimate, this catches the inconsistency."""
    expected_output = (
        WIKI_TRIAGE_MAX_EVENTS_PER_BATCH * WIKI_TRIAGE_VERDICT_TOKENS
    )
    assert expected_output <= WIKI_TRIAGE_MAX_OUTPUT_TOKENS, (
        f"WIKI_TRIAGE_MAX_EVENTS_PER_BATCH ({WIKI_TRIAGE_MAX_EVENTS_PER_BATCH}) "
        f"x WIKI_TRIAGE_VERDICT_TOKENS ({WIKI_TRIAGE_VERDICT_TOKENS}) = "
        f"{expected_output} would exceed WIKI_TRIAGE_MAX_OUTPUT_TOKENS "
        f"({WIKI_TRIAGE_MAX_OUTPUT_TOKENS}). Lower the event cap or raise "
        f"max_tokens."
    )


def test_pack_input_budget_binds_first_for_large_events() -> None:
    """When events are large enough that the input budget binds before
    the event cap, batches must still respect the input budget — i.e.
    the cap doesn't artificially force more events into a batch when
    they wouldn't fit."""
    # Each event ~50K cl100k -> ~65K Anthropic. At 150K budget, only ~2
    # events per batch fit by input. Event cap of 50 would never bind.
    events = [_ev(i, 50_000) for i in range(10)]
    batches, oversized = pack_into_batches(events)

    assert oversized == []
    for batch in batches:
        # Input budget is the binder: each batch should be 1-2 events.
        assert len(batch) <= 3, (
            f"batch of {len(batch)} large events should not pack so wide"
        )
