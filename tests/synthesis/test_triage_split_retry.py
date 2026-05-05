"""Unit tests for `call_triage_with_split_retry`.

Defense-in-depth on top of PR #100's tightened triage packer. Even with
the Anthropic tokenizer multiplier and per-event framing accounted for,
tokenizer drift, a prompt-template change, or an unusually dense doc
could still push a wire request past Anthropic's 200K hard limit. When
that happens we want to halve-and-retry instead of DLQ'ing the whole
batch.

These tests pin the wrapper behavior:

  1. A single oversize 400 on the full batch → split into halves, both
     halves succeed → all verdicts returned, exactly 3 Anthropic calls.
  2. Repeated overflows recursing all the way down to single events →
     every event's verdict still surfaces (some as success, some as
     rejected).
  3. A single-event batch that overflows is reported as
     `triage.oversized_event_at_call_time` rejected, no further recursion.
  4. A non-oversize 400 (e.g. authentication) propagates unchanged.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from anthropic import BadRequestError

from services.synthesis.models import TriageInput, TriageOutput
from services.synthesis.triage import (
    OVERSIZED_AT_CALL_TIME_REASON,
    call_triage_with_split_retry,
    is_anthropic_oversize_error,
)

NOW = datetime(2026, 5, 5, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _ev(qid: int) -> TriageInput:
    return TriageInput(
        queue_id=qid,
        doc_id=f"doc:{qid}",
        doc_type="github.commit",
        source_system="github",
        title=f"Doc {qid}",
        author_id="alice",
        body="x",
        body_token_count=100,
    )


def _bad_request(message: str) -> BadRequestError:
    """Build a BadRequestError that mirrors what the Anthropic SDK
    raises on a real 400 from the wire."""
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return BadRequestError(
        message,
        response=httpx.Response(400, request=req),
        body={
            "type": "error",
            "error": {"type": "invalid_request_error", "message": message},
        },
    )


def _success_response(events: list[TriageInput]) -> SimpleNamespace:
    """Build an Anthropic tool_use response that scores every event in
    `events`. The shape matches what `_extract_tool_use_input` consumes
    in `services/synthesis/providers.py`."""
    payload = {
        "verdicts": {
            str(ev.queue_id): {
                "important": True,
                "score": 7.0,
                "reason": "ok",
            }
            for ev in events
        }
    }
    block = SimpleNamespace(type="tool_use", name="record_triage", input=payload)
    return SimpleNamespace(content=[block])


def _make_client_with_responses(responses: list[object]) -> object:
    """Build a fake AsyncAnthropic whose `messages.create` returns (or
    raises) `responses` in order. Each entry is either a response object
    (returned) or an Exception subclass instance (raised)."""
    iterator = iter(responses)

    async def _create(**kwargs: object) -> object:
        item = next(iterator)
        if isinstance(item, BaseException):
            raise item
        return item

    client = SimpleNamespace()
    client.messages = SimpleNamespace(create=AsyncMock(side_effect=_create))
    return client


# ---------------------------------------------------------------------------
# is_anthropic_oversize_error — detector unit tests
# ---------------------------------------------------------------------------


def test_detector_matches_prompt_too_long() -> None:
    err = _bad_request("prompt is too long: 211739 tokens > 200000 maximum")
    assert is_anthropic_oversize_error(err) is True


def test_detector_matches_tokens_maximum_phrasing() -> None:
    err = _bad_request("input has 250000 tokens > 200000 maximum context window")
    assert is_anthropic_oversize_error(err) is True


def test_detector_skips_unrelated_400() -> None:
    err = _bad_request("invalid api key")
    assert is_anthropic_oversize_error(err) is False


def test_detector_skips_non_bad_request() -> None:
    assert is_anthropic_oversize_error(RuntimeError("boom")) is False


# ---------------------------------------------------------------------------
# Split-on-overflow behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_batch_splits_into_two_successful_halves() -> None:
    """4 events: full batch overflows once, both halves succeed."""
    events = [_ev(i) for i in range(4)]
    overflow = _bad_request("prompt is too long: 211739 tokens > 200000 maximum")
    # Order: full batch raises, then left half (events 0,1) succeeds,
    # then right half (events 2,3) succeeds.
    client = _make_client_with_responses(
        [
            overflow,
            _success_response(events[:2]),
            _success_response(events[2:]),
        ]
    )

    out = await call_triage_with_split_retry(client, events, now=NOW)

    assert isinstance(out, TriageOutput)
    assert sorted(int(k) for k in out.verdicts) == [0, 1, 2, 3]
    # 1 failed call + 2 successful calls = 3 wire calls total.
    assert client.messages.create.await_count == 3


@pytest.mark.asyncio
async def test_repeated_overflows_recurse_to_single_events() -> None:
    """8 events: full batch overflows, first half overflows, first
    quarter overflows. Eventually every leaf either succeeds or gets
    surfaced as a rejected single-event verdict.

    Recursion tree (• = call attempt):
        • [0..7]            -> overflow -> split
          • [0..3]          -> overflow -> split
            • [0,1]         -> overflow -> split
              • [0]         -> overflow -> rejected (no further recursion)
              • [1]         -> success
            • [2,3]         -> success
          • [4..7]          -> success
    """
    events = [_ev(i) for i in range(8)]
    overflow = _bad_request("prompt is too long: 250000 tokens > 200000 maximum")

    # Each entry corresponds to one call to `messages.create` in the
    # order the wrapper makes them. Pre-order DFS: full -> left -> ...
    client = _make_client_with_responses(
        [
            overflow,                     # [0..7]
            overflow,                     # [0..3]
            overflow,                     # [0,1]
            overflow,                     # [0] alone -> rejected, no call beyond this
            _success_response([events[1]]),    # [1]
            _success_response(events[2:4]),    # [2,3]
            _success_response(events[4:8]),    # [4..7]
        ]
    )

    out = await call_triage_with_split_retry(client, events, now=NOW)

    # All 8 queue_ids must appear in the merged output.
    assert sorted(int(k) for k in out.verdicts) == list(range(8))
    # qid 0 should be tagged with the call-time-oversize reason.
    assert out.verdicts["0"].important is False
    assert out.verdicts["0"].score == 0.0
    assert out.verdicts["0"].reason == OVERSIZED_AT_CALL_TIME_REASON
    # The other 7 came back as scored success verdicts.
    for qid in range(1, 8):
        assert out.verdicts[str(qid)].important is True


@pytest.mark.asyncio
async def test_single_event_overflow_marked_rejected() -> None:
    """A single-event batch that overflows is returned as a rejected
    verdict tagged with the call-time-oversize reason; no recursion
    happens beyond this leaf."""
    events = [_ev(42)]
    overflow = _bad_request("prompt is too long: 220000 tokens > 200000 maximum")
    client = _make_client_with_responses([overflow])

    out = await call_triage_with_split_retry(client, events, now=NOW)

    assert list(out.verdicts.keys()) == ["42"]
    verdict = out.verdicts["42"]
    assert verdict.important is False
    assert verdict.score == 0.0
    assert verdict.reason == OVERSIZED_AT_CALL_TIME_REASON
    # Exactly one wire call — no retry on single-event leaf.
    assert client.messages.create.await_count == 1


@pytest.mark.asyncio
async def test_non_oversize_400_propagates() -> None:
    """A 400 that is NOT the oversize-prompt variant (e.g. bad api key)
    must propagate unchanged. We must not split-retry on it — that would
    just multiply the failures."""
    events = [_ev(1), _ev(2), _ev(3)]
    other_400 = _bad_request("authentication_error: invalid x-api-key header")
    client = _make_client_with_responses([other_400])

    with pytest.raises(BadRequestError) as exc_info:
        await call_triage_with_split_retry(client, events, now=NOW)
    assert "invalid x-api-key" in str(exc_info.value)
    # No retry: the wrapper saw a non-oversize 400 and re-raised.
    assert client.messages.create.await_count == 1


@pytest.mark.asyncio
async def test_non_anthropic_exception_propagates() -> None:
    """Non-Anthropic exceptions (network, parse error, 5xx) must not
    trigger split-retry either."""
    events = [_ev(1), _ev(2)]
    client = _make_client_with_responses([RuntimeError("kaboom")])

    with pytest.raises(RuntimeError, match="kaboom"):
        await call_triage_with_split_retry(client, events, now=NOW)
    assert client.messages.create.await_count == 1
