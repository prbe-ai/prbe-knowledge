"""Unit tests for the triage stage.

- pack_into_batches: token-budget bin-packing.
- call_triage: round-trip via mocked AsyncAnthropic, validates against TriageOutput.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from services.synthesis.models import TriageInput, TriageOutput
from services.synthesis.triage import (
    TriageParseError,
    call_triage,
    pack_into_batches,
)


def _ev(qid: int, body_tokens: int, *, body: str = "x") -> TriageInput:
    return TriageInput(
        queue_id=qid,
        doc_id=f"doc:{qid}",
        doc_type="github.commit",
        source_system="github",
        title=f"Doc {qid}",
        author_id="alice",
        body=body,
        body_token_count=body_tokens,
    )


def _tool_use_response(payload: dict) -> SimpleNamespace:
    block = SimpleNamespace(type="tool_use", name="record_triage", input=payload)
    return SimpleNamespace(content=[block])


# ---------------------------------------------------------------------------
# pack_into_batches
# ---------------------------------------------------------------------------


def test_pack_tiny_events_fit_in_one_batch() -> None:
    events = [_ev(i, 100) for i in range(10)]
    batches, oversized = pack_into_batches(events, budget=150_000)
    assert oversized == []
    assert len(batches) == 1
    assert [e.queue_id for e in batches[0]] == list(range(10))


def test_pack_splits_when_budget_exceeded() -> None:
    # Each event: 50_000 cl100k * 1.30 = 65_000 + 80 framing = 65_080
    # estimated Anthropic tokens. Budget 150_000 - 2_000 overhead =
    # 148_000 available -> [65_080 + 65_080 = 130_160], next event
    # would push to 195_240 > 148_000, so it goes in a second batch.
    events = [_ev(i, 50_000) for i in range(3)]
    batches, oversized = pack_into_batches(events, budget=150_000)
    assert oversized == []
    assert len(batches) == 2
    assert [e.queue_id for e in batches[0]] == [0, 1]
    assert [e.queue_id for e in batches[1]] == [2]


def test_pack_oversized_event_is_dropped_not_packed() -> None:
    # 200K cl100k * 1.30 = 260K Anthropic tokens — over the 150K
    # OVERSIZED_EVENT_TOKENS cap. It cannot fit even alone; the packer
    # surfaces it via the oversized list so the worker can DLQ it.
    events = [_ev(0, 10), _ev(1, 200_000), _ev(2, 10)]
    batches, oversized = pack_into_batches(events, budget=150_000)
    assert len(oversized) == 1
    assert oversized[0].queue_id == 1
    # The two surviving small events fit in one batch.
    assert len(batches) == 1
    assert [e.queue_id for e in batches[0]] == [0, 2]


def test_pack_minimum_per_event_charge() -> None:
    # Even zero-token bodies cost framing tokens (EVENT_FRAMING_TOKENS =
    # 80) so 100 zero-byte events cannot collapse into one call when the
    # budget is small.
    events = [_ev(i, 0) for i in range(100)]
    # 100 events at min 80 each = 8_000 Anthropic tokens. With budget
    # 1_000 available after overhead, that's 12+ batches.
    batches, oversized = pack_into_batches(events, budget=3_000)
    assert oversized == []
    # available = 3_000 - 2_000 overhead = 1_000; per-event ~80 -> ~12 per batch.
    assert len(batches) >= 8


# ---------------------------------------------------------------------------
# call_triage
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_call_triage_round_trip() -> None:
    events = [_ev(1, 100), _ev(2, 100)]
    # v4: triage produces score-only verdicts. No targets / wiki_type /
    # slug — the wiki agent decides downstream.
    payload = {
        "verdicts": {
            "1": {
                "important": True,
                "score": 7.5,
                "reason": "Auth incident",
            },
            "2": {
                "important": False,
                "score": 1.0,
                "reason": "Routine ack",
            },
        }
    }
    client = SimpleNamespace()
    client.messages = SimpleNamespace(create=AsyncMock(return_value=_tool_use_response(payload)))
    out = await call_triage(client, events, now=datetime(2026, 5, 2, tzinfo=UTC))
    assert isinstance(out, TriageOutput)
    assert out.verdicts["1"].important is True
    assert out.verdicts["1"].score == pytest.approx(7.5)
    assert out.verdicts["1"].reason == "Auth incident"
    assert out.verdicts["2"].important is False
    # No targets field on TriageVerdict in v4.
    assert not hasattr(out.verdicts["1"], "targets")


@pytest.mark.asyncio
async def test_call_triage_empty_input_short_circuits() -> None:
    client = SimpleNamespace()
    # No call should be made; verify by pre-failing the mock.
    client.messages = SimpleNamespace(create=AsyncMock(side_effect=AssertionError("must not call")))
    out = await call_triage(client, [], now=datetime(2026, 5, 2, tzinfo=UTC))
    assert out.verdicts == {}


@pytest.mark.asyncio
async def test_call_triage_raises_on_missing_tool_block() -> None:
    events = [_ev(1, 10)]
    text_only = SimpleNamespace(content=[SimpleNamespace(type="text", text="hi")])
    client = SimpleNamespace()
    client.messages = SimpleNamespace(create=AsyncMock(return_value=text_only))
    with pytest.raises(TriageParseError):
        await call_triage(client, events, now=datetime(2026, 5, 2, tzinfo=UTC))


@pytest.mark.asyncio
async def test_call_triage_raises_on_validation_failure() -> None:
    events = [_ev(1, 10)]
    bad = {
        "verdicts": {
            "1": {
                "important": True,
                "score": "not-a-number",  # invalid
            }
        }
    }
    client = SimpleNamespace()
    client.messages = SimpleNamespace(create=AsyncMock(return_value=_tool_use_response(bad)))
    with pytest.raises(TriageParseError):
        await call_triage(client, events, now=datetime(2026, 5, 2, tzinfo=UTC))
