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
    batches = pack_into_batches(events, budget=120_000)
    assert len(batches) == 1
    assert [e.queue_id for e in batches[0]] == list(range(10))


def test_pack_splits_when_budget_exceeded() -> None:
    # 3 events at 50K tokens each = 150K. Budget 120K -> [50K+50K], [50K].
    events = [_ev(i, 50_000) for i in range(3)]
    batches = pack_into_batches(events, budget=120_000)
    assert len(batches) == 2
    assert [e.queue_id for e in batches[0]] == [0, 1]
    assert [e.queue_id for e in batches[1]] == [2]


def test_pack_oversized_event_gets_own_batch() -> None:
    events = [_ev(0, 10), _ev(1, 200_000), _ev(2, 10)]
    batches = pack_into_batches(events, budget=120_000)
    assert len(batches) == 3
    assert batches[0][0].queue_id == 0
    assert batches[1][0].queue_id == 1
    assert batches[2][0].queue_id == 2


def test_pack_minimum_50_token_charge() -> None:
    # 100 zero-cost events shouldn't all collapse into one batch with budget 100;
    # the 50-token-per-event floor forces splits.
    events = [_ev(i, 0) for i in range(100)]
    batches = pack_into_batches(events, budget=100)
    assert len(batches) >= 50  # at least 2 events per batch (100 budget / 50 floor)


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
