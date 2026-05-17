"""Unit tests for middleware._persist_trace_blob_r2.

Exercises the BackgroundTask in isolation — no ASGITransport, no live_db.
The function must:
  * No-op when request.state.search_agent_should_persist is missing/False
  * No-op when customer_id is missing (auth never ran)
  * Set request.state.trace_blob_key on a successful R2 PUT
  * NOT set trace_blob_key when persist_trace_blob_to_r2 returns None
  * Swallow every exception including CancelledError so the BackgroundTask
    chain stays intact even when something explodes inside the helper.

Per feedback_no_real_cli_in_tests.md: no real R2 traffic; we monkeypatch
persist_trace_blob_to_r2 to return canned values.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from services.retrieval.agent.models import (
    DroppedCandidate,
    GathererNotes,
    GathererOutput,
)
from services.retrieval.middleware import _persist_trace_blob_r2


def _mk_request(**state_attrs: Any) -> SimpleNamespace:
    """Build a fake request whose .state carries the named attrs."""
    state = SimpleNamespace(**state_attrs)
    return SimpleNamespace(state=state)


def _mk_minimal_gathered() -> GathererOutput:
    return GathererOutput(
        entities=[],
        chunks=[],
        gatherer_notes=GathererNotes(
            turns_used=1,
            tools_called=[],
            confidence="low",
            dropped=[DroppedCandidate(canonical_id="x", reason="test")],
        ),
    )


@pytest.mark.asyncio
async def test_noop_when_should_persist_missing() -> None:
    """No request.state.search_agent_should_persist => no R2 call, no
    trace_blob_key. The helper must not raise."""
    request = _mk_request()  # empty state
    await _persist_trace_blob_r2(request)
    assert not hasattr(request.state, "trace_blob_key")


@pytest.mark.asyncio
async def test_noop_when_should_persist_false() -> None:
    """Explicit False also short-circuits."""
    request = _mk_request(search_agent_should_persist=False)
    await _persist_trace_blob_r2(request)
    assert not hasattr(request.state, "trace_blob_key")


@pytest.mark.asyncio
async def test_noop_when_customer_id_missing() -> None:
    """Auth never ran => no customer_id to scope the bucket to => skip."""
    request = _mk_request(
        search_agent_should_persist=True,
        search_agent_trace_id="t-1",
        # search_agent_customer_id NOT set, customer_id NOT set
    )
    await _persist_trace_blob_r2(request)
    assert not hasattr(request.state, "trace_blob_key")


@pytest.mark.asyncio
async def test_noop_when_trace_id_missing() -> None:
    """No trace_id => can't compute the blob key => skip."""
    request = _mk_request(
        search_agent_should_persist=True,
        search_agent_customer_id="cust-1",
        # search_agent_trace_id NOT set
    )
    await _persist_trace_blob_r2(request)
    assert not hasattr(request.state, "trace_blob_key")


@pytest.mark.asyncio
async def test_happy_path_sets_trace_blob_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Successful PUT sets request.state.trace_blob_key for the next
    BackgroundTask in the chain (_build_and_write_trace) to read."""
    canned_key = "search-traces/2026-05-17/abc.json.gz"
    persist_mock = AsyncMock(return_value=canned_key)
    monkeypatch.setattr(
        "services.retrieval.agent.trace_blob.persist_trace_blob_to_r2",
        persist_mock,
    )
    monkeypatch.setattr(
        "services.retrieval.agent.trace_blob.compute_blob_key",
        lambda trace_id, now: canned_key,
    )

    request = _mk_request(
        search_agent_should_persist=True,
        search_agent_customer_id="cust-1",
        search_agent_trace_id="abc",
        search_agent_loop_state=None,
        search_agent_gathered=_mk_minimal_gathered(),
        search_agent_status="ok",
        search_agent_timing={"grounding_ms": 12.0},
        search_agent_query="hello",
        search_agent_model="accounts/fireworks/models/gpt-oss-120b",
    )
    await _persist_trace_blob_r2(request)

    assert request.state.trace_blob_key == canned_key
    assert persist_mock.await_count == 1
    args, _kwargs = persist_mock.await_args
    # signature: persist_trace_blob_to_r2(customer_id, key, payload)
    assert args[0] == "cust-1"
    assert args[1] == canned_key
    payload = args[2]
    assert payload["trace_id"] == "abc"
    assert payload["status"] == "ok"


@pytest.mark.asyncio
async def test_r2_failure_does_not_set_trace_blob_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When persist_trace_blob_to_r2 returns None (R2 outage, etc),
    the DB row must still be writable — trace_blob_key stays unset so
    _build_and_write_trace's `getattr(..., None)` writes NULL."""
    monkeypatch.setattr(
        "services.retrieval.agent.trace_blob.persist_trace_blob_to_r2",
        AsyncMock(return_value=None),
    )
    request = _mk_request(
        search_agent_should_persist=True,
        search_agent_customer_id="cust-1",
        search_agent_trace_id="abc",
        search_agent_loop_state=None,
        search_agent_gathered=_mk_minimal_gathered(),
        search_agent_status="ok",
        search_agent_timing={},
        search_agent_query="x",
        search_agent_model="m",
    )
    await _persist_trace_blob_r2(request)
    assert not hasattr(request.state, "trace_blob_key")


@pytest.mark.asyncio
async def test_swallows_exceptions_from_build_trace_blob(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Any exception inside build_trace_blob (e.g. unexpected state shape)
    must NOT propagate — would break the BackgroundTask chain and lose
    the query_traces row too."""
    monkeypatch.setattr(
        "services.retrieval.agent.trace_blob.build_trace_blob",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("kaboom")),
    )
    request = _mk_request(
        search_agent_should_persist=True,
        search_agent_customer_id="cust-1",
        search_agent_trace_id="abc",
        search_agent_loop_state=None,
        search_agent_gathered=None,
        search_agent_status="ok",
        search_agent_timing={},
        search_agent_query="x",
        search_agent_model="m",
    )
    # Should not raise
    await _persist_trace_blob_r2(request)
    assert not hasattr(request.state, "trace_blob_key")


@pytest.mark.asyncio
async def test_swallows_cancelled_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """CancelledError became BaseException-rooted in Python 3.11+; the
    helper's broad-except clause must include it explicitly so a
    shutting-down server doesn't crash through to the BackgroundTask
    chain."""
    async def cancel(*_args: Any, **_kwargs: Any) -> str:
        raise asyncio.CancelledError()

    monkeypatch.setattr(
        "services.retrieval.agent.trace_blob.persist_trace_blob_to_r2",
        cancel,
    )
    request = _mk_request(
        search_agent_should_persist=True,
        search_agent_customer_id="cust-1",
        search_agent_trace_id="abc",
        search_agent_loop_state=None,
        search_agent_gathered=_mk_minimal_gathered(),
        search_agent_status="ok",
        search_agent_timing={},
        search_agent_query="x",
        search_agent_model="m",
    )
    # Should not raise.
    await _persist_trace_blob_r2(request)
