"""Tests for the trace blob assembler + R2 persister.

Pure-function tests (build_trace_blob, compute_blob_key) run without any
external dependency. The persister test monkey-patches the ObjectStore so
no real R2/MinIO traffic happens (per feedback_no_real_cli_in_tests.md).
"""

from __future__ import annotations

import gzip
import json
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from services.retrieval.agent.loop import LoopState
from services.retrieval.agent.models import (
    GatheredChunk,
    GathererNotes,
    GathererOutput,
)
from services.retrieval.agent.trace_blob import (
    TRACE_BLOB_SCHEMA_VERSION,
    build_trace_blob,
    compute_blob_key,
    persist_trace_blob_to_r2,
)
from shared.exceptions import StorageUnavailable

# ============================================================
# Fixtures
# ============================================================


def _mk_state(**overrides: Any) -> LoopState:
    defaults: dict[str, Any] = {
        "customer_id": "cust-123",
        "trace_id": "trace-abc",
        "query": "what shipped this week",
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "sys"}]},
            {"role": "user", "content": "what shipped this week"},
            {"role": "assistant", "content": "tool_calls"},
            {"role": "tool", "tool_call_id": "c1", "content": '{"hits":[]}'},
        ],
        "tools_fired": ["vector_search", "bm25_search"],
        "turn_count": 2,
        "tool_calls_count": 5,
        "extensions_used": 0,
        "cache_hit_rates": [0.0, 0.8],
        "turn_1_tools_fired": ["vector_search", "bm25_search", "graph_search", "inferred_edge_search"],
        "turn_latencies_ms": [510.2, 420.7],
        "tool_latencies_ms": [120.0, 95.5, 80.3],
        "prose_retries": 0,
        "prefanout": {"vector": {"hits": []}, "bm25": {"hits": []}},
        "prefanout_hit_counts": {"vector": 3, "bm25": 2, "graph": 0, "inferred_edge": 1},
    }
    defaults.update(overrides)
    return LoopState(**defaults)


def _mk_gathered(confidence: str = "high") -> GathererOutput:
    return GathererOutput(
        entities=[],
        chunks=[
            GatheredChunk(
                doc_id="doc-1",
                chunk_id="chunk-1",
                content="body",
                matched_via=["vector"],
                why_relevant="surfaced via vector channel",
            ),
        ],
        gatherer_notes=GathererNotes(
            turns_used=2,
            tools_called=["vector_search"],
            confidence=confidence,
            dropped=[],
        ),
    )


# ============================================================
# build_trace_blob
# ============================================================


def test_build_trace_blob_includes_all_fields() -> None:
    state = _mk_state()
    gathered = _mk_gathered(confidence="medium")
    blob = build_trace_blob(
        state=state,
        gathered=gathered,
        status="ok",
        timing={"grounding_ms": 12.5, "prefanout_ms": 480.1, "agent_ms": 1100.0},
        query="what shipped this week",
        customer_id="cust-123",
        trace_id="trace-abc",
        model="accounts/fireworks/models/gpt-oss-120b",
    )
    # Top-level keys
    assert blob["schema_version"] == TRACE_BLOB_SCHEMA_VERSION
    assert blob["trace_id"] == "trace-abc"
    assert blob["customer_id"] == "cust-123"
    assert blob["query"] == "what shipped this week"
    assert blob["model"] == "accounts/fireworks/models/gpt-oss-120b"
    assert blob["status"] == "ok"
    assert blob["timing_ms"]["grounding_ms"] == 12.5

    # LoopState fields preserved
    assert blob["messages"] == state.messages
    assert blob["tools_fired"] == state.tools_fired
    assert blob["turn_1_tools_fired"] == state.turn_1_tools_fired
    assert blob["turn_count"] == 2
    assert blob["tool_calls_count"] == 5
    assert blob["extensions_used"] == 0
    assert blob["cache_hit_rates"] == [0.0, 0.8]
    assert blob["turn_latencies_ms"] == [510.2, 420.7]
    assert blob["tool_latencies_ms"] == [120.0, 95.5, 80.3]
    assert blob["prose_retries"] == 0
    assert blob["prefanout"] == state.prefanout
    assert blob["prefanout_hit_counts"]["graph"] == 0

    # GathererOutput round-trips through model_dump
    assert blob["gathered"]["gatherer_notes"]["confidence"] == "medium"
    assert blob["gathered"]["chunks"][0]["doc_id"] == "doc-1"


def test_build_trace_blob_serializes_to_json() -> None:
    """The blob must be json.dumps-able — no datetimes, no Pydantic
    leaks, no non-serializable types from state.messages.
    """
    state = _mk_state()
    blob = build_trace_blob(
        state=state,
        gathered=_mk_gathered(),
        status="ok",
        timing={"grounding_ms": 12.5},
        query="q",
        customer_id="c",
        trace_id="t",
        model="m",
    )
    # default=str is what persist_trace_blob_to_r2 uses; equivalent here.
    serialized = json.dumps(blob, default=str)
    # Round-trip back; should not raise
    parsed = json.loads(serialized)
    assert parsed["schema_version"] == TRACE_BLOB_SCHEMA_VERSION


def test_build_trace_blob_handles_none_state() -> None:
    """Pre-loop failure path — state was never constructed. Blob still
    produces structurally complete output so analyzer schema is stable.
    """
    blob = build_trace_blob(
        state=None,
        gathered=None,
        status="fatal_provider_error",
        timing={"grounding_ms": 5.0},
        query="q",
        customer_id="c",
        trace_id="t",
        model="m",
    )
    assert blob["status"] == "fatal_provider_error"
    assert blob["gathered"] is None
    assert blob["messages"] == []
    assert blob["turn_count"] == 0
    assert blob["prefanout"] == {}


def test_build_trace_blob_handles_none_gathered() -> None:
    """503 path — state is set (loop began) but gathered is None
    (LLMError raised before final emission).
    """
    state = _mk_state(turn_count=1, tool_calls_count=1)
    blob = build_trace_blob(
        state=state,
        gathered=None,
        status="fatal_provider_error",
        timing={"grounding_ms": 5.0, "agent_ms": 250.0},
        query="q",
        customer_id="c",
        trace_id="t",
        model="m",
    )
    assert blob["status"] == "fatal_provider_error"
    assert blob["gathered"] is None
    assert blob["turn_count"] == 1
    # State partial-fields still recorded for analyst review
    assert blob["tools_fired"] == state.tools_fired


def test_build_trace_blob_handles_none_status() -> None:
    """Defensive — should never happen in practice but tolerate it."""
    blob = build_trace_blob(
        state=None,
        gathered=None,
        status=None,
        timing={},
        query="q",
        customer_id="c",
        trace_id="t",
        model="m",
    )
    assert blob["status"] is None


# ============================================================
# compute_blob_key
# ============================================================


def test_compute_blob_key_format() -> None:
    now = datetime(2026, 5, 17, 14, 30, tzinfo=UTC)
    key = compute_blob_key("trace-abc-123", now)
    assert key == "search-traces/2026-05-17/trace-abc-123.json.gz"


def test_compute_blob_key_uses_utc_date() -> None:
    # Same UTC-instant; key uses the UTC date portion.
    now = datetime(2026, 12, 31, 23, 59, tzinfo=UTC)
    assert compute_blob_key("x", now) == "search-traces/2026-12-31/x.json.gz"


# ============================================================
# persist_trace_blob_to_r2 — never raises
# ============================================================


class _FakeStore:
    def __init__(self) -> None:
        self.put_calls: list[tuple[str, str, bytes, str]] = []
        self.bucket_resolve_called: list[str] = []

    async def bucket_for(self, customer_id: str) -> str:
        self.bucket_resolve_called.append(customer_id)
        return f"bucket-{customer_id}"

    async def put(
        self, bucket: str, key: str, body: bytes, content_type: str = "application/json"
    ) -> SimpleNamespace:
        self.put_calls.append((bucket, key, body, content_type))
        return SimpleNamespace(bucket=bucket, key=key)


@pytest.mark.asyncio
async def test_persist_trace_blob_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeStore()
    monkeypatch.setattr(
        "services.retrieval.agent.trace_blob.get_store", lambda: fake
    )
    payload = {"schema_version": 1, "trace_id": "t", "messages": [{"a": "b"}]}
    result = await persist_trace_blob_to_r2(
        "cust-1", "search-traces/2026-05-17/t.json.gz", payload
    )
    assert result == "search-traces/2026-05-17/t.json.gz"
    assert fake.bucket_resolve_called == ["cust-1"]
    assert len(fake.put_calls) == 1
    bucket, key, body, content_type = fake.put_calls[0]
    assert bucket == "bucket-cust-1"
    assert key == "search-traces/2026-05-17/t.json.gz"
    assert content_type == "application/json"
    # Body is gzipped JSON — decompress and verify round-trip
    decoded = json.loads(gzip.decompress(body).decode("utf-8"))
    assert decoded == payload


@pytest.mark.asyncio
async def test_persist_trace_blob_swallows_storage_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeStore()
    fake.put = AsyncMock(side_effect=StorageUnavailable("R2 down"))  # type: ignore[assignment]
    monkeypatch.setattr(
        "services.retrieval.agent.trace_blob.get_store", lambda: fake
    )
    result = await persist_trace_blob_to_r2("cust-1", "key", {"a": 1})
    assert result is None


@pytest.mark.asyncio
async def test_persist_trace_blob_swallows_arbitrary_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Defensive: any unexpected exception (TypeError, KeyError, OOM)
    must NOT escape — the BackgroundTask chain would break otherwise.
    """
    fake = _FakeStore()
    fake.put = AsyncMock(side_effect=RuntimeError("unexpected"))  # type: ignore[assignment]
    monkeypatch.setattr(
        "services.retrieval.agent.trace_blob.get_store", lambda: fake
    )
    result = await persist_trace_blob_to_r2("cust-1", "key", {"a": 1})
    assert result is None


@pytest.mark.asyncio
async def test_persist_trace_blob_bucket_lookup_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """bucket_for raising must also be swallowed."""
    fake = _FakeStore()
    fake.bucket_for = AsyncMock(  # type: ignore[assignment]
        side_effect=StorageUnavailable("no customers row")
    )
    monkeypatch.setattr(
        "services.retrieval.agent.trace_blob.get_store", lambda: fake
    )
    result = await persist_trace_blob_to_r2("cust-1", "key", {"a": 1})
    assert result is None
