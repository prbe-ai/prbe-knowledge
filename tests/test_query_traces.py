"""query_traces: unit tests for write_query_trace + RLS + migration shape.

Each test exercises one slice of the contract:
  * happy path: row persisted with all fields populated
  * DB outage: write_query_trace swallows the exception
  * CancelledError swallow (Python 3.11+ doesn't catch it via except Exception)
  * 256KB response cap → truncation marker, response_truncated=true
  * chunk-count pre-check → truncation without paying for full serialization
  * error response shape: {error_class, error_message}
  * schema_version stamped on every row
  * RLS isolation: customer A's traces invisible under with_tenant("B")
  * Migration applies FORCE ROW LEVEL SECURITY

Mirrors the test patterns established by tests/test_backfill_reclaim.py and
tests/test_usage_middleware.py.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import secrets
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from engine.retrieval.usage import (
    EVENT_TYPE_GET_SOURCE,
    EVENT_TYPE_QUERY,
    EVENT_TYPE_RETRIEVE,
    QUERY_TRACE_SCHEMA_VERSION,
    QueryTrace,
    write_query_trace,
)
from engine.shared.constants import SourceSystem
from engine.shared.db import raw_conn, with_tenant
from engine.shared.models import (
    AnswerResponse,
    QueryChunk,
    QueryDocumentResult,
    QueryRequest,
    QueryResponse,
    SourceResponse,
)


async def _seed_customer(customer_id: str) -> str:
    api_key = secrets.token_urlsafe(32)
    api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO customers (customer_id, display_name, api_key_hash)
            VALUES ($1, $2, $3)
            ON CONFLICT (customer_id) DO UPDATE SET api_key_hash = EXCLUDED.api_key_hash
            """,
            customer_id,
            f"{customer_id} display",
            api_key_hash,
        )
    return api_key


def _make_query_response(chunk_count: int = 3, content_size: int = 100) -> QueryResponse:
    """Build a polymorphic QueryResponse with `chunk_count` doc results.

    One QueryDocumentResult per index, each carrying a single QueryChunk.
    The size-gating tests interpret `chunk_count` as TOTAL chunks, which
    matches `_total_chunk_count` in the polymorphic usage path.
    """
    now = datetime.now(UTC)
    docs = [
        QueryDocumentResult(
            canonical_id=f"doc-{i}",
            doc_id=f"doc-{i}",
            doc_version=1,
            source_system=SourceSystem.SLACK,
            source_url=f"https://example/{i}",
            title=f"chunk {i}",
            created_at=now,
            updated_at=now,
            score=1.0 - i * 0.01,
            rank=i + 1,
            chunks=[
                QueryChunk(
                    chunk_id=f"c{i}",
                    content="x" * content_size,
                    score=1.0 - i * 0.01,
                    rank_in_doc=1,
                )
            ],
            chunk_count=1,
        )
        for i in range(chunk_count)
    ]
    return QueryResponse(
        query="test query",
        results=list(docs),
        total_candidates=chunk_count,
        router_hit_cache=False,
        timing_ms={"router_ms": 1.0},
        trace_id="trace-test",
    )


def _jsonb(value: Any) -> dict[str, Any] | None:
    """asyncpg returns JSONB as either a dict (with codec) or a JSON string.
    Normalize to dict; return None on parse failure or non-object shapes."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


async def _fetch_traces(customer_id: str) -> list[Any]:
    async with raw_conn() as conn:
        await conn.execute(
            "SELECT set_config('app.current_customer_id', $1, false)",
            customer_id,
        )
        rows = list(
            await conn.fetch(
                "SELECT * FROM query_traces WHERE customer_id = $1 "
                "ORDER BY occurred_at",
                customer_id,
            )
        )
        await conn.execute("SELECT set_config('app.current_customer_id', '', false)")
    return rows


def test_canonical_schema_includes_search_agent_summary_columns() -> None:
    """Fresh installs use schema.sql rather than replaying migration 0078."""
    schema = (
        Path(__file__).resolve().parents[1] / "db" / "schema.sql"
    ).read_text()

    for column in (
        "gatherer_status",
        "tool_calls_count",
        "need_deeper_extensions",
        "confidence",
        "dropped_count",
        "cache_hit_rate",
    ):
        assert column in schema


@pytest.mark.asyncio
async def test_write_query_trace_includes_search_agent_summary_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The existing migration-0078 columns are present in the INSERT."""
    captured: list[tuple[str, tuple[Any, ...]]] = []

    class _CaptureConn:
        async def execute(self, statement: str, *args: Any) -> None:
            captured.append((statement, args))

    class _CaptureContext:
        async def __aenter__(self) -> _CaptureConn:
            return _CaptureConn()

        async def __aexit__(self, *args: Any) -> None:
            return None

    monkeypatch.setattr(
        "engine.retrieval.usage.with_tenant",
        lambda _customer_id: _CaptureContext(),
    )

    await write_query_trace(QueryTrace(
        customer_id="cust-summary",
        request_id=str(uuid.uuid4()),
        event_type=EVENT_TYPE_RETRIEVE,
        request_payload={"query": "fallback"},
        response_payload={"results": []},
        gatherer_status="provider_error_prefanout_fallback",
        tool_calls_count=3,
        need_deeper_extensions=1,
        confidence="low",
        dropped_count=2,
        cache_hit_rate=0.875,
        trace_blob_key="search-traces/2026-07-15/trace.json.gz",
    ))

    assert len(captured) == 1
    statement, args = captured[0]
    for column in (
        "gatherer_status",
        "tool_calls_count",
        "need_deeper_extensions",
        "confidence",
        "dropped_count",
        "cache_hit_rate",
    ):
        assert column in statement
    assert args[-7:] == (
        "provider_error_prefanout_fallback",
        3,
        1,
        "low",
        2,
        Decimal("0.875"),
        "search-traces/2026-07-15/trace.json.gz",
    )


@pytest.mark.asyncio
async def test_write_query_trace_persists_row(live_db) -> None:
    """Happy path: a retrieve trace lands with all fields populated."""
    await _seed_customer("cust-trace-ok")
    request_id = str(uuid.uuid4())
    req = QueryRequest(query="how does Pebble battery life work?", top_k=5)
    resp = _make_query_response(chunk_count=3)

    await write_query_trace(
        QueryTrace(
            customer_id="cust-trace-ok",
            request_id=request_id,
            event_type=EVENT_TYPE_RETRIEVE,
            request_payload=req,
            response_payload=resp,
            gatherer_status="ok",
            tool_calls_count=2,
            need_deeper_extensions=0,
            confidence="high",
            dropped_count=1,
            cache_hit_rate=0.875,
        )
    )

    rows = await _fetch_traces("cust-trace-ok")
    assert len(rows) == 1
    row = rows[0]
    assert row["customer_id"] == "cust-trace-ok"
    assert str(row["request_id"]) == request_id
    assert row["event_type"] == EVENT_TYPE_RETRIEVE
    assert row["schema_version"] == QUERY_TRACE_SCHEMA_VERSION
    assert row["response_truncated"] is False
    assert row["response_size_bytes"] > 0
    assert row["gatherer_status"] == "ok"
    assert row["tool_calls_count"] == 2
    assert row["need_deeper_extensions"] == 0
    assert row["confidence"] == "high"
    assert row["dropped_count"] == 1
    assert row["cache_hit_rate"] == Decimal("0.875")
    # The request column round-trips the parsed pydantic model.
    request = _jsonb(row["request"])
    assert request is not None
    assert request["query"] == "how does Pebble battery life work?"
    assert request["top_k"] == 5
    # The response column carries the polymorphic results list with each
    # Document's body chunks nested under it (PR feat/polymorphic-search-results).
    response = _jsonb(row["response"])
    assert response is not None
    assert len(response["results"]) == 3
    assert response["results"][0]["doc_id"] == "doc-0"


@pytest.mark.asyncio
async def test_write_query_trace_swallows_db_errors(
    live_db, monkeypatch
) -> None:
    """DB unreachable mid-write must not propagate. The user-visible request
    has already returned by the time the BackgroundTask runs; a logging
    failure has to degrade silently to a missing row, never a 500."""
    await _seed_customer("cust-trace-dberr")

    class _BoomConn:
        async def execute(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("simulated DB outage")

    class _BoomCtx:
        async def __aenter__(self) -> _BoomConn:
            return _BoomConn()

        async def __aexit__(self, *args: Any) -> None:
            return None

    def fake_with_tenant(_customer_id: str) -> _BoomCtx:
        return _BoomCtx()

    monkeypatch.setattr("engine.retrieval.usage.with_tenant", fake_with_tenant)

    # Must NOT raise.
    await write_query_trace(
        QueryTrace(
            customer_id="cust-trace-dberr",
            request_id=str(uuid.uuid4()),
            event_type=EVENT_TYPE_RETRIEVE,
            request_payload={"query": "x"},
            response_payload={"chunks": []},
        )
    )

    rows = await _fetch_traces("cust-trace-dberr")
    assert rows == []


@pytest.mark.asyncio
async def test_write_query_trace_swallows_cancelled_error(
    live_db, monkeypatch
) -> None:
    """Python 3.11+: asyncio.CancelledError is BaseException-rooted and is NOT
    caught by `except Exception`. The write_query_trace handler explicitly
    catches it so client disconnect / shutdown doesn't crash the task."""
    await _seed_customer("cust-trace-cancel")

    class _CancellingConn:
        async def execute(self, *args: Any, **kwargs: Any) -> None:
            raise asyncio.CancelledError()

    class _CancellingCtx:
        async def __aenter__(self) -> _CancellingConn:
            return _CancellingConn()

        async def __aexit__(self, *args: Any) -> None:
            return None

    monkeypatch.setattr(
        "engine.retrieval.usage.with_tenant",
        lambda _cid: _CancellingCtx(),
    )

    # Must NOT raise CancelledError out to the caller.
    await write_query_trace(
        QueryTrace(
            customer_id="cust-trace-cancel",
            request_id=str(uuid.uuid4()),
            event_type=EVENT_TYPE_RETRIEVE,
            request_payload={"query": "x"},
            response_payload={"chunks": []},
        )
    )

    rows = await _fetch_traces("cust-trace-cancel")
    assert rows == []


@pytest.mark.asyncio
async def test_write_query_trace_truncates_oversized_response(live_db) -> None:
    """When the serialized response exceeds RESPONSE_MAX_BYTES, store the
    truncation marker shape and flip response_truncated=true. The request
    payload remains intact."""
    await _seed_customer("cust-trace-toobig")
    # Single chunk well under chunk-count gate but with huge content so
    # the post-serialization size cap fires (256KB).
    resp = _make_query_response(chunk_count=10, content_size=40 * 1024)

    await write_query_trace(
        QueryTrace(
            customer_id="cust-trace-toobig",
            request_id=str(uuid.uuid4()),
            event_type=EVENT_TYPE_RETRIEVE,
            request_payload=QueryRequest(query="big response", top_k=5),
            response_payload=resp,
        )
    )

    rows = await _fetch_traces("cust-trace-toobig")
    assert len(rows) == 1
    row = rows[0]
    assert row["response_truncated"] is True
    response = _jsonb(row["response"])
    assert response is not None
    # Marker shape: size_bytes + chunk_count, no `chunks` array.
    assert "size_bytes" in response
    assert response.get("chunk_count") == 10
    assert "chunks" not in response
    # Request stays intact even when response is truncated.
    request = _jsonb(row["request"])
    assert request is not None
    assert request["query"] == "big response"


@pytest.mark.asyncio
async def test_write_query_trace_truncates_excess_chunk_count(live_db) -> None:
    """The pre-serialization gate fires for pathological top_k values:
    > MAX_CHUNK_COUNT_BEFORE_TRUNCATE chunks → truncation marker without
    paying for full model_dump."""
    await _seed_customer("cust-trace-many")
    # 250 chunks * small content. Tiny per-chunk size, but count alone
    # trips the chunk-count gate.
    resp = _make_query_response(chunk_count=250, content_size=10)

    await write_query_trace(
        QueryTrace(
            customer_id="cust-trace-many",
            request_id=str(uuid.uuid4()),
            event_type=EVENT_TYPE_RETRIEVE,
            request_payload=QueryRequest(query="huge top_k", top_k=250),
            response_payload=resp,
        )
    )

    rows = await _fetch_traces("cust-trace-many")
    assert len(rows) == 1
    row = rows[0]
    assert row["response_truncated"] is True
    response = _jsonb(row["response"])
    assert response is not None
    assert response.get("chunk_count") == 250
    assert "chunks" not in response


@pytest.mark.asyncio
async def test_write_query_trace_persists_error_response(live_db) -> None:
    """When the handler raised, response_payload is None and we record the
    error class + message in the response JSONB instead. response_truncated
    stays false (this isn't a truncation; it's a structurally different shape)."""
    await _seed_customer("cust-trace-err")
    request_id = str(uuid.uuid4())

    await write_query_trace(
        QueryTrace(
            customer_id="cust-trace-err",
            request_id=request_id,
            event_type=EVENT_TYPE_QUERY,
            request_payload=QueryRequest(query="failing", top_k=5),
            response_payload=None,
            error_class="HTTPException",
            error_message="upstream synthesis failed",
        )
    )

    rows = await _fetch_traces("cust-trace-err")
    assert len(rows) == 1
    row = rows[0]
    assert row["response_truncated"] is False
    response = _jsonb(row["response"])
    assert response is not None
    assert response["error_class"] == "HTTPException"
    assert response["error_message"] == "upstream synthesis failed"
    # Request still recorded — that's the data we most want for failed queries.
    request = _jsonb(row["request"])
    assert request is not None
    assert request["query"] == "failing"


@pytest.mark.asyncio
async def test_write_query_trace_stamps_schema_version(live_db) -> None:
    """Every row carries QUERY_TRACE_SCHEMA_VERSION so downstream consumers
    can filter by version when the JSONB shape changes."""
    await _seed_customer("cust-trace-ver")
    await write_query_trace(
        QueryTrace(
            customer_id="cust-trace-ver",
            request_id=str(uuid.uuid4()),
            event_type=EVENT_TYPE_GET_SOURCE,
            request_payload={"doc_id": "github:foo/bar:pr:1"},
            response_payload=SourceResponse(
                doc_id="github:foo/bar:pr:1",
                doc_version=1,
                source_system=SourceSystem.GITHUB,
                source_id="pr:1",
                source_url="https://example",
                title="t",
                content="c",
                chunk_count=1,
                body_size_bytes=1,
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
                ingested_at=datetime.now(UTC),
            ),
        )
    )

    rows = await _fetch_traces("cust-trace-ver")
    assert len(rows) == 1
    assert rows[0]["schema_version"] == QUERY_TRACE_SCHEMA_VERSION


@pytest.mark.asyncio
async def test_query_traces_rls_blocks_cross_tenant(live_db) -> None:
    """Customer A's traces must not surface under with_tenant('B'). RLS
    misfires have bitten this codebase before (see
    feedback_graph_nodes_rls_force.md); FORCE ROW LEVEL SECURITY ensures
    even a session running as the table owner is blocked.

    Mirrors test_multitenant_isolation.py: local docker's `prbe` role is
    a superuser with BYPASSRLS, so we SET LOCAL ROLE to a non-super test
    role to actually exercise the policy. In Neon prod the app role has
    no BYPASSRLS so this is the same code path."""
    await _seed_customer("cust-trace-a")
    await _seed_customer("cust-trace-b")

    await write_query_trace(
        QueryTrace(
            customer_id="cust-trace-a",
            request_id=str(uuid.uuid4()),
            event_type=EVENT_TYPE_RETRIEVE,
            request_payload={"query": "tenant a secret"},
            response_payload=QueryResponse(
                query="x", results=[], total_candidates=0,
                router_hit_cache=False, timing_ms={}, trace_id="t",
            ),
        )
    )

    async with raw_conn() as conn:
        await conn.execute(
            """
            DO $$
            BEGIN
                IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'prbe_rls_test') THEN
                    CREATE ROLE prbe_rls_test NOSUPERUSER NOBYPASSRLS;
                END IF;
            END $$;
            """
        )
        await conn.execute("GRANT USAGE ON SCHEMA public TO prbe_rls_test")
        await conn.execute("GRANT ALL ON ALL TABLES IN SCHEMA public TO prbe_rls_test")
        await conn.execute("GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO prbe_rls_test")

    # Read as tenant B (under a non-superuser role) — must see zero rows.
    async with with_tenant("cust-trace-b") as conn:
        await conn.execute("SET LOCAL ROLE prbe_rls_test")
        rows_b = await conn.fetch(
            "SELECT trace_id FROM query_traces WHERE customer_id = $1",
            "cust-trace-a",
        )
    assert rows_b == []

    # Sanity: tenant A still sees their own row under the same role.
    async with with_tenant("cust-trace-a") as conn:
        await conn.execute("SET LOCAL ROLE prbe_rls_test")
        rows_a = await conn.fetch(
            "SELECT trace_id FROM query_traces WHERE customer_id = $1",
            "cust-trace-a",
        )
    assert len(rows_a) == 1


@pytest.mark.asyncio
async def test_query_traces_force_rls_active_after_migration(live_db) -> None:
    """Verify the migration installed FORCE RLS, not just ENABLE RLS.
    Without FORCE, table-owner sessions bypass the policy — and the asyncpg
    pool runs as the owner."""
    async with raw_conn() as conn:
        row = await conn.fetchrow(
            """
            SELECT relrowsecurity, relforcerowsecurity
            FROM pg_class
            WHERE relname = 'query_traces'
            """
        )
    assert row is not None
    assert row["relrowsecurity"] is True
    assert row["relforcerowsecurity"] is True


# Sanity check: AnswerResponse round-trips through write_query_trace —
# its `answer` field is plain text plus citations. This catches any
# pydantic field that doesn't survive model_dump(mode="json").
@pytest.mark.asyncio
async def test_write_query_trace_persists_answer_response(live_db) -> None:
    await _seed_customer("cust-trace-answer")
    results = _make_query_response(chunk_count=2).results
    answer = AnswerResponse(
        query="why?",
        answer="because.",
        citations=[{"chunk_id": "c0", "page": 1}],
        insufficient_context=False,
        model="anthropic/claude-sonnet-4-6",
        results=results,
        total_candidates=2,
        router_hit_cache=False,
        timing_ms={"synthesis_ms": 50.0},
        trace_id="trace-answer",
    )
    await write_query_trace(
        QueryTrace(
            customer_id="cust-trace-answer",
            request_id=str(uuid.uuid4()),
            event_type=EVENT_TYPE_QUERY,
            request_payload={"query": "why?"},
            response_payload=answer,
        )
    )
    rows = await _fetch_traces("cust-trace-answer")
    assert len(rows) == 1
    response = _jsonb(rows[0]["response"])
    assert response is not None
    assert response["answer"] == "because."
    assert response["citations"][0]["chunk_id"] == "c0"
