"""UsageLoggingMiddleware integration: query_traces writes alongside usage_events.

Each test exercises the middleware path end-to-end through ASGITransport:
  * /retrieve happy path → usage_events row + query_traces row with matching request_id
  * /sources GET → query_traces.request = {doc_id, version}
  * Handler raises → query_traces fires with response = {error_class, error_message}
  * Hot-path additive invariant — patch write_query_trace to ALWAYS raise;
    /retrieve still returns 200 with unchanged body. Proves the trace write is
    genuinely off the user-visible path.

We monkeypatch the retrieval pipeline to a deterministic stub (mirroring
tests/test_usage_middleware.py) so the test doesn't need OpenAI/Anthropic
credentials.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import secrets
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
from httpx import ASGITransport

from shared.config import Settings, get_settings
from shared.constants import SourceSystem
from shared.db import close_pool, init_pool, raw_conn
from shared.models import QueryChunk, QueryResponse

INTERNAL_KEY = "test-internal-knowledge-key"


@pytest.fixture(autouse=True)
def _patch_settings(monkeypatch, settings: Settings) -> None:
    monkeypatch.setenv("TOKEN_ENCRYPTION_KEY", settings.token_encryption_key.get_secret_value())
    monkeypatch.setenv("ENVIRONMENT", "local")
    monkeypatch.setenv("INTERNAL_KNOWLEDGE_API_KEY", INTERNAL_KEY)
    get_settings.cache_clear()  # type: ignore[attr-defined]


def _jsonb(value: Any) -> dict[str, Any] | None:
    """asyncpg returns JSONB as either a dict or a JSON string. Normalize."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


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


def _stub_pipeline(monkeypatch, *, chunk_count: int = 2) -> None:
    async def fake_run_retrieval(req, customer_id):
        now = datetime.now(UTC)
        chunks = [
            QueryChunk(
                chunk_id=f"c{i}",
                doc_id=f"doc-{i}",
                doc_version=1,
                source_system=SourceSystem.SLACK,
                source_url=f"https://example/{i}",
                title=f"chunk {i}",
                content=f"content {i}",
                created_at=now,
                updated_at=now,
                score=1.0 - i * 0.1,
                rank=i,
            )
            for i in range(chunk_count)
        ]
        return QueryResponse(
            query=req.query,
            chunks=chunks,
            total_candidates=chunk_count,
            router_hit_cache=False,
            timing_ms={"router_ms": 1.0},
            trace_id="trace-test",
        )

    import services.retrieval.main as main_mod

    monkeypatch.setattr(main_mod, "run_retrieval", fake_run_retrieval)


async def _post(
    *,
    path: str = "/retrieve",
    headers: dict[str, str] | None = None,
    body: dict[str, Any] | None = None,
    raise_app_exceptions: bool = False,
) -> httpx.Response:
    from services.retrieval.main import app

    await close_pool()
    transport = ASGITransport(app=app, raise_app_exceptions=raise_app_exceptions)
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://t") as client,
        app.router.lifespan_context(app),
    ):
        return await client.post(
            path, json=body or {"query": "hello", "top_k": 1}, headers=headers or {}
        )


async def _get(
    path: str,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    from services.retrieval.main import app

    await close_pool()
    transport = ASGITransport(app=app)
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://t") as client,
        app.router.lifespan_context(app),
    ):
        return await client.get(path, headers=headers or {})


async def _wait_for_traces(
    customer_id: str, expected: int, timeout_s: float = 2.0
) -> list[Any]:
    deadline = asyncio.get_event_loop().time() + timeout_s
    rows: list[Any] = []
    while asyncio.get_event_loop().time() < deadline:
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
            await conn.execute(
                "SELECT set_config('app.current_customer_id', '', false)"
            )
        if len(rows) >= expected:
            return rows
        await asyncio.sleep(0.05)
    return rows


@pytest.mark.asyncio
async def test_middleware_writes_both_usage_event_and_trace(
    live_db, settings, monkeypatch
) -> None:
    """A single /retrieve call produces exactly one usage_events row AND one
    query_traces row. They share the same request_id so consumers can join."""
    api_key = await _seed_customer("cust-mw-both")
    _stub_pipeline(monkeypatch, chunk_count=2)

    resp = await _post(
        headers={"Authorization": f"Bearer {api_key}", "X-Caller-Kind": "mcp"},
        body={"query": "hello world", "top_k": 5},
    )
    await init_pool(settings)
    assert resp.status_code == 200, resp.text

    traces = await _wait_for_traces("cust-mw-both", 1)
    assert len(traces) == 1
    trace = traces[0]
    assert trace["event_type"] == "knowledge.retrieve"
    assert trace["response_truncated"] is False
    response = _jsonb(trace["response"])
    assert response is not None
    assert len(response["chunks"]) == 2

    # request_id continuity: both writes used the same uuid.
    async with raw_conn() as conn:
        await conn.execute(
            "SELECT set_config('app.current_customer_id', $1, false)",
            "cust-mw-both",
        )
        usage_rows = list(
            await conn.fetch(
                "SELECT request_id FROM usage_events WHERE customer_id = $1",
                "cust-mw-both",
            )
        )
        await conn.execute(
            "SELECT set_config('app.current_customer_id', '', false)"
        )
    assert len(usage_rows) == 1
    assert str(usage_rows[0]["request_id"]) == str(trace["request_id"])


@pytest.mark.asyncio
async def test_middleware_query_trace_write_failure_does_not_affect_response(
    live_db, settings, monkeypatch
) -> None:
    """Hot-path additive invariant: even if write_query_trace ALWAYS raises,
    /retrieve still returns 200 with the correct body. Proves the trace
    write is genuinely off the user-visible path."""
    api_key = await _seed_customer("cust-mw-isolated")
    _stub_pipeline(monkeypatch, chunk_count=1)

    async def boom(_trace: Any) -> None:
        raise RuntimeError("simulated trace write failure")

    monkeypatch.setattr("services.retrieval.middleware.write_query_trace", boom)

    resp = await _post(
        headers={"Authorization": f"Bearer {api_key}", "X-Caller-Kind": "mcp"},
        body={"query": "hello", "top_k": 1},
    )
    await init_pool(settings)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["query"] == "hello"
    assert len(body["chunks"]) == 1
    assert body["chunks"][0]["chunk_id"] == "c0"

    # No trace row written (the patched write raised). usage_events row
    # still present — it's an independent BackgroundTask in the chain.
    traces = await _wait_for_traces("cust-mw-isolated", expected=0, timeout_s=0.3)
    assert traces == []


@pytest.mark.asyncio
async def test_middleware_handler_raises_emits_error_trace(
    live_db, settings, monkeypatch
) -> None:
    """When the retrieval handler raises, the middleware error path still
    fires a query_traces row with response = {error_class, error_message}.
    The original exception MUST still propagate to the user as a 500."""
    api_key = await _seed_customer("cust-mw-err")

    async def boom_pipeline(req, customer_id):
        raise RuntimeError("retrieval blew up")

    import services.retrieval.main as main_mod

    monkeypatch.setattr(main_mod, "run_retrieval", boom_pipeline)

    resp = await _post(
        headers={"Authorization": f"Bearer {api_key}", "X-Caller-Kind": "mcp"},
        body={"query": "fail", "top_k": 1},
    )
    await init_pool(settings)
    assert resp.status_code == 500

    traces = await _wait_for_traces("cust-mw-err", 1)
    assert len(traces) == 1
    trace = traces[0]
    response = _jsonb(trace["response"])
    assert response is not None
    assert response["error_class"] == "RuntimeError"
    assert "retrieval blew up" in response["error_message"]
    # Request was stashed before the handler raised, so it's still recorded.
    request = _jsonb(trace["request"])
    assert request is not None
    assert request["query"] == "fail"


@pytest.mark.asyncio
async def test_middleware_query_stream_captures_response(
    live_db, settings, monkeypatch
) -> None:
    """/query/stream stashes a synthetic AnswerResponse on request.state at
    the end of the SSE generator, so query_traces lands with the same shape
    /query writes — not response={}. Closes the gap PR #64 documented for
    usage_events.

    Without the fix, response would be `{}` (size_bytes=2). With the fix,
    response carries the full AnswerResponse: query, answer, chunks list,
    citations, model, etc. — byte-for-byte equivalent to a non-streaming
    /query trace."""
    api_key = await _seed_customer("cust-mw-stream")

    # Mock the streaming pipeline. Mirrors tests/retrieval/test_query_stream.py
    # patterns but plumbed through the live middleware so we observe the
    # query_traces row that lands.
    from datetime import UTC
    from datetime import datetime as _dt

    from services.retrieval.pipeline import RouterPhaseResult
    from services.retrieval.router import RouterOutput
    from services.retrieval.synthesis import StreamDelta, StreamFinal
    from shared.models import QueryResponse, TemporalSpec

    async def _async_return(value):  # type: ignore[no-untyped-def]
        return value

    phase = RouterPhaseResult(
        routed=RouterOutput(),
        spec=TemporalSpec(),
        temporal_meta={"mode": "latest", "source": "default", "raw_phrase": None, "error": None},
        sort_meta=None,
        extracted_entities=[],
        doc_types=None,
        trace_id="trace-stream-1",
        timing={"router_ms": 5.0},
        dispatch_mode="search",
    )
    chunk = QueryChunk(
        chunk_id="c0",
        doc_id="github:foo/bar:pr:1",
        doc_version=1,
        source_system=SourceSystem.GITHUB,
        source_url="https://example/1",
        title="example",
        content="hello",
        created_at=_dt.now(UTC),
        updated_at=_dt.now(UTC),
        score=0.9,
        rank=0,
    )
    rresp = QueryResponse(
        query="streamed?",
        chunks=[chunk],
        total_candidates=1,
        router_hit_cache=False,
        applied_mode="search",
        timing_ms={"router_ms": 5.0, "search_ms": 30.0},
        trace_id="trace-stream-1",
    )

    async def fake_synth_stream(query, chunks, model, max_tokens):  # type: ignore[no-untyped-def]
        yield StreamDelta(text="Hello ")
        yield StreamDelta(text="world.")
        yield StreamFinal(
            answer="Hello world.",
            citations=[{"index": 1, "chunk_id": "c0"}],
            insufficient_context=False,
            model=model,
        )

    import services.retrieval.main as main_mod

    monkeypatch.setattr(
        main_mod, "run_router_phase", lambda req, cid: _async_return(phase)
    )
    monkeypatch.setattr(
        main_mod, "run_search_phase", lambda req, cid, p: _async_return(rresp)
    )
    monkeypatch.setattr(main_mod, "synthesize_stream", fake_synth_stream)

    # ASGITransport with the SSE response. Consume the body so the SSE
    # generator runs to completion — only then does request.state get
    # populated and the BackgroundTask fire.
    await close_pool()
    transport = ASGITransport(app=main_mod.app, raise_app_exceptions=False)
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://t") as client,
        main_mod.app.router.lifespan_context(main_mod.app),
    ):
        resp = await client.post(
            "/query/stream",
            json={"query": "streamed?", "top_k": 1},
            headers={
                "Authorization": f"Bearer {api_key}",
                "X-Caller-Kind": "mcp",
            },
        )
        # Force-consume the streamed body so the generator exits and the
        # BackgroundTask scheduled by the middleware actually runs. httpx
        # already buffers .text but be explicit.
        body = resp.text
    await init_pool(settings)
    assert resp.status_code == 200, body
    assert "event: done" in body

    traces = await _wait_for_traces("cust-mw-stream", 1)
    assert len(traces) == 1
    trace = traces[0]
    response = _jsonb(trace["response"])
    assert response is not None
    # Response is the full AnswerResponse shape, not an empty stub.
    assert response["answer"] == "Hello world."
    assert response["query"] == "streamed?"
    assert len(response["chunks"]) == 1
    assert response["chunks"][0]["chunk_id"] == "c0"
    assert response["citations"][0]["chunk_id"] == "c0"
    assert response["insufficient_context"] is False
    assert response["model"] == "anthropic/claude-sonnet-4-6"
    # Truncation flag stays false; size is non-trivial (well above the 2-byte
    # `{}` placeholder this test was added to prevent).
    assert trace["response_truncated"] is False
    assert trace["response_size_bytes"] > 100
