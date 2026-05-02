"""Integration tests for POST /query/stream — SSE event sequence + error path.

The pipeline (router + search + synthesis) is fully mocked so these tests
don't need a live DB or Anthropic key. Auth is bypassed via FastAPI's
dependency_overrides. We assert on the raw SSE bytes emitted to the
client so any change to the event names, ordering, or payload shape
will surface here before it ships.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
from httpx import ASGITransport

from services.retrieval.auth import authenticate_query
from services.retrieval.pipeline import RouterPhaseResult
from services.retrieval.router import RouterOutput
from services.retrieval.synthesis import StreamDelta, StreamFinal, SynthesisError
from shared.models import (
    QueryChunk,
    QueryRequest,
    QueryResponse,
    SourceSystem,
    TemporalSpec,
)

pytestmark = pytest.mark.asyncio


def _parse_sse(body: str) -> list[tuple[str, dict]]:
    """Parse a `text/event-stream` body into [(event_name, data_dict), ...].

    SSE frames are blank-line-separated. Each frame may carry an `event:`
    line and one or more `data:` lines (we only emit one). Anything we
    don't recognize is ignored.
    """
    out: list[tuple[str, dict]] = []
    for frame in body.strip().split("\n\n"):
        event = ""
        data_lines: list[str] = []
        for line in frame.splitlines():
            if line.startswith("event:"):
                event = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:") :].strip())
        if event:
            payload = json.loads("\n".join(data_lines)) if data_lines else {}
            out.append((event, payload))
    return out


def _phase_result() -> RouterPhaseResult:
    return RouterPhaseResult(
        routed=RouterOutput(),
        spec=TemporalSpec(),
        temporal_meta={"mode": "latest", "source": "default", "raw_phrase": None, "error": None},
        sort_meta=None,
        extracted_entities=[
            {"entity_type": "repo", "canonical_id": "prbe", "display_name": "prbe", "confidence": 0.9}
        ],
        doc_types=None,
        trace_id="q-test-1",
        timing={"router_ms": 10.0},
        dispatch_mode="search",
    )


def _query_response() -> QueryResponse:
    chunk = QueryChunk(
        chunk_id="chunk-1",
        doc_id="github:prbe-ai/x:pr:1",
        doc_version=1,
        source_system=SourceSystem.GITHUB,
        source_url="https://example.com/x/pr/1",
        title="example",
        content="hello world",
        author_id="alice",
        created_at=datetime(2026, 4, 1, tzinfo=UTC),
        updated_at=datetime(2026, 4, 2, tzinfo=UTC),
        score=0.9,
        rank=1,
        retriever_scores={"vector": 0.9},
    )
    return QueryResponse(
        query="q",
        chunks=[chunk],
        total_candidates=1,
        router_hit_cache=False,
        applied_temporal={"mode": "latest", "source": "default", "raw_phrase": None, "error": None},
        applied_sort=None,
        applied_entity_filter=None,
        applied_mode="search",
        applied_doc_types=None,
        extracted_entities=[
            {"entity_type": "repo", "canonical_id": "prbe", "display_name": "prbe", "confidence": 0.9}
        ],
        aggregation=None,
        timing_ms={"router_ms": 10.0, "search_ms": 50.0},
        trace_id="q-test-1",
    )


async def _post(body: dict[str, Any]) -> httpx.Response:
    """POST to /query/stream with auth dependency overridden to a fixed customer."""
    from services.retrieval.main import app

    app.dependency_overrides[authenticate_query] = lambda: "cust-test"
    try:
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://t"
        ) as client:
            return await client.post("/query/stream", json=body)
    finally:
        app.dependency_overrides.pop(authenticate_query, None)


async def test_query_stream_emits_full_event_sequence(monkeypatch) -> None:
    """Happy path: refining → entities → searching → chunks → synthesizing →
    delta(*) → done. Verifies ordering, payload shape, and that text deltas
    are forwarded one-for-one from synthesize_stream.
    """
    monkeypatch.setattr(
        "services.retrieval.main.run_router_phase",
        lambda req, customer_id: _async_return(_phase_result()),
    )
    monkeypatch.setattr(
        "services.retrieval.main.run_search_phase",
        lambda req, customer_id, phase: _async_return(_query_response()),
    )

    delta_texts = ["Hello ", "world ", "[chunk:1]."]

    async def fake_stream(query, chunks, model, max_tokens):  # type: ignore[no-untyped-def]
        for t in delta_texts:
            yield StreamDelta(text=t)
        yield StreamFinal(
            answer="Hello world [chunk:1].",
            citations=[{"index": 1, "chunk_id": "chunk-1"}],
            insufficient_context=False,
            model=model,
        )

    monkeypatch.setattr("services.retrieval.main.synthesize_stream", fake_stream)

    resp = await _post({"query": "what shipped?", "top_k": 5})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse(resp.text)
    names = [e for e, _ in events]
    assert names == [
        "step",
        "entities",
        "step",
        "chunks",
        "step",
        "delta",
        "delta",
        "delta",
        "done",
    ]

    # Step values fire in the right order.
    step_values = [data["step"] for name, data in events if name == "step"]
    assert step_values == ["refining", "searching", "synthesizing"]

    # Entities event carries the router output.
    entities_payload = next(d for n, d in events if n == "entities")
    assert entities_payload["extracted_entities"][0]["canonical_id"] == "prbe"
    assert entities_payload["applied_mode"] == "search"
    assert entities_payload["trace_id"] == "q-test-1"

    # Chunks event is dumped as JSON-ready dicts.
    chunks_payload = next(d for n, d in events if n == "chunks")
    assert len(chunks_payload["chunks"]) == 1
    assert chunks_payload["chunks"][0]["chunk_id"] == "chunk-1"
    assert chunks_payload["total_candidates"] == 1

    # Deltas preserve order and content.
    deltas = [d["text"] for n, d in events if n == "delta"]
    assert deltas == delta_texts

    # Done event has the final assembled answer + citations + timing.
    done = next(d for n, d in events if n == "done")
    assert done["answer"] == "Hello world [chunk:1]."
    assert done["citations"] == [{"index": 1, "chunk_id": "chunk-1"}]
    assert done["insufficient_context"] is False
    assert "synthesis_ms" in done["timing_ms"]
    assert "total_ms" in done["timing_ms"]


async def test_query_stream_emits_error_event_on_synthesis_failure(monkeypatch) -> None:
    """Synthesis blow-up surfaces as a single SSE `error` frame mid-stream
    rather than tearing the HTTP response down with a 5xx. The frames that
    succeeded before the failure (refining, entities, searching, chunks,
    synthesizing) still reach the client.
    """
    monkeypatch.setattr(
        "services.retrieval.main.run_router_phase",
        lambda req, customer_id: _async_return(_phase_result()),
    )
    monkeypatch.setattr(
        "services.retrieval.main.run_search_phase",
        lambda req, customer_id, phase: _async_return(_query_response()),
    )

    async def boom_stream(query, chunks, model, max_tokens):  # type: ignore[no-untyped-def]
        # Yield no events — raise immediately, mimicking an Anthropic auth
        # failure or model-not-allowed at the start of synthesis.
        raise SynthesisError("ANTHROPIC_API_KEY not configured")
        yield  # pragma: no cover — keeps the function an async generator

    monkeypatch.setattr("services.retrieval.main.synthesize_stream", boom_stream)

    resp = await _post({"query": "what shipped?", "top_k": 5})
    assert resp.status_code == 200  # SSE always 200; status lives in the data channel
    events = _parse_sse(resp.text)
    names = [e for e, _ in events]
    # Earlier phases still arrive; only the synthesis stream itself is replaced
    # with the error event.
    assert names[:5] == ["step", "entities", "step", "chunks", "step"]
    assert names[-1] == "error"
    err = events[-1][1]
    assert err["status"] == 502
    assert "synthesis failed" in err["detail"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _async_return(value):  # type: ignore[no-untyped-def]
    """Return `value` from an awaitable. Used to make plain values
    monkeypatch-substitutable for `async def` functions in main.py.
    """
    return value


# Sanity: the QueryRequest import has to resolve so tests fail fast
# rather than silently passing on a typo.
_ = QueryRequest
