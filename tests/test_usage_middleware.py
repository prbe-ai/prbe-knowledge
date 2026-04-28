"""UsageLoggingMiddleware: per-call write to usage_events.

Each test exercises one slice of the middleware contract:
  * happy path: /retrieve writes one row with all expected fields
  * error path: handler exception → status='error' row, exception still 500s
  * missing X-Caller-Kind → caller_kind='unknown'
  * /health and /usage/* are NOT logged
  * write_usage_event() failure is swallowed — request still 200s

We monkeypatch the retrieval pipeline to a deterministic stub so the test
doesn't need OpenAI/Anthropic credentials. The middleware is the unit
under test; the pipeline's correctness is covered elsewhere.
"""

from __future__ import annotations

import asyncio
import hashlib
import secrets
from typing import Any

import httpx
import pytest
from httpx import ASGITransport

from shared.config import Settings, get_settings
from shared.db import close_pool, init_pool, raw_conn

INTERNAL_KEY = "test-internal-knowledge-key"


@pytest.fixture(autouse=True)
def _patch_settings(monkeypatch, settings: Settings) -> None:
    monkeypatch.setenv("TOKEN_ENCRYPTION_KEY", settings.token_encryption_key.get_secret_value())
    monkeypatch.setenv("ENVIRONMENT", "local")
    monkeypatch.setenv("INTERNAL_KNOWLEDGE_API_KEY", INTERNAL_KEY)
    get_settings.cache_clear()  # type: ignore[attr-defined]


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


def _stub_pipeline(monkeypatch, *, chunk_count: int = 3) -> None:
    """Replace run_retrieval with a deterministic stub returning N chunks."""
    from datetime import UTC, datetime

    from shared.constants import SourceSystem
    from shared.models import QueryChunk, QueryResponse

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


async def _post_retrieve(
    *,
    headers: dict[str, str] | None = None,
    body: dict[str, Any] | None = None,
    path: str = "/retrieve",
    raise_app_exceptions: bool = False,
) -> httpx.Response:
    from services.retrieval.main import app

    await close_pool()
    # raise_app_exceptions=False so a handler RuntimeError surfaces as a
    # 500 Response (matching prod behavior) instead of bubbling out of the
    # transport. The middleware test asserts on the 500 + the usage row.
    transport = ASGITransport(app=app, raise_app_exceptions=raise_app_exceptions)
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://t") as client,
        app.router.lifespan_context(app),
    ):
        return await client.post(path, json=body or {"query": "hello", "top_k": 1}, headers=headers or {})


async def _wait_for_usage_rows(customer_id: str, expected: int, timeout_s: float = 2.0) -> list[Any]:
    """Poll usage_events for the expected number of rows.

    BackgroundTask runs after the response is written, so the row may not
    be visible immediately when the test thread continues. Poll until it
    is, with a short cap.
    """
    deadline = asyncio.get_event_loop().time() + timeout_s
    rows: list[Any] = []
    while asyncio.get_event_loop().time() < deadline:
        async with raw_conn() as conn:
            await conn.execute("SELECT set_config('app.current_customer_id', $1, false)", customer_id)
            rows = list(
                await conn.fetch(
                    "SELECT * FROM usage_events WHERE customer_id = $1 ORDER BY occurred_at",
                    customer_id,
                )
            )
            await conn.execute("SELECT set_config('app.current_customer_id', '', false)")
        if len(rows) >= expected:
            return rows
        await asyncio.sleep(0.05)
    return rows


@pytest.mark.asyncio
async def test_middleware_writes_row_on_successful_retrieve(live_db, settings, monkeypatch) -> None:
    api_key = await _seed_customer("cust-mw-ok")
    _stub_pipeline(monkeypatch, chunk_count=2)

    resp = await _post_retrieve(
        headers={"Authorization": f"Bearer {api_key}", "X-Caller-Kind": "mcp"},
        body={"query": "hello world", "top_k": 5},
    )
    await init_pool(settings)
    assert resp.status_code == 200, resp.text

    rows = await _wait_for_usage_rows("cust-mw-ok", 1)
    assert len(rows) == 1
    row = rows[0]
    assert row["customer_id"] == "cust-mw-ok"
    assert row["caller_kind"] == "mcp"
    assert row["event_type"] == "knowledge.retrieve"
    assert row["endpoint"] == "/retrieve"
    assert row["status"] == "ok"
    assert row["error_class"] is None
    assert row["latency_ms"] is not None and row["latency_ms"] >= 0
    assert row["result_count"] == 2
    assert row["summary"] == "hello world"
    assert row["request_id"] is not None


@pytest.mark.asyncio
async def test_middleware_records_error_when_handler_raises(live_db, settings, monkeypatch) -> None:
    api_key = await _seed_customer("cust-mw-err")

    async def boom(req, customer_id):
        raise RuntimeError("pipeline kaboom")

    import services.retrieval.main as main_mod

    monkeypatch.setattr(main_mod, "run_retrieval", boom)

    resp = await _post_retrieve(
        headers={"Authorization": f"Bearer {api_key}", "X-Caller-Kind": "mcp"},
        body={"query": "kaboom", "top_k": 1},
    )
    await init_pool(settings)
    assert resp.status_code == 500, resp.text  # original exception still 500s

    rows = await _wait_for_usage_rows("cust-mw-err", 1)
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "error"
    assert row["error_class"] == "RuntimeError"
    assert row["event_type"] == "knowledge.retrieve"


@pytest.mark.asyncio
async def test_middleware_defaults_caller_kind_to_unknown(live_db, settings, monkeypatch) -> None:
    """Missing X-Caller-Kind → 'unknown'. Backwards-compatible."""
    api_key = await _seed_customer("cust-mw-noheader")
    _stub_pipeline(monkeypatch)
    resp = await _post_retrieve(
        headers={"Authorization": f"Bearer {api_key}"},  # no X-Caller-Kind
        body={"query": "hi", "top_k": 1},
    )
    await init_pool(settings)
    assert resp.status_code == 200, resp.text

    rows = await _wait_for_usage_rows("cust-mw-noheader", 1)
    assert rows and rows[0]["caller_kind"] == "unknown"


@pytest.mark.asyncio
async def test_middleware_skips_health_and_usage_paths(live_db, settings, monkeypatch) -> None:
    """/health and /usage/* must not produce usage_events rows."""
    api_key = await _seed_customer("cust-mw-skip")
    _stub_pipeline(monkeypatch)

    from services.retrieval.main import app

    await close_pool()
    transport = ASGITransport(app=app)
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://t") as client,
        app.router.lifespan_context(app),
    ):
        h = await client.get("/health")
        assert h.status_code in (200, 503)

        u = await client.get(
            "/usage/feed?window=24h",
            headers={"Authorization": f"Bearer {api_key}", "X-Caller-Kind": "dashboard"},
        )
        assert u.status_code == 200, u.text

    await init_pool(settings)

    # Quick wait then assert no rows for either skipped path.
    await asyncio.sleep(0.1)
    async with raw_conn() as conn:
        await conn.execute("SELECT set_config('app.current_customer_id', 'cust-mw-skip', false)")
        n = await conn.fetchval(
            "SELECT COUNT(*) FROM usage_events WHERE customer_id = 'cust-mw-skip'"
        )
    assert n == 0


@pytest.mark.asyncio
async def test_middleware_write_failure_does_not_500_request(live_db, settings, monkeypatch) -> None:
    """If write_usage_event() raises (DB down), the original /retrieve
    response must still be 200. write_usage_event has its own try/except
    so this should be the case unconditionally — we verify by patching
    write_usage_event to a function that raises, then confirming the
    request still succeeds and (separately) no row is written."""
    api_key = await _seed_customer("cust-mw-writefail")
    _stub_pipeline(monkeypatch)

    raised = {"count": 0}

    async def explode(_event):
        raised["count"] += 1
        raise RuntimeError("db simulated outage")

    # Patch the symbol the middleware imported — middleware.py did
    # `from services.retrieval.usage import write_usage_event`.
    import services.retrieval.middleware as mw_mod

    monkeypatch.setattr(mw_mod, "write_usage_event", explode)

    resp = await _post_retrieve(
        headers={"Authorization": f"Bearer {api_key}", "X-Caller-Kind": "mcp"},
        body={"query": "still works", "top_k": 1},
    )
    await init_pool(settings)
    assert resp.status_code == 200, resp.text

    # Give the BackgroundTask a chance to fire (and explode).
    await asyncio.sleep(0.1)

    # Even though our patch raised, no row should have been inserted —
    # and crucially the 200 already came back.
    async with raw_conn() as conn:
        await conn.execute(
            "SELECT set_config('app.current_customer_id', 'cust-mw-writefail', false)"
        )
        n = await conn.fetchval(
            "SELECT COUNT(*) FROM usage_events WHERE customer_id = 'cust-mw-writefail'"
        )
    assert n == 0
