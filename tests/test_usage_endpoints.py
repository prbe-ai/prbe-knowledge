"""GET /usage/{feed,stats,search}: read endpoints over usage_events.

Each endpoint test exercises:
  * the response shape (Pydantic model contract lane D consumes)
  * RLS isolation (tenant A can never see tenant B's events)
  * the spec-mandated filters / window math
  * defensive behavior on edge inputs (FTS injection, empty windows)

The dashboard's /query/usage page is the primary consumer; lane B/D will
spec their UI off this contract.
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from httpx import ASGITransport

from engine.shared.config import Settings, get_settings
from engine.shared.db import close_pool, init_pool, raw_conn

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


async def _insert_event(
    customer_id: str,
    *,
    occurred_at: datetime | None = None,
    caller_kind: str = "mcp",
    event_type: str = "knowledge.retrieve",
    endpoint: str = "/retrieve",
    summary: str = "alpha bravo",
    status: str = "ok",
    latency_ms: int | None = 100,
    error_class: str | None = None,
) -> None:
    if occurred_at is None:
        occurred_at = datetime.now(UTC)
    async with raw_conn() as conn:
        await conn.execute(
            "SELECT set_config('app.current_customer_id', $1, false)", customer_id
        )
        await conn.execute(
            """
            INSERT INTO usage_events (
                customer_id, occurred_at, caller_kind, event_type, endpoint,
                summary, status, latency_ms, error_class
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """,
            customer_id,
            occurred_at,
            caller_kind,
            event_type,
            endpoint,
            summary,
            status,
            latency_ms,
            error_class,
        )
        await conn.execute("SELECT set_config('app.current_customer_id', '', false)")


async def _client_get(path: str, *, headers: dict[str, str]) -> httpx.Response:
    from engine.retrieval.main import app

    await close_pool()
    transport = ASGITransport(app=app)
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://t") as client,
        app.router.lifespan_context(app),
    ):
        return await client.get(path, headers=headers)


# ---------------------------------------------------------------------------
# /usage/feed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_feed_returns_events_ordered_desc(live_db, settings) -> None:
    api_key = await _seed_customer("cust-feed-order")
    now = datetime.now(UTC)
    for i in range(5):
        await _insert_event("cust-feed-order", occurred_at=now - timedelta(minutes=i))

    resp = await _client_get(
        "/usage/feed?window=24h", headers={"Authorization": f"Bearer {api_key}"}
    )
    await init_pool(settings)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["count"] == 5
    assert body["window"] == "24h"
    occurred = [e["occurred_at"] for e in body["events"]]
    assert occurred == sorted(occurred, reverse=True)


@pytest.mark.asyncio
async def test_feed_window_24h_excludes_older_rows(live_db, settings) -> None:
    api_key = await _seed_customer("cust-feed-window")
    now = datetime.now(UTC)
    await _insert_event("cust-feed-window", occurred_at=now - timedelta(hours=1))
    await _insert_event("cust-feed-window", occurred_at=now - timedelta(hours=25))

    resp = await _client_get(
        "/usage/feed?window=24h", headers={"Authorization": f"Bearer {api_key}"}
    )
    await init_pool(settings)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["count"] == 1


@pytest.mark.asyncio
async def test_feed_honors_caller_kind_and_event_type(live_db, settings) -> None:
    api_key = await _seed_customer("cust-feed-filters")
    await _insert_event("cust-feed-filters", caller_kind="mcp", event_type="knowledge.retrieve")
    await _insert_event("cust-feed-filters", caller_kind="dashboard", event_type="knowledge.retrieve")
    await _insert_event("cust-feed-filters", caller_kind="mcp", event_type="knowledge.query")

    resp = await _client_get(
        "/usage/feed?caller_kind=mcp", headers={"Authorization": f"Bearer {api_key}"}
    )
    await init_pool(settings)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["count"] == 2
    assert all(e["caller_kind"] == "mcp" for e in body["events"])

    resp = await _client_get(
        "/usage/feed?event_type=knowledge.query",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    await init_pool(settings)
    body = resp.json()
    assert body["count"] == 1
    assert body["events"][0]["event_type"] == "knowledge.query"


@pytest.mark.asyncio
async def test_feed_rls_isolation(live_db, settings) -> None:
    """Tenant A's /usage/feed must never include tenant B's rows."""
    api_key_a = await _seed_customer("cust-A")
    await _seed_customer("cust-B")
    await _insert_event("cust-A", summary="alpha-only")
    await _insert_event("cust-B", summary="bravo-only")

    resp = await _client_get(
        "/usage/feed", headers={"Authorization": f"Bearer {api_key_a}"}
    )
    await init_pool(settings)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["count"] == 1
    assert body["events"][0]["summary"] == "alpha-only"


@pytest.mark.asyncio
async def test_feed_rejects_unknown_window(live_db, settings) -> None:
    api_key = await _seed_customer("cust-feed-bad-window")
    resp = await _client_get(
        "/usage/feed?window=42years",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    await init_pool(settings)
    assert resp.status_code == 400, resp.text


# ---------------------------------------------------------------------------
# /usage/stats
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stats_returns_counts_and_percentiles(live_db, settings) -> None:
    api_key = await _seed_customer("cust-stats")
    # Five OK rows with latencies 100..500 → p50=300, p95=480.
    for i, ms in enumerate([100, 200, 300, 400, 500]):
        await _insert_event(
            "cust-stats",
            caller_kind="mcp" if i % 2 == 0 else "dashboard",
            event_type="knowledge.retrieve" if i < 3 else "knowledge.query",
            latency_ms=ms,
            status="ok",
        )
    # One error row — should NOT contribute to latency, should bump error_count.
    await _insert_event(
        "cust-stats", status="error", error_class="RuntimeError", latency_ms=9999
    )

    resp = await _client_get(
        "/usage/stats?window=24h", headers={"Authorization": f"Bearer {api_key}"}
    )
    await init_pool(settings)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["total"] == 6
    assert body["error_count"] == 1
    assert body["window"] == "24h"
    # p50 of [100,200,300,400,500] is 300 (percentile_cont).
    assert body["latency_p50_ms"] == 300
    # p95 of [100,200,300,400,500] is 480.
    assert body["latency_p95_ms"] == 480
    assert body["by_caller_kind"] == {"mcp": 4, "dashboard": 2}
    assert body["by_event_type"] == {"knowledge.retrieve": 4, "knowledge.query": 2}


@pytest.mark.asyncio
async def test_stats_empty_window_returns_zeros(live_db, settings) -> None:
    """No events in the window → zeros, not 500."""
    api_key = await _seed_customer("cust-stats-empty")
    resp = await _client_get(
        "/usage/stats?window=24h", headers={"Authorization": f"Bearer {api_key}"}
    )
    await init_pool(settings)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["total"] == 0
    assert body["error_count"] == 0
    assert body["latency_p50_ms"] is None
    assert body["latency_p95_ms"] is None
    assert body["by_caller_kind"] == {}
    assert body["by_event_type"] == {}


# ---------------------------------------------------------------------------
# /usage/search
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_finds_summary_text(live_db, settings) -> None:
    api_key = await _seed_customer("cust-search")
    await _insert_event("cust-search", summary="show me linear ticket ABC-123")
    await _insert_event("cust-search", summary="status of klavis integration")
    await _insert_event("cust-search", summary="something unrelated entirely")

    resp = await _client_get(
        "/usage/search?q=klavis", headers={"Authorization": f"Bearer {api_key}"}
    )
    await init_pool(settings)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["count"] == 1
    assert "klavis" in body["events"][0]["summary"]


@pytest.mark.asyncio
async def test_search_uses_plainto_tsquery(live_db, settings) -> None:
    """plainto_tsquery treats operator characters as literal text — no
    tsquery injection. A query like 'foo & bar' must NOT be parsed as
    'foo AND bar' by the FTS layer; it should match summaries containing
    both tokens but only because plainto turns whitespace separators into
    AND under the hood, not because '&' did something hostile."""
    api_key = await _seed_customer("cust-search-injection")
    # Summary with literal ampersand.
    await _insert_event(
        "cust-search-injection",
        summary="auth & retry handling for 401",
    )
    # Summary that would match a malicious to_tsquery('foo | bar') if we
    # were vulnerable — but plainto doesn't do operator parsing.
    await _insert_event(
        "cust-search-injection",
        summary="random unrelated thing",
    )

    # plainto_tsquery('foo & bar') turns into 'foo' & 'bar' as a phrase
    # token search — '&' becomes whitespace which becomes AND. Asserting
    # the response shape doesn't error AND filters by content correctly.
    resp = await _client_get(
        "/usage/search?q=auth retry",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    await init_pool(settings)
    assert resp.status_code == 200, resp.text
    assert resp.json()["count"] == 1

    # Now a payload with raw operator syntax. Must NOT 500 and must NOT
    # match the unrelated row (no operator semantics in plainto).
    resp = await _client_get(
        "/usage/search?q=foo & bar | baz",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    await init_pool(settings)
    assert resp.status_code == 200, resp.text
    # 'foo bar baz' AND'd against neither summary → 0 hits.
    assert resp.json()["count"] == 0


@pytest.mark.asyncio
async def test_search_rls_isolation(live_db, settings) -> None:
    api_key_a = await _seed_customer("cust-srch-A")
    await _seed_customer("cust-srch-B")
    await _insert_event("cust-srch-A", summary="apricot")
    await _insert_event("cust-srch-B", summary="apricot")

    resp = await _client_get(
        "/usage/search?q=apricot",
        headers={"Authorization": f"Bearer {api_key_a}"},
    )
    await init_pool(settings)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["count"] == 1


# ---------------------------------------------------------------------------
# Auth gating sanity
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_endpoints_require_auth(live_db, settings) -> None:
    """All three endpoints reject unauthenticated requests."""
    for path in ("/usage/feed", "/usage/stats", "/usage/search?q=x"):
        resp = await _client_get(path, headers={})
        assert resp.status_code == 401, f"{path}: {resp.text}"
    await init_pool(settings)
