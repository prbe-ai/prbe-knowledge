"""Tests for the KG read API: GET /kg/classes/{id} and GET /kg/classes.

Covers the four contract guarantees from spec §5.2 / §12.3:

  1. ``test_get_class_returns_envelope`` — a stored class round-trips back as
     a ``BugClass``-shaped JSON envelope (`frontmatter` + `body`).
  2. ``test_get_unknown_class_404`` — missing class_id is a 404, not a 500.
  3. ``test_list_classes_returns_metadata_only`` — the list view returns
     id + description only, never the body. The body would blow up the
     payload (markdown can be many KB) and isn't useful for a list.
  4. ``test_cross_tenant_query_returns_nothing`` — the security-critical
     RLS check: tenant tA's class is invisible to tenant tB even when tB
     guesses the exact class_id. The 404 surfaces because RLS filters
     the row out before the WHERE clause sees it (USING-only policy on
     ``app.current_customer_id``).

All four require a live Postgres + the kg_classes table. The
``tests/kg/conftest.py:pytest_collection_modifyitems`` hook auto-skips
when Postgres isn't reachable, so this file is safe to ship in CI envs
that don't run docker-compose.

Self-cleans the kg_classes / customers rows it inserts (the parent
``live_db`` fixture's TRUNCATE_SQL excludes kg_* tables — see
``tests/kg/test_embedding_query.py`` for the same pattern).
"""

from __future__ import annotations

import hashlib
import secrets
from collections.abc import AsyncIterator

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport

from services.kg.api.read import router as read_router
from shared.config import Settings, get_settings
from shared.db import close_pool, init_pool, raw_conn

# Module-local internal key. Same pattern as tests/test_query_auth.py.
INTERNAL_KEY = "test-internal-knowledge-key"


@pytest.fixture(autouse=True)
def _patch_settings(monkeypatch, settings: Settings) -> None:
    """Set the internal-key env var so the auth dep accepts our test header."""
    monkeypatch.setenv("ENVIRONMENT", "local")
    monkeypatch.setenv("INTERNAL_KNOWLEDGE_API_KEY", INTERNAL_KEY)
    get_settings.cache_clear()


@pytest.fixture
def kg_app() -> FastAPI:
    """Tiny FastAPI app with just the KG read router mounted at /kg.

    Built per-test so the test-suite never carries app state between cases.
    No lifespan: the live_db fixture handles pool init/close.
    """
    app = FastAPI()
    app.include_router(read_router, prefix="/kg")
    return app


@pytest.fixture
def headers_for():
    """Return a callable that builds auth headers for a given customer_id.

    Uses the X-Internal-Knowledge-Key + X-Prbe-Customer path (same as
    service-to-service callers). No DB roundtrip needed — convenient
    for tests that just want to assert RLS filtering.
    """

    def _make(customer_id: str) -> dict[str, str]:
        return {
            "X-Internal-Knowledge-Key": INTERNAL_KEY,
            "X-Prbe-Customer": customer_id,
        }

    return _make


async def _seed_customer(customer_id: str) -> None:
    api_key_hash = hashlib.sha256(secrets.token_urlsafe(32).encode()).hexdigest()
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO customers (customer_id, display_name, api_key_hash, status)
            VALUES ($1, $2, $3, 'active')
            ON CONFLICT (customer_id) DO NOTHING
            """,
            customer_id,
            f"{customer_id} display",
            api_key_hash,
        )


async def _seed_class(
    *,
    customer_id: str,
    class_id: str,
    description: str = "401 from upstream after JWT refresh",
    body: str = "## When this fires\n401 ...",
) -> None:
    """Insert a minimal valid kg_classes row.

    Frontmatter shape mirrors the Pydantic ``Frontmatter`` model so the
    GET-by-id handler can parse it back into a ``BugClass``.
    """
    frontmatter = {
        "id": class_id,
        "type": "bug-class",
        "description": description,
        "signature": {
            "must_match": ["status_code == 401"],
            "embedding_seed": "jwt refresh expired clock-skew",
        },
        "related": {
            "analogous_to": [],
            "overlaps_with": [],
            "often_confused_with": [],
            "regressed_by": [],
        },
        "context_sources": [],
        "evidence": {
            "match_count": 0,
            "last_updated": None,
            "recent_refinements": [],
        },
    }
    async with raw_conn() as conn:
        # Set the GUC inline so RLS lets the INSERT through (RLS USING is
        # checked on writes too because FORCE is enabled and there's no
        # bypass role here).
        await conn.execute(
            "SELECT set_config('app.current_customer_id', $1, true)",
            customer_id,
        )
        await conn.execute(
            """
            INSERT INTO kg_classes (customer_id, class_id, frontmatter, body)
            VALUES ($1, $2, $3::jsonb, $4)
            """,
            customer_id,
            class_id,
            __import__("json").dumps(frontmatter),
            body,
        )


async def _cleanup(customer_ids: list[str]) -> None:
    """Remove rows we inserted; live_db's TRUNCATE_SQL doesn't cover kg_*."""
    async with raw_conn() as conn:
        for cid in customer_ids:
            await conn.execute(
                "DELETE FROM kg_classes WHERE customer_id = $1", cid
            )
            await conn.execute(
                "DELETE FROM customers WHERE customer_id = $1", cid
            )


@pytest_asyncio.fixture
async def seeded_classes(live_db: None) -> AsyncIterator[None]:
    """Seed two tenants with known classes; clean up afterward.

    tA gets two classes (so list-view tests have >1 row); tB gets none
    (so the cross-tenant test has a clean negative case).
    """
    customers = ["tA", "tB"]
    try:
        for cid in customers:
            await _seed_customer(cid)
        await _seed_class(customer_id="tA", class_id="auth-401-jwt-refresh")
        await _seed_class(
            customer_id="tA",
            class_id="db-timeout-replica-lag",
            description="DB timeout on replica lag",
            body="## When this fires\nreplica lag exceeded ...",
        )
        yield None
    finally:
        await _cleanup(customers)


async def _client_get(
    kg_app: FastAPI,
    path: str,
    headers: dict[str, str],
) -> httpx.Response:
    """One-shot GET via ASGITransport; mirrors tests/test_query_auth.py."""
    transport = ASGITransport(app=kg_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        return await client.get(path, headers=headers)


@pytest.mark.asyncio
async def test_get_class_returns_envelope(
    kg_app: FastAPI,
    seeded_classes: None,
    headers_for,
) -> None:
    resp = await _client_get(
        kg_app,
        "/kg/classes/auth-401-jwt-refresh",
        headers_for("tA"),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["frontmatter"]["id"] == "auth-401-jwt-refresh"
    assert body["frontmatter"]["type"] == "bug-class"
    assert "body" in body
    assert isinstance(body["body"], str)


@pytest.mark.asyncio
async def test_get_unknown_class_404(
    kg_app: FastAPI,
    seeded_classes: None,
    headers_for,
) -> None:
    resp = await _client_get(
        kg_app,
        "/kg/classes/does-not-exist",
        headers_for("tA"),
    )
    assert resp.status_code == 404, resp.text


@pytest.mark.asyncio
async def test_list_classes_returns_metadata_only(
    kg_app: FastAPI,
    seeded_classes: None,
    headers_for,
) -> None:
    resp = await _client_get(kg_app, "/kg/classes", headers_for("tA"))
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    items = payload["items"]
    # tA has two seeded classes, ordered by class_id ascending.
    assert len(items) == 2
    assert items[0]["id"] == "auth-401-jwt-refresh"
    assert items[1]["id"] == "db-timeout-replica-lag"
    assert all("description" in i and i["description"] for i in items)
    # Body must NOT leak into the list view.
    assert all("body" not in i for i in items)


@pytest.mark.asyncio
async def test_cross_tenant_query_returns_nothing(
    kg_app: FastAPI,
    seeded_classes: None,
    headers_for,
) -> None:
    """Security-critical: tA's class must be invisible to tB.

    RLS filters the row out of the SELECT before the WHERE clause sees
    it; the handler then surfaces "row not found" as a 404. If this ever
    returned 200 with tA's payload, that's a tenant-isolation breach.
    """
    resp = await _client_get(
        kg_app,
        "/kg/classes/auth-401-jwt-refresh",
        headers_for("tB"),
    )
    assert resp.status_code == 404, resp.text


# Auth tests don't need the DB, but they DO depend on the auth dep being
# wired correctly. Keep them here so the read-API contract is checked
# end-to-end in one file.


@pytest.mark.asyncio
async def test_missing_auth_returns_401(kg_app: FastAPI) -> None:
    """No headers → 401, not 200/404. Belt-and-suspenders for the auth dep."""
    # No DB needed; reset the pool reference so an env without Postgres
    # doesn't error out on pool acquisition before the auth check runs.
    await close_pool()
    transport = ASGITransport(app=kg_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await client.get("/kg/classes/anything")
    assert resp.status_code == 401, resp.text


@pytest.mark.asyncio
async def test_missing_auth_on_list_returns_401(kg_app: FastAPI) -> None:
    await close_pool()
    transport = ASGITransport(app=kg_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await client.get("/kg/classes")
    assert resp.status_code == 401, resp.text
