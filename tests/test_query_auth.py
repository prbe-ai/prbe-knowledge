"""Auth on POST /retrieve (retrieval service).

Covers:
  - 401 when no auth headers are present (in any environment)
  - Valid Bearer → 200 with customer_id derived from customers row
  - Invalid Bearer → 401
  - X-Internal-Knowledge-Key + X-Prbe-Customer → 200 (service-to-service path,
    works in local dev too)
  - Body `customer_id` field is silently ignored — schema no longer accepts it
"""

from __future__ import annotations

import hashlib
import secrets

import httpx
import pytest
from httpx import ASGITransport

from engine.shared.config import Settings, get_settings
from engine.shared.db import close_pool, init_pool, raw_conn
from engine.shared.embeddings import reset_embedder
from engine.shared.storage import reset_store

# Test internal-knowledge key. Set via monkeypatch so each test gets a clean
# settings instance (the autouse fixture clears the lru_cache).
INTERNAL_KEY = "test-internal-knowledge-key"


@pytest.fixture(autouse=True)
def _patch_settings(monkeypatch, settings: Settings) -> None:
    monkeypatch.setenv(
        "TOKEN_ENCRYPTION_KEY", settings.token_encryption_key.get_secret_value()
    )
    monkeypatch.setenv("ENVIRONMENT", "local")
    monkeypatch.setenv("INTERNAL_KNOWLEDGE_API_KEY", INTERNAL_KEY)
    reset_embedder()
    reset_store()
    get_settings.cache_clear()  # type: ignore[attr-defined]


async def _seed_customer(customer_id: str) -> str:
    """Insert a customer row and return the plaintext api_key."""
    api_key = secrets.token_urlsafe(32)
    api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO customers (customer_id, display_name, api_key_hash)
            VALUES ($1, $2, $3)
            ON CONFLICT (customer_id) DO UPDATE
            SET api_key_hash = EXCLUDED.api_key_hash
            """,
            customer_id,
            f"{customer_id} display",
            api_key_hash,
        )
    return api_key


async def _post_query(
    headers: dict[str, str] | None = None,
    body: dict | None = None,
) -> httpx.Response:
    from engine.retrieval.main import app as retrieval_app

    await close_pool()
    transport = ASGITransport(app=retrieval_app)
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://t") as client,
        retrieval_app.router.lifespan_context(retrieval_app),
    ):
        resp = await client.post(
            "/retrieve",
            json=body or {"query": "anything", "top_k": 1},
            headers=headers or {},
        )
    return resp


@pytest.mark.asyncio
async def test_query_requires_auth_in_non_local(live_db, settings, monkeypatch) -> None:
    monkeypatch.setenv("ENVIRONMENT", "dev")
    get_settings.cache_clear()  # type: ignore[attr-defined]
    try:
        resp = await _post_query(body={"query": "hi", "top_k": 1})
        assert resp.status_code == 401, resp.text
        assert resp.headers.get("www-authenticate", "").lower() == "bearer"
    finally:
        await init_pool(settings)


@pytest.mark.asyncio
async def test_query_requires_auth_in_local(live_db, settings) -> None:
    """No body-fallback in local: missing headers → 401, same as prod."""
    resp = await _post_query(body={"query": "hi", "top_k": 1})
    await init_pool(settings)
    assert resp.status_code == 401, resp.text
    assert resp.headers.get("www-authenticate", "").lower() == "bearer"


@pytest.mark.asyncio
async def test_query_accepts_valid_bearer(live_db, settings) -> None:
    api_key = await _seed_customer("cust-auth-ok")
    resp = await _post_query(
        headers={"Authorization": f"Bearer {api_key}"},
        body={"query": "hello world", "top_k": 1},
    )
    await init_pool(settings)
    assert resp.status_code == 200, resp.text
    assert "results" in resp.json()


@pytest.mark.asyncio
async def test_query_rejects_invalid_bearer(live_db, settings) -> None:
    await _seed_customer("cust-auth-bad")
    resp = await _post_query(
        headers={"Authorization": "Bearer not-a-real-key"},
        body={"query": "hi", "top_k": 1},
    )
    await init_pool(settings)
    assert resp.status_code == 401, resp.text


@pytest.mark.asyncio
async def test_query_accepts_internal_key_with_customer_header(live_db, settings) -> None:
    """Service-to-service path: shared internal key + X-Prbe-Customer header.

    Used by prbe-orchestrator and prbe-knowledge-mcp; also the way local-dev
    callers authenticate now that the body bypass is gone.
    """
    await _seed_customer("cust-internal-ok")
    resp = await _post_query(
        headers={
            "X-Internal-Knowledge-Key": INTERNAL_KEY,
            "X-Prbe-Customer": "cust-internal-ok",
        },
        body={"query": "hello world", "top_k": 1},
    )
    await init_pool(settings)
    assert resp.status_code == 200, resp.text


@pytest.mark.asyncio
async def test_query_body_customer_id_is_silently_ignored(live_db, settings) -> None:
    """`customer_id` is no longer part of the request schema. Pydantic's
    default `extra=ignore` means callers passing it as a vestigial field
    aren't rejected — but the value has no effect. Tenant comes from auth."""
    api_key = await _seed_customer("cust-real-tenant")
    await _seed_customer("cust-other-tenant")
    resp = await _post_query(
        headers={"Authorization": f"Bearer {api_key}"},
        # customer_id in the body would have routed to the wrong tenant under
        # the old code path. Under the new code it's dropped on parse.
        body={"query": "hi", "customer_id": "cust-other-tenant", "top_k": 1},
    )
    await init_pool(settings)
    assert resp.status_code == 200, resp.text
