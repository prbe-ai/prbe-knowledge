"""Bearer auth on POST /retrieve (retrieval service).

Covers:
  - 401 when Authorization header is missing in non-local env
  - Local bypass: no header + body customer_id → 200
  - Valid Bearer → 200 with customer_id derived from customers row
  - Invalid Bearer → 401
  - Header tenant ≠ body tenant → 400
"""

from __future__ import annotations

import hashlib
import secrets

import httpx
import pytest
from httpx import ASGITransport

from shared.config import Settings, get_settings
from shared.db import close_pool, init_pool, raw_conn
from shared.embeddings import reset_embedder
from shared.storage import reset_store


@pytest.fixture(autouse=True)
def _patch_settings(monkeypatch, settings: Settings) -> None:
    monkeypatch.setenv(
        "TOKEN_ENCRYPTION_KEY", settings.token_encryption_key.get_secret_value()
    )
    monkeypatch.setenv("ENVIRONMENT", "local")
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
    from services.retrieval.main import app as retrieval_app

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
async def test_query_requires_bearer_in_non_local(live_db, settings, monkeypatch) -> None:
    monkeypatch.setenv("ENVIRONMENT", "dev")
    get_settings.cache_clear()  # type: ignore[attr-defined]
    try:
        resp = await _post_query(body={"query": "hi", "top_k": 1})
        assert resp.status_code == 401, resp.text
        assert resp.headers.get("www-authenticate", "").lower() == "bearer"
    finally:
        await init_pool(settings)


@pytest.mark.asyncio
async def test_query_accepts_valid_bearer(live_db, settings) -> None:
    api_key = await _seed_customer("cust-auth-ok")
    resp = await _post_query(
        headers={"Authorization": f"Bearer {api_key}"},
        body={"query": "hello world", "top_k": 1},
    )
    await init_pool(settings)
    assert resp.status_code == 200, resp.text
    assert "chunks" in resp.json()


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
async def test_query_rejects_mismatched_body_customer_id(live_db, settings) -> None:
    api_key_a = await _seed_customer("tenant-a")
    await _seed_customer("tenant-b")
    resp = await _post_query(
        headers={"Authorization": f"Bearer {api_key_a}"},
        body={"query": "hi", "customer_id": "tenant-b", "top_k": 1},
    )
    await init_pool(settings)
    assert resp.status_code == 400, resp.text


@pytest.mark.asyncio
async def test_query_local_bypass_works(live_db, settings) -> None:
    await _seed_customer("cust-local-bypass")
    resp = await _post_query(
        body={"query": "hi", "customer_id": "cust-local-bypass", "top_k": 1}
    )
    await init_pool(settings)
    assert resp.status_code == 200, resp.text
