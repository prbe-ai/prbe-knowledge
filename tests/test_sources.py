"""GET /sources/{doc_id} — full source reassembly.

Covers:
  - Bearer required (401 without)
  - Live version returned by default; chunks reassembled in order
  - 404 for unknown doc_id
  - 404 for cross-tenant doc_id (RLS path)
  - Specific version via ?version=
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import UTC, datetime

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
            "test",
            api_key_hash,
        )
    return api_key


async def _seed_doc_with_chunks(
    customer_id: str,
    doc_id: str,
    *,
    chunks: list[str],
    version: int = 1,
    title: str | None = "Test Doc",
) -> None:
    """Insert a single-version document and N ordered chunks."""
    now = datetime.now(UTC)
    body_size = sum(len(c.encode()) for c in chunks)
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO documents (
                doc_id, version, customer_id,
                source_system, source_id, source_url,
                doc_class, doc_type, content_type,
                content_hash, title, body_size_bytes, body_token_count,
                created_at, updated_at, valid_from, ingested_at,
                acl
            ) VALUES (
                $1, $2, $3,
                'slack', 'msg-1', 'https://example.slack.com/archives/C/p1',
                'raw_source', 'slack_message', 'text/plain',
                'hash', $4, $5, 0,
                $6, $6, $6, $6,
                '{}'::jsonb
            )
            """,
            doc_id,
            version,
            customer_id,
            title,
            body_size,
            now,
        )
        for idx, content in enumerate(chunks):
            await conn.execute(
                """
                INSERT INTO chunks (
                    chunk_id, doc_id, customer_id,
                    chunk_index, content, content_hash, token_count,
                    embedding, first_seen_version, last_seen_version
                ) VALUES (
                    $1, $2, $3,
                    $4, $5, $6, 5,
                    array_fill(0::real, ARRAY[3072])::halfvec,
                    $7, $7
                )
                """,
                # Include version in chunk_id so seeding multiple versions of
                # the same doc in tests doesn't collide on chunks_pkey.
                f"{doc_id}:c{idx}:v{version}",
                doc_id,
                customer_id,
                idx,
                content,
                f"hash-{idx}-v{version}",
                version,
            )


async def _get_source(
    doc_id: str,
    headers: dict[str, str] | None = None,
    *,
    query: str = "",
) -> httpx.Response:
    from services.retrieval.main import app as retrieval_app

    await close_pool()
    transport = ASGITransport(app=retrieval_app)
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://t") as client,
        retrieval_app.router.lifespan_context(retrieval_app),
    ):
        # FastAPI's :path converter handles colons inside doc_id.
        url = f"/sources/{doc_id}{query}"
        resp = await client.get(url, headers=headers or {})
    return resp


@pytest.mark.asyncio
async def test_sources_requires_bearer(live_db, settings) -> None:
    resp = await _get_source("anything")
    await init_pool(settings)
    assert resp.status_code == 401, resp.text


@pytest.mark.asyncio
async def test_sources_returns_full_reassembled_content(live_db, settings) -> None:
    api_key = await _seed_customer("cust-src")
    await _seed_doc_with_chunks(
        "cust-src",
        "slack:T1:C1:1234.5678",
        chunks=["first chunk text", "second chunk text", "third chunk text"],
        title="Big Slack Thread",
    )

    resp = await _get_source(
        "slack:T1:C1:1234.5678",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    await init_pool(settings)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["doc_id"] == "slack:T1:C1:1234.5678"
    assert body["doc_version"] == 1
    assert body["title"] == "Big Slack Thread"
    assert body["chunk_count"] == 3
    # Chunks concatenated in chunk_index order with double-newline separator.
    assert body["content"] == "first chunk text\n\nsecond chunk text\n\nthird chunk text"
    assert body["source_system"] == "slack"
    assert "created_at" in body
    assert "updated_at" in body


@pytest.mark.asyncio
async def test_sources_404_for_unknown_doc(live_db, settings) -> None:
    api_key = await _seed_customer("cust-404")
    resp = await _get_source(
        "slack:NOPE:NOPE:0",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    await init_pool(settings)
    assert resp.status_code == 404, resp.text


@pytest.mark.asyncio
async def test_sources_404_for_cross_tenant_doc(live_db, settings) -> None:
    """Tenant A asks for a doc that belongs to tenant B → 404 (not 403),
    we don't leak the doc's existence."""
    api_key_a = await _seed_customer("tenant-a")
    await _seed_customer("tenant-b")
    await _seed_doc_with_chunks(
        "tenant-b",
        "slack:T2:C2:9999.0001",
        chunks=["secret tenant b content"],
    )

    resp = await _get_source(
        "slack:T2:C2:9999.0001",
        headers={"Authorization": f"Bearer {api_key_a}"},
    )
    await init_pool(settings)
    assert resp.status_code == 404, resp.text


@pytest.mark.asyncio
async def test_sources_specific_version(live_db, settings) -> None:
    api_key = await _seed_customer("cust-ver")
    # Seed two versions of the same doc.
    await _seed_doc_with_chunks(
        "cust-ver",
        "slack:T:C:1.1",
        chunks=["v1 content"],
        version=1,
    )
    # Mark v1 as superseded so v2 is the live one.
    async with raw_conn() as conn:
        await conn.execute(
            "UPDATE documents SET valid_to = NOW() WHERE customer_id='cust-ver' AND version=1"
        )
    await _seed_doc_with_chunks(
        "cust-ver",
        "slack:T:C:1.1",
        chunks=["v2 different content"],
        version=2,
    )

    # Default = live (v2).
    resp_live = await _get_source(
        "slack:T:C:1.1",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert resp_live.status_code == 200, resp_live.text
    assert resp_live.json()["doc_version"] == 2

    # Explicit v1.
    resp_v1 = await _get_source(
        "slack:T:C:1.1",
        headers={"Authorization": f"Bearer {api_key}"},
        query="?version=1",
    )
    await init_pool(settings)
    assert resp_v1.status_code == 200, resp_v1.text
    body = resp_v1.json()
    assert body["doc_version"] == 1
    assert body["content"] == "v1 content"
