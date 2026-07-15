"""Integration test: list pipeline's representative chunk is always
kind='content', never the synthetic metadata chunk.

Seeds a doc with both content and metadata chunks and confirms `sql_list`
returns the content chunk's text, not the metadata key:value text.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from engine.retrieval.retrievers.sql import sql_list
from engine.shared.config import Settings, get_settings
from engine.shared.db import raw_conn
from engine.shared.embeddings import reset_embedder
from engine.shared.storage import reset_store

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _patch_settings(monkeypatch, settings: Settings) -> None:
    monkeypatch.setenv("TOKEN_ENCRYPTION_KEY", settings.token_encryption_key.get_secret_value())
    monkeypatch.setenv("ENVIRONMENT", "local")
    reset_embedder()
    reset_store()
    get_settings.cache_clear()  # type: ignore[attr-defined]


async def _seed_customer(customer_id: str) -> None:
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO customers (customer_id, display_name, api_key_hash)
            VALUES ($1, 'test', 'h-' || $1)
            ON CONFLICT (customer_id) DO NOTHING
            """,
            customer_id,
        )


async def _seed_doc_with_both_chunks(
    customer_id: str,
    doc_id: str,
    *,
    content_text: str,
    metadata_text: str,
) -> None:
    """Seed one doc, one CONTENT chunk, and one METADATA chunk."""
    now = datetime(2026, 4, 28, tzinfo=UTC)
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
                $1, 1, $2,
                'github', $3, 'https://example/x',
                'raw_source', 'github.commit', 'text/plain',
                $4, 'commit title', 100, 0,
                $5, $5, $5, $5,
                '{}'::jsonb
            )
            """,
            doc_id,
            customer_id,
            doc_id + "-src",
            f"hash-{doc_id}",
            now,
        )
        # Content chunk (kind defaults to 'content').
        await conn.execute(
            """
            INSERT INTO chunks (
                chunk_id, doc_id, customer_id,
                chunk_index, content, content_hash, token_count,
                embedding, first_seen_version, last_seen_version
            ) VALUES (
                $1, $2, $3, 0, $4, $5, 5,
                array_fill(0::real, ARRAY[3072])::halfvec,
                1, 1
            )
            """,
            f"{doc_id}:c_content",
            doc_id,
            customer_id,
            content_text,
            f"hash-content-{doc_id}",
        )
        # Metadata chunk (kind='metadata', sentinel chunk_index = -1).
        await conn.execute(
            """
            INSERT INTO chunks (
                chunk_id, doc_id, customer_id,
                chunk_index, content, content_hash, token_count,
                embedding, first_seen_version, last_seen_version, kind
            ) VALUES (
                $1, $2, $3, -1, $4, $5, 5,
                array_fill(0::real, ARRAY[3072])::halfvec,
                1, 1, 'metadata'
            )
            """,
            f"{doc_id}:m_meta",
            doc_id,
            customer_id,
            metadata_text,
            f"hash-meta-{doc_id}",
        )


async def test_sql_list_returns_content_chunk_not_metadata(live_db) -> None:
    cust = "cust-list-kind"
    await _seed_customer(cust)
    await _seed_doc_with_both_chunks(
        cust,
        "doc:1",
        content_text="This is the actual commit body text.",
        metadata_text="title: x\nrepo: prbe-backend\nauthor: alice\n",
    )

    hits = await sql_list(cust, top_k=10, doc_types=["github.commit"])
    assert len(hits) == 1
    # The representative chunk MUST be the content one. Synthetic
    # key:value text never escapes through the list path.
    assert hits[0].content == "This is the actual commit body text."
    assert "title:" not in hits[0].content
    assert "repo:" not in hits[0].content
