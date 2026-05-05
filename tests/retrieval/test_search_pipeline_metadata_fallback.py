from __future__ import annotations

from datetime import UTC, datetime

import pytest

from services.retrieval.retrievers.bm25 import BM25Hit
from services.retrieval.search_pipeline import _content_fallbacks_for_metadata_only_agent_hits
from shared.db import raw_conn
from shared.models import TemporalSpec

pytestmark = pytest.mark.asyncio


async def test_content_fallbacks_for_metadata_only_codex_hits(live_db) -> None:
    customer_id = "cust-metadata-fallback"
    doc_id = "codex:cust-metadata-fallback:session-1"
    now = datetime(2026, 5, 5, tzinfo=UTC)

    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO customers (customer_id, display_name, api_key_hash)
            VALUES ($1, 'metadata fallback test', 'hash')
            ON CONFLICT (customer_id) DO NOTHING
            """,
            customer_id,
        )
        await conn.execute(
            """
            INSERT INTO documents (
                doc_id, version, customer_id,
                source_system, source_id, source_url,
                doc_class, doc_type, content_type,
                content_hash, title, body_size_bytes, body_token_count,
                author_id,
                created_at, updated_at, valid_from, ingested_at,
                acl
            ) VALUES (
                $1, 1, $2,
                'codex', 'session-1', 'https://example/session-1',
                'raw_source', 'claude_code.session', 'text/plain',
                'doc-hash', $3, 24, 4,
                'user-1',
                $4, $4, $4, $4,
                '{}'::jsonb
            )
            """,
            doc_id,
            customer_id,
            "Richard Wei's Codex session",
            now,
        )
        await conn.execute(
            """
            INSERT INTO chunks (
                chunk_id, doc_id, customer_id,
                chunk_index, content, content_hash, token_count,
                embedding, first_seen_version, last_seen_version, kind
            ) VALUES (
                $1, $2, $3,
                0, 'real transcript content', 'content-hash', 3,
                array_fill(0::real, ARRAY[3072])::halfvec,
                1, 1, 'content'
            )
            """,
            f"{doc_id}:c0",
            doc_id,
            customer_id,
        )

    metadata_hit = BM25Hit(
        chunk_id=f"{doc_id}:m1",
        doc_id=doc_id,
        doc_version=1,
        source_system="codex",
        source_url="https://example/session-1",
        title="Richard Wei's Codex session",
        content="title: Richard Wei's Codex session\nsource: codex",
        created_at=now,
        updated_at=now,
        score=1.0,
        author_id="user-1",
        kind="metadata",
    )

    fallbacks = await _content_fallbacks_for_metadata_only_agent_hits(
        customer_id,
        {"bm25": [metadata_hit], "vector": [], "graph": []},
        TemporalSpec(),
    )

    assert len(fallbacks) == 1
    assert fallbacks[0].doc_id == doc_id
    assert fallbacks[0].kind == "content_fallback"
    assert fallbacks[0].content == "real transcript content"
