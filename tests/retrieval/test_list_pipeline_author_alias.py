"""Test author filter expansion: a list-mode query with author_ids=[ALIAS]
must match documents whose author_id is PRIMARY (or any other alias).

documents.author_id is never rewritten on merge — Phase 2 expands at
filter time via entity_aliases.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

from services.retrieval.list_pipeline import run_list
from services.retrieval.router import RouterEntity, RouterOutput
from shared.config import Settings, get_settings
from shared.db import raw_conn
from shared.embeddings import reset_embedder
from shared.models import QueryRequest, TemporalSpec
from shared.storage import reset_store

pytestmark = pytest.mark.asyncio


CUSTOMER_ID = "list-author-alias-cust"
PRIMARY = "richardwei6"
ALIAS = "mahit@prbe.ai"


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


async def _seed_doc_with_author(
    customer_id: str, *, doc_id: str, author_id: str, title: str
) -> None:
    now = datetime(2026, 4, 28, tzinfo=UTC)
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO documents (
                doc_id, version, customer_id,
                source_system, source_id, source_url,
                doc_class, doc_type, content_type,
                content_hash, title, body_size_bytes, body_token_count,
                created_at, updated_at, valid_from, ingested_at, acl,
                author_id
            ) VALUES (
                $1, 1, $2,
                'github', $3, 'https://example/' || $1,
                'raw_source', 'github.commit', 'text/plain',
                'h-' || $1, $4, 100, 0,
                $5, $5, $5, $5, '{}'::jsonb,
                $6
            )
            """,
            doc_id, customer_id, f"commit:{doc_id}", title, now, author_id,
        )
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
            f"{doc_id}:c0", doc_id, customer_id,
            f"body {doc_id}", f"chash-{doc_id}",
        )


async def _seed_cluster(customer_id: str) -> None:
    merge_id = str(uuid.uuid4())
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO entity_merge_audit (
                merge_id, customer_id, label, primary_canonical_id,
                merged_alias_canonical_ids, performed_by_user_id, status
            ) VALUES ($1, $2, 'Person', $3, ARRAY[$4]::text[],
                      '11111111-1111-1111-1111-111111111111', 'active')
            """,
            merge_id, customer_id, PRIMARY, ALIAS,
        )
        await conn.execute(
            """
            INSERT INTO entity_aliases (
                customer_id, label, alias_canonical_id,
                primary_canonical_id, merge_id
            ) VALUES ($1, 'Person', $2, $3, $4)
            """,
            customer_id, ALIAS, PRIMARY, merge_id,
        )


def _routed_with_person(canonical_id: str) -> RouterOutput:
    """Build a minimal RouterOutput with one Person entity."""
    return RouterOutput(
        operation="list",
        mode="list",
        entities=[
            RouterEntity(
                entity_type="person",
                canonical_id=canonical_id,
                display_name=canonical_id,
                confidence=0.9,
            )
        ],
    )


async def test_alias_author_expands_to_cluster(live_db):
    """Asking for documents authored by ALIAS returns docs authored
    by PRIMARY too (after the cluster expansion).
    """
    await _seed_customer(CUSTOMER_ID)
    await _seed_doc_with_author(CUSTOMER_ID, doc_id="doc-1", author_id=PRIMARY, title="primary doc")
    await _seed_doc_with_author(CUSTOMER_ID, doc_id="doc-2", author_id="someone-else", title="unrelated")
    await _seed_cluster(CUSTOMER_ID)

    req = QueryRequest(query="anything", top_k=10, entity_must_match=True)
    spec = TemporalSpec()
    routed = _routed_with_person(ALIAS)

    response = await run_list(
        req=req,
        customer_id=CUSTOMER_ID,
        routed=routed,
        spec=spec,
        temporal_meta={},
        sort_meta=None,
        extracted_entities=[{"canonical_id": ALIAS, "type": "person"}],
        doc_types=None,
        trace_id="t-1",
        timing={},
    )

    doc_ids = {r.doc_id for r in response.results}
    assert "doc-1" in doc_ids, "Alias query should match primary-authored doc post-expansion"
    assert "doc-2" not in doc_ids


async def test_unmerged_author_passes_through(live_db):
    """Asking for documents by an unmerged author_id behaves as before."""
    await _seed_customer(CUSTOMER_ID)
    await _seed_doc_with_author(CUSTOMER_ID, doc_id="doc-3", author_id="loner-id", title="loner doc")

    req = QueryRequest(query="anything", top_k=10, entity_must_match=True)
    spec = TemporalSpec()
    routed = _routed_with_person("loner-id")

    response = await run_list(
        req=req,
        customer_id=CUSTOMER_ID,
        routed=routed,
        spec=spec,
        temporal_meta={},
        sort_meta=None,
        extracted_entities=[{"canonical_id": "loner-id", "type": "person"}],
        doc_types=None,
        trace_id="t-2",
        timing={},
    )
    doc_ids = {r.doc_id for r in response.results}
    assert doc_ids == {"doc-3"}
