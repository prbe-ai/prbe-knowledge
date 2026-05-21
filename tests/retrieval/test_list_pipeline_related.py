"""Integration tests for `related_entities` wire-in on `run_list`.

Covers (per locked plan section 6):
- Doc-shaped list response: related_entities populated.
- Aggregation list response (count / group_by): related_entities is None,
  walk SKIPPED entirely (codex-B2 -- no result docs to walk from).
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from services.retrieval.list_pipeline import run_list
from services.retrieval.retrievers.bm25 import BM25Hit
from services.retrieval.router import Intent
from shared.config import Settings, get_settings
from shared.constants import EdgeType, NodeLabel
from shared.db import raw_conn
from shared.embeddings import reset_embedder
from shared.models import QueryRequest, TemporalSpec
from shared.storage import reset_store

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _patch_settings(monkeypatch, settings: Settings) -> None:
    monkeypatch.setenv("TOKEN_ENCRYPTION_KEY", settings.token_encryption_key.get_secret_value())
    monkeypatch.setenv("ENVIRONMENT", "local")
    reset_embedder()
    reset_store()
    get_settings.cache_clear()  # type: ignore[attr-defined]


# ---- helpers --------------------------------------------------------------


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


async def _seed_doc(customer_id: str, *, doc_id: str, title: str = "doc") -> None:
    """Seed documents row + chunk + Document graph_node. `source_id`
    follows the `<kind>:<uuid>` pattern per memory."""
    now = datetime(2026, 4, 28, tzinfo=UTC)
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO documents (
                doc_id, version, customer_id,
                source_system, source_id, source_url,
                doc_class, doc_type, content_type,
                content_hash, title, body_size_bytes, body_token_count,
                created_at, updated_at, valid_from, ingested_at, acl
            ) VALUES (
                $1, 1, $2,
                'github', $3, 'https://example/' || $1,
                'raw_source', 'github.commit', 'text/plain',
                'h-' || $1, $4, 100, 0,
                $5, $5, $5, $5, '{}'::jsonb
            )
            """,
            doc_id, customer_id, f"commit:{doc_id}", title, now,
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
            f"body of {doc_id}", f"chash-{doc_id}",
        )
        await conn.execute(
            """
            INSERT INTO graph_nodes (customer_id, label, canonical_id, properties)
            VALUES ($1, $2, $3, '{}'::jsonb)
            ON CONFLICT (customer_id, label, canonical_id) DO NOTHING
            """,
            customer_id, NodeLabel.DOCUMENT.value, doc_id,
        )


async def _seed_neighbor(
    customer_id: str, *, label: str, canonical_id: str, name: str | None = None,
) -> None:
    properties_json = "{}" if name is None else f'{{"name": "{name}"}}'
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO graph_nodes (customer_id, label, canonical_id, properties)
            VALUES ($1, $2, $3, $4::jsonb)
            ON CONFLICT (customer_id, label, canonical_id) DO NOTHING
            """,
            customer_id, label, canonical_id, properties_json,
        )


async def _seed_edge(
    customer_id: str,
    *,
    doc_id: str,
    label: str,
    canonical_id: str,
    edge_type: str = EdgeType.MENTIONS.value,
) -> None:
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO graph_edges (
                customer_id, edge_type, from_node_id, to_node_id,
                confidence, valid_from
            )
            SELECT $1, $2, f.node_id, t.node_id, 'EXTRACTED', NOW()
            FROM graph_nodes f, graph_nodes t
            WHERE f.customer_id = $1 AND f.label = $5 AND f.canonical_id = $3
              AND t.customer_id = $1 AND t.label = $6 AND t.canonical_id = $4
            ON CONFLICT DO NOTHING
            """,
            customer_id, edge_type, doc_id, canonical_id,
            NodeLabel.DOCUMENT.value, label,
        )


def _bm25_hit(doc_id: str) -> BM25Hit:
    """sql_list returns BM25Hit-shaped objects -- mirror the BM25Hit row shape."""
    now = datetime(2026, 4, 28, tzinfo=UTC)
    return BM25Hit(
        chunk_id=f"{doc_id}:c0",
        doc_id=doc_id,
        doc_version=1,
        source_system="github",
        source_url=f"https://example/{doc_id}",
        title=doc_id,
        content=f"body of {doc_id}",
        created_at=now,
        updated_at=now,
        score=1.0,
        kind="content",
    )


# ---- tests ---------------------------------------------------------------


async def test_list_response_populates_related_entities(live_db) -> None:
    """list-mode (operation='list') -> related_entities walked + populated."""
    cust = "cust-list-related"
    await _seed_customer(cust)
    await _seed_doc(cust, doc_id="doc:1", title="d1")
    await _seed_neighbor(
        cust, label=NodeLabel.DOCUMENT.value, canonical_id="r", name="r",
    )
    await _seed_edge(cust, doc_id="doc:1", label=NodeLabel.DOCUMENT.value, canonical_id="r")

    intent = Intent(query_text="x", mode="list", confidence=0.9, operation="list")
    req = QueryRequest(query="x", top_k=5, top_k_related=10)

    with patch(
        "services.retrieval.list_pipeline.sql_list",
        new=AsyncMock(return_value=[_bm25_hit("doc:1")]),
    ):
        resp = await run_list(
            req=req,
            customer_id=cust,
            intent=intent,
            intent_idx=0,
            spec=TemporalSpec(),
            temporal_meta={},
            sort_meta=None,
            extracted_entities=[],
            doc_types=None,
            trace_id="t-1",
            timing={},
        )

    assert resp.aggregation is None
    assert resp.related_entities is not None
    assert resp.related_entities_error is None
    cids = {e.canonical_id for e in resp.related_entities}
    assert "r" in cids


async def test_count_aggregation_skips_walk(live_db) -> None:
    """operation='count' -> aggregation populated, related_entities is None,
    walk_result_doc_neighbors NOT called (codex-B2)."""
    cust = "cust-list-count"
    await _seed_customer(cust)

    intent = Intent(query_text="x", mode="list", confidence=0.9, operation="count")
    req = QueryRequest(query="x", top_k=5, top_k_related=10)

    with patch(
        "services.retrieval.list_pipeline.sql_count",
        new=AsyncMock(return_value=42),
    ), patch(
        "services.retrieval.list_pipeline.walk_result_doc_neighbors",
        new=AsyncMock(return_value=[]),
    ) as m_walk:
        resp = await run_list(
            req=req,
            customer_id=cust,
            intent=intent,
            intent_idx=0,
            spec=TemporalSpec(),
            temporal_meta={},
            sort_meta=None,
            extracted_entities=[],
            doc_types=None,
            trace_id="t-1",
            timing={},
        )

    assert resp.aggregation == {"count": 42}
    assert resp.related_entities is None
    assert resp.related_entities_error is None
    m_walk.assert_not_called()


async def test_group_by_aggregation_skips_walk(live_db) -> None:
    """operation='group_by' -> aggregation rows, related_entities is None,
    walk skipped (codex-B2)."""
    cust = "cust-list-groupby"
    await _seed_customer(cust)

    intent = Intent(
        query_text="x",
        mode="list",
        confidence=0.9,
        operation="group_by",
        group_by_key="source_system",
    )
    req = QueryRequest(query="x", top_k=5, top_k_related=10)

    with patch(
        "services.retrieval.list_pipeline.sql_group_by",
        new=AsyncMock(return_value=[{"source_system": "github", "count": 7}]),
    ), patch(
        "services.retrieval.list_pipeline.walk_result_doc_neighbors",
        new=AsyncMock(return_value=[]),
    ) as m_walk:
        resp = await run_list(
            req=req,
            customer_id=cust,
            intent=intent,
            intent_idx=0,
            spec=TemporalSpec(),
            temporal_meta={},
            sort_meta=None,
            extracted_entities=[],
            doc_types=None,
            trace_id="t-1",
            timing={},
        )

    assert resp.aggregation is not None
    assert resp.aggregation["key"] == "source_system"
    assert resp.related_entities is None
    assert resp.related_entities_error is None
    m_walk.assert_not_called()


async def test_list_top_k_related_zero_skips_walk(live_db) -> None:
    """top_k_related=0 on a doc-shaped list response also skips the walk."""
    cust = "cust-list-skip"
    await _seed_customer(cust)
    await _seed_doc(cust, doc_id="doc:1", title="d1")

    intent = Intent(query_text="x", mode="list", confidence=0.9, operation="list")
    req = QueryRequest(query="x", top_k=5, top_k_related=0)

    with patch(
        "services.retrieval.list_pipeline.sql_list",
        new=AsyncMock(return_value=[_bm25_hit("doc:1")]),
    ), patch(
        "services.retrieval.list_pipeline.walk_result_doc_neighbors",
        new=AsyncMock(return_value=[]),
    ) as m_walk:
        resp = await run_list(
            req=req,
            customer_id=cust,
            intent=intent,
            intent_idx=0,
            spec=TemporalSpec(),
            temporal_meta={},
            sort_meta=None,
            extracted_entities=[],
            doc_types=None,
            trace_id="t-1",
            timing={},
        )

    assert resp.aggregation is None
    assert resp.related_entities is None
    m_walk.assert_not_called()
