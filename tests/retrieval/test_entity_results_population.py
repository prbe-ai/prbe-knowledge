"""Integration tests for QueryEntityResult population in `run_search`
(PR feat/polymorphic-search-results).

The pipeline turns each routed entity that resolves to a graph_node into
one QueryEntityResult. The result carries:
- canonical_id / label / display_name (from properties->>'name')
- attached_doc_ids: 1-hop Documents, capped at 5, ordered by recency
- doc_count: TOTAL 1-hop Documents (uncapped)
- edge_types: distinct edge types observed on the 1-hop neighborhood
- properties: the full graph_node properties dict

Cases covered:
1. Routed entity -> QueryEntityResult with display_name from props.name.
2. attached_doc_ids capped at 5; doc_count is total (uncapped).
3. Two routed entities -> two QueryEntityResults (no cross-pollination).
4. Unmapped entity_type (no NodeLabel) -> dropped silently.
5. 'session' entity_type -> dropped (handled via id_lookup as Document).
6. Non-existent graph_node for a routed entity -> dropped silently.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from services.retrieval.router import Intent, RouterEntity
from services.retrieval.search_pipeline import run_search
from shared.config import Settings, get_settings
from shared.constants import NodeLabel
from shared.db import raw_conn
from shared.embeddings import reset_embedder
from shared.models import (
    QueryEntityResult,
    QueryRequest,
    TemporalSpec,
)
from shared.storage import reset_store

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _patch_settings(monkeypatch, settings: Settings) -> None:
    monkeypatch.setenv(
        "TOKEN_ENCRYPTION_KEY", settings.token_encryption_key.get_secret_value()
    )
    monkeypatch.setenv("ENVIRONMENT", "local")
    reset_embedder()
    reset_store()
    get_settings.cache_clear()  # type: ignore[attr-defined]


# ---- seed helpers ---------------------------------------------------------


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


async def _seed_doc(
    customer_id: str,
    *,
    doc_id: str,
    updated_at: datetime,
) -> None:
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
                'h-' || $1, $1, 100, 0,
                $4, $4, $4, $4, '{}'::jsonb
            )
            """,
            doc_id, customer_id, f"commit:{doc_id}", updated_at,
        )
        await conn.execute(
            """
            INSERT INTO graph_nodes (customer_id, label, canonical_id, properties)
            VALUES ($1, $2, $3, '{}'::jsonb)
            ON CONFLICT (customer_id, label, canonical_id) DO NOTHING
            """,
            customer_id, NodeLabel.DOCUMENT.value, doc_id,
        )


async def _seed_entity_node(
    customer_id: str,
    *,
    label: str,
    canonical_id: str,
    properties_json: str = "{}",
) -> None:
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO graph_nodes (customer_id, label, canonical_id, properties)
            VALUES ($1, $2, $3, $4::jsonb)
            ON CONFLICT (customer_id, label, canonical_id) DO NOTHING
            """,
            customer_id, label, canonical_id, properties_json,
        )


async def _seed_doc_to_entity_edge(
    customer_id: str,
    *,
    doc_id: str,
    label: str,
    canonical_id: str,
    edge_type: str = "MENTIONS",
) -> None:
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO graph_edges (
                customer_id, edge_type, from_node_id, to_node_id, valid_from
            )
            SELECT $1, $2, f.node_id, t.node_id, NOW()
            FROM graph_nodes f, graph_nodes t
            WHERE f.customer_id = $1 AND f.label = 'Document' AND f.canonical_id = $3
              AND t.customer_id = $1 AND t.label = $4 AND t.canonical_id = $5
            ON CONFLICT DO NOTHING
            """,
            customer_id, edge_type, doc_id, label, canonical_id,
        )


def _patch_retrievers():
    """All four primary channels return empty -- entity result construction
    happens independently of fused hits, so this isolates the entity path.
    """
    return [
        patch(
            "services.retrieval.search_pipeline.vector_search",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "services.retrieval.search_pipeline.bm25_search",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "services.retrieval.search_pipeline.graph_search",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "services.retrieval.search_pipeline.id_lookup_search",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "services.retrieval.search_pipeline.embeddings_for_chunks",
            new=AsyncMock(return_value={}),
        ),
        patch(
            "services.retrieval.search_pipeline.filter_by_acl",
            new=AsyncMock(side_effect=lambda _c, _u, hits: hits),
        ),
    ]


async def _run(cust: str, intent: Intent, top_k: int = 5):
    req = QueryRequest(query="x", top_k=top_k, top_k_related=0)
    patches = _patch_retrievers()
    with (
        patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]
    ):
        return await run_search(
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


# ---- tests ---------------------------------------------------------------


async def test_routed_entity_becomes_query_entity_result(live_db) -> None:
    """A routed Service entity with a graph_node -> one QueryEntityResult
    with display_name pulled from properties->>'name'."""
    cust = "cust-entity-basic"
    await _seed_customer(cust)
    await _seed_entity_node(
        cust, label=NodeLabel.SERVICE.value,
        canonical_id="prbe-backend",
        properties_json='{"name": "prbe-backend", "team": "platform"}',
    )
    intent = Intent(
        query_text="x",
        mode="search",
        confidence=0.9,
        entities=[
            RouterEntity(
                entity_type="service",
                canonical_id="prbe-backend",
                display_name="prbe-backend",
                confidence=0.9,
            )
        ],
    )
    resp = await _run(cust, intent)
    entities = [r for r in resp.results if isinstance(r, QueryEntityResult)]
    assert len(entities) == 1
    e = entities[0]
    assert e.canonical_id == "prbe-backend"
    assert e.label == NodeLabel.SERVICE.value
    assert e.display_name == "prbe-backend"
    assert e.properties.get("team") == "platform"
    # No attached docs -> doc_count 0 + empty pool.
    assert e.doc_count == 0
    assert e.attached_doc_ids == []


async def test_attached_docs_capped_at_five_doc_count_uncapped(live_db) -> None:
    """8 docs attached to the entity -> attached_doc_ids capped at 5;
    doc_count = 8."""
    cust = "cust-entity-cap"
    await _seed_customer(cust)
    await _seed_entity_node(
        cust, label=NodeLabel.REPO.value, canonical_id="big-repo",
        properties_json='{"name": "big-repo"}',
    )
    base = datetime(2026, 4, 1, tzinfo=UTC)
    for i in range(8):
        await _seed_doc(cust, doc_id=f"doc:{i}", updated_at=base + timedelta(days=i))
        await _seed_doc_to_entity_edge(
            cust, doc_id=f"doc:{i}",
            label=NodeLabel.REPO.value, canonical_id="big-repo",
        )
    intent = Intent(
        query_text="x",
        mode="search",
        confidence=0.9,
        entities=[
            RouterEntity(
                entity_type="repo", canonical_id="big-repo",
                display_name="big-repo", confidence=0.95,
            )
        ],
    )
    resp = await _run(cust, intent)
    entities = [r for r in resp.results if isinstance(r, QueryEntityResult)]
    assert len(entities) == 1
    e = entities[0]
    assert e.doc_count == 8
    assert len(e.attached_doc_ids) == 5  # capped
    # The 5 most recent (highest updated_at) are doc:7 .. doc:3.
    assert e.attached_doc_ids[0] == "doc:7"


async def test_two_routed_entities_emit_two_results(live_db) -> None:
    """Two routed entities both resolve -> two QueryEntityResults, no
    cross-pollination of edge_types or attached docs."""
    cust = "cust-entity-pair"
    await _seed_customer(cust)
    await _seed_entity_node(
        cust, label=NodeLabel.SERVICE.value, canonical_id="svc-1",
        properties_json='{"name": "svc-1"}',
    )
    await _seed_entity_node(
        cust, label=NodeLabel.REPO.value, canonical_id="repo-1",
        properties_json='{"name": "repo-1"}',
    )
    base = datetime(2026, 4, 28, tzinfo=UTC)
    await _seed_doc(cust, doc_id="doc:s1", updated_at=base)
    await _seed_doc(cust, doc_id="doc:r1", updated_at=base)
    await _seed_doc_to_entity_edge(
        cust, doc_id="doc:s1", label=NodeLabel.SERVICE.value,
        canonical_id="svc-1", edge_type="OWNS",
    )
    await _seed_doc_to_entity_edge(
        cust, doc_id="doc:r1", label=NodeLabel.REPO.value,
        canonical_id="repo-1", edge_type="TOUCHES",
    )

    intent = Intent(
        query_text="x",
        mode="search",
        confidence=0.9,
        entities=[
            RouterEntity(
                entity_type="service", canonical_id="svc-1",
                display_name="svc-1", confidence=0.9,
            ),
            RouterEntity(
                entity_type="repo", canonical_id="repo-1",
                display_name="repo-1", confidence=0.9,
            ),
        ],
    )
    resp = await _run(cust, intent)
    entities = {
        e.canonical_id: e
        for e in resp.results
        if isinstance(e, QueryEntityResult)
    }
    assert "svc-1" in entities
    assert "repo-1" in entities
    assert entities["svc-1"].label == NodeLabel.SERVICE.value
    assert entities["repo-1"].label == NodeLabel.REPO.value
    # No bleed: svc-1's docs are svc-1's; repo-1's docs are repo-1's.
    assert entities["svc-1"].attached_doc_ids == ["doc:s1"]
    assert entities["repo-1"].attached_doc_ids == ["doc:r1"]
    assert entities["svc-1"].edge_types == ["OWNS"]
    assert entities["repo-1"].edge_types == ["TOUCHES"]


async def test_unmapped_entity_type_dropped(live_db) -> None:
    """An entity_type not in ROUTER_ENTITY_TO_LABEL is silently dropped --
    we have no NodeLabel to look it up under."""
    cust = "cust-entity-unmapped"
    await _seed_customer(cust)
    intent = Intent(
        query_text="x",
        mode="search",
        confidence=0.9,
        entities=[
            RouterEntity(
                entity_type="alien_type", canonical_id="weird",
                display_name="weird", confidence=0.95,
            ),
        ],
    )
    resp = await _run(cust, intent)
    entities = [r for r in resp.results if isinstance(r, QueryEntityResult)]
    assert entities == []


async def test_session_entity_type_dropped_handled_via_id_lookup(
    live_db,
) -> None:
    """The 'session' entity_type maps to NodeLabel.DOCUMENT and is handled
    via the id_lookup channel -- the entity-result builder skips it so the
    same node doesn't surface twice (once as Document via id_lookup, once
    as Entity here)."""
    cust = "cust-entity-session"
    await _seed_customer(cust)
    # Even if a Document graph_node exists for this id, the session entity
    # type must NOT produce a QueryEntityResult.
    await _seed_entity_node(
        cust, label=NodeLabel.DOCUMENT.value,
        canonical_id="3c325e11-2008-46a9-83f7-fc40d11eaf82",
    )
    intent = Intent(
        query_text="x",
        mode="search",
        confidence=0.9,
        entities=[
            RouterEntity(
                entity_type="session",
                canonical_id="3c325e11-2008-46a9-83f7-fc40d11eaf82",
                display_name="session 3c325e11", confidence=0.95,
            ),
        ],
    )
    resp = await _run(cust, intent)
    entities = [r for r in resp.results if isinstance(r, QueryEntityResult)]
    assert entities == []


async def test_routed_entity_with_no_graph_node_is_dropped(live_db) -> None:
    """If the routed entity's (label, canonical_id) doesn't match any
    graph_node row, no QueryEntityResult is emitted -- the user asked
    about something we have no graph evidence for."""
    cust = "cust-entity-missing"
    await _seed_customer(cust)
    # Note: NO graph_node seeded for this canonical_id.
    intent = Intent(
        query_text="x",
        mode="search",
        confidence=0.9,
        entities=[
            RouterEntity(
                entity_type="service", canonical_id="not-in-graph",
                display_name="not-in-graph", confidence=0.95,
            ),
        ],
    )
    resp = await _run(cust, intent)
    entities = [r for r in resp.results if isinstance(r, QueryEntityResult)]
    assert entities == []
