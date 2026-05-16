"""Integration tests for `related_entities` wire-in on `run_search`.

Covers (per locked plan section 6):
- Walk runs, response.related_entities populated, sorted by score desc,
  excludes any extracted_entities[].canonical_id under matching label.
- req.top_k_related=0 SKIPS the SQL call (no walk, no timing key).
- Same doc surfaced via two top chunks -> dedup-best-rank passes one
  (doc_id, rank) tuple to the SQL with the lowest rank.
- Failure isolation (codex-B4): inject an exception in the walk;
  chunks still return; related_entities is None;
  related_entities_error == "<exc-type>".
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from services.retrieval.retrievers.bm25 import BM25Hit
from services.retrieval.retrievers.graph import GraphHit
from services.retrieval.retrievers.vector import VectorHit
from services.retrieval.router import Intent, RouterEntity
from services.retrieval.search_pipeline import run_search
from shared.config import Settings, get_settings
from shared.constants import EdgeType, NodeLabel
from shared.db import raw_conn
from shared.embeddings import reset_embedder
from shared.models import QueryRequest, RelatedEntity, TemporalSpec
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


async def _seed_doc(
    customer_id: str,
    *,
    doc_id: str,
    title: str = "doc",
) -> None:
    """Seed a documents row + chunk + matching Document graph_node.

    `documents.source_id` follows the `<kind>:<uuid>` shape per memory
    `feedback_documents_source_id_format.md`.
    """
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
    confidence: str = "EXTRACTED",
) -> None:
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO graph_edges (
                customer_id, edge_type, from_node_id, to_node_id,
                confidence, valid_from
            )
            SELECT $1, $2, f.node_id, t.node_id, $5, NOW()
            FROM graph_nodes f, graph_nodes t
            WHERE f.customer_id = $1 AND f.label = $6 AND f.canonical_id = $3
              AND t.customer_id = $1 AND t.label = $7 AND t.canonical_id = $4
            ON CONFLICT DO NOTHING
            """,
            customer_id,
            edge_type,
            doc_id,
            canonical_id,
            confidence,
            NodeLabel.DOCUMENT.value,
            label,
        )


def _bm25_hit(doc_id: str) -> BM25Hit:
    """A BM25Hit shaped like what `bm25_search` would return for the seeded
    docs above. Used to drive run_search via patched retrievers."""
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


def _patch_retrievers(*, bm25_hits: list[BM25Hit] | None = None,
                      vec_hits: list[VectorHit] | None = None,
                      graph_hits: list[GraphHit] | None = None,
                      id_hits: list | None = None,
                      embeddings: dict | None = None,
                      acl_passthrough: bool = True):
    """Patch the four retrievers + embeddings + ACL so run_search becomes
    deterministic. The walk runs against the live DB."""
    bm25 = bm25_hits or []
    vec = vec_hits or []
    graph = graph_hits or []
    ids = id_hits or []
    embeds = embeddings or {}

    return {
        "vector": patch(
            "services.retrieval.search_pipeline.vector_search",
            new=AsyncMock(return_value=vec),
        ),
        "bm25": patch(
            "services.retrieval.search_pipeline.bm25_search",
            new=AsyncMock(return_value=bm25),
        ),
        "graph": patch(
            "services.retrieval.search_pipeline.graph_search",
            new=AsyncMock(return_value=graph),
        ),
        "id_lookup": patch(
            "services.retrieval.search_pipeline.id_lookup_search",
            new=AsyncMock(return_value=ids),
        ),
        "embeddings": patch(
            "services.retrieval.search_pipeline.embeddings_for_chunks",
            new=AsyncMock(return_value=embeds),
        ),
        "acl": patch(
            "services.retrieval.search_pipeline.filter_by_acl",
            new=AsyncMock(side_effect=lambda _c, _u, hits: hits)
            if acl_passthrough
            else AsyncMock(return_value=[]),
        ),
    }


# ---- tests ---------------------------------------------------------------


async def test_related_entities_populated_and_excludes_routed_entity(
    live_db,
) -> None:
    """Walk runs, response.related_entities populated, ranked by score,
    and excludes the routed entity (matching label, canonical_id)."""
    cust = "cust-search-related"
    await _seed_customer(cust)
    await _seed_doc(cust, doc_id="doc:1", title="d1")
    await _seed_doc(cust, doc_id="doc:2", title="d2")
    # Repo attached to both docs (will be the visible related entity).
    await _seed_neighbor(
        cust, label=NodeLabel.REPO.value,
        canonical_id="prbe-ai/prbe-knowledge", name="prbe-knowledge",
    )
    await _seed_edge(
        cust, doc_id="doc:1", label=NodeLabel.REPO.value,
        canonical_id="prbe-ai/prbe-knowledge",
    )
    await _seed_edge(
        cust, doc_id="doc:2", label=NodeLabel.REPO.value,
        canonical_id="prbe-ai/prbe-knowledge",
    )
    # Service routed by the query -- attached to both docs but must be
    # excluded because the LLM already has its handle from extracted_entities.
    await _seed_neighbor(
        cust, label=NodeLabel.SERVICE.value,
        canonical_id="prbe-backend", name="prbe-backend",
    )
    await _seed_edge(
        cust, doc_id="doc:1", label=NodeLabel.SERVICE.value,
        canonical_id="prbe-backend",
    )
    await _seed_edge(
        cust, doc_id="doc:2", label=NodeLabel.SERVICE.value,
        canonical_id="prbe-backend",
    )

    intent = Intent(
        query_text="prbe-backend",
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
    req = QueryRequest(query="prbe-backend", top_k=2, top_k_related=10)

    patches = _patch_retrievers(
        bm25_hits=[_bm25_hit("doc:1"), _bm25_hit("doc:2")],
    )
    with patches["vector"], patches["bm25"], patches["graph"], \
            patches["id_lookup"], patches["embeddings"], patches["acl"]:
        resp = await run_search(
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

    assert resp.related_entities is not None
    assert resp.related_entities_error is None
    cids = {e.canonical_id for e in resp.related_entities}
    assert "prbe-ai/prbe-knowledge" in cids
    # The routed Service was excluded.
    assert "prbe-backend" not in cids
    # Ordering is score desc.
    scores = [e.score for e in resp.related_entities]
    assert scores == sorted(scores, reverse=True)


async def test_top_k_related_zero_skips_walk(live_db) -> None:
    """req.top_k_related=0 -> related_entities is None, no error,
    no `related_entities_ms` timing key."""
    cust = "cust-search-skip"
    await _seed_customer(cust)
    await _seed_doc(cust, doc_id="doc:1", title="d1")
    await _seed_neighbor(
        cust, label=NodeLabel.REPO.value, canonical_id="repo:x", name="x",
    )
    await _seed_edge(cust, doc_id="doc:1", label=NodeLabel.REPO.value, canonical_id="repo:x")

    req = QueryRequest(query="x", top_k=2, top_k_related=0)
    patches = _patch_retrievers(bm25_hits=[_bm25_hit("doc:1")])

    timing: dict = {}
    with patches["vector"], patches["bm25"], patches["graph"], \
            patches["embeddings"], patches["acl"], \
            patch(
                "services.retrieval.search_pipeline.walk_result_doc_neighbors",
                new=AsyncMock(return_value=[]),
            ) as m_walk:
        resp = await run_search(
            req=req,
            customer_id=cust,
            intent=Intent(query_text="x", mode="search", confidence=0.9),
            intent_idx=0,
            spec=TemporalSpec(),
            temporal_meta={},
            sort_meta=None,
            extracted_entities=[],
            doc_types=None,
            trace_id="t-1",
            timing=timing,
        )

    assert resp.related_entities is None
    assert resp.related_entities_error is None
    m_walk.assert_not_called()
    assert "related_entities_ms" not in timing


async def test_dedup_best_rank_per_doc(live_db) -> None:
    """Same doc surfaced via two top chunks -> walk receives a single
    (doc_id, rank) tuple keyed on the lowest rank.
    """
    cust = "cust-search-dedup"
    await _seed_customer(cust)
    await _seed_doc(cust, doc_id="doc:1", title="d1")
    await _seed_doc(cust, doc_id="doc:2", title="d2")
    await _seed_neighbor(cust, label=NodeLabel.REPO.value, canonical_id="r", name="r")
    await _seed_edge(cust, doc_id="doc:1", label=NodeLabel.REPO.value, canonical_id="r")
    await _seed_edge(cust, doc_id="doc:2", label=NodeLabel.REPO.value, canonical_id="r")

    # Two BM25 hits on doc:1 (different chunk_ids, same doc) plus one on doc:2.
    # filtered top_k=3 -> ranks 1, 2, 3 with two from doc:1.
    now = datetime(2026, 4, 28, tzinfo=UTC)
    hit1a = _bm25_hit("doc:1")
    hit1b = BM25Hit(
        chunk_id="doc:1:c1", doc_id="doc:1", doc_version=1,
        source_system="github", source_url="https://example/doc:1",
        title="d1", content="body 1b",
        created_at=now, updated_at=now, score=0.9, kind="content",
    )
    hit2 = _bm25_hit("doc:2")

    req = QueryRequest(query="r", top_k=3, top_k_related=10)
    patches = _patch_retrievers(bm25_hits=[hit1a, hit1b, hit2])

    captured_call: dict = {}

    async def fake_walk(customer_id, *, ranked_result_docs, exclude_node_keys,
                        min_confidence, top_n):
        captured_call["ranked"] = list(ranked_result_docs)
        return []

    with patches["vector"], patches["bm25"], patches["graph"], \
            patches["embeddings"], patches["acl"], \
            patch(
                "services.retrieval.search_pipeline.walk_result_doc_neighbors",
                new=fake_walk,
            ):
        await run_search(
            req=req,
            customer_id=cust,
            intent=Intent(query_text="r", mode="search", confidence=0.9),
            intent_idx=0,
            spec=TemporalSpec(),
            temporal_meta={},
            sort_meta=None,
            extracted_entities=[],
            doc_types=None,
            trace_id="t-1",
            timing={},
        )

    ranked = captured_call["ranked"]
    # Each doc appears AT MOST ONCE -- the dedup-best-rank pass collapses
    # two chunks of doc:1 into one (doc_id, rank) tuple. doc:2 may or may
    # not appear depending on what the upstream dedup/ACL kept.
    doc_ids = [d for d, _ in ranked]
    assert len(doc_ids) == len(set(doc_ids))
    assert "doc:1" in doc_ids
    by_doc = dict(ranked)
    # doc:1 came in first (rank 1) -- best (lowest) rank wins; the second
    # doc:1 chunk does NOT bump it to rank 2.
    assert by_doc["doc:1"] == 1
    # The ranks returned must be a contiguous prefix of (1..len(top))
    # from the perspective of post-fusion/dedup chunks (i.e. set of
    # ranks is exactly {1..N} for N the number of distinct docs in top).
    assert sorted(by_doc.values()) == list(range(1, len(by_doc) + 1))


async def test_walk_failure_isolation_chunks_still_return(live_db) -> None:
    """Inject an exception in walk_result_doc_neighbors. The host search
    must still return chunks; related_entities is None;
    related_entities_error == "<exc-type>"; timing_ms carries the elapsed
    walk duration but NOT a sentinel error key (timing_ms is durations
    only -- error info lives on the dedicated response field).
    """
    cust = "cust-search-failure"
    await _seed_customer(cust)
    await _seed_doc(cust, doc_id="doc:1", title="d1")

    req = QueryRequest(query="x", top_k=2, top_k_related=10)
    patches = _patch_retrievers(bm25_hits=[_bm25_hit("doc:1")])

    class FakeWalkError(RuntimeError):
        pass

    timing: dict = {}
    with patches["vector"], patches["bm25"], patches["graph"], \
            patches["embeddings"], patches["acl"], \
            patch(
                "services.retrieval.search_pipeline.walk_result_doc_neighbors",
                new=AsyncMock(side_effect=FakeWalkError("boom")),
            ):
        resp = await run_search(
            req=req,
            customer_id=cust,
            intent=Intent(query_text="x", mode="search", confidence=0.9),
            intent_idx=0,
            spec=TemporalSpec(),
            temporal_meta={},
            sort_meta=None,
            extracted_entities=[],
            doc_types=None,
            trace_id="t-1",
            timing=timing,
        )

    # Polymorphic shape (PR feat/polymorphic-search-results): primary
    # results carry one Document with the chunk nested under it.
    from shared.models import QueryDocumentResult
    docs = [r for r in resp.results if isinstance(r, QueryDocumentResult)]
    assert docs  # docs still flow through
    assert docs[0].doc_id == "doc:1"
    assert resp.related_entities is None
    assert resp.related_entities_error == "FakeWalkError"
    assert "related_entities_ms" in timing
    # timing_ms is durations only -- error sentinel must NOT appear here.
    assert "related_entities_error" not in timing


async def test_walk_legitimate_empty_returns_empty_list(live_db) -> None:
    """When the walk runs successfully but finds no neighbors, the response
    carries `related_entities=[]` (NOT None). Distinguishing the empty
    case is the whole point of the three-state contract (codex-B4).
    """
    cust = "cust-search-empty"
    await _seed_customer(cust)
    await _seed_doc(cust, doc_id="doc:1", title="d1")
    # No neighbor edges seeded -- walk runs but finds nothing.

    req = QueryRequest(query="x", top_k=2, top_k_related=10)
    patches = _patch_retrievers(bm25_hits=[_bm25_hit("doc:1")])

    with patches["vector"], patches["bm25"], patches["graph"], \
            patches["id_lookup"], patches["embeddings"], patches["acl"]:
        resp = await run_search(
            req=req,
            customer_id=cust,
            intent=Intent(query_text="x", mode="search", confidence=0.9),
            intent_idx=0,
            spec=TemporalSpec(),
            temporal_meta={},
            sort_meta=None,
            extracted_entities=[],
            doc_types=None,
            trace_id="t-1",
            timing={},
        )

    # The list must be present (not None) and empty.
    assert resp.related_entities == []
    assert resp.related_entities_error is None
    assert isinstance(resp.related_entities, list)
    # Sanity: it's actually a list of RelatedEntity (zero of them).
    assert all(isinstance(e, RelatedEntity) for e in resp.related_entities)
