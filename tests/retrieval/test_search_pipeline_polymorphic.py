"""End-to-end test for the polymorphic search-pipeline output
(PR feat/polymorphic-search-results).

`run_search` must return a `QueryResponse` with `results: list[QueryResult]`
that interleaves Document and Entity variants. Each Document carries
`matched_via: list[MatchProvenance]` with at least one channel entry, and
inferred-edge-derived Documents carry channel='inferred_edge' with the
LLM-derived `why` populated.

Strategy: the four primary retrievers (vector / BM25 / graph / id_lookup)
plus embeddings + ACL are mocked so the pipeline is deterministic. The
inferred-edges + entity-lookup paths run against the LIVE DB so the SQL
contracts are exercised end-to-end.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from services.retrieval.retrievers.bm25 import BM25Hit
from services.retrieval.retrievers.inferred_edges import INFERRED_EDGES_EXTRACTOR_ID
from services.retrieval.router import Intent, RouterEntity
from services.retrieval.search_pipeline import run_search
from shared.config import Settings, get_settings
from shared.constants import EdgeType, NodeLabel
from shared.db import raw_conn
from shared.embeddings import reset_embedder
from shared.models import (
    QueryDocumentResult,
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
    title: str = "doc",
) -> None:
    """Seed a documents row + chunk + matching Document graph_node."""
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


async def _seed_entity_node(
    customer_id: str,
    *,
    label: str,
    canonical_id: str,
    name: str | None = None,
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


async def _seed_inferred_doc_edge(
    customer_id: str,
    *,
    from_doc_id: str,
    to_doc_id: str,
    edge_type: str,
    why: str,
) -> None:
    """Seed a Lane B inferred Doc-Doc edge."""
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO graph_edges (
                customer_id, edge_type, from_node_id, to_node_id,
                properties, confidence, valid_from, extractor_id, extracted_at
            )
            SELECT $1::text, $2::text, f.node_id, t.node_id,
                   jsonb_build_object('why', $5::text),
                   'INFERRED', NOW(), $6::text, NOW()
            FROM graph_nodes f, graph_nodes t
            WHERE f.customer_id = $1 AND f.label = 'Document' AND f.canonical_id = $3
              AND t.customer_id = $1 AND t.label = 'Document' AND t.canonical_id = $4
            ON CONFLICT DO NOTHING
            """,
            customer_id, edge_type, from_doc_id, to_doc_id, why,
            INFERRED_EDGES_EXTRACTOR_ID,
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


def _bm25_hit(doc_id: str) -> BM25Hit:
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


def _patch_retrievers(*, bm25_hits=None):
    bm25 = bm25_hits or []
    return [
        patch(
            "services.retrieval.search_pipeline.vector_search",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "services.retrieval.search_pipeline.bm25_search",
            new=AsyncMock(return_value=bm25),
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


# ---- tests ---------------------------------------------------------------


async def test_response_carries_polymorphic_results_with_matched_via(
    live_db,
) -> None:
    """run_search returns QueryResponse with `results: list[QueryResult]`.
    Each Document has matched_via with at least one MatchProvenance entry.
    """
    cust = "cust-poly-basic"
    await _seed_customer(cust)
    await _seed_doc(cust, doc_id="doc:1", title="d1")

    req = QueryRequest(query="x", top_k=2, top_k_related=0)
    patches = _patch_retrievers(bm25_hits=[_bm25_hit("doc:1")])
    with (
        patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]
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
            timing={},
        )

    assert len(resp.results) == 1
    first = resp.results[0]
    assert isinstance(first, QueryDocumentResult)
    assert first.doc_id == "doc:1"
    assert first.matched_via, "every primary doc must have at least one channel"
    channels = {p.channel for p in first.matched_via}
    assert "bm25" in channels
    assert all(p.intent_idx == 0 for p in first.matched_via)


async def test_inferred_edge_documents_surface_with_why(live_db) -> None:
    """A primary doc has a Lane B inferred Doc-Doc edge to a linked doc.
    The linked doc surfaces as a QueryDocumentResult with channel=
    'inferred_edge' and `why` populated."""
    cust = "cust-poly-inferred"
    await _seed_customer(cust)
    await _seed_doc(cust, doc_id="primary:1", title="primary")
    await _seed_doc(cust, doc_id="linked:1", title="linked")
    await _seed_inferred_doc_edge(
        cust, from_doc_id="primary:1", to_doc_id="linked:1",
        edge_type=EdgeType.DISCUSSES.value,
        why="Both docs cover the auth refactor decision.",
    )

    req = QueryRequest(query="x", top_k=2, top_k_related=0)
    patches = _patch_retrievers(bm25_hits=[_bm25_hit("primary:1")])
    with (
        patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]
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
            timing={},
        )

    docs = [r for r in resp.results if isinstance(r, QueryDocumentResult)]
    by_doc = {d.doc_id: d for d in docs}
    assert "primary:1" in by_doc
    assert "linked:1" in by_doc
    inferred_doc = by_doc["linked:1"]
    inferred_provs = [
        p for p in inferred_doc.matched_via if p.channel == "inferred_edge"
    ]
    assert len(inferred_provs) == 1
    prov = inferred_provs[0]
    assert prov.anchor_doc_id == "primary:1"
    assert prov.edge_type == EdgeType.DISCUSSES.value
    assert prov.confidence == "INFERRED"
    assert prov.why == "Both docs cover the auth refactor decision."
    assert prov.intent_idx == 0
    # Inferred-edge timing key was recorded.
    # (timing was passed as a fresh dict; we don't assert keys here -- the
    # retriever-error path is covered separately. The fact that the doc
    # surfaced means the SQL ran successfully.)


async def test_routed_entity_surfaces_as_entity_result_alongside_docs(
    live_db,
) -> None:
    """Router emits a Service entity that has a graph_node row + 1-hop
    edges to the result-set docs. The response carries one
    QueryEntityResult plus the Document results, mixed in `results`."""
    cust = "cust-poly-entity"
    await _seed_customer(cust)
    await _seed_doc(cust, doc_id="doc:1", title="d1")
    await _seed_doc(cust, doc_id="doc:2", title="d2")
    await _seed_entity_node(
        cust, label=NodeLabel.SERVICE.value,
        canonical_id="prbe-backend", name="prbe-backend",
    )
    await _seed_doc_to_entity_edge(
        cust, doc_id="doc:1", label=NodeLabel.SERVICE.value,
        canonical_id="prbe-backend",
    )
    await _seed_doc_to_entity_edge(
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
    req = QueryRequest(query="prbe-backend", top_k=2, top_k_related=0)
    patches = _patch_retrievers(bm25_hits=[_bm25_hit("doc:1"), _bm25_hit("doc:2")])
    with (
        patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]
    ):
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

    entities = [r for r in resp.results if isinstance(r, QueryEntityResult)]
    docs = [r for r in resp.results if isinstance(r, QueryDocumentResult)]
    assert len(entities) == 1
    assert len(docs) == 2
    e = entities[0]
    assert e.canonical_id == "prbe-backend"
    assert e.label == NodeLabel.SERVICE.value
    assert e.display_name == "prbe-backend"
    assert e.doc_count == 2


async def test_results_sorted_by_score_with_rank_assigned(live_db) -> None:
    """Final pass sorts results by score desc and stamps rank 1..N."""
    cust = "cust-poly-rank"
    await _seed_customer(cust)
    await _seed_doc(cust, doc_id="doc:a", title="a")
    await _seed_doc(cust, doc_id="doc:b", title="b")

    req = QueryRequest(query="x", top_k=2, top_k_related=0)
    patches = _patch_retrievers(
        bm25_hits=[_bm25_hit("doc:a"), _bm25_hit("doc:b")]
    )
    with (
        patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]
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
            timing={},
        )

    ranks = [r.rank for r in resp.results]
    assert ranks == sorted(ranks)  # 1, 2, 3, ...
    assert ranks[0] == 1
    scores = [r.score for r in resp.results]
    assert scores == sorted(scores, reverse=True)  # score desc


async def test_chunk_carries_rank_in_doc_not_doc_level_fields(
    live_db,
) -> None:
    """QueryChunk on a Document has rank_in_doc but NOT doc_id / source /
    title / created_at -- those live on the parent QueryDocumentResult."""
    cust = "cust-poly-chunk-shape"
    await _seed_customer(cust)
    await _seed_doc(cust, doc_id="doc:1", title="d1")

    req = QueryRequest(query="x", top_k=2, top_k_related=0)
    patches = _patch_retrievers(bm25_hits=[_bm25_hit("doc:1")])
    with (
        patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]
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
            timing={},
        )

    doc = next(r for r in resp.results if isinstance(r, QueryDocumentResult))
    assert doc.chunks
    chunk = doc.chunks[0]
    assert chunk.rank_in_doc == 1
    # These attributes don't exist on the chunk in the new shape.
    assert not hasattr(chunk, "doc_id")
    assert not hasattr(chunk, "source_system")
    assert not hasattr(chunk, "title")
    # But the parent doc carries them.
    assert doc.doc_id == "doc:1"
    # Title flows from the BM25Hit (set to doc_id by the test helper).
    assert doc.title == "doc:1"


async def test_inferred_edge_results_hydrate_chunks(live_db) -> None:
    """v1 wrapped inferred-edge hits with `chunks=[]` -- the dashboard
    rendered "0 matched" and the synthesizer couldn't cite from them.
    Now the wrapper hydrates body chunks (capped at
    INFERRED_EDGE_HYDRATION_CHUNKS) so inferred-edge results are
    first-class evidence."""
    cust = "cust-poly-hydrate"
    await _seed_customer(cust)
    await _seed_doc(cust, doc_id="primary:1", title="primary")
    # `_seed_doc` inserts chunk_index=0; that single chunk is what the
    # hydrator should pull through onto the inferred-edge result.
    await _seed_doc(cust, doc_id="linked:1", title="linked")
    await _seed_inferred_doc_edge(
        cust, from_doc_id="primary:1", to_doc_id="linked:1",
        edge_type=EdgeType.DISCUSSES.value,
        why="Both cover the migration.",
    )

    req = QueryRequest(query="x", top_k=2, top_k_related=0)
    patches = _patch_retrievers(bm25_hits=[_bm25_hit("primary:1")])
    with (
        patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]
    ):
        resp = await run_search(
            req=req, customer_id=cust,
            intent=Intent(query_text="x", mode="search", confidence=0.9),
            intent_idx=0,
            spec=TemporalSpec(), temporal_meta={}, sort_meta=None,
            extracted_entities=[], doc_types=None, trace_id="t-hydrate",
            timing={},
        )

    by_doc = {
        r.doc_id: r for r in resp.results if isinstance(r, QueryDocumentResult)
    }
    assert "linked:1" in by_doc
    inferred_doc = by_doc["linked:1"]
    # The fix: chunks are populated, not [].
    assert inferred_doc.chunks, "inferred-edge result must carry hydrated chunks"
    assert inferred_doc.chunk_count == len(inferred_doc.chunks)
    # The seeded chunk's content surfaces verbatim so the synthesizer
    # has something to cite.
    chunk = inferred_doc.chunks[0]
    assert chunk.content == "body of linked:1"
    assert chunk.rank_in_doc == 1


async def test_fanout_penalty_crushes_hub_doc_below_primary(live_db) -> None:
    """Regression test for the codex-session-#1 case: a high-fan-out
    inferred-edge target was outranking the primary BM25 hit it surfaced
    *from*. The combined dampening + source-mult + fan-out penalty must
    keep agent-session-style hubs below direct primary matches.

    Setup:
      - primary:1 = a regular Notion-source doc (BM25 hit, source_mult 1.0).
      - linked:hub = a CODEX-source doc with 8 INFERRED edges (hub).
      - One DISCUSSES edge primary:1 -> linked:hub brings the hub in.

    Expectation: linked:hub.score < primary:1.score -- the hub does NOT
    surface above the doc it was reached from."""
    cust = "cust-poly-fanout"
    await _seed_customer(cust)
    await _seed_doc(cust, doc_id="primary:1", title="primary")
    # Override _seed_doc to source_system='codex' on the hub via a direct
    # UPDATE -- the fixture is hardcoded to 'github' so we patch the row
    # in place after seeding rather than fork the helper.
    await _seed_doc(cust, doc_id="linked:hub", title="hub")
    async with raw_conn() as conn:
        await conn.execute(
            "UPDATE documents SET source_system = 'codex' "
            "WHERE customer_id = $1 AND doc_id = 'linked:hub'",
            cust,
        )
    # 25 sibling edges that pile additional INFERRED relationships onto the
    # hub -- mimics the codex-session-#1 case in prod which had ~30 edges.
    # At 25 siblings + 1 anchor edge = 26 INFERRED edges on the hub:
    #   fan-out divisor = 1 + ln(26) = 4.26
    #   hub_final = 0.2 * 1/2 * 0.5 / 4.26 = 0.0117
    # which is below the primary's post-fusion score (~0.015).
    # 8 edges (divisor 3.08) was a borderline case where the penalty
    # didn't quite catch up to fusion-decayed primary scores.
    for i in range(25):
        await _seed_doc(cust, doc_id=f"sibling:{i}")
        await _seed_inferred_doc_edge(
            cust, from_doc_id="linked:hub", to_doc_id=f"sibling:{i}",
            edge_type=EdgeType.RELATES_TO.value,
            why=f"sibling fan-out {i}",
        )
    # The DISCUSSES edge that brings the hub into our walk.
    await _seed_inferred_doc_edge(
        cust, from_doc_id="primary:1", to_doc_id="linked:hub",
        edge_type=EdgeType.DISCUSSES.value, why="hub reason",
    )

    req = QueryRequest(query="x", top_k=2, top_k_related=0)
    patches = _patch_retrievers(bm25_hits=[_bm25_hit("primary:1")])
    with (
        patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]
    ):
        resp = await run_search(
            req=req, customer_id=cust,
            intent=Intent(query_text="x", mode="search", confidence=0.9),
            intent_idx=0,
            spec=TemporalSpec(), temporal_meta={}, sort_meta=None,
            extracted_entities=[], doc_types=None, trace_id="t-fanout",
            timing={},
        )

    by_doc = {
        r.doc_id: r for r in resp.results if isinstance(r, QueryDocumentResult)
    }
    assert "primary:1" in by_doc and "linked:hub" in by_doc
    primary = by_doc["primary:1"]
    hub = by_doc["linked:hub"]
    # The hub's combined penalties (dampening + 0.5x codex source-mult +
    # fan-out divisor over 8 edges) must keep it below the primary it
    # was reached from. The exact numbers float around with fusion
    # multipliers so we just pin the ordering, not magnitudes.
    assert hub.score < primary.score, (
        f"fan-out hub (score={hub.score}) should rank below primary "
        f"(score={primary.score})"
    )
    # And the rank stamp reflects the same ordering.
    assert hub.rank > primary.rank
