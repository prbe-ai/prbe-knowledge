"""End-to-end test: search → directed → response.

Pins the wire-up that the unit tests can't catch:
  - directed_search runs in the asyncio.gather batch
  - directed_hits flow into fuse() as the new param
  - QueryDocument.retriever_scores carries 'directed'
  - the page's CONTENT chunk is what surfaces (directed text never leaks
    to the response)

Drives run_search with patched chunk-retrievers + a real directed_search
against a seeded directed_vectors row, so the directed path is exercised
end-to-end.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from services.retrieval.retrievers.bm25 import BM25Hit
from services.retrieval.router import RouterOutput
from services.retrieval.search_pipeline import run_search
from shared.config import Settings, get_settings
from shared.db import raw_conn, with_tenant
from shared.embeddings import get_embedder_v2, reset_embedder
from shared.models import QueryDocumentResult, QueryRequest, TemporalSpec
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


_NOW = datetime(2026, 5, 9, tzinfo=UTC)


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


async def _seed_doc(customer_id: str, *, doc_id: str, title: str) -> None:
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
                'wiki', $1, 'https://wiki/' || $1,
                'compiled_wiki', 'wiki.runbook', 'text/markdown',
                'h-' || $1, $3, 100, 0,
                $4, $4, $4, $4, '{}'::jsonb
            )
            """,
            doc_id, customer_id, title, _NOW,
        )
        await conn.execute(
            """
            INSERT INTO chunks (
                chunk_id, doc_id, customer_id,
                chunk_index, content, content_hash, token_count,
                embedding, first_seen_version, last_seen_version, kind
            ) VALUES (
                $1, $2, $3, 0, $4, $5, 5,
                array_fill(0::real, ARRAY[3072])::halfvec,
                1, 1, 'content'
            )
            """,
            f"{doc_id}:c0", doc_id, customer_id,
            f"runbook body for {doc_id}", f"chash-{doc_id}",
        )


async def _seed_directed(customer_id: str, doc_id: str, phrase: str) -> None:
    embedder = get_embedder_v2()
    [vec] = (await embedder.embed_many([phrase])).embedded[:]
    literal = "[" + ",".join(f"{x:.7f}" for x in vec.embedding) + "]"
    h = hashlib.sha256(phrase.lower().encode("utf-8")).digest()
    async with with_tenant(customer_id) as conn:
        await conn.execute(
            """
            INSERT INTO directed_vectors
              (customer_id, doc_id, embedding, source_text, source,
               synthesis_run_id, content_hash)
            VALUES ($1, $2, $3::halfvec, $4, 'human', NULL, $5)
            """,
            customer_id, doc_id, literal, phrase, h,
        )


def _bm25_hit(doc_id: str, content: str) -> BM25Hit:
    return BM25Hit(
        chunk_id=f"{doc_id}:c0",
        doc_id=doc_id,
        doc_version=1,
        source_system="wiki",
        source_url=f"https://wiki/{doc_id}",
        title=doc_id,
        content=content,
        created_at=_NOW,
        updated_at=_NOW,
        score=1.0,
        kind="content",
    )


async def test_directed_signal_flows_end_to_end_to_query_response(
    live_db,
) -> None:
    """Seed a wiki page with a directed phrase whose text differs from the
    page body. Run a search with a query that semantically matches the
    directed phrase (under the stub embedder, identical strings hash to
    identical vectors → cosine similarity ≈ 1.0). Patch chunk-retrievers
    to return ONE bm25 hit for the page (so the doc has a content chunk
    in the pool). Verify:

      - the response surfaces the doc
      - retriever_scores['directed'] is set
      - retriever_scores['directed'] is on RRF magnitude (~1/61), NOT raw
        similarity (~0.99) — pins the P1 score-scale fix
      - QueryDocument.chunks carries the page's CONTENT, never the
        directed phrase text
    """
    cust = "cust-search-directed-e2e"
    await _seed_customer(cust)
    doc_id = "wiki:runbook:rollback"
    await _seed_doc(cust, doc_id=doc_id, title="Production rollback playbook")
    # Directed phrase whose TEXT differs from the page body. The query
    # below matches the phrase but not the body.
    await _seed_directed(cust, doc_id, "deploy keeps timing out")

    routed = RouterOutput(entities=[])
    req = QueryRequest(query="deploy keeps timing out", top_k=5, top_k_related=0)

    patches = {
        "vector": patch(
            "services.retrieval.search_pipeline.vector_search",
            new=AsyncMock(return_value=[]),
        ),
        # bm25 returns a single content chunk for the page so fusion has
        # something to surface; without this, the doc would be dropped
        # (directed signal alone is not a sole source).
        "bm25": patch(
            "services.retrieval.search_pipeline.bm25_search",
            new=AsyncMock(return_value=[_bm25_hit(doc_id, f"runbook body for {doc_id}")]),
        ),
        "graph": patch(
            "services.retrieval.search_pipeline.graph_search",
            new=AsyncMock(return_value=[]),
        ),
        "id_lookup": patch(
            "services.retrieval.search_pipeline.id_lookup_search",
            new=AsyncMock(return_value=[]),
        ),
        "embeddings": patch(
            "services.retrieval.search_pipeline.embeddings_for_chunks",
            new=AsyncMock(return_value={}),
        ),
        "acl": patch(
            "services.retrieval.search_pipeline.filter_by_acl",
            new=AsyncMock(side_effect=lambda _c, _u, hits: hits),
        ),
    }
    with patches["vector"], patches["bm25"], patches["graph"], patches[
        "id_lookup"
    ], patches["embeddings"], patches["acl"]:
        resp = await run_search(
            req=req,
            customer_id=cust,
            routed=routed,
            spec=TemporalSpec(),
            temporal_meta={},
            sort_meta=None,
            extracted_entities=[],
            doc_types=None,
            trace_id="t-directed-e2e",
            timing={},
        )

    # Doc surfaced. The polymorphic response shape carries documents in
    # `results` discriminated by type — filter to QueryDocumentResult.
    docs = [r for r in resp.results if isinstance(r, QueryDocumentResult)]
    assert len(docs) == 1
    doc = docs[0]
    assert doc.doc_id == doc_id

    # directed in retriever_scores breakdown — proves the signal flowed
    # all the way through fusion to the response.
    assert "directed" in doc.retriever_scores

    # And on RRF magnitude, NOT raw similarity. The P1 score-scale fix
    # converts directed list-rank to RRF (rank-1 = 1/61 ≈ 0.0164). If
    # this regresses to raw similarity, the value would be ~0.99 — pin
    # the magnitude ceiling so the ranking-domination bug stays dead.
    assert doc.retriever_scores["directed"] < 0.05, (
        f"directed should be RRF-scale (~0.016), got "
        f"{doc.retriever_scores['directed']}"
    )

    # Response chunks carry the page CONTENT, never the directed phrase.
    assert any("runbook body" in c.content for c in doc.chunks)
    assert all("deploy keeps timing out" not in c.content for c in doc.chunks)

    # MCP-visible signal: matched_via must include a 'directed' provenance.
    # retriever_scores is dropped by the MCP serializer; matched_via is
    # the ONLY user-visible trace. Without this entry, agents and dashboards
    # have no way to tell directed surfaced this doc.
    channels = [m.channel for m in doc.matched_via]
    assert "directed" in channels, (
        f"matched_via must include a 'directed' MatchProvenance entry, "
        f"got channels={channels}. retriever_scores is invisible to MCP "
        f"clients; matched_via is the canonical trace."
    )
    directed_prov = next(m for m in doc.matched_via if m.channel == "directed")
    assert directed_prov.rank == 1, "single hit -> rank 1 in directed list"
    # MatchProvenance.score on directed carries the cosine SIMILARITY
    # (DirectedHit.score), NOT the RRF contribution. The RRF value is in
    # retriever_scores; matched_via shows the raw signal strength so
    # consumers can judge how good the match was.
    assert directed_prov.score >= 0.99, (
        f"identical phrase + stub embedder -> similarity ~1.0, "
        f"got {directed_prov.score}"
    )


async def test_directed_runner_skips_directed_search_when_tenant_has_no_rows(
    live_db,
) -> None:
    """Pre-check optimization: the _directed_runner short-circuits before
    calling directed_search (and the embedding API) when the tenant has
    zero rows in directed_vectors. Saves the embed round-trip for the
    common case where a tenant hasn't enabled the feature.

    Setup: seed a tenant with a doc + a content chunk for bm25 to surface,
    but NO directed_vectors. Run search and patch directed_search to
    raise — proving it was never called.
    """
    cust = "cust-search-no-directed"
    await _seed_customer(cust)
    doc_id = "wiki:runbook:no-directed"
    await _seed_doc(cust, doc_id=doc_id, title="Empty directed page")
    # Note: deliberately NOT seeding directed_vectors — the runner's
    # pre-check should skip the call entirely.

    routed = RouterOutput(entities=[])
    req = QueryRequest(query="anything", top_k=5, top_k_related=0)

    # If the pre-check fails, directed_search runs and trips this assertion.
    directed_search_called = False

    async def _boom(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal directed_search_called
        directed_search_called = True
        raise AssertionError(
            "directed_search must not be called for tenants with zero "
            "directed_vectors rows; the _directed_runner pre-check should "
            "have short-circuited."
        )

    patches = {
        "vector": patch(
            "services.retrieval.search_pipeline.vector_search",
            new=AsyncMock(return_value=[]),
        ),
        "bm25": patch(
            "services.retrieval.search_pipeline.bm25_search",
            new=AsyncMock(return_value=[_bm25_hit(doc_id, f"runbook body for {doc_id}")]),
        ),
        "graph": patch(
            "services.retrieval.search_pipeline.graph_search",
            new=AsyncMock(return_value=[]),
        ),
        "id_lookup": patch(
            "services.retrieval.search_pipeline.id_lookup_search",
            new=AsyncMock(return_value=[]),
        ),
        "directed": patch(
            "services.retrieval.search_pipeline.directed_search",
            new=AsyncMock(side_effect=_boom),
        ),
        "embeddings": patch(
            "services.retrieval.search_pipeline.embeddings_for_chunks",
            new=AsyncMock(return_value={}),
        ),
        "acl": patch(
            "services.retrieval.search_pipeline.filter_by_acl",
            new=AsyncMock(side_effect=lambda _c, _u, hits: hits),
        ),
    }
    with patches["vector"], patches["bm25"], patches["graph"], patches[
        "id_lookup"
    ], patches["directed"], patches["embeddings"], patches["acl"]:
        resp = await run_search(
            req=req,
            customer_id=cust,
            routed=routed,
            spec=TemporalSpec(),
            temporal_meta={},
            sort_meta=None,
            extracted_entities=[],
            doc_types=None,
            trace_id="t-no-directed",
            timing={},
        )

    assert directed_search_called is False, (
        "Pre-check failed: directed_search was called for a tenant with "
        "zero directed_vectors rows."
    )
    # Sanity: the rest of the pipeline still works.
    docs = [r for r in resp.results if isinstance(r, QueryDocumentResult)]
    assert len(docs) == 1
    assert docs[0].doc_id == doc_id
    # No 'directed' key in retriever_scores because the signal didn't run.
    assert "directed" not in docs[0].retriever_scores
    # And no 'directed' provenance in matched_via either (the canonical
    # MCP-visible trace).
    assert all(m.channel != "directed" for m in docs[0].matched_via)
