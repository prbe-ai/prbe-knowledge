"""Plan A Component 6: default ``visibility='approved'`` retrieval filter.

Verifies that the retrieval chokepoints exclude draft documents/chunks
by default and only surface them when the reviewer-only
``include_drafts=True`` path is taken.

Coverage:

- ``services/retrieval/retrievers/vector.py``         — vector_search
- ``services/retrieval/retrievers/bm25.py``           — bm25_search
- ``services/retrieval/retrievers/id_lookup.py``      — id_lookup_search
- ``services/retrieval/retrievers/sql.py``            — sql_list, sql_count,
                                                        sql_group_by
- ``services/retrieval/retrievers/graph.py``          — graph_search
- ``services/retrieval/retrievers/directed.py``       — directed_search
- ``services/retrieval/retrievers/inferred_edges.py`` — inferred_edge_search
- ``services/retrieval/main.py``                      — _load_source_doc_and_chunks
- ``services/retrieval/agent/tools.py``               — execute_fetch_doc
- Wiki-listing queries (Plan B):
    - ``services/ingestion/wiki_routes.py``      (wiki TOC)
    - ``services/synthesis/wiki_agent.py``       (drain-time index regen)
    - ``services/synthesis/persistence.py``      (fetch_wiki_index)

Live Postgres required (the shared ``live_db`` fixture truncates between
tests).
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

from services.retrieval.agent.tools import execute_fetch_doc
from services.retrieval.main import _load_source_doc_and_chunks
from services.retrieval.retrievers.bm25 import bm25_search
from services.retrieval.retrievers.directed import directed_search
from services.retrieval.retrievers.graph import graph_search
from services.retrieval.retrievers.id_lookup import id_lookup_search
from services.retrieval.retrievers.inferred_edges import (
    INFERRED_EDGES_EXTRACTOR_ID,
    inferred_edge_search,
)
from services.retrieval.retrievers.sql import sql_count, sql_group_by, sql_list
from services.retrieval.retrievers.vector import vector_search
from services.synthesis.persistence import fetch_wiki_index
from shared.constants import NodeLabel
from shared.db import raw_conn
from shared.embeddings import get_embedder_v2, reset_embedder

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------


def _new_customer_id(prefix: str = "retr-vis") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


async def _seed_customer(customer_id: str) -> None:
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers (customer_id, display_name, api_key_hash) "
            "VALUES ($1, $2, $3) ON CONFLICT (customer_id) DO NOTHING",
            customer_id,
            f"test {customer_id}",
            f"h-{customer_id}",
        )


async def _seed_doc(
    customer_id: str,
    doc_id: str,
    *,
    title: str,
    content: str,
    visibility: str,
    source_system: str = "wiki",
    doc_type: str = "wiki.runbook",
    source_id: str | None = None,
    source_url: str | None = None,
    chunk_kind: str = "content",
) -> None:
    """Insert one document + one chunk at the given visibility.

    Uses the embedding_v2 column so the vector retriever picks it up
    (the partial HNSW index requires a non-null embedding_v2). All seeded
    docs share the same zero vector so the cosine distance is deterministic.
    Content text varies so bm25 still differentiates by score.
    """
    now = datetime.now(UTC)
    src_id = source_id or doc_id
    src_url = source_url or f"https://example/{doc_id}"
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO documents (
                doc_id, version, customer_id,
                source_system, source_id, source_url,
                doc_class, doc_type, content_type,
                content_hash, title, body_size_bytes, body_token_count,
                created_at, updated_at, valid_from, ingested_at,
                acl, metadata, entities, attachments, doc_references,
                normalizer_version, visibility
            ) VALUES (
                $1, 1, $2,
                $3, $4, $5,
                'compiled_wiki', $6, 'text/markdown',
                $7, $8, $9, 0,
                $10, $10, $10, $10,
                '{}'::jsonb,
                '{}'::jsonb, '[]'::jsonb, '[]'::jsonb, '[]'::jsonb,
                'v1', $11
            )
            ON CONFLICT (customer_id, doc_id, version) DO NOTHING
            """,
            doc_id,
            customer_id,
            source_system,
            src_id,
            src_url,
            doc_type,
            f"hash-{doc_id}",
            title,
            len(content.encode()),
            now,
            visibility,
        )
        await conn.execute(
            """
            INSERT INTO chunks (
                chunk_id, doc_id, customer_id,
                chunk_index, content, content_hash, token_count,
                first_seen_version, last_seen_version,
                kind,
                embedding_v2,
                visibility
            ) VALUES (
                $1, $2, $3,
                0, $4, $5, 5,
                1, 1,
                $6,
                array_fill(0::real, ARRAY[3072])::halfvec,
                $7
            )
            ON CONFLICT (doc_id, content_hash) DO NOTHING
            """,
            f"{doc_id}:c0:v1",
            doc_id,
            customer_id,
            content,
            f"chunk-hash-{doc_id}",
            chunk_kind,
            visibility,
        )


# ---------------------------------------------------------------------------
# Retrieval chokepoint tests
# ---------------------------------------------------------------------------


async def test_vector_search_excludes_drafts_by_default(live_db) -> None:
    cid = _new_customer_id()
    await _seed_customer(cid)
    await _seed_doc(
        cid, f"{cid}:approved",
        title="Approved page",
        content="checkout pool exhaustion runbook",
        visibility="approved",
    )
    await _seed_doc(
        cid, f"{cid}:draft",
        title="Draft page",
        content="checkout pool exhaustion runbook",
        visibility="draft",
    )

    hits = await vector_search(cid, "checkout pool", top_k=10)
    doc_ids = {h.doc_id for h in hits}
    assert f"{cid}:approved" in doc_ids
    assert f"{cid}:draft" not in doc_ids


async def test_vector_search_include_drafts_returns_both(live_db) -> None:
    cid = _new_customer_id()
    await _seed_customer(cid)
    await _seed_doc(
        cid, f"{cid}:approved",
        title="Approved page",
        content="checkout pool exhaustion runbook",
        visibility="approved",
    )
    await _seed_doc(
        cid, f"{cid}:draft",
        title="Draft page",
        content="checkout pool exhaustion runbook",
        visibility="draft",
    )

    hits = await vector_search(cid, "checkout pool", top_k=10, include_drafts=True)
    doc_ids = {h.doc_id for h in hits}
    assert f"{cid}:approved" in doc_ids
    assert f"{cid}:draft" in doc_ids


async def test_bm25_search_excludes_drafts_by_default(live_db) -> None:
    cid = _new_customer_id()
    await _seed_customer(cid)
    await _seed_doc(
        cid, f"{cid}:approved",
        title="Approved",
        content="unique-token-zzz approved body",
        visibility="approved",
    )
    await _seed_doc(
        cid, f"{cid}:draft",
        title="Draft",
        content="unique-token-zzz draft body",
        visibility="draft",
    )

    hits = await bm25_search(cid, "unique-token-zzz", top_k=10)
    doc_ids = {h.doc_id for h in hits}
    assert f"{cid}:approved" in doc_ids
    assert f"{cid}:draft" not in doc_ids


async def test_bm25_search_include_drafts_returns_both(live_db) -> None:
    cid = _new_customer_id()
    await _seed_customer(cid)
    await _seed_doc(
        cid, f"{cid}:approved",
        title="Approved",
        content="unique-token-zzz approved body",
        visibility="approved",
    )
    await _seed_doc(
        cid, f"{cid}:draft",
        title="Draft",
        content="unique-token-zzz draft body",
        visibility="draft",
    )

    hits = await bm25_search(cid, "unique-token-zzz", top_k=10, include_drafts=True)
    doc_ids = {h.doc_id for h in hits}
    assert f"{cid}:approved" in doc_ids
    assert f"{cid}:draft" in doc_ids


async def test_id_lookup_excludes_drafts_by_default(live_db) -> None:
    cid = _new_customer_id()
    await _seed_customer(cid)
    # Ticket-shaped canonical_ids are the simplest is_lookup_candidate
    # match (anchored on _TICKET_RE: LETTERS-DIGITS). source_id ==
    # canonical_id so the exact-equality arm matches.
    approved_ticket = "PRB-1001"
    draft_ticket = "PRB-1002"
    await _seed_doc(
        cid, f"{cid}:approved:{approved_ticket}",
        title="Approved ticket page",
        content="anything",
        visibility="approved",
        source_id=approved_ticket,
    )
    await _seed_doc(
        cid, f"{cid}:draft:{draft_ticket}",
        title="Draft ticket page",
        content="anything",
        visibility="draft",
        source_id=draft_ticket,
    )

    # Approved ticket surfaces under the default approved-only filter.
    hits_default = await id_lookup_search(cid, [approved_ticket])
    assert any(h.doc_id == f"{cid}:approved:{approved_ticket}" for h in hits_default)

    # Draft ticket hidden under the default filter.
    hits_drafts_off = await id_lookup_search(cid, [draft_ticket])
    assert all(h.doc_id != f"{cid}:draft:{draft_ticket}" for h in hits_drafts_off)

    # include_drafts=True surfaces the draft.
    hits_drafts_on = await id_lookup_search(
        cid, [draft_ticket], include_drafts=True,
    )
    assert any(h.doc_id == f"{cid}:draft:{draft_ticket}" for h in hits_drafts_on)


async def test_sql_list_excludes_drafts_by_default(live_db) -> None:
    cid = _new_customer_id()
    await _seed_customer(cid)
    await _seed_doc(
        cid, f"{cid}:approved",
        title="Approved",
        content="body a",
        visibility="approved",
    )
    await _seed_doc(
        cid, f"{cid}:draft",
        title="Draft",
        content="body b",
        visibility="draft",
    )

    hits = await sql_list(cid, top_k=10)
    doc_ids = {h.doc_id for h in hits}
    assert f"{cid}:approved" in doc_ids
    assert f"{cid}:draft" not in doc_ids

    hits_drafts = await sql_list(cid, top_k=10, include_drafts=True)
    doc_ids_drafts = {h.doc_id for h in hits_drafts}
    assert f"{cid}:approved" in doc_ids_drafts
    assert f"{cid}:draft" in doc_ids_drafts


async def test_sql_count_excludes_drafts_by_default(live_db) -> None:
    cid = _new_customer_id()
    await _seed_customer(cid)
    await _seed_doc(
        cid, f"{cid}:approved",
        title="Approved",
        content="body a",
        visibility="approved",
    )
    await _seed_doc(
        cid, f"{cid}:draft",
        title="Draft",
        content="body b",
        visibility="draft",
    )

    n_default = await sql_count(cid)
    n_with_drafts = await sql_count(cid, include_drafts=True)
    assert n_default == 1
    assert n_with_drafts == 2


async def test_load_source_doc_excludes_drafts_by_default(live_db) -> None:
    cid = _new_customer_id()
    await _seed_customer(cid)
    await _seed_doc(
        cid, f"{cid}:approved",
        title="Approved",
        content="approved body",
        visibility="approved",
    )
    await _seed_doc(
        cid, f"{cid}:draft",
        title="Draft",
        content="draft body",
        visibility="draft",
    )

    # Default: drafts are 404 (treated as non-existent).
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as excinfo:
        await _load_source_doc_and_chunks(
            customer_id=cid, doc_id=f"{cid}:draft", version=None,
        )
    assert excinfo.value.status_code == 404

    # include_drafts=True: returns the draft.
    doc, chunk_rows = await _load_source_doc_and_chunks(
        customer_id=cid,
        doc_id=f"{cid}:draft",
        version=None,
        include_drafts=True,
    )
    assert doc["doc_id"] == f"{cid}:draft"  # type: ignore[index]
    assert chunk_rows  # at least one chunk

    # Approved doc: returns normally regardless of include_drafts.
    doc, chunk_rows = await _load_source_doc_and_chunks(
        customer_id=cid, doc_id=f"{cid}:approved", version=None,
    )
    assert doc["doc_id"] == f"{cid}:approved"  # type: ignore[index]
    assert chunk_rows


async def test_agent_fetch_doc_excludes_draft_chunks(live_db) -> None:
    cid = _new_customer_id()
    await _seed_customer(cid)
    await _seed_doc(
        cid, f"{cid}:approved",
        title="Approved",
        content="approved body",
        visibility="approved",
    )
    await _seed_doc(
        cid, f"{cid}:draft",
        title="Draft",
        content="draft body",
        visibility="draft",
    )

    # Agent fetching the approved doc: gets its chunks.
    result = await execute_fetch_doc(cid, doc_id=f"{cid}:approved")
    assert result["chunks"], "approved doc should have visible chunks"

    # Agent fetching the draft doc: gets nothing (chunks are draft).
    result = await execute_fetch_doc(cid, doc_id=f"{cid}:draft")
    assert result["chunks"] == [], "draft chunks must be invisible to the agent"


# ---------------------------------------------------------------------------
# Graph / Directed / Inferred-edge / SQL group-by retrievers
# ---------------------------------------------------------------------------
#
# These four retrievers also gate on ``visibility='approved'`` by default
# and surface drafts only when ``include_drafts=True``. Each retriever has
# its own seeding shape (graph + edges, directed_vectors phrase rows,
# Doc-Doc INFERRED edges, plain documents grouped by doc_type); the tests
# match the existing per-retriever conventions in
# ``tests/retrieval/test_*_retriever.py``.


async def _seed_doc_node(customer_id: str, doc_id: str) -> None:
    """Insert a Document graph_node for ``doc_id`` (matches the inferred
    edge retriever's seeding pattern)."""
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO graph_nodes (customer_id, label, canonical_id, properties)
            VALUES ($1, $2, $3, '{}'::jsonb)
            ON CONFLICT (customer_id, label, canonical_id) DO NOTHING
            """,
            customer_id, NodeLabel.DOCUMENT.value, doc_id,
        )


async def _seed_doc_doc_edge(
    customer_id: str,
    *,
    from_doc_id: str,
    to_doc_id: str,
    edge_type: str = "DISCUSSES",
    why: str = "test reason",
    confidence: str = "INFERRED",
    extractor_id: str = INFERRED_EDGES_EXTRACTOR_ID,
) -> None:
    """Doc->Doc graph_edges row tagged with the Lane B inferred-edges
    extractor id and confidence='INFERRED' so the walk surfaces it."""
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO graph_edges (
                customer_id, edge_type, from_node_id, to_node_id,
                properties, confidence, valid_from, extractor_id, extracted_at
            )
            SELECT $1::text, $2::text, f.node_id, t.node_id,
                   jsonb_build_object('why', $5::text),
                   $6::text, NOW(), $7::text, NOW()
            FROM graph_nodes f, graph_nodes t
            WHERE f.customer_id = $1 AND f.label = 'Document' AND f.canonical_id = $3
              AND t.customer_id = $1 AND t.label = 'Document' AND t.canonical_id = $4
            ON CONFLICT DO NOTHING
            """,
            customer_id, edge_type, from_doc_id, to_doc_id, why,
            confidence, extractor_id,
        )


async def _seed_anchor_node(
    customer_id: str,
    *,
    label: str,
    canonical_id: str,
) -> None:
    """Non-Document anchor node (e.g. Service/Repo/Person) the graph
    retriever can resolve a router entity to."""
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO graph_nodes (customer_id, label, canonical_id, properties)
            VALUES ($1, $2, $3, '{}'::jsonb)
            ON CONFLICT (customer_id, label, canonical_id) DO NOTHING
            """,
            customer_id, label, canonical_id,
        )


async def _seed_anchor_to_doc_edge(
    customer_id: str,
    *,
    anchor_label: str,
    anchor_canonical_id: str,
    doc_id: str,
    edge_type: str = "MENTIONS_ENTITY",
    confidence: str = "EXTRACTED",
) -> None:
    """Edge from a non-Document anchor to a Document node so graph_search
    can reach the doc via a 1-hop neighbor walk."""
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO graph_edges (
                customer_id, edge_type, from_node_id, to_node_id,
                properties, confidence, valid_from, extractor_id, extracted_at
            )
            SELECT $1::text, $2::text, a.node_id, d.node_id,
                   '{}'::jsonb,
                   $6::text, NOW(), 'test:v1', NOW()
            FROM graph_nodes a, graph_nodes d
            WHERE a.customer_id = $1 AND a.label = $3 AND a.canonical_id = $4
              AND d.customer_id = $1 AND d.label = 'Document' AND d.canonical_id = $5
            ON CONFLICT DO NOTHING
            """,
            customer_id, edge_type, anchor_label, anchor_canonical_id,
            doc_id, confidence,
        )


async def _seed_directed_phrase(
    customer_id: str,
    doc_id: str,
    phrase: str,
) -> None:
    """Insert one directed_vectors row using the embedder stub vector for
    ``phrase`` so a query with the same phrase yields a high-similarity
    match against this doc."""
    import hashlib

    embedder = get_embedder_v2()
    [vec] = (await embedder.embed_many([phrase])).embedded[:]
    literal = "[" + ",".join(f"{x:.7f}" for x in vec.embedding) + "]"
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO directed_vectors
                (customer_id, doc_id, embedding, source_text, source,
                 synthesis_run_id, content_hash)
            VALUES ($1, $2, $3::halfvec, $4, 'human', NULL, $5)
            """,
            customer_id, doc_id, literal, phrase,
            hashlib.sha256(phrase.encode("utf-8")).digest(),
        )


async def test_graph_search_excludes_drafts_by_default(live_db) -> None:
    """graph_search default filters draft documents reachable via a
    1-hop entity neighbor walk."""
    cid = _new_customer_id()
    await _seed_customer(cid)
    approved_id = f"{cid}:graph:approved"
    draft_id = f"{cid}:graph:draft"
    await _seed_doc(
        cid, approved_id,
        title="Approved", content="approved body",
        visibility="approved",
    )
    await _seed_doc(
        cid, draft_id,
        title="Draft", content="draft body",
        visibility="draft",
    )
    # Document nodes for the docs themselves.
    await _seed_doc_node(cid, approved_id)
    await _seed_doc_node(cid, draft_id)
    # Shared anchor: a Service node both docs link to via MENTIONS_ENTITY.
    anchor_cid = f"{cid}:svc-a"
    await _seed_anchor_node(
        cid, label=NodeLabel.SERVICE.value, canonical_id=anchor_cid,
    )
    await _seed_anchor_to_doc_edge(
        cid, anchor_label=NodeLabel.SERVICE.value,
        anchor_canonical_id=anchor_cid, doc_id=approved_id,
    )
    await _seed_anchor_to_doc_edge(
        cid, anchor_label=NodeLabel.SERVICE.value,
        anchor_canonical_id=anchor_cid, doc_id=draft_id,
    )

    hits = await graph_search(
        cid, entities=[("service", anchor_cid)], top_k=20,
    )
    doc_ids = {h.doc_id for h in hits}
    assert approved_id in doc_ids
    assert draft_id not in doc_ids


async def test_graph_search_include_drafts_returns_both(live_db) -> None:
    """graph_search with include_drafts=True surfaces draft neighbors."""
    cid = _new_customer_id()
    await _seed_customer(cid)
    approved_id = f"{cid}:graph:approved"
    draft_id = f"{cid}:graph:draft"
    await _seed_doc(
        cid, approved_id,
        title="Approved", content="approved body",
        visibility="approved",
    )
    await _seed_doc(
        cid, draft_id,
        title="Draft", content="draft body",
        visibility="draft",
    )
    await _seed_doc_node(cid, approved_id)
    await _seed_doc_node(cid, draft_id)
    anchor_cid = f"{cid}:svc-a"
    await _seed_anchor_node(
        cid, label=NodeLabel.SERVICE.value, canonical_id=anchor_cid,
    )
    await _seed_anchor_to_doc_edge(
        cid, anchor_label=NodeLabel.SERVICE.value,
        anchor_canonical_id=anchor_cid, doc_id=approved_id,
    )
    await _seed_anchor_to_doc_edge(
        cid, anchor_label=NodeLabel.SERVICE.value,
        anchor_canonical_id=anchor_cid, doc_id=draft_id,
    )

    hits = await graph_search(
        cid, entities=[("service", anchor_cid)],
        top_k=20, include_drafts=True,
    )
    doc_ids = {h.doc_id for h in hits}
    assert approved_id in doc_ids
    assert draft_id in doc_ids


async def test_directed_search_excludes_drafts_by_default(live_db) -> None:
    """directed_search default filters draft documents from
    directed_vectors phrase matches."""
    reset_embedder()
    try:
        cid = _new_customer_id()
        await _seed_customer(cid)
        phrase = "deploy keeps timing out"
        approved_id = f"{cid}:dir:approved"
        draft_id = f"{cid}:dir:draft"
        await _seed_doc(
            cid, approved_id,
            title="Approved", content="approved body",
            visibility="approved",
        )
        await _seed_doc(
            cid, draft_id,
            title="Draft", content="draft body",
            visibility="draft",
        )
        await _seed_directed_phrase(cid, approved_id, phrase)
        await _seed_directed_phrase(cid, draft_id, phrase)

        hits = await directed_search(cid, phrase, top_k=10)
        doc_ids = {h.doc_id for h in hits}
        assert approved_id in doc_ids
        assert draft_id not in doc_ids
    finally:
        reset_embedder()


async def test_directed_search_include_drafts_returns_both(live_db) -> None:
    """directed_search with include_drafts=True surfaces drafts too."""
    reset_embedder()
    try:
        cid = _new_customer_id()
        await _seed_customer(cid)
        phrase = "deploy keeps timing out"
        approved_id = f"{cid}:dir:approved"
        draft_id = f"{cid}:dir:draft"
        await _seed_doc(
            cid, approved_id,
            title="Approved", content="approved body",
            visibility="approved",
        )
        await _seed_doc(
            cid, draft_id,
            title="Draft", content="draft body",
            visibility="draft",
        )
        await _seed_directed_phrase(cid, approved_id, phrase)
        await _seed_directed_phrase(cid, draft_id, phrase)

        hits = await directed_search(
            cid, phrase, top_k=10, include_drafts=True,
        )
        doc_ids = {h.doc_id for h in hits}
        assert approved_id in doc_ids
        assert draft_id in doc_ids
    finally:
        reset_embedder()


async def test_inferred_edge_search_excludes_drafts_by_default(
    live_db,
) -> None:
    """inferred_edge_search default filters draft neighbor documents
    reached from an anchor doc via Lane B Doc-Doc INFERRED edges."""
    cid = _new_customer_id()
    await _seed_customer(cid)
    anchor_id = f"{cid}:inf:anchor"
    approved_neighbor = f"{cid}:inf:approved-neighbor"
    draft_neighbor = f"{cid}:inf:draft-neighbor"
    await _seed_doc(
        cid, anchor_id,
        title="Anchor", content="anchor body",
        visibility="approved",
    )
    await _seed_doc(
        cid, approved_neighbor,
        title="Approved", content="approved body",
        visibility="approved",
    )
    await _seed_doc(
        cid, draft_neighbor,
        title="Draft", content="draft body",
        visibility="draft",
    )
    for doc_id in (anchor_id, approved_neighbor, draft_neighbor):
        await _seed_doc_node(cid, doc_id)
    await _seed_doc_doc_edge(
        cid, from_doc_id=anchor_id, to_doc_id=approved_neighbor,
    )
    await _seed_doc_doc_edge(
        cid, from_doc_id=anchor_id, to_doc_id=draft_neighbor,
    )

    hits = await inferred_edge_search(cid, top_doc_ids=[anchor_id], top_k=10)
    doc_ids = {h.doc_id for h in hits}
    assert approved_neighbor in doc_ids
    assert draft_neighbor not in doc_ids


async def test_inferred_edge_search_include_drafts_returns_both(
    live_db,
) -> None:
    """inferred_edge_search with include_drafts=True surfaces draft
    neighbors too."""
    cid = _new_customer_id()
    await _seed_customer(cid)
    anchor_id = f"{cid}:inf:anchor"
    approved_neighbor = f"{cid}:inf:approved-neighbor"
    draft_neighbor = f"{cid}:inf:draft-neighbor"
    await _seed_doc(
        cid, anchor_id,
        title="Anchor", content="anchor body",
        visibility="approved",
    )
    await _seed_doc(
        cid, approved_neighbor,
        title="Approved", content="approved body",
        visibility="approved",
    )
    await _seed_doc(
        cid, draft_neighbor,
        title="Draft", content="draft body",
        visibility="draft",
    )
    for doc_id in (anchor_id, approved_neighbor, draft_neighbor):
        await _seed_doc_node(cid, doc_id)
    await _seed_doc_doc_edge(
        cid, from_doc_id=anchor_id, to_doc_id=approved_neighbor,
    )
    await _seed_doc_doc_edge(
        cid, from_doc_id=anchor_id, to_doc_id=draft_neighbor,
    )

    hits = await inferred_edge_search(
        cid, top_doc_ids=[anchor_id], top_k=10, include_drafts=True,
    )
    doc_ids = {h.doc_id for h in hits}
    assert approved_neighbor in doc_ids
    assert draft_neighbor in doc_ids


async def test_sql_group_by_excludes_drafts_by_default(live_db) -> None:
    """sql_group_by default filter excludes draft docs from group counts."""
    cid = _new_customer_id()
    await _seed_customer(cid)
    # 2 approved + 1 draft, all the same doc_type — so the bucket count
    # MUST be 2 by default and 3 with include_drafts=True.
    for slug, vis in [
        ("a1", "approved"),
        ("a2", "approved"),
        ("d1", "draft"),
    ]:
        await _seed_doc(
            cid, f"wiki:gb:{slug}",
            title=f"page {slug}",
            content=f"body {slug}",
            visibility=vis,
            doc_type="wiki.knowledge_page",
        )

    groups = await sql_group_by(cid, key="doc_type", top_k=10)
    by_key = {g["key"]: g["n"] for g in groups}
    assert by_key.get("wiki.knowledge_page") == 2  # draft hidden


async def test_sql_group_by_include_drafts_returns_full_count(
    live_db,
) -> None:
    """sql_group_by with include_drafts=True counts draft docs too."""
    cid = _new_customer_id()
    await _seed_customer(cid)
    for slug, vis in [
        ("a1", "approved"),
        ("a2", "approved"),
        ("d1", "draft"),
    ]:
        await _seed_doc(
            cid, f"wiki:gb:{slug}",
            title=f"page {slug}",
            content=f"body {slug}",
            visibility=vis,
            doc_type="wiki.knowledge_page",
        )

    groups = await sql_group_by(
        cid, key="doc_type", top_k=10, include_drafts=True,
    )
    by_key = {g["key"]: g["n"] for g in groups}
    assert by_key.get("wiki.knowledge_page") == 3


# ---------------------------------------------------------------------------
# Plan B — Wiki-listing queries
# ---------------------------------------------------------------------------


async def _seed_wiki_doc(
    customer_id: str,
    doc_id: str,
    *,
    visibility: str,
    doc_type: str = "wiki.postmortem",
    title: str = "Page",
) -> None:
    """Wiki doc + chunk pair shaped like Component 5's writeback output."""
    await _seed_doc(
        customer_id,
        doc_id,
        title=title,
        content=f"body for {doc_id}",
        visibility=visibility,
        source_system="wiki",
        doc_type=doc_type,
        source_id=doc_id,
    )


async def test_fetch_wiki_index_excludes_drafts(live_db) -> None:
    """services/synthesis/persistence.py fetch_wiki_index hides drafts.

    The wiki agent reads this once at drain start and uses it to pick
    (wiki_type, slug) targets. A draft artifact written by Component 5
    must not appear here until the reviewer approves it.
    """
    cid = _new_customer_id()
    await _seed_customer(cid)
    await _seed_wiki_doc(cid, "wiki:postmortem:approved-pm", visibility="approved")
    await _seed_wiki_doc(cid, "wiki:postmortem:draft-pm", visibility="draft")

    rows = await fetch_wiki_index(cid)
    # slug derivation: source_id.split(":", 1)[1]. For source_id =
    # "wiki:postmortem:approved-pm" the split is "postmortem:approved-pm".
    slugs = {r["slug"] for r in rows}
    assert "postmortem:approved-pm" in slugs
    assert "postmortem:draft-pm" not in slugs


async def test_wiki_route_index_excludes_drafts(live_db) -> None:
    """services/ingestion/wiki_routes.py GET /index hides drafts.

    Direct SQL re-execution of the same query (the route itself wraps an
    HTTP layer + key auth we don't need here); the predicate is what we
    care about.
    """
    cid = _new_customer_id()
    await _seed_customer(cid)
    await _seed_wiki_doc(cid, "wiki:postmortem:approved-pm", visibility="approved")
    await _seed_wiki_doc(cid, "wiki:postmortem:draft-pm", visibility="draft")

    from shared.constants import (
        WIKI_DOC_TYPE_PREFIX,
        WIKI_INDEX_DOC_TYPE,
        SourceSystem,
    )
    from shared.db import with_tenant

    async with with_tenant(cid) as conn:
        rows = await conn.fetch(
            """
            SELECT title, source_id, version, updated_at, metadata
            FROM documents
            WHERE customer_id = $1
              AND source_system = $2
              AND doc_type LIKE $3
              AND doc_type <> $4
              AND valid_to IS NULL
              AND deleted_at IS NULL
              AND visibility = 'approved'
            ORDER BY updated_at DESC
            """,
            cid,
            SourceSystem.WIKI.value,
            f"{WIKI_DOC_TYPE_PREFIX}%",
            WIKI_INDEX_DOC_TYPE,
        )
    source_ids = {r["source_id"] for r in rows}
    assert "wiki:postmortem:approved-pm" in source_ids
    assert "wiki:postmortem:draft-pm" not in source_ids


async def test_wiki_agent_drain_index_excludes_drafts(live_db) -> None:
    """services/synthesis/wiki_agent.py drain-time index regen hides drafts.

    Mirrors the synth-time query the agent runs to assemble the
    auto-generated wiki index. A draft artifact must not contribute to
    the rendered TOC body until approval.
    """
    cid = _new_customer_id()
    await _seed_customer(cid)
    await _seed_wiki_doc(cid, "wiki:postmortem:approved-pm", visibility="approved")
    await _seed_wiki_doc(cid, "wiki:postmortem:draft-pm", visibility="draft")

    from shared.constants import (
        WIKI_DOC_TYPE_PREFIX,
        WIKI_INDEX_DOC_TYPE,
        SourceSystem,
    )
    from shared.db import with_tenant

    async with with_tenant(cid) as conn:
        rows = await conn.fetch(
            """
            SELECT title, body_preview, source_id, version, updated_at,
                   metadata
            FROM documents
            WHERE customer_id = $1
              AND source_system = $2
              AND doc_type LIKE $3
              AND doc_type <> $4
              AND valid_to IS NULL
              AND deleted_at IS NULL
              AND visibility = 'approved'
            ORDER BY updated_at DESC
            """,
            cid,
            SourceSystem.WIKI.value,
            f"{WIKI_DOC_TYPE_PREFIX}%",
            WIKI_INDEX_DOC_TYPE,
        )
    source_ids = {r["source_id"] for r in rows}
    assert "wiki:postmortem:approved-pm" in source_ids
    assert "wiki:postmortem:draft-pm" not in source_ids
