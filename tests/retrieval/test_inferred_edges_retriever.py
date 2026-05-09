"""Integration tests for `inferred_edge_search`.

The 4th retrieval channel walks Lane B Doc-Doc edges (extractor_id =
'inferred_edges:v1', confidence='INFERRED') from anchor docs and surfaces
linked Documents with their LLM-derived `properties.why` justification.

Cases covered:
1. Two-anchor walk: each anchor links to one neighbor doc, both surface.
2. top_k cap: a 5-edge graph caps at top_k=2.
3. AMBIGUOUS-confidence edges are NOT walked.
4. Edges with extractor_id != 'inferred_edges:v1' are NOT walked.
5. Cross-tenant RLS isolation -- tenant A walking with tenant B's doc IDs
   returns empty.
6. Self-anchor exclusion: an anchor doc never appears as its own neighbor.
7. Edge-type ordering: DISCUSSES outranks RELATES_TO.
8. Multi-anchor neighbor: a doc reached from anchors at ranks 1 AND 2
   keeps the BEST anchor (rank 1).
9. Dampened score: a rank-1 anchor produces 0.5 * 1/(1+1) = 0.25 with
   default dampening=0.5.
10. Empty input short-circuits without a SQL call.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from services.retrieval.retrievers.inferred_edges import (
    INFERRED_EDGES_EXTRACTOR_ID,
    inferred_edge_search,
)
from shared.config import Settings, get_settings
from shared.constants import EdgeType, NodeLabel
from shared.db import raw_conn

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _patch_settings(monkeypatch, settings: Settings) -> None:
    monkeypatch.setenv(
        "TOKEN_ENCRYPTION_KEY", settings.token_encryption_key.get_secret_value()
    )
    monkeypatch.setenv("ENVIRONMENT", "local")
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


async def _seed_doc_with_node(
    customer_id: str,
    *,
    doc_id: str,
    title: str = "doc",
    updated_at: datetime | None = None,
) -> None:
    """Seed a documents row + a Document graph_node row (matched on
    canonical_id == doc_id)."""
    if updated_at is None:
        updated_at = datetime(2026, 4, 28, 12, 0, tzinfo=UTC)
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
            doc_id, customer_id, f"commit:{doc_id}", title, updated_at,
        )
        await conn.execute(
            """
            INSERT INTO graph_nodes (customer_id, label, canonical_id, properties)
            VALUES ($1, $2, $3, '{}'::jsonb)
            ON CONFLICT (customer_id, label, canonical_id) DO NOTHING
            """,
            customer_id, NodeLabel.DOCUMENT.value, doc_id,
        )


async def _seed_inferred_edge(
    customer_id: str,
    *,
    from_doc_id: str,
    to_doc_id: str,
    edge_type: str,
    why: str,
    confidence: str = "INFERRED",
    extractor_id: str | None = INFERRED_EDGES_EXTRACTOR_ID,
) -> None:
    """Seed a Doc-Doc graph_edges row. By default carries the Lane B
    extractor_id + confidence='INFERRED' tag; tests override to exercise
    filtering."""
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
            customer_id,
            edge_type,
            from_doc_id,
            to_doc_id,
            why,
            confidence,
            extractor_id,
        )


# ---- tests ----------------------------------------------------------------


async def test_empty_input_returns_empty_no_sql_call() -> None:
    """Empty top_doc_ids -> [] short-circuit, no DB call needed."""
    out = await inferred_edge_search("any-tenant", [])
    assert out == []


async def test_walk_returns_linked_doc_with_why(live_db) -> None:
    """Two anchors, each linked to a distinct neighbor doc via a DISCUSSES
    edge with `properties.why`. Both neighbors surface; each carries
    anchor_doc_id + why."""
    cust = "cust-inferred-walk"
    await _seed_customer(cust)
    await _seed_doc_with_node(cust, doc_id="primary:1")
    await _seed_doc_with_node(cust, doc_id="primary:2")
    await _seed_doc_with_node(cust, doc_id="linked:a")
    await _seed_doc_with_node(cust, doc_id="linked:b")
    await _seed_inferred_edge(
        cust, from_doc_id="primary:1", to_doc_id="linked:a",
        edge_type=EdgeType.DISCUSSES.value,
        why="Both docs cover the auth refactor decision.",
    )
    await _seed_inferred_edge(
        cust, from_doc_id="primary:2", to_doc_id="linked:b",
        edge_type=EdgeType.DISCUSSES.value,
        why="Both reference the migration 0042 plan.",
    )

    out = await inferred_edge_search(
        cust, top_doc_ids=["primary:1", "primary:2"], top_k=5
    )
    by_doc = {h.doc_id: h for h in out}
    assert "linked:a" in by_doc
    assert "linked:b" in by_doc
    assert by_doc["linked:a"].anchor_doc_id == "primary:1"
    assert by_doc["linked:a"].why.startswith("Both docs cover")
    assert by_doc["linked:b"].anchor_doc_id == "primary:2"
    assert by_doc["linked:a"].confidence == "INFERRED"
    assert by_doc["linked:a"].edge_type == EdgeType.DISCUSSES.value


async def test_top_k_caps_response_to_two(live_db) -> None:
    """5 outgoing edges from one anchor -> top_k=2 caps at 2."""
    cust = "cust-inferred-cap"
    await _seed_customer(cust)
    await _seed_doc_with_node(cust, doc_id="anchor:1")
    for i in range(5):
        await _seed_doc_with_node(cust, doc_id=f"link:{i}")
        await _seed_inferred_edge(
            cust, from_doc_id="anchor:1", to_doc_id=f"link:{i}",
            edge_type=EdgeType.RELATES_TO.value,
            why=f"link {i} reason",
        )

    out = await inferred_edge_search(cust, top_doc_ids=["anchor:1"], top_k=2)
    assert len(out) == 2


async def test_ambiguous_confidence_not_walked(live_db) -> None:
    """An edge with confidence='AMBIGUOUS' must NOT surface -- the SQL
    filters on `confidence = 'INFERRED'` so anything else is excluded."""
    cust = "cust-inferred-ambig"
    await _seed_customer(cust)
    await _seed_doc_with_node(cust, doc_id="primary:1")
    await _seed_doc_with_node(cust, doc_id="ambiguous-link")
    await _seed_inferred_edge(
        cust, from_doc_id="primary:1", to_doc_id="ambiguous-link",
        edge_type=EdgeType.DISCUSSES.value,
        why="ambiguous reason",
        confidence="AMBIGUOUS",
    )
    out = await inferred_edge_search(cust, top_doc_ids=["primary:1"])
    assert out == []


async def test_extractor_id_mismatch_not_walked(live_db) -> None:
    """An edge from a different extractor_id must NOT surface -- the
    Lane B retriever scopes itself to its own provenance tag."""
    cust = "cust-inferred-extractor"
    await _seed_customer(cust)
    await _seed_doc_with_node(cust, doc_id="primary:1")
    await _seed_doc_with_node(cust, doc_id="other-extractor-link")
    # Same shape as a Lane B edge but from a different extractor.
    await _seed_inferred_edge(
        cust, from_doc_id="primary:1", to_doc_id="other-extractor-link",
        edge_type=EdgeType.DISCUSSES.value,
        why="other extractor reason",
        extractor_id="some_other_extractor:v1",
    )
    out = await inferred_edge_search(cust, top_doc_ids=["primary:1"])
    assert out == []


async def test_rls_does_not_leak_other_tenant_neighbors(live_db) -> None:
    """Tenant A walking with tenant B's doc IDs returns empty -- the
    `with_tenant()` context binds the GUC and RLS hides everything else."""
    await _seed_customer("tenant-A")
    await _seed_customer("tenant-B")
    # Tenant B has a primary doc + an inferred-edge link.
    await _seed_doc_with_node("tenant-B", doc_id="b-primary")
    await _seed_doc_with_node("tenant-B", doc_id="b-link")
    await _seed_inferred_edge(
        "tenant-B", from_doc_id="b-primary", to_doc_id="b-link",
        edge_type=EdgeType.DISCUSSES.value, why="secret",
    )

    # Walk as tenant A using tenant B's primary doc id -- RLS hides
    # the doc node, so the CTE finds no anchors and the result is empty.
    out = await inferred_edge_search("tenant-A", top_doc_ids=["b-primary"])
    assert out == []


async def test_self_anchor_doc_excluded_from_neighbors(live_db) -> None:
    """A doc appearing in top_doc_ids must NEVER reappear as its own
    neighbor, even if there's a self-loop or it's reachable as the to_node
    of another anchor."""
    cust = "cust-inferred-self"
    await _seed_customer(cust)
    await _seed_doc_with_node(cust, doc_id="primary:1")
    await _seed_doc_with_node(cust, doc_id="primary:2")
    # A directional edge from primary:1 -> primary:2. primary:2 is in
    # top_doc_ids itself, so it must not surface as a neighbor.
    await _seed_inferred_edge(
        cust, from_doc_id="primary:1", to_doc_id="primary:2",
        edge_type=EdgeType.DISCUSSES.value, why="reason",
    )
    out = await inferred_edge_search(
        cust, top_doc_ids=["primary:1", "primary:2"]
    )
    cids = {h.doc_id for h in out}
    assert "primary:1" not in cids
    assert "primary:2" not in cids


async def test_edge_type_priority_orders_discusses_before_relates_to(
    live_db,
) -> None:
    """Two neighbors reached via different edge types from the same anchor:
    DISCUSSES (priority 1) outranks RELATES_TO (priority 5)."""
    cust = "cust-inferred-priority"
    await _seed_customer(cust)
    await _seed_doc_with_node(cust, doc_id="anchor:1")
    await _seed_doc_with_node(cust, doc_id="link:discusses")
    await _seed_doc_with_node(cust, doc_id="link:relates")
    await _seed_inferred_edge(
        cust, from_doc_id="anchor:1", to_doc_id="link:discusses",
        edge_type=EdgeType.DISCUSSES.value, why="d",
    )
    await _seed_inferred_edge(
        cust, from_doc_id="anchor:1", to_doc_id="link:relates",
        edge_type=EdgeType.RELATES_TO.value, why="r",
    )
    out = await inferred_edge_search(cust, top_doc_ids=["anchor:1"], top_k=5)
    doc_ids = [h.doc_id for h in out]
    assert doc_ids.index("link:discusses") < doc_ids.index("link:relates")


async def test_multi_anchor_neighbor_keeps_best_anchor(live_db) -> None:
    """One neighbor reached from anchor at rank 1 AND anchor at rank 2 ->
    the surviving record has anchor_rank=1 (best/lowest)."""
    cust = "cust-inferred-multi-anchor"
    await _seed_customer(cust)
    await _seed_doc_with_node(cust, doc_id="anchor:1")
    await _seed_doc_with_node(cust, doc_id="anchor:2")
    await _seed_doc_with_node(cust, doc_id="shared-link")
    await _seed_inferred_edge(
        cust, from_doc_id="anchor:1", to_doc_id="shared-link",
        edge_type=EdgeType.DISCUSSES.value, why="from anchor 1",
    )
    await _seed_inferred_edge(
        cust, from_doc_id="anchor:2", to_doc_id="shared-link",
        edge_type=EdgeType.DISCUSSES.value, why="from anchor 2",
    )
    out = await inferred_edge_search(
        cust, top_doc_ids=["anchor:1", "anchor:2"]
    )
    assert len(out) == 1
    hit = out[0]
    assert hit.doc_id == "shared-link"
    # Best (lowest) anchor_rank wins.
    assert hit.anchor_rank == 1
    assert hit.anchor_doc_id == "anchor:1"


async def test_dampened_score_for_rank_one_anchor(live_db) -> None:
    """default dampening=0.5, anchor_rank=1 -> score = 0.5 * 1/(1+1) = 0.25."""
    cust = "cust-inferred-score"
    await _seed_customer(cust)
    await _seed_doc_with_node(cust, doc_id="anchor:1")
    await _seed_doc_with_node(cust, doc_id="link:1")
    await _seed_inferred_edge(
        cust, from_doc_id="anchor:1", to_doc_id="link:1",
        edge_type=EdgeType.DISCUSSES.value, why="reason",
    )
    out = await inferred_edge_search(cust, top_doc_ids=["anchor:1"])
    assert len(out) == 1
    assert out[0].score == pytest.approx(0.25, rel=0.001)


async def test_bidirectional_walk_uses_to_node_branch(live_db) -> None:
    """Edges where the anchor is the to_node (not from_node) must still
    surface their from_node neighbor -- the UNION ALL covers both
    directions."""
    cust = "cust-inferred-bidir"
    await _seed_customer(cust)
    await _seed_doc_with_node(cust, doc_id="anchor:1")
    await _seed_doc_with_node(cust, doc_id="upstream-link")
    # Edge points TO the anchor.
    await _seed_inferred_edge(
        cust, from_doc_id="upstream-link", to_doc_id="anchor:1",
        edge_type=EdgeType.DOCUMENTS.value, why="upstream reason",
    )
    out = await inferred_edge_search(cust, top_doc_ids=["anchor:1"])
    assert len(out) == 1
    assert out[0].doc_id == "upstream-link"
    assert out[0].edge_type == EdgeType.DOCUMENTS.value
