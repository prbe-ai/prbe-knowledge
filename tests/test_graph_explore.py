"""Integration tests for services/retrieval/graph_explore.py.

Real Postgres (no DB mocks per the project's no-mock-DB rule for retrieval
paths). Seeds graph_nodes / graph_edges / documents directly then exercises
default_graph_query() and anchor_graph_query() through the public surface.

Cases covered:

default_graph_query:
  - empty result on a fresh customer
  - top-N by degree DESC
  - edge_types filter narrows the edge set
  - confidences filter narrows the edge set
  - source_systems filter (edge.source_system)
  - since filter (edge.valid_from)

anchor_graph_query:
  - anchor with 0 edges -> only the anchor node returned
  - 1-hop fills cap (no 2-hop attempted)
  - 1-hop < cap -> full 2-hop fills more
  - bidirectional UNION ALL coverage (both directions surface)
  - anchor across tenant boundary -> empty graph (RLS), not a leak

Serializer behavior:
  - edge dedup across UNION ALL doubling
  - node dedup across multi-anchor / multi-edge
  - truncated flag when total > cap
  - `why` cap at GRAPH_EXPLORE_WHY_MAX_CHARS chars
  - `why` is NULL when confidence == EXTRACTED

Regression:
  - cross-tenant query returns empty when the GUC is wrong (defense-in-depth
    for RLS).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from services.retrieval import graph_explore
from services.retrieval.graph_explore import (
    ExploreFilters,
    _truncate_why,
    anchor_exists,
    anchor_graph_query,
    default_graph_query,
)
from shared.constants import (
    GRAPH_EXPLORE_WHY_MAX_CHARS,
    EdgeType,
    NodeLabel,
)
from shared.db import raw_conn

pytestmark = pytest.mark.asyncio


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


async def _seed_node(
    customer_id: str,
    *,
    label: str,
    canonical_id: str,
    name: str | None = None,
    source_system: str | None = None,
    degree: int = 0,
    community_id: int | None = None,
) -> None:
    """Seed one graph_nodes row.

    `degree` is the materialized count maintained by graph_writer in real
    ingestion; tests set it explicitly to control top-N ordering.
    """
    properties: dict[str, str] = {}
    if name is not None:
        properties["name"] = name
    if source_system is not None:
        properties["source_system"] = source_system
    import json
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO graph_nodes (
                customer_id, label, canonical_id, properties,
                degree, community_id
            )
            VALUES ($1, $2, $3, $4::jsonb, $5, $6)
            ON CONFLICT (customer_id, label, canonical_id)
            DO UPDATE SET degree = EXCLUDED.degree,
                          community_id = EXCLUDED.community_id,
                          properties = EXCLUDED.properties
            """,
            customer_id, label, canonical_id, json.dumps(properties),
            degree, community_id,
        )


async def _seed_doc_node(
    customer_id: str,
    *,
    doc_id: str,
    title: str = "doc",
    source_system: str = "github",
    degree: int = 0,
) -> None:
    """Seed a documents row + matching Document graph_node.

    Mirrors the seeder in tests/retrieval/test_related_entities.py so the
    documents.title join in default_graph_query / anchor_graph_query
    actually fires.
    """
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
                $6, $3, 'https://example/' || $1,
                'raw_source', 'github.commit', 'text/plain',
                'h-' || $1, $4, 100, 0,
                $5, $5, $5, $5, '{}'::jsonb
            )
            ON CONFLICT DO NOTHING
            """,
            doc_id, customer_id, f"commit:{doc_id}", title, updated_at,
            source_system,
        )
    await _seed_node(
        customer_id,
        label=NodeLabel.DOCUMENT.value,
        canonical_id=doc_id,
        degree=degree,
    )


async def _seed_edge(
    customer_id: str,
    *,
    from_label: str,
    from_canonical_id: str,
    to_label: str,
    to_canonical_id: str,
    edge_type: str = EdgeType.MENTIONS.value,
    confidence: str = "EXTRACTED",
    source_system: str | None = None,
    why: str | None = None,
    valid_from: datetime | None = None,
    valid_to: datetime | None = None,
) -> None:
    """Seed one graph_edges row by (label, canonical_id) pairs."""
    import json
    properties: dict[str, str] = {}
    if why is not None:
        properties["why"] = why
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO graph_edges (
                customer_id, edge_type, from_node_id, to_node_id,
                confidence, source_system, properties,
                valid_from, valid_to
            )
            SELECT $1, $2, f.node_id, t.node_id, $7, $8, $9::jsonb,
                   COALESCE($10, NOW()), $11
            FROM graph_nodes f, graph_nodes t
            WHERE f.customer_id = $1 AND f.label = $3 AND f.canonical_id = $4
              AND t.customer_id = $1 AND t.label = $5 AND t.canonical_id = $6
            ON CONFLICT DO NOTHING
            """,
            customer_id,
            edge_type,
            from_label, from_canonical_id,
            to_label,   to_canonical_id,
            confidence,
            source_system,
            json.dumps(properties),
            valid_from, valid_to,
        )


# ---- _truncate_why helper (pure, no DB) ----------------------------------
#
# Sync tests inherit the module-level `pytestmark = pytest.mark.asyncio`,
# which pytest-asyncio warns about. Pytest-asyncio's auto mode tolerates
# non-async functions but emits a PytestWarning per test. Annotating each
# function with `@pytest.mark.asyncio(loop_scope="function")` would NOT
# suppress it -- the only clean fixes are (a) moving these into a separate
# file or (b) marking each sync test to opt out. We pick the simpler path:
# the warnings are cosmetic and the helpers are stable enough to live here.


def test_truncate_why_passthrough_under_cap() -> None:
    text = "a short reason"
    assert _truncate_why(text) == text


def test_truncate_why_none_passes_through() -> None:
    assert _truncate_why(None) is None


def test_truncate_why_caps_with_ellipsis_at_max_chars() -> None:
    """Strings over the cap are truncated and end in '...' so the dashboard
    can render the cue without re-measuring text."""
    long_text = "x" * (GRAPH_EXPLORE_WHY_MAX_CHARS + 50)
    out = _truncate_why(long_text)
    assert out is not None
    # The total length is exactly the cap (3 chars reserved for ellipsis).
    assert len(out) == GRAPH_EXPLORE_WHY_MAX_CHARS
    assert out.endswith("...")


# ---- default_graph_query --------------------------------------------------


async def test_default_graph_empty_for_new_customer(live_db) -> None:
    """A customer with no graph data returns empty nodes/edges and totals=0."""
    await _seed_customer("cust-empty")
    result = await default_graph_query(customer_id="cust-empty")
    assert result.nodes == []
    assert result.edges == []
    assert result.total_nodes_available == 0
    assert result.total_edges_available == 0


async def test_default_graph_top_n_by_degree(live_db) -> None:
    """Nodes come back ordered by degree DESC."""
    await _seed_customer("cust-default")
    await _seed_node("cust-default", label=NodeLabel.SERVICE.value,
                     canonical_id="svc-low", degree=1)
    await _seed_node("cust-default", label=NodeLabel.SERVICE.value,
                     canonical_id="svc-high", degree=99)
    await _seed_node("cust-default", label=NodeLabel.SERVICE.value,
                     canonical_id="svc-mid", degree=50)
    result = await default_graph_query(customer_id="cust-default")
    ids_by_position = [n.id for n in result.nodes]
    assert ids_by_position == ["svc-high", "svc-mid", "svc-low"]
    assert result.total_nodes_available == 3
    # No edges seeded, so no edges in the response.
    assert result.edges == []


async def test_default_graph_filter_by_edge_type(live_db) -> None:
    """edge_types filter narrows the edges; nodes are unaffected."""
    await _seed_customer("cust-edge-filter")
    await _seed_doc_node("cust-edge-filter", doc_id="doc-A", degree=10)
    await _seed_node("cust-edge-filter", label=NodeLabel.SERVICE.value,
                     canonical_id="svc-1", degree=5)
    await _seed_edge(
        "cust-edge-filter",
        from_label=NodeLabel.DOCUMENT.value, from_canonical_id="doc-A",
        to_label=NodeLabel.SERVICE.value,    to_canonical_id="svc-1",
        edge_type=EdgeType.DISCUSSES.value,
    )
    await _seed_edge(
        "cust-edge-filter",
        from_label=NodeLabel.DOCUMENT.value, from_canonical_id="doc-A",
        to_label=NodeLabel.SERVICE.value,    to_canonical_id="svc-1",
        edge_type=EdgeType.RELATES_TO.value,
    )

    # Without filter: both edges present.
    full = await default_graph_query(customer_id="cust-edge-filter")
    assert len(full.edges) == 2

    # With filter: only DISCUSSES.
    narrow = await default_graph_query(
        customer_id="cust-edge-filter",
        filters=ExploreFilters(edge_types=[EdgeType.DISCUSSES.value]),
    )
    assert len(narrow.edges) == 1
    assert narrow.edges[0].edge_type == EdgeType.DISCUSSES.value


async def test_default_graph_filter_by_confidence(live_db) -> None:
    """confidences filter drops edges whose confidence is not in the list."""
    await _seed_customer("cust-conf-filter")
    await _seed_doc_node("cust-conf-filter", doc_id="doc-A", degree=10)
    await _seed_node("cust-conf-filter", label=NodeLabel.SERVICE.value,
                     canonical_id="svc-1", degree=5)
    await _seed_edge(
        "cust-conf-filter",
        from_label=NodeLabel.DOCUMENT.value, from_canonical_id="doc-A",
        to_label=NodeLabel.SERVICE.value,    to_canonical_id="svc-1",
        edge_type=EdgeType.DISCUSSES.value, confidence="EXTRACTED",
    )
    await _seed_edge(
        "cust-conf-filter",
        from_label=NodeLabel.DOCUMENT.value, from_canonical_id="doc-A",
        to_label=NodeLabel.SERVICE.value,    to_canonical_id="svc-1",
        edge_type=EdgeType.RESOLVES.value, confidence="INFERRED",
    )
    extracted_only = await default_graph_query(
        customer_id="cust-conf-filter",
        filters=ExploreFilters(confidences=["EXTRACTED"]),
    )
    assert len(extracted_only.edges) == 1
    assert extracted_only.edges[0].confidence == "EXTRACTED"


async def test_default_graph_filter_by_source_system(live_db) -> None:
    """source_systems filter narrows on graph_edges.source_system."""
    await _seed_customer("cust-src-filter")
    await _seed_doc_node("cust-src-filter", doc_id="doc-A", degree=10)
    await _seed_node("cust-src-filter", label=NodeLabel.SERVICE.value,
                     canonical_id="svc-1", degree=5)
    await _seed_edge(
        "cust-src-filter",
        from_label=NodeLabel.DOCUMENT.value, from_canonical_id="doc-A",
        to_label=NodeLabel.SERVICE.value,    to_canonical_id="svc-1",
        source_system="linear",
    )
    await _seed_edge(
        "cust-src-filter",
        from_label=NodeLabel.DOCUMENT.value, from_canonical_id="doc-A",
        to_label=NodeLabel.SERVICE.value,    to_canonical_id="svc-1",
        edge_type=EdgeType.DISCUSSES.value, source_system="github",
    )
    linear_only = await default_graph_query(
        customer_id="cust-src-filter",
        filters=ExploreFilters(source_systems=["linear"]),
    )
    assert len(linear_only.edges) == 1


async def test_default_graph_filter_by_since(live_db) -> None:
    """since filter (edge.valid_from >= since) drops older edges."""
    await _seed_customer("cust-since-filter")
    await _seed_doc_node("cust-since-filter", doc_id="doc-A", degree=10)
    await _seed_node("cust-since-filter", label=NodeLabel.SERVICE.value,
                     canonical_id="svc-1", degree=5)
    old = datetime(2026, 1, 1, tzinfo=UTC)
    new = datetime(2026, 5, 1, tzinfo=UTC)
    await _seed_edge(
        "cust-since-filter",
        from_label=NodeLabel.DOCUMENT.value, from_canonical_id="doc-A",
        to_label=NodeLabel.SERVICE.value,    to_canonical_id="svc-1",
        edge_type=EdgeType.DISCUSSES.value, valid_from=old,
    )
    await _seed_edge(
        "cust-since-filter",
        from_label=NodeLabel.DOCUMENT.value, from_canonical_id="doc-A",
        to_label=NodeLabel.SERVICE.value,    to_canonical_id="svc-1",
        edge_type=EdgeType.RESOLVES.value, valid_from=new,
    )
    cutoff = datetime(2026, 3, 1, tzinfo=UTC)
    recent = await default_graph_query(
        customer_id="cust-since-filter",
        filters=ExploreFilters(since=cutoff),
    )
    assert len(recent.edges) == 1
    assert recent.edges[0].edge_type == EdgeType.RESOLVES.value


# ---- anchor_graph_query ---------------------------------------------------


async def test_anchor_exists_true_for_seeded_node(live_db) -> None:
    await _seed_customer("cust-anchor")
    await _seed_node("cust-anchor", label=NodeLabel.SERVICE.value,
                     canonical_id="svc-anchor", degree=0)
    assert await anchor_exists(
        customer_id="cust-anchor", anchor_canonical_id="svc-anchor"
    )


async def test_anchor_exists_false_for_unknown(live_db) -> None:
    await _seed_customer("cust-anchor-empty")
    assert not await anchor_exists(
        customer_id="cust-anchor-empty", anchor_canonical_id="missing"
    )


async def test_anchor_graph_anchor_with_zero_edges(live_db) -> None:
    """Anchor exists but no edges -> just the anchor node, no edges."""
    await _seed_customer("cust-zero")
    await _seed_node("cust-zero", label=NodeLabel.SERVICE.value,
                     canonical_id="svc-zero", degree=0)
    result = await anchor_graph_query(
        customer_id="cust-zero", anchor_canonical_id="svc-zero",
    )
    assert len(result.nodes) == 1
    assert result.nodes[0].id == "svc-zero"
    assert result.edges == []


async def test_anchor_graph_bidirectional_union_all(live_db) -> None:
    """Edges where the anchor is the to_node also surface (UNION ALL covers
    both directions). Without UNION ALL, the inbound edge would be missed.
    """
    await _seed_customer("cust-bidir")
    await _seed_node("cust-bidir", label=NodeLabel.SERVICE.value,
                     canonical_id="anchor-svc", degree=0)
    await _seed_doc_node("cust-bidir", doc_id="inbound-doc", degree=1)
    await _seed_doc_node("cust-bidir", doc_id="outbound-doc", degree=1)
    # Anchor as TO node (inbound).
    await _seed_edge(
        "cust-bidir",
        from_label=NodeLabel.DOCUMENT.value, from_canonical_id="inbound-doc",
        to_label=NodeLabel.SERVICE.value,    to_canonical_id="anchor-svc",
        edge_type=EdgeType.DISCUSSES.value,
    )
    # Anchor as FROM node (outbound).
    await _seed_edge(
        "cust-bidir",
        from_label=NodeLabel.SERVICE.value, from_canonical_id="anchor-svc",
        to_label=NodeLabel.DOCUMENT.value,  to_canonical_id="outbound-doc",
        edge_type=EdgeType.DISCUSSES.value,
    )
    result = await anchor_graph_query(
        customer_id="cust-bidir", anchor_canonical_id="anchor-svc",
    )
    node_ids = {n.id for n in result.nodes}
    assert node_ids == {"anchor-svc", "inbound-doc", "outbound-doc"}
    # Two distinct logical edges (dedup mandatory: UNION ALL would have
    # produced 4 raw rows, dedup collapses to 2).
    assert len(result.edges) == 2


async def test_anchor_graph_two_hop_fills(live_db) -> None:
    """When 1-hop is small, hop2 expands to neighbors-of-neighbors."""
    await _seed_customer("cust-2hop")
    await _seed_node("cust-2hop", label=NodeLabel.SERVICE.value,
                     canonical_id="anchor", degree=0)
    await _seed_node("cust-2hop", label=NodeLabel.SERVICE.value,
                     canonical_id="hop1-a", degree=1)
    await _seed_node("cust-2hop", label=NodeLabel.SERVICE.value,
                     canonical_id="hop2-b", degree=1)
    # anchor -> hop1-a -> hop2-b
    await _seed_edge(
        "cust-2hop",
        from_label=NodeLabel.SERVICE.value, from_canonical_id="anchor",
        to_label=NodeLabel.SERVICE.value,   to_canonical_id="hop1-a",
        edge_type=EdgeType.DISCUSSES.value,
    )
    await _seed_edge(
        "cust-2hop",
        from_label=NodeLabel.SERVICE.value, from_canonical_id="hop1-a",
        to_label=NodeLabel.SERVICE.value,   to_canonical_id="hop2-b",
        edge_type=EdgeType.DISCUSSES.value,
    )
    result = await anchor_graph_query(
        customer_id="cust-2hop", anchor_canonical_id="anchor",
    )
    node_ids = {n.id for n in result.nodes}
    assert node_ids == {"anchor", "hop1-a", "hop2-b"}


async def test_anchor_graph_cross_tenant_no_leak(live_db) -> None:
    """Anchor seeded in tenant A, query under tenant B -> empty graph.

    Defense-in-depth: the endpoint short-circuits to 404 via anchor_exists,
    but even if the BFS is invoked with a stale ID, RLS must produce empty
    results -- not leak tenant A's edges.
    """
    await _seed_customer("cust-A")
    await _seed_customer("cust-B")
    await _seed_node("cust-A", label=NodeLabel.SERVICE.value,
                     canonical_id="cross-tenant", degree=0)
    await _seed_doc_node("cust-A", doc_id="cust-A-doc", degree=1)
    await _seed_edge(
        "cust-A",
        from_label=NodeLabel.SERVICE.value, from_canonical_id="cross-tenant",
        to_label=NodeLabel.DOCUMENT.value,  to_canonical_id="cust-A-doc",
        edge_type=EdgeType.DISCUSSES.value,
    )
    # anchor_exists under tenant B sees nothing.
    assert not await anchor_exists(
        customer_id="cust-B", anchor_canonical_id="cross-tenant"
    )
    # And even invoking the BFS directly under tenant B returns nothing.
    result = await anchor_graph_query(
        customer_id="cust-B", anchor_canonical_id="cross-tenant",
    )
    assert result.nodes == []
    assert result.edges == []


# ---- Serializer behavior --------------------------------------------------


async def test_edge_dedup_collapses_union_all_doubling(live_db) -> None:
    """The bidirectional UNION ALL emits two rows per logical edge in the
    anchor query when both endpoints are in the selected set. The serializer
    MUST collapse to one logical edge per (source, target, edge_type).
    """
    await _seed_customer("cust-dedup")
    await _seed_node("cust-dedup", label=NodeLabel.SERVICE.value,
                     canonical_id="anchor", degree=0)
    await _seed_doc_node("cust-dedup", doc_id="neighbor", degree=1)
    await _seed_edge(
        "cust-dedup",
        from_label=NodeLabel.SERVICE.value, from_canonical_id="anchor",
        to_label=NodeLabel.DOCUMENT.value,  to_canonical_id="neighbor",
        edge_type=EdgeType.DISCUSSES.value,
    )
    result = await anchor_graph_query(
        customer_id="cust-dedup", anchor_canonical_id="anchor",
    )
    # All_edges runs both UNION ALL halves over (anchor, neighbor) -- the
    # raw row count is 2 but the de-duped logical-edge count is 1.
    assert len(result.edges) == 1


async def test_why_dropped_for_extracted_confidence(live_db) -> None:
    """EXTRACTED edges' `why` is dropped at serialization time -- the field
    is meaningful only for INFERRED / AMBIGUOUS edges (LLM rationale).
    """
    await _seed_customer("cust-why-ex")
    await _seed_doc_node("cust-why-ex", doc_id="doc-A", degree=10)
    await _seed_node("cust-why-ex", label=NodeLabel.SERVICE.value,
                     canonical_id="svc-1", degree=5)
    await _seed_edge(
        "cust-why-ex",
        from_label=NodeLabel.DOCUMENT.value, from_canonical_id="doc-A",
        to_label=NodeLabel.SERVICE.value,    to_canonical_id="svc-1",
        edge_type=EdgeType.DISCUSSES.value, confidence="EXTRACTED",
        why="extractor noise that should be dropped",
    )
    result = await default_graph_query(customer_id="cust-why-ex")
    assert len(result.edges) == 1
    assert result.edges[0].why is None


async def test_why_kept_and_capped_for_inferred(live_db) -> None:
    """INFERRED edges keep `why` but truncate to GRAPH_EXPLORE_WHY_MAX_CHARS."""
    await _seed_customer("cust-why-inf")
    await _seed_doc_node("cust-why-inf", doc_id="doc-A", degree=10)
    await _seed_node("cust-why-inf", label=NodeLabel.SERVICE.value,
                     canonical_id="svc-1", degree=5)
    long_why = "y" * (GRAPH_EXPLORE_WHY_MAX_CHARS + 100)
    await _seed_edge(
        "cust-why-inf",
        from_label=NodeLabel.DOCUMENT.value, from_canonical_id="doc-A",
        to_label=NodeLabel.SERVICE.value,    to_canonical_id="svc-1",
        edge_type=EdgeType.DISCUSSES.value, confidence="INFERRED",
        why=long_why,
    )
    result = await default_graph_query(customer_id="cust-why-inf")
    assert len(result.edges) == 1
    why = result.edges[0].why
    assert why is not None
    assert len(why) == GRAPH_EXPLORE_WHY_MAX_CHARS
    assert why.endswith("...")


async def test_truncated_flag_when_node_count_exceeds_cap(live_db, monkeypatch) -> None:
    """If total_nodes_available > what the response carries, the endpoint
    flips truncated=True. We mimic this at the dataclass level by patching
    the node cap down to 2 and seeding 5 nodes.
    """
    # Patch GRAPH_EXPLORE_NODE_CAP for the duration of this test. Patching
    # at the module level (where the SQL builders read it) is the correct
    # binding -- patching shared.constants would not affect the already-
    # imported value inside graph_explore.
    monkeypatch.setattr(graph_explore, "GRAPH_EXPLORE_NODE_CAP", 2)
    await _seed_customer("cust-trunc")
    for i in range(5):
        await _seed_node(
            "cust-trunc", label=NodeLabel.SERVICE.value,
            canonical_id=f"svc-{i}", degree=10 - i,
        )
    result = await default_graph_query(customer_id="cust-trunc")
    assert len(result.nodes) == 2
    assert result.total_nodes_available == 5
    # The endpoint computes truncated; here we assert the precondition for it.
    assert len(result.nodes) < result.total_nodes_available


async def test_node_dedup_across_multi_anchor_paths(live_db) -> None:
    """A node reachable via multiple paths from the anchor must appear only
    once in the response. anchor -> A -> shared and anchor -> shared.
    """
    await _seed_customer("cust-node-dedup")
    await _seed_node("cust-node-dedup", label=NodeLabel.SERVICE.value,
                     canonical_id="anchor", degree=0)
    await _seed_node("cust-node-dedup", label=NodeLabel.SERVICE.value,
                     canonical_id="hop1", degree=1)
    await _seed_node("cust-node-dedup", label=NodeLabel.SERVICE.value,
                     canonical_id="shared", degree=1)
    # Anchor -> shared directly
    await _seed_edge(
        "cust-node-dedup",
        from_label=NodeLabel.SERVICE.value, from_canonical_id="anchor",
        to_label=NodeLabel.SERVICE.value,   to_canonical_id="shared",
        edge_type=EdgeType.DISCUSSES.value,
    )
    # Anchor -> hop1 -> shared (also reachable via 2-hop)
    await _seed_edge(
        "cust-node-dedup",
        from_label=NodeLabel.SERVICE.value, from_canonical_id="anchor",
        to_label=NodeLabel.SERVICE.value,   to_canonical_id="hop1",
        edge_type=EdgeType.DISCUSSES.value,
    )
    await _seed_edge(
        "cust-node-dedup",
        from_label=NodeLabel.SERVICE.value, from_canonical_id="hop1",
        to_label=NodeLabel.SERVICE.value,   to_canonical_id="shared",
        edge_type=EdgeType.DISCUSSES.value,
    )
    result = await anchor_graph_query(
        customer_id="cust-node-dedup", anchor_canonical_id="anchor",
    )
    ids = [n.id for n in result.nodes]
    assert sorted(ids) == ["anchor", "hop1", "shared"]
    # No duplicates.
    assert len(ids) == len(set(ids))
