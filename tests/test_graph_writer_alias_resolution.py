"""Graph_writer alias-resolution invariants.

Pins the behaviour we wire into upsert_nodes + upsert_edges so post-merge
webhook ingest correctly routes aliased canonical_ids to the primary.

Invariants:
  * Empty entity_aliases → no rewrite. Pre-merge ingest is a no-op for
    the new code path.
  * upsert_nodes with an aliased canonical_id → INSERT lands on the
    primary's (label, canonical_id) row.
  * upsert_edges with an aliased endpoint → INSERT row's
    from_node_id (or to_node_id) is the primary's; aliased_from_canonical_id
    (or aliased_to) is set to the original alias.
  * upsert_edges where both endpoints resolve to the same canonical
    (post-merge self-loop) → row is dropped, dropped['self_edge_post_alias']
    counter is incremented.
"""

from __future__ import annotations

import uuid

import pytest

from services.ingestion.graph_writer import upsert_edges, upsert_nodes
from shared.constants import EdgeType, NodeLabel
from shared.db import raw_conn, with_tenant
from shared.models import GraphEdgeSpec, GraphNodeSpec


async def _seed_customer(conn, customer_id: str) -> None:
    await conn.execute(
        "INSERT INTO customers(customer_id, display_name, api_key_hash) "
        "VALUES ($1, 'mig', 'mig-hash') ON CONFLICT DO NOTHING",
        customer_id,
    )


async def _insert_audit(conn, *, customer_id, primary, aliases) -> uuid.UUID:
    merge_id = uuid.uuid4()
    await conn.execute(
        """
        INSERT INTO entity_merge_audit
          (merge_id, customer_id, label, primary_canonical_id,
           merged_alias_canonical_ids, performed_by_user_id)
        VALUES ($1, $2, 'Person', $3, $4, $5)
        """,
        merge_id, customer_id, primary, aliases, uuid.uuid4(),
    )
    return merge_id


async def _insert_alias_row(conn, *, customer_id, alias, primary, merge_id) -> None:
    await conn.execute(
        """
        INSERT INTO entity_aliases
          (customer_id, label, alias_canonical_id, primary_canonical_id, merge_id)
        VALUES ($1, 'Person', $2, $3, $4)
        """,
        customer_id, alias, primary, merge_id,
    )


# ---------------------------------------------------------------------------
# upsert_nodes alias resolution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_nodes_no_aliases_no_rewrite(live_db) -> None:
    cust = "alres-noop"
    async with raw_conn() as conn:
        await _seed_customer(conn, cust)
    async with with_tenant(cust) as conn:
        node_ids = await upsert_nodes(
            conn,
            customer_id=cust,
            nodes=[GraphNodeSpec(label=NodeLabel.PERSON, canonical_id="richardwei6")],
            source_system="github",
        )
        assert ("Person", "richardwei6") in node_ids
        row = await conn.fetchrow(
            "SELECT canonical_id FROM graph_nodes WHERE node_id = $1",
            node_ids[("Person", "richardwei6")],
        )
        assert row["canonical_id"] == "richardwei6"


@pytest.mark.asyncio
async def test_upsert_nodes_rewrites_aliased_canonical_id(live_db) -> None:
    cust = "alres-node-rewrite"
    async with raw_conn() as conn:
        await _seed_customer(conn, cust)
    async with with_tenant(cust) as conn:
        # Pre-seed the primary so the alias row's FK pre-check is happy.
        await conn.execute(
            "INSERT INTO graph_nodes(customer_id, label, canonical_id) "
            "VALUES ($1, 'Person', 'richardwei6')",
            cust,
        )
        merge_id = await _insert_audit(
            conn, customer_id=cust, primary="richardwei6",
            aliases=["mahit@prbe.ai"],
        )
        await _insert_alias_row(
            conn, customer_id=cust, alias="mahit@prbe.ai",
            primary="richardwei6", merge_id=merge_id,
        )
        # Incoming webhook: ingest "mahit@prbe.ai" — should resolve.
        node_ids = await upsert_nodes(
            conn,
            customer_id=cust,
            nodes=[GraphNodeSpec(label=NodeLabel.PERSON, canonical_id="mahit@prbe.ai")],
            source_system="github",
        )
        # The returned map keys on the ORIGINAL (label, canonical_id) for callers,
        # but the underlying graph_nodes row is the primary's.
        # Implementations may choose either convention; assert the underlying row.
        assert len(node_ids) == 1
        node_id = next(iter(node_ids.values()))
        row = await conn.fetchrow(
            "SELECT canonical_id FROM graph_nodes WHERE node_id = $1",
            node_id,
        )
        assert row["canonical_id"] == "richardwei6"
        # No row was created with canonical_id='mahit@prbe.ai'.
        leaked = await conn.fetchrow(
            "SELECT 1 FROM graph_nodes "
            "WHERE customer_id = $1 AND label = 'Person' AND canonical_id = 'mahit@prbe.ai'",
            cust,
        )
        assert leaked is None


# ---------------------------------------------------------------------------
# upsert_edges alias resolution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_edges_rewrites_aliased_endpoint(live_db) -> None:
    cust = "alres-edge-rewrite"
    async with raw_conn() as conn:
        await _seed_customer(conn, cust)
    async with with_tenant(cust) as conn:
        await conn.execute(
            "INSERT INTO graph_nodes(customer_id, label, canonical_id) "
            "VALUES ($1, 'Person', 'richardwei6'), ($1, 'Document', 'doc-1')",
            cust,
        )
        merge_id = await _insert_audit(
            conn, customer_id=cust, primary="richardwei6",
            aliases=["mahit@prbe.ai"],
        )
        await _insert_alias_row(
            conn, customer_id=cust, alias="mahit@prbe.ai",
            primary="richardwei6", merge_id=merge_id,
        )
        # Upsert nodes first (graph_writer's normal call sequence).
        node_ids = await upsert_nodes(
            conn,
            customer_id=cust,
            nodes=[
                GraphNodeSpec(label=NodeLabel.PERSON, canonical_id="mahit@prbe.ai"),
                GraphNodeSpec(label=NodeLabel.DOCUMENT, canonical_id="doc-1"),
            ],
            source_system="github",
        )
        # Inbound edge from the (aliased) Slack user → doc-1.
        await upsert_edges(
            conn,
            customer_id=cust,
            edges=[
                GraphEdgeSpec(
                    edge_type=EdgeType.AUTHORED,
                    from_label=NodeLabel.PERSON,
                    from_canonical_id="mahit@prbe.ai",
                    to_label=NodeLabel.DOCUMENT,
                    to_canonical_id="doc-1",
                ),
            ],
            node_ids=node_ids,
            source_system="github",
        )
        # Assert the edge landed on the primary's node and stamped aliased_from.
        rows = await conn.fetch(
            """
            SELECT ge.aliased_from_canonical_id, ge.aliased_to_canonical_id,
                   gn_from.canonical_id AS from_canonical
            FROM graph_edges ge
            JOIN graph_nodes gn_from ON gn_from.node_id = ge.from_node_id
            WHERE ge.customer_id = $1 AND ge.edge_type = 'AUTHORED'
            """,
            cust,
        )
        assert len(rows) == 1
        assert rows[0]["from_canonical"] == "richardwei6"
        assert rows[0]["aliased_from_canonical_id"] == "mahit@prbe.ai"
        assert rows[0]["aliased_to_canonical_id"] is None


@pytest.mark.asyncio
async def test_upsert_edges_drops_self_loop_after_resolution(live_db) -> None:
    """Both endpoints resolve to the same canonical → row dropped."""
    cust = "alres-selfloop"
    async with raw_conn() as conn:
        await _seed_customer(conn, cust)
    async with with_tenant(cust) as conn:
        await conn.execute(
            "INSERT INTO graph_nodes(customer_id, label, canonical_id) "
            "VALUES ($1, 'Person', 'richardwei6')",
            cust,
        )
        merge_id = await _insert_audit(
            conn, customer_id=cust, primary="richardwei6",
            aliases=["mahit@prbe.ai", "U07ABC123"],
        )
        await _insert_alias_row(
            conn, customer_id=cust, alias="mahit@prbe.ai",
            primary="richardwei6", merge_id=merge_id,
        )
        await _insert_alias_row(
            conn, customer_id=cust, alias="U07ABC123",
            primary="richardwei6", merge_id=merge_id,
        )
        # Upsert nodes for both aliases — both resolve to richardwei6.
        node_ids = await upsert_nodes(
            conn,
            customer_id=cust,
            nodes=[
                GraphNodeSpec(label=NodeLabel.PERSON, canonical_id="mahit@prbe.ai"),
                GraphNodeSpec(label=NodeLabel.PERSON, canonical_id="U07ABC123"),
            ],
            source_system="github",
        )
        # Try to write Person↔Person edge between two aliases of the same
        # cluster — post-resolution it's a self-loop.
        await upsert_edges(
            conn,
            customer_id=cust,
            edges=[
                GraphEdgeSpec(
                    edge_type=EdgeType.MENTIONS,
                    from_label=NodeLabel.PERSON,
                    from_canonical_id="mahit@prbe.ai",
                    to_label=NodeLabel.PERSON,
                    to_canonical_id="U07ABC123",
                ),
            ],
            node_ids=node_ids,
            source_system="github",
        )
        # Either:
        #   - upsert_edges returns 0 and no row was written, OR
        #   - the dropped counter records the drop (implementation choice).
        # The DB is the authoritative check.
        rows = await conn.fetch(
            "SELECT 1 FROM graph_edges WHERE customer_id = $1 AND edge_type = 'MENTIONS'",
            cust,
        )
        assert rows == [], "self-loop edge should not be persisted"
