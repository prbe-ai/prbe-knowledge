"""Unit tests for degree-maintenance logic in graph_writer.upsert_edges.

Tests use a mock asyncpg Connection to avoid needing a live DB. The mock
simulates the RETURNING rows from the INSERT...ON CONFLICT statement, and
verifies that:

  - A fresh insert -> degree incremented for both endpoints
  - A conflict (ON CONFLICT DO UPDATE) -> degree NOT incremented (no double-count)
  - A batch with mixed inserts+conflicts -> only new edges bump degree
  - A batch of multiple new edges sharing a node -> degree bumped by correct sum
  - An empty edge list -> no SQL executed at all
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from services.ingestion.graph_writer import upsert_edges
from shared.models import GraphEdgeSpec, GraphNodeSpec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _edge(
    edge_type: str = "REFERENCES",
    from_label: str = "Service",
    from_cid: str = "svc-a",
    to_label: str = "Service",
    to_cid: str = "svc-b",
    confidence: str = "EXTRACTED",
) -> GraphEdgeSpec:
    from shared.models import EdgeType, NodeLabel

    return GraphEdgeSpec(
        edge_type=EdgeType(edge_type),
        from_label=NodeLabel(from_label),
        from_canonical_id=from_cid,
        to_label=NodeLabel(to_label),
        to_canonical_id=to_cid,
        confidence=confidence,
    )


def _node_ids(*pairs: tuple[str, str, int]) -> dict[tuple[str, str], int]:
    """Build a node_ids lookup from (label, canonical_id, node_id) triples."""
    return {(label, cid): nid for label, cid, nid in pairs}


def _returning_row(from_node_id: int, to_node_id: int, inserted: bool) -> MagicMock:
    """Simulate an asyncpg Record returned by INSERT...RETURNING."""
    row = MagicMock()
    row.__getitem__ = lambda self, key: {
        "from_node_id": from_node_id,
        "to_node_id": to_node_id,
        "inserted": inserted,
    }[key]
    return row


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_insert_increments_degree_for_both_endpoints() -> None:
    """A genuine INSERT (inserted=True) -> both endpoints get degree++."""
    conn = AsyncMock()
    # fetch() simulates the INSERT RETURNING; execute() is the degree UPDATE
    conn.fetch = AsyncMock(
        return_value=[_returning_row(from_node_id=10, to_node_id=20, inserted=True)]
    )
    conn.execute = AsyncMock()

    node_ids = _node_ids(
        ("Service", "svc-a", 10),
        ("Service", "svc-b", 20),
    )
    await upsert_edges(conn, "cust-1", [_edge()], node_ids, "slack")

    # The INSERT fetch was called once
    assert conn.fetch.call_count == 1

    # The degree UPDATE must have been called
    assert conn.execute.call_count == 1
    update_sql = conn.execute.call_args[0][0]
    assert "degree" in update_sql.lower() and "update" in update_sql.lower()

    # Check the arguments: both node_ids should be in the update call
    update_args = conn.execute.call_args[0]
    inc_node_ids = update_args[1]
    inc_amounts = update_args[2]
    assert set(inc_node_ids) == {10, 20}
    # Each endpoint appears once (one new edge)
    for nid, amt in zip(inc_node_ids, inc_amounts):
        assert amt == 1


@pytest.mark.asyncio
async def test_conflict_does_not_increment_degree() -> None:
    """ON CONFLICT DO UPDATE (inserted=False) -> degree NOT touched."""
    conn = AsyncMock()
    conn.fetch = AsyncMock(
        return_value=[_returning_row(from_node_id=10, to_node_id=20, inserted=False)]
    )
    conn.execute = AsyncMock()

    node_ids = _node_ids(
        ("Service", "svc-a", 10),
        ("Service", "svc-b", 20),
    )
    await upsert_edges(conn, "cust-1", [_edge()], node_ids, "slack")

    # INSERT fetch ran
    assert conn.fetch.call_count == 1
    # Degree UPDATE must NOT have been called (no inserted rows)
    conn.execute.assert_not_called()


@pytest.mark.asyncio
async def test_mixed_batch_only_bumps_new_edges() -> None:
    """Batch with one new edge and one conflict: only new edge bumps degree."""
    conn = AsyncMock()
    # Two rows returned: first is new, second is conflict
    conn.fetch = AsyncMock(
        return_value=[
            _returning_row(from_node_id=10, to_node_id=20, inserted=True),
            _returning_row(from_node_id=30, to_node_id=40, inserted=False),
        ]
    )
    conn.execute = AsyncMock()

    node_ids = _node_ids(
        ("Service", "svc-a", 10),
        ("Service", "svc-b", 20),
        ("Service", "svc-c", 30),
        ("Service", "svc-d", 40),
    )
    edges = [
        _edge(from_cid="svc-a", to_cid="svc-b"),
        _edge(from_cid="svc-c", to_cid="svc-d"),
    ]
    await upsert_edges(conn, "cust-1", edges, node_ids, "slack")

    # Degree UPDATE was called once
    assert conn.execute.call_count == 1
    update_args = conn.execute.call_args[0]
    inc_node_ids = update_args[1]
    # Only nodes 10 and 20 (first edge) should be updated; 30+40 are conflicts
    assert set(inc_node_ids) == {10, 20}


@pytest.mark.asyncio
async def test_shared_node_degree_incremented_by_correct_sum() -> None:
    """Node shared by 3 new edges -> degree bumped by 3 in one UPDATE."""
    # Node 10 is the hub; edges: 10-20, 10-30, 10-40 (all new inserts)
    conn = AsyncMock()
    conn.fetch = AsyncMock(
        return_value=[
            _returning_row(from_node_id=10, to_node_id=20, inserted=True),
            _returning_row(from_node_id=10, to_node_id=30, inserted=True),
            _returning_row(from_node_id=10, to_node_id=40, inserted=True),
        ]
    )
    conn.execute = AsyncMock()

    node_ids = _node_ids(
        ("Service", "hub", 10),
        ("Service", "a", 20),
        ("Service", "b", 30),
        ("Service", "c", 40),
    )
    edges = [
        _edge(from_cid="hub", to_cid="a"),
        _edge(from_cid="hub", to_cid="b"),
        _edge(from_cid="hub", to_cid="c"),
    ]
    # Need distinct edge_types to avoid dedup collision in upsert_edges
    from shared.models import EdgeType, NodeLabel

    spec_edges = [
        GraphEdgeSpec(
            edge_type=EdgeType("REFERENCES"),
            from_label=NodeLabel("Service"),
            from_canonical_id="hub",
            to_label=NodeLabel("Service"),
            to_canonical_id="a",
            confidence="EXTRACTED",
        ),
        GraphEdgeSpec(
            edge_type=EdgeType("MENTIONS"),
            from_label=NodeLabel("Service"),
            from_canonical_id="hub",
            to_label=NodeLabel("Service"),
            to_canonical_id="b",
            confidence="EXTRACTED",
        ),
        GraphEdgeSpec(
            edge_type=EdgeType("OWNS"),
            from_label=NodeLabel("Service"),
            from_canonical_id="hub",
            to_label=NodeLabel("Service"),
            to_canonical_id="c",
            confidence="EXTRACTED",
        ),
    ]
    await upsert_edges(conn, "cust-1", spec_edges, node_ids, "slack")

    assert conn.execute.call_count == 1
    update_args = conn.execute.call_args[0]
    inc_node_ids = update_args[1]
    inc_amounts = update_args[2]
    degree_map = dict(zip(inc_node_ids, inc_amounts))

    # Hub node 10 is endpoint of 3 new edges
    assert degree_map[10] == 3
    # Leaf nodes 20, 30, 40 are each endpoints of 1 new edge
    assert degree_map[20] == 1
    assert degree_map[30] == 1
    assert degree_map[40] == 1


@pytest.mark.asyncio
async def test_empty_edge_list_returns_zero_no_sql() -> None:
    """Empty edges list -> return 0, no SQL calls."""
    conn = AsyncMock()
    result = await upsert_edges(conn, "cust-1", [], {}, "slack")
    assert result == 0
    conn.fetch.assert_not_called()
    conn.execute.assert_not_called()
