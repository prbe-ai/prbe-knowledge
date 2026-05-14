"""Regression tests: upsert_edges back-compat for extractor_id/extracted_at.

CRITICAL: Existing EXTRACTED writes (all callers that don't pass
extractor_id/extracted_at) must still produce NULL extractor_id and
NULL extracted_at. The new optional parameters must default to NULL
without changing any existing behavior.

These tests exercise graph_writer.upsert_edges directly without a live DB
(pure unit tests using mocked connections) so they're fast and always run.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from services.ingestion.graph_writer import upsert_edges
from shared.constants import EdgeType, NodeLabel
from shared.models import GraphEdgeSpec

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _edge(
    edge_type: EdgeType = EdgeType.MENTIONS,
    from_label: NodeLabel = NodeLabel.DOCUMENT,
    from_cid: str = "doc1",
    to_label: NodeLabel = NodeLabel.TICKET,
    to_cid: str = "AUTH-123",
    confidence: str = "EXTRACTED",
) -> GraphEdgeSpec:
    return GraphEdgeSpec(
        edge_type=edge_type,
        from_label=from_label,
        from_canonical_id=from_cid,
        to_label=to_label,
        to_canonical_id=to_cid,
        confidence=confidence,
    )


def _node_ids(*pairs: tuple[str, str, int]) -> dict[tuple[str, str], int]:
    """Build node_ids dict from (label, canonical_id, node_id) triples."""
    return {(label, cid): node_id for label, cid, node_id in pairs}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


# NOTE: After Lane A merged, upsert_edges issues the INSERT via conn.fetch
# (to capture RETURNING (xmax = 0) AS inserted for degree maintenance) and
# only uses conn.execute for the follow-up degree UPDATE. So these tests
# inspect conn.fetch.call_args, not conn.execute.call_args.


def _make_conn() -> AsyncMock:
    """Build a conn that returns [] from fetch (no rows -> no degree UPDATE).

    `upsert_edges` issues two fetches when there are real edges to write:
    (1) the entity_aliases lookup (no rows here = no rewrite) and
    (2) the INSERT ... RETURNING. Both return [] so the post-INSERT degree
    UPDATE never runs.
    """
    conn = AsyncMock()
    conn.fetch = AsyncMock(return_value=[])
    conn.execute = AsyncMock()
    return conn


def _insert_call(conn: AsyncMock):
    """Return the conn.fetch call_args corresponding to the INSERT.

    `_fetch_aliases` issues a SELECT against entity_aliases first; the
    INSERT is the call whose SQL mentions `INSERT INTO graph_edges`.
    """
    for call in conn.fetch.call_args_list:
        sql = call[0][0]
        if "INSERT INTO graph_edges" in sql:
            return call
    raise AssertionError("conn.fetch was never invoked with the graph_edges INSERT")


@pytest.mark.asyncio
async def test_upsert_edges_null_extractor_id_by_default() -> None:
    """When extractor_id/extracted_at are not passed, they default to NULL.

    The INSERT must be called with $10=None and $11=None.
    """
    conn = _make_conn()

    edges = [_edge()]
    node_ids = _node_ids(
        ("Document", "doc1", 1),
        ("Ticket", "AUTH-123", 2),
    )

    await upsert_edges(conn, "cust-test", edges, node_ids, "slack")

    call_args = _insert_call(conn)[0]
    # Positional args: (sql, customer_id, source_system, edge_types, from_ids,
    #   to_ids, properties_json, valid_from_list, valid_to_list, confidences,
    #   extractor_id, extracted_at, aliased_from_list, aliased_to_list)
    # Index: 0=sql, 1=$1, ..., 9=$9=confidences, 10=$10=extractor_id, 11=$11=extracted_at
    assert call_args[10] is None, f"extractor_id should be None, got {call_args[10]}"
    assert call_args[11] is None, f"extracted_at should be None, got {call_args[11]}"


@pytest.mark.asyncio
async def test_upsert_edges_with_extractor_id_passed() -> None:
    """When extractor_id/extracted_at are passed, they are forwarded to INSERT."""
    conn = _make_conn()

    edges = [_edge(confidence="INFERRED")]
    node_ids = _node_ids(
        ("Document", "doc1", 1),
        ("Ticket", "AUTH-123", 2),
    )
    extracted_at = datetime(2026, 5, 7, 12, 0, 0, tzinfo=UTC)

    await upsert_edges(
        conn,
        "cust-test",
        edges,
        node_ids,
        "inferred_edges:v1",
        extractor_id="inferred_edges:v1",
        extracted_at=extracted_at,
    )

    call_args = _insert_call(conn)[0]
    assert call_args[10] == "inferred_edges:v1"
    assert call_args[11] == extracted_at


@pytest.mark.asyncio
async def test_upsert_edges_empty_list_returns_zero() -> None:
    """Empty edge list returns 0 without calling execute or fetch."""
    conn = _make_conn()

    result = await upsert_edges(conn, "cust-test", [], {}, "slack")

    assert result == 0
    conn.fetch.assert_not_called()
    conn.execute.assert_not_called()


@pytest.mark.asyncio
async def test_upsert_edges_skips_unknown_endpoints() -> None:
    """Edges whose endpoints are not in node_ids are silently skipped."""
    conn = _make_conn()

    edges = [_edge(from_cid="unknown-doc", to_cid="AUTH-123")]
    node_ids = _node_ids(("Ticket", "AUTH-123", 2))
    # "Document:unknown-doc" is NOT in node_ids

    result = await upsert_edges(conn, "cust-test", edges, node_ids, "slack")

    # All edges dropped (endpoint not found). One fetch fires regardless of
    # endpoint resolution — the entity_aliases lookup at the top of
    # upsert_edges runs before dedup — but no INSERT is issued.
    assert result == 0
    for call in conn.fetch.call_args_list:
        assert "INSERT INTO graph_edges" not in call[0][0]
    conn.execute.assert_not_called()


@pytest.mark.asyncio
async def test_upsert_edges_extracted_confidence_never_demoted() -> None:
    """REGRESSION: an existing EXTRACTED edge must not be demoted by a later
    INFERRED write. The SQL ON CONFLICT clause must preserve the higher tier.

    We can't test the SQL logic here (no live DB), but we verify that the
    confidence value passed to the INSERT matches the edge's confidence
    (so the DB-level CASE expression has the right input).
    """
    conn = _make_conn()

    edges = [_edge(confidence="EXTRACTED")]
    node_ids = _node_ids(
        ("Document", "doc1", 1),
        ("Ticket", "AUTH-123", 2),
    )

    await upsert_edges(conn, "cust-test", edges, node_ids, "slack")

    call_args = _insert_call(conn)[0]
    # Positional args (0=sql, then $1..$13 at indices 1..13):
    #   0=sql, 1=customer_id, 2=source_system, 3=edge_types, 4=from_ids,
    #   5=to_ids, 6=properties, 7=valid_from, 8=valid_to, 9=confidences,
    #   10=extractor_id, 11=extracted_at, 12=aliased_from_list, 13=aliased_to_list
    confidences_list = call_args[9]
    assert confidences_list == ["EXTRACTED"]
