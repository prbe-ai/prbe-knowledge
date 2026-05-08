"""Regression tests for the flag-gated sort in graph_search().

What this protects
------------------
With SURPRISE_SCORE_ENABLED=true, graph_search() must return hits sorted
by score DESC so the highest-surprise edge lands at rank 1. Without the
sort, hits arrive in heap-scan order and fusion's RRF (1/(k+rank)) gives
the biggest graph-side contribution to an arbitrary neighbor instead of
the most informative one.

Flag-off path must remain a no-op: hits return in DB order with score=1.0,
matching the pre-flag behavior. This file pins both branches.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_row(
    *,
    chunk_id: str,
    source_system: str = "slack",
    confidence: str | None = "INFERRED",
    via_source_system: str | None = "slack",
    community_id: int | None = None,
    via_community: int | None = None,
    degree: int = 3,
    via_degree: int = 3,
    via_label: str = "Service",
    edge_type: str = "REFERENCES",
) -> MagicMock:
    now = datetime.now(UTC)
    data = {
        "chunk_id": chunk_id,
        "doc_id": f"doc-{chunk_id}",
        "doc_version": 1,
        "source_system": source_system,
        "source_url": "http://example.com",
        "title": None,
        "author_id": None,
        "content": "some content",
        "created_at": now,
        "updated_at": now,
        "via_entity": "svc-a",
        "via_label": via_label,
        "edge_type": edge_type,
        "confidence": confidence,
        "via_source_system": via_source_system,
        "community_id": community_id,
        "via_community": via_community,
        "degree": degree,
        "via_degree": via_degree,
    }
    row = MagicMock()
    row.__getitem__ = lambda self, key: data[key]
    return row


def _patched_conn(rows: list) -> MagicMock:
    mock_conn = AsyncMock()
    mock_conn.fetch = AsyncMock(return_value=rows)
    mock_ctx = MagicMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)
    return mock_ctx


@pytest.mark.asyncio
async def test_flag_off_preserves_db_order() -> None:
    """Flag=false: hits come back in DB iteration order, score=1.0.

    Pin pre-flag behavior so a future change can't accidentally shuffle
    the flag-off path. DB returns rows in [low, high] surprise order;
    flag-off must keep that order even though high would sort first.
    """
    import services.retrieval.retrievers.graph as graph_mod

    # Row 1: low surprise (same-source, same-community, INFERRED) -> score 1.0
    low_surprise = _make_row(
        chunk_id="row_first_low_surprise",
        source_system="slack",
        via_source_system="slack",
        community_id=1,
        via_community=1,
        confidence="INFERRED",
    )
    # Row 2: high surprise (cross-source, cross-community, INFERRED) -> score 1.0 (flag off)
    high_surprise = _make_row(
        chunk_id="row_second_high_surprise",
        source_system="github",
        via_source_system="slack",
        community_id=99,
        via_community=1,
        confidence="INFERRED",
    )

    with patch.object(graph_mod, "SURPRISE_SCORE_ENABLED", False):
        ctx = _patched_conn([low_surprise, high_surprise])
        with patch("services.retrieval.retrievers.graph.with_tenant", return_value=ctx):
            hits = await graph_mod.graph_search(
                customer_id="cust-test",
                entities=[("service", "svc-a")],
            )

    assert len(hits) == 2
    # DB order preserved.
    assert hits[0].chunk_id == "row_first_low_surprise"
    assert hits[1].chunk_id == "row_second_high_surprise"
    # Flag-off path: score is the flat 1.0 baseline for every hit.
    assert hits[0].score == pytest.approx(1.0)
    assert hits[1].score == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_flag_on_sorts_by_score_desc() -> None:
    """Flag=true: highest-surprise hit is at rank 1.

    DB returns rows in [low, high] order; with the flag on graph_search
    must sort to [high, low] so RRF gives the big graph contribution to
    the informative cross-source/cross-community edge.
    """
    import services.retrieval.retrievers.graph as graph_mod

    low_surprise = _make_row(
        chunk_id="low",
        source_system="slack",
        via_source_system="slack",
        community_id=1,
        via_community=1,
        confidence="INFERRED",  # 1.25
    )
    high_surprise = _make_row(
        chunk_id="high",
        source_system="github",
        via_source_system="slack",
        community_id=99,
        via_community=1,
        confidence="INFERRED",  # 1.25 * 1.5 (cross-source) * 1.4 (cross-community) = 2.625
    )

    with patch.object(graph_mod, "SURPRISE_SCORE_ENABLED", True):
        # DB delivers low BEFORE high; sort must flip that.
        ctx = _patched_conn([low_surprise, high_surprise])
        with patch("services.retrieval.retrievers.graph.with_tenant", return_value=ctx):
            hits = await graph_mod.graph_search(
                customer_id="cust-test",
                entities=[("service", "svc-a")],
            )

    assert len(hits) == 2
    assert hits[0].chunk_id == "high", "high-surprise hit must rank 1 when flag on"
    assert hits[1].chunk_id == "low"
    assert hits[0].score == pytest.approx(1.25 * 1.5 * 1.4)
    assert hits[1].score == pytest.approx(1.25)
    assert hits[0].score > hits[1].score


@pytest.mark.asyncio
async def test_flag_on_tie_break_by_chunk_id() -> None:
    """Flag=true: equal-score hits ordered by chunk_id ASC.

    Determinism matters because RRF assigns rank from list position --
    a non-deterministic tie order would make rank 1 unstable across
    runs of the same query, which would jitter retriever_scores in
    user-visible responses.
    """
    import services.retrieval.retrievers.graph as graph_mod

    # Two rows with identical surprise inputs: same source, same community,
    # same confidence -- score will be exactly equal.
    row_b = _make_row(
        chunk_id="b_chunk",
        source_system="slack",
        via_source_system="slack",
        community_id=1,
        via_community=1,
        confidence="INFERRED",
    )
    row_a = _make_row(
        chunk_id="a_chunk",
        source_system="slack",
        via_source_system="slack",
        community_id=1,
        via_community=1,
        confidence="INFERRED",
    )

    with patch.object(graph_mod, "SURPRISE_SCORE_ENABLED", True):
        # DB returns b BEFORE a; deterministic sort must flip to alphabetical.
        ctx = _patched_conn([row_b, row_a])
        with patch("services.retrieval.retrievers.graph.with_tenant", return_value=ctx):
            hits = await graph_mod.graph_search(
                customer_id="cust-test",
                entities=[("service", "svc-a")],
            )

    assert len(hits) == 2
    # Equal scores -> tie-break by chunk_id ascending.
    assert hits[0].score == pytest.approx(hits[1].score)
    assert hits[0].chunk_id == "a_chunk"
    assert hits[1].chunk_id == "b_chunk"
