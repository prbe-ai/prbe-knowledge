"""Regression tests for graph_search()'s always-on surprise-driven sort.

What this protects
------------------
graph_search() must always return hits sorted by surprise score DESC so the
highest-surprise edge lands at rank 1 and gets the biggest graph-side RRF
contribution (1/61) in fusion. Without the sort, hits would arrive in
arbitrary heap-scan order and fusion's RRF would weight an arbitrary
neighbor highest.

The sort is unconditional: prior to this PR it was gated behind a feature
flag (SURPRISE_SCORE_ENABLED). Empirical 4-run A/B on acme proved
the always-on path is safe and strictly better than heap-scan order. The
flag was deleted; this file pins the always-on contract.
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
    # Defaults stay below the hub-to-hub anti-bonus threshold (min_deg >= 3)
    # so the sort-order tests don't pick up Component 5's penalty.
    degree: int = 2,
    via_degree: int = 2,
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
async def test_sorts_by_surprise_score_desc() -> None:
    """Highest-surprise hit lands at rank 1 regardless of DB return order.

    DB returns rows in [low, high] surprise order; graph_search must
    flip to [high, low] so fusion's RRF assigns the biggest 1/(k+rank)
    contribution to the most informative cross-source/cross-community
    edge.
    """
    import engine.retrieval.retrievers.graph as graph_mod

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
        confidence="INFERRED",  # 1.25 * 1.5 cross-source * 1.4 cross-community = 2.625
    )

    # DB delivers low BEFORE high; sort must flip that.
    ctx = _patched_conn([low_surprise, high_surprise])
    with patch("engine.retrieval.retrievers.graph.with_tenant", return_value=ctx):
        hits = await graph_mod.graph_search(
            customer_id="cust-test",
            entities=[("service", "svc-a")],
        )

    assert len(hits) == 2
    assert hits[0].chunk_id == "high", "high-surprise hit must rank 1"
    assert hits[1].chunk_id == "low"
    assert hits[0].score == pytest.approx(1.25 * 1.5 * 1.4)
    assert hits[1].score == pytest.approx(1.25)
    assert hits[0].score > hits[1].score


@pytest.mark.asyncio
async def test_score_field_is_unconditional_surprise() -> None:
    """hit.score equals the computed surprise score on every hit.

    Pre-PR there was a flag-gated path that wrote score=1.0 when off.
    The flag was removed; score is now always the surprise value, so
    fusion (focus or discovery mode) gets the same input.
    """
    import engine.retrieval.retrievers.graph as graph_mod

    cross_source = _make_row(
        chunk_id="cross",
        source_system="github",
        via_source_system="slack",
        community_id=1,
        via_community=1,
        confidence="INFERRED",  # 1.25 * 1.5 = 1.875
    )

    ctx = _patched_conn([cross_source])
    with patch("engine.retrieval.retrievers.graph.with_tenant", return_value=ctx):
        hits = await graph_mod.graph_search(
            customer_id="cust-test",
            entities=[("service", "svc-a")],
        )

    assert len(hits) == 1
    # score is the actual surprise value, not 1.0.
    assert hits[0].score == pytest.approx(1.25 * 1.5)
    # Also exposed in retriever_scores.surprise telemetry.
    assert hits[0].retriever_scores is not None
    assert hits[0].retriever_scores["surprise"] == pytest.approx(1.25 * 1.5)


@pytest.mark.asyncio
async def test_tie_break_by_chunk_id() -> None:
    """Equal-score hits ordered by chunk_id ASC.

    Determinism matters because fusion's RRF assigns rank from list
    position -- a non-deterministic tie order would make rank 1 unstable
    across runs of the same query, jittering retriever_scores in
    user-visible MCP responses.
    """
    import engine.retrieval.retrievers.graph as graph_mod

    # Identical surprise inputs -> exactly equal scores.
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

    # DB returns b BEFORE a; deterministic sort must flip to alphabetical.
    ctx = _patched_conn([row_b, row_a])
    with patch("engine.retrieval.retrievers.graph.with_tenant", return_value=ctx):
        hits = await graph_mod.graph_search(
            customer_id="cust-test",
            entities=[("service", "svc-a")],
        )

    assert len(hits) == 2
    assert hits[0].score == pytest.approx(hits[1].score)
    assert hits[0].chunk_id == "a_chunk"
    assert hits[1].chunk_id == "b_chunk"
