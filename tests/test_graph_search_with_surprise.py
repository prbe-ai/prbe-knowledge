"""Integration tests for the surprise-score wiring in graph.py.

The surprise score is computed unconditionally for every graph hit and
written to both hit.score (for fusion's RRF input) and
retriever_scores['surprise'] (for telemetry round-trip). Earlier
versions of this file patched a feature flag (SURPRISE_SCORE_ENABLED);
the flag was deleted along with its branches, so these tests now run
the always-on path directly.

Covers:
- Cross-source hits score higher than same-source hits
- Cross-community adds bonus on top of cross-source
- Tenant isolation: scores use per-row data only, no shared state
- Every hit carries retriever_scores['surprise']
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_row(
    *,
    chunk_id: str = "c1",
    doc_id: str = "doc1",
    source_system: str = "slack",
    confidence: str | None = "INFERRED",
    via_source_system: str | None = "slack",
    community_id: int | None = None,
    via_community: int | None = None,
    # Defaults stay below the hub-to-hub anti-bonus threshold (min_deg >= 3
    # in surprise.py) so these source/community-focused tests don't pick
    # up Component 5's penalty as a side effect.
    degree: int = 2,
    via_degree: int = 2,
    via_label: str = "Service",
    edge_type: str = "REFERENCES",
) -> MagicMock:
    now = datetime.now(UTC)
    data = {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
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
async def test_cross_source_hits_rank_above_same_source() -> None:
    """Cross-source edge ranks higher than same-source edge."""
    import services.retrieval.retrievers.graph as graph_mod

    same_source_row = _make_row(
        chunk_id="same_source",
        source_system="slack",
        confidence="INFERRED",
        via_source_system="slack",
        community_id=1,
        via_community=1,
    )
    cross_source_row = _make_row(
        chunk_id="cross_source",
        source_system="github",
        confidence="INFERRED",
        via_source_system="slack",
        community_id=1,
        via_community=1,
    )

    ctx = _patched_conn([same_source_row, cross_source_row])
    with patch("services.retrieval.retrievers.graph.with_tenant", return_value=ctx):
        hits = await graph_mod.graph_search(
            customer_id="cust-test",
            entities=[("service", "svc-a")],
        )

    assert len(hits) == 2
    by_id = {h.chunk_id: h for h in hits}

    same = by_id["same_source"]
    cross = by_id["cross_source"]

    # Cross-source must score higher.
    assert cross.score > same.score
    # Same-source INFERRED, same community: only INFERRED weight = 1.25.
    assert same.score == pytest.approx(1.25)
    # Cross-source: INFERRED * cross-source = 1.25 * 1.5 = 1.875.
    assert cross.score == pytest.approx(1.875)


@pytest.mark.asyncio
async def test_all_hits_have_retriever_scores_set() -> None:
    """Every hit carries retriever_scores['surprise'] for telemetry."""
    import services.retrieval.retrievers.graph as graph_mod

    rows = [
        _make_row(chunk_id="c1", confidence="INFERRED"),
        _make_row(chunk_id="c2", confidence="EXTRACTED"),
    ]

    ctx = _patched_conn(rows)
    with patch("services.retrieval.retrievers.graph.with_tenant", return_value=ctx):
        hits = await graph_mod.graph_search(
            customer_id="cust-test",
            entities=[("service", "svc-a")],
        )

    assert len(hits) == 2
    for hit in hits:
        assert hit.retriever_scores is not None
        assert "surprise" in hit.retriever_scores
        assert isinstance(hit.retriever_scores["surprise"], float)
        # surprise telemetry equals the rank-driving score.
        assert hit.retriever_scores["surprise"] == pytest.approx(hit.score)


@pytest.mark.asyncio
async def test_tenant_isolation_retriever_scores_use_row_data_only() -> None:
    """Scores are computed purely from per-row data; no shared global state."""
    import services.retrieval.retrievers.graph as graph_mod

    # Tenant A: cross-source, cross-community.
    tenant_a_row = _make_row(
        chunk_id="a1",
        source_system="github",
        confidence="INFERRED",
        via_source_system="slack",
        community_id=1,
        via_community=99,
    )
    # Tenant B: same-source, same-community.
    tenant_b_row = _make_row(
        chunk_id="b1",
        source_system="slack",
        confidence="INFERRED",
        via_source_system="slack",
        community_id=5,
        via_community=5,
    )

    for customer_id, row, expected_bonus in [
        ("cust-a", tenant_a_row, True),
        ("cust-b", tenant_b_row, False),
    ]:
        ctx = _patched_conn([row])
        with patch("services.retrieval.retrievers.graph.with_tenant", return_value=ctx):
            hits = await graph_mod.graph_search(
                customer_id=customer_id,
                entities=[("service", "svc-a")],
            )

        assert len(hits) == 1
        hit = hits[0]
        if expected_bonus:
            assert hit.score > 1.0, f"Expected surprise bonus for {customer_id}"
        else:
            assert hit.score == pytest.approx(1.25), (
                f"Expected INFERRED-only score=1.25 for {customer_id}, got {hit.score}"
            )


@pytest.mark.asyncio
async def test_cross_community_adds_bonus_on_top_of_cross_source() -> None:
    """Cross-community adds 1.4x on top of cross-source bonus."""
    import services.retrieval.retrievers.graph as graph_mod

    no_community_row = _make_row(
        chunk_id="same_comm",
        source_system="github",
        confidence="INFERRED",
        via_source_system="slack",
        community_id=1,
        via_community=1,
    )
    cross_community_row = _make_row(
        chunk_id="cross_comm",
        source_system="github",
        confidence="INFERRED",
        via_source_system="slack",
        community_id=1,
        via_community=99,
    )

    ctx = _patched_conn([no_community_row, cross_community_row])
    with patch("services.retrieval.retrievers.graph.with_tenant", return_value=ctx):
        hits = await graph_mod.graph_search(
            customer_id="cust-test",
            entities=[("service", "svc-a")],
        )

    assert len(hits) == 2
    by_id = {h.chunk_id: h for h in hits}

    same = by_id["same_comm"]
    cross = by_id["cross_comm"]

    # INFERRED * cross-source = 1.25 * 1.5 = 1.875.
    assert same.score == pytest.approx(1.25 * 1.5)
    # INFERRED * cross-source * cross-community = 1.25 * 1.5 * 1.4 = 2.625.
    assert cross.score == pytest.approx(1.25 * 1.5 * 1.4)
    assert cross.score > same.score
