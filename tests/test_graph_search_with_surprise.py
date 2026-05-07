"""Integration tests for the surprise-score wiring in graph.py.

Uses mock DB rows to verify:
- Flag on -> cross-source hits score higher than same-source hits
- Tenant isolation: retriever_scores uses only per-row data (no cross-tenant leak)
- All rows get retriever_scores['surprise'] regardless of flag state
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
    degree: int = 3,
    via_degree: int = 3,
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


@pytest.mark.asyncio
async def test_flag_on_cross_source_hits_rank_above_same_source() -> None:
    """Flag=true: cross-source edge ranks higher than same-source edge."""
    import services.retrieval.retrievers.graph as graph_mod

    with patch.object(graph_mod, "SURPRISE_SCORE_ENABLED", True):
        # Same-source (slack -> slack): INFERRED, same community
        same_source_row = _make_row(
            chunk_id="same_source",
            source_system="slack",
            confidence="INFERRED",
            via_source_system="slack",
            community_id=1,
            via_community=1,
        )
        # Cross-source (github -> slack): INFERRED, same community
        cross_source_row = _make_row(
            chunk_id="cross_source",
            source_system="github",
            confidence="INFERRED",
            via_source_system="slack",
            community_id=1,
            via_community=1,
        )

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[same_source_row, cross_source_row])
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("services.retrieval.retrievers.graph.with_tenant", return_value=mock_ctx):
            hits = await graph_mod.graph_search(
                customer_id="cust-test",
                entities=[("service", "svc-a")],
            )

    assert len(hits) == 2
    by_id = {h.chunk_id: h for h in hits}

    same = by_id["same_source"]
    cross = by_id["cross_source"]

    # Cross-source hit must score higher
    assert cross.score > same.score
    # Same-source with INFERRED, same community: only INFERRED weight = 1.25
    assert same.score == pytest.approx(1.25)
    # Cross-source: INFERRED * cross-source = 1.25 * 1.5 = 1.875
    assert cross.score == pytest.approx(1.875)


@pytest.mark.asyncio
async def test_all_hits_have_retriever_scores_set() -> None:
    """Every hit (regardless of flag state) has retriever_scores['surprise'] set."""
    import services.retrieval.retrievers.graph as graph_mod

    for flag_value in (True, False):
        with patch.object(graph_mod, "SURPRISE_SCORE_ENABLED", flag_value):
            rows = [
                _make_row(chunk_id="c1", confidence="INFERRED"),
                _make_row(chunk_id="c2", confidence="EXTRACTED"),
            ]

            mock_conn = AsyncMock()
            mock_conn.fetch = AsyncMock(return_value=rows)
            mock_ctx = MagicMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)

            with patch("services.retrieval.retrievers.graph.with_tenant", return_value=mock_ctx):
                hits = await graph_mod.graph_search(
                    customer_id="cust-test",
                    entities=[("service", "svc-a")],
                )

        assert len(hits) == 2, f"Expected 2 hits with flag={flag_value}"
        for hit in hits:
            assert hit.retriever_scores is not None
            assert "surprise" in hit.retriever_scores
            assert isinstance(hit.retriever_scores["surprise"], float)


@pytest.mark.asyncio
async def test_tenant_isolation_retriever_scores_use_row_data_only() -> None:
    """retriever_scores are computed purely from per-row data; no shared global state."""
    import services.retrieval.retrievers.graph as graph_mod

    with patch.object(graph_mod, "SURPRISE_SCORE_ENABLED", True):
        # Tenant A: cross-source, cross-community row
        tenant_a_row = _make_row(
            chunk_id="a1",
            source_system="github",
            confidence="INFERRED",
            via_source_system="slack",
            community_id=1,
            via_community=99,
        )
        # Tenant B: same-source, same-community row (as if from another tenant)
        tenant_b_row = _make_row(
            chunk_id="b1",
            source_system="slack",
            confidence="INFERRED",
            via_source_system="slack",
            community_id=5,
            via_community=5,
        )

        # Each call uses separate mock conn (simulates separate tenant contexts)
        for customer_id, row, expected_bonus in [
            ("cust-a", tenant_a_row, True),   # cross-source + cross-community
            ("cust-b", tenant_b_row, False),  # same-source + same-community
        ]:
            mock_conn = AsyncMock()
            mock_conn.fetch = AsyncMock(return_value=[row])
            mock_ctx = MagicMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)

            with patch("services.retrieval.retrievers.graph.with_tenant", return_value=mock_ctx):
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

    with patch.object(graph_mod, "SURPRISE_SCORE_ENABLED", True):
        # Same community
        no_community_row = _make_row(
            chunk_id="same_comm",
            source_system="github",
            confidence="INFERRED",
            via_source_system="slack",
            community_id=1,
            via_community=1,
        )
        # Different community
        cross_community_row = _make_row(
            chunk_id="cross_comm",
            source_system="github",
            confidence="INFERRED",
            via_source_system="slack",
            community_id=1,
            via_community=99,
        )

        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[no_community_row, cross_community_row])
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("services.retrieval.retrievers.graph.with_tenant", return_value=mock_ctx):
            hits = await graph_mod.graph_search(
                customer_id="cust-test",
                entities=[("service", "svc-a")],
            )

    assert len(hits) == 2
    by_id = {h.chunk_id: h for h in hits}

    same = by_id["same_comm"]
    cross = by_id["cross_comm"]

    # INFERRED * cross-source = 1.25 * 1.5 = 1.875
    assert same.score == pytest.approx(1.25 * 1.5)
    # INFERRED * cross-source * cross-community = 1.25 * 1.5 * 1.4 = 2.625
    assert cross.score == pytest.approx(1.25 * 1.5 * 1.4)
    assert cross.score > same.score
