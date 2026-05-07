"""Regression guard: with SURPRISE_SCORE_ENABLED=false, graph hits return score=1.0.

This test is the IRON RULE regression guard for Lane A. It must pass before
any prod flag flip. It verifies that the graph retriever's external behaviour
is unchanged when the flag is off.

We mock graph_search's DB layer to avoid needing a live Postgres instance.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure flag is false before importing the module under test.
# The module-level constant is evaluated at import time; we patch the env
# var and then force re-evaluation by reloading.
os.environ["SURPRISE_SCORE_ENABLED"] = "false"


def _make_row(
    *,
    chunk_id: str = "c1",
    doc_id: str = "doc1",
    source_system: str = "slack",
    confidence: str | None = "EXTRACTED",
    via_source_system: str | None = "slack",
    community_id: int | None = None,
    via_community: int | None = None,
    degree: int = 3,
    via_degree: int = 3,
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
        "via_label": "Service",
        "edge_type": "REFERENCES",
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
async def test_flag_false_always_returns_score_1() -> None:
    """With SURPRISE_SCORE_ENABLED=false, every hit has score=1.0 regardless of inputs."""
    import services.retrieval.retrievers.graph as graph_mod

    # Force flag to false at module level (already set via env var above, but
    # override the in-process variable to guarantee the test is not flaky).
    with patch.object(graph_mod, "SURPRISE_SCORE_ENABLED", False):
        # Use INFERRED confidence (passes default min_confidence="INFERRED") but
        # inputs that would produce high surprise if flag were on (cross-source,
        # cross-community, peripheral-to-hub).
        rows = [
            _make_row(
                chunk_id="c1",
                source_system="github",
                confidence="INFERRED",
                via_source_system="slack",
                community_id=1,
                via_community=99,
                degree=1,
                via_degree=20,
            ),
            _make_row(
                chunk_id="c2",
                source_system="notion",
                confidence="INFERRED",
                via_source_system="sentry",
                community_id=2,
                via_community=3,
                degree=2,
                via_degree=7,
            ),
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

    assert len(hits) == 2
    for hit in hits:
        assert hit.score == 1.0, f"Expected score=1.0 but got {hit.score} for {hit.chunk_id}"


@pytest.mark.asyncio
async def test_flag_false_still_populates_retriever_scores_for_telemetry() -> None:
    """Even with flag off, retriever_scores['surprise'] is populated (non-zero) for telemetry."""
    import services.retrieval.retrievers.graph as graph_mod

    with patch.object(graph_mod, "SURPRISE_SCORE_ENABLED", False):
        # Use INFERRED (passes default filter) + cross-source -> would-be score > 1.0
        rows = [
            _make_row(
                chunk_id="c1",
                source_system="github",
                confidence="INFERRED",
                via_source_system="slack",  # cross-source -> should produce > 1.0
            ),
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

    assert len(hits) == 1
    hit = hits[0]
    # Score is flat 1.0 (flag off)
    assert hit.score == 1.0
    # But retriever_scores carries the would-be value for telemetry
    assert hit.retriever_scores is not None
    assert "surprise" in hit.retriever_scores
    # INFERRED + cross-source = 1.25 * 1.5 = 1.875 > 1.0
    assert hit.retriever_scores["surprise"] > 1.0


@pytest.mark.asyncio
async def test_flag_true_uses_surprise_score() -> None:
    """With SURPRISE_SCORE_ENABLED=true, score equals the computed surprise value."""
    import services.retrieval.retrievers.graph as graph_mod

    with patch.object(graph_mod, "SURPRISE_SCORE_ENABLED", True):
        # Use INFERRED confidence (passes default min_confidence filter) + cross-source
        rows = [
            _make_row(
                chunk_id="c1",
                source_system="github",
                confidence="INFERRED",
                via_source_system="slack",  # cross-source
                community_id=None,
                via_community=None,
                degree=3,
                via_degree=3,
            ),
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

    assert len(hits) == 1
    hit = hits[0]
    # INFERRED * cross-source = 1.25 * 1.5 = 1.875
    expected = 1.25 * 1.5
    assert abs(hit.score - expected) < 1e-9
    assert hit.retriever_scores is not None
    assert abs(hit.retriever_scores["surprise"] - expected) < 1e-9


@pytest.mark.asyncio
async def test_flag_false_same_source_same_community_score_1() -> None:
    """All-neutral row: flag off -> score=1.0; retriever_scores['surprise']=1.0."""
    import services.retrieval.retrievers.graph as graph_mod

    with patch.object(graph_mod, "SURPRISE_SCORE_ENABLED", False):
        rows = [
            _make_row(
                chunk_id="c1",
                source_system="slack",
                confidence="EXTRACTED",
                via_source_system="slack",
                community_id=5,
                via_community=5,
                degree=3,
                via_degree=3,
            ),
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

    assert len(hits) == 1
    hit = hits[0]
    assert hit.score == 1.0
    assert hit.retriever_scores is not None
    assert hit.retriever_scores["surprise"] == pytest.approx(1.0)
