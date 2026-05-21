"""Unit tests for list-pipeline entity-filter gating on `entity_must_match`.

The list path used to derive `author_ids` (from `person` entities) and
`graph_entity_filters` (from `service`/`repo`/`ticket`/`pr`/`channel`
entities) from the router output and pass them as hard SQL `WHERE`
clauses unconditionally. When a router-extracted entity didn't have a
matching `graph_nodes` row or `documents.author_id`, the SQL returned
zero results — even when the broader query intent could have been
satisfied by sort/temporal alone.

This test pins the new behavior:

  - `entity_must_match=False` (default, matches the MCP) → list pipeline
    passes `author_ids=None` and `graph_entity_filters=[]` (or None) to
    the SQL helpers, regardless of what the router extracted.
  - `entity_must_match=True` → list pipeline derives both from the
    router output and passes them through.

Mocks the SQL helpers at the module boundary so this runs without a DB.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from services.retrieval.list_pipeline import run_list
from services.retrieval.router import Intent, RouterEntity
from shared.models import QueryRequest, TemporalSpec

pytestmark = pytest.mark.asyncio


def _intent_with_repo_and_person() -> Intent:
    """Intent with one narrowing repo entity AND one person
    entity — exercises both the `graph_entity_filters` and `author_ids`
    branches in a single test."""
    return Intent(
        query_text="recent prbe-backend commits by alice",
        mode="list",
        confidence=0.9,
        entities=[
            RouterEntity(
                entity_type="repo",
                canonical_id="prbe-ai/prbe-backend",
                display_name="prbe-backend",
                confidence=0.9,
            ),
            RouterEntity(
                entity_type="person",
                canonical_id="user:alice",
                display_name="alice",
                confidence=0.9,
            ),
        ],
        sort={"field": "updated_at", "direction": "desc"},
        operation="list",
    )


class TestListPipelineEntityGating:
    async def test_flag_off_skips_entity_filters(self) -> None:
        """Default `entity_must_match=False` → SQL helper sees no
        author_ids and no graph_entity_filters even when the router
        extracted both."""
        req = QueryRequest(query="recent prbe-backend commits by alice", top_k=5)
        assert req.entity_must_match is False  # belt-and-suspenders

        intent = _intent_with_repo_and_person()

        # Phase 2 Chunk 8: list pipeline's related-entities branch now calls
        # expand_exclude_keys_with_aliases, which opens a `with_tenant`
        # connection. Mock to passthrough so this unit test doesn't need a
        # live DB pool; integration coverage lives in
        # test_related_entities_clusters.py.
        async def _passthrough_exclude(_cid, _ents, keys, **_kw):
            return keys

        with patch(
            "services.retrieval.list_pipeline.sql_list", new=AsyncMock(return_value=[])
        ) as m_list, patch(
            "services.retrieval.list_pipeline.expand_exclude_keys_with_aliases",
            new=AsyncMock(side_effect=_passthrough_exclude),
        ):
            await run_list(
                req=req,
                customer_id="cust-1",
                intent=intent,
                intent_idx=0,
                spec=TemporalSpec(),
                temporal_meta={},
                sort_meta=None,
                extracted_entities=[],
                doc_types=None,
                trace_id="t-1",
                timing={},
            )

        m_list.assert_called_once()
        kwargs = m_list.call_args.kwargs
        assert kwargs["author_ids"] is None
        # The pipeline passes `graph_entity_filters or None`; with the
        # gate off it's []  → coalesces to None.
        assert kwargs["graph_entity_filters"] is None

    async def test_flag_on_passes_entity_filters(self) -> None:
        """`entity_must_match=True` → author_ids and graph_entity_filters
        derived from the router output are passed through to the SQL
        helper."""
        req = QueryRequest(
            query="recent prbe-backend commits by alice",
            top_k=5,
            entity_must_match=True,
        )

        intent = _intent_with_repo_and_person()

        # Post-0091: list pipeline expands author_ids to (cluster members) +
        # (Lane E property values on members) via expand_to_author_id_set.
        # Mock the helper so this unit test doesn't require a live DB pool;
        # expansion semantics are covered by test_list_pipeline_author_alias.py.
        async def _passthrough_exclude(_cid, _ents, keys, **_kw):
            return keys

        with patch(
            "services.retrieval.list_pipeline.sql_list", new=AsyncMock(return_value=[])
        ) as m_list, patch(
            "services.retrieval.list_pipeline.with_tenant"
        ) as m_with_tenant, patch(
            "services.retrieval.list_pipeline.expand_to_author_id_set",
            new=AsyncMock(return_value=["user:alice"]),
        ), patch(
            "services.retrieval.list_pipeline.expand_exclude_keys_with_aliases",
            new=AsyncMock(side_effect=_passthrough_exclude),
        ):
            m_with_tenant.return_value.__aenter__ = AsyncMock(return_value=None)
            m_with_tenant.return_value.__aexit__ = AsyncMock(return_value=None)
            await run_list(
                req=req,
                customer_id="cust-1",
                intent=intent,
                intent_idx=0,
                spec=TemporalSpec(),
                temporal_meta={},
                sort_meta=None,
                extracted_entities=[],
                doc_types=None,
                trace_id="t-1",
                timing={},
            )

        m_list.assert_called_once()
        kwargs = m_list.call_args.kwargs
        assert kwargs["author_ids"] == ["user:alice"]
        gef = kwargs["graph_entity_filters"]
        assert gef is not None and len(gef) == 1
        # Post-0091: repo entity_type narrows to Document label (collapsed).
        assert gef[0].label == "Document"
        assert "prbe-ai/prbe-backend" in gef[0].values
        assert "prbe-backend" in gef[0].values

    async def test_flag_off_count_branch_skips_entity_filters(self) -> None:
        """Same gating applies to the `count` branch."""
        req = QueryRequest(query="how many backend commits", top_k=5)
        intent = _intent_with_repo_and_person()
        intent.operation = "count"

        with patch(
            "services.retrieval.list_pipeline.sql_count", new=AsyncMock(return_value=0)
        ) as m_count:
            await run_list(
                req=req,
                customer_id="cust-1",
                intent=intent,
                intent_idx=0,
                spec=TemporalSpec(),
                temporal_meta={},
                sort_meta=None,
                extracted_entities=[],
                doc_types=None,
                trace_id="t-1",
                timing={},
            )

        kwargs = m_count.call_args.kwargs
        assert kwargs["author_ids"] is None
        assert kwargs["graph_entity_filters"] is None

    async def test_flag_off_group_by_branch_skips_entity_filters(self) -> None:
        """Same gating applies to the `group_by` branch."""
        req = QueryRequest(query="commits by author", top_k=5)
        intent = _intent_with_repo_and_person()
        intent.operation = "group_by"
        intent.group_by_key = "author_id"

        with patch(
            "services.retrieval.list_pipeline.sql_group_by",
            new=AsyncMock(return_value=[]),
        ) as m_group:
            await run_list(
                req=req,
                customer_id="cust-1",
                intent=intent,
                intent_idx=0,
                spec=TemporalSpec(),
                temporal_meta={},
                sort_meta=None,
                extracted_entities=[],
                doc_types=None,
                trace_id="t-1",
                timing={},
            )

        kwargs = m_group.call_args.kwargs
        assert kwargs["author_ids"] is None
        assert kwargs["graph_entity_filters"] is None
