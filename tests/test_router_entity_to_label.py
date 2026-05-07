"""Unit tests for the router entity_type -> NodeLabel mapping.

A: validates that ROUTER_ENTITY_TO_LABEL in shared/constants.py covers
   every entity_type the router prompt advertises. If the router prompt
   gains a new type and nobody updates the dict, this test fails at
   build time -- before the silent-zero-results bug we hit in prod.

C: validates that graph_search() degrades gracefully when an unknown
   entity_type comes in: it still resolves the canonical_id (via the
   SQL UNION fallback) AND emits a warning log so the drift is visible.
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import patch

import pytest

from shared.constants import ROUTER_ENTITY_TO_LABEL, NodeLabel

# ---------------------------------------------------------------------------
# A. Coverage: every router-emitted entity_type has a NodeLabel mapping.
# ---------------------------------------------------------------------------


def _router_prompt_entity_types() -> set[str]:
    """Pull the router prompt's entity_type enum out of router.py.

    Doing it via regex against the file is intentionally crude: the alternative
    (importing the router module) loads heavy LLM client deps. The regex
    targets the JSON schema literal in services/retrieval/router.py:
        "entity_type": {
            "type": "string",
            "enum": ["service", "repo", "person", ...],
        }
    Match the bracketed list, then split out string literals.
    """
    src = Path(__file__).resolve().parent.parent / "services/retrieval/router.py"
    text = src.read_text(encoding="utf-8")
    m = re.search(
        r'"entity_type":\s*\{\s*"type":\s*"string",\s*"enum":\s*\[([^\]]+)\]',
        text,
    )
    assert m, "router.py JSON schema entity_type enum not found"
    return set(re.findall(r'"([^"]+)"', m.group(1)))


def test_every_router_entity_type_has_label_mapping() -> None:
    """Every entity_type the router can emit MUST be in ROUTER_ENTITY_TO_LABEL.

    If this fails, the router added a new type and nobody updated the dict
    -- meaning the graph retriever drops it silently. Add the new type
    to ROUTER_ENTITY_TO_LABEL in shared/constants.py.
    """
    router_types = _router_prompt_entity_types()
    mapped_types = set(ROUTER_ENTITY_TO_LABEL.keys())
    missing = router_types - mapped_types
    assert not missing, (
        f"Router emits these entity_types but ROUTER_ENTITY_TO_LABEL has no "
        f"mapping for them: {sorted(missing)}. Add them to "
        f"shared/constants.py:ROUTER_ENTITY_TO_LABEL or the graph retriever "
        f"will fall back to canonical_id-only lookup (degraded mode)."
    )


def test_session_maps_to_document() -> None:
    """Regression for the production bug: session UUIDs are claude_code
    Document canonical_ids, so the graph retriever must anchor on
    NodeLabel.DOCUMENT.
    """
    assert ROUTER_ENTITY_TO_LABEL["session"] == NodeLabel.DOCUMENT


def test_file_path_maps_to_repo() -> None:
    """file_path is a sub-entity of a Repo (no FilePath label exists).
    The graph retriever anchors on the Repo so the 1-hop walk surfaces
    the rest of the repo's content."""
    assert ROUTER_ENTITY_TO_LABEL["file_path"] == NodeLabel.REPO


def test_all_values_are_real_node_labels() -> None:
    """Every value in the mapping must be a real NodeLabel enum member.
    mypy catches this statically; this is the runtime safety net.
    """
    for entity_type, label in ROUTER_ENTITY_TO_LABEL.items():
        assert isinstance(label, NodeLabel), (
            f"{entity_type!r} maps to {label!r} which is not a NodeLabel"
        )


# ---------------------------------------------------------------------------
# C. Loud failure + canonical_id-only fallback.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unknown_entity_type_falls_back_and_logs() -> None:
    """An unknown entity_type triggers a warning log AND its canonical_id
    is passed to the SQL anchors fallback branch. Previously the entity
    was silently dropped; now drift is visible AND the graph hit still
    surfaces if the canonical_id matches a node.
    """
    from services.retrieval.retrievers import graph as graph_module

    # Mock the DB connection: with_tenant() yields an async context manager
    # whose connection.fetch() returns no rows (we only care about the
    # SQL parameters that get passed in).
    captured_params: list[tuple] = []

    class _FakeConn:
        async def fetch(self, sql, *args):
            captured_params.append(args)
            return []

    class _FakeContextManager:
        async def __aenter__(self):
            return _FakeConn()

        async def __aexit__(self, *exc):
            return False

    # Patch log.warning directly because structlog bypasses pytest's caplog
    # by default (it has its own processor pipeline). Asserting on the call
    # is more direct anyway -- we control exactly which kwargs we expect.
    with (
        patch.object(graph_module, "with_tenant", lambda _cid: _FakeContextManager()),
        patch.object(graph_module.log, "warning") as mock_warning,
    ):
        hits = await graph_module.graph_search(
            customer_id="cust-test",
            entities=[
                ("pr", "175"),                   # known type
                ("session", "1b39163a"),         # known (post-fix)
                ("brand_new_type", "foo-1"),     # unknown -> fallback
                ("another_unknown", "bar-2"),    # unknown -> fallback
            ],
        )

    assert hits == []  # _FakeConn returns no rows; that's fine

    # Param order: (customer_id, labels, cids, top_k, fallback_cids, ...)
    assert len(captured_params) == 1
    args = captured_params[0]
    customer_id, labels, cids, _top_k, fallback_cids = args[:5]

    assert customer_id == "cust-test"
    # Both known types resolved to their NodeLabel.value (PR, Document)
    assert set(labels) == {"PR", "Document"}
    assert set(cids) == {"175", "1b39163a"}
    # Both unknown types' canonical_ids landed in the fallback list
    assert set(fallback_cids) == {"foo-1", "bar-2"}

    # The warning identifies which types were unknown and points at the fix.
    mock_warning.assert_called_once()
    call_args = mock_warning.call_args
    assert call_args.args[0] == "graph.unknown_entity_types_fallback"
    assert call_args.kwargs["customer"] == "cust-test"
    assert sorted(call_args.kwargs["unknown_types"]) == [
        "another_unknown",
        "brand_new_type",
    ]
    assert call_args.kwargs["fallback_cid_count"] == 2
    assert "ROUTER_ENTITY_TO_LABEL" in call_args.kwargs["fix_hint"]


@pytest.mark.asyncio
async def test_all_known_types_no_warning() -> None:
    """When every entity_type is in the mapping, no warning fires and
    fallback_cids is empty.
    """
    from services.retrieval.retrievers import graph as graph_module

    captured_params: list[tuple] = []

    class _FakeConn:
        async def fetch(self, sql, *args):
            captured_params.append(args)
            return []

    class _FakeContextManager:
        async def __aenter__(self):
            return _FakeConn()

        async def __aexit__(self, *exc):
            return False

    with (
        patch.object(graph_module, "with_tenant", lambda _cid: _FakeContextManager()),
        patch.object(graph_module.log, "warning") as mock_warning,
    ):
        await graph_module.graph_search(
            customer_id="cust-test",
            entities=[("pr", "175"), ("repo", "prbe-ai/prbe-knowledge")],
        )

    assert len(captured_params) == 1
    fallback_cids = captured_params[0][4]
    assert fallback_cids == []

    # No unknown_entity_types_fallback warning should fire when every type
    # resolves cleanly.
    mock_warning.assert_not_called()


@pytest.mark.asyncio
async def test_all_unknown_types_still_runs_query() -> None:
    """If every entity is unknown, we still issue the SQL query (so the
    fallback branch can match by canonical_id alone). Previous behaviour
    was to early-return [] which silently dropped every entity.
    """
    from services.retrieval.retrievers import graph as graph_module

    captured_params: list[tuple] = []

    class _FakeConn:
        async def fetch(self, sql, *args):
            captured_params.append(args)
            return []

    class _FakeContextManager:
        async def __aenter__(self):
            return _FakeConn()

        async def __aexit__(self, *exc):
            return False

    with patch.object(
        graph_module, "with_tenant", lambda _cid: _FakeContextManager()
    ):
        hits = await graph_module.graph_search(
            customer_id="cust-test",
            entities=[("brand_new_type", "abc"), ("another_new", "def")],
        )

    assert hits == []
    # The query still ran -- fallback branch handles it.
    assert len(captured_params) == 1
    labels, cids, fallback_cids = (
        captured_params[0][1],
        captured_params[0][2],
        captured_params[0][4],
    )
    assert labels == []      # no typed lookups
    assert cids == []
    assert set(fallback_cids) == {"abc", "def"}


@pytest.mark.asyncio
async def test_no_entities_short_circuits() -> None:
    """Backwards-compat: passing no entities at all still returns []
    without issuing a SQL query.
    """
    from services.retrieval.retrievers import graph as graph_module

    captured_params: list[tuple] = []

    class _FakeConn:
        async def fetch(self, sql, *args):
            captured_params.append(args)
            return []

    class _FakeContextManager:
        async def __aenter__(self):
            return _FakeConn()

        async def __aexit__(self, *exc):
            return False

    with patch.object(
        graph_module, "with_tenant", lambda _cid: _FakeContextManager()
    ):
        hits = await graph_module.graph_search(
            customer_id="cust-test",
            entities=[],
        )

    assert hits == []
    assert captured_params == []  # short-circuit; no query
