"""Regression tests for the graph retriever's one-chunk-per-doc cap.

Bug it prevents
---------------
Without the cap, a giant anchor doc (e.g. a claude_code session with
200+ chunks) crowds out every other neighbor: the LIMIT $4 fills with
chunks of one doc before a single chunk of any other neighbor makes it
through. The user-facing symptom: graph_search() returns 20 chunks of
the session itself and 0 chunks of any inferred-edge neighbor.

The cap is implemented as
    ROW_NUMBER() OVER (PARTITION BY c.doc_id ORDER BY c.chunk_index ...)
in the graph retriever's SQL. We verify both:

1. The SQL string contains the partition-by-doc_id pattern (cheap; no DB).
2. The graph_search() entry point passes the right parameters and the SQL
   the conn receives is the new one-chunk-per-doc shape.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

_GRAPH_PY = (
    Path(__file__).resolve().parent.parent
    / "services/retrieval/retrievers/graph.py"
)


def _load_graph_source() -> str:
    return _GRAPH_PY.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Static SQL string assertions: cheap regression catch.
# ---------------------------------------------------------------------------


def test_sql_contains_row_number_partition_by_doc_id() -> None:
    """The SQL must use ROW_NUMBER() PARTITION BY c.doc_id to cap chunks per
    doc. Removing this is the bug; this test catches a regression."""
    src = _load_graph_source()
    assert "ROW_NUMBER() OVER" in src, (
        "graph retriever no longer uses ROW_NUMBER() — one-chunk-per-doc "
        "cap removed. A giant anchor doc will crowd out all neighbors."
    )
    assert "PARTITION BY c.doc_id" in src, (
        "graph retriever no longer partitions by c.doc_id — chunks from "
        "one giant doc will fill the LIMIT before any neighbor gets a slot."
    )


def test_sql_filters_to_first_chunk_only() -> None:
    """The outer query filters rn_in_doc = 1 so each doc contributes
    exactly one chunk."""
    src = _load_graph_source()
    assert "rn_in_doc = 1" in src, (
        "graph retriever no longer filters to rn_in_doc = 1 — multiple "
        "chunks per doc will return again."
    )


def test_sql_orders_by_chunk_index_for_determinism() -> None:
    """Pick the lowest chunk_index per doc, deterministically. chunk_id
    breaks ties on the rare doc with collision in chunk_index."""
    src = _load_graph_source()
    assert "ORDER BY c.chunk_index ASC, c.chunk_id ASC" in src


# ---------------------------------------------------------------------------
# Behavioural test: with mocked DB, verify the SQL reaching conn.fetch is
# the one-chunk-per-doc shape and the right parameters are passed through.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_graph_search_issues_per_doc_cap_sql() -> None:
    """End-to-end through graph_search(): the SQL the connection receives
    must contain the per-doc cap pattern, and the right top_k flows through
    to the LIMIT."""
    from services.retrieval.retrievers import graph as graph_module

    captured_sql: list[str] = []
    captured_params: list[tuple] = []

    class _FakeConn:
        async def fetch(self, sql, *args):
            captured_sql.append(sql)
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
        await graph_module.graph_search(
            customer_id="cust-test",
            entities=[
                ("session", "claude_code:acme:1b39163a-...")
            ],
            top_k=20,
        )

    assert len(captured_sql) == 1
    sql = captured_sql[0]
    # The shape that prevents giant-anchor crowd-out
    assert "ROW_NUMBER() OVER" in sql
    assert "PARTITION BY c.doc_id" in sql
    assert "rn_in_doc = 1" in sql
    # top_k still controls the final cap (now docs not chunks, but same
    # parameter index $4)
    assert captured_params[0][3] == 20


@pytest.mark.asyncio
async def test_graph_search_param_order_unchanged() -> None:
    """Backwards-compat: the parameter order graph_search passes to conn.fetch
    must remain (customer_id, labels, cids, top_k, fallback_cids, ...). The
    per-doc cap is added entirely inside the SQL; positional params are
    untouched. Tests that depend on this order keep passing."""
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
        await graph_module.graph_search(
            customer_id="cust-test",
            entities=[("pr", "175")],
            top_k=42,
        )

    args = captured_params[0]
    assert args[0] == "cust-test"           # $1
    assert list(args[1]) == ["PR"]          # $2 labels
    assert list(args[2]) == ["175"]         # $3 cids
    assert args[3] == 42                    # $4 top_k
    assert list(args[4]) == []              # $5 fallback_cids (none here)
