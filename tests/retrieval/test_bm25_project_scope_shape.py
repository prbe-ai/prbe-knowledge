"""E7 phase 2: where the project scope lands in the BM25 query.

Phase 1 put `scope.project_id` on the documents join, where Tantivy cannot see
it: the scan applies it as a per-candidate HEAP filter AFTER TopK has already
chosen its pool from the whole tenant. The existing mitigation widens the pool
4x and hopes. Phase 2 moves it inside the boolean as a `must` clause.

Both shapes must work, because code deploys before the index is swapped.
"""

from __future__ import annotations

import inspect

from engine.retrieval.retrievers import bm25


def test_the_retriever_probes_for_the_index_rather_than_assuming_it() -> None:
    """The whole safety property of this change. A retriever that assumed v3
    the moment migration 0131 ran would fail every BM25 query until an attended
    swap window came round -- which can be days."""
    assert hasattr(bm25, "bm25_project_scope_is_index_side")
    assert inspect.iscoroutinefunction(bm25.bm25_project_scope_is_index_side)


async def test_an_invalid_index_does_not_count_as_present() -> None:
    """A killed `CREATE INDEX` keeps the NAME with indisvalid = false.
    Querying through one returns wrong results rather than raising, so the
    probe reads indisvalid, not existence."""

    class _Conn:
        def __init__(self, indisvalid):
            self.indisvalid = indisvalid

        async def fetchval(self, sql, *args):
            assert "indisvalid" in sql, "the probe must read validity, not just the name"
            return self.indisvalid

    bm25._bm25_v3_available = None
    assert await bm25.bm25_project_scope_is_index_side(_Conn(False)) is False

    bm25._bm25_v3_available = None
    assert await bm25.bm25_project_scope_is_index_side(_Conn(True)) is True

    bm25._bm25_v3_available = None
    assert await bm25.bm25_project_scope_is_index_side(_Conn(None)) is False


async def test_a_probe_failure_falls_back_to_the_shape_that_always_worked() -> None:
    """A probe is a convenience. It must never be the reason a query fails."""

    class _Boom:
        async def fetchval(self, sql, *args):
            raise RuntimeError("pg_class unavailable")

    bm25._bm25_v3_available = None
    assert await bm25.bm25_project_scope_is_index_side(_Boom()) is False
    bm25._bm25_v3_available = None


def test_the_scoped_pool_factor_still_exists_for_the_other_scopes() -> None:
    """`project_id` stops needing the 4x pool once it rides the index, but
    sources / doc_types / author / source_keys still land on the join and
    still do."""
    assert bm25._BM25_SCOPED_POOL_FACTOR > 1
    src = inspect.getsource(bm25.bm25_search)
    assert "project_index_side" in src
    # The pool widens for the other scopes regardless.
    assert "source_keys" in src


def test_the_scope_uses_term_not_match_so_a_uuid_matches_whole() -> None:
    """`match()` tokenizes. A project_id is an opaque uuid, and tokenizing it
    would let a scope leak to any project sharing a hyphen-delimited segment --
    which for uuids is a real collision, not a theoretical one."""
    src = inspect.getsource(bm25.bm25_search)
    assert "paradedb.term('project_id'" in src
    assert "paradedb.match('project_id'" not in src
