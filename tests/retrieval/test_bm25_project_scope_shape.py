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


def test_the_scope_uses_match_not_term_because_a_uuid_is_all_hyphens() -> None:
    """THE INVERSE of what this test first asserted, and the reason matters.

    `term()` looks up a TOKEN. `project_id` is indexed under the default
    tokenizer, which splits on hyphens, and a project_id is a uuid -- which is
    nothing but hyphens. Verified against a real pg_search index:
    `term('project_id', '240f2b75-a2ee-4189-ad05-9883bd5514a5')` matches ZERO
    rows, because the whole-string token does not exist.

    Shipping that would have made every project-scoped BM25 query return
    nothing, silently, under `state: "ok"` -- and it is the identical trap the
    customer_id clause in this same query already documents, one field over.
    """
    src = inspect.getsource(bm25.bm25_search)
    assert "paradedb.match('project_id'" in src
    assert "conjunction_mode => true" in src
    assert "paradedb.term('project_id'" not in src


def test_the_sql_predicate_is_kept_as_the_correctness_filter() -> None:
    """The index leg is a PRE-FILTER, not the filter.

    conjunction_mode requires every token of the id, which makes it a sound
    pre-filter but not an equality test. The SQL predicate on the documents
    join is what makes the answer exact, so it runs whether or not the index
    leg applied -- the same belt-and-braces, in the same order, that the
    tenant and visibility filters already use.
    """
    src = inspect.getsource(bm25.bm25_search)
    filter_line = [
        ln for ln in src.splitlines()
        if "project_filter = project_scope_predicate" in ln
    ]
    assert len(filter_line) == 1, "the SQL predicate must be built exactly once"
    assert not filter_line[0].strip().startswith("#")
    # It must NOT sit inside the `if project_index_side:` branch -- that is the
    # shape that made the two mutually exclusive.
    indent = len(filter_line[0]) - len(filter_line[0].lstrip())
    branch = next(ln for ln in src.splitlines() if "if project_index_side" in ln)
    assert indent <= len(branch) - len(branch.lstrip()), (
        "the SQL predicate is nested under the index-side branch, so an "
        "index-side query has no correctness filter"
    )
