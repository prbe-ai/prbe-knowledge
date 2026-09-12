"""E7 phase 2: two BM25 index generations coexist in the fleet, and every
unattended path has to stay correct while they do.

pg_search allows exactly ONE `USING bm25` index per relation, so v2 (no
project_id) and v3 (project_id as a fast field) are mutually exclusive. Code
always deploys before an index is rebuilt, so there is a window -- possibly
days, because the swap waits for an attended slot -- where the new code runs
against the old index. These tests pin that window.
"""

from __future__ import annotations

import pytest

from engine.shared.pg_search_guardian import (
    BM25_INDEX_V2,
    BM25_INDEX_V3,
    LEGACY_INDEX_DDL,
    REQUIRED_INDEX_DDL,
    find_absent_required_indexes,
    index_ddl,
    rebuild_absent_index,
    required_bm25_index,
)


class _FakeConn:
    """A connection that answers only the three questions these paths ask."""

    def __init__(self, *, has_column: bool, existing: set[str]) -> None:
        self.has_column = has_column
        self.existing = existing
        self.executed: list[str] = []

    async def fetchval(self, sql: str, *args):
        if "information_schema.columns" in sql:
            return self.has_column
        if "to_regclass" in sql:
            name = args[0]
            # Tables the guardian checks for are always present here; only
            # index existence is under test.
            if name == "chunks":
                return "chunks"
            return name if name in self.existing else None
        if "pg_try_advisory_lock" in sql:
            return True
        if "count(*)" in sql:
            return 1
        raise AssertionError(f"unexpected query: {sql[:80]}")

    async def execute(self, sql: str, *args, **kwargs):
        self.executed.append(sql)


# ------------------------------------------------------- which one is wanted

@pytest.mark.asyncio
async def test_a_database_without_the_column_still_wants_v2() -> None:
    """Pre-0131. Creating v3 there fails -- the fast field has no column."""
    conn = _FakeConn(has_column=False, existing={BM25_INDEX_V2})
    assert await required_bm25_index(conn) == BM25_INDEX_V2


@pytest.mark.asyncio
async def test_a_database_with_the_column_wants_v3() -> None:
    conn = _FakeConn(has_column=True, existing={BM25_INDEX_V2})
    assert await required_bm25_index(conn) == BM25_INDEX_V3


@pytest.mark.asyncio
async def test_the_guardians_repair_path_cannot_silently_regress_to_v2() -> None:
    """The failure this discriminator exists to stop.

    The guardian DROPS a broken index, leaving none, and the rebuild job then
    recreates whichever generation is "required". Hard-coded to v2, a database
    already swapped to v3 would quietly fall back the first time its index
    broke -- lexical search would still work, just without index-side project
    scope, and nothing anywhere would say so.
    """
    conn = _FakeConn(has_column=True, existing=set())
    assert await find_absent_required_indexes(conn) == [BM25_INDEX_V3]


# ------------------------------------------ one index per relation, enforced

@pytest.mark.asyncio
async def test_the_other_generation_standing_there_is_not_an_absence() -> None:
    """Both serve lexical search. Reporting v3 absent while v2 stands would
    have an unattended job try to CREATE a second bm25 index on `chunks`, which
    pg_search refuses outright -- a red CronJob on a healthy database."""
    conn = _FakeConn(has_column=True, existing={BM25_INDEX_V2})
    assert await find_absent_required_indexes(conn) == []

    conn = _FakeConn(has_column=False, existing={BM25_INDEX_V3})
    assert await find_absent_required_indexes(conn) == []


@pytest.mark.asyncio
async def test_rebuild_refuses_to_build_across_generations() -> None:
    """Belt to the braces above: even asked directly, the rebuild will not
    create v3 while v2 stands. Moving between generations is a deliberate,
    attended swap because it takes lexical search DOWN for the build."""
    conn = _FakeConn(has_column=True, existing={BM25_INDEX_V2})
    built = await rebuild_absent_index(conn, BM25_INDEX_V3)
    assert built is False
    assert not conn.executed, "it must not have run any DDL"


# ------------------------------------------------------------------ the DDL

def test_both_generations_are_still_buildable() -> None:
    """v2 has to stay buildable: a database where 0131 has not run has no
    `chunks.project_id`, so v3's DDL fails there and v2 is the only index it
    can have. It lives in LEGACY_INDEX_DDL rather than REQUIRED_INDEX_DDL
    because the latter is pinned against schema.sql, which declares the
    CURRENT generation only."""
    assert index_ddl(BM25_INDEX_V3) is not None
    assert index_ddl(BM25_INDEX_V2) is not None
    assert BM25_INDEX_V3 in REQUIRED_INDEX_DDL
    assert BM25_INDEX_V2 in LEGACY_INDEX_DDL
    assert BM25_INDEX_V2 not in REQUIRED_INDEX_DDL


def test_only_v3_carries_project_id() -> None:
    assert "project_id" in index_ddl(BM25_INDEX_V3)
    assert "project_id" not in index_ddl(BM25_INDEX_V2)


def test_the_v3_ddl_matches_what_schema_sql_declares() -> None:
    """A rebuild that recreates a DIFFERENT index definition than the schema
    declares is worse than no rebuild, because the result LOOKS healthy. Same
    check the existing index contracts make, applied to the new generation."""
    import pathlib
    import re

    schema = pathlib.Path(__file__).resolve().parents[1] / "db" / "schema.sql"
    text = schema.read_text()
    assert "idx_chunks_bm25_v3" in text, "schema.sql must declare v3"

    def _fields(ddl: str) -> set[str]:
        body = re.search(r"USING bm25 \(([^)]*)\)", ddl, re.S)
        assert body, f"no field list found in: {ddl[:80]}"
        return {f.strip() for f in body.group(1).split(",") if f.strip()}

    schema_block = text[text.index("CREATE INDEX IF NOT EXISTS idx_chunks_bm25_v3"):]
    assert _fields(REQUIRED_INDEX_DDL[BM25_INDEX_V3]) == _fields(schema_block)
