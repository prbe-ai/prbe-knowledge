"""The rebuild's copy of the index DDL must not drift from db/schema.sql.

WHY THIS EXISTS
---------------
`REQUIRED_INDEX_DDL` duplicates a CREATE INDEX statement that db/schema.sql
already owns, so that the unattended rebuild does not have to parse SQL at
runtime. Duplication is the right trade here and it has exactly one failure
mode: schema.sql changes, the copy does not, and the rebuild silently recreates
last month's index. That result LOOKS healthy -- the index exists, it is valid,
the guardian goes quiet -- while serving a definition nobody chose. This test is
the thing that makes the duplication safe.

Same shape as tests for index_contracts.py, and for the same reason.
"""

from __future__ import annotations

import re
from pathlib import Path

from engine.shared.pg_search_guardian import (
    REQUIRED_INDEX_DDL,
    REQUIRED_PG_SEARCH_INDEXES,
)

SCHEMA_SQL = Path(__file__).resolve().parents[1] / "db" / "schema.sql"


def _normalise(sql: str) -> str:
    """Strip `--` comments and collapse whitespace.

    Comments are stripped because schema.sql carries a long explanation of the
    source_code tokenizer inside the WITH block that the runtime copy has no
    reason to repeat. Whitespace is collapsed because the two live at different
    indentation depths.
    """
    without_comments = re.sub(r"--[^\n]*", " ", sql)
    return re.sub(r"\s+", " ", without_comments).strip()


def test_every_required_index_has_rebuild_ddl() -> None:
    """A required index with no DDL is an index the rebuild cannot restore.

    This is the failure that would otherwise surface only during an incident,
    as a ValueError from a CronJob at 3am.
    """
    missing = set(REQUIRED_PG_SEARCH_INDEXES) - set(REQUIRED_INDEX_DDL)
    assert not missing, (
        f"required pg_search indexes with no rebuild DDL: {sorted(missing)}. "
        "Add the CREATE INDEX statement to REQUIRED_INDEX_DDL."
    )


def test_rebuild_ddl_has_no_index_the_guardian_does_not_require() -> None:
    """The reverse drift: DDL for something nothing asks the guardian to miss."""
    extra = set(REQUIRED_INDEX_DDL) - set(REQUIRED_PG_SEARCH_INDEXES)
    assert not extra, (
        f"rebuild DDL declared for non-required indexes: {sorted(extra)}. "
        "Either add them to REQUIRED_PG_SEARCH_INDEXES or drop the DDL."
    )


def test_rebuild_ddl_matches_schema_sql() -> None:
    """The statement the rebuild runs is the statement schema.sql declares."""
    schema = _normalise(SCHEMA_SQL.read_text())
    for index_name, ddl in sorted(REQUIRED_INDEX_DDL.items()):
        normalised = _normalise(ddl)
        assert normalised in schema, (
            f"{index_name}: REQUIRED_INDEX_DDL has drifted from db/schema.sql.\n"
            f"  rebuild copy: {normalised}\n"
            "Fix whichever one moved. Do NOT edit the copy to match a schema "
            "change without confirming the new definition is the one you want "
            "rebuilt unattended."
        )


def test_rebuild_uses_a_bounded_memory_budget() -> None:
    """The rebuild must not inherit the instance's maintenance_work_mem.

    512MB x an already-tight 4Gi container is how the primary got OOM
    group-killed twice; the rebuild is the single largest thing this codebase
    starts on purpose, so its budget is pinned rather than inherited.
    """
    from engine.shared.pg_search_guardian import (
        REBUILD_MAINTENANCE_WORK_MEM,
        REBUILD_PARALLEL_WORKERS,
    )

    assert REBUILD_MAINTENANCE_WORK_MEM.endswith("MB")
    assert int(REBUILD_MAINTENANCE_WORK_MEM.removesuffix("MB")) <= 256
    assert REBUILD_PARALLEL_WORKERS == 0
