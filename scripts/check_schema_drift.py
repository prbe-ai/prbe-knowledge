#!/usr/bin/env python3
"""Compare the TABLE and INDEX names db/schema.sql declares against a live DB.

    uv run python scripts/check_schema_drift.py            # uses DATABASE_URL
    uv run python scripts/check_schema_drift.py --json     # machine-readable

Exit 0 = no drift. Exit 1 = something schema.sql declares is missing. Exit 2 =
could not check (bad DSN, unreachable).

WHY THIS EXISTS
---------------
A fresh database is bootstrapped by `scripts/migrate.py` from `db/schema.sql`
followed by `alembic stamp head` -- the chain is never replayed, because
migrations 0007+ duplicate state schema.sql already creates. The consequence is
that **anything added to schema.sql after a plane was bootstrapped is missing on
that plane forever**, while `alembic_version` cheerfully reports head.

That is not hypothetical. `system_settings` was added to schema.sql on 2026-07-15
(251bdc9) and was still absent from managed-shared on 2026-08-20, five weeks and
many deploys later. Its reader fails open, so the only symptom was 4,271 log
lines a day and a global ingestion killswitch that silently did nothing.

WHAT THIS CATCHES, AND WHAT IT DOES NOT
---------------------------------------
Read this before trusting a green result.

    CATCHES    a table declared in schema.sql that is absent from the database
               an index declared in schema.sql that is absent from the database

    MISSES     missing or wrong COLUMNS on a table that exists
               missing CONSTRAINTS, triggers, defaults
               missing GRANTS  <-- the neon_auth bug of 2026-08-20 was exactly
                                   this, and this check would NOT have caught it
               missing ROW LEVEL SECURITY policies
               missing SEED ROWS (an empty system_settings passes this check)
               extensions, ownership, function bodies

    ALSO       drift in the other direction is reported as INFO, not failure.
               The database legitimately holds tables migrations added that were
               never folded back into schema.sql, and schema.sql legitimately
               holds stale declarations (research is missing
               idx_chunks_fts_content because the BM25 cleanup dropped it and
               the file was never updated).

A green run means "no declared table or index is missing". It does NOT mean "the
schema is correct". Full semantic parity between the two construction paths is a
real project -- it needs one authority generated from the other -- and this is
deliberately not that. It is the cheap check that would have caught the bug that
actually happened.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

import asyncpg

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCHEMA_SQL = _REPO_ROOT / "db" / "schema.sql"

#: Schemas a declared object may legitimately live in. prbe-knowledge's tables
#: sit in ag_catalog on some deployments (migration 0066) and public on others,
#: so a name found in either counts as present.
_SEARCH_SCHEMAS = ("public", "ag_catalog")

_TABLE_RE = re.compile(r"CREATE TABLE (?:IF NOT EXISTS )?([a-z0-9_]+)", re.IGNORECASE)
_INDEX_RE = re.compile(
    r"CREATE (?:UNIQUE )?INDEX (?:CONCURRENTLY )?(?:IF NOT EXISTS )?([a-z0-9_]+)",
    re.IGNORECASE,
)
#: `DO $$ ... $$;` bodies. Everything inside one is CONDITIONAL by construction
#: and must not be read as a declaration -- schema.sql guards the pg_search BM25
#: index that way because the extension ships in `prbe-postgres` but not in the
#: `pgvector/pgvector` image local dev and CI use. Scanning inside the block made
#: every pgvector-based database report drift for an index it is correct not to
#: have, which is precisely the false positive that gets a checker ignored.
_DO_BLOCK_RE = re.compile(r"DO\s*\$\$.*?\$\$\s*;", re.DOTALL | re.IGNORECASE)


def declared_objects(schema_sql: str) -> tuple[set[str], set[str]]:
    """Table and index names schema.sql declares UNCONDITIONALLY.

    Regex, not a SQL parser, and that is a deliberate trade: the alternative is
    a parser dependency for a check whose whole value is being cheap enough to
    run on every deploy. The patterns only match statement-leading CREATE forms,
    which is the shape schema.sql uses throughout.
    """
    unconditional = _DO_BLOCK_RE.sub("", schema_sql)
    return set(_TABLE_RE.findall(unconditional)), set(_INDEX_RE.findall(unconditional))


async def live_objects(conn: asyncpg.Connection) -> tuple[set[str], set[str]]:
    tables = {
        r["relname"]
        for r in await conn.fetch(
            """
            SELECT c.relname
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relkind = 'r' AND n.nspname = ANY($1::text[])
            """,
            list(_SEARCH_SCHEMAS),
        )
    }
    indexes = {
        r["indexname"]
        for r in await conn.fetch(
            "SELECT indexname FROM pg_indexes WHERE schemaname = ANY($1::text[])",
            list(_SEARCH_SCHEMAS),
        )
    }
    return tables, indexes


async def check(dsn: str) -> dict:
    declared_t, declared_i = declared_objects(_SCHEMA_SQL.read_text())
    conn = await asyncpg.connect(dsn)
    try:
        live_t, live_i = await live_objects(conn)
        version = await conn.fetchval(
            "SELECT version_num FROM alembic_version LIMIT 1"
        )
    finally:
        await conn.close()

    return {
        "alembic_version": version,
        "missing_tables": sorted(declared_t - live_t),
        "missing_indexes": sorted(declared_i - live_i),
        "declared_tables": len(declared_t),
        "declared_indexes": len(declared_i),
    }


def _render(result: dict) -> bool:
    """Print a human report. Returns True when drift was found."""
    missing_t = result["missing_tables"]
    missing_i = result["missing_indexes"]
    print(f"alembic_version: {result['alembic_version']}")
    print(
        f"declared in schema.sql: {result['declared_tables']} tables, "
        f"{result['declared_indexes']} indexes"
    )
    if not missing_t and not missing_i:
        print("OK - every declared table and index is present.")
        print("NOTE: this does not check columns, constraints, grants, RLS or seed rows.")
        return False

    print("\nDRIFT - schema.sql declares objects this database does not have.")
    print("This is the stamp-head bootstrap gap: alembic reports head, but the")
    print("migration that would have created these never ran here and never will.")
    for name in missing_t:
        print(f"  MISSING TABLE: {name}")
    for name in missing_i:
        print(f"  MISSING INDEX: {name}")
    print("\nFix with a repair migration (see 0112_repair_bootstrap_drift), not by")
    print("hand-applying DDL to one plane -- the other planes need it too.")
    return True


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", default=None, help="defaults to $DATABASE_URL")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()

    dsn = args.dsn or os.environ.get("DATABASE_URL", "")
    if not dsn:
        print("no DSN: pass --dsn or set DATABASE_URL", file=sys.stderr)
        return 2
    dsn = dsn.replace("postgresql+asyncpg://", "postgresql://")

    try:
        result = await check(dsn)
    except Exception as exc:  # unreachable DB, bad DSN, missing alembic_version
        print(f"could not check: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result, indent=2))
        return 1 if (result["missing_tables"] or result["missing_indexes"]) else 0

    return 1 if _render(result) else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
