"""Workflow memory's tables are gone, and migration 0142 is what removes them.

The feature (team rules) was removed in #595 with its tables and rows kept.
Migration 0142 drops the five tables with their rows, the two trigger functions,
and the six capability keys 0115 wrote into `customers.preferences` (owner
decision 2026-09-26; irreversible).

CI builds its database from db/schema.sql and stamps head -- it never replays
the chain -- so two separate things need proving, and neither implies the other:

* the MIGRATION removes everything 0114-0118 created, on a database that has
  it, which is what production is; and
* SCHEMA.SQL creates none of it, which is what every fresh database is.

Both run in scratch databases created here, never in the shared test database.
Leftovers are found by NAME PATTERN rather than by the lists below, so an object
those lists forgot still counts as a leftover.
"""

from __future__ import annotations

import importlib.util
import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from types import ModuleType
from urllib.parse import urlparse, urlunparse
from uuid import uuid4

import asyncpg
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_VERSIONS_DIR = _REPO_ROOT / "db" / "migrations" / "versions"
_SCHEMA_PATH = _REPO_ROOT / "db" / "schema.sql"

#: The migrations that built workflow memory, in apply order. 0119 is left out:
#: it only backfills rows through op.get_bind() and creates no object.
_CREATING_MIGRATIONS = (
    "20260820_0114_workflow_memory_store.py",
    "20260820_0115_wfmem_capability_prefs.py",
    "20260820_0116_wfmem_clause_publication.py",
    "20260820_0117_wfmem_clause_embedding.py",
    "20260821_0118_wfmem_situation_fallback.py",
)
_DROP_MIGRATION = "20260926_0142_drop_workflow_memory.py"

_TABLES = ("situations", "clauses", "clause_situation_edges", "clause_evidence", "serve_ledger")
_FUNCTIONS = ("wfmem_touch_updated_at", "wfmem_clear_stale_clause_embedding")

#: Tables, indexes, the serve_ledger sequence, row types and their array types
#: (`_clauses`), policies on those tables.
_RELATION_RE = "^_?(situation|clause|serve_ledger|wfmem_)"
#: Functions, and triggers calling them wherever they sit.
_FUNCTION_RE = "^wfmem_"

#: schema.sql references neon_auth, which Neon provisions in prod and CI shims
#: in by hand (.github/workflows/tests.yml, "Provision neon_auth shim").
_NEON_AUTH_SHIM = """
CREATE SCHEMA IF NOT EXISTS neon_auth;
CREATE TABLE IF NOT EXISTS neon_auth.organization (id UUID PRIMARY KEY);
CREATE TABLE IF NOT EXISTS neon_auth."user" (
    id              UUID PRIMARY KEY,
    organization_id UUID REFERENCES neon_auth.organization(id),
    email           TEXT,
    name            TEXT
);
"""

_LEFTOVERS_SQL = """
    SELECT 'relation', relname FROM pg_class
     WHERE relnamespace = 'public'::regnamespace AND relname ~ $1
    UNION ALL
    SELECT 'type', typname FROM pg_type
     WHERE typnamespace = 'public'::regnamespace AND typname ~ $1
    UNION ALL
    SELECT 'policy', tablename || '.' || policyname FROM pg_policies
     WHERE schemaname = 'public' AND tablename ~ $1
    UNION ALL
    SELECT 'function', proname FROM pg_proc
     WHERE pronamespace = 'public'::regnamespace AND proname ~ $2
    UNION ALL
    SELECT 'trigger', t.tgname FROM pg_trigger t JOIN pg_proc p ON p.oid = t.tgfoid
     WHERE NOT t.tgisinternal AND p.proname ~ $2
    ORDER BY 1, 2
"""


async def _leftovers(conn: asyncpg.Connection) -> list[tuple[str, str]]:
    return [tuple(r) for r in await conn.fetch(_LEFTOVERS_SQL, _RELATION_RE, _FUNCTION_RE)]


class _Recorder:
    """Stands in for alembic's `op`, collecting what `op.execute` is given.

    Anything else raises: a migration step this cannot replay would otherwise be
    skipped, and the test would pass on a migration it had only half run.
    """

    def __init__(self) -> None:
        self.statements: list[str] = []

    def execute(self, sql: object) -> None:
        self.statements.append(str(sql))

    def __getattr__(self, name: str) -> object:
        raise AssertionError(
            f"migration used op.{name}(), which this test cannot replay; "
            "write it as op.execute(...) with raw SQL"
        )


def _load(filename: str) -> ModuleType:
    path = _VERSIONS_DIR / filename
    assert path.exists(), f"{filename} is missing from db/migrations/versions"
    spec = importlib.util.spec_from_file_location(f"wfmem_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _upgrade_statements(filename: str) -> list[str]:
    module = _load(filename)
    recorder = _Recorder()
    module.op = recorder  # type: ignore[attr-defined]
    module.upgrade()
    return recorder.statements


def _dsn_for(dbname: str) -> str:
    parsed = urlparse(os.environ["DATABASE_URL"])
    return urlunparse(parsed._replace(path=f"/{dbname}"))


@asynccontextmanager
async def _scratch_db(label: str) -> AsyncIterator[asyncpg.Connection]:
    name = f"wfmem_{label}_{os.getpid()}_{uuid4().hex[:8]}"
    admin = await asyncpg.connect(_dsn_for("postgres"))
    try:
        await admin.execute(f'CREATE DATABASE "{name}"')
    finally:
        await admin.close()
    try:
        conn = await asyncpg.connect(_dsn_for(name))
        try:
            # clauses.body_embedding is halfvec (0117); a fresh database has no
            # pgvector until someone creates it.
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            yield conn
        finally:
            await conn.close()
    finally:
        admin = await asyncpg.connect(_dsn_for("postgres"))
        try:
            await admin.execute(f'DROP DATABASE IF EXISTS "{name}"')
        finally:
            await admin.close()


async def _seed_one_row_per_table(conn: asyncpg.Connection, customer_id: str) -> None:
    """Rows in all five, so the drop is shown to work on tables that hold data.

    Runs as the test role, a superuser, so FORCE RLS does not filter these
    inserts. That is fine here: what is under test is DDL, which RLS never
    filters.
    """
    situation_id = await conn.fetchval(
        "INSERT INTO situations (customer_id, slug, label, description) "
        "VALUES ($1, 'open-pr', 'Open a PR', 'd') RETURNING id",
        customer_id,
    )
    clause_id = await conn.fetchval(
        "INSERT INTO clauses (customer_id, kind, body, status, author_ref) "
        "VALUES ($1, 'step', 'Run the tests first', 'declared', 'user:seed') RETURNING id",
        customer_id,
    )
    await conn.execute(
        "INSERT INTO clause_situation_edges (customer_id, clause_id, situation_id) "
        "VALUES ($1, $2, $3)",
        customer_id,
        clause_id,
        situation_id,
    )
    await conn.execute(
        "INSERT INTO clause_evidence "
        "(customer_id, clause_id, source_class, source_ref, exposure_tainted, ts) "
        "VALUES ($1, $2, 'declared', '{\"session\": \"s1\"}', false, now())",
        customer_id,
        clause_id,
    )
    await conn.execute(
        "INSERT INTO serve_ledger (customer_id, clause_ids, session_id, channel) "
        "VALUES ($1, ARRAY[$2::uuid], 's1', 'retrieved')",
        customer_id,
        clause_id,
    )


@pytest.mark.asyncio
async def test_0142_removes_everything_the_workflow_memory_migrations_created(live_db):
    async with _scratch_db("drop") as conn:
        # Only what the five tables and 0115 need from customers.
        await conn.execute(
            "CREATE TABLE customers ("
            " customer_id TEXT PRIMARY KEY,"
            " preferences JSONB NOT NULL DEFAULT '{}'::jsonb)"
        )
        await conn.execute(
            """
            INSERT INTO customers (customer_id, preferences) VALUES
                ('tenant', '{"code_graph_branch_overrides": {"acme/api": "develop"}}'),
                ('junk-prefs', '"not an object"')
            """
        )
        for filename in _CREATING_MIGRATIONS:
            for statement in _upgrade_statements(filename):
                await conn.execute(statement)
        await _seed_one_row_per_table(conn, "tenant")

        # Non-vacuity: the objects are really there before the drop, so an
        # empty result afterwards means they were removed, not never made.
        before = await _leftovers(conn)
        present = {name for _, name in before}
        assert set(_TABLES) <= present, before
        assert set(_FUNCTIONS) <= present, before
        assert any(kind == "policy" for kind, _ in before), before
        assert any(kind == "trigger" for kind, _ in before), before
        before_prefs = json.loads(
            await conn.fetchval("SELECT preferences FROM customers WHERE customer_id = 'tenant'")
        )
        assert any(key.startswith("wfmem_") for key in before_prefs), before_prefs

        for statement in _upgrade_statements(_DROP_MIGRATION):
            await conn.execute(statement)

        assert await _leftovers(conn) == []
        # customers itself survives, and only the wfmem keys left its preferences.
        prefs = {
            row["customer_id"]: json.loads(row["preferences"])
            for row in await conn.fetch("SELECT customer_id, preferences FROM customers")
        }
        assert prefs == {
            "tenant": {"code_graph_branch_overrides": {"acme/api": "develop"}},
            "junk-prefs": "not an object",
        }


@pytest.mark.asyncio
async def test_schema_sql_creates_no_workflow_memory_object(live_db):
    async with _scratch_db("schema") as conn:
        await conn.execute(_NEON_AUTH_SHIM)
        await conn.execute(_SCHEMA_PATH.read_text())
        # Non-vacuity: schema.sql really ran.
        assert await conn.fetchval("SELECT to_regclass('public.customers')") is not None
        assert await _leftovers(conn) == []


def test_0142_downgrade_refuses() -> None:
    """The rows were deleted, not archived; a downgrade must say so, not pretend."""
    module = _load(_DROP_MIGRATION)
    recorder = _Recorder()
    module.op = recorder  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError, match="not reversible"):
        module.downgrade()
    assert recorder.statements == []
