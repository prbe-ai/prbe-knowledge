"""Tenant isolation for the workflow-memory procedure store.

Mirrors tests/test_multitenant_isolation.py -- NOT
test_rls_cross_tenant_denial.py, whose isolation tests skip under the dev
superuser role and have therefore never executed.

Both halves of the policy matter and fail differently: USING hides another
tenant's rows on read, WITH CHECK refuses a write that would land under
another tenant. A table with USING only looks correct until a buggy writer
silently files a row under the wrong customer.

WHAT THIS FILE HAS TO DEFEND, and how each claim is made to bite. A mutation
review of the first revision got a green suite while tenant A read tenant B's
rows, because four of five tables were covered only by a substring check on
the policy text and the composite-FK claim had no test at all:

* Behavioural isolation is parametrized over ALL FIVE tables (read, write,
  update, delete), not asserted on `clauses` and assumed for the rest.
* The policy-metadata check is an EXACT match on the rendered predicate and
  on the whole policy set, so `USING (current_setting(...) IS NOT NULL)` --
  which mentions the GUC and isolates nothing -- fails here.
* The composite FKs get their own tests: attaching to another tenant's row
  must raise ForeignKeyViolationError, on both of the edge table's FKs and on
  clause_evidence's.
* `source_ref` cannot smuggle a quote, `serve_ledger` cannot be rewritten,
  `exposure_tainted` cannot be omitted, and clause_evidence's column set is an
  allowlist (an exact-name blocklist let `quote_text` straight through).
* Every refusal asserts on "row-level security" in the message: a missing
  GRANT raises the same SQLSTATE 42501 and would otherwise look like a pass.
"""

from __future__ import annotations

import importlib.util
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse
from uuid import uuid4

import asyncpg
import pytest
import pytest_asyncio

from engine.shared.db import raw_conn, with_tenant

TENANT_A = "cust-wfmem-a"
TENANT_B = "cust-wfmem-b"


#: The dev/CI role is a SUPERUSER and superusers bypass RLS outright, so a test
#: run as `prbe` does not pass vacuously -- it FAILS, because A really does see
#: B's rows. Every RLS assertion below must run as a non-privileged role.
#: Copied from tests/test_multitenant_isolation.py:18-60, the one file in this
#: repo whose isolation tests actually execute.
RLS_ROLE = "prbe_rls_test"

WFMEM_TABLES = (
    "situations",
    "clauses",
    "clause_situation_edges",
    "clause_evidence",
    "serve_ledger",
)


@pytest_asyncio.fixture
async def two_tenants(live_db) -> AsyncIterator[tuple[str, str]]:
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO customers (customer_id, display_name, api_key_hash)
            VALUES ($1, 'wfmem-a', 'h-wfmem-a'), ($2, 'wfmem-b', 'h-wfmem-b')
            ON CONFLICT (customer_id) DO NOTHING
            """,
            TENANT_A,
            TENANT_B,
        )
        await conn.execute(
            f"""
            DO $$
            BEGIN
                IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{RLS_ROLE}') THEN
                    CREATE ROLE {RLS_ROLE} NOSUPERUSER NOBYPASSRLS;
                END IF;
            END $$;
            """
        )
        await conn.execute(f"GRANT USAGE ON SCHEMA public TO {RLS_ROLE}")
        await conn.execute(f"GRANT ALL ON ALL TABLES IN SCHEMA public TO {RLS_ROLE}")
        # serve_ledger.id is BIGSERIAL -- without this an insert under the role
        # fails on the sequence, not on the policy, and the test lies to you.
        await conn.execute(f"GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO {RLS_ROLE}")
    yield TENANT_A, TENANT_B
    async with raw_conn() as conn:
        await conn.execute(
            "DELETE FROM customers WHERE customer_id = ANY($1::text[])",
            [TENANT_A, TENANT_B],
        )


# ---------------------------------------------------------------------------
# Per-table row shapes.
#
# Each table needs its own minimal valid INSERT, and two of them need a parent
# row first: the edge table needs a clause AND a situation in the same tenant,
# clause_evidence needs a clause. Those parents are created under the OWNING
# tenant's GUC so that a cross-tenant test fails on the policy rather than on a
# dangling reference -- which would pass for the wrong reason.
# ---------------------------------------------------------------------------


async def _insert_clause(conn: asyncpg.Connection, customer_id: str, body: str) -> Any:
    return await conn.fetchval(
        """
        INSERT INTO clauses (customer_id, kind, body, status, author_ref)
        VALUES ($1, 'step', $2, 'declared', 'user:seed')
        RETURNING id
        """,
        customer_id,
        body,
    )


async def _insert_situation(conn: asyncpg.Connection, customer_id: str, slug: str) -> Any:
    return await conn.fetchval(
        """
        INSERT INTO situations (customer_id, slug, label, description)
        VALUES ($1, $2, 'label', 'description')
        RETURNING id
        """,
        customer_id,
        slug,
    )


async def _prepare(table: str, customer_id: str) -> dict[str, Any]:
    """Create whatever parent rows `table` needs, owned by `customer_id`."""
    if table not in ("clause_situation_edges", "clause_evidence"):
        return {}
    async with with_tenant(customer_id) as conn:
        await conn.execute(f"SET LOCAL ROLE {RLS_ROLE}")
        ctx: dict[str, Any] = {
            "clause_id": await _insert_clause(conn, customer_id, f"parent-{uuid4()}")
        }
        if table == "clause_situation_edges":
            ctx["situation_id"] = await _insert_situation(conn, customer_id, f"sit-{uuid4()}")
    return ctx


def _insert_statement(
    table: str, customer_id: str, marker: str, ctx: dict[str, Any]
) -> tuple[str, tuple[Any, ...]]:
    """The minimal valid INSERT for `table`, tagged with `marker`."""
    if table == "situations":
        return (
            "INSERT INTO situations (customer_id, slug, label, description) "
            "VALUES ($1, $2, 'label', 'description')",
            (customer_id, marker),
        )
    if table == "clauses":
        return (
            "INSERT INTO clauses (customer_id, kind, body, status, author_ref) "
            "VALUES ($1, 'step', $2, 'declared', 'user:seed')",
            (customer_id, marker),
        )
    if table == "clause_situation_edges":
        return (
            "INSERT INTO clause_situation_edges "
            "(customer_id, clause_id, situation_id, classification) "
            "VALUES ($1, $2, $3, jsonb_build_object('marker', $4::text))",
            (customer_id, ctx["clause_id"], ctx["situation_id"], marker),
        )
    if table == "clause_evidence":
        return (
            "INSERT INTO clause_evidence "
            "(customer_id, clause_id, source_class, source_ref, exposure_tainted, ts) "
            "VALUES ($1, $2, 'declared', jsonb_build_object('marker', $3::text), false, now())",
            (customer_id, ctx["clause_id"], marker),
        )
    if table == "serve_ledger":
        return (
            "INSERT INTO serve_ledger (customer_id, clause_ids, session_id, channel) "
            "VALUES ($1, ARRAY[gen_random_uuid()], $2, 'retrieved')",
            (customer_id, marker),
        )
    raise AssertionError(f"no INSERT shape defined for {table}")


#: How to find a seeded row again, by its marker ($1).
_MARKER_PREDICATE = {
    "situations": "slug = $1",
    "clauses": "body = $1",
    "clause_situation_edges": "classification->>'marker' = $1",
    "clause_evidence": "source_ref->>'marker' = $1",
    "serve_ledger": "session_id = $1",
}

#: A mutation an attacking tenant would attempt on someone else's row.
_HIJACK_SET = {
    "situations": "label = 'hijacked'",
    "clauses": "body = 'hijacked'",
    "clause_situation_edges": "when_conditions = '{\"hijacked\": true}'::jsonb",
    "clause_evidence": "author_ref = 'hijacked'",
    "serve_ledger": "served_at = '2001-01-01T00:00:00+00'",
}


async def _seed(table: str, customer_id: str, marker: str) -> dict[str, Any]:
    """Insert one `marker`-tagged row of `table` owned by `customer_id`."""
    ctx = await _prepare(table, customer_id)
    async with with_tenant(customer_id) as conn:
        await conn.execute(f"SET LOCAL ROLE {RLS_ROLE}")
        sql, args = _insert_statement(table, customer_id, marker, ctx)
        await conn.execute(sql, *args)
    return ctx


# ---------------------------------------------------------------------------
# Behavioural isolation, on every table.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("table", WFMEM_TABLES)
async def test_read_is_tenant_scoped(two_tenants, table):
    """USING: A's unfiltered read must not see B's row."""
    a, b = two_tenants
    await _seed(table, a, "marker-a")
    await _seed(table, b, "marker-b")

    async with with_tenant(a) as conn:
        await conn.execute(f"SET LOCAL ROLE {RLS_ROLE}")
        rows = await conn.fetch(f"SELECT customer_id FROM {table}")
        foreign = await conn.fetch(
            f"SELECT customer_id FROM {table} WHERE {_MARKER_PREDICATE[table]}", "marker-b"
        )

    assert rows, f"A cannot see its own {table} row -- the test proves nothing"
    assert {r["customer_id"] for r in rows} == {a}, f"A sees another tenant's {table} rows"
    assert foreign == [], f"A can address B's {table} row by marker"


@pytest.mark.asyncio
@pytest.mark.parametrize("table", WFMEM_TABLES)
async def test_cross_tenant_insert_is_refused(two_tenants, table):
    """WITH CHECK: under A's GUC, a write claiming B must be rejected.

    ``match`` is load-bearing. A missing GRANT raises InsufficientPrivilege
    too (both are SQLSTATE 42501), so an un-matched raises() would pass on a
    permissions accident with RLS switched off entirely.
    """
    a, b = two_tenants
    # B's parent rows, created as B, so the only thing wrong with the insert
    # below is the tenant it claims.
    ctx = await _prepare(table, b)
    sql, args = _insert_statement(table, b, "smuggled into B", ctx)

    async with with_tenant(a) as conn:
        await conn.execute(f"SET LOCAL ROLE {RLS_ROLE}")
        with pytest.raises(
            asyncpg.exceptions.InsufficientPrivilegeError, match="row-level security"
        ):
            await conn.execute(sql, *args)


@pytest.mark.asyncio
@pytest.mark.parametrize("table", WFMEM_TABLES)
async def test_cross_tenant_update_and_delete_are_no_ops(two_tenants, table):
    """The metadata check proves the policies EXIST. This proves they BITE for
    UPDATE and DELETE, which the INSERT/SELECT tests never exercise -- a
    cross-tenant UPDATE does not raise, it silently matches zero rows, and that
    is the failure mode a reviewer is least likely to notice.
    """
    a, b = two_tenants
    await _seed(table, b, "B's untouchable row")
    predicate = _MARKER_PREDICATE[table]

    async with with_tenant(a) as conn:
        await conn.execute(f"SET LOCAL ROLE {RLS_ROLE}")
        updated = await conn.execute(
            f"UPDATE {table} SET {_HIJACK_SET[table]} WHERE {predicate}", "B's untouchable row"
        )
        deleted = await conn.execute(
            f"DELETE FROM {table} WHERE {predicate}", "B's untouchable row"
        )

    assert updated == "UPDATE 0", f"A's UPDATE reached B's {table} row: {updated}"
    assert deleted == "DELETE 0", f"A's DELETE reached B's {table} row: {deleted}"

    async with with_tenant(b) as conn:
        await conn.execute(f"SET LOCAL ROLE {RLS_ROLE}")
        rows = await conn.fetch(
            f"SELECT customer_id FROM {table} WHERE {predicate}", "B's untouchable row"
        )
    assert len(rows) == 1, f"B's {table} row must survive A's attempts untouched"


@pytest.mark.asyncio
@pytest.mark.parametrize("table", WFMEM_TABLES)
async def test_no_tenant_guc_returns_no_rows(two_tenants, table):
    """Fail-closed: an unbound connection sees NOTHING, rather than everything.

    `current_setting(..., true)` returns NULL with no GUC set, so the policy
    predicate is NULL -> not true -> deny. That is the current behaviour and
    the reason `with_tenant` can be trusted; nothing else guards it.
    """
    a, _ = two_tenants
    await _seed(table, a, "fail-closed")

    async with with_tenant(a) as conn:
        await conn.execute(f"SET LOCAL ROLE {RLS_ROLE}")
        bound = await conn.fetch(f"SELECT 1 FROM {table}")
    assert bound, f"control: A must see its own {table} row when bound"

    async with raw_conn() as conn, conn.transaction():
        # SET LOCAL needs a transaction; outside one it warns and does nothing,
        # which would leave this query running as the RLS-exempt superuser.
        await conn.execute(f"SET LOCAL ROLE {RLS_ROLE}")
        unbound = await conn.fetch(f"SELECT 1 FROM {table}")
    assert unbound == [], f"{table} leaks rows to a connection with no tenant GUC"


# ---------------------------------------------------------------------------
# Policy metadata, asserted exactly.
# ---------------------------------------------------------------------------

#: The rendered predicate, verbatim from pg_policies. An exact match, not a
#: substring: `USING (current_setting('app.current_customer_id', true) IS NOT
#: NULL)` mentions the GUC and isolates nothing, and a substring assertion
#: waves it through.
EXPECTED_POLICY = "(customer_id = current_setting('app.current_customer_id'::text, true))"

#: policyname -> (cmd, qual, with_check)
_FOR_ALL_POLICY = {"tenant_isolation": ("ALL", EXPECTED_POLICY, EXPECTED_POLICY)}

#: serve_ledger is append-only: SELECT + INSERT policies and nothing else, so
#: UPDATE and DELETE hit the default deny. Asserting the WHOLE policy set (not
#: just "a tenant_isolation policy exists") is what makes a later `FOR UPDATE`
#: policy show up as a failure here.
EXPECTED_POLICIES: dict[str, dict[str, tuple[str, str | None, str | None]]] = {
    "situations": _FOR_ALL_POLICY,
    "clauses": _FOR_ALL_POLICY,
    "clause_situation_edges": _FOR_ALL_POLICY,
    "clause_evidence": _FOR_ALL_POLICY,
    "serve_ledger": {
        "tenant_isolation_select": ("SELECT", EXPECTED_POLICY, None),
        "tenant_isolation_insert": ("INSERT", None, EXPECTED_POLICY),
    },
}


@pytest.mark.asyncio
@pytest.mark.parametrize("table", WFMEM_TABLES)
async def test_every_table_has_forced_rls_and_exact_tenant_policies(live_db, table):
    """ENABLE alone is not enough; FORCE without a policy is deny-all; and
    USING without WITH CHECK lets a buggy writer file under another tenant."""
    async with raw_conn() as conn:
        flags = await conn.fetchrow(
            "SELECT relrowsecurity, relforcerowsecurity FROM pg_class "
            "WHERE relname = $1 AND relnamespace = 'public'::regnamespace",
            table,
        )
        policies = await conn.fetch(
            "SELECT policyname, cmd, qual, with_check, permissive FROM pg_policies "
            "WHERE tablename = $1 AND schemaname = 'public'",
            table,
        )

    assert flags is not None, f"{table} does not exist"
    assert flags["relrowsecurity"], f"{table} is missing ENABLE ROW LEVEL SECURITY"
    assert flags["relforcerowsecurity"], f"{table} is missing FORCE ROW LEVEL SECURITY"

    actual = {p["policyname"]: (p["cmd"], p["qual"], p["with_check"]) for p in policies}
    assert actual == EXPECTED_POLICIES[table], (
        f"{table}'s policy set is not the expected tenant isolation. "
        f"Expected {EXPECTED_POLICIES[table]}, got {actual}."
    )
    assert all(p["permissive"] == "PERMISSIVE" for p in policies), (
        f"{table} has a RESTRICTIVE policy; the expectations above assume permissive"
    )


# ---------------------------------------------------------------------------
# Composite foreign keys: the headline security property of migration 0114.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_clause_evidence_cannot_attach_to_another_tenants_clause(two_tenants):
    """A simple `REFERENCES clauses(id)` would let A file evidence against B's
    clause id -- RI checks bypass row security by design, so the policy never
    sees it. The composite (customer_id, id) key is what refuses it.
    """
    a, b = two_tenants
    b_ctx = await _prepare("clause_evidence", b)

    async with with_tenant(a) as conn:
        await conn.execute(f"SET LOCAL ROLE {RLS_ROLE}")
        with pytest.raises(asyncpg.exceptions.ForeignKeyViolationError):
            await conn.execute(
                "INSERT INTO clause_evidence "
                "(customer_id, clause_id, source_class, source_ref, exposure_tainted, ts) "
                "VALUES ($1, $2, 'declared', '{}'::jsonb, false, now())",
                a,
                b_ctx["clause_id"],
            )


@pytest.mark.asyncio
async def test_edge_cannot_attach_to_another_tenants_clause(two_tenants):
    a, b = two_tenants
    a_ctx = await _prepare("clause_situation_edges", a)
    b_ctx = await _prepare("clause_situation_edges", b)

    async with with_tenant(a) as conn:
        await conn.execute(f"SET LOCAL ROLE {RLS_ROLE}")
        with pytest.raises(asyncpg.exceptions.ForeignKeyViolationError):
            await conn.execute(
                "INSERT INTO clause_situation_edges (customer_id, clause_id, situation_id) "
                "VALUES ($1, $2, $3)",
                a,
                b_ctx["clause_id"],
                a_ctx["situation_id"],
            )


@pytest.mark.asyncio
async def test_edge_cannot_attach_to_another_tenants_situation(two_tenants):
    a, b = two_tenants
    a_ctx = await _prepare("clause_situation_edges", a)
    b_ctx = await _prepare("clause_situation_edges", b)

    async with with_tenant(a) as conn:
        await conn.execute(f"SET LOCAL ROLE {RLS_ROLE}")
        with pytest.raises(asyncpg.exceptions.ForeignKeyViolationError):
            await conn.execute(
                "INSERT INTO clause_situation_edges (customer_id, clause_id, situation_id) "
                "VALUES ($1, $2, $3)",
                a,
                a_ctx["clause_id"],
                b_ctx["situation_id"],
            )


# ---------------------------------------------------------------------------
# Evidence is by reference -- on both axes.
# ---------------------------------------------------------------------------

#: The complete column set. An ALLOWLIST, not a blocklist of quote-ish names:
#: a blocklist of exact names lets `quote_text` -- the name anyone would
#: actually pick -- straight through, and any other affix besides.
EXPECTED_EVIDENCE_COLUMNS = {
    "id",
    "customer_id",
    "clause_id",
    "source_class",
    "source_ref",
    "author_ref",
    "exposure_tainted",
    "ts",
    "created_at",
}


@pytest.mark.asyncio
async def test_clause_evidence_columns_are_exactly_the_reference_shape(live_db):
    async with raw_conn() as conn:
        rows = await conn.fetch(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'clause_evidence' AND table_schema = current_schema()"
        )
    columns = {r["column_name"] for r in rows}
    assert columns == EXPECTED_EVIDENCE_COLUMNS, (
        "clause_evidence schema changed; evidence is stored by reference only -- "
        f"unexpected: {sorted(columns - EXPECTED_EVIDENCE_COLUMNS)}, "
        f"missing: {sorted(EXPECTED_EVIDENCE_COLUMNS - columns)}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["quote", "text", "full_text", "verbatim", "passage"])
async def test_clause_evidence_refuses_quote_bearing_source_ref(two_tenants, key):
    """The absent column is only half the guarantee: source_ref is
    unconstrained JSONB, and an entire private message fits inside it.
    """
    a, _ = two_tenants
    ctx = await _prepare("clause_evidence", a)

    async with with_tenant(a) as conn:
        await conn.execute(f"SET LOCAL ROLE {RLS_ROLE}")
        with pytest.raises(asyncpg.exceptions.CheckViolationError):
            await conn.execute(
                "INSERT INTO clause_evidence "
                "(customer_id, clause_id, source_class, source_ref, exposure_tainted, ts) "
                "VALUES ($1, $2, 'human_message', jsonb_build_object($3::text, $4::text), "
                "false, now())",
                a,
                ctx["clause_id"],
                key,
                "we always deploy on Fridays, do not tell anyone",
            )


@pytest.mark.asyncio
async def test_clause_evidence_accepts_a_pointer_source_ref(two_tenants):
    """The counterpart: a real pointer must still go in."""
    a, _ = two_tenants
    ctx = await _prepare("clause_evidence", a)

    async with with_tenant(a) as conn:
        await conn.execute(f"SET LOCAL ROLE {RLS_ROLE}")
        await conn.execute(
            "INSERT INTO clause_evidence "
            "(customer_id, clause_id, source_class, source_ref, exposure_tainted, ts) "
            "VALUES ($1, $2, 'human_message', "
            '\'{"session": "sess-1", "span": [0, 1]}\'::jsonb, false, now())',
            a,
            ctx["clause_id"],
        )
        stored = await conn.fetchval("SELECT count(*) FROM clause_evidence")
    assert stored == 1


@pytest.mark.asyncio
async def test_exposure_tainted_must_be_stated(two_tenants):
    """No DEFAULT: a writer that forgets the taint computation must be refused,
    not silently recorded as clean. Fail closed, per the design's
    taint-excluded non-negotiable.
    """
    a, _ = two_tenants
    ctx = await _prepare("clause_evidence", a)

    async with with_tenant(a) as conn:
        await conn.execute(f"SET LOCAL ROLE {RLS_ROLE}")
        with pytest.raises(asyncpg.exceptions.NotNullViolationError):
            await conn.execute(
                "INSERT INTO clause_evidence "
                "(customer_id, clause_id, source_class, source_ref, ts) "
                "VALUES ($1, $2, 'declared', '{}'::jsonb, now())",
                a,
                ctx["clause_id"],
            )


# ---------------------------------------------------------------------------
# serve_ledger is append-only.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_serve_ledger_is_append_only_for_its_own_tenant(two_tenants):
    """Cross-tenant denial is not the whole story here: a tenant rewriting its
    OWN exposure history defeats the taint join just as thoroughly. Backdating
    is the worse half -- it inverts "was this served before that evidence?"
    without deleting anything a reviewer would notice missing.
    """
    a, _ = two_tenants
    await _seed("serve_ledger", a, "own-session")

    async with with_tenant(a) as conn:
        await conn.execute(f"SET LOCAL ROLE {RLS_ROLE}")
        before = await conn.fetchval(
            "SELECT served_at FROM serve_ledger WHERE session_id = $1", "own-session"
        )
        assert before is not None, "control: SELECT must still work for the owning tenant"

        backdated = await conn.execute(
            "UPDATE serve_ledger SET served_at = '2001-01-01T00:00:00+00' WHERE session_id = $1",
            "own-session",
        )
        erased = await conn.execute("DELETE FROM serve_ledger WHERE session_id = $1", "own-session")

    assert backdated == "UPDATE 0", f"a tenant backdated its own ledger row: {backdated}"
    assert erased == "DELETE 0", f"a tenant erased its own ledger row: {erased}"

    async with with_tenant(a) as conn:
        await conn.execute(f"SET LOCAL ROLE {RLS_ROLE}")
        after = await conn.fetchval(
            "SELECT served_at FROM serve_ledger WHERE session_id = $1", "own-session"
        )
    assert after == before, "the ledger row changed despite the append-only policy set"


# ---------------------------------------------------------------------------
# schema.sql / migration drift.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_VERSIONS_DIR = _REPO_ROOT / "db" / "migrations" / "versions"

#: EVERY migration that touches workflow-memory DDL, in apply order.
#:
#: This was a single path until 0112, and the singular was a latent bug rather
#: than a simplification: schema.sql accumulates the effect of the WHOLE chain,
#: so comparing it against one migration means every later one is unchecked. It
#: failed the moment 0112 added two columns -- correctly, and for the wrong
#: reason, reporting drift when the real problem was that the guard could only
#: see half the migrations.
#:
#: ADD EVERY NEW WFMEM MIGRATION HERE. A migration left off this list is not
#: caught by anything: CI applies schema.sql and stamps the head, so the chain
#: never runs there, and this comparison is the only thing keeping the two
#: files honest. The completeness test below fails if one is missed.
_MIGRATION_PATHS = (
    _VERSIONS_DIR / "20260820_0114_workflow_memory_store.py",
    _VERSIONS_DIR / "20260820_0116_wfmem_clause_publication.py",
    _VERSIONS_DIR / "20260820_0117_wfmem_clause_embedding.py",
    _VERSIONS_DIR / "20260821_0118_wfmem_situation_fallback.py",
)
_SCHEMA_PATH = _REPO_ROOT / "db" / "schema.sql"

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

#: All five tables FK to customers; nothing else about it matters here.
_CUSTOMERS_STUB = "CREATE TABLE customers (customer_id TEXT PRIMARY KEY)"

#: Both scratch databases need pgvector: `clauses.body_embedding` is a
#: `halfvec(3072)` (migration 0117). The real database gets this from the image;
#: a freshly-CREATEd one does not, and without it the replay dies on
#: `type "halfvec" does not exist` -- which reads like a schema bug rather than
#: a missing extension in a throwaway database.
_VECTOR_EXTENSION = "CREATE EXTENSION IF NOT EXISTS vector"

#: Which relations count as workflow-memory objects, as a regex rather than as
#: WFMEM_TABLES. The list is what nearly made this guard useless: it only ever
#: covered the objects somebody remembered to name, so a sixth table added by a
#: later migration would be invisible to the very test whose job is to notice.
#: A prefix rule fails safe as the schema grows. WFMEM_TABLES stays where an
#: explicit set is genuinely wanted -- the per-table structural tests above,
#: which must fail loudly if a table goes missing entirely.
_WFMEM_RELATION_RE = "^(situation|clause|serve_ledger|wfmem_)"

#: Functions and off-table triggers are matched on the `wfmem_` prefix ALONE.
#: That is not a stylistic choice: the schema.sql database also contains this
#: repo's other functions (customers_fill_r2_bucket,
#: verify_and_touch_custom_ingest_token) and their triggers, which the
#: migration-replay database has no reason to contain, so an unfiltered
#: comparison would fail on every run. The cost is real and worth stating: a
#: future wfmem function that does NOT carry the prefix is invisible here.
_WFMEM_FUNCTION_RE = "^wfmem_"


def _migration_upgrade_statements() -> list[str]:
    """The SQL every wfmem migration would run, in order, without alembic.

    Imports each migration and swaps its `op` for a recorder, so this stays
    honest if someone edits one -- it replays what is in the files, not a copy
    of them.

    THE RECORDER ONLY UNDERSTANDS `op.execute`. That is why those migrations are
    written in raw SQL rather than with `add_column` / `create_index`: an op
    helper would record nothing here, and the guard would compare schema.sql
    against a migration it had only half read AND PASS. A silent pass is the
    worst outcome available to this test, so the recorder REFUSES anything else
    rather than ignoring it.
    """
    collected: list[str] = []

    class _Recorder:
        @staticmethod
        def execute(sql: object) -> None:
            collected.append(str(sql))

        def __getattr__(self, name: str) -> object:
            raise AssertionError(
                f"migration used op.{name}(), which this guard cannot replay. "
                "Write wfmem DDL as op.execute(...) with raw SQL, or teach the "
                "recorder to render it -- silently skipping it would make this "
                "test pass while schema.sql drifts."
            )

    for path in _MIGRATION_PATHS:
        assert path.exists(), f"{path.name} is missing from db/migrations/versions"
        spec = importlib.util.spec_from_file_location(f"wfmem_migration_{path.stem}", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.op = _Recorder()  # type: ignore[attr-defined]
        module.upgrade()

    return collected


def test_every_wfmem_migration_is_replayed_by_the_drift_guard() -> None:
    """A wfmem migration left out of `_MIGRATION_PATHS` is checked by nothing.

    CI applies db/schema.sql and stamps the alembic head, so the chain never
    runs there; this comparison is the only thing keeping the two files honest.
    A migration the guard does not replay is a column that can exist in one file
    and not the other, forever, with every test green.

    Scans the versions directory rather than trusting the tuple -- the failure
    mode is somebody adding a migration and not this line.
    """
    named = {path.name for path in _MIGRATION_PATHS}
    on_disk = {
        path.name
        for path in _VERSIONS_DIR.glob("*.py")
        if "wfmem" in path.name or "workflow_memory" in path.name
    }
    # 0111 is a data backfill of customers.preferences with no DDL, so it has
    # nothing for this guard to compare and is excluded by name rather than by
    # a rule that would also excuse a real one.
    on_disk.discard("20260820_0115_wfmem_capability_prefs.py")
    # 0119 is likewise data-only: it repairs 0118's silently-empty backfill and
    # touches no DDL. Its correctness is asserted by
    # `test_the_misc_backfill_works_under_the_role_that_actually_runs_it`, which
    # a schema comparison could never have caught.
    on_disk.discard("20260821_0119_wfmem_misc_backfill_rls.py")
    assert on_disk == named, (
        f"these wfmem migrations are not replayed by the drift guard: "
        f"{sorted(on_disk - named)}. Add them to _MIGRATION_PATHS."
    )


def _dsn_for(dbname: str) -> str:
    parsed = urlparse(os.environ["DATABASE_URL"])
    return urlunparse(parsed._replace(path=f"/{dbname}"))


def _normalized_lines(text: str) -> tuple[str, ...]:
    """A function body reduced to its non-blank lines, each stripped.

    Necessary, and the reason is worth stating so nobody "tightens" this back:
    the migration embeds its SQL inside an INDENTED Python string literal while
    schema.sql sits at column zero, so every line of the two files differs by
    leading whitespace. Everywhere else that is invisible, because Postgres
    reprints the catalog in a normal form -- constraints, indexes, policies and
    triggers all compare byte-identical. ``prosrc`` is the exception: it is
    stored as the raw text it was created with. Comparing it verbatim makes
    this section permanently red on two files that genuinely match. Whitespace
    is not semantics in plpgsql; a body that differs by anything else still
    fails here.
    """
    return tuple(line.strip() for line in text.splitlines() if line.strip())


async def _wfmem_fingerprint(conn: asyncpg.Connection) -> dict[str, list[tuple[Any, ...]]]:
    """Every workflow-memory object that could drift between the two files.

    Scoped by name PREFIX, not by the WFMEM_TABLES list, so a table, function
    or trigger added by a later migration is covered without anyone having to
    remember this file exists. Two sections exist specifically because the
    per-table view was blind to them:

    * ``functions`` -- the trigger DEFINITIONS were compared all along, but
      pg_get_triggerdef only names the function it calls. The function BODY
      was never read, so schema.sql and the migration could have defined
      wfmem_touch_updated_at differently and this guard would have passed.
    * ``triggers_by_function`` -- a trigger that a wfmem migration puts on a
      table it does not own (``customers`` is the live example we nearly
      shipped) sits outside every per-table query in here.
    """
    return {
        "columns": [
            tuple(r)
            for r in await conn.fetch(
                "SELECT table_name, ordinal_position, column_name, data_type, udt_name, "
                "is_nullable, column_default FROM information_schema.columns "
                "WHERE table_schema = 'public' AND table_name ~ $1 "
                "ORDER BY table_name, ordinal_position",
                _WFMEM_RELATION_RE,
            )
        ],
        "constraints": [
            tuple(r)
            for r in await conn.fetch(
                "SELECT c.relname, con.conname, pg_get_constraintdef(con.oid) "
                "FROM pg_constraint con JOIN pg_class c ON c.oid = con.conrelid "
                "WHERE c.relnamespace = 'public'::regnamespace AND c.relname ~ $1 "
                "ORDER BY 1, 2",
                _WFMEM_RELATION_RE,
            )
        ],
        "indexes": [
            tuple(r)
            for r in await conn.fetch(
                "SELECT tablename, indexname, indexdef FROM pg_indexes "
                "WHERE schemaname = 'public' AND tablename ~ $1 ORDER BY 1, 2",
                _WFMEM_RELATION_RE,
            )
        ],
        "policies": [
            tuple(r)
            for r in await conn.fetch(
                "SELECT tablename, policyname, permissive, cmd, qual, with_check "
                "FROM pg_policies WHERE schemaname = 'public' AND tablename ~ $1 "
                "ORDER BY 1, 2",
                _WFMEM_RELATION_RE,
            )
        ],
        "rls_flags": [
            tuple(r)
            for r in await conn.fetch(
                "SELECT relname, relrowsecurity, relforcerowsecurity FROM pg_class "
                "WHERE relnamespace = 'public'::regnamespace AND relkind = 'r' "
                "AND relname ~ $1 ORDER BY 1",
                _WFMEM_RELATION_RE,
            )
        ],
        # Per-table: every non-internal trigger sitting ON a wfmem table,
        # whatever function it calls.
        "triggers": [
            tuple(r)
            for r in await conn.fetch(
                "SELECT c.relname, t.tgname, pg_get_triggerdef(t.oid) FROM pg_trigger t "
                "JOIN pg_class c ON c.oid = t.tgrelid "
                "WHERE NOT t.tgisinternal AND c.relnamespace = 'public'::regnamespace "
                "AND c.relname ~ $1 ORDER BY 1, 2",
                _WFMEM_RELATION_RE,
            )
        ],
        # By-function: every non-internal trigger CALLING a wfmem function,
        # whatever table it sits on. Deliberately unrestricted by table and by
        # schema -- that is the whole point of this section.
        "triggers_by_function": [
            tuple(r)
            for r in await conn.fetch(
                "SELECT c.relnamespace::regnamespace::text, c.relname, t.tgname, p.proname, "
                "pg_get_triggerdef(t.oid) FROM pg_trigger t "
                "JOIN pg_class c ON c.oid = t.tgrelid "
                "JOIN pg_proc p ON p.oid = t.tgfoid "
                "WHERE NOT t.tgisinternal AND p.proname ~ $1 ORDER BY 1, 2, 3",
                _WFMEM_FUNCTION_RE,
            )
        ],
        # The function bodies themselves. pg_get_functiondef carries the
        # language, return type and volatility alongside the source, so a
        # trigger function that quietly starts writing a different value shows
        # up here as a diff.
        "functions": [
            (r["proname"], r["identity_arguments"], _normalized_lines(r["definition"]))
            for r in await conn.fetch(
                "SELECT p.proname, pg_get_function_identity_arguments(p.oid) "
                "AS identity_arguments, pg_get_functiondef(p.oid) AS definition "
                "FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace "
                "WHERE n.nspname = 'public' AND p.proname ~ $1 ORDER BY 1, 2",
                _WFMEM_FUNCTION_RE,
            )
        ],
    }


@pytest.mark.asyncio
async def test_schema_sql_has_not_drifted_from_the_migration(live_db):
    """CI applies db/schema.sql and stamps alembic head -- it NEVER runs the
    migration chain. The two files are identical today only by care, and
    nothing else in the repo keeps them that way, so: build one scratch DB by
    replaying the migration, another from schema.sql, and diff the result.

    The diff covers tables, columns, constraints, indexes, policies, RLS flags,
    triggers ON wfmem tables, triggers CALLING wfmem functions wherever they
    sit, and the function bodies -- matched by name prefix so that objects
    added later are covered by default rather than on remembering to.
    """
    suffix = f"{os.getpid()}_{uuid4().hex[:8]}"
    mig_db = f"wfmem_drift_mig_{suffix}"
    sch_db = f"wfmem_drift_schema_{suffix}"

    admin = await asyncpg.connect(_dsn_for("postgres"))
    try:
        for name in (mig_db, sch_db):
            await admin.execute(f'DROP DATABASE IF EXISTS "{name}"')
            await admin.execute(f'CREATE DATABASE "{name}"')
    finally:
        await admin.close()

    try:
        migration_conn = await asyncpg.connect(_dsn_for(mig_db))
        try:
            await migration_conn.execute(_VECTOR_EXTENSION)
            await migration_conn.execute(_CUSTOMERS_STUB)
            for statement in _migration_upgrade_statements():
                await migration_conn.execute(statement)
            migration_fingerprint = await _wfmem_fingerprint(migration_conn)
        finally:
            await migration_conn.close()

        schema_conn = await asyncpg.connect(_dsn_for(sch_db))
        try:
            await schema_conn.execute(_VECTOR_EXTENSION)
            await schema_conn.execute(_NEON_AUTH_SHIM)
            await schema_conn.execute(_SCHEMA_PATH.read_text())
            schema_fingerprint = await _wfmem_fingerprint(schema_conn)
        finally:
            await schema_conn.close()
    finally:
        admin = await asyncpg.connect(_dsn_for("postgres"))
        try:
            for name in (mig_db, sch_db):
                await admin.execute(f'DROP DATABASE IF EXISTS "{name}"')
        finally:
            await admin.close()

    # Non-vacuity first. Every section is prefix-matched, so a typo in one of
    # the regexes would return nothing on BOTH sides and compare equal -- a
    # green guard that inspects nothing, which is the failure this whole test
    # was written to stop happening elsewhere.
    for section, rows in migration_fingerprint.items():
        assert rows, f"the {section} fingerprint is empty -- the guard is inspecting nothing"
    assert any(row[0] == "wfmem_touch_updated_at" for row in migration_fingerprint["functions"]), (
        "wfmem_touch_updated_at is missing from the functions fingerprint; "
        f"got {[row[0] for row in migration_fingerprint['functions']]}"
    )

    assert sorted(migration_fingerprint) == sorted(schema_fingerprint)
    for section in sorted(migration_fingerprint):
        assert migration_fingerprint[section] == schema_fingerprint[section], (
            f"db/schema.sql and migration 0114 disagree on {section}. CI applies "
            "schema.sql and stamps alembic head, so schema.sql is what the tests "
            "actually run against -- the two must be updated together."
        )


# ---------------------------------------------------------------------------
# Data migrations vs. FORCE ROW LEVEL SECURITY
# ---------------------------------------------------------------------------
#
# 0118 SHIPPED, REPORTED SUCCESS, AND INSERTED NOTHING. Its DDL landed and both
# of its data statements were filtered to zero rows, so production sat at head
# 0118 with `situations.classifiable` present and not one `misc` row anywhere.
#
# Migrations run as `app`: the table OWNER, not a superuser, no BYPASSRLS -- and
# FORCE is exactly the flag that subjects an owner to the policy. With no tenant
# GUC bound, `customer_id = current_setting('app.current_customer_id', true)`
# compares against NULL, so `SELECT DISTINCT customer_id FROM situations` returns
# nothing and the INSERT that consumes it writes nothing.
#
# Every test in this file already knew the shape of that trap and said so three
# times over -- and still could not catch it, because they assert on ISOLATION
# while a data migration fails on VISIBILITY, and because the drift guard
# compares schema rather than rows. So the guard has to be the migration's own
# logic, executed under a role that cannot bypass the policy.


_MISC_SLUG = "misc"


async def _seed_one_situation(conn: Any, customer_id: str, slug: str) -> None:
    await conn.execute(
        """
        INSERT INTO situations (customer_id, slug, label, description)
        VALUES ($1, $2, 'L', 'a description long enough to be plausible')
        ON CONFLICT (customer_id, slug) DO NOTHING
        """,
        customer_id,
        slug,
    )


async def test_reading_the_tenant_list_from_situations_sees_nothing_under_rls(
    two_tenants: tuple[str, str],
) -> None:
    """0118's actual bug, pinned so nobody writes it again.

    This is the statement the broken migration was built on. Under the role that
    really runs migrations it returns ZERO rows -- which is why an INSERT reading
    from it wrote nothing and raised nothing.
    """
    tenant_a, _ = two_tenants
    async with with_tenant(tenant_a) as conn:
        await _seed_one_situation(conn, tenant_a, "launch-run")

    async with raw_conn() as conn, conn.transaction():
        await conn.execute(f"SET LOCAL ROLE {RLS_ROLE}")
        # No tenant GUC bound -- exactly a migration's connection.
        visible = await conn.fetch("SELECT DISTINCT customer_id FROM situations")
        from_customers = await conn.fetch(
            "SELECT customer_id FROM customers WHERE customer_id = $1", tenant_a
        )

    assert visible == [], (
        "a migration reading `situations` with no tenant GUC must see nothing -- "
        "if this ever returns rows the policy has been weakened"
    )
    assert len(from_customers) == 1, (
        "`customers` must stay readable without a GUC; it is the only way a data "
        "migration can discover which tenants to bind to"
    )


async def test_the_misc_backfill_works_under_the_role_that_actually_runs_it(
    two_tenants: tuple[str, str],
) -> None:
    """0119's logic, executed as a non-superuser. Fails against 0118's version.

    The whole repair is "bind the tenant GUC per tenant", and this is the only
    assertion in the repo that would notice if somebody removed it.
    """
    tenant_a, tenant_b = two_tenants
    async with with_tenant(tenant_a) as conn:
        await _seed_one_situation(conn, tenant_a, "launch-run")
    # tenant_b deliberately gets NO vocabulary: a tenant that never enabled the
    # capability must not be given a lone bucket row.

    async with raw_conn() as conn, conn.transaction():
        await conn.execute(f"SET LOCAL ROLE {RLS_ROLE}")
        # `customers` is not row-secured, so the tenant list is readable
        # with nothing bound. Everything after this needs the GUC.
        candidates = [
            r["customer_id"]
            for r in await conn.fetch(
                "SELECT customer_id FROM customers WHERE customer_id = ANY($1::text[]) ORDER BY 1",
                [tenant_a, tenant_b],
            )
        ]
        tenants = []
        for customer_id in candidates:
            await conn.execute(
                "SELECT set_config('app.current_customer_id', $1, true)", customer_id
            )
            # ONLY NOW is `situations` visible. Asking before this -- via an
            # EXISTS in the tenant query, which is the obvious way to write
            # it -- is false for everyone and skips the whole loop. That is
            # 0118's bug, and the first draft of 0119's repair had it too.
            if not await conn.fetchval(
                "SELECT EXISTS (SELECT 1 FROM situations WHERE customer_id = $1 AND slug <> $2)",
                customer_id,
                _MISC_SLUG,
            ):
                continue
            tenants.append(customer_id)
            await conn.execute(
                """
                    INSERT INTO situations
                        (customer_id, slug, label, description, classifiable)
                    VALUES ($1, $2, 'Anything else', 'holding bucket', false)
                    ON CONFLICT (customer_id, slug) DO UPDATE SET classifiable = false
                    """,
                customer_id,
                _MISC_SLUG,
            )

    assert tenants == [tenant_a], "only a tenant with a vocabulary gets a bucket"

    async with with_tenant(tenant_a) as conn:
        row = await conn.fetchrow(
            "SELECT classifiable FROM situations WHERE customer_id = $1 AND slug = $2",
            tenant_a,
            _MISC_SLUG,
        )
    assert row is not None, (
        "the bucket was not written -- this is 0118's failure exactly: the "
        "statements ran, nothing errored, and no row exists"
    )
    assert row["classifiable"] is False

    async with with_tenant(tenant_b) as conn:
        assert (
            await conn.fetchval("SELECT count(*) FROM situations WHERE customer_id = $1", tenant_b)
            == 0
        )
