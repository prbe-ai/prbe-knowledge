"""Tenant isolation for the workflow-memory procedure store.

Mirrors tests/test_multitenant_isolation.py -- NOT
test_rls_cross_tenant_denial.py, whose isolation tests skip under the dev
superuser role and have therefore never executed.

Both halves of the policy matter and fail differently: USING hides another
tenant's rows on read, WITH CHECK refuses a write that would land under
another tenant. A table with USING only looks correct until a buggy writer
silently files a row under the wrong customer.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import asyncpg
import pytest
import pytest_asyncio

from shared.db import raw_conn, with_tenant

TENANT_A = "cust-wfmem-a"
TENANT_B = "cust-wfmem-b"


#: The dev/CI role is a SUPERUSER and superusers bypass RLS outright, so a test
#: run as `prbe` does not pass vacuously -- it FAILS, because A really does see
#: B's rows. Every RLS assertion below must run as a non-privileged role.
#: Copied from tests/test_multitenant_isolation.py:18-60, the one file in this
#: repo whose isolation tests actually execute.
RLS_ROLE = "prbe_rls_test"


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


async def _seed_clause(customer_id: str, body: str) -> None:
    async with with_tenant(customer_id) as conn:
        await conn.execute(f"SET LOCAL ROLE {RLS_ROLE}")
        await conn.execute(
            """
            INSERT INTO clauses (customer_id, kind, body, status, author_ref)
            VALUES ($1, 'step', $2, 'declared', 'user:seed')
            """,
            customer_id,
            body,
        )


@pytest.mark.asyncio
async def test_clauses_read_is_tenant_scoped(two_tenants):
    """USING: A's unfiltered read must not see B's clause."""
    a, b = two_tenants
    await _seed_clause(a, "rule belonging to A")
    await _seed_clause(b, "rule belonging to B")

    async with with_tenant(a) as conn:
        await conn.execute(f"SET LOCAL ROLE {RLS_ROLE}")
        rows = await conn.fetch("SELECT body, customer_id FROM clauses")

    bodies = {r["body"] for r in rows}
    assert "rule belonging to A" in bodies
    assert "rule belonging to B" not in bodies
    assert {r["customer_id"] for r in rows} == {a}


@pytest.mark.asyncio
async def test_clauses_cross_tenant_insert_is_refused(two_tenants):
    """WITH CHECK: under A's GUC, a write claiming B must be rejected."""
    a, b = two_tenants
    with pytest.raises(asyncpg.exceptions.InsufficientPrivilegeError):
        async with with_tenant(a) as conn:
            await conn.execute(f"SET LOCAL ROLE {RLS_ROLE}")
            await conn.execute(
                """
                INSERT INTO clauses (customer_id, kind, body, status, author_ref)
                VALUES ($1, 'step', 'smuggled into B', 'declared', 'user:attacker')
                """,
                b,
            )


WFMEM_TABLES = (
    "situations",
    "clauses",
    "clause_situation_edges",
    "clause_evidence",
    "serve_ledger",
)


@pytest.mark.asyncio
@pytest.mark.parametrize("table", WFMEM_TABLES)
async def test_every_table_has_forced_rls_and_both_policy_halves(live_db, table):
    """ENABLE alone is not enough; FORCE without a policy is deny-all; and
    USING without WITH CHECK lets a buggy writer file under another tenant."""
    async with raw_conn() as conn:
        flags = await conn.fetchrow(
            "SELECT relrowsecurity, relforcerowsecurity FROM pg_class WHERE relname = $1",
            table,
        )
        policy = await conn.fetchrow(
            "SELECT qual, with_check FROM pg_policies "
            "WHERE tablename = $1 AND policyname = 'tenant_isolation'",
            table,
        )

    assert flags is not None, f"{table} does not exist"
    assert flags["relrowsecurity"], f"{table} is missing ENABLE ROW LEVEL SECURITY"
    assert flags["relforcerowsecurity"], f"{table} is missing FORCE ROW LEVEL SECURITY"
    assert policy is not None, f"{table} has no tenant_isolation policy"
    assert policy["qual"] is not None, f"{table} policy has no USING clause"
    assert policy["with_check"] is not None, f"{table} policy has no WITH CHECK clause"
    # Not merely "a clause exists" -- USING (true) would satisfy that. Assert the
    # policy actually keys on the tenant GUC.
    assert "app.current_customer_id" in policy["qual"], (
        f"{table} USING clause does not reference the tenant GUC: {policy['qual']}"
    )
    assert "app.current_customer_id" in policy["with_check"], (
        f"{table} WITH CHECK does not reference the tenant GUC: {policy['with_check']}"
    )


@pytest.mark.asyncio
async def test_cross_tenant_update_and_delete_are_no_ops(two_tenants):
    """The structural check above proves the clauses EXIST. This proves they
    BITE for UPDATE and DELETE, which the INSERT/SELECT tests never exercise --
    a cross-tenant UPDATE does not raise, it silently matches zero rows, and
    that is the failure mode a reviewer is least likely to notice.
    """
    a, b = two_tenants
    await _seed_clause(b, "B's untouchable rule")

    async with with_tenant(a) as conn:
        await conn.execute(f"SET LOCAL ROLE {RLS_ROLE}")
        updated = await conn.execute(
            "UPDATE clauses SET body = 'hijacked' WHERE body = $1", "B's untouchable rule"
        )
        deleted = await conn.execute("DELETE FROM clauses WHERE body = $1", "B's untouchable rule")

    assert updated == "UPDATE 0", f"A's UPDATE reached B's row: {updated}"
    assert deleted == "DELETE 0", f"A's DELETE reached B's row: {deleted}"

    async with with_tenant(b) as conn:
        await conn.execute(f"SET LOCAL ROLE {RLS_ROLE}")
        rows = await conn.fetch("SELECT body FROM clauses WHERE body = $1", "B's untouchable rule")
    assert len(rows) == 1, "B's row must survive A's attempts untouched"


#: Column names that would mean a quote had been copied into the row rather
#: than pointed at. Evidence resolves at view time through the viewer's ACL;
#: a baked-in quote escapes that check and, per the design's red team, makes
#: the product un-sellable. Absence is the enforcement.
_FORBIDDEN_EVIDENCE_COLUMNS = {
    "quote",
    "text",
    "body",
    "content",
    "excerpt",
    "snippet",
    "raw",
}


@pytest.mark.asyncio
async def test_clause_evidence_stores_no_quote_text(live_db):
    async with raw_conn() as conn:
        rows = await conn.fetch(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'clause_evidence'"
        )
    columns = {r["column_name"] for r in rows}
    offending = columns & _FORBIDDEN_EVIDENCE_COLUMNS
    assert not offending, (
        f"clause_evidence must store references only, found {sorted(offending)}. "
        "Quotes resolve at view time through the viewer's ACL."
    )
    assert "source_ref" in columns, "clause_evidence must keep its pointer column"
