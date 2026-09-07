"""Companion tables: tenant isolation under a non-superuser role.

The dev/CI role is a SUPERUSER and superusers bypass RLS outright, so a test
run as `prbe` does not pass vacuously -- it FAILS, because A really does see
B's rows. Every RLS assertion here runs as `prbe_rls_test` (NOSUPERUSER
NOBYPASSRLS), the fixture shape copied from tests/test_workflow_memory_isolation.py.

Append-only tables (`companion_mailbox`, `companion_deliveries`) carry
SELECT + INSERT policies only. Under FORCE RLS the ABSENCE of an UPDATE/DELETE
policy is the deny -- and it is a QUIET deny: the statement succeeds and touches
zero rows, it does not raise. The assertions below check the row count and the
row's persistence, not an exception, so nobody "fixes" the policy to make an
error appear. `companion_claims` is the one mutable table and gets all four.

Run with the isolated database (these fixtures TRUNCATE):

    PRBE_TEST_DATABASE_URL=postgresql://prbe:prbe@localhost:55442/prbe_knowledge \
        .venv/bin/pytest tests/test_companion_isolation.py -q
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

import asyncpg
import pytest
import pytest_asyncio

from engine.shared.db import raw_conn, with_tenant

TENANT_A = "cust-companion-a"
TENANT_B = "cust-companion-b"
RLS_ROLE = "prbe_rls_test"
TABLES = ("companion_mailbox", "companion_deliveries", "companion_claims")


@pytest_asyncio.fixture
async def two_tenants(live_db) -> AsyncIterator[tuple[str, str]]:
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO customers (customer_id, display_name, api_key_hash)
            VALUES ($1, 'companion-a', 'h-companion-a'), ($2, 'companion-b', 'h-companion-b')
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
        # companion_deliveries.id is BIGSERIAL -- without this an insert under
        # the role fails on the sequence, not on the policy, and the test lies.
        await conn.execute(f"GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO {RLS_ROLE}")
    yield TENANT_A, TENANT_B
    async with raw_conn() as conn:
        await conn.execute(
            "DELETE FROM customers WHERE customer_id = ANY($1::text[])",
            [TENANT_A, TENANT_B],
        )


async def _seed_card(conn: asyncpg.Connection, customer_id: str, session_id: str | None) -> Any:
    return await conn.fetchval(
        """
        INSERT INTO companion_mailbox
            (customer_id, recipient, session_id, class, body, dedupe_key, source,
             trial_id, expires_at)
        VALUES ($1, 'user:seed', $2, 'seam', '[probe companion] hello', $3, 'driver',
                $4, now() + interval '10 minutes')
        RETURNING id
        """,
        customer_id,
        session_id,
        f"k-{uuid4()}",
        uuid4(),
    )


@pytest.mark.asyncio
async def test_tables_exist(live_db) -> None:
    async with raw_conn() as conn:
        for table in TABLES:
            assert await conn.fetchval("SELECT to_regclass($1) IS NOT NULL", table), table


@pytest.mark.asyncio
async def test_cross_tenant_insert_rejected(two_tenants) -> None:
    a, b = two_tenants
    async with with_tenant(a) as conn:
        await conn.execute(f"SET LOCAL ROLE {RLS_ROLE}")
        with pytest.raises(asyncpg.InsufficientPrivilegeError):
            await _seed_card(conn, b, "sess-1")


@pytest.mark.asyncio
async def test_reads_are_tenant_scoped(two_tenants) -> None:
    a, b = two_tenants
    async with with_tenant(a) as conn:
        await _seed_card(conn, a, "sess-a")
    async with with_tenant(b) as conn:
        await _seed_card(conn, b, "sess-b")
    async with with_tenant(a) as conn:
        await conn.execute(f"SET LOCAL ROLE {RLS_ROLE}")
        rows = await conn.fetch("SELECT customer_id FROM companion_mailbox")
        assert {r["customer_id"] for r in rows} == {a}


@pytest.mark.asyncio
async def test_append_only_tables_quietly_deny_update_and_delete(two_tenants) -> None:
    a, _ = two_tenants
    async with with_tenant(a) as conn:
        mid = await _seed_card(conn, a, "sess-a")
    async with with_tenant(a) as conn:
        await conn.execute(f"SET LOCAL ROLE {RLS_ROLE}")
        assert (
            await conn.execute("UPDATE companion_mailbox SET body = 'x' WHERE id = $1", mid)
            == "UPDATE 0"
        )
        assert await conn.execute("DELETE FROM companion_mailbox WHERE id = $1", mid) == "DELETE 0"
        body = await conn.fetchval("SELECT body FROM companion_mailbox WHERE id = $1", mid)
        assert body == "[probe companion] hello"


@pytest.mark.asyncio
async def test_cross_tenant_delivery_edge_unrepresentable(two_tenants) -> None:
    """The composite FK: tenant B cannot ack tenant A's card even as superuser."""
    a, b = two_tenants
    async with with_tenant(a) as conn:
        mid = await _seed_card(conn, a, "sess-a")
    async with with_tenant(b) as conn:
        with pytest.raises(asyncpg.ForeignKeyViolationError):
            await conn.execute(
                """
                INSERT INTO companion_deliveries
                    (customer_id, mailbox_id, attempt_id, seam, outcome, receiving_instance)
                VALUES ($1, $2, $3, 'stop', 'emitted', 'test')
                """,
                b,
                mid,
                uuid4(),
            )


@pytest.mark.asyncio
async def test_claims_are_mutable_within_tenant_only(two_tenants) -> None:
    a, b = two_tenants
    async with with_tenant(a) as conn:
        mid = await _seed_card(conn, a, None)
    async with with_tenant(a) as conn:
        await conn.execute(f"SET LOCAL ROLE {RLS_ROLE}")
        await conn.execute(
            "INSERT INTO companion_claims VALUES ($1, $2, 'ingest:t', now() + interval '1 minute')",
            a,
            mid,
        )
        assert (
            await conn.execute(
                "UPDATE companion_claims SET lease_until = now() WHERE mailbox_id = $1", mid
            )
            == "UPDATE 1"
        )
    async with with_tenant(b) as conn:
        await conn.execute(f"SET LOCAL ROLE {RLS_ROLE}")
        assert await conn.fetchval("SELECT count(*) FROM companion_claims") == 0
