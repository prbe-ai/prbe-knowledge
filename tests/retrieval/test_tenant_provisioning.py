"""kb_provision_tenant() and its Python callers (plan T4, migration 0144).

Pins: Python and SQL agree on partition names; the function is idempotent,
copies every parent policy onto the leaf and refuses a stale same-named table;
concurrent creates do not deadlock when provisioning runs after the customer
commits; bound-based lookup finds pre-0144 (sha1-named) partitions; the
guardian backstop finds tenants with no partition.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from pathlib import Path

import pytest

import engine.shared.db as db_module
from engine.shared.partitions import (
    PARTITION_PREFIXES,
    ensure_tenant_partition,
    ensure_tenant_partitions,
    find_tenants_missing_partitions,
    partition_name_for,
    partition_of,
    partitioned_parents,
)

REPO = Path(__file__).resolve().parents[2]


async def _add_customer(conn, customer_id: str) -> None:
    # The research-os kb_mirror shape: insert-or-nothing, committed on its own.
    await conn.execute(
        "INSERT INTO customers (customer_id, display_name, api_key_hash) "
        "VALUES ($1, $1, $1) ON CONFLICT DO NOTHING",
        customer_id,
    )


def test_schema_sql_carries_0144_bodies_verbatim():
    migration = (REPO / "db/migrations/versions/20260929_0144_kb_provision_tenant.py").read_text()
    schema = (REPO / "db/schema.sql").read_text()
    for name in ("PARTITION_NAME_SQL", "PROVISION_SQL"):
        body = re.search(rf'{name} = r"""(.*?)"""', migration, re.S).group(1)
        assert body in schema, f"{name} drifted between 0144 and db/schema.sql"


@pytest.mark.integration
@pytest.mark.parametrize("parent", sorted(PARTITION_PREFIXES))
@pytest.mark.parametrize(
    "customer_id", ["probe", "new-workspace", "a.b_c-D9", "x" * 63, "Bucket-Robotics"]
)
async def test_python_and_sql_name_partitions_identically(live_db, parent, customer_id):
    async with db_module.raw_conn() as conn:
        sql_name = await conn.fetchval("SELECT kb_partition_name($1, $2)", parent, customer_id)
    assert sql_name == partition_name_for(customer_id, parent)
    assert len(sql_name) <= 63
    assert sql_name.endswith(hashlib.sha256(customer_id.encode()).hexdigest()[:8])


@pytest.mark.integration
async def test_provision_creates_attached_leaf_with_every_parent_policy(live_db):
    async with db_module.raw_conn() as conn:
        await _add_customer(conn, "prov-a")
        assert await ensure_tenant_partition(conn, "prov-a") is True
        # Idempotent: nothing left to create.
        assert await ensure_tenant_partition(conn, "prov-a") is False
        for parent in await partitioned_parents(conn):
            leaf = await partition_of(conn, "prov-a", parent=parent)
            assert leaf == partition_name_for("prov-a", parent)
            rls = await conn.fetchrow(
                "SELECT relrowsecurity, relforcerowsecurity FROM pg_class WHERE relname = $1",
                leaf,
            )
            assert rls["relrowsecurity"] and rls["relforcerowsecurity"]
            parent_pols = await conn.fetch(
                "SELECT polname, polcmd, pg_get_expr(polqual, polrelid) q, "
                "pg_get_expr(polwithcheck, polrelid) w FROM pg_policy "
                "WHERE polrelid = to_regclass($1) ORDER BY polname",
                parent,
            )
            leaf_pols = await conn.fetch(
                "SELECT polname, polcmd, pg_get_expr(polqual, polrelid) q, "
                "pg_get_expr(polwithcheck, polrelid) w FROM pg_policy "
                "WHERE polrelid = to_regclass($1) ORDER BY polname",
                leaf,
            )
            assert [tuple(p) for p in leaf_pols] == [tuple(p) for p in parent_pols]
            assert parent_pols, f"{parent} has no policy to copy"


@pytest.mark.integration
async def test_refuses_inside_the_customer_insert_transaction(live_db):
    async with db_module.raw_conn() as conn, conn.transaction():
        await _add_customer(conn, "prov-in-txn")
        with pytest.raises(RuntimeError, match="own transaction"):
            await ensure_tenant_partition(conn, "prov-in-txn")


@pytest.mark.integration
async def test_concurrent_creates_then_provision_do_not_deadlock(live_db):
    tenants = [f"prov-conc-{i}" for i in range(6)]

    async def create(t: str) -> None:
        async with db_module.raw_conn() as conn:
            await _add_customer(conn, t)  # committed on its own
        await ensure_tenant_partitions(t)

    await asyncio.wait_for(asyncio.gather(*(create(t) for t in tenants)), timeout=60)
    async with db_module.raw_conn() as conn:
        for t in tenants:
            assert await partition_of(conn, t) is not None


@pytest.mark.integration
async def test_refuses_a_stale_same_named_table(live_db):
    name = partition_name_for("prov-stale")
    async with db_module.raw_conn() as conn:
        await _add_customer(conn, "prov-stale")
        await conn.execute(f'CREATE TABLE "{name}" (LIKE chunks INCLUDING ALL)')
        try:
            with pytest.raises(Exception, match="exists but is not attached"):
                await ensure_tenant_partition(conn, "prov-stale")
        finally:
            await conn.execute(f'DROP TABLE IF EXISTS "{name}"')


@pytest.mark.integration
async def test_bound_lookup_finds_a_pre_0144_sha1_named_partition(live_db):
    legacy = "chunks_p_prov_legacy_" + hashlib.sha1(b"prov-legacy").hexdigest()[:8]
    async with db_module.raw_conn() as conn:
        await _add_customer(conn, "prov-legacy")
        await conn.execute(f'CREATE TABLE "{legacy}" (LIKE chunks INCLUDING ALL)')
        await conn.execute(
            f"""ALTER TABLE chunks ATTACH PARTITION "{legacy}" FOR VALUES IN ('prov-legacy')"""
        )
        assert await partition_of(conn, "prov-legacy") == legacy
        # Already provisioned by bound: no second chunks partition.
        await ensure_tenant_partition(conn, "prov-legacy")
        count = await conn.fetchval(
            """
            SELECT count(*) FROM pg_inherits h JOIN pg_class c ON c.oid = h.inhrelid
            WHERE h.inhparent = 'chunks'::regclass
              AND pg_get_expr(c.relpartbound, c.oid) = 'FOR VALUES IN (''prov-legacy'')'
            """
        )
        assert count == 1


@pytest.mark.integration
async def test_backstop_finds_and_fixes_a_tenant_without_partitions(live_db):
    async with db_module.raw_conn() as conn:
        await _add_customer(conn, "prov-missed")
        assert "prov-missed" in await find_tenants_missing_partitions(conn)
    assert await ensure_tenant_partitions("prov-missed") is True
    async with db_module.raw_conn() as conn:
        assert "prov-missed" not in await find_tenants_missing_partitions(conn)


@pytest.mark.integration
async def test_unsafe_customer_id_is_refused_by_the_function(live_db):
    async with db_module.raw_conn() as conn:
        with pytest.raises(Exception, match="refusing unsafe customer_id"):
            await conn.fetchval("SELECT kb_provision_tenant($1)", "bad id; drop")


@pytest.mark.integration
async def test_refuses_a_customer_that_does_not_exist(live_db):
    async with db_module.raw_conn() as conn:
        with pytest.raises(Exception, match="no customer"):
            await conn.fetchval("SELECT kb_provision_tenant($1)", "prov-nobody")
        assert await partition_of(conn, "prov-nobody") is None
