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
import time
from pathlib import Path

import asyncpg
import pytest

import engine.shared.db as db_module
from engine.shared.partitions import (
    PARTITION_DDL_LOCK_SQL,
    PARTITION_PREFIXES,
    default_partition_name,
    ensure_tenant_partition,
    ensure_tenant_partitions,
    find_tenants_missing_partitions,
    partition_name_for,
    partition_of,
    partitioned_parents,
    split_default,
)
from engine.shared.provisioning import create_customer

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


async def _strand_in_default(conn, tenant: str, n: int = 3) -> None:
    """A document and `n` chunks for a tenant that has NO partition: DEFAULT."""
    await conn.execute(
        """
        INSERT INTO documents (customer_id, doc_id, version, source_system,
                               source_id, source_url, doc_type, content_hash,
                               created_at, updated_at, valid_from, acl,
                               title, body_preview)
        VALUES ($2, $1, 1, 'custom_ingest', $1, 'https://x', 'custom.note',
                'dh', NOW(), NOW(), NOW(), '{}'::jsonb, 'T', 'p')
        """,
        f"{tenant}:d1",
        tenant,
    )
    await conn.execute(
        """
        INSERT INTO chunks (
            chunk_id, doc_id, customer_id, chunk_index, content,
            content_hash, token_count, chunker_version, first_seen_version,
            last_seen_version, kind, visibility
        )
        SELECT $1 || ':c_' || g, $2, $1, g, 'c' || g, 'h' || g, 3, 'v1',
               1, 1, 'content', 'approved'
        FROM generate_series(1, $3::int) g
        """,
        tenant,
        f"{tenant}:d1",
        n,
    )


@pytest.mark.integration
async def test_refuses_inside_a_transaction_that_wrote_customers_in_sql(live_db):
    # research-os calls the function straight from SQL, so the guard against
    # the documented deadlock has to live in the function, not only in Python.
    async with db_module.raw_conn() as conn:
        with pytest.raises(asyncpg.exceptions.ActiveSQLTransactionError, match="commit it first"):
            async with conn.transaction():
                await _add_customer(conn, "prov-sql-in-txn")
                await conn.fetchval("SELECT kb_provision_tenant($1)", "prov-sql-in-txn")


@pytest.mark.integration
async def test_advisory_lock_wait_is_bounded_by_the_lock_timeout(live_db):
    # The 3 s lock_timeout is a SET clause of the function, so it bounds the
    # wait for the partition-DDL lock too (it used to be set after it: an
    # attended split_default holding the lock hung every provision).
    async with db_module.raw_conn() as holder, db_module.raw_conn() as conn:
        await _add_customer(conn, "prov-bounded")
        await holder.execute(f"SELECT pg_advisory_lock({PARTITION_DDL_LOCK_SQL})")
        try:
            started = time.monotonic()
            with pytest.raises(asyncpg.exceptions.LockNotAvailableError):
                await ensure_tenant_partition(conn, "prov-bounded")
            assert time.monotonic() - started < 10
        finally:
            await holder.execute(f"SELECT pg_advisory_unlock({PARTITION_DDL_LOCK_SQL})")
        assert await ensure_tenant_partition(conn, "prov-bounded") is True


@pytest.mark.integration
async def test_function_settings_do_not_leak_into_the_callers_transaction(live_db):
    async with db_module.raw_conn() as conn:
        await _add_customer(conn, "prov-leak")
        async with conn.transaction():
            await conn.execute("SET LOCAL lock_timeout = '17s'")
            await conn.execute("SELECT set_config('app.current_customer_id', 'someone', true)")
            assert await conn.fetchval("SELECT kb_provision_tenant($1)", "prov-leak") >= 1
            assert await conn.fetchval("SHOW lock_timeout") == "17s"
            assert await conn.fetchval("SELECT current_setting('app.current_customer_id')") == "someone"


@pytest.mark.integration
async def test_a_tenant_already_in_default_is_refused_then_split(live_db):
    tenant = "prov-in-default"
    async with db_module.raw_conn() as conn:
        await _add_customer(conn, tenant)
        await _strand_in_default(conn, tenant)
        default_name = await default_partition_name(conn)
        with pytest.raises(
            asyncpg.exceptions.ObjectNotInPrerequisiteStateError, match="split_default"
        ):
            await ensure_tenant_partition(conn, tenant)
        assert await partition_of(conn, tenant) is None
        # The backstop keeps listing it; the attended fix clears it.
        assert tenant in await find_tenants_missing_partitions(conn)
        assert await split_default(conn, tenant) == 3
        assert await partition_of(conn, tenant) == partition_name_for(tenant)
        assert await conn.fetchval(
            f'SELECT count(*) FROM ONLY "{default_name}" WHERE customer_id = $1', tenant
        ) == 0


@pytest.mark.integration
async def test_default_check_sees_rows_under_rls_as_a_non_superuser(live_db):
    # Prod's DEFAULT is under FORCE RLS and the function runs as `app`, not a
    # superuser: without its function-scoped tenant GUC the check would see no
    # rows and go on to lock DEFAULT and fail the ATTACH.
    tenant = "prov-rls-default"
    async with db_module.raw_conn() as conn:
        await _add_customer(conn, tenant)
        default_name = await default_partition_name(conn)
        tx = conn.transaction()
        await tx.start()
        try:
            await _strand_in_default(conn, tenant)
            await conn.execute(f'ALTER TABLE "{default_name}" ENABLE ROW LEVEL SECURITY')
            await conn.execute(
                f'CREATE POLICY prov_test_iso ON "{default_name}" USING '
                "(customer_id = current_setting('app.current_customer_id', true))"
            )
            await conn.execute("CREATE ROLE prov_test_app NOLOGIN")
            await conn.execute(f'GRANT SELECT ON customers, "{default_name}" TO prov_test_app')
            await conn.execute("GRANT EXECUTE ON FUNCTION kb_provision_tenant(text) TO prov_test_app")
            await conn.execute("SET LOCAL ROLE prov_test_app")
            with pytest.raises(
                asyncpg.exceptions.ObjectNotInPrerequisiteStateError, match="already has rows"
            ):
                await conn.fetchval("SELECT kb_provision_tenant($1)", tenant)
        finally:
            await tx.rollback()


@pytest.mark.integration
async def test_leaf_takes_the_parents_owner_and_grants(live_db):
    # A superuser calling this by hand (probe-peek runs as one) must not leave
    # a leaf the parent's owner cannot ALTER: later migrations recurse into
    # every leaf as that owner. Rolled back: nothing here outlives the test.
    async with db_module.raw_conn() as conn:
        await _add_customer(conn, "prov-owner")
        tx = conn.transaction()
        await tx.start()
        try:
            await conn.execute("CREATE ROLE prov_test_owner NOLOGIN")
            await conn.execute("CREATE ROLE prov_test_reader NOLOGIN")
            await conn.execute("CREATE ROLE prov_test_extra NOLOGIN")
            await conn.execute("ALTER TABLE chunks OWNER TO prov_test_owner")
            await conn.execute("GRANT SELECT ON chunks TO prov_test_reader")
            # The creating role's default privileges must not leak onto the leaf.
            await conn.execute(
                "ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO prov_test_extra"
            )
            assert await conn.fetchval("SELECT kb_provision_tenant($1)", "prov-owner") >= 1
            leaf = partition_name_for("prov-owner")
            assert await conn.fetchval(
                "SELECT relowner::regrole::text FROM pg_class WHERE relname = $1", leaf
            ) == "prov_test_owner"
            assert await conn.fetchval(
                "SELECT has_table_privilege('prov_test_reader', $1::regclass, 'SELECT')", leaf
            )
            acl = (
                "SELECT array_agg(g.grantee::regrole::text || ':' || g.privilege_type "
                "ORDER BY 1) FROM pg_class c, aclexplode(c.relacl) g "
                "WHERE c.oid = $1::regclass AND g.grantee <> c.relowner"
            )
            assert await conn.fetchval(acl, leaf) == await conn.fetchval(acl, "chunks")
            assert not await conn.fetchval(
                "SELECT has_table_privilege('prov_test_extra', $1::regclass, 'SELECT')", leaf
            )
        finally:
            await tx.rollback()


@pytest.mark.integration
async def test_a_parent_without_a_prefix_is_skipped_not_fatal(live_db):
    # A conversion's scratch table is LIST(customer_id) too; it used to make
    # every provision raise "no partition prefix".
    async with db_module.raw_conn() as conn:
        await conn.execute(
            "CREATE TABLE chunks_part (customer_id text NOT NULL, n int) "
            "PARTITION BY LIST (customer_id)"
        )
        try:
            await _add_customer(conn, "prov-scratch")
            assert "chunks_part" not in await partitioned_parents(conn)
            assert await ensure_tenant_partition(conn, "prov-scratch") is True
            assert "prov-scratch" not in await find_tenants_missing_partitions(conn)
        finally:
            await conn.execute("DROP TABLE chunks_part")


@pytest.mark.integration
async def test_create_customer_refuses_an_unsafe_id_before_inserting(live_db):
    with pytest.raises(ValueError, match="refusing"):
        await create_customer("bad id; x", "Bad")
    async with db_module.raw_conn() as conn:
        assert not await conn.fetchval(
            "SELECT EXISTS (SELECT 1 FROM customers WHERE customer_id = $1)", "bad id; x"
        )


@pytest.mark.integration
async def test_create_customer_returns_the_key_and_provisions(live_db):
    key = await create_customer("prov-created", "Created")
    assert key
    async with db_module.raw_conn() as conn:
        assert await partition_of(conn, "prov-created") == partition_name_for("prov-created")
