"""Per-tenant partitions of `chunks`.

WHY THIS EXISTS
---------------
`chunks` is partitioned BY LIST (customer_id) so that each tenant's ANN index is
priced at that tenant's size. Before partitioning there was ONE HNSW index over
every tenant, and pgvector prices an `ORDER BY <distance>` scan of it at the
whole-index cost -- measured 3,808,868 cost units, IDENTICAL for every tenant,
while the competing brute-force plan is priced on that tenant's own rows. Small
tenants therefore lost the planner's comparison and were pushed onto a
1,003 ms / 646,471-buffer scan where the index would have taken 80 ms. One
tenant's ingestion could push another tenant off the index: `probe` grew 53%
over two weeks and `anthrogen`'s share of the shared index fell 16.4% -> 11.8%,
which is what produced the 14.7 s search on 2026-09-14.

A tenant with no partition of its own lands in DEFAULT -- a shared relation with
a shared index -- which silently restores exactly that fault for them. So a
partition is part of creating a tenant, not a later cleanup.

WHY CREATE + ATTACH AND NOT `CREATE TABLE ... PARTITION OF`
-----------------------------------------------------------
`CREATE TABLE ... PARTITION OF` takes ACCESS EXCLUSIVE on the parent. Measured on
paradedb 0.23.4-pg16 with the production index set in shape: with a 6-second
search holding ACCESS SHARE on the parent it blocked, hit a 3 s lock_timeout,
and while it waited it queued every later query behind itself.

`CREATE TABLE (LIKE parent INCLUDING ALL)` + `ALTER TABLE ... ATTACH PARTITION`
takes SHARE UPDATE EXCLUSIVE instead: 7.1 ms + 0.9 ms under the same held lock,
no wait. ATTACH also creates whatever partitioned-index children LIKE did not
produce -- verified for both the pgvector HNSW index and the pg_search BM25
index -- so the new partition is fully indexed the moment it is attached.

    tenant created ─► ensure_tenant_partition()
                          │
                          ├─ CREATE TABLE chunks_p_<slug>_<hash> (LIKE chunks INCLUDING ALL)
                          │     (standalone table; no lock on the parent)
                          │
                          ├─ ALTER TABLE chunks ATTACH PARTITION ... FOR VALUES IN ('<tenant>')
                          │     SHARE UPDATE EXCLUSIVE, ~1 ms, index children filled in
                          │
                          └─ ENABLE + FORCE RLS + tenant_isolation policy on the partition
                                (parent policy covers parent-routed queries; this
                                 covers anything that names the partition directly)

ONE CAVEAT ON ATTACH: with a DEFAULT partition present, ATTACH must scan DEFAULT
to prove it holds no rows for the incoming tenant. That is instant while DEFAULT
is empty, which is what the guardian's DEFAULT alarm exists to keep true. If
DEFAULT has acquired rows for this tenant, use `split_default()` instead.
"""

from __future__ import annotations

import hashlib
import re

import asyncpg

from engine.shared.logging import get_logger

log = get_logger(__name__)

#: The partitioned table and the column it is partitioned on.
CHUNKS_PARENT = "chunks"
PARTITION_KEY = "customer_id"

#: Identifier prefix for a tenant partition. Kept short so the hashed suffix
#: fits inside PostgreSQL's 63-byte identifier limit.
PARTITION_PREFIX = "chunks_p_"

#: DDL waits this long for a lock before giving up. Provisioning must not hang
#: behind a long-running search or an autovacuum; the caller retries.
#: `ATTACH` needs only SHARE UPDATE EXCLUSIVE, so in practice it does not queue.
PARTITION_LOCK_TIMEOUT = "3s"

#: `customer_id` values that may be interpolated into DDL. DDL cannot take bind
#: parameters, so this is the boundary that makes interpolation safe. Deliberately
#: narrower than what `customers.customer_id` accepts: a value outside this set
#: is refused loudly rather than quoted and hoped for.
_SAFE_CUSTOMER_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,62}$")


class UnsafeCustomerId(ValueError):
    """A customer_id that must never be interpolated into DDL."""


def partition_name_for(customer_id: str) -> str:
    """Deterministic, collision-free partition name for a tenant.

    Slug plus a hash of the ORIGINAL id. The slug alone is not enough: `a-b` and
    `a_b` are different tenants that sanitize to the same identifier, and the
    collision would surface as "relation already exists" during provisioning --
    or worse, as an ATTACH pointing a second tenant at the first one's table.
    The hash is taken before sanitizing, so distinct ids always get distinct
    names.
    """
    if not _SAFE_CUSTOMER_ID.match(customer_id):
        raise UnsafeCustomerId(f"refusing to build DDL for customer_id: {customer_id!r}")
    slug = re.sub(r"[^a-z0-9]+", "_", customer_id.lower()).strip("_")[:40]
    digest = hashlib.sha1(customer_id.encode()).hexdigest()[:8]
    return f"{PARTITION_PREFIX}{slug}_{digest}"


async def is_partitioned(conn: asyncpg.Connection, table: str = CHUNKS_PARENT) -> bool:
    """True once `table` is a partitioned parent.

    Every caller below is a no-op while this is False, so the same code ships
    before and after the conversion runs. That is what lets T4 merge and deploy
    ahead of T1 instead of having to land in the same breath as a 26 GB rebuild.
    """
    return await conn.fetchval(
        "SELECT relkind = 'p' FROM pg_class WHERE oid = to_regclass($1)", table
    ) or False


async def partition_exists(
    conn: asyncpg.Connection, customer_id: str, *, parent: str = CHUNKS_PARENT
) -> bool:
    """True only when the partition exists AND is attached to the parent.

    Name existence alone is the wrong test. `CREATE TABLE` and `ATTACH` are two
    statements; if ATTACH fails the standalone table survives, and a
    name-existence check would then report "already done" forever -- leaving
    that tenant writing into DEFAULT with nothing ever retrying. Asking
    `pg_inherits` makes a half-built partition look unbuilt, which is what lets
    the next call finish the job.
    """
    return (
        await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT 1 FROM pg_inherits h
                WHERE h.inhrelid = to_regclass($1)
                  AND h.inhparent = to_regclass($2)
            )
            """,
            partition_name_for(customer_id),
            parent,
        )
        or False
    )


async def ensure_tenant_partition(
    conn: asyncpg.Connection,
    customer_id: str,
    *,
    parent: str = CHUNKS_PARENT,
    lock_timeout: str = PARTITION_LOCK_TIMEOUT,
) -> bool:
    """Create and attach this tenant's partition. Returns True if it created one.

    Idempotent: an existing partition is left alone. Safe to call on a database
    where the conversion has not run (returns False, does nothing).

    NOT swallowed on failure. A tenant whose partition could not be created is a
    tenant whose search will be slow and whose rows sit in DEFAULT; provisioning
    should fail loudly rather than hand back a degraded tenant.
    """
    if not await is_partitioned(conn, parent):
        return False
    if await partition_exists(conn, customer_id, parent=parent):
        return False

    part = partition_name_for(customer_id)
    # Safe: `part` comes from partition_name_for (regex-gated, then sanitized to
    # [a-z0-9_]), and `customer_id` passed the same regex. Neither can carry a
    # quote. DDL cannot take bind parameters, so this is the only option and the
    # gate above is what makes it sound.
    literal = customer_id.replace("'", "''")

    # ONE TRANSACTION, for two reasons that are easy to miss.
    #
    # 1. `SET LOCAL` outside a transaction block is a NO-OP -- Postgres emits
    #    `WARNING: SET LOCAL can only be used in transaction blocks` and moves
    #    on. Both production callers use `raw_conn()`, which is autocommit, so
    #    the documented lock cap simply would not exist and ATTACH could queue
    #    behind autovacuum indefinitely. (Verified on the live database: after a
    #    bare `SET LOCAL lock_timeout='3s'`, `SHOW lock_timeout` still reads 0.)
    # 2. `CREATE TABLE` then `ATTACH` as separate autocommit statements leaves
    #    an orphaned standalone table if ATTACH fails. Wrapped, a failure leaves
    #    nothing behind and the next call starts clean.
    async with conn.transaction():
        await conn.execute(f"SET LOCAL lock_timeout = '{lock_timeout}'")
        # Standalone first: this touches the parent not at all, so a slow build
        # cannot block a reader. INCLUDING ALL brings the column defaults,
        # checks, storage parameters and index definitions across; ATTACH then
        # matches them to the parent's partitioned indexes and builds any that
        # LIKE could not express (the BM25 child among them).
        await conn.execute(f'CREATE TABLE "{part}" (LIKE "{parent}" INCLUDING ALL)')
        await conn.execute(
            f'ALTER TABLE "{parent}" ATTACH PARTITION "{part}" '
            f"FOR VALUES IN ('{literal}')"
        )
        # Parent policies govern parent-routed queries, which is every
        # application path. This covers the other door: a query naming the
        # partition directly would otherwise see every row in it with no tenant
        # check at all.
        await conn.execute(f'ALTER TABLE "{part}" ENABLE ROW LEVEL SECURITY')
        await conn.execute(f'ALTER TABLE "{part}" FORCE ROW LEVEL SECURITY')
        await conn.execute(
            f'CREATE POLICY tenant_isolation ON "{part}" '
            f"USING ({PARTITION_KEY} = current_setting('app.current_customer_id', true))"
        )
    log.info(
        "partitions.created", customer=customer_id, partition=part, parent=parent
    )
    return True


async def default_partition_name(
    conn: asyncpg.Connection, parent: str = CHUNKS_PARENT
) -> str | None:
    return await conn.fetchval(
        """
        SELECT part.relname
        FROM pg_inherits h
        JOIN pg_class part ON part.oid = h.inhrelid
        WHERE h.inhparent = to_regclass($1)
          AND pg_get_expr(part.relpartbound, part.oid) = 'DEFAULT'
        """,
        parent,
    )


async def split_default(
    conn: asyncpg.Connection,
    customer_id: str,
    *,
    parent: str = CHUNKS_PARENT,
    lock_timeout: str = PARTITION_LOCK_TIMEOUT,
) -> int:
    """Move one tenant's rows out of DEFAULT into their own partition.

    Returns the number of rows moved.

    This is the reconciliation path for a tenant created by something that did
    not call `ensure_tenant_partition` -- a seed script, a restore, a direct
    INSERT into `customers`. The guardian alarms when DEFAULT is non-empty; this
    is what an operator runs in response.

    ATTACH cannot be used while DEFAULT holds the tenant's rows (it would have
    to scan DEFAULT and would find conflicting rows), so this does the honest
    thing: create the partition attached for the tenant's values AFTER removing
    their rows from DEFAULT, then put the rows back through the parent so they
    route correctly.

    Runs in ONE transaction. A half-done split would leave a tenant's rows
    deleted from DEFAULT and not yet inserted anywhere, which is data loss; the
    transaction is what makes the failure mode "nothing happened".
    """
    if not await is_partitioned(conn, parent):
        return 0
    default_name = await default_partition_name(conn, parent)
    if default_name is None:
        return 0
    part = partition_name_for(customer_id)  # validates customer_id

    # GENERATED columns must be named in neither the RETURNING list nor the
    # INSERT: Postgres refuses `cannot insert a non-DEFAULT value into column
    # "content_tsv"`. `RETURNING *` would sweep them up, so the column list is
    # built explicitly from the catalog with `attgenerated = ''` as the filter.
    # Found by tests/retrieval/test_chunks_partition_pruning.py.
    columns = [
        r["attname"]
        for r in await conn.fetch(
            """
            SELECT attname FROM pg_attribute
            WHERE attrelid = to_regclass($1)
              AND attnum > 0 AND NOT attisdropped AND attgenerated = ''
            ORDER BY attnum
            """,
            parent,
        )
    ]
    collist = ", ".join(f'"{c}"' for c in columns)

    async with conn.transaction():
        await conn.execute(f"SET LOCAL lock_timeout = '{lock_timeout}'")
        # The re-INSERT goes through the parent, and `chunks`' tenant_isolation
        # policy carries a WITH CHECK clause (verified on the live database), so
        # without the GUC every row is rejected with "new row violates row-level
        # security policy". The transaction would roll back cleanly -- no data
        # loss -- but the documented remedy for the guardian's
        # `kb_chunks_default_partition_nonempty` alarm would never once succeed.
        await conn.execute(
            "SELECT set_config('app.current_customer_id', $1, true)", customer_id
        )
        moved = await conn.fetch(
            f'DELETE FROM ONLY "{default_name}" WHERE {PARTITION_KEY} = $1 '
            f"RETURNING {collist}",
            customer_id,
        )
        if not moved:
            # Nothing to split; just make sure the partition exists so the next
            # write lands correctly.
            await ensure_tenant_partition(conn, customer_id, parent=parent)
            return 0
        await ensure_tenant_partition(conn, customer_id, parent=parent)
        placeholders = ", ".join(f"${i + 1}" for i in range(len(columns)))
        await conn.executemany(
            f'INSERT INTO "{parent}" ({collist}) VALUES ({placeholders})',
            [tuple(r[c] for c in columns) for r in moved],
        )
    log.info(
        "partitions.split_default",
        customer=customer_id,
        partition=part,
        rows=len(moved),
    )
    return len(moved)


__all__ = [
    "CHUNKS_PARENT",
    "UnsafeCustomerId",
    "default_partition_name",
    "ensure_tenant_partition",
    "is_partitioned",
    "partition_exists",
    "partition_name_for",
    "split_default",
]
