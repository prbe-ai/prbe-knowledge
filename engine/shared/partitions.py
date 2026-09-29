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

#: Creation-name prefix per partitioned parent. Mirrored by the SQL function
#: `kb_partition_name()` (migration 0144), which is what actually names new
#: partitions; tests/retrieval/test_tenant_provisioning.py pins the two equal.
#: The longest prefix + 40-char slug + "_" + 8 hex is 58 bytes, so the hash is
#: never truncated by the 63-byte identifier limit.
PARTITION_PREFIXES: dict[str, str] = {
    "chunks": PARTITION_PREFIX,
    "documents": "doc_p_",
    "usage_events": "ue_p_",
    "graph_nodes": "gn_p_",
    "graph_edges": "ge_p_",
    "graph_node_provenance": "gnp_p_",
}

#: DDL waits this long for a lock before giving up. Provisioning must not hang
#: behind a long-running search or an autovacuum; the caller retries.
#: `ATTACH` needs only SHARE UPDATE EXCLUSIVE, so in practice it does not queue.
PARTITION_LOCK_TIMEOUT = "3s"

#: SQL for the advisory-lock key every partition DDL takes (kb_provision_tenant,
#: drop, conversion swaps, purge detach), so they queue instead of deadlocking.
PARTITION_DDL_LOCK_SQL = "hashtextextended('kb_partition_ddl', 0)"

#: `customer_id` values that may be interpolated into DDL. DDL cannot take bind
#: parameters, so this is the boundary that makes interpolation safe. Deliberately
#: narrower than what `customers.customer_id` accepts: a value outside this set
#: is refused loudly rather than quoted and hoped for.
_SAFE_CUSTOMER_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,62}$")


class UnsafeCustomerId(ValueError):
    """A customer_id that must never be interpolated into DDL."""


def partition_name_for(customer_id: str, parent: str = CHUNKS_PARENT) -> str:
    """The name a NEW partition of `parent` gets for this tenant.

    Slug plus a hash of the ORIGINAL id. The slug alone is not enough: `a-b` and
    `a_b` are different tenants that sanitize to the same identifier, and the
    collision would surface as "relation already exists" during provisioning --
    or worse, as an ATTACH pointing a second tenant at the first one's table.
    The hash is taken before sanitizing, so distinct ids always get distinct
    names.

    A CREATION name only. Nothing looks a partition up by name: partitions made
    before migration 0144 carry sha1 names, later ones sha256 (computable in
    SQL, where kb_provision_tenant() names them), and `partition_of` finds
    either by its bound.
    """
    if not _SAFE_CUSTOMER_ID.match(customer_id):
        raise UnsafeCustomerId(f"refusing to build DDL for customer_id: {customer_id!r}")
    base = parent.removesuffix("__conv")
    prefix = PARTITION_PREFIXES.get(base)
    if prefix is None:
        raise ValueError(f"no partition prefix declared for parent {parent!r}")
    slug = re.sub(r"[^a-z0-9]+", "_", customer_id.lower()).strip("_")[:40]
    digest = hashlib.sha256(customer_id.encode()).hexdigest()[:8]
    return f"{prefix}{slug}_{digest}"


async def is_partitioned(conn: asyncpg.Connection, table: str = CHUNKS_PARENT) -> bool:
    """True once `table` is a partitioned parent.

    Every caller below is a no-op while this is False, so the same code ships
    before and after the conversion runs. That is what lets T4 merge and deploy
    ahead of T1 instead of having to land in the same breath as a 26 GB rebuild.
    """
    return await conn.fetchval(
        "SELECT relkind = 'p' FROM pg_class WHERE oid = to_regclass($1)", table
    ) or False


async def partition_of(
    conn: asyncpg.Connection, customer_id: str, *, parent: str = CHUNKS_PARENT
) -> str | None:
    """The name of this tenant's ATTACHED partition of `parent`, or None.

    Found by the partition's BOUND, not its name: names changed scheme in
    migration 0144 (sha1 -> sha256), and a name alone would also report a
    half-built standalone table as done. Asking `pg_inherits` makes a table
    whose ATTACH failed look unbuilt, which is what lets the next call finish
    the job.
    """
    if not _SAFE_CUSTOMER_ID.match(customer_id):
        raise UnsafeCustomerId(f"refusing to build DDL for customer_id: {customer_id!r}")
    return await conn.fetchval(
        """
        SELECT leaf.relname
        FROM pg_inherits h
        JOIN pg_class leaf ON leaf.oid = h.inhrelid
        WHERE h.inhparent = to_regclass($1)
          AND pg_get_expr(leaf.relpartbound, leaf.oid) = format('FOR VALUES IN (%L)', $2::text)
        """,
        parent,
        customer_id,
    )


async def partition_exists(
    conn: asyncpg.Connection, customer_id: str, *, parent: str = CHUNKS_PARENT
) -> bool:
    """True only when this tenant has a partition ATTACHED to the parent."""
    return await partition_of(conn, customer_id, parent=parent) is not None


async def partitioned_parents(conn: asyncpg.Connection) -> list[str]:
    """Every public table LIST-partitioned on customer_id, referenced ones first.

    The same catalog query kb_provision_tenant() iterates, so the guardian's
    reconciler and the provisioner agree on what "all partitions" means.
    """
    rows = await conn.fetch(
        """
        SELECT c.relname
        FROM pg_partitioned_table pt
        JOIN pg_class c ON c.oid = pt.partrelid
        JOIN pg_namespace ns ON ns.oid = c.relnamespace
        JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum = pt.partattrs[0]
        WHERE ns.nspname = 'public' AND pt.partstrat = 'l' AND pt.partnatts = 1
          AND a.attname = 'customer_id'
        ORDER BY (
            SELECT count(*) FROM pg_constraint k
            WHERE k.conrelid = c.oid AND k.contype = 'f'
              AND k.confrelid IN (SELECT partrelid FROM pg_partitioned_table)
        ), c.relname
        """
    )
    return [r["relname"] for r in rows]


async def find_orphan_partitions(
    conn: asyncpg.Connection, *, parent: str = CHUNKS_PARENT
) -> list[tuple[str, str]]:
    """Attached partitions whose tenant no longer exists. (partition, tenant).

    `ensure_tenant_partition` had no counterpart, so deleting a tenant left its
    partition attached forever. The rows go -- `chunks.customer_id` carries
    ON DELETE CASCADE from `customers` -- but the TABLE and its 15 indexes
    stay, and nothing will ever route a row to them again because no
    `customer_id` matches the bound.

    Observed on the research plane 2026-09-15: purging `richards-research-team`
    left `chunks_p_richards_research_team_40743306` holding 609 MB of dead
    tuples with the customer row already gone.

    That is not only wasted space. Locks are taken per RELATION, so every
    orphan permanently adds itself plus its indexes to the lock footprint of
    any query that does not prune -- the cost grows with the number of tenants
    ever deleted, not the number that exist.

    The bound value is read from the catalog rather than reversed out of the
    relation name, which is sanitized and hashed and cannot be turned back into
    a customer_id. The DEFAULT partition is never reported: it holds no bound
    and is meant to be empty.
    """
    rows = await conn.fetch(
        r"""
        SELECT c.relname AS partition,
               substring(pg_get_expr(c.relpartbound, c.oid)
                         from 'FOR VALUES IN \(''(.*)''\)') AS tenant
        FROM pg_class c
        JOIN pg_inherits h ON h.inhrelid = c.oid
        WHERE h.inhparent = to_regclass($1)
          AND c.relpartbound IS NOT NULL
          AND pg_get_expr(c.relpartbound, c.oid) <> 'DEFAULT'
        ORDER BY c.relname
        """,
        parent,
    )
    orphans: list[tuple[str, str]] = []
    for r in rows:
        tenant = r["tenant"]
        if tenant is None:
            continue
        exists = await conn.fetchval(
            "SELECT EXISTS (SELECT 1 FROM customers WHERE customer_id = $1)", tenant
        )
        if not exists:
            orphans.append((r["partition"], tenant))
    return orphans


async def drop_tenant_partition(
    conn: asyncpg.Connection,
    customer_id: str,
    *,
    parent: str = CHUNKS_PARENT,
    lock_timeout: str = PARTITION_LOCK_TIMEOUT,
) -> bool:
    """DETACH then DROP one tenant's partition. Returns True if it dropped one.

    REFUSES while the tenant still exists in `customers`. This drops a table
    and every row in it, and the only thing that makes that safe is that the
    tenant is already gone -- at which point the rows are gone too, by cascade,
    and the partition can never receive another. Without that gate this is a
    one-call way to delete a live tenant's entire corpus.

    DETACH before DROP, and both in one transaction. Dropping an attached
    partition works, but DETACH first means a failure between the two leaves a
    standalone table rather than a parent that briefly had a partition
    disappear underneath a concurrent plan.
    """
    if not await is_partitioned(conn, parent):
        return False
    part = await partition_of(conn, customer_id, parent=parent)
    if part is None:
        return False
    still_there = await conn.fetchval(
        "SELECT EXISTS (SELECT 1 FROM customers WHERE customer_id = $1)", customer_id
    )
    if still_there:
        raise ValueError(
            f"refusing to drop the partition for {customer_id!r}: the tenant is "
            f"still in `customers`. Delete the tenant first; the rows go by "
            f"cascade and this reclaims what is left."
        )
    async with conn.transaction():
        # The same lock kb_provision_tenant() takes: partition DDL queues.
        await conn.execute(f"SELECT pg_advisory_xact_lock({PARTITION_DDL_LOCK_SQL})")
        await conn.execute(f"SET LOCAL lock_timeout = '{lock_timeout}'")
        await conn.execute(f'ALTER TABLE {parent} DETACH PARTITION "{part}"')
        await conn.execute(f'DROP TABLE "{part}"')
    return True


async def ensure_tenant_partition(
    conn: asyncpg.Connection,
    customer_id: str,
    *,
    parent: str = CHUNKS_PARENT,
    lock_timeout: str = PARTITION_LOCK_TIMEOUT,
) -> bool:
    """Give this tenant a partition on EVERY partitioned parent. True if any was created.

    Delegates to the SQL function kb_provision_tenant() (migration 0144), the
    one implementation research-os also calls. `parent` and `lock_timeout` are
    kept for callers' signatures; the function covers every parent and uses
    its own 3 s lock_timeout.

    MUST NOT run inside the transaction that inserted the customer: ATTACH
    takes SHARE ROW EXCLUSIVE on `customers` through the cloned FK, and two
    such transactions deadlock (see 0144). Callers commit the customer first.
    If `conn` is already inside a transaction this runs as a savepoint of it,
    which is exactly that mistake, so it refuses.

    NOT swallowed on failure: a tenant with no partition writes into DEFAULT
    (or, once DEFAULT is gone, cannot write at all). The pg_search guardian
    reconciles missing partitions every minute as the backstop.
    """
    del parent, lock_timeout  # every parent, the function's own timeout
    if not _SAFE_CUSTOMER_ID.match(customer_id):
        raise UnsafeCustomerId(f"refusing to build DDL for customer_id: {customer_id!r}")
    if conn.is_in_transaction():
        raise RuntimeError(
            "ensure_tenant_partition must run in its own transaction, not inside "
            "the one that inserted the customer (deadlock; see migration 0144)"
        )
    async with conn.transaction():
        created = await conn.fetchval("SELECT kb_provision_tenant($1)", customer_id)
    if created:
        log.info("partitions.created", customer=customer_id, partitions=created)
    return bool(created)


async def ensure_tenant_partitions(customer_id: str) -> bool:
    """`ensure_tenant_partition` on a fresh pooled connection (its own transaction)."""
    from engine.shared.db import raw_conn

    async with raw_conn() as conn:
        return await ensure_tenant_partition(conn, customer_id)


async def find_tenants_missing_partitions(conn: asyncpg.Connection) -> list[str]:
    """Active customers lacking an attached partition on some partitioned parent."""
    parents = await partitioned_parents(conn)
    if not parents:
        return []
    rows = await conn.fetch(
        """
        SELECT c.customer_id
        FROM customers c
        WHERE c.status = 'active'
          AND c.customer_id ~ '^[a-zA-Z0-9][a-zA-Z0-9._-]{0,62}$'
          AND EXISTS (
              SELECT 1 FROM unnest($1::text[]) AS p(parent)
              WHERE NOT EXISTS (
                  SELECT 1
                  FROM pg_inherits h
                  JOIN pg_class leaf ON leaf.oid = h.inhrelid
                  WHERE h.inhparent = to_regclass(p.parent)
                    AND pg_get_expr(leaf.relpartbound, leaf.oid)
                        = format('FOR VALUES IN (%L)', c.customer_id)
              )
          )
        ORDER BY c.customer_id
        """,
        parents,
    )
    return [r["customer_id"] for r in rows]


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
        # Called directly, inside THIS transaction: the rows just left DEFAULT
        # in it, so ATTACH's scan of DEFAULT finds none. No customers row is
        # written here, so the deadlock ensure_tenant_partition guards against
        # cannot arise.
        await conn.fetchval("SELECT kb_provision_tenant($1)", customer_id)
        if not moved:
            # Nothing to split; the partition now exists so the next write
            # lands correctly.
            return 0
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
    "drop_tenant_partition",
    "ensure_tenant_partition",
    "ensure_tenant_partitions",
    "find_orphan_partitions",
    "find_tenants_missing_partitions",
    "is_partitioned",
    "partition_exists",
    "partition_name_for",
    "partition_of",
    "partitioned_parents",
    "split_default",
]
