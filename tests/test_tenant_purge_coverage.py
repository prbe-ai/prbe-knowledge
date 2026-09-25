"""No kb table that holds a tenant's rows may escape the tenant purge.

research-os deletes a tenant from this database in `purge_customer`
(research-os `app/core/kb_mirror.py`, `_DRAIN_PLAN_SQL`), in three steps:

  1. batch-delete every table with a SINGLE-COLUMN foreign key to
     `customers(customer_id)` ON DELETE CASCADE (partitions through their parent);
  2. batch-delete the tables it names in `_UNLINKED_TENANT_TABLES` (today
     'session_batch_receipts', 'github_backfill_jobs',
     'github_backfill_retry_receipts');
  3. `DELETE FROM customers`, whose cascade reaches every table chained below.

A table a migration adds with a `customer_id` column and no cascade path to
`customers` is deleted by none of that. It outlives the tenant, and the purge
still reports success -- nothing about it fails. So every table (any schema)
with a `customer_id` column must be one of:

  (a) DIRECT: FK `customer_id -> customers(customer_id)` ON DELETE CASCADE,
      the exact shape step 1 selects;
  (b) CHAINED: FK ON DELETE CASCADE that maps its `customer_id` onto the
      `customer_id` of a table that is itself (a) or (b), so step 3 reaches it
      (and it can only point at rows of the SAME tenant);
  (c) ALLOWED_UNLINKED below, with the reason and how research-os deletes it.

Either FK must also be VALIDATED: a `NOT VALID` constraint never checked the
rows that predate it, so rows pointing at a parent that no longer exists are
reached by no cascade. And every column of a CHAINED FK must be NOT NULL: under
the default MATCH SIMPLE a row with a NULL in any FK column is not constrained
at all, so `(customer_id='t', installation_id=NULL)` has no parent whose
deletion would cascade to it.

WHEN ALLOWED_UNLINKED CHANGES, research-os's `_UNLINKED_TENANT_TABLES` must
change with it -- a table added here and not there is exactly the leak this
file exists to stop. research-os's three named tables are all chained here
(composite FKs through `session_streams` / `github_installations`), so its
explicit drain of them is a second path, not the only one; the test pins that.

Runs against the database CI builds (`db/schema.sql` + stamp), read-only
except for a rolled-back probe that proves the check can fail.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from engine.shared.db import raw_conn

#: Tables with a `customer_id` and no cascade path, and how each is deleted.
#: Keep in step with research-os `app/core/kb_mirror.py:_UNLINKED_TENANT_TABLES`.
ALLOWED_UNLINKED: dict[str, str] = {}

#: research-os drains these by name as well (`_UNLINKED_TENANT_TABLES`).
RESEARCH_OS_NAMED = ("session_batch_receipts", "github_backfill_jobs", "github_backfill_retry_receipts")

_TENANT_TABLES_SQL = """
    SELECT c.oid::regclass::text AS tbl
      FROM pg_class c
      JOIN pg_namespace n ON n.oid = c.relnamespace
      JOIN pg_attribute a
        ON a.attrelid = c.oid AND a.attname = 'customer_id' AND NOT a.attisdropped
     WHERE c.relkind IN ('r', 'p')
       AND NOT c.relispartition            -- purged through the parent
       AND n.nspname NOT IN ('pg_catalog', 'information_schema')
       AND c.oid <> 'customers'::regclass
"""

#: Every declared FK (not a partition's inherited copy), columns in key order.
_FOREIGN_KEYS_SQL = """
    SELECT con.conrelid::regclass::text AS child,
           con.confrelid::regclass::text AS parent,
           con.confdeltype::text AS on_delete,   -- "char": bytes to asyncpg otherwise
           con.convalidated AS validated,
           (SELECT bool_and(a.attnotnull) FROM unnest(con.conkey) k(num)
              JOIN pg_attribute a ON a.attrelid = con.conrelid AND a.attnum = k.num
           ) AS child_not_null,
           ARRAY(SELECT a.attname FROM unnest(con.conkey) WITH ORDINALITY k(num, ord)
                 JOIN pg_attribute a ON a.attrelid = con.conrelid AND a.attnum = k.num
                 ORDER BY k.ord)::text[] AS child_cols,
           ARRAY(SELECT a.attname FROM unnest(con.confkey) WITH ORDINALITY k(num, ord)
                 JOIN pg_attribute a ON a.attrelid = con.confrelid AND a.attnum = k.num
                 ORDER BY k.ord)::text[] AS parent_cols
      FROM pg_constraint con
     WHERE con.contype = 'f' AND con.conparentid = 0
"""

_CASCADE = "c"
_ON_DELETE = {"a": "NO ACTION", "r": "RESTRICT", "n": "SET NULL", "d": "SET DEFAULT", "c": "CASCADE"}


@dataclass(frozen=True)
class Coverage:
    direct: frozenset[str]
    chained: frozenset[str]
    uncovered: frozenset[str]
    #: FKs onto `customers` that do not cascade: `DELETE FROM customers` fails.
    blocking: frozenset[str]


async def _coverage(conn) -> Coverage:
    customers = await conn.fetchval("SELECT 'customers'::regclass::text")
    tenant_tables = {r["tbl"] for r in await conn.fetch(_TENANT_TABLES_SQL)}
    fks = await conn.fetch(_FOREIGN_KEYS_SQL)

    direct = {
        fk["child"]
        for fk in fks
        if fk["parent"] == customers
        and fk["on_delete"] == _CASCADE
        and fk["validated"]
        and list(fk["child_cols"]) == ["customer_id"]
        and list(fk["parent_cols"]) == ["customer_id"]
    }
    covered = set(direct)
    while True:
        reached = {
            fk["child"]
            for fk in fks
            if fk["parent"] in covered
            and fk["child"] not in covered
            and fk["on_delete"] == _CASCADE
            and fk["validated"]
            and fk["child_not_null"]
            and "customer_id" in fk["child_cols"]
            and fk["parent_cols"][list(fk["child_cols"]).index("customer_id")] == "customer_id"
        }
        if not reached:
            break
        covered |= reached
    blocking = {
        f"{fk['child']}({', '.join(fk['child_cols'])}) ON DELETE {_ON_DELETE[fk['on_delete']]}"
        for fk in fks
        if fk["parent"] == customers and fk["on_delete"] != _CASCADE
    }
    return Coverage(
        direct=frozenset(direct & tenant_tables),
        chained=frozenset((covered - direct) & tenant_tables),
        uncovered=frozenset(tenant_tables - covered),
        blocking=frozenset(blocking),
    )


@pytest.mark.asyncio
async def test_every_tenant_table_is_reached_by_the_purge(live_db) -> None:
    async with raw_conn() as conn:
        coverage = await _coverage(conn)

    escaped = sorted(coverage.uncovered - ALLOWED_UNLINKED.keys())
    assert not escaped, (
        f"{escaped} hold a customer_id but nothing deletes them when a tenant is purged. "
        "Give each an FK ON DELETE CASCADE -- `customer_id` to customers(customer_id), or "
        "customer_id-to-customer_id to a tenant table that has one, validated and with "
        "every FK column NOT NULL -- or add it to "
        "ALLOWED_UNLINKED with its reason AND to research-os's `_UNLINKED_TENANT_TABLES`."
    )
    stale = sorted(ALLOWED_UNLINKED.keys() - coverage.uncovered)
    assert not stale, f"{stale} now cascade (or no longer exist); drop them from ALLOWED_UNLINKED"


@pytest.mark.asyncio
async def test_nothing_blocks_the_final_delete_from_customers(live_db) -> None:
    """A non-cascading FK onto `customers` makes the purge's last step fail on
    every attempt, leaving the tenant half-deleted for good."""
    async with raw_conn() as conn:
        coverage = await _coverage(conn)
    assert not coverage.blocking, sorted(coverage.blocking)


@pytest.mark.asyncio
async def test_the_check_sees_what_it_should(live_db) -> None:
    """Pinned so a catalog query that quietly matched nothing could not pass the
    two tests above: the big tables are direct, research-os's named ones are
    chained, and the count is the whole schema's, not a handful."""
    async with raw_conn() as conn:
        coverage = await _coverage(conn)
    assert {"chunks", "documents", "ingestion_queue", "session_streams"} <= coverage.direct
    assert set(RESEARCH_OS_NAMED) <= coverage.chained
    assert len(coverage.direct | coverage.chained) >= 40


@pytest.mark.asyncio
async def test_the_check_fails_on_a_table_that_escapes(live_db) -> None:
    """A negative control, rolled back: an unlinked tenant table is caught, a
    non-cascading FK is caught, a chained FK a row can slip past (a nullable
    FK column, or a NOT VALID constraint) is caught, and a correctly chained
    table is not."""
    async with raw_conn() as conn:
        tx = conn.transaction()
        await tx.start()
        try:
            await conn.execute(
                """
                CREATE TABLE purge_probe_orphan (customer_id TEXT NOT NULL);
                CREATE TABLE purge_probe_restrict (
                    customer_id TEXT NOT NULL REFERENCES customers(customer_id));
                CREATE TABLE purge_probe_chained (
                    customer_id TEXT NOT NULL, installation_id TEXT NOT NULL,
                    FOREIGN KEY (customer_id, installation_id)
                      REFERENCES github_installations(customer_id, installation_id)
                      ON DELETE CASCADE);
                -- MATCH SIMPLE: a NULL installation_id leaves the row unchecked,
                -- with no parent to cascade from.
                CREATE TABLE purge_probe_nullable (
                    customer_id TEXT NOT NULL, installation_id TEXT,
                    FOREIGN KEY (customer_id, installation_id)
                      REFERENCES github_installations(customer_id, installation_id)
                      ON DELETE CASCADE);
                -- NOT VALID: rows older than the constraint were never checked.
                CREATE TABLE purge_probe_not_valid (
                    customer_id TEXT NOT NULL, installation_id TEXT NOT NULL);
                ALTER TABLE purge_probe_not_valid
                  ADD FOREIGN KEY (customer_id, installation_id)
                  REFERENCES github_installations(customer_id, installation_id)
                  ON DELETE CASCADE NOT VALID;
                """
            )
            coverage = await _coverage(conn)
        finally:
            await tx.rollback()

    assert {
        "purge_probe_orphan",
        "purge_probe_restrict",
        "purge_probe_nullable",
        "purge_probe_not_valid",
    } <= coverage.uncovered
    assert "purge_probe_chained" in coverage.chained
    assert any(b.startswith("purge_probe_restrict(") for b in coverage.blocking)
