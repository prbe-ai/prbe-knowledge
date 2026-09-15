"""chunks: tenant-qualified unique key, additive — step (a) of the partition sequence

Revision ID: 0134_chunks_tenant_unique
Revises: 0133_retire_session_user_acls
Create Date: 2026-09-15

WHY
---
`chunks` is being converted to a table PARTITIONED BY LIST (customer_id) so that
each tenant's ANN index is priced at that tenant's size. Postgres requires every
UNIQUE constraint on a partitioned table to contain the partition key, and
`chunks` has two that do not:

    chunks_doc_id_content_hash_key   UNIQUE (doc_id, content_hash)
    chunks_chunk_id_unique           UNIQUE (chunk_id)

This migration is the ADDITIVE half of the change, and it is deliberately the
only half that ships on a deploy. The sequence is:

    (a) THIS migration      add UNIQUE (customer_id, doc_id, content_hash).
                            Both old uniques stay. Nothing breaks; old and new
                            ingestion code are both valid against the table.
    (b) same release        `_insert_chunk` / `_insert_chunks_batch` switch their
                            ON CONFLICT target to the new key. Valid because (a)
                            already ran -- the kb-migrate Job is a `pre-upgrade`
                            hook (weight 6), so it completes before the engine
                            Deployment rolls.
    (c) drain               old ingestion workers finish; every writer now names
                            the new key.
    (d) conversion script   builds the partitioned table with ONLY the
                            tenant-qualified unique, and the old indexes die with
                            the old table at swap time.

There is deliberately NO drop step here. Dropping `chunks_chunk_id_unique` or
the old composite before (c) would break in-flight writers, and after (d) there
is nothing left to drop. A release that ships (a)+(b) is independently
revertible: reverting the code leaves a redundant index, not a broken table.

NO DUPLICATE RISK. `(doc_id, content_hash)` is already unique table-wide, so a
superset of those columns cannot have duplicates. The CONCURRENTLY build can
still fail on a lock timeout or a crash, which leaves an INVALID index behind --
hence the drop-if-invalid guard, mirroring 0124.

WHY (customer_id, doc_id, content_hash) AND NOT (doc_id, content_hash, customer_id)
Leading with `customer_id` matches `chunks_pkey (customer_id, chunk_id)` and every
RLS-filtered access path, so the index is usable for tenant-scoped prefix scans
as well as for the constraint. Trailing it would make the index useless for
anything but the uniqueness check.
"""

from __future__ import annotations

from alembic import op

revision = "0134_chunks_tenant_unique"
down_revision = "0133_retire_session_user_acls"
branch_labels = None
depends_on = None

INDEX_NAME = "chunks_customer_doc_hash_key"


def upgrade() -> None:
    # A previous attempt that was interrupted leaves an INVALID index which
    # CREATE INDEX ... IF NOT EXISTS will happily skip over, leaving the
    # database permanently without a usable index while reporting success.
    op.execute(
        f"""
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM pg_class c
                JOIN pg_index i ON i.indexrelid = c.oid
                WHERE c.relname = '{INDEX_NAME}' AND NOT i.indisvalid
            ) THEN
                EXECUTE 'DROP INDEX {INDEX_NAME}';
            END IF;
        END $$;
        """
    )
    with op.get_context().autocommit_block():
        op.execute(
            f"CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS {INDEX_NAME} "
            "ON chunks (customer_id, doc_id, content_hash)"
        )
    op.execute("ANALYZE chunks")


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {INDEX_NAME}")
