"""Index tombstoned document versions, so the tombstone purge can find them.

scripts/cron_tombstone_purge.py hard-deletes documents whose current version
has been a tombstone (`deleted_at` set) for longer than the retention window.
It walks each tenant's tombstones oldest first, keyset-paginated on
(deleted_at, doc_id). No existing index serves that: `idx_documents_live` and
`idx_documents_stats_live` are partial on `valid_to IS NULL`, the second with
`deleted_at IS NULL` -- the opposite predicate -- so the scan would read every
live document of the tenant to find the handful that are tombstones.

Partial on `deleted_at IS NOT NULL` rather than also `valid_to IS NULL`: a
code-graph repo disconnect tombstones in place and closes the row
(kb/handlers/codegraph.py), so its tombstones have no live row and a
`valid_to IS NULL` index would never show them to the purge. Tombstone versions
are a small minority of `documents`, so the index stays small either way.

PLAIN CREATE INDEX, not CONCURRENTLY, and IF NOT EXISTS: the same convention as
0105/0124/0135. The plain form holds a SHARE lock on `documents` (writes wait)
for the length of one scan of the table -- 4 GB on the research plane -- so
BUILD IT ATTENDED FIRST there, before this migration deploys:

    CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_tombstones
        ON documents (customer_id, deleted_at, doc_id)
        WHERE deleted_at IS NOT NULL;

after which this migration is a no-op on that plane. Fresh installs and small
self-hosts take the plain build, whose lock is a non-event at their size. The
INVALID-leftover guard is why `IF NOT EXISTS` alone is not enough: it matches
on NAME, so an interrupted CONCURRENTLY build leaves a corpse the planner
ignores and this migration would record itself applied against it.
"""

from alembic import op

revision = "0139_tombstone_purge_index"
down_revision = "0138_queue_extraction_outcome"
branch_labels = None
depends_on = None

_NAME = "idx_documents_tombstones"


def upgrade() -> None:
    # Drop an INVALID leftover FIRST -- see the module docstring.
    op.execute(
        f"""
        DO $$
        DECLARE
            invalid_name text;
        BEGIN
            SELECT c.relname INTO invalid_name
            FROM pg_index i
            JOIN pg_class c ON c.oid = i.indexrelid
            WHERE c.relname = '{_NAME}'
              AND NOT i.indisvalid;
            IF invalid_name IS NOT NULL THEN
                RAISE NOTICE 'dropping INVALID %', invalid_name;
                EXECUTE 'DROP INDEX ' || quote_ident(invalid_name);
            END IF;
        END $$
        """
    )
    op.execute(
        f"""
        CREATE INDEX IF NOT EXISTS {_NAME}
            ON documents (customer_id, deleted_at, doc_id)
            WHERE deleted_at IS NOT NULL
        """
    )


def downgrade() -> None:
    op.execute(f"DROP INDEX IF EXISTS {_NAME}")
