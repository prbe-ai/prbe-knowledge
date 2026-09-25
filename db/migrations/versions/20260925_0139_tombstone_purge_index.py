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

CONCURRENTLY, in alembic's autocommit block (the 0122/0130/0137 shape). This
migration runs UNATTENDED: merging rolls the managed plane through its
`managed-migrate` hook, and the research plane's `engine-kb-migrate` hook runs
it on the next research-os deploy. Both hold a large `documents` table (4 GB on
research). A plain CREATE INDEX holds a SHARE lock for one full scan of it, and
every ingest write -- every document version, every tombstone -- waits for the
whole build. CONCURRENTLY blocks no reads or writes.

Its known cost, from 0105: CONCURRENTLY waits for every transaction older than
the build before it marks the index valid, and a deploy hook with a timeout can
be killed in that wait, leaving an INVALID index behind. Two things make that
recoverable rather than an outage:

  * the INVALID-leftover guard below. `IF NOT EXISTS` matches on NAME, so
    without it a retry would record itself applied against a corpse the planner
    ignores. With it, the next deploy drops the corpse and builds again.
  * nothing reads this index on the request path. Until it is valid the purge
    walks its tombstones more slowly; no user-facing query changes plan.

If a deploy does stall here, the first thing to look for is a long-lived
transaction (`pg_stat_activity` ordered by `xact_start`). The index can also be
built by hand first, after which this migration is a no-op:

    CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_tombstones
        ON documents (customer_id, deleted_at, doc_id)
        WHERE deleted_at IS NOT NULL;
"""

import sqlalchemy as sa
from alembic import op

revision = "0139_tombstone_purge_index"
down_revision = "0138_queue_extraction_outcome"
branch_labels = None
depends_on = None

_NAME = "idx_documents_tombstones"


def upgrade() -> None:
    # CREATE INDEX CONCURRENTLY cannot run inside a transaction; the autocommit
    # block commits alembic's transaction around these statements only.
    with op.get_context().autocommit_block():
        # Drop an INVALID leftover FIRST -- see the module docstring. Guarded
        # on indisvalid, so a healthy index on a re-run is left alone. Checked
        # here rather than in a DO block (0062/0101) so the drop can be
        # CONCURRENTLY too: a plain DROP INDEX takes ACCESS EXCLUSIVE on
        # `documents`, and one queued behind a long transaction stalls every
        # read queued behind it.
        invalid = op.get_bind().execute(
            sa.text(
                """
                SELECT 1
                FROM pg_class c
                JOIN pg_index i ON i.indexrelid = c.oid
                WHERE c.relname = :name AND NOT i.indisvalid
                """
            ),
            {"name": _NAME},
        ).first()
        if invalid is not None:
            op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {_NAME}")
        op.execute(
            f"""
            CREATE INDEX CONCURRENTLY IF NOT EXISTS {_NAME}
                ON documents (customer_id, deleted_at, doc_id)
                WHERE deleted_at IS NOT NULL
            """
        )


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {_NAME}")
