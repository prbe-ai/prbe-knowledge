"""Index the rows the tombstone purge deletes by document id.

scripts/cron_tombstone_purge.py (0139, #593) finishes each batch of up to 200
deleted documents by removing every row that names them. Three of those
deletes had no index to use, so each batch scanned the tenant's whole share of
the table -- and the first attended run on the research plane (2026-09-25)
timed out at the 300 s client command_timeout on two tenants:

  * acl_snapshots by resource_id. The only index with resource_id leads with
    (customer_id, resource_type), and the purge does not know the type, so the
    delete merge-joined all ~372k of the tenant's rows. Now served by
    idx_acl_snapshots_resource: (customer_id, resource_id).
  * pending_edges by a Document endpoint's canonical id, on either side.
    idx_pending_edges_missing keys the MISSING endpoint; a row parked on the
    OTHER end (a Run -> Document edge waiting for its Run) is found only by
    from/to canonical id, so the delete read all ~517k tenant rows. Now two
    partial indexes, one per side, restricted to Document endpoints -- the
    only label the purge asks about:
    idx_pending_edges_from_document, idx_pending_edges_to_document.
  * inferred_edges_queue by anchor_doc_id. The only index with it is partial
    on `done_at IS NULL`, and nearly every row is done, so the delete read
    the whole table (every tenant) per batch. Now served by
    idx_inferred_edges_queue_anchor: (customer_id, anchor_doc_id).

The ACL query changed with them: its surviving-document check,
`x.doc_id = a.resource_id OR x.source_id = a.resource_id`, probed the trigram
index on documents.source_id for every matched row whether or not these
indexes exist. pending_edges is deleted one side per statement, each restating
its index's predicate. The script's docstrings say how.

CONCURRENTLY, in alembic's autocommit block, one index at a time -- 0139's
shape. This migration runs UNATTENDED on both planes (managed-migrate hook on
merge; engine-kb-migrate on the next research-os deploy), and these tables take
writes on every ingest. A plain CREATE INDEX holds a SHARE lock for a full scan
of each; CONCURRENTLY blocks no reads or writes.

Its known cost, from 0105/0139: CONCURRENTLY waits for every transaction older
than the build before marking the index valid, and a deploy hook with a timeout
can be killed in that wait, leaving an INVALID index behind. Each index is
guarded the way 0139 guards its one: an INVALID leftover of the same name is
dropped (CONCURRENTLY) and rebuilt, because `IF NOT EXISTS` matches on NAME and
would otherwise record the migration applied against a corpse the planner
ignores. No user-facing query needs these indexes: until they are valid only
the purge and the per-document deletes (kb/session_deletion.py,
engine/ingest/purge.py, kb/github_control_purge.py) are slower.

It runs at a different time on each plane: on merge for managed, and on the
next research-os deploy for research. Check for long-lived transactions
(`pg_stat_activity` ordered by `xact_start`) before EACH of those, and first
if a deploy stalls here. The indexes can also be built by
hand beforehand, after which this migration is a no-op -- the statements are
_INDEXES below, each run as `CREATE INDEX CONCURRENTLY IF NOT EXISTS`. Paste
them exactly: `IF NOT EXISTS` and the INVALID guard both match on NAME only, so
a valid hand-built index with another column list or predicate (`'document'`)
is kept, 0141 is stamped applied, and the purge keeps scanning. Check afterwards
that each `pg_get_indexdef(to_regclass('<name>'))` shows the columns and the
`'Document'` predicate below.
"""

import sqlalchemy as sa
from alembic import op

revision = "0141_tombstone_purge_side_idx"
down_revision = "0140_session_deletions"
branch_labels = None
depends_on = None

# (name, "ON table (...) [WHERE ...]"). db/schema.sql declares the same four.
_INDEXES: tuple[tuple[str, str], ...] = (
    (
        "idx_acl_snapshots_resource",
        "ON acl_snapshots (customer_id, resource_id)",
    ),
    (
        "idx_pending_edges_from_document",
        "ON pending_edges (customer_id, from_canonical_id) WHERE from_label = 'Document'",
    ),
    (
        "idx_pending_edges_to_document",
        "ON pending_edges (customer_id, to_canonical_id) WHERE to_label = 'Document'",
    ),
    (
        "idx_inferred_edges_queue_anchor",
        "ON inferred_edges_queue (customer_id, anchor_doc_id)",
    ),
)


def upgrade() -> None:
    # CREATE INDEX CONCURRENTLY cannot run inside a transaction; the autocommit
    # block commits alembic's transaction around these statements only.
    with op.get_context().autocommit_block():
        bind = op.get_bind()
        for name, definition in _INDEXES:
            # Drop an INVALID leftover FIRST (see the module docstring), and
            # CONCURRENTLY: a plain DROP INDEX takes ACCESS EXCLUSIVE on the
            # table, and one queued behind a long transaction stalls every
            # read queued behind it. A healthy index on a re-run is left alone.
            invalid = bind.execute(
                sa.text(
                    """
                    SELECT 1
                    FROM pg_class c
                    JOIN pg_index i ON i.indexrelid = c.oid
                    WHERE c.relname = :name AND NOT i.indisvalid
                    """
                ),
                {"name": name},
            ).first()
            if invalid is not None:
                op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {name}")
            op.execute(f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {name} {definition}")


def downgrade() -> None:
    with op.get_context().autocommit_block():
        for name, _definition in reversed(_INDEXES):
            op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {name}")
