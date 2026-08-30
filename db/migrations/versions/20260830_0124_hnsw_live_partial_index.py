"""A live-only HNSW index, so ANN walks stop visiting dead versions.

Only ~35% of chunks are live (`valid_to IS NULL`); the other 65% are
superseded versions the full index still carries -- 7.7GB of graph, growing
~580MB/day, where ~2.7GB would do. Every default-mode ANN query filters
`valid_to IS NULL` (TemporalMode.LATEST's chunk predicate), so
`hnsw.iterative_scan` widens through dead candidates it will always discard,
and the memory cliff (index vs cache) arrives ~3x sooner than it needs to.

The partial index serves that default path with no query change: the LATEST
predicate implies the index predicate, so the planner can pick it today.
Temporal modes (`ALL`, `AS_OF`) and the inferred-edges bundle keep using the
FULL index, which this migration deliberately does not drop -- the hot path
migrates, the full index goes cold in cache, and dropping it is a separate
decision once its remaining consumers are audited.

Plain CREATE INDEX, not CONCURRENTLY: 0105 (2026-08-16) lost two hours to a
CONCURRENTLY build wedged behind old transactions inside a deploy. On the
production database the index is expected to ALREADY EXIST -- built
attended via CIC on 2026-08-30 -- making this a no-op there; the plain form
is for fresh installs and self-hosts, whose chunk counts make the write
lock a non-event.
"""

from alembic import op

revision = "0124_hnsw_live_partial_index"
down_revision = "0123_pg_prewarm_extension"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Drop an INVALID leftover FIRST. `CREATE INDEX IF NOT EXISTS` matches on
    # NAME alone, and an interrupted CONCURRENTLY build -- the exact
    # production path this migration leans on -- leaves an INVALID index the
    # planner ignores. Without this guard the migration would no-op against
    # the corpse, record itself applied, and every LATEST-mode query would
    # keep walking the full index with nothing anywhere to say so (the
    # guardian's debris scan covers pg_search access methods only). Same
    # convention as 0062/0099/0105.
    op.execute(
        """
        DO $$
        DECLARE
            invalid_name text;
        BEGIN
            SELECT c.relname INTO invalid_name
            FROM pg_index i
            JOIN pg_class c ON c.oid = i.indexrelid
            WHERE c.relname = 'idx_chunks_embedding_v2_hnsw_live'
              AND NOT i.indisvalid;
            IF invalid_name IS NOT NULL THEN
                RAISE NOTICE 'dropping INVALID %', invalid_name;
                EXECUTE 'DROP INDEX ' || quote_ident(invalid_name);
            END IF;
        END $$
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_chunks_embedding_v2_hnsw_live "
        "ON chunks USING hnsw (embedding_v2 halfvec_cosine_ops) "
        "WHERE valid_to IS NULL"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_chunks_embedding_v2_hnsw_live")
