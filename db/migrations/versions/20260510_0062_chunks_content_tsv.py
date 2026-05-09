"""chunks_content_tsv: stored to_tsvector + GIN to kill BM25 re-tokenization

Revision ID: 0062_chunks_content_tsv
Revises: 0061_directed_vectors
Create Date: 2026-05-10

Materialize to_tsvector('english', content) as a STORED generated column on
chunks so BM25 stops re-tokenizing on every query. EXPLAIN ANALYZE on a
representative production query against probe-founders showed ~5.7s of a
5.9s BM25 query was per-row tokenization in the Bitmap Heap Scan recheck +
ts_rank_cd. The expression-based index `idx_chunks_fts_content` finds
candidates fast (~110ms) but tokenization is still done per row at recheck
time because the tsvector isn't on the heap.

This migration is the EXPAND half of an expand/contract:
  - Adds `content_tsv tsvector GENERATED ALWAYS AS (...) STORED`.
  - Builds new GIN index `idx_chunks_content_tsv` CONCURRENTLY.
  - Leaves the old `idx_chunks_fts_content` IN PLACE so old code instances
    running during the rolling deploy still hit a real index.

A follow-up migration drops the old index after the deploy is verified.

Lessons reminder: revision string MUST be <=32 chars (alembic_version
column is varchar(32)); '0062_chunks_content_tsv' is 23 chars - fine.
"""

from __future__ import annotations

from alembic import op


revision = "0062_chunks_content_tsv"
down_revision = "0061_directed_vectors"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ADD COLUMN GENERATED triggers a full table rewrite. AccessExclusiveLock
    # is held on `chunks` for the duration (chunks: ~444 MB heap, 90k rows;
    # expect 3-7 min in prod). Reads on `chunks` block for that window --
    # BM25, vector, dedup, and ingestion all queue. Other tables unaffected.
    # The follow-up cleanup PR's DROP INDEX is metadata-only and does not
    # repeat this stall.
    #
    # IF NOT EXISTS only helps the narrow case where ADD COLUMN already
    # committed and CREATE INDEX CONCURRENTLY (below) failed afterward; a
    # mid-rewrite kill rolls back via alembic's transaction wrapper and the
    # next run re-rewrites from scratch.
    op.execute(
        """
        ALTER TABLE chunks
        ADD COLUMN IF NOT EXISTS content_tsv tsvector
        GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
        """
    )

    # CREATE INDEX CONCURRENTLY cannot run inside a transaction. autocommit_block
    # breaks alembic's wrapper for these statements only.
    with op.get_context().autocommit_block():
        # Recover from a prior partial run: a CREATE INDEX CONCURRENTLY that
        # was interrupted (timeout, OOM, killed by Fly's release-command
        # deadline) leaves an INVALID index on disk. Plain `IF NOT EXISTS`
        # below would skip by name alone and never rebuild, leaving the
        # INVALID index in place -- queries hit it but the planner ignores
        # it for read serving, so BM25 silently falls back to seq scan.
        # Drop any INVALID-marked instance first so the recreate below
        # actually rebuilds.
        op.execute(
            """
            DO $$
            BEGIN
                IF EXISTS (
                    SELECT 1
                    FROM pg_class c
                    JOIN pg_index i ON i.indexrelid = c.oid
                    WHERE c.relname = 'idx_chunks_content_tsv'
                      AND NOT i.indisvalid
                ) THEN
                    EXECUTE 'DROP INDEX idx_chunks_content_tsv';
                END IF;
            END
            $$;
            """
        )
        op.execute(
            """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_chunks_content_tsv
            ON chunks USING gin(content_tsv)
            """
        )


def downgrade() -> None:
    # WARNING: this downgrade hard-breaks BM25 retrieval the moment DROP
    # COLUMN runs. Any retrieval pod still serving the new code references
    # `c.content_tsv` and will start raising `column "content_tsv" does
    # not exist` until the pod is restarted on the rolled-back binary.
    # Recovery sequence in an emergency:
    #   1. Roll back the retrieval / mcp / cron app images first.
    #   2. Confirm no live process references content_tsv.
    #   3. Then run alembic downgrade.
    with op.get_context().autocommit_block():
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS idx_chunks_content_tsv")
    op.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS content_tsv")
