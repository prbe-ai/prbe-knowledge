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
    # held for the duration (chunks: ~444 MB heap, 90k rows; expect 3-7 min
    # in prod). Reads keep working through the cutover -- the existing GIN
    # index covers BM25 traffic until the new index is online and code cuts
    # over.
    op.execute(
        """
        ALTER TABLE chunks
        ADD COLUMN content_tsv tsvector
        GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
        """
    )

    # CREATE INDEX CONCURRENTLY cannot run inside a transaction. autocommit_block
    # breaks alembic's wrapper for this statement only. IF NOT EXISTS makes the
    # migration restart-safe if the build is interrupted (e.g. timeout, OOM).
    with op.get_context().autocommit_block():
        op.execute(
            """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_chunks_content_tsv
            ON chunks USING gin(content_tsv)
            """
        )


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS idx_chunks_content_tsv")
    op.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS content_tsv")
