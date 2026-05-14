"""chunks: drop NOT NULL on v1 embedding columns + drop v1 HNSW index

Revision ID: 0071_chunks_embed_v1_nullable
Revises: 0070_gnp_rls
Create Date: 2026-05-14

OpenAI -> Gemini embedding cutover (Stage 4, final).

After this migration:
- `chunks.embedding`, `chunks.embedding_model`, `chunks.embedding_dim` are
  NULLABLE. New rows written by the post-cutover normalizer leave them
  NULL. Existing rows (the handful of legacy OpenAI vectors) are
  preserved exactly as they were — no data is read from these columns by
  any production code path post-cutover, but the eval harness can still
  reference them for apples-to-apples retrieval comparisons.
- `idx_chunks_embedding_hnsw` (HNSW over the v1 column) is DROPPED. No
  production query traverses it; keeping it just costs write
  amplification on every chunk insert/update. `idx_chunks_embedding_v2_hnsw`
  (built in migration 0063) is what every retrieval now uses.

DROP INDEX CONCURRENTLY can't run inside a transaction; the autocommit
block mirrors 0063_embedding_v2_hnsw's pattern.

Downgrade NOTE: restoring NOT NULL would require backfilling NULL rows
with a placeholder vector (post-cutover rows all have NULL here). The
downgrade re-creates the index but does NOT re-impose NOT NULL — the
constraint can't be safely re-added without a data migration that no
caller would want to run. If you genuinely need to roll back the
cutover, do it at the application layer (revert the code change) rather
than at the schema layer.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0071_chunks_embed_v1_nullable"
down_revision = "0070_gnp_rls"
branch_labels = None
depends_on = None


V1_INDEX_NAME = "idx_chunks_embedding_hnsw"


def upgrade() -> None:
    # Phase 1: drop NOT NULL AND the DEFAULT clauses inside a regular
    # transaction. Keeping the defaults would inject misleading metadata on
    # post-cutover INSERTs — every new row would land with
    # embedding_model='openai/text-embedding-3-large' and embedding_dim=3072
    # despite embedding being NULL. Drop them so an unset value is clearly
    # absent.
    op.execute(
        """
        ALTER TABLE chunks
            ALTER COLUMN embedding DROP NOT NULL,
            ALTER COLUMN embedding_model DROP NOT NULL,
            ALTER COLUMN embedding_model DROP DEFAULT,
            ALTER COLUMN embedding_dim DROP NOT NULL,
            ALTER COLUMN embedding_dim DROP DEFAULT
        """
    )

    # Phase 2: drop the v1 HNSW index. CONCURRENTLY so writes keep flowing
    # on `chunks` during the drop; mirrors 0063's autocommit_block pattern.
    with op.get_context().autocommit_block():
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {V1_INDEX_NAME}")


def downgrade() -> None:
    # Rebuild the v1 HNSW index. Won't apply cleanly if any post-cutover
    # rows exist with embedding IS NULL — pgvector HNSW skips NULLs at
    # build time (the rebuild still succeeds; those rows are just missing
    # from the index). Use the same parameters as the original
    # idx_chunks_embedding_hnsw definition in schema.sql.
    with op.get_context().autocommit_block():
        op.execute(
            f"""
            CREATE INDEX CONCURRENTLY IF NOT EXISTS {V1_INDEX_NAME}
                ON chunks USING hnsw (embedding halfvec_cosine_ops)
            """
        )
    # Deliberately NOT restoring NOT NULL — see module docstring.


_ = sa  # retained for symmetry with neighbor migrations
