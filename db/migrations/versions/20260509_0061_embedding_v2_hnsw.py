"""chunks: HNSW index on embedding_v2 (Stage 3 of Gemini migration)

Revision ID: 0061_embedding_v2_hnsw
Revises: 0060_add_embedding_v2_cols
Create Date: 2026-05-09

Stage 3 of the OpenAI -> Gemini embedding migration. Builds the HNSW
index over the embedding_v2 column so the Stage 4 cutover (vector.py
+ helpers.py + dedup.py + synthesis_worker.py + inferred_edges/bundle.py
all switching reads to embedding_v2) gets the same index-backed search
quality the legacy embedding column already has.

DO NOT MERGE until Stage 2 backfill (scripts/backfill_embedding_v2.py)
reports zero remaining NULL chunks. Building HNSW over a NULL-heavy
column wastes work; the partial-index alternative is more code for
marginal benefit.

Verify before merge::

    SELECT COUNT(*) FROM chunks WHERE embedding_v2 IS NULL;
    -- must return 0

CONCURRENTLY: chunks is hot (ingest is constantly writing). ACCESS
EXCLUSIVE for a regular CREATE INDEX would block all writes for the
duration of the build, which on prod-scale chunk counts is tens of
minutes. Concurrent build keeps writes flowing at the cost of a slower
build wall-clock and a second internal scan.

m=16, ef_construction=64 are pgvector's defaults and match the existing
idx_chunks_embedding_hnsw (see db/schema.sql line ~203). Keeping them
identical so any retrieval tuning that lands on the v1 index translates
1:1 to the v2 path after cutover.
"""

from __future__ import annotations

from alembic import op

revision = "0061_embedding_v2_hnsw"
down_revision = "0060_add_embedding_v2_cols"
branch_labels = None
depends_on = None


INDEX_NAME = "idx_chunks_embedding_v2_hnsw"


def upgrade() -> None:
    # autocommit_block: CREATE INDEX CONCURRENTLY can't run inside a
    # transaction. Same pattern as 0056_documents_id_trgm_idx and
    # 0019_graph_nodes_loose_match_indexes.
    with op.get_context().autocommit_block():
        op.execute(
            f"""
            CREATE INDEX CONCURRENTLY IF NOT EXISTS {INDEX_NAME}
                ON chunks USING hnsw (embedding_v2 halfvec_cosine_ops)
                WITH (m = 16, ef_construction = 64)
            """
        )


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {INDEX_NAME}")
