"""chunks: add embedding_v2 columns for Gemini dual-write

Revision ID: 0060_add_embedding_v2_cols
Revises: 0059_agent_runs_model_col
Create Date: 2026-05-08

Stage 0 of the OpenAI -> Gemini embedding migration.

Adds three nullable columns alongside the existing `embedding` column:

    embedding_v2        halfvec(3072)  NULL  -- Gemini-2 vector
    embedding_v2_model  TEXT           NULL  -- e.g. 'google/gemini-embedding-2-preview'
    embedding_v2_dim    INT            NULL

No HNSW index yet -- the index gets built in a later migration once the
backfill has populated v2 across the full table. Building HNSW on a
mostly-NULL column wastes work; deferring keeps Stage 0 zero-risk.

Purely additive. Existing readers that don't know about embedding_v2 are
unaffected; existing writers continue writing only `embedding`. Stage 1
of the migration ships the dual-write code in the same deploy as this
revision so freshly-ingested chunks populate v2 immediately.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0060_add_embedding_v2_cols"
down_revision = "0059_agent_runs_model_col"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # halfvec is the pgvector type; SQLAlchemy doesn't know it natively.
    op.execute(
        """
        ALTER TABLE chunks
            ADD COLUMN IF NOT EXISTS embedding_v2 halfvec(3072) NULL,
            ADD COLUMN IF NOT EXISTS embedding_v2_model TEXT NULL,
            ADD COLUMN IF NOT EXISTS embedding_v2_dim INT NULL
        """
    )


def downgrade() -> None:
    op.execute(
        """
        ALTER TABLE chunks
            DROP COLUMN IF EXISTS embedding_v2,
            DROP COLUMN IF EXISTS embedding_v2_model,
            DROP COLUMN IF EXISTS embedding_v2_dim
        """
    )


# sa import retained for symmetry with neighbor migrations even though the
# raw-SQL path above is what actually runs.
_ = sa
