"""drop cross_repo_reverify_dlq

Revision ID: 0054_drop_cross_repo_dlq
Revises: 0053_cross_repo_reverify_dlq
Create Date: 2026-05-07

The DLQ landed in 0053 to handle Gemini-unavailable cases on push
webhooks. Reverting that decision: webhooks now do removed-files-only
edge updates (no LLM call needed), so there's nothing to fail and
nothing to retry. The nightly cross-repo refresh is the canonical
LLM-using path; it has its own retry by virtue of running every night.

Downgrade re-creates the table with the same shape as 0053 in case
we change our minds.
"""

from __future__ import annotations

from alembic import op

revision = "0054_drop_cross_repo_dlq"
down_revision = "0053_cross_repo_reverify_dlq"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("DROP TABLE IF EXISTS cross_repo_reverify_dlq")


def downgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS cross_repo_reverify_dlq (
            dlq_id              BIGSERIAL PRIMARY KEY,
            customer_id         TEXT NOT NULL
                                REFERENCES customers(customer_id) ON DELETE CASCADE,
            source_repo         TEXT NOT NULL,
            sha                 TEXT NOT NULL,
            removed_files       JSONB NOT NULL DEFAULT '[]',
            modified_files      JSONB NOT NULL DEFAULT '[]',
            integration_token_id UUID,
            status              TEXT NOT NULL DEFAULT 'pending'
                                CHECK (status IN ('pending','processing','done','failed')),
            attempts            INT  NOT NULL DEFAULT 0,
            last_error          TEXT,
            last_attempt_at     TIMESTAMPTZ,
            enqueued_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_cross_repo_reverify_dlq_pending
            ON cross_repo_reverify_dlq (customer_id, enqueued_at)
            WHERE status = 'pending'
        """
    )
