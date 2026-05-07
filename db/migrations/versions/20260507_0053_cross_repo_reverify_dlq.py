"""cross_repo_reverify_dlq table

Revision ID: 0053_cross_repo_reverify_dlq
Revises: 0052_codegraph_file_as_doc
Create Date: 2026-05-07

When a push webhook can't reach the Gemini classifier (down, rate-
limited, errored response), `update_edges_after_push` falls back to
"keep all evidence" so the system doesn't incorrectly delete edges
based on a missing verifier. The deferred verification work is
written here for a later retry pass (drained by the nightly cron).

Schema is intentionally lean: store enough metadata to re-run the
verification by re-fetching file contents from GitHub at the
recorded sha. We do NOT store the file contents themselves — at
retry time the file may have been superseded anyway, and storing
post-push contents would blow up table size.

Status state machine: pending → processing → done | failed.
The retry caps attempts so a permanently-broken token doesn't loop
forever.
"""

from __future__ import annotations

from alembic import op

revision = "0053_cross_repo_reverify_dlq"
down_revision = "0052_codegraph_file_as_document"
branch_labels = None
depends_on = None


def upgrade() -> None:
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
    # Drain query filters on (customer_id, status='pending') ordered by
    # enqueued_at; partial index keeps the scan tight as the done/failed
    # tail grows.
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_cross_repo_reverify_dlq_pending
            ON cross_repo_reverify_dlq (customer_id, enqueued_at)
            WHERE status = 'pending'
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS cross_repo_reverify_dlq")
