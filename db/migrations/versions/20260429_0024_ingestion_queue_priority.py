"""ingestion_queue priority column

Revision ID: 0024_queue_priority
Revises: 0023_customer_prefs
Create Date: 2026-04-29

Adds a `priority` SMALLINT to `ingestion_queue` so the worker can claim
high-priority (live webhook) rows ahead of low-priority (backfill) rows.
Without this, a single customer's install-time backfill can head-of-line
block live webhooks for hours under Tier 1's 1M TPM ceiling.

- Default priority is 100 (live webhooks).
- `backfill_runner` inserts at 50 so backfills drain in the gaps.

Index swap: the existing `idx_queue_pending` covers (status, enqueued_at)
WHERE status='pending'. The claim ORDER BY now sorts by (priority DESC,
enqueued_at), so we replace it with `idx_queue_pending_priority` covering
the new ordering. Same row count, same WHERE clause — pure swap.
"""

from __future__ import annotations

from alembic import op

revision = "0024_queue_priority"
down_revision = "0023_customer_prefs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE ingestion_queue
        ADD COLUMN IF NOT EXISTS priority SMALLINT NOT NULL DEFAULT 100
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_queue_pending_priority
        ON ingestion_queue (priority DESC, enqueued_at)
        WHERE status = 'pending'
        """
    )
    op.execute("DROP INDEX IF EXISTS idx_queue_pending")


def downgrade() -> None:
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_queue_pending
        ON ingestion_queue (status, enqueued_at)
        WHERE status = 'pending'
        """
    )
    op.execute("DROP INDEX IF EXISTS idx_queue_pending_priority")
    op.execute("ALTER TABLE ingestion_queue DROP COLUMN IF EXISTS priority")
