"""backfill progress columns.

Revision ID: 0004_backfill_progress
Revises: 0003_customer_source_mapping
Create Date: 2026-04-23

Adds `events_enqueued` counter + `heartbeat_at` to `backfill_state` so the
backfill drain loop can emit progress (status endpoint) and so stuck-job
reclaim has a heartbeat to compare against.
"""

from __future__ import annotations

from alembic import op

revision = "0004_backfill_progress"
down_revision = "0003_customer_source_mapping"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Idempotent: schema.sql already adds these columns on fresh provision.
    op.execute(
        "ALTER TABLE backfill_state ADD COLUMN IF NOT EXISTS events_enqueued INT NOT NULL DEFAULT 0"
    )
    op.execute(
        "ALTER TABLE backfill_state ADD COLUMN IF NOT EXISTS heartbeat_at TIMESTAMPTZ"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_backfill_state_pending "
        "ON backfill_state (status, started_at) WHERE status = 'pending'"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_backfill_state_running "
        "ON backfill_state (status, heartbeat_at) WHERE status = 'running'"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_backfill_state_pending")
    op.execute("DROP INDEX IF EXISTS idx_backfill_state_running")
    op.execute("ALTER TABLE backfill_state DROP COLUMN IF EXISTS heartbeat_at")
    op.execute("ALTER TABLE backfill_state DROP COLUMN IF EXISTS events_enqueued")
