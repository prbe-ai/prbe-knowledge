"""enrichment_runs + ingestion_queue retry columns

Revision ID: 0011_enrichment_retry_columns
Revises: 0010_own_organization_tables
Create Date: 2026-04-26

Adds the columns required for the heartbeat + backoff + DLQ pattern in
both queue tables. enrichment_runs is owned by prbe-orchestrator's worker;
ingestion_queue by this repo's services/ingestion/worker.py. Same shape so
both workers can use the same backoff helper logic.

Why both at once: deferring backoff in either was rejected during eng
review (see prbe-orchestrator/docs/enrichment-retry-reconciler.md). A
single migration prevents schema drift between the two structurally
identical queue tables.

Numbered 0011 because 0010_own_organization_tables landed on
feature/coding-agent-ingestion-v2 first (also dated 2026-04-26).
"""

from __future__ import annotations

from alembic import op

revision = "0011_enrichment_retry_columns"
down_revision = "0010_own_organization_tables"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        -- enrichment_runs (orchestrator)
        ALTER TABLE enrichment_runs
          ADD COLUMN attempt_count   INT          NOT NULL DEFAULT 0,
          ADD COLUMN error_class     TEXT,
          ADD COLUMN last_error      TEXT,
          ADD COLUMN next_retry_at   TIMESTAMPTZ,
          ADD COLUMN heartbeat_at    TIMESTAMPTZ,
          ADD COLUMN started_at      TIMESTAMPTZ,
          ADD COLUMN finished_at     TIMESTAMPTZ,
          ADD COLUMN payload         JSONB        NOT NULL DEFAULT '{}'::jsonb,
          ADD COLUMN payload_version INT          NOT NULL DEFAULT 1;

        ALTER TABLE enrichment_runs
          DROP CONSTRAINT enrichment_runs_status_check,
          ADD CONSTRAINT enrichment_runs_status_check
            CHECK (status IN ('pending','processing','succeeded','failed','skipped','dlq'));

        CREATE INDEX enrichment_runs_due_idx
          ON enrichment_runs (next_retry_at)
          WHERE status IN ('pending','failed') AND next_retry_at IS NOT NULL;

        CREATE INDEX enrichment_runs_heartbeat_idx
          ON enrichment_runs (heartbeat_at)
          WHERE status = 'processing';

        -- ingestion_queue (knowledge worker)
        ALTER TABLE ingestion_queue
          ADD COLUMN next_retry_at TIMESTAMPTZ;

        CREATE INDEX ingestion_queue_due_idx
          ON ingestion_queue (next_retry_at)
          WHERE status = 'pending' AND next_retry_at IS NOT NULL;
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DROP INDEX IF EXISTS ingestion_queue_due_idx;
        ALTER TABLE ingestion_queue DROP COLUMN IF EXISTS next_retry_at;

        DROP INDEX IF EXISTS enrichment_runs_heartbeat_idx;
        DROP INDEX IF EXISTS enrichment_runs_due_idx;
        ALTER TABLE enrichment_runs
          DROP CONSTRAINT IF EXISTS enrichment_runs_status_check,
          ADD CONSTRAINT enrichment_runs_status_check
            CHECK (status IN ('pending','processing','succeeded','failed','skipped'));
        ALTER TABLE enrichment_runs
          DROP COLUMN IF EXISTS payload_version,
          DROP COLUMN IF EXISTS payload,
          DROP COLUMN IF EXISTS finished_at,
          DROP COLUMN IF EXISTS started_at,
          DROP COLUMN IF EXISTS heartbeat_at,
          DROP COLUMN IF EXISTS next_retry_at,
          DROP COLUMN IF EXISTS last_error,
          DROP COLUMN IF EXISTS error_class,
          DROP COLUMN IF EXISTS attempt_count;
        """
    )
