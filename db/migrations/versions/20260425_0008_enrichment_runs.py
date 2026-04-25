"""enrichment_runs

Revision ID: 0008_enrichment_runs
Revises: 0007_dashboard_organization_link
Create Date: 2026-04-25

Track per-event enrichment work performed by prbe-orchestrator. Holds the
idempotency key (customer_id + source + source_event_id) so a webhook
redelivery can't trigger duplicate Linear comments, plus the comment id we
posted for audit/replay.

Lives in prbe-knowledge's alembic chain because knowledge owns the schema
of the shared Neon database — orchestrator just reads/writes through it.
"""

from __future__ import annotations

from alembic import op

revision = "0008_enrichment_runs"
down_revision = "0007_dashboard_organization_link"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE enrichment_runs (
            run_id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            customer_id       TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
            source            TEXT NOT NULL,
            source_event_id   TEXT NOT NULL,
            ticket_id         TEXT,
            status            TEXT NOT NULL DEFAULT 'pending',
            comment_id        TEXT,
            error             TEXT,
            created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CONSTRAINT enrichment_runs_status_check
                CHECK (status IN ('pending', 'processing', 'succeeded', 'failed', 'skipped'))
        );

        CREATE UNIQUE INDEX enrichment_runs_idempotency
            ON enrichment_runs (customer_id, source, source_event_id);

        CREATE INDEX enrichment_runs_status_created
            ON enrichment_runs (status, created_at);
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS enrichment_runs CASCADE;")
