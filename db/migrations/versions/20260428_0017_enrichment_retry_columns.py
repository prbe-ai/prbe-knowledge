"""enrichment_runs: add retry/heartbeat/payload columns

Revision ID: 0017_enrichment_retry_columns
Revises: 0016_acl_snapshots_unique
Create Date: 2026-04-28

prbe-orchestrator's retry+worker+DLQ pipeline (PR #1) and the Pydantic AI
enrichment agent (PR #4) both assume the following columns exist on
enrichment_runs. The orchestrator's startup precondition refuses to serve
without them, so this migration is a hard prereq for the next deploy.

Columns added:
  * heartbeat_at      TIMESTAMPTZ  — refreshed by _heartbeat task while a
                                     run is processing; lets the boot sweep
                                     identify rows abandoned by a crashed
                                     worker.
  * next_retry_at     TIMESTAMPTZ  — exponential-backoff gate; claim() refuses
                                     to pick up a row before this time.
  * payload           JSONB        — the original webhook envelope, so the
                                     worker can re-process after a crash
                                     without re-querying the source.
  * payload_version   SMALLINT     — schema version for `payload`. Bumping
                                     this lets the worker reject rows it
                                     doesn't know how to parse.
  * attempt_count     INTEGER      — incremented atomically by claim(); the
                                     optimistic lock in mark_*() compares
                                     against this.
  * started_at        TIMESTAMPTZ  — set on claim, cleared on transient retry.
  * finished_at       TIMESTAMPTZ  — set by mark_done / mark_skipped /
                                     on_error_terminal.
  * error_class       TEXT         — taxonomy slug from EnrichError (e.g.
                                     'agent_mcp_401', 'linear_429').
  * last_error        TEXT         — truncated error message, debug surface.

Status CHECK constraint is also extended with 'dlq' so on_error_terminal
can transition rows to the dead-letter state.

Indexes added:
  * (status, next_retry_at) WHERE status IN ('pending','failed') — backs the
    worker's scheduling poll.
  * (heartbeat_at) WHERE status = 'processing' — backs the boot-sweep + the
    stale-heartbeat reclaim path.

NULL-safety: existing columns NOT added with NOT NULL because there are
zero rows in prod today (verified 2026-04-28). For dev branches that may
have legacy rows, the defaults on attempt_count and payload_version mean
even pre-existing rows can be claimed without a backfill.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0017_enrichment_retry_columns"
down_revision = "0016_acl_snapshots_unique"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "enrichment_runs",
        sa.Column("heartbeat_at", postgresql.TIMESTAMP(timezone=True), nullable=True),
    )
    op.add_column(
        "enrichment_runs",
        sa.Column("next_retry_at", postgresql.TIMESTAMP(timezone=True), nullable=True),
    )
    op.add_column(
        "enrichment_runs",
        sa.Column("payload", postgresql.JSONB(), nullable=True),
    )
    op.add_column(
        "enrichment_runs",
        sa.Column(
            "payload_version",
            sa.SmallInteger(),
            nullable=False,
            server_default=sa.text("1"),
        ),
    )
    op.add_column(
        "enrichment_runs",
        sa.Column(
            "attempt_count",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
    )
    op.add_column(
        "enrichment_runs",
        sa.Column("started_at", postgresql.TIMESTAMP(timezone=True), nullable=True),
    )
    op.add_column(
        "enrichment_runs",
        sa.Column("finished_at", postgresql.TIMESTAMP(timezone=True), nullable=True),
    )
    op.add_column(
        "enrichment_runs",
        sa.Column("error_class", sa.Text(), nullable=True),
    )
    op.add_column(
        "enrichment_runs",
        sa.Column("last_error", sa.Text(), nullable=True),
    )

    op.execute(
        "ALTER TABLE enrichment_runs DROP CONSTRAINT IF EXISTS enrichment_runs_status_check"
    )
    op.execute(
        """
        ALTER TABLE enrichment_runs ADD CONSTRAINT enrichment_runs_status_check
        CHECK (status IN ('pending','processing','succeeded','failed','skipped','dlq'))
        """
    )

    op.execute(
        """
        CREATE INDEX IF NOT EXISTS enrichment_runs_claim_idx
            ON enrichment_runs (next_retry_at)
            WHERE status IN ('pending', 'failed')
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS enrichment_runs_heartbeat_idx
            ON enrichment_runs (heartbeat_at)
            WHERE status = 'processing'
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS enrichment_runs_heartbeat_idx")
    op.execute("DROP INDEX IF EXISTS enrichment_runs_claim_idx")

    op.execute(
        "ALTER TABLE enrichment_runs DROP CONSTRAINT IF EXISTS enrichment_runs_status_check"
    )
    op.execute(
        """
        ALTER TABLE enrichment_runs ADD CONSTRAINT enrichment_runs_status_check
        CHECK (status IN ('pending','processing','succeeded','failed','skipped'))
        """
    )

    op.drop_column("enrichment_runs", "last_error")
    op.drop_column("enrichment_runs", "error_class")
    op.drop_column("enrichment_runs", "finished_at")
    op.drop_column("enrichment_runs", "started_at")
    op.drop_column("enrichment_runs", "attempt_count")
    op.drop_column("enrichment_runs", "payload_version")
    op.drop_column("enrichment_runs", "payload")
    op.drop_column("enrichment_runs", "next_retry_at")
    op.drop_column("enrichment_runs", "heartbeat_at")
