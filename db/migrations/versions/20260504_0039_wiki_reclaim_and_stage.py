"""wiki_synthesis_queue heartbeat + wiki_synthesis_runs stage discriminator

Revision ID: 0039_wiki_reclaim_and_stage
Revises: 0038_backfill_prefs_off
Create Date: 2026-05-04

Two adds for the wiki-worker / wiki-synthesis fly app split:

1. `wiki_synthesis_queue.heartbeat_at` — stamped at claim time so the
   reclaim loop can detect rows wedged at status='triaging' or
   'synthesizing' after a worker SIGKILL/OOM. The reclaim sweep
   resets stale rows back to their prior state ('pending'/'triaged')
   if attempts < WIKI_SYNTHESIS_MAX_ATTEMPTS, else terminal 'failed' so
   ops can investigate via the dashboard. Mirrors the
   ingestion_queue.heartbeat_at + ReclaimLoop pattern.

2. `wiki_synthesis_runs.stage` — discriminator distinguishing the two
   workers' run rows. Triage and synthesis each open their own run
   per drain, so the existing `kind` column ('wake'/'scheduled'/
   'onboarding') doesn't tell the status endpoint which row to pick
   for `last_run_pages_*`. Adding stage='triage'|'synthesis' lets the
   query filter cleanly. Default 'synthesis' for backfill is a
   conservative choice — historic rows pre-split mostly completed
   the synthesize half.
"""

from __future__ import annotations

from alembic import op

revision = "0039_wiki_reclaim_and_stage"
down_revision = "0038_backfill_prefs_off"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ---- wiki_synthesis_queue.heartbeat_at -----------------------------
    op.execute(
        """
        ALTER TABLE wiki_synthesis_queue
            ADD COLUMN IF NOT EXISTS heartbeat_at TIMESTAMPTZ
        """
    )
    # Partial index — reclaim only ever scans rows in 'triaging' or
    # 'synthesizing'. Skips the long tail of 'done'/'rejected' rows.
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_wsq_heartbeat_reclaim
        ON wiki_synthesis_queue (heartbeat_at)
        WHERE status IN ('triaging', 'synthesizing')
        """
    )

    # ---- wiki_synthesis_runs.stage -------------------------------------
    op.execute(
        """
        ALTER TABLE wiki_synthesis_runs
            ADD COLUMN IF NOT EXISTS stage TEXT
                NOT NULL DEFAULT 'synthesis'
        """
    )
    op.execute(
        """
        ALTER TABLE wiki_synthesis_runs
            ADD CONSTRAINT ck_wsr_stage CHECK (stage IN ('triage', 'synthesis'))
        """
    )
    # The status endpoint reads the latest synthesis-stage run per
    # customer for pages_* counts; index the access pattern.
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_wsr_stage_started
        ON wiki_synthesis_runs (customer_id, stage, started_at DESC)
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_wsr_stage_started")
    op.execute(
        "ALTER TABLE wiki_synthesis_runs DROP CONSTRAINT IF EXISTS ck_wsr_stage"
    )
    op.execute(
        "ALTER TABLE wiki_synthesis_runs DROP COLUMN IF EXISTS stage"
    )
    op.execute("DROP INDEX IF EXISTS idx_wsq_heartbeat_reclaim")
    op.execute(
        "ALTER TABLE wiki_synthesis_queue DROP COLUMN IF EXISTS heartbeat_at"
    )
