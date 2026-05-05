"""wiki v4: drop triage_targets, add source_ts, expand status with dlq

Revision ID: 0041_wiki_v4_agent_loop
Revises: 0040_backfill_cc_doc_titles
Create Date: 2026-05-05

Five linked changes that flip the wiki pipeline from procedural triage +
verifier + synthesize into a single Gemini 3.1 Pro agent loop:

1. Drop `triage_targets` — the cheap model no longer picks (wiki_type,
   slug). The agent does that downstream by reading the day in time
   order, so the column has nothing to store.

2. Add `source_ts` — populated at insert by Normalizer from the source
   event's own timestamp (Slack ts, GitHub created_at, Linear updatedAt,
   Granola startedAt, Notion last_edited_time, falling back to
   documents.created_at). The agent reads events ordered by source_ts
   ASC so a 09:00 Slack flap and the 14:00 Notion postmortem doc that
   resolves it both flow into the same wiki page edit.

3. Add `dlq_reason` + `dlq_at` — DLQ is the universal "needs attention"
   surface in v4. Triage batch failures, agent halts (turn cap, stall,
   compactor crash, Gemini outage) all park the customer's rows here
   with a structured reason. Admin reset (POST .../dlq/reset) flips
   them back to pending or triaged depending on which stage they were
   in.

4. Status CHECK update — rename `verifier_rejected` -> `synthesis_skipped`
   (the agent's `skip_events` tool now produces this state too, not just
   the verifier), add `dlq`. Drop the old verifier-shaped name from the
   accepted set. The downgrade reverses both renames so v3 binaries can
   roll back.

5. Composite index `ix_wsq_drain_cursor` on
   (customer_id, status, source_ts, queue_id) — supports next_events()
   pagination cursor lookups in the agent, which page through triaged
   rows ordered by source_ts ASC, queue_id ASC.

Backfill: existing rows get source_ts = documents.created_at. Close
enough for one drain's worth of staleness; new inserts populate from
extracted source metadata. After the backfill, the column is set NOT
NULL so future code can rely on it without coalesce checks.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0041_wiki_v4_agent_loop"
down_revision = "0040_backfill_cc_doc_titles"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Drop the column the cheap model used to fill in.
    op.execute(
        "ALTER TABLE wiki_synthesis_queue DROP COLUMN IF EXISTS triage_targets"
    )

    # 2. Add source_ts. Backfill from documents.created_at, then NOT NULL.
    op.execute(
        "ALTER TABLE wiki_synthesis_queue "
        "ADD COLUMN IF NOT EXISTS source_ts TIMESTAMPTZ"
    )
    # Customer-scoped UPDATE; documents PK includes customer_id so the
    # join must include it.
    op.execute(
        """
        UPDATE wiki_synthesis_queue q
        SET source_ts = d.created_at
        FROM documents d
        WHERE d.customer_id = q.customer_id
          AND d.doc_id = q.doc_id
          AND d.version = q.doc_version
          AND q.source_ts IS NULL
        """
    )
    # Any leftover row (doc deleted, etc.) gets enqueued_at as a final
    # fallback so the NOT NULL transition succeeds.
    op.execute(
        "UPDATE wiki_synthesis_queue "
        "SET source_ts = enqueued_at WHERE source_ts IS NULL"
    )
    op.alter_column("wiki_synthesis_queue", "source_ts", nullable=False)

    # 3. DLQ tracking columns.
    op.execute(
        "ALTER TABLE wiki_synthesis_queue "
        "ADD COLUMN IF NOT EXISTS dlq_reason TEXT"
    )
    op.execute(
        "ALTER TABLE wiki_synthesis_queue "
        "ADD COLUMN IF NOT EXISTS dlq_at TIMESTAMPTZ"
    )

    # 4. Migrate verifier_rejected rows, then expand the CHECK constraint.
    op.execute(
        "UPDATE wiki_synthesis_queue "
        "SET status = 'synthesis_skipped' WHERE status = 'verifier_rejected'"
    )
    op.execute(
        "ALTER TABLE wiki_synthesis_queue DROP CONSTRAINT IF EXISTS ck_wsq_status"
    )
    op.execute(
        """
        ALTER TABLE wiki_synthesis_queue ADD CONSTRAINT ck_wsq_status CHECK (
            status IN ('pending','triaging','triaged','rejected',
                       'synthesizing','done','failed',
                       'synthesis_skipped','dlq')
        )
        """
    )

    # 5. Composite index for next_events() pagination.
    op.create_index(
        "ix_wsq_drain_cursor",
        "wiki_synthesis_queue",
        ["customer_id", "status", "source_ts", "queue_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_wsq_drain_cursor", table_name="wiki_synthesis_queue")
    op.execute(
        "ALTER TABLE wiki_synthesis_queue DROP CONSTRAINT IF EXISTS ck_wsq_status"
    )
    # Map states back into the v3 accepted set.
    op.execute(
        "UPDATE wiki_synthesis_queue "
        "SET status = 'verifier_rejected' WHERE status = 'synthesis_skipped'"
    )
    # 'dlq' rows have no clean v3 home; park them as 'failed' so the
    # constraint passes. Append the dlq_reason into synthesis_error so
    # the audit trail isn't lost.
    op.execute(
        """
        UPDATE wiki_synthesis_queue
        SET status = 'failed',
            synthesis_error = COALESCE(synthesis_error, '')
                              || CASE
                                   WHEN dlq_reason IS NULL THEN ''
                                   ELSE ' | downgraded from dlq: ' || dlq_reason
                                 END
        WHERE status = 'dlq'
        """
    )
    op.execute(
        """
        ALTER TABLE wiki_synthesis_queue ADD CONSTRAINT ck_wsq_status CHECK (
            status IN ('pending','triaging','triaged','rejected',
                       'synthesizing','done','failed','verifier_rejected')
        )
        """
    )
    op.execute(
        "ALTER TABLE wiki_synthesis_queue DROP COLUMN IF EXISTS dlq_at"
    )
    op.execute(
        "ALTER TABLE wiki_synthesis_queue DROP COLUMN IF EXISTS dlq_reason"
    )
    op.execute(
        "ALTER TABLE wiki_synthesis_queue DROP COLUMN IF EXISTS source_ts"
    )
    # Re-add the dropped column. JSONB matches the v3 shape; existing
    # rows get NULL.
    op.add_column(
        "wiki_synthesis_queue",
        sa.Column("triage_targets", sa.dialects.postgresql.JSONB(), nullable=True),
    )
