"""wiki_synthesis_queue + wiki_synthesis_runs

Revision ID: 0033_wiki_synthesis
Revises: 0032_manual_uploads
Create Date: 2026-05-03

Two tables for the Phase 2 LLM-Wiki synthesis loop.

`wiki_synthesis_queue` is the per-document staging table the cron drains.
`Normalizer._persist` appends one row per persisted document (skipping
source_system='wiki' so the cron's own writes don't loop back into it).
A `pg_notify('wiki_synthesize', customer_id)` follows the insert; the
cron LISTEN-er wakes within seconds and drains the row through Haiku
triage + Sonnet synthesis. Triage decisions land back on the same row
(triage_score / triage_targets / status), so the queue is also the
provenance log: which docs influenced which wiki pages.

`wiki_synthesis_runs` is the audit row per cron tick. Onboarding,
notify-driven wakes, and scheduled defensive ticks all open one of
these. The dashboard polls it for "wiki being generated, X / Y events"
progress (separate session).

RLS via `app.current_customer_id` GUC, mirroring usage_events (0020).
Both tables get FORCE so application code can't bypass the policy.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0033_wiki_synthesis"
down_revision = "0032_manual_uploads"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "wiki_synthesis_queue",
        sa.Column(
            "queue_id",
            sa.BigInteger(),
            primary_key=True,
            autoincrement=True,
            nullable=False,
        ),
        sa.Column(
            "customer_id",
            sa.Text(),
            sa.ForeignKey("customers.customer_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("doc_id", sa.Text(), nullable=False),
        sa.Column("doc_version", sa.Integer(), nullable=False),
        sa.Column("source_system", sa.Text(), nullable=False),
        sa.Column("doc_type", sa.Text(), nullable=False),
        sa.Column(
            "enqueued_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.Text(),
            server_default=sa.text("'pending'"),
            nullable=False,
        ),
        sa.Column("triage_score", sa.Float(), nullable=True),
        sa.Column(
            "triage_targets",
            postgresql.JSONB(),
            nullable=True,
        ),
        sa.Column("triage_error", sa.Text(), nullable=True),
        sa.Column(
            "triage_completed_at",
            postgresql.TIMESTAMP(timezone=True),
            nullable=True,
        ),
        sa.Column("attempts", sa.Integer(), server_default="0", nullable=False),
        sa.Column("synthesis_run_id", sa.BigInteger(), nullable=True),
        sa.Column(
            "synthesis_completed_at",
            postgresql.TIMESTAMP(timezone=True),
            nullable=True,
        ),
        sa.Column("synthesis_error", sa.Text(), nullable=True),
        sa.UniqueConstraint(
            "customer_id",
            "doc_id",
            "doc_version",
            name="uq_wsq_customer_doc_version",
        ),
        sa.CheckConstraint(
            "status IN ('pending','triaging','triaged','rejected',"
            "'synthesizing','done','failed')",
            name="ck_wsq_status",
        ),
    )

    op.create_index(
        "idx_wsq_drain",
        "wiki_synthesis_queue",
        ["customer_id", "status", "enqueued_at"],
    )

    op.execute("ALTER TABLE wiki_synthesis_queue ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE wiki_synthesis_queue FORCE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY wiki_synthesis_queue_tenant_isolation ON wiki_synthesis_queue
            USING (customer_id = current_setting('app.current_customer_id', true))
        """
    )

    op.create_table(
        "wiki_synthesis_runs",
        sa.Column(
            "run_id",
            sa.BigInteger(),
            primary_key=True,
            autoincrement=True,
            nullable=False,
        ),
        sa.Column(
            "customer_id",
            sa.Text(),
            sa.ForeignKey("customers.customer_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column(
            "started_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.Column(
            "finished_at",
            postgresql.TIMESTAMP(timezone=True),
            nullable=True,
        ),
        sa.Column(
            "events_total", sa.Integer(), server_default="0", nullable=False
        ),
        sa.Column(
            "events_triaged", sa.Integer(), server_default="0", nullable=False
        ),
        sa.Column(
            "events_kept", sa.Integer(), server_default="0", nullable=False
        ),
        sa.Column(
            "pages_updated", sa.Integer(), server_default="0", nullable=False
        ),
        sa.Column(
            "pages_created", sa.Integer(), server_default="0", nullable=False
        ),
        sa.Column(
            "status",
            sa.Text(),
            server_default=sa.text("'running'"),
            nullable=False,
        ),
        sa.Column("error", sa.Text(), nullable=True),
        sa.CheckConstraint(
            "kind IN ('onboarding','wake','scheduled')", name="ck_wsr_kind"
        ),
        sa.CheckConstraint(
            "status IN ('running','complete','failed','partial')",
            name="ck_wsr_status",
        ),
    )

    op.create_index(
        "idx_wsr_customer",
        "wiki_synthesis_runs",
        ["customer_id", sa.text("started_at DESC")],
    )

    op.execute("ALTER TABLE wiki_synthesis_runs ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE wiki_synthesis_runs FORCE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY wiki_synthesis_runs_tenant_isolation ON wiki_synthesis_runs
            USING (customer_id = current_setting('app.current_customer_id', true))
        """
    )


def downgrade() -> None:
    op.execute(
        "DROP POLICY IF EXISTS wiki_synthesis_runs_tenant_isolation "
        "ON wiki_synthesis_runs"
    )
    op.drop_index("idx_wsr_customer", table_name="wiki_synthesis_runs")
    op.drop_table("wiki_synthesis_runs")

    op.execute(
        "DROP POLICY IF EXISTS wiki_synthesis_queue_tenant_isolation "
        "ON wiki_synthesis_queue"
    )
    op.drop_index("idx_wsq_drain", table_name="wiki_synthesis_queue")
    op.drop_table("wiki_synthesis_queue")
