"""enrichment_runs: agent_kind + subject_kind + paused status + by-subject index

Revision ID: 0029_enrichment_agent_kind_columns
Revises: 0028_backfill_cc_person_name_email
Create Date: 2026-04-30

Lands four schema changes on enrichment_runs in a single migration:

  * `agent_kind TEXT NOT NULL DEFAULT 'debug'` — discriminator for which
    agent produced the run. The orchestrator currently runs one agent
    kind ('debug', for ticket enrichment); a follow-up PR adds a second
    kind ('development', for PR review). Existing rows backfill to
    'debug' since they all came from the ticket enrichment path.
  * `subject_kind TEXT NOT NULL DEFAULT 'issue'` — discriminator for what
    was enriched. Today every row enriches an issue/ticket; the
    development agent's runs target a 'pr' subject. Default 'issue'
    faithfully reflects the historical population.
  * Replace the existing `enrichment_runs_status_check` CHECK constraint
    to include `'paused'` alongside the current values. Postgres doesn't
    support ALTER on CHECK constraints in place, so this is DROP + ADD.
    The current set (per migration 0017) is
    ('pending','processing','succeeded','failed','skipped','dlq');
    extending to ('pending','processing','succeeded','failed','skipped',
    'dlq','paused'). 'paused' is set by the orchestrator when a run is
    blocked on customer action (e.g. an integration token expired) and
    shouldn't be retried until the user resolves it.
  * `enrichment_runs_subject_idx` btree composite on
    (customer_id, source, ticket_id, finished_at DESC). Backs the
    upcoming "history for this subject" query path
    (`SELECT ... FROM enrichment_runs WHERE customer_id = $1 AND source = $2
    AND ticket_id = $3 ORDER BY finished_at DESC LIMIT $k`). Built
    CONCURRENTLY because enrichment_runs grows unboundedly per tenant
    and an ACCESS EXCLUSIVE lock would stall the orchestrator's worker
    drain on any tenant past a few hundred thousand runs.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0029_agent_kind_subject_kind"
down_revision = "0028_backfill_cc_person_props"
branch_labels = None
depends_on = None


SUBJECT_INDEX = "enrichment_runs_subject_idx"


def upgrade() -> None:
    # 1. Add the two discriminator columns. server_default backfills
    # existing rows to the historical values ('debug', 'issue'); the
    # NOT NULL constraint then holds for every row, old and new.
    op.add_column(
        "enrichment_runs",
        sa.Column(
            "agent_kind",
            sa.Text(),
            nullable=False,
            server_default=sa.text("'debug'"),
        ),
    )
    op.add_column(
        "enrichment_runs",
        sa.Column(
            "subject_kind",
            sa.Text(),
            nullable=False,
            server_default=sa.text("'issue'"),
        ),
    )

    # 2. Swap the status CHECK constraint to add 'paused'. Postgres
    # doesn't support ALTER on CHECK constraints in place, so DROP + ADD.
    # 'dlq' came in via 0017; keep it.
    op.execute(
        "ALTER TABLE enrichment_runs DROP CONSTRAINT IF EXISTS enrichment_runs_status_check"
    )
    op.execute(
        """
        ALTER TABLE enrichment_runs ADD CONSTRAINT enrichment_runs_status_check
        CHECK (status IN ('pending','processing','succeeded','failed','skipped','dlq','paused'))
        """
    )

    # 3. Composite index for the by-subject history lookup. CONCURRENTLY
    # can't run inside a transaction, so wrap in autocommit_block (same
    # pattern as 0015 / 0019 / 0022).
    with op.get_context().autocommit_block():
        op.create_index(
            SUBJECT_INDEX,
            "enrichment_runs",
            ["customer_id", "source", "ticket_id", "finished_at"],
            unique=False,
            postgresql_using="btree",
            postgresql_concurrently=True,
            postgresql_ops={"finished_at": "DESC"},
            if_not_exists=True,
        )


def downgrade() -> None:
    # 1. Drop the composite index first (CONCURRENTLY, autocommit block).
    with op.get_context().autocommit_block():
        op.drop_index(
            SUBJECT_INDEX,
            table_name="enrichment_runs",
            postgresql_concurrently=True,
            if_exists=True,
        )

    # 2. Restore the pre-paused CHECK constraint set. Mirrors the upgrade
    # set minus 'paused' — i.e. what 0017 left behind.
    op.execute(
        "ALTER TABLE enrichment_runs DROP CONSTRAINT IF EXISTS enrichment_runs_status_check"
    )
    op.execute(
        """
        ALTER TABLE enrichment_runs ADD CONSTRAINT enrichment_runs_status_check
        CHECK (status IN ('pending','processing','succeeded','failed','skipped','dlq'))
        """
    )

    # 3. Drop the two discriminator columns.
    op.drop_column("enrichment_runs", "subject_kind")
    op.drop_column("enrichment_runs", "agent_kind")
