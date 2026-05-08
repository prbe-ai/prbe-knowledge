"""enrichment_runs: Lane A fix-mode columns + indexes

Revision ID: 0057_fix_run_columns_a
Revises: 0056_documents_id_trgm_idx
Create Date: 2026-05-07

Lane A's auto-fix flow lives on the same enrichment_runs table as
ticket / dev enrichment. This migration extends the table with three
fix-specific columns and adds the supporting concurrency index that
fix_dispatcher.py needs to keep its tenant-cap check fast.

Schema changes:
  * `fly_machine_id text` - the Fly machine running the agent. Recovery
    on startup uses this to reattach or fail rows whose machine has gone
    away. Nullable because non-fix rows never set it.
  * `pr_urls jsonb DEFAULT '[]'::jsonb` - growing list of PR URLs the
    agent has opened during this run. recovery.rollback_partial_prs()
    iterates over this on partial-failure cleanup. Default empty array
    so callers can append unconditionally.
  * `repos_cloned jsonb DEFAULT '[]'::jsonb` - audit-style list of
    repos the agent fetched a clone token for. Per-tenant per-run
    visibility for security audits without going to the
    repo_clone_token_audit table.

Enum changes:
  * Extend `enrichment_runs_agent_kind_check` to allow 'fix' alongside
    the existing 'ticket' / 'debug' / 'dev' values. Postgres doesn't
    support ALTER on CHECK constraints in place; this is DROP + ADD.
    NOTE: the current check name and value set come from migration
    0029_enrichment_agent_kind_columns.

Index:
  * `idx_enr_runs_tnt_kind_status` partial composite on
    (tenant_id [customer_id], agent_kind, status) WHERE status IN
    ('processing', 'queued'). Covers fix_dispatcher's per-tenant
    in-flight count query and recovery's "find stale fix runs"
    sweep. CONCURRENTLY because enrichment_runs is hot per-tenant.

Lessons reminder: revision string MUST be <=32 chars (alembic_version
column is varchar(32)); '0057_fix_run_columns_a' is 22 chars - fine.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0057_fix_run_columns_a"
down_revision = "0056_documents_id_trgm_idx"
branch_labels = None
depends_on = None


CONCURRENCY_INDEX = "idx_enr_runs_tnt_kind_status"
AGENT_KIND_CHECK = "enrichment_runs_agent_kind_check"


def upgrade() -> None:
    # 1. New columns. server_default keeps the NOT NULL guarantee
    # backward-compatible for non-fix rows.
    op.add_column(
        "enrichment_runs",
        sa.Column("fly_machine_id", sa.Text(), nullable=True),
    )
    op.add_column(
        "enrichment_runs",
        sa.Column(
            "pr_urls",
            sa.dialects.postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
    )
    op.add_column(
        "enrichment_runs",
        sa.Column(
            "repos_cloned",
            sa.dialects.postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
    )
    # app_id ties the row to a specific GitHub App / workspace install -
    # required by the scoped JWT mint in fix_dispatcher. Nullable
    # because legacy enrichment rows never had one.
    op.add_column(
        "enrichment_runs",
        sa.Column("app_id", sa.Text(), nullable=True),
    )

    # 2. Extend the agent_kind CHECK constraint to allow 'fix'.
    # DROP + ADD because Postgres doesn't ALTER CHECKs in place.
    op.execute(
        f"ALTER TABLE enrichment_runs DROP CONSTRAINT IF EXISTS {AGENT_KIND_CHECK}"
    )
    op.execute(
        f"ALTER TABLE enrichment_runs ADD CONSTRAINT {AGENT_KIND_CHECK} "
        "CHECK (agent_kind IN ('ticket', 'debug', 'dev', 'fix'))"
    )

    # 3. Concurrency-cap composite index. CONCURRENTLY requires
    # commit-per-statement and disables the surrounding transaction.
    with op.get_context().autocommit_block():
        op.execute(
            f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {CONCURRENCY_INDEX} "
            "ON enrichment_runs (customer_id, agent_kind, status) "
            "WHERE status IN ('processing', 'queued')"
        )


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {CONCURRENCY_INDEX}")
    # Restore the prior agent_kind constraint shape.
    op.execute(
        f"ALTER TABLE enrichment_runs DROP CONSTRAINT IF EXISTS {AGENT_KIND_CHECK}"
    )
    op.execute(
        f"ALTER TABLE enrichment_runs ADD CONSTRAINT {AGENT_KIND_CHECK} "
        "CHECK (agent_kind IN ('ticket', 'debug', 'dev'))"
    )
    op.drop_column("enrichment_runs", "app_id")
    op.drop_column("enrichment_runs", "repos_cloned")
    op.drop_column("enrichment_runs", "pr_urls")
    op.drop_column("enrichment_runs", "fly_machine_id")
