"""enrichment_runs: add report + token usage columns

Revision ID: 0014_enrichment_report_columns
Revises: 0013_merge_heads
Create Date: 2026-04-27

prbe-orchestrator's new ticket-enrichment agent produces a structured
EnrichmentReport (Pydantic model serialized to JSON) and tracks Sonnet
token usage per run. Persist both on the existing enrichment_runs row so
the dashboard and replay tooling can read the structured output without
having to re-run the agent.

All four columns are NULLable: rows written by PR #1's worker (the
pre-agent code path) and any future runs that fail before the agent
completes will simply leave them NULL. Purely additive — no rewrites of
existing data, no constraints that could fail on backfill, no risk to
running services that haven't been upgraded yet.

report_schema_version is a forward-compat hook: current value is 1,
written by the orchestrator on every successful agent run. Future
schema changes to the EnrichmentReport shape can bump this without
needing another migration.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0014_enrichment_report_columns"
down_revision = "0013_merge_heads"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "enrichment_runs",
        sa.Column("report", postgresql.JSONB(), nullable=True),
    )
    op.add_column(
        "enrichment_runs",
        sa.Column("report_schema_version", sa.SmallInteger(), nullable=True),
    )
    op.add_column(
        "enrichment_runs",
        sa.Column("token_usage_input", sa.Integer(), nullable=True),
    )
    op.add_column(
        "enrichment_runs",
        sa.Column("token_usage_output", sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("enrichment_runs", "token_usage_output")
    op.drop_column("enrichment_runs", "token_usage_input")
    op.drop_column("enrichment_runs", "report_schema_version")
    op.drop_column("enrichment_runs", "report")
