"""add target column + composite index to wiki_synthesis_runs

Revision ID: 0050_add_target_to_wsr
Revises: 0049_codegraph_phase_a
Create Date: 2026-05-06

Phase 2 fan-out: per-(customer, source) backfill rows can now also be
scoped to a single target (repo for GitHub, channel for Slack later).
Phase 1 rows have target=NULL; Phase 2 rows carry target='owner/repo'.

The orchestrator's post-Phase-1 hook in ``services/synthesis/backfill_app.py``
queries the source's BackfillFanout discoverer at done() and inserts
one row per target. Existing _claim_one() picks them up.

Adds a composite index covering the dashboard's per-target status
aggregation (filter on customer + kind + source + target with started_at
recency).

Downgrade drops the index + column. Existing rows are unaffected on
downgrade since the column is nullable.
"""

from __future__ import annotations

from alembic import op

revision = "0050_add_target_to_wsr"
down_revision = "0049_codegraph_phase_a"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE wiki_synthesis_runs ADD COLUMN IF NOT EXISTS target TEXT")
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_wsr_kind_source_target "
        "ON wiki_synthesis_runs (customer_id, kind, source, target, started_at DESC)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_wsr_kind_source_target")
    op.execute("ALTER TABLE wiki_synthesis_runs DROP COLUMN IF EXISTS target")
