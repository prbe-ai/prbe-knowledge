"""Merge two parallel heads on 0072_ingestion_cursors.

PR #273 (``0073_customers_r2_bucket``) and PR #269 (``0073_code_repo_state_cascade``)
both forked from ``0072_ingestion_cursors`` and landed on main, leaving
alembic with two heads. The first ``alembic upgrade head`` after both merged
crashes with "Multiple head revisions are present for given argument 'head'",
which is exactly what blocked the managed-postgres migrate-job at deploy time.

This migration is the alembic-standard fix: a no-op revision whose
``down_revision`` is the tuple of both heads. After applying, ``alembic
heads`` reports a single head again. No DDL — both feature migrations are
already applied independently when this runs.

See feedback_alembic_check_heads_post_rebase.md / feedback_alembic_merge_revalidate_before_merge.md
for why this keeps happening and how to avoid it next time.

Revision ID: 0074_merge_r2_cascade
Revises: 0073_customers_r2_bucket, 0073_code_repo_state_cascade
Create Date: 2026-05-15
"""

from __future__ import annotations

revision = "0074_merge_r2_cascade"
down_revision = ("0073_customers_r2_bucket", "0073_code_repo_state_cascade")
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
