"""Merge two sibling 0073 heads created by concurrent PRs.

Both heads descend from 0072_ingestion_cursors:
  * 0073_code_repo_state_cascade (PR #269 — ON DELETE CASCADE FK on
    code_repo_state.customer_id, fixes the tenant-teardown orphan bug)
  * 0073_customers_r2_bucket     (PR #273 — customers.r2_bucket column
    + backfill for the prbe-<slug> bucket-naming rollout)

Both PRs were rebased on origin/main at author-time but merged within
seconds of each other without re-checking ``alembic heads``, so main
ended up with two heads. The managed-postgres-0 migrate-job (Helm
post-install,post-upgrade hook) refused ``alembic upgrade head`` with
"Multiple head revisions are present for given argument 'head'", which
in turn blocked every helm upgrade of prbe-data-plane on the cluster
and stalled the rollout.

No-op merge: schema state is identical pre- and post-merge; only the
alembic_version row advances to ``0074_merge_0073_heads``. Both sibling
upgrades MUST have applied before this runs — alembic enforces that
via the down_revision tuple.

Revision ID: 0074_merge_0073_heads
Revises: 0073_code_repo_state_cascade, 0073_customers_r2_bucket
Create Date: 2026-05-14
"""

from __future__ import annotations


revision = "0074_merge_0073_heads"
down_revision = (
    "0073_code_repo_state_cascade",
    "0073_customers_r2_bucket",
)
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
