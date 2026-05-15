"""Add ON DELETE CASCADE FK from ``code_repo_state.customer_id`` to
``customers.customer_id``.

Why
---
``code_repo_state`` (migration 0049, codegraph phase A) was created with a
bare ``customer_id TEXT NOT NULL`` column and no FK. Every other per-tenant
table in this chain (documents, chunks, graph_nodes, integration_tokens,
...) carries a cascade FK, so the data plane's
``DELETE /internal/admin/customer/{id}`` endpoint — which relies on a
single ``DELETE FROM customers WHERE customer_id = $1`` cascading through
those FKs — quietly leaves ``code_repo_state`` rows behind on every
tenant teardown.

Hit in prod 2026-05-14 against the probe-founders teardown: the CP
``delete_team`` endpoint (prbe-backend PR #276 fixes the missing DP
mirror call) cascades the rest of the schema fine, but
``code_repo_state`` orphans because it never had the cascade FK.

Idempotent — drops a pre-existing constraint with the default Postgres
name before adding so a re-run is safe.

Skipped under non-Postgres dialects.

Revision ID: 0073_code_repo_state_cascade
Revises: 0072_ingestion_cursors
Create Date: 2026-05-14
"""
from __future__ import annotations

from alembic import op

revision = "0073_code_repo_state_cascade"
down_revision = "0072_ingestion_cursors"
branch_labels = None
depends_on = None


_FK_NAME = "code_repo_state_customer_id_fkey"


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return
    op.execute(
        f"ALTER TABLE code_repo_state DROP CONSTRAINT IF EXISTS {_FK_NAME}"
    )
    op.execute(
        f"ALTER TABLE code_repo_state "
        f"ADD CONSTRAINT {_FK_NAME} "
        f"FOREIGN KEY (customer_id) "
        f"REFERENCES customers (customer_id) ON DELETE CASCADE"
    )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return
    op.execute(
        f"ALTER TABLE code_repo_state DROP CONSTRAINT IF EXISTS {_FK_NAME}"
    )
