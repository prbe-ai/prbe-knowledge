"""merge two 0011 heads

Revision ID: 0013_merge_heads
Revises: 0012_per_tenant_pk, 0011_integration_tokens_devices
Create Date: 2026-04-27

Two parallel migrations both branched from 0010:
  * 0011_graph_source_system  → 0012_per_tenant_pk  (graph provenance + per-tenant PKs)
  * 0011_integration_tokens_devices                  (device_id + surrogate UUID PK)

`alembic upgrade head` refused to run while both were unmerged heads, which
broke deploys on every push since the fork. This is a no-op merge that
collapses the DAG so future deploys can upgrade cleanly. No DDL — both
parent revisions already carry their own.
"""

from __future__ import annotations

revision = "0013_merge_heads"
down_revision = ("0012_per_tenant_pk", "0011_integration_tokens_devices")
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
