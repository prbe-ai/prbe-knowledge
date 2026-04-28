"""acl_snapshots: idempotency UNIQUE

Revision ID: 0013_acl_snapshots_unique
Revises: 0012_per_tenant_pk
Create Date: 2026-04-26

Reclaim retries and webhook redeliveries that hit the normalizer re-run
_insert_acl_snapshots, which was a naked INSERT loop. Without a UNIQUE
constraint, every replay multiplied ACL rows by N — pure bloat (not
data loss), but unbounded under any retry storm.

The UNIQUE key matches "the same assertion at the same valid_from is
the same row." Updates that legitimately change a permission produce a
distinct (permission, valid_from) tuple and remain insertable.
"""

from __future__ import annotations

from alembic import op

revision = "0013_acl_snapshots_unique"
down_revision = "0012_per_tenant_pk"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Drop any duplicates accumulated before this constraint existed.
    # Keep the lowest snapshot_id for each unique tuple.
    op.execute(
        """
        DELETE FROM acl_snapshots a
        USING acl_snapshots b
        WHERE a.snapshot_id > b.snapshot_id
          AND a.customer_id    = b.customer_id
          AND a.source_system  = b.source_system
          AND a.principal_type = b.principal_type
          AND a.principal_id   = b.principal_id
          AND a.resource_type  = b.resource_type
          AND a.resource_id    = b.resource_id
          AND a.permission     = b.permission
          AND a.valid_from     = b.valid_from
        """
    )

    op.execute(
        """
        ALTER TABLE acl_snapshots
        ADD CONSTRAINT acl_snapshots_assertion_unique UNIQUE (
            customer_id, source_system,
            principal_type, principal_id,
            resource_type, resource_id,
            permission, valid_from
        )
        """
    )


def downgrade() -> None:
    op.execute(
        "ALTER TABLE acl_snapshots DROP CONSTRAINT IF EXISTS acl_snapshots_assertion_unique"
    )
