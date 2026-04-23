"""customer_source_mapping table.

Revision ID: 0003_customer_source_mapping
Revises: 0002_force_rls
Create Date: 2026-04-23

Populated at OAuth install time; read at webhook time to resolve customer_id
from the source-side workspace/team/org identifier carried in the payload.
"""

from __future__ import annotations

from alembic import op

revision = "0003_customer_source_mapping"
down_revision = "0002_force_rls"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Idempotent: schema.sql (run verbatim by 0001) already creates this table,
    # so a fresh DB provision skips the CREATE. This migration matters only for
    # databases that provisioned from an older schema.sql revision.
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS customer_source_mapping (
            source_system   TEXT NOT NULL,
            external_id     TEXT NOT NULL,
            customer_id     TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
            external_name   TEXT,
            metadata        JSONB NOT NULL DEFAULT '{}',
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (source_system, external_id)
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_customer_source_mapping_customer "
        "ON customer_source_mapping (customer_id, source_system)"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS customer_source_mapping")
