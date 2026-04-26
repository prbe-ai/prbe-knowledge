"""integration_tokens: add device_id, device_metadata, surrogate PK, partial unique indexes

Revision ID: 0011_integration_tokens_devices
Revises: 0010_own_organization_tables
Create Date: 2026-04-26

The composite primary key (customer_id, source_system) is replaced with a
surrogate token_id UUID PK so that device-scoped sources (Phase 1: claude_code)
can have many rows per (customer, source). The original uniqueness contract is
preserved for non-device sources via a partial unique index. A second partial
unique index handles the multi-device case.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0011_integration_tokens_devices"
down_revision = "0010_own_organization_tables"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("integration_tokens", sa.Column("device_id", sa.Text(), nullable=True))
    op.add_column("integration_tokens", sa.Column("device_metadata", sa.dialects.postgresql.JSONB(), nullable=True))

    op.drop_constraint("integration_tokens_pkey", "integration_tokens", type_="primary")
    op.add_column(
        "integration_tokens",
        sa.Column(
            "token_id",
            sa.dialects.postgresql.UUID(as_uuid=False),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
    )
    op.create_primary_key("integration_tokens_pkey", "integration_tokens", ["token_id"])

    op.execute(
        """
        CREATE UNIQUE INDEX integration_tokens_unique_per_source
        ON integration_tokens (customer_id, source_system)
        WHERE device_id IS NULL
        """
    )
    op.execute(
        """
        CREATE UNIQUE INDEX integration_tokens_unique_per_device
        ON integration_tokens (customer_id, source_system, device_id)
        WHERE device_id IS NOT NULL
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS integration_tokens_unique_per_device")
    op.execute("DROP INDEX IF EXISTS integration_tokens_unique_per_source")

    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM integration_tokens WHERE device_id IS NOT NULL) THEN
                RAISE EXCEPTION 'Cannot downgrade: device-scoped rows exist. Delete them first.';
            END IF;
        END $$;
        """
    )

    op.drop_constraint("integration_tokens_pkey", "integration_tokens", type_="primary")
    op.drop_column("integration_tokens", "token_id")
    op.create_primary_key("integration_tokens_pkey", "integration_tokens", ["customer_id", "source_system"])
    op.drop_column("integration_tokens", "device_metadata")
    op.drop_column("integration_tokens", "device_id")
