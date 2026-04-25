"""dashboard organization link

Revision ID: 0007_dashboard_organization_link
Revises: 0006_drop_query_cache
Create Date: 2026-04-25

Bridges customers (our tenant primitive) to Better Auth organizations
(Neon Auth's neon_auth.organization). Each team-managed customer points
to exactly one organization; the inverse is enforced by the partial
unique index. ON DELETE RESTRICT prevents Better Auth's
organization.delete from cascading into per-tenant data — the dashboard
soft-deletes via customers.status instead.

Depends on Neon Auth being provisioned on the target branch (creates
the neon_auth schema). If the schema is missing, the FK constraint
will fail at upgrade.
"""

from __future__ import annotations

from alembic import op

revision = "0007_dashboard_organization_link"
down_revision = "0006_drop_query_cache"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_namespace WHERE nspname = 'neon_auth') THEN
                RAISE EXCEPTION
                    'neon_auth schema not found. Provision Neon Auth on this branch first.';
            END IF;
        END $$;
        """
    )

    op.execute(
        """
        ALTER TABLE customers
        ADD COLUMN organization_id UUID
        REFERENCES neon_auth.organization(id) ON DELETE RESTRICT
        """
    )

    op.execute(
        """
        CREATE UNIQUE INDEX customers_organization_id_unique
        ON customers (organization_id)
        WHERE organization_id IS NOT NULL
        """
    )

    op.execute(
        """
        CREATE INDEX idx_customers_active
        ON customers (customer_id)
        WHERE status = 'active'
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_customers_active")
    op.execute("DROP INDEX IF EXISTS customers_organization_id_unique")
    op.execute("ALTER TABLE customers DROP COLUMN IF EXISTS organization_id")
