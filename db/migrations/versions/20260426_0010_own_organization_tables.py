"""own organization tables

Revision ID: 0010_own_organization_tables
Revises: 0009_mcp_oauth
Create Date: 2026-04-26

Move organization / membership / invitation ownership from Neon Auth's
Better Auth org-plugin tables (neon_auth.organization, .member,
.invitation) into our own public.* tables. Reasons:

  * Better Auth's API endpoints (/organization/create, /invite-member,
    etc.) authenticate via the user's session cookie. Our BFF only
    has the JWT, so it can't drive those endpoints. Owning the tables
    ourselves means the BFF can write to them directly inside a single
    transaction with the customers row (no orphan-org races).

  * No coupling to Better Auth's schema. Their column names, FK shapes,
    and id formats are theirs to evolve. We pin our own.

  * organization_limit=1 + creator_role=owner enforcement moves into
    our SQL (UNIQUE constraint + explicit role insert) instead of
    living in upstream config we can't see.

The neon_auth.* tables stay in place — Neon Auth still mints them on
provisioning and they cost nothing sitting empty. Disable the org
plugin in the Neon Auth console at your leisure to stop them refilling.

Migration is destructive on customers.organization_id FK direction:
the column previously pointed at neon_auth.organization(id), now
points at public.organizations(id). The DB was truncated before this
landed, so no data preservation needed.
"""

from __future__ import annotations

from alembic import op

revision = "0010_own_organization_tables"
down_revision = "0009_mcp_oauth"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE organizations (
            id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            name        text NOT NULL,
            slug        text NOT NULL UNIQUE,
            created_at  timestamptz NOT NULL DEFAULT now(),
            updated_at  timestamptz NOT NULL DEFAULT now()
        )
        """
    )

    op.execute(
        """
        CREATE TABLE memberships (
            id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            organization_id uuid NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
            user_id         uuid NOT NULL REFERENCES neon_auth."user"(id) ON DELETE CASCADE,
            role            text NOT NULL CHECK (role IN ('owner', 'admin', 'member')),
            created_at      timestamptz NOT NULL DEFAULT now(),
            UNIQUE (user_id)
        )
        """
    )

    op.execute(
        "CREATE INDEX idx_memberships_org ON memberships(organization_id)"
    )

    op.execute(
        """
        CREATE TABLE invitations (
            id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            organization_id uuid NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
            inviter_id      uuid NOT NULL REFERENCES neon_auth."user"(id) ON DELETE CASCADE,
            email           text NOT NULL,
            role            text NOT NULL CHECK (role IN ('owner', 'admin', 'member')),
            status          text NOT NULL DEFAULT 'pending'
                            CHECK (status IN ('pending', 'accepted', 'cancelled', 'expired')),
            expires_at      timestamptz NOT NULL,
            created_at      timestamptz NOT NULL DEFAULT now()
        )
        """
    )

    op.execute(
        """
        CREATE UNIQUE INDEX idx_invitations_pending_unique
        ON invitations(organization_id, email)
        WHERE status = 'pending'
        """
    )

    op.execute(
        "CREATE INDEX idx_invitations_org ON invitations(organization_id)"
    )

    # Re-point customers FK from neon_auth.organization → public.organizations.
    # The previous FK was added in 0007. We drop it and re-add against the
    # new table; the column itself stays (uuid).
    op.execute(
        "ALTER TABLE customers DROP CONSTRAINT IF EXISTS customers_organization_id_fkey"
    )
    op.execute(
        """
        ALTER TABLE customers
        ADD CONSTRAINT customers_organization_id_fkey
        FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE RESTRICT
        """
    )


def downgrade() -> None:
    op.execute(
        "ALTER TABLE customers DROP CONSTRAINT IF EXISTS customers_organization_id_fkey"
    )
    op.execute(
        """
        ALTER TABLE customers
        ADD CONSTRAINT customers_organization_id_fkey
        FOREIGN KEY (organization_id) REFERENCES neon_auth.organization(id) ON DELETE RESTRICT
        """
    )

    op.execute("DROP INDEX IF EXISTS idx_invitations_org")
    op.execute("DROP INDEX IF EXISTS idx_invitations_pending_unique")
    op.execute("DROP TABLE IF EXISTS invitations")

    op.execute("DROP INDEX IF EXISTS idx_memberships_org")
    op.execute("DROP TABLE IF EXISTS memberships")

    op.execute("DROP TABLE IF EXISTS organizations")
