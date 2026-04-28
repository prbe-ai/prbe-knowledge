"""organizations: add dev_enabled flag for per-org dev-mode gating

Revision ID: 0021_organizations_dev_enabled
Revises: 0020_usage_events
Create Date: 2026-04-28

Per-organization boolean that gates dev-only UI features in prbe-dashboard.
Any member of an org with `dev_enabled = true` automatically sees the dev
features — there's no per-user toggle, just an org-level flag flipped by an
admin (today, manually via SQL; later, via the dashboard settings page).

Read paths:
  * prbe-backend's `_resolve_active_membership` JOINs this column when it
    looks up the caller's active org/role, so /me returns it in a single
    query (no extra roundtrip).
  * prbe-dashboard reads `dev_enabled` off the /me payload and uses it to
    show/hide dev surfaces (debug panels, internal tools, beta features).

Default false: existing rows backfill to false via `server_default`, so no
data migration is needed. The Probe org gets flipped to true by hand:
    UPDATE organizations SET dev_enabled = true WHERE slug = 'probe';

Index:
  * Partial index on (dev_enabled) WHERE dev_enabled = true. Tiny — only
    indexes the few "dev" orgs — and makes "list all dev orgs" instant for
    admin tooling. Non-dev orgs aren't indexed at all.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0021_organizations_dev_enabled"
down_revision = "0020_usage_events"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "organizations",
        sa.Column(
            "dev_enabled",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )

    op.create_index(
        "ix_organizations_dev_enabled",
        "organizations",
        ["dev_enabled"],
        postgresql_where=sa.text("dev_enabled = true"),
    )


def downgrade() -> None:
    op.drop_index("ix_organizations_dev_enabled", table_name="organizations")
    op.drop_column("organizations", "dev_enabled")
