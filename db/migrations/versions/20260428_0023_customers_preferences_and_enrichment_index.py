"""customers preferences + enrichment list index

Revision ID: 0023_customer_prefs
Revises: 0022_graph_nodes_alnum_indexes
Create Date: 2026-04-28

Two changes that together unblock the dashboard's Tickets surface:

1. `customers.preferences JSONB`: per-tenant feature toggles, starting
   with `ticket_enrichment_enabled`. JSONB-not-typed-table because we
   only have one key today and the cross-org query story isn't worth a
   dedicated table yet. Default `{}`; readers default missing keys to
   ON so a botched deploy can't silently kill enrichment.

2. Partial covering index for the upcoming
   `GET /dashboard/tickets/enrichments` listing. The list query is
   tenant-scoped to succeeded runs ordered by finished_at DESC; without
   this index it scans the table to satisfy ORDER BY+LIMIT on growth.
"""

from __future__ import annotations

from alembic import op

revision = "0023_customer_prefs"
down_revision = "0022_graph_nodes_alnum_indexes"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE customers
        ADD COLUMN IF NOT EXISTS preferences JSONB NOT NULL DEFAULT '{}'::jsonb
        """
    )

    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_enrichment_runs_listing
        ON enrichment_runs (customer_id, finished_at DESC)
        WHERE status = 'succeeded'
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_enrichment_runs_listing")
    op.execute("ALTER TABLE customers DROP COLUMN IF EXISTS preferences")
