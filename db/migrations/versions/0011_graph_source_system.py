"""graph node provenance + edge source_system

Revision ID: 0011_graph_source_system
Revises: 0010_own_organization_tables
Create Date: 2026-04-26

Track which source system(s) asserted each graph node so that
disconnect-integration can do correct multi-source cleanup: a node touched
by multiple connectors must survive disconnection of any single one.

  * graph_node_provenance: composite (node_id, source_system) records each
    asserting source. ON DELETE CASCADE on node_id keeps it tidy if a node
    is ever hard-deleted; customer_id is denormalized for cheap
    customer-scoped purges.

  * graph_edges.source_system: which connector asserted this edge. The
    upserter preserves the original asserting source on conflict (edges
    are not multi-sourced today; first writer wins).
"""

from __future__ import annotations

from alembic import op

revision = "0011_graph_source_system"
down_revision = "0010_own_organization_tables"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE graph_node_provenance (
            node_id        BIGINT NOT NULL REFERENCES graph_nodes(node_id) ON DELETE CASCADE,
            customer_id    TEXT   NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
            source_system  TEXT   NOT NULL,
            first_seen_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            last_seen_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (node_id, source_system)
        )
        """
    )

    op.execute(
        """
        CREATE INDEX idx_provenance_customer_source
            ON graph_node_provenance (customer_id, source_system)
        """
    )

    op.execute(
        "ALTER TABLE graph_edges ADD COLUMN source_system TEXT"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE graph_edges DROP COLUMN IF EXISTS source_system")
    op.execute("DROP INDEX IF EXISTS idx_provenance_customer_source")
    op.execute("DROP TABLE IF EXISTS graph_node_provenance")
