"""graph_nodes: add degree + community_id columns for surprise-score.

Revision ID: 0054_graph_node_degree_community
Revises: 0053_cross_repo_reverify_dlq
Create Date: 2026-05-07

Lane A of the surprise-score feature (fix 1).

- degree     INT NOT NULL DEFAULT 0  -- count of incident edges, maintained
                                        write-time by graph_writer.upsert_edges
- community_id INT NULL               -- Leiden partition id, populated nightly
                                        by services/community/leiden.py cron

Backfills degree from current graph_edges data.

CRITICAL: the UPDATE is wrapped in NO FORCE / FORCE RLS because Alembic
runs as a superuser role that Postgres still restricts via FORCE ROW LEVEL
SECURITY on graph_nodes. Without the toggle the UPDATE silently zero-matches
(every updated row count = 0). This is the established pattern in this repo
-- see feedback_graph_nodes_rls_force in project memory.

Forward-only. No downgrade (migrations in this codebase are never reverted
in production; dropping degree/community_id would lose data).
"""

from __future__ import annotations

from alembic import op

revision = "0054_graph_node_degree_community"
down_revision = "0053_cross_repo_reverify_dlq"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add columns.
    op.execute(
        """
        ALTER TABLE graph_nodes
            ADD COLUMN IF NOT EXISTS degree       INT NOT NULL DEFAULT 0,
            ADD COLUMN IF NOT EXISTS community_id INT NULL
        """
    )

    # Index for community-scoped queries (surprise_score cross-community check,
    # Leiden cron per-customer scan). Partial: NULL community_id rows are
    # unpartitioned nodes (< 100-edge tenants) -- exclude them from the index
    # so the planner can use it for selective lookups.
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_graph_nodes_customer_community
            ON graph_nodes (customer_id, community_id)
            WHERE community_id IS NOT NULL
        """
    )

    # Backfill degree.
    # Wrap in NO FORCE / FORCE RLS toggle: Alembic runs as superuser, but
    # graph_nodes has FORCE ROW LEVEL SECURITY -- UPDATE silently zero-matches
    # without this toggle even for superuser. This pattern is required for any
    # bulk UPDATE on graph_nodes in this codebase.
    op.execute("ALTER TABLE graph_nodes NO FORCE ROW LEVEL SECURITY")
    op.execute(
        """
        UPDATE graph_nodes
        SET degree = subq.cnt
        FROM (
            SELECT n.node_id, COUNT(*) AS cnt
            FROM graph_nodes n
            LEFT JOIN graph_edges e
              ON e.from_node_id = n.node_id
              OR e.to_node_id   = n.node_id
            GROUP BY n.node_id
        ) subq
        WHERE graph_nodes.node_id = subq.node_id
        """
    )
    op.execute("ALTER TABLE graph_nodes FORCE ROW LEVEL SECURITY")


def downgrade() -> None:
    # Forward-only migration -- no downgrade implemented.
    # Dropping these columns would destroy backfilled degree data and any
    # community assignments written by the nightly cron.
    pass
