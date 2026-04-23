"""force RLS on graph tables.

Revision ID: 0002_force_rls
Revises: 0001_initial_schema
Create Date: 2026-04-22

Postgres exempts the table owner + superusers from RLS unless FORCE is set.
Applying FORCE here so the tenant_isolation policy actually holds when the
app role is the table owner (the default on Neon).
"""

from __future__ import annotations

from alembic import op

revision = "0002_force_rls"
down_revision = "0001_initial_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE graph_nodes FORCE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE graph_edges FORCE ROW LEVEL SECURITY")


def downgrade() -> None:
    op.execute("ALTER TABLE graph_nodes NO FORCE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE graph_edges NO FORCE ROW LEVEL SECURITY")
