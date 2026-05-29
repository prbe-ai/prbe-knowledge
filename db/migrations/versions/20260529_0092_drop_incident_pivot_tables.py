"""drop incident / on-call pivot tables (Spec B Part 2)

The pivot code that wrote these four tables was removed in the strip (Spec B
Part 1); they have been dormant since. This drops them. A pg_dump backup of all
four was taken before this runs (incident_investigations held 7 historical rows;
the other three were empty).

All four are children of customers(customer_id) ON DELETE CASCADE with no
incoming FKs, so CASCADE cleans up their RLS policies + indexes and order is
irrelevant.

NOTE: revision id "0092_drop_incident_pivot_tables" is 31 chars (<=32) per
alembic_version.version_num cap (feedback_alembic_version_32char_cap).
"""

from __future__ import annotations

from alembic import op

revision = "0092_drop_incident_pivot_tables"
down_revision = "0091_collapse_node_labels"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("DROP TABLE IF EXISTS incident_investigations CASCADE")
    op.execute("DROP TABLE IF EXISTS customer_incident_mcp_servers CASCADE")
    op.execute("DROP TABLE IF EXISTS wiki_review_queue CASCADE")
    op.execute("DROP TABLE IF EXISTS customer_postmortem_templates CASCADE")


def downgrade() -> None:
    # Irreversible: the pivot feature and its code are gone. Recreate the table
    # DDL from db/schema.sql history and restore data from the pre-drop pg_dump
    # backup if ever needed (see Spec B Part 2).
    raise NotImplementedError(
        "drop_incident_pivot_tables is irreversible; restore from the pg_dump "
        "backup taken before the drop."
    )
