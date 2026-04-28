"""graph_nodes: functional indexes for alphanumeric-normalized entity match

Revision ID: 0022_graph_nodes_alnum_indexes
Revises: 0021_organizations_dev_enabled
Create Date: 2026-04-28

PR #18 added two new arms to the list-pipeline entity filter that strip
non-alphanumeric chars on both sides before comparing, so
`"external investigations"` matches a graph node whose canonical_id is
`"external-investigations"`. Those arms are correct but slow — they
break the existing `idx_graph_nodes_lower_canonical` and
`idx_graph_nodes_lower_props_name` functional indexes, falling back to
a seq scan over the (customer_id, label)-narrowed subset.

At Phase 0 sizes (a few hundred graph_nodes per tenant) the seq scan
is microseconds. But the cost is unbounded — once a tenant gets a
larger Slack workspace or many GitHub repos, this turns into a real
hot path. Cheap to fix now while the table is small.

Two functional indexes, mirroring the existing LOWER() ones but with
the regexp_replace expression that the new arms use. CONCURRENTLY
because graph_nodes is on the request path for every list query.
"""

from __future__ import annotations

from alembic import op

revision = "0022_graph_nodes_alnum_indexes"
down_revision = "0021_organizations_dev_enabled"
branch_labels = None
depends_on = None


CANONICAL_INDEX = "idx_graph_nodes_alnum_canonical"
NAME_INDEX = "idx_graph_nodes_alnum_props_name"


def upgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute(
            f"""
            CREATE INDEX CONCURRENTLY IF NOT EXISTS {CANONICAL_INDEX}
                ON graph_nodes (
                    customer_id,
                    label,
                    regexp_replace(LOWER(canonical_id), '[^a-z0-9]+', '', 'g')
                )
            """
        )
        op.execute(
            f"""
            CREATE INDEX CONCURRENTLY IF NOT EXISTS {NAME_INDEX}
                ON graph_nodes (
                    customer_id,
                    label,
                    regexp_replace(LOWER(properties ->> 'name'), '[^a-z0-9]+', '', 'g')
                )
            """
        )


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {NAME_INDEX}")
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {CANONICAL_INDEX}")
