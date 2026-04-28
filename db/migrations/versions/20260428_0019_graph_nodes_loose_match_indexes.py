"""graph_nodes: functional indexes for loose-match entity filter

Revision ID: 0019_graph_nodes_loose_match
Revises: 0018_chunks_kind
Create Date: 2026-04-28

The list pipeline's entity filter joins graph_edges + graph_nodes with
case-insensitive matching across two dimensions:

  LOWER(canonical_id) = LOWER($X)
  OR LOWER(canonical_id) LIKE '%/' || LOWER($X)
  OR LOWER(properties->>'name') = LOWER($X)

The equality arms benefit from functional indexes on
(customer_id, label, LOWER(canonical_id)) and
(customer_id, label, LOWER(properties->>'name')).

The suffix-LIKE arm (`'%/' || LOWER($X)`) has a leading wildcard that
btree CAN'T index — it falls back to seq-scan-of-subset over rows
matching (customer_id, label). That's accepted: graph_nodes per tenant
is small (a few hundred at most for any plausible horizon), the seq
scan over that filtered subset is microseconds, and adding a reverse-
string index for the ~rare bare-name case isn't worth the cognitive
cost. Revisit if any tenant ever exceeds 100k graph_nodes.

CONCURRENTLY because graph_nodes can grow as integrations are added —
no point taking ACCESS EXCLUSIVE on a fresh tenant onboarding tomorrow.
"""

from __future__ import annotations

from alembic import op

revision = "0019_graph_nodes_loose_match"
down_revision = "0018_chunks_kind"
branch_labels = None
depends_on = None


CANONICAL_INDEX = "idx_graph_nodes_lower_canonical"
NAME_INDEX = "idx_graph_nodes_lower_props_name"


def upgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute(
            f"""
            CREATE INDEX CONCURRENTLY IF NOT EXISTS {CANONICAL_INDEX}
                ON graph_nodes (customer_id, label, LOWER(canonical_id))
            """
        )
        op.execute(
            f"""
            CREATE INDEX CONCURRENTLY IF NOT EXISTS {NAME_INDEX}
                ON graph_nodes (customer_id, label, LOWER(properties ->> 'name'))
            """
        )


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {NAME_INDEX}")
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {CANONICAL_INDEX}")
