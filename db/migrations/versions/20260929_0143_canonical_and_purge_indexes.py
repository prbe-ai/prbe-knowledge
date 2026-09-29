"""Index graph_nodes by canonical id alone, and pending_edges by tenant.

Two lookups had no index they could use (measured as `app` on the research
plane, 2026-09-29):

  * graph_nodes by (customer_id, canonical_id) with no label. The subgraph
    anchor check (graph_explore.anchor_exists, called per hop from
    agent/tools.py), the one-hop walk's anchor CTE and the adapter's entity
    name fetch (agent/adapter.py) all look a node up by canonical id without
    knowing its label. The unique key leads (customer_id, label, ...), so each
    read the tenant's whole key prefix: 29-111 ms warm, 780 ms cold for
    `probe`. `texteq` is LEAKPROOF, so this btree is usable under FORCE RLS.
    Now idx_graph_nodes_customer_canonical.
  * pending_edges by customer_id. Its FK to customers had only partial
    indexes, so every tenant-purge batch (research-os kb_mirror drain) was a
    full scan of the 643 MB table. Now idx_pending_edges_customer.

CONCURRENTLY, in alembic's autocommit block, with 0141's INVALID-leftover
guard: this runs unattended in the engine-kb-migrate hook and both tables take
writes on every ingest. See 0141's docstring for the CONCURRENTLY wait caveat.
"""

import sqlalchemy as sa
from alembic import op

revision = "0143_canonical_and_purge_idx"
down_revision = "0142_drop_workflow_memory"
branch_labels = None
depends_on = None

# (name, "ON table (...)"). db/schema.sql declares the same two.
_INDEXES: tuple[tuple[str, str], ...] = (
    ("idx_graph_nodes_customer_canonical", "ON graph_nodes (customer_id, canonical_id)"),
    ("idx_pending_edges_customer", "ON pending_edges (customer_id)"),
)


def upgrade() -> None:
    # CREATE INDEX CONCURRENTLY cannot run inside a transaction.
    with op.get_context().autocommit_block():
        bind = op.get_bind()
        for name, definition in _INDEXES:
            invalid = bind.execute(
                sa.text(
                    """
                    SELECT 1
                    FROM pg_class c
                    JOIN pg_index i ON i.indexrelid = c.oid
                    WHERE c.relname = :name AND NOT i.indisvalid
                    """
                ),
                {"name": name},
            ).first()
            if invalid is not None:
                op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {name}")
            op.execute(f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {name} {definition}")


def downgrade() -> None:
    with op.get_context().autocommit_block():
        for name, _definition in reversed(_INDEXES):
            op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {name}")
