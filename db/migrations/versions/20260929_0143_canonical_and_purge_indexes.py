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

CONCURRENTLY, in alembic's autocommit block: this runs unattended in the
engine-kb-migrate hook and both tables take writes on every ingest. What makes
that safe to leave unattended (0141 shipped without these; review of #608):

  * A session advisory lock serializes migrators across the autocommit
    boundary, so a second runner never mistakes a build in progress (which is
    INVALID until it finishes) for a dead one and drops it.
  * Explicit timeouts, set here rather than inherited: 0142 leaves a session
    `lock_timeout = 5s` behind on an upgrade that runs through it. The build
    may wait on old transactions (lock_timeout 0), bounded as a whole by a
    25-minute statement_timeout that fits the hook's 30-minute deadline, and
    client_connection_check_interval makes a killed hook pod's backend notice
    and stop instead of building on alone.
  * An existing index of the same name is accepted only when it is valid AND
    its definition is exactly the one below. Anything else (an INVALID
    leftover, a hand-built index with another key order or predicate) is
    dropped CONCURRENTLY and rebuilt, and the result is verified before the
    migration is recorded. `IF NOT EXISTS` alone matches on the name only.
  * Names are schema-qualified (public), so a same-named index in another
    schema can neither satisfy nor be dropped by this check.

Before the first research deploy that runs it, check pg_stat_activity ordered by
xact_start: the build waits for every transaction older than itself.
"""

import sqlalchemy as sa
from alembic import op

revision = "0143_canonical_and_purge_idx"
down_revision = "0142_drop_workflow_memory"
branch_labels = None
depends_on = None

#: (name, table, column list). db/schema.sql declares the same two.
INDEXES: tuple[tuple[str, str, str], ...] = (
    ("idx_graph_nodes_customer_canonical", "graph_nodes", "customer_id, canonical_id"),
    ("idx_pending_edges_customer", "pending_edges", "customer_id"),
)

_MIGRATOR_LOCK = "hashtextextended('prbe-knowledge:concurrent-index-migration', 0)"


def expected_definition(name: str, table: str, columns: str) -> str:
    return f"CREATE INDEX {name} ON public.{table} USING btree ({columns})"


def _current(bind, name: str):
    return bind.execute(
        sa.text(
            """
            SELECT i.indisvalid AND i.indisready AS usable, pg_get_indexdef(c.oid) AS definition
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            JOIN pg_index i ON i.indexrelid = c.oid
            WHERE n.nspname = 'public' AND c.relname = :name
            """
        ),
        {"name": name},
    ).first()


def ensure_index(bind, name: str, table: str, columns: str) -> bool:
    """Build `public.<name>` unless an identical valid one exists. True if built."""
    want = expected_definition(name, table, columns)
    row = _current(bind, name)
    if row is not None and row.usable and row.definition == want:
        return False
    if row is not None:
        bind.execute(sa.text(f"DROP INDEX CONCURRENTLY IF EXISTS public.{name}"))
    bind.execute(sa.text(f"CREATE INDEX CONCURRENTLY {name} ON public.{table} ({columns})"))
    row = _current(bind, name)
    if row is None or not row.usable or row.definition != want:
        raise RuntimeError(f"{name} is not usable after the build: {row}")
    return True


def upgrade() -> None:
    # CREATE INDEX CONCURRENTLY cannot run inside a transaction.
    with op.get_context().autocommit_block():
        bind = op.get_bind()
        bind.execute(sa.text(f"SELECT pg_advisory_lock({_MIGRATOR_LOCK})"))
        try:
            bind.execute(sa.text("SET lock_timeout = 0"))
            bind.execute(sa.text("SET statement_timeout = '25min'"))
            bind.execute(sa.text("SET client_connection_check_interval = '10s'"))
            for name, table, columns in INDEXES:
                ensure_index(bind, name, table, columns)
        finally:
            bind.execute(sa.text("RESET client_connection_check_interval"))
            bind.execute(sa.text("RESET statement_timeout"))
            bind.execute(sa.text("RESET lock_timeout"))
            bind.execute(sa.text(f"SELECT pg_advisory_unlock({_MIGRATOR_LOCK})"))


def downgrade() -> None:
    with op.get_context().autocommit_block():
        for name, _table, _columns in reversed(INDEXES):
            op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS public.{name}")
