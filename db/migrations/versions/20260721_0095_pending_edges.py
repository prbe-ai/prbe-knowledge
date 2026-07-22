"""Deferred-edge queue: pending_edges

Revision ID: 0095_pending_edges
Revises: 0094_documents_chunks_rls
Create Date: 2026-07-21

Why
---
``graph_writer.upsert_edges`` resolves an edge's endpoints only against the
node set upserted in the SAME batch (``node_ids``). An edge whose endpoint is
not in that map is silently dropped -- a bare ``continue`` with no log, no
counter, no retry. That is fine when a batch always carries every endpoint it
references, which was true for the flat Document + Person + AUTHORED shape.

It is NOT true for the client-asserted graph payload (custom-ingest ``nodes`` /
``edges``): a run can be ingested before its experiment ever has, and the
research-os outbox delivers out of order (multi-replica ``SKIP LOCKED`` + retry
re-queue). A DB lookup for the missing endpoint does not fix this -- a genuinely
not-yet-ingested parent is absent everywhere, so the lookup fails too and the
edge is lost permanently, silently, differently on every deploy.

This table parks such an edge keyed by the MISSING endpoint's
(label, canonical_id). When that node is later written, the post-write worker
drains the matching rows and materialises the edges. Order stops mattering, and
queue depth becomes an observable completeness signal (the drop was invisible
before).

Shape
-----
Deliberately mirrors ``node_post_write_queue``: tenant-scoped, RLS-forced, a
``locked_until`` lease column, and a partial index on the pending set. A
``created_at`` supports the TTL reaper that sweeps edges whose counterpart
never arrives (an unbounded queue otherwise). RLS matches every other
tenant-scoped graph table (graph_nodes / graph_edges / graph_node_provenance):
ENABLE + FORCE + a ``tenant_isolation`` policy on the
``app.current_customer_id`` GUC, so a missing ``with_tenant`` scope fails
closed rather than leaking across tenants.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

# Keep <=32 chars (alembic_version.version_num is varchar(32)).
revision: str = "0095_pending_edges"
down_revision: str | Sequence[str] | None = "0094_documents_chunks_rls"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # CREATE TABLE takes no lock on existing tables; keep a short bound anyway
    # so a migrate hook never wedges behind an unrelated long transaction.
    op.execute("SET lock_timeout = '5s'")
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS pending_edges (
            id              BIGSERIAL PRIMARY KEY,
            customer_id     TEXT NOT NULL
                            REFERENCES customers(customer_id) ON DELETE CASCADE,
            -- The endpoint that could NOT be resolved when the edge arrived.
            -- The drain keys on this.
            missing_label        TEXT NOT NULL,
            missing_canonical_id TEXT NOT NULL,
            -- The full edge, so it can be replayed verbatim once resolvable.
            edge_type            TEXT NOT NULL,
            from_label           TEXT NOT NULL,
            from_canonical_id    TEXT NOT NULL,
            to_label             TEXT NOT NULL,
            to_canonical_id      TEXT NOT NULL,
            source_system        TEXT NOT NULL,
            properties           JSONB NOT NULL DEFAULT '{}'::JSONB,
            valid_from           TIMESTAMPTZ NULL,
            created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            locked_until         TIMESTAMPTZ NULL
        )
        """
    )
    # Drain lookup: "give me pending edges waiting on THIS node", newest-safe
    # ordering is irrelevant so no sort column in the index.
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_pending_edges_missing
            ON pending_edges (customer_id, missing_label, missing_canonical_id)
            WHERE locked_until IS NULL
        """
    )
    # Reaper scan: oldest-first over the unlocked backlog.
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_pending_edges_created
            ON pending_edges (created_at)
            WHERE locked_until IS NULL
        """
    )
    op.execute("ALTER TABLE pending_edges ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE pending_edges FORCE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY tenant_isolation ON pending_edges
            USING (customer_id = current_setting('app.current_customer_id', true))
            WITH CHECK (customer_id = current_setting('app.current_customer_id', true))
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS pending_edges")
