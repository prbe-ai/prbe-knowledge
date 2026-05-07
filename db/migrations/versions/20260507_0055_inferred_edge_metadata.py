"""inferred_edge_metadata: extractor tracking on graph_edges + inferred_edges_queue

Revision ID: 0055_inferred_edge_metadata
Revises: 0054_drop_cross_repo_dlq, 0054_graph_node_degree_community
Create Date: 2026-05-07

Lane B (Cross-source LLM-inferred edges):

1. Two new nullable columns on graph_edges for provenance tracking:
   - extractor_id: which prompt version produced this edge (e.g. "inferred_edges:v1")
   - extracted_at: when this edge was produced by the extractor

2. A new side-queue table inferred_edges_queue:
   - One row per (customer, anchor_doc) whenever a doc is ingested
   - Side-queue worker drains it: builds bundle -> LLM call -> upsert edges
   - RLS mirrors other customer-scoped tables in this codebase

This is also an alembic MERGE migration: it chains both 0054_drop_cross_repo_dlq
(landed via PR #166) and 0054_graph_node_degree_community (landed via PR #168)
into a single linear head. Both 0054 revs branched off 0053; main currently has
two heads, which alembic refuses to upgrade. After this migration applies, head
is single again at 0055_inferred_edge_metadata.

Forward-only. No downgrade (we do not undeploy migrations in this codebase).
"""

from __future__ import annotations

from alembic import op

revision = "0055_inferred_edge_metadata"
down_revision = ("0054_drop_cross_repo_dlq", "0054_graph_node_degree_community")
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ------------------------------------------------------------------ #
    # graph_edges: add extractor provenance columns                       #
    # ------------------------------------------------------------------ #
    op.execute(
        """
        ALTER TABLE graph_edges
            ADD COLUMN IF NOT EXISTS extractor_id  TEXT NULL,
            ADD COLUMN IF NOT EXISTS extracted_at  TIMESTAMPTZ NULL
        """
    )

    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_graph_edges_customer_extractor
            ON graph_edges(customer_id, extractor_id)
            WHERE extractor_id IS NOT NULL
        """
    )

    # ------------------------------------------------------------------ #
    # inferred_edges_queue                                                 #
    # ------------------------------------------------------------------ #
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS inferred_edges_queue (
            id                      BIGSERIAL PRIMARY KEY,
            customer_id             TEXT NOT NULL
                                    REFERENCES customers(customer_id) ON DELETE CASCADE,
            anchor_doc_id           TEXT NOT NULL,
            enqueued_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            processing_started_at   TIMESTAMPTZ NULL,
            processing_worker_id    TEXT NULL,
            attempts                INT NOT NULL DEFAULT 0,
            extractor_id            TEXT NOT NULL,
            done_at                 TIMESTAMPTZ NULL,
            error                   TEXT NULL
        )
        """
    )

    # Partial index: drain queries filter on pending rows only.
    # As done/failed rows accumulate the tail stays outside this index.
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_inferred_edges_queue_pending
            ON inferred_edges_queue(enqueued_at)
            WHERE processing_started_at IS NULL AND done_at IS NULL
        """
    )

    # ------------------------------------------------------------------ #
    # RLS on inferred_edges_queue: same pattern as other customer-scoped  #
    # tables (USING / WITH CHECK on app.current_customer_id GUC).         #
    # ------------------------------------------------------------------ #
    op.execute("ALTER TABLE inferred_edges_queue ENABLE ROW LEVEL SECURITY")

    op.execute(
        """
        CREATE POLICY inferred_edges_queue_tenant_isolation
            ON inferred_edges_queue
            USING (customer_id = current_setting('app.current_customer_id', true))
            WITH CHECK (customer_id = current_setting('app.current_customer_id', true))
        """
    )

    # Service role bypasses RLS so the worker can use raw_conn for drain
    # queries without setting the GUC (matches other queue tables).
    op.execute(
        """
        ALTER TABLE inferred_edges_queue FORCE ROW LEVEL SECURITY
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS inferred_edges_queue")
    op.execute(
        """
        ALTER TABLE graph_edges
            DROP COLUMN IF EXISTS extractor_id,
            DROP COLUMN IF EXISTS extracted_at
        """
    )
