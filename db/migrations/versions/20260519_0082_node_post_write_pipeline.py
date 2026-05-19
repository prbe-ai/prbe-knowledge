"""node_post_write_pipeline

Infrastructure for the unified post-node-write pipeline that runs all
registered NodeAnalyzers (InferredEdges + AutoMerge at launch) on every
node touched by graph_writer.upsert_nodes.

Creates:

  * ``node_post_write_queue`` -- shared queue, JSONB per-analyzer status.
    NO FORCE RLS (mirrors ``inferred_edges_queue`` post-0068) so the
    side-worker can drain cross-tenant via app.current_customer_id.

  * ``graph_nodes.embedding halfvec(3072)`` + HNSW cosine index --
    same dim as ``chunks.embedding_v2`` so the LiteLLM-gateway
    GeminiEmbedder output writes in-place. Nullable initially; the
    backfill script fills existing 8606 rows.

  * Trigram GIN indexes on ``LOWER(canonical_id)`` and
    ``LOWER(properties->>'name')`` for the AutoMergeAnalyzer's
    candidate-generation pre-filter (pg_trgm is already enabled per
    db/schema.sql).

  * ``entity_merge_suggestions`` -- medium/low-confidence verdicts
    surfaced in the dashboard's /graph cluster admin UI. FORCE RLS
    (tenant-isolated for the app role; admin BFF reads with the
    customer context set).

See: entity auto-merge plan, /plan-eng-review 2026-05-19.

Revision ID: 0082_node_post_write_pipeline
Revises: 0081_incident_mcp_servers
Create Date: 2026-05-19
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0082_node_post_write_pipeline"
down_revision: str | Sequence[str] | None = "0081_incident_mcp_servers"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


_TENANT_EXPR = "customer_id = current_setting('app.current_customer_id', true)"


def upgrade() -> None:
    # ----- node_post_write_queue (shared queue, NO FORCE RLS) -----
    op.execute(
        """
        CREATE TABLE node_post_write_queue (
            customer_id      TEXT NOT NULL
                             REFERENCES customers(customer_id) ON DELETE CASCADE,
            node_id          BIGINT NOT NULL,
            enqueued_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            analyzer_status  JSONB NOT NULL DEFAULT '{}'::JSONB,
            locked_until     TIMESTAMPTZ NULL,
            PRIMARY KEY (customer_id, node_id)
        )
        """
    )
    # Partial index that the drain loop's FOR UPDATE SKIP LOCKED selects against.
    # Worker selects rows where locked_until IS NULL OR locked_until < NOW();
    # the index covers the IS NULL branch (the common case — locked_until is
    # only set during active processing) and the small live-locked set
    # seq-scans cheaply. Partial-index predicates must be IMMUTABLE so NOW()
    # can't live in the WHERE clause.
    op.execute(
        """
        CREATE INDEX idx_node_post_write_queue_pending
        ON node_post_write_queue (enqueued_at)
        WHERE locked_until IS NULL
        """
    )
    # NO FORCE RLS pattern (mirrors inferred_edges_queue post-migration 0068):
    # tenant policy is set up so app role gets per-tenant isolation when
    # app.current_customer_id is set, but the worker (probe_app) can also
    # poll cross-tenant by unsetting the GUC. Bypass for table owner / no
    # tenant context.
    op.execute("ALTER TABLE node_post_write_queue ENABLE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY node_post_write_queue_tenant_isolation
            ON node_post_write_queue
            USING (customer_id = current_setting('app.current_customer_id', true))
            WITH CHECK (customer_id = current_setting('app.current_customer_id', true))
        """
    )
    op.execute("ALTER TABLE node_post_write_queue OWNER TO probe")
    op.execute("GRANT ALL ON node_post_write_queue TO probe_app, probe_admin")

    # ----- graph_nodes.embedding column + HNSW index -----
    op.execute("ALTER TABLE graph_nodes ADD COLUMN embedding halfvec(3072) NULL")
    # HNSW with cosine ops (halfvec_cosine_ops indexes up to 4000 dims per
    # pgvector — matches the pattern already used on chunks.embedding_v2).
    op.execute(
        """
        CREATE INDEX idx_graph_nodes_embedding_hnsw
        ON graph_nodes USING hnsw (embedding halfvec_cosine_ops)
        """
    )

    # ----- Trigram GIN indexes on graph_nodes (pg_trgm confirmed live in prod) -----
    # Defensive CREATE EXTENSION in case a fresh DB doesn't have it; no-op if
    # already enabled.
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    op.execute(
        """
        CREATE INDEX idx_graph_nodes_canonical_id_trgm
        ON graph_nodes USING gin (LOWER(canonical_id) gin_trgm_ops)
        """
    )
    op.execute(
        """
        CREATE INDEX idx_graph_nodes_name_trgm
        ON graph_nodes USING gin (LOWER(properties->>'name') gin_trgm_ops)
        """
    )

    # ----- entity_merge_suggestions (FORCE RLS, dashboard-surfaced) -----
    op.execute(
        """
        CREATE TABLE entity_merge_suggestions (
            suggestion_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            customer_id            TEXT NOT NULL
                                   REFERENCES customers(customer_id) ON DELETE CASCADE,
            label                  TEXT NOT NULL,
            primary_canonical_id   TEXT NOT NULL,
            candidate_canonical_id TEXT NOT NULL,
            confidence             TEXT NOT NULL
                                   CHECK (confidence IN ('high','medium','low')),
            rationale              TEXT NULL,
            llm_model              TEXT NOT NULL,
            run_id                 UUID NULL,
            status                 TEXT NOT NULL DEFAULT 'pending'
                                   CHECK (status IN ('pending','approved','dismissed','applied')),
            created_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            decided_at             TIMESTAMPTZ NULL,
            decided_by_user_id     UUID NULL
        )
        """
    )
    op.execute(
        """
        CREATE INDEX idx_entity_merge_suggestions_lookup
        ON entity_merge_suggestions (customer_id, status, created_at DESC)
        """
    )
    # Unique on (customer, label, primary, candidate) for pending rows only,
    # so re-running the analyzer on the same pair doesn't double-insert.
    op.execute(
        """
        CREATE UNIQUE INDEX uq_entity_merge_suggestions_pair
        ON entity_merge_suggestions (customer_id, label, primary_canonical_id, candidate_canonical_id)
        WHERE status = 'pending'
        """
    )
    op.execute("ALTER TABLE entity_merge_suggestions ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE entity_merge_suggestions FORCE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY entity_merge_suggestions_tenant_isolation
            ON entity_merge_suggestions
            USING (customer_id = current_setting('app.current_customer_id', true))
            WITH CHECK (customer_id = current_setting('app.current_customer_id', true))
        """
    )
    op.execute("ALTER TABLE entity_merge_suggestions OWNER TO probe")
    op.execute("GRANT ALL ON entity_merge_suggestions TO probe_app, probe_admin")


def downgrade() -> None:
    # Reverse order: suggestions table, trigram indexes, embedding column +
    # index, queue table. pg_trgm extension stays — other consumers depend on it.
    op.execute("DROP TABLE IF EXISTS entity_merge_suggestions")
    op.execute("DROP INDEX IF EXISTS idx_graph_nodes_name_trgm")
    op.execute("DROP INDEX IF EXISTS idx_graph_nodes_canonical_id_trgm")
    op.execute("DROP INDEX IF EXISTS idx_graph_nodes_embedding_hnsw")
    op.execute("ALTER TABLE graph_nodes DROP COLUMN IF EXISTS embedding")
    op.execute("DROP TABLE IF EXISTS node_post_write_queue")
