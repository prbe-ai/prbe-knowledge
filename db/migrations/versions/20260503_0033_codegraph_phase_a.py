"""CodeGraph PR-A precursor: confidence column + code_repo_state

Revision ID: 0033_codegraph_phase_a
Revises: 0032_manual_uploads
Create Date: 2026-05-03

Sets up the schema PR-A's CodeGraph connector requires:

  * graph_edges.confidence — text NOT NULL DEFAULT 'EXTRACTED' with a CHECK
    constraint covering the three tiers ('EXTRACTED','INFERRED','AMBIGUOUS').
    The default populates existing rows atomically — on PG 11+ this is a
    catalog-only fact, no table rewrite. PR-B (Graphify) writes 'INFERRED'
    and 'AMBIGUOUS' rows; PR-A's deterministic extractor stays at the default.
  * idx_graph_edges_confidence on (customer_id, edge_type, confidence) —
    backs the retrieval-side min_confidence filter.
  * code_repo_state — per-(customer, repo, file) extraction cache. The PR-A
    incremental-push flow short-circuits on content_hash match so steady-state
    pushes do zero re-embedding.

CONCURRENTLY on the new index because prbe-knowledge auto-deploys on push to
main; the migration runs against a live populated graph_edges table. A plain
CREATE INDEX would take SHARE lock and briefly block concurrent ingestion writes.

The ADD COLUMN with default is metadata-only and not row-touching, so the
NO FORCE / FORCE RLS toggle (per the documented graph_nodes/graph_edges
UPDATE pitfall) is not required here.
"""

from __future__ import annotations

from alembic import op

revision = "0033_codegraph_phase_a"
down_revision = "0032_manual_uploads"
branch_labels = None
depends_on = None


CONFIDENCE_INDEX = "idx_graph_edges_confidence"


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE graph_edges
            ADD COLUMN confidence TEXT NOT NULL DEFAULT 'EXTRACTED'
                CONSTRAINT graph_edges_confidence_check
                CHECK (confidence IN ('EXTRACTED', 'INFERRED', 'AMBIGUOUS'))
        """
    )

    with op.get_context().autocommit_block():
        op.execute(
            f"""
            CREATE INDEX CONCURRENTLY IF NOT EXISTS {CONFIDENCE_INDEX}
                ON graph_edges (customer_id, edge_type, confidence)
            """
        )

    op.execute(
        """
        CREATE TABLE code_repo_state (
            customer_id            TEXT NOT NULL,
            repo                   TEXT NOT NULL,
            file_path              TEXT NOT NULL,
            content_hash           TEXT NOT NULL,
            language               TEXT NOT NULL,
            symbol_count           INT  NOT NULL DEFAULT 0,
            last_extracted_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            last_extractor_version TEXT NOT NULL,
            PRIMARY KEY (customer_id, repo, file_path)
        )
        """
    )
    # Match the existing tenancy GUC (`app.current_customer_id`) rather than the
    # `app.tenant_id` name shown in the spec — `with_tenant()` only sets the
    # former, so a policy keyed on the latter would silently match nothing.
    # FORCE pairs with ENABLE so superusers don't bypass (mirrors the
    # usage_events / query_traces / graph_nodes convention).
    op.execute("ALTER TABLE code_repo_state ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE code_repo_state FORCE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY code_repo_state_tenant_isolation ON code_repo_state
            USING (customer_id = current_setting('app.current_customer_id', true))
        """
    )


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS code_repo_state_tenant_isolation ON code_repo_state")
    op.execute("DROP TABLE IF EXISTS code_repo_state")

    with op.get_context().autocommit_block():
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {CONFIDENCE_INDEX}")

    op.execute(
        "ALTER TABLE graph_edges DROP CONSTRAINT IF EXISTS graph_edges_confidence_check"
    )
    op.execute("ALTER TABLE graph_edges DROP COLUMN IF EXISTS confidence")
