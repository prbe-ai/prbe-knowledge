"""graph_nodes: index on (customer_id, degree DESC) for /graph/explore.

Revision ID: 0064_graph_nodes_degree_idx
Revises: 0063_embedding_v2_hnsw
Create Date: 2026-05-10

POST /graph/explore (default mode) selects the top-N graph_nodes per tenant
ordered by `degree DESC`. Without a matching index the planner falls back
to a sequential scan over (customer_id, ...) and sorts in memory --
tolerable on small tenants, expensive once a customer crosses the
~100k-node mark (acme is already there for code_graph nodes).

`(customer_id, degree DESC)` matches the WHERE + ORDER BY shape exactly:
  WHERE customer_id = $1 ORDER BY degree DESC LIMIT $N

CONCURRENTLY because graph_nodes is hot (every webhook ingestion writes
into it via graph_writer); no point taking ACCESS EXCLUSIVE for a
read-side index. Pattern mirrors 0056_documents_id_trgm_idx and
0062_chunks_content_tsv -- autocommit_block + IF NOT EXISTS.

This migration creates an index only and does not touch any tenant rows,
so the NO FORCE / FORCE RLS toggle pattern (see
feedback_graph_nodes_rls_force) does not apply -- no UPDATE/DELETE on
graph_nodes is performed.

Lessons reminder: revision string MUST be <=32 chars (alembic_version
column is varchar(32)); '0064_graph_nodes_degree_idx' is 27 chars - fine.
"""

from __future__ import annotations

from alembic import op

revision = "0064_graph_nodes_degree_idx"
down_revision = "0063_embedding_v2_hnsw"
branch_labels = None
depends_on = None


INDEX_NAME = "idx_graph_nodes_customer_degree"


def upgrade() -> None:
    # CREATE INDEX CONCURRENTLY cannot run inside a transaction.
    # autocommit_block breaks alembic's transaction wrapper for these
    # statements only.
    with op.get_context().autocommit_block():
        # Recover from a prior partial run: a CREATE INDEX CONCURRENTLY
        # that was interrupted (timeout, OOM, killed by Fly's
        # release-command deadline) leaves an INVALID index on disk.
        # Plain `IF NOT EXISTS` below would skip by name alone and never
        # rebuild, leaving the INVALID index in place -- queries hit
        # planner-ignored garbage and silently fall back to seq scan.
        op.execute(
            f"""
            DO $$
            BEGIN
                IF EXISTS (
                    SELECT 1
                    FROM pg_class c
                    JOIN pg_index i ON i.indexrelid = c.oid
                    WHERE c.relname = '{INDEX_NAME}'
                      AND NOT i.indisvalid
                ) THEN
                    EXECUTE 'DROP INDEX {INDEX_NAME}';
                END IF;
            END
            $$;
            """
        )
        op.execute(
            f"""
            CREATE INDEX CONCURRENTLY IF NOT EXISTS {INDEX_NAME}
                ON graph_nodes (customer_id, degree DESC)
            """
        )


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {INDEX_NAME}")
