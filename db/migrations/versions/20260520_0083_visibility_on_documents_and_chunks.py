"""visibility_on_documents_and_chunks

Adds a ``visibility`` text column to ``documents`` + ``chunks`` with a
``'approved'`` server default. Existing rows keep the default value
(no backfill required). New wiki-artifact writes set ``'draft'``; the
review approve path flips them to ``'approved'`` in a single
transaction. Retrieval gains a default ``WHERE visibility = 'approved'``
filter in a later task.

Two partial indexes back the approved-only retrieval path:
- ``chunks_visibility_approved_idx`` (customer_id, doc_id) WHERE
  ``visibility = 'approved'`` — keeps retrieval's per-doc chunk fetch
  index-only after draft rows start appearing.
- ``documents_visibility_approved_idx`` (customer_id, doc_type) WHERE
  ``visibility = 'approved'`` — keeps the doc-type listing path from
  scanning draft rows.

Both indexes are built CONCURRENTLY: ``chunks`` and ``documents`` are
the two hottest tables in this database (every ingest writes them),
and we cannot take ACCESS EXCLUSIVE for a non-blocking-read index.
Pattern mirrors 0056_documents_id_trgm_idx, 0063_embedding_v2_hnsw,
and 0064_graph_nodes_degree_idx.

Revision ID: 0083_visibility_columns
Revises: 0081_incident_mcp_servers
Create Date: 2026-05-18
"""

import sqlalchemy as sa
from alembic import op

# revision identifiers
revision = "0083_visibility_columns"
down_revision = "0082_node_post_write_pipeline"
branch_labels = None
depends_on = None


CHUNKS_INDEX = "chunks_visibility_approved_idx"
DOCS_INDEX = "documents_visibility_approved_idx"


def upgrade() -> None:
    # ADD COLUMN with a server_default backfills every existing row to
    # 'approved' in a single rewrite under ACCESS EXCLUSIVE. With the
    # server_default keeping every existing row at 'approved', the new
    # column is safe to mark NOT NULL immediately.
    op.add_column(
        "documents",
        sa.Column(
            "visibility", sa.Text(), nullable=False, server_default="approved",
        ),
    )
    op.add_column(
        "chunks",
        sa.Column(
            "visibility", sa.Text(), nullable=False, server_default="approved",
        ),
    )
    op.create_check_constraint(
        "documents_visibility_chk",
        "documents",
        "visibility IN ('draft','approved')",
    )
    op.create_check_constraint(
        "chunks_visibility_chk",
        "chunks",
        "visibility IN ('draft','approved')",
    )

    # CREATE INDEX CONCURRENTLY cannot run inside a transaction;
    # autocommit_block breaks alembic's wrapper for these statements
    # only. Same pattern as 0063_embedding_v2_hnsw and 0064_graph_nodes_degree_idx.
    with op.get_context().autocommit_block():
        op.execute(
            f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {CHUNKS_INDEX} "
            "ON chunks (customer_id, doc_id) WHERE visibility = 'approved'"
        )
        op.execute(
            f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {DOCS_INDEX} "
            "ON documents (customer_id, doc_type) WHERE visibility = 'approved'"
        )


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {CHUNKS_INDEX}")
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {DOCS_INDEX}")
    op.drop_constraint("chunks_visibility_chk", "chunks", type_="check")
    op.drop_constraint("documents_visibility_chk", "documents", type_="check")
    op.drop_column("chunks", "visibility")
    op.drop_column("documents", "visibility")
