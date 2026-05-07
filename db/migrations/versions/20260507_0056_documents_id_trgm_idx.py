"""documents: trigram GIN indexes for id_lookup suffix match

Revision ID: 0056_documents_id_trgm_idx
Revises: 0055_inferred_edge_metadata
Create Date: 2026-05-07

The id_lookup retriever (services/retrieval/retrievers/id_lookup.py)
matches handler-prefixed source_ids and doc_ids via leading-wildcard
LIKE:

    d.source_id LIKE '%:<canonical_id>'
    d.doc_id    LIKE '%:<canonical_id>'

A leading `%` defeats btree, so without these indexes the planner falls
back to a sequential scan over `documents` filtered only by customer_id.
Tolerable on small tenants, expensive once a customer carries a million+
docs.

`pg_trgm` GIN supports leading-wildcard LIKE by indexing trigrams of the
column value; the planner uses it for `LIKE '%foo'`, `LIKE '%foo%'`, and
`LIKE 'foo%'` alike. Extension already enabled in schema.sql.

CONCURRENTLY because `documents` is hot — no point taking ACCESS
EXCLUSIVE on a tenant ingesting webhooks right now.
"""

from __future__ import annotations

from alembic import op

revision = "0056_documents_id_trgm_idx"
down_revision = "0055_inferred_edge_metadata"
branch_labels = None
depends_on = None


SOURCE_ID_INDEX = "idx_documents_source_id_trgm"
DOC_ID_INDEX = "idx_documents_doc_id_trgm"


def upgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute(
            f"""
            CREATE INDEX CONCURRENTLY IF NOT EXISTS {SOURCE_ID_INDEX}
                ON documents USING GIN (source_id gin_trgm_ops)
            """
        )
        op.execute(
            f"""
            CREATE INDEX CONCURRENTLY IF NOT EXISTS {DOC_ID_INDEX}
                ON documents USING GIN (doc_id gin_trgm_ops)
            """
        )


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {DOC_ID_INDEX}")
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {SOURCE_ID_INDEX}")
