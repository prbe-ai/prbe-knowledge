"""documents: composite + partial index for the deterministic list path

Revision ID: 0015_documents_listing_index
Revises: 0014_enrichment_report_columns
Create Date: 2026-04-28

The retrieval service's new list pipeline runs queries shaped like

    SELECT ...
    FROM documents
    WHERE customer_id = $1
      AND source_system = ANY($2)
      AND doc_type = ANY($3)
      AND valid_to IS NULL
    ORDER BY updated_at DESC
    LIMIT $k

(plus aggregate variants: COUNT(*), GROUP BY author_id/source_system).

Existing indexes don't cleanly serve this:
  - idx_documents_customer_updated covers customer_id + sort but no source/type filter
  - idx_documents_customer_source covers source filter but not the sort
  - idx_documents_customer_class covers doc_type but not the sort

The planner picks customer_updated and post-filters; fast when matches are
dense, slow when sparse (e.g. a heavy-PR-week customer asking for 3 commits).
A composite + partial directly serves the list path and the aggregate paths.

CONCURRENTLY is mandatory: the documents table grows unboundedly per tenant
and a default ACCESS EXCLUSIVE lock would take the service down on any
tenant past ~10M rows. CONCURRENTLY trades ~2-3x build time for zero
downtime, which is the right call for a production index.
"""

from __future__ import annotations

from alembic import op

revision = "0015_documents_listing_index"
down_revision = "0014_enrichment_report_columns"
branch_labels = None
depends_on = None


INDEX_NAME = "idx_documents_customer_source_doctype_updated"


def upgrade() -> None:
    # CONCURRENTLY can't run inside a transaction. Alembic's autocommit_block
    # opens an autonomous block so the create_index doesn't crash on the
    # implicit BEGIN that wraps every migration.
    with op.get_context().autocommit_block():
        op.create_index(
            INDEX_NAME,
            "documents",
            ["customer_id", "source_system", "doc_type", "updated_at"],
            unique=False,
            postgresql_using="btree",
            postgresql_concurrently=True,
            postgresql_ops={"updated_at": "DESC"},
            postgresql_where="valid_to IS NULL",
            if_not_exists=True,
        )


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.drop_index(
            INDEX_NAME,
            table_name="documents",
            postgresql_concurrently=True,
            if_exists=True,
        )
