"""documents: pg_trgm GIN index on title for grounding-time fuzzy match.

Revision ID: 0089_documents_title_trgm
Revises: 0088_inv_metadata
Create Date: 2026-05-20

The retrieval grounding step (services/retrieval/grounding.py) gains a
new channel `_fuzzy_match_document_titles` that probes documents.title
with pg_trgm `%` and tsvector FTS so concept queries (e.g.
"multi-granola", "shared-managed pivot") deterministically anchor on
the canonical doc instead of falling through to vector/BM25 lottery.

`idx_documents_fts_title_preview` already serves the tsvector half of
the predicate. The pg_trgm half currently has no supporting index on
`title` so similarity() against title is a sequential scan. This
migration adds a GIN trgm index. Without it the new channel still
works correctly but scans the full table per query (~6k rows today,
unbounded as tenants scale).

CONCURRENTLY is mandatory: the documents table grows unboundedly per
tenant and a default ACCESS EXCLUSIVE lock would take retrieval down
on any tenant past ~10M rows. CONCURRENTLY trades ~2-3x build time for
zero downtime — same call as 0015 (idx_documents_customer_source_doctype_updated)
and 0019 (graph_nodes_loose_match_indexes).

Follow-up: `idx_documents_fts_title_preview` is NOT partial on
`valid_to IS NULL`. The new channel's FTS branch will trigger a heap
recheck for every match to verify the live-row filter. Cost is
negligible at current scale (~6k docs) but worth revisiting past
~10M docs by adding a partial variant of the FTS index. Tracked as
[TODO: partial FTS index follow-up].
"""

from __future__ import annotations

from alembic import op

revision = "0089_documents_title_trgm"
down_revision = "0088_inv_metadata"
branch_labels = None
depends_on = None


INDEX_NAME = "idx_documents_title_trgm"


def upgrade() -> None:
    # CONCURRENTLY can't run inside a transaction. Alembic's autocommit_block
    # opens an autonomous block so the create_index doesn't crash on the
    # implicit BEGIN that wraps every migration.
    with op.get_context().autocommit_block():
        op.create_index(
            INDEX_NAME,
            "documents",
            ["title"],
            unique=False,
            postgresql_using="gin",
            postgresql_concurrently=True,
            postgresql_ops={"title": "gin_trgm_ops"},
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
