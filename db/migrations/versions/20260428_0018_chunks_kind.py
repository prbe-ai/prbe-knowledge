"""chunks: add `kind` column for content vs metadata distinction

Revision ID: 0018_chunks_kind
Revises: 0017_enrichment_retry_columns
Create Date: 2026-04-28

The retrieval pipeline needs to distinguish two kinds of chunks:

  - kind='content'  — body text from the source document, the existing
                      shape. Default for all current rows.
  - kind='metadata' — synthetic key:value text per document
                      (title, repo, author, source URL) generated at
                      ingestion. Embedded + FTS-indexed for search-path
                      ranking on metadata-keyed queries
                      ("what's going on with prbe-backend") that today
                      have nothing in chunk content to match.

The list pipeline's LATERAL join filters `kind='content'` so the
representative chunk per doc is always body, not synthetic text.
The vector / BM25 retrievers do NOT filter — metadata IS searchable;
that's the whole point. The fusion layer is kind-aware so a doc that
surfaces via its metadata chunk has its score combined with the doc's
best content chunk score, and only the content chunk reaches the
response (synthetic key:value text never escapes to agents).

DEFAULT 'content' on the existing rows means the migration is a
metadata-only schema change for the current corpus — Postgres 11+
records the default as a catalog-only fact and doesn't rewrite the
table. Old rows read back as 'content' without an UPDATE storm.

The partial index serves the backfill script's "skip docs that already
have a metadata chunk" idempotency check and the ingestion-side
duplicate-prevention. It's tiny (one row per doc).
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0018_chunks_kind"
down_revision = "0017_enrichment_retry_columns"
branch_labels = None
depends_on = None


INDEX_NAME = "idx_chunks_metadata_kind"


def upgrade() -> None:
    # ADD COLUMN with DEFAULT does not rewrite the table on PG 11+.
    op.add_column(
        "chunks",
        sa.Column(
            "kind",
            sa.Text(),
            nullable=False,
            server_default=sa.text("'content'"),
        ),
    )

    # Partial index for "find / skip metadata chunks" lookups. Tiny; one
    # row per doc maximum. CONCURRENTLY because chunks is the heaviest
    # table — never block writes mid-build.
    with op.get_context().autocommit_block():
        op.create_index(
            INDEX_NAME,
            "chunks",
            ["customer_id", "doc_id"],
            unique=False,
            postgresql_using="btree",
            postgresql_concurrently=True,
            postgresql_where="kind = 'metadata'",
            if_not_exists=True,
        )


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.drop_index(
            INDEX_NAME,
            table_name="chunks",
            postgresql_concurrently=True,
            if_exists=True,
        )
    op.drop_column("chunks", "kind")
