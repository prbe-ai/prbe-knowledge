"""Strip the redundant `body` key from documents.metadata.

Revision ID: 0036_strip_metadata_body
Revises: 0035_wiki_verifier_rejected
Create Date: 2026-05-03

WHY
---
Every connector was stuffing the full normalized body into
`documents.metadata['body']` — ~440 MB raw / ~180 MB TOASTed of pure
duplication. The canonical source of truth for full text is
`chunks.content` (joined by chunk_index for the live version). The
metadata.body key was originally added to feed
`Normalizer._stringify_body` and stayed by convention; with that
function now reading `Document.body` (a transient field), the key has
no remaining reader and can be removed.

This migration is the backfill: it strips `metadata.body` from every
existing row in the `documents` table. The corresponding handler change
prevents new rows from accumulating it.

WHAT
----
`UPDATE documents SET metadata = metadata - 'body' WHERE metadata ?
'body'`. The jsonb minus operator removes the key without touching any
sibling keys.

The `documents` table has neither RLS nor FORCE RLS (verified against
schema.sql + the 0002_force_rls migration), so a global UPDATE under
the migration's role hits every row. No NO FORCE / FORCE wrapping is
required (unlike graph_nodes/graph_edges; see migration 0028 for the
contrast).

Storage reclaimed
-----------------
~440 MB raw text, ~180 MB after compression. The on-disk reclaim does
not happen until autovacuum runs (or `VACUUM FULL documents` is run
manually). VACUUM FULL takes an exclusive lock — schedule for a
maintenance window or use pg_repack for zero-downtime. Do NOT run it
inside this migration: Alembic wraps each migration in a single
transaction, and VACUUM FULL is not transaction-safe.

Manual followup after deploy
----------------------------
    VACUUM FULL documents;  -- or: pg_repack -t documents

IDEMPOTENT
----------
Running twice produces the same result. The WHERE clause excludes rows
that have already been stripped.

DOWNGRADE
---------
No-op. The body data is now sourced from chunks.content; reconstructing
it back into metadata.body would re-introduce the duplication this
migration eliminates. If a rollback is ever needed, the body can be
materialized on demand via the same chunk-join used by
fetch_live_body_from_chunks.
"""

from __future__ import annotations

from alembic import op

revision = "0036_strip_metadata_body"
down_revision = "0035_wiki_verifier_rejected"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        UPDATE documents
        SET metadata = metadata - 'body'
        WHERE metadata ? 'body'
        """
    )


def downgrade() -> None:
    # No-op. See module docstring.
    pass
