"""kg indexes + updated_at trigger: GIN on related, ivfflat on embeddings

Revision ID: 0033_kg_indexes
Revises: 0032_kg_candidates
Create Date: 2026-04-30

Fourth migration in the Phase 1 foundation of the debugging knowledge
graph (see docs/superpowers/specs/2026-04-29-debugging-knowledge-graph-design.md
§5.1, §6, §7.3). Adds the remaining required indexes deferred from the
table-creation migrations (0030, 0032) plus an updated_at trigger on
kg_classes.

Indexes:
  * kg_classes_related_gin — GIN on `frontmatter->'related'` with
    jsonb_path_ops, supporting the compute-on-read edge query
    "for each related id, fetch the target class" used by 1-hop
    expand (spec §6 step 5). jsonb_path_ops is the right choice over
    the default jsonb_ops because we only need containment (`@>`) on
    the related array, never key existence checks.
  * kg_classes_embedding_ivfflat — pgvector ivfflat with
    vector_cosine_ops on `signature_embedding`, supporting the
    classifier's cosine-similarity step over class signatures
    (spec §6 step 2). lists=100 is the de-facto floor for ivfflat
    and is appropriate for our expected scale (50-500 classes per
    tenant; sqrt(N) is the textbook target but pg's ivfflat doesn't
    perform well below 100 lists). Revisit if a tenant exceeds ~10K
    classes.
  * kg_candidates_notes_embedding_ivfflat — pgvector ivfflat with
    vector_cosine_ops on `notes_embedding`, supporting the layer-2
    dedup confirmation step (spec §7.3): on layer-1 hash collision,
    compare notes embeddings against pending candidates and only
    increment repeat_count if cosine > 0.85. Same lists=100 rationale.

The clustering index `kg_candidates_dedup` was already created by
0032_kg_candidates and is NOT re-added here.

Trigger:
  * kg_set_updated_at() function + kg_classes_updated trigger so
    every UPDATE on kg_classes refreshes `updated_at` automatically,
    without requiring callers to remember. The function is
    kg-prefixed to avoid colliding with any future generic
    set_updated_at() helper; no such helper exists in the schema
    today (verified via grep on schema.sql).

Out of scope for this migration:
  * RLS enable + tenant_isolation policies on kg_* tables (Task 5).

The vector extension is already created in db/schema.sql; no
CREATE EXTENSION needed here.

Why raw SQL via op.execute: same rationale as 0030/0031/0032 — keeps
the pattern consistent and sidesteps SQLAlchemy core's lack of
first-class pgvector / ivfflat / GIN-with-opclass support.
"""

from __future__ import annotations

from alembic import op

revision = "0033_kg_indexes"
down_revision = "0032_kg_candidates"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE INDEX kg_classes_related_gin
            ON kg_classes
            USING GIN ((frontmatter->'related') jsonb_path_ops)
        """
    )
    op.execute(
        """
        CREATE INDEX kg_classes_embedding_ivfflat
            ON kg_classes
            USING ivfflat (signature_embedding vector_cosine_ops)
            WITH (lists = 100)
        """
    )
    op.execute(
        """
        CREATE INDEX kg_candidates_notes_embedding_ivfflat
            ON kg_candidates
            USING ivfflat (notes_embedding vector_cosine_ops)
            WITH (lists = 100)
        """
    )
    op.execute(
        """
        CREATE OR REPLACE FUNCTION kg_set_updated_at()
        RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
            NEW.updated_at = now();
            RETURN NEW;
        END $$
        """
    )
    op.execute(
        """
        CREATE TRIGGER kg_classes_updated
            BEFORE UPDATE ON kg_classes
            FOR EACH ROW EXECUTE FUNCTION kg_set_updated_at()
        """
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS kg_classes_updated ON kg_classes")
    op.execute("DROP FUNCTION IF EXISTS kg_set_updated_at()")
    op.execute("DROP INDEX IF EXISTS kg_candidates_notes_embedding_ivfflat")
    op.execute("DROP INDEX IF EXISTS kg_classes_embedding_ivfflat")
    op.execute("DROP INDEX IF EXISTS kg_classes_related_gin")
