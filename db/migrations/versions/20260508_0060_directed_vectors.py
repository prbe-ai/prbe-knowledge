"""directed_vectors: per-doc trigger phrases for retrieval booster

Revision ID: 0060_directed_vectors
Revises: 0059_agent_runs_model_col
Create Date: 2026-05-08

Wiki pages need to be retrievable by semantic match against engineer-pinned
or LLM-generated *trigger phrases* — short strings describing problems /
situations a page should be surfaced for. Phrases are stored as embeddings
in this new table; the directed retriever joins them to documents and
contributes a doc-level booster to RRF fusion (see services/retrieval/
fusion.py and services/retrieval/retrievers/directed.py).

Authoring is hybrid:
  - source='human': engineer-pinned via wiki frontmatter `directed:` block.
                    synthesis_run_id is NULL. Never overwritten by the LLM.
  - source='llm':   LLM-generated from the page body during synthesis.
                    synthesis_run_id is the run that produced the row;
                    later runs delete prior runs' rows for the same doc.

Identity is (customer_id, doc_id, content_hash). The hash is computed
from the normalized phrase so the same engineer pin re-emitted across
runs is idempotent.

No FK to documents — same rationale chunks uses (chunks have no FK to
documents either): documents PK is (customer_id, doc_id, version) and
a directed_vector is doc-level, not version-level. ON DELETE CASCADE
flows through `customer_id REFERENCES customers` for tenant teardown,
which is the only delete path in normal operation. Doc-level cleanup
on doc-delete (rare) is handled by the synthesis path that owns the
doc, not by FK cascade.

Lessons reminder: revision string MUST be <=32 chars (alembic_version
column is varchar(32)); '0060_directed_vectors' is 21 chars - fine.
"""

from __future__ import annotations

from alembic import op

revision = "0060_directed_vectors"
down_revision = "0059_agent_runs_model_col"
branch_labels = None
depends_on = None


_HNSW_INDEX = "idx_directed_vectors_embedding_hnsw"
_LOOKUP_INDEX = "idx_directed_vectors_customer_doc"


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE directed_vectors (
            vector_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            customer_id      TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
            doc_id           TEXT NOT NULL,
            embedding        halfvec(3072) NOT NULL,
            source_text      TEXT NOT NULL,
            source           TEXT NOT NULL,
            synthesis_run_id BIGINT NULL,
            content_hash     BYTEA NOT NULL,
            created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CONSTRAINT ck_dv_source CHECK (source IN ('human','llm')),
            CONSTRAINT ck_dv_run_for_llm CHECK (
                (source = 'llm'   AND synthesis_run_id IS NOT NULL) OR
                (source = 'human' AND synthesis_run_id IS NULL)
            ),
            CONSTRAINT uq_dv_doc_hash UNIQUE (customer_id, doc_id, content_hash)
        )
        """
    )

    # HNSW for the per-query ANN lookup (one search per /retrieve). Uses
    # halfvec_cosine_ops to match the chunks.embedding index — we share
    # the same query embedding across both retrievers.
    op.execute(
        f"""
        CREATE INDEX {_HNSW_INDEX}
            ON directed_vectors USING hnsw (embedding halfvec_cosine_ops)
        """
    )

    # Btree for the post-fusion / synthesis-side per-doc lookups (e.g.
    # 'find all directed_vectors for this doc to delete on regen').
    op.execute(
        f"""
        CREATE INDEX {_LOOKUP_INDEX}
            ON directed_vectors (customer_id, doc_id)
        """
    )

    op.execute("ALTER TABLE directed_vectors ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE directed_vectors FORCE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY directed_vectors_tenant_isolation ON directed_vectors
            USING (customer_id = current_setting('app.current_customer_id', true))
            WITH CHECK (customer_id = current_setting('app.current_customer_id', true))
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS directed_vectors CASCADE")
