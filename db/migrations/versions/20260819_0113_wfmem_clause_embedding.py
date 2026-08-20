"""store each clause's body embedding so neighbour search stops re-embedding the corpus

Revision ID: 0113_wfmem_clause_embedding
Revises: 0112_wfmem_clause_publication
Create Date: 2026-08-19

WHAT THIS REPLACES. `find_neighbours` fetched up to 500 clause rows on every
`/procedures/preview`, embedded all 500 plus the draft in one batch, computed
500 cosines, and threw every vector away -- then did it again on the next
declaration. Two costs, and the second is the one that mattered:

  * Latency and spend: ~500x the embedding work the question needs, added to a
    call where a human is already waiting on a structuring pass.
  * A SILENT CORRECTNESS CEILING. The `LIMIT 500` was ordered `updated_at DESC`,
    so past 500 clauses the oldest ones could never surface as neighbours. A
    feature whose entire job is preventing duplicates would quietly stop seeing
    the rules most likely to have been forgotten. Nothing errors; the duplicate
    just gets written.

With the vector stored, preview embeds ONE text and runs an indexed
nearest-neighbour query over the whole corpus. The ceiling disappears rather
than moving.

`halfvec(3072)` + HNSW `halfvec_cosine_ops` is the house pattern -- `chunks`,
`directed_vectors` and `graph_nodes` all use it, and `chunks` also carries an
`embedding_v2_model` column, which is the guard copied here.

WHY THE MODEL COLUMN IS NOT OPTIONAL. A cosine between vectors from two
different embedders is a plausible-looking number rather than an error, so a
row embedded by an older model must be EXCLUDED from a search rather than
silently compared. Same reasoning that puts `model_id` in the classifier's
vocabulary cache key. The CHECK keeps the pair together: a vector with no model
id cannot be safely used, and a model id with no vector says nothing.

WHY A TRIGGER CLEARS THE VECTOR ON A BODY EDIT. A stored embedding goes stale
the moment the text it describes changes, and staleness here is invisible --
the search still returns results, just wrong ones. `wfmem_clear_stale_clause_
embedding` sets both columns NULL when `body` changes, so a stale vector can
never be compared against. The clause drops out of neighbour search until
something re-embeds it, which is the safe direction to fail. Nothing in v0
updates a body (there is no edit path yet); this exists so that when one is
added, it cannot introduce the bug by omission.

NO BACKFILL. Existing clauses keep NULL and are simply absent from neighbour
results until re-embedded. That is a visible gap rather than a wrong answer,
and Phase 0 corpora are small enough that re-declaring is cheaper than a
backfill job nobody has needed yet.
"""

from __future__ import annotations

from alembic import op

revision = "0113_wfmem_clause_embedding"
down_revision = "0112_wfmem_clause_publication"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE clauses
            ADD COLUMN body_embedding halfvec(3072),
            ADD COLUMN body_embedding_model TEXT
        """
    )
    op.execute(
        """
        ALTER TABLE clauses
            ADD CONSTRAINT ck_clauses_embedding_names_its_model
            CHECK ((body_embedding IS NULL) = (body_embedding_model IS NULL))
        """
    )
    op.execute(
        """
        CREATE INDEX clauses_body_embedding_hnsw
            ON clauses USING hnsw (body_embedding halfvec_cosine_ops)
        """
    )
    op.execute(
        """
        CREATE OR REPLACE FUNCTION wfmem_clear_stale_clause_embedding()
        RETURNS TRIGGER AS $$
        BEGIN
            IF NEW.body IS DISTINCT FROM OLD.body THEN
                NEW.body_embedding := NULL;
                NEW.body_embedding_model := NULL;
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql
        """
    )
    op.execute(
        """
        CREATE TRIGGER clauses_clear_stale_embedding_trg
            BEFORE UPDATE ON clauses
            FOR EACH ROW
            EXECUTE FUNCTION wfmem_clear_stale_clause_embedding()
        """
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS clauses_clear_stale_embedding_trg ON clauses")
    op.execute("DROP FUNCTION IF EXISTS wfmem_clear_stale_clause_embedding()")
    op.execute("DROP INDEX IF EXISTS clauses_body_embedding_hnsw")
    op.execute(
        "ALTER TABLE clauses DROP CONSTRAINT IF EXISTS ck_clauses_embedding_names_its_model"
    )
    op.execute("ALTER TABLE clauses DROP COLUMN IF EXISTS body_embedding_model")
    op.execute("ALTER TABLE clauses DROP COLUMN IF EXISTS body_embedding")
