"""index_parity: recreate indexes that exist on one data plane but not the other

Revision ID: 0101_index_parity
Revises: 0100_chunks_title_bm25
Create Date: 2026-08-05

Two data planes running the same alembic head had different indexes:

    object                      managed   research
    idx_documents_title_trgm      yes        NO
    chunks_chunk_id_unique        yes        NO

Both report `0099_documents_title_tsv`.

WHY THEY DIVERGED
-----------------
`scripts/migrate.py` bootstraps a FRESH database from `db/schema.sql` and then
`alembic stamp head` -- it deliberately does not replay the chain, because
migrations 0007+ duplicate what schema.sql already creates. Its own docstring
calls schema.sql "the canonical latest schema".

It is not. `idx_documents_title_trgm` is created by migration 0089 and appears
ZERO times in schema.sql. So any database born fresh after 0089 skips straight
past it, stamps head, and is permanently missing the index while reporting that
it has every migration.

The research `kb` database was born that way. Measured consequence, same query
on both planes:

    managed  (index present)   77 ms   BitmapOr across both branches
    research (index absent)   677 ms   bitmap scan on the wrong index

That is `grounding.py`'s doc_title subtask, which #454 had just optimised. The
optimisation landed on research and did nothing there, invisibly, because the
index it depends on was never created.

ADDITIVE ONLY
-------------
Everything here is CREATE. Nothing is dropped, on purpose:

`chunks_chunk_id_unique` is obsolete as a pg_search requirement (0.23.4 no
longer needs a single-column unique key_field -- verified by building a bm25
index on a table with no unique index at all). The tempting cleanup is to drop
its 36 MB from managed. But it is a UNIQUE index, so dropping it removes an
enforced guarantee that chunk_id stays unique, and it is not backed by a
constraint that would make that intent obvious to the next reader. Parity by
addition costs ~14 MB on research and removes nothing. Reversing that trade
later is one migration; un-corrupting duplicate chunk_ids is not.

The real fix for the drift CLASS is backporting migration-only objects into
schema.sql so fresh databases stop being born incomplete. Done alongside this
migration; this one repairs the databases that already exist.

Revision string is 17 chars, inside the 32-char alembic_version limit.
"""

from __future__ import annotations

from alembic import op

revision = "0101_index_parity"
down_revision = "0100_chunks_title_bm25"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.get_context().autocommit_block():
        # 0089's index, with the INVALID-recovery guard 0089 itself lacks.
        #
        # This matters here specifically: `IF NOT EXISTS` matches on NAME, so an
        # interrupted CONCURRENTLY build leaves an INVALID index that the
        # planner ignores and that every subsequent run silently skips. That is
        # a plausible second explanation for the research divergence, and the
        # guard costs one DO block.
        op.execute(
            """
            DO $$
            BEGIN
                IF EXISTS (
                    SELECT 1 FROM pg_class c JOIN pg_index i ON i.indexrelid=c.oid
                    WHERE c.relname='idx_documents_title_trgm' AND NOT i.indisvalid
                ) THEN
                    EXECUTE 'DROP INDEX idx_documents_title_trgm';
                END IF;
            END
            $$;
            """
        )
        # Expression and predicate must match 0089 EXACTLY -- an expression
        # index only serves the expression it was built on, and grounding.py
        # queries `d.title % $2` under `valid_to IS NULL`.
        op.execute(
            """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_title_trgm
            ON documents USING gin (title gin_trgm_ops)
            WHERE valid_to IS NULL
            """
        )

        op.execute(
            """
            DO $$
            BEGIN
                IF EXISTS (
                    SELECT 1 FROM pg_class c JOIN pg_index i ON i.indexrelid=c.oid
                    WHERE c.relname='chunks_chunk_id_unique' AND NOT i.indisvalid
                ) THEN
                    EXECUTE 'DROP INDEX chunks_chunk_id_unique';
                END IF;
            END
            $$;
            """
        )
        # Enforces the uniqueness that chunk_id already has in practice
        # (`{doc_id}:{prefix}{content_hash[:16]}`, 0 duplicates measured on
        # managed). Restores parity without removing a guarantee from managed.
        op.execute(
            """
            CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS chunks_chunk_id_unique
            ON chunks (chunk_id)
            """
        )


def downgrade() -> None:
    # Reverting parity means re-introducing drift, so this drops only what this
    # migration could have added. On managed both predate it and dropping them
    # here would undo state 0089 (and an out-of-band operator) created.
    with op.get_context().autocommit_block():
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS chunks_chunk_id_unique")
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS idx_documents_title_trgm")
