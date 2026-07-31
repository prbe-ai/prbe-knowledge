"""documents_title_tsv: make document titles reachable from BM25

Revision ID: 0099_documents_title_tsv
Revises: 0098_agent_session_names
Create Date: 2026-07-31

BM25 could not match a document's TITLE. `chunks.content_tsv` is a GENERATED
column, and a generation expression may only reference its own row, so it can
never reach `documents.title` — the two live in different tables. The exclusion
was a side effect of materializing the tsvector for speed (0062), not a choice:
nothing in the query ever read the title, and nothing said why.

The practical effect: a file named `model.ckpt`, or a PR titled "Fix the retry
loop", was findable by keyword ONLY if those words also appeared in the body.
Vector search did see titles — chunks embed as `title: {title} | text: {content}`
— but an exact filename is precisely the query where semantic similarity is
weakest and lexical matching should win.

This adds the mirror of 0062 on the documents side:
  - `title_tsv tsvector GENERATED ALWAYS AS (...) STORED`
  - GIN index `idx_documents_title_tsv`, built CONCURRENTLY

Weighted 'A' via setweight. `chunks.content_tsv` is unweighted, so it ranks at
the default 'D' (0.1) while a title match ranks 'A' (1.0) under ts_rank_cd's
default weights. That 10x is the whole point — someone typing a filename wants
the file, not a document that happens to mention it — and it comes from
Postgres' own weighting rather than a magic constant in the query.

`coalesce(title, '')` because title is nullable; a NULL title yields an empty
tsvector, which matches nothing and costs nothing.

Cost: documents is ~12k rows / 44 MB, so the ADD COLUMN rewrite is seconds, not
the minutes 0062 cost on chunks (~444 MB). AccessExclusiveLock is still held on
`documents` for that window and BM25/vector/ingestion all join it, so it is not
free — just short.

Revision string is 23 chars, inside the 32-char alembic_version limit.
"""

from __future__ import annotations

from alembic import op

revision = "0099_documents_title_tsv"
down_revision = "0098_agent_session_names"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ADD COLUMN GENERATED rewrites the table under AccessExclusiveLock. At
    # ~12k rows / 44 MB this is a seconds-long stall, unlike 0062's multi-minute
    # rewrite of chunks. Reads on `documents` queue for that window; every
    # retrieval channel joins it, so this is short but not invisible.
    #
    # setweight() and the two-argument to_tsvector(regconfig, text) are both
    # IMMUTABLE, which a generated column requires. The one-argument
    # to_tsvector(text) is only STABLE (it reads default_text_search_config)
    # and would be rejected here — hence the explicit 'english'.
    # The title is indexed TWICE over: once verbatim, once with path/filename
    # punctuation flattened to spaces. Both, not either.
    #
    # Postgres' `english` parser treats `model.ckpt` and `checkpoints/model.pt`
    # as SINGLE `file` tokens, but the BM25 retriever splits an incoming query
    # on alphanumeric runs (`_TOKEN_RE`), so searching "model.ckpt" asks for
    # `model | ckpt` and matches neither. Verified against this database:
    #
    #   to_tsvector('english','model.ckpt')             -> 'model.ckpt':1
    #   ... @@ to_tsquery('english','model | ckpt')     -> FALSE
    #
    # Indexing only the verbatim form would therefore have left filenames --
    # the exact case this migration exists for -- still unfindable. The
    # translate() pass adds `model` and `ckpt` alongside `model.ckpt`, so both
    # the split query and a verbatim one hit. translate() is IMMUTABLE, which
    # the generated column requires.
    op.execute(
        r"""
        ALTER TABLE documents
        ADD COLUMN IF NOT EXISTS title_tsv tsvector
        GENERATED ALWAYS AS (
            setweight(
                to_tsvector(
                    'english',
                    coalesce(title, '') || ' ' ||
                    translate(coalesce(title, ''), './\-_:', '      ')
                ),
                'A'
            )
        ) STORED
        """
    )

    # CREATE INDEX CONCURRENTLY cannot run inside a transaction.
    with op.get_context().autocommit_block():
        # Same recovery as 0062: an interrupted CREATE INDEX CONCURRENTLY
        # leaves an INVALID index behind, and a plain IF NOT EXISTS would skip
        # it by name and never rebuild — the planner ignores INVALID indexes,
        # so title search would silently fall back to a seq scan.
        op.execute(
            """
            DO $$
            BEGIN
                IF EXISTS (
                    SELECT 1
                    FROM pg_class c
                    JOIN pg_index i ON i.indexrelid = c.oid
                    WHERE c.relname = 'idx_documents_title_tsv'
                      AND NOT i.indisvalid
                ) THEN
                    EXECUTE 'DROP INDEX idx_documents_title_tsv';
                END IF;
            END
            $$;
            """
        )
        op.execute(
            """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_title_tsv
            ON documents USING gin(title_tsv)
            """
        )


def downgrade() -> None:
    # Ordering matters, same as 0062: a retrieval pod still running the new
    # code references `d.title_tsv` and will raise `column "title_tsv" does not
    # exist` the moment DROP COLUMN lands. Roll the app images back first,
    # confirm nothing references title_tsv, then downgrade.
    with op.get_context().autocommit_block():
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS idx_documents_title_tsv")
    op.execute("ALTER TABLE documents DROP COLUMN IF EXISTS title_tsv")
