r"""bm25_title_tokenizer: tokenize chunks.title as source_code, not prose

Revision ID: 0102_bm25_title_tok
Revises: 0101_index_parity
Create Date: 2026-08-05

0100 indexed `chunks.title` with pg_search's default tokenizer. That tokenizer
splits on `/` but NOT on `.`, which silently breaks the exact case titles were
made searchable for. Measured on a title of `checkpoints/model.ckpt`:

    paradedb.match('title', 'checkpoints')  -> 1 hit
    paradedb.match('title', 'model')        -> 0 hits
    paradedb.match('title', 'ckpt')         -> 0 hits

So the indexed terms are `checkpoints` and `model.ckpt`, and a user typing
`model.ckpt` gets nothing, because the retriever tokenizes the QUERY on
alphanumeric runs (`_TOKEN_RE`) and asks for `model ckpt`.

This is not a new problem, it is the same one migration 0099 already solved on
the tsvector side, and its docstring says so explicitly: Postgres' `english`
parser also emits `model.ckpt` as one lexeme, so 0099 stores the title twice --
verbatim AND with `./\-_:` translated to spaces. Moving BM25 to pg_search
without carrying that over would have regressed filename search relative to the
ranker being replaced.

`source_code` is pg_search's tokenizer for identifier-shaped text. Verified on
the prod-parity image, same index, all four cases:

    model        -> 1     (dotted filename part, the regression)
    ckpt         -> 1     (dotted filename part, the regression)
    retry        -> 1     (ordinary prose title "Fix the retry loop")
    model.ckpt   -> 1     (verbatim form still matches)
    MODEL        -> 1     (case-insensitive)

so it is strictly better here than the default, not a trade.

Only `title` gets it. `content` stays on the default tokenizer: it is prose, it
is 1.2 KB per chunk on average, and identifier-splitting every code fence in
every transcript would inflate the term dictionary for no recall the title
field does not already provide.

Rebuilding rather than altering because pg_search has no in-place tokenizer
change, and the one-index-per-relation rule means the drop must come first --
creating the replacement alongside raises "a relation may only have one
`USING bm25` index".

DEPLOY NOTE: this DEGRADES rather than breaks if it runs after the retriever
that depends on it (managed's migrate job is a post-upgrade hook). Between the
pods rolling and this migration completing, title matching works for ordinary
words and misses dotted filenames -- exactly today's pre-Phase-2 behaviour.
Content matching is unaffected throughout.

Revision string is 18 chars, inside the 32-char alembic_version limit.
"""

from __future__ import annotations

from alembic import op

revision = "0102_bm25_title_tok"
down_revision = "0101_index_parity"
branch_labels = None
depends_on = None

_FIELDS = """
    chunk_id, content, title, customer_id, doc_id, kind,
    chunk_index, first_seen_version, last_seen_version, visibility
"""


def upgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS idx_chunks_bm25_v2")
        op.execute(
            f"""
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_chunks_bm25_v2
            ON chunks USING bm25 ({_FIELDS})
            WITH (
                key_field=chunk_id,
                text_fields='{{"title": {{"tokenizer": {{"type": "source_code"}}}}}}'
            )
            """
        )


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS idx_chunks_bm25_v2")
        op.execute(
            f"""
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_chunks_bm25_v2
            ON chunks USING bm25 ({_FIELDS})
            WITH (key_field=chunk_id)
            """
        )
