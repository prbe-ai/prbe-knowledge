"""chunks_title_bm25: denormalize documents.title onto chunks; index it for BM25

Revision ID: 0100_chunks_title_bm25
Revises: 0099_documents_title_tsv
Create Date: 2026-08-05

Phase 2 of the retrieval-latency work. Phase 1 (#454) made the ANN path
index-usable; this makes the KEYWORD path index-usable, by moving BM25 off the
hand-rolled `ts_rank_cd` ranker and onto the pg_search index that has been
installed and unused since it was created.

WHY A COLUMN AND NOT A JOIN
---------------------------
`bm25.py` matched titles with:

    (c.content_tsv @@ q) OR (d.title_tsv @@ q AND c.chunk_index = 0)

An OR across two JOINed tables cannot be served by any single index -- the
planner pushes the chunks-side disjunct down as `(content_tsv @@ q OR
chunk_index = 0)`, which nothing covers, and seq-scans 4.4 GB. Measured on the
managed data plane, 204,637 chunks:

    ts_rank_cd, cross-table OR      30,000 ms (statement timeout)
    ts_rank_cd, OR rewritten UNION   2,997 ms
    pg_search @@@ (title on chunk)     191 ms   TopKScanExecState

Denormalizing the title onto the chunk makes the whole query single-table, so
the cross-table OR that caused this becomes structurally impossible to write.

THE DEDUPE IS LOAD-BEARING
--------------------------
`chunks JOIN documents ON version BETWEEN first_seen_version AND
last_seen_version` is NOT 1:1. Measured on this database:

    join rows                   254,697
    distinct chunk_id           209,218
    chunk_ids with >1 title          78

A chunk spans a version RANGE, and a document can be retitled inside that
range, so 78 chunks legitimately have two candidate titles. A naive
`UPDATE ... FROM documents` would pick one arbitrarily and be non-deterministic
across re-runs. `DISTINCT ON (chunk_id) ... ORDER BY d.version DESC` pins it to
the newest title in the chunk's range, which is the one a searcher means.

CONSISTENCY IS ENFORCED, NOT OBSERVED
-------------------------------------
A denormalized column needs a rule, not a test. Two triggers, because there are
two ways to get it wrong and they are different:

  * `documents` UPDATE OF title -> re-stamp that document's chunks. This is the
    path that actually drifts: normalizer.py:1327 updates a title IN PLACE
    without rewriting chunks, so content-identical chunks survive a retitle and
    would keep the old title forever.
  * `chunks` BEFORE INSERT with a NULL title -> fill from `documents`. The hot
    ingest path sets it explicitly (cheaper, no per-row lookup), but migrations,
    backfill scripts and manual SQL do not, and those are exactly the callers
    nobody remembers to update.

Together they mean a chunk cannot exist with a stale or missing title regardless
of which door it came through.

COST
----
Measured by building the equivalent table on production data: 209k rows
backfilled in 14.6 s, bm25 index built in 46 s. `ADD COLUMN` with no default is
a catalog-only change in PG11+, so the table is not rewritten and the
AccessExclusiveLock is held for milliseconds. The backfill is batched so no
single statement holds a long transaction over a 4.4 GB table.

Revision string is 22 chars, inside the 32-char alembic_version limit.
"""

from __future__ import annotations

from alembic import op

revision = "0100_chunks_title_bm25"
down_revision = "0099_documents_title_tsv"
branch_labels = None
depends_on = None

# Backfill batch size. Small enough that each statement is a short transaction
# on a 4.4 GB table, large enough that 209k rows is ~40 statements.
_BATCH = 5000


def upgrade() -> None:
    # Catalog-only in PG11+ (no DEFAULT), so no table rewrite.
    op.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS title text")

    # Batched backfill. DISTINCT ON picks the newest title in the chunk's
    # version range -- see the dedupe note in the docstring.
    op.execute(
        f"""
        DO $$
        DECLARE
            n integer;
        BEGIN
            LOOP
                WITH batch AS (
                    SELECT c.chunk_id
                    FROM chunks c
                    WHERE c.title IS NULL
                    LIMIT {_BATCH}
                ),
                resolved AS (
                    SELECT DISTINCT ON (c.chunk_id)
                           c.chunk_id,
                           coalesce(d.title, '') AS title
                    FROM chunks c
                    JOIN batch b ON b.chunk_id = c.chunk_id
                    LEFT JOIN documents d
                      ON d.doc_id = c.doc_id
                     AND d.customer_id = c.customer_id
                     AND d.version BETWEEN c.first_seen_version
                                       AND c.last_seen_version
                    ORDER BY c.chunk_id, d.version DESC
                )
                UPDATE chunks c
                SET title = r.title
                FROM resolved r
                WHERE c.chunk_id = r.chunk_id;

                GET DIAGNOSTICS n = ROW_COUNT;
                EXIT WHEN n = 0;
            END LOOP;
        END
        $$;
        """
    )

    # Backstop: a chunk with no title is a bug, and '' is the honest value for
    # "document has no title". NOT NULL makes the trigger below the only way a
    # chunk can be created, rather than one of two ways.
    op.execute("ALTER TABLE chunks ALTER COLUMN title SET DEFAULT ''")
    op.execute("UPDATE chunks SET title = '' WHERE title IS NULL")
    op.execute("ALTER TABLE chunks ALTER COLUMN title SET NOT NULL")

    # --- consistency enforcement -------------------------------------------
    # Path 1: an in-place retitle. normalizer.py:1327 updates documents.title
    # without touching chunks, so without this the chunk keeps the old title
    # until the document's CONTENT changes.
    op.execute(
        """
        CREATE OR REPLACE FUNCTION chunks_sync_title_from_document()
        RETURNS trigger AS $$
        BEGIN
            UPDATE chunks c
            SET title = coalesce(NEW.title, '')
            WHERE c.customer_id = NEW.customer_id
              AND c.doc_id = NEW.doc_id
              AND NEW.version BETWEEN c.first_seen_version AND c.last_seen_version
              AND c.title IS DISTINCT FROM coalesce(NEW.title, '');
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    op.execute("DROP TRIGGER IF EXISTS trg_chunks_sync_title ON documents")
    op.execute(
        """
        CREATE TRIGGER trg_chunks_sync_title
        AFTER UPDATE OF title ON documents
        FOR EACH ROW
        WHEN (OLD.title IS DISTINCT FROM NEW.title)
        EXECUTE FUNCTION chunks_sync_title_from_document();
        """
    )

    # Path 2: a chunk inserted by something that does not know about the
    # column -- a backfill script, a migration, manual SQL. The hot ingest
    # path passes the title explicitly and skips this lookup entirely.
    op.execute(
        """
        CREATE OR REPLACE FUNCTION chunks_fill_title_on_insert()
        RETURNS trigger AS $$
        BEGIN
            IF NEW.title IS NULL OR NEW.title = '' THEN
                SELECT coalesce(d.title, '')
                INTO NEW.title
                FROM documents d
                WHERE d.doc_id = NEW.doc_id
                  AND d.customer_id = NEW.customer_id
                  AND d.version BETWEEN NEW.first_seen_version
                                    AND NEW.last_seen_version
                ORDER BY d.version DESC
                LIMIT 1;
                NEW.title := coalesce(NEW.title, '');
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    op.execute("DROP TRIGGER IF EXISTS trg_chunks_fill_title ON chunks")
    op.execute(
        """
        CREATE TRIGGER trg_chunks_fill_title
        BEFORE INSERT ON chunks
        FOR EACH ROW
        EXECUTE FUNCTION chunks_fill_title_on_insert();
        """
    )

    # --- the index this all exists for --------------------------------------
    # Replaces idx_chunks_bm25, which indexed content but NOT title and was
    # therefore unusable for the title half of the query.
    #
    # The DROP is mandatory, not tidying. pg_search enforces ONE bm25 index per
    # relation:
    #
    #     ERROR: a relation may only have one `USING bm25` index
    #
    # so creating the replacement alongside the old one fails outright.
    # Reproduced on the prod-parity image (pg_search 0.23.4) before writing
    # this. Dropping first leaves a window with no bm25 index on `chunks`,
    # which is safe here for a specific reason rather than by assumption:
    # idx_chunks_bm25 has idx_scan = 0 and pg_stat_database.stats_reset is
    # NULL, so it has never been read in this database's entire history. No
    # query plan can regress on its removal because no plan uses it.
    #
    # Ordering with the app image: this migration is a pre-upgrade hook, so the
    # new index exists before any pod serving `@@@` starts. Rolling the app
    # forward without the migration would 500 every BM25 query.
    with op.get_context().autocommit_block():
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS idx_chunks_bm25")
        op.execute(
            """
            DO $$
            BEGIN
                IF EXISTS (
                    SELECT 1 FROM pg_class c JOIN pg_index i ON i.indexrelid=c.oid
                    WHERE c.relname='idx_chunks_bm25_v2' AND NOT i.indisvalid
                ) THEN
                    EXECUTE 'DROP INDEX idx_chunks_bm25_v2';
                END IF;
            END
            $$;
            """
        )
        op.execute(
            """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_chunks_bm25_v2
            ON chunks USING bm25 (
                chunk_id, content, title, customer_id, doc_id, kind,
                chunk_index, first_seen_version, last_seen_version, visibility
            )
            WITH (key_field=chunk_id)
            """
        )


def downgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS idx_chunks_bm25_v2")
        # Restore the pre-Phase-2 index. Content-only, matching what 0100
        # replaced -- the one-bm25-index-per-relation rule means this can only
        # be created after the v2 drop above.
        op.execute(
            """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_chunks_bm25
            ON chunks USING bm25 (
                chunk_id, content, customer_id, doc_id, kind,
                first_seen_version, last_seen_version
            )
            WITH (key_field=chunk_id)
            """
        )
    op.execute("DROP TRIGGER IF EXISTS trg_chunks_fill_title ON chunks")
    op.execute("DROP TRIGGER IF EXISTS trg_chunks_sync_title ON documents")
    op.execute("DROP FUNCTION IF EXISTS chunks_fill_title_on_insert()")
    op.execute("DROP FUNCTION IF EXISTS chunks_sync_title_from_document()")
    # Ordering: retrieval pods running the new code select c.title, so roll the
    # app image back BEFORE dropping the column.
    op.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS title")
