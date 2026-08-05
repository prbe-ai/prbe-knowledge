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
    # NOT NULL DEFAULT '' in the SAME statement, and that is load-bearing.
    #
    # The obvious form -- bare ADD COLUMN, backfill, then SET NOT NULL -- has a
    # race that took production to find. ADD COLUMN is catalog-only, so a row
    # inserted by ingestion WHILE this long transaction runs is physically
    # missing the attribute; once this commits it reads as NULL, and the
    # `UPDATE ... WHERE title IS NULL` that was supposed to catch it already
    # ran. `SET NOT NULL` then fails:
    #
    #     NotNullViolation: column "title" of relation "chunks"
    #     contains null values
    #
    # Observed exactly that on the research plane, which had live ingestion
    # during the backfill. Managed survived the same migration purely on
    # timing, which is the worst kind of pass.
    #
    # A constant DEFAULT is still catalog-only in PG11+ (attmissingval, no
    # rewrite), and it applies to rows that lack the attribute -- including
    # ones inserted concurrently. So the constraint holds from the first
    # instant the column exists and there is no window to lose.
    op.execute(
        "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS title text NOT NULL DEFAULT ''"
    )

    # Batched backfill, walked by the PRIMARY KEY cursor (customer_id,
    # chunk_id) rather than by `WHERE title IS NULL`.
    #
    # Two reasons, both learned the hard way on the first production run:
    #
    #   * A `WHERE title <needs work>` predicate has no index, so EVERY batch
    #     re-scanned the whole 4.4 GB table -- 42 batches, 42 scans, ~15
    #     minutes for work that is seconds of actual writing.
    #   * Exiting on ROW_COUNT = 0 is wrong once the column defaults to ''.
    #     A batch whose documents genuinely have no title updates nothing, and
    #     the loop would stop there with the rest of the table unprocessed.
    #     Exiting on an EMPTY BATCH is the correct termination condition.
    #
    # `(customer_id, chunk_id) > (cursor)` matches chunks_pkey exactly, so each
    # batch is an index range scan on both data planes -- including research,
    # which does not have the single-column chunk_id index (0101 adds it, and
    # runs after this).
    op.execute(
        f"""
        DO $$
        DECLARE
            last_cust text := '';
            last_id   text := '';
            b_cust    text;
            b_id      text;
        BEGIN
            LOOP
                WITH batch AS (
                    SELECT c.customer_id, c.chunk_id, c.doc_id,
                           c.first_seen_version, c.last_seen_version
                    FROM chunks c
                    WHERE (c.customer_id, c.chunk_id) > (last_cust, last_id)
                    ORDER BY c.customer_id, c.chunk_id
                    LIMIT {_BATCH}
                ),
                resolved AS (
                    SELECT DISTINCT ON (b.customer_id, b.chunk_id)
                           b.customer_id, b.chunk_id,
                           coalesce(d.title, '') AS title
                    FROM batch b
                    LEFT JOIN documents d
                      ON d.doc_id = b.doc_id
                     AND d.customer_id = b.customer_id
                     AND d.version BETWEEN b.first_seen_version
                                       AND b.last_seen_version
                    ORDER BY b.customer_id, b.chunk_id, d.version DESC
                ),
                upd AS (
                    UPDATE chunks c
                    SET title = r.title
                    FROM resolved r
                    WHERE c.customer_id = r.customer_id
                      AND c.chunk_id = r.chunk_id
                      AND c.title IS DISTINCT FROM r.title
                    RETURNING 1
                )
                -- The LAST row in sort order, not max() of each column
                -- independently: with rows (custA,'z') and (custB,'a'),
                -- independent maxima give the cursor (custB,'z') and silently
                -- skip everything between (custB,'a') and (custB,'z').
                SELECT b.customer_id, b.chunk_id
                INTO b_cust, b_id
                FROM batch b
                ORDER BY b.customer_id DESC, b.chunk_id DESC
                LIMIT 1;

                EXIT WHEN b_id IS NULL;
                last_cust := b_cust;
                last_id   := b_id;
            END LOOP;
        END
        $$;
        """
    )

    # No SET NOT NULL step: the column was created NOT NULL above. Removing it
    # is the fix, not an omission -- see the race note at the ADD COLUMN.

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
