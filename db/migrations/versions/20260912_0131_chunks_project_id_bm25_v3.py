"""chunks_project_id_bm25_v3: denormalize project_id onto chunks; index it for BM25

Revision ID: 0131_chunks_project_id_bm25_v3
Revises: 0130_documents_project_id_expr
Create Date: 2026-09-12

Phase 2 of pre-search project scope. Phase 1 (#545) put `scope.project_id`
into every channel as a hard predicate; on BM25 it lands on the DOCUMENTS
join, after Tantivy has already chosen its TopK pool.

WHY A COLUMN AND NOT A JOIN
---------------------------
Exactly the argument migration 0100 made for `title`, and the reason that
migration is the template for this one.

`project_id` lives in `documents.metadata`, so the scope filter cannot ride
the BM25 index the way the tenant and visibility filters do. It is applied as
a per-candidate HEAP filter after TopK, which means a scoped query ranks the
whole tenant and then throws most of it away. The existing mitigation is
`_BM25_SCOPED_POOL_FACTOR = 4`: widen the pool 4x so the post-filter has
something left to keep. That is a workaround with a failure mode -- a project
holding a small enough share of a large tenant still empties a 4x pool, and
the answer is silently thin rather than wrong, which is the worst shape for a
recall bug.

Denormalizing project_id onto the chunk makes the scope expressible as a
Tantivy `must` clause, so it filters INDEX-side and TopK returns pool_size
rows that are already in scope.

ONE BM25 INDEX PER RELATION
---------------------------
pg_search permits exactly one `USING bm25` index per relation -- a second
raises "a relation may only have one `USING bm25` index". So adding a fast
field means DROP v2, CREATE v3. There is no add-a-column path, and no window
in which both exist.

That makes the index swap an OPERATIONAL EVENT, not a schema detail, and it is
deliberately NOT performed here. This migration adds the column, the
backfill and the triggers; `scripts/cron_pg_search_rebuild.py` owns the index
itself, under the guardian, on a planned schedule. Rebuilding ad hoc is how
the standby ends up with a 0-byte index it believes is valid (see that
script's docstring -- it is the best record of this hazard we have).

THE RETRIEVER WORKS EITHER WAY, AND THAT IS THE POINT
-----------------------------------------------------
Code deploys before the index is rebuilt, always. `bm25.py` detects which
index is present and picks its clause accordingly: with v3 the scope is a
`must` clause, with v2 it stays on the documents join exactly as today. If it
assumed v3 the moment this migration ran, every BM25 query would fail in the
window between deploy and rebuild -- which can be days, because the rebuild
waits for a window.

THE DEDUPE IS LOAD-BEARING
--------------------------
`chunks JOIN documents ON version BETWEEN first_seen_version AND
last_seen_version` is not 1:1 -- a chunk spans a version RANGE and a document
can be re-projected inside that range, so a chunk can have two candidate
project_ids. `DISTINCT ON (chunk_id) ... ORDER BY d.version DESC` pins it to
the newest, which is the one a searcher means. Same reasoning as 0100's title
dedupe, same shape.

CONSISTENCY IS ENFORCED, NOT OBSERVED
-------------------------------------
Two triggers, mirroring 0100: fill on chunk insert, sync on document update.
Without the sync trigger a document that moves between projects leaves every
existing chunk claiming the old one, and a scoped search returns documents
that are no longer in the project -- a scope LEAK, which is the failure this
whole line of work exists to prevent.
"""

from __future__ import annotations

from alembic import op

revision = "0131_chunks_project_id_bm25_v3"
down_revision = "0130_documents_project_id_expr"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Nullable with no default: NULL means "this chunk's document belongs to no
    # project", which is a real and common state, and is distinct from '' (a
    # project whose id is the empty string, which cannot happen but would
    # compare equal to a scope asking for it).
    op.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS project_id text")

    # Backfill. Newest document version in each chunk's range wins; see the
    # dedupe note above.
    op.execute(
        """
        UPDATE chunks c
           SET project_id = src.project_id
          FROM (
                SELECT DISTINCT ON (c2.chunk_id)
                       c2.chunk_id,
                       d.metadata->>'project_id' AS project_id
                  FROM chunks c2
                  JOIN documents d
                    ON d.doc_id = c2.doc_id
                   AND d.customer_id = c2.customer_id
                   AND d.version BETWEEN c2.first_seen_version
                                     AND c2.last_seen_version
                 WHERE d.metadata->>'project_id' IS NOT NULL
                 ORDER BY c2.chunk_id, d.version DESC
               ) src
         WHERE c.chunk_id = src.chunk_id
           AND c.project_id IS DISTINCT FROM src.project_id
        """
    )

    op.execute(
        """
        CREATE OR REPLACE FUNCTION chunks_fill_project_id_on_insert()
        RETURNS trigger AS $fn$
        BEGIN
            IF NEW.project_id IS NULL THEN
                SELECT d.metadata->>'project_id' INTO NEW.project_id
                FROM documents d
                WHERE d.doc_id = NEW.doc_id
                  AND d.customer_id = NEW.customer_id
                  AND d.version BETWEEN NEW.first_seen_version
                                    AND NEW.last_seen_version
                ORDER BY d.version DESC
                LIMIT 1;
            END IF;
            RETURN NEW;
        END;
        $fn$ LANGUAGE plpgsql;
        """
    )
    op.execute(
        """
        CREATE OR REPLACE FUNCTION chunks_sync_project_id_from_document()
        RETURNS trigger AS $fn$
        BEGIN
            UPDATE chunks c
               SET project_id = NEW.metadata->>'project_id'
             WHERE c.doc_id = NEW.doc_id
               AND c.customer_id = NEW.customer_id
               AND NEW.version BETWEEN c.first_seen_version
                                   AND c.last_seen_version
               AND c.project_id IS DISTINCT FROM NEW.metadata->>'project_id';
            RETURN NEW;
        END;
        $fn$ LANGUAGE plpgsql;
        """
    )
    op.execute("DROP TRIGGER IF EXISTS trg_chunks_fill_project_id ON chunks")
    op.execute(
        """
        CREATE TRIGGER trg_chunks_fill_project_id
            BEFORE INSERT ON chunks
            FOR EACH ROW
            EXECUTE FUNCTION chunks_fill_project_id_on_insert()
        """
    )
    op.execute("DROP TRIGGER IF EXISTS trg_chunks_sync_project_id ON documents")
    # Fires on any metadata write, not only on a project_id change: metadata is
    # a single jsonb column, so Postgres cannot give us `AFTER UPDATE OF
    # project_id`. The WHEN clause below narrows it to updates that actually
    # move the key, so an unrelated metadata edit costs one comparison and no
    # write.
    op.execute(
        """
        CREATE TRIGGER trg_chunks_sync_project_id
            AFTER UPDATE OF metadata ON documents
            FOR EACH ROW
            WHEN (OLD.metadata->>'project_id'
                  IS DISTINCT FROM NEW.metadata->>'project_id')
            EXECUTE FUNCTION chunks_sync_project_id_from_document()
        """
    )

    # A plain btree for the NON-bm25 paths (vector, sql, the fetch tools) that
    # also filter by project. Partial: rows with no project are the majority on
    # most tenants and are never selected by a project scope.
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_chunks_project_id
            ON chunks (customer_id, project_id)
         WHERE project_id IS NOT NULL
        """
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS trg_chunks_sync_project_id ON documents")
    op.execute("DROP TRIGGER IF EXISTS trg_chunks_fill_project_id ON chunks")
    op.execute("DROP FUNCTION IF EXISTS chunks_sync_project_id_from_document()")
    op.execute("DROP FUNCTION IF EXISTS chunks_fill_project_id_on_insert()")
    op.execute("DROP INDEX IF EXISTS idx_chunks_project_id")
    op.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS project_id")
