"""drop the team wiki: its compiled pages, their chunks, and its eight tables

Revision ID: 0110_drop_wiki
Revises: 0109_documents_title_preview_tsv
Create Date: 2026-08-19

The wiki is removed. Generation stopped fleet-wide on 2026-08-18, the HTTP
surface and the synthesis package are deleted in this release, and this is the
schema half.

WHY THE PURGE IS NOT OPTIONAL, and why it runs BEFORE the code it belongs to.
`SourceSystem.WIKI` leaves the enum in the same release. Four retrieval paths
build `SourceSystem(row["source_system"])` straight from a document row --
`engine/retrieval/main.py` (three) and `list_pipeline.py` -- with no guard, so a
surviving `source_system='wiki'` row turns every `/query`, `/retrieve` and
`get_source` that touches it into a 500. Migrations run as a pre-upgrade hook,
which is exactly the ordering that fixes it: the rows are gone before the code
that cannot parse them starts. Deleting the rows is the right half to fix
rather than widening those four response models to `str`, because the rows are
being deleted anyway and a widened model would outlive the reason for it.

NO EXPORT, DELIBERATELY. The decision (2026-08-18) is that generated wiki prose
is dead: it was LLM-distilled from sources that still exist, and the replacement
is agent-written notes with provenance at the moment of work. Nothing here is
archived first. That is a decision about THIS content, not a precedent.

WHAT THE COUNTS LOOKED LIKE when this was written, on the research plane's `kb`
database: 0 wiki documents, 0 directed_vectors, 0 rows in four of the eight
tables -- and 139,635 rows in `wiki_synthesis_queue` with 270 runs and 30 links.
That queue is the whole argument for dropping rather than leaving it: ingest
kept filling it on every webhook for tenants whose preference was still true,
and nothing has drained it since the workers went. A tenant with live pages
(the managed-shared plane had some) is handled by the same statements; they are
written to be correct at any count, including zero.

FORCE RLS. `documents` and `chunks` both have it, so the DELETEs bind the tenant
GUC per customer -- the `_for_each_customer` shape from 0108, which exists
because 0107 ran one unscoped UPDATE, matched nothing, reported success and let
alembic stamp the revision. DDL is not row-filtered, so the DROPs run once.

VERIFIED under production-shaped conditions before shipping, because the two
halves fail in opposite ways and neither failure is loud. On a scratch database
seeded with two tenants' wiki pages plus one non-wiki document, run by a role
with NOBYPASSRLS that OWNS the tables (which is what `app` is on the research
plane -- checked, `rolbypassrls = false`, and it owns all eight):

  * that role sees ZERO rows of `documents` with no GUC bound. That is the
    exact condition 0107 hit, and it is why the DELETEs loop per customer
    rather than running once.
  * after the migration: both tenants' wiki documents gone, the non-wiki
    document untouched, all eight tables dropped, revision stamped.

The DROPs need OWNERSHIP, not just privileges -- a non-owner gets "must be
owner of table", which is loud rather than silent, but would still fail a
deploy. `app` owns them today; a deployment whose migrator does not will stop
here instead of half-finishing.

DOWNGRADE REFUSES, and the function says why at length. The rows were deleted
rather than archived, so no migration can restore them; and an earlier draft
that recreated the tables "shape only" got the shapes wrong in four places,
which would have failed a rollback later and less legibly than refusing does.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0110_drop_wiki"
down_revision = "0109_documents_title_preview_tsv"
branch_labels = None
depends_on = None

#: Every table the wiki owned outright. `directed_vectors` is here despite its
#: neutral name: migration 0061 created it for "engineer-pinned wiki frontmatter
#: `directed:` blocks", its only retriever is deleted in this release, and both
#: of its backfill scripts are gone. Leaving it would strand a table whose
#: producer, consumer and rationale had all left.
_WIKI_TABLES = (
    "wiki_synthesis_queue",
    "wiki_synthesis_runs",
    "wiki_raw_data",
    "wiki_links",
    "wiki_timeline_entries",
    "wiki_page_settings",
    "wiki_append_idempotency",
    "directed_vectors",
)

#: A wiki document, by any of the three marks it carries. All three, not one:
#: `source_system` is what the retrieval models choke on, `doc_type` is what the
#: routes addressed, and `doc_class` is what the compiler stamped. A row that
#: carries only one of them is still a wiki row and still breaks the same read.
_WIKI_DOCS = """
    source_system = 'wiki'
    OR doc_type LIKE 'wiki.%'
    OR doc_class = 'compiled_wiki'
"""


def _for_each_customer(sql: str) -> None:
    """Run `sql` once per customer with the tenant GUC bound to that customer.

    `set_config(..., false)` -- session scope, not transaction scope. Alembic
    runs the whole migration in one transaction, and a `true` here would reset
    the GUC at the first statement boundary, putting us straight back to the
    NULL that made 0107 a no-op.
    """
    conn = op.get_bind()
    bind_tenant = sa.text("SELECT set_config('app.current_customer_id', :customer_id, false)")
    customers = [
        row[0] for row in conn.execute(sa.text("SELECT customer_id FROM customers")).fetchall()
    ]
    for customer_id in customers:
        conn.execute(bind_tenant, {"customer_id": customer_id})
        conn.execute(sa.text(sql))
    # Leave no tenant bound behind.
    conn.execute(bind_tenant, {"customer_id": ""})


def upgrade() -> None:
    # Chunks first: they reference the documents by (customer_id, doc_id), and
    # deleting the parents first would leave the child rows unreachable by this
    # predicate -- the orphan is the thing that would still be embedded and
    # still be searchable.
    _for_each_customer(
        f"""
        DELETE FROM chunks c
         WHERE EXISTS (
               SELECT 1 FROM documents d
                WHERE d.customer_id = c.customer_id
                  AND d.doc_id = c.doc_id
                  AND ({_WIKI_DOCS})
         )
        """
    )
    _for_each_customer(f"DELETE FROM documents WHERE {_WIKI_DOCS}")

    # DDL: not row-filtered, so once each. CASCADE because the sidecars carry
    # foreign keys at each other and the drop order should not be a puzzle.
    for table in _WIKI_TABLES:
        op.execute(f"DROP TABLE IF EXISTS {table} CASCADE")

    # The last wiki-shaped object in the live schema. It capped a live wiki
    # page's `body_size_bytes` and is vacuously true the moment no row can
    # carry `source_system='wiki'` -- so it breaks nothing either way, and that
    # is exactly why it has to go now rather than later: left in place it reads
    # as live policy to the next person who opens `documents`.
    op.execute("ALTER TABLE documents DROP CONSTRAINT IF EXISTS ck_wiki_live_page_size")


def downgrade() -> None:
    """Refuses. The wiki cannot be restored from here, and a half-restore is worse.

    This function used to recreate the eight tables behind an environment flag,
    "shape only, for a rollback to code that expects them to exist". Review
    found that the shapes were WRONG -- `queue_id` written as `id`,
    `src_wiki_type`/`src_slug` collapsed to `src_doc_id`, NOT-NULL columns
    (`text_sha256`, `version`, `entry_date`) missing, and the
    `uq_wsq_customer_doc_version` constraint absent, which is the exact target
    of pre-0110 `_enqueue_wiki_synthesis`'s `ON CONFLICT`. So the rollback the
    flag existed to enable would have failed on its first query anyway, with
    `UndefinedColumnError` instead of `UndefinedTable`.

    It also dropped FORCE RLS and the `tenant_isolation` policies that
    `directed_vectors`, `wiki_page_settings` and `wiki_append_idempotency`
    carried. Recreating those tables WITHOUT tenant isolation, for the
    NOBYPASSRLS `app` role that reads them, is a cross-tenant hole opened by a
    rollback -- the one moment nobody is looking for new holes.

    A schema that is close but wrong is worse than no schema: the first fails
    at the first query with a message about the real problem, the second fails
    later, further away, and looks like a code bug.

    TO ACTUALLY ROLL BACK: take the DDL from git, where it is exact and
    complete --

        git show <this-revision>^:db/schema.sql

    -- and restore the rows from a database backup. The rows were deleted, not
    archived, so no migration can bring them back.
    """
    raise RuntimeError(
        "0110_drop_wiki is not reversible. The wiki's pages, chunks and queue "
        "rows were deleted, not archived, and the eight tables cannot be "
        "recreated correctly from here -- their exact DDL, including FORCE RLS "
        "and the tenant_isolation policies, is in git: "
        "`git show <this-revision>^:db/schema.sql`. Restore rows from a "
        "database backup."
    )
