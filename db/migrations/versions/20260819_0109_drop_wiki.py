"""drop the team wiki: its compiled pages, their chunks, and its seven tables

Revision ID: 0109_drop_wiki
Revises: 0108_retire_pp_under_rls
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
database: 0 wiki documents, 0 directed_vectors, 0 rows in four of the seven
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

DOWNGRADE RECREATES THE TABLES, EMPTY, AND SAYS SO. The DDL is reversible; the
rows are not. A downgrade that silently produced an empty wiki would look like
a restore, so it raises unless the operator sets `PRBE_ALLOW_EMPTY_WIKI_RESTORE`
-- the schema comes back for a rollback that needs the shape, and nobody
mistakes it for the data.
"""

from __future__ import annotations

import os

import sqlalchemy as sa
from alembic import op

revision = "0109_drop_wiki"
down_revision = "0108_retire_pp_under_rls"
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


def downgrade() -> None:
    if not os.environ.get("PRBE_ALLOW_EMPTY_WIKI_RESTORE"):
        raise RuntimeError(
            "0109 cannot restore the wiki: its pages, chunks and queue rows were "
            "deleted, not archived, and this downgrade would recreate seven empty "
            "tables that look like a restore and are not. Set "
            "PRBE_ALLOW_EMPTY_WIKI_RESTORE=1 if the empty SCHEMA is what you need "
            "(a rollback to code that expects the tables to exist)."
        )
    # Shape only, and only enough of it for older code to start against. The
    # indexes, RLS policies and constraints are NOT restored: they exist to make
    # a working pipeline correct, and there is no pipeline to be correct for.
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS wiki_synthesis_queue (
            id BIGSERIAL PRIMARY KEY,
            customer_id TEXT NOT NULL,
            doc_id TEXT NOT NULL,
            doc_version INT,
            source_system TEXT,
            doc_type TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        CREATE TABLE IF NOT EXISTS wiki_synthesis_runs (
            id BIGSERIAL PRIMARY KEY,
            customer_id TEXT NOT NULL,
            kind TEXT,
            status TEXT,
            started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            error TEXT
        );
        CREATE TABLE IF NOT EXISTS wiki_raw_data (
            id BIGSERIAL PRIMARY KEY,
            customer_id TEXT NOT NULL,
            payload JSONB
        );
        CREATE TABLE IF NOT EXISTS wiki_links (
            id BIGSERIAL PRIMARY KEY,
            customer_id TEXT NOT NULL,
            src_doc_id TEXT,
            dst TEXT
        );
        CREATE TABLE IF NOT EXISTS wiki_timeline_entries (
            id BIGSERIAL PRIMARY KEY,
            customer_id TEXT NOT NULL,
            doc_id TEXT,
            occurred_at TIMESTAMPTZ
        );
        CREATE TABLE IF NOT EXISTS wiki_page_settings (
            customer_id TEXT NOT NULL,
            wiki_type TEXT NOT NULL,
            slug TEXT NOT NULL,
            pipeline_updates BOOLEAN NOT NULL DEFAULT TRUE,
            PRIMARY KEY (customer_id, wiki_type, slug)
        );
        CREATE TABLE IF NOT EXISTS wiki_append_idempotency (
            customer_id TEXT NOT NULL,
            wiki_type TEXT NOT NULL,
            slug TEXT NOT NULL,
            idempotency_key TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (customer_id, wiki_type, slug, idempotency_key)
        );
        -- `halfvec` needs the vector extension, which every deployment that ran
        -- this migration already has.
        CREATE TABLE IF NOT EXISTS directed_vectors (
            vector_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            customer_id TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
            doc_id TEXT NOT NULL,
            embedding halfvec(3072) NOT NULL,
            source_text TEXT NOT NULL,
            source TEXT NOT NULL,
            synthesis_run_id BIGINT NULL,
            content_hash BYTEA NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """
    )
