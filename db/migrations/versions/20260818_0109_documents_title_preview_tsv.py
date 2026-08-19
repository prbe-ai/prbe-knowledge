"""documents: materialize the title+preview tsvector so RLS can index it

Revision ID: 0109_documents_title_preview_tsv
Revises: 0108_retire_pp_under_rls
Create Date: 2026-08-18

WHY
---
`grounding._fuzzy_match_document_titles` qualifies rows with

    d.title % $2
    OR to_tsvector('english', coalesce(d.title,'') || ' '
                              || coalesce(d.body_preview,'')) @@ plainto_tsquery(...)

and `idx_documents_fts_title_preview` is a GIN index on exactly that
expression. As the table owner or a superuser the planner uses it happily.
The retrieval service never does, because `documents` carries ENABLE + FORCE
ROW LEVEL SECURITY and the service is not BYPASSRLS (the same FORCE that made
0107 a silent no-op).

Under RLS the planner may not use an EXPRESSION index here: matching one means
evaluating the indexed expression on rows before the security qual has been
applied, which it refuses to do for anything not provably leakproof. It falls
back to `idx_documents_parent_live` (customer_id) and filters, rebuilding the
tsvector for every live document in the tenant.

MEASURED on the managed plane, tenant probe-founders, role probe_app with
`row_security = on`, single probe 'grounding':

    today (expression index)                    1052 ms
    with to_tsvector/ts_match_vq/plainto_tsquery
      marked LEAKPROOF                           874 ms   <-- still not indexed
    against a STORED tsvector column              23 ms   <-- indexable/cheap

LEAKPROOF does not rescue an expression index; only materializing the
expression does. A STORED generated column is a plain column reference, so the
index on it is an ordinary column index and the barrier does not apply.

The full grounding predicate (trgm OR fts) measured 1052 ms today and 238 ms
against a stored column with no other change -- a 4.4x cut with no semantic
change and no security decision. Getting the remaining ~9x needs
`similarity_op` marked LEAKPROOF so the pg_trgm half can use its own index;
that is a separate, reviewable security trade and is deliberately NOT bundled
here.

SEMANTICS
---------
The generated expression is character-for-character the expression the dropped
index was built on, so qualification is unchanged. This is deliberately NOT the
existing `title_tsv` column: that one is title-only and setweight'd, and
`fts_hit` QUALIFIES rows rather than merely ranking them, so a document
matching only through `body_preview` would silently stop qualifying. There is
a test guarding exactly that
(`test_fuzzy_match_document_titles_fts_only_path_hits_body_preview`).

`to_tsvector(regconfig, text)` -- the two-argument form with an explicit
configuration -- is IMMUTABLE, which is what makes it legal in a generated
column. The one-argument form is only STABLE and would be rejected.

COST
----
The column measures 6849 kB over 24,220 rows on a 101 MB table. The GIN index
lands near the 4104 kB of the expression index it replaces, which this
migration drops -- `grounding.py` was its only caller (verified by grep), and
under RLS that caller could never use it, so nothing loses an access path it
actually had. Write cost is a wash: the dropped index already computed this
same tsvector on every insert and update.

WHY NOT CONCURRENTLY
--------------------
Same reasoning as 0105, which took prod deploys down for two hours learning it:
CONCURRENTLY waits indefinitely on older transactions and has no place in a
deploy hook with a timeout. This table is 24,220 rows / 101 MB and a plain
build is well under a second.

Note the ADD COLUMN itself rewrites the table under ACCESS EXCLUSIVE regardless
-- a generated STORED column has to be materialized for every existing row.

MEASURED on the managed plane (24,235 rows / 101 MB), in a rolled-back
transaction: ADD COLUMN 9.05 s, CREATE INDEX 0.40 s. So this migration holds
ACCESS EXCLUSIVE on `documents` for roughly 9.5 SECONDS, not the "seconds" an
earlier draft of this note claimed. `documents` is the hottest table in the
system and the managed plane carries five active tenants, so this is a
customer-visible stall and belongs in a low-traffic window. The follow-on
ANALYZE (4.2 s) does not take an exclusive lock.

Equivalence was verified on the same prod snapshot: 24,235 rows,
0 rows where the generated column differs from the expression it replaces.
"""
from __future__ import annotations

from alembic import op

revision = "0109_documents_title_preview_tsv"
down_revision = "0108_retire_pp_under_rls"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE documents
            ADD COLUMN IF NOT EXISTS title_preview_tsv tsvector
            GENERATED ALWAYS AS (
                to_tsvector(
                    'english'::regconfig,
                    coalesce(title, '') || ' ' || coalesce(body_preview, '')
                )
            ) STORED
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_documents_title_preview_tsv
            ON documents USING GIN (title_preview_tsv)
        """
    )
    # Dropped only after the replacement exists. Safe to drop ahead of the code
    # change that stops referencing the raw expression: under RLS the retrieval
    # service could never use this index anyway, so the window where old code
    # meets a missing index costs it nothing it was getting.
    op.execute("DROP INDEX IF EXISTS idx_documents_fts_title_preview")


def downgrade() -> None:
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_documents_fts_title_preview
            ON documents USING GIN (
                to_tsvector(
                    'english'::regconfig,
                    coalesce(title, '') || ' ' || coalesce(body_preview, '')
                )
            )
        """
    )
    op.execute("DROP INDEX IF EXISTS idx_documents_title_preview_tsv")
    op.execute("ALTER TABLE documents DROP COLUMN IF EXISTS title_preview_tsv")
