"""redo 0107's retire per-customer — FORCE RLS made it a silent no-op

Revision ID: 0108_retire_pp_under_rls
Revises: 0107_retire_project_person
Create Date: 2026-08-18

WHAT WENT WRONG
---------------
0107 retired every live `project` / `person` page with one unscoped statement:

    UPDATE documents SET valid_to = ... WHERE source_system = 'wiki' AND ...

`documents` carries `ENABLE ROW LEVEL SECURITY` **and** `FORCE ROW LEVEL
SECURITY` with a `tenant_isolation` policy on the `app.current_customer_id`
GUC. FORCE is the part that matters: without it RLS exempts the table owner,
and with it the owner is subject to the policy like anyone else. Only a role
holding BYPASSRLS escapes.

A migration sets no GUC, so `current_setting('app.current_customer_id', true)`
is NULL, `customer_id = NULL` is NULL for every row, and the UPDATE matches
NOTHING. It does not error. It reports success, alembic stamps the revision,
and the deploy goes green.

Whether 0107 did anything therefore came down to which role ran it, which
differs per deployment:

  * managed-shared connects as `probe`, which holds BYPASSRLS, so 0107 applied
    correctly there (3 pages retired for probe-founders, verified 2026-08-18).
  * research-os's `engine-kb-migrate` hook connects as `app`, which does not.
    0107 matched zero rows, leaving 20 live pages (`probe` 15, `anthrogen` 5)
    of two kinds `WikiType` no longer contains.

That second state is exactly what 0107 existed to prevent: the kind is a path
segment and a doc_id component validated against the closed set, so a live page
carrying a retired kind is unreachable — 400 on read, and 400 on the DELETE
that would have cleaned it up. It was invisible only because the engine pods on
that cluster run a floating image tag and had not yet rolled to the code that
drops the kinds.

WHY THE LOOP, NOT A PRIVILEGE
-----------------------------
Three ways out, and the other two are worse:

  * Run migrations as a BYPASSRLS role. That is a privilege change on every
    deployment to fix one statement, and it silently re-arms the same trap for
    the next migration on any deployment that does not make the change.
  * `ALTER TABLE documents NO FORCE ROW LEVEL SECURITY` around the statement.
    This is what 0051 tried on `graph_nodes`: it needs an AccessExclusiveLock
    against live ingestion traffic, and waiting for it is what blew that
    migration's 15-minute release window.
  * Bind the GUC per customer and scope the statement — what the application
    itself does on every write (`engine.shared.db.with_tenant`). It needs no
    privilege, takes no DDL lock, and is correct on both planes.

`customers` has no RLS of its own (it is the tenant registry the policy is
keyed on), so enumerating it from inside the migration is safe.

SAME STAMP AS 0107, DELIBERATELY
--------------------------------
Rows closed here carry 0107's `valid_to` value rather than a new one, so "the
pages this taxonomy change retired" stays ONE addressable value across both
planes. The undo guard in `_restore_retired_pages` keys on the page KIND rather
than on this timestamp, so it is unaffected either way — but a second stamp
would mean anyone reading these rows later has to know which plane they are on
and which migration got there first.

IDEMPOTENT
----------
`valid_to IS NULL` means the rows 0107 already closed on managed-shared are not
re-stamped, and re-running this changes nothing. The sidecar deletes 0107 ran
unscoped DID take effect everywhere (those three tables carry no RLS), so they
are repeated here only to cover rows written between the two migrations.

DOWNGRADE
---------
Same shape and same bound as 0107's: un-retire only rows carrying this stamp,
and only where nothing live has since taken the doc_id, since two live versions
of one doc_id would break every `valid_to IS NULL` read in the engine.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0108_retire_pp_under_rls"
down_revision = "0107_retire_project_person"
branch_labels = None
depends_on = None

#: 0107's stamp, reused. See "SAME STAMP AS 0107" above.
_RETIRED_AT = "TIMESTAMPTZ '2026-08-18 00:00:00+00'"

#: Spelled as SQL literals rather than imported from `WikiType`: this migration
#: has to keep describing what it retired after those members are gone from the
#: enum, and one whose target set is read from application code changes meaning
#: every time that code does.
#:
#: Both halves of the identity are checked. `doc_type` is what the connector
#: stamps (`wiki.` + kind) and `doc_id` is what the routes address
#: (`wiki:{kind}:{slug}`); requiring both leaves a row that disagrees with
#: itself for a human rather than sweeping it up on a guess.
_TARGET = """
           AND doc_type IN ('wiki.project', 'wiki.person')
           AND (doc_id LIKE 'wiki:project:%' OR doc_id LIKE 'wiki:person:%')
"""


def _for_each_customer(sql: str) -> None:
    """Run `sql` once per customer with the tenant GUC bound to that customer.

    `set_config(..., false)` — session scope, not transaction scope. Alembic
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
    # Leave no tenant bound behind: a later statement in this transaction that
    # forgot to scope itself should match nothing, not silently inherit
    # whichever customer happened to sort last.
    conn.execute(bind_tenant, {"customer_id": ""})


def upgrade() -> None:
    # Sidecars carry no RLS, so these run once, unscoped, exactly as in 0107.
    op.execute(
        """
        DELETE FROM wiki_links
         WHERE src_wiki_type IN ('project', 'person')
            OR dst_wiki_type IN ('project', 'person')
        """
    )
    op.execute("DELETE FROM wiki_timeline_entries WHERE wiki_type IN ('project', 'person')")
    op.execute("DELETE FROM wiki_raw_data WHERE wiki_type IN ('project', 'person')")

    _for_each_customer(
        f"""
        UPDATE documents
           SET valid_to = {_RETIRED_AT}
         WHERE source_system = 'wiki'
           AND valid_to IS NULL
           {_TARGET}
        """
    )


def downgrade() -> None:
    _for_each_customer(
        f"""
        UPDATE documents
           SET valid_to = NULL
         WHERE source_system = 'wiki'
           AND valid_to = {_RETIRED_AT}
           {_TARGET}
           AND NOT EXISTS (
               SELECT 1
                 FROM documents live
                WHERE live.customer_id = documents.customer_id
                  AND live.doc_id = documents.doc_id
                  AND live.valid_to IS NULL
           )
        """
    )
    # Sidecars are NOT restored: they were deleted, and they are pure
    # derivations rewritten on the next write of their page.
