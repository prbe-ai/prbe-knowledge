"""retire every live `project` / `person` wiki page

Revision ID: 0107_retire_project_person
Revises: 0106_wiki_append_idem
Create Date: 2026-08-18

`project` and `person` left `WikiType` in the same change that adds this
migration. The reasoning lives next to the enum (kb/synthesis/models.py); in
one line, both page kinds restated what the platform already holds — projects,
experiments and runs are live entities in research-os, and authorship /
ownership are edges the ingestion pipeline maintains continuously — so each
page was a second copy that went stale silently and gave a reader no way to
tell which copy was current. `decision` stays, because a decision exists in no
system of record.

WHY THE PAGES CANNOT JUST BE LEFT ALONE
---------------------------------------
The kind is not only a label. It is a path segment
(`/api/wiki/pages/{wiki_type}/{slug}`) and a doc_id component
(`wiki:{wiki_type}:{slug}`), and both the human routes and the ingestion gate
validate it against the closed set. The moment `WikiType` loses a member, every
page still carrying it is live but unreachable: 400 on read, 400 on edit, and
400 on the DELETE that would have been the way to clean it up by hand.

Leaving that state in place is the worst of the three options — the wiki still
"has" the folder as far as the database is concerned, nothing can open it, and
the only way out is another migration. So the pages go in the same deploy as
the enum change.

RETIRE, NOT DELETE
------------------
`documents` is bitemporal: the primary key is (customer_id, doc_id, version)
and "live" means `valid_to IS NULL`. Stamping `valid_to` is exactly what an
ordinary supersede does, and every page read path already filters on it, so
retiring is indistinguishable from deleting to every reader while leaving the
bodies and their version history on disk. `_wipe_wiki_for_customer` made the
same call for the same reason (#494); this is the narrowed, one-way version.

That distinction is the difference between a reversible migration and an
irreversible one. A wrong call about a page kind is recoverable from
`valid_to`; it is not recoverable from a DELETE.

ALL DOC CLASSES, NOT ONLY THE GENERATED ONES
--------------------------------------------
`compiled_wiki` (synthesized) and `manual_entry` (hand-written) pages are both
retired. Retiring only the generated ones would leave hand-written `person`
pages live and unreachable, which is the exact state described above — and a
hand-written page is the one most likely to be missed, because no nightly drain
will ever touch it again to make the breakage visible.

THE SIDECARS
------------
`wiki_links`, `wiki_timeline_entries` and `wiki_raw_data` rows for these pages
are deleted rather than retired: they carry no version history, are pure
derivations of a page body, and are rewritten wholesale on the next write of
their page.

Inbound links FROM surviving pages go too, and that is narrower than it sounds.
A repo page whose frontmatter says `owners: [person:maison]` keeps that text,
and the `[[person: ...]]` link still resolves to the canonical Person graph
node exactly as before — a wiki link points at a graph ENTITY, and several
kinds it can name have never had pages (`service`, `ticket`). Only the
`wiki_links` row recording page-to-page adjacency is dropped, and the next
write of that page re-derives its links from the body.

`graph_nodes` is not touched, following 0051: the lock it would need against
live ingestion traffic is what blew that migration's release window, and nodes
whose documents are gone are unreachable orphans either way.

`documents` has neither RLS nor FORCE RLS (see 0037), so these statements run
unscoped across every tenant under the migration role, which is what a taxonomy
change requires.

DOWNGRADE
---------
Un-retires exactly what this migration retired — possible only because it
retired instead of deleting. Bounded to rows carrying this migration's own
stamp so it cannot resurrect a page a human deleted last week, and skipping any
doc that has since acquired a live version, since two live versions of one
doc_id would break every `valid_to IS NULL` read in the engine.
"""

from __future__ import annotations

from alembic import op

revision = "0107_retire_project_person"
down_revision = "0106_wiki_append_idem"
branch_labels = None
depends_on = None

#: One instant shared by every row this migration closes, rather than NOW().
#: That is what makes the downgrade addressable — "the pages 0107 retired" is a
#: value to match on, not a time range someone has to guess at afterwards.
_RETIRED_AT = "TIMESTAMPTZ '2026-08-18 00:00:00+00'"

#: The two kinds are spelled out as SQL literals rather than imported from
#: `WikiType`: this migration has to keep describing what it retired after those
#: members are gone from the enum, and a migration whose target set is read from
#: application code changes meaning every time that code does.
#:
#: Both halves of the identity are checked, not just one. `doc_type` is what the
#: connector stamps (`wiki.` + kind) and `doc_id` is what the routes address
#: (`wiki:{kind}:{slug}`). Requiring both means a row that disagrees with itself
#: — the shape a bad backfill leaves behind — is left for a human to look at
#: rather than swept up on a guess.
_TARGET = """
           AND doc_type IN ('wiki.project', 'wiki.person')
           AND (doc_id LIKE 'wiki:project:%' OR doc_id LIKE 'wiki:person:%')
"""


def upgrade() -> None:
    # Sidecars first, keyed on (wiki_type, slug) — the pair these tables index
    # on. Both directions of wiki_links go: a row is a fact about an adjacency,
    # and one of its endpoints is about to stop existing.
    op.execute(
        """
        DELETE FROM wiki_links
         WHERE src_wiki_type IN ('project', 'person')
            OR dst_wiki_type IN ('project', 'person')
        """
    )
    op.execute("DELETE FROM wiki_timeline_entries WHERE wiki_type IN ('project', 'person')")
    op.execute("DELETE FROM wiki_raw_data WHERE wiki_type IN ('project', 'person')")

    # `valid_to IS NULL` keeps this idempotent AND keeps the downgrade honest:
    # versions an earlier supersede already closed keep their original stamp and
    # are correctly not part of what this migration retired.
    op.execute(
        f"""
        UPDATE documents
           SET valid_to = {_RETIRED_AT}
         WHERE source_system = 'wiki'
           AND valid_to IS NULL
           {_TARGET}
        """
    )


def downgrade() -> None:
    op.execute(
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
    # The sidecars are NOT restored: they were deleted, and they are pure
    # derivations rewritten on the next write of their page. A downgraded page
    # comes back with its body and its history and an empty link graph — the
    # same trade `_wipe_wiki_for_customer` documents for the rebuild-undo path.
