"""wiki_page_settings: make "the pipeline updates this page" an explicit setting

Revision ID: 0103_wiki_page_settings
Revises: 0102_bm25_title_tokenizer
Create Date: 2026-08-12

WHAT THIS REPLACES
------------------
Until now, "will the nightly synthesis run touch this page?" was answered by
`documents.doc_class`. Editing a page by hand stamped it `manual_entry`
(`kb/wiki_routes.py::_reject_non_manual_doc_class` forces that value on every
public write), and the agent then skipped it forever
(`kb/synthesis/wiki_agent.py::_persist_update`). No API, CLI or UI path could
set it back -- only hand-written SQL against this database.

So a single hand edit was a one-way door, and nothing warned anyone before
they walked through it. The page just stopped updating and went stale.

This migration separates the two things that were conflated:

    doc_class         who wrote THIS VERSION      (provenance, per-version)
    pipeline_updates  may the agent rewrite it    (setting, per-page)

`doc_class` keeps its meaning and its validator: a public write really is a
human write, and the history list uses that to tell a person's revision from
the agent's. Nothing is dropped by this migration or by a later one -- the
expand/contract rule does not apply because there is no contract phase.

WHY ITS OWN TABLE AND NOT A COLUMN ON `documents`
-------------------------------------------------
`documents` is a version chain -- PK (customer_id, doc_id, version), and
`Normalizer._persist` INSERTs a brand-new row on every write built wholly from
the incoming payload. Nothing carries forward. A column there would be a fact
about one VERSION, so all four writers (dashboard BFF, `probe wiki write`,
research-os `PUT /v1/wiki/pages`, the agent's own `_build_wiki_event`) would
each have to remember to re-send it, and any that forgot would silently reset
the page to the default. `revert` is worse: it rebuilds its event from the
metadata of the version being reverted TO, so restoring last week's text would
restore last week's freeze state with it.

And a setting is not a revision. Flipping it changes no content, so it must
not mint a version -- otherwise page history fills with revisions that changed
nothing. Its own row gets that for free, plus `updated_at` / `updated_by`,
which the version chain cannot record for a change that makes no version.

MIGRATION DECISION: EVERY EXISTING PAGE COMES BACK ON
-----------------------------------------------------
This migration inserts NO ROWS. An absent row reads as pipeline_updates=TRUE
(see `fetch_page_pipeline_updates`), so every page that exists today -- including
every page currently sitting at `doc_class='manual_entry'` -- resumes receiving
nightly updates.

That is deliberate, and it is a behaviour change for those pages. Almost none
of them were frozen on purpose: they were frozen by the trapdoor described
above, as a side effect of somebody fixing a typo. Carrying that state forward
would preserve an outcome nobody chose. Anyone who does want a page frozen can
now say so explicitly, and say it in a way they can undo.

The blast radius is bounded by two properties that did not exist when the skip
was written: the agent now READS the existing body as authoritative context
rather than clobbering it, and every version is retained with one-click revert.

Revision string is 22 chars, inside the 32-char alembic_version limit.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0103_wiki_page_settings"
down_revision = "0102_bm25_title_tokenizer"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Keyed on (customer_id, wiki_type, slug) rather than on doc_id: that is
    # the identity `wiki_links` and `wiki_timeline_entries` already use, and the
    # one the routes and the agent both carry. There is no FK to `documents`
    # available in either shape -- its PK includes `version`.
    op.create_table(
        "wiki_page_settings",
        sa.Column(
            "customer_id",
            sa.Text(),
            sa.ForeignKey("customers.customer_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("wiki_type", sa.Text(), nullable=False),
        sa.Column("slug", sa.Text(), nullable=False),
        sa.Column(
            "pipeline_updates",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("TRUE"),
        ),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column("updated_by", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("customer_id", "wiki_type", "slug"),
    )

    # Same tenant guard as every other tenant table here. FORCE so the policy
    # applies to the table owner too -- the ingestion role owns these tables,
    # and without FORCE it would read across tenants.
    op.execute("ALTER TABLE wiki_page_settings ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE wiki_page_settings FORCE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY tenant_isolation ON wiki_page_settings
            USING (customer_id = current_setting('app.current_customer_id', true))
            WITH CHECK (customer_id = current_setting('app.current_customer_id', true))
        """
    )


def downgrade() -> None:
    # Dropping the table restores the DEFAULT (pipeline updates on) for every
    # page, not the old doc_class-based freeze -- the agent code that read
    # doc_class is gone by then. A downgrade therefore un-freezes deliberately
    # frozen pages. Stated rather than guarded against: the alternative is
    # writing doc_class='manual_entry' back onto those pages, which would
    # rewrite content provenance to encode a setting, which is the bug.
    op.execute("DROP POLICY IF EXISTS tenant_isolation ON wiki_page_settings")
    op.drop_table("wiki_page_settings")
