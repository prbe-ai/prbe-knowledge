"""wiki: cap the size of LIVE wiki pages

Revision ID: 0104_wiki_live_page_size
Revises: 0103_wiki_page_settings
Create Date: 2026-08-13

The backstop under the staged-commit preflight, and deliberately weaker than
it. The preflight is the real enforcement: it sees the whole batch, knows
which pages are splits, and can tell the agent WHY in words it can act on.
This catches anything that never goes through it -- the manual
`PUT /v1/wiki/pages` route, a future writer, a migration.

IT CONSTRAINS A COUNTER, NOT THE CONTENT. There is no page `body` column:
bodies live across `chunks` and are reassembled on read, so no CHECK can
measure one. `documents.body_size_bytes` is written by the handler as
`len(body.encode("utf-8"))`, so this rejects an over-cap write only from a
writer that reports its own size honestly. That is defence in depth against
an ordinary bug, not a guarantee against a lying caller, and calling it
anything stronger would be false comfort.

THREE ESCAPES, each load-bearing:

  source_system <> 'wiki'
      `documents` holds every connector. Slack threads, GitHub PRs and Claude
      Code transcripts are routinely far larger than 8 KB and have no business
      being split; an unscoped constraint would break every ingest path in the
      product.

  doc_type = 'wiki.index'
      The front page is generated whole from every other page. Splitting it is
      meaningless, and capping it would simply fail the nightly render.

  valid_to IS NOT NULL
      THE ONE THAT DECIDES WHETHER THIS MIGRATION APPLIES AT ALL. `documents`
      is temporal: superseding a page sets `valid_to` and keeps the old row.
      Measured on the probe tenant when this was written, 97 historical wiki
      versions are over the cap (research_os alone reached 37,540 bytes) while
      0 live ones are. Without this clause `ADD CONSTRAINT` fails outright on
      those 97 rows -- and rewriting published history to satisfy a new rule
      would corrupt the audit chain the version list exists to provide.

      It also protects the supersede path: setting `valid_to` on a row is an
      UPDATE, and an UPDATE re-checks the constraint. A row that was live and
      legal stays legal as it ages out.

Validated, not NOT VALID. 44,794 documents is a cheap scan, and the predicate
was confirmed to hold against production before this was written -- a
NOT VALID constraint would leave a rule nobody has ever checked, which is the
kind of guard that certifies its own rot.
"""

from __future__ import annotations

from alembic import op

revision = "0104_wiki_live_page_size"
down_revision = "0103_wiki_page_settings"
branch_labels = None
depends_on = None

#: Mirrors kb.synthesis.staged_graph.PAGE_CAP_BYTES. Duplicated as a literal
#: because a migration must keep meaning what it meant on the day it ran: if the
#: application constant is raised later, already-applied databases still carry
#: the old number, and a migration that silently tracked the constant would
#: describe a state no database is actually in.
_CAP_BYTES = 8 * 1024

_CONSTRAINT = "ck_wiki_live_page_size"

#: The predicate, as one string, so a test can compare it against db/schema.sql
#: without regexing a template. schema-parity would catch a divergence too, but
#: only after a full Postgres round trip in CI.
PREDICATE = (
    "source_system <> 'wiki' "
    "OR doc_type = 'wiki.index' "
    "OR valid_to IS NOT NULL "
    f"OR body_size_bytes <= {_CAP_BYTES}"
)


def upgrade() -> None:
    op.execute(
        f"""
        ALTER TABLE documents ADD CONSTRAINT {_CONSTRAINT} CHECK ({PREDICATE})
        """
    )


def downgrade() -> None:
    op.execute(f"ALTER TABLE documents DROP CONSTRAINT IF EXISTS {_CONSTRAINT}")
