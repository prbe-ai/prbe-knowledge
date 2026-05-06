"""wiki bootstrap: length caps on dedup-bearing text fields

Revision ID: 0045_wiki_field_length_caps
Revises: 0044_wiki_bootstrap_schema
Create Date: 2026-05-06

Two CHECK constraints to keep agent-generated text from overflowing
btree key limits and from misusing summary as a long-form field:

1. `wiki_timeline_entries.summary` — capped at 1000 chars. The column
   is part of the `uq_wiki_timeline_dedup` btree key, which Postgres
   limits to ~2704 bytes per row. The other columns in the key total
   roughly 150 bytes (customer_id, wiki_type, slug short strings +
   entry_date), leaving ~2500 bytes for summary. 1000 is comfortably
   under that and well above any sensible "one-line audit entry"
   length. If a crawler wants to write longer text it goes in
   `detail` (no length cap, not in the unique key).

2. `wiki_links.context` — capped at 200 chars. The column comment
   already says "~80 chars surrounding the link site", and the
   parser that populates it slices a ~80-char window. 200 is 2.5x
   the convention, leaving headroom for multibyte characters and
   future window adjustments without re-migrating. Unlike the
   summary case, `context` is NOT in the unique key — this cap is
   purely a "don't misuse the field" guard rail, not a btree
   protection.

Both caps are forward-compatible with all existing rows (none yet —
migration 0044 created these tables empty).

Downgrade drops both constraints.
"""

from __future__ import annotations

from alembic import op

revision = "0045_wiki_field_length_caps"
down_revision = "0044_wiki_bootstrap_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE wiki_timeline_entries "
        "ADD CONSTRAINT ck_wiki_timeline_summary_len "
        "CHECK (length(summary) <= 1000)"
    )
    op.execute(
        "ALTER TABLE wiki_links "
        "ADD CONSTRAINT ck_wiki_links_context_len "
        "CHECK (length(context) <= 200)"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE wiki_links DROP CONSTRAINT IF EXISTS ck_wiki_links_context_len")
    op.execute(
        "ALTER TABLE wiki_timeline_entries DROP CONSTRAINT IF EXISTS ck_wiki_timeline_summary_len"
    )
