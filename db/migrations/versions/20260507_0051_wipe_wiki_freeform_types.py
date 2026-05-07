"""wipe all existing wiki data + drop decision/feature/vendor doc_types

Revision ID: 0051_wipe_wiki_freeform_types
Revises: 0050_add_target_to_wsr
Create Date: 2026-05-07

The wiki taxonomy is being narrowed: ``service_card`` is renamed to
``repo`` and ``decision``, ``feature``, ``vendor`` are dropped entirely
(too product-specific for the wiki's slow-moving knowledge scope). The
LLM also takes over index organization (no hardcoded sections), so any
existing index pages are stale.

The user explicitly asked for a wholesale wipe of pre-existing wiki
data — re-bootstrap regenerates everything cleanly under the new shape.
This migration:

  1. Deletes every wiki document and its chunks (every ``doc_type LIKE
     'wiki.%'`` row in ``documents``, plus every ``chunks`` row whose
     ``doc_id`` starts with ``wiki:``).
  2. Wipes the synthesis queue (any pending rows would re-emit pages
     under the old taxonomy).
  3. Wipes wiki-specific side tables (``wiki_links``,
     ``wiki_raw_data``, ``wiki_timeline_entries``).
  4. Deletes any ``failed_chunks`` rows referencing wiki doc_ids.

``wiki_synthesis_runs`` is preserved for audit history.

The orphan ``graph_nodes`` rows whose ``label`` was a wiki taxonomy
member (``ServiceCard`` / ``Decision`` / ``Feature`` / ``Runbook`` /
``WikiPerson``) are NOT touched here. The first attempt held an
``ALTER TABLE graph_nodes NO FORCE ROW LEVEL SECURITY`` waiting for
``AccessExclusiveLock`` against live ingestion traffic and timed out
the 15-minute ``release_command`` window. Since the wiki documents
themselves are gone, those graph nodes are inaccessible orphans — a
follow-up out-of-band cleanup (or a future migration that takes the
lock when ingestion is paused) can drop them safely.

Downgrade is a no-op — the wipe is one-way, and re-running this
migration after a fresh bootstrap would just delete the new pages.
"""

from __future__ import annotations

from alembic import op

revision = "0051_wipe_wiki_freeform_types"
down_revision = "0050_add_target_to_wsr"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Chunks first (no FK to documents, would orphan otherwise).
    op.execute("DELETE FROM chunks WHERE doc_id LIKE 'wiki:%'")
    op.execute("DELETE FROM failed_chunks WHERE doc_id LIKE 'wiki:%'")

    # 2. The documents themselves. source_system='wiki' is the
    #    authoritative filter; doc_type LIKE 'wiki.%' is a safety net
    #    in case anything stamped a wiki doc_type under a different
    #    source_system.
    op.execute(
        "DELETE FROM documents "
        "WHERE source_system = 'wiki' OR doc_type LIKE 'wiki.%'"
    )

    # 3. Synthesis queue — pending rows would re-emit pages under the
    #    old taxonomy. The wiki-cron will re-enqueue from scratch on
    #    the next nightly tick (or via the next bootstrap).
    op.execute("DELETE FROM wiki_synthesis_queue")

    # 4. Wiki-specific side tables.
    op.execute("DELETE FROM wiki_links")
    op.execute("DELETE FROM wiki_raw_data")
    op.execute("DELETE FROM wiki_timeline_entries")

    # graph_nodes cleanup intentionally omitted — see module docstring.
    # The previous attempt blocked on `ALTER TABLE NO FORCE ROW LEVEL
    # SECURITY` waiting for an AccessExclusiveLock against live
    # ingestion traffic and exceeded the release_command timeout.
    # Orphan wiki-labeled nodes are inaccessible without their
    # documents and can be cleaned up out-of-band.


def downgrade() -> None:
    # No-op. The wipe is one-way; re-running upgrade after a fresh
    # bootstrap would erase the new pages, so the inverse direction
    # cannot meaningfully restore data.
    pass
