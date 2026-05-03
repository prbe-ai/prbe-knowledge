"""Coalesce superseded claude_code session versions into one row.

Revision ID: 0036_cleanup_cc_superseded
Revises: 0035_strip_metadata_body
Create Date: 2026-05-03

WHY
---
Pre-handler-fix, every batch of a long-running Claude Code session
opened a new SCD2 version of the session document. One willow-voice
session accumulated 5+ versions, each carrying the entire transcript.
Across all CC sessions: 1,726 superseded rows holding ~251 MB of
redundant body data.

The handler change in this branch coalesces in-place while the session
is still incomplete (`session_complete=false`), so no new amplification
will accrue. This migration cleans up what already accumulated.

WHAT
----
For each `(customer_id, source_id)` in claude_code documents with more
than one version, keep ONLY the latest version (highest `version`
number) and DELETE the rest. The latest is the most authoritative — it
contains the most events. Earlier versions are strict subsets.

Cascading
---------
The `chunks` table has NO FK to documents (only customer_id CASCADE);
chunks span versions via [first_seen_version, last_seen_version], so
they're never tied to a specific doc row's lifecycle. Deleting old
document rows leaves all chunks intact. The chunk reuse machinery
already keeps content-addressed deduplication, so the live version's
chunks (the only ones still queryable as live) are untouched.

Other tables that reference documents:
  * `wiki_synthesis_queue` — references (customer_id, doc_id, doc_version).
    No FK; rows are status-tracked queue entries. Stale rows for
    already-deleted versions are harmless (they'll be filtered by the
    cron's join against documents and marked as missing).
  * `failed_chunks` — references (customer_id, doc_id, doc_version).
    No FK; rows are diagnostic. Safe to leave; they'll go stale.

The `documents` table has neither RLS nor FORCE RLS, so a global DELETE
under the migration's role hits every row.

Storage reclaimed
-----------------
Estimated ~251 MB of `metadata.body` (mostly already removed by
migration 0035, but the document rows themselves also carry per-version
metadata + body_preview that adds up).

The on-disk reclaim does not happen until autovacuum runs (or
`VACUUM FULL documents` is run manually). See 0035 for the followup.

IDEMPOTENT
----------
Running twice is safe. The DELETE keys on (customer_id, doc_id,
version) and excludes the latest version per (customer_id, source_id);
on a second run there is at most one row per group, so the EXCEPT set
is empty.

DOWNGRADE
---------
No-op. The deleted versions are by definition strict subsets of the
retained latest version; reconstructing them would require replaying
the original ingestion, which is beyond a migration's scope.
"""

from __future__ import annotations

from alembic import op

revision = "0036_cleanup_cc_superseded"
down_revision = "0035_strip_metadata_body"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # DELETE every claude_code/codex session row that is NOT the latest
    # version for its (customer_id, source_id) group. The CTE picks the
    # max version per group; the outer DELETE keys on the per-tenant PK.
    #
    # Restricted to source_system IN ('claude_code','codex') because both
    # share the SCD2 amplification pattern (Codex inherits from
    # ClaudeCodeConnector). All other sources keep one-version-per-edit
    # by design and must not be touched.
    op.execute(
        """
        WITH latest_per_session AS (
            SELECT customer_id, source_id, MAX(version) AS keep_version
            FROM documents
            WHERE source_system IN ('claude_code', 'codex')
              AND doc_type = 'claude_code.session'
            GROUP BY customer_id, source_id
            HAVING COUNT(*) > 1
        )
        DELETE FROM documents d
        USING latest_per_session lp
        WHERE d.customer_id = lp.customer_id
          AND d.source_id = lp.source_id
          AND d.source_system IN ('claude_code', 'codex')
          AND d.doc_type = 'claude_code.session'
          AND d.version <> lp.keep_version
        """
    )


def downgrade() -> None:
    # No-op. See module docstring.
    pass
