"""Inferred-edges queue: outstanding-row dedup + unique index

Revision ID: 0096_inferred_edges_dedup
Revises: 0095_pending_edges
Create Date: 2026-07-23

Why
---
``normalizer._enqueue_inferred_edges`` inserts with ``ON CONFLICT DO NOTHING``
intending "one outstanding row per (customer, anchor_doc, extractor)", but the
table never had a UNIQUE constraint for that clause to arbitrate on -- so the
ON CONFLICT was a silent no-op. Every re-persist of a doc inserted a fresh row.

Agent-session transcripts (claude_code / codex) re-persist the SAME anchor doc
on every batch append, so a single live session enqueued the same anchor
dozens-to-hundreds of times (observed: one codex session = 902 rows), each a
full ~300K-token bundle LLM call -- a large, avoidable volume of redundant
extraction work.

The enqueue side is now gated to session-finalization for those two sources
(engine/ingest/normalizer._inferred_edge_doc_ids). This migration is the
defense-in-depth half: give the ON CONFLICT something to bite on for EVERY
source (repo re-scans, connector re-delivers, ...), and drop the redundant
pending backlog that accumulated before the gate landed.

Shape
-----
Partial unique index on ``(customer_id, anchor_doc_id, extractor_id) WHERE
done_at IS NULL``: at most one OUTSTANDING row per key, while still allowing a
genuine re-extraction after a row completes (prompt-version bump, real content
change) to enqueue a fresh row. Mirrors the existing partial
``idx_inferred_edges_queue_pending`` predicate style.

Existing duplicate outstanding rows must be collapsed first or the index build
fails. We take a SHARE ROW EXCLUSIVE lock (blocks concurrent writers, permits
readers) so the dedup DELETE and the index build cannot race a fresh insert,
then keep the lowest ``id`` per key. Deleting a row a worker happens to be
mid-processing is harmless: its later ``done_at`` update simply matches zero
rows, and edge upserts are idempotent.
"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op

# Keep <=32 chars (alembic_version.version_num is varchar(32)).
revision: str = "0096_inferred_edges_dedup"
down_revision: str | Sequence[str] | None = "0095_pending_edges"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Fail fast rather than wedge behind an active drain; a collision is
    # retriable. The queue table is small so the held window is sub-second.
    op.execute("SET lock_timeout = '10s'")
    op.execute(
        "LOCK TABLE inferred_edges_queue IN SHARE ROW EXCLUSIVE MODE"
    )
    # Collapse duplicate OUTSTANDING (not-yet-done) rows, keeping the lowest id
    # per (customer, anchor_doc, extractor). Done rows are untouched.
    op.execute(
        """
        DELETE FROM inferred_edges_queue a
        USING inferred_edges_queue b
        WHERE a.done_at IS NULL
          AND b.done_at IS NULL
          AND a.customer_id   = b.customer_id
          AND a.anchor_doc_id = b.anchor_doc_id
          AND a.extractor_id  = b.extractor_id
          AND a.id > b.id
        """
    )
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_inferred_edges_queue_outstanding
            ON inferred_edges_queue (customer_id, anchor_doc_id, extractor_id)
            WHERE done_at IS NULL
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_inferred_edges_queue_outstanding")
