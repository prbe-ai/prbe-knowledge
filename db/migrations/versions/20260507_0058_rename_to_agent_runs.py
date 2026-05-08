"""rename enrichment_runs -> agent_runs; widen idempotency; allow dedupe kind

Revision ID: 0058_rename_to_agent_runs
Revises: 0057_fix_run_columns_a
Create Date: 2026-05-07

The orchestrator is generalizing from a single "enrichment" pipeline to
a multi-stage "agent_runs" pipeline. The first concrete instance: a new
'dedupe' agent_kind runs *before* the existing 'ticket' enrichment for
the same inbound webhook event, and both rows must coexist in the runs
table -- one per agent_kind -- so each stage's lifecycle (claim, retry,
DLQ, paused) is tracked independently.

This migration does four things:

  * Renames the table from `enrichment_runs` to `agent_runs`, plus its
    indexes and the status / agent_kind CHECK constraints, so the table
    name reflects the orchestrator's vocabulary instead of the historical
    "enrichment" framing. All renames are metadata-only and cheap; no
    data rewrite.
  * Replaces the old 3-column idempotency UNIQUE
    (customer_id, source, source_event_id) with a 4-column UNIQUE
    (customer_id, source, source_event_id, agent_kind). The old
    constraint blocked the dedupe -> ticket pipeline because both rows
    share (customer_id, source, source_event_id) by design -- they're
    distinct work units for the same webhook, differentiated only by
    agent_kind.
  * Renames the Lane A index `idx_enr_runs_tnt_kind_status` (added by
    0057_fix_run_columns) to `idx_agnt_runs_tnt_kind_status` so it stays
    discoverable under the new table name.
  * Extends the agent_kind CHECK constraint to allow 'dedupe' alongside
    the existing 'ticket' / 'debug' / 'dev' / 'fix' values. Without this
    every dedupe insert from the orchestrator would violate the check
    and 4xx the webhook.

Chain note: this lands AFTER 0057_fix_run_columns (Lane A's enrichment_
runs columns + concurrency index). Both originally targeted
0056_documents_id_trgm_idx; ordering them serially keeps alembic's
revision DAG linear.

Downgrade WILL FAIL on `CREATE UNIQUE INDEX enrichment_runs_idempotency`
if any (customer_id, source, source_event_id) tuple already has multiple
rows differing only in agent_kind. That is the correct behavior -- those
rows are real distinct work units, and silently collapsing them would
lose data. To downgrade in that situation, the operator must first
delete (or merge) the dedupe rows by hand.
"""

from __future__ import annotations

from alembic import op

revision = "0058_rename_to_agent_runs"
down_revision = "0057_fix_run_columns_a"
branch_labels = None
depends_on = None


# Constraint and index names borrowed from upstream so the rename hits
# the right objects.
_LANE_A_INDEX_OLD = "idx_enr_runs_tnt_kind_status"
_LANE_A_INDEX_NEW = "idx_agnt_runs_tnt_kind_status"
_AGENT_KIND_CHECK_OLD = "enrichment_runs_agent_kind_check"
_AGENT_KIND_CHECK_NEW = "agent_runs_agent_kind_check"


def upgrade() -> None:
    # 1. Rename the table itself. Cheap metadata-only operation.
    op.execute("ALTER TABLE enrichment_runs RENAME TO agent_runs")

    # 2. Rename indexes to match the new table name. The idempotency
    # index is renamed to a temporary "_old" name so we can build the
    # new wider UNIQUE under the canonical name first, then drop the
    # legacy one — keeps the dedupe guarantee live across the swap.
    op.execute(
        "ALTER INDEX enrichment_runs_idempotency RENAME TO agent_runs_idempotency_old"
    )
    op.execute(
        "ALTER INDEX enrichment_runs_status_created RENAME TO agent_runs_status_created"
    )
    op.execute("ALTER INDEX enrichment_runs_claim_idx RENAME TO agent_runs_claim_idx")
    op.execute(
        "ALTER INDEX enrichment_runs_heartbeat_idx RENAME TO agent_runs_heartbeat_idx"
    )
    op.execute("ALTER INDEX idx_enrichment_runs_listing RENAME TO idx_agent_runs_listing")
    op.execute(
        "ALTER INDEX enrichment_runs_subject_idx RENAME TO agent_runs_subject_idx"
    )

    # 3. Rename the Lane A concurrency index added by 0057_fix_run_columns.
    # Doing this in the same migration as the table rename keeps the
    # naming consistent under the new `agent_runs` table.
    op.execute(
        f"ALTER INDEX {_LANE_A_INDEX_OLD} RENAME TO {_LANE_A_INDEX_NEW}"
    )

    # 4. Rename the status CHECK constraint to match the new table name.
    op.execute(
        "ALTER TABLE agent_runs RENAME CONSTRAINT enrichment_runs_status_check TO agent_runs_status_check"
    )

    # 5. Rename the agent_kind CHECK constraint AND extend it to allow
    # 'dedupe'. The orchestrator inserts agent_kind='dedupe' rows in the
    # new pipeline; without this addition every dedupe insert would
    # 4xx with a CHECK violation. Postgres can't ALTER a CHECK in place;
    # rename + drop + add is the standard pattern (mirrors what
    # 0057_fix_run_columns did to add 'fix').
    op.execute(
        f"ALTER TABLE agent_runs RENAME CONSTRAINT {_AGENT_KIND_CHECK_OLD} TO {_AGENT_KIND_CHECK_NEW}"
    )
    op.execute(
        f"ALTER TABLE agent_runs DROP CONSTRAINT IF EXISTS {_AGENT_KIND_CHECK_NEW}"
    )
    op.execute(
        f"ALTER TABLE agent_runs ADD CONSTRAINT {_AGENT_KIND_CHECK_NEW} "
        "CHECK (agent_kind IN ('ticket', 'debug', 'dev', 'fix', 'dedupe'))"
    )

    # 6. Widen idempotency to include agent_kind so two rows can coexist
    # for the same webhook (one per agent_kind in the new dedupe ->
    # ticket pipeline). Build the new UNIQUE first, then drop the old
    # 3-col one so the swap commits atomically (the whole migration
    # runs under ACCESS EXCLUSIVE, so this isn't about live-writer
    # safety -- it's about not committing a transaction that has no
    # idempotency UNIQUE on the table).
    op.execute(
        """
        CREATE UNIQUE INDEX agent_runs_idempotency
            ON agent_runs (customer_id, source, source_event_id, agent_kind)
        """
    )
    op.execute("DROP INDEX agent_runs_idempotency_old")


def downgrade() -> None:
    # 1. Recreate the original 3-column UNIQUE under its historical
    # name. WILL FAIL if any (customer_id, source, source_event_id)
    # tuple has multiple rows differing only in agent_kind -- that's
    # correct, those rows are real distinct work units and silently
    # collapsing them would lose data. Operator must delete dedupe
    # rows by hand before retrying downgrade.
    op.execute("DROP INDEX agent_runs_idempotency")
    op.execute(
        """
        CREATE UNIQUE INDEX enrichment_runs_idempotency
            ON agent_runs (customer_id, source, source_event_id)
        """
    )

    # 2. Restore the agent_kind CHECK to its pre-dedupe value set, then
    # rename it back to the historical name.
    op.execute(
        f"ALTER TABLE agent_runs DROP CONSTRAINT IF EXISTS {_AGENT_KIND_CHECK_NEW}"
    )
    op.execute(
        f"ALTER TABLE agent_runs ADD CONSTRAINT {_AGENT_KIND_CHECK_NEW} "
        "CHECK (agent_kind IN ('ticket', 'debug', 'dev', 'fix'))"
    )
    op.execute(
        f"ALTER TABLE agent_runs RENAME CONSTRAINT {_AGENT_KIND_CHECK_NEW} TO {_AGENT_KIND_CHECK_OLD}"
    )

    # 3. Rename the status CHECK constraint back.
    op.execute(
        "ALTER TABLE agent_runs RENAME CONSTRAINT agent_runs_status_check TO enrichment_runs_status_check"
    )

    # 4. Rename indexes back to their historical names.
    op.execute(
        f"ALTER INDEX {_LANE_A_INDEX_NEW} RENAME TO {_LANE_A_INDEX_OLD}"
    )
    op.execute(
        "ALTER INDEX agent_runs_subject_idx RENAME TO enrichment_runs_subject_idx"
    )
    op.execute("ALTER INDEX idx_agent_runs_listing RENAME TO idx_enrichment_runs_listing")
    op.execute(
        "ALTER INDEX agent_runs_heartbeat_idx RENAME TO enrichment_runs_heartbeat_idx"
    )
    op.execute("ALTER INDEX agent_runs_claim_idx RENAME TO enrichment_runs_claim_idx")
    op.execute(
        "ALTER INDEX agent_runs_status_created RENAME TO enrichment_runs_status_created"
    )

    # 5. Rename the table back.
    op.execute("ALTER TABLE agent_runs RENAME TO enrichment_runs")
