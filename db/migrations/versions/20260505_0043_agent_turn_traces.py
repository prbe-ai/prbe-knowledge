"""Per-turn structured trace events for enrichment agent runs.

Revision ID: 0043_agent_turn_traces
Revises: 0042_backfill_codex_doc_titles
Create Date: 2026-05-05

WHY
---
Task 4 (prbe-orchestrator) writes one row per agent turn to capture the
full event stream for replay, debugging, and cost attribution. Each row
records the event kind (user_prompt, tool_calls, tool_returns,
model_response, complete, soft_cap_tripped, finalize_start,
finalize_end, unknown), the structured payload, and optional per-turn
token counts.

SCHEMA
------
agent_turn_traces
  trace_id      uuid PK   DEFAULT gen_random_uuid()
  run_id        uuid NOT NULL REFERENCES enrichment_runs(run_id) ON DELETE CASCADE
  customer_id   text NOT NULL
  turn_idx      int  NOT NULL
  event_kind    text NOT NULL
  payload       jsonb NOT NULL
  input_tokens  int  (nullable)
  output_tokens int  (nullable)
  created_at    timestamptz NOT NULL DEFAULT NOW()

INDEXES
-------
agent_turn_traces_run_idx      ON (run_id, turn_idx)
  -- fast replay: fetch all turns for a run in order.
agent_turn_traces_customer_idx ON (customer_id, created_at DESC)
  -- tenant-scoped time-range scans for the debug dashboard.

FK
--
run_id -> enrichment_runs(run_id) ON DELETE CASCADE so traces vanish
automatically when a run is purged.

NOTE: revision id kept under 32 chars because alembic_version.version_num
is varchar(32) by default; longer ids overflow the column and break the
deploy on first attempt (see migration 0028 for the gotcha). The
auto-generated id '89063d360118' is 12 chars -- safe.
"""

from __future__ import annotations

from alembic import op

revision = "0043_agent_turn_traces"
down_revision = "0042_backfill_codex_doc_titles"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_turn_traces (
            trace_id      UUID        NOT NULL DEFAULT gen_random_uuid(),
            run_id        UUID        NOT NULL,
            customer_id   TEXT        NOT NULL,
            turn_idx      INTEGER     NOT NULL,
            event_kind    TEXT        NOT NULL,
            payload       JSONB       NOT NULL,
            input_tokens  INTEGER,
            output_tokens INTEGER,
            created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CONSTRAINT agent_turn_traces_pkey PRIMARY KEY (trace_id),
            CONSTRAINT fk_agent_turn_traces_run_id
                FOREIGN KEY (run_id)
                REFERENCES enrichment_runs (run_id)
                ON DELETE CASCADE
        )
        """
    )
    # Fast replay: fetch all turns for a run in turn order.
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS agent_turn_traces_run_idx
        ON agent_turn_traces (run_id, turn_idx)
        """
    )
    # Tenant-scoped time-range scans for the debug dashboard.
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS agent_turn_traces_customer_idx
        ON agent_turn_traces (customer_id, created_at DESC)
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS agent_turn_traces_customer_idx")
    op.execute("DROP INDEX IF EXISTS agent_turn_traces_run_idx")
    op.execute("DROP TABLE IF EXISTS agent_turn_traces")
