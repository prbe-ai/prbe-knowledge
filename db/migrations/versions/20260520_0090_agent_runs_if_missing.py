"""agent_runs + agent_turn_traces — defensive create-if-missing

Revision ID: 0090_agent_runs_if_missing
Revises: 0089_documents_title_trgm
Create Date: 2026-05-20

Some managed environments were bootstrapped via the `db/schema.sql` +
`alembic stamp head` shortcut path (see `scripts/neon-migrate.sh` —
"Fresh local DB detected — applying schema.sql + stamping alembic head").
schema.sql intentionally does NOT include `agent_runs` /
`agent_turn_traces` (they were owned by the now-archived prbe-orchestrator
service and never folded into the canonical seed). Result: alembic
reports head 0088+ but the tables created in migrations 0008
(`enrichment_runs`) → 0058 (rename to `agent_runs`) → 0017
(retry/heartbeat cols) → 0041/0043 (`agent_turn_traces`) → 0059
(`model` col) were never run.

After the orchestrator port into prbe-apps (May 2026) the apps plane
relies on `agent_runs` for every investigation + post-approval run.
Without these tables, every PD/incident.io investigation lands in
`failed_pending_review` because `INSERT INTO agent_runs` throws
`UndefinedTable`.

This migration creates BOTH tables with their post-rename, all-columns-
included final shape using `CREATE TABLE IF NOT EXISTS`. It is a no-op
on any environment that already has the tables (either via the original
migration chain or via my prior local manual fixup) and a recovery for
the schema-sql-stamped environments where they're absent.

Indexes + the FK from agent_turn_traces → agent_runs use IF NOT EXISTS
forms too. The CHECK constraint is added via a DO block since Postgres
lacks `ADD CONSTRAINT IF NOT EXISTS`.
"""

from __future__ import annotations

from alembic import op

revision = "0090_agent_runs_if_missing"
down_revision = "0089_documents_title_trgm"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_runs (
            run_id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            customer_id            TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
            source                 TEXT NOT NULL,
            source_event_id        TEXT NOT NULL,
            ticket_id              TEXT,
            agent_kind             TEXT NOT NULL DEFAULT 'enrichment',
            subject_kind           TEXT NOT NULL DEFAULT 'ticket',
            status                 TEXT NOT NULL DEFAULT 'pending',
            comment_id             TEXT,
            report                 JSONB,
            report_schema_version  SMALLINT,
            token_usage_input      INTEGER,
            token_usage_output     INTEGER,
            model                  TEXT,
            payload                JSONB NOT NULL DEFAULT '{}'::jsonb,
            payload_version        SMALLINT NOT NULL DEFAULT 1,
            attempt_count          INTEGER NOT NULL DEFAULT 0,
            heartbeat_at           TIMESTAMPTZ,
            next_retry_at          TIMESTAMPTZ,
            started_at             TIMESTAMPTZ,
            finished_at            TIMESTAMPTZ,
            error                  TEXT,
            error_class            TEXT,
            last_error             TEXT,
            created_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at             TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );

        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint
                WHERE conname = 'agent_runs_status_check'
            ) THEN
                ALTER TABLE agent_runs ADD CONSTRAINT agent_runs_status_check
                    CHECK (status IN ('pending','processing','succeeded','failed','skipped','dlq'));
            END IF;
        END$$;

        CREATE UNIQUE INDEX IF NOT EXISTS agent_runs_idempotency_v2
            ON agent_runs (customer_id, source, source_event_id, agent_kind);

        CREATE INDEX IF NOT EXISTS agent_runs_status_created
            ON agent_runs (status, created_at);

        CREATE TABLE IF NOT EXISTS agent_turn_traces (
            trace_id      UUID NOT NULL DEFAULT gen_random_uuid(),
            run_id        UUID NOT NULL,
            customer_id   TEXT NOT NULL,
            turn_idx      INTEGER NOT NULL,
            event_kind    TEXT NOT NULL,
            payload       JSONB NOT NULL,
            input_tokens  INTEGER,
            output_tokens INTEGER,
            created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CONSTRAINT agent_turn_traces_pkey PRIMARY KEY (trace_id),
            CONSTRAINT fk_agent_turn_traces_run_id
                FOREIGN KEY (run_id) REFERENCES agent_runs(run_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS agent_turn_traces_run_idx
            ON agent_turn_traces (run_id, turn_idx);

        CREATE INDEX IF NOT EXISTS agent_turn_traces_customer_idx
            ON agent_turn_traces (customer_id, created_at DESC);
        """
    )


def downgrade() -> None:
    # No-op: this migration's job is recovery on environments that should
    # have had the tables for years (since 0008 in April 2026). Dropping
    # them in a downgrade would destroy real run history on the rare
    # environments where the original migration chain DID run. If a
    # downgrade is truly needed, do it manually with explicit DROP TABLE
    # statements after a backup.
    pass
