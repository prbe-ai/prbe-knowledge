"""Per-source purge bookkeeping: purge_runs

Revision ID: 0097_purge_runs
Revises: 0096_inferred_edges_dedup
Create Date: 2026-07-24

Why
---
Disconnecting an integration has to delete everything that integration ever
ingested. The engine gained ``POST /purge`` for that, and the caller
(research-os) removes its own record of the connection ONLY after the purge
reports ``verified=true``.

That contract needs an outcome that outlives the HTTP request. A client whose
connection dropped, a pod that restarted mid-cascade, or a purge that ran
longer than the caller's timeout all leave the caller unable to answer "did it
finish?" -- and re-running blind is not an answer when the alternative is
leaving tenant data indexed after a user asked for it to be gone.

``purge_runs`` records one row per purge attempt: which source, whether the
postcondition pass verified clean, per-table row counts, and any residue found.
The status endpoint reads it, so the answer survives everything above.

Shape
-----
Tenant-scoped and RLS-forced like every other per-customer table, so a missing
``with_tenant`` scope fails closed instead of leaking another tenant's purge
history. ``result`` is JSONB rather than columns because the shape is
diagnostic detail (per-table counts, residue map) that will grow as the cascade
grows, and nothing queries inside it.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

# Keep <=32 chars (alembic_version.version_num is varchar(32)).
revision: str = "0097_purge_runs"
down_revision: str | Sequence[str] | None = "0096_inferred_edges_dedup"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # CREATE TABLE takes no lock on existing tables; keep a short bound anyway
    # so a migrate hook never wedges behind an unrelated long transaction.
    op.execute("SET lock_timeout = '5s'")
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS purge_runs (
            purge_id       UUID PRIMARY KEY,
            customer_id    TEXT NOT NULL
                           REFERENCES customers(customer_id) ON DELETE CASCADE,
            source_system  TEXT NOT NULL,
            status         TEXT NOT NULL DEFAULT 'running'
                           CHECK (status IN ('running','done','failed')),
            result         JSONB,
            error          TEXT,
            started_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            finished_at    TIMESTAMPTZ
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_purge_runs_customer_source
            ON purge_runs (customer_id, source_system, started_at DESC)
        """
    )
    op.execute("ALTER TABLE purge_runs ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE purge_runs FORCE ROW LEVEL SECURITY")
    op.execute("DROP POLICY IF EXISTS tenant_isolation ON purge_runs")
    op.execute(
        """
        CREATE POLICY tenant_isolation ON purge_runs
            USING (customer_id = current_setting('app.current_customer_id', true))
            WITH CHECK (customer_id = current_setting('app.current_customer_id', true))
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS purge_runs")
