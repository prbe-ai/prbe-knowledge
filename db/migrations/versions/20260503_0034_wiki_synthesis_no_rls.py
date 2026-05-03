"""Drop RLS from wiki_synthesis_queue + wiki_synthesis_runs

Revision ID: 0034_wiki_synthesis_no_rls
Revises: 0033_wiki_synthesis
Create Date: 2026-05-03

Migration 0033 created both tables with `ENABLE` + `FORCE ROW LEVEL
SECURITY` and a tenant-isolation policy. That was wrong: these are
**internal queue tables** drained cross-customer by the worker (the
synthesis cron's `_tick` does `SELECT DISTINCT customer_id FROM
wiki_synthesis_queue WHERE status = 'pending'`), and FORCE RLS would
make that SELECT silently return zero rows under the worker's
application role.

The convention for internal queue tables in this codebase is to leave
RLS off entirely — see `ingestion_queue` and `backfill_state`, both
RLS-disabled. Tenant scoping for *per-customer* operations is enforced
by application code: every per-customer query uses
`shared.db.with_tenant(customer_id)` AND filters on `customer_id`
explicitly. The cron loop uses the same discipline.

This migration brings 0033 into line with that convention by dropping
the RLS policy and disabling RLS on both tables.
"""

from __future__ import annotations

from alembic import op

revision = "0034_wiki_synthesis_no_rls"
down_revision = "0033_wiki_synthesis"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "DROP POLICY IF EXISTS wiki_synthesis_queue_tenant_isolation "
        "ON wiki_synthesis_queue"
    )
    op.execute("ALTER TABLE wiki_synthesis_queue NO FORCE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE wiki_synthesis_queue DISABLE ROW LEVEL SECURITY")

    op.execute(
        "DROP POLICY IF EXISTS wiki_synthesis_runs_tenant_isolation "
        "ON wiki_synthesis_runs"
    )
    op.execute("ALTER TABLE wiki_synthesis_runs NO FORCE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE wiki_synthesis_runs DISABLE ROW LEVEL SECURITY")


def downgrade() -> None:
    op.execute("ALTER TABLE wiki_synthesis_queue ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE wiki_synthesis_queue FORCE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY wiki_synthesis_queue_tenant_isolation ON wiki_synthesis_queue
            USING (customer_id = current_setting('app.current_customer_id', true))
        """
    )

    op.execute("ALTER TABLE wiki_synthesis_runs ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE wiki_synthesis_runs FORCE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY wiki_synthesis_runs_tenant_isolation ON wiki_synthesis_runs
            USING (customer_id = current_setting('app.current_customer_id', true))
        """
    )
