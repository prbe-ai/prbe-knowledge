"""Drop FORCE RLS from inferred_edges_queue (shared-managed prep)

Revision ID: 0068_inferred_edges_no_force
Revises: 0067_with_check_tenant_pol
Create Date: 2026-05-13

Background
----------
The Phase 4 shared-managed migration moves prbe-knowledge from connecting
as the ``probe`` SUPERUSER (which bypasses RLS entirely) to the
non-privileged ``probe_app`` role. Under ``probe_app``, every RLS policy
is enforced — including against the table OWNER when ``FORCE ROW LEVEL
SECURITY`` is set.

Migration 0055 created ``inferred_edges_queue`` with ENABLE + FORCE RLS
and a USING/WITH CHECK policy on ``app.current_customer_id``. The
inferred-edges worker (``services/ingestion/inferred_edges/worker.py``)
drains this queue **cross-tenant**: ``_claim_one()`` does ``SELECT ...
FROM inferred_edges_queue WHERE processing_started_at IS NULL ...
FOR UPDATE SKIP LOCKED LIMIT 1`` without a customer_id filter, picking
up whichever pending row is oldest.

Under FORCE RLS on the new ``probe_app`` role, that claim SELECT
silently returns zero rows (the GUC isn't set yet — there's no
customer_id to set it to until we've claimed the row). The worker would
spin doing nothing forever, and inferred-edges extraction would halt
across the fleet.

This migration brings ``inferred_edges_queue`` in line with the
established convention for internal queue tables in this codebase:

  * ``ingestion_queue``           — no RLS (queue-style drain)
  * ``backfill_state``            — no RLS (queue-style drain)
  * ``wiki_synthesis_queue``      — RLS dropped in 0034 (same reason)
  * ``wiki_synthesis_runs``       — RLS dropped in 0034 (same reason)

The pattern: internal queue tables drained cross-customer leave RLS off
entirely; tenant scoping for per-customer operations is enforced by
application code via ``shared.db.with_tenant(customer_id)`` AND an
explicit ``customer_id = $1`` WHERE clause. The inferred-edges worker
already follows this — line 141 of worker.py wraps the per-row processing
in ``with_tenant(customer_id)`` after claim.

This migration drops the policy + RLS on the queue table only. The
graph_edges writes the worker eventually makes (via
``graph_writer.upsert_edges`` inside ``with_tenant``) remain
RLS-protected — that's the actual tenant data plane.

Postgres-only — RLS is a PG-only feature. The migration uses raw DDL
under ``op.execute(...)`` consistent with surrounding migrations.

The reverse (downgrade) recreates the policy + FORCE state from 0055
for safety, even though re-enabling it would break the worker again.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0068_inferred_edges_no_force"
down_revision: str | Sequence[str] | None = "0067_with_check_tenant_pol"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "DROP POLICY IF EXISTS inferred_edges_queue_tenant_isolation "
        "ON inferred_edges_queue"
    )
    op.execute("ALTER TABLE inferred_edges_queue NO FORCE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE inferred_edges_queue DISABLE ROW LEVEL SECURITY")


def downgrade() -> None:
    op.execute("ALTER TABLE inferred_edges_queue ENABLE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY inferred_edges_queue_tenant_isolation
            ON inferred_edges_queue
            USING (customer_id = current_setting('app.current_customer_id', true))
            WITH CHECK (customer_id = current_setting('app.current_customer_id', true))
        """
    )
    op.execute("ALTER TABLE inferred_edges_queue FORCE ROW LEVEL SECURITY")
