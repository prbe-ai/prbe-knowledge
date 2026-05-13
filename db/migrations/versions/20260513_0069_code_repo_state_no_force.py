"""Drop FORCE RLS from code_repo_state (preemptive — matches 0068 pattern)

Revision ID: 0069_code_repo_state_no_force
Revises: 0068_inferred_edges_no_force
Create Date: 2026-05-13

Background
----------
``services/synthesis/nightly_trigger.py::refresh_cross_repo_edges`` runs a
cross-tenant aggregator:

    SELECT customer_id, ARRAY_AGG(DISTINCT repo) AS repos
    FROM code_repo_state
    GROUP BY customer_id

Under the new ``probe_app`` role (post Phase 4 cutover) this returns zero
rows silently because ``code_repo_state`` was created with FORCE ROW LEVEL
SECURITY in migration 0049 — the policy keys off
``app.current_customer_id``, which is unset for a bare cross-tenant query.

The aggregator is currently behind a ``DO NOT REVIVE WITHOUT FIXING RLS``
guard comment, but the failure mode (silent zero rows + no errors) is a
landmine for whoever flips it back on. We're fixing it preemptively here,
mirroring the precedent set in migration 0068 for ``inferred_edges_queue``:

  * ``ingestion_queue``           — no RLS (queue-style table)
  * ``backfill_state``            — no RLS (queue-style table)
  * ``wiki_synthesis_queue``      — RLS dropped in 0034
  * ``wiki_synthesis_runs``       — RLS dropped in 0034
  * ``inferred_edges_queue``      — RLS dropped in 0068

``code_repo_state`` is a per-(customer, repo, file) extraction cache —
it's not user-facing tenant data, it's the bookkeeping table for the
code-graph extractor. Tenant scoping for per-customer reads/writes
remains enforced by the application code: every call site already wraps
its ``code_repo_state`` UPSERT / SELECT in ``with_tenant(customer_id)``
AND an explicit ``customer_id = $1`` WHERE clause. The cross-tenant
aggregator becomes safe-to-revive without per-tenant looping.

The actual tenant data plane (graph_nodes / graph_edges / documents /
chunks etc.) stays RLS-protected — that's where the real content lives.

The reverse (downgrade) recreates the policy + FORCE state from 0049
for safety, even though re-enabling it re-introduces the silent-zero
landmine for the cross-tenant aggregator.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0069_code_repo_state_no_force"
down_revision: str | Sequence[str] | None = "0068_inferred_edges_no_force"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "DROP POLICY IF EXISTS code_repo_state_tenant_isolation "
        "ON code_repo_state"
    )
    op.execute("ALTER TABLE code_repo_state NO FORCE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE code_repo_state DISABLE ROW LEVEL SECURITY")


def downgrade() -> None:
    op.execute("ALTER TABLE code_repo_state ENABLE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY code_repo_state_tenant_isolation ON code_repo_state
            USING (customer_id = current_setting('app.current_customer_id', true))
        """
    )
    op.execute("ALTER TABLE code_repo_state FORCE ROW LEVEL SECURITY")
