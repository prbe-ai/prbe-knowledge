"""kg RLS: tenant_isolation policies on kg_classes, kg_evidence, kg_candidates

Revision ID: 0034_kg_rls
Revises: 0033_kg_indexes
Create Date: 2026-04-30

Fifth and final migration in the Phase 1 foundation of the debugging
knowledge graph (see docs/superpowers/specs/2026-04-29-debugging-
knowledge-graph-design.md §5.1, §12.3). Enables and forces row-level
security on the three kg_* tables introduced by 0030/0031/0032, and
attaches a tenant_isolation policy filtering on the existing
`app.current_customer_id` GUC that `shared/db.with_tenant()` sets.

Pattern matches the canonical RLS migration 0020_usage_events exactly:

  ALTER TABLE <t> ENABLE ROW LEVEL SECURITY;
  ALTER TABLE <t> FORCE ROW LEVEL SECURITY;
  CREATE POLICY <t>_tenant_isolation ON <t>
      USING (customer_id = current_setting('app.current_customer_id', true));

Notes:
  * USING only, no WITH CHECK. 0020 (and the older graph_nodes /
    graph_edges policies) use USING-only; this is fail-closed for reads,
    and the same predicate would gate writes via standard RLS write
    semantics if WITH CHECK were specified. We deliberately mirror 0020
    rather than introducing WITH CHECK as a one-off here — uniformity
    matters more than the marginal write-side guarantee, and any future
    hardening should sweep all RLS-protected tables together.
  * No bypass role. The repo's existing RLS-protected tables
    (graph_nodes, graph_edges, usage_events, plus the cascade tables
    in 0005) do not use one. Services needing cross-tenant access
    enter `with_tenant(<id>)` per request; bootstrap / cron / infra
    paths use `raw_conn()` (which acquires a connection without setting
    the GUC, so RLS-protected SELECTs return zero rows — that's by
    design; such ops should not touch tenant-scoped tables).
  * FORCE is required so the policy applies to the table owner too
    (asyncpg connects as the owner role in this codebase).
  * Failure mode: any code path that touches kg_* outside `with_tenant()`
    will silently see zero rows. This is the explicit contract.

Out of scope:
  * Bypass / staff role. Not an existing pattern in this repo; the
    Phase 1 plan's mention of `kg_staff` was reconciled against the
    actual codebase before writing this migration.
  * Per-write WITH CHECK (see note above).
"""

from __future__ import annotations

from alembic import op

revision = "0034_kg_rls"
down_revision = "0033_kg_indexes"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE kg_classes ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE kg_classes FORCE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY kg_classes_tenant_isolation ON kg_classes
            USING (customer_id = current_setting('app.current_customer_id', true))
        """
    )

    op.execute("ALTER TABLE kg_evidence ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE kg_evidence FORCE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY kg_evidence_tenant_isolation ON kg_evidence
            USING (customer_id = current_setting('app.current_customer_id', true))
        """
    )

    op.execute("ALTER TABLE kg_candidates ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE kg_candidates FORCE ROW LEVEL SECURITY")
    op.execute(
        """
        CREATE POLICY kg_candidates_tenant_isolation ON kg_candidates
            USING (customer_id = current_setting('app.current_customer_id', true))
        """
    )


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS kg_candidates_tenant_isolation ON kg_candidates")
    op.execute("ALTER TABLE kg_candidates NO FORCE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE kg_candidates DISABLE ROW LEVEL SECURITY")

    op.execute("DROP POLICY IF EXISTS kg_evidence_tenant_isolation ON kg_evidence")
    op.execute("ALTER TABLE kg_evidence NO FORCE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE kg_evidence DISABLE ROW LEVEL SECURITY")

    op.execute("DROP POLICY IF EXISTS kg_classes_tenant_isolation ON kg_classes")
    op.execute("ALTER TABLE kg_classes NO FORCE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE kg_classes DISABLE ROW LEVEL SECURITY")
