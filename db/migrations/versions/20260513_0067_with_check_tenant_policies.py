"""WITH CHECK on tenant-scoped RLS policies (Phase 4 defense-in-depth)

Revision ID: 0067_with_check_tenant_pol
Revises: 0066_cit_search_path_fix
Create Date: 2026-05-13

Background
----------
Phase 4's cross-tenant denial integration test (prbe-backend PR #200) audited
the tenant-scoped RLS policies on the data-plane schema and found that only
``directed_vectors`` (migration 0061) and ``inferred_edges_queue`` (migration
0055) had BOTH ``USING`` and ``WITH CHECK`` clauses on their tenant-isolation
policies.

The other tenant-scoped tables had ``USING``-only policies. ``USING`` filters
what existing rows are visible (SELECT/UPDATE/DELETE) but does NOT block an
INSERT or an UPDATE that would CREATE a row for a different ``customer_id``
than the GUC's value. With FORCE ROW LEVEL SECURITY on those tables, a buggy
or malicious code path could insert rows under tenant A's connection that are
attributed to tenant B's ``customer_id`` -- and the row would simply not be
visible back to tenant A under its own GUC, "succeeding" silently.

The ``WITH CHECK`` clause is the symmetric defense: it asserts the row's
``customer_id`` matches the GUC at write time and rejects the INSERT/UPDATE
otherwise.

Tables touched
--------------
This migration adds ``WITH CHECK`` to the USING-only tenant policies on:

  * ``graph_nodes``           (policy ``tenant_isolation``)
  * ``graph_edges``           (policy ``tenant_isolation``)
  * ``usage_events``          (policy ``usage_events_tenant_isolation``)
  * ``query_traces``          (policy ``query_traces_tenant_isolation``)
  * ``code_repo_state``       (policy ``code_repo_state_tenant_isolation``)

Note: ``directed_vectors``, ``inferred_edges_queue``, and
``custom_ingest_tokens`` already have ``WITH CHECK`` (added in their own
migrations) -- this migration leaves them alone.

Idempotency
-----------
``ALTER POLICY`` cannot add ``WITH CHECK`` to an existing policy (it can
modify USING/WITH CHECK expressions, but only when both already exist on
the policy). The portable idempotent pattern is DROP + CREATE wrapped in a
``DO $$ ... $$`` guard that checks ``pg_policies.with_check`` first and
skips when it's already populated -- so re-running this migration after a
partial apply is a no-op.

Postgres-only
-------------
RLS is a Postgres-only feature. CI for prbe-knowledge runs migrations
against a real Postgres (see ``.github/workflows/tests.yml``), so a dialect
gate is unnecessary. The DDL is still wrapped in ``op.execute(...)`` for
consistency with the surrounding migrations.

Verification
------------
After applying, ``pg_policies`` shows the new ``with_check`` clause::

    SELECT polname, polrelid::regclass AS tbl, polqual::text AS using_expr,
           polwithcheck::text AS check_expr
    FROM   pg_policy
    WHERE  polrelid::regclass::text IN
           ('graph_nodes','graph_edges','usage_events',
            'query_traces','code_repo_state');
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0067_with_check_tenant_pol"
down_revision: str | Sequence[str] | None = "0066_cit_search_path_fix"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# Per-table tuple: (policy_name, table_name).
# The USING/WITH CHECK expression is identical across all five policies, so
# the migration body inlines it once. customer_id is TEXT in this schema --
# do NOT cast to uuid (it's an opaque slug like ``probe-founders``).
_TARGETS: tuple[tuple[str, str], ...] = (
    ("tenant_isolation", "graph_nodes"),
    ("tenant_isolation", "graph_edges"),
    ("usage_events_tenant_isolation", "usage_events"),
    ("query_traces_tenant_isolation", "query_traces"),
    ("code_repo_state_tenant_isolation", "code_repo_state"),
)


_POLICY_EXPR = "customer_id = current_setting('app.current_customer_id', true)"


def _add_with_check(policy: str, table: str) -> None:
    """Idempotently add WITH CHECK to an existing USING-only policy.

    DO-block guard: if pg_policies already shows a non-null ``with_check``
    for ``(table, policy)``, skip. Otherwise DROP and re-CREATE with both
    clauses present. CREATE POLICY has no ``IF NOT EXISTS`` in Postgres
    16, so drop-then-create is the portable idempotent shape (matches the
    pattern in migration 0061_directed_vectors).
    """
    op.execute(
        f"""
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1
                FROM   pg_policies
                WHERE  schemaname = current_schema()
                  AND  tablename  = '{table}'
                  AND  policyname = '{policy}'
                  AND  with_check IS NOT NULL
            ) THEN
                -- Already has WITH CHECK; nothing to do.
                RETURN;
            END IF;

            EXECUTE 'DROP POLICY IF EXISTS {policy} ON {table}';
            EXECUTE $ddl$
                CREATE POLICY {policy} ON {table}
                    USING ({_POLICY_EXPR})
                    WITH CHECK ({_POLICY_EXPR})
            $ddl$;
        END $$;
        """
    )


def _drop_with_check(policy: str, table: str) -> None:
    """Downgrade: revert to USING-only (the pre-migration shape)."""
    op.execute(f"DROP POLICY IF EXISTS {policy} ON {table}")
    op.execute(
        f"""
        CREATE POLICY {policy} ON {table}
            USING ({_POLICY_EXPR})
        """
    )


def upgrade() -> None:
    for policy, table in _TARGETS:
        _add_with_check(policy, table)


def downgrade() -> None:
    # Walk in reverse for symmetry; each table is independent so order
    # doesn't actually matter.
    for policy, table in reversed(_TARGETS):
        _drop_with_check(policy, table)
