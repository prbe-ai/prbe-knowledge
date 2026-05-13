"""Enable RLS + tenant policy on graph_node_provenance (audit fix #51)

Revision ID: 0070_gnp_rls
Revises: 0069_code_repo_state_no_force
Create Date: 2026-05-13

Audit finding (#51)
-------------------
``graph_node_provenance`` was created in migration 0011 WITHOUT enabling
row level security or creating a tenant_isolation policy. Migration 0052
(``codegraph_file_as_document``) later toggles
``FORCE ROW LEVEL SECURITY`` on the table -- but ``FORCE`` is a no-op
unless RLS is also ``ENABLE``d, so today the table relies entirely on the
application layer using ``with_tenant()``.

That works for current callers (graph_writer.py, codegraph handler, and
the retrieval graph retriever all wrap their access in ``with_tenant``),
but it's brittle:

  * Defense in depth: every other tenant-scoped table in this schema
    (graph_nodes, graph_edges, code_repo_state, query_traces,
    usage_events, etc.) has ENABLE + FORCE + a policy. Provenance is the
    odd one out.
  * Migration 0052 explicitly ``FORCE``s the table assuming RLS is on,
    and migration 0067 implies the audit intent is "every tenant-scoped
    table has USING + WITH CHECK". A future contributor adding ENABLE
    without a policy would brick the table silently (FORCE + ENABLE +
    no policy = deny-all).

This migration brings ``graph_node_provenance`` in line with the rest of
the schema:

  1. ``ENABLE ROW LEVEL SECURITY``
  2. ``FORCE ROW LEVEL SECURITY`` (idempotent re-affirm; 0052 already
      issued this)
  3. ``CREATE POLICY tenant_isolation`` with both ``USING`` and
      ``WITH CHECK`` keyed on
      ``current_setting('app.current_customer_id', true)``

Pattern mirrors migration 0067 (with_check_tenant_policies): drop-first
guarded by ``DO $$`` so re-running after a partial apply is a no-op.

Verification
------------

    SELECT relname,
           relrowsecurity AS rls_enabled,
           relforcerowsecurity AS rls_forced
    FROM   pg_class
    WHERE  relname = 'graph_node_provenance';

    SELECT polname, polqual::text, polwithcheck::text
    FROM   pg_policy
    WHERE  polrelid = 'graph_node_provenance'::regclass;
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

# Keep <=32 chars (alembic_version.version_num is varchar(32)).
revision: str = "0070_gnp_rls"
down_revision: str | Sequence[str] | None = "0069_code_repo_state_no_force"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


_POLICY_EXPR = "customer_id = current_setting('app.current_customer_id', true)"


def upgrade() -> None:
    # 1 + 2. ENABLE + FORCE RLS. Both are idempotent at the catalog level
    # (Postgres no-ops if already on). FORCE was already issued by
    # migration 0052; ENABLE is the missing piece this migration adds.
    op.execute("ALTER TABLE graph_node_provenance ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE graph_node_provenance FORCE ROW LEVEL SECURITY")

    # 3. Policy. DO $$ guard mirrors migration 0067 so a partial apply +
    # re-run is a no-op (CREATE POLICY has no IF NOT EXISTS in PG 16).
    op.execute(
        f"""
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1
                FROM   pg_policies
                WHERE  schemaname = current_schema()
                  AND  tablename  = 'graph_node_provenance'
                  AND  policyname = 'tenant_isolation'
                  AND  with_check IS NOT NULL
            ) THEN
                RETURN;
            END IF;

            EXECUTE 'DROP POLICY IF EXISTS tenant_isolation ON graph_node_provenance';
            EXECUTE $ddl$
                CREATE POLICY tenant_isolation ON graph_node_provenance
                    USING ({_POLICY_EXPR})
                    WITH CHECK ({_POLICY_EXPR})
            $ddl$;
        END $$;
        """
    )


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS tenant_isolation ON graph_node_provenance")
    # Leave FORCE in place (matches the pre-migration state -- 0052 set it).
    op.execute("ALTER TABLE graph_node_provenance DISABLE ROW LEVEL SECURITY")
