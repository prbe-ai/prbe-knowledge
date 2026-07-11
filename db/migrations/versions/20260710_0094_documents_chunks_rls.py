"""Enable RLS + tenant policy on documents / chunks / failed_chunks

Revision ID: 0094_documents_chunks_rls
Revises: 0093_drop_agent_run_tables
Create Date: 2026-07-10

Security fix
------------
``documents``, ``chunks`` and ``failed_chunks`` -- the tables holding the
actual customer content -- were created WITHOUT row level security and
relied entirely on the application layer adding an explicit
``customer_id = $1`` filter to every query. Every other tenant-scoped
table in this schema (graph_nodes, graph_edges, graph_node_provenance,
directed_vectors, usage_events, query_traces, entity_* ...) already has
ENABLE + FORCE + a ``tenant_isolation`` policy with both USING and
WITH CHECK. This migration brings the three content tables in line.

Access audit (2026-07-10)
-------------------------
Every live service read/write on these tables runs inside
``with_tenant(customer_id)`` (normalizer persist + failed-chunk DLQ,
wiki routes, codegraph handler, inferred-edges bundle, synthesis
persistence / wiki agent / diagram + index renderers, every retrieval
retriever and the /sources + agent tool paths). The one exception --
``GET /api/manual-uploads``'s chunk_count subquery on a bare pool
connection -- is converted to ``with_tenant`` in the same change.

Cross-tenant operator scripts (backfill_embedding_v2, the global
SELECT in backfill_cc_metadata_chunks, inferred_edges_backfill_existing,
backfill_directed_phrases' customer discovery, smoke_phase2_clusters,
scripts/synth) follow the existing repo convention of running under the
BYPASSRLS ``prbe`` role -- the same convention those scripts already
rely on for the FORCE-RLS'd graph tables (see e.g.
scripts/backfill_directed_phrases.py::_all_customers_with_wiki).

FK-cascade deletes from ``DELETE FROM customers`` (deprovisioning,
shared/provisioning.py) bypass RLS by Postgres referential-integrity
design, so tenant teardown is unaffected.

Behavior note: chunks carries a non-tenant-scoped
``UNIQUE (doc_id, content_hash)``. If two tenants ever ingest a
byte-identical doc_id + content, the second tenant's
``ON CONFLICT DO UPDATE`` previously updated the FIRST tenant's row
silently (cross-tenant corruption); under FORCE RLS it now raises an
RLS violation instead -- a loud failure for what was already a bug.

Pattern mirrors migration 0070 (graph_node_provenance RLS): ENABLE +
FORCE + CREATE POLICY tenant_isolation, guarded by ``DO $$`` so a
partial apply + re-run is a no-op (CREATE POLICY has no IF NOT EXISTS
in PG 16). Unqualified table names resolve via the migrating role's
search_path (ag_catalog first on the managed cluster, public on
self-host) -- same as every prior migration in this chain.

Verification
------------

    SELECT relname,
           relrowsecurity AS rls_enabled,
           relforcerowsecurity AS rls_forced
    FROM   pg_class
    WHERE  relname IN ('documents', 'chunks', 'failed_chunks');

    SELECT polrelid::regclass::text AS tbl, polname,
           pg_get_expr(polqual, polrelid)      AS using_expr,
           pg_get_expr(polwithcheck, polrelid) AS with_check_expr
    FROM   pg_policy
    WHERE  polrelid IN ('documents'::regclass, 'chunks'::regclass,
                        'failed_chunks'::regclass);
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

# Keep <=32 chars (alembic_version.version_num is varchar(32)).
revision: str = "0094_documents_chunks_rls"
down_revision: str | Sequence[str] | None = "0093_drop_agent_run_tables"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


_POLICY_EXPR = "customer_id = current_setting('app.current_customer_id', true)"

_TABLES: tuple[str, ...] = ("documents", "chunks", "failed_chunks")


def upgrade() -> None:
    # These DDLs are catalog-only (no table scan/rewrite) so they are
    # near-instant, but each ALTER TABLE grabs a brief ACCESS EXCLUSIVE lock
    # on documents/chunks — the two hottest tables. Bound the wait so the
    # migration fails fast and retries (idempotent) instead of queueing
    # behind a long retrieval scan and stalling new queries on those tables.
    op.execute("SET lock_timeout = '5s'")
    for table in _TABLES:
        # ENABLE + FORCE RLS. Both are idempotent at the catalog level
        # (Postgres no-ops if already on). FORCE matters: the app roles
        # may own these tables, and owners bypass non-FORCEd RLS.
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")

        # Policy. DO $$ guard mirrors migrations 0067/0070 so a partial
        # apply + re-run is a no-op.
        op.execute(
            f"""
            DO $$
            BEGIN
                IF EXISTS (
                    SELECT 1
                    FROM   pg_policies
                    WHERE  schemaname = current_schema()
                      AND  tablename  = '{table}'
                      AND  policyname = 'tenant_isolation'
                      AND  with_check IS NOT NULL
                ) THEN
                    RETURN;
                END IF;

                EXECUTE 'DROP POLICY IF EXISTS tenant_isolation ON {table}';
                EXECUTE $ddl$
                    CREATE POLICY tenant_isolation ON {table}
                        USING ({_POLICY_EXPR})
                        WITH CHECK ({_POLICY_EXPR})
                $ddl$;
            END $$;
            """
        )


def downgrade() -> None:
    # Pre-0094 state had no RLS at all on these tables, so (unlike 0070,
    # where 0052 had already FORCEd the table) downgrade removes FORCE too.
    for table in _TABLES:
        op.execute(f"DROP POLICY IF EXISTS tenant_isolation ON {table}")
        op.execute(f"ALTER TABLE {table} NO FORCE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY")
