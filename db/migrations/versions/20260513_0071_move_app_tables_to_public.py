"""Move prbe-knowledge app tables out of ag_catalog into public.

Background — the AGE-extension search_path hijack:
  Apache AGE installs into ``ag_catalog`` and prepends that schema to the
  session ``search_path`` during ``CREATE EXTENSION age``. Every
  ``CREATE TABLE`` the migrate role ran with AGE loaded landed in
  ``ag_catalog`` instead of ``public`` — 30 application tables today,
  including ``customers``, ``documents``, ``chunks``, ``graph_nodes``,
  ``integration_tokens``, and the entire wiki + mcp_oauth surface.

Two latent problems with leaving them there:
  1. ``ALTER EXTENSION age UPDATE`` could touch or collide with our
     tables — ``ag_catalog`` is AGE's namespace, not ours.
  2. Anyone debugging the DB looks in ``public`` first; missing tables
     look like a missing migration.

Fix — ``ALTER TABLE ag_catalog.<t> SET SCHEMA public`` for each app
table. The catalog flip is metadata-only (~milliseconds, atomic);
indexes, sequences, PKs/FKs, RLS policies, and grants move with the
table. AGE's own ``ag_graph`` / ``ag_label`` catalog tables stay in
ag_catalog; the probe_graph label tables stay in their own schema.

Idempotent: each move is guarded on the table still existing in
ag_catalog (and not yet existing in public). Re-running the migration on
a DB where the move already happened is a no-op.

Companion changes outside this migration:
  * ``db/migrations/env.py`` sets ``search_path = public, ag_catalog``
    at session start so every future migration's unqualified
    ``CREATE TABLE`` lands in public regardless of AGE state.
  * ``db/schema.sql`` updates the ``verify_and_touch_custom_ingest_token``
    function definition to ``SET search_path = public, "$user", ag_catalog``
    so fresh-install paths land with the right per-function config.
  * The prbe-backend ``apps/data_plane/db/migrations`` chain has a
    sibling migration (``0010_probe_role_public_first``) that flips
    ``ALTER ROLE probe SET search_path`` to public-first (overriding
    the legacy 0005 setting) and moves ``webhook_secrets`` back to
    public; the ``apps/data_plane/db_migrations`` chain has a sibling
    ``0002`` migration that moves ``data_plane_secrets`` back to public.

Revision ID: 0071_move_app_tables_to_public
Revises: 0070_gnp_rls
Create Date: 2026-05-13
"""
from __future__ import annotations

from alembic import op

# Keep <=32 chars (alembic_version.version_num is varchar(32)).
revision = "0071_move_app_tables_to_public"
down_revision = "0070_gnp_rls"
branch_labels = None
depends_on = None


# Hardcoded list — derived from a live inventory of managed-postgres-0
# on 2026-05-13. Hardcoding rather than deriving from pg_class at runtime
# so the migration body is identical on SQLite (where the inventory query
# would return nothing useful) and on a fresh Postgres where ag_catalog
# is empty of our tables (the existence guard makes the migration a
# no-op there).
#
# NOT included on purpose:
#   * ``ag_graph`` / ``ag_label``    — AGE's own catalog tables
#   * ``alembic_version``            — this chain's version table; moving
#                                     it mid-migration would break the
#                                     same migration's commit. Stays in
#                                     ag_catalog; subsequent runs find
#                                     it via the public,ag_catalog
#                                     search_path fallback.
#   * ``data_plane_secrets`` /
#     ``dp_secrets_alembic_version`` — owned by prbe-backend's
#                                     db_migrations chain.
#   * ``webhook_secrets`` /
#     ``dp_auth_alembic_version``    — owned by prbe-backend's
#                                     apps/data_plane/db/migrations chain.
APP_TABLES: tuple[str, ...] = (
    "acl_snapshots",
    "audit_log",
    "backfill_state",
    "chunks",
    "code_repo_state",
    "custom_ingest_tokens",
    "customer_source_mapping",
    "customers",
    "directed_vectors",
    "documents",
    "failed_chunks",
    "graph_edges",
    "graph_node_provenance",
    "graph_nodes",
    "inferred_edges_queue",
    "ingestion_events",
    "ingestion_queue",
    "integration_tokens",
    "manual_uploads",
    "mcp_oauth_clients",
    "mcp_oauth_codes",
    "mcp_oauth_refresh_tokens",
    "mcp_oauth_sessions",
    "query_traces",
    "usage_events",
    "wiki_links",
    "wiki_raw_data",
    "wiki_synthesis_queue",
    "wiki_synthesis_runs",
    "wiki_timeline_entries",
)


def _move_one(tbl: str, src: str, dst: str) -> str:
    """Build an idempotent ALTER TABLE ... SET SCHEMA DO-block.

    Branch logic:
      * src has it, dst doesn't  → ALTER TABLE SET SCHEMA (the normal case).
      * src missing, dst has it  → no-op (idempotent re-run after success).
      * src missing, dst missing → no-op (SQLite path / fresh DB without the table).
      * src AND dst both have it → RAISE EXCEPTION. This is split-brain
        (a manually-recovered backup or an out-of-band restore that
        doubled up rows during a crash); silent skip would leave a
        ghost copy in the wrong schema. Fail loudly so an operator
        reconciles before the migration claims success.
    """
    return f"""
        DO $$
        DECLARE
            in_src BOOLEAN;
            in_dst BOOLEAN;
        BEGIN
            in_src := EXISTS (
                SELECT 1 FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE c.relname = '{tbl}' AND n.nspname = '{src}'
            );
            in_dst := EXISTS (
                SELECT 1 FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE c.relname = '{tbl}' AND n.nspname = '{dst}'
            );
            IF in_src AND in_dst THEN
                RAISE EXCEPTION
                    'split-brain: % exists in both {src} and {dst} -- manual reconciliation required',
                    '{tbl}';
            ELSIF in_src AND NOT in_dst THEN
                EXECUTE 'ALTER TABLE {src}.{tbl} SET SCHEMA {dst}';
            END IF;
        END $$;
    """


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        # SQLite tests don't have schemas; everything's in the main
        # database, and these tables (if they exist) were created via
        # the same alembic chain without any ag_catalog involvement.
        return

    # Pin this migration's search_path so any unqualified DDL we issue
    # below lands in public (defensive — the body below is all
    # schema-qualified, but env.py's session-start pin handles the rest
    # of the chain regardless).
    op.execute('SET LOCAL search_path TO public, ag_catalog')

    # Fail fast if any table is locked by a concurrent transaction. The
    # data-plane chart has NO migrate Job — this migration runs via
    # kubectl exec against a Postgres serving live traffic, so a
    # contended ALTER TABLE could otherwise hang indefinitely on
    # AccessExclusiveLock and stall the deploy. 5s is enough to outlast
    # any reasonable request handler; longer waits should bail and let
    # an operator pick a quieter window.
    op.execute("SET LOCAL lock_timeout = '5s'")

    for tbl in APP_TABLES:
        op.execute(_move_one(tbl, "ag_catalog", "public"))

    # Schema-qualified — the function was created in ag_catalog by 0046
    # (AGE hijack landed it there) and 0066 altered its per-call
    # search_path without moving it. Don't rely on search_path
    # resolution for this ALTER FUNCTION; a same-named function in
    # public would silently mis-target.
    op.execute(
        'ALTER FUNCTION ag_catalog.verify_and_touch_custom_ingest_token(text) '
        'SET search_path = public, "$user", ag_catalog'
    )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return

    op.execute('SET LOCAL search_path TO ag_catalog, public')
    op.execute("SET LOCAL lock_timeout = '5s'")

    # Restore 0066's setting first so the function points at ag_catalog
    # before we move tables back. Schema-qualified — see upgrade() note.
    op.execute(
        'ALTER FUNCTION ag_catalog.verify_and_touch_custom_ingest_token(text) '
        'SET search_path = ag_catalog, "$user", public'
    )

    for tbl in APP_TABLES:
        op.execute(_move_one(tbl, "public", "ag_catalog"))
