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

    Only fires when the table still exists in ``src`` AND does not yet
    exist in ``dst``. Both halves of the guard matter — a partial replay
    after a crash could leave the table already in ``dst``; the second
    half prevents a same-name collision from raising.
    """
    return f"""
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE c.relname = '{tbl}' AND n.nspname = '{src}'
            ) AND NOT EXISTS (
                SELECT 1 FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE c.relname = '{tbl}' AND n.nspname = '{dst}'
            ) THEN
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

    for tbl in APP_TABLES:
        op.execute(_move_one(tbl, "ag_catalog", "public"))

    # Flip the SECURITY DEFINER function's per-call search_path to
    # public-first now that the table it touches lives there. The
    # 0066 setting (``ag_catalog, "$user", public``) would keep working
    # because public is still in the list, but the semantic order is
    # wrong post-move — public is where the table is, ag_catalog is
    # only there as a fallback for AGE's own catalog access.
    op.execute(
        'ALTER FUNCTION verify_and_touch_custom_ingest_token(text) '
        'SET search_path = public, "$user", ag_catalog'
    )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return

    op.execute('SET LOCAL search_path TO ag_catalog, public')

    # Restore 0066's setting first so the function points at ag_catalog
    # before we move tables back.
    op.execute(
        'ALTER FUNCTION verify_and_touch_custom_ingest_token(text) '
        'SET search_path = ag_catalog, "$user", public'
    )

    for tbl in APP_TABLES:
        op.execute(_move_one(tbl, "public", "ag_catalog"))
