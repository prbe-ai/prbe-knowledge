"""RLS + explicit-filter isolation: customer A must never see customer B data.

The graph tables enforce tenant isolation via RLS. Documents/chunks rely on
explicit `customer_id = $1` filters in every query. This test asserts both
layers: bind tenant A, insert a graph row, bind tenant B, confirm zero rows.
"""

from __future__ import annotations

from datetime import UTC

import pytest

from shared.db import raw_conn, with_tenant


@pytest.mark.asyncio
async def test_rls_graph_isolation(live_db) -> None:
    """RLS enforcement against the graph tables.

    Postgres exempts superusers and users with BYPASSRLS from row-level policies
    even when the table is FORCE-enabled. The local docker-compose Postgres
    creates the `prbe` role as a superuser, so we create a non-super test role
    here, SET ROLE into it, and then verify isolation. In Neon production the
    app connects as a role without BYPASSRLS by default, so this matches prod.
    """
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers (customer_id, display_name, api_key_hash) VALUES ($1,'A','x'), ($2,'B','x') ON CONFLICT DO NOTHING",
            "cust-A",
            "cust-B",
        )
        await conn.execute(
            """
            DO $$
            BEGIN
                IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'prbe_rls_test') THEN
                    CREATE ROLE prbe_rls_test NOSUPERUSER NOBYPASSRLS;
                END IF;
            END $$;
            """
        )
        await conn.execute("GRANT USAGE ON SCHEMA public TO prbe_rls_test")
        await conn.execute("GRANT ALL ON ALL TABLES IN SCHEMA public TO prbe_rls_test")
        await conn.execute("GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO prbe_rls_test")

    async with with_tenant("cust-A") as conn:
        await conn.execute("SET LOCAL ROLE prbe_rls_test")
        await conn.execute(
            """
            INSERT INTO graph_nodes (customer_id, label, canonical_id)
            VALUES ('cust-A', 'Service', 'payments')
            """
        )
        rows_a = await conn.fetch("SELECT * FROM graph_nodes")
    assert len(rows_a) == 1

    async with with_tenant("cust-B") as conn:
        await conn.execute("SET LOCAL ROLE prbe_rls_test")
        rows_b = await conn.fetch("SELECT * FROM graph_nodes")
    assert len(rows_b) == 0  # RLS hides cust-A's node


@pytest.mark.asyncio
async def test_documents_filter_isolation(live_db) -> None:
    from datetime import datetime

    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers (customer_id, display_name, api_key_hash) VALUES ($1,'A','x'), ($2,'B','x') ON CONFLICT DO NOTHING",
            "cust-A",
            "cust-B",
        )
        now = datetime.now(UTC)
        for cid in ("cust-A", "cust-B"):
            await conn.execute(
                """
                INSERT INTO documents
                    (doc_id, version, customer_id, source_system, source_id, source_url,
                     doc_type, content_hash, created_at, updated_at, valid_from, ingested_at,
                     acl)
                VALUES ($1, 1, $2, 'slack', $1, 'u', 'slack.message', 'h', $3, $3, $3, $3, '{}'::jsonb)
                """,
                f"doc-{cid}",
                cid,
                now,
            )

    async with raw_conn() as conn:
        # Without explicit filter — returns both (vulnerable shape)
        rows = await conn.fetch("SELECT doc_id FROM documents")
        assert len(rows) == 2

        # Application-code layer always adds customer_id = $1
        rows = await conn.fetch("SELECT doc_id FROM documents WHERE customer_id=$1", "cust-A")
        assert len(rows) == 1
        assert rows[0]["doc_id"] == "doc-cust-A"


# WITH CHECK enforcement matrix. Each tuple is
# (table_name, insert_sql_template, insert_args_factory) where the SQL has a
# single ``{cid}`` placeholder for the row's customer_id (the value the test
# attempts to write, which is intentionally != the GUC's value). The factory
# returns extra positional args for asyncpg's ``execute`` (none of the
# minimal-required-columns rows below need any).
#
# Phase 4 (PR #200 audit) finding: prior to migration 0067 these tables had
# USING-only tenant policies, meaning a cross-tenant INSERT under tenant A's
# GUC writing a row for tenant B would silently succeed and become invisible
# to both tenants. WITH CHECK rejects the INSERT at the DB layer.
_WITH_CHECK_INSERTS: tuple[tuple[str, str], ...] = (
    (
        "graph_nodes",
        "INSERT INTO graph_nodes (customer_id, label, canonical_id) "
        "VALUES ('{cid}', 'Service', 'payments-x')",
    ),
    (
        "usage_events",
        "INSERT INTO usage_events (customer_id, caller_kind, event_type, endpoint, status) "
        "VALUES ('{cid}', 'test', 'test', '/test', 'ok')",
    ),
    (
        "query_traces",
        "INSERT INTO query_traces "
        "(request_id, customer_id, event_type, request, response, response_size_bytes) "
        "VALUES (gen_random_uuid(), '{cid}', 'retrieve', '{{}}'::jsonb, '{{}}'::jsonb, 0)",
    ),
    (
        "code_repo_state",
        "INSERT INTO code_repo_state "
        "(customer_id, repo, file_path, content_hash, language, last_extractor_version) "
        "VALUES ('{cid}', 'r', 'f.py', 'h', 'python', 'v0')",
    ),
)


@pytest.mark.parametrize("table,insert_template", _WITH_CHECK_INSERTS)
@pytest.mark.asyncio
async def test_with_check_blocks_cross_tenant_insert(
    live_db, table: str, insert_template: str
) -> None:
    """WITH CHECK on tenant-scoped policies blocks writes for the wrong tenant.

    Mirrors test_rls_graph_isolation's role/GUC setup. Bind tenant A, attempt
    to INSERT a row stamped with tenant B's customer_id. WITH CHECK should
    raise InsufficientPrivilegeError (Postgres' RLS-violation code).
    """
    import asyncpg

    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers (customer_id, display_name, api_key_hash) "
            "VALUES ($1,'A','x'), ($2,'B','x') ON CONFLICT DO NOTHING",
            "wc-cust-A",
            "wc-cust-B",
        )
        await conn.execute(
            """
            DO $$
            BEGIN
                IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'prbe_rls_test') THEN
                    CREATE ROLE prbe_rls_test NOSUPERUSER NOBYPASSRLS;
                END IF;
            END $$;
            """
        )
        await conn.execute("GRANT USAGE ON SCHEMA public TO prbe_rls_test")
        await conn.execute("GRANT ALL ON ALL TABLES IN SCHEMA public TO prbe_rls_test")
        await conn.execute("GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO prbe_rls_test")

    # Bind to tenant A. Attempt to write a row with customer_id = 'wc-cust-B'.
    async with with_tenant("wc-cust-A") as conn:
        await conn.execute("SET LOCAL ROLE prbe_rls_test")
        with pytest.raises(asyncpg.InsufficientPrivilegeError):
            await conn.execute(insert_template.format(cid="wc-cust-B"))

    # Sanity: same insert with the CORRECT customer_id succeeds.
    async with with_tenant("wc-cust-A") as conn:
        await conn.execute("SET LOCAL ROLE prbe_rls_test")
        await conn.execute(insert_template.format(cid="wc-cust-A"))
        rows = await conn.fetch(f"SELECT customer_id FROM {table}")
        assert all(r["customer_id"] == "wc-cust-A" for r in rows)


@pytest.mark.asyncio
async def test_with_check_blocks_graph_edges_cross_tenant_insert(live_db) -> None:
    """graph_edges WITH CHECK enforcement.

    Separated from the parametrized test because graph_edges requires two
    pre-existing graph_nodes (FK constraints) — extra setup that doesn't
    fit the table-agnostic INSERT template.
    """
    import asyncpg

    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers (customer_id, display_name, api_key_hash) "
            "VALUES ($1,'A','x'), ($2,'B','x') ON CONFLICT DO NOTHING",
            "wc-edge-A",
            "wc-edge-B",
        )
        await conn.execute(
            """
            DO $$
            BEGIN
                IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'prbe_rls_test') THEN
                    CREATE ROLE prbe_rls_test NOSUPERUSER NOBYPASSRLS;
                END IF;
            END $$;
            """
        )
        await conn.execute("GRANT USAGE ON SCHEMA public TO prbe_rls_test")
        await conn.execute("GRANT ALL ON ALL TABLES IN SCHEMA public TO prbe_rls_test")
        await conn.execute("GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO prbe_rls_test")

    # Insert two nodes for tenant A (under A's GUC, as superuser owner role
    # for the FK target — we revert to test role only for the negative case).
    async with with_tenant("wc-edge-A") as conn:
        row_from = await conn.fetchrow(
            "INSERT INTO graph_nodes (customer_id, label, canonical_id) "
            "VALUES ('wc-edge-A','Service','svc-from-x') RETURNING node_id"
        )
        row_to = await conn.fetchrow(
            "INSERT INTO graph_nodes (customer_id, label, canonical_id) "
            "VALUES ('wc-edge-A','Service','svc-to-x') RETURNING node_id"
        )
    from_id = row_from["node_id"]
    to_id = row_to["node_id"]

    # Under tenant A's GUC, attempt to write an edge stamped with tenant B's
    # customer_id. WITH CHECK rejects.
    async with with_tenant("wc-edge-A") as conn:
        await conn.execute("SET LOCAL ROLE prbe_rls_test")
        with pytest.raises(asyncpg.InsufficientPrivilegeError):
            await conn.execute(
                "INSERT INTO graph_edges "
                "(customer_id, edge_type, from_node_id, to_node_id) "
                "VALUES ('wc-edge-B', 'calls', $1, $2)",
                from_id,
                to_id,
            )
