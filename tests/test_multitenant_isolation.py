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
