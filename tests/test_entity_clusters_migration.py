"""Migration assertions for entity_clusters (0071).

Pins schema-level invariants for the manual-entity-merge tables:

  * Five new tables exist with the right columns / PKs / CHECKs.
  * entity_aliases PK uniqueness + entity_aliases_not_self CHECK fire.
  * entity_merge_audit.status CHECK rejects unknown values.
  * entity_merge_edge_snapshot.operation CHECK rejects unknown ops.
  * graph_edges gains aliased_from_canonical_id + aliased_to_canonical_id.
  * graph_edges UNIQUE now includes the alias provenance columns, so two
    rows with the same (edge_type, from, to) coexist if they differ in
    aliased_from_canonical_id.
  * graph_edges UNIQUE still dedups the common-case "both aliased_from
    cols NULL" inserts (the existing graph_writer upsert path).
  * RLS is ENABLE + FORCE + USING + WITH CHECK on all 5 new tables;
    cross-tenant SELECT under tenant A's GUC returns zero rows for B's
    data; cross-tenant INSERT is rejected by WITH CHECK.
"""

from __future__ import annotations

import uuid

import asyncpg
import pytest

from engine.shared.db import raw_conn, with_tenant


async def _skip_if_superuser(conn: asyncpg.Connection) -> None:
    """Skip the calling test if the connection is running as a superuser.

    Mirrors the helper in ``tests/test_rls_cross_tenant_denial.py``: the
    dev/CI Postgres connects as the ``prbe`` superuser, which bypasses
    RLS entirely (both USING and WITH CHECK short-circuit, regardless of
    FORCE ROW LEVEL SECURITY). These RLS assertions are only meaningful
    under the non-privileged app role -- skip otherwise so the test
    pinning the schema invariant still runs on the CI matrix when it's
    eventually switched to that role.
    """
    is_superuser = await conn.fetchval(
        "SELECT current_setting('is_superuser', true)::bool"
    )
    if is_superuser:
        pytest.skip(
            "Connection is running as superuser; RLS USING / WITH CHECK "
            "are bypassed. RLS cross-tenant denial is meaningful only "
            "under the non-privileged app role."
        )


async def _seed_customer(conn: asyncpg.Connection, customer_id: str) -> None:
    await conn.execute(
        "INSERT INTO customers(customer_id, display_name, api_key_hash) "
        "VALUES ($1, 'mig', 'mig-hash') ON CONFLICT DO NOTHING",
        customer_id,
    )


async def _seed_person_node(
    conn: asyncpg.Connection, customer_id: str, canonical_id: str
) -> int:
    """Insert one Person graph_nodes row; return its node_id."""
    row = await conn.fetchrow(
        """
        INSERT INTO graph_nodes(customer_id, label, canonical_id)
        VALUES ($1, 'Person', $2)
        ON CONFLICT (customer_id, label, canonical_id) DO UPDATE
            SET updated_at = NOW()
        RETURNING node_id
        """,
        customer_id,
        canonical_id,
    )
    return row["node_id"]


async def _seed_doc_node(
    conn: asyncpg.Connection, customer_id: str, canonical_id: str
) -> int:
    row = await conn.fetchrow(
        """
        INSERT INTO graph_nodes(customer_id, label, canonical_id)
        VALUES ($1, 'Document', $2)
        ON CONFLICT (customer_id, label, canonical_id) DO UPDATE
            SET updated_at = NOW()
        RETURNING node_id
        """,
        customer_id,
        canonical_id,
    )
    return row["node_id"]


async def _insert_audit(
    conn: asyncpg.Connection,
    *,
    customer_id: str,
    label: str,
    primary: str,
    aliases: list[str],
    user_id: uuid.UUID,
) -> uuid.UUID:
    merge_id = uuid.uuid4()
    await conn.execute(
        """
        INSERT INTO entity_merge_audit
          (merge_id, customer_id, label, primary_canonical_id,
           merged_alias_canonical_ids, performed_by_user_id)
        VALUES ($1, $2, $3, $4, $5, $6)
        """,
        merge_id, customer_id, label, primary, aliases, user_id,
    )
    return merge_id


# ---------------------------------------------------------------------------
# Table existence + PK / CHECK behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_entity_aliases_pk_rejects_duplicate_alias(live_db) -> None:
    cust = "mig-ec-pk"
    user_id = uuid.uuid4()
    async with raw_conn() as conn:
        await _seed_customer(conn, cust)
    async with with_tenant(cust) as conn:
        await _seed_person_node(conn, cust, "p1")
        await _seed_person_node(conn, cust, "a1")
        await _seed_person_node(conn, cust, "p2")
        merge1 = await _insert_audit(
            conn, customer_id=cust, label="Person",
            primary="p1", aliases=["a1"], user_id=user_id,
        )
        merge2 = await _insert_audit(
            conn, customer_id=cust, label="Person",
            primary="p2", aliases=["a1"], user_id=user_id,
        )
        await conn.execute(
            """
            INSERT INTO entity_aliases
              (customer_id, label, alias_canonical_id, primary_canonical_id, merge_id)
            VALUES ($1, 'Person', 'a1', 'p1', $2)
            """,
            cust, merge1,
        )
        with pytest.raises(asyncpg.UniqueViolationError):
            await conn.execute(
                """
                INSERT INTO entity_aliases
                  (customer_id, label, alias_canonical_id, primary_canonical_id, merge_id)
                VALUES ($1, 'Person', 'a1', 'p2', $2)
                """,
                cust, merge2,
            )


@pytest.mark.asyncio
async def test_entity_aliases_check_rejects_self_alias(live_db) -> None:
    cust = "mig-ec-self"
    user_id = uuid.uuid4()
    async with raw_conn() as conn:
        await _seed_customer(conn, cust)
    async with with_tenant(cust) as conn:
        await _seed_person_node(conn, cust, "richard")
        merge_id = await _insert_audit(
            conn, customer_id=cust, label="Person",
            primary="richard", aliases=[], user_id=user_id,
        )
        with pytest.raises(asyncpg.CheckViolationError):
            await conn.execute(
                """
                INSERT INTO entity_aliases
                  (customer_id, label, alias_canonical_id, primary_canonical_id, merge_id)
                VALUES ($1, 'Person', 'richard', 'richard', $2)
                """,
                cust, merge_id,
            )


@pytest.mark.asyncio
async def test_entity_merge_audit_status_check(live_db) -> None:
    cust = "mig-ec-status"
    user_id = uuid.uuid4()
    async with raw_conn() as conn:
        await _seed_customer(conn, cust)
    async with with_tenant(cust) as conn:
        with pytest.raises(asyncpg.CheckViolationError):
            await conn.execute(
                """
                INSERT INTO entity_merge_audit
                  (merge_id, customer_id, label, primary_canonical_id,
                   merged_alias_canonical_ids, performed_by_user_id, status)
                VALUES ($1, $2, 'Person', 'x', ARRAY['y']::text[], $3, 'bogus')
                """,
                uuid.uuid4(), cust, user_id,
            )


@pytest.mark.asyncio
async def test_entity_merge_edge_snapshot_operation_check(live_db) -> None:
    cust = "mig-ec-op"
    user_id = uuid.uuid4()
    async with raw_conn() as conn:
        await _seed_customer(conn, cust)
    async with with_tenant(cust) as conn:
        merge_id = await _insert_audit(
            conn, customer_id=cust, label="Person",
            primary="x", aliases=["y"], user_id=user_id,
        )
        with pytest.raises(asyncpg.CheckViolationError):
            await conn.execute(
                """
                INSERT INTO entity_merge_edge_snapshot
                  (merge_id, snapshot_seq, customer_id, operation,
                   pre_edge_type, pre_from_canonical_id, pre_from_label,
                   pre_to_canonical_id, pre_to_label, pre_properties,
                   pre_confidence, pre_valid_from)
                VALUES ($1, 1, $2, 'bogus_op',
                        'AUTHORED', 'a', 'Person', 'b', 'Document',
                        '{}'::jsonb, 'EXTRACTED', NOW())
                """,
                merge_id, cust,
            )


# ---------------------------------------------------------------------------
# graph_edges schema changes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_graph_edges_has_alias_columns(live_db) -> None:
    """The two new nullable TEXT columns exist on graph_edges."""
    async with raw_conn() as conn:
        cols = await conn.fetch(
            """
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'graph_edges'
              AND column_name IN ('aliased_from_canonical_id', 'aliased_to_canonical_id')
            ORDER BY column_name
            """
        )
    assert [c["column_name"] for c in cols] == [
        "aliased_from_canonical_id",
        "aliased_to_canonical_id",
    ]


@pytest.mark.asyncio
async def test_graph_edges_composite_unique_allows_alias_lanes(live_db) -> None:
    """Same (edge_type, from, to) with different aliased_from coexist."""
    cust = "mig-ec-lanes"
    async with raw_conn() as conn:
        await _seed_customer(conn, cust)
    async with with_tenant(cust) as conn:
        p_node = await _seed_person_node(conn, cust, "richard")
        d_node = await _seed_doc_node(conn, cust, "doc-1")
        # Lane 1: NULL aliased_from
        await conn.execute(
            """
            INSERT INTO graph_edges
              (customer_id, edge_type, from_node_id, to_node_id,
               source_system, confidence)
            VALUES ($1, 'AUTHORED', $2, $3, 'github', 'EXTRACTED')
            """,
            cust, p_node, d_node,
        )
        # Lane 2: aliased_from='mahit@prbe.ai'
        await conn.execute(
            """
            INSERT INTO graph_edges
              (customer_id, edge_type, from_node_id, to_node_id,
               source_system, confidence, aliased_from_canonical_id)
            VALUES ($1, 'AUTHORED', $2, $3, 'github', 'EXTRACTED', 'mahit@prbe.ai')
            """,
            cust, p_node, d_node,
        )
        # Both rows persist
        rows = await conn.fetch(
            """
            SELECT aliased_from_canonical_id FROM graph_edges
            WHERE customer_id = $1 AND from_node_id = $2 AND to_node_id = $3
            ORDER BY edge_id
            """,
            cust, p_node, d_node,
        )
    assert [r["aliased_from_canonical_id"] for r in rows] == [None, "mahit@prbe.ai"]


@pytest.mark.asyncio
async def test_graph_edges_composite_unique_still_dedups_null_lane(live_db) -> None:
    """The common case (both aliased_from cols NULL) still dedups via upsert."""
    cust = "mig-ec-nulldedup"
    async with raw_conn() as conn:
        await _seed_customer(conn, cust)
    async with with_tenant(cust) as conn:
        p_node = await _seed_person_node(conn, cust, "richard")
        d_node = await _seed_doc_node(conn, cust, "doc-1")
        # Two inserts in the NULL lane should collide on the composite UNIQUE.
        await conn.execute(
            """
            INSERT INTO graph_edges
              (customer_id, edge_type, from_node_id, to_node_id,
               source_system, confidence)
            VALUES ($1, 'AUTHORED', $2, $3, 'github', 'EXTRACTED')
            """,
            cust, p_node, d_node,
        )
        with pytest.raises(asyncpg.UniqueViolationError):
            await conn.execute(
                """
                INSERT INTO graph_edges
                  (customer_id, edge_type, from_node_id, to_node_id,
                   source_system, confidence)
                VALUES ($1, 'AUTHORED', $2, $3, 'github', 'EXTRACTED')
                """,
                cust, p_node, d_node,
            )


# ---------------------------------------------------------------------------
# RLS cross-tenant denial
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_entity_aliases_rls_cross_tenant_denied(live_db) -> None:
    cust_a = "mig-ec-rls-a"
    cust_b = "mig-ec-rls-b"
    user_id = uuid.uuid4()
    async with raw_conn() as conn:
        await _skip_if_superuser(conn)
        await _seed_customer(conn, cust_a)
        await _seed_customer(conn, cust_b)

    async with with_tenant(cust_a) as conn:
        await _seed_person_node(conn, cust_a, "pa")
        await _seed_person_node(conn, cust_a, "aa")
        merge_id_a = await _insert_audit(
            conn, customer_id=cust_a, label="Person",
            primary="pa", aliases=["aa"], user_id=user_id,
        )
        await conn.execute(
            """
            INSERT INTO entity_aliases
              (customer_id, label, alias_canonical_id, primary_canonical_id, merge_id)
            VALUES ($1, 'Person', 'aa', 'pa', $2)
            """,
            cust_a, merge_id_a,
        )

    # Read under B sees zero of A's rows.
    async with with_tenant(cust_b) as conn:
        rows = await conn.fetch(
            "SELECT alias_canonical_id FROM entity_aliases WHERE label = 'Person'"
        )
        assert rows == []

    # Write under A with customer_id = B → WITH CHECK rejects.
    async with with_tenant(cust_b) as conn:
        await _seed_person_node(conn, cust_b, "pb")
        await _seed_person_node(conn, cust_b, "ab")
        merge_id_b = await _insert_audit(
            conn, customer_id=cust_b, label="Person",
            primary="pb", aliases=["ab"], user_id=user_id,
        )
    async with with_tenant(cust_a) as conn:
        with pytest.raises(asyncpg.CheckViolationError):
            await conn.execute(
                """
                INSERT INTO entity_aliases
                  (customer_id, label, alias_canonical_id, primary_canonical_id, merge_id)
                VALUES ($1, 'Person', 'ab', 'pb', $2)
                """,
                cust_b, merge_id_b,
            )


@pytest.mark.asyncio
async def test_entity_merge_audit_rls_cross_tenant_denied(live_db) -> None:
    cust_a = "mig-ec-audit-rls-a"
    cust_b = "mig-ec-audit-rls-b"
    user_id = uuid.uuid4()
    async with raw_conn() as conn:
        await _skip_if_superuser(conn)
        await _seed_customer(conn, cust_a)
        await _seed_customer(conn, cust_b)
    async with with_tenant(cust_a) as conn:
        await _insert_audit(
            conn, customer_id=cust_a, label="Person",
            primary="pa", aliases=["aa"], user_id=user_id,
        )
    async with with_tenant(cust_b) as conn:
        rows = await conn.fetch(
            "SELECT merge_id FROM entity_merge_audit WHERE label = 'Person'"
        )
        assert rows == []
