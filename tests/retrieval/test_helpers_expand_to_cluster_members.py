"""Unit tests for `expand_to_cluster_members` (Phase 2 retrieval helper).

Real Postgres (no DB mocks for retrieval per project convention). The
``live_db`` fixture truncates between tests. Helper runs under
``with_tenant(customer_id)`` because it queries an RLS-protected table.
"""

from __future__ import annotations

import uuid

import pytest

from services.retrieval.helpers import expand_to_cluster_members
from shared.db import raw_conn, with_tenant

pytestmark = pytest.mark.asyncio


CUSTOMER_ID = "expand-cluster-cust"
PRIMARY = "richardwei6"
ALIAS_A = "mahit@prbe.ai"
ALIAS_B = "U07ABC123"


async def _seed_customer(customer_id: str) -> None:
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO customers (customer_id, display_name, api_key_hash)
            VALUES ($1, 'test', 'h-' || $1)
            ON CONFLICT (customer_id) DO NOTHING
            """,
            customer_id,
        )


async def _seed_cluster(customer_id: str) -> None:
    """Seed Person cluster: PRIMARY with two aliases ALIAS_A and ALIAS_B.

    entity_merge_audit.merge_id is UUID PRIMARY KEY with no default; we
    generate it client-side per the convention in
    tests/test_graph_writer_alias_resolution.py.
    """
    merge_id = str(uuid.uuid4())
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO entity_merge_audit (
                merge_id, customer_id, label, primary_canonical_id,
                merged_alias_canonical_ids, performed_by_user_id, status
            ) VALUES ($1, $2, 'Person', $3, ARRAY[$4, $5]::text[],
                      '11111111-1111-1111-1111-111111111111', 'active')
            """,
            merge_id, customer_id, PRIMARY, ALIAS_A, ALIAS_B,
        )
        await conn.executemany(
            """
            INSERT INTO entity_aliases (
                customer_id, label, alias_canonical_id,
                primary_canonical_id, merge_id
            ) VALUES ($1, 'Person', $2, $3, $4)
            """,
            [
                (customer_id, ALIAS_A, PRIMARY, merge_id),
                (customer_id, ALIAS_B, PRIMARY, merge_id),
            ],
        )


async def test_expand_from_primary_returns_full_cluster(live_db):
    await _seed_customer(CUSTOMER_ID)
    await _seed_cluster(CUSTOMER_ID)
    async with with_tenant(CUSTOMER_ID) as conn:
        out = await expand_to_cluster_members(
            conn, CUSTOMER_ID, "Person", canonical_ids=[PRIMARY],
        )
    assert sorted(out[PRIMARY]) == sorted([PRIMARY, ALIAS_A, ALIAS_B])


async def test_expand_from_alias_returns_full_cluster(live_db):
    """Querying with an alias id returns the same full cluster."""
    await _seed_customer(CUSTOMER_ID)
    await _seed_cluster(CUSTOMER_ID)
    async with with_tenant(CUSTOMER_ID) as conn:
        out = await expand_to_cluster_members(
            conn, CUSTOMER_ID, "Person", canonical_ids=[ALIAS_A],
        )
    assert sorted(out[ALIAS_A]) == sorted([PRIMARY, ALIAS_A, ALIAS_B])


async def test_expand_unmerged_id_returns_self(live_db):
    """An id that is neither a primary nor an alias maps to a singleton."""
    await _seed_customer(CUSTOMER_ID)
    async with with_tenant(CUSTOMER_ID) as conn:
        out = await expand_to_cluster_members(
            conn, CUSTOMER_ID, "Person", canonical_ids=["loner-id"],
        )
    assert out == {"loner-id": ["loner-id"]}


async def test_expand_mixed_input_collapses_correctly(live_db):
    """Mixed input: one alias, one primary, one unmerged."""
    await _seed_customer(CUSTOMER_ID)
    await _seed_cluster(CUSTOMER_ID)
    async with with_tenant(CUSTOMER_ID) as conn:
        out = await expand_to_cluster_members(
            conn, CUSTOMER_ID, "Person",
            canonical_ids=[ALIAS_A, PRIMARY, "loner-id"],
        )
    expected_cluster = sorted([PRIMARY, ALIAS_A, ALIAS_B])
    assert sorted(out[ALIAS_A]) == expected_cluster
    assert sorted(out[PRIMARY]) == expected_cluster
    assert out["loner-id"] == ["loner-id"]


async def test_expand_empty_input_returns_empty_dict(live_db):
    async with with_tenant(CUSTOMER_ID) as conn:
        out = await expand_to_cluster_members(
            conn, CUSTOMER_ID, "Person", canonical_ids=[],
        )
    assert out == {}


async def test_expand_is_label_scoped(live_db):
    """Same canonical_id under two different labels must not collide.

    Seed Person:alice aliased to Person:bob, then query with label='Repo'
    and canonical_id='alice' — the Repo lookup must NOT see the Person
    cluster, since membership is label-scoped.
    """
    customer_id = "expand-cluster-label-cust"
    await _seed_customer(customer_id)
    merge_id = str(uuid.uuid4())
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO entity_merge_audit (
                merge_id, customer_id, label, primary_canonical_id,
                merged_alias_canonical_ids, performed_by_user_id, status
            ) VALUES ($1, $2, 'Person', 'bob', ARRAY['alice']::text[],
                      '11111111-1111-1111-1111-111111111111', 'active')
            """,
            merge_id, customer_id,
        )
        await conn.execute(
            """
            INSERT INTO entity_aliases (
                customer_id, label, alias_canonical_id,
                primary_canonical_id, merge_id
            ) VALUES ($1, 'Person', 'alice', 'bob', $2)
            """,
            customer_id, merge_id,
        )
    async with with_tenant(customer_id) as conn:
        # Query with label='Repo' — must NOT see the Person cluster.
        repo_out = await expand_to_cluster_members(
            conn, customer_id, "Repo", canonical_ids=["alice"],
        )
        # Sanity: Person lookup DOES see the cluster.
        person_out = await expand_to_cluster_members(
            conn, customer_id, "Person", canonical_ids=["alice"],
        )
    assert repo_out == {"alice": ["alice"]}
    assert sorted(person_out["alice"]) == sorted(["alice", "bob"])


async def test_expand_is_tenant_scoped(live_db):
    """A cluster in tenant A must NOT be visible from tenant B.

    Mirrors `test_resolve_aliases_is_tenant_scoped` in the sibling
    helper. RLS is a load-bearing security boundary.
    """
    customer_a = "expand-cluster-tenant-a"
    customer_b = "expand-cluster-tenant-b"
    await _seed_customer(customer_a)
    await _seed_customer(customer_b)
    await _seed_cluster(customer_a)
    # Query from B — must see neither the alias nor the primary.
    async with with_tenant(customer_b) as conn:
        out_alias = await expand_to_cluster_members(
            conn, customer_b, "Person", canonical_ids=[ALIAS_A],
        )
        out_primary = await expand_to_cluster_members(
            conn, customer_b, "Person", canonical_ids=[PRIMARY],
        )
    # Both must collapse to self (B sees no cluster relationship).
    assert out_alias == {ALIAS_A: [ALIAS_A]}
    assert out_primary == {PRIMARY: [PRIMARY]}
