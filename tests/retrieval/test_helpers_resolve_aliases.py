"""Unit tests for `resolve_aliases` (Phase 2 retrieval-side alias helper).

Real Postgres (no DB mocks for retrieval per project convention). The
``live_db`` fixture truncates between tests. Helper runs under
``with_tenant(customer_id)`` because it queries an RLS-protected table.
"""

from __future__ import annotations

import uuid

import pytest

from engine.retrieval.helpers import resolve_aliases
from engine.shared.db import raw_conn, with_tenant

pytestmark = pytest.mark.asyncio


CUSTOMER_ID = "resolve-aliases-cust"
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


async def _seed_audit(customer_id: str) -> str:
    merge_id = uuid.uuid4()
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
    return str(merge_id)


async def _seed_aliases(customer_id: str, merge_id: str) -> None:
    async with raw_conn() as conn:
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


async def test_resolve_aliases_returns_primary_for_known_aliases(live_db):
    await _seed_customer(CUSTOMER_ID)
    merge_id = await _seed_audit(CUSTOMER_ID)
    await _seed_aliases(CUSTOMER_ID, merge_id)
    async with with_tenant(CUSTOMER_ID) as conn:
        out = await resolve_aliases(
            conn, CUSTOMER_ID,
            refs=[("Person", ALIAS_A), ("Person", ALIAS_B)],
        )
    assert out == {("Person", ALIAS_A): PRIMARY, ("Person", ALIAS_B): PRIMARY}


async def test_resolve_aliases_omits_non_aliases(live_db):
    await _seed_customer(CUSTOMER_ID)
    merge_id = await _seed_audit(CUSTOMER_ID)
    await _seed_aliases(CUSTOMER_ID, merge_id)
    async with with_tenant(CUSTOMER_ID) as conn:
        out = await resolve_aliases(
            conn, CUSTOMER_ID,
            refs=[("Person", ALIAS_A), ("Person", "nobody"), ("Repo", "r1")],
        )
    assert out == {("Person", ALIAS_A): PRIMARY}


async def test_resolve_aliases_empty_input_returns_empty_dict(live_db):
    async with with_tenant(CUSTOMER_ID) as conn:
        out = await resolve_aliases(conn, CUSTOMER_ID, refs=[])
    assert out == {}


async def test_resolve_aliases_is_tenant_scoped(live_db):
    """An alias in tenant A must NOT resolve when queried from tenant B."""
    await _seed_customer(CUSTOMER_ID)
    merge_id = await _seed_audit(CUSTOMER_ID)
    await _seed_aliases(CUSTOMER_ID, merge_id)
    await _seed_customer("resolve-aliases-other-tenant")
    async with with_tenant("resolve-aliases-other-tenant") as conn:
        out = await resolve_aliases(
            conn, "resolve-aliases-other-tenant",
            refs=[("Person", ALIAS_A)],
        )
    assert out == {}
