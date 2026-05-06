"""Migration assertions for custom_ingest_tokens (0046).

Specifically: the SECURITY DEFINER `verify_and_touch_custom_ingest_token`
function must work from a connection that has NOT set
`app.current_customer_id`. That's the production verifier path -- the
caller doesn't know which tenant the bearer belongs to until *after* the
lookup.

If the migration ever re-introduces FORCE ROW LEVEL SECURITY on this
table, this test fails: the SECURITY DEFINER function (running as the
table owner) would also be subject to the policy, and with no GUC set
the policy predicate evaluates against an empty current_setting() and
filters every row out. See feedback_graph_nodes_rls_force.md.
"""

from __future__ import annotations

import hashlib
import uuid

import pytest

from shared.db import raw_conn


@pytest.mark.asyncio
async def test_verify_and_touch_returns_row_without_tenant_guc(live_db) -> None:
    customer_id = "mig-test-cit-cust"
    raw_token = f"prbe_test_{uuid.uuid4().hex}"
    token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
    async with raw_conn() as conn:
        # Seed customer + token row directly (owner role bypasses non-FORCE'd RLS).
        await conn.execute(
            "INSERT INTO customers(customer_id, display_name, api_key_hash) "
            "VALUES ($1, 'mig', 'mig-test-hash') ON CONFLICT DO NOTHING",
            customer_id,
        )
        await conn.execute("DELETE FROM custom_ingest_tokens WHERE customer_id = $1", customer_id)
        await conn.execute(
            """
            INSERT INTO custom_ingest_tokens (customer_id, name, token_hash, token_prefix)
            VALUES ($1, 'mig-test', $2, $3)
            """,
            customer_id,
            token_hash,
            raw_token[:8],
        )
        try:
            # Call the verifier from a connection with NO app.current_customer_id set.
            # Under FORCE'd RLS this would return zero rows; under ENABLE-only it
            # returns the seeded row because the function owner is RLS-exempt.
            rows = await conn.fetch(
                "SELECT token_id, customer_id FROM verify_and_touch_custom_ingest_token($1)",
                token_hash,
            )
            assert len(rows) == 1, "verifier must return exactly one row for a valid token"
            assert rows[0]["customer_id"] == customer_id
        finally:
            await conn.execute(
                "DELETE FROM custom_ingest_tokens WHERE customer_id = $1", customer_id
            )
            await conn.execute("DELETE FROM customers WHERE customer_id = $1", customer_id)


@pytest.mark.asyncio
async def test_custom_ingest_tokens_rls_enabled_but_not_forced(live_db) -> None:
    """RLS is enabled but NOT forced -- forcing would break the verifier path."""
    async with raw_conn() as conn:
        row = await conn.fetchrow(
            """
            SELECT relrowsecurity, relforcerowsecurity
            FROM pg_class
            WHERE relname = 'custom_ingest_tokens'
            """
        )
    assert row is not None
    assert row["relrowsecurity"] is True, "RLS must be enabled (defense in depth)"
    assert row["relforcerowsecurity"] is False, (
        "FORCE must NOT be set: SECURITY DEFINER verifier relies on owner exemption"
    )
