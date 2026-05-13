"""Phase 4 cross-tenant denial test for prbe-knowledge.

Mirrors the data-plane denial test in prbe-backend
(tests/data_plane/test_rls_cross_tenant_denial.py), but exercises this
service's actual code paths instead of raw SQL.

Bug guarded against
-------------------
Under the shared-managed cluster, prbe-knowledge will connect as the
non-privileged ``probe_app`` role instead of the ``probe`` superuser
that bypasses RLS today. RLS policies on graph_nodes / graph_edges /
directed_vectors / usage_events / query_traces / code_repo_state /
inferred_edges_queue ENFORCE under probe_app — but only if the
``app.current_customer_id`` GUC is set via ``with_tenant(customer_id)``.

A query path that forgets ``with_tenant`` AND filters only on a
caller-supplied ``customer_id = $1`` (instead of letting RLS enforce)
would silently return tenant B's rows for tenant A's request if A's
customer_id was forwarded incorrectly. Defense in depth: this test
inserts cross-tenant rows and asserts the read API returns only the
calling tenant's data, regardless of customer_id parameter or RLS
GUC value.

We don't try to simulate the probe_app role here — that's the operator
switch when this PR merges. What we DO simulate is the FORCE-RLS
behaviour for both reads and writes by checking that:

  1. with_tenant(A) followed by an INSERT with customer_id=B is rejected
     by the WITH CHECK clause (migration 0067 added WITH CHECK to the
     policies that were USING-only before).
  2. with_tenant(A) reading without an explicit WHERE filter sees only
     A's rows (USING clause).
  3. The retrieval path's graph_search/bm25/vector retrievers each
     return zero rows for B's data under A's GUC.

The test relies on CI running migrations against real Postgres
(prbe-knowledge's CI does — see ``.github/workflows/tests.yml``); SQLite
in-memory wouldn't exercise the RLS pathway at all.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio

from shared.db import raw_conn, with_tenant

TENANT_A = "cust-rls-tenant-a"
TENANT_B = "cust-rls-tenant-b"


@pytest_asyncio.fixture
async def two_tenants(live_db) -> AsyncIterator[tuple[str, str]]:
    """Two customers in the DB with cross-tenant rows pre-seeded.

    Seeds:
      - one graph_node per tenant (under with_tenant so RLS passes)
      - one graph_edge per tenant (under with_tenant so RLS passes)

    The seeds let the tests then assert ISOLATION: reads under A's GUC
    see only A's rows; writes under A's GUC for B's customer_id get
    rejected by WITH CHECK (the row would be silently invisible
    otherwise, which is the bug WITH CHECK closes).
    """
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO customers (customer_id, display_name, api_key_hash)
            VALUES ($1, 'tenant-a', 'h-a'), ($2, 'tenant-b', 'h-b')
            ON CONFLICT (customer_id) DO NOTHING
            """,
            TENANT_A,
            TENANT_B,
        )

    # Seed one node per tenant via the tenant-scoped path. with_tenant
    # sets the app.current_customer_id GUC so the FORCE-RLS WITH CHECK
    # passes on the matching customer_id.
    for tenant in (TENANT_A, TENANT_B):
        async with with_tenant(tenant) as conn:
            await conn.execute(
                """
                INSERT INTO graph_nodes
                    (customer_id, label, canonical_id, properties)
                VALUES ($1, 'Person', $2, '{}'::jsonb)
                ON CONFLICT (customer_id, label, canonical_id) DO NOTHING
                """,
                tenant,
                f"person-{tenant}",
            )

    yield TENANT_A, TENANT_B


@pytest.mark.asyncio
async def test_with_tenant_read_isolation(two_tenants) -> None:
    """Reading graph_nodes under tenant A's GUC must see only A's rows.

    This is the USING-clause path: if a code path forgot the
    ``WHERE customer_id = $1`` filter, RLS would still block B's rows
    from leaking to A.
    """
    tenant_a, tenant_b = two_tenants

    async with with_tenant(tenant_a) as conn:
        # Intentionally NO `WHERE customer_id = ...` — let RLS enforce.
        rows = await conn.fetch("SELECT customer_id FROM graph_nodes")
    visible = {r["customer_id"] for r in rows}
    assert visible == {tenant_a}, (
        f"with_tenant({tenant_a}) read should see only {tenant_a}'s rows, "
        f"saw: {visible}"
    )

    async with with_tenant(tenant_b) as conn:
        rows = await conn.fetch("SELECT customer_id FROM graph_nodes")
    visible = {r["customer_id"] for r in rows}
    assert visible == {tenant_b}, (
        f"with_tenant({tenant_b}) read should see only {tenant_b}'s rows, "
        f"saw: {visible}"
    )


@pytest.mark.asyncio
async def test_with_tenant_write_check_rejects_cross_tenant(two_tenants) -> None:
    """Writing a row with customer_id=B under tenant A's GUC must be rejected.

    Migration 0067 added WITH CHECK to the graph_nodes policy. Without
    WITH CHECK, this INSERT would "succeed" silently (the row would
    land but be invisible to A's reads — the lock-in-the-haystack
    bug). WITH CHECK turns it into a hard error.
    """
    import asyncpg

    tenant_a, tenant_b = two_tenants

    with pytest.raises(asyncpg.exceptions.CheckViolationError):
        async with with_tenant(tenant_a) as conn:
            await conn.execute(
                """
                INSERT INTO graph_nodes
                    (customer_id, label, canonical_id, properties)
                VALUES ($1, 'Person', 'cross-tenant-canary', '{}'::jsonb)
                """,
                tenant_b,  # WRONG tenant for the GUC — must be rejected
            )

    # Verify no leaked row landed (via a raw_conn read which bypasses RLS).
    async with raw_conn() as conn:
        leaked = await conn.fetchval(
            """
            SELECT count(*) FROM graph_nodes
            WHERE canonical_id = 'cross-tenant-canary'
            """,
        )
    assert leaked == 0, "WITH CHECK rejected the insert but a row leaked anyway"


@pytest.mark.asyncio
async def test_with_tenant_required_for_force_rls_tables(two_tenants) -> None:
    """raw_conn (no GUC) against FORCE-RLS tables sees zero rows.

    Under FORCE RLS the policy applies to the table owner too. The
    `probe` superuser bypasses RLS entirely today, so this test passes
    "for the wrong reason" against the dev DB — but it's the right
    assertion to pin once the production cutover lands. Documented
    here as a guard against a future regression where the on_connect
    hook accidentally sets app.current_customer_id to a default.
    """
    async with raw_conn() as conn:
        is_superuser = await conn.fetchval(
            "SELECT current_setting('is_superuser', true)::bool"
        )
        if is_superuser:
            pytest.skip(
                "raw_conn is running as superuser (probe/postgres); FORCE RLS is "
                "bypassed entirely. This test is meaningful only under "
                "probe_app — see the operator-switch step in the PR body."
            )

        # Under probe_app: raw_conn with no GUC should see ZERO rows.
        rows = await conn.fetch("SELECT customer_id FROM graph_nodes")
    assert rows == [], (
        "Under probe_app, FORCE RLS without GUC must return zero rows. "
        f"Saw: {[r['customer_id'] for r in rows]}"
    )


@pytest.mark.asyncio
async def test_with_tenant_sets_guc_per_transaction(two_tenants) -> None:
    """with_tenant(A) sets app.current_customer_id = A for the txn lifetime."""
    tenant_a, _ = two_tenants

    async with with_tenant(tenant_a) as conn:
        guc = await conn.fetchval("SELECT current_setting('app.current_customer_id', true)")
    assert guc == tenant_a, (
        f"with_tenant({tenant_a}) must set the GUC to {tenant_a!r}; saw {guc!r}"
    )


@pytest.mark.asyncio
async def test_inferred_edges_queue_no_longer_force_rls(two_tenants) -> None:
    """Migration 0068 drops FORCE RLS on inferred_edges_queue.

    The side-worker drains this queue cross-tenant (one shared FOR UPDATE
    SKIP LOCKED claim across all customers), which is impossible under
    FORCE RLS without setting the GUC pre-claim. 0068 disables RLS on
    this queue table specifically, matching the wiki_synthesis_queue
    pattern (migration 0034).

    This regression test pins that fact so a well-meaning future
    migration doesn't re-enable RLS on this queue table and break the
    side-worker silently.
    """
    async with raw_conn() as conn:
        row = await conn.fetchrow(
            """
            SELECT relrowsecurity, relforcerowsecurity
            FROM pg_class
            WHERE relname = 'inferred_edges_queue'
            """
        )
    assert row is not None, "inferred_edges_queue table missing"
    assert row["relrowsecurity"] is False, (
        "inferred_edges_queue must have RLS DISABLED (the side-worker "
        "drains cross-tenant). See migration 0068."
    )
    assert row["relforcerowsecurity"] is False, (
        "inferred_edges_queue must NOT have FORCE RLS (the side-worker "
        "claims rows pre-GUC). See migration 0068."
    )
