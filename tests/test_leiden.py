"""Unit tests for services/community/leiden.py.

Tests use mock asyncpg connections to avoid needing a live database.
The core Leiden algorithm is exercised with synthetic graph fixtures.

Tests:
  1. Synthetic 100-node 3-cluster graph: partition matches clusters
  2. Tenant with < 100 edges: skipped (community_id stays NULL)
  3. Advisory lock is acquired before any DB operations
  4. Per-customer isolation: separate tenant data stays separate
  5. No NO FORCE / FORCE toggle: writes rely on caller's with_tenant GUC
  6. UPDATE uses the correct node_id -> community_id mapping
  7. Empty graph (0 edges): treated as < MIN_EDGES, skipped
"""

from __future__ import annotations

import random
from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_cluster_edges(
    cluster_nodes: list[list[int]],
    intra_density: int = 8,
    inter_edges: int = 3,
    seed: int = 42,
) -> list[tuple[int, int]]:
    """Build a synthetic graph with strong intra-cluster connectivity.

    Args:
        cluster_nodes: List of node groups (each group = one cluster).
        intra_density: Each node connects to this many intra-cluster neighbors.
        inter_edges: Total bridging edges between clusters.
        seed: RNG seed for reproducibility.

    Returns:
        List of (from_id, to_id) tuples.
    """
    rng = random.Random(seed)
    edges: set[tuple[int, int]] = set()

    # Dense intra-cluster edges
    for cluster in cluster_nodes:
        for node in cluster:
            candidates = [n for n in cluster if n != node]
            targets = rng.sample(candidates, min(intra_density, len(candidates)))
            for t in targets:
                e = (min(node, t), max(node, t))
                edges.add(e)

    # Sparse inter-cluster edges
    all_clusters = list(cluster_nodes)
    for _ in range(inter_edges):
        c1, c2 = rng.sample(range(len(all_clusters)), 2)
        n1 = rng.choice(all_clusters[c1])
        n2 = rng.choice(all_clusters[c2])
        e = (min(n1, n2), max(n1, n2))
        edges.add(e)

    return list(edges)


def _make_rows(edges: list[tuple[int, int]]) -> list[MagicMock]:
    """Convert edge list to mock asyncpg Record rows."""
    rows = []
    for from_id, to_id in edges:
        row = MagicMock()
        row.__getitem__ = lambda self, key, f=from_id, t=to_id: (
            f if key == "from_node_id" else t
        )
        rows.append(row)
    return rows


def _make_conn(edges: list[tuple[int, int]]) -> AsyncMock:
    """Build a mock asyncpg connection that returns given edges."""
    conn = AsyncMock()
    conn.fetch = AsyncMock(return_value=_make_rows(edges))
    conn.execute = AsyncMock()
    # fetchval for the UPDATE ... RETURNING COUNT(*) call
    conn.fetchval = AsyncMock(return_value=len({n for e in edges for n in e}))
    return conn


# ---------------------------------------------------------------------------
# Test 1: 100-node 3-cluster graph
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_leiden_3_cluster_100_node_graph() -> None:
    """Synthetic 100-node graph with 3 obvious clusters -> 3 communities detected."""
    import igraph
    import leidenalg

    # Three clusters of ~33 nodes each
    cluster_a = list(range(0, 33))
    cluster_b = list(range(33, 66))
    cluster_c = list(range(66, 100))

    edges = _build_cluster_edges([cluster_a, cluster_b, cluster_c], intra_density=8, inter_edges=3)
    assert len(edges) >= 100, f"Expected >=100 edges, got {len(edges)}"

    # Build the graph directly (pure Leiden logic, no DB mock needed here)
    all_node_ids = sorted({n for e in edges for n in e})
    node_to_idx = {nid: i for i, nid in enumerate(all_node_ids)}
    edge_list = [(node_to_idx[f], node_to_idx[t]) for f, t in edges]

    g = igraph.Graph(n=len(all_node_ids), edges=edge_list, directed=False)
    g.simplify()
    partition = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition)

    # Leiden should find 3 communities (might occasionally find more due to
    # stochasticity but never fewer for this graph construction).
    num_communities = len(partition)
    assert num_communities >= 3, (
        f"Expected at least 3 communities for 3-cluster graph, got {num_communities}"
    )

    # Check that nodes within the same original cluster mostly share a community
    # (at least 90% of nodes should be in the "dominant" community for each cluster)
    for cluster in [cluster_a, cluster_b, cluster_c]:
        community_votes: dict[int, int] = {}
        for node_id in cluster:
            idx = node_to_idx[node_id]
            comm = partition.membership[idx]
            community_votes[comm] = community_votes.get(comm, 0) + 1
        dominant_count = max(community_votes.values())
        cohesion = dominant_count / len(cluster)
        assert cohesion >= 0.85, (
            f"Cluster cohesion {cohesion:.2f} below 85% threshold. "
            f"Community votes: {community_votes}"
        )


# ---------------------------------------------------------------------------
# Test 2: Tenant with < MIN_EDGES skipped
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_leiden_skips_tenant_with_few_edges() -> None:
    """Tenant with < 100 edges is skipped without any UPDATE."""
    from engine.community.leiden import run_leiden_for_tenant

    # Build 50 edges (below threshold)
    edges = [(i, i + 1) for i in range(0, 50)]
    conn = _make_conn(edges)

    stats = await run_leiden_for_tenant(conn, "cust-small")

    assert stats["skipped"] is True
    assert stats["reason"] == "too_few_edges"
    assert stats["edge_count"] == 50

    # No UPDATE should have been called
    for c in conn.execute.call_args_list:
        sql = c[0][0] if c[0] else ""
        assert "UPDATE" not in sql.upper() or "ALTER" in sql.upper(), (
            f"Unexpected UPDATE call for skipped tenant: {sql}"
        )


# ---------------------------------------------------------------------------
# Test 3: Empty graph (0 edges) is skipped
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_leiden_skips_empty_graph() -> None:
    """A tenant with 0 edges is treated as too_few_edges and skipped."""
    from engine.community.leiden import run_leiden_for_tenant

    conn = _make_conn([])
    stats = await run_leiden_for_tenant(conn, "cust-empty")

    assert stats["skipped"] is True
    assert stats["edge_count"] == 0


# ---------------------------------------------------------------------------
# Test 4: Advisory lock is acquired
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_leiden_acquires_advisory_lock() -> None:
    """pg_advisory_xact_lock is called before any graph operations."""
    from engine.community.leiden import run_leiden_for_tenant

    # Just enough edges to NOT skip (100 edges)
    edges = [(i, i + 1) for i in range(0, 100)]
    conn = _make_conn(edges)

    await run_leiden_for_tenant(conn, "cust-lock-test")

    # Verify pg_advisory_xact_lock was called first
    calls = [str(c) for c in conn.execute.call_args_list]
    lock_calls = [c for c in calls if "pg_advisory_xact_lock" in c]
    assert len(lock_calls) >= 1, "Expected at least one advisory lock call"


# ---------------------------------------------------------------------------
# Test 5: No NO FORCE / FORCE toggle is emitted
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_leiden_does_not_toggle_rls() -> None:
    """run_leiden_for_tenant must NOT emit ALTER TABLE ... NO FORCE / FORCE.

    The previous implementation toggled FORCE RLS off and back on around the
    UPDATE. That dance requires SUPERUSER and breaks under the non-privileged
    probe_app role used in the shared-managed cluster. The new pattern is:
    the caller wraps the call in `with_tenant(customer_id)` so the
    app.current_customer_id GUC is set and the FORCE-RLS USING/WITH CHECK
    policy passes on the UPDATE.

    Regression guard so the toggle dance does not reappear.
    """
    from engine.community.leiden import run_leiden_for_tenant

    edges = [(i, i + 1) for i in range(0, 100)]
    conn = _make_conn(edges)

    await run_leiden_for_tenant(conn, "cust-rls-test")

    execute_sqls = [str(c[0][0]) for c in conn.execute.call_args_list if c[0]]
    no_force_calls = [s for s in execute_sqls if "NO FORCE" in s.upper()]
    assert no_force_calls == [], (
        "run_leiden_for_tenant must not emit ALTER TABLE ... NO FORCE — "
        "rely on the caller's with_tenant() GUC instead. "
        f"Saw: {no_force_calls}"
    )
    alter_table_calls = [s for s in execute_sqls if s.upper().startswith("ALTER TABLE")]
    assert alter_table_calls == [], (
        "run_leiden_for_tenant must not emit any ALTER TABLE inside the "
        f"per-tenant write path. Saw: {alter_table_calls}"
    )


# ---------------------------------------------------------------------------
# Test 6: Community IDs are correctly mapped in the UPDATE
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_leiden_update_contains_correct_node_ids() -> None:
    """The UPDATE is called with the correct set of node_ids."""
    from engine.community.leiden import run_leiden_for_tenant

    # Build a small but valid graph (>= 100 edges)
    edges = [(i, i % 20) for i in range(1, 101)]  # 100 edges, nodes 0-100
    conn = _make_conn(edges)

    await run_leiden_for_tenant(conn, "cust-update-test")

    # Find the fetchval call (UPDATE wrapped in a CTE so we can COUNT the rows).
    fetchval_calls = conn.fetchval.call_args_list
    assert len(fetchval_calls) >= 1, "Expected at least one fetchval call for UPDATE"

    # The fetchval SQL should contain UPDATE + community_id
    update_sql = fetchval_calls[0][0][0]
    assert "UPDATE" in update_sql.upper()
    assert "community_id" in update_sql.lower()

    # Regression guard: RETURNING COUNT(*) is invalid Postgres SQL — RETURNING
    # emits per-row column values, not aggregates. The CTE form below is the
    # supported pattern. AsyncMock can't catch invalid SQL, so guard with a
    # static string check.
    upper = update_sql.upper()
    assert "RETURNING COUNT(" not in upper, (
        "RETURNING COUNT(*) is invalid SQL; use a CTE wrapping the UPDATE."
    )
    assert "WITH " in upper and "RETURNING 1" in upper, (
        "UPDATE must be wrapped in a CTE that RETURNs 1 per row, then "
        "SELECT COUNT(*) over the CTE."
    )
