"""Nightly Leiden community-detection cron.

Runs per-tenant: builds an igraph.Graph from graph_edges, partitions it
with the Leiden algorithm (ModularityVertexPartition), and writes the
resulting community_id back to graph_nodes.

Scheduled by .github/workflows/knowledge-cron.yml (cron `0 3 * * *` UTC),
which fires `flyctl machine run --command "python -m scripts.leiden_one_shot --all-customers --yes"`
against the prbe-knowledge-cron Fly app. Can also be run one-shot
manually via the same script with --customer-id.

Design decisions:
  D1 -- Leiden (full algorithm, not source_system proxy): accuracy > speed.
  D5 -- Materialize degree at write-time; community_id written here.

Tenants with < 100 edges are skipped: Leiden is meaningless at that scale
and community_id stays NULL. surprise_score() gracefully handles NULL by
skipping the cross-community bonus.

Memory budget: 1 GB per Fly machine. igraph + leidenalg are C extensions
and handle 1M+ edge graphs comfortably. Tenant-by-tenant processing keeps
peak memory = single largest tenant graph.

CRITICAL: UPDATE on graph_nodes uses `SET LOCAL row_security = OFF`
(txn-scoped) to write across the FORCE ROW LEVEL SECURITY policy. The
older ALTER TABLE NO FORCE / FORCE toggle took ACCESS EXCLUSIVE on
graph_nodes and stalled every other tenant's read for the duration of
the cron run -- on the shared cluster that's minutes of fleet-wide
blocking (Bug #70). `row_security = off` requires the connection to be
a SUPERUSER or have BYPASSRLS, which the leiden worker's DSN already
satisfies. See feedback_graph_nodes_rls_force.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

import asyncpg
import igraph
import leidenalg

from shared.config import get_settings
from shared.db import close_pool, init_pool, raw_conn
from shared.locks import advisory_lock_key
from shared.logging import configure_logging, get_logger

log = get_logger(__name__)

# Skip tenants with fewer than this many edges.
# Leiden community detection is not meaningful at small scales;
# partitioning a 10-node graph produces trivially correct but useless results.
MIN_EDGES_FOR_LEIDEN = 100

# Salt for advisory lock keys (namespaced to avoid collisions with other salts).
_LOCK_SALT = "leiden-community"


async def run_leiden_for_tenant(
    conn: asyncpg.Connection,
    customer_id: str,
) -> dict[str, Any]:
    """Run Leiden community detection for a single tenant.

    Returns a stats dict for logging.
    """
    # Acquire per-customer advisory lock. This prevents two concurrent cron
    # runs (or a manual one-shot + cron overlap) from racing on community_id
    # writes for the same customer. pg_advisory_xact_lock is held for the
    # duration of the transaction (released on commit/rollback).
    lock_key = advisory_lock_key(_LOCK_SALT, customer_id)
    await conn.execute("SELECT pg_advisory_xact_lock($1)", lock_key)

    # Fetch all edges for this customer (raw, bypassing RLS by using
    # raw_conn which is outside with_tenant). The `SET LOCAL row_security = OFF`
    # below is the write-side RLS bypass; we read here without RLS because
    # this is an administrative cron operation, not a tenant-initiated query.
    rows = await conn.fetch(
        """
        SELECT from_node_id, to_node_id
        FROM graph_edges
        WHERE customer_id = $1
        """,
        customer_id,
    )

    edge_count = len(rows)
    if edge_count < MIN_EDGES_FOR_LEIDEN:
        log.info(
            "leiden.skip_small_tenant",
            customer_id=customer_id,
            edge_count=edge_count,
            min_edges=MIN_EDGES_FOR_LEIDEN,
        )
        return {
            "customer_id": customer_id,
            "skipped": True,
            "reason": "too_few_edges",
            "edge_count": edge_count,
        }

    # Build node index: Leiden/igraph work with 0-based integer vertex IDs.
    # Map each node_id (bigint PK) to a 0-based index.
    node_id_set: set[int] = set()
    for row in rows:
        node_id_set.add(row["from_node_id"])
        node_id_set.add(row["to_node_id"])

    node_list = sorted(node_id_set)
    node_to_idx: dict[int, int] = {nid: i for i, nid in enumerate(node_list)}

    # Build igraph.Graph (undirected; edge direction not meaningful for
    # community detection on a knowledge graph).
    edge_list = [(node_to_idx[r["from_node_id"]], node_to_idx[r["to_node_id"]]) for r in rows]
    g = igraph.Graph(n=len(node_list), edges=edge_list, directed=False)
    g.simplify()  # remove self-loops and duplicate edges

    # Run Leiden with ModularityVertexPartition.
    partition = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition)

    # Build mapping from node_id (bigint) to community_id (int, 0-based).
    community_by_node: dict[int, int] = {}
    for community_idx, member_vertex_ids in enumerate(partition):
        for vertex_idx in member_vertex_ids:
            db_node_id = node_list[vertex_idx]
            community_by_node[db_node_id] = community_idx

    num_communities = len(partition)
    node_count = len(node_list)

    # Batch UPDATE graph_nodes.community_id.
    #
    # This UPDATE runs on a raw (non-tenant) connection that needs to write
    # across the FORCE ROW LEVEL SECURITY policy. Two paths previously
    # considered:
    #   (a) ALTER TABLE NO FORCE ... ALTER TABLE FORCE around the UPDATE.
    #       Works but takes ACCESS EXCLUSIVE on graph_nodes, blocking every
    #       other tenant's READ on the table for the duration -- on the
    #       shared cluster that means the cron stalls all tenant graph
    #       traffic for minutes. (Bug #70.)
    #   (b) SET LOCAL row_security = OFF: session-scoped (txn-local because
    #       the outer caller wraps in conn.transaction()), takes NO lock,
    #       overrides FORCE for this session only. Requires the connection
    #       role to be either a SUPERUSER or to have the BYPASSRLS attribute
    #       -- the worker DSN already runs as the probe superuser (see the
    #       leiden cron's deploy env), so the precondition holds.
    # We use (b). The toggle is bounded to the same txn the UPDATE runs in,
    # and the connection is returned to the pool with row_security default.
    update_node_ids = list(community_by_node.keys())
    update_community_ids = [community_by_node[nid] for nid in update_node_ids]

    await conn.execute("SET LOCAL row_security = OFF")
    # RETURNING emits per-row column values, not aggregates — wrap in a CTE
    # so we can COUNT(*) over the rows that actually got updated.
    updated = await conn.fetchval(
        """
        WITH updated AS (
            UPDATE graph_nodes
            SET community_id = t.community_id
            FROM unnest($1::bigint[], $2::int[]) AS t(node_id, community_id)
            WHERE graph_nodes.node_id = t.node_id
              AND graph_nodes.customer_id = $3
            RETURNING 1
        )
        SELECT COUNT(*) FROM updated
        """,
        update_node_ids,
        update_community_ids,
        customer_id,
    )

    stats = {
        "customer_id": customer_id,
        "skipped": False,
        "edge_count": edge_count,
        "node_count": node_count,
        "num_communities": num_communities,
        "nodes_updated": updated,
    }
    log.info("leiden.tenant_done", **stats)
    return stats


async def run_leiden_all_tenants() -> None:
    """Entry point: iterate all active customers and run Leiden for each."""
    settings = get_settings()
    await init_pool(settings)

    try:
        async with raw_conn() as list_conn:
            customers = await list_conn.fetch(
                "SELECT customer_id FROM customers ORDER BY customer_id"
            )

        customer_ids = [r["customer_id"] for r in customers]
        log.info("leiden.start", tenant_count=len(customer_ids))

        results = []
        for customer_id in customer_ids:
            try:
                # Each tenant gets its own transaction for the advisory lock
                # and community_id UPDATE. Failure on one tenant doesn't
                # abort the others.
                async with raw_conn() as conn, conn.transaction():
                    stats = await run_leiden_for_tenant(conn, customer_id)
                    results.append(stats)
            except Exception:
                log.exception(
                    "leiden.tenant_error",
                    customer_id=customer_id,
                )
                results.append(
                    {
                        "customer_id": customer_id,
                        "skipped": False,
                        "error": True,
                    }
                )

        processed = sum(1 for r in results if not r.get("skipped") and not r.get("error"))
        skipped = sum(1 for r in results if r.get("skipped"))
        errored = sum(1 for r in results if r.get("error"))
        log.info(
            "leiden.complete",
            total=len(results),
            processed=processed,
            skipped=skipped,
            errored=errored,
        )

    finally:
        await close_pool()


def main() -> None:
    configure_logging()
    log.info("leiden.cron_starting")
    asyncio.run(run_leiden_all_tenants())
    log.info("leiden.cron_finished")


if __name__ == "__main__":
    sys.exit(main())
