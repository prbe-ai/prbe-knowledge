"""Retire the `Experiment` graph nodes: fold each into the `Project` node that
now names the same thing.

research-os made every experiment a Project of kind 'experiment' sharing the
experiment's id, and its projections now emit `Project` / `project:<uuid>`
where they emitted `Experiment` / `experiment:<uuid>`. The label is gone from
NodeLabel (an arriving one is mapped at validation, constants.
RETIRED_NODE_LABELS), which leaves the nodes already in the graph. This moves
them.

Usage:
    .venv/bin/python -m scripts.retire_experiment_nodes --customer cust-x --dry-run
    .venv/bin/python -m scripts.retire_experiment_nodes --customer cust-x
    .venv/bin/python -m scripts.retire_experiment_nodes --all-tenants

ROLLOUT ORDER, and it is not optional:
  1. research-os stops emitting `Experiment` (the search doc-type retire) and
     that image FULLY rolls. Check its `index_outbox` holds no pending row
     naming the label (the PR that ships this script says how).
  2. This engine release deploys (the label leaves NodeLabel; a straggler is
     mapped, never refused).
  3. `--dry-run` on ONE tenant, then a real run on that tenant, then look at it
     (G4 in the experiments-as-subprojects plan). Then `--all-tenants`.
Run in the other order and research-os re-creates Experiment nodes behind you.

WHY FOLD AND NOT DELETE. Most edges into an experiment node come back on their
own: research-os re-pushes the run, group and experiment documents because
their edges are part of their content hash. The experiment-anchored ARTIFACT
documents do not -- their hash is field-derived, so nothing re-pushes them --
and deleting the node would silently cut every such file off from its parent
(585 of them in production when this was written). Re-pointing keeps them.

Per node, one transaction:
  * a `Project` node with the mapped canonical id EXISTS (the experiment's own
    document has been re-pushed): re-point the old node's edges onto it,
    dropping any edge the project node already has (graph_edges_unique_lane),
    carry its provenance over, delete it, recompute `degree` for every node
    whose edge count moved.
  * it does NOT exist yet: relabel the node in place. Its node_id, edges,
    provenance and degree all stay correct; only its name changes.
`pending_edges` parked on the old label are rewritten the same way and their
leases cleared; any whose Project node has ALREADY landed are drained on the
spot, because that node's post-write pass is over and nothing else would.

Idempotent: a second run finds no `Experiment` node and changes nothing.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass, field

import asyncpg

from engine.ingest.graph_writer import drain_pending_edges
from engine.shared.config import get_settings
from engine.shared.constants import RETIRED_NODE_LABELS
from engine.shared.db import close_pool, init_pool, raw_conn, with_tenant
from engine.shared.logging import configure_logging, get_logger

log = get_logger(__name__)

RETIRED_LABEL = "Experiment"
_RETIRED = RETIRED_NODE_LABELS[RETIRED_LABEL]
_NEW_LABEL = _RETIRED.replacement.value
_OLD_PREFIX = _RETIRED.canonical_prefix
_NEW_PREFIX = _RETIRED.replacement_prefix

#: Tables keyed on (label, canonical_id) TEXT rather than node_id. They are
#: reported, not rewritten: they hold within-label merge history (auto-merge
#: of two experiment nodes), which has no meaning across a label change, and
#: nothing reads them for a label that no longer arrives.
_LABEL_KEYED_TABLES = ("entity_aliases", "entity_cluster_metadata", "entity_merge_audit")


@dataclass
class RetireStats:
    customer_id: str
    nodes: int = 0
    merged: int = 0
    relabelled: int = 0
    edges_repointed: int = 0
    duplicate_edges_dropped: int = 0
    pending_rewritten: int = 0
    pending_drained: int = 0
    skipped_ids: list[str] = field(default_factory=list)
    label_keyed_rows: dict[str, int] = field(default_factory=dict)


def mapped_canonical_id(canonical_id: str) -> str | None:
    """`project:<uuid>` for `experiment:<uuid>`; None for an id this script
    does not understand, which is reported and left alone rather than guessed
    at -- a wrong merge joins two unrelated entities."""
    if not canonical_id.startswith(_OLD_PREFIX):
        return None
    return _NEW_PREFIX + canonical_id.removeprefix(_OLD_PREFIX)


# Edges of the old node that would collide with an edge the project node
# already has once re-pointed: same type, same other end, same alias pair.
# Dropped first, or the re-point UPDATE violates graph_edges_unique_lane.
# An edge directly between the two nodes is dropped too: re-pointed, it
# becomes a self-loop on the project node that no projection ever asserted.
_DROP_DUPLICATES_SQL = """
DELETE FROM graph_edges x
 WHERE x.customer_id = $1
   AND (x.from_node_id = $2 OR x.to_node_id = $2)
   AND (
        x.from_node_id = $3 OR x.to_node_id = $3
        OR EXISTS (
            SELECT 1 FROM graph_edges y
             WHERE y.customer_id = x.customer_id
               AND y.edge_type = x.edge_type
               AND y.from_node_id = CASE WHEN x.from_node_id = $2 THEN $3 ELSE x.from_node_id END
               AND y.to_node_id   = CASE WHEN x.to_node_id   = $2 THEN $3 ELSE x.to_node_id END
               AND COALESCE(y.aliased_from_canonical_id, '') = COALESCE(x.aliased_from_canonical_id, '')
               AND COALESCE(y.aliased_to_canonical_id, '')   = COALESCE(x.aliased_to_canonical_id, '')
        )
   )
RETURNING x.from_node_id, x.to_node_id
"""

_REPOINT_FROM_SQL = (
    "UPDATE graph_edges SET from_node_id = $3 WHERE customer_id = $1 AND from_node_id = $2"
)
_REPOINT_TO_SQL = (
    "UPDATE graph_edges SET to_node_id = $3 WHERE customer_id = $1 AND to_node_id = $2"
)

_MOVE_PROVENANCE_SQL = """
INSERT INTO graph_node_provenance (node_id, customer_id, source_system, first_seen_at, last_seen_at)
SELECT $3, customer_id, source_system, first_seen_at, last_seen_at
  FROM graph_node_provenance
 WHERE customer_id = $1 AND node_id = $2
ON CONFLICT (node_id, source_system) DO UPDATE
   SET first_seen_at = LEAST(graph_node_provenance.first_seen_at, EXCLUDED.first_seen_at),
       last_seen_at  = GREATEST(graph_node_provenance.last_seen_at, EXCLUDED.last_seen_at)
"""

# Degree is maintained incrementally (+1 per endpoint on insert, -1 on
# delete). Re-pointing moves counts between nodes and dropping a duplicate
# removes one from each end, so the affected nodes are recounted from the rows
# rather than adjusted -- an adjustment inherits any drift already there.
_RECOUNT_DEGREE_SQL = """
UPDATE graph_nodes n
   SET degree = (SELECT count(*) FROM graph_edges e WHERE e.from_node_id = n.node_id)
              + (SELECT count(*) FROM graph_edges e WHERE e.to_node_id = n.node_id)
 WHERE n.customer_id = $1 AND n.node_id = ANY($2::bigint[])
"""

# Parked edges waiting on an experiment node, or asserted FROM/TO one. Each
# column pair is rewritten only where IT names the retired label with the
# mapped id shape -- guarded per pair, or a row selected for one pair would
# rewrite another pair's odd id into `project:` plus a truncated tail.
#
# `locked_until = NULL` clears a lease a drain left behind: a drain that hit an
# `Experiment` endpoint after the label left NodeLabel raised past its claim,
# and a leased row is never claimed again. The caller then DRAINS every
# rewritten row whose Project node already exists -- that node has landed and
# its post-write pass is over, so nothing else would ever materialise them.
_PENDING_PAIR = "{label} = '" + RETIRED_LABEL + "' AND {cid} LIKE '" + _OLD_PREFIX + "%'"


def _rewrite(label: str, cid: str) -> str:
    guard = _PENDING_PAIR.format(label=label, cid=cid)
    return (
        f"{label} = CASE WHEN {guard} THEN '{_NEW_LABEL}' ELSE {label} END, "
        f"{cid} = CASE WHEN {guard} "
        f"THEN '{_NEW_PREFIX}' || substr({cid}, {len(_OLD_PREFIX) + 1}) ELSE {cid} END"
    )


_PENDING_WHERE = " OR ".join(
    "(" + _PENDING_PAIR.format(label=f"{side}_label", cid=f"{side}_canonical_id") + ")"
    for side in ("missing", "from", "to")
)

_REWRITE_PENDING_SQL = f"""
UPDATE pending_edges
   SET {_rewrite("missing_label", "missing_canonical_id")},
       {_rewrite("from_label", "from_canonical_id")},
       {_rewrite("to_label", "to_canonical_id")},
       locked_until = NULL
 WHERE customer_id = $1 AND ({_PENDING_WHERE})
RETURNING missing_label, missing_canonical_id
"""

_COUNT_PENDING_SQL = f"SELECT count(*) FROM pending_edges WHERE customer_id = $1 AND ({_PENDING_WHERE})"


async def _retire_one(conn, customer_id: str, node_id: int, target_id: str, stats: RetireStats) -> None:
    """Fold or relabel one node. The caller owns the transaction."""
    project_node = await conn.fetchval(
        "SELECT node_id FROM graph_nodes WHERE customer_id = $1 AND label = $2 AND canonical_id = $3",
        customer_id,
        _NEW_LABEL,
        target_id,
    )
    if project_node is None:
        await conn.execute(
            "UPDATE graph_nodes SET label = $3, canonical_id = $4, updated_at = now() "
            "WHERE customer_id = $1 AND node_id = $2",
            customer_id,
            node_id,
            _NEW_LABEL,
            target_id,
        )
        stats.relabelled += 1
        return

    dropped = await conn.fetch(_DROP_DUPLICATES_SQL, customer_id, node_id, project_node)
    touched = {project_node}
    for row in dropped:
        touched.update((row["from_node_id"], row["to_node_id"]))
    moved_from = await conn.execute(_REPOINT_FROM_SQL, customer_id, node_id, project_node)
    moved_to = await conn.execute(_REPOINT_TO_SQL, customer_id, node_id, project_node)
    await conn.execute(_MOVE_PROVENANCE_SQL, customer_id, node_id, project_node)
    await conn.execute(
        "DELETE FROM node_post_write_queue WHERE customer_id = $1 AND node_id = $2",
        customer_id,
        node_id,
    )
    await conn.execute(
        "DELETE FROM graph_nodes WHERE customer_id = $1 AND node_id = $2", customer_id, node_id
    )
    touched.discard(node_id)
    await conn.execute(_RECOUNT_DEGREE_SQL, customer_id, sorted(touched))
    stats.merged += 1
    stats.duplicate_edges_dropped += len(dropped)
    stats.edges_repointed += int(moved_from.rsplit(" ", 1)[-1]) + int(moved_to.rsplit(" ", 1)[-1])


async def retire_customer(customer_id: str, *, dry_run: bool = False) -> RetireStats:
    """Retire every Experiment node of one tenant. One transaction per node, so
    ingest racing a relabel onto the same project id costs that node only (it
    is reported in `skipped_ids`); re-running picks it up."""
    stats = RetireStats(customer_id=customer_id)
    async with with_tenant(customer_id) as conn:
        nodes = await conn.fetch(
            "SELECT node_id, canonical_id FROM graph_nodes "
            "WHERE customer_id = $1 AND label = $2 ORDER BY node_id",
            customer_id,
            RETIRED_LABEL,
        )
        pending = int(await conn.fetchval(_COUNT_PENDING_SQL, customer_id) or 0)
        for table in _LABEL_KEYED_TABLES:
            count = int(
                await conn.fetchval(
                    f"SELECT count(*) FROM {table} WHERE customer_id = $1 AND label = $2",
                    customer_id,
                    RETIRED_LABEL,
                )
                or 0
            )
            if count:
                stats.label_keyed_rows[table] = count
    stats.nodes = len(nodes)

    if dry_run:
        async with with_tenant(customer_id) as conn:
            for node in nodes:
                target = mapped_canonical_id(node["canonical_id"])
                if target is None:
                    stats.skipped_ids.append(node["canonical_id"])
                    continue
                exists = await conn.fetchval(
                    "SELECT 1 FROM graph_nodes WHERE customer_id = $1 AND label = $2 "
                    "AND canonical_id = $3",
                    customer_id,
                    _NEW_LABEL,
                    target,
                )
                if exists:
                    stats.merged += 1
                else:
                    stats.relabelled += 1
        stats.pending_rewritten = pending
        log.info("retire_experiment_nodes.dry_run", **_log_fields(stats))
        return stats

    for node in nodes:
        target = mapped_canonical_id(node["canonical_id"])
        if target is None:
            stats.skipped_ids.append(node["canonical_id"])
            continue
        try:
            async with with_tenant(customer_id) as conn:
                await _retire_one(conn, customer_id, node["node_id"], target, stats)
        except asyncpg.UniqueViolationError:
            # Live ingest created the Project node between the lookup and the
            # relabel. This node's transaction rolled back; a re-run folds it.
            log.warning(
                "retire_experiment_nodes.node_raced_ingest",
                customer_id=customer_id,
                canonical_id=node["canonical_id"],
            )
            stats.skipped_ids.append(node["canonical_id"])
    async with with_tenant(customer_id) as conn:
        rewritten = await conn.fetch(_REWRITE_PENDING_SQL, customer_id)
    stats.pending_rewritten = len(rewritten)
    waiting_on = {
        (r["missing_label"], r["missing_canonical_id"])
        for r in rewritten
        if r["missing_label"] == _NEW_LABEL
    }
    for label, canonical_id in sorted(waiting_on):
        async with with_tenant(customer_id) as conn:
            landed = await conn.fetchval(
                "SELECT 1 FROM graph_nodes WHERE customer_id = $1 AND label = $2 "
                "AND canonical_id = $3",
                customer_id,
                label,
                canonical_id,
            )
            if landed:
                stats.pending_drained += await drain_pending_edges(
                    conn, customer_id, label, canonical_id
                )
    log.info("retire_experiment_nodes.done", **_log_fields(stats))
    return stats


def _log_fields(stats: RetireStats) -> dict:
    return {
        "customer_id": stats.customer_id,
        "nodes": stats.nodes,
        "merged": stats.merged,
        "relabelled": stats.relabelled,
        "edges_repointed": stats.edges_repointed,
        "duplicate_edges_dropped": stats.duplicate_edges_dropped,
        "pending_rewritten": stats.pending_rewritten,
        "pending_drained": stats.pending_drained,
        "skipped": len(stats.skipped_ids),
        "label_keyed_rows": stats.label_keyed_rows,
    }


async def _list_customers() -> list[str]:
    async with raw_conn() as conn:
        rows = await conn.fetch(
            "SELECT customer_id FROM customers WHERE status = 'active' ORDER BY customer_id"
        )
    return [r["customer_id"] for r in rows]


async def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--customer", help="one tenant")
    target.add_argument("--all-tenants", action="store_true", help="every active tenant")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(settings.log_level)
    await init_pool(settings)
    try:
        customers = [args.customer] if args.customer else await _list_customers()
        remaining = 0
        for cid in customers:
            stats = await retire_customer(cid, dry_run=args.dry_run)
            remaining += len(stats.skipped_ids)
        log.info(
            "retire_experiment_nodes.finished",
            tenants=len(customers),
            dry_run=args.dry_run,
            skipped=remaining,
        )
    finally:
        await close_pool()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main(sys.argv[1:])))
