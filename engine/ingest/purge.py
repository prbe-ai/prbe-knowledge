"""Per-source purge: delete everything one integration ever ingested.

Disconnecting an integration must leave nothing queryable behind. Before this
module the only bulk delete was `provisioning.delete_customer` (whole tenant),
so a caller wanting "forget this one source" had no engine-side operation and
would have to reach into this schema from outside — which is how derived
artifacts got orphaned in the past (github left `code_graph` docs, chunks, DLQ
rows and graph nodes behind).

The cascade lives here, next to the schema it deletes from, so a migration that
adds a source-tagged table can extend `_CASCADE_STEPS` in the same commit.

    ┌──────────────────────────────────────────────────────────────┐
    │ phase 1  gate close      COMMITTED ON ITS OWN                │
    │          integration_tokens + customer_source_mapping        │
    │          → connectedness.is_source_connected() now False, so │
    │            no worker can enqueue while phases 2-4 run.       │
    ├──────────────────────────────────────────────────────────────┤
    │ phase 2  cascade         ONE TRANSACTION                     │
    │          queues → doc-joined rows → documents → per-source   │
    │          state → wiki → graph provenance → orphan nodes      │
    ├──────────────────────────────────────────────────────────────┤
    │ phase 3  R2 sweep        raw/<source>/<customer>/            │
    ├──────────────────────────────────────────────────────────────┤
    │ phase 4  verify + re-sweep, looping until a pass finds       │
    │          nothing. Closes the straggler window: a worker that │
    │          claimed a queue row before phase 1 can land a       │
    │          document during phase 2.                            │
    └──────────────────────────────────────────────────────────────┘

Order inside phase 2 is not arbitrary. `chunks`, `directed_vectors` and
`failed_chunks` carry NO foreign key to `documents` (deliberate — see
schema.sql), so they must be deleted through a doc_id sub-select BEFORE the
documents they point at disappear, or they are orphaned permanently.

What `verified=True` does and does not mean
-------------------------------------------
It means: a full postcondition pass, run under the same tenant fence as the
deletes, found zero rows for this cascade and zero raw objects in R2.

It does NOT mean the source can never reappear, and callers should not read it
that way. Known limits, all inherent to the schema rather than to this code:

* **Not a quiescence barrier.** The gate closes first, so nothing NEW can be
  enqueued, but a worker that claimed a queue row before the gate shut can
  still commit a document after the final verification pass. The re-sweep loop
  catches anything that lands while the purge is running; a straggler slower
  than the loop needs another purge. Re-running is cheap and idempotent.
* **Shared graph nodes keep merged properties.** Node writes merge properties
  into one row and only provenance is per-source, so a node another source
  still asserts survives with values this source contributed. Removing those
  would need per-source property provenance.
* **Edges carry a single `source_system`.** The edge upsert key excludes it and
  first writer wins, so an edge asserted by two sources is tagged with one of
  them. Deleting by tag is the best signal the schema offers; it can drop an
  edge a surviving source also asserted, or keep one this source contributed
  to under another source's tag.
* **Wiki artifacts synthesized from this source survive.** They are written as
  `source_system='wiki'`, so they are outside the cascade by construction and
  outside the verification count too.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any

import structlog

from engine.shared.constants import SourceSystem
from engine.shared.db import with_tenant
from engine.shared.storage import StorageUnavailable, get_store

log = structlog.get_logger(__name__)

# Sources whose data a disconnect must also remove. A derived source has no
# integration of its own: it exists only because its parent ingested content,
# so it is orphaned the moment the parent goes.
_DERIVED_SOURCES: dict[SourceSystem, tuple[SourceSystem, ...]] = {
    SourceSystem.GITHUB: (SourceSystem.GITHUB, SourceSystem.CODE_GRAPH),
}

# Extra R2 prefixes beyond `raw/<source>/<customer>/` for each cascaded source.
_MANUAL_UPLOAD_STAGING = "manual_uploads/staging/{customer_id}/"


def cascade_for(source: SourceSystem) -> tuple[SourceSystem, ...]:
    """Every source_system value a disconnect of `source` must delete."""
    return _DERIVED_SOURCES.get(source, (source,))


def r2_prefixes_for(source: SourceSystem, customer_id: str) -> list[str]:
    """Raw-payload prefixes written by the ingestion path for this cascade."""
    prefixes = [f"raw/{s.value}/{customer_id}/" for s in cascade_for(source)]
    if source is SourceSystem.MANUAL_UPLOAD:
        prefixes.append(_MANUAL_UPLOAD_STAGING.format(customer_id=customer_id))
    return prefixes


@dataclass
class _Step:
    """One DELETE in the cascade, plus the COUNT that proves it worked.

    `sql` and `count_sql` take ($1 customer_id, $2 cascade source list).
    """

    table: str
    sql: str
    count_sql: str
    # Restrict this step to specific parent sources (e.g. code_repo_state is
    # only meaningful for github). Empty means "every source".
    only_for: tuple[SourceSystem, ...] = field(default_factory=tuple)


# Rows joined to documents by a doc-id column. These have no FK, so they MUST
# precede the documents delete or they are orphaned permanently.
def _doc_joined(table: str, column: str = "doc_id") -> _Step:
    where = (
        f"customer_id = $1 AND {column} IN ("
        "  SELECT doc_id FROM documents"
        "  WHERE customer_id = $1 AND source_system = ANY($2::text[])"
        ")"
    )
    return _Step(
        table=table,
        sql=f"DELETE FROM {table} WHERE {where}",
        count_sql=f"SELECT COUNT(*) FROM {table} WHERE {where}",
    )


def _by_source(table: str, column: str = "source_system") -> _Step:
    where = f"customer_id = $1 AND {column} = ANY($2::text[])"
    return _Step(
        table=table,
        sql=f"DELETE FROM {table} WHERE {where}",
        count_sql=f"SELECT COUNT(*) FROM {table} WHERE {where}",
    )


def _by_customer(table: str, only_for: tuple[SourceSystem, ...]) -> _Step:
    where = "customer_id = $1"
    return _Step(
        table=table,
        sql=f"DELETE FROM {table} WHERE {where}",
        count_sql=f"SELECT COUNT(*) FROM {table} WHERE {where}",
        only_for=only_for,
    )


# Ordered. Queues first so nothing re-materialises mid-cascade; doc-joined
# rows before documents; graph last (provenance drives node orphaning).
_CASCADE_STEPS: tuple[_Step, ...] = (
    # -- queues and in-flight work -------------------------------------
    _by_source("ingestion_queue"),
    _by_source("backfill_state"),
    _by_source("ingestion_cursors", column="source"),
    _by_source("pending_edges"),
    _by_source("wiki_synthesis_queue"),
    # inferred_edges_queue reaches documents through `anchor_doc_id`.
    _doc_joined("inferred_edges_queue", column="anchor_doc_id"),
    # -- document-joined rows (no FK to documents) ---------------------
    _doc_joined("chunks"),
    _doc_joined("directed_vectors"),
    _doc_joined("failed_chunks"),
    # -- the documents themselves --------------------------------------
    _by_source("documents"),
    # -- per-source state ----------------------------------------------
    _by_source("acl_snapshots"),
    _by_source("ingestion_events"),
    # wiki_raw_data.source / wiki_timeline_entries.source hold the source
    # system ('github' — kb/synthesis/crawlers/github.py:82), so these are
    # genuinely source-scoped. wiki_links is NOT: its `link_source` column is
    # the link KIND ('markdown'/'frontmatter'/'manual', enforced by
    # ck_wiki_links_source), so matching it against a source system would
    # delete nothing while still reporting clean.
    _by_source("wiki_timeline_entries", column="source"),
    _by_source("wiki_raw_data", column="source"),
    _by_customer("code_repo_state", only_for=(SourceSystem.GITHUB,)),
)

# Graph is handled separately: nodes are SHARED across sources, so a node only
# dies when this cascade removed its last provenance row. Edges asserted by the
# cascaded sources go regardless; edges between surviving nodes stay.
_GRAPH_PROVENANCE_SQL = (
    "DELETE FROM graph_node_provenance "
    "WHERE customer_id = $1 AND source_system = ANY($2::text[]) "
    "RETURNING node_id"
)
_GRAPH_EDGES_SQL = (
    "DELETE FROM graph_edges "
    "WHERE customer_id = $1 AND source_system = ANY($2::text[])"
)
# Deleting a node cascades its remaining provenance + edges by FK.
#
# Restricted to the nodes THIS purge just de-provenanced ($2). A blanket
# "every provenance-less node" sweep would also delete nodes that legitimately
# never had provenance — cross_repo_deps.py:1024 inserts repo Document nodes
# straight into graph_nodes with no provenance row — so purging one source
# would silently take another source's nodes with it.
_GRAPH_ORPHAN_NODES_SQL = """
    DELETE FROM graph_nodes n
    WHERE n.customer_id = $1
      AND n.node_id = ANY($2::bigint[])
      AND NOT EXISTS (
          SELECT 1 FROM graph_node_provenance p
          WHERE p.node_id = n.node_id AND p.customer_id = n.customer_id
      )
"""
_GRAPH_PROVENANCE_COUNT = (
    "SELECT COUNT(*) FROM graph_node_provenance "
    "WHERE customer_id = $1 AND source_system = ANY($2::text[])"
)
_GRAPH_EDGES_COUNT = (
    "SELECT COUNT(*) FROM graph_edges "
    "WHERE customer_id = $1 AND source_system = ANY($2::text[])"
)
# Residue check for the graph: any node still carrying provenance from a
# cascaded source. (Nodes with no provenance at all are NOT residue — they may
# predate provenance tracking or come from a writer that never records it.)
_GRAPH_ORPHAN_COUNT = """
    SELECT COUNT(*) FROM graph_nodes n
    WHERE n.customer_id = $1
      AND EXISTS (
          SELECT 1 FROM graph_node_provenance p
          WHERE p.node_id = n.node_id AND p.customer_id = n.customer_id
            AND p.source_system = ANY($2::text[])
      )
"""
# node_post_write_queue references graph_nodes by id but has NO foreign key to
# it (only to customers), so deleting an orphaned node strands its queue row
# forever. Sweep by "node no longer exists" AFTER the node delete.
_GRAPH_QUEUE_ORPHANS_SQL = """
    DELETE FROM node_post_write_queue q
    WHERE q.customer_id = $1
      AND NOT EXISTS (
          SELECT 1 FROM graph_nodes n
          WHERE n.node_id = q.node_id AND n.customer_id = q.customer_id
      )
"""
_GRAPH_QUEUE_ORPHANS_COUNT = """
    SELECT COUNT(*) FROM node_post_write_queue q
    WHERE q.customer_id = $1
      AND NOT EXISTS (
          SELECT 1 FROM graph_nodes n
          WHERE n.node_id = q.node_id AND n.customer_id = q.customer_id
      )
"""

# The gate. Deleting these is what makes is_source_connected() return False.
# integration_tokens is keyed (customer_id, source_system) and code_graph has
# no token of its own, so only the parent source's row exists to delete.
_GATE_TOKENS_SQL = (
    "DELETE FROM integration_tokens WHERE customer_id = $1 AND source_system = $2"
)
_GATE_MAPPING_SQL = (
    "DELETE FROM customer_source_mapping "
    "WHERE customer_id = $1 AND source_system = ANY($2::text[])"
)

# How many verify→re-sweep rounds before declaring the purge stuck. Each round
# only runs if the previous one found residue, so a clean purge does exactly
# one verification pass.
_MAX_VERIFY_ROUNDS = 5


async def _close_gate(customer_id: str, source: SourceSystem) -> None:
    """Phase 1. Own transaction, committed before the cascade begins.

    This has to commit separately: `is_source_connected` reads through a fresh
    READ COMMITTED connection, so a token delete still inside an open
    transaction is invisible to every worker and the gate would stay open for
    the whole cascade.
    """
    cascade = [s.value for s in cascade_for(source)]
    async with with_tenant(customer_id) as conn:
        await conn.execute(_GATE_TOKENS_SQL, customer_id, source.value)
        await conn.execute(_GATE_MAPPING_SQL, customer_id, cascade)
    log.info("purge.gate_closed", customer=customer_id, source=source.value)


async def _run_cascade(
    customer_id: str, source: SourceSystem
) -> dict[str, int]:
    """Phase 2. One transaction; returns rows deleted per table."""
    cascade = [s.value for s in cascade_for(source)]
    deleted: dict[str, int] = {}
    async with with_tenant(customer_id) as conn:
        for step in _CASCADE_STEPS:
            if step.only_for and source not in step.only_for:
                continue
            if step.sql.count("$2"):
                status = await conn.execute(step.sql, customer_id, cascade)
            else:
                status = await conn.execute(step.sql, customer_id)
            deleted[step.table] = _rows_from_status(status)

        # RETURNING gives the exact node set this purge de-provenanced, which
        # is what bounds the orphan sweep below.
        touched = await conn.fetch(_GRAPH_PROVENANCE_SQL, customer_id, cascade)
        deleted["graph_node_provenance"] = len(touched)
        node_ids = [r["node_id"] for r in touched]

        status = await conn.execute(_GRAPH_EDGES_SQL, customer_id, cascade)
        deleted["graph_edges"] = _rows_from_status(status)

        if node_ids:
            status = await conn.execute(
                _GRAPH_ORPHAN_NODES_SQL, customer_id, node_ids
            )
            deleted["graph_nodes"] = _rows_from_status(status)
        else:
            deleted["graph_nodes"] = 0
        status = await conn.execute(_GRAPH_QUEUE_ORPHANS_SQL, customer_id)
        deleted["node_post_write_queue"] = _rows_from_status(status)
    return deleted


def _rows_from_status(status: str) -> int:
    """asyncpg returns 'DELETE <n>'; pull the count out."""
    try:
        return int(status.rsplit(" ", 1)[1])
    except (IndexError, ValueError):
        return 0


async def _sweep_r2(customer_id: str, source: SourceSystem) -> tuple[int, int]:
    """Phase 3. Returns (objects deleted, per-key errors).

    Storage being unconfigured is not a failure: a deployment without R2 has
    no raw payloads to strand.
    """
    try:
        store = get_store()
        bucket = await store.bucket_for(customer_id)
    except StorageUnavailable as exc:
        log.warning(
            "purge.r2_unconfigured", customer=customer_id, error=str(exc)
        )
        return 0, 0
    deleted = errors = 0
    for prefix in r2_prefixes_for(source, customer_id):
        d, e = await store.delete_prefix(bucket, prefix)
        deleted += d
        errors += e
    return deleted, errors


async def _count_residue(
    customer_id: str, source: SourceSystem
) -> dict[str, int]:
    """Phase 4 postcondition. Every non-zero entry is something still there.

    Counts run under the tenant GUC exactly like the deletes, so a count of
    zero means the same fence the delete ran behind sees nothing — not that
    RLS silently hid rows from an unscoped connection.

    Scope note: doc-joined tables are counted THROUGH documents, so this
    proves "no chunk belongs to a surviving document of this source", not
    "no chunk was ever orphaned". The cascade deletes doc-joined rows before
    their documents inside one transaction, so an orphan can't be produced
    here; a straggler that writes a fresh document is caught by the
    source-scoped documents count and swept on the next round.
    """
    cascade = [s.value for s in cascade_for(source)]
    residue: dict[str, int] = {}
    async with with_tenant(customer_id) as conn:
        for step in _CASCADE_STEPS:
            if step.only_for and source not in step.only_for:
                continue
            if step.count_sql.count("$2"):
                n = await conn.fetchval(step.count_sql, customer_id, cascade)
            else:
                n = await conn.fetchval(step.count_sql, customer_id)
            if n:
                residue[step.table] = int(n)
        for name, sql in (
            ("graph_node_provenance", _GRAPH_PROVENANCE_COUNT),
            ("graph_edges", _GRAPH_EDGES_COUNT),
        ):
            n = await conn.fetchval(sql, customer_id, cascade)
            if n:
                residue[name] = int(n)
        n = await conn.fetchval(_GRAPH_ORPHAN_COUNT, customer_id, cascade)
        if n:
            residue["graph_nodes"] = int(n)
        n = await conn.fetchval(_GRAPH_QUEUE_ORPHANS_COUNT, customer_id)
        if n:
            residue["node_post_write_queue"] = int(n)
    return residue


async def _count_r2_residue(customer_id: str, source: SourceSystem) -> int:
    try:
        store = get_store()
        bucket = await store.bucket_for(customer_id)
    except StorageUnavailable:
        return 0
    total = 0
    for prefix in r2_prefixes_for(source, customer_id):
        total += await store.count_prefix(bucket, prefix)
    return total


async def purge_source(
    customer_id: str, source: SourceSystem, purge_id: str
) -> dict[str, Any]:
    """Delete every trace of `source` for `customer_id`.

    Idempotent: a second run over an already-purged source deletes nothing and
    verifies clean, which is what makes a failed purge safe to retry.

    Returns a result dict persisted to `purge_runs`. `verified` is True only
    when a full postcondition pass found zero residue in the database AND zero
    remaining raw objects in R2 — callers gate irreversible follow-up work
    (like dropping their own record of the connection) on that flag.
    """
    totals: dict[str, int] = {}
    r2_deleted = r2_errors = 0

    await _close_gate(customer_id, source)

    deleted = await _run_cascade(customer_id, source)
    for table, n in deleted.items():
        totals[table] = totals.get(table, 0) + n

    d, e = await _sweep_r2(customer_id, source)
    r2_deleted += d
    r2_errors += e

    # Verify, and re-run for anything that landed while we were deleting.
    residue: dict[str, int] = {}
    r2_residue = 0
    rounds = 0
    for rounds in range(1, _MAX_VERIFY_ROUNDS + 1):
        residue = await _count_residue(customer_id, source)
        r2_residue = await _count_r2_residue(customer_id, source)
        if not residue and not r2_residue:
            break
        log.warning(
            "purge.residue_found",
            customer=customer_id,
            source=source.value,
            purge_id=purge_id,
            round=rounds,
            residue=residue,
            r2_residue=r2_residue,
        )
        if rounds == _MAX_VERIFY_ROUNDS:
            break
        again = await _run_cascade(customer_id, source)
        for table, n in again.items():
            totals[table] = totals.get(table, 0) + n
        d, e = await _sweep_r2(customer_id, source)
        r2_deleted += d
        r2_errors += e

    verified = not residue and not r2_residue and r2_errors == 0
    result: dict[str, Any] = {
        "purge_id": purge_id,
        "customer_id": customer_id,
        "source": source.value,
        "cascade": [s.value for s in cascade_for(source)],
        "verified": verified,
        "rows_deleted": {k: v for k, v in totals.items() if v},
        "total_rows_deleted": sum(totals.values()),
        "r2_objects_deleted": r2_deleted,
        "r2_errors": r2_errors,
        "residue": residue,
        "r2_residue": r2_residue,
        "verify_rounds": rounds,
    }
    log.info(
        "purge.completed",
        customer=customer_id,
        source=source.value,
        purge_id=purge_id,
        verified=verified,
        total_rows=result["total_rows_deleted"],
        r2_deleted=r2_deleted,
        r2_errors=r2_errors,
    )
    return result


# ---------------------------------------------------------------------------
# purge_runs bookkeeping — durable status so a caller that lost its HTTP
# response (timeout, pod restart) can still learn the outcome.
# ---------------------------------------------------------------------------


async def create_purge_run(customer_id: str, source: SourceSystem) -> str:
    purge_id = str(uuid.uuid4())
    # purge_runs is FORCE RLS, so every access needs the tenant GUC bound —
    # a raw connection would fail the INSERT's WITH CHECK and read back zero
    # rows on the status endpoint.
    async with with_tenant(customer_id) as conn:
        await conn.execute(
            """
            INSERT INTO purge_runs (purge_id, customer_id, source_system, status)
            VALUES ($1, $2, $3, 'running')
            """,
            purge_id,
            customer_id,
            source.value,
        )
    return purge_id


async def finish_purge_run(
    customer_id: str,
    purge_id: str,
    result: dict[str, Any] | None,
    error: str | None = None,
) -> None:
    status = "failed" if error or not (result and result.get("verified")) else "done"
    async with with_tenant(customer_id) as conn:
        await conn.execute(
            """
            UPDATE purge_runs
            SET status = $2, result = $3::jsonb, error = $4, finished_at = NOW()
            WHERE purge_id = $1
            """,
            purge_id,
            status,
            json.dumps(result or {}),
            error,
        )


async def get_purge_run(customer_id: str, purge_id: str) -> dict[str, Any] | None:
    async with with_tenant(customer_id) as conn:
        row = await conn.fetchrow(
            """
            SELECT purge_id, customer_id, source_system, status, result, error,
                   started_at, finished_at
            FROM purge_runs
            WHERE purge_id = $1 AND customer_id = $2
            """,
            purge_id,
            customer_id,
        )
    return dict(row) if row else None


async def latest_purge_run(
    customer_id: str, source: SourceSystem
) -> dict[str, Any] | None:
    async with with_tenant(customer_id) as conn:
        row = await conn.fetchrow(
            """
            SELECT purge_id, customer_id, source_system, status, result, error,
                   started_at, finished_at
            FROM purge_runs
            WHERE customer_id = $1 AND source_system = $2
            ORDER BY started_at DESC
            LIMIT 1
            """,
            customer_id,
            source.value,
        )
    return dict(row) if row else None
