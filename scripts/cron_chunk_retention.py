"""Delete superseded chunks past the retention window.

WHY THIS EXISTS. Only ~35% of chunks were live on the research plane
(2026-08-30: 986,918 rows, 347,062 live), growing ~76k rows/day, and every
cost in retrieval scaled with the dead majority: the HNSW index tripled in
size and walk length, the memory-cliff arithmetic ran ~3x faster than it
needed to, and rebuild times grew with the table. Superseded chunks are
version history nobody reads -- the temporal request modes that could
(`AS_OF`, `ALL`) measured ZERO uses across 1,059 requests in 14 days.

WHAT MAY BE DELETED, and why this was blocked until 2026-08-31: a chunk is
reapable only when `valid_to` says dead AND is older than the window. Before
kb#528 the two liveness markers disagreed (11k chunks marked dead whose
version range still claimed the live doc version), so a valid_to-keyed
policy and a version-keyed one would have deleted different rows. #528 made
removal cap the version range and backfilled (0125), so `valid_to` is now
THE liveness marker and this predicate is unambiguous.

WHAT IS GIVEN UP: chunk-level time travel older than the window. `AS_OF`
queries inside the window still work; older ones under-return. That is the
priced trade (measured usage: zero), owner-approved 2026-08-31, and the
window is env-tunable if it is ever re-priced.

WHAT IS NOT TOUCHED: live chunks (valid_to IS NULL) -- the partial HNSW
index's rows -- and old `documents` versions (doc metadata rows are small;
their retention is a separate decision). Resurrection is unaffected:
deleting a dead chunk removes its (doc_id, content_hash) row entirely, so
re-appearing content INSERTs fresh instead of hitting the ON CONFLICT
resurrect path -- same end state either way.

PER-TENANT GUC LOOP because chunks carries FORCE RLS and this runs as the
`app` role -- the same rule migrations 0119/0125 established: an unscoped
DELETE matches zero rows and reports success. BATCHED because one statement
deleting ~600k rows holds locks and generates a WAL burst the standby has
to swallow; batches bound both, and each batch is its own transaction so a
kill mid-run loses nothing.

EXIT CODES mirror the guardian: 0 = done (including nothing to do),
1 = a database operation failed.

    python -m scripts.cron_chunk_retention [--dry-run]
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from engine.shared.config import get_settings
from engine.shared.constants import (
    CHUNK_RETENTION_BATCH_SIZE,
    CHUNK_RETENTION_DAYS,
)
from engine.shared.db import close_pool, get_pool, init_pool, with_tenant
from engine.shared.logging import configure_logging, get_logger

log = get_logger(__name__)


async def reap_tenant(customer_id: str, *, dry_run: bool = False) -> int:
    """Delete this tenant's reapable chunks in batches. Returns rows deleted."""
    deleted = 0
    while True:
        async with with_tenant(customer_id) as conn:
            if dry_run:
                count = await conn.fetchval(
                    """
                    SELECT count(*) FROM chunks
                    WHERE customer_id = $1
                      AND valid_to IS NOT NULL
                      AND valid_to < now() - make_interval(days => $2)
                    """,
                    customer_id,
                    CHUNK_RETENTION_DAYS,
                )
                log.info(
                    "retention.would_delete",
                    customer_id=customer_id,
                    rows=count,
                    retention_days=CHUNK_RETENTION_DAYS,
                )
                return 0
            # ctid batch: cheap to select, exact to delete, and immune to
            # the predicate shifting under us between batches.
            batch = await conn.fetchval(
                """
                WITH doomed AS (
                    SELECT ctid FROM chunks
                    WHERE customer_id = $1
                      AND valid_to IS NOT NULL
                      AND valid_to < now() - make_interval(days => $2)
                    LIMIT $3
                ),
                gone AS (
                    DELETE FROM chunks
                    WHERE ctid IN (SELECT ctid FROM doomed)
                    RETURNING 1
                )
                SELECT count(*) FROM gone
                """,
                customer_id,
                CHUNK_RETENTION_DAYS,
                CHUNK_RETENTION_BATCH_SIZE,
            )
        if not batch:
            break
        deleted += batch
        log.info(
            "retention.batch_deleted",
            customer_id=customer_id,
            batch=batch,
            total=deleted,
        )
        # A breath between batches: the point of batching is that nothing
        # else queues behind this job -- replication, autovacuum, or a
        # search's own writes.
        await asyncio.sleep(0.2)
    if deleted:
        log.info(
            "retention.tenant_done",
            customer_id=customer_id,
            deleted=deleted,
            retention_days=CHUNK_RETENTION_DAYS,
        )
    return deleted


async def run_once(*, dry_run: bool = False) -> int:
    pool = get_pool()
    async with pool.acquire() as conn:
        # `customers` is deliberately not row-secured -- the readable tenant
        # list is what makes the per-tenant GUC loop possible at all (0119).
        tenants = [
            r["customer_id"]
            for r in await conn.fetch("SELECT customer_id FROM customers ORDER BY 1")
        ]
    total = 0
    for customer_id in tenants:
        try:
            total += await reap_tenant(customer_id, dry_run=dry_run)
        except Exception as exc:
            # One tenant's failure must not starve the rest, but it DOES
            # make the run red -- a silent partial reap would read as "the
            # table just grows slower now".
            log.error(
                "retention.tenant_failed", customer_id=customer_id, error=str(exc)
            )
            return 1
    log.info("retention.done", tenants=len(tenants), deleted=total, dry_run=dry_run)
    return 0


async def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)
    await init_pool(settings)
    try:
        return await run_once(dry_run=args.dry_run)
    finally:
        await close_pool()


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
