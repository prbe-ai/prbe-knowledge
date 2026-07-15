"""Trace blob loader: iterate yesterday's query_traces + pull blobs from R2.

RLS-aware: `query_traces` is FORCE-ROW-LEVEL-SECURITY (migration 0030);
a raw `SELECT * FROM query_traces` under the prod role returns ZERO rows.
We iterate the `customers` table and run each per-tenant query inside
`with_tenant(customer_id)` so the GUC is set correctly.

Range-predicate SQL (not `occurred_at::date = $1`) so the existing
`idx_query_traces_customer_time` index is usable. The cast variant
forces a seq scan — fine on a small table today, slow once query_traces
hits millions of rows. Regression-guarded by a static-SQL test.

Failure modes:
- Customer's R2 bucket missing or blob deleted: log + skip, continue.
- Blob gzip-decompress fails: log + skip, continue.
- Single trace row can't be loaded: never aborts the whole iteration.
"""

from __future__ import annotations

import gzip
import json
import logging
from collections.abc import AsyncIterator
from datetime import UTC, date, datetime, time, timedelta
from typing import Any

from engine.shared.db import raw_conn, with_tenant
from engine.shared.exceptions import StorageNotFound, StorageUnavailable
from engine.shared.storage import get_store

log = logging.getLogger(__name__)

# SQL kept at module scope so the regression-guard test can assert on it
# verbatim (prevents a future "fix" reintroducing the seq-scan-forcing
# `::date` cast).
_LOAD_SQL = """
SELECT request_id, customer_id, occurred_at, trace_blob_key,
       event_type, response_size_bytes,
       gatherer_status, tool_calls_count, confidence,
       cache_hit_rate, dropped_count, need_deeper_extensions
FROM query_traces
WHERE occurred_at >= $1 AND occurred_at < $2
  AND trace_blob_key IS NOT NULL
"""


async def _iter_customer_ids() -> list[str]:
    """Return all live customer_ids. Read outside `with_tenant` because
    `customers` is the routing table and not itself RLS-tenant-scoped.

    Filter: `status = 'active'` matches the schema as of migration
    chain head (no soft-delete `deleted_at` column exists; the lifecycle
    is encoded in `status` instead — value 'active' for live, anything
    else (e.g. 'suspended', 'deleted') is excluded from the nightly).
    """
    async with raw_conn() as conn:
        rows = await conn.fetch(
            "SELECT customer_id FROM customers WHERE status = 'active'"
        )
    return [r["customer_id"] for r in rows]


async def iter_trace_blobs(target_date: date) -> AsyncIterator[dict[str, Any]]:
    """Yield one trace blob dict per `query_traces` row whose
    `trace_blob_key` is set and whose `occurred_at` falls within
    `target_date` (UTC).

    Each yielded dict is the raw blob shape from `build_trace_blob`,
    plus a `_db` sub-dict carrying the summary row's columns so the
    digest layer doesn't need a second DB round-trip.
    """
    start_ts = datetime.combine(target_date, time.min, tzinfo=UTC)
    end_ts = start_ts + timedelta(days=1)
    customers = await _iter_customer_ids()
    log.info(
        "trace_analyzer.loader.start",
        extra={
            "target_date": target_date.isoformat(),
            "customer_count": len(customers),
        },
    )

    store = get_store()
    total_yielded = 0
    total_skipped = 0
    for customer_id in customers:
        async with with_tenant(customer_id) as conn:
            rows = await conn.fetch(_LOAD_SQL, start_ts, end_ts)
        if not rows:
            continue
        try:
            bucket = await store.bucket_for(customer_id)
        except StorageUnavailable as exc:
            log.warning(
                "trace_analyzer.loader.bucket_lookup_failed",
                extra={"customer_id": customer_id, "error": str(exc)},
            )
            total_skipped += len(rows)
            continue
        for r in rows:
            key = r["trace_blob_key"]
            try:
                body = await store.get(bucket, key)
            except StorageNotFound:
                # DB row points at an R2 object that no longer exists
                # (lifecycle TTL aged it out, or the upload silently
                # never landed). Log + skip — common enough that we
                # don't want a single missing blob to abort the run.
                log.warning(
                    "trace_analyzer.loader.blob_missing",
                    extra={"customer_id": customer_id, "key": key},
                )
                total_skipped += 1
                continue
            except StorageUnavailable as exc:
                log.warning(
                    "trace_analyzer.loader.r2_get_failed",
                    extra={
                        "customer_id": customer_id,
                        "key": key,
                        "error": str(exc),
                    },
                )
                total_skipped += 1
                continue
            try:
                blob = json.loads(gzip.decompress(body))
            except (OSError, json.JSONDecodeError) as exc:
                log.warning(
                    "trace_analyzer.loader.blob_decode_failed",
                    extra={
                        "customer_id": customer_id,
                        "key": key,
                        "error": str(exc),
                    },
                )
                total_skipped += 1
                continue
            # Stitch DB summary columns into the yielded dict for easy
            # access by the digest layer.
            blob["_db"] = {
                "request_id": str(r["request_id"]),
                "customer_id": r["customer_id"],
                "occurred_at": r["occurred_at"].isoformat(),
                "event_type": r["event_type"],
                "response_size_bytes": r["response_size_bytes"],
                "gatherer_status": r["gatherer_status"],
                "tool_calls_count": r["tool_calls_count"],
                "confidence": r["confidence"],
                "cache_hit_rate": (
                    float(r["cache_hit_rate"]) if r["cache_hit_rate"] is not None else None
                ),
                "dropped_count": r["dropped_count"],
                "need_deeper_extensions": r["need_deeper_extensions"],
                "trace_blob_key": key,
                "bucket_name": bucket,
            }
            total_yielded += 1
            yield blob

    log.info(
        "trace_analyzer.loader.complete",
        extra={
            "target_date": target_date.isoformat(),
            "yielded": total_yielded,
            "skipped": total_skipped,
        },
    )


__all__ = ["iter_trace_blobs"]
