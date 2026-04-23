"""Reclaim ingestion_queue rows whose worker died mid-process.

A healthy worker bumps `heartbeat_at` every QUEUE_HEARTBEAT_INTERVAL_SECONDS.
If a row has status='processing' but heartbeat is older than the reclaim
threshold, the worker is dead or wedged — reset the row so another worker
can pick it up. Attempts counter is preserved so retries still tip into DLQ.

Schedule: every 2 minutes via Fly cron / GH Actions.
"""

from __future__ import annotations

import asyncio

from shared.constants import QUEUE_RECLAIM_THRESHOLD_SECONDS, QueueStatus
from shared.db import close_pool, init_pool, raw_conn
from shared.logging import configure_logging, get_logger
from shared.metrics import counter

log = get_logger(__name__)


async def reclaim() -> int:
    configure_logging()
    await init_pool()
    async with raw_conn() as conn:
        rows = await conn.fetch(
            """
            UPDATE ingestion_queue
            SET status = $1, heartbeat_at = NULL, started_at = NULL,
                error = 'reclaimed: heartbeat stale'
            WHERE status = $2
              AND heartbeat_at < NOW() - make_interval(secs => $3)
            RETURNING queue_id, customer_id, source_system
            """,
            QueueStatus.PENDING.value,
            QueueStatus.PROCESSING.value,
            QUEUE_RECLAIM_THRESHOLD_SECONDS,
        )
    await close_pool()

    if rows:
        for r in rows:
            log.warning(
                "queue.reclaimed",
                queue_id=r["queue_id"],
                customer=r["customer_id"],
                source=r["source_system"],
            )
        counter("queue.reclaimed", len(rows))
    return len(rows)


if __name__ == "__main__":  # pragma: no cover
    n = asyncio.run(reclaim())
    print(f"reclaimed {n} stuck rows")
