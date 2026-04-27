"""Periodic finalizer for Claude Code sessions that go idle.

For each (customer, session) where the most recent ingestion_queue row is
older than `idle_minutes` and we haven't already enqueued a finalize event,
write a placeholder R2 object and INSERT a synthetic '<session>:finalize'
queue row that the worker will dispatch to ClaudeCodeConnector.normalize
with session_complete=True (via the :finalize suffix detection in
fetch_supplementary).
"""
from __future__ import annotations

import orjson

from shared.constants import SourceSystem
from shared.db import get_pool
from shared.logging import get_logger
from shared.storage import get_store

log = get_logger(__name__)


async def enqueue_idle_session_finalizers(idle_minutes: int) -> int:
    find_sql = """
    WITH idle_sessions AS (
        SELECT customer_id,
               split_part(source_event_id, ':', 1) AS session_id,
               max(enqueued_at) AS last_seen
          FROM ingestion_queue
         WHERE source_system = $1
           AND source_event_id NOT LIKE '%:finalize'
         GROUP BY customer_id, session_id
        HAVING max(enqueued_at) < NOW() - make_interval(mins => $2)
    )
    SELECT i.customer_id, i.session_id
      FROM idle_sessions i
      LEFT JOIN ingestion_queue q
        ON q.customer_id = i.customer_id
       AND q.source_system = $1
       AND q.source_event_id = i.session_id || ':finalize'
     WHERE q.source_event_id IS NULL
    """
    insert_sql = """
    INSERT INTO ingestion_queue
        (customer_id, source_system, source_event_id, payload_s3_key, status, enqueued_at)
    VALUES ($1, $2, $3, $4, 'pending', NOW())
    ON CONFLICT (customer_id, source_system, source_event_id) DO NOTHING
    """

    store = get_store()
    enqueued = 0
    async with get_pool().acquire() as conn:
        rows = await conn.fetch(find_sql, SourceSystem.CLAUDE_CODE.value, idle_minutes)
        seen_buckets: set[str] = set()
        for r in rows:
            customer_id = r["customer_id"]
            session_id = r["session_id"]
            bucket = store.bucket_for(customer_id)
            if bucket not in seen_buckets:
                await store.ensure_bucket(bucket)
                seen_buckets.add(bucket)
            placeholder_key = f"raw/claude_code/{customer_id}/{session_id}/finalize.marker"
            placeholder_body = orjson.dumps({
                "device_id": "cron-finalize",
                "session_id": session_id,
                "batch_seq": -1,
                "cwd": None,
                "events": [],
                "finalize": True,
            })
            await store.put(bucket, placeholder_key, placeholder_body)
            status = await conn.execute(
                insert_sql,
                customer_id,
                SourceSystem.CLAUDE_CODE.value,
                f"{session_id}:finalize",
                placeholder_key,
            )
            # asyncpg returns 'INSERT 0 1' on success, 'INSERT 0 0' when ON CONFLICT fires.
            if status.endswith(" 1"):
                enqueued += 1
    log.info(
        "session_completer.run",
        extra={"idle_minutes": idle_minutes, "enqueued": enqueued, "candidates": len(rows)},
    )
    return enqueued
