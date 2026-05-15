"""Periodic finalizer for agent-session sources (Claude Code, Codex) that go idle.

For each (customer, session) where the most recent ingestion_queue activity is
older than `idle_minutes`, write a finalize.marker placeholder to R2 and
UPSERT it into the live session row's `payload_s3_keys` array. The worker,
on next claim, sees the marker key and triggers `session_complete=True`
extraction (qa, code_change, decision, file_ref unit docs).

Post-migration 0026 the live session row is keyed on bare session_id (no
`:batch_seq` suffix), and finalize is no longer a separate row — it
coalesces into the same row as live batches via the same UPSERT path.

Codex sessions need the same finalizer treatment as Claude Code — both
ingest in coalescing mode where idle sessions otherwise stay `pending`
forever. We loop over both sources and write the marker under the
source-prefixed R2 path (raw/claude_code/... vs raw/codex/...) so each
source's marker collides correctly with that source's live batches and
nothing else.
"""
from __future__ import annotations

import orjson

from shared.constants import (
    DEFAULT_INGESTION_PRIORITY,
    SOURCE_INGESTION_PRIORITY,
    SourceSystem,
)
from shared.db import get_pool
from shared.logging import get_logger
from shared.storage import get_store

log = get_logger(__name__)


async def enqueue_idle_session_finalizers(idle_minutes: int) -> int:
    # Find sessions where the most recent activity (across both new-format
    # rows with bare session_id and any in-flight legacy `:batch_seq`/
    # `:finalize` rows) is older than the idle window. The split_part
    # collapses both shapes onto the session_id key.
    #
    # We additionally filter out sessions whose live row already has a
    # finalize.marker in payload_s3_keys — that's the post-coalescing
    # signal that the cron has already finalized this session.
    find_sql = """
    WITH idle_sessions AS (
        SELECT customer_id,
               split_part(source_event_id, ':', 1) AS session_id,
               MAX(enqueued_at) AS last_seen
          FROM ingestion_queue
         WHERE source_system = $1
         GROUP BY customer_id, session_id
        HAVING MAX(enqueued_at) < NOW() - make_interval(mins => $2)
    )
    SELECT i.customer_id, i.session_id
      FROM idle_sessions i
      LEFT JOIN ingestion_queue q
        ON q.customer_id = i.customer_id
       AND q.source_system = $1
       AND q.source_event_id = i.session_id
     WHERE q.queue_id IS NULL
        OR NOT EXISTS (
            SELECT 1 FROM unnest(q.payload_s3_keys) AS k
            WHERE k LIKE '%/finalize.marker'
        )
    """

    # UPSERT the finalize.marker into the live session row. Same shape as
    # services/ingestion/main.py:_enqueue's CC path: append marker key,
    # bump version, refresh status, bump enqueued_at. If no live row
    # exists for this session_id (cleanly archived sessions, or sessions
    # that only ever had legacy `:batch_seq` rows that all completed),
    # this INSERTs a fresh row whose payload_s3_keys contains only the
    # marker — the worker will process it once and emit complete=True
    # with an empty event list (no unit docs, no harm).
    #
    # Intentionally NOT gated by services.ingestion.connectedness:
    # finalize markers only fire for CLAUDE_CODE / CODEX, which don't
    # have integration_tokens rows (agent sessions, no OAuth lifecycle).
    upsert_sql = """
    INSERT INTO ingestion_queue
        (customer_id, source_system, source_event_id,
         payload_s3_key, payload_s3_keys, status, priority,
         version, enqueued_at)
    VALUES ($1, $2, $3, $4, ARRAY[$4], 'pending', $5, 1, NOW())
    ON CONFLICT (customer_id, source_system, source_event_id) DO UPDATE
        SET payload_s3_keys = ingestion_queue.payload_s3_keys
                              || EXCLUDED.payload_s3_keys,
            status = 'pending',
            version = ingestion_queue.version + 1,
            completed_at = NULL,
            error = NULL,
            enqueued_at = NOW()
    """

    # Both agent-session sources ingest in coalescing mode and need
    # finalize markers when idle. We finalize each source independently so
    # the R2 marker key lives under the source-prefixed namespace and
    # collides with the right live batches.
    AGENT_SOURCES = (SourceSystem.CLAUDE_CODE, SourceSystem.CODEX)
    store = get_store()
    enqueued = 0
    total_candidates = 0
    async with get_pool().acquire() as conn:
        seen_buckets: set[str] = set()
        for source in AGENT_SOURCES:
            priority = SOURCE_INGESTION_PRIORITY.get(source, DEFAULT_INGESTION_PRIORITY)
            rows = await conn.fetch(find_sql, source.value, idle_minutes)
            total_candidates += len(rows)
            for r in rows:
                customer_id = r["customer_id"]
                session_id = r["session_id"]
                bucket = await store.bucket_for(customer_id)
                if bucket not in seen_buckets:
                    await store.ensure_bucket(bucket)
                    seen_buckets.add(bucket)
                placeholder_key = (
                    f"raw/{source.value}/{customer_id}/{session_id}/finalize.marker"
                )
                placeholder_body = orjson.dumps({
                    "device_id": "cron-finalize",
                    "session_id": session_id,
                    "batch_seq": -1,
                    "cwd": None,
                    "events": [],
                    "finalize": True,
                })
                await store.put(bucket, placeholder_key, placeholder_body)
                await conn.execute(
                    upsert_sql,
                    customer_id,
                    source.value,
                    session_id,  # bare session_id — coalescing key
                    placeholder_key,
                    priority,
                )
                enqueued += 1
    log.info(
        "session_completer.run",
        extra={
            "idle_minutes": idle_minutes,
            "enqueued": enqueued,
            "candidates": total_candidates,
        },
    )
    return enqueued
