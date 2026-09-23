"""Periodic finalizer for agent-session sources (Claude Code, Codex, pi) that go idle.

A session is mined only once it has ENDED (engine.shared.session_signals: the
newest key on its queue row is an end signal). Clients end their own sessions;
this sweep ends the ones nobody did -- a hard-killed terminal, a laptop that
never came back -- by appending a `finalize.marker` key to the live row once it
has been idle for `idle_minutes`. The worker then mines it once.

A session whose newest key is ALREADY an end signal is left alone. That one
check is the whole cost bound: the previous version asked instead whether a
marker key was still present anywhere, while the worker deleted that key after
mining, so every idle session was re-ended and fully re-mined once a day.

Rows are only ever UPDATED. A session with no live row has nothing to mine: the
old INSERT path created marker-only rows that dead-lettered on "missing
employee_id" (288 of them on research).

Protocol-2 sessions are skipped here: the client journal finalizes them, and
covering the ones it misses is a separate change.
"""

from __future__ import annotations

import orjson

from engine.shared.constants import SourceSystem
from engine.shared.db import get_pool
from engine.shared.logging import get_logger
from engine.shared.session_signals import (
    cron_marker_key,
    has_v2_key_sql,
    last_key_ends_v1_session_sql,
)
from engine.shared.storage import get_store
from kb.session_receipts import _lock

log = get_logger(__name__)

#: Every agent-session source ingests in coalescing mode and needs ending when
#: idle. Each is swept on its own so the marker lands under its own R2 prefix.
AGENT_SOURCES = (SourceSystem.CLAUDE_CODE, SourceSystem.CODEX, SourceSystem.PI)

#: Idle, not already ended, protocol 1, not being processed right now. The same
#: predicate is re-checked in the UPDATE under the per-session lock, because a
#: batch can land between this read and that write.
_ELIGIBLE = f"""
       status IS DISTINCT FROM 'processing'
   AND enqueued_at < NOW() - make_interval(mins => $2)
   AND cardinality(payload_s3_keys) > 0
   AND NOT {last_key_ends_v1_session_sql()}
   AND NOT {has_v2_key_sql()}
"""

_FIND_SQL = f"""
SELECT queue_id, customer_id, source_event_id AS session_id
  FROM ingestion_queue
 WHERE source_system = $1
   AND {_ELIGIBLE}
 ORDER BY enqueued_at
 LIMIT $3
"""

#: Append the marker, return the row to the worker. Conditioned on the same
#: eligibility, so a row that changed since it was found is skipped, not ended.
_END_SQL = f"""
UPDATE ingestion_queue
   SET payload_s3_keys = payload_s3_keys || ARRAY[$3]::text[],
       status = 'pending',
       version = version + 1,
       completed_at = NULL,
       error = NULL,
       enqueued_at = NOW()
 WHERE queue_id = $4
   AND source_system = $1
   AND {_ELIGIBLE}
RETURNING queue_id
"""


async def enqueue_idle_session_finalizers(
    idle_minutes: int,
    *,
    limit: int = 1000,
    dry_run: bool = False,
) -> int:
    """End idle sessions that nobody ended, so the worker mines them once.

    `limit` is per source, and it is a COST bound rather than a correctness one:
    every session this ends buys one multi-segment extraction. Anything not
    reached this run is reached on the next.

    `dry_run` counts what would be ended and writes nothing.

    Returns how many sessions were ended (or would be, on a dry run).
    """
    store = get_store()
    enqueued = 0
    candidates = 0
    skipped = 0
    capped = False
    async with get_pool().acquire() as conn:
        seen_buckets: set[str] = set()
        for source in AGENT_SOURCES:
            rows = await conn.fetch(_FIND_SQL, source.value, idle_minutes, limit)
            candidates += len(rows)
            capped = capped or len(rows) >= limit
            for r in rows:
                customer_id = r["customer_id"]
                session_id = r["session_id"]
                async with conn.transaction():
                    await conn.execute(
                        "SELECT set_config('app.current_customer_id', $1, true)", customer_id
                    )
                    await _lock(conn, customer_id, source.value, session_id)
                    # A protocol-2 stream owns this session even when no key on
                    # the row says so (session_streams is the authority).
                    if await conn.fetchval(
                        "SELECT 1 FROM session_streams WHERE customer_id=$1 AND source_system=$2 AND session_id=$3",
                        customer_id,
                        source.value,
                        session_id,
                    ):
                        skipped += 1
                        continue
                    if dry_run:
                        enqueued += 1
                        continue
                    key = cron_marker_key(source.value, customer_id, session_id)
                    # The object must exist before any row references it: a
                    # claim that cannot fetch a key fails the whole row. One
                    # object per session, rewritten identically, so a write
                    # whose UPDATE is then skipped leaves nothing dangling.
                    bucket = await store.bucket_for(customer_id)
                    if bucket not in seen_buckets:
                        await store.ensure_bucket(bucket)
                        seen_buckets.add(bucket)
                    await store.put(bucket, key, _marker_body(session_id))
                    ended = await conn.fetchval(
                        _END_SQL, source.value, idle_minutes, key, r["queue_id"]
                    )
                    if ended is None:
                        skipped += 1
                        continue
                    enqueued += 1
    log.info(
        "session_completer.run",
        idle_minutes=idle_minutes,
        enqueued=enqueued,
        candidates=candidates,
        skipped=skipped,
        limit=limit,
        dry_run=dry_run,
        # A run that hits the cap left work behind. Silent truncation here
        # would read as "the corpus is fully swept".
        capped=capped,
    )
    return enqueued


def _marker_body(session_id: str) -> bytes:
    """The placeholder the worker reads for a marker key: no events."""
    return orjson.dumps(
        {
            "device_id": "cron-finalize",
            "session_id": session_id,
            "batch_seq": -1,
            "cwd": None,
            "events": [],
            "finalize": True,
        }
    )
