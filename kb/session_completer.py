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

Protocol-2 sessions are covered the same way. Their client's own finalize is
recorded in `session_streams.finalized` (true exactly when the newest accepted
batch was the finalize); a v2 session is left alone when that is true and a v2
batch is still the newest key, or when the sweep's own marker is. The marker
never touches `session_streams`: it is a server observation, not a client claim.

"Ended" is not "mined". A session whose last complete pass was partial
(`ingestion_queue.extraction_outcome.authoritative` false: a segment failed or
the model declined the tool) is re-queued once it is idle, up to
MAX_EXTRACTION_RETRIES times. A pass skipped because extraction was switched
off is re-queued up to MAX_DISABLED_RETRIES times (daily), and not at all while
this sweep's own settings say extraction is off. A session that only hit the
segment cap is final: the same transcript hits the same cap every time.

Only ACTIVE tenants are swept, for both halves (shared.tenant_status): a
terminated tenant's sessions are held for its purge, never ended or re-mined.
"""

from __future__ import annotations

from datetime import UTC, datetime

from engine.shared.config import get_settings
from engine.shared.constants import SourceSystem
from engine.shared.db import get_pool
from engine.shared.logging import get_logger
from engine.shared.session_signals import (
    V2_KEY_SEGMENT,
    cron_marker_body,
    cron_marker_key,
    ends_v1_session_sql,
    is_cron_marker_key,
    last_key_sql,
)
from engine.shared.storage import get_store
from engine.shared.tenant_status import ACTIVE_TENANTS_SQL, active_tenant_sql
from kb.session_receipts import _lock

log = get_logger(__name__)

#: Every agent-session source ingests in coalescing mode and needs ending when
#: idle. Each is swept on its own so the marker lands under its own R2 prefix.
AGENT_SOURCES = (SourceSystem.CLAUDE_CODE, SourceSystem.CODEX, SourceSystem.PI)

#: How many times the sweep re-queues a session whose last pass was partial.
#: A segment that always fails must not become a new daily loop.
MAX_EXTRACTION_RETRIES = 3

#: A switched-off pass spends no model call, only a re-read; a week of daily
#: retries covers an emergency stop without re-reading forever.
MAX_DISABLED_RETRIES = 7


#: Idle, not already ended by a key-visible signal, not being processed right
#: now. A protocol-2 client finalize is not visible in the key; the find query
#: checks `session_streams` for it, and `_end_one` again under the lock. The
#: same predicate is re-checked in the UPDATE, because a batch can land between
#: this read and that write.
def _eligible(last_key: str) -> str:
    return f"""
       status IS DISTINCT FROM 'processing'
   AND enqueued_at < NOW() - make_interval(mins => $2)
   AND cardinality(payload_s3_keys) > 0
   -- A pre-0026 legacy identity (`<session>:<batch>`) is not a session.
   AND strpos(source_event_id, ':') = 0
   AND NOT {ends_v1_session_sql(last_key)}
"""


#: The last key is read once per row: repeating the array subscript makes
#: Postgres de-TOAST the whole array per reference (rows hold up to ~2,700 keys).
#:
#: Per tenant, under that tenant's RLS setting, so a protocol-2 session its
#: client already finalized is excluded HERE rather than skipped per row: most
#: v2 rows are finalized, and an oldest-first LIMIT spent skipping them would
#: re-read the same finished rows every run and never reach a real candidate.
_FIND_SQL = f"""
SELECT q.queue_id, q.customer_id, q.source_event_id AS session_id, q.enqueued_at
  FROM ingestion_queue q
  CROSS JOIN LATERAL (SELECT {last_key_sql("q.payload_s3_keys")} AS last_key OFFSET 0) lk
 WHERE q.source_system = $1
   AND q.customer_id = $4
   AND {_eligible("lk.last_key")}
   AND NOT EXISTS (
        SELECT 1 FROM session_streams s
         WHERE s.customer_id = q.customer_id
           AND s.source_system = q.source_system
           AND s.session_id = q.source_event_id
           AND s.finalized
           AND strpos(lk.last_key, '{V2_KEY_SEGMENT}') > 0)
   AND (q.enqueued_at, q.queue_id) > ($5::timestamptz, $6::bigint)
 ORDER BY q.enqueued_at, q.queue_id
 LIMIT $3
"""

#: Rows that fail keep their place (oldest first), so a run pages past them
#: until it has ENDED `limit` sessions or run out of candidates. The keyset
#: cursor only moves forward through a finite set, so this always terminates;
#: a page cap would let enough persistent failures starve every row behind
#: them, run after run.

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
   AND {_eligible(last_key_sql())}
RETURNING queue_id
"""

#: A done row whose last complete pass was partial, and which has not changed
#: since that pass (keys only ever get appended, so an equal count means the
#: pass's end signal is still the newest key). `$3` is the retry bound.
_RETRYABLE = f"""
       source_system = $1
   AND status = 'done'
   AND enqueued_at < NOW() - make_interval(mins => $2)
   AND (extraction_outcome ->> 'authoritative') = 'false'
   AND (extraction_outcome ->> 'keys')::int = cardinality(payload_s3_keys)
   AND extraction_outcome ->> 'reason' <> 'capped'
   AND COALESCE((extraction_outcome ->> 'retries')::int, 0) < CASE
         WHEN extraction_outcome ->> 'reason' = 'disabled' THEN {MAX_DISABLED_RETRIES}
         ELSE $3 END
"""

_RETRY_FIND_SQL = f"""
SELECT queue_id FROM ingestion_queue
 WHERE {_RETRYABLE}
   AND {active_tenant_sql("ingestion_queue.customer_id")}
 ORDER BY enqueued_at
 LIMIT $4
"""

#: Hand it back to the worker as it is: its end signal is already on top.
_RETRY_SQL = f"""
UPDATE ingestion_queue
   SET status = 'pending',
       version = version + 1,
       completed_at = NULL,
       error = NULL,
       enqueued_at = NOW(),
       extraction_outcome = jsonb_set(extraction_outcome, '{{retries}}',
                        to_jsonb(COALESCE((extraction_outcome ->> 'retries')::int, 0) + 1))
 WHERE queue_id = $4
   AND {_RETRYABLE}
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
    failed = 0
    capped = False
    async with get_pool().acquire() as conn:
        seen_buckets: set[str] = set()
        # ACTIVE tenants only: a terminated tenant's sessions are held, not
        # mined (shared.tenant_status). This used to be every `customers` row.
        tenants = [r["customer_id"] for r in await conn.fetch(ACTIVE_TENANTS_SQL)]
        for source in AGENT_SOURCES:
            ended_here = 0
            for tenant in tenants:
                if ended_here >= limit:
                    break
                after = (datetime(1970, 1, 1, tzinfo=UTC), 0)
                while True:
                    async with conn.transaction():
                        await conn.execute(
                            "SELECT set_config('app.current_customer_id', $1, true)", tenant
                        )
                        rows = await conn.fetch(
                            _FIND_SQL, source.value, idle_minutes, limit - ended_here, tenant, *after
                        )
                    if not rows:
                        break
                    candidates += len(rows)
                    after = (rows[-1]["enqueued_at"], rows[-1]["queue_id"])
                    for r in rows:
                        try:
                            outcome = await _end_one(
                                conn, store, seen_buckets, source, r, idle_minutes, dry_run=dry_run
                            )
                        except Exception as exc:
                            # One bad row (a tenant with no bucket, an R2
                            # error) must not end the run, nor hold its slot:
                            # the next page goes past it.
                            log.warning(
                                "session_completer.row_failed",
                                queue_id=r["queue_id"],
                                error=f"{type(exc).__name__}: {str(exc)[:200]}",
                            )
                            failed += 1
                            continue
                        if outcome:
                            enqueued += 1
                            ended_here += 1
                        else:
                            skipped += 1
                    if ended_here >= limit:
                        capped = True
                        break
        retried = await _retry_partial_passes(conn, idle_minutes, limit=limit, dry_run=dry_run)
    log.info(
        "session_completer.run",
        idle_minutes=idle_minutes,
        enqueued=enqueued,
        candidates=candidates,
        skipped=skipped,
        failed=failed,
        retried=retried,
        limit=limit,
        dry_run=dry_run,
        # A run that hits the cap left work behind. Silent truncation here
        # would read as "the corpus is fully swept".
        capped=capped,
    )
    return enqueued + retried


async def _end_one(conn, store, seen_buckets: set[str], source, r, idle_minutes: int, *, dry_run: bool) -> bool:
    """End one idle session under its lock. False when it turned out not to
    need ending (a v2 stream owns it, or a batch landed since it was found)."""
    customer_id, session_id = r["customer_id"], r["session_id"]
    async with conn.transaction():
        await conn.execute("SELECT set_config('app.current_customer_id', $1, true)", customer_id)
        await _lock(conn, customer_id, source.value, session_id)
        if await _v2_client_already_ended(conn, r["queue_id"], customer_id, source.value, session_id):
            return False
        if dry_run:
            return True
        key = cron_marker_key(source.value, customer_id, session_id)
        # The object must exist before any row references it. One object per
        # session, rewritten identically, so a write whose UPDATE is then
        # skipped leaves nothing dangling.
        bucket = await store.bucket_for(customer_id)
        if bucket not in seen_buckets:
            await store.ensure_bucket(bucket)
            seen_buckets.add(bucket)
        await store.put(bucket, key, cron_marker_body(session_id))
        return await conn.fetchval(_END_SQL, source.value, idle_minutes, key, r["queue_id"]) is not None


async def _v2_client_already_ended(conn, queue_id: int, customer_id: str, source: str, session_id: str) -> bool:
    """True when a protocol-2 client finalize is still the newest thing on the row.

    `session_streams.finalized` is true exactly when the newest ACCEPTED v2
    batch was the finalize (kb/session_receipts.accept admits batches strictly
    in sequence). It only ends the session while a v2 batch is also the newest
    key: anything appended after it -- the sweep's own earlier marker included --
    means the key-visible rule already decided. Caller holds the session lock.
    """
    finalized = await conn.fetchval(
        "SELECT finalized FROM session_streams WHERE customer_id=$1 AND source_system=$2 AND session_id=$3",
        customer_id,
        source,
        session_id,
    )
    if not finalized:
        return False
    last = await conn.fetchval(
        f"SELECT {last_key_sql()} FROM ingestion_queue WHERE queue_id=$1", queue_id
    )
    return bool(last) and V2_KEY_SEGMENT in last and not is_cron_marker_key(last)


async def _retry_partial_passes(conn, idle_minutes: int, *, limit: int, dry_run: bool) -> int:
    """Re-queue ended sessions whose last complete pass was partial."""
    if not get_settings().claude_code_extraction_enabled:
        # Re-queueing now would only produce more switched-off passes.
        log.info("session_completer.retry_skipped", reason="extraction disabled")
        return 0
    if not await conn.fetchval(
        "SELECT 1 FROM information_schema.columns WHERE table_schema = current_schema() "
        "AND table_name = 'ingestion_queue' AND column_name = 'extraction_outcome'"
    ):
        # Research applies kb migrations on its own deploy, which can trail the
        # worker's image. Ending sessions does not depend on this; retrying does.
        log.warning("session_completer.retry_skipped", reason="extraction_outcome column missing")
        return 0
    retried = 0
    for source in AGENT_SOURCES:
        rows = await conn.fetch(
            _RETRY_FIND_SQL, source.value, idle_minutes, MAX_EXTRACTION_RETRIES, limit
        )
        if dry_run:
            retried += len(rows)
            continue
        for r in rows:
            if await conn.fetchval(
                _RETRY_SQL, source.value, idle_minutes, MAX_EXTRACTION_RETRIES, r["queue_id"]
            ) is not None:
                retried += 1
    return retried


async def has_v2_stream(conn, customer_id: str, source: str, session_id: str) -> bool:
    """A protocol-2 stream owns this session, whatever its keys look like.

    `session_streams` is the authority and is FORCE RLS: the caller must hold
    a transaction with `app.current_customer_id` set to `customer_id`.
    """
    return bool(
        await conn.fetchval(
            "SELECT 1 FROM session_streams WHERE customer_id=$1 AND source_system=$2 AND session_id=$3",
            customer_id,
            source,
            session_id,
        )
    )
