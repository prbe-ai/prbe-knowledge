"""Reusable backfill runner.

One function, used by two callers:
  - `BackfillWorker` (services/ingestion/worker.py) — runs in the worker process,
    drains `backfill_state` rows with status='pending'.
  - `scripts/backfill.py` CLI — operator-triggered ad-hoc runs.

Responsibilities:
  - Resolve the integration token for the (customer, source)
  - Call `connector.backfill(customer_id, token, cursor)`
  - For each yielded WebhookEvent: put raw envelope to R2, insert into
    ingestion_queue. The regular ingestion worker picks it up like any webhook.
  - Update progress (`last_cursor`, `events_enqueued`, `last_progress_at`,
    `heartbeat_at`) every N events.
  - Mark done / failed at the end.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from datetime import UTC, datetime

from services.ingestion.handlers.base import ConnectorContext
from services.ingestion.handlers.registry import build_connector
from shared.constants import BackfillStatus, QueueStatus, SourceSystem
from shared.db import get_pool, raw_conn
from shared.encryption import decrypt_token
from shared.exceptions import NotSupportedByConnector, PermanentSourceError
from shared.logging import get_logger
from shared.metrics import counter
from shared.models import IntegrationToken
from shared.storage import get_store

log = get_logger(__name__)

PROGRESS_EVERY_N_EVENTS = 25
DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 30.0
# How often to re-check `integration_tokens.status='active'` mid-backfill.
# A SELECT every event is wasteful; once per ~50 events bounds the disconnect-race
# window to one Granola page (~250ms) without flooding the DB. See _token_still_active.
TOKEN_RECHECK_EVERY_N_EVENTS = 50


async def run_backfill(
    ctx: ConnectorContext,
    customer_id: str,
    source: SourceSystem,
    heartbeat_interval_seconds: float = DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
) -> int:
    """Execute a backfill for (customer, source). Returns events enqueued.

    Assumes backfill_state row already exists (status='pending' or 'running').
    Responsible for marking it complete/failed.
    """
    token = await _load_token(customer_id, source)
    if token is None:
        await _mark_failed(customer_id, source, "no active integration_tokens row")
        return 0

    cursor = await _load_cursor(customer_id, source)
    store = get_store()
    bucket = store.bucket_for(customer_id)
    await store.ensure_bucket(bucket)

    connector = build_connector(source, ctx)

    await _mark_running(customer_id, source)
    log.info(
        "backfill.start",
        customer=customer_id,
        source=source.value,
        resume_cursor=bool(cursor),
    )

    # Liveness ping is decoupled from progress writes. The reaper looks at
    # heartbeat_at; if we only updated it on enqueue (every 25 events), a
    # healthy runner blocked on a slow Slack page would look dead.
    heartbeat_task = asyncio.create_task(
        _heartbeat_loop(customer_id, source, heartbeat_interval_seconds)
    )

    enqueued = 0
    try:
        try:
            events = connector.backfill(customer_id, token, cursor)
        except NotSupportedByConnector as exc:
            # Not auth-related; passing exc is harmless since _mark_failed
            # only flips the token on PermanentSourceError with 401/403.
            await _mark_failed(
                customer_id, source, f"not supported: {exc}", exc=exc
            )
            return 0

        latest_cursor = cursor
        async for event in events:
            # Disconnect race: if the user disconnected mid-backfill, the
            # integration_tokens row was deleted (or status flipped). Bail
            # before writing more rows so we don't leave zombie R2 objects +
            # ingestion_queue entries for a now-disconnected source.
            #
            # Checking once per N events rather than every event keeps the
            # SELECT cost bounded (~250ms race window for Granola at ~50
            # events/page).
            if (
                enqueued % TOKEN_RECHECK_EVERY_N_EVENTS == 0
                and not await _token_still_active(customer_id, source)
            ):
                log.info(
                    "backfill.aborted_disconnect",
                    customer=customer_id,
                    source=source.value,
                    enqueued=enqueued,
                )
                # Do NOT call _mark_done — leave backfill_state for cleanup.
                return enqueued

            # Cursor-only checkpoint event (e.g. Granola end-of-pagination).
            # The connector is telling us the watermark may safely advance now
            # that pagination completed cleanly. Persist immediately; do NOT
            # enqueue into ingestion_queue or R2. `payload.get(...)` is
            # defensive against `raw_payload=None` or missing key.
            payload = event.raw_payload or {}
            if payload.get("_checkpoint"):
                cursor_str = payload.get("_cursor")
                if cursor_str is not None:
                    latest_cursor = str(cursor_str)
                    await _update_progress(
                        customer_id, source, latest_cursor, enqueued
                    )
                continue

            envelope = json.dumps(
                {
                    "_headers": event.headers,
                    "payload": event.raw_payload,
                    "received_at": event.received_at.isoformat(),
                    "trace_id": f"backfill-{customer_id}-{source.value}-{enqueued}",
                    "_backfill": True,
                }
            ).encode()
            key = (
                f"raw/{source.value}/{customer_id}/backfill/"
                f"{event.source_event_id.replace('/', '_')}.json"
            )
            await store.put(bucket, key, envelope)

            async with get_pool().acquire() as conn:
                # Backfill rows always land at priority 50 (never block live).
                # Both columns are populated for the migration window:
                # `payload_s3_key` for back-compat readers, `payload_s3_keys`
                # for the new array-based normalizer/worker path.
                await conn.execute(
                    """
                    INSERT INTO ingestion_queue
                        (customer_id, source_system, source_event_id,
                         payload_s3_key, payload_s3_keys, status, priority)
                    VALUES ($1, $2, $3, $4, ARRAY[$4], $5, 50)
                    ON CONFLICT DO NOTHING
                    """,
                    customer_id,
                    source.value,
                    event.source_event_id,
                    key,
                    QueueStatus.PENDING.value,
                )

            # The runner can discover a new cursor via the event's parse_hint
            # or via the connector yielding a tuple — keep it simple for now:
            # most connectors set a `_cursor` on their synthesized events
            # specifically so the runner can persist it.
            possible_cursor = payload.get("_cursor")
            if possible_cursor is not None:
                latest_cursor = str(possible_cursor)

            enqueued += 1
            if enqueued % PROGRESS_EVERY_N_EVENTS == 0:
                await _update_progress(customer_id, source, latest_cursor, enqueued)

        await _mark_done(customer_id, source, enqueued, latest_cursor)
        counter(
            "backfill.completed",
            1,
            source=source.value,
            events=enqueued,
        )
        log.info(
            "backfill.done", customer=customer_id, source=source.value, events=enqueued
        )
    except Exception as exc:
        # Pass exc so _mark_failed can detect PermanentSourceError(401/403)
        # raised by a connector mid-backfill (e.g. Granola key revoked) and
        # flip integration_tokens.status='auth_failed' atomically.
        await _mark_failed(customer_id, source, str(exc), exc=exc)
        counter("backfill.failed", 1, source=source.value)
        log.exception("backfill.failed", customer=customer_id, source=source.value)
        raise
    finally:
        heartbeat_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await heartbeat_task

    return enqueued


async def _heartbeat_loop(
    customer_id: str,
    source: SourceSystem,
    interval_seconds: float,
) -> None:
    """Unconditionally ping heartbeat_at every interval_seconds while running.

    The WHERE clause filters on status='running' so a row that's been marked
    done/failed/reclaimed mid-loop won't get a stale heartbeat written. DB
    errors are logged and swallowed: a transient blip should not kill the
    only liveness signal — if Postgres is truly down, the next reaper tick
    will catch it via the stale heartbeat anyway.
    """
    while True:
        try:
            await asyncio.sleep(interval_seconds)
            async with raw_conn() as conn:
                await conn.execute(
                    """
                    UPDATE backfill_state
                       SET heartbeat_at = NOW()
                     WHERE customer_id = $1
                       AND source_system = $2
                       AND status = $3
                    """,
                    customer_id,
                    source.value,
                    BackfillStatus.RUNNING.value,
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception(
                "backfill.heartbeat_loop.error",
                customer=customer_id,
                source=source.value,
            )


# ---- backfill_state writes -----------------------------------------------


async def enqueue_backfill(customer_id: str, source: SourceSystem) -> None:
    """Insert a `pending` backfill_state row so a worker picks it up.

    Resets cursor to NULL — use for INITIAL syncs only. For incremental
    re-polls of already-synced integrations (Granola steady-state), use
    `re_enqueue_for_polling` to preserve the cursor watermark.
    """
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO backfill_state
                (customer_id, source_system, status, started_at, events_enqueued)
            VALUES ($1, $2, $3, NULL, 0)
            ON CONFLICT (customer_id, source_system)
            DO UPDATE SET status          = EXCLUDED.status,
                          last_cursor     = NULL,
                          last_error      = NULL,
                          events_enqueued = 0,
                          started_at      = NULL,
                          heartbeat_at    = NULL,
                          completed_at    = NULL
            """,
            customer_id,
            source.value,
            BackfillStatus.PENDING.value,
        )


async def re_enqueue_for_polling(customer_id: str, source: SourceSystem) -> bool:
    """Mark a backfill_state row pending again WITHOUT clearing the cursor.

    Used for incremental polling on connectors with no webhooks (Granola).
    Preserves last_cursor so the connector resumes from its watermark.

    No-ops if the row is already pending or running (don't restart in-flight
    work). Returns True if a re-enqueue actually happened, False if it was
    a no-op or no row existed.
    """
    async with raw_conn() as conn:
        row = await conn.fetchrow(
            """
            UPDATE backfill_state
            SET status = $1,
                last_error = NULL,
                started_at = NULL,
                heartbeat_at = NULL,
                completed_at = NULL
            WHERE customer_id = $2 AND source_system = $3
              AND status NOT IN ($4, $5)
            RETURNING customer_id
            """,
            BackfillStatus.PENDING.value,
            customer_id,
            source.value,
            BackfillStatus.PENDING.value,
            BackfillStatus.RUNNING.value,
        )
    return row is not None


async def _mark_running(customer_id: str, source: SourceSystem) -> None:
    async with raw_conn() as conn:
        await conn.execute(
            """
            UPDATE backfill_state
            SET status = $1, started_at = COALESCE(started_at, NOW()), heartbeat_at = NOW()
            WHERE customer_id = $2 AND source_system = $3
            """,
            BackfillStatus.RUNNING.value,
            customer_id,
            source.value,
        )


async def _update_progress(
    customer_id: str,
    source: SourceSystem,
    cursor: str | None,
    enqueued: int,
) -> None:
    async with raw_conn() as conn:
        await conn.execute(
            """
            UPDATE backfill_state
            SET last_cursor      = $1,
                events_enqueued  = $2,
                last_progress_at = NOW(),
                heartbeat_at     = NOW()
            WHERE customer_id = $3 AND source_system = $4
            """,
            cursor,
            enqueued,
            customer_id,
            source.value,
        )


async def _mark_done(
    customer_id: str,
    source: SourceSystem,
    enqueued: int,
    cursor: str | None,
) -> None:
    async with raw_conn() as conn:
        await conn.execute(
            """
            UPDATE backfill_state
            SET status           = $1,
                last_cursor      = $2,
                events_enqueued  = $3,
                last_progress_at = NOW(),
                heartbeat_at     = NOW(),
                completed_at     = NOW()
            WHERE customer_id = $4 AND source_system = $5
            """,
            BackfillStatus.COMPLETE.value,
            cursor,
            enqueued,
            customer_id,
            source.value,
        )


async def _mark_failed(
    customer_id: str,
    source: SourceSystem,
    error: str,
    *,
    exc: Exception | None = None,
) -> None:
    """Mark a backfill_state row as failed.

    When `exc` is a `PermanentSourceError` carrying a 401/403 status, ALSO flip
    the active integration_tokens row to status='auth_failed'. This is what
    surfaces "Reconnect" in the dashboard for revoked Granola keys without
    flipping on Granola 503s (transient) or non-auth permanent errors.

    The `WHERE status='active'` filter on the token UPDATE means a concurrent
    disconnect (which deletes the row) leaves us with a 0-row UPDATE — silent,
    no error.

    Both updates run in a single transaction on the same connection so a crash
    never leaves backfill_state='failed' but integration_tokens still 'active'.
    """
    # PermanentSourceError stores kwargs in `self.context`, not as attributes
    # (see shared/exceptions.PrbeError). Granola raises with status=401/403.
    is_auth_failure = False
    if isinstance(exc, PermanentSourceError):
        status_val = exc.context.get("status", 0) if exc.context else 0
        is_auth_failure = status_val in {401, 403}

    async with raw_conn() as conn, conn.transaction():
        await conn.execute(
            """
                UPDATE backfill_state
                SET status       = $1,
                    last_error   = $2,
                    heartbeat_at = NOW()
                WHERE customer_id = $3 AND source_system = $4
                """,
            BackfillStatus.FAILED.value,
            error[:1000],
            customer_id,
            source.value,
        )
        if is_auth_failure:
            # The status='active' filter means a concurrent disconnect
            # (DELETE FROM integration_tokens) leaves this UPDATE matching
            # 0 rows — safe no-op, no error raised.
            await conn.execute(
                """
                    UPDATE integration_tokens
                    SET status             = 'auth_failed',
                        last_refresh_error = $1,
                        updated_at         = NOW()
                    WHERE customer_id   = $2
                      AND source_system = $3
                      AND status        = 'active'
                    """,
                error[:500],
                customer_id,
                source.value,
            )


async def _token_still_active(customer_id: str, source: SourceSystem) -> bool:
    """Re-check integration_tokens.status='active' mid-backfill.

    Used to bail out of `run_backfill` when a concurrent disconnect deletes
    the token row (or flips status). Cheap one-row SELECT keyed on the unique
    index (customer_id, source_system).
    """
    async with raw_conn() as conn:
        status = await conn.fetchval(
            "SELECT status FROM integration_tokens "
            "WHERE customer_id=$1 AND source_system=$2",
            customer_id,
            source.value,
        )
    return status == "active"


async def _load_cursor(customer_id: str, source: SourceSystem) -> str | None:
    async with raw_conn() as conn:
        row = await conn.fetchrow(
            "SELECT last_cursor FROM backfill_state WHERE customer_id=$1 AND source_system=$2",
            customer_id,
            source.value,
        )
    return row["last_cursor"] if row else None


async def _load_token(
    customer_id: str, source: SourceSystem
) -> IntegrationToken | None:
    async with raw_conn() as conn:
        row = await conn.fetchrow(
            """
            SELECT access_token_encrypted, refresh_token_encrypted, expires_at, scope,
                   webhook_secret
            FROM integration_tokens
            WHERE customer_id=$1 AND source_system=$2 AND status='active'
            """,
            customer_id,
            source.value,
        )
    if row is None:
        return None
    return IntegrationToken(
        customer_id=customer_id,
        source_system=source,
        access_token=decrypt_token(row["access_token_encrypted"]),
        refresh_token=(
            decrypt_token(row["refresh_token_encrypted"])
            if row["refresh_token_encrypted"]
            else None
        ),
        expires_at=row["expires_at"],
        scope=row["scope"],
        webhook_secret=row["webhook_secret"],
    )


async def claim_pending_backfill() -> tuple[str, SourceSystem] | None:
    """Pick up one pending backfill_state row atomically. SKIP LOCKED for concurrency."""
    async with get_pool().acquire() as conn, conn.transaction():
        row = await conn.fetchrow(
            """
            SELECT customer_id, source_system
            FROM backfill_state
            WHERE status = 'pending'
            ORDER BY customer_id, source_system
            FOR UPDATE SKIP LOCKED
            LIMIT 1
            """
        )
        if row is None:
            return None
        # Claim it by setting status=running immediately (inside the same tx).
        await conn.execute(
            """
            UPDATE backfill_state
            SET status = $1, started_at = NOW(), heartbeat_at = NOW()
            WHERE customer_id = $2 AND source_system = $3
            """,
            BackfillStatus.RUNNING.value,
            row["customer_id"],
            row["source_system"],
        )
    return row["customer_id"], SourceSystem(row["source_system"])


# Date/time helpers used in tests.
def utcnow() -> datetime:
    return datetime.now(UTC)
