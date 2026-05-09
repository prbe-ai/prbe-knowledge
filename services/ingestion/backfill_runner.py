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
from dataclasses import dataclass
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


class BackfillReclaimedError(Exception):
    """The runner's claim was preempted (status flipped or started_at advanced).

    Raised when an in-loop UPDATE matches 0 rows, meaning the reaper or another
    worker now owns this (customer, source). The run loop bails without calling
    _mark_failed since the row is no longer ours to mutate.
    """


@dataclass(frozen=True)
class _ResumeState:
    cursor: str | None
    events_enqueued: int
    started_at: datetime | None


@dataclass(frozen=True)
class ChannelBackfillEnqueueResult:
    queued: bool
    reason: str


def _affected(command_tag: str) -> int:
    # asyncpg execute() returns e.g. "UPDATE 1" / "UPDATE 0"; last token is rowcount.
    parts = command_tag.split()
    return int(parts[-1]) if parts and parts[-1].isdigit() else 0


def _decode_json_object(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _slack_state_from_cursor(raw: str | None) -> dict:
    data = _decode_json_object(raw)
    if isinstance(data.get("active"), dict):
        return {
            "active": dict(data["active"]),
            "done": list(data.get("done", [])),
        }

    if "channels_remaining" in data or "current_channel" in data:
        active = {ch: None for ch in (data.get("channels_remaining") or []) if ch}
        current = data.get("current_channel")
        if current:
            active[current] = data.get("history_cursor")
        return {"active": active, "done": []}

    return {"active": {}, "done": []}


def _slack_deferred_channels(raw: str | None) -> dict[str, str | None]:
    data = _decode_json_object(raw)
    pending = data.get("pending_channels")
    if not isinstance(pending, dict):
        return {}
    return {str(ch): cursor for ch, cursor in pending.items() if ch}


def _slack_channel_cursor(channels: dict[str, str | None]) -> str:
    return json.dumps(
        {
            "active": channels,
            "done": [],
            "mode": "channel_join",
        },
        sort_keys=True,
    )


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
    state = await _load_resume_state(customer_id, source)
    cursor = state.cursor if state else None
    initial_enqueued = state.events_enqueued if state else 0

    token = await _load_token(customer_id, source)
    if token is None:
        # Mark failed against the existing claim if there is one. If the row
        # has no started_at (CLI fresh enqueue without claim), skip the mark
        # — there's no claim to invalidate.
        if state is not None and state.started_at is not None:
            await _mark_failed(
                customer_id,
                source,
                "no active integration_tokens row",
                claim_token=state.started_at,
            )
        return 0

    store = get_store()
    bucket = store.bucket_for(customer_id)
    await store.ensure_bucket(bucket)

    connector = build_connector(source, ctx)

    claim_token = await _mark_running(customer_id, source)
    log.info(
        "backfill.start",
        customer=customer_id,
        source=source.value,
        resume_cursor=bool(cursor),
        resume_events_enqueued=initial_enqueued,
    )

    # Liveness ping is decoupled from progress writes. The reaper looks at
    # heartbeat_at; if we only updated it on enqueue (every 25 events), a
    # healthy runner blocked on a slow Slack page would look dead.
    heartbeat_task = asyncio.create_task(
        _heartbeat_loop(
            customer_id, source, heartbeat_interval_seconds, claim_token
        )
    )

    # Cumulative across resumes: a reclaimed run continues incrementing the
    # prior run's count rather than starting from zero.
    enqueued = initial_enqueued
    try:
        try:
            events = connector.backfill(customer_id, token, cursor)
        except NotSupportedByConnector as exc:
            # Not auth-related; passing exc is harmless since _mark_failed
            # only flips the token on PermanentSourceError with 401/403.
            await _mark_failed(
                customer_id,
                source,
                f"not supported: {exc}",
                exc=exc,
                claim_token=claim_token,
            )
            return enqueued

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
                        customer_id, source, latest_cursor, enqueued, claim_token
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
                await _update_progress(
                    customer_id, source, latest_cursor, enqueued, claim_token
                )

        await _mark_done(
            customer_id, source, enqueued, latest_cursor, claim_token
        )
        counter(
            "backfill.completed",
            1,
            source=source.value,
            events=enqueued,
        )
        log.info(
            "backfill.done", customer=customer_id, source=source.value, events=enqueued
        )
    except BackfillReclaimedError:
        # Reaper or a competing claim took the row. The new owner is responsible
        # for it now; do NOT call _mark_failed (would clobber their state).
        log.warning(
            "backfill.preempted_by_reclaim",
            customer=customer_id,
            source=source.value,
            enqueued=enqueued,
        )
        counter("backfill.preempted", 1, source=source.value)
    except asyncio.CancelledError:
        # Process is shutting down (SIGTERM during a deploy) mid-backfill.
        # Release the claim now so the next worker resumes from `last_cursor`
        # within seconds, rather than waiting for the 5-min stale-heartbeat
        # reclaim cron. The asyncpg call inside the handler runs to completion
        # because asyncio.gather hasn't torn down yet — it's still waiting on
        # this task to exit.
        log.warning(
            "backfill.released_on_cancel",
            customer=customer_id,
            source=source.value,
            enqueued=enqueued,
        )
        counter("backfill.released_on_cancel", 1, source=source.value)
        try:
            # Shielded so a cascading cancel can't abort the UPDATE roundtrip mid-flight.
            await asyncio.shield(
                _release_for_resume(customer_id, source, claim_token)
            )
        except (Exception, asyncio.CancelledError):
            log.exception(
                "backfill.release_on_cancel_failed",
                customer=customer_id,
                source=source.value,
            )
        raise
    except Exception as exc:
        # Pass exc so _mark_failed can detect PermanentSourceError(401/403)
        # raised by a connector mid-backfill (e.g. Granola key revoked) and
        # flip integration_tokens.status='auth_failed' atomically.
        await _mark_failed(
            customer_id, source, str(exc), exc=exc, claim_token=claim_token
        )
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
    claim_token: datetime,
) -> None:
    """Unconditionally ping heartbeat_at every interval_seconds while running.

    The WHERE clause is gated on (status='running' AND started_at = claim_token)
    so a row that's been marked done/failed/reclaimed or re-claimed by another
    worker won't get a stale heartbeat written. DB errors are logged and
    swallowed: a transient blip should not kill the only liveness signal — if
    Postgres is truly down, the next reaper tick will catch it via the stale
    heartbeat anyway.
    """
    while True:
        try:
            await asyncio.sleep(interval_seconds)
            async with raw_conn() as conn:
                await conn.execute(
                    """
                    UPDATE backfill_state
                       SET heartbeat_at = NOW()
                     WHERE customer_id   = $1
                       AND source_system = $2
                       AND status        = $3
                       AND started_at    = $4
                    """,
                    customer_id,
                    source.value,
                    BackfillStatus.RUNNING.value,
                    claim_token,
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


async def enqueue_slack_channel_backfill(
    customer_id: str,
    channel_id: str,
) -> ChannelBackfillEnqueueResult:
    """Queue a Slack backfill for one newly visible channel.

    Slack has a singleton backfill_state row per customer/source. For normal
    idle/complete/failed states we can replace that row with a channel-scoped
    cursor. If the source-wide Slack backfill is currently running, defer the
    channel in `pending_channels`; `_mark_done` will immediately requeue a
    follow-up channel run instead of marking the row complete.
    """
    if not channel_id:
        return ChannelBackfillEnqueueResult(queued=False, reason="missing_channel")

    async with raw_conn() as conn, conn.transaction():
        row = await conn.fetchrow(
            """
            SELECT status, last_cursor
            FROM backfill_state
            WHERE customer_id = $1 AND source_system = $2
            FOR UPDATE
            """,
            customer_id,
            SourceSystem.SLACK.value,
        )

        if row is None:
            await conn.execute(
                """
                INSERT INTO backfill_state
                    (customer_id, source_system, status, last_cursor,
                     started_at, events_enqueued)
                VALUES ($1, $2, $3, $4, NULL, 0)
                """,
                customer_id,
                SourceSystem.SLACK.value,
                BackfillStatus.PENDING.value,
                _slack_channel_cursor({channel_id: None}),
            )
            return ChannelBackfillEnqueueResult(queued=True, reason="inserted")

        status = row["status"]
        last_cursor = row["last_cursor"]

        if status == BackfillStatus.PENDING.value and last_cursor is None:
            return ChannelBackfillEnqueueResult(
                queued=False,
                reason="covered_by_full_backfill",
            )

        if status == BackfillStatus.RUNNING.value:
            data = _decode_json_object(last_cursor)
            pending = _slack_deferred_channels(last_cursor)
            if channel_id in pending:
                return ChannelBackfillEnqueueResult(
                    queued=False,
                    reason="already_deferred",
                )
            pending[channel_id] = None
            data["pending_channels"] = pending
            await conn.execute(
                """
                UPDATE backfill_state
                SET last_cursor = $1,
                    last_error = NULL,
                    heartbeat_at = NOW()
                WHERE customer_id = $2 AND source_system = $3
                """,
                json.dumps(data, sort_keys=True),
                customer_id,
                SourceSystem.SLACK.value,
            )
            return ChannelBackfillEnqueueResult(
                queued=True,
                reason="deferred_until_running_backfill_finishes",
            )

        state = _slack_state_from_cursor(last_cursor)
        active: dict[str, str | None] = {
            str(ch): cursor for ch, cursor in state["active"].items() if ch
        }
        if channel_id in active and status == BackfillStatus.PENDING.value:
            return ChannelBackfillEnqueueResult(queued=False, reason="already_pending")
        active[channel_id] = None

        await conn.execute(
            """
            UPDATE backfill_state
            SET status = $1,
                last_cursor = $2,
                last_error = NULL,
                events_enqueued = 0,
                started_at = NULL,
                heartbeat_at = NULL,
                completed_at = NULL
            WHERE customer_id = $3 AND source_system = $4
            """,
            BackfillStatus.PENDING.value,
            _slack_channel_cursor(active),
            customer_id,
            SourceSystem.SLACK.value,
        )
        return ChannelBackfillEnqueueResult(queued=True, reason="queued")


async def _release_for_resume(
    customer_id: str,
    source: SourceSystem,
    claim_token: datetime,
) -> bool:
    """Flip the row back to 'pending' so the next worker resumes immediately.

    Used when this process is shutting down mid-backfill (SIGTERM during a
    rolling deploy). Releasing the claim now means the next worker can
    re-claim within seconds instead of waiting for the 5-min stale-heartbeat
    reclaim threshold.

    Fenced on (status='running' AND started_at = claim_token) so we don't
    clobber a row that was already reclaimed or re-claimed by a competing
    worker. last_cursor and events_enqueued are preserved so the resume
    continues exactly where we stopped.

    Returns True if a row was released, False if already preempted / not ours.
    """
    async with raw_conn() as conn:
        tag = await conn.execute(
            """
            UPDATE backfill_state
            SET status       = $1,
                started_at   = NULL,
                heartbeat_at = NULL
            WHERE customer_id   = $2
              AND source_system = $3
              AND status        = $4
              AND started_at    = $5
            """,
            BackfillStatus.PENDING.value,
            customer_id,
            source.value,
            BackfillStatus.RUNNING.value,
            claim_token,
        )
    return _affected(tag) > 0


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


async def _mark_running(customer_id: str, source: SourceSystem) -> datetime:
    """Flip to running and return started_at — the claim ownership token.

    started_at uniquely identifies this claim because every fresh claim
    (claim_pending_backfill, reaper-then-claim, fresh enqueue-then-mark_running)
    sets it to NOW() at claim time. All in-loop UPDATEs filter on this value
    to detect preemption by the reaper or a competing claim.
    """
    async with raw_conn() as conn:
        row = await conn.fetchrow(
            """
            UPDATE backfill_state
            SET status = $1, started_at = COALESCE(started_at, NOW()), heartbeat_at = NOW()
            WHERE customer_id = $2 AND source_system = $3
            RETURNING started_at
            """,
            BackfillStatus.RUNNING.value,
            customer_id,
            source.value,
        )
    if row is None or row["started_at"] is None:
        raise BackfillReclaimedError(
            f"backfill_state row vanished or has NULL started_at: "
            f"{customer_id}/{source.value}"
        )
    return row["started_at"]


async def _update_progress(
    customer_id: str,
    source: SourceSystem,
    cursor: str | None,
    enqueued: int,
    claim_token: datetime,
) -> None:
    """Write progress, gated on the row still being ours.

    Filters on (status='running' AND started_at = claim_token). If the reaper
    already flipped the row to 'pending' or another worker has re-claimed it
    (advancing started_at), the UPDATE matches 0 rows and we raise so the
    run loop can bail without overwriting the new owner's state.
    """
    async with raw_conn() as conn, conn.transaction():
        if source == SourceSystem.SLACK and cursor is not None:
            row = await conn.fetchrow(
                """
                SELECT last_cursor
                FROM backfill_state
                WHERE customer_id   = $1
                  AND source_system = $2
                  AND status        = $3
                  AND started_at    = $4
                FOR UPDATE
                """,
                customer_id,
                source.value,
                BackfillStatus.RUNNING.value,
                claim_token,
            )
            if row is not None:
                pending = _slack_deferred_channels(row["last_cursor"])
                if pending:
                    data = _decode_json_object(cursor)
                    data["pending_channels"] = {
                        **_slack_deferred_channels(cursor),
                        **pending,
                    }
                    cursor = json.dumps(data, sort_keys=True)

        tag = await conn.execute(
            """
            UPDATE backfill_state
            SET last_cursor      = $1,
                events_enqueued  = $2,
                last_progress_at = NOW(),
                heartbeat_at     = NOW()
            WHERE customer_id   = $3
              AND source_system = $4
              AND status        = $5
              AND started_at    = $6
            """,
            cursor,
            enqueued,
            customer_id,
            source.value,
            BackfillStatus.RUNNING.value,
            claim_token,
        )
    if _affected(tag) == 0:
        raise BackfillReclaimedError(
            f"progress write preempted: {customer_id}/{source.value}"
        )


async def _mark_done(
    customer_id: str,
    source: SourceSystem,
    enqueued: int,
    cursor: str | None,
    claim_token: datetime,
) -> None:
    """Mark complete only if the row is still ours. No-op silently otherwise."""
    async with raw_conn() as conn, conn.transaction():
        if source == SourceSystem.SLACK:
            row = await conn.fetchrow(
                """
                SELECT last_cursor
                FROM backfill_state
                WHERE customer_id   = $1
                  AND source_system = $2
                  AND status        = $3
                  AND started_at    = $4
                FOR UPDATE
                """,
                customer_id,
                source.value,
                BackfillStatus.RUNNING.value,
                claim_token,
            )
            pending = _slack_deferred_channels(row["last_cursor"] if row else None)
            if pending:
                await conn.execute(
                    """
                    UPDATE backfill_state
                    SET status           = $1,
                        last_cursor      = $2,
                        events_enqueued  = 0,
                        last_progress_at = NOW(),
                        heartbeat_at     = NULL,
                        started_at       = NULL,
                        completed_at     = NULL
                    WHERE customer_id   = $3
                      AND source_system = $4
                      AND status        = $5
                      AND started_at    = $6
                    """,
                    BackfillStatus.PENDING.value,
                    _slack_channel_cursor(pending),
                    customer_id,
                    source.value,
                    BackfillStatus.RUNNING.value,
                    claim_token,
                )
                return

        await conn.execute(
            """
            UPDATE backfill_state
            SET status           = $1,
                last_cursor      = $2,
                events_enqueued  = $3,
                last_progress_at = NOW(),
                heartbeat_at     = NOW(),
                completed_at     = NOW()
            WHERE customer_id   = $4
              AND source_system = $5
              AND status        = $6
              AND started_at    = $7
            """,
            BackfillStatus.COMPLETE.value,
            cursor,
            enqueued,
            customer_id,
            source.value,
            BackfillStatus.RUNNING.value,
            claim_token,
        )


async def _mark_failed(
    customer_id: str,
    source: SourceSystem,
    error: str,
    *,
    exc: Exception | None = None,
    claim_token: datetime,
) -> None:
    """Mark a backfill_state row as failed, gated on it still being ours.

    The backfill_state UPDATE filters on (status='running' AND started_at =
    claim_token). If the row was reclaimed and another worker has re-claimed
    it, we no-op silently — the new owner's run is independent of our error.

    When `exc` is a `PermanentSourceError` carrying a 401/403 status AND the
    backfill_state UPDATE actually fired, we ALSO flip the active
    integration_tokens row to status='auth_failed'. Sequencing these together
    surfaces "Reconnect" in the dashboard for revoked Granola keys without
    flipping on transient errors or on rows we don't own anymore.
    """
    # PermanentSourceError stores kwargs in `self.context`, not as attributes
    # (see shared/exceptions.PrbeError). Granola raises with status=401/403.
    is_auth_failure = False
    if isinstance(exc, PermanentSourceError):
        status_val = exc.context.get("status", 0) if exc.context else 0
        is_auth_failure = status_val in {401, 403}

    async with raw_conn() as conn, conn.transaction():
        tag = await conn.execute(
            """
                UPDATE backfill_state
                SET status       = $1,
                    last_error   = $2,
                    heartbeat_at = NOW()
                WHERE customer_id   = $3
                  AND source_system = $4
                  AND status        = $5
                  AND started_at    = $6
                """,
            BackfillStatus.FAILED.value,
            error[:1000],
            customer_id,
            source.value,
            BackfillStatus.RUNNING.value,
            claim_token,
        )
        # Token flip is gated on backfill_state actually flipping. If we
        # were preempted, the row belongs to a new claim; their fate is
        # not ours to decide.
        if is_auth_failure and _affected(tag) > 0:
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


async def _load_resume_state(
    customer_id: str, source: SourceSystem
) -> _ResumeState | None:
    """Load the cursor + cumulative event count + current ownership token.

    events_enqueued is cumulative across resumes — the run loop initializes
    its local counter from this value so that progress writes after a reclaim
    or restart preserve the running total instead of clobbering it with a
    fresh-from-zero count.

    Returns None if no row exists (caller should treat as fresh state).
    """
    async with raw_conn() as conn:
        row = await conn.fetchrow(
            """
            SELECT last_cursor, events_enqueued, started_at
              FROM backfill_state
             WHERE customer_id = $1 AND source_system = $2
            """,
            customer_id,
            source.value,
        )
    if row is None:
        return None
    return _ResumeState(
        cursor=row["last_cursor"],
        events_enqueued=row["events_enqueued"],
        started_at=row["started_at"],
    )


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
