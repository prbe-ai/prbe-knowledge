"""Tests for `_mark_failed` token flip + `run_backfill` disconnect-race abort.

Covers Lane A of the Granola integration plan:

  - `_mark_failed(..., exc=PermanentSourceError(status=401|403))` flips the
    matching `integration_tokens` row to `status='auth_failed'`.
  - `_mark_failed(..., exc=PermanentSourceError(status=400|404|...))` does NOT
    flip — the error is permanent but not auth-related.
  - `_mark_failed(..., exc=TransientSourceError(...))` does NOT flip — Granola
    503s should leave the token alone for the poller's next tick.
  - `_mark_failed(..., exc=None)` does NOT flip.
  - `_mark_failed` against a (customer, source) with no token row succeeds
    silently (UPDATE matches 0 rows).
  - `_mark_failed` against an already-`auth_failed` token does not double-flip
    (the `WHERE status='active'` filter matches 0 rows).
  - `run_backfill` aborts when the token row is deleted mid-pull (disconnect
    race): no further `ingestion_queue` rows after the disconnect commits.
  - `run_backfill` happy path: completes when the token stays active.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest

from services.ingestion.backfill_runner import (
    _mark_failed,
    enqueue_backfill,
    run_backfill,
)
from services.ingestion.connectedness import is_source_connected
from services.ingestion.handlers.base import Connector, ConnectorContext
from shared.config import Settings, get_settings
from shared.constants import BackfillStatus, SourceSystem
from shared.db import raw_conn
from shared.embeddings import reset_embedder
from shared.encryption import encrypt_token
from shared.exceptions import (
    PermanentSourceError,
    TransientSourceError,
)
from shared.models import IntegrationToken, WebhookEvent
from shared.storage import reset_store

# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch(monkeypatch, settings: Settings):
    monkeypatch.setenv(
        "TOKEN_ENCRYPTION_KEY", settings.token_encryption_key.get_secret_value()
    )
    monkeypatch.setenv("ENVIRONMENT", "local")
    reset_embedder()
    reset_store()
    get_settings.cache_clear()  # type: ignore[attr-defined]


CUSTOMER_ID = "cust-mark-failed"
SOURCE = SourceSystem.GRANOLA


async def _seed_customer() -> None:
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers (customer_id, display_name, api_key_hash) "
            "VALUES ($1, 'mf-test', 'mf-hash') ON CONFLICT DO NOTHING",
            CUSTOMER_ID,
        )


async def _seed_token(status: str = "active") -> None:
    """Seed a single (customer, source) integration_tokens row at the given status."""
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO integration_tokens
                (customer_id, source_system, access_token_encrypted, status)
            VALUES ($1, $2, $3, $4)
            """,
            CUSTOMER_ID,
            SOURCE.value,
            encrypt_token("grn_test_TOKEN"),
            status,
        )


async def _seed_backfill_state(status: str = "running"):
    """Seed a backfill_state row in the desired starting state.

    Returns the started_at the row was claimed at — tests use this as the
    claim ownership token when calling _mark_failed directly.
    """
    async with raw_conn() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO backfill_state
                (customer_id, source_system, status, events_enqueued, started_at)
            VALUES ($1, $2, $3, 0, NOW())
            ON CONFLICT (customer_id, source_system)
            DO UPDATE SET status = EXCLUDED.status,
                          events_enqueued = 0,
                          last_error = NULL,
                          started_at = NOW()
            RETURNING started_at
            """,
            CUSTOMER_ID,
            SOURCE.value,
            status,
        )
    return row["started_at"]


async def _backfill_state_row():
    async with raw_conn() as conn:
        return await conn.fetchrow(
            "SELECT status, last_error FROM backfill_state "
            "WHERE customer_id=$1 AND source_system=$2",
            CUSTOMER_ID,
            SOURCE.value,
        )


async def _token_row():
    async with raw_conn() as conn:
        return await conn.fetchrow(
            "SELECT status, last_refresh_error FROM integration_tokens "
            "WHERE customer_id=$1 AND source_system=$2",
            CUSTOMER_ID,
            SOURCE.value,
        )


# ---------------------------------------------------------------------------
# _mark_failed: token-flip gating
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mark_failed_permanent_auth_flips_token(live_db) -> None:
    """PermanentSourceError with status=401 flips integration_tokens.status."""
    await _seed_customer()
    await _seed_token(status="active")
    claim_token = await _seed_backfill_state(status="running")

    err = "granola auth failure: 401"
    exc = PermanentSourceError(err, url="https://api.granola.so/v1/notes", status=401)
    await _mark_failed(CUSTOMER_ID, SOURCE, str(exc), exc=exc, claim_token=claim_token)

    bf = await _backfill_state_row()
    assert bf is not None
    assert bf["status"] == BackfillStatus.FAILED.value
    assert bf["last_error"] is not None and "401" in bf["last_error"]

    tk = await _token_row()
    assert tk is not None
    assert tk["status"] == "auth_failed"
    assert tk["last_refresh_error"] is not None and "401" in tk["last_refresh_error"]


@pytest.mark.asyncio
async def test_mark_failed_permanent_403_flips_token(live_db) -> None:
    """403 is also an auth failure (Granola returns it for revoked keys)."""
    await _seed_customer()
    await _seed_token(status="active")
    claim_token = await _seed_backfill_state(status="running")

    exc = PermanentSourceError("granola 403", status=403)
    await _mark_failed(CUSTOMER_ID, SOURCE, str(exc), exc=exc, claim_token=claim_token)

    tk = await _token_row()
    assert tk is not None
    assert tk["status"] == "auth_failed"


@pytest.mark.asyncio
async def test_mark_failed_permanent_non_auth_no_flip(live_db) -> None:
    """PermanentSourceError with non-401/403 status flips backfill_state but not token."""
    await _seed_customer()
    await _seed_token(status="active")
    claim_token = await _seed_backfill_state(status="running")

    exc = PermanentSourceError("granola 404 missing", status=404)
    await _mark_failed(CUSTOMER_ID, SOURCE, str(exc), exc=exc, claim_token=claim_token)

    bf = await _backfill_state_row()
    assert bf is not None
    assert bf["status"] == BackfillStatus.FAILED.value

    tk = await _token_row()
    assert tk is not None
    assert tk["status"] == "active"
    assert tk["last_refresh_error"] is None


@pytest.mark.asyncio
async def test_mark_failed_transient_no_flip(live_db) -> None:
    """TransientSourceError (e.g. Granola 503) must not flip the token."""
    await _seed_customer()
    await _seed_token(status="active")
    claim_token = await _seed_backfill_state(status="running")

    exc = TransientSourceError("granola 5xx: 503", status=503)
    await _mark_failed(CUSTOMER_ID, SOURCE, str(exc), exc=exc, claim_token=claim_token)

    bf = await _backfill_state_row()
    assert bf is not None
    assert bf["status"] == BackfillStatus.FAILED.value

    tk = await _token_row()
    assert tk is not None
    assert tk["status"] == "active"


@pytest.mark.asyncio
async def test_mark_failed_no_exc_no_flip(live_db) -> None:
    """`exc=None` (e.g. the no-active-token branch) must not flip the token."""
    await _seed_customer()
    await _seed_token(status="active")
    claim_token = await _seed_backfill_state(status="running")

    await _mark_failed(
        CUSTOMER_ID, SOURCE, "some non-exception error string", claim_token=claim_token
    )

    bf = await _backfill_state_row()
    assert bf is not None
    assert bf["status"] == BackfillStatus.FAILED.value

    tk = await _token_row()
    assert tk is not None
    assert tk["status"] == "active"


@pytest.mark.asyncio
async def test_mark_failed_no_token_row_silent(live_db) -> None:
    """No integration_tokens row exists (disconnect already happened); UPDATE
    matches 0 rows and `_mark_failed` returns without raising."""
    await _seed_customer()
    # Intentionally NO token seeded.
    claim_token = await _seed_backfill_state(status="running")

    exc = PermanentSourceError("granola 401", status=401)
    # Should not raise.
    await _mark_failed(CUSTOMER_ID, SOURCE, str(exc), exc=exc, claim_token=claim_token)

    bf = await _backfill_state_row()
    assert bf is not None
    assert bf["status"] == BackfillStatus.FAILED.value

    tk = await _token_row()
    assert tk is None


@pytest.mark.asyncio
async def test_mark_failed_token_already_flipped(live_db) -> None:
    """Token already auth_failed: `WHERE status='active'` matches 0 rows;
    last_refresh_error stays whatever it was set to first."""
    await _seed_customer()
    await _seed_token(status="active")
    claim_token = await _seed_backfill_state(status="running")

    exc = PermanentSourceError("first failure 401", status=401)
    await _mark_failed(CUSTOMER_ID, SOURCE, str(exc), exc=exc, claim_token=claim_token)

    tk_first = await _token_row()
    assert tk_first is not None
    assert tk_first["status"] == "auth_failed"
    first_err = tk_first["last_refresh_error"]
    assert first_err is not None

    # Second permanent-auth failure with a different message — should NOT
    # double-flip and SHOULD NOT overwrite last_refresh_error (since the
    # WHERE status='active' filter excludes the row).
    # Re-seed (gives a fresh claim) so the backfill_state UPDATE matches.
    claim_token = await _seed_backfill_state(status="running")
    exc2 = PermanentSourceError("second failure 401", status=401)
    await _mark_failed(CUSTOMER_ID, SOURCE, str(exc2), exc=exc2, claim_token=claim_token)

    tk_after = await _token_row()
    assert tk_after is not None
    assert tk_after["status"] == "auth_failed"
    assert tk_after["last_refresh_error"] == first_err


# ---------------------------------------------------------------------------
# is_source_connected helper (used by run_backfill mid-loop and _enqueue)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_is_source_connected_returns_false_when_deleted(live_db) -> None:
    await _seed_customer()
    await _seed_token(status="active")
    assert await is_source_connected(CUSTOMER_ID, SOURCE) is True

    async with raw_conn() as conn:
        await conn.execute(
            "DELETE FROM integration_tokens WHERE customer_id=$1 AND source_system=$2",
            CUSTOMER_ID,
            SOURCE.value,
        )
    assert await is_source_connected(CUSTOMER_ID, SOURCE) is False


@pytest.mark.asyncio
async def test_is_source_connected_returns_false_when_auth_failed(live_db) -> None:
    await _seed_customer()
    await _seed_token(status="auth_failed")
    assert await is_source_connected(CUSTOMER_ID, SOURCE) is False


# ---------------------------------------------------------------------------
# run_backfill: disconnect-race abort + happy path
# ---------------------------------------------------------------------------


def _evt(idx: int) -> WebhookEvent:
    """Tiny synthetic event for the fake connector."""
    return WebhookEvent(
        customer_id=CUSTOMER_ID,
        source_system=SOURCE,
        source_event_id=f"fake-{idx}",
        received_at=datetime.now(UTC),
        payload_s3_key="raw/granola/fake/key.json",
        raw_payload={"idx": idx},
    )


class _FakeConnector(Connector):
    """Minimal Connector that yields N synthetic events.

    Optionally invokes `between_events` (an async coroutine) once, the moment
    we yield event index `disconnect_at`. Tests use that hook to simulate a
    concurrent disconnect mid-iteration.
    """

    source_system = SOURCE

    def __init__(self, ctx: ConnectorContext, total: int = 200, *, disconnect_at: int | None = None, on_disconnect=None) -> None:
        super().__init__(ctx)
        self._total = total
        self._disconnect_at = disconnect_at
        self._on_disconnect = on_disconnect
        self.yielded = 0

    # The abstract API surface is large; tests only exercise backfill().
    def verify_signature(self, headers, raw_body):  # pragma: no cover
        return True

    def parse_webhook_event(self, customer_id, headers, raw_payload):  # pragma: no cover
        return None

    async def normalize(self, event, hydrated):  # pragma: no cover
        raise NotImplementedError

    async def backfill(  # type: ignore[override]
        self,
        customer_id: str,
        token: IntegrationToken,
        cursor: str | None = None,
    ) -> AsyncIterator[WebhookEvent]:
        for i in range(self._total):
            if self._disconnect_at is not None and i == self._disconnect_at and self._on_disconnect:
                await self._on_disconnect()
            self.yielded = i + 1
            yield _evt(i)


def _ctx() -> ConnectorContext:
    import httpx

    return ConnectorContext(settings=Settings(environment="local"), http=httpx.AsyncClient())


async def _delete_token() -> None:
    async with raw_conn() as conn:
        await conn.execute(
            "DELETE FROM integration_tokens WHERE customer_id=$1 AND source_system=$2",
            CUSTOMER_ID,
            SOURCE.value,
        )


async def _ingestion_queue_count() -> int:
    async with raw_conn() as conn:
        return await conn.fetchval(
            "SELECT count(*) FROM ingestion_queue WHERE customer_id=$1 AND source_system=$2",
            CUSTOMER_ID,
            SOURCE.value,
        )


@pytest.mark.asyncio
async def test_run_backfill_aborts_when_token_deleted(live_db, monkeypatch) -> None:
    """If the token row is deleted mid-iteration, `run_backfill` must bail
    on the next event without writing it.

    Post-batching semantics: the connectedness check fires on every event
    (immediate abort). Already-flushed batches stay; the in-flight batch is
    intentionally DROPPED so a disconnected source never gets new rows. So
    queue_count = floor(disconnect_at / batch_size) * batch_size.
    """
    from shared.config import get_settings as _get_settings

    await _seed_customer()
    await _seed_token(status="active")
    await enqueue_backfill(CUSTOMER_ID, SOURCE)

    batch_size = _get_settings().backfill_batch_size  # default 100
    # Disconnect after 2 full batches plus 50 in-flight: 2 batches land,
    # the trailing 50 are dropped on disconnect.
    disconnect_at = batch_size * 2 + 50
    expected_landed = (disconnect_at // batch_size) * batch_size
    fake = _FakeConnector(
        _ctx(),
        total=disconnect_at + 100,  # headroom; abort stops us before total
        disconnect_at=disconnect_at,
        on_disconnect=_delete_token,
    )

    monkeypatch.setattr(
        "services.ingestion.backfill_runner.build_connector", lambda src, ctx: fake
    )

    enqueued = await run_backfill(_ctx(), CUSTOMER_ID, SOURCE)

    queue_count = await _ingestion_queue_count()
    assert queue_count == expected_landed, (
        f"expected {expected_landed} queue rows (2 full batches) before "
        f"disconnect-abort dropped the in-flight batch, got {queue_count}"
    )
    assert enqueued == expected_landed

    # backfill_state was NOT marked done (we returned early); status still 'running'.
    bf = await _backfill_state_row()
    assert bf is not None
    assert bf["status"] == BackfillStatus.RUNNING.value


@pytest.mark.asyncio
async def test_run_backfill_continues_when_token_active(live_db, monkeypatch) -> None:
    """Happy path regression: token stays active → all events get written and
    `backfill_state.status` flips to 'complete'."""
    await _seed_customer()
    await _seed_token(status="active")
    await enqueue_backfill(CUSTOMER_ID, SOURCE)

    total = 55
    fake = _FakeConnector(_ctx(), total=total)

    monkeypatch.setattr(
        "services.ingestion.backfill_runner.build_connector", lambda src, ctx: fake
    )

    enqueued = await run_backfill(_ctx(), CUSTOMER_ID, SOURCE)

    assert enqueued == total
    queue_count = await _ingestion_queue_count()
    assert queue_count == total

    bf = await _backfill_state_row()
    assert bf is not None
    assert bf["status"] == BackfillStatus.COMPLETE.value


# ---------------------------------------------------------------------------
# run_backfill: cursor-only `_checkpoint` event (Granola end-of-pagination)
# ---------------------------------------------------------------------------


class _CheckpointConnector(Connector):
    """Yields one normal event then one cursor-only `_checkpoint` event.

    Mirrors the shape Granola's connector emits at the end of a clean run.
    The runner must persist the checkpoint cursor without enqueueing it.
    """

    source_system = SOURCE

    def __init__(
        self,
        ctx: ConnectorContext,
        *,
        normal_cursor: str,
        checkpoint_cursor: str,
    ) -> None:
        super().__init__(ctx)
        self._normal_cursor = normal_cursor
        self._checkpoint_cursor = checkpoint_cursor

    def verify_signature(self, headers, raw_body):  # pragma: no cover
        return True

    def parse_webhook_event(self, customer_id, headers, raw_payload):  # pragma: no cover
        return None

    async def normalize(self, event, hydrated):  # pragma: no cover
        raise NotImplementedError

    async def backfill(  # type: ignore[override]
        self,
        customer_id: str,
        token: IntegrationToken,
        cursor: str | None = None,
    ) -> AsyncIterator[WebhookEvent]:
        # Normal event with the input watermark.
        yield WebhookEvent(
            customer_id=customer_id,
            source_system=SOURCE,
            source_event_id="real-1",
            received_at=datetime.now(UTC),
            payload_s3_key="",
            raw_payload={"note": {"id": "real-1"}, "_cursor": self._normal_cursor},
            headers={},
        )
        # Cursor-only checkpoint event.
        yield WebhookEvent(
            customer_id=customer_id,
            source_system=SOURCE,
            source_event_id="__cursor_checkpoint__",
            received_at=datetime.now(UTC),
            payload_s3_key="",
            raw_payload={
                "_cursor": self._checkpoint_cursor,
                "_checkpoint": True,
            },
            headers={},
        )


async def _backfill_state_cursor() -> str | None:
    async with raw_conn() as conn:
        return await conn.fetchval(
            "SELECT last_cursor FROM backfill_state "
            "WHERE customer_id=$1 AND source_system=$2",
            CUSTOMER_ID,
            SOURCE.value,
        )


@pytest.mark.asyncio
async def test_runner_checkpoint_event_persists_cursor_without_enqueue(
    live_db, monkeypatch
) -> None:
    """`_checkpoint` events update last_cursor but don't add a queue row.

    Regression for the Granola watermark bug: the connector now defers the
    watermark advance to a final `_checkpoint=True` event that the runner
    must recognize as cursor-only (no R2 put, no ingestion_queue insert).
    """
    import json as _json

    await _seed_customer()
    await _seed_token(status="active")
    await enqueue_backfill(CUSTOMER_ID, SOURCE)

    normal_cursor = _json.dumps({"watermark": None, "page_cursor": None})
    checkpoint_cursor = _json.dumps(
        {"watermark": "2026-04-27T00:00:00Z", "page_cursor": None}
    )

    fake = _CheckpointConnector(
        _ctx(),
        normal_cursor=normal_cursor,
        checkpoint_cursor=checkpoint_cursor,
    )
    monkeypatch.setattr(
        "services.ingestion.backfill_runner.build_connector", lambda src, ctx: fake
    )

    enqueued = await run_backfill(_ctx(), CUSTOMER_ID, SOURCE)

    # Only the normal event made it to ingestion_queue. The checkpoint did
    # NOT get enqueued, but its cursor IS persisted.
    assert enqueued == 1
    queue_count = await _ingestion_queue_count()
    assert queue_count == 1

    persisted = await _backfill_state_cursor()
    assert persisted == checkpoint_cursor

    bf = await _backfill_state_row()
    assert bf is not None
    assert bf["status"] == BackfillStatus.COMPLETE.value
