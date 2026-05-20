"""IntegrationPoller tests.

Ports the SQL-predicate cases from the retired test_granola_scheduler.py
(verifying _fetch_due_customers against a live Postgres) and adds new
cases covering the registry-discovery layer + per-source tick behavior.
"""

from __future__ import annotations

from collections.abc import Mapping
from unittest.mock import AsyncMock, patch

import pytest

from services.ingestion.handlers.base import PollConfig
from services.ingestion.poller import IntegrationPoller
from shared.constants import (
    BackfillStatus,
    IntegrationStatus,
    SourceSystem,
)
from shared.db import raw_conn

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _granola_config(interval_seconds: int = 300) -> PollConfig:
    return PollConfig(
        interval_seconds=interval_seconds,
        eligible_statuses=(BackfillStatus.COMPLETE, BackfillStatus.FAILED),
        notify_channel="granola_refresh",
    )


async def _seed_customer(customer_id: str) -> None:
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO customers (customer_id, display_name, api_key_hash, status)
            VALUES ($1, $2, $3, 'active')
            """,
            customer_id,
            customer_id,
            f"hash-{customer_id}",
        )


async def _seed_integration(
    customer_id: str,
    *,
    source: SourceSystem = SourceSystem.GRANOLA,
    token_status: str = IntegrationStatus.ACTIVE.value,
    bf_status: str = BackfillStatus.COMPLETE.value,
    last_progress_minutes_ago: float | None = 10.0,
) -> None:
    """Insert an integration_tokens row + matching backfill_state row."""
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO integration_tokens
                (customer_id, source_system, access_token_encrypted,
                 scope, status)
            VALUES ($1, $2, $3, 'tier:enterprise', $4)
            """,
            customer_id,
            source.value,
            "encrypted-stub",
            token_status,
        )
        if last_progress_minutes_ago is None:
            await conn.execute(
                """
                INSERT INTO backfill_state
                    (customer_id, source_system, status, events_enqueued,
                     last_progress_at)
                VALUES ($1, $2, $3, 0, NULL)
                """,
                customer_id,
                source.value,
                bf_status,
            )
        else:
            await conn.execute(
                """
                INSERT INTO backfill_state
                    (customer_id, source_system, status, events_enqueued,
                     last_progress_at)
                VALUES ($1, $2, $3, 0,
                        NOW() - make_interval(secs => $4))
                """,
                customer_id,
                source.value,
                bf_status,
                last_progress_minutes_ago * 60,
            )


# ---------------------------------------------------------------------------
# _discover() — registry-walk tests (no DB)
# ---------------------------------------------------------------------------


def test_discover_empty_when_no_connectors_have_poll_config() -> None:
    """If list_registered() returns nothing, _discover returns an empty dict."""
    with patch("services.ingestion.poller.list_registered", return_value=[]):
        assert IntegrationPoller._discover() == {}


def test_discover_skips_connectors_with_none_poll_config() -> None:
    """Webhook-only connectors (poll_config is None, the default) are excluded."""

    class _NullCfg:
        poll_config = None

    with (
        patch(
            "services.ingestion.poller.list_registered",
            return_value=[SourceSystem.SLACK],
        ),
        patch(
            "services.ingestion.poller.get_connector_class",
            return_value=_NullCfg,
        ),
    ):
        assert IntegrationPoller._discover() == {}


def test_discover_includes_connectors_with_set_poll_config() -> None:
    """Connectors with a non-None poll_config end up in the returned dict."""
    cfg = _granola_config()

    class _GranolaLike:
        poll_config = cfg

    with (
        patch(
            "services.ingestion.poller.list_registered",
            return_value=[SourceSystem.GRANOLA],
        ),
        patch(
            "services.ingestion.poller.get_connector_class",
            return_value=_GranolaLike,
        ),
    ):
        assert IntegrationPoller._discover() == {SourceSystem.GRANOLA: cfg}


def test_discover_picks_up_real_granola_connector() -> None:
    """Smoke: the real GranolaConnector subclass exposes poll_config so the
    registry-walk discovers it without any explicit wiring."""
    # Import side effect populates the connector registry.
    import services.ingestion.handlers  # noqa: F401

    discovered = IntegrationPoller._discover()
    assert SourceSystem.GRANOLA in discovered
    cfg = discovered[SourceSystem.GRANOLA]
    assert cfg.notify_channel == "granola_refresh"
    assert BackfillStatus.FAILED in cfg.eligible_statuses


# ---------------------------------------------------------------------------
# _tick_source() — mocked DB so we can drive every branch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tick_source_empty_customers_is_noop() -> None:
    poller = IntegrationPoller(configs={SourceSystem.GRANOLA: _granola_config()})
    with (
        patch.object(poller, "_fetch_due_customers", AsyncMock(return_value=[])),
        patch(
            "services.ingestion.poller.re_enqueue_for_polling",
            new=AsyncMock(),
        ) as mock_re,
        patch("services.ingestion.poller.get_pool") as mock_pool,
    ):
        await poller._tick_source(SourceSystem.GRANOLA, _granola_config())
        mock_re.assert_not_awaited()
        mock_pool.assert_not_called()


@pytest.mark.asyncio
async def test_tick_source_swallows_fetch_failure() -> None:
    """A DB blip on the SELECT logs+returns — the surrounding run() loop continues."""
    poller = IntegrationPoller(configs={SourceSystem.GRANOLA: _granola_config()})
    with patch.object(
        poller,
        "_fetch_due_customers",
        AsyncMock(side_effect=RuntimeError("db down")),
    ):
        # Must not raise.
        await poller._tick_source(SourceSystem.GRANOLA, _granola_config())


@pytest.mark.asyncio
async def test_tick_source_skips_notify_when_re_enqueue_returns_false() -> None:
    """re_enqueue_for_polling returns False when the row is already pending/running."""
    poller = IntegrationPoller(configs={SourceSystem.GRANOLA: _granola_config()})

    notify_calls: list[tuple[str, str]] = []

    class _FakeConn:
        async def execute(self, *args: object) -> None:
            notify_calls.append((args[1], args[2]))  # type: ignore[arg-type]

    class _FakePool:
        def acquire(self):
            class _Ctx:
                async def __aenter__(self_inner) -> _FakeConn:
                    return _FakeConn()

                async def __aexit__(self_inner, *_a: object) -> None:
                    return None

            return _Ctx()

    with (
        patch.object(poller, "_fetch_due_customers", AsyncMock(return_value=["c1"])),
        patch(
            "services.ingestion.poller.re_enqueue_for_polling",
            new=AsyncMock(return_value=False),
        ),
        patch("services.ingestion.poller.get_pool", return_value=_FakePool()),
    ):
        await poller._tick_source(SourceSystem.GRANOLA, _granola_config())

    assert notify_calls == []


@pytest.mark.asyncio
async def test_tick_source_notifies_on_successful_re_enqueue() -> None:
    """The happy path: re_enqueue returns True → NOTIFY fires on the configured channel."""
    poller = IntegrationPoller(configs={SourceSystem.GRANOLA: _granola_config()})

    notify_calls: list[tuple[str, str]] = []

    class _FakeConn:
        async def execute(self, *args: object) -> None:
            # args[0] is the SQL, args[1] = channel, args[2] = customer.
            notify_calls.append((args[1], args[2]))  # type: ignore[arg-type]

    class _FakePool:
        def acquire(self):
            class _Ctx:
                async def __aenter__(self_inner) -> _FakeConn:
                    return _FakeConn()

                async def __aexit__(self_inner, *_a: object) -> None:
                    return None

            return _Ctx()

    with (
        patch.object(poller, "_fetch_due_customers", AsyncMock(return_value=["c1", "c2"])),
        patch(
            "services.ingestion.poller.re_enqueue_for_polling",
            new=AsyncMock(return_value=True),
        ),
        patch("services.ingestion.poller.get_pool", return_value=_FakePool()),
    ):
        await poller._tick_source(SourceSystem.GRANOLA, _granola_config())

    assert notify_calls == [
        ("granola_refresh", "c1"),
        ("granola_refresh", "c2"),
    ]


@pytest.mark.asyncio
async def test_tick_source_continues_past_per_customer_error() -> None:
    """re_enqueue raising for c1 must NOT block re_enqueue for c2."""
    poller = IntegrationPoller(configs={SourceSystem.GRANOLA: _granola_config()})

    re_enqueue_mock = AsyncMock(side_effect=[RuntimeError("boom"), True])
    notify_calls: list[tuple[str, str]] = []

    class _FakeConn:
        async def execute(self, *args: object) -> None:
            notify_calls.append((args[1], args[2]))  # type: ignore[arg-type]

    class _FakePool:
        def acquire(self):
            class _Ctx:
                async def __aenter__(self_inner) -> _FakeConn:
                    return _FakeConn()

                async def __aexit__(self_inner, *_a: object) -> None:
                    return None

            return _Ctx()

    with (
        patch.object(poller, "_fetch_due_customers", AsyncMock(return_value=["c1", "c2"])),
        patch(
            "services.ingestion.poller.re_enqueue_for_polling",
            new=re_enqueue_mock,
        ),
        patch("services.ingestion.poller.get_pool", return_value=_FakePool()),
    ):
        await poller._tick_source(SourceSystem.GRANOLA, _granola_config())

    # c2 still notified.
    assert notify_calls == [("granola_refresh", "c2")]


# ---------------------------------------------------------------------------
# _fetch_due_customers — live-DB SQL coverage (ports test_granola_scheduler.py)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_picks_up_completed_stale_active(live_db) -> None:
    await _seed_customer("cust-stale")
    await _seed_integration("cust-stale", last_progress_minutes_ago=10.0)

    poller = IntegrationPoller(configs={SourceSystem.GRANOLA: _granola_config()})
    due = await poller._fetch_due_customers(SourceSystem.GRANOLA, _granola_config())

    assert due == ["cust-stale"]


@pytest.mark.asyncio
async def test_fetch_skips_recently_progressed(live_db) -> None:
    await _seed_customer("cust-fresh")
    await _seed_integration("cust-fresh", last_progress_minutes_ago=1.0)

    poller = IntegrationPoller(configs={SourceSystem.GRANOLA: _granola_config()})
    due = await poller._fetch_due_customers(SourceSystem.GRANOLA, _granola_config())

    assert due == []


@pytest.mark.asyncio
async def test_fetch_skips_pending_and_running(live_db) -> None:
    await _seed_customer("cust-pending")
    await _seed_customer("cust-running")
    await _seed_integration(
        "cust-pending",
        bf_status=BackfillStatus.PENDING.value,
        last_progress_minutes_ago=10.0,
    )
    await _seed_integration(
        "cust-running",
        bf_status=BackfillStatus.RUNNING.value,
        last_progress_minutes_ago=10.0,
    )

    poller = IntegrationPoller(configs={SourceSystem.GRANOLA: _granola_config()})
    due = await poller._fetch_due_customers(SourceSystem.GRANOLA, _granola_config())

    assert due == []


@pytest.mark.asyncio
async def test_fetch_includes_failed_for_retry(live_db) -> None:
    """FAILED is in granola's eligible_statuses — auto-retry on next tick."""
    await _seed_customer("cust-failed")
    await _seed_integration(
        "cust-failed",
        bf_status=BackfillStatus.FAILED.value,
        last_progress_minutes_ago=10.0,
    )

    poller = IntegrationPoller(configs={SourceSystem.GRANOLA: _granola_config()})
    due = await poller._fetch_due_customers(SourceSystem.GRANOLA, _granola_config())

    assert due == ["cust-failed"]


@pytest.mark.asyncio
async def test_fetch_skips_revoked_tokens(live_db) -> None:
    await _seed_customer("cust-revoked")
    await _seed_integration(
        "cust-revoked",
        token_status=IntegrationStatus.REVOKED.value,
        last_progress_minutes_ago=10.0,
    )

    poller = IntegrationPoller(configs={SourceSystem.GRANOLA: _granola_config()})
    due = await poller._fetch_due_customers(SourceSystem.GRANOLA, _granola_config())

    assert due == []


@pytest.mark.asyncio
async def test_fetch_includes_never_progressed(live_db) -> None:
    """last_progress_at IS NULL — first poll, include it."""
    await _seed_customer("cust-null")
    await _seed_integration(
        "cust-null",
        bf_status=BackfillStatus.COMPLETE.value,
        last_progress_minutes_ago=None,
    )

    poller = IntegrationPoller(configs={SourceSystem.GRANOLA: _granola_config()})
    due = await poller._fetch_due_customers(SourceSystem.GRANOLA, _granola_config())

    assert due == ["cust-null"]


@pytest.mark.asyncio
async def test_fetch_orders_oldest_first(live_db) -> None:
    await _seed_customer("cust-old")
    await _seed_customer("cust-newer")
    await _seed_integration("cust-old", last_progress_minutes_ago=30.0)
    await _seed_integration("cust-newer", last_progress_minutes_ago=10.0)

    poller = IntegrationPoller(configs={SourceSystem.GRANOLA: _granola_config()})
    due = await poller._fetch_due_customers(SourceSystem.GRANOLA, _granola_config())

    assert due == ["cust-old", "cust-newer"]


@pytest.mark.asyncio
async def test_fetch_respects_eligible_statuses_subset(live_db) -> None:
    """A config that doesn't include FAILED skips failed rows. Verifies the
    eligible_statuses parameter actually drives the SQL ANY() filter, not
    just the previously-hardcoded (COMPLETE, FAILED) pair."""
    await _seed_customer("cust-failed-skipped")
    await _seed_integration(
        "cust-failed-skipped",
        bf_status=BackfillStatus.FAILED.value,
        last_progress_minutes_ago=10.0,
    )

    complete_only = PollConfig(
        interval_seconds=300,
        eligible_statuses=(BackfillStatus.COMPLETE,),
        notify_channel="granola_refresh",
    )

    poller = IntegrationPoller(configs={SourceSystem.GRANOLA: complete_only})
    due = await poller._fetch_due_customers(SourceSystem.GRANOLA, complete_only)

    assert due == []


# ---------------------------------------------------------------------------
# run() lifecycle — drive with explicit configs + asyncio.Event
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_returns_immediately_when_no_configs() -> None:
    """Empty configs path: log + return, no infinite loop."""
    poller = IntegrationPoller(configs={})
    # If this hangs, pytest-asyncio's default 5-min timeout would catch it,
    # but we expect immediate return.
    await poller.run()


@pytest.mark.asyncio
async def test_run_shutdown_after_first_tick_exits_cleanly() -> None:
    """run() does a boot tick, sleeps, then sees shutdown → exits."""
    configs: Mapping[SourceSystem, PollConfig] = {
        SourceSystem.GRANOLA: _granola_config(interval_seconds=1),
    }
    poller = IntegrationPoller(configs=configs)

    tick_calls: list[SourceSystem] = []

    async def _stub_tick(source: SourceSystem, cfg: PollConfig) -> None:
        tick_calls.append(source)
        # After the boot tick fires, signal shutdown so run() exits at next sleep.
        if len(tick_calls) == 1:
            poller.shutdown()

    with patch.object(poller, "_tick_source", _stub_tick):
        await poller.run()

    # Boot tick fired at least once. May or may not loop once more depending
    # on timing; assert at least 1.
    assert tick_calls and tick_calls[0] == SourceSystem.GRANOLA
