"""Real legacy producer/purge barriers; only provider and object storage are fake."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
import pytest_asyncio

from engine.ingest import connectedness
from engine.ingest import purge as source_purge
from engine.shared.config import get_settings
from engine.shared.constants import SourceSystem
from engine.shared.db import raw_conn, with_tenant
from engine.shared.encryption import encrypt_token
from engine.shared.models import WebhookEvent
from kb import backfill_runner
from kb.github_control import adoption_lock, set_source_purge_gate

pytestmark = pytest.mark.integration
TENANT = "github-legacy-purge-barrier"
SOURCE = SourceSystem.GITHUB


class BarrierStore:
    def __init__(self, *, block_put=False, block_delete=False):
        self.objects = {}
        self.put_started = asyncio.Event()
        self.put_release = asyncio.Event()
        self.delete_started = asyncio.Event()
        self.delete_release = asyncio.Event()
        self.puts = 0
        self.sweeps = 0
        if not block_put:
            self.put_release.set()
        if not block_delete:
            self.delete_release.set()

    async def bucket_for(self, customer_id):
        return f"fixture-{customer_id}"

    async def ensure_bucket(self, bucket):
        pass

    async def put(self, bucket, key, payload):
        self.puts += 1
        self.put_started.set()
        await self.put_release.wait()
        self.objects[key] = payload

    async def delete(self, bucket, key):
        self.delete_started.set()
        await self.delete_release.wait()
        self.objects.pop(key, None)

    async def delete_prefix(self, bucket, prefix):
        self.sweeps += 1
        keys = [key for key in self.objects if key.startswith(prefix)]
        for key in keys:
            del self.objects[key]
        return len(keys), 0

    async def count_prefix(self, bucket, prefix):
        return sum(key.startswith(prefix) for key in self.objects)


class FixtureConnector:
    async def backfill(self, customer_id, token, cursor):
        for index in range(2):
            yield WebhookEvent(
                customer_id=customer_id,
                source_system=SOURCE,
                source_event_id=f"fixture-event-{index}",
                received_at=datetime(2026, 9, 7, tzinfo=UTC),
                raw_payload={"fixture": index},
            )


@pytest_asyncio.fixture
async def legacy_source(live_db, monkeypatch):
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers(customer_id,display_name,api_key_hash) VALUES ($1,$1,$1)",
            TENANT,
        )
        await conn.execute(
            """INSERT INTO integration_tokens
            (customer_id,source_system,access_token_encrypted,status)
            VALUES ($1,'github',$2,'active')""",
            TENANT, encrypt_token("synthetic-legacy-token"),
        )
        await conn.execute(
            """INSERT INTO backfill_state(customer_id,source_system,status)
            VALUES ($1,'github','pending')""",
            TENANT,
        )
    monkeypatch.setattr(get_settings(), "backfill_batch_size", 1)
    monkeypatch.setattr(backfill_runner, "build_connector", lambda *_: FixtureConnector())

    def install_store(store):
        monkeypatch.setattr(backfill_runner, "get_store", lambda: store)
        monkeypatch.setattr(source_purge, "get_store", lambda: store)

    return install_store


async def source_gate():
    async with with_tenant(TENANT) as conn:
        return await conn.fetchval(
            "SELECT purge_in_progress FROM github_source_gates WHERE customer_id=$1",
            TENANT,
        )


async def producer_status():
    async with with_tenant(TENANT) as conn:
        return await conn.fetchval(
            "SELECT status FROM backfill_state WHERE customer_id=$1 AND source_system='github'",
            TENANT,
        )


async def producer_claim():
    async with with_tenant(TENANT) as conn:
        return await conn.fetchval(
            "SELECT started_at FROM backfill_state WHERE customer_id=$1 AND source_system='github'",
            TENANT,
        )


async def wait_for_gate():
    async with asyncio.timeout(2):
        while not await source_gate():
            await asyncio.sleep(0.005)


async def assert_no_queue():
    async with with_tenant(TENANT) as conn:
        assert await conn.fetchval(
            "SELECT count(*) FROM ingestion_queue WHERE customer_id=$1", TENANT
        ) == 0


async def settle_tasks(store, *tasks):
    store.put_release.set()
    store.delete_release.set()
    try:
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 3)
    except TimeoutError:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def test_verified_purge_waits_for_active_flush_and_its_object_cleanup(legacy_source):
    store = BarrierStore(block_put=True, block_delete=True)
    legacy_source(store)
    producer = asyncio.create_task(backfill_runner.run_backfill(SimpleNamespace(), TENANT, SOURCE))
    purge = None
    try:
        await asyncio.wait_for(store.put_started.wait(), 2)
        assert await producer_status() == "running"
        purge = asyncio.create_task(source_purge.purge_source(TENANT, SOURCE, "fixture-purge"))
        await wait_for_gate()
        assert not purge.done() and store.sweeps == 0
        assert not await connectedness.is_source_connected(TENANT, SOURCE)

        # The write was already admitted before gate close. Cleanup must finish
        # before the producer acknowledgement can let the final sweep proceed.
        store.put_release.set()
        await asyncio.wait_for(store.delete_started.wait(), 2)
        assert store.objects and await producer_status() == "running"
        assert not purge.done() and store.sweeps == 0
        await assert_no_queue()
        store.delete_release.set()
        assert await asyncio.wait_for(producer, 2) == 0
        result = await asyncio.wait_for(purge, 2)
        assert result["verified"] is True
        assert result["residue"] == {} and result["r2_residue"] == 0
        assert store.puts == 1 and not store.objects and store.sweeps > 0
        await assert_no_queue()
        assert await producer_status() is None
        assert not await connectedness.is_source_connected(TENANT, SOURCE)

        # Even a stale direct flush after verified completion cannot resurrect
        # queue work or leave an object after its rejected write settles.
        accepted = await backfill_runner._flush_batch(
            store, "fixture", TENANT, SOURCE,
            [(f"raw/github/{TENANT}/late.json", b"{}", "late-event")],
            asyncio.Semaphore(1),
        )
        assert accepted is False and not store.objects
        await assert_no_queue()
    finally:
        await settle_tasks(store, producer, *([purge] if purge is not None else []))


async def test_stalled_flush_cannot_report_verified_and_retry_keeps_gate_closed(
    legacy_source, monkeypatch,
):
    store = BarrierStore(block_put=True)
    legacy_source(store)
    monkeypatch.setattr(source_purge, "_GITHUB_LEGACY_STOP_TIMEOUT_SECONDS", 0.02)
    producer = asyncio.create_task(backfill_runner.run_backfill(SimpleNamespace(), TENANT, SOURCE))
    try:
        await asyncio.wait_for(store.put_started.wait(), 2)
        with pytest.raises(RuntimeError, match="still stopping"):
            await source_purge.purge_source(TENANT, SOURCE, "fixture-timeout")
        assert await source_gate() is True
        assert await producer_status() == "running" and store.sweeps == 0
        assert not await connectedness.is_source_connected(TENANT, SOURCE)

        # The singleton token still exists at this point: the durable gate,
        # not token deletion, must stop this future batch from being admitted.
        future_store = BarrierStore()
        original_claim = await producer_claim()
        assert await backfill_runner._flush_batch(
            future_store, "fixture", TENANT, SOURCE,
            [(f"raw/github/{TENANT}/future.json", b"{}", "future-event")],
            asyncio.Semaphore(1),
            claim_token=original_claim,
        ) is False
        assert future_store.puts == 1 and not future_store.objects
        await assert_no_queue()
        store.put_release.set()
        assert await asyncio.wait_for(producer, 2) == 0
        assert await producer_status() == "failed"
        assert not store.objects
        result = await source_purge.purge_source(TENANT, SOURCE, "fixture-retry")
        assert result["verified"] is True and await source_gate() is False
        await assert_no_queue()
    finally:
        await settle_tasks(store, producer)


async def test_closed_gate_stops_future_runner_before_any_object_write(legacy_source):
    store = BarrierStore()
    legacy_source(store)
    async with with_tenant(TENANT) as conn:
        await adoption_lock(conn, TENANT)
        await set_source_purge_gate(conn, TENANT, True)
    assert not await connectedness.is_source_connected(TENANT, SOURCE)
    assert await backfill_runner.run_backfill(SimpleNamespace(), TENANT, SOURCE) == 0
    assert await producer_status() == "failed"
    assert store.puts == 0 and not store.objects
    await assert_no_queue()


async def test_delayed_old_flush_cannot_resurrect_after_verified_purge_and_reseed(legacy_source):
    store = BarrierStore(block_put=True)
    legacy_source(store)
    producer = asyncio.create_task(backfill_runner.run_backfill(SimpleNamespace(), TENANT, SOURCE))
    try:
        await asyncio.wait_for(store.put_started.wait(), 2)
        original_claim = await producer_claim()
        assert original_claim is not None
        # Model a reaper releasing a claim while the old provider operation is
        # still in flight. The purge's running poll cannot see this old task.
        async with with_tenant(TENANT) as conn:
            await conn.execute(
                """UPDATE backfill_state SET status='pending',started_at=NULL
                WHERE customer_id=$1 AND source_system='github'""", TENANT,
            )
        result = await source_purge.purge_source(TENANT, SOURCE, "fixture-before-reseed")
        assert result["verified"] is True and not producer.done()
        assert await source_gate() is False and await producer_status() is None
        assert not store.objects

        # Reconnection is legitimate after verified completion. A fresh token
        # plus open source gate must not authorize the old producer's bytes.
        async with with_tenant(TENANT) as conn:
            await conn.execute(
                """INSERT INTO integration_tokens
                (customer_id,source_system,access_token_encrypted,status)
                VALUES ($1,'github',$2,'active')""",
                TENANT, encrypt_token("synthetic-reconnected-token"),
            )
        await backfill_runner.enqueue_backfill(TENANT, SOURCE)
        new_claim = await backfill_runner._mark_running(TENANT, SOURCE)
        assert new_claim != original_claim
        assert await connectedness.is_source_connected(TENANT, SOURCE)
        store.put_release.set()
        enqueued = await asyncio.wait_for(producer, 2)
        await assert_no_queue()
        assert enqueued == 0 and not store.objects and store.puts == 1
        assert await producer_status() == "running"
        assert await producer_claim() == new_claim
    finally:
        await settle_tasks(store, producer)
