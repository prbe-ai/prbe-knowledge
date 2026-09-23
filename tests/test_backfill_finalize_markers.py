"""The one-time backfill re-attaches an end signal only with evidence of a mine.

Getting this wrong in the permissive direction is silent data loss: a session
marked ended without having been mined is never mined, because the sweep then
skips it. So every evidence condition gets a row that fails only it.
"""
from __future__ import annotations

import pytest

from engine.shared.db import get_pool
from engine.shared.session_signals import cron_marker_key
from engine.shared.storage import get_store
from scripts.backfill_finalize_markers import backfill

C = "backfill-evidence-cust"


def _batch(sid: str, day: str) -> str:
    return f"raw/claude_code/{C}/{day}/{sid}:0.json"


async def _seed(conn, sid: str, keys: list[str], *, enqueued: str, completed: str,
                status: str = "done") -> int:
    return await conn.fetchval(
        """
        INSERT INTO ingestion_queue
            (customer_id, source_system, source_event_id, payload_s3_key,
             payload_s3_keys, status, enqueued_at, completed_at, priority, version)
        VALUES ($1, 'claude_code', $2, ($3::text[])[1], $3::text[], $4,
                $5::text::timestamptz, $6::text::timestamptz, 60, 7)
        RETURNING queue_id
        """,
        C, sid, keys, status, enqueued, completed,
    )


async def _keys(conn, queue_id: int) -> list[str]:
    return list(await conn.fetchval("SELECT payload_s3_keys FROM ingestion_queue WHERE queue_id=$1", queue_id))


@pytest.fixture
async def rows(live_db: None):
    store = get_store()
    async with get_pool().acquire() as conn:
        await conn.execute(
            "INSERT INTO customers(customer_id, display_name, api_key_hash) "
            "VALUES ($1, 'b', 'b-hash') ON CONFLICT DO NOTHING",
            C,
        )
        await conn.execute("DELETE FROM ingestion_queue WHERE customer_id = $1", C)
        ids = {
            # Mined: the last upsert (a day after the newest batch) was an end
            # signal, the pass finished after it, the key is gone, the object is not.
            "mined": await _seed(conn, "s-mined", [_batch("s-mined", "2026/04/29")],
                                 enqueued="2026-05-02 07:00Z", completed="2026-05-02 07:05Z"),
            # Same, but the marker object is gone: missing evidence, never forged.
            "no_object": await _seed(conn, "s-noobj", [_batch("s-noobj", "2026/04/29")],
                                     enqueued="2026-05-02 07:00Z", completed="2026-05-02 07:05Z"),
            # The last upsert WAS the newest batch: a live pass, never mined.
            "live_pass": await _seed(conn, "s-live", [_batch("s-live", "2026/05/02")],
                                     enqueued="2026-05-02 07:00Z", completed="2026-05-02 07:05Z"),
            # Upserted two minutes after the batch's UTC day ended: inside the
            # clock margin, so not evidence either.
            "day_edge": await _seed(conn, "s-edge", [_batch("s-edge", "2026/05/01")],
                                    enqueued="2026-05-02 00:02Z", completed="2026-05-02 00:03Z"),
            # Upserted after the last pass finished: that pass did not see it.
            "stale_pass": await _seed(conn, "s-stale", [_batch("s-stale", "2026/04/29")],
                                      enqueued="2026-05-02 07:00Z", completed="2026-05-02 06:00Z"),
            # Already ends in an end signal: kept by a non-authoritative pass.
            "already": await _seed(
                conn, "s-already",
                [_batch("s-already", "2026/04/29"), cron_marker_key("claude_code", C, "s-already")],
                enqueued="2026-05-02 07:00Z", completed="2026-05-02 07:05Z",
            ),
            # Not done: out of scope entirely.
            "pending": await _seed(conn, "s-pending", [_batch("s-pending", "2026/04/29")],
                                   enqueued="2026-05-02 07:00Z", completed="2026-05-02 07:05Z",
                                   status="pending"),
            # Protocol 2: out of scope entirely.
            "v2": await _seed(conn, "s-v2", [f"raw/claude_code/{C}/sessions-v2/s-v2/0-abc.json"],
                              enqueued="2026-05-02 07:00Z", completed="2026-05-02 07:05Z"),
        }
    bucket = await store.bucket_for(C)
    await store.ensure_bucket(bucket)
    for sid in ("s-mined", "s-live", "s-stale", "s-edge"):
        await store.put(bucket, cron_marker_key("claude_code", C, sid), b'{"finalize": true}')
    yield ids
    async with get_pool().acquire() as conn:
        await conn.execute("DELETE FROM ingestion_queue WHERE customer_id = $1", C)


@pytest.mark.asyncio
async def test_dry_run_counts_every_row_and_writes_nothing(rows) -> None:
    async with get_pool().acquire() as conn:
        before = {k: await _keys(conn, q) for k, q in rows.items()}
    report = await backfill(dry_run=True, customers=[C])
    assert report.candidates == 2
    assert (report.would_relink, report.missing_marker_object, report.relinked) == (1, 1, 0)
    assert report.sweep_will_mine_once == 3, "the live pass, the day edge, the stale pass"
    assert report.already_ended_queue_ids == [rows["already"]]
    async with get_pool().acquire() as conn:
        assert {k: await _keys(conn, q) for k, q in rows.items()} == before


@pytest.mark.asyncio
async def test_only_the_evidenced_row_gets_its_marker_back(rows) -> None:
    async with get_pool().acquire() as conn:
        before = {k: await _keys(conn, q) for k, q in rows.items()}
        mined_row = await conn.fetchrow(
            "SELECT version, status, completed_at FROM ingestion_queue WHERE queue_id=$1", rows["mined"]
        )
    report = await backfill(dry_run=False, customers=[C])
    assert (report.relinked, report.missing_marker_object, report.skipped_changed) == (1, 1, 0)
    async with get_pool().acquire() as conn:
        after = {k: await _keys(conn, q) for k, q in rows.items()}
        mined_after = await conn.fetchrow(
            "SELECT version, status, completed_at FROM ingestion_queue WHERE queue_id=$1", rows["mined"]
        )
    assert after["mined"] == [*before["mined"], cron_marker_key("claude_code", C, "s-mined")]
    # No worker pass follows: version, status and completion are untouched.
    assert tuple(mined_after) == tuple(mined_row)
    for name in ("no_object", "live_pass", "day_edge", "stale_pass", "already", "pending", "v2"):
        assert after[name] == before[name], name

    # Idempotent: a second run re-links nothing; the row without its object
    # stays a candidate every run and is never forged.
    again = await backfill(dry_run=False, customers=[C])
    assert (again.candidates, again.relinked, again.missing_marker_object) == (1, 0, 1)


@pytest.mark.asyncio
async def test_a_row_that_changes_under_the_backfill_is_skipped(rows, monkeypatch) -> None:
    store = get_store()
    real_exists = store.exists

    async def exists_then_a_batch_lands(bucket, key):
        async with get_pool().acquire() as conn:
            await conn.execute(
                "UPDATE ingestion_queue SET payload_s3_keys = payload_s3_keys || ARRAY['late']::text[], "
                "version = version + 1, status = 'pending' WHERE queue_id = $1",
                rows["mined"],
            )
        return await real_exists(bucket, key)

    monkeypatch.setattr(store, "exists", exists_then_a_batch_lands)
    monkeypatch.setattr("scripts.backfill_finalize_markers.get_store", lambda: store)
    report = await backfill(dry_run=False, customers=[C])
    assert report.skipped_changed == 1 and report.relinked == 0
    async with get_pool().acquire() as conn:
        keys = await _keys(conn, rows["mined"])
    assert keys[-1] == "late"
