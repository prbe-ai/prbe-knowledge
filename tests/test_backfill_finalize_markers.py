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
from scripts.backfill_finalize_markers import Outcome, backfill

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
            # The next day, hours later: an upload that stalled across midnight
            # would look exactly like this, so it is not evidence (margin = a day).
            "next_day": await _seed(conn, "s-next", [_batch("s-next", "2026/05/01")],
                                    enqueued="2026-05-02 07:00Z", completed="2026-05-02 07:05Z"),
            # Several batches: the NEWEST day decides. The last write is a batch.
            "multi_live": await _seed(
                conn, "s-multi-live",
                [_batch("s-multi-live", "2026/04/29"), f"raw/claude_code/{C}/2026/05/02/s-multi-live:1.json"],
                enqueued="2026-05-02 07:00Z", completed="2026-05-02 07:05Z",
            ),
            # Several older batches, then an end signal days later: mined.
            "multi_mined": await _seed(
                conn, "s-multi-mined",
                [_batch("s-multi-mined", "2026/04/28"), f"raw/claude_code/{C}/2026/04/29/s-multi-mined:1.json"],
                enqueued="2026-05-02 07:00Z", completed="2026-05-02 07:05Z",
            ),
            # Evidence holds, but a protocol-2 stream owns the session.
            "streamed": await _seed(conn, "s-streamed", [_batch("s-streamed", "2026/04/29")],
                                    enqueued="2026-05-02 07:00Z", completed="2026-05-02 07:05Z"),
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
    for sid in ("s-mined", "s-live", "s-stale", "s-edge", "s-next", "s-multi-live",
                "s-multi-mined", "s-streamed"):
        await store.put(bucket, cron_marker_key("claude_code", C, sid), b'{"finalize": true}')
    async with get_pool().acquire() as conn:
        await conn.execute("DELETE FROM session_streams WHERE customer_id = $1", C)
        await conn.execute(
            "INSERT INTO session_streams (customer_id, source_system, session_id, stream_id, "
            "protocol_version, last_seq, source_byte_end, source_line_end, event_end, prefix_sha256, "
            "finalized) VALUES ($1, 'claude_code', 's-streamed', 's', 2, 0, 0, 0, 0, '', false)",
            C,
        )
    yield ids
    async with get_pool().acquire() as conn:
        await conn.execute("DELETE FROM ingestion_queue WHERE customer_id = $1", C)
        await conn.execute("DELETE FROM session_streams WHERE customer_id = $1", C)


@pytest.mark.asyncio
async def test_dry_run_counts_every_row_and_writes_nothing(rows) -> None:
    async with get_pool().acquire() as conn:
        before = {k: await _keys(conn, q) for k, q in rows.items()}
    report = await backfill(dry_run=True, customers=[C])
    assert report.candidates == 4, "mined, no_object, multi_mined, streamed"
    assert report.outcomes == {
        Outcome.WOULD_RELINK: 2, Outcome.MISSING_MARKER_OBJECT: 1, Outcome.SKIPPED_STREAM: 1,
    }
    assert report.sweep_will_mine_once == 5, "live, day edge, next day, multi live, stale"
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
    assert report.outcomes == {
        Outcome.RELINKED: 2, Outcome.MISSING_MARKER_OBJECT: 1, Outcome.SKIPPED_STREAM: 1,
    }
    async with get_pool().acquire() as conn:
        after = {k: await _keys(conn, q) for k, q in rows.items()}
        mined_after = await conn.fetchrow(
            "SELECT version, status, completed_at FROM ingestion_queue WHERE queue_id=$1", rows["mined"]
        )
    assert after["mined"] == [*before["mined"], cron_marker_key("claude_code", C, "s-mined")]
    # No worker pass follows: version, status and completion are untouched.
    assert tuple(mined_after) == tuple(mined_row)
    assert after["multi_mined"][-1] == cron_marker_key("claude_code", C, "s-multi-mined")
    for name in ("no_object", "live_pass", "day_edge", "next_day", "multi_live", "streamed",
                 "stale_pass", "already", "pending", "v2"):
        assert after[name] == before[name], name

    # Idempotent: a second run re-links nothing; the row without its object
    # stays a candidate every run and is never forged.
    again = await backfill(dry_run=False, customers=[C])
    assert again.candidates == 2 and again.outcomes[Outcome.RELINKED] == 0
    assert again.outcomes[Outcome.MISSING_MARKER_OBJECT] == 1


@pytest.mark.asyncio
async def test_a_row_that_changes_under_the_backfill_is_skipped(rows, monkeypatch) -> None:
    store = get_store()
    real_exists = store.exists

    async def exists_then_a_batch_lands(bucket, key):
        async with get_pool().acquire() as conn:
            await conn.execute(
                "UPDATE ingestion_queue SET payload_s3_keys = payload_s3_keys || ARRAY['late']::text[], "
                "version = version + 1, status = 'pending' WHERE queue_id = ANY($1::bigint[])",
                [rows["mined"], rows["multi_mined"]],
            )
        return await real_exists(bucket, key)

    monkeypatch.setattr(store, "exists", exists_then_a_batch_lands)
    monkeypatch.setattr("scripts.backfill_finalize_markers.get_store", lambda: store)
    report = await backfill(dry_run=False, customers=[C])
    assert report.outcomes[Outcome.SKIPPED_CHANGED] == 2, "both evidenced rows changed under it"
    assert report.outcomes[Outcome.RELINKED] == 0
    async with get_pool().acquire() as conn:
        keys = await _keys(conn, rows["mined"])
    assert keys[-1] == "late"


@pytest.mark.asyncio
async def test_one_failing_row_is_counted_not_fatal(rows, monkeypatch) -> None:
    """A transient R2 error on one row must not abort the one-time run and
    lose the report for every other row."""
    store = get_store()
    real_exists = store.exists

    async def flaky(bucket, key):
        if "s-mined" in key:
            raise RuntimeError("R2 hiccup")
        return await real_exists(bucket, key)

    monkeypatch.setattr(store, "exists", flaky)
    monkeypatch.setattr("scripts.backfill_finalize_markers.get_store", lambda: store)
    report = await backfill(dry_run=False, customers=[C])
    assert report.outcomes[Outcome.ERRORED] == 1
    assert report.outcomes[Outcome.RELINKED] == 1, "multi_mined still went through"


@pytest.mark.asyncio
async def test_legacy_partial_passes_are_handed_to_the_retry(rows) -> None:
    """Only rows the report named BEFORE the relink run: a row the relink just
    proved mined also ends in an end signal with no outcome, and stamping it
    would pay to mine it again. Rows mined after the cutoff are left alone."""
    from datetime import UTC, datetime

    from scripts.backfill_finalize_markers import stamp_legacy_retry

    before_relink = await backfill(dry_run=True, customers=[C])
    captured = before_relink.already_ended_queue_ids
    assert captured == [rows["already"]]
    await backfill(dry_run=False, customers=[C])  # relinks "mined" and "multi_mined"

    cutoff = datetime(2026, 5, 3, tzinfo=UTC)
    async with get_pool().acquire() as conn:
        after_cutoff = await _seed(
            conn, "s-new-worker",
            [_batch("s-new-worker", "2026/05/04"), cron_marker_key("claude_code", C, "s-new-worker")],
            enqueued="2026-05-04 07:00Z", completed="2026-05-04 07:05Z",
        )
    ids = [*captured, after_cutoff]  # the cutoff still guards a row mined since
    assert await stamp_legacy_retry(queue_ids=ids, completed_before=cutoff, dry_run=True) == 1
    assert await stamp_legacy_retry(queue_ids=ids, completed_before=cutoff, dry_run=False) == 1
    async with get_pool().acquire() as conn:
        import json

        stamped = json.loads(await conn.fetchval(
            "SELECT extraction_outcome FROM ingestion_queue WHERE queue_id=$1", rows["already"]
        ))
        untouched = await conn.fetchval(
            "SELECT extraction_outcome FROM ingestion_queue WHERE queue_id=$1", after_cutoff
        )
        relinked = await conn.fetchval(
            "SELECT extraction_outcome FROM ingestion_queue WHERE queue_id=$1", rows["mined"]
        )
    assert (stamped["authoritative"], stamped["reason"], stamped["keys"], stamped["retries"]) == (
        False, "legacy_unconsumed", 2, 0
    )
    assert untouched is None
    assert relinked is None, "a row the relink proved mined is never handed to the retry"
    assert await stamp_legacy_retry(queue_ids=ids, completed_before=cutoff, dry_run=False) == 0
