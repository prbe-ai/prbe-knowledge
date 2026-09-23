"""The idle sweep ends each idle session once, by appending a finalize.marker
to its live queue row -- and leaves alone any session whose NEWEST key is
already an end signal (engine/shared/session_signals.py).

Post-coalescing (migration 0026) a session is one queue row keyed on the bare
session id; the sweep only UPDATEs that row, never inserts one. The worker
treats a marker on top of the row as the session having ended.
"""
from __future__ import annotations

import pytest

from engine.shared.db import get_pool
from kb.session_completer import enqueue_idle_session_finalizers


@pytest.mark.asyncio
async def test_idle_session_gets_finalize_marker_appended(live_db: None) -> None:
    """The sweep appends the finalize.marker key to the existing live session row."""
    customer = "completer-test-cust"
    session_id = "sess-idle"
    live_key = f"raw/claude_code/{customer}/2026/04/29/{session_id}:0.json"

    async with get_pool().acquire() as conn:
        await conn.execute(
            "INSERT INTO customers(customer_id, display_name, api_key_hash) "
            "VALUES ($1, 'c', 'c-hash') ON CONFLICT DO NOTHING",
            customer,
        )
        await conn.execute(
            "DELETE FROM ingestion_queue WHERE customer_id = $1", customer
        )
        # An idle live session row (10 minutes old) using the new coalescing
        # shape: source_event_id is the bare session_id, payload_s3_keys is
        # the array of batch keys.
        await conn.execute(
            """
            INSERT INTO ingestion_queue
                (customer_id, source_system, source_event_id,
                 payload_s3_key, payload_s3_keys, status, enqueued_at,
                 priority, version)
            VALUES ($1, 'claude_code', $2, $3, ARRAY[$3], 'done',
                    NOW() - INTERVAL '10 minutes', 75, 1)
            """,
            customer, session_id, live_key,
        )
        # A fresh session (1 minute old) — should NOT be finalized.
        fresh_key = f"raw/claude_code/{customer}/2026/04/29/sess-fresh:0.json"
        await conn.execute(
            """
            INSERT INTO ingestion_queue
                (customer_id, source_system, source_event_id,
                 payload_s3_key, payload_s3_keys, status, enqueued_at,
                 priority, version)
            VALUES ($1, 'claude_code', 'sess-fresh', $2, ARRAY[$2], 'done',
                    NOW() - INTERVAL '1 minute', 75, 1)
            """,
            customer, fresh_key,
        )

    n = await enqueue_idle_session_finalizers(idle_minutes=5)
    assert n == 1, f"expected exactly one finalize enqueue, got {n}"

    async with get_pool().acquire() as conn:
        idle_row = await conn.fetchrow(
            """
            SELECT payload_s3_keys, status, version
            FROM ingestion_queue
            WHERE customer_id = $1 AND source_event_id = $2
            """,
            customer, session_id,
        )
        fresh_row = await conn.fetchrow(
            """
            SELECT payload_s3_keys, status, version
            FROM ingestion_queue
            WHERE customer_id = $1 AND source_event_id = 'sess-fresh'
            """,
            customer,
        )

        await conn.execute(
            "DELETE FROM ingestion_queue WHERE customer_id = $1", customer
        )

    # Idle session: live row got the finalize.marker appended, status reset
    # to pending so the worker re-claims, version bumped.
    assert idle_row is not None
    assert idle_row["status"] == "pending", "idle row should be re-marked pending"
    assert idle_row["version"] == 2, f"version should be bumped from 1 to 2, got {idle_row['version']}"
    assert len(idle_row["payload_s3_keys"]) == 2
    assert any(k.endswith("/finalize.marker") for k in idle_row["payload_s3_keys"]), (
        f"expected finalize.marker in payload_s3_keys, got {idle_row['payload_s3_keys']}"
    )
    assert live_key in idle_row["payload_s3_keys"], (
        "original live batch key must be preserved alongside the marker"
    )

    # Fresh session: untouched.
    assert fresh_row is not None
    assert fresh_row["status"] == "done"
    assert fresh_row["version"] == 1
    assert len(fresh_row["payload_s3_keys"]) == 1
    assert not any(
        k.endswith("/finalize.marker") for k in fresh_row["payload_s3_keys"]
    )


@pytest.mark.asyncio
async def test_finalize_event_processes_through_normalizer(
    live_db: None, monkeypatch
) -> None:
    """End-to-end: cron-injected finalize.marker is detected by
    fetch_supplementary, session_complete=True triggers unit extraction,
    no DLQ.
    """
    import orjson

    from engine.ingest.handlers.base import make_default_context
    from engine.ingest.normalizer import Normalizer
    from engine.shared import claude_code_extraction as _ext
    from engine.shared.claude_code_extraction import UnitBundle
    from engine.shared.constants import SourceSystem
    from engine.shared.customer_mapping import record_mapping
    from engine.shared.models import IntegrationToken
    from engine.shared.storage import get_store
    from engine.shared.tokens import save_device_token

    customer = "completer-e2e-cust"
    session_id = "sess-e2e-final"
    employee_id = "emp-final-e2e"

    async def _noop_extract(**kwargs):  # type: ignore[no-untyped-def]
        return UnitBundle(qa=[], code_change=[], decision=[], file_ref=[])

    monkeypatch.setattr(_ext, "extract_units_from_session", _noop_extract)

    async with get_pool().acquire() as conn:
        await conn.execute(
            "INSERT INTO customers(customer_id, display_name, api_key_hash) "
            "VALUES ($1, 'cf', 'cf-hash') ON CONFLICT DO NOTHING",
            customer,
        )
    await save_device_token(IntegrationToken(
        customer_id=customer,
        source_system=SourceSystem.CLAUDE_CODE,
        access_token="x",
        webhook_secret="test-secret-hash",
        device_id="cron-finalize-device",
        device_metadata={"hostname": "h"},
    ))
    await record_mapping(
        customer_id=customer,
        source_system=SourceSystem.CLAUDE_CODE,
        external_id="cron-finalize-device",
        external_name="h",
        metadata={},
    )

    # Stage one live batch envelope at the date-partitioned key. After the
    # cron runs, payload_s3_keys will be [live_key, finalize.marker] and
    # fetch_supplementary will read both.
    store = get_store()
    bucket = await store.bucket_for(customer)
    await store.ensure_bucket(bucket)
    live_key = f"raw/claude_code/{customer}/2026/04/29/{session_id}:0.json"
    live_envelope = orjson.dumps({
        "_headers": {},
        "payload": {
            "device_id": "cron-finalize-device",
            "session_id": session_id,
            "batch_seq": 0,
            "cwd": None,
            "events": [{
                "line_no": 0,
                "employee_id": employee_id,
                "raw": {"role": "user", "content": "finalize test prompt"},
            }],
        },
    })
    await store.put(bucket, live_key, live_envelope)

    # Insert the live session row (idle for 10 minutes).
    async with get_pool().acquire() as conn:
        await conn.execute(
            """
            INSERT INTO ingestion_queue
                (customer_id, source_system, source_event_id,
                 payload_s3_key, payload_s3_keys, status, enqueued_at,
                 priority, version)
            VALUES ($1, 'claude_code', $2, $3, ARRAY[$3], 'done',
                    NOW() - INTERVAL '10 minutes', 75, 1)
            """,
            customer, session_id, live_key,
        )

    n = await enqueue_idle_session_finalizers(idle_minutes=5)
    assert n >= 1, f"expected at least one finalize enqueue, got {n}"

    # The cron upserted the marker into the same row — find it.
    async with get_pool().acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT queue_id, source_event_id, payload_s3_key, payload_s3_keys
            FROM ingestion_queue
            WHERE customer_id = $1 AND source_event_id = $2
            """,
            customer, session_id,
        )
    assert row is not None
    assert any(
        k.endswith("/finalize.marker") for k in row["payload_s3_keys"]
    ), "finalize.marker not appended"
    assert live_key in row["payload_s3_keys"]

    # Drive the normalizer with the full coalesced array.
    ctx = make_default_context()
    try:
        normalizer = Normalizer(ctx)
        outcome = await normalizer.process_queue_row(
            queue_id=row["queue_id"],
            customer_id=customer,
            source_system=SourceSystem.CLAUDE_CODE,
            source_event_id=row["source_event_id"],
            payload_s3_keys=list(row["payload_s3_keys"]),
        )
    finally:
        await ctx.http.aclose()

    assert outcome.doc_ids, "normalizer produced no doc_ids — DLQ would have fired"

    async with get_pool().acquire() as conn:
        doc_rows = await conn.fetch(
            "SELECT doc_type FROM documents WHERE customer_id = $1",
            customer,
        )
    types = {r["doc_type"] for r in doc_rows}
    assert "claude_code.session" in types, (
        f"expected claude_code.session doc, found: {types}"
    )


class TestCronIdleWindowPrecedence:
    """`--idle-minutes` is the whole point of the sweep.

    Research passes 1440 (the managed nightly passes 360). Ending a session
    mines it; a session ended while someone is still using it is mined, then
    mined again when it next goes idle. If the flag were silently ignored and
    the 5-minute settings default won, the sweep would end sessions people are
    still using -- which is exactly the failure this tests against.
    """

    def test_explicit_flag_wins_over_settings_default(self) -> None:
        from scripts.cron_session_completer import _parse_args, resolve_idle_minutes

        args = _parse_args(["--idle-minutes", "360"])
        assert resolve_idle_minutes(args.idle_minutes) == 360

    def test_omitted_flag_falls_back_to_settings(self) -> None:
        from engine.shared.config import get_settings
        from scripts.cron_session_completer import _parse_args, resolve_idle_minutes

        args = _parse_args([])
        assert args.idle_minutes is None
        assert resolve_idle_minutes(None) == get_settings().claude_code_session_idle_minutes

    @pytest.mark.asyncio
    async def test_a_window_under_an_hour_is_refused(self, monkeypatch) -> None:
        """The flagless default is 5 minutes. Ending a session mines it, so a
        run with it would re-mine every session after every pause."""
        import scripts.cron_session_completer as script

        async def no_pool(*a, **k):
            return None

        monkeypatch.setattr(script, "init_pool", no_pool)
        with pytest.raises(SystemExit, match="below 60"):
            await script._main([])

    def test_nonpositive_window_is_rejected(self) -> None:
        """A 0 would finalize every live session on the next sweep."""
        from scripts.cron_session_completer import _parse_args

        with pytest.raises(SystemExit):
            _parse_args(["--idle-minutes", "0"])


class TestSweepCostBounds:
    """The sweep buys a full multi-segment extraction per session it finalizes.
    Those properties are the difference between a safe first run and a bill."""

    def test_limit_and_dry_run_are_rejected_when_nonsensical(self) -> None:
        from scripts.cron_session_completer import _parse_args

        with pytest.raises(SystemExit):
            _parse_args(["--limit", "0"])

    def test_dry_run_and_limit_reach_the_sweep(self) -> None:
        from scripts.cron_session_completer import _parse_args

        args = _parse_args(["--idle-minutes", "360", "--limit", "50", "--dry-run"])
        assert (args.idle_minutes, args.limit, args.dry_run) == (360, 50, True)

    def test_defaults_are_bounded(self) -> None:
        """An unbounded default is how a first run against a never-swept corpus
        becomes a surprise invoice."""
        from scripts.cron_session_completer import _parse_args

        args = _parse_args([])
        assert args.limit == 1000
        assert args.dry_run is False



# ---- the sweep ends each idle session ONCE ------------------------------------
#
# The regression these pin: the sweep asked "is a finalize key still here?"
# while the worker deleted that key after mining, so every idle session was
# re-ended and fully re-mined once a day (docs/plans/extraction-spend-plan.md).
# The sweep now asks the worker's own question -- is an end signal the NEWEST
# key (engine.shared.session_signals) -- and nothing deletes it.


async def _seed(conn, customer: str, session_id: str, keys: list[str], *,
                status: str = "done", idle: str = "2 days") -> None:
    await conn.execute(
        "INSERT INTO customers(customer_id, display_name, api_key_hash) "
        "VALUES ($1, 'c', $1 || '-hash') ON CONFLICT DO NOTHING",
        customer,
    )
    await conn.execute(
        """
        INSERT INTO ingestion_queue
            (customer_id, source_system, source_event_id, payload_s3_key,
             payload_s3_keys, status, enqueued_at, completed_at, priority, version)
        VALUES ($1, 'claude_code', $2, ($3::text[])[1], $3::text[], $4,
                NOW() - $5::text::interval, NOW() - $5::text::interval, 60, 1)
        """,
        customer, session_id, keys, status, idle,
    )


async def _row(conn, customer: str, session_id: str):
    return await conn.fetchrow(
        "SELECT queue_id, payload_s3_keys, status, version FROM ingestion_queue "
        "WHERE customer_id = $1 AND source_event_id = $2",
        customer, session_id,
    )


@pytest.mark.asyncio
async def test_a_mined_session_is_not_re_ended_by_the_next_sweep(live_db: None, monkeypatch) -> None:
    """CRITICAL. Sweep -> worker mines it -> sweep again must do NOTHING.

    Drives the real normalizer between the two sweeps, because the bug lived
    in the handshake: the worker's pass removed the very key the next sweep
    checked for. On the old code the second sweep returns 1.
    """
    import orjson

    from engine.ingest.handlers.base import make_default_context
    from engine.ingest.normalizer import Normalizer
    from engine.shared import claude_code_extraction as _ext
    from engine.shared.constants import SourceSystem
    from engine.shared.storage import get_store

    customer, session_id = "completer-once-cust", "sess-once"
    passes = {"n": 0}

    async def mine(**kwargs):
        passes["n"] += 1
        return _ext.UnitBundle(qa=[_ext.QA(prompt="p", outcome="o")])

    monkeypatch.setattr(_ext, "extract_units_from_session", mine)
    store = get_store()
    live_key = f"raw/claude_code/{customer}/2026/04/29/{session_id}:0.json"
    async with get_pool().acquire() as conn:
        await conn.execute("DELETE FROM ingestion_queue WHERE customer_id = $1", customer)
        await _seed(conn, customer, session_id, [live_key])
    bucket = await store.bucket_for(customer)
    await store.ensure_bucket(bucket)
    await store.put(bucket, live_key, orjson.dumps({"_headers": {}, "payload": {
        "device_id": "d", "session_id": session_id, "batch_seq": 0, "employee_id": "emp",
        "events": [{"line_no": 0, "employee_id": "emp", "raw": {"type": "user", "content": "hi"}}],
    }}))

    assert await enqueue_idle_session_finalizers(idle_minutes=1440) == 1

    async with get_pool().acquire() as conn:
        row = await _row(conn, customer, session_id)
    ctx = make_default_context()
    try:
        await Normalizer(ctx).process_queue_row(
            queue_id=row["queue_id"], customer_id=customer,
            source_system=SourceSystem.CLAUDE_CODE, source_event_id=session_id,
            payload_s3_keys=list(row["payload_s3_keys"]),
        )
    finally:
        await ctx.http.aclose()
    assert passes["n"] == 1
    async with get_pool().acquire() as conn:
        # What the worker's CAS commit does, then a day and a half of quiet.
        await conn.execute(
            "UPDATE ingestion_queue SET status='done', completed_at=NOW(), "
            "enqueued_at=NOW()-INTERVAL '36 hours' WHERE queue_id=$1",
            row["queue_id"],
        )
        mined = await _row(conn, customer, session_id)

    assert await enqueue_idle_session_finalizers(idle_minutes=1440) == 0
    async with get_pool().acquire() as conn:
        after = await _row(conn, customer, session_id)
        await conn.execute("DELETE FROM ingestion_queue WHERE customer_id = $1", customer)
    assert list(after["payload_s3_keys"]) == list(mined["payload_s3_keys"])
    assert after["payload_s3_keys"][-1].endswith("/finalize.marker")
    assert after["version"] == mined["version"] and after["status"] == "done"


@pytest.mark.parametrize(
    "last",
    ["raw/claude_code/{c}/{s}/finalize.marker", "raw/claude_code/{c}/2026/04/29/{s}.json"],
    ids=["cron_marker", "v1_client_finalize"],
)
@pytest.mark.asyncio
async def test_a_session_already_ended_is_left_alone(live_db: None, last: str) -> None:
    customer, session_id = "completer-ended-cust", "sess-ended"
    keys = [f"raw/claude_code/{customer}/2026/04/29/{session_id}:0.json",
            last.format(c=customer, s=session_id)]
    async with get_pool().acquire() as conn:
        await conn.execute("DELETE FROM ingestion_queue WHERE customer_id = $1", customer)
        await _seed(conn, customer, session_id, keys)
    assert await enqueue_idle_session_finalizers(idle_minutes=1440) == 0
    async with get_pool().acquire() as conn:
        row = await _row(conn, customer, session_id)
        await conn.execute("DELETE FROM ingestion_queue WHERE customer_id = $1", customer)
    assert list(row["payload_s3_keys"]) == keys and row["version"] == 1


@pytest.mark.asyncio
async def test_a_batch_after_an_ending_is_ended_again_once(live_db: None) -> None:
    """A resumed session is live again; once it goes quiet it is ended once
    more -- and then left alone."""
    customer, session_id = "completer-resumed-cust", "sess-resumed"
    keys = [
        f"raw/claude_code/{customer}/2026/04/29/{session_id}:0.json",
        f"raw/claude_code/{customer}/{session_id}/finalize.marker",
        f"raw/claude_code/{customer}/2026/04/30/{session_id}:1.json",
    ]
    async with get_pool().acquire() as conn:
        await conn.execute("DELETE FROM ingestion_queue WHERE customer_id = $1", customer)
        await _seed(conn, customer, session_id, keys)
    assert await enqueue_idle_session_finalizers(idle_minutes=1440) == 1
    async with get_pool().acquire() as conn:
        row = await _row(conn, customer, session_id)
        assert row["payload_s3_keys"][-1].endswith("/finalize.marker")
        assert (row["status"], row["version"]) == ("pending", 2)
        await conn.execute(
            "UPDATE ingestion_queue SET status='done', enqueued_at=NOW()-INTERVAL '2 days' "
            "WHERE queue_id=$1", row["queue_id"],
        )
    assert await enqueue_idle_session_finalizers(idle_minutes=1440) == 0
    async with get_pool().acquire() as conn:
        await conn.execute("DELETE FROM ingestion_queue WHERE customer_id = $1", customer)


@pytest.mark.asyncio
async def test_the_sweep_never_inserts_and_skips_rows_in_flight(live_db: None) -> None:
    """The shape the OLD sweep turned into an INSERT -- an idle session known
    only by a legacy `<session>:<batch>` row, no bare-session-id row -- made a
    marker-only row that dead-lettered (288 on research). Now nothing is
    inserted and the legacy identity is not treated as a session. A row being
    processed right now is not touched either."""
    customer = "completer-noinsert-cust"
    legacy_key = f"raw/claude_code/{customer}/2026/04/29/sess-legacy:0.json"
    async with get_pool().acquire() as conn:
        await conn.execute("DELETE FROM ingestion_queue WHERE customer_id = $1", customer)
        await _seed(conn, customer, "sess-legacy:0", [legacy_key])
        await _seed(conn, customer, "sess-busy",
                    [f"raw/claude_code/{customer}/2026/04/29/sess-busy:0.json"], status="processing")
        before = await conn.fetchval("SELECT count(*) FROM ingestion_queue")
    assert await enqueue_idle_session_finalizers(idle_minutes=1440) == 0
    async with get_pool().acquire() as conn:
        assert await conn.fetchval("SELECT count(*) FROM ingestion_queue") == before
        legacy = await _row(conn, customer, "sess-legacy:0")
        busy = await _row(conn, customer, "sess-busy")
        await conn.execute("DELETE FROM ingestion_queue WHERE customer_id = $1", customer)
    assert list(legacy["payload_s3_keys"]) == [legacy_key]
    assert len(busy["payload_s3_keys"]) == 1 and busy["status"] == "processing"


@pytest.mark.asyncio
async def test_a_leftover_marker_only_row_is_skipped_not_dead_lettered(live_db: None) -> None:
    """Through the REAL normalizer: an empty result needs a reason, or it
    raises NormalizationError and the worker dead-letters the row."""
    from engine.ingest.handlers.base import make_default_context
    from engine.ingest.normalizer import Normalizer
    from engine.shared.constants import SourceSystem
    from engine.shared.exceptions import DuplicateEventIgnored
    from engine.shared.session_signals import cron_marker_body, cron_marker_key
    from engine.shared.storage import get_store

    customer, sid = "completer-markeronly-cust", "sess-markeronly"
    key = cron_marker_key("claude_code", customer, sid)
    async with get_pool().acquire() as conn:
        await conn.execute("DELETE FROM ingestion_queue WHERE customer_id = $1", customer)
        await _seed(conn, customer, sid, [key], status="dlq")
        row = await _row(conn, customer, sid)
    store = get_store()
    bucket = await store.bucket_for(customer)
    await store.ensure_bucket(bucket)
    await store.put(bucket, key, cron_marker_body(sid))
    ctx = make_default_context()
    try:
        with pytest.raises(DuplicateEventIgnored):
            await Normalizer(ctx).process_queue_row(
                queue_id=row["queue_id"], customer_id=customer,
                source_system=SourceSystem.CLAUDE_CODE, source_event_id=sid,
                payload_s3_keys=[key],
            )
    finally:
        await ctx.http.aclose()
        async with get_pool().acquire() as conn:
            await conn.execute("DELETE FROM ingestion_queue WHERE customer_id = $1", customer)


@pytest.mark.asyncio
async def test_a_batch_landing_mid_sweep_is_not_ended(live_db: None, monkeypatch) -> None:
    """Candidates are read before the per-session lock. A batch that lands in
    between makes the session active again, and the conditioned UPDATE must
    skip it rather than end a session someone is using."""
    import kb.session_completer as completer

    customer, session_id = "completer-race-cust", "sess-race"
    first = f"raw/claude_code/{customer}/2026/04/29/{session_id}:0.json"
    async with get_pool().acquire() as conn:
        await conn.execute("DELETE FROM ingestion_queue WHERE customer_id = $1", customer)
        await _seed(conn, customer, session_id, [first])

    real_lock = completer._lock

    async def lock_after_a_batch_lands(conn, *args):
        async with get_pool().acquire() as other:
            await other.execute(
                "UPDATE ingestion_queue SET payload_s3_keys = payload_s3_keys || $3::text[], "
                "version = version + 1, enqueued_at = NOW() "
                "WHERE customer_id = $1 AND source_event_id = $2",
                customer, session_id, [first.replace(":0.json", ":1.json")],
            )
        await real_lock(conn, *args)

    monkeypatch.setattr(completer, "_lock", lock_after_a_batch_lands)
    assert await enqueue_idle_session_finalizers(idle_minutes=1440) == 0
    async with get_pool().acquire() as conn:
        row = await _row(conn, customer, session_id)
        await conn.execute("DELETE FROM ingestion_queue WHERE customer_id = $1", customer)
    assert not any(k.endswith("/finalize.marker") for k in row["payload_s3_keys"])


@pytest.mark.asyncio
async def test_a_dead_letter_is_retried_once_not_daily(live_db: None) -> None:
    customer, session_id = "completer-dlq-cust", "sess-dlq"
    async with get_pool().acquire() as conn:
        await conn.execute("DELETE FROM ingestion_queue WHERE customer_id = $1", customer)
        await _seed(conn, customer, session_id,
                    [f"raw/claude_code/{customer}/2026/04/29/{session_id}:0.json"], status="dlq")
    assert await enqueue_idle_session_finalizers(idle_minutes=1440) == 1
    async with get_pool().acquire() as conn:
        await conn.execute(
            "UPDATE ingestion_queue SET status='dlq', enqueued_at=NOW()-INTERVAL '2 days' "
            "WHERE customer_id=$1", customer,
        )
    assert await enqueue_idle_session_finalizers(idle_minutes=1440) == 0
    async with get_pool().acquire() as conn:
        await conn.execute("DELETE FROM ingestion_queue WHERE customer_id = $1", customer)



@pytest.mark.asyncio
async def test_one_failing_row_does_not_stop_the_sweep(live_db: None, monkeypatch) -> None:
    """Rows are taken oldest first. If one row raised out of the run, the same
    row would come up first every hour and nothing would be ended again."""
    from engine.shared.storage import get_store

    customer = "completer-badrow-cust"
    async with get_pool().acquire() as conn:
        await conn.execute("DELETE FROM ingestion_queue WHERE customer_id = $1", customer)
        await _seed(conn, customer, "sess-bad", [f"raw/claude_code/{customer}/2026/04/29/sess-bad:0.json"],
                    idle="3 days")
        await _seed(conn, customer, "sess-good", [f"raw/claude_code/{customer}/2026/04/29/sess-good:0.json"],
                    idle="2 days")
    store = get_store()
    real_put = store.put

    async def put(bucket, key, body):
        if "sess-bad" in key:
            raise RuntimeError("R2 refused")
        return await real_put(bucket, key, body)

    monkeypatch.setattr(store, "put", put)
    monkeypatch.setattr("kb.session_completer.get_store", lambda: store)
    assert await enqueue_idle_session_finalizers(idle_minutes=1440) == 1
    async with get_pool().acquire() as conn:
        good = await _row(conn, customer, "sess-good")
        bad = await _row(conn, customer, "sess-bad")
        await conn.execute("DELETE FROM ingestion_queue WHERE customer_id = $1", customer)
    assert good["payload_s3_keys"][-1].endswith("/finalize.marker")
    assert len(bad["payload_s3_keys"]) == 1
