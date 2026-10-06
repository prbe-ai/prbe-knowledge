"""scripts/atif_sessions.py over sessions built through the real pipeline
(receipts -> queue -> worker pass), on Postgres and MinIO.

replay must read a session the way the worker does and pass the gate on it;
backfill must be a dry run unless told otherwise, write only trajectory.json,
and never write one for a deleted session.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from engine.ingest.atif.store import read_trajectory, trajectory_key
from scripts import atif_sessions
from tests.test_session_deletion import (  # the deletion suite's real-pipeline sessions
    CC,
    _sid,
    env,  # noqa: F401  # pytest fixture, used by name
    v1_session,
    v2_session,
)


async def _run(capsys: pytest.CaptureFixture[str], argv: list[str]) -> list[dict[str, Any]]:
    args = atif_sessions._parse_args(argv)
    await (atif_sessions.replay(args) if args.command == "replay" else atif_sessions.backfill(args))
    # Log lines share stdout; the script's own records are the JSON objects with a kind.
    return [json.loads(x) for x in capsys.readouterr().out.splitlines() if x.startswith('{"kind"')]


@pytest.mark.asyncio
async def test_replay_passes_the_gate_on_real_sessions(env, capsys) -> None:  # noqa: F811
    (a, _b), _store = env
    await v2_session(a, _sid())
    await v1_session(a, _sid())
    records = await _run(capsys, ["replay", "--customer", a, "--sample", "10",
                                  "--batchwise", "2", "--points", "3"])
    sessions = [r for r in records if r["kind"] == "session"]
    summary = records[-1]
    assert len(sessions) == 2 and all(r["same"] for r in sessions), sessions
    assert summary["kind"] == "summary" and summary["gate_passed"] is True, summary
    assert summary["batchwise_sessions"] == 2 and summary["batchwise_disagreements"] == 0
    # Ids, counts and timings only: no transcript text reaches the output.
    assert "hello" not in json.dumps(records).lower()


@pytest.mark.asyncio
async def test_backfill_is_a_dry_run_until_told_and_skips_deleted(env, capsys) -> None:  # noqa: F811
    (a, _b), store = env
    keep, gone = _sid(), _sid()
    await v2_session(a, keep)
    await v2_session(a, gone)
    bucket = await store.bucket_for(a)
    for sid in (keep, gone):  # as if they ended before the engine wrote trajectories
        await store.delete(bucket, trajectory_key(CC.value, a, sid))
    from engine.shared import db as db_module

    async with db_module.with_tenant(a) as conn:
        # What the worker does after a pass; backfill leaves rows it is about to run.
        await conn.execute("UPDATE ingestion_queue SET status = 'done' WHERE customer_id = $1", a)
        await conn.execute(
            "INSERT INTO session_deletions "
            "(customer_id, source_system, session_id, deletion_id, reason, status) "
            "VALUES ($1, $2, $3, gen_random_uuid(), 'test', 'done')",
            a, CC.value, gone,
        )

    assert (await _run(capsys, ["backfill", "--customer", a]))[-1]["would_write"] == 1
    assert await read_trajectory(store, bucket, trajectory_key(CC.value, a, keep)) is None

    summary = (await _run(capsys, ["backfill", "--customer", a, "--write"]))[-1]
    assert (summary["written"], summary["deleted"]) == (1, 1), summary
    written = await read_trajectory(store, bucket, trajectory_key(CC.value, a, keep))
    assert written is not None and written["session_id"] == keep
    assert await read_trajectory(store, bucket, trajectory_key(CC.value, a, gone)) is None

    again = (await _run(capsys, ["backfill", "--customer", a, "--write"]))[-1]
    assert again["already_present"] == 1, "a second run rewrites nothing"
