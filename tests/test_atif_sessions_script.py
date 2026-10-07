"""scripts/atif_sessions.py over sessions built through the real pipeline
(receipts -> queue -> worker pass), on Postgres and MinIO.

replay must read a session the way the worker does and pass the gate on it;
backfill must be a dry run unless told otherwise, write only trajectory.json,
and never write one for a deleted session or one still running (as the live
path: only a session's completing pass writes it).
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from engine.ingest.atif.store import read_trajectory, trajectory_key
from scripts import atif_sessions
from tests.test_session_deletion import (  # the deletion suite's real-pipeline sessions
    ALICE,
    ALICE_EMAIL,
    CC,
    _mine,
    _sid,
    _v2_batches,
    env,  # noqa: F401  # pytest fixture, used by name
    sr,
    v1_session,
    v2_session,
)


async def running_session(customer: str, sid: str) -> None:
    """A protocol-2 session that has not ended: its batches, no finalize."""
    from engine.shared.storage import get_store

    for payload in _v2_batches(sid, ALICE, ALICE_EMAIL)[:-1]:
        assert (await sr.accept(payload, customer, CC, get_store()))["status"] == "accepted"
    await _mine(customer, CC, sid)


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
    keep, gone, running = _sid(), _sid(), _sid()
    await v2_session(a, keep)
    await v2_session(a, gone)
    await running_session(a, running)
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
    # The running session: the backfill never writes it. With live copies on, the
    # worker's own live pass already did (session_ended false).
    live = await read_trajectory(store, bucket, trajectory_key(CC.value, a, running))
    assert live is not None and live["extra"]["session_ended"] is False
    written = await read_trajectory(store, bucket, trajectory_key(CC.value, a, keep))
    assert written is not None and written["session_id"] == keep
    assert await read_trajectory(store, bucket, trajectory_key(CC.value, a, gone)) is None

    assert written["extra"]["session_ended"] is True
    again = (await _run(capsys, ["backfill", "--customer", a, "--write"]))[-1]
    assert again["already_present"] == 2, "a second run rewrites nothing"
    assert "written" not in again


@pytest.mark.asyncio
async def test_replay_fails_the_gate_when_the_trajectory_lines_differ(
    env, capsys, monkeypatch  # noqa: F811
) -> None:
    (a, _b), _store = env
    await v2_session(a, _sid())
    real = atif_sessions.lines_from_trajectory

    def one_line_off(trajectory):
        lines = real(trajectory)
        return [lines[0].__class__(line_no=lines[0].line_no, text="USER: something else"),
                *lines[1:]]

    monkeypatch.setattr(atif_sessions, "lines_from_trajectory", one_line_off)
    records = await _run(capsys, ["replay", "--customer", a, "--sample", "5", "--batchwise", "0"])
    [session] = [r for r in records if r["kind"] == "session"]
    assert (session["same"], session["diff_kind"]) == (False, "text")
    assert records[-1]["gate_passed"] is False


@pytest.mark.asyncio
async def test_replay_counts_a_builder_crash_against_the_gate(env, capsys, monkeypatch) -> None:  # noqa: F811
    (a, _b), _store = env
    await v2_session(a, _sid())

    def crash(*_a, **_k):
        raise TypeError("a shape the builder never saw")

    monkeypatch.setattr(atif_sessions, "build_trajectory", crash)
    summary = (await _run(capsys, ["replay", "--customer", a, "--sample", "5",
                                   "--batchwise", "0"]))[-1]
    assert summary["build_errors"] == 1 and summary["build_error_kinds"] == ["TypeError"]
    assert summary["gate_passed"] is False


def test_every_stratum_is_sampled_even_when_the_sample_is_small() -> None:
    rows = [{"customer_id": f"t{i % 22}", "source_system": "claude_code", "source_event_id": str(i)}
            for i in range(400)]
    picked = atif_sessions._sample(rows, 10, seed=1)
    assert {r["customer_id"] for r in picked} == {f"t{i}" for i in range(22)}


async def _done(customer: str) -> None:
    from engine.shared import db as db_module

    async with db_module.with_tenant(customer) as conn:
        await conn.execute("UPDATE ingestion_queue SET status = 'done' WHERE customer_id = $1",
                           customer)


@pytest.mark.asyncio
@pytest.mark.parametrize("overtaker", ["new_batch", "deletion"])
async def test_backfill_removes_a_write_something_overtook(
    env, capsys, monkeypatch, overtaker  # noqa: F811
) -> None:
    (a, _b), store = env
    sid = _sid()
    await v2_session(a, sid)
    bucket = await store.bucket_for(a)
    key = trajectory_key(CC.value, a, sid)
    await store.delete(bucket, key)
    await _done(a)
    real_write = atif_sessions.write_trajectory

    async def write_then_overtake(*args, **kwargs):
        result = await real_write(*args, **kwargs)
        from engine.shared import db as db_module

        async with db_module.with_tenant(a) as conn:
            if overtaker == "new_batch":  # what the ingestion door does on a new batch
                await conn.execute(
                    "UPDATE ingestion_queue SET version = version + 1 WHERE customer_id = $1", a
                )
            else:
                await conn.execute(
                    "INSERT INTO session_deletions "
                    "(customer_id, source_system, session_id, deletion_id, reason, status) "
                    "VALUES ($1, $2, $3, gen_random_uuid(), 'test', 'done')",
                    a, CC.value, sid,
                )
        return result

    monkeypatch.setattr(atif_sessions, "write_trajectory", write_then_overtake)
    summary = (await _run(capsys, ["backfill", "--customer", a, "--write"]))[-1]
    assert summary.get("overtaken") == 1 and "written" not in summary, summary
    assert await read_trajectory(store, bucket, key) is None


@pytest.mark.asyncio
async def test_backfill_leaves_a_row_the_worker_is_about_to_run(
    env, capsys, monkeypatch  # noqa: F811
) -> None:
    (a, _b), store = env
    sid = _sid()
    await v2_session(a, sid)
    bucket = await store.bucket_for(a)
    await store.delete(bucket, trajectory_key(CC.value, a, sid))
    from engine.shared import db as db_module

    # The tenant's list says `done` (a snapshot); the row went pending since.
    snapshot = await atif_sessions._queue_rows(a, [CC.value])
    async with db_module.with_tenant(a) as conn:
        await conn.execute(
            "UPDATE ingestion_queue SET status = 'pending' WHERE customer_id = $1", a
        )

    async def stale(_customer, _sources):
        return [{**r, "status": "done"} for r in snapshot]

    monkeypatch.setattr(atif_sessions, "_queue_rows", stale)
    summary = (await _run(capsys, ["backfill", "--customer", a, "--write"]))[-1]
    assert summary.get("busy") == 1 and "written" not in summary, summary
    assert await read_trajectory(store, bucket, trajectory_key(CC.value, a, sid)) is None


@pytest.mark.asyncio
async def test_replay_compare_builders_agrees_with_itself_and_the_stored_copy(
    env,  # noqa: F811
    capsys,
) -> None:
    (a, _b), _store = env
    await v2_session(a, _sid())
    await v1_session(a, _sid())
    records = await _run(
        capsys,
        [
            "replay",
            "--customer",
            a,
            "--sample",
            "10",
            "--batchwise",
            "2",
            "--points",
            "3",
            "--compare-builders",
        ],
    )
    sessions = [r for r in records if r["kind"] == "session"]
    summary = records[-1]
    assert len(sessions) == 2, sessions
    for r in sessions:
        assert (r["builders_same"], r["builders_diff"], r["builders_error"]) == (True, None, None)
        # Both ended through the pipeline, so the worker wrote their final copies.
        assert (r["stored"], r["stored_fold_same"], r["stored_reference_same"]) == (
            "final",
            True,
            True,
        ), r
        assert r["batchwise"]["prefix_builder_disagreements"] == 0
    assert summary["builders_gate_passed"] is True and summary["gate_passed"] is True
    assert (summary["builders_compared"], summary["builders_identical"]) == (2, 2)
    assert summary["stored"] == {"final": 2} and summary["stored_fold_identical"] == 2
    assert summary["builders_prefix_disagreements"] == 0 and summary["stored_fold_lost"] == 0
    assert "hello" not in json.dumps(records).lower()


@pytest.mark.asyncio
async def test_replay_compare_builders_names_where_fold_differs_and_no_content(
    env,  # noqa: F811
    capsys,
    monkeypatch,
) -> None:
    (a, _b), _store = env
    await v2_session(a, _sid())
    import engine.ingest.atif.build as build_mod

    real = build_mod.fold

    def tampered(fragments, **kwargs):
        built = real(fragments, **kwargs)
        built.trajectory["steps"][0]["message"] = "TAMPERED-TEXT"
        return built

    monkeypatch.setattr(build_mod, "fold", tampered)
    records = await _run(
        capsys,
        [
            "replay",
            "--customer",
            a,
            "--sample",
            "5",
            "--batchwise",
            "1",
            "--points",
            "2",
            "--compare-builders",
        ],
    )
    [session] = [r for r in records if r["kind"] == "session"]
    summary = records[-1]
    assert session["builders_same"] is False
    assert session["builders_diff"] == {"path": "trajectory.steps[0].message", "kind": "value"}
    assert (session["stored_fold_same"], session["stored_reference_same"]) == (False, True)
    assert session["stored_fold_diff"] == {"path": "steps[0].message", "kind": "value"}
    assert session["batchwise"]["prefix_builder_disagreements"] == 2
    assert summary["builders_gate_passed"] is False and summary["stored_fold_lost"] == 1
    assert summary["builders_diff_paths"] == {"trajectory.steps[].message": 1}
    assert summary["gate_passed"] is True, "the render gate runs the configured (reference) builder"
    assert "TAMPERED" not in json.dumps(records) and "hello" not in json.dumps(records).lower()


def test_a_difference_is_reported_by_schema_path_and_counts_only() -> None:
    diff = atif_sessions._first_difference
    assert diff({"a": 1}, {"a": 1}) is None
    assert diff({"steps": [1, 2]}, {"steps": [1]}) == {
        "path": "steps",
        "kind": "length",
        "left": 2,
        "right": 1,
    }
    assert diff({"steps": [{"text": "x"}]}, {"steps": [{"text": 5}]}) == {
        "path": "steps[0].text",
        "kind": "type",
        "left": "str",
        "right": "int",
    }
    # A key outside the document's vocabulary (a harness extra, anything a
    # client chose) never reaches the report.
    assert diff(
        {"extra": {"codex_extras": [{"api_token_9f": "a"}]}},
        {"extra": {"codex_extras": [{"api_token_9f": "b"}]}},
    ) == {"path": "extra.codex_extras[0].*", "kind": "value"}
    assert diff({"extra": {}}, {"extra": {"my secret key": 1}}) == {
        "path": "extra.*",
        "kind": "missing_left",
    }
    assert diff({"x-secret": 1}, {}) == {"path": "*", "kind": "missing_right"}


@pytest.mark.asyncio
async def test_replay_compare_builders_counts_a_fold_crash_when_fold_is_configured(
    env,  # noqa: F811
    capsys,
    monkeypatch,
) -> None:
    """With SESSION_ATIF_BUILDER=fold the render gate's own build crashes too,
    and that row leaves `compared`; the builders gate must still see it."""
    (a, _b), _store = env
    await v2_session(a, _sid())
    import engine.ingest.atif.build as build_mod
    from engine.shared.config import get_settings

    def crash(*_a, **_k):
        raise TypeError("a fragment shape fold never saw")

    monkeypatch.setattr(build_mod, "fold", crash)
    monkeypatch.setenv("SESSION_ATIF_BUILDER", "fold")
    get_settings.cache_clear()  # type: ignore[attr-defined]
    records = await _run(
        capsys,
        ["replay", "--customer", a, "--sample", "5", "--batchwise", "0", "--compare-builders"],
    )
    [session] = [r for r in records if r["kind"] == "session"]
    summary = records[-1]
    assert session["build_error"] == "TypeError" and session["builders_error"] == "fold:TypeError"
    assert summary["build_errors"] == 1 and summary["gate_passed"] is False
    assert (summary["builders_compared"], summary["builders_errors"]) == (1, 1)
    assert summary["builders_error_kinds"] == ["fold:TypeError"]
    assert summary["builders_gate_passed"] is False
