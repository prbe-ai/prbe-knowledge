"""scripts/strip_session_payloads.py over a session stored through the real
pipeline (receipts -> R2 -> queue -> worker pass), on Postgres and MinIO."""

from __future__ import annotations

import json
from typing import Any

import orjson
import pytest

import tests.test_session_deletion as deletion_suite
from engine.shared.transcript_render import lines_from_events
from scripts import strip_session_payloads as strip_mod
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
)

LEAK = "TOOL-OUTPUT-9b2c"


async def old_tap_session(customer: str, sid: str) -> None:
    """A protocol-2 session as a tap before 0.9.10 stored it: toolUseResult rides along."""
    from engine.shared.storage import get_store

    for payload in _v2_batches(sid, ALICE, ALICE_EMAIL):
        for event in payload.get("events") or []:
            event["raw"]["toolUseResult"] = {"stdout": LEAK}
        assert (await sr.accept(payload, customer, CC, get_store()))["status"] == "accepted"
    await _mine(customer, CC, sid)


async def _stored(store, customer: str) -> dict[str, bytes]:
    bucket = await store.bucket_for(customer)
    keys = [k for k in await store.list_keys(bucket, f"raw/{CC.value}/{customer}/sessions-v2/")]
    return {k: await store.get(bucket, k) for k in keys}


async def _run(capsys: pytest.CaptureFixture[str], argv: list[str]) -> list[dict[str, Any]]:
    await strip_mod.strip(strip_mod._parse_args(argv))
    return [json.loads(x) for x in capsys.readouterr().out.splitlines() if x.startswith('{"kind"')]


def _lines(bodies: dict[str, bytes]) -> list[Any]:
    events = []
    for body in bodies.values():
        events.extend((orjson.loads(body).get("payload") or {}).get("events") or [])
    return lines_from_events(sorted(events, key=lambda e: e["line_no"]))


@pytest.mark.asyncio
async def test_strip_is_a_dry_run_until_told_then_removes_only_what_nothing_reads(
    env, capsys  # noqa: F811
) -> None:
    (a, _b), store = env
    await old_tap_session(a, _sid())
    before = await _stored(store, a)
    assert before and all(LEAK.encode() in body for body in before.values() if b'"events":[{' in body)

    dry = (await _run(capsys, ["strip", "--customer", a]))[-1]
    assert dry["stripped"] >= 1 and "written" not in dry and dry["bytes_removed"] > 0
    assert await _stored(store, a) == before, "a dry run writes nothing"

    done = (await _run(capsys, ["strip", "--customer", a, "--write"]))[-1]
    assert done["written"] == dry["stripped"], done
    after = await _stored(store, a)
    assert set(after) == set(before), "same keys: the receipts still point at them"
    assert not any(LEAK.encode() in body for body in after.values())
    assert _lines(after) == _lines(before), "the indexed text cannot move"

    again = (await _run(capsys, ["strip", "--customer", a, "--write"]))[-1]
    assert "written" not in again and again["clean"] >= 1


@pytest.mark.asyncio
async def test_a_batch_whose_text_would_change_is_left_alone(env, capsys, monkeypatch) -> None:  # noqa: F811
    (a, _b), store = env
    await old_tap_session(a, _sid())
    before = await _stored(store, a)

    def rewrites_text(event: Any) -> Any:
        if isinstance(event, dict) and isinstance(event.get("message"), dict):
            return {**event, "message": {**event["message"], "content": "something else"}}
        return event

    monkeypatch.setattr(strip_mod, "project_event", rewrites_text)
    summary = (await _run(capsys, ["strip", "--customer", a, "--write"]))[-1]
    assert summary.get("lines_changed", 0) >= 1 and "written" not in summary
    assert await _stored(store, a) == before


@pytest.mark.asyncio
async def test_a_deleted_session_is_never_rewritten(env, capsys) -> None:  # noqa: F811
    (a, _b), store = env
    sid = _sid()
    await old_tap_session(a, sid)
    before = await _stored(store, a)
    from engine.shared import db as db_module

    async with db_module.with_tenant(a) as conn:
        await conn.execute(
            "INSERT INTO session_deletions "
            "(customer_id, source_system, session_id, deletion_id, reason, status) "
            "VALUES ($1, $2, $3, gen_random_uuid(), 'test', 'done')",
            a, CC.value, sid,
        )
    summary = (await _run(capsys, ["strip", "--customer", a, "--write"]))[-1]
    assert summary.get("skipped:deleted", 0) >= 1 and "written" not in summary
    assert await _stored(store, a) == before


@pytest.mark.asyncio
async def test_a_tenant_under_legal_hold_is_never_rewritten(env, capsys) -> None:  # noqa: F811
    (a, _b), store = env
    await old_tap_session(a, _sid())
    before = await _stored(store, a)
    from engine.shared import db as db_module

    async with db_module.raw_conn() as conn:
        await conn.execute(
            "UPDATE customers SET metadata = coalesce(metadata, '{}'::jsonb) "
            "|| '{\"legal_hold\": \"litigation\"}'::jsonb WHERE customer_id = $1", a)
    try:
        records = await _run(capsys, ["strip", "--customer", a, "--write"])
    finally:
        async with db_module.raw_conn() as conn:
            await conn.execute(
                "UPDATE customers SET metadata = metadata - 'legal_hold' WHERE customer_id = $1", a)
    assert any(r.get("skipped") == "legal_hold" for r in records if r["kind"] == "tenant")
    assert "written" not in records[-1]
    assert await _stored(store, a) == before


@pytest.mark.asyncio
async def test_a_deletion_landing_during_the_write_gets_its_copy_removed(
    env, capsys, monkeypatch  # noqa: F811
) -> None:
    (a, _b), store = env
    sid = _sid()
    await old_tap_session(a, sid)
    from engine.shared import db as db_module

    real_put = store.put

    async def put_then_delete(bucket: str, key: str, body: bytes, *args: Any, **kw: Any) -> None:
        await real_put(bucket, key, body, *args, **kw)
        async with db_module.with_tenant(a) as conn:
            await conn.execute(
                "INSERT INTO session_deletions "
                "(customer_id, source_system, session_id, deletion_id, reason, status) "
                "VALUES ($1, $2, $3, gen_random_uuid(), 'test', 'done') ON CONFLICT DO NOTHING",
                a, CC.value, sid,
            )

    before = await _stored(store, a)
    monkeypatch.setattr(store, "put", put_then_delete)
    summary = (await _run(capsys, ["strip", "--customer", a, "--write"]))[-1]
    overtaken = summary.get("overtaken", 0)
    assert overtaken >= 1 and "written" not in summary, summary
    after = await _stored(store, a)
    # Batches written as the deletion landed (several can be in flight) are
    # removed again; the rest are left exactly as they were for its sweep.
    assert len(after) == len(before) - overtaken
    assert all(after[k] == before[k] for k in after)


@pytest.mark.asyncio
async def test_a_protocol_1_session_is_stripped_too(env, capsys, monkeypatch) -> None:  # noqa: F811
    (a, _b), store = env
    real_event = deletion_suite._event

    def leaky(i: int, text: str) -> dict:
        event = real_event(i, text)
        event["raw"]["toolUseResult"] = {"stdout": LEAK}
        return event

    monkeypatch.setattr(deletion_suite, "_event", leaky)
    keys = await v1_session(a, _sid())
    bucket = await store.bucket_for(a)
    before = {k: await store.get(bucket, k) for k in keys}
    assert any(LEAK.encode() in b for b in before.values())
    summary = (await _run(capsys, ["strip", "--customer", a, "--write"]))[-1]
    assert summary.get("written", 0) >= 1, summary
    after = {k: await store.get(bucket, k) for k in keys}
    assert not any(LEAK.encode() in b for b in after.values())
