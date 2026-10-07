"""The readers of stored batches over protocol-3 sessions, stored through the
real pipeline (receipts -> R2 -> queue -> worker pass), on Postgres and MinIO.

scripts/atif_sessions.py replay and backfill read a protocol-3 session from its
fragments, as the worker does; scripts/strip_session_payloads.py never touches
its fragments and projects the canary's events only where they still fragment
the same; scripts/rechunk_collapsed_sessions.py re-renders its body from its
fragments' Lines.
"""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

import orjson
import pytest

from engine.ingest.atif.store import trajectory_key
from engine.ingest.handlers.base import make_default_context
from engine.ingest.normalizer import Normalizer
from engine.shared import claude_code_extraction as ext
from engine.shared.config import Settings
from engine.shared.transcript_render import lines_from_events, render_lines
from scripts import atif_sessions
from scripts import rechunk_collapsed_sessions as rechunk
from scripts import strip_session_payloads as strip_mod
from tests.test_session_deletion import (  # the deletion suite's real-pipeline sessions
    CC,
    _mine,
    env,  # noqa: F401  # pytest fixture, used by name
    sr,
    v2_session,
)
from tests.test_session_protocol3 import SESSIONS, _numbered, session_batches

LEAK = "TOOL-OUTPUT-5e1d"
EVENTS = SESSIONS["golden:claude_code.expected.jsonl"][0]


@pytest.fixture(autouse=True)
def _protocol3_and_no_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """The door advertises protocol 3; the model call is the only thing stubbed."""
    settings = Settings(session_protocol3_all=True)
    monkeypatch.setattr(sr, "get_settings", lambda: settings)

    async def extract(**_kw: Any) -> ext.UnitBundle:
        return ext.UnitBundle(qa=[ext.QA(prompt="why?", outcome="synthetic")])

    monkeypatch.setattr("kb.handlers.claude_code._ext.extract_units_from_session", extract)


async def p3_session(
    customer: str,
    sid: str,
    *,
    with_events: bool = False,
    finalize: bool = True,
    tamper: bool = False,
    leak: bool = False,
    drop_events_of: int | None = None,
) -> list[dict[str, Any]]:
    from engine.shared.storage import get_store

    batches = session_batches(
        EVENTS, protocol=3, sid=sid, with_events=with_events, finalize=finalize
    )
    if drop_events_of is not None:
        # A batch the tap sent without its events (too large to carry both).
        del batches[drop_events_of]["events"]
    if tamper:
        batches[0]["fragments"][0]["line"]["text"] = "USER: TAMPERED"
        batches[0]["fragments"][0]["message"] = "TAMPERED"
    if leak:
        for body in batches:
            for event in body.get("events") or []:
                event["raw"]["toolUseResult"] = {"stdout": LEAK}
    for body in batches:
        result = await sr.accept(body, customer, CC, get_store())
        assert result["status"] == "accepted", result
    await _mine(customer, CC, sid)
    return batches


async def _replay(capsys: pytest.CaptureFixture[str], argv: list[str]) -> list[dict[str, Any]]:
    args = atif_sessions._parse_args(argv)
    await (atif_sessions.replay(args) if args.command == "replay" else atif_sessions.backfill(args))
    return [json.loads(x) for x in capsys.readouterr().out.splitlines() if x.startswith('{"kind"')]


# -- scripts/atif_sessions.py ----------------------------------------------------------


@pytest.mark.asyncio
async def test_replay_reads_protocol_3_from_its_fragments_and_passes_both_gates(
    env,  # noqa: F811
    capsys,
) -> None:
    (a, _b), _store = env
    canary, plain, v2 = str(uuid4()), str(uuid4()), str(uuid4())
    await p3_session(a, canary, with_events=True)
    await p3_session(a, plain)
    await v2_session(a, v2)
    records = await _replay(
        capsys,
        [
            "replay",
            "--customer",
            a,
            "--sample",
            "10",
            "--batchwise",
            "3",
            "--points",
            "3",
            "--compare-builders",
        ],
    )
    sessions = {r["session_id"]: r for r in records if r["kind"] == "session"}
    summary = records[-1]
    for sid in (canary, plain):
        record = sessions[sid]
        assert record["protocol"] == 3 and record["same"] and record["segments_same"], record
        assert record["events"] == len(EVENTS) and record["fragments_degraded"] is False
        assert record["stored"] == "final" and record["stored_fold_same"] is True, record
        assert record["batchwise"]["prefix_disagreements"] == 0
    assert sessions[canary]["builders_same"] is True and sessions[canary]["fragments_same"] is True
    assert sessions[canary]["stored_reference_same"] is True
    assert sessions[plain]["builders_skipped"] == "no_events"
    assert sessions[plain]["stored_reference_same"] is None
    assert "protocol" not in sessions[v2]
    assert summary["gate_passed"] is True and summary["uploaded"] == 2, summary
    assert summary["builders_gate_passed"] is True, summary
    assert (summary["fragments_compared"], summary["fragments_identical"]) == (1, 1)
    assert summary["builders_skipped"] == 1
    assert summary["stored"] == {"final": 3} and summary["stored_fold_identical"] == 3


@pytest.mark.asyncio
async def test_replay_names_a_client_fragment_that_disagrees_without_content(
    env,  # noqa: F811
    capsys,
) -> None:
    (a, _b), _store = env
    sid = str(uuid4())
    await p3_session(a, sid, with_events=True, tamper=True)
    records = await _replay(
        capsys,
        ["replay", "--customer", a, "--sample", "5", "--batchwise", "0", "--compare-builders"],
    )
    [record] = [r for r in records if r["kind"] == "session"]
    summary = records[-1]
    assert record["fragments_same"] is False
    assert record["fragments_diff"] == {"ordinal": 0, "path": "line.text", "kind": "value"}
    assert record["builders_same"] is False
    # The served Lines are the client's and fold renders them back: the render gate holds.
    assert record["same"] is True and summary["gate_passed"] is True
    assert summary["builders_gate_passed"] is False
    assert "TAMPERED" not in json.dumps(records)


@pytest.mark.asyncio
async def test_replay_compares_only_the_ordinals_the_canary_events_cover(
    env,  # noqa: F811
    capsys,
) -> None:
    (a, _b), _store = env
    sid = str(uuid4())
    batches = await p3_session(a, sid, with_events=True, drop_events_of=1)
    records = await _replay(
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
            "3",
            "--compare-builders",
        ],
    )
    [record] = [r for r in records if r["kind"] == "session"]
    missing = batches[1]["event_end"] - batches[1]["event_start"]
    assert record["events_missing"] == missing > 0
    assert (record["builders_same"], record["fragments_same"]) == (True, True), record
    # The stored copy is the fold of every fragment; partial events cannot reproduce it.
    assert (record["stored_fold_same"], record["stored_reference_same"]) == (True, None)
    assert record["batchwise"]["prefix_builder_disagreements"] == 0
    assert records[-1]["builders_gate_passed"] is True and records[-1]["events_missing"] == missing


@pytest.mark.asyncio
async def test_backfill_writes_the_fold_of_the_stored_fragments(env, capsys) -> None:  # noqa: F811
    (a, _b), store = env
    sid = str(uuid4())
    await p3_session(a, sid)
    bucket = await store.bucket_for(a)
    key = trajectory_key(CC.value, a, sid)
    written_by_worker = await store.get(bucket, key)
    await store.delete(bucket, key)  # as if it ended before the engine wrote trajectories
    from engine.shared import db as db_module

    async with db_module.with_tenant(a) as conn:
        await conn.execute("UPDATE ingestion_queue SET status = 'done' WHERE customer_id = $1", a)
    dry = await _replay(capsys, ["backfill", "--customer", a])
    assert dry[-1]["would_write"] == 1
    done = await _replay(capsys, ["backfill", "--customer", a, "--write"])
    assert done[-1]["written"] == 1, done
    assert orjson.loads(await store.get(bucket, key)) == orjson.loads(written_by_worker)


# -- scripts/strip_session_payloads.py ---------------------------------------------------


async def _bodies(store: Any, customer: str) -> dict[str, dict[str, Any]]:
    bucket = await store.bucket_for(customer)
    keys = await store.list_keys(bucket, f"raw/{CC.value}/{customer}/sessions-v2/")
    return {k: orjson.loads(await store.get(bucket, k))["payload"] for k in keys}


async def _strip(capsys: pytest.CaptureFixture[str], argv: list[str]) -> dict[str, Any]:
    await strip_mod.strip(strip_mod._parse_args(argv))
    out = [json.loads(x) for x in capsys.readouterr().out.splitlines() if x.startswith('{"kind"')]
    return out[-1]


@pytest.mark.asyncio
async def test_strip_projects_the_canary_events_and_never_touches_fragments(
    env,  # noqa: F811
    capsys,
) -> None:
    (a, _b), store = env
    await p3_session(a, str(uuid4()), with_events=True, leak=True)
    await p3_session(a, str(uuid4()))  # no canary events: nothing to strip
    before = await _bodies(store, a)
    assert any(LEAK in json.dumps(p) for p in before.values())
    summary = await _strip(capsys, ["strip", "--customer", a, "--write"])
    with_events = sum(1 for p in before.values() if p.get("events"))
    assert summary["written"] == summary["stripped"] == with_events, summary
    # Batches without events: the second session's, and both finalizes.
    assert summary["no_events"] == len(before) - with_events
    after = await _bodies(store, a)
    assert set(after) == set(before)
    for key, payload in after.items():
        assert payload.get("fragments") == before[key].get("fragments"), key
        assert LEAK not in json.dumps(payload)
        events = payload.get("events") or []
        assert lines_from_events(events) == lines_from_events(before[key].get("events") or [])


@pytest.mark.asyncio
async def test_strip_leaves_canary_events_whose_fragments_would_change(
    env,  # noqa: F811
    capsys,
    monkeypatch,
) -> None:
    """The canary's events are the evidence the fragment shadow compares the
    client's fragments with: a projection that changes what they fragment to
    (here: drops their timestamps, which no Line prints) is refused."""
    (a, _b), store = env
    await p3_session(a, str(uuid4()), with_events=True, leak=True)
    before = await _bodies(store, a)
    real = strip_mod.project_event
    monkeypatch.setattr(
        strip_mod,
        "project_event",
        lambda raw: {k: v for k, v in real(raw).items() if k != "timestamp"},
    )
    summary = await _strip(capsys, ["strip", "--customer", a, "--write"])
    assert summary.get("fragments_changed", 0) >= 1 and "written" not in summary, summary
    assert await _bodies(store, a) == before


def test_a_protocol_3_batch_without_events_has_nothing_to_strip() -> None:
    body = session_batches(EVENTS, protocol=3, sid=str(uuid4()))[0]
    new_body, outcome, saved, dropped = strip_mod.strip_batch(orjson.dumps({"payload": body}))
    assert (new_body, outcome, saved, dropped) == (None, "no_events", 0, [])


# -- scripts/rechunk_collapsed_sessions.py ------------------------------------------------


@pytest.mark.asyncio
async def test_rechunk_renders_a_protocol_3_body_from_its_fragments(env) -> None:  # noqa: F811
    (a, _b), store = env
    sid = str(uuid4())
    await p3_session(a, sid)
    from engine.shared import db as db_module

    async with db_module.with_tenant(a) as conn:
        keys = await conn.fetchval(
            "SELECT payload_s3_keys FROM ingestion_queue WHERE customer_id=$1 "
            "AND source_event_id=$2",
            a,
            sid,
        )
    normalizer = Normalizer(make_default_context())
    doc = await rechunk._render(normalizer, store, a, CC, list(keys))
    assert doc.body == render_lines(lines_from_events(_numbered(EVENTS)))
    assert doc.metadata["event_count"] == len(EVENTS)
