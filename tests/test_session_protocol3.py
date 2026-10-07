"""Protocol 3 through the real door, the stored object and the worker's pass.

A protocol-3 client uploads one ATIF fragment per event (engine/ingest/atif/
fragment.py) instead of the events. What must hold:

  * the same events uploaded as protocol 2 and as protocol 3 index the same
    text, evidence spans and extraction input, and store the same
    trajectory.json, for every fixture session: the four harness goldens, the
    builder's CASES, the real-journal batches and the ingest pass's EVENTS;
  * a protocol-3 session runs the live throttle, the end-of-turn bypass, the
    completing pass and its finalize as protocol 2 does, and its author is
    recorded unverified exactly as protocol 2's is;
  * the fragment shadow (SESSION_FRAGMENT_SHADOW) logs one content-free line:
    agreement when the canary's events match, the first differing path when a
    client fragment does not -- and the client's Lines are served regardless,
    because the fragment is the client's word and the events only evidence;
  * no fragment can fail a pass, and every Line keeps its ordinal.

The fragments are REAL fragment() outputs of the fixture events: what a tap
running the vendored fragment.py sends. Runs on the receipt suite's dedicated
database (PRBE_RECEIPT_TEST_DATABASE_URL; its fixture drops the schema).
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
from collections import OrderedDict
from typing import Any
from uuid import uuid4

import orjson
import pytest
from structlog.testing import capture_logs

from engine.ingest.atif import compare as compare_mod
from engine.ingest.atif import uploaded
from engine.ingest.atif.fragment import FRAGMENT_VERSION, fragment
from engine.ingest.atif.mode import render_mode
from engine.ingest.atif.store import trajectory_key
from engine.ingest.handlers.base import make_default_context
from engine.ingest.normalizer import Normalizer
from engine.shared import claude_code_extraction as ext
from engine.shared.config import Settings
from engine.shared.constants import EdgeType, SourceSystem
from engine.shared.exceptions import StorageNotFound
from engine.shared.session_signals import SessionProtocol
from engine.shared.transcript_render import Line, lines_from_events, render_lines_indexed
from kb import session_receipts as sr
from kb.handlers import claude_code as cc_mod
from kb.handlers.claude_code import ClaudeCodeConnector
from kb.session_deletion import SessionRef
from tests.test_atif_build import CASES as BUILD_CASES
from tests.test_atif_build import FIXTURES as JOURNAL_FIXTURES
from tests.test_atif_ingest import EVENTS as INGEST_EVENTS
from tests.test_probe_events_project import GOLDENS, _events
from tests.test_session_receipts import database  # noqa: F401  # pytest fixture, used by name

P2, P3 = "tenant-a", "tenant-b"
IDENTITY = {
    "employee_id": "uploader-1",
    "employee_name": "Up Loader",
    "employee_email": "uploader@example.test",
    "employee_hostname": "upload-host",
    "device_id": "upload-device",
}


def _golden(path: Any) -> list[dict[str, Any]]:
    return [{"line_no": n, "raw": raw} for n, raw in enumerate(_events(path))]


def _journal(source: str) -> list[dict[str, Any]]:
    fixture = json.loads((JOURNAL_FIXTURES / f"{source}.json").read_text())
    return [e for batch in fixture["batches"] for e in batch.get("events") or []]


#: A session document previews its first event's top-level `content`, which
#: none of the sessions above starts with.
PREVIEWED: dict[str, list[dict[str, Any]]] = {
    "system first": [
        {"raw": {"type": "system", "subtype": "init", "content": "session started in /work"}},
        {"raw": {"type": "user", "message": {"role": "user", "content": "hi"}}},
    ],
    "other type first": [
        {"raw": {"type": "summary", "content": "an earlier session, summarised"}},
        {"raw": {"type": "user", "message": {"role": "user", "content": "go on"}}},
    ],
}

#: name -> (events, source). Every one is renderable and has object events.
SESSIONS: dict[str, tuple[list[dict[str, Any]], str]] = {
    **{f"golden:{p.name}": (_golden(p), p.name.split(".")[0]) for p in GOLDENS},
    **{f"build:{name}": (events, "claude_code") for name, events in BUILD_CASES.items()},
    **{f"journal:{s}": (_journal(s), s) for s in ("claude_code", "codex", "pi")},
    "ingest:EVENTS": (INGEST_EVENTS, "claude_code"),
    **{f"preview:{name}": (events, "claude_code") for name, events in PREVIEWED.items()},
}
NAMES = sorted(SESSIONS)


def _numbered(events: list[Any]) -> list[dict[str, Any]]:
    """As the door requires: object events, ordinals 0..n-1."""
    return [dict(e, line_no=i) for i, e in enumerate(e for e in events if isinstance(e, dict))]


def session_batches(
    events: list[Any],
    *,
    protocol: int,
    sid: str,
    cuts: list[int] | None = None,
    with_events: bool = False,
    finalize: bool = True,
) -> list[dict[str, Any]]:
    """A session's batches as a tap sends them: the events split at `cuts`,
    then (unless `finalize` is False) the finalize that certifies the cursor."""
    events = _numbered(events)
    n = len(events)
    if cuts is None:
        cuts = sorted({n // 3, 2 * n // 3} - {0, n})
    bounds = [0, *cuts, n]
    stream = str(uuid4())
    prefix = hashlib.sha256(b"").hexdigest()
    out: list[dict[str, Any]] = []
    for seq, (a, b) in enumerate(itertools.pairwise(bounds)):
        prefix = hashlib.sha256(f"{sid}:{seq}".encode()).hexdigest()
        body: dict[str, Any] = dict(
            protocol_version=protocol,
            session_id=sid,
            stream_id=stream,
            batch_seq=seq,
            source_byte_start=a * 100,
            source_byte_end=b * 100,
            source_line_start=a,
            source_line_end=b,
            event_start=a,
            event_end=b,
            prefix_sha256=prefix,
            cwd="/work",
            **IDENTITY,
        )
        chunk = copy.deepcopy(events[a:b])
        if protocol == SessionProtocol.FRAGMENTS:
            body["fragments"] = [fragment(e) for e in chunk]
            body["fragment_version"] = FRAGMENT_VERSION
            if with_events:
                body["events"] = chunk
        else:
            body["events"] = chunk
        out.append(body)
    if finalize:
        end: dict[str, Any] = dict(
            protocol_version=protocol,
            session_id=sid,
            stream_id=stream,
            batch_seq=len(out),
            source_byte_start=n * 100,
            source_byte_end=n * 100,
            source_line_start=n,
            source_line_end=n,
            event_start=n,
            event_end=n,
            prefix_sha256=prefix,
            finalize=True,
            **IDENTITY,
        )
        if protocol == SessionProtocol.FRAGMENTS:
            end["fragment_version"] = FRAGMENT_VERSION
        out.append(end)
    return out


class Store:
    """The production store's surface, in memory; a bucket per customer."""

    def __init__(self) -> None:
        self.blobs: dict[tuple[str, str], bytes] = {}

    async def bucket_for(self, customer: str) -> str:
        return customer

    async def ensure_bucket(self, bucket: str) -> None:
        pass

    async def put(self, bucket: str, key: str, body: bytes, content_type: str = "") -> None:
        self.blobs[bucket, key] = body

    async def get(self, bucket: str, key: str) -> bytes:
        try:
            return self.blobs[bucket, key]
        except KeyError:
            raise StorageNotFound(key) from None

    async def delete(self, bucket: str, key: str) -> None:
        self.blobs.pop((bucket, key), None)


class Worker:
    """The door and the worker's pass, with the model call recorded."""

    def __init__(self, admin: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        self.admin = admin
        self.monkeypatch = monkeypatch
        self.store = Store()
        self.mined: list[dict[str, Any]] = []
        self.normalizer = Normalizer(make_default_context(), store=self.store, embedder=object())

        async def token(*_a: Any) -> None:
            return None

        async def extract(**kw: Any) -> ext.UnitBundle:
            self.mined.append(kw)
            return ext.UnitBundle(qa=[ext.QA(prompt="why?", outcome="synthetic")])

        async def no_cache(**_kw: Any) -> None:
            return None

        async def not_deleted(*_a: Any) -> bool:
            return False

        monkeypatch.setattr(self.normalizer, "_load_token", token)
        monkeypatch.setattr(cc_mod, "get_store", lambda: self.store)
        monkeypatch.setattr(cc_mod._ext, "extract_units_from_session", extract)
        monkeypatch.setattr(cc_mod._ext_cache.SegmentCache, "for_session", no_cache)
        monkeypatch.setattr(cc_mod, "is_session_deleted", not_deleted)
        self.configure()

    def configure(self, **values: Any) -> Settings:
        """One Settings object for the door and the worker. Protocol 3 is
        advertised to P3 only; the segment cache is off (nothing is mined)."""
        settings = Settings(
            **{
                "session_protocol3_customers": P3,
                "claude_code_extraction_segment_cache": False,
                **values,
            }
        )
        self.monkeypatch.setattr(sr, "get_settings", lambda: settings)
        self.monkeypatch.setattr(cc_mod, "get_settings", lambda: settings)
        self.monkeypatch.setattr(
            cc_mod, "render_mode", lambda customer: render_mode(customer, settings)
        )
        return settings

    async def upload(self, customer: str, source: str, batches: list[dict[str, Any]]) -> None:
        for body in batches:
            result = await sr.accept(body, customer, SourceSystem(source), self.store)
            assert result["status"] == "accepted", result

    async def keys(self, customer: str, source: str, sid: str) -> list[str]:
        return list(
            await self.admin.fetchval(
                "SELECT payload_s3_keys FROM ingestion_queue WHERE customer_id=$1 "
                "AND source_system=$2 AND source_event_id=$3",
                customer,
                source,
                sid,
            )
        )

    async def run(self, customer: str, source: str, sid: str) -> Any:
        """The worker's pass over the session's queue row as it is now."""
        keys = await self.keys(customer, source, sid)
        return await self.normalizer._normalize_only(customer, SourceSystem(source), keys)

    def trajectory(self, customer: str, source: str, sid: str) -> bytes | None:
        return self.store.blobs.get((customer, trajectory_key(source, customer, sid)))


@pytest.fixture
def worker(database, monkeypatch: pytest.MonkeyPatch) -> Worker:  # noqa: F811
    _tenant, admin = database
    return Worker(admin, monkeypatch)


@pytest.fixture(autouse=True)
def _fresh_live_throttle(monkeypatch: pytest.MonkeyPatch) -> None:
    for table in ("_LIVE_WRITES", "_LIVE_TOO_LARGE", "_LIVE_TURNS"):
        monkeypatch.setattr(cc_mod, table, OrderedDict())


def _served(kw: dict[str, Any]) -> list[Line]:
    """The Lines extraction segments: what it was handed, or its events' Lines."""
    if kw.get("lines") is not None:
        return list(kw["lines"])
    return lines_from_events(kw["events"])


def _extraction_input(lines: list[Line]) -> list[tuple[Any, ...]]:
    """Per segment exactly what the model would be sent (claude_code_extraction
    `_extract_one`): boundary, rendered transcript, its spans, line bounds."""
    segments, capped = ext._segment_session(lines)
    out = []
    for segment, boundary in segments:
        kept = segment if capped else [x for x in segment if not ext._is_compact_summary(x)]
        transcript, spans = render_lines_indexed(ext._as_lines(kept))
        out.append((boundary, transcript, spans, *ext._line_bounds(segment)))
    return out


def _metadata(doc: Any) -> dict[str, Any]:
    return {k: v for k, v in doc.metadata.items() if k != "protocol_version"}


# -- the same events on either protocol ---------------------------------------------


def test_every_kind_of_fixture_session_is_here() -> None:
    assert len(GOLDENS) == 4 and len(SESSIONS) == 4 + len(BUILD_CASES) + 3 + 1 + len(PREVIEWED)
    assert {s for _e, s in SESSIONS.values()} == {"claude_code", "codex", "pi", "kimi_code"}


@pytest.mark.asyncio
@pytest.mark.parametrize("name", NAMES)
async def test_the_same_events_on_either_protocol_index_mine_and_store_the_same(
    worker: Worker, name: str
) -> None:
    events, source = SESSIONS[name]
    sid = str(uuid4())
    # The same session id in two tenants, so the documents may be compared byte for byte.
    await worker.upload(P2, source, session_batches(events, protocol=2, sid=sid))
    await worker.upload(P3, source, session_batches(events, protocol=3, sid=sid))
    v2 = await worker.run(P2, source, sid)
    mined_v2 = worker.mined[-1]
    v3 = await worker.run(P3, source, sid)
    mined_v3 = worker.mined[-1]

    doc2, doc3 = v2.documents[0], v3.documents[0]
    assert doc2.metadata["session_complete"] and doc3.metadata["session_complete"]
    # The indexed text and everything derived from it.
    assert doc3.body == doc2.body
    assert (doc3.content_hash, doc3.body_size_bytes) == (doc2.content_hash, doc2.body_size_bytes)
    assert (doc3.title, doc3.body_preview) == (doc2.title, doc2.body_preview)
    if name.startswith("preview:"):
        assert doc3.body_preview == SESSIONS[name][0][0]["raw"]["content"]
    assert _metadata(doc3) == _metadata(doc2)
    assert (doc2.metadata["protocol_version"], doc3.metadata["protocol_version"]) == (2, 3)
    # Extraction: the same Lines, so the same segments, transcripts and spans.
    assert "lines" in mined_v3, "protocol 3 hands extraction the fragments' Lines"
    lines2, lines3 = _served(mined_v2), _served(mined_v3)
    assert lines3 == lines2
    assert render_lines_indexed(lines3) == render_lines_indexed(lines2)
    assert _extraction_input(lines3) == _extraction_input(lines2)
    assert [d.body for d in v3.documents[1:]] == [d.body for d in v2.documents[1:]]
    outcome2, outcome3 = v2.extraction_outcome, v3.extraction_outcome
    for key in ("authoritative", "reason", "units", "keys", "completed_by"):
        assert outcome3[key] == outcome2[key], key
    # The trajectory: the final copies are the same bytes (no build stamp
    # differs: `extra.session_ended` is true on both, the rest is the build).
    stored2, stored3 = worker.trajectory(P2, source, sid), worker.trajectory(P3, source, sid)
    assert stored2 is not None and stored3 == stored2
    assert orjson.loads(stored3)["extra"]["session_ended"] is True


# -- the session's life on protocol 3 -----------------------------------------------


@pytest.mark.asyncio
async def test_a_finalize_ends_the_session_and_a_later_batch_reopens_it(worker: Worker) -> None:
    events, source = SESSIONS["golden:claude_code.expected.jsonl"]
    sid = str(uuid4())
    batches = session_batches(events, protocol=3, sid=sid)
    await worker.upload(P3, source, batches[:-1])
    live = await worker.run(P3, source, sid)
    assert live.documents[0].metadata["session_complete"] is False and len(live.documents) == 1
    assert live.extraction_outcome is None, "a live pass never mines"
    await worker.upload(P3, source, batches[-1:])
    ended = await worker.run(P3, source, sid)
    doc = ended.documents[0]
    assert doc.metadata["session_complete"] is True
    assert doc.metadata["completed_by"] == "v2_finalize"
    assert doc.metadata["event_count"] == len(events) and len(ended.documents) > 1
    # A resume: the stream continues past its finalize, and the session is live again.
    final = batches[-1]
    n = final["event_end"]
    tail = dict(
        final,
        batch_seq=final["batch_seq"] + 1,
        finalize=False,
        event_end=n + 1,
        source_byte_end=final["source_byte_end"] + 10,
        source_line_end=n + 1,
        prefix_sha256="c" * 64,
        fragments=[
            fragment({"line_no": n, "raw": {"type": "user", "message": {"content": "more"}}})
        ],
    )
    del tail["finalize"]
    await worker.upload(P3, source, [tail])
    reopened = await worker.run(P3, source, sid)
    assert reopened.documents[0].metadata["session_complete"] is False
    assert reopened.documents[0].metadata["event_count"] == len(events) + 1
    assert reopened.documents[0].body.endswith("USER: more")


@pytest.mark.asyncio
async def test_the_uploader_is_recorded_and_never_credited_as_the_author(worker: Worker) -> None:
    events, source = SESSIONS["golden:pi.expected.jsonl"]
    sid = str(uuid4())
    batches = session_batches(events, protocol=3, sid=sid)
    batches[0]["provenance"] = {"original_author": "unverified", "native_session_id": sid}
    await worker.upload(P3, source, batches)
    result = await worker.run(P3, source, sid)
    assert len(result.documents) > 1
    assert not any(edge.edge_type == EdgeType.AUTHORED for edge in result.graph_edges)
    for doc in result.documents:
        assert doc.author_id is None and "Up Loader" not in doc.title
        assert doc.metadata["protocol_version"] == 3
        assert doc.metadata["author_verification"] == "unverified"
        assert doc.metadata["native_session_id"] == sid
        assert (doc.metadata["uploader_id"], doc.metadata["uploader_device_id"]) == (
            "uploader-1",
            "upload-device",
        )
        assert not any(key.startswith("employee_") for key in doc.metadata)


@pytest.mark.asyncio
async def test_a_protocol_3_batch_lands_where_deletion_and_the_sweep_look(worker: Worker) -> None:
    """Deletion, the purge and the idle sweep find a session's batches by key
    (`sessions-v2/<sid>/`) and its stream row: protocol 3 changes neither."""
    events, source = SESSIONS["golden:codex.expected.jsonl"]
    sid = str(uuid4())
    await worker.upload(P3, source, session_batches(events, protocol=3, sid=sid))
    prefix = SessionRef(source=source, session_id=sid).v2_prefix(P3)
    keys = await worker.keys(P3, source, sid)
    assert keys and all(key.startswith(prefix) for key in keys)
    assert await worker.admin.fetchval(
        "SELECT finalized FROM session_streams WHERE customer_id=$1 AND session_id=$2", P3, sid
    )


# -- the live trajectory --------------------------------------------------------------


def _spy(monkeypatch: pytest.MonkeyPatch, name: str) -> list[Any]:
    calls: list[Any] = []
    real = getattr(cc_mod, name)

    def spy(*args: Any) -> Any:
        calls.append(args)
        return real(*args)

    monkeypatch.setattr(cc_mod, name, spy)
    return calls


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "cuts", "expected"),
    [
        # pi: 10 ends a turn (`_pi_extras.stop_reason` "stop"), 11 is the next prompt.
        ("golden:pi.expected.jsonl", [8, 11, 12], [1, 2, 2, 2]),
        # Codex: 3 is a `final_answer`, 4 the next prompt, 13 the last answer.
        ("golden:codex.expected.jsonl", [2, 4, 5], [1, 2, 2, 3]),
    ],
)
async def test_live_copies_are_throttled_and_a_turn_end_is_written_at_once(
    worker: Worker, monkeypatch: pytest.MonkeyPatch, name: str, cuts: list[int], expected: list[int]
) -> None:
    """Per live pass, how many builds have run: the first write, then a turn
    end inside the interval, then nothing until the session ends -- the same
    pacing protocol 2 has for the same events."""
    worker.configure(session_trajectory_live_interval_s=3600)
    folds = _spy(monkeypatch, "fold_fragments")
    builds = _spy(monkeypatch, "build_and_render")
    events, source = SESSIONS[name]
    pacing: dict[int, list[int]] = {2: [], 3: []}
    for protocol, customer in ((2, P2), (3, P3)):
        sid = str(uuid4())
        batches = session_batches(events, protocol=protocol, sid=sid, cuts=cuts)
        calls = folds if protocol == 3 else builds
        for body in batches[:-1]:
            await worker.upload(customer, source, [body])
            await worker.run(customer, source, sid)
            pacing[protocol].append(len(calls))
            stored = worker.trajectory(customer, source, sid)
            assert stored is not None and orjson.loads(stored)["extra"] == {"session_ended": False}
        await worker.upload(customer, source, batches[-1:])
        await worker.run(customer, source, sid)
        assert len(calls) == expected[-1] + 1, "the completing pass always builds"
        final = orjson.loads(worker.trajectory(customer, source, sid))
        assert final["extra"] == {"session_ended": True}
    assert pacing == {2: expected, 3: expected}
    assert all(args[0] for args in folds), "protocol 3 folds the client's fragments"


@pytest.mark.asyncio
async def test_claude_codes_end_turn_is_not_in_a_version_1_fragment(
    worker: Worker, monkeypatch: pytest.MonkeyPatch
) -> None:
    """KNOWN GAP (engine/ingest/atif/fragment.py keeps a stop reason only when
    the index prints it, and `end_turn` never prints): protocol 2 writes a
    Claude Code turn end live at once, protocol 3 waits for the interval. When
    a fragment version carries every stop reason, this flips."""
    worker.configure(session_trajectory_live_interval_s=3600)
    folds = _spy(monkeypatch, "fold_fragments")
    builds = _spy(monkeypatch, "build_and_render")
    events, source = SESSIONS["golden:claude_code.expected.jsonl"]
    for protocol, customer in ((2, P2), (3, P3)):
        sid = str(uuid4())
        # 11 is the assistant's `end_turn`; 12 a system note after it.
        for body in session_batches(
            events, protocol=protocol, sid=sid, cuts=[8, 13], finalize=False
        )[:2]:
            await worker.upload(customer, source, [body])
            await worker.run(customer, source, sid)
    assert (len(builds), len(folds)) == (2, 1)


# -- the fragment shadow -------------------------------------------------------------


def _compared(logs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [e for e in logs if e["event"] == "session_fragments.compared"]


@pytest.mark.asyncio
@pytest.mark.parametrize("name", [n for n in NAMES if n.startswith(("golden:", "journal:"))])
async def test_the_shadow_agrees_when_the_canary_events_match(worker: Worker, name: str) -> None:
    events, source = SESSIONS[name]
    sid = str(uuid4())
    await worker.upload(P3, source, session_batches(events, protocol=3, sid=sid, with_events=True))
    with capture_logs() as logs:
        await worker.run(P3, source, sid)
    [line] = _compared(logs)
    n = len(_numbered(events))
    assert line["log_level"] == "info"
    assert (line["protocol"], line["source"], line["session_id"]) == (3, source, sid)
    assert (line["same_fragments"], line["same_trajectory"]) == (True, True)
    assert (line["fragments_diff"], line["trajectory_diff"], line["fragments_differing"]) == (
        None,
        None,
        0,
    )
    assert (line["fragments"], line["events"]) == (n, n)
    assert line["steps"] == line["reference_steps"] > 0


@pytest.mark.asyncio
async def test_a_tampered_client_fragment_is_named_by_path_and_still_served(
    worker: Worker,
) -> None:
    """The client's fragment IS what the session is read from: the shadow only
    says, without content, that it disagrees with the canary's events."""
    events, source = SESSIONS["golden:claude_code.expected.jsonl"]
    sid = str(uuid4())
    batches = session_batches(events, protocol=3, sid=sid, with_events=True)
    tampered = batches[0]["fragments"][0]
    assert tampered["kind"] == "user" and tampered["line"]["line_no"] == 0
    tampered["line"]["text"] = "USER: TAMPERED-LINE"
    tampered["message"] = "TAMPERED-MESSAGE"
    await worker.upload(P3, source, batches)
    with capture_logs() as logs:
        result = await worker.run(P3, source, sid)
    [line] = _compared(logs)
    assert line["log_level"] == "warning"
    assert line["same_fragments"] is False and line["fragments_differing"] == 1
    assert line["fragments_diff"] == {"ordinal": 0, "path": "line.text", "kind": "value"}
    assert line["same_trajectory"] is False
    assert line["trajectory_diff"]["path"] == "trajectory.steps[0].message"
    assert "TAMPERED" not in json.dumps(_compared(logs), default=str)
    assert result.documents[0].body.startswith("USER: TAMPERED-LINE")


@pytest.mark.asyncio
async def test_a_protocol_2_session_compares_the_fold_with_the_frozen_builder(
    worker: Worker,
) -> None:
    events, source = SESSIONS["golden:kimi_code.expected.jsonl"]
    for builder in ("reference", "fold"):
        worker.configure(session_atif_builder=builder)
        sid = str(uuid4())
        await worker.upload(P2, source, session_batches(events, protocol=2, sid=sid))
        with capture_logs() as logs:
            await worker.run(P2, source, sid)
        [line] = _compared(logs)
        assert (line["protocol"], line["same_fragments"], line["same_trajectory"]) == (
            2,
            None,
            True,
        )
        assert line["events"] == len(events) and "fragments" not in line


@pytest.mark.asyncio
async def test_the_shadow_reuses_the_pass_s_build_and_computes_only_the_other_side(
    worker: Worker, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[Any] = []
    real = compare_mod.shadow

    def spy(events: Any, fragments: Any, built: Any, *rest: Any) -> Any:
        seen.append((fragments is None, built is not None))
        return real(events, fragments, built, *rest)

    monkeypatch.setattr(compare_mod, "shadow", spy)
    events, source = SESSIONS["golden:pi.expected.jsonl"]
    sid = str(uuid4())
    await worker.upload(P2, source, session_batches(events, protocol=2, sid=sid))
    await worker.upload(P3, source, session_batches(events, protocol=3, sid=sid, with_events=True))
    await worker.run(P2, source, sid)
    await worker.run(P3, source, sid)
    assert seen == [(True, True), (False, True)]


@pytest.mark.asyncio
async def test_no_shadow_without_events_on_a_live_pass_or_when_switched_off(
    worker: Worker,
) -> None:
    events, source = SESSIONS["golden:codex.expected.jsonl"]
    no_events, live, off = (str(uuid4()) for _ in range(3))
    await worker.upload(P3, source, session_batches(events, protocol=3, sid=no_events))
    await worker.upload(
        P3, source, session_batches(events, protocol=3, sid=live, with_events=True, finalize=False)
    )
    await worker.upload(P3, source, session_batches(events, protocol=3, sid=off, with_events=True))
    with capture_logs() as logs:
        await worker.run(P3, source, no_events)
        await worker.run(P3, source, live)
        worker.configure(session_fragment_shadow=False)
        await worker.run(P3, source, off)
    assert _compared(logs) == []


@pytest.mark.asyncio
async def test_a_failing_comparison_is_logged_and_never_fails_the_pass(
    worker: Worker, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_a: Any) -> Any:
        raise RuntimeError("bug")

    monkeypatch.setattr(compare_mod, "shadow", boom)
    events, source = SESSIONS["golden:pi.expected.jsonl"]
    sid = str(uuid4())
    await worker.upload(P3, source, session_batches(events, protocol=3, sid=sid, with_events=True))
    with capture_logs() as logs:
        result = await worker.run(P3, source, sid)
    [failed] = [e for e in logs if e["event"] == "session_fragments.compare_failed"]
    assert (failed["error"], failed["protocol"]) == ("RuntimeError", 3)
    assert result.documents[0].metadata["session_complete"] and len(result.documents) > 1
    assert worker.trajectory(P3, source, sid) is not None


# -- fragments no pass trusts --------------------------------------------------------


def _line_numbers(lines: list[Line]) -> list[int | None]:
    return [line.line_no for line in lines]


@pytest.mark.asyncio
async def test_a_clients_fallback_fragment_keeps_its_place_and_fails_nothing(
    worker: Worker,
) -> None:
    """research-os #2409's fallback for an event its fragment() raised on:
    `{"v", "line": {"line_no", "text": ""}, "error"}` and no kind. Its Line is
    empty, every later Line keeps its ordinal, and fold counts it unparsed."""
    events, source = SESSIONS["golden:claude_code.expected.jsonl"]
    sid = str(uuid4())
    batches = session_batches(events, protocol=3, sid=sid)
    batches[1]["fragments"][0] = {
        "v": FRAGMENT_VERSION,
        "line": {"line_no": batches[1]["event_start"], "text": ""},
        "error": "KeyError",
    }
    await worker.upload(P3, source, batches)
    with capture_logs() as logs:
        result = await worker.run(P3, source, sid)
    lines = _served(worker.mined[-1])
    assert _line_numbers(lines) == list(range(len(events)))
    assert lines[batches[1]["event_start"]].text == ""
    [degraded] = [e for e in logs if e["event"] == "session_fragments.degraded"]
    assert (degraded["unmapped"], degraded["invalid_lines"], degraded["fragments"]) == (
        1,
        0,
        len(events),
    )
    [unparsed] = [e for e in logs if e["event"] == "atif.event_unparsed"]
    assert (unparsed["line_no"], unparsed["error"]) == (
        batches[1]["event_start"],
        "InvalidFragment",
    )
    stored = orjson.loads(worker.trajectory(P3, source, sid))
    subtypes = [(s.get("extra") or {}).get("subtype") for s in stored["steps"]]
    assert subtypes.count("unparsed") == 1
    assert result.extraction_outcome["authoritative"] is True


@pytest.mark.asyncio
async def test_lines_of_the_wrong_types_are_empty_lines_at_their_ordinal(worker: Worker) -> None:
    events, source = SESSIONS["golden:pi.expected.jsonl"]
    sid = str(uuid4())
    batches = session_batches(events, protocol=3, sid=sid)
    hostile = batches[0]["fragments"]
    hostile[1]["line"]["text"] = 5
    hostile[2]["line"]["user_turn"] = "yes"
    hostile[3]["line"] = {"line_no": 3, "text": ["not", "text"], "compact_boundary": None}
    hostile[4]["kind"] = {"nested": "junk"}
    await worker.upload(P3, source, batches)
    with capture_logs() as logs:
        result = await worker.run(P3, source, sid)
    lines = _served(worker.mined[-1])
    assert _line_numbers(lines) == list(range(len(events)))
    assert [lines[i].text for i in (1, 2, 3)] == ["", "", ""]
    [degraded] = [e for e in logs if e["event"] == "session_fragments.degraded"]
    assert degraded["invalid_lines"] == 3
    assert result.documents[0].metadata["event_count"] == len(events)


@pytest.mark.asyncio
async def test_a_giant_line_is_cut_and_counted(
    worker: Worker, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(uploaded, "FRAGMENT_LINE_MAX_CHARS", 1_000)
    events, source = SESSIONS["golden:codex.expected.jsonl"]
    sid = str(uuid4())
    batches = session_batches(events, protocol=3, sid=sid)
    batches[0]["fragments"][0]["line"]["text"] = "SYSTEM: " + "x" * 5_000
    await worker.upload(P3, source, batches)
    with capture_logs() as logs:
        result = await worker.run(P3, source, sid)
    lines = _served(worker.mined[-1])
    assert len(lines[0].text) == 1_000 and _line_numbers(lines) == list(range(len(events)))
    [degraded] = [e for e in logs if e["event"] == "session_fragments.degraded"]
    assert degraded["truncated"] == 1
    assert result.documents[0].metadata["session_complete"]


def test_the_line_cap_never_cuts_what_the_gateway_lets_through() -> None:
    # research-os app/ingestion/sessions_router.py MAX_BODY_BYTES: one batch.
    assert uploaded.FRAGMENT_LINE_MAX_CHARS >= 2_000_000


@pytest.mark.parametrize(
    "junk",
    [
        None,
        5,
        "text",
        [],
        {},
        {"line": "junk"},
        {"line": {"line_no": "1", "text": ""}},
        {"line": {"line_no": -1, "text": ""}},
        {"line": {"line_no": 2**70, "text": ""}},
        {"line": {"line_no": True, "text": "x"}},
        {"line": {"line_no": 1, "text": "x", "compact_summary": 1}},
    ],
)
def test_any_fragment_reads_as_one_line(junk: Any) -> None:
    read = uploaded.fragment_lines([junk, fragment({"line_no": 9, "raw": {"type": "system"}})])
    assert len(read.lines) == 2 and read.invalid == 1
    assert read.lines[0].text == "" and read.lines[1].line_no == 9


@pytest.mark.parametrize(
    "item",
    [
        {"kind": "assistant", "stop": {"reason": []}, "line": {"line_no": 1}},
        {"kind": "assistant", "stop": [], "extras": {"pi_extras": {"stop_reason": {}}}},
        {"kind": "assistant", "extras": {"codex_extras": {"phase": ["final_answer"]}}},
        {"kind": ["assistant"]},
        {"kind": "assistant", "extras": "junk"},
    ],
)
def test_a_hostile_reply_is_never_a_turn_end_and_never_raises(item: dict[str, Any]) -> None:
    assert cc_mod._turn_end_line([item], fragments=True) is None


def test_an_unhashable_stop_reason_on_an_event_no_longer_raises() -> None:
    events = [{"line_no": 3, "raw": {"type": "assistant", "message": {"stop_reason": []}}}]
    assert cc_mod._turn_end_line(events) is None


@pytest.mark.parametrize(
    ("item", "line"),
    [
        ({"kind": "assistant", "stop": {"seq": 1, "reason": "stop_sequence"}}, 4),
        ({"kind": "assistant", "extras": {"pi_extras": {"stop_reason": "stop"}}}, 4),
        ({"kind": "assistant", "extras": {"codex_extras": {"phase": "final_answer"}}}, 4),
        ({"kind": "assistant", "extras": {"codex_extras": {"phase": "commentary"}}}, None),
        ({"kind": "user", "message": "next"}, None),
    ],
)
def test_which_fragments_end_a_turn(item: dict[str, Any], line: int | None) -> None:
    fragments = [
        dict(item, line={"line_no": 4, "text": ""}),
        {"kind": "system", "line": {"line_no": 5}},
    ]
    assert cc_mod._turn_end_line(fragments, fragments=True) == line


# -- a stored session mixing protocols --------------------------------------------------


@pytest.mark.asyncio
async def test_mixed_protocols_are_each_read_by_their_own_and_logged(
    worker: Worker,
) -> None:
    """The door pins a stream to one protocol; if stored batches mix anyway,
    each batch is read by its own protocol and the Lines keep ordinal order."""
    events, source = SESSIONS["golden:claude_code.expected.jsonl"]
    sid = str(uuid4())
    v3 = session_batches(events, protocol=3, sid=sid, cuts=[10], finalize=False)
    v2 = session_batches(events, protocol=2, sid=sid, cuts=[10], finalize=False)
    keys = [f"raw/{source}/{P3}/sessions-v2/{sid}/{seq}-x.json" for seq in (0, 1)]
    # Out of order on purpose: a protocol-2 batch for the tail, protocol 3 for the head.
    worker.store.blobs[P3, keys[0]] = orjson.dumps({"payload": v3[0]})
    worker.store.blobs[P3, keys[1]] = orjson.dumps({"payload": v2[1]})
    event = sr_event(source, sid, v3[0], list(reversed(keys)))
    connector = ClaudeCodeConnector(make_default_context())
    with capture_logs() as logs:
        hydrated = await connector.fetch_supplementary(event, None)
        result = await connector.normalize(event, hydrated)
    [mixed] = [e for e in logs if e["event"] == "claude_code.mixed_protocols"]
    assert mixed["protocols"] == [2, 3]
    assert hydrated["protocol_version"] == 3 and hydrated["events"] == []
    assert result.documents[0].body == render_lines_indexed(lines_from_events(_numbered(events)))[0]


def sr_event(source: str, sid: str, payload: dict[str, Any], keys: list[str]) -> Any:
    from datetime import UTC, datetime

    from engine.shared.models import WebhookEvent

    return WebhookEvent(
        customer_id=P3,
        source_system=SourceSystem(source),
        source_event_id=sid,
        received_at=datetime.now(UTC),
        payload_s3_key=keys[0],
        payload_s3_keys=keys,
        raw_payload=payload,
        headers={},
    )


# -- parsing a stored protocol-3 batch --------------------------------------------------


def test_the_worker_parses_a_stored_batch_whose_version_is_no_longer_listed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The door accepted it; withdrawing its version later must not fail the
    session (its Lines need no fold; fold reports what it cannot read)."""
    monkeypatch.setattr(sr, "get_settings", lambda: Settings(session_fragment_versions=""))
    body = session_batches(SESSIONS["golden:pi.expected.jsonl"][0], protocol=3, sid=str(uuid4()))[0]
    with pytest.raises(Exception, match="unsupported fragment version"):
        sr.validate_payload(body)
    parsed = ClaudeCodeConnector(make_default_context()).parse_webhook_event("c", {}, body)
    assert parsed is not None and parsed.source_event_id == body["session_id"]


def test_a_cursor_only_protocol_3_batch_still_identifies_its_session() -> None:
    body = session_batches([{"raw": {"type": "system"}}], protocol=3, sid=str(uuid4()))[0]
    body.update(event_end=0, fragments=[], source_line_end=0, source_byte_end=0)
    parsed = ClaudeCodeConnector(make_default_context()).parse_webhook_event("c", {}, body)
    assert parsed is not None


def test_the_protocols_the_worker_reads_are_the_doors() -> None:
    assert (SessionProtocol.EVENTS, SessionProtocol.FRAGMENTS) == sr.SESSION_PROTOCOLS
    assert sr.fragment_ordinal is uploaded.fragment_ordinal
    assert SessionProtocol.of({"protocol_version": 3}) is SessionProtocol.FRAGMENTS
    assert SessionProtocol.of({}) is SessionProtocol.LEGACY
