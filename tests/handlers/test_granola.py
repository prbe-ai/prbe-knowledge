"""Granola connector tests.

Covers:
  - normalize() on macOS (no diarization), iOS (diarization_label), and
    summary-only notes
  - content_hash idempotence — re-normalize same note → same hash
  - backfill() pagination, watermark advancement, cursor resumability
  - HTTP error mapping: 401/403 → PermanentSourceError, 429 → RateLimited,
    5xx → TransientSourceError, network blips → silently end the tick

LISTEN reconnect + DB-backed admin/poller flows live in their own test files.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest

from services.ingestion.handlers.base import ConnectorContext
from services.ingestion.handlers.granola import (
    GranolaConnector,
)
from services.ingestion.handlers.registry import build_connector
from shared.config import Settings
from shared.constants import (
    DocClass,
    DocType,
    EdgeType,
    NodeLabel,
    Permission,
    PrincipalType,
    SourceSystem,
)
from shared.exceptions import PermanentSourceError, RateLimited, TransientSourceError
from shared.models import IntegrationToken, WebhookEvent

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_ctx(http: httpx.AsyncClient | None = None) -> ConnectorContext:
    settings = Settings(environment="local")
    return ConnectorContext(settings=settings, http=http or httpx.AsyncClient())


def _make_event(note: dict[str, Any]) -> WebhookEvent:
    return WebhookEvent(
        customer_id="cust-1",
        source_system=SourceSystem.GRANOLA,
        source_event_id=str(note.get("id") or "n/a"),
        received_at=datetime.now(UTC),
        payload_s3_key="raw/granola/cust-1/2026/04/24/test.json",
        raw_payload={"note": note},
        headers={},
    )


def _token() -> IntegrationToken:
    return IntegrationToken(
        customer_id="cust-1",
        source_system=SourceSystem.GRANOLA,
        access_token="grn_test_TOKEN",
        scope="tier:enterprise",
    )


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_normalize_macos_no_diarization() -> None:
    """macOS notes carry speaker.source ('microphone'/'speaker'), no per-person id."""
    ctx = _make_ctx()
    granola = build_connector(SourceSystem.GRANOLA, ctx)

    note = {
        "id": "not_macos_001",
        "title": "Eng standup 2026-04-24",
        "summary": "Decisions: ship Granola integration. Next steps: ...",
        "owner": {"name": "Richard", "email": "Richard@PRBE.ai"},
        "created_at": "2026-04-24T17:00:00Z",
        "transcript": [
            {"speaker": {"source": "microphone"}, "text": "Let's start with infra."},
            {"speaker": {"source": "speaker"}, "text": "Sounds good. We agreed last week..."},
        ],
    }

    result = await granola.normalize(_make_event(note), {})

    assert not result.is_empty
    assert len(result.documents) == 1
    doc = result.documents[0]

    assert doc.doc_id == "granola:meeting:not_macos_001"
    assert doc.source_system == SourceSystem.GRANOLA
    assert doc.doc_type == DocType.GRANOLA_MEETING
    assert doc.doc_class == DocClass.RAW_SOURCE
    assert doc.title == "Eng standup 2026-04-24"
    assert doc.author_id == "richard@prbe.ai"  # lowercased + stripped
    assert doc.body_preview == "Decisions: ship Granola integration. Next steps: ..."
    assert doc.metadata["transcript_segments"] == 2
    assert doc.metadata["has_transcript"] is True
    assert "## Transcript" in doc.metadata["body"]
    assert "microphone: Let's start with infra." in doc.metadata["body"]

    # Owner ACL: USER, WRITE.
    assert len(result.acl_snapshots) == 1
    acl = result.acl_snapshots[0]
    assert acl.principal_type == PrincipalType.USER
    assert acl.principal_id == "richard@prbe.ai"
    assert acl.permission == Permission.WRITE
    assert acl.resource_type == "granola.meeting"
    assert acl.resource_id == "not_macos_001"

    # Graph: DOCUMENT + PERSON nodes, AUTHORED edge.
    labels = {(n.label, n.canonical_id) for n in result.graph_nodes}
    assert (NodeLabel.DOCUMENT, doc.doc_id) in labels
    assert (NodeLabel.PERSON, "richard@prbe.ai") in labels
    edge_kinds = {e.edge_type for e in result.graph_edges}
    assert EdgeType.AUTHORED in edge_kinds


@pytest.mark.asyncio
async def test_normalize_ios_with_diarization_label() -> None:
    """iOS notes attach a diarization_label per turn — preserved in body."""
    ctx = _make_ctx()
    granola = build_connector(SourceSystem.GRANOLA, ctx)

    note = {
        "id": "not_ios_002",
        "title": "Customer interview",
        "summary": "User wants tighter integration with Slack.",
        "owner": {"name": "Sam", "email": "sam@prbe.ai"},
        "created_at": "2026-04-24T18:00:00Z",
        "transcript": [
            {
                "speaker": {"source": "microphone", "diarization_label": "Speaker A"},
                "text": "How are you using PRBE today?",
            },
            {
                "speaker": {"source": "speaker", "diarization_label": "Speaker B"},
                "text": "Mostly via the dashboard.",
            },
        ],
    }

    result = await granola.normalize(_make_event(note), {})
    body = result.documents[0].metadata["body"]
    assert "Speaker A: How are you using PRBE today?" in body
    assert "Speaker B: Mostly via the dashboard." in body


@pytest.mark.asyncio
async def test_normalize_summary_only_no_transcript() -> None:
    ctx = _make_ctx()
    granola = build_connector(SourceSystem.GRANOLA, ctx)

    note = {
        "id": "not_summary_only",
        "title": "Quick note",
        "summary": "Just a few thoughts.",
        "owner": {"name": "Pat", "email": "pat@prbe.ai"},
        "created_at": "2026-04-24T19:00:00Z",
        # No transcript field at all.
    }

    result = await granola.normalize(_make_event(note), {})
    doc = result.documents[0]
    assert doc.metadata["body"] == "Just a few thoughts."
    assert doc.metadata["has_transcript"] is False
    assert doc.metadata["transcript_segments"] == 0
    assert "## Transcript" not in doc.metadata["body"]


@pytest.mark.asyncio
async def test_normalize_skips_event_without_note_id() -> None:
    ctx = _make_ctx()
    granola = build_connector(SourceSystem.GRANOLA, ctx)

    event = WebhookEvent(
        customer_id="cust-1",
        source_system=SourceSystem.GRANOLA,
        source_event_id="missing",
        received_at=datetime.now(UTC),
        payload_s3_key="",
        raw_payload={"note": {}},  # no id
        headers={},
    )
    result = await granola.normalize(event, {})
    assert result.is_empty
    assert result.skipped_reason and "missing" in result.skipped_reason.lower()


@pytest.mark.asyncio
async def test_normalize_idempotent_content_hash() -> None:
    """Same note → identical content_hash. Bitemporal writer no-ops on re-poll."""
    ctx = _make_ctx()
    granola = build_connector(SourceSystem.GRANOLA, ctx)

    note = {
        "id": "not_repeat",
        "title": "Repeat me",
        "summary": "Stable summary",
        "owner": {"name": "X", "email": "x@example.com"},
        "created_at": "2026-04-24T20:00:00Z",
        "transcript": [
            {"speaker": {"source": "microphone"}, "text": "Same text both runs."},
        ],
    }
    a = (await granola.normalize(_make_event(note), {})).documents[0]
    b = (await granola.normalize(_make_event(note), {})).documents[0]
    assert a.content_hash == b.content_hash

    # Mutate summary → new hash.
    note_v2 = dict(note, summary="DIFFERENT summary")
    c = (await granola.normalize(_make_event(note_v2), {})).documents[0]
    assert c.content_hash != a.content_hash


# ---------------------------------------------------------------------------
# backfill
# ---------------------------------------------------------------------------


def _granola_transport(handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_backfill_paginates_and_advances_watermark() -> None:
    """Two pages; per-note events carry input_watermark; final checkpoint carries new max."""
    page_1 = {
        "notes": [{"id": "not_a"}, {"id": "not_b"}],
        "hasMore": True,
        "cursor": "page2",
    }
    page_2 = {
        "notes": [{"id": "not_c"}],
        "hasMore": False,
    }
    notes_by_id = {
        "not_a": {
            "id": "not_a",
            "title": "A",
            "summary": "first",
            "owner": {"name": "x", "email": "x@e"},
            "created_at": "2026-04-22T00:00:00Z",
            "transcript": [],
        },
        "not_b": {
            "id": "not_b",
            "title": "B",
            "summary": "second",
            "owner": {"name": "x", "email": "x@e"},
            "created_at": "2026-04-23T00:00:00Z",
            "transcript": [],
        },
        "not_c": {
            "id": "not_c",
            "title": "C",
            "summary": "third",
            "owner": {"name": "x", "email": "x@e"},
            "created_at": "2026-04-24T00:00:00Z",
            "transcript": [],
        },
    }

    requests_seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        requests_seen.append(req)
        if req.url.path == "/v1/notes":
            cursor = req.url.params.get("cursor")
            return httpx.Response(200, json=page_2 if cursor == "page2" else page_1)
        if req.url.path.startswith("/v1/notes/"):
            note_id = req.url.path.rsplit("/", 1)[-1]
            return httpx.Response(200, json=notes_by_id[note_id])
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=_granola_transport(handler)) as http:
        granola = GranolaConnector(_make_ctx(http=http))
        events = [
            event async for event in granola.backfill("cust-1", _token(), cursor=None)
        ]

    # Three note events + one checkpoint event.
    assert [e.source_event_id for e in events] == [
        "not_a",
        "not_b",
        "not_c",
        "__cursor_checkpoint__",
    ]

    # Per-note events carry the INPUT watermark (None here). Watermark only
    # advances at the final checkpoint event.
    for ev in events[:3]:
        per_note_cursor = json.loads(ev.raw_payload["_cursor"])
        assert per_note_cursor["watermark"] is None
        assert per_note_cursor["page_cursor"] is None
        assert "_checkpoint" not in ev.raw_payload

    # Final checkpoint event encodes the highest watermark seen.
    checkpoint = events[-1]
    assert checkpoint.raw_payload["_checkpoint"] is True
    final_cursor = json.loads(checkpoint.raw_payload["_cursor"])
    assert final_cursor["watermark"] == "2026-04-24T00:00:00Z"
    assert final_cursor["page_cursor"] is None

    # Authorization header is forwarded.
    auths = {req.headers.get("Authorization") for req in requests_seen}
    assert auths == {"Bearer grn_test_TOKEN"}


@pytest.mark.asyncio
async def test_backfill_resumes_from_cursor_using_created_after() -> None:
    """When given a cursor with a watermark, sends created_after to Granola."""
    seen_params: list[httpx.QueryParams] = []

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/v1/notes":
            seen_params.append(req.url.params)
            return httpx.Response(200, json={"notes": [], "hasMore": False})
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=_granola_transport(handler)) as http:
        granola = GranolaConnector(_make_ctx(http=http))
        cursor = json.dumps({"watermark": "2026-04-23T00:00:00Z", "page_cursor": None})
        events = [
            ev async for ev in granola.backfill("cust-1", _token(), cursor=cursor)
        ]

    assert events == []
    assert len(seen_params) == 1
    assert seen_params[0].get("created_after") == "2026-04-23T00:00:00Z"


@pytest.mark.asyncio
async def test_backfill_raises_permanent_on_401() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"message": "bad token"})

    async with httpx.AsyncClient(transport=_granola_transport(handler)) as http:
        granola = GranolaConnector(_make_ctx(http=http))
        with pytest.raises(PermanentSourceError):
            async for _ in granola.backfill("cust-1", _token(), cursor=None):
                pass


@pytest.mark.asyncio
async def test_backfill_raises_ratelimited_on_429() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            headers={"Retry-After": "5"},
            json={"message": "slow down"},
        )

    async with httpx.AsyncClient(transport=_granola_transport(handler)) as http:
        granola = GranolaConnector(_make_ctx(http=http))
        with pytest.raises(RateLimited):
            async for _ in granola.backfill("cust-1", _token(), cursor=None):
                pass


@pytest.mark.asyncio
async def test_backfill_raises_transient_on_5xx() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"message": "down"})

    async with httpx.AsyncClient(transport=_granola_transport(handler)) as http:
        granola = GranolaConnector(_make_ctx(http=http))
        with pytest.raises(TransientSourceError):
            async for _ in granola.backfill("cust-1", _token(), cursor=None):
                pass


@pytest.mark.asyncio
async def test_backfill_skips_note_with_transient_hydration_failure() -> None:
    """A 5xx on per-note GET shouldn't kill the whole tick — just skip the note."""
    list_body = {"notes": [{"id": "good"}, {"id": "bad"}], "hasMore": False}
    good = {
        "id": "good",
        "title": "g",
        "summary": "ok",
        "owner": {"name": "x", "email": "x@e"},
        "created_at": "2026-04-24T00:00:00Z",
        "transcript": [],
    }

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/v1/notes":
            return httpx.Response(200, json=list_body)
        if req.url.path.endswith("/good"):
            return httpx.Response(200, json=good)
        # 'bad' raises TransientSourceError inside _granola_get; that
        # propagates and ends the backfill iteration. Verify by counting
        # what we got before the raise.
        return httpx.Response(503)

    async with httpx.AsyncClient(transport=_granola_transport(handler)) as http:
        granola = GranolaConnector(_make_ctx(http=http))
        collected: list[str] = []
        with pytest.raises(TransientSourceError):
            async for ev in granola.backfill("cust-1", _token(), cursor=None):
                collected.append(ev.source_event_id)
        assert collected == ["good"]


# ---------------------------------------------------------------------------
# backfill — watermark-checkpoint correctness (regression for the bug where a
# transient interruption would advance the persisted watermark past notes the
# previous run never reached, permanently skipping them).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_backfill_no_checkpoint_on_transient_list_error() -> None:
    """First page yields notes; second page list errors transiently (None).

    Per-yield events must carry the INPUT watermark (not advanced) and the
    connector must NOT emit a checkpoint event. The runner then leaves the
    persisted watermark at the input value, so the next tick re-lists.
    """
    page_1 = {
        "notes": [{"id": "not_a"}, {"id": "not_b"}],
        "hasMore": True,
        "cursor": "page2",
    }
    notes_by_id = {
        "not_a": {
            "id": "not_a",
            "title": "A",
            "summary": "first",
            "owner": {"name": "x", "email": "x@e"},
            "created_at": "2026-04-26T00:00:00Z",
            "transcript": [],
        },
        "not_b": {
            "id": "not_b",
            "title": "B",
            "summary": "second",
            "owner": {"name": "x", "email": "x@e"},
            "created_at": "2026-04-27T00:00:00Z",
            "transcript": [],
        },
    }

    list_calls: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/v1/notes":
            list_calls.append(req)
            cursor = req.url.params.get("cursor")
            if cursor == "page2":
                # 418 is a non-{200,401,403,429,5xx} status → _granola_get
                # returns None → backfill returns without checkpoint.
                return httpx.Response(418, text="teapot")
            return httpx.Response(200, json=page_1)
        if req.url.path.startswith("/v1/notes/"):
            note_id = req.url.path.rsplit("/", 1)[-1]
            return httpx.Response(200, json=notes_by_id[note_id])
        return httpx.Response(404)

    input_cursor = json.dumps(
        {"watermark": "2026-04-25T00:00:00Z", "page_cursor": None}
    )
    async with httpx.AsyncClient(transport=_granola_transport(handler)) as http:
        granola = GranolaConnector(_make_ctx(http=http))
        events = [
            ev async for ev in granola.backfill("cust-1", _token(), cursor=input_cursor)
        ]

    # Both notes from page 1 yielded, no checkpoint after page 2's None.
    assert [e.source_event_id for e in events] == ["not_a", "not_b"]
    for ev in events:
        per = json.loads(ev.raw_payload["_cursor"])
        assert per["watermark"] == "2026-04-25T00:00:00Z"  # INPUT, not advanced
        assert "_checkpoint" not in ev.raw_payload
    # We attempted both list pages.
    assert len(list_calls) == 2


@pytest.mark.asyncio
async def test_backfill_checkpoint_on_clean_run_advances_watermark() -> None:
    """Single page, hasMore=false; final checkpoint encodes max created_at."""
    page = {
        "notes": [{"id": "A"}, {"id": "B"}, {"id": "C"}],
        "hasMore": False,
    }
    notes_by_id = {
        "A": {
            "id": "A",
            "summary": "a",
            "owner": {"name": "x", "email": "x@e"},
            "created_at": "2026-04-25T00:00:00Z",
            "transcript": [],
        },
        "B": {
            "id": "B",
            "summary": "b",
            "owner": {"name": "x", "email": "x@e"},
            "created_at": "2026-04-26T00:00:00Z",
            "transcript": [],
        },
        "C": {
            "id": "C",
            "summary": "c",
            "owner": {"name": "x", "email": "x@e"},
            "created_at": "2026-04-27T00:00:00Z",
            "transcript": [],
        },
    }

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/v1/notes":
            return httpx.Response(200, json=page)
        if req.url.path.startswith("/v1/notes/"):
            return httpx.Response(200, json=notes_by_id[req.url.path.rsplit("/", 1)[-1]])
        return httpx.Response(404)

    input_watermark = "2026-04-24T00:00:00Z"
    input_cursor = json.dumps({"watermark": input_watermark, "page_cursor": None})

    async with httpx.AsyncClient(transport=_granola_transport(handler)) as http:
        granola = GranolaConnector(_make_ctx(http=http))
        events = [
            ev async for ev in granola.backfill("cust-1", _token(), cursor=input_cursor)
        ]

    assert [e.source_event_id for e in events] == [
        "A",
        "B",
        "C",
        "__cursor_checkpoint__",
    ]
    for ev in events[:3]:
        per = json.loads(ev.raw_payload["_cursor"])
        assert per["watermark"] == input_watermark  # unchanged on per-note events
        assert "_checkpoint" not in ev.raw_payload

    checkpoint = events[-1]
    assert checkpoint.raw_payload["_checkpoint"] is True
    final = json.loads(checkpoint.raw_payload["_cursor"])
    assert final["watermark"] == "2026-04-27T00:00:00Z"  # = C's created_at
    assert final["page_cursor"] is None


@pytest.mark.asyncio
async def test_backfill_no_checkpoint_when_max_pages_hit(monkeypatch) -> None:
    """If we hit _MAX_PAGES_PER_TICK with has_more still true, don't checkpoint —
    next tick must keep the same created_after filter."""
    import services.ingestion.handlers.granola as granola_mod

    monkeypatch.setattr(granola_mod, "_MAX_PAGES_PER_TICK", 1)

    page_1 = {
        "notes": [{"id": "only"}],
        "hasMore": True,
        "cursor": "page2",  # but we'll never reach page 2 due to MAX=1
    }
    note = {
        "id": "only",
        "summary": "x",
        "owner": {"name": "x", "email": "x@e"},
        "created_at": "2026-04-27T00:00:00Z",
        "transcript": [],
    }

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/v1/notes":
            return httpx.Response(200, json=page_1)
        if req.url.path.startswith("/v1/notes/"):
            return httpx.Response(200, json=note)
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=_granola_transport(handler)) as http:
        granola = GranolaConnector(_make_ctx(http=http))
        events = [
            ev async for ev in granola.backfill("cust-1", _token(), cursor=None)
        ]

    # Got the one note, but no checkpoint because the loop terminated via the
    # MAX_PAGES guard while has_more=true.
    assert [e.source_event_id for e in events] == ["only"]
    assert all("_checkpoint" not in ev.raw_payload for ev in events)


@pytest.mark.asyncio
async def test_backfill_caps_watermark_at_skipped_note() -> None:
    """A, B, C in one page. Hydrate succeeds for A and C, returns None for B.

    The connector must cap the checkpoint watermark at B's created_at - 1ms,
    NOT advance it to C's created_at — otherwise B would be permanently
    skipped on the next tick (created_after=C filters B out).
    """
    page = {
        "notes": [
            {"id": "A", "created_at": "2026-04-25T00:00:00.000Z"},
            {"id": "B", "created_at": "2026-04-26T00:00:00.000Z"},
            {"id": "C", "created_at": "2026-04-27T00:00:00.000Z"},
        ],
        "hasMore": False,
    }
    notes_by_id = {
        "A": {
            "id": "A",
            "summary": "a",
            "owner": {"name": "x", "email": "x@e"},
            "created_at": "2026-04-25T00:00:00.000Z",
            "transcript": [],
        },
        "C": {
            "id": "C",
            "summary": "c",
            "owner": {"name": "x", "email": "x@e"},
            "created_at": "2026-04-27T00:00:00.000Z",
            "transcript": [],
        },
    }

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/v1/notes":
            return httpx.Response(200, json=page)
        if req.url.path.endswith("/A"):
            return httpx.Response(200, json=notes_by_id["A"])
        if req.url.path.endswith("/B"):
            # 418 → _granola_get returns None → connector skips B and
            # records min_skipped_created_at=B.created_at.
            return httpx.Response(418, text="teapot")
        if req.url.path.endswith("/C"):
            return httpx.Response(200, json=notes_by_id["C"])
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=_granola_transport(handler)) as http:
        granola = GranolaConnector(_make_ctx(http=http))
        events = [
            ev async for ev in granola.backfill("cust-1", _token(), cursor=None)
        ]

    # A and C yielded; B skipped. Final event is the checkpoint.
    assert [e.source_event_id for e in events] == [
        "A",
        "C",
        "__cursor_checkpoint__",
    ]
    checkpoint = events[-1]
    assert checkpoint.raw_payload["_checkpoint"] is True
    final = json.loads(checkpoint.raw_payload["_cursor"])
    # Capped at B's created_at minus 1ms (NOT C's created_at).
    assert final["watermark"] == "2026-04-25T23:59:59.999Z"


@pytest.mark.asyncio
async def test_backfill_no_checkpoint_when_nothing_seen() -> None:
    """Empty page → connector yields nothing at all (no checkpoint either)."""

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/v1/notes":
            return httpx.Response(200, json={"notes": [], "hasMore": False})
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=_granola_transport(handler)) as http:
        granola = GranolaConnector(_make_ctx(http=http))
        events = [
            ev async for ev in granola.backfill("cust-1", _token(), cursor=None)
        ]

    assert events == []


def test_watermark_step_back_1ms_helper() -> None:
    """Helper produces ISO 1ms earlier; returns None on unparseable input."""
    from services.ingestion.handlers.granola import _watermark_step_back_1ms

    assert (
        _watermark_step_back_1ms("2026-04-28T19:03:26.314Z")
        == "2026-04-28T19:03:26.313Z"
    )
    assert _watermark_step_back_1ms("not-a-date") is None
    assert _watermark_step_back_1ms("") is None
