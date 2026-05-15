"""SlackPoller tests (Phase 2 PR E2).

Exercises ``SlackPoller.poll`` end-to-end against an ``httpx.MockTransport``
that stands in for Slack's REST API. The DB-side token-load path is
monkeypatched to a fixed ``IntegrationToken`` so the tests don't need
a live ``integration_tokens`` row.

Surface:

  * First poll (cursor=None): 3 messages on one page, cursor advances
    to the newest ``ts``.
  * Subsequent poll (cursor set): ``oldest`` is the stored cursor,
    cursor advances past the new batch.
  * Pagination: page 1 returns ``response_metadata.next_cursor``;
    page 2 returns it empty. Both pages merge into one PollResult.
  * Slack API soft-error (``ok: false, error: ratelimited``) →
    ``PollResult.error`` set, no cursor change.
  * Auth-test failure → ``PollResult.error`` set.
  * Missing token → ``PollResult.error="missing_active_token"``.
  * Empty channel: no documents, ``next_cursor`` is None (no advance).
  * Non-message types + empty/text-less rows are filtered out before
    becoming documents.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from services.ingestion.polling import slack as slack_poller_mod
from services.ingestion.polling.slack import SlackPoller
from shared.constants import SourceSystem
from shared.models import IntegrationToken

# Asyncio mode is "auto" in pyproject; explicit decorators are still OK
# and make the test surface obvious to readers.
pytestmark = pytest.mark.asyncio


_CUSTOMER = "test-customer-1"
_CHANNEL = "C0123ABC"
_TEAM = "T0TESTTEAM"
_TOKEN_VALUE = "xoxb-test-token"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_token() -> IntegrationToken:
    return IntegrationToken(
        customer_id=_CUSTOMER,
        source_system=SourceSystem.SLACK,
        access_token=_TOKEN_VALUE,
        scope="channels:history,channels:read,users:read,team:read",
    )


def _auth_test_ok() -> httpx.Response:
    return httpx.Response(
        200,
        json={"ok": True, "team_id": _TEAM, "team": "TestCo", "user_id": "UBOT"},
    )


def _message(ts: str, *, text: str = "hello world", user: str = "U001") -> dict[str, Any]:
    """Shape that conversations.history returns inside ``messages[]``."""
    return {
        "type": "message",
        "ts": ts,
        "text": text,
        "user": user,
    }


def _history_ok(
    messages: list[dict[str, Any]],
    *,
    next_cursor: str | None = None,
) -> httpx.Response:
    body: dict[str, Any] = {"ok": True, "messages": messages, "has_more": bool(next_cursor)}
    if next_cursor:
        body["response_metadata"] = {"next_cursor": next_cursor}
    return httpx.Response(200, json=body)


def _install_mock_transport(
    monkeypatch: pytest.MonkeyPatch,
    handler,
) -> list[httpx.Request]:
    """Wire ``SlackPoller.http_client_factory`` to a MockTransport-backed
    client and capture every request the poller makes. Returns the
    captured-request list so the test can assert on order + body shape."""
    captured: list[httpx.Request] = []

    def _wrapped(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return handler(request)

    def _factory() -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(_wrapped))

    monkeypatch.setattr(SlackPoller, "http_client_factory", staticmethod(_factory))
    return captured


def _install_token(monkeypatch: pytest.MonkeyPatch, token: IntegrationToken | None) -> None:
    """Bypass the DB load_token call; the polling tests don't need
    real ``integration_tokens`` rows."""

    async def _fake_load_token(customer_id: str, source_system: SourceSystem):
        assert customer_id == _CUSTOMER
        assert source_system is SourceSystem.SLACK
        return token

    monkeypatch.setattr(slack_poller_mod, "load_token", _fake_load_token)


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


async def test_first_poll_pulls_seven_day_window_and_advances_cursor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cold start: cursor=None → ``oldest`` is a Unix timestamp string
    (the 7-day-ago cutoff). Three messages come back; the cursor
    advances to the largest ``ts``."""
    _install_token(monkeypatch, _make_token())

    messages = [
        _message("1704067203.000300", text="newest"),
        _message("1704067202.000200", text="middle"),
        _message("1704067201.000100", text="oldest"),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/auth.test":
            return _auth_test_ok()
        if request.url.path == "/api/conversations.history":
            return _history_ok(messages)
        return httpx.Response(404, json={"ok": False, "error": "unmocked_path"})

    captured = _install_mock_transport(monkeypatch, handler)

    poller = SlackPoller()
    result = await poller.poll(
        customer_id=_CUSTOMER,
        resource_id=_CHANNEL,
        cursor=None,
    )

    assert result.error is None
    assert len(result.documents) == 3
    assert result.next_cursor == "1704067203.000300"

    # auth.test fires once before the history call.
    paths = [r.url.path for r in captured]
    assert paths == ["/api/auth.test", "/api/conversations.history"]

    # First-call body uses a 7-day-ago oldest (Unix-epoch string with 6
    # decimals); it must NOT be a Slack-style ``ts`` and must be inclusive=False.
    history_req = captured[1]
    body = _json_body(history_req)
    assert body["channel"] == _CHANNEL
    assert body["inclusive"] is False
    assert body["limit"] == 200
    assert "cursor" not in body
    oldest = body["oldest"]
    # 7 days ≈ 604800s; the cutoff is in the past relative to "now".
    assert float(oldest) > 0
    assert float(oldest) < 2_000_000_000  # before year 2033, sanity bound

    # Webhook-envelope shape: each document is event_callback-shaped so
    # the existing SlackConnector.normalize() can consume it unchanged.
    doc = result.documents[0]
    assert doc["type"] == "event_callback"
    assert doc["team_id"] == _TEAM
    assert doc["event"]["type"] == "message"
    assert doc["event"]["channel"] == _CHANNEL
    assert doc["event"]["ts"] in {m["ts"] for m in messages}


async def test_subsequent_poll_uses_stored_cursor_as_oldest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When a cursor is stored, the next tick passes it as ``oldest``
    so Slack only returns strictly-newer messages."""
    _install_token(monkeypatch, _make_token())

    stored_cursor = "1704067200.000000"
    new_messages = [
        _message("1704067210.000500"),
        _message("1704067205.000400"),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/auth.test":
            return _auth_test_ok()
        return _history_ok(new_messages)

    captured = _install_mock_transport(monkeypatch, handler)

    poller = SlackPoller()
    result = await poller.poll(
        customer_id=_CUSTOMER,
        resource_id=_CHANNEL,
        cursor=stored_cursor,
    )

    assert result.error is None
    assert len(result.documents) == 2
    assert result.next_cursor == "1704067210.000500"

    history_req = next(r for r in captured if r.url.path == "/api/conversations.history")
    body = _json_body(history_req)
    assert body["oldest"] == stored_cursor


async def test_pagination_merges_pages_via_response_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """response_metadata.next_cursor on page 1 → page 2 fetched with
    ``cursor`` in the request body. Both pages' messages land in one
    PollResult and the max ts across pages wins."""
    _install_token(monkeypatch, _make_token())

    page1 = [_message("1704067210.000500"), _message("1704067209.000400")]
    page2 = [_message("1704067208.000300"), _message("1704067207.000200")]

    history_calls: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/auth.test":
            return _auth_test_ok()
        body = _json_body(request)
        history_calls.append(body)
        if "cursor" not in body:
            return _history_ok(page1, next_cursor="PAGE2_CURSOR")
        if body["cursor"] == "PAGE2_CURSOR":
            return _history_ok(page2)
        return httpx.Response(500, json={"ok": False, "error": "unexpected_cursor"})

    _install_mock_transport(monkeypatch, handler)

    poller = SlackPoller()
    result = await poller.poll(
        customer_id=_CUSTOMER,
        resource_id=_CHANNEL,
        cursor="1704067200.000000",
    )

    assert result.error is None
    assert len(result.documents) == 4
    # Largest ts across both pages.
    assert result.next_cursor == "1704067210.000500"
    # Page 1 had no cursor; page 2 carried it.
    assert len(history_calls) == 2
    assert "cursor" not in history_calls[0]
    assert history_calls[1]["cursor"] == "PAGE2_CURSOR"


async def test_ratelimited_returns_error_and_no_cursor_advance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``ok: false, error: ratelimited`` → PollResult.error stamped;
    documents empty; next_cursor None so the scheduler doesn't try to
    advance past unread messages."""
    _install_token(monkeypatch, _make_token())

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/auth.test":
            return _auth_test_ok()
        return httpx.Response(200, json={"ok": False, "error": "ratelimited"})

    _install_mock_transport(monkeypatch, handler)

    poller = SlackPoller()
    result = await poller.poll(
        customer_id=_CUSTOMER,
        resource_id=_CHANNEL,
        cursor="1704067200.000000",
    )

    assert result.error == "ratelimited"
    assert result.documents == []
    assert result.next_cursor is None


async def test_empty_channel_returns_no_documents_and_no_cursor_advance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A channel with no new messages since ``oldest`` → empty messages
    array. PollResult should have no documents and next_cursor=None,
    which the framework's ``advance_cursor`` interprets as "keep the
    existing cursor"."""
    _install_token(monkeypatch, _make_token())

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/auth.test":
            return _auth_test_ok()
        return _history_ok([])

    _install_mock_transport(monkeypatch, handler)

    poller = SlackPoller()
    result = await poller.poll(
        customer_id=_CUSTOMER,
        resource_id=_CHANNEL,
        cursor="1704067200.000000",
    )

    assert result.error is None
    assert result.documents == []
    assert result.next_cursor is None


async def test_non_message_types_are_filtered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """conversations.history sometimes returns ``channel_join`` /
    ``channel_topic`` rows under ``type``; the poller must drop
    non-message and content-less rows so they don't reach the
    normalizer."""
    _install_token(monkeypatch, _make_token())

    messages = [
        _message("1704067210.000500", text="real message"),
        # Wrong type: not a user message.
        {"type": "channel_join", "ts": "1704067209.000400", "user": "U002"},
        # Right type, no text + no files: noise.
        {"type": "message", "ts": "1704067208.000300", "user": "U001"},
        # Right type, has files: keep even without text.
        {
            "type": "message",
            "ts": "1704067207.000200",
            "user": "U001",
            "files": [{"id": "F1", "url_private": "https://example.com/f1"}],
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/auth.test":
            return _auth_test_ok()
        return _history_ok(messages)

    _install_mock_transport(monkeypatch, handler)

    poller = SlackPoller()
    result = await poller.poll(
        customer_id=_CUSTOMER,
        resource_id=_CHANNEL,
        cursor="1704067200.000000",
    )

    assert result.error is None
    kept_ts = {d["event"]["ts"] for d in result.documents}
    assert kept_ts == {"1704067210.000500", "1704067207.000200"}
    # Cursor advances to the max ts among messages that WERE kept;
    # filtered rows' ts (the channel_join at .400 and the empty msg at
    # .300) must not influence the cursor either, so the max stays at .500.
    assert result.next_cursor == "1704067210.000500"


async def test_auth_test_failure_surfaces_as_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If auth.test returns ok:false, we can't build webhook payloads
    (no team_id), so the tick fails soft without hitting history."""
    _install_token(monkeypatch, _make_token())

    history_called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal history_called
        if request.url.path == "/api/auth.test":
            return httpx.Response(200, json={"ok": False, "error": "invalid_auth"})
        if request.url.path == "/api/conversations.history":
            history_called = True
            return _history_ok([])
        return httpx.Response(404, json={"ok": False, "error": "unmocked_path"})

    _install_mock_transport(monkeypatch, handler)

    poller = SlackPoller()
    result = await poller.poll(
        customer_id=_CUSTOMER,
        resource_id=_CHANNEL,
        cursor=None,
    )

    assert result.error == "auth_test_failed"
    assert result.documents == []
    assert result.next_cursor is None
    assert history_called is False, "history should not be called when auth.test fails"


async def test_missing_token_returns_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """No active integration_tokens row → soft error so the scheduler
    can stamp it. The customer's cursor is preserved for after they
    re-authorize."""
    _install_token(monkeypatch, None)

    # No transport should be hit; install a handler that explodes if so.
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected HTTP call to {request.url}")

    _install_mock_transport(monkeypatch, handler)

    poller = SlackPoller()
    result = await poller.poll(
        customer_id=_CUSTOMER,
        resource_id=_CHANNEL,
        cursor=None,
    )
    assert result.error == "missing_active_token"
    assert result.documents == []
    assert result.next_cursor is None


# ---------------------------------------------------------------------------
# private helpers
# ---------------------------------------------------------------------------


def _json_body(request: httpx.Request) -> dict[str, Any]:
    """Decode an httpx.Request body that we know is JSON. MockTransport
    hands us a Request whose ``content`` is the encoded body — same
    surface as on the wire."""
    import json as _json

    return _json.loads(request.content.decode("utf-8"))
