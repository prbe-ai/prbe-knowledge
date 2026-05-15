"""Sentry source poller tests (PR E5).

These tests are intentionally hermetic — no live Postgres, no real
network. We patch ``load_token`` (the only DB-touching call in the
poller) and drive the HTTP layer through ``httpx.MockTransport`` so
each test pins exactly the request the poller makes and exactly the
response the upstream returns.

Coverage:
  * Happy first-poll path → statsPeriod=7d, no cursor, documents
    shaped like the webhook handler consumes.
  * Cursor advances via Link-header ``rel="next"`` parsing across two
    sequential polls.
  * Sentry 429 → ``PollResult(error=...)``, no documents, no cursor
    advance.
  * Empty event list → ``PollResult`` with no documents and no cursor
    (link header has ``rel="next"; results="false"``).
"""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from services.ingestion.polling import sentry as sentry_poller
from services.ingestion.polling.base import PollResult
from services.ingestion.polling.sentry import (
    SentryPoller,
    _parse_next_cursor,
    _parse_next_link,
    _parse_resource_id,
)
from shared.constants import SourceSystem
from shared.models import IntegrationToken

# --- shared fixtures ------------------------------------------------------

_TENANT = "test-sentry-tenant"
_ORG = "acme"
_PROJECT = "api"
_RESOURCE_ID = f"{_ORG}/{_PROJECT}"
_TOKEN = "test-sentry-access-token"


def _fake_token() -> IntegrationToken:
    """A minimal IntegrationToken that ``load_token`` would return for a
    healthy sentry install. Only ``access_token`` is read by the poller."""
    return IntegrationToken(
        customer_id=_TENANT,
        source_system=SourceSystem.SENTRY,
        access_token=_TOKEN,
    )


def _install_mock_transport(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[httpx.Request], httpx.Response],
) -> list[httpx.Request]:
    """Wire a MockTransport into the poller's HTTP client factory and
    return a list that captures every request the poller issued (in order).
    Tests assert against this list to pin the URL + query params + headers.
    """
    captured: list[httpx.Request] = []

    def _capturing(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return handler(request)

    transport = httpx.MockTransport(_capturing)
    monkeypatch.setattr(
        SentryPoller,
        "_http_client_factory",
        lambda: httpx.AsyncClient(transport=transport),
    )
    return captured


def _patch_load_token(
    monkeypatch: pytest.MonkeyPatch,
    token: IntegrationToken | None = None,
) -> None:
    """Replace the load_token import inside the poller module."""

    async def _fake_load_token(customer_id: str, source: SourceSystem) -> IntegrationToken | None:
        assert customer_id == _TENANT
        assert source == SourceSystem.SENTRY
        return token

    monkeypatch.setattr(sentry_poller, "load_token", _fake_load_token)


# --- pure helpers --------------------------------------------------------


def test_parse_resource_id_splits_org_and_project() -> None:
    assert _parse_resource_id("acme/api") == ("acme", "api")


def test_parse_resource_id_rejects_missing_slash() -> None:
    assert _parse_resource_id("acme") == (None, None)


def test_parse_resource_id_rejects_too_many_slashes() -> None:
    # We split on exactly one slash — Sentry slugs never contain "/".
    assert _parse_resource_id("acme/api/v2") == (None, None)


def test_parse_resource_id_rejects_empty_input() -> None:
    assert _parse_resource_id("") == (None, None)


def test_parse_next_link_extracts_next_url() -> None:
    header = (
        '<https://sentry.io/api/0/projects/acme/api/events/?cursor=PREV>; '
        'rel="previous"; results="false"; cursor="0:0:1", '
        '<https://sentry.io/api/0/projects/acme/api/events/?cursor=NEXT_TOK>; '
        'rel="next"; results="true"; cursor="0:100:0"'
    )
    assert _parse_next_link(header) == (
        "https://sentry.io/api/0/projects/acme/api/events/?cursor=NEXT_TOK"
    )


def test_parse_next_link_returns_none_when_results_false() -> None:
    header = (
        '<https://sentry.io/api/0/projects/acme/api/events/?cursor=X>; '
        'rel="next"; results="false"; cursor="0:200:0"'
    )
    assert _parse_next_link(header) is None


def test_parse_next_cursor_pulls_cursor_query_param() -> None:
    header = (
        '<https://sentry.io/api/0/projects/acme/api/events/'
        '?cursor=ABC123&statsPeriod=7d>; '
        'rel="next"; results="true"; cursor="0:100:0"'
    )
    assert _parse_next_cursor(header) == "ABC123"


def test_parse_next_cursor_returns_none_with_no_link() -> None:
    assert _parse_next_cursor("") is None


# --- happy path: first poll ----------------------------------------------


async def test_first_poll_uses_stats_period_and_emits_documents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First poll for a fresh resource (cursor=None) should query
    statsPeriod=7d, send the bearer token, and emit one webhook-shaped
    doc per event Sentry returns."""

    def handler(request: httpx.Request) -> httpx.Response:
        # The URL should hit the project events endpoint with statsPeriod=7d.
        assert request.url.host == "sentry.io"
        assert request.url.path == f"/api/0/projects/{_ORG}/{_PROJECT}/events/"
        assert request.url.params.get("statsPeriod") == "7d"
        # No cursor on the first poll.
        assert request.url.params.get("cursor") is None
        assert request.headers["authorization"] == f"Bearer {_TOKEN}"
        return httpx.Response(
            200,
            json=[
                {
                    "id": "evt-1",
                    "eventID": "evt-1",
                    "groupID": "9001",
                    "dateCreated": "2026-05-14T12:00:00Z",
                    "title": "TypeError: foo is undefined",
                    "culprit": "api.handlers.thing",
                    "tags": [],
                    "entries": [],
                },
                {
                    "id": "evt-2",
                    "eventID": "evt-2",
                    "groupID": "9002",
                    "dateCreated": "2026-05-14T12:01:00Z",
                    "title": "Boom",
                    "culprit": "api.other",
                },
            ],
            headers={
                # No more pages.
                "link": (
                    "<https://sentry.io/...?cursor=PREV>; "
                    'rel="previous"; results="false"; cursor="0:0:1"'
                ),
            },
        )

    captured = _install_mock_transport(monkeypatch, handler)
    _patch_load_token(monkeypatch, _fake_token())

    poller = SentryPoller()
    result = await poller.poll(
        customer_id=_TENANT,
        resource_id=_RESOURCE_ID,
        cursor=None,
    )

    assert isinstance(result, PollResult)
    assert result.error is None
    assert result.next_cursor is None  # no next page
    assert len(result.documents) == 2
    assert len(captured) == 1

    # Webhook-shaped: action / data.event / project.slug / organization.slug.
    doc = result.documents[0]
    assert doc["action"] == "triggered"
    assert doc["project"]["slug"] == _PROJECT
    assert doc["organization"]["slug"] == _ORG
    assert doc["_poller_source"] == "sentry"

    ev = doc["data"]["event"]
    # event_id mirrored under the snake_case key the webhook handler reads.
    assert ev["event_id"] == "evt-1"
    # groupID also exposed as group_id for handler convenience.
    assert ev["group_id"] == "9001"
    # timestamp filled from dateCreated when not present.
    assert ev["timestamp"] == "2026-05-14T12:00:00Z"


# --- cursor advances on second poll --------------------------------------


async def test_subsequent_poll_passes_cursor_and_extracts_next(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second poll passes the previous tick's cursor; the new
    next_cursor comes from the response's Link header rel="next"."""

    def handler(request: httpx.Request) -> httpx.Response:
        # statsPeriod is omitted on subsequent polls — we walk the cursor.
        assert request.url.params.get("statsPeriod") is None
        assert request.url.params.get("cursor") == "PAGE_2"
        return httpx.Response(
            200,
            json=[
                {
                    "id": "evt-9",
                    "eventID": "evt-9",
                    "groupID": "9999",
                    "dateCreated": "2026-05-14T13:00:00Z",
                    "title": "Late",
                },
            ],
            headers={
                "link": (
                    "<https://sentry.io/api/0/projects/acme/api/events/"
                    '?cursor=PAGE_2>; rel="previous"; results="true"; cursor="0:0:1", '
                    "<https://sentry.io/api/0/projects/acme/api/events/"
                    '?cursor=PAGE_3&statsPeriod=7d>; rel="next"; '
                    'results="true"; cursor="0:100:0"'
                ),
            },
        )

    captured = _install_mock_transport(monkeypatch, handler)
    _patch_load_token(monkeypatch, _fake_token())

    poller = SentryPoller()
    result = await poller.poll(
        customer_id=_TENANT,
        resource_id=_RESOURCE_ID,
        cursor="PAGE_2",
    )

    assert result.error is None
    assert result.next_cursor == "PAGE_3"
    assert len(result.documents) == 1
    assert len(captured) == 1


# --- 429 rate-limit path -------------------------------------------------


async def test_429_returns_error_no_documents(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 429 is a soft failure — PollResult.error is set, the scheduler
    stamps it onto the cursor row, and the cursor is NOT advanced."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            text="rate limited; retry after 60s",
            headers={"retry-after": "60"},
        )

    _install_mock_transport(monkeypatch, handler)
    _patch_load_token(monkeypatch, _fake_token())

    poller = SentryPoller()
    result = await poller.poll(
        customer_id=_TENANT,
        resource_id=_RESOURCE_ID,
        cursor="PREV_CURSOR",
    )

    assert result.documents == []
    assert result.next_cursor is None  # cursor NOT advanced
    assert result.error is not None
    assert "429" in result.error


async def test_5xx_returns_error_no_documents(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 5xx is also a soft failure — same shape as 429."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="sentry temporarily unavailable")

    _install_mock_transport(monkeypatch, handler)
    _patch_load_token(monkeypatch, _fake_token())

    poller = SentryPoller()
    result = await poller.poll(
        customer_id=_TENANT,
        resource_id=_RESOURCE_ID,
        cursor=None,
    )
    assert result.documents == []
    assert result.next_cursor is None
    assert result.error is not None
    assert "503" in result.error


# --- empty page ----------------------------------------------------------


async def test_empty_event_list_returns_no_documents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty 200 response (quiet project) returns zero docs, no
    next_cursor, and no error. The scheduler will COALESCE-keep the
    existing cursor on its side via advance_cursor(None)."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[],
            headers={
                "link": (
                    "<https://sentry.io/...?cursor=X>; "
                    'rel="next"; results="false"; cursor="0:0:0"'
                ),
            },
        )

    _install_mock_transport(monkeypatch, handler)
    _patch_load_token(monkeypatch, _fake_token())

    poller = SentryPoller()
    result = await poller.poll(
        customer_id=_TENANT,
        resource_id=_RESOURCE_ID,
        cursor="EARLIER",
    )

    assert result.documents == []
    assert result.next_cursor is None
    assert result.error is None


# --- guard rails ---------------------------------------------------------


async def test_invalid_resource_id_returns_soft_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bad resource_id never hits the network — the poller short-circuits
    with a descriptive soft error."""
    # No HTTP fixture needed; assert no request goes out.
    captured = _install_mock_transport(
        monkeypatch,
        lambda req: httpx.Response(500, text="should not be called"),
    )
    _patch_load_token(monkeypatch, _fake_token())

    poller = SentryPoller()
    result = await poller.poll(
        customer_id=_TENANT,
        resource_id="not-a-valid-resource",
        cursor=None,
    )
    assert result.documents == []
    assert result.error is not None
    assert "invalid sentry resource_id" in result.error
    assert captured == []


async def test_missing_token_returns_soft_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No active integration row → soft error, no HTTP call."""
    captured = _install_mock_transport(
        monkeypatch,
        lambda req: httpx.Response(500, text="should not be called"),
    )
    _patch_load_token(monkeypatch, None)

    poller = SentryPoller()
    result = await poller.poll(
        customer_id=_TENANT,
        resource_id=_RESOURCE_ID,
        cursor=None,
    )
    assert result.documents == []
    assert result.error is not None
    assert "sentry integration token" in result.error
    assert captured == []


async def test_4xx_other_than_429_returns_soft_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 403 (expired token / wrong project) surfaces as a soft error
    rather than throwing — operators triage via cursor.last_error."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="invalid token")

    _install_mock_transport(monkeypatch, handler)
    _patch_load_token(monkeypatch, _fake_token())

    poller = SentryPoller()
    result = await poller.poll(
        customer_id=_TENANT,
        resource_id=_RESOURCE_ID,
        cursor=None,
    )
    assert result.documents == []
    assert result.error is not None
    assert "403" in result.error


async def test_events_missing_ids_are_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A response row with no id / eventID / event_id can't anchor a
    document and is dropped silently — the poll itself succeeds."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[
                {"id": "ok", "eventID": "ok", "groupID": "g1"},
                {"groupID": "g2", "title": "missing id"},  # dropped
                {"id": "ok-2", "eventID": "ok-2", "groupID": "g3"},
            ],
        )

    _install_mock_transport(monkeypatch, handler)
    _patch_load_token(monkeypatch, _fake_token())

    poller = SentryPoller()
    result = await poller.poll(
        customer_id=_TENANT,
        resource_id=_RESOURCE_ID,
        cursor=None,
    )

    assert result.error is None
    assert len(result.documents) == 2
    assert {d["data"]["event"]["event_id"] for d in result.documents} == {
        "ok",
        "ok-2",
    }


async def test_registered_for_sentry_source() -> None:
    """The module's import-time ``register_poller`` call must wire
    SentryPoller to SourceSystem.SENTRY so the scheduler resolves it."""
    from services.ingestion.polling.base import get_poller

    assert get_poller(SourceSystem.SENTRY) is SentryPoller
