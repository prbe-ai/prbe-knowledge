"""Unit tests for the Notion source poller (PR E4).

httpx.MockTransport handles the upstream API; ``load_token`` is patched
to a static token so the tests don't need a live DB. Each test exercises
one branch of ``NotionPoller.poll``:

  * happy path on a fresh cursor (no prior cursor → 7-day lookback)
  * pagination via Notion's ``next_cursor`` / body ``start_cursor``
  * client-side filter — pages with ``last_edited_time`` <= cursor are
    dropped, and the cursor only advances to the max edited time of the
    NEW pages
  * Notion ``{"object": "error"}`` envelope → ``PollResult.error`` set,
    cursor untouched
  * missing token → soft error, empty documents
  * unparseable cursor → falls back to lookback without raising
  * empty results → ``next_cursor=None`` so the scheduler keeps the
    existing stored cursor (COALESCE in advance_cursor)
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from engine.shared.constants import SourceSystem
from engine.shared.models import IntegrationToken
from kb.polling.notion import (
    NotionPoller,
    _format_cursor,
    _parse_cursor,
)

_CUSTOMER = "cust_notion_test"
_WORKSPACE = "ws_test_42"
_TOKEN = "secret_test_token"


def _make_token() -> IntegrationToken:
    return IntegrationToken(
        customer_id=_CUSTOMER,
        source_system=SourceSystem.NOTION,
        access_token=_TOKEN,
    )


def _page(
    page_id: str,
    last_edited: str,
    *,
    title: str = "Untitled",
) -> dict[str, Any]:
    """Minimal Notion search-result page shape — enough for the poller's
    extract + the downstream webhook handler's parse step. The full
    entity is fetched again at hydration time, so we don't need every
    real-world field here."""
    return {
        "object": "page",
        "id": page_id,
        "last_edited_time": last_edited,
        "created_time": last_edited,
        "url": f"https://www.notion.so/{page_id.replace('-', '')}",
        "properties": {
            "title": {
                "type": "title",
                "title": [{"plain_text": title, "type": "text"}],
            }
        },
        "parent": {"type": "workspace", "workspace": True},
    }


# Stash the real AsyncClient before any patching shadows it — the
# _FakeAsyncClient wrapper needs to instantiate the genuine class.
_REAL_ASYNC_CLIENT = httpx.AsyncClient


class _FakeAsyncClient:
    """Drop-in stand-in for ``httpx.AsyncClient`` that defers to a
    user-supplied MockTransport. Constructed by patching ``httpx.AsyncClient``
    in the poller module so the inner ``async with httpx.AsyncClient(...)``
    yields this object instead of opening a real socket."""

    def __init__(self, transport: httpx.MockTransport):
        self._client = _REAL_ASYNC_CLIENT(transport=transport)

    async def __aenter__(self) -> _FakeAsyncClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._client.aclose()

    async def post(self, *args: Any, **kwargs: Any) -> httpx.Response:
        return await self._client.post(*args, **kwargs)


def _patch_client(transport: httpx.MockTransport) -> Any:
    """Patch the AsyncClient constructor inside the poller module so any
    call (with any args) hands back our MockTransport-backed client."""

    def _factory(*_args: Any, **_kwargs: Any) -> _FakeAsyncClient:
        return _FakeAsyncClient(transport)

    return patch(
        "kb.polling.notion.httpx.AsyncClient",
        side_effect=_factory,
    )


def _patch_load_token(token: IntegrationToken | None) -> Any:
    async def _loader(*_args: Any, **_kwargs: Any) -> IntegrationToken | None:
        return token

    return patch(
        "kb.polling.notion.load_token",
        side_effect=_loader,
    )


# ---------------------------------------------------------------------------
# happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_first_poll_returns_pages_within_lookback() -> None:
    """No cursor stored → 7-day lookback. Two recent pages come back,
    one older-than-7-days page is filtered out."""
    now = datetime.now(UTC)
    recent_a = (now - timedelta(hours=2)).isoformat().replace("+00:00", "Z")
    recent_b = (now - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    too_old = (now - timedelta(days=30)).isoformat().replace("+00:00", "Z")

    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/v1/search"
        body = json.loads(request.content)
        captured.append(body)
        # Verify the request body matches the documented Notion shape.
        assert body["filter"] == {"property": "object", "value": "page"}
        assert body["sort"] == {
            "timestamp": "last_edited_time",
            "direction": "ascending",
        }
        assert body["page_size"] == 100
        # Verify auth + version headers.
        assert request.headers["Authorization"] == f"Bearer {_TOKEN}"
        assert request.headers["Notion-Version"] == "2022-06-28"
        return httpx.Response(
            200,
            json={
                "results": [
                    _page("page-too-old", too_old, title="ancient"),
                    _page("page-a", recent_a, title="A"),
                    _page("page-b", recent_b, title="B"),
                ],
                "has_more": False,
                "next_cursor": None,
            },
        )

    transport = httpx.MockTransport(handler)
    with _patch_client(transport), _patch_load_token(_make_token()):
        result = await NotionPoller().poll(
            customer_id=_CUSTOMER,
            resource_id=_WORKSPACE,
            cursor=None,
        )

    assert result.error is None
    # Only the two recent pages survive the lookback floor.
    assert len(result.documents) == 2
    ids = [d["entity"]["id"] for d in result.documents]
    assert ids == ["page-a", "page-b"]

    # Each doc is shaped like a real Notion webhook event so the
    # downstream connector's `_is_notion_webhook` branch picks it up.
    doc_a = result.documents[0]
    assert doc_a["type"] == "page.created"
    assert doc_a["entity"]["type"] == "page"
    assert doc_a["entity"]["workspace_id"] == _WORKSPACE
    assert doc_a["workspace_id"] == _WORKSPACE
    assert doc_a["data"]["id"] == "page-a"
    assert doc_a["_source"] == "poller"

    # Cursor advances to the LATEST last_edited we ingested (recent_b).
    assert result.next_cursor == _format_cursor(_parse_cursor(recent_b))
    # First request body has no start_cursor (no pagination yet).
    assert "start_cursor" not in captured[0]


@pytest.mark.asyncio
async def test_empty_results_returns_no_cursor_advance() -> None:
    """When Notion returns an empty results list, ``next_cursor`` is
    ``None`` so ``advance_cursor`` keeps the existing stored value via
    its COALESCE clause."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [], "has_more": False, "next_cursor": None})

    transport = httpx.MockTransport(handler)
    with _patch_client(transport), _patch_load_token(_make_token()):
        result = await NotionPoller().poll(
            customer_id=_CUSTOMER,
            resource_id=_WORKSPACE,
            cursor="2026-05-01T00:00:00Z",
        )

    assert result.error is None
    assert result.documents == []
    assert result.next_cursor is None


# ---------------------------------------------------------------------------
# pagination
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_paginates_via_next_cursor() -> None:
    """Two pages of /v1/search; the second is fetched only when the first
    returns ``has_more=true`` + ``next_cursor``. The poller must forward
    that value as ``start_cursor`` on the next POST body."""
    now = datetime.now(UTC)
    t1 = (now - timedelta(hours=4)).isoformat().replace("+00:00", "Z")
    t2 = (now - timedelta(hours=3)).isoformat().replace("+00:00", "Z")
    t3 = (now - timedelta(hours=2)).isoformat().replace("+00:00", "Z")
    t4 = (now - timedelta(hours=1)).isoformat().replace("+00:00", "Z")

    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured.append(body)
        if "start_cursor" not in body:
            return httpx.Response(
                200,
                json={
                    "results": [_page("p1", t1), _page("p2", t2)],
                    "has_more": True,
                    "next_cursor": "page2-cursor",
                },
            )
        assert body["start_cursor"] == "page2-cursor"
        return httpx.Response(
            200,
            json={
                "results": [_page("p3", t3), _page("p4", t4)],
                "has_more": False,
                "next_cursor": None,
            },
        )

    transport = httpx.MockTransport(handler)
    with _patch_client(transport), _patch_load_token(_make_token()):
        result = await NotionPoller().poll(
            customer_id=_CUSTOMER,
            resource_id=_WORKSPACE,
            cursor=None,
        )

    assert result.error is None
    assert len(result.documents) == 4
    assert [d["entity"]["id"] for d in result.documents] == ["p1", "p2", "p3", "p4"]
    # Cursor is the max across BOTH pages.
    assert result.next_cursor == _format_cursor(_parse_cursor(t4))
    # Two POST calls were made.
    assert len(captured) == 2


# ---------------------------------------------------------------------------
# client-side cursor filter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_filters_pages_at_or_before_cursor() -> None:
    """Notion search returns ALL pages regardless of cursor — the
    poller must drop anything with ``last_edited_time <= cursor`` so we
    don't re-emit work the previous tick already enqueued."""
    cursor_dt = datetime(2026, 5, 10, 12, 0, 0, tzinfo=UTC)
    cursor_str = _format_cursor(cursor_dt)

    # One page strictly older, one exactly equal (must also be dropped —
    # "strict greater than" is the semantic), one newer.
    older = "2026-05-09T08:00:00Z"
    equal = cursor_str
    newer = "2026-05-11T15:30:00Z"

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    _page("p-older", older),
                    _page("p-equal", equal),
                    _page("p-newer", newer),
                ],
                "has_more": False,
                "next_cursor": None,
            },
        )

    transport = httpx.MockTransport(handler)
    with _patch_client(transport), _patch_load_token(_make_token()):
        result = await NotionPoller().poll(
            customer_id=_CUSTOMER,
            resource_id=_WORKSPACE,
            cursor=cursor_str,
        )

    assert result.error is None
    assert [d["entity"]["id"] for d in result.documents] == ["p-newer"]
    # Cursor advances to the new page's edit time, not the cursor we
    # were called with.
    assert result.next_cursor == _format_cursor(_parse_cursor(newer))


@pytest.mark.asyncio
async def test_skips_non_page_objects_and_unparseable_timestamps() -> None:
    """Search can return ``database`` objects (we filter to pages
    app-side as a belt-and-suspenders to Notion's server-side filter)
    plus pages without parseable ``last_edited_time``."""
    now = datetime.now(UTC)
    t_good = (now - timedelta(hours=1)).isoformat().replace("+00:00", "Z")

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    # Wrong object type — Notion shouldn't return this
                    # given our filter, but we defend in depth.
                    {"object": "database", "id": "db-x", "last_edited_time": t_good},
                    # Page with garbage timestamp — dropped silently.
                    {
                        "object": "page",
                        "id": "p-bad-ts",
                        "last_edited_time": "not-a-date",
                    },
                    # Page with no id — dropped.
                    {"object": "page", "id": "", "last_edited_time": t_good},
                    # Valid page.
                    _page("p-good", t_good),
                ],
                "has_more": False,
                "next_cursor": None,
            },
        )

    transport = httpx.MockTransport(handler)
    with _patch_client(transport), _patch_load_token(_make_token()):
        result = await NotionPoller().poll(
            customer_id=_CUSTOMER,
            resource_id=_WORKSPACE,
            cursor=None,
        )

    assert result.error is None
    assert [d["entity"]["id"] for d in result.documents] == ["p-good"]


# ---------------------------------------------------------------------------
# error envelopes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_notion_error_envelope_sets_poll_error() -> None:
    """Notion's documented error shape: 200/4xx with body
    ``{"object": "error", "status": ..., "code": ..., "message": ...}``.
    The poller must surface that as ``PollResult.error`` and leave
    ``next_cursor`` None so the scheduler doesn't advance past a known
    failure."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={
                "object": "error",
                "status": 401,
                "code": "unauthorized",
                "message": "API token is invalid.",
            },
        )

    transport = httpx.MockTransport(handler)
    with _patch_client(transport), _patch_load_token(_make_token()):
        result = await NotionPoller().poll(
            customer_id=_CUSTOMER,
            resource_id=_WORKSPACE,
            cursor=None,
        )

    assert result.documents == []
    assert result.next_cursor is None
    assert result.error is not None
    assert "unauthorized" in result.error
    assert "API token is invalid" in result.error
    assert "401" in result.error


@pytest.mark.asyncio
async def test_non_json_response_surfaces_error() -> None:
    """A wedged upstream proxy returning HTML must not crash the poller —
    it should surface as a parse-style error so the cursor row's
    ``last_error`` carries the http status."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            502,
            content=b"<html>bad gateway</html>",
            headers={"content-type": "text/html"},
        )

    transport = httpx.MockTransport(handler)
    with _patch_client(transport), _patch_load_token(_make_token()):
        result = await NotionPoller().poll(
            customer_id=_CUSTOMER,
            resource_id=_WORKSPACE,
            cursor=None,
        )

    assert result.documents == []
    assert result.next_cursor is None
    assert result.error is not None
    assert "502" in result.error


@pytest.mark.asyncio
async def test_missing_token_returns_soft_error() -> None:
    """No active integration_tokens row → the poller can't make the
    request. Surface as an error so the operator dashboard shows
    "needs reauth" instead of the cursor silently spinning."""
    transport = httpx.MockTransport(lambda _r: httpx.Response(500, json={"never": "called"}))
    with _patch_client(transport), _patch_load_token(None):
        result = await NotionPoller().poll(
            customer_id=_CUSTOMER,
            resource_id=_WORKSPACE,
            cursor=None,
        )

    assert result.documents == []
    assert result.next_cursor is None
    assert result.error is not None
    assert "no active integration_tokens" in result.error


@pytest.mark.asyncio
async def test_http_transport_error_surfaces_with_partial_docs() -> None:
    """If page 1 succeeds and page 2's POST raises, the poller returns
    the docs it DID collect, advances the cursor to that page's max,
    and stamps the transport error. Partial progress beats restarting
    from scratch every tick."""
    now = datetime.now(UTC)
    t1 = (now - timedelta(hours=3)).isoformat().replace("+00:00", "Z")
    call_count = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return httpx.Response(
                200,
                json={
                    "results": [_page("p-first", t1)],
                    "has_more": True,
                    "next_cursor": "page2",
                },
            )
        raise httpx.ConnectError("connection reset")

    transport = httpx.MockTransport(handler)
    with _patch_client(transport), _patch_load_token(_make_token()):
        result = await NotionPoller().poll(
            customer_id=_CUSTOMER,
            resource_id=_WORKSPACE,
            cursor=None,
        )

    # First page's doc survived.
    assert len(result.documents) == 1
    assert result.documents[0]["entity"]["id"] == "p-first"
    # Cursor advanced to that page's edit time.
    assert result.next_cursor == _format_cursor(_parse_cursor(t1))
    # Error captured.
    assert result.error is not None
    assert "ConnectError" in result.error


# ---------------------------------------------------------------------------
# cursor helpers
# ---------------------------------------------------------------------------


def test_parse_cursor_none_returns_lookback() -> None:
    """``None`` (first poll) decodes to "_FIRST_POLL_LOOKBACK_DAYS_ ago"."""
    out = _parse_cursor(None)
    delta = datetime.now(UTC) - out
    # 7 days ± a few seconds of slop for the comparison.
    assert timedelta(days=6, hours=23) < delta < timedelta(days=7, hours=1)


def test_parse_cursor_round_trip_with_z_suffix() -> None:
    iso = "2026-05-14T10:30:00Z"
    out = _parse_cursor(iso)
    assert out.tzinfo is not None
    assert _format_cursor(out) == iso


def test_parse_cursor_unparseable_falls_back_to_lookback() -> None:
    """A corrupt cursor (manual DB edit, schema drift, etc.) must not
    crash the poller — it falls back to the same lookback ``None`` uses."""
    out = _parse_cursor("not-an-iso")
    delta = datetime.now(UTC) - out
    assert timedelta(days=6, hours=23) < delta < timedelta(days=7, hours=1)


def test_parse_cursor_naive_iso_gets_utc() -> None:
    """ISO strings without a tz suffix are assumed UTC (Notion always
    serializes UTC; if we ever wrote a naive cursor by mistake, the
    fallback shouldn't shift the comparison by the local offset)."""
    out = _parse_cursor("2026-05-14T10:30:00")
    assert out.tzinfo is UTC


# ---------------------------------------------------------------------------
# registry wiring
# ---------------------------------------------------------------------------


def test_notion_poller_registers_on_import() -> None:
    from kb.polling.base import get_poller

    cls = get_poller(SourceSystem.NOTION)
    assert cls is NotionPoller
    assert cls.source is SourceSystem.NOTION
