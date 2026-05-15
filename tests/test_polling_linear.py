"""Linear poller tests (PR E3).

Exercises ``LinearPoller.poll`` against an in-memory ``httpx.MockTransport``
that captures GraphQL POSTs to ``https://api.linear.app/graphql``. The
DB-side ``_load_linear_api_key`` is monkeypatched so these tests run
without a live Postgres (the polling-framework tests already cover the
cursor-table side end-to-end against ``live_db``).

Surface covered:

  * First poll (cursor=None) uses an updatedAfter floor of ~7d ago and
    drains a single page of issues; next_cursor advances to the max
    ``updatedAt`` seen.
  * Pagination: ``pageInfo.hasNextPage=true`` walks ``after=endCursor``
    until exhausted; documents from BOTH pages come back; next_cursor
    is the max across the run.
  * GraphQL ``errors[]`` body on 200 sets ``PollResult.error`` and does
    NOT advance the cursor past anything we already had.
  * Empty result -> zero documents, next_cursor unchanged.
  * Non-200 response sets ``PollResult.error``.
  * Missing integration token row -> error, no HTTP calls.
  * The persisted Linear API key rides as the ``Authorization`` header
    verbatim (no ``Bearer`` prefix).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest

from services.ingestion.polling import linear as linear_poller_mod
from services.ingestion.polling.base import PollResult
from services.ingestion.polling.linear import (
    RESOURCE_ID_WILDCARD,
    LinearPoller,
)
from shared.constants import SourceSystem

_CUSTOMER = "test-linear-cust"
_API_KEY = "lin_api_test_abc123"


# --- helpers ----------------------------------------------------------------


def _stub_api_key(monkeypatch: pytest.MonkeyPatch, value: str | None = _API_KEY) -> None:
    """Bypass the integration_tokens DB lookup so tests don't need live_db."""

    async def _fake_load(customer_id: str) -> str | None:
        return value

    monkeypatch.setattr(linear_poller_mod, "_load_linear_api_key", _fake_load)


def _patch_http(
    monkeypatch: pytest.MonkeyPatch,
    handler: Any,
) -> list[httpx.Request]:
    """Replace ``httpx.AsyncClient`` with one wired to a MockTransport.

    Returns the captured-requests list. The handler may either be a plain
    callable (``request -> Response``) or a list of responses to iterate
    through page-by-page.
    """
    captured: list[httpx.Request] = []

    if callable(handler):
        callable_handler = handler
    else:
        responses = iter(handler)

        def _seq_handler(request: httpx.Request) -> httpx.Response:
            return next(responses)

        callable_handler = _seq_handler

    def _capture(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return callable_handler(request)

    transport = httpx.MockTransport(_capture)

    real_async_client = httpx.AsyncClient

    def _patched(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = transport
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr(linear_poller_mod.httpx, "AsyncClient", _patched)
    return captured


def _decode_body(req: httpx.Request) -> dict[str, Any]:
    return json.loads(req.content.decode("utf-8"))


def _issue_node(
    *,
    issue_id: str,
    identifier: str,
    updated_at: str,
    title: str = "an issue",
) -> dict[str, Any]:
    return {
        "id": issue_id,
        "identifier": identifier,
        "title": title,
        "description": "body",
        "url": f"https://linear.app/test/issue/{identifier}",
        "createdAt": "2026-05-01T00:00:00.000Z",
        "updatedAt": updated_at,
        "state": {"name": "In Progress", "type": "started"},
        "priority": 2,
        "team": {"id": "team-A", "key": "ENG", "name": "Engineering"},
        "creator": {"id": "user-1", "name": "Author", "email": "a@x"},
        "assignee": {"id": "user-2", "name": "Assignee", "email": "b@x"},
    }


def _ok_body(
    nodes: list[dict[str, Any]],
    *,
    has_next_page: bool = False,
    end_cursor: str | None = None,
) -> dict[str, Any]:
    return {
        "data": {
            "issues": {
                "nodes": nodes,
                "pageInfo": {
                    "hasNextPage": has_next_page,
                    "endCursor": end_cursor,
                },
            }
        }
    }


# --- registration -----------------------------------------------------------


def test_linear_poller_registers_itself() -> None:
    """Importing the module should have registered ``LinearPoller`` against
    ``SourceSystem.LINEAR`` (the ``register_poller`` call at module scope)."""
    from services.ingestion.polling.base import get_poller

    # Re-import to make sure registration ran for this process.
    assert get_poller(SourceSystem.LINEAR) is LinearPoller


def test_linear_poller_source_attribute() -> None:
    assert LinearPoller.source is SourceSystem.LINEAR


# --- happy path -------------------------------------------------------------


@pytest.mark.asyncio
async def test_first_poll_uses_seven_day_floor_and_returns_docs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cursor=None => updatedAfter is roughly now - 7d, three issues come
    back as webhook-shaped docs, next_cursor advances to the newest
    updatedAt."""
    _stub_api_key(monkeypatch)

    nodes = [
        _issue_node(
            issue_id=f"iss-{i}",
            identifier=f"ENG-{i}",
            updated_at=f"2026-05-1{i}T12:00:00.000Z",
        )
        for i in range(1, 4)
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_ok_body(nodes))

    captured = _patch_http(monkeypatch, handler)

    result = await LinearPoller().poll(
        customer_id=_CUSTOMER,
        resource_id=RESOURCE_ID_WILDCARD,
        cursor=None,
    )

    assert isinstance(result, PollResult)
    assert result.error is None
    assert len(result.documents) == 3
    for doc in result.documents:
        assert doc["type"] == "Issue"
        assert doc["action"] == "create"
        assert doc["_origin"] == "poll"

    # next_cursor is ISO; newest updatedAt was 2026-05-13.
    assert result.next_cursor is not None
    assert "2026-05-13" in result.next_cursor

    # Exactly one HTTP call (no pagination), and the API key rides
    # verbatim as the Authorization header.
    assert len(captured) == 1
    req = captured[0]
    assert str(req.url) == "https://api.linear.app/graphql"
    assert req.headers["authorization"] == _API_KEY

    # Variables: updatedAfter ~ now - 7d, first=50, no after.
    body = _decode_body(req)
    assert body["variables"]["first"] == 50
    assert "after" not in body["variables"]
    after_dt = datetime.fromisoformat(body["variables"]["updatedAfter"])
    now = datetime.now(UTC)
    assert timedelta(days=6, hours=23) < (now - after_dt) < timedelta(days=7, hours=1)


@pytest.mark.asyncio
async def test_poll_with_cursor_passes_it_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_api_key(monkeypatch)

    cursor_iso = "2026-05-10T00:00:00+00:00"

    def handler(request: httpx.Request) -> httpx.Response:
        body = _decode_body(request)
        # Cursor passed verbatim as updatedAfter.
        assert body["variables"]["updatedAfter"] == cursor_iso
        return httpx.Response(200, json=_ok_body([]))

    _patch_http(monkeypatch, handler)

    result = await LinearPoller().poll(
        customer_id=_CUSTOMER,
        resource_id=RESOURCE_ID_WILDCARD,
        cursor=cursor_iso,
    )

    assert result.error is None
    assert result.documents == []


# --- pagination -------------------------------------------------------------


@pytest.mark.asyncio
async def test_pagination_walks_endcursor_and_unions_docs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """hasNextPage=true on page 1 should drive a second call with
    after=<endCursor>; docs from both pages come back; next_cursor is the
    max updatedAt across both pages."""
    _stub_api_key(monkeypatch)

    page1 = _ok_body(
        [
            _issue_node(
                issue_id="A",
                identifier="ENG-1",
                updated_at="2026-05-10T10:00:00.000Z",
            ),
            _issue_node(
                issue_id="B",
                identifier="ENG-2",
                updated_at="2026-05-11T10:00:00.000Z",
            ),
        ],
        has_next_page=True,
        end_cursor="cur-page1-end",
    )
    page2 = _ok_body(
        [
            _issue_node(
                issue_id="C",
                identifier="ENG-3",
                updated_at="2026-05-12T10:00:00.000Z",
            ),
        ],
        has_next_page=False,
        end_cursor=None,
    )

    responses = [
        httpx.Response(200, json=page1),
        httpx.Response(200, json=page2),
    ]
    captured = _patch_http(monkeypatch, responses)

    result = await LinearPoller().poll(
        customer_id=_CUSTOMER,
        resource_id=RESOURCE_ID_WILDCARD,
        cursor=None,
    )

    assert result.error is None
    assert len(result.documents) == 3
    assert {doc["data"]["id"] for doc in result.documents} == {"A", "B", "C"}

    # next_cursor = max(updatedAt) = 2026-05-12.
    assert result.next_cursor is not None
    assert "2026-05-12" in result.next_cursor

    # Two HTTP calls; the second carries after=<page1.endCursor>.
    assert len(captured) == 2
    body2 = _decode_body(captured[1])
    assert body2["variables"]["after"] == "cur-page1-end"
    # First page had no `after`.
    body1 = _decode_body(captured[0])
    assert "after" not in body1["variables"]


# --- error paths ------------------------------------------------------------


@pytest.mark.asyncio
async def test_graphql_errors_array_surfaces_as_poll_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Linear returns 200 with non-empty ``errors[]`` on validation
    failures. The poller must surface that as ``PollResult.error`` instead
    of silently advancing the cursor with zero docs."""
    _stub_api_key(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "errors": [
                    {"message": "Field 'badness' doesn't exist on type 'Issue'."}
                ],
                "data": None,
            },
        )

    _patch_http(monkeypatch, handler)

    result = await LinearPoller().poll(
        customer_id=_CUSTOMER,
        resource_id=RESOURCE_ID_WILDCARD,
        cursor=None,
    )

    assert result.error is not None
    assert "Field 'badness'" in result.error
    assert result.documents == []


@pytest.mark.asyncio
async def test_non_200_response_sets_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_api_key(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="unauthorized")

    _patch_http(monkeypatch, handler)

    result = await LinearPoller().poll(
        customer_id=_CUSTOMER,
        resource_id=RESOURCE_ID_WILDCARD,
        cursor=None,
    )

    assert result.error is not None
    assert "401" in result.error
    assert result.documents == []


@pytest.mark.asyncio
async def test_missing_api_key_returns_soft_error_without_http(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_api_key(monkeypatch, value=None)

    # Set up a transport that would fail loudly if it got called.
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("poller should not call HTTP when no token exists")

    captured = _patch_http(monkeypatch, handler)

    result = await LinearPoller().poll(
        customer_id=_CUSTOMER,
        resource_id=RESOURCE_ID_WILDCARD,
        cursor="2026-05-10T00:00:00+00:00",
    )

    assert result.error is not None
    assert "integration_tokens" in result.error
    assert result.documents == []
    assert captured == []


# --- empty / boundary -------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_result_keeps_cursor_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty page should produce zero documents and a next_cursor equal
    to the incoming cursor (so the scheduler's COALESCE keeps the old
    watermark)."""
    _stub_api_key(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_ok_body([]))

    _patch_http(monkeypatch, handler)

    in_cursor = "2026-05-10T00:00:00+00:00"
    result = await LinearPoller().poll(
        customer_id=_CUSTOMER,
        resource_id=RESOURCE_ID_WILDCARD,
        cursor=in_cursor,
    )

    assert result.error is None
    assert result.documents == []
    # We preserve the inbound watermark verbatim (parsed and re-serialized).
    assert result.next_cursor is not None
    assert result.next_cursor.startswith("2026-05-10")


@pytest.mark.asyncio
async def test_empty_result_first_poll_returns_none_cursor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First poll + empty result: there's no prior watermark and no new
    nodes, so ``next_cursor`` is None — the scheduler's COALESCE leaves
    the column NULL and the next tick re-runs the 7d-floor query."""
    _stub_api_key(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_ok_body([]))

    _patch_http(monkeypatch, handler)

    result = await LinearPoller().poll(
        customer_id=_CUSTOMER,
        resource_id=RESOURCE_ID_WILDCARD,
        cursor=None,
    )

    assert result.error is None
    assert result.documents == []
    assert result.next_cursor is None


# --- webhook envelope shape -------------------------------------------------


@pytest.mark.asyncio
async def test_node_without_id_is_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    """A malformed node (no ``id``) should NOT make it into the documents
    list — the existing connector's parse step would reject it anyway and
    we'd rather not enqueue a row that's destined to fail."""
    _stub_api_key(monkeypatch)

    nodes = [
        _issue_node(
            issue_id="good",
            identifier="ENG-1",
            updated_at="2026-05-10T10:00:00.000Z",
        ),
        # Missing `id`.
        {
            "identifier": "ENG-X",
            "title": "headless",
            "updatedAt": "2026-05-11T10:00:00.000Z",
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_ok_body(nodes))

    _patch_http(monkeypatch, handler)

    result = await LinearPoller().poll(
        customer_id=_CUSTOMER,
        resource_id=RESOURCE_ID_WILDCARD,
        cursor=None,
    )

    assert result.error is None
    assert len(result.documents) == 1
    assert result.documents[0]["data"]["id"] == "good"


@pytest.mark.asyncio
async def test_envelope_matches_connector_webhook_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The envelope keys ``type``, ``action``, ``data``, ``createdAt``
    are exactly the fields ``LinearConnector.parse_webhook_event`` reads.
    Locking this shape prevents the poller from drifting away from the
    webhook handler."""
    _stub_api_key(monkeypatch)

    node = _issue_node(
        issue_id="iss-1",
        identifier="ENG-1",
        updated_at="2026-05-10T10:00:00.000Z",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_ok_body([node]))

    _patch_http(monkeypatch, handler)

    result = await LinearPoller().poll(
        customer_id=_CUSTOMER,
        resource_id=RESOURCE_ID_WILDCARD,
        cursor=None,
    )

    assert len(result.documents) == 1
    doc = result.documents[0]
    assert set(["type", "action", "data", "organizationId", "createdAt"]).issubset(doc)
    assert doc["type"] == "Issue"
    assert doc["action"] == "create"
    assert doc["data"]["id"] == "iss-1"
    # createdAt clock falls back to updatedAt for the source_event_id hash.
    assert doc["createdAt"] == "2026-05-10T10:00:00.000Z"
