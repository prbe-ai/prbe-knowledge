"""Unit tests for the GitHub source poller (PR E1).

These tests exercise ``GitHubPoller.poll`` end-to-end using
``httpx.MockTransport`` for the upstream + monkeypatched ``load_token``
and ``fetch_github_installation_token`` for the auth side. No DB, no
live HTTP — pure pure pure.

Coverage:
  * Happy path: first poll (cursor=None) returns issues + PRs, cursor
    advances to the max updated_at, document shape matches what the
    webhook normalizer expects.
  * Pagination: a ``Link: <...>; rel="next"`` header walks the poller
    onto a second page.
  * 429: returns ``PollResult.error`` and zero documents; cursor would
    not be advanced (the scheduler stamps the error and re-uses the
    prior cursor on the next tick).
  * Empty response: ``next_cursor`` is None so the cursor row's
    existing value is preserved by the scheduler's COALESCE.
  * Malformed resource_id is rejected before any HTTP call.
  * Missing integration token surfaces as an error string.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

import httpx
import pytest

import kb.polling.github as gh_poller
from engine.shared.constants import GITHUB_INSTALLATION_SCOPE_PREFIX, SourceSystem
from engine.shared.models import IntegrationToken
from kb.polling.github import GitHubPoller

_CUST = "test-cust-poll"
_RESOURCE = "acme/api"
_PAT_BEARER = "pat-bearer-xyz"
_APP_BEARER = "app-bearer-abc"


# --- fixtures ---------------------------------------------------------------


def _make_token(scope: str | None = None, access_token: str = _PAT_BEARER) -> IntegrationToken:
    return IntegrationToken(
        customer_id=_CUST,
        source_system=SourceSystem.GITHUB,
        access_token=access_token,
        scope=scope,
    )


def _patch_load_token(
    monkeypatch: pytest.MonkeyPatch,
    *,
    token: IntegrationToken | None,
) -> None:
    async def fake_load_token(customer_id: str, source_system: SourceSystem):
        assert customer_id == _CUST
        assert source_system is SourceSystem.GITHUB
        return token

    monkeypatch.setattr(gh_poller, "load_token", fake_load_token)


def _patch_fetch_installation_token(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_fetch(http, *, customer_id):
        assert customer_id == _CUST
        return _APP_BEARER, datetime.now(UTC)

    monkeypatch.setattr(
        gh_poller, "fetch_github_installation_token", fake_fetch
    )


def _install_transport(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[httpx.Request], httpx.Response],
) -> list[httpx.Request]:
    """Patch ``httpx.AsyncClient`` so the poller's ``async with httpx.AsyncClient``
    returns a client wired to a MockTransport. Returns a list that records
    every request the poller made (in call order) for assertions.
    """
    captured: list[httpx.Request] = []

    def capturing_handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return handler(request)

    transport = httpx.MockTransport(capturing_handler)
    real_async_client = httpx.AsyncClient

    def fake_async_client(*args, **kwargs):
        kwargs["transport"] = transport
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr(gh_poller.httpx, "AsyncClient", fake_async_client)
    return captured


# --- happy path -------------------------------------------------------------


@pytest.mark.asyncio
async def test_first_poll_returns_issues_and_pulls_and_advances_cursor(
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_load_token(monkeypatch, token=_make_token())

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/repos/acme/api/issues"
        assert req.headers["Authorization"] == f"Bearer {_PAT_BEARER}"
        assert req.url.params.get("state") == "all"
        assert req.url.params.get("direction") == "asc"
        assert req.url.params.get("per_page") == "100"
        # First poll: cursor is None so since= falls back to ~7 days ago.
        assert req.url.params.get("since"), "missing since param"

        return httpx.Response(
            200,
            json=[
                {
                    "number": 10,
                    "title": "first issue",
                    "updated_at": "2026-05-10T10:00:00Z",
                    "user": {"login": "alice"},
                    "html_url": "https://github.com/acme/api/issues/10",
                },
                {
                    "number": 11,
                    "title": "a PR",
                    "updated_at": "2026-05-11T10:00:00Z",
                    "user": {"login": "bob"},
                    "html_url": "https://github.com/acme/api/pull/11",
                    "pull_request": {"url": "..."},
                },
                {
                    "number": 12,
                    "title": "later issue",
                    "updated_at": "2026-05-12T10:00:00Z",
                    "user": {"login": "carol"},
                    "html_url": "https://github.com/acme/api/issues/12",
                },
            ],
        )

    requests = _install_transport(monkeypatch, handler)

    result = await GitHubPoller().poll(
        customer_id=_CUST, resource_id=_RESOURCE, cursor=None
    )

    assert result.error is None
    assert len(result.documents) == 3
    # Cursor advances to the max updated_at across issues + PRs.
    assert result.next_cursor == "2026-05-12T10:00:00Z"
    assert len(requests) == 1

    # Shape mirrors shared.models.WebhookEvent field-for-field so the
    # PR C sink can lift dicts into WebhookEvent without remapping.
    by_event = {d["headers"]["X-GitHub-Event"]: d for d in result.documents}
    assert "issues" in by_event
    assert "pull_request" in by_event

    issue_doc = next(
        d for d in result.documents
        if d["headers"]["X-GitHub-Event"] == "issues"
        and d["raw_payload"]["issue"]["number"] == 10
    )
    assert issue_doc["customer_id"] == _CUST
    assert issue_doc["source_system"] == SourceSystem.GITHUB.value
    assert issue_doc["raw_payload"]["repository"]["full_name"] == "acme/api"
    assert issue_doc["raw_payload"]["repository"]["owner"]["login"] == "acme"
    assert issue_doc["raw_payload"]["action"] == "opened"
    assert issue_doc["received_at"] == "2026-05-10T10:00:00Z"
    assert issue_doc["source_event_id"].startswith("issue:acme/api:10:")

    pr_doc = next(
        d for d in result.documents
        if d["headers"]["X-GitHub-Event"] == "pull_request"
    )
    assert pr_doc["raw_payload"]["pull_request"]["number"] == 11
    assert pr_doc["source_event_id"].startswith("pr:acme/api:11:")


@pytest.mark.asyncio
async def test_subsequent_poll_uses_cursor_as_since(
    monkeypatch: pytest.MonkeyPatch,
):
    """When a prior cursor is present, it is passed as ``since=`` verbatim."""
    _patch_load_token(monkeypatch, token=_make_token())

    seen_since: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen_since.append(req.url.params.get("since") or "")
        return httpx.Response(200, json=[])

    _install_transport(monkeypatch, handler)

    cursor = "2026-05-01T00:00:00Z"
    result = await GitHubPoller().poll(
        customer_id=_CUST, resource_id=_RESOURCE, cursor=cursor
    )

    assert result.error is None
    assert seen_since == [cursor]


# --- pagination -------------------------------------------------------------


@pytest.mark.asyncio
async def test_paginates_via_link_header(monkeypatch: pytest.MonkeyPatch):
    _patch_load_token(monkeypatch, token=_make_token())

    page2_url = (
        "https://api.github.com/repos/acme/api/issues?since=2020-01-01T00%3A00%3A00Z"
        "&state=all&direction=asc&sort=updated&per_page=100&page=2"
    )

    def handler(req: httpx.Request) -> httpx.Response:
        # Page 1 carries the rel=next header; page 2 does not.
        if req.url.params.get("page") == "2":
            return httpx.Response(
                200,
                json=[
                    {
                        "number": 200,
                        "updated_at": "2026-05-14T00:00:00Z",
                        "title": "page2 issue",
                    },
                ],
            )

        return httpx.Response(
            200,
            headers={"Link": f'<{page2_url}>; rel="next", <...>; rel="last"'},
            json=[
                {
                    "number": 100,
                    "updated_at": "2026-05-13T00:00:00Z",
                    "title": "page1 issue",
                },
            ],
        )

    requests = _install_transport(monkeypatch, handler)

    result = await GitHubPoller().poll(
        customer_id=_CUST, resource_id=_RESOURCE, cursor="2020-01-01T00:00:00Z"
    )

    assert result.error is None
    assert len(result.documents) == 2
    # Cursor advances to the max across both pages.
    assert result.next_cursor == "2026-05-14T00:00:00Z"
    # Two HTTP calls — initial + the rel=next follow.
    assert len(requests) == 2
    assert requests[1].url.params.get("page") == "2"


# --- error paths ------------------------------------------------------------


@pytest.mark.asyncio
async def test_rate_limit_429_returns_error_and_no_documents(
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_load_token(monkeypatch, token=_make_token())

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            headers={"Retry-After": "30"},
            json={"message": "too many requests"},
        )

    _install_transport(monkeypatch, handler)

    result = await GitHubPoller().poll(
        customer_id=_CUST, resource_id=_RESOURCE, cursor=None
    )

    assert result.documents == []
    assert result.next_cursor is None
    assert result.error is not None
    assert "rate limited" in result.error
    assert "429" in result.error


@pytest.mark.asyncio
async def test_secondary_rate_limit_403_with_zero_remaining_returns_error(
    monkeypatch: pytest.MonkeyPatch,
):
    """GitHub uses 403 with ``x-ratelimit-remaining: 0`` for the secondary
    rate limit; treat it the same as a 429."""
    _patch_load_token(monkeypatch, token=_make_token())

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            headers={"x-ratelimit-remaining": "0", "Retry-After": "60"},
            json={"message": "secondary rate limit"},
        )

    _install_transport(monkeypatch, handler)

    result = await GitHubPoller().poll(
        customer_id=_CUST, resource_id=_RESOURCE, cursor=None
    )

    assert result.documents == []
    assert result.next_cursor is None
    assert result.error is not None
    assert "rate limited" in result.error


@pytest.mark.asyncio
async def test_5xx_returns_error(monkeypatch: pytest.MonkeyPatch):
    _patch_load_token(monkeypatch, token=_make_token())

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="upstream down")

    _install_transport(monkeypatch, handler)

    result = await GitHubPoller().poll(
        customer_id=_CUST, resource_id=_RESOURCE, cursor=None
    )

    assert result.documents == []
    assert result.next_cursor is None
    assert result.error is not None
    assert "503" in result.error


@pytest.mark.asyncio
async def test_empty_response_returns_no_cursor_advance(
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_load_token(monkeypatch, token=_make_token())

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[])

    _install_transport(monkeypatch, handler)

    result = await GitHubPoller().poll(
        customer_id=_CUST, resource_id=_RESOURCE, cursor="2026-05-10T00:00:00Z"
    )

    assert result.error is None
    assert result.documents == []
    # next_cursor=None tells the scheduler to keep the existing cursor
    # (advance_cursor's COALESCE is the load-bearing piece here).
    assert result.next_cursor is None


# --- auth paths -------------------------------------------------------------


@pytest.mark.asyncio
async def test_installation_scope_fetches_app_bearer(
    monkeypatch: pytest.MonkeyPatch,
):
    """When token.scope is ``installation:<id>``, the poller mints a fresh
    App installation token via prbe-backend instead of using the stored
    access_token."""
    _patch_load_token(
        monkeypatch,
        token=_make_token(
            scope=f"{GITHUB_INSTALLATION_SCOPE_PREFIX}42",
            access_token="placeholder-not-used",
        ),
    )
    _patch_fetch_installation_token(monkeypatch)

    seen_auth: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen_auth.append(req.headers.get("Authorization", ""))
        return httpx.Response(200, json=[])

    _install_transport(monkeypatch, handler)

    result = await GitHubPoller().poll(
        customer_id=_CUST, resource_id=_RESOURCE, cursor=None
    )

    assert result.error is None
    assert seen_auth == [f"Bearer {_APP_BEARER}"]


@pytest.mark.asyncio
async def test_missing_token_returns_error(monkeypatch: pytest.MonkeyPatch):
    _patch_load_token(monkeypatch, token=None)
    # No transport patched — the poller must not reach an HTTP call.

    result = await GitHubPoller().poll(
        customer_id=_CUST, resource_id=_RESOURCE, cursor=None
    )

    assert result.documents == []
    assert result.next_cursor is None
    assert result.error is not None
    assert "no active github integration_tokens" in result.error


# --- resource_id validation -------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "just-a-word",
        "/missing-owner",
        "missing-repo/",
        "too/many/slashes",
    ],
)
@pytest.mark.asyncio
async def test_malformed_resource_id_returns_error_without_http(
    monkeypatch: pytest.MonkeyPatch, bad: str
):
    # No load_token patch — we should never reach it.
    result = await GitHubPoller().poll(
        customer_id=_CUST, resource_id=bad, cursor=None
    )
    assert result.documents == []
    assert result.next_cursor is None
    assert result.error is not None
    assert "invalid resource_id" in result.error


# --- registry --------------------------------------------------------------


def test_module_import_registers_poller():
    """Just importing the module wires GitHubPoller into the registry."""
    from kb.polling.base import get_poller

    assert get_poller(SourceSystem.GITHUB) is GitHubPoller
