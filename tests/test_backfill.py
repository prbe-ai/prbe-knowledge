"""Backfill runner + per-connector pagination tests.

Exercises:
  - enqueue_backfill → backfill_state row
  - claim_pending_backfill → atomic SKIP LOCKED claim
  - Slack backfill paginates conversations.list + conversations.history
  - Linear backfill paginates issues + nested comments
  - Notion backfill paginates /search
  - Sentry backfill paginates issues via Link-header cursor
  - run_backfill wires a connector to the queue + records progress
"""

from __future__ import annotations

import httpx
import pytest

from services.ingestion.backfill_runner import (
    claim_pending_backfill,
    enqueue_backfill,
)
from services.ingestion.handlers.base import ConnectorContext
from shared.config import Settings, get_settings
from shared.constants import BackfillStatus, SourceSystem
from shared.db import raw_conn
from shared.embeddings import reset_embedder
from shared.storage import reset_store


@pytest.fixture(autouse=True)
def _patch(monkeypatch, settings: Settings):
    monkeypatch.setenv("TOKEN_ENCRYPTION_KEY", settings.token_encryption_key.get_secret_value())
    monkeypatch.setenv("ENVIRONMENT", "local")
    reset_embedder()
    reset_store()
    get_settings.cache_clear()  # type: ignore[attr-defined]


# -------------------------------- runner ------------------------------------


@pytest.mark.asyncio
async def test_enqueue_and_claim(live_db) -> None:
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers (customer_id, display_name, api_key_hash) VALUES ('cust-bf','x','y') ON CONFLICT DO NOTHING"
        )

    await enqueue_backfill("cust-bf", SourceSystem.SLACK)

    async with raw_conn() as conn:
        row = await conn.fetchrow(
            "SELECT status FROM backfill_state WHERE customer_id='cust-bf'"
        )
    assert row["status"] == BackfillStatus.PENDING.value

    claimed = await claim_pending_backfill()
    assert claimed == ("cust-bf", SourceSystem.SLACK)

    # After claim, row transitions to running; a second claim returns None.
    async with raw_conn() as conn:
        row = await conn.fetchrow(
            "SELECT status FROM backfill_state WHERE customer_id='cust-bf'"
        )
    assert row["status"] == BackfillStatus.RUNNING.value
    assert await claim_pending_backfill() is None


# -------------------------------- slack -------------------------------------


def _mock_transport(routes: dict) -> httpx.MockTransport:
    """Build a MockTransport that dispatches by (method, path, params)."""

    def handler(request: httpx.Request) -> httpx.Response:
        key = (request.method, request.url.path)
        for (method, path_prefix), responder in routes.items():
            if method == request.method and request.url.path == path_prefix:
                return responder(request)
        return httpx.Response(404, json={"error": f"unmocked {key}"})

    return httpx.MockTransport(handler)


def _ctx_with_transport(transport: httpx.MockTransport) -> ConnectorContext:
    from shared.config import Settings as _S

    return ConnectorContext(
        settings=_S(),
        http=httpx.AsyncClient(transport=transport),
    )


@pytest.mark.asyncio
async def test_slack_backfill_paginates_channels_and_history() -> None:
    """Slack: list channels, then paginate history per channel. All messages yielded."""
    calls: list[str] = []

    def auth_test(req):
        return httpx.Response(200, json={"ok": True, "team_id": "T1", "team": "Acme"})

    def list_channels(req):
        return httpx.Response(
            200,
            json={
                "ok": True,
                "channels": [
                    {"id": "C1", "is_member": True},
                    {"id": "C2", "is_member": True},
                ],
                "response_metadata": {"next_cursor": ""},
            },
        )

    def history(req):
        channel = req.url.params["channel"]
        calls.append(f"history:{channel}")
        # Use real-looking Slack ts values (unix_secs.microsecs).
        base = 1713628800 if channel == "C1" else 1713628900
        msgs = [
            {"type": "message", "channel": channel, "ts": f"{base}.000100", "text": "hi", "user": "U1"},
            {"type": "message", "channel": channel, "ts": f"{base + 1}.000200", "text": "ho", "user": "U2"},
        ]
        return httpx.Response(
            200,
            json={"ok": True, "messages": msgs, "response_metadata": {"next_cursor": ""}},
        )

    transport = _mock_transport(
        {
            ("POST", "/api/auth.test"): auth_test,
            ("GET", "/api/conversations.list"): list_channels,
            ("GET", "/api/conversations.history"): history,
        }
    )

    from services.ingestion.handlers.registry import build_connector
    from shared.models import IntegrationToken

    slack = build_connector(SourceSystem.SLACK, _ctx_with_transport(transport))
    token = IntegrationToken(
        customer_id="cust", source_system=SourceSystem.SLACK, access_token="x"
    )

    events = [e async for e in slack.backfill("cust", token)]
    assert len(events) == 4  # 2 channels, 2 messages each
    assert {e.raw_payload["event"]["channel"] for e in events} == {"C1", "C2"}
    assert calls == ["history:C1", "history:C2"]


# -------------------------------- linear ------------------------------------


@pytest.mark.asyncio
async def test_linear_backfill_paginates_issues() -> None:
    """Linear: GraphQL pagination. Two pages, each with 1 issue + 1 comment."""
    page = {"ix": 0}

    def graphql(req):
        payload = req.read()
        body = payload.decode()
        # Org-id probe is the short `{ organization { id } }` query.
        if "issues(first" not in body:
            return httpx.Response(200, json={"data": {"organization": {"id": "ORG"}}})
        # Backfill issues query.
        if page["ix"] == 0:
            page["ix"] = 1
            return httpx.Response(
                200,
                json={
                    "data": {
                        "issues": {
                            "pageInfo": {"hasNextPage": True, "endCursor": "cur1"},
                            "nodes": [
                                {
                                    "id": "I1",
                                    "identifier": "ENG-1",
                                    "title": "first",
                                    "description": "",
                                    "url": "https://linear.app/x/I1",
                                    "createdAt": "2026-04-01T00:00:00Z",
                                    "updatedAt": "2026-04-01T00:00:00Z",
                                    "team": {"id": "T1", "key": "ENG", "name": "Eng"},
                                    "teamId": "T1",
                                    "comments": {
                                        "nodes": [
                                            {
                                                "id": "C1",
                                                "body": "hi",
                                                "url": "https://linear.app/x/I1#c1",
                                                "createdAt": "2026-04-01T00:10:00Z",
                                                "updatedAt": "2026-04-01T00:10:00Z",
                                                "user": {"id": "U1", "name": "alice"},
                                            }
                                        ]
                                    },
                                }
                            ],
                        }
                    }
                },
            )
        return httpx.Response(
            200,
            json={
                "data": {
                    "issues": {
                        "pageInfo": {"hasNextPage": False, "endCursor": None},
                        "nodes": [
                            {
                                "id": "I2",
                                "identifier": "ENG-2",
                                "title": "second",
                                "description": "",
                                "url": "https://linear.app/x/I2",
                                "createdAt": "2026-04-02T00:00:00Z",
                                "updatedAt": "2026-04-02T00:00:00Z",
                                "team": {"id": "T1", "key": "ENG", "name": "Eng"},
                                "teamId": "T1",
                                "comments": {"nodes": []},
                            }
                        ],
                    }
                }
            },
        )

    transport = _mock_transport({("POST", "/graphql"): graphql})

    from services.ingestion.handlers.registry import build_connector
    from shared.models import IntegrationToken

    linear = build_connector(SourceSystem.LINEAR, _ctx_with_transport(transport))
    token = IntegrationToken(
        customer_id="c", source_system=SourceSystem.LINEAR, access_token="x"
    )
    events = [e async for e in linear.backfill("c", token)]

    # 2 issues + 1 comment = 3 events
    assert len(events) == 3
    types = [e.raw_payload["type"] for e in events]
    assert types.count("Issue") == 2
    assert types.count("Comment") == 1


# -------------------------------- notion ------------------------------------


@pytest.mark.asyncio
async def test_notion_backfill_paginates_search() -> None:
    """Notion: paginated /search. Yields one event per page entity."""
    page = {"ix": 0}

    def users_me(req):
        return httpx.Response(
            200,
            json={"bot": {"workspace_id": "WS1", "workspace_name": "Acme WS"}},
        )

    def search(req):
        if page["ix"] == 0:
            page["ix"] = 1
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "object": "page",
                            "id": "P1",
                            "last_edited_time": "2026-04-01T00:00:00.000Z",
                        }
                    ],
                    "has_more": True,
                    "next_cursor": "cur1",
                },
            )
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "object": "database",
                        "id": "D1",
                        "last_edited_time": "2026-04-02T00:00:00.000Z",
                    }
                ],
                "has_more": False,
                "next_cursor": None,
            },
        )

    transport = _mock_transport(
        {
            ("GET", "/v1/users/me"): users_me,
            ("POST", "/v1/search"): search,
        }
    )

    from services.ingestion.handlers.registry import build_connector
    from shared.models import IntegrationToken

    notion = build_connector(SourceSystem.NOTION, _ctx_with_transport(transport))
    token = IntegrationToken(
        customer_id="c", source_system=SourceSystem.NOTION, access_token="x"
    )
    events = [e async for e in notion.backfill("c", token)]
    assert len(events) == 2
    kinds = [e.raw_payload["entity"]["type"] for e in events]
    assert set(kinds) == {"page", "database"}


# -------------------------------- sentry ------------------------------------


@pytest.mark.asyncio
async def test_sentry_backfill_paginates_issues() -> None:
    """Sentry: Link-header cursor pagination. Two pages of issues."""
    page = {"ix": 0}

    def orgs(req):
        return httpx.Response(200, json=[{"slug": "acme"}])

    def issues(req):
        if page["ix"] == 0:
            page["ix"] = 1
            return httpx.Response(
                200,
                json=[
                    {"id": "1", "lastSeen": "2026-04-01T00:00:00Z", "title": "a"},
                    {"id": "2", "lastSeen": "2026-04-01T00:01:00Z", "title": "b"},
                ],
                headers={
                    "link": (
                        '<https://sentry.io/api/0/organizations/acme/issues/?cursor=p2>; '
                        'rel="next"; results="true"; cursor="p2"'
                    )
                },
            )
        return httpx.Response(
            200,
            json=[{"id": "3", "lastSeen": "2026-04-02T00:00:00Z", "title": "c"}],
            headers={
                "link": (
                    '<https://sentry.io/api/0/organizations/acme/issues/?cursor=end>; '
                    'rel="next"; results="false"'
                )
            },
        )

    transport = _mock_transport(
        {
            ("GET", "/api/0/organizations/"): orgs,
            ("GET", "/api/0/organizations/acme/issues/"): issues,
        }
    )

    from services.ingestion.handlers.registry import build_connector
    from shared.models import IntegrationToken

    sentry = build_connector(SourceSystem.SENTRY, _ctx_with_transport(transport))
    token = IntegrationToken(
        customer_id="c", source_system=SourceSystem.SENTRY, access_token="x"
    )
    events = [e async for e in sentry.backfill("c", token)]
    assert len(events) == 3
    assert {e.source_event_id for e in events} == {
        "issue:1:backfill",
        "issue:2:backfill",
        "issue:3:backfill",
    }


# -------------------------------- github -----------------------------------


@pytest.mark.asyncio
async def test_github_backfill_paginates_repos_pulls_and_issues() -> None:
    """GitHub: list installation repos, then pulls + issues per repo. Issue-shaped PRs filtered."""

    def installation_repos(req):
        return httpx.Response(
            200,
            json={
                "repositories": [
                    {
                        "full_name": "acme/api",
                        "name": "api",
                        "owner": {"login": "acme"},
                        "private": True,
                    },
                    {
                        "full_name": "acme/web",
                        "name": "web",
                        "owner": {"login": "acme"},
                        "private": False,
                    },
                ]
            },
        )

    def api_pulls(req):
        return httpx.Response(
            200,
            json=[
                {"number": 1, "updated_at": "2026-04-01T00:00:00Z", "title": "pr1"},
                {"number": 2, "updated_at": "2026-04-02T00:00:00Z", "title": "pr2"},
            ],
        )

    def api_issues(req):
        return httpx.Response(
            200,
            json=[
                {"number": 10, "updated_at": "2026-04-03T00:00:00Z", "title": "issue"},
                # Issue-shaped PR — must be filtered out by the backfill.
                {
                    "number": 11,
                    "updated_at": "2026-04-04T00:00:00Z",
                    "title": "pr-as-issue",
                    "pull_request": {"url": "..."},
                },
            ],
        )

    def web_pulls(req):
        return httpx.Response(
            200,
            json=[
                {"number": 3, "updated_at": "2026-04-05T00:00:00Z", "title": "pr3"},
            ],
        )

    def web_issues(req):
        return httpx.Response(200, json=[])

    transport = _mock_transport(
        {
            ("GET", "/installation/repositories"): installation_repos,
            ("GET", "/repos/acme/api/pulls"): api_pulls,
            ("GET", "/repos/acme/api/issues"): api_issues,
            ("GET", "/repos/acme/web/pulls"): web_pulls,
            ("GET", "/repos/acme/web/issues"): web_issues,
        }
    )

    from services.ingestion.handlers.registry import build_connector
    from shared.models import IntegrationToken

    gh = build_connector(SourceSystem.GITHUB, _ctx_with_transport(transport))
    token = IntegrationToken(
        customer_id="cust", source_system=SourceSystem.GITHUB, access_token="x"
    )

    events = [e async for e in gh.backfill("cust", token)]

    # 3 PRs + 1 issue = 4 events; the issue-shaped PR was filtered.
    assert len(events) == 4

    pr_events = [e for e in events if e.headers.get("X-GitHub-Event") == "pull_request"]
    issue_events = [e for e in events if e.headers.get("X-GitHub-Event") == "issues"]
    assert len(pr_events) == 3
    assert len(issue_events) == 1

    # Confirm the issue-shaped PR (number=11) was filtered.
    issue_numbers = {e.raw_payload["issue"]["number"] for e in issue_events}
    assert issue_numbers == {10}

    pr_numbers = {e.raw_payload["pull_request"]["number"] for e in pr_events}
    assert pr_numbers == {1, 2, 3}


# --------------------- backfill status endpoint -----------------------------


@pytest.mark.asyncio
async def test_backfill_status_endpoint(live_db) -> None:
    """GET /backfill/status returns per-source rows for a customer."""
    from httpx import ASGITransport

    from shared.db import close_pool, init_pool

    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers (customer_id, display_name, api_key_hash) VALUES ('cust-s','x','y') ON CONFLICT DO NOTHING"
        )

    await enqueue_backfill("cust-s", SourceSystem.SLACK)
    await enqueue_backfill("cust-s", SourceSystem.LINEAR)

    from services.ingestion.main import app as ingestion_app

    await close_pool()
    transport = ASGITransport(app=ingestion_app)
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://t") as client,
        ingestion_app.router.lifespan_context(ingestion_app),
    ):
        resp = await client.get("/backfill/status?customer_id=cust-s")
    await init_pool(settings=None)

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["customer_id"] == "cust-s"
    sources = {s["source"]: s for s in body["sources"]}
    assert set(sources.keys()) == {"slack", "linear"}
    assert sources["slack"]["status"] == "pending"
