"""GitHub API client. Tests use respx to mock httpx.AsyncClient."""

from __future__ import annotations

import httpx
import pytest
import respx

from scripts.synth.extractor.github_api import GithubClient, parse_repo_url


def test_parse_repo_url_https_form() -> None:
    assert parse_repo_url("github.com/prbe-ai/prbe-knowledge") == ("prbe-ai", "prbe-knowledge")
    assert parse_repo_url("https://github.com/prbe-ai/prbe-knowledge") == ("prbe-ai", "prbe-knowledge")
    assert parse_repo_url("https://github.com/prbe-ai/prbe-knowledge.git") == ("prbe-ai", "prbe-knowledge")


def test_parse_repo_url_rejects_non_github() -> None:
    with pytest.raises(ValueError):
        parse_repo_url("gitlab.com/x/y")


@pytest.mark.asyncio
async def test_fetch_contributors_returns_username_and_display() -> None:
    with respx.mock(base_url="https://api.github.com") as router:
        router.get("/repos/x/y/contributors").mock(
            return_value=httpx.Response(
                200,
                json=[
                    {"login": "alice", "id": 1, "contributions": 42},
                    {"login": "bob", "id": 2, "contributions": 17},
                ],
            )
        )
        router.get("/users/alice").mock(
            return_value=httpx.Response(200, json={"login": "alice", "name": "Alice X", "email": "alice@example.com"})
        )
        router.get("/users/bob").mock(
            return_value=httpx.Response(200, json={"login": "bob", "name": None, "email": None})
        )

        client = GithubClient(token="t")
        contributors = await client.fetch_contributors("x", "y")
        await client.close()

    assert {c.gh_username for c in contributors} == {"alice", "bob"}
    alice = next(c for c in contributors if c.gh_username == "alice")
    assert alice.display_name == "Alice X"
    assert alice.email_aliases == ("alice@example.com",)


@pytest.mark.asyncio
async def test_fetch_issues_paginates_and_strips_pull_requests() -> None:
    with respx.mock(base_url="https://api.github.com") as router:
        page1 = [
            {
                "number": 1, "title": "issue 1", "body": "b1",
                "state": "open", "labels": [{"name": "bug"}],
                "assignees": [{"login": "alice"}],
                "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-02T00:00:00Z",
            },
            {
                "number": 2, "title": "PR linked as issue", "body": "p",
                "state": "open", "labels": [],
                "assignees": [],
                "pull_request": {"url": "..."},
                "created_at": "2026-01-03T00:00:00Z", "updated_at": "2026-01-04T00:00:00Z",
            },
        ]
        router.get("/repos/x/y/issues").mock(
            return_value=httpx.Response(200, json=page1)
        )

        client = GithubClient(token="t")
        issues = await client.fetch_issues("x", "y", limit=200)
        await client.close()

    # PR-shaped entries dropped
    assert len(issues) == 1
    assert issues[0].number == 1
    assert issues[0].labels == ("bug",)


@pytest.mark.asyncio
async def test_fetch_handles_404_as_none() -> None:
    """If a repo doesn't exist or the token lacks access, fetch_* returns
    an empty list rather than crashing — the caller treats no-data the
    same way the no-token path does."""
    with respx.mock(base_url="https://api.github.com") as router:
        router.get("/repos/x/missing/contributors").mock(
            return_value=httpx.Response(404, json={"message": "Not Found"})
        )
        client = GithubClient(token="t")
        contributors = await client.fetch_contributors("x", "missing")
        await client.close()
    assert contributors == []
