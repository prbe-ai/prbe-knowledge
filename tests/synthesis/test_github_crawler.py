"""Unit tests for ``GitHubCrawlerAgent`` (Lane D).

These exercise the crawler's plumbing in isolation: tool dispatch
routing, cursor encoding, the 12-month bound on PRs/issues vs all-time
on commits, soft-halt handling, and the no-bearer / no-repos paths.

The snapshot tests in tests/synthesis/evals/github/ are the integration
gate; this file is the per-method gate.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
import respx

from services.synthesis.api_clients.github import (
    GitHubAPIClient,
    GitHubRateLimitExhausted,
)
from services.synthesis.crawlers.github import (
    BootstrapWikiRuntime,
    GitHubCrawlerAgent,
    _Cursor,
)
from shared.exceptions import ToolValidationError

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def http() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient() as client:
        yield client


def _make_agent(
    http_client: httpx.AsyncClient,
    *,
    bearer: str | None = "ghs_x",
    runtime: BootstrapWikiRuntime | None = None,
    client: GitHubAPIClient | None = None,
    llm_client: Any | None = None,
) -> GitHubCrawlerAgent:
    """Construct an agent with a deterministic bearer resolver."""

    async def _resolver() -> str | None:
        return bearer

    return GitHubCrawlerAgent(
        customer_id="test-customer",
        run_id=99,
        bearer_resolver=_resolver,
        http=http_client,
        settings=None,
        llm_client=llm_client,
        runtime=runtime,
        client=client,
    )


class _NoopRuntime(BootstrapWikiRuntime):
    """Bootstrap runtime initialized without a Normalizer / store / DB.

    Mirrors the eval harness's in-memory runtime: bypass __init__, set
    state by hand. The dispatch_tool path still routes through the
    parent's pydantic validators.
    """

    def __init__(self) -> None:
        self.customer_id = "test-customer"
        self.agent_run_id = "noop-runtime"
        self._run_id = 99
        self._run_kind = "bootstrap"
        self._normalizer = None
        self._store = None
        self._ctx = None
        self._pending_updates = {}
        self._pending_creates = {}
        self._applied_queue_ids = set()
        self._skipped_queue_ids = set()
        self.is_done = False
        self._wiki_index_cache = []

    async def commit(self) -> None:
        return None

    async def _tool_read_page(self, args: Any) -> dict[str, Any]:
        # Skip the DB-touching parent implementation; treat all reads as
        # page-not-found unless the agent already staged a create.
        return {"error": "page_not_found", "wiki_type": args.wiki_type, "slug": args.slug}


# ---------------------------------------------------------------------------
# Cursor round-trip
# ---------------------------------------------------------------------------


def test_cursor_roundtrips_through_encode_decode() -> None:
    cur = _Cursor(full_name="x/y", offset=50)
    decoded = _Cursor.decode(cur.encode())
    assert decoded == cur


def test_cursor_decode_none_returns_none() -> None:
    assert _Cursor.decode(None) is None
    assert _Cursor.decode("") is None


def test_cursor_decode_garbage_raises_validation_error() -> None:
    with pytest.raises(ToolValidationError):
        _Cursor.decode("definitely_not_base64_json")


# ---------------------------------------------------------------------------
# Tool dispatch routing — source vs wiki
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_router_dispatches_source_tools_to_source_handler(
    http: httpx.AsyncClient,
) -> None:
    """``list_repos`` flows through the source-tool path; ``read_page``
    flows through the runtime's parent dispatch.
    """
    runtime = _NoopRuntime()
    agent = _make_agent(http, runtime=runtime)
    # Stamp a client so list_repos has something to call against.
    with respx.mock(assert_all_called=False) as router:
        router.get("https://api.github.com/installation/repositories").mock(
            return_value=httpx.Response(200, json={"repositories": []})
        )
        agent._client = GitHubAPIClient("ghs_x", http, target_rps=1000.0)
        router_fn = agent._build_router(runtime.dispatch_tool)
        repos_result = await router_fn("list_repos", {})
        assert repos_result == {"repos": [], "next_cursor": None}

    # Wiki tool path: read_page on a missing page returns the runtime's
    # typed page_not_found result, not the source handler's shape.
    read_result = await router_fn("read_page", {"wiki_type": "decision", "slug": "missing"})
    assert read_result.get("error") == "page_not_found"


# ---------------------------------------------------------------------------
# 12-month bound for PRs/issues — but NOT commits
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_pulls_applies_twelve_month_bound(
    http: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The crawler's _fetch_pulls passes since= to the API client."""
    captured: dict[str, Any] = {}

    class _FakeClient:
        async def list_pulls(
            self, full_name: str, *, since=None, until=None
        ) -> AsyncIterator[dict[str, Any]]:
            captured["since"] = since
            captured["until"] = until
            if False:  # pragma: no cover — make this an async iterator
                yield {}

        async def list_commits(
            self, full_name: str, *, since=None, until=None
        ) -> AsyncIterator[dict[str, Any]]:
            captured["since"] = since
            captured["until"] = until
            if False:  # pragma: no cover
                yield {}

    runtime = _NoopRuntime()
    agent = _make_agent(http, runtime=runtime)
    agent._client = _FakeClient()  # type: ignore[assignment]
    await agent._fetch_pulls("x/y")
    # The cutoff is `now - 365 days`; we just assert it's set and naive-aware.
    assert captured["since"] is not None
    # commit-listing path should not pass a since.
    captured.clear()

    await agent._fetch_commits("x/y")
    assert captured["since"] is None
    assert captured["until"] is None


# ---------------------------------------------------------------------------
# Soft halt: rate-limit exhaustion -> halt_reason='rate_limited'
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_repos_rate_limit_exhaustion_sets_counter(
    http: httpx.AsyncClient,
) -> None:
    """When the API client raises ``GitHubRateLimitExhausted``, the source
    tool returns a typed result and flips ``counters.rate_limited``.
    """

    class _RaisingClient:
        async def list_installation_repos(
            self,
        ) -> AsyncIterator[dict[str, Any]]:
            raise GitHubRateLimitExhausted("test")
            if False:  # pragma: no cover — keep mypy happy
                yield {}

    runtime = _NoopRuntime()
    agent = _make_agent(http, runtime=runtime)
    agent._client = _RaisingClient()  # type: ignore[assignment]

    out = await agent._tool_list_repos()
    assert out["error"] == "rate_limited"
    assert agent.counters.rate_limited is True


# ---------------------------------------------------------------------------
# Empty installation: no repos -> agent.run() short-circuits cleanly
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_with_no_bearer_returns_auth_missing(
    http: httpx.AsyncClient,
) -> None:
    """A None-returning bearer_resolver yields a ``halt_reason='auth.missing'``
    result without crashing.
    """
    agent = _make_agent(http, bearer=None)
    result = await agent.run()
    assert result.halt_reason == "auth.missing"
    assert result.error is None
    assert result.pages_created == 0
    assert result.pages_updated == 0


@pytest.mark.asyncio
async def test_run_with_failing_bearer_resolver_returns_error_result(
    http: httpx.AsyncClient,
) -> None:
    async def _raising_resolver() -> str | None:
        raise RuntimeError("backend down")

    agent = GitHubCrawlerAgent(
        customer_id="test-customer",
        run_id=99,
        bearer_resolver=_raising_resolver,
        http=http,
        settings=None,
    )
    result = await agent.run()
    assert result.error is not None
    assert "backend down" in result.error


@pytest.mark.asyncio
async def test_blocked_tools_raise_in_bootstrap_runtime(
    http: httpx.AsyncClient,
) -> None:
    """next_events / get_event_body / skip_events are intentionally
    unsupported in bootstrap mode. Calls return a typed ToolValidationError
    so the harness can route the error back to the LLM as a tool result.
    """
    runtime = _NoopRuntime()
    with pytest.raises(ToolValidationError):
        await runtime.dispatch_tool("next_events", {"count": 50})
    with pytest.raises(ToolValidationError):
        await runtime.dispatch_tool("skip_events", {"queue_ids": [1], "reason": "x"})
    with pytest.raises(ToolValidationError):
        await runtime.dispatch_tool("get_event_body", {"queue_id": 1})


# ---------------------------------------------------------------------------
# Pagination: cursor invalid -> typed error, valid -> next page slice
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_pulls_cursor_invalid_returns_typed_error(
    http: httpx.AsyncClient,
) -> None:
    runtime = _NoopRuntime()
    agent = _make_agent(http, runtime=runtime)

    class _EmptyClient:
        async def list_pulls(
            self, full_name: str, *, since=None, until=None
        ) -> AsyncIterator[dict[str, Any]]:
            if False:  # pragma: no cover
                yield {}

    agent._client = _EmptyClient()  # type: ignore[assignment]
    out = await agent.dispatch_source_tool("list_pulls", {"full_name": "x/y", "cursor": "garbage"})
    # The garbage cursor is rejected from inside _paged_list with a typed error
    # rather than a tool exception.
    assert out["error"] == "invalid_cursor"


@pytest.mark.asyncio
async def test_list_pulls_returns_next_cursor_when_more_remaining(
    http: httpx.AsyncClient,
) -> None:
    """A repo with > 50 PRs returns a ``next_cursor`` on the first call."""
    fake_pulls = [
        {
            "number": i,
            "title": f"pr {i}",
            "user": {"login": "u"},
            "labels": [],
            "requested_reviewers": [],
        }
        for i in range(60)
    ]

    class _BulkClient:
        async def list_pulls(
            self, full_name: str, *, since=None, until=None
        ) -> AsyncIterator[dict[str, Any]]:
            for pr in fake_pulls:
                yield pr

    runtime = _NoopRuntime()
    agent = _make_agent(http, runtime=runtime)
    agent._client = _BulkClient()  # type: ignore[assignment]

    first = await agent.dispatch_source_tool("list_pulls", {"full_name": "x/y"})
    assert len(first["pulls"]) == 50
    assert first["next_cursor"] is not None
    assert first["remaining"] == 10

    second = await agent.dispatch_source_tool(
        "list_pulls", {"full_name": "x/y", "cursor": first["next_cursor"]}
    )
    assert len(second["pulls"]) == 10
    assert second["next_cursor"] is None
    assert second["remaining"] == 0


# ---------------------------------------------------------------------------
# Source-tool validators reject malformed input
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_source_tool_unknown_name_raises(
    http: httpx.AsyncClient,
) -> None:
    runtime = _NoopRuntime()
    agent = _make_agent(http, runtime=runtime)
    with pytest.raises(ToolValidationError):
        await agent.dispatch_source_tool("not_a_tool", {})


@pytest.mark.asyncio
async def test_dispatch_source_tool_invalid_args_raise(
    http: httpx.AsyncClient,
) -> None:
    runtime = _NoopRuntime()
    agent = _make_agent(http, runtime=runtime)
    # missing required `full_name`
    with pytest.raises(ToolValidationError):
        await agent.dispatch_source_tool("list_pulls", {})


# ---------------------------------------------------------------------------
# Registry: GitHubCrawlerAgent is registered under "github"
# ---------------------------------------------------------------------------


def test_registry_includes_github_crawler() -> None:
    from services.synthesis.crawlers import REGISTRY

    assert REGISTRY.get("github") is GitHubCrawlerAgent
