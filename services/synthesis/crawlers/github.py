"""``GitHubCrawlerAgent`` — Lane D's first concrete BootstrapAgent.

Walks every accessible installation repo for the customer, reading
PRs/issues/commits/reviews via ``GitHubAPIClient``, and emits wiki pages
through the shared ``WikiAgentRuntime`` write tools (update_page /
create_page / done). Recency-first: each repo's PRs/issues are bounded
to ``WIKI_BOOTSTRAP_GITHUB_PRS_DAYS`` (12 months by default per the
locked plan), commits walk all-time so old structural commits ("first
added auth middleware") still surface even when ticket history is
bounded.

Architecture (see PR body for the (a)/(b) write-only-runtime decision):

    GitHubCrawlerAgent
      ├─ GitHubAPIClient         (token bucket + 429 ladder, Lane E)
      ├─ BootstrapWikiRuntime    (queue-less subclass of WikiAgentRuntime)
      └─ AgentLoop                (the v4 harness, reused verbatim)

Tool dispatch routing (in ``_dispatch``):

    list_repos / list_pulls / list_issues / list_commits / get_pull_reviews
    / get_repo / wiki_raw_save / record_timeline   →  self._dispatch_source
    list_wiki_pages / read_page / update_page / create_page / done
                                                  →  runtime.dispatch_tool

Bookkeeping tools that the daily-replay agent uses (next_events,
get_event_body, skip_events) are intentionally NOT in the bootstrap
palette: there is no triaged-events queue to drain on bootstrap. The
agent walks the live source via list_pulls / list_issues / list_commits
and decides per-item whether to write a page or move on.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any, ClassVar, get_args

from pydantic import BaseModel, Field, field_validator

from services.synthesis.agent_harness import AgentLoop, new_agent_run_id
from services.synthesis.agent_tools import (
    CREATE_PAGE_TOOL,
    DONE_TOOL,
    LIST_WIKI_PAGES_TOOL,
    READ_PAGE_TOOL,
    UPDATE_PAGE_TOOL,
)
from services.synthesis.api_clients.github import (
    GitHubAPIClient,
    GitHubAPIError,
    GitHubRateLimitExhausted,
)
from services.synthesis.crawlers.base import (
    BootstrapAgent,
    BootstrapAgentResult,
    empty_result,
)
from services.synthesis.gemini_agent_client import GeminiAgentClient
from services.synthesis.models import WikiType
from services.synthesis.prompts import build_github_crawler_system_prompt
from services.synthesis.wiki_agent import WikiAgentRuntime
from shared.constants import (
    WIKI_BOOTSTRAP_GITHUB_PRS_DAYS,
    WIKI_BOOTSTRAP_MODEL_GITHUB,
    WIKI_BOOTSTRAP_QUIET_STREAK,
)
from shared.db import with_tenant
from shared.exceptions import AgentHaltError, ToolValidationError
from shared.logging import get_logger

log = get_logger(__name__)

_PAGE_SIZE = 50
_REVIEW_PAGE_SIZE = 100
_SOURCE_KIND = "github"

# Wiki type literal — pulled from the Pydantic literal so adding a new
# wiki_type in models.py automatically widens the validators.
_WIKI_TYPES = list(get_args(WikiType))


# ---------------------------------------------------------------------------
# Source-tool input validators (Pydantic)
# ---------------------------------------------------------------------------


class _ListReposArgs(BaseModel):
    cursor: str | None = None


class _ListPullsArgs(BaseModel):
    full_name: str = Field(min_length=1, max_length=200)
    cursor: str | None = None


class _ListIssuesArgs(BaseModel):
    full_name: str = Field(min_length=1, max_length=200)
    cursor: str | None = None


class _ListCommitsArgs(BaseModel):
    full_name: str = Field(min_length=1, max_length=200)
    cursor: str | None = None


class _GetPullReviewsArgs(BaseModel):
    full_name: str = Field(min_length=1, max_length=200)
    pull_number: int = Field(ge=1)


class _GetRepoArgs(BaseModel):
    full_name: str = Field(min_length=1, max_length=200)


class _WikiRawSaveArgs(BaseModel):
    source_ref: str = Field(min_length=1, max_length=240)
    wiki_type: WikiType
    slug: str = Field(min_length=1, max_length=64)
    payload: dict[str, Any]


class _RecordTimelineArgs(BaseModel):
    wiki_type: WikiType
    slug: str = Field(min_length=1, max_length=64)
    entry_date: str = Field(min_length=10, max_length=10)  # YYYY-MM-DD
    source_ref: str | None = None
    summary: str = Field(min_length=1, max_length=240)
    detail: str = ""

    @field_validator("entry_date")
    @classmethod
    def _validate_entry_date(cls, value: str) -> str:
        # Reject malformed dates here so the failure surfaces as a typed
        # tool-validation error rather than a Postgres ::date cast error
        # downstream (which would surface as tool_exception).
        try:
            date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"entry_date must be YYYY-MM-DD, got {value!r}") from exc
        return value


_SOURCE_TOOL_VALIDATORS: dict[str, type[BaseModel]] = {
    "list_repos": _ListReposArgs,
    "list_pulls": _ListPullsArgs,
    "list_issues": _ListIssuesArgs,
    "list_commits": _ListCommitsArgs,
    "get_pull_reviews": _GetPullReviewsArgs,
    "get_repo": _GetRepoArgs,
    "wiki_raw_save": _WikiRawSaveArgs,
    "record_timeline": _RecordTimelineArgs,
}


# ---------------------------------------------------------------------------
# Source-tool schema (Gemini function-call shape)
# ---------------------------------------------------------------------------


def _list_repos_tool() -> dict[str, Any]:
    return {
        "name": "list_repos",
        "description": (
            "List every repo accessible to the GitHub App installation, "
            "newest pushed first. Use cursor= from the previous result "
            "to keep walking; absent cursor means start at the top."
        ),
        "parameters": {
            "type": "object",
            "properties": {"cursor": {"type": "string"}},
        },
    }


def _list_pulls_tool() -> dict[str, Any]:
    return {
        "name": "list_pulls",
        "description": (
            "List pulls (PRs) for a repo, newest-updated first, bounded "
            "to the last 12 months per the locked bootstrap plan. Returns "
            "up to 50 PRs per call. Pass cursor from the previous result "
            "to continue; absent cursor restarts at the newest PR."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "full_name": {"type": "string"},
                "cursor": {"type": "string"},
            },
            "required": ["full_name"],
        },
    }


def _list_issues_tool() -> dict[str, Any]:
    return {
        "name": "list_issues",
        "description": (
            "List issues for a repo (PR-shaped issues already filtered "
            "out), newest-updated first, bounded to the last 12 months. "
            "Up to 50 issues per call; pass cursor to continue."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "full_name": {"type": "string"},
                "cursor": {"type": "string"},
            },
            "required": ["full_name"],
        },
    }


def _list_commits_tool() -> dict[str, Any]:
    return {
        "name": "list_commits",
        "description": (
            "List commits on the default branch, newest first, all-time. "
            "Up to 50 commits per call; pass cursor to continue. Stop "
            "calling when consecutive commits stop changing the wiki."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "full_name": {"type": "string"},
                "cursor": {"type": "string"},
            },
            "required": ["full_name"],
        },
    }


def _get_pull_reviews_tool() -> dict[str, Any]:
    return {
        "name": "get_pull_reviews",
        "description": (
            "Fetch the full review thread for one PR. Returns up to 100 "
            "reviews — PRs effectively don't get more than that, so this "
            "tool doesn't paginate."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "full_name": {"type": "string"},
                "pull_number": {"type": "integer", "minimum": 1},
            },
            "required": ["full_name", "pull_number"],
        },
    }


def _get_repo_tool() -> dict[str, Any]:
    return {
        "name": "get_repo",
        "description": "Fetch a single repo's full metadata (description, topics, default_branch).",
        "parameters": {
            "type": "object",
            "properties": {"full_name": {"type": "string"}},
            "required": ["full_name"],
        },
    }


def _wiki_raw_save_tool() -> dict[str, Any]:
    return {
        "name": "wiki_raw_save",
        "description": (
            "Persist the raw GitHub API payload that drove a wiki page "
            "into wiki_raw_data, so future readers can trace 'why does "
            "this page say X' back to the exact PR/issue/commit. Call "
            "this BEFORE update_page / create_page so the source is "
            "linked. UNIQUE constraint dedups re-saves automatically."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "source_ref": {"type": "string"},
                "wiki_type": {"type": "string", "enum": _WIKI_TYPES},
                "slug": {"type": "string"},
                "payload": {"type": "object"},
            },
            "required": ["source_ref", "wiki_type", "slug", "payload"],
        },
    }


def _record_timeline_tool() -> dict[str, Any]:
    return {
        "name": "record_timeline",
        "description": (
            "Append a chronological audit entry to wiki_timeline_entries "
            "for a wiki page. Call when a source event contributed to a "
            "page; UNIQUE constraint dedups duplicates. entry_date is "
            "YYYY-MM-DD."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "wiki_type": {"type": "string", "enum": _WIKI_TYPES},
                "slug": {"type": "string"},
                "entry_date": {"type": "string"},
                "source_ref": {"type": "string"},
                "summary": {"type": "string"},
                "detail": {"type": "string"},
            },
            "required": ["wiki_type", "slug", "entry_date", "summary"],
        },
    }


def _all_source_tools() -> list[dict[str, Any]]:
    return [
        _list_repos_tool(),
        _list_pulls_tool(),
        _list_issues_tool(),
        _list_commits_tool(),
        _get_pull_reviews_tool(),
        _get_repo_tool(),
        _wiki_raw_save_tool(),
        _record_timeline_tool(),
    ]


def _bootstrap_wiki_tools() -> list[dict[str, Any]]:
    """Subset of agent_tools.ALL_TOOLS that bootstrap exposes.

    Bootstrap drops next_events / get_event_body / skip_events because
    there's no triaged queue to drain. The agent walks the source live.
    """
    return [
        LIST_WIKI_PAGES_TOOL,
        READ_PAGE_TOOL,
        UPDATE_PAGE_TOOL,
        CREATE_PAGE_TOOL,
        DONE_TOOL,
    ]


# ---------------------------------------------------------------------------
# Cursor encoding — opaque base64-JSON
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Cursor:
    """Opaque pagination cursor handed back to the agent.

    Captures (full_name, offset_index). The agent passes the same string
    back via ``cursor=`` to continue iterating. Encoding is base64-JSON
    so it round-trips through the LLM without escaping surprises.
    """

    full_name: str
    offset: int

    def encode(self) -> str:
        payload = {"full_name": self.full_name, "offset": self.offset}
        return base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")

    @classmethod
    def decode(cls, raw: str | None) -> _Cursor | None:
        if not raw:
            return None
        try:
            data = json.loads(base64.urlsafe_b64decode(raw.encode("ascii")).decode("utf-8"))
        except (ValueError, json.JSONDecodeError):
            raise ToolValidationError(f"invalid cursor: {raw!r}") from None
        if not isinstance(data, dict) or "full_name" not in data:
            raise ToolValidationError(f"invalid cursor payload: {data!r}")
        return cls(
            full_name=str(data["full_name"]),
            offset=int(data.get("offset") or 0),
        )


# ---------------------------------------------------------------------------
# Bootstrap-mode wiki runtime (option (b) per the design discussion)
# ---------------------------------------------------------------------------


class BootstrapWikiRuntime(WikiAgentRuntime):
    """Queue-less subclass of WikiAgentRuntime for bootstrap crawlers.

    Reuses the page-write machinery (``_persist_update`` / ``_persist_create``
    / link extraction / index regeneration / snapshot-restore) verbatim.
    Strips the queue lifecycle: bootstrap never reads from
    wiki_synthesis_queue, so next_events / get_event_body / skip_events
    are unsupported and ``commit()`` skips the queue mark-done /
    mark-skipped steps. ``initial_manifest`` returns an empty payload so
    the AgentLoop's cache build still works.

    The agent still uses ``state_snapshot_for_summary``, ``wiki_index``,
    and ``dispatch_tool`` — but only for the bootstrap-permitted tools.
    Calls to next_events / get_event_body / skip_events from a misbehaving
    LLM produce a typed ``ToolValidationError`` so the harness routes the
    error back as a tool result instead of crashing the loop.
    """

    _BLOCKED_TOOLS = frozenset({"next_events", "get_event_body", "skip_events"})

    async def _dispatch_validated(self, name: str, validated: Any) -> dict[str, Any]:
        if name in self._BLOCKED_TOOLS:
            raise ToolValidationError(f"tool {name!r} is not available in bootstrap mode")
        return await super()._dispatch_validated(name, validated)

    async def initial_manifest(self, count: int) -> dict[str, Any]:
        # No queue rows in bootstrap. Return a minimal, well-shaped payload
        # so the AgentLoop's cache builder still works.
        return {"events": [], "remaining": 0, "drain_complete": True}

    @property
    def pages_created_count(self) -> int:
        return len(self._pending_creates)

    @property
    def pages_updated_count(self) -> int:
        return len(self._pending_updates)

    async def commit(self) -> None:
        """Atomic-ish commit of staged updates + creates.

        Same per-page persist + link-extraction path as the parent runtime
        (so wiki_links rows show up exactly the same way), but skips the
        queue mark-done / mark-skipped / regenerate-index steps. Bootstrap
        regenerates the index at the end of run() instead — the per-source
        crawler doesn't own the customer-wide index lifecycle.
        """
        for update in self._pending_updates.values():
            await self._persist_update(update)
        for create in self._pending_creates.values():
            await self._persist_create(create)


# ---------------------------------------------------------------------------
# wiki_raw_data + wiki_timeline_entries persistence
# ---------------------------------------------------------------------------


async def _persist_wiki_raw(
    *,
    customer_id: str,
    source: str,
    source_ref: str,
    wiki_type: str,
    slug: str,
    payload: dict[str, Any],
) -> None:
    """Insert (or no-op on dedup) a wiki_raw_data row.

    UNIQUE constraint on (customer_id, wiki_type, slug, source, source_ref)
    absorbs re-saves so the agent can call wiki_raw_save freely.
    """
    async with with_tenant(customer_id) as conn:
        await conn.execute(
            """
            INSERT INTO wiki_raw_data
                (customer_id, wiki_type, slug, source, source_ref, data)
            VALUES ($1, $2, $3, $4, $5, $6::jsonb)
            ON CONFLICT (customer_id, wiki_type, slug, source, source_ref)
            DO NOTHING
            """,
            customer_id,
            wiki_type,
            slug,
            source,
            source_ref,
            json.dumps(payload),
        )


async def _persist_timeline_entry(
    *,
    customer_id: str,
    source: str,
    wiki_type: str,
    slug: str,
    entry_date: str,
    source_ref: str | None,
    summary: str,
    detail: str,
) -> None:
    """Insert (or no-op on dedup) a wiki_timeline_entries row.

    UNIQUE constraint on (cust, type, slug, entry_date, summary) absorbs
    duplicate appends from re-runs.
    """
    async with with_tenant(customer_id) as conn:
        await conn.execute(
            """
            INSERT INTO wiki_timeline_entries
                (customer_id, wiki_type, slug, entry_date,
                 source, source_ref, summary, detail)
            VALUES ($1, $2, $3, $4::date, $5, $6, $7, $8)
            ON CONFLICT ON CONSTRAINT uq_wiki_timeline_dedup DO NOTHING
            """,
            customer_id,
            wiki_type,
            slug,
            entry_date,
            source,
            source_ref,
            summary,
            detail,
        )


# ---------------------------------------------------------------------------
# GitHubCrawlerAgent
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _Counters:
    items_processed: int = 0
    rate_limited: bool = False
    quiet_streak: int = 0
    consequential_streak_break: bool = False
    skipped_pr_numbers: list[int] = field(default_factory=list)
    skipped_issue_numbers: list[int] = field(default_factory=list)
    skipped_commit_shas: list[str] = field(default_factory=list)


class GitHubCrawlerAgent(BootstrapAgent):
    """Per-customer GitHub bootstrap crawler.

    Walks every accessible installation repo for the customer, reading
    PRs/issues/commits/reviews via ``GitHubAPIClient``, and emits wiki
    pages through the shared write tools (``WikiAgentRuntime``).
    Recency-first: PRs/issues are bounded to the last 12 months per the
    locked plan, commits walk all-time.
    """

    source: ClassVar[str] = "github"
    # Per locked plan #1: GitHub PRs/issues 12mo, commits all-time.
    # The single attr can't represent two windows; the run() implementation
    # applies the per-resource bounding (see _since_for_prs vs commits).
    time_horizon_days: ClassVar[int | None] = WIKI_BOOTSTRAP_GITHUB_PRS_DAYS

    def __init__(
        self,
        *,
        customer_id: str,
        run_id: int,
        bearer_resolver: Any,
        http: Any,
        settings: Any,
        llm_client: Any | None = None,
        runtime: BootstrapWikiRuntime | None = None,
        client: GitHubAPIClient | None = None,
    ) -> None:
        super().__init__(
            customer_id=customer_id,
            run_id=run_id,
            bearer_resolver=bearer_resolver,
            http=http,
            settings=settings,
        )
        # Optional injection points used by the eval harness + unit tests:
        # production code passes None and the run() path constructs the
        # production-flavored objects from the resolved bearer.
        self._llm_client = llm_client
        self._runtime: BootstrapWikiRuntime | None = runtime
        self._client: GitHubAPIClient | None = client
        self._counters = _Counters()
        self._cached_repos: list[dict[str, Any]] | None = None
        # Per-repo cached resource lists. The synchronous list-then-yield
        # shape is OK because each repo's per_page=100 fits in one collection
        # for the bootstrap volumes we see in practice.
        self._pulls_cache: dict[str, list[dict[str, Any]]] = {}
        self._issues_cache: dict[str, list[dict[str, Any]]] = {}
        self._commits_cache: dict[str, list[dict[str, Any]]] = {}

    # -----------------------------------------------------------------------
    # BootstrapAgent surface
    # -----------------------------------------------------------------------

    def system_prompt(self) -> str:
        return build_github_crawler_system_prompt(
            customer_id=self.customer_id,
            quiet_streak=WIKI_BOOTSTRAP_QUIET_STREAK,
        )

    def source_api_tools(self) -> list[dict[str, Any]]:
        return _all_source_tools()

    async def dispatch_source_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        validator = _SOURCE_TOOL_VALIDATORS.get(name)
        if validator is None:
            raise ToolValidationError(f"unknown source tool: {name}")
        try:
            validated = validator.model_validate(args)
        except Exception as exc:
            raise ToolValidationError(f"invalid args for {name}: {exc}") from exc
        return await self._dispatch_source_validated(name, validated)

    async def run(self) -> BootstrapAgentResult:
        started_at = datetime.now(UTC)

        # Resolve bearer. None -> auth not connected; return a clean result.
        try:
            bearer = await self._bearer_resolver()
        except Exception as exc:
            log.warning(
                "github_crawler.bearer_resolve_failed",
                customer=self.customer_id,
                run_id=self.run_id,
                error=str(exc),
            )
            return self._empty_result(started_at, error=f"bearer resolve failed: {exc}")
        if not bearer:
            return self._empty_result(started_at, halt_reason="auth.missing")

        # Build the API client + runtime if the test harness didn't inject
        # them. Production always falls through this branch.
        if self._client is None:
            self._client = GitHubAPIClient(bearer=bearer, http=self._http)
        if self._runtime is None:
            self._runtime = BootstrapWikiRuntime(
                self.customer_id,
                agent_run_id=new_agent_run_id(),
                run_id=self.run_id,
                run_kind="bootstrap",
            )
        if self._llm_client is None:
            self._llm_client = GeminiAgentClient(model=WIKI_BOOTSTRAP_MODEL_GITHUB)

        # Wire the AgentLoop. Tools = source-API tools + write subset of the
        # daily-replay tool palette. Dispatch is routed in _dispatch.
        loop = AgentLoop(
            runtime=self._runtime,
            llm=self._llm_client,
            system_prompt=self.system_prompt(),
            tool_schemas=self.source_api_tools() + _bootstrap_wiki_tools(),
            model=WIKI_BOOTSTRAP_MODEL_GITHUB,
        )
        # The AgentLoop dispatches via runtime.dispatch_tool. Wrap the
        # runtime so source tools route through self.dispatch_source_tool
        # while still passing wiki tools through to the runtime.
        original_dispatch = self._runtime.dispatch_tool
        self._runtime.dispatch_tool = self._build_router(original_dispatch)  # type: ignore[method-assign]

        halt_reason: str | None = None
        try:
            metrics = await loop.run()
        except AgentHaltError as halt:
            halt_reason = halt.reason
            metrics = loop.metrics
            log.warning(
                "github_crawler.halt",
                customer=self.customer_id,
                run_id=self.run_id,
                reason=halt_reason,
            )
        except GitHubRateLimitExhausted:
            halt_reason = "rate_limited"
            metrics = loop.metrics
            log.warning(
                "github_crawler.rate_limited",
                customer=self.customer_id,
                run_id=self.run_id,
            )
        finally:
            self._runtime.dispatch_tool = original_dispatch  # type: ignore[method-assign]

        # If the agent never reached done(), still flush any staged writes.
        # commit() is idempotent here (a clean done() already called it).
        if not self._runtime.is_done:
            try:
                await self._runtime.commit()
            except Exception as exc:
                log.warning(
                    "github_crawler.commit_failed",
                    customer=self.customer_id,
                    run_id=self.run_id,
                    error=str(exc),
                )
                halt_reason = halt_reason or "commit_failed"

        # Regenerate the singleton wiki index now that this source's pages
        # are persisted. Cross-source race is acceptable: multiple per-source
        # crawlers each regen the index page in parallel; last writer wins,
        # and since the index body is just a TOC of all live pages, all
        # writers converge on the same content.
        try:
            await self._runtime._regenerate_index()
        except Exception as exc:
            log.warning(
                "github_crawler.index_regen_failed",
                customer=self.customer_id,
                run_id=self.run_id,
                error=str(exc),
            )

        if self._counters.rate_limited and halt_reason is None:
            halt_reason = "rate_limited"

        finished_at = datetime.now(UTC)
        return BootstrapAgentResult(
            source=self.source,
            customer_id=self.customer_id,
            run_id=self.run_id,
            pages_created=self._runtime.pages_created_count,
            pages_updated=self._runtime.pages_updated_count,
            items_processed=self._counters.items_processed,
            halt_reason=halt_reason,
            started_at=started_at,
            finished_at=finished_at,
            turns=metrics.turns,
            compaction_count=metrics.compaction_count,
            cache_hit_rate=metrics.cache_hit_rate,
            total_input_tokens=metrics.total_input_tokens,
            total_cached_tokens=metrics.total_cached_tokens,
            total_output_tokens=metrics.total_output_tokens,
        )

    # -----------------------------------------------------------------------
    # Test / eval-harness surface — counters + skip lists
    # -----------------------------------------------------------------------

    @property
    def counters(self) -> _Counters:
        return self._counters

    @property
    def runtime(self) -> BootstrapWikiRuntime | None:
        return self._runtime

    # -----------------------------------------------------------------------
    # Routing
    # -----------------------------------------------------------------------

    def _build_router(self, runtime_dispatch: Any) -> Any:
        source_names = set(_SOURCE_TOOL_VALIDATORS.keys())

        async def router(name: str, args: dict[str, Any]) -> dict[str, Any]:
            if name in source_names:
                return await self.dispatch_source_tool(name, args)
            return await runtime_dispatch(name, args)

        return router

    # -----------------------------------------------------------------------
    # Source-tool dispatch
    # -----------------------------------------------------------------------

    async def _dispatch_source_validated(self, name: str, validated: BaseModel) -> dict[str, Any]:
        if name == "list_repos":
            return await self._tool_list_repos()
        if name == "list_pulls":
            return await self._tool_list_pulls(validated)  # type: ignore[arg-type]
        if name == "list_issues":
            return await self._tool_list_issues(validated)  # type: ignore[arg-type]
        if name == "list_commits":
            return await self._tool_list_commits(validated)  # type: ignore[arg-type]
        if name == "get_pull_reviews":
            return await self._tool_get_pull_reviews(validated)  # type: ignore[arg-type]
        if name == "get_repo":
            return await self._tool_get_repo(validated)  # type: ignore[arg-type]
        if name == "wiki_raw_save":
            return await self._tool_wiki_raw_save(validated)  # type: ignore[arg-type]
        if name == "record_timeline":
            return await self._tool_record_timeline(validated)  # type: ignore[arg-type]
        raise ToolValidationError(f"unknown source tool: {name}")

    # -----------------------------------------------------------------------
    # Source-tool handlers
    # -----------------------------------------------------------------------

    async def _tool_list_repos(self) -> dict[str, Any]:
        client = self._require_client()
        if self._cached_repos is None:
            try:
                repos: list[dict[str, Any]] = []
                async for repo in client.list_installation_repos():
                    repos.append(repo)
                self._cached_repos = repos
            except GitHubRateLimitExhausted:
                self._counters.rate_limited = True
                return {"error": "rate_limited", "repos": []}
            except GitHubAPIError as exc:
                return {"error": f"github.{exc.status}", "detail": str(exc)[:200], "repos": []}
        out = [
            {
                "name": r.get("name"),
                "full_name": r.get("full_name"),
                "description": r.get("description"),
                "pushed_at": r.get("pushed_at"),
                "default_branch": r.get("default_branch"),
            }
            for r in self._cached_repos
        ]
        self._counters.items_processed += len(out)
        return {"repos": out, "next_cursor": None}

    async def _tool_list_pulls(self, args: _ListPullsArgs) -> dict[str, Any]:
        return await self._paged_list(
            cache=self._pulls_cache,
            full_name=args.full_name,
            cursor=args.cursor,
            fetch=self._fetch_pulls,
            shape="pulls",
        )

    async def _tool_list_issues(self, args: _ListIssuesArgs) -> dict[str, Any]:
        return await self._paged_list(
            cache=self._issues_cache,
            full_name=args.full_name,
            cursor=args.cursor,
            fetch=self._fetch_issues,
            shape="issues",
        )

    async def _tool_list_commits(self, args: _ListCommitsArgs) -> dict[str, Any]:
        return await self._paged_list(
            cache=self._commits_cache,
            full_name=args.full_name,
            cursor=args.cursor,
            fetch=self._fetch_commits,
            shape="commits",
        )

    async def _tool_get_pull_reviews(self, args: _GetPullReviewsArgs) -> dict[str, Any]:
        client = self._require_client()
        try:
            reviews: list[dict[str, Any]] = []
            async for r in client.list_pull_reviews(args.full_name, args.pull_number):
                reviews.append(self._shrink_review(r))
                if len(reviews) >= _REVIEW_PAGE_SIZE:
                    break
        except GitHubRateLimitExhausted:
            self._counters.rate_limited = True
            return {"error": "rate_limited", "reviews": []}
        except GitHubAPIError as exc:
            return {"error": f"github.{exc.status}", "detail": str(exc)[:200], "reviews": []}
        self._counters.items_processed += len(reviews)
        return {"full_name": args.full_name, "pull_number": args.pull_number, "reviews": reviews}

    async def _tool_get_repo(self, args: _GetRepoArgs) -> dict[str, Any]:
        client = self._require_client()
        try:
            repo = await client.get_repo(args.full_name)
        except GitHubRateLimitExhausted:
            self._counters.rate_limited = True
            return {"error": "rate_limited"}
        except GitHubAPIError as exc:
            return {"error": f"github.{exc.status}", "detail": str(exc)[:200]}
        return {"repo": self._shrink_repo(repo)}

    async def _tool_wiki_raw_save(self, args: _WikiRawSaveArgs) -> dict[str, Any]:
        await _persist_wiki_raw(
            customer_id=self.customer_id,
            source=_SOURCE_KIND,
            source_ref=args.source_ref,
            wiki_type=args.wiki_type,
            slug=args.slug,
            payload=args.payload,
        )
        return {"status": "saved", "source_ref": args.source_ref}

    async def _tool_record_timeline(self, args: _RecordTimelineArgs) -> dict[str, Any]:
        await _persist_timeline_entry(
            customer_id=self.customer_id,
            source=_SOURCE_KIND,
            wiki_type=args.wiki_type,
            slug=args.slug,
            entry_date=args.entry_date,
            source_ref=args.source_ref,
            summary=args.summary,
            detail=args.detail,
        )
        return {"status": "appended"}

    # -----------------------------------------------------------------------
    # Pagination scaffolding
    # -----------------------------------------------------------------------

    async def _paged_list(
        self,
        *,
        cache: dict[str, list[dict[str, Any]]],
        full_name: str,
        cursor: str | None,
        fetch: Any,
        shape: str,
    ) -> dict[str, Any]:
        try:
            cur = _Cursor.decode(cursor)
        except ToolValidationError as exc:
            return {"error": "invalid_cursor", "detail": str(exc), shape: []}
        offset = cur.offset if cur is not None else 0

        if full_name not in cache:
            try:
                cache[full_name] = await fetch(full_name)
            except GitHubRateLimitExhausted:
                self._counters.rate_limited = True
                return {"error": "rate_limited", shape: []}
            except GitHubAPIError as exc:
                return {"error": f"github.{exc.status}", "detail": str(exc)[:200], shape: []}

        rows = cache[full_name]
        slice_end = offset + _PAGE_SIZE
        chunk = rows[offset:slice_end]
        next_cursor: str | None = None
        if slice_end < len(rows):
            next_cursor = _Cursor(full_name=full_name, offset=slice_end).encode()
        self._counters.items_processed += len(chunk)
        return {
            "full_name": full_name,
            shape: chunk,
            "next_cursor": next_cursor,
            "remaining": max(0, len(rows) - slice_end),
        }

    async def _fetch_pulls(self, full_name: str) -> list[dict[str, Any]]:
        client = self._require_client()
        since = self._since_for_prs()
        out: list[dict[str, Any]] = []
        async for pr in client.list_pulls(full_name, since=since):
            out.append(self._shrink_pull(pr))
        return out

    async def _fetch_issues(self, full_name: str) -> list[dict[str, Any]]:
        client = self._require_client()
        since = self._since_for_prs()
        out: list[dict[str, Any]] = []
        async for issue in client.list_issues(full_name, since=since):
            out.append(self._shrink_issue(issue))
        return out

    async def _fetch_commits(self, full_name: str) -> list[dict[str, Any]]:
        # All-time per the locked plan — no since filter.
        client = self._require_client()
        out: list[dict[str, Any]] = []
        async for commit in client.list_commits(full_name):
            out.append(self._shrink_commit(commit))
        return out

    def _since_for_prs(self) -> datetime:
        """Cutoff for PR + issue listing. Commits use no cutoff."""
        return datetime.now(UTC) - timedelta(days=WIKI_BOOTSTRAP_GITHUB_PRS_DAYS)

    # -----------------------------------------------------------------------
    # Shrinkers — drop fields the agent doesn't need so the tool result
    # stays compact in the LLM context.
    # -----------------------------------------------------------------------

    @staticmethod
    def _shrink_pull(pr: dict[str, Any]) -> dict[str, Any]:
        user = pr.get("user") or {}
        return {
            "number": pr.get("number"),
            "title": pr.get("title"),
            "body": pr.get("body"),
            "state": pr.get("state"),
            "created_at": pr.get("created_at"),
            "updated_at": pr.get("updated_at"),
            "merged_at": pr.get("merged_at"),
            "closed_at": pr.get("closed_at"),
            "user_login": user.get("login") if isinstance(user, dict) else None,
            "html_url": pr.get("html_url"),
            "requested_reviewers": [
                rr.get("login")
                for rr in (pr.get("requested_reviewers") or [])
                if isinstance(rr, dict)
            ],
            "labels": [lb.get("name") for lb in (pr.get("labels") or []) if isinstance(lb, dict)],
        }

    @staticmethod
    def _shrink_issue(issue: dict[str, Any]) -> dict[str, Any]:
        user = issue.get("user") or {}
        return {
            "number": issue.get("number"),
            "title": issue.get("title"),
            "body": issue.get("body"),
            "state": issue.get("state"),
            "created_at": issue.get("created_at"),
            "updated_at": issue.get("updated_at"),
            "closed_at": issue.get("closed_at"),
            "state_reason": issue.get("state_reason"),
            "user_login": user.get("login") if isinstance(user, dict) else None,
            "html_url": issue.get("html_url"),
            "labels": [
                lb.get("name") for lb in (issue.get("labels") or []) if isinstance(lb, dict)
            ],
        }

    @staticmethod
    def _shrink_commit(commit: dict[str, Any]) -> dict[str, Any]:
        commit_meta = commit.get("commit") or {}
        author_meta = commit_meta.get("author") if isinstance(commit_meta, dict) else None
        author_outer = commit.get("author") or {}
        return {
            "sha": commit.get("sha"),
            "message": commit_meta.get("message") if isinstance(commit_meta, dict) else None,
            "author_login": (author_outer or {}).get("login")
            if isinstance(author_outer, dict)
            else None,
            "author_name": (author_meta or {}).get("name")
            if isinstance(author_meta, dict)
            else None,
            "author_date": (author_meta or {}).get("date")
            if isinstance(author_meta, dict)
            else None,
            "html_url": commit.get("html_url"),
        }

    @staticmethod
    def _shrink_review(review: dict[str, Any]) -> dict[str, Any]:
        user = review.get("user") or {}
        return {
            "id": review.get("id"),
            "state": review.get("state"),
            "body": review.get("body"),
            "submitted_at": review.get("submitted_at"),
            "user_login": user.get("login") if isinstance(user, dict) else None,
            "html_url": review.get("html_url"),
        }

    @staticmethod
    def _shrink_repo(repo: dict[str, Any]) -> dict[str, Any]:
        owner = repo.get("owner") or {}
        return {
            "name": repo.get("name"),
            "full_name": repo.get("full_name"),
            "description": repo.get("description"),
            "default_branch": repo.get("default_branch"),
            "topics": repo.get("topics") or [],
            "owner_login": owner.get("login") if isinstance(owner, dict) else None,
            "pushed_at": repo.get("pushed_at"),
            "language": repo.get("language"),
        }

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _require_client(self) -> GitHubAPIClient:
        if self._client is None:
            raise RuntimeError(
                "GitHubCrawlerAgent._client is None — run() must resolve the bearer first"
            )
        return self._client

    def _empty_result(
        self,
        started_at: datetime,
        *,
        halt_reason: str | None = None,
        error: str | None = None,
    ) -> BootstrapAgentResult:
        result = empty_result(
            source=self.source,
            customer_id=self.customer_id,
            run_id=self.run_id,
            halt_reason=halt_reason,
            error=error,
        )
        # ``empty_result`` stamps started_at = finished_at = now(); reuse
        # the started_at the caller measured so the row's window matches
        # the orchestrator's bootstrap.start log.
        return result.model_copy(
            update={"started_at": started_at, "finished_at": datetime.now(UTC)}
        )


__all__ = [
    "BootstrapWikiRuntime",
    "GitHubCrawlerAgent",
]
