"""``BackfillAgent`` ABC — per-source crawler agent skeleton.

Each concrete crawler (Slack, GitHub, Linear, ...) subclasses this. The
base class owns the lifecycle the orchestrator expects:

  ``__init__(customer_id, run_id, bearer_resolver, http, settings)``
  -> ``await self.run()``
  -> returns a ``BackfillAgentResult``

Subclasses provide:

  - ``source``  — class-level string identifier ("github", "slack", ...).
  - ``time_horizon_days`` — class-level cutoff or ``None`` for all-time
    (locked decisions table #1; e.g. Slack=365, Linear=None, Granola=730).
  - ``system_prompt()`` — source-specialized agent prompt.
  - ``source_api_tools()`` — Gemini tool schemas for source-specific
    APIs (``list_pulls``, ``list_messages``, ...).
  - ``dispatch_source_tool(name, args)`` — handler for those tools.
    The base class handles the shared wiki write tools by delegating to
    a ``WikiAgentRuntime`` instance the subclass receives via ``self.wiki``.

The harness (``AgentLoop``) is reused verbatim. Compaction, halt, snapshot
semantics, cap enforcement — all the v4 daily-replay machinery applies.

The orchestrator instantiates the subclass, calls ``run()`` once, and
records the returned ``BackfillAgentResult`` against the
``wiki_synthesis_runs`` row it opened for this crawler.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any, ClassVar, Protocol

import httpx
from pydantic import BaseModel, Field

from shared.config import Settings


class BearerResolver(Protocol):
    """Async callable returning a fresh bearer for the source API.

    Implementations live in ``shared.backend_client`` for tokens that
    flow through the BFF (GitHub installation tokens, Slack OAuth, ...).
    Returning ``None`` means the customer hasn't connected this source —
    the orchestrator skips the crawler with a clean ``halt_reason``.
    """

    async def __call__(self) -> str | None: ...


# Type alias — useful when crawler factories want to accept a callable
# directly without buying into the Protocol.
BearerResolverCallable = Callable[[], Awaitable[str | None]]


class BackfillAgentResult(BaseModel):
    """One crawler's outcome. Written into ``wiki_synthesis_runs`` by
    the BackfillWorker after the crawler returns or crashes.

    ``error`` is set if the crawler raised before completing; the
    worker catches the exception, fills this field via ``empty_result``,
    and the run row's status is set to 'failed' by ``_close_run``.
    """

    source: str
    customer_id: str
    run_id: int
    # Phase 2 fan-out target. None for Phase 1 rows; 'owner/repo' (or
    # equivalent per-source identifier) for Phase 2. Round-tripped through
    # the worker into ``wiki_synthesis_runs.target`` on close.
    target: str | None = None
    pages_created: int = 0
    pages_updated: int = 0
    items_processed: int = 0
    halt_reason: str | None = None
    started_at: datetime
    finished_at: datetime
    turns: int = 0
    compaction_count: int = 0
    cache_hit_rate: float | None = None
    total_input_tokens: int = 0
    total_cached_tokens: int = 0
    total_output_tokens: int = 0
    error: str | None = None


class BackfillAgent(ABC):
    """Per-source crawler agent. Concrete subclasses implement the
    source-specific tool palette + system prompt; the base class owns
    the harness, the wiki write tools, and lifecycle bookkeeping.

    The orchestrator calls ``await self.run()`` exactly once. Each
    instance corresponds to one ``wiki_synthesis_runs`` row.
    """

    # ClassVar discriminators that the registry + orchestrator key off.
    # Concrete subclasses MUST override these.
    source: ClassVar[str] = ""
    time_horizon_days: ClassVar[int | None] = None

    def __init__(
        self,
        *,
        customer_id: str,
        run_id: int,
        bearer_resolver: BearerResolver | BearerResolverCallable,
        http: httpx.AsyncClient,
        settings: Settings,
        target: str | None = None,
    ) -> None:
        if not self.source:
            raise ValueError(f"{type(self).__name__} must set a non-empty `source` ClassVar")
        self.customer_id = customer_id
        self.run_id = run_id
        # Phase 2 fan-out target. None for Phase 1 broad-pass; a specific
        # target ('owner/repo' for GitHub) when the orchestrator fanned
        # out per-target subtasks after Phase 1 completed.
        self.target = target
        self._bearer_resolver = bearer_resolver
        self._http = http
        self._settings = settings

    # ------------------------------------------------------------------
    # Subclass surface — override these.
    # ------------------------------------------------------------------

    @abstractmethod
    def system_prompt(self) -> str:
        """Source-specialized system prompt. Lane D's GitHub crawler
        will return something like:

            "You are a GitHub analyst for {customer}. Read every PR/
            issue/commit newest-first. Extract durable knowledge into
            the wiki using update_page / create_page."
        """

    @abstractmethod
    def source_api_tools(self) -> list[dict[str, Any]]:
        """Gemini-shaped tool schemas for source-specific tools.

        Returns ONLY the source-API tools (``list_pulls``,
        ``get_thread``, ...). The harness combines these with the
        shared wiki write tools at run time.
        """

    @abstractmethod
    async def dispatch_source_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Dispatch a source-specific tool call.

        The base class handles all shared wiki write tools
        (``update_page``, ``create_page``, ``list_wiki_pages``, ...) by
        delegating to its internal ``WikiAgentRuntime``. Subclasses only
        see calls for tools they declared in ``source_api_tools()``.
        """

    # ------------------------------------------------------------------
    # Public entry point.
    # ------------------------------------------------------------------

    @abstractmethod
    async def run(self) -> BackfillAgentResult:
        """Drive the crawler from construction to completion or halt.

        The expected shape (concrete crawlers should follow):

            1. Resolve bearer via ``self._bearer_resolver()``. If None,
               return a ``BackfillAgentResult`` with halt_reason=
               'auth.missing' and zero counters.
            2. Build the ``WikiAgentRuntime`` (shared with v4) and the
               combined tool palette (source tools + wiki tools).
            3. Build an ``AgentLoop`` around that runtime + Gemini
               client + the system prompt.
            4. Call ``await loop.run()`` and translate its
               ``AgentMetrics`` + the runtime's commit counters into a
               ``BackfillAgentResult``.
            5. On exception, return a ``BackfillAgentResult`` with
               ``error`` set; do NOT re-raise (the orchestrator catches
               via ``return_exceptions=True``, but a clean result keeps
               the per-source row write path simple).

        Lane C ships this signature only — Lane D's GitHub crawler is
        the first concrete implementation.
        """

    # ------------------------------------------------------------------
    # Helpers shared with concrete subclasses.
    # ------------------------------------------------------------------

    async def resolve_bearer(self) -> str | None:
        """Convenience wrapper around the resolver. Concrete
        subclasses can override if they need to cache or mutate the
        token before stamping it onto an API client.
        """
        return await self._bearer_resolver()

    @property
    def http(self) -> httpx.AsyncClient:
        return self._http

    @property
    def settings(self) -> Settings:
        return self._settings


def empty_result(
    *,
    source: str,
    customer_id: str,
    run_id: int,
    target: str | None = None,
    halt_reason: str | None = None,
    error: str | None = None,
) -> BackfillAgentResult:
    """Build a ``BackfillAgentResult`` for the early-exit paths.

    Used by:
      - Auth-not-connected paths (halt_reason='auth.missing').
      - Pre-run failures (constructor raised, etc.).
    All counters land at zero; ``started_at == finished_at == now``.
    """
    now = datetime.now(UTC)
    return BackfillAgentResult(
        source=source,
        customer_id=customer_id,
        run_id=run_id,
        target=target,
        started_at=now,
        finished_at=now,
        halt_reason=halt_reason,
        error=error,
    )


__all__ = [
    "BackfillAgent",
    "BackfillAgentResult",
    "BearerResolver",
    "BearerResolverCallable",
    "Field",
    "empty_result",
]
