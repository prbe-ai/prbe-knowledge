"""Wiki bootstrap orchestrator.

Launches per-source ``BootstrapAgent`` crawlers in parallel for one
customer, tracks one ``wiki_synthesis_runs`` row per crawler, isolates
per-source failures, and (by default) wipes the customer's existing
wiki rows first so re-bootstrap starts clean.

Locked plan decisions reified here:

  - **#2 Parallel from day 1**, optimistic concurrency on doc version.
    Implemented via ``asyncio.gather(..., return_exceptions=True)``.
    Optimistic concurrency itself lives inside ``update_page`` — the
    orchestrator just lets ``STALE_VERSION`` propagate through the
    agent which retries.
  - **#9 Re-bootstrap = wipe first**. Default ``wipe_first=True``.
    The ``_wipe_wiki_for_customer`` helper drops:
        wiki_links, wiki_timeline_entries, wiki_raw_data,
        documents WHERE doc_class = 'compiled_wiki'
    in one txn for the customer. Manual-entry pages (human-authored)
    are preserved. wiki_synthesis_queue rows are NOT touched —
    bootstrap absorbs them via the daily-loop coexistence path
    (decision #3).

Crawler factories: tests pass ``crawler_factories=`` to inject mocks.
Production passes ``None`` and the orchestrator looks up the registry.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import httpx
from pydantic import BaseModel, Field

from services.synthesis.crawlers import REGISTRY, BootstrapAgent, BootstrapAgentResult
from services.synthesis.crawlers.base import BearerResolver, empty_result
from shared.config import Settings
from shared.db import raw_conn, with_tenant
from shared.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


class BootstrapResult(BaseModel):
    """Aggregate outcome across every crawler the orchestrator launched."""

    customer_id: str
    started_at: datetime
    finished_at: datetime
    sources_attempted: list[str] = Field(default_factory=list)
    sources_succeeded: list[str] = Field(default_factory=list)
    sources_failed: dict[str, str] = Field(default_factory=dict)
    per_source: list[BootstrapAgentResult] = Field(default_factory=list)
    wiped: bool = False


# Type alias: callable that returns a BootstrapAgent given the keyword
# arguments the orchestrator constructs (customer_id, run_id, http,
# settings, bearer_resolver). Used by tests to inject a mock crawler.
CrawlerFactory = Callable[..., BootstrapAgent]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class BootstrapOrchestrator:
    """Fan out crawlers in parallel, track runs, isolate failures."""

    def __init__(
        self,
        *,
        settings: Settings,
        http: httpx.AsyncClient,
        bearer_resolver_factory: Callable[[str, str], BearerResolver] | None = None,
    ) -> None:
        """``bearer_resolver_factory`` resolves (customer_id, source) ->
        ``BearerResolver`` (an async callable returning the token).
        Tests can pass a stub; production wires it to ``shared.backend_client``.
        Lane C ships ``None`` since no real crawlers are registered yet —
        the GitHub crawler in Lane D will pull this in.
        """
        self._settings = settings
        self._http = http
        self._bearer_resolver_factory = bearer_resolver_factory or _no_bearer_factory

    async def bootstrap(
        self,
        *,
        customer_id: str,
        sources: list[str] | None = None,
        wipe_first: bool = True,
        reason: str = "bootstrap",
        crawler_factories: dict[str, CrawlerFactory] | None = None,
    ) -> BootstrapResult:
        """Run bootstrap for ``customer_id`` across the requested sources.

        ``sources=None`` means "all registered crawlers". When tests pass
        ``crawler_factories``, the registry is bypassed and the requested
        sources are looked up there instead.
        """
        started_at = datetime.now(UTC)
        factories = self._resolve_factories(sources, crawler_factories)
        attempted = sorted(factories.keys())
        log.info(
            "bootstrap.start",
            customer=customer_id,
            sources=attempted,
            wipe_first=wipe_first,
            reason=reason,
        )

        wiped = False
        if wipe_first:
            await _wipe_wiki_for_customer(customer_id)
            wiped = True
            log.info("bootstrap.wiped", customer=customer_id)

        # Open one wiki_synthesis_runs row per source BEFORE launching the
        # crawlers so a crashed crawler still has a row to mark failed.
        run_ids: dict[str, int] = {}
        for source in attempted:
            run_ids[source] = await _open_bootstrap_run(customer_id=customer_id, source=source)

        # Build + launch crawlers in parallel.
        agents: dict[str, BootstrapAgent] = {}
        construct_errors: dict[str, str] = {}
        for source, factory in factories.items():
            try:
                agents[source] = factory(
                    customer_id=customer_id,
                    run_id=run_ids[source],
                    bearer_resolver=self._bearer_resolver_factory(customer_id, source),
                    http=self._http,
                    settings=self._settings,
                )
            except Exception as exc:
                construct_errors[source] = f"{type(exc).__name__}: {exc}"
                log.warning(
                    "bootstrap.construct_failed",
                    customer=customer_id,
                    source=source,
                    error=str(exc),
                )

        ordered_sources = [s for s in attempted if s in agents]
        gather_results: list[BootstrapAgentResult | BaseException] = []
        if ordered_sources:
            gather_results = await asyncio.gather(
                *(agents[s].run() for s in ordered_sources),
                return_exceptions=True,
            )

        succeeded: list[str] = []
        failed: dict[str, str] = dict(construct_errors)
        per_source: list[BootstrapAgentResult] = []

        # Materialize results from gather, including pre-construction failures.
        for source, ce in construct_errors.items():
            per_source.append(
                empty_result(
                    source=source,
                    customer_id=customer_id,
                    run_id=run_ids[source],
                    error=ce,
                )
            )

        for source, result in zip(ordered_sources, gather_results, strict=True):
            if isinstance(result, BaseException):
                err = f"{type(result).__name__}: {result}"
                failed[source] = err
                per_source.append(
                    empty_result(
                        source=source,
                        customer_id=customer_id,
                        run_id=run_ids[source],
                        error=err,
                    )
                )
                log.warning(
                    "bootstrap.crawler_exception",
                    customer=customer_id,
                    source=source,
                    error=str(result),
                    error_class=type(result).__name__,
                )
            else:
                # Crawler returned a result. It may still describe a
                # halt — that's a successful return, not a failure.
                # Only the `error` field flips it to "failed".
                per_source.append(result)
                if result.error:
                    failed[source] = result.error
                else:
                    succeeded.append(source)

        # Close every run row according to its outcome.
        for result in per_source:
            await _close_bootstrap_run(
                run_id=result.run_id,
                customer_id=customer_id,
                source=result.source,
                status="failed" if result.error else "complete",
                pages_updated=result.pages_updated,
                pages_created=result.pages_created,
                error=result.error,
            )

        finished_at = datetime.now(UTC)
        log.info(
            "bootstrap.done",
            customer=customer_id,
            sources_attempted=attempted,
            sources_succeeded=succeeded,
            sources_failed=list(failed.keys()),
            duration_seconds=(finished_at - started_at).total_seconds(),
        )

        return BootstrapResult(
            customer_id=customer_id,
            started_at=started_at,
            finished_at=finished_at,
            sources_attempted=attempted,
            sources_succeeded=succeeded,
            sources_failed=failed,
            per_source=per_source,
            wiped=wiped,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_factories(
        self,
        sources: list[str] | None,
        override: dict[str, CrawlerFactory] | None,
    ) -> dict[str, CrawlerFactory]:
        """Pick the active factory map for this run.

        Tests pass ``override`` which wins. Production passes ``None`` and
        we read REGISTRY. ``sources`` filters whichever map we settled on;
        unknown source names are dropped with a warning.
        """
        base: dict[str, CrawlerFactory] = dict(override) if override is not None else dict(REGISTRY)
        if sources is None:
            return base
        out: dict[str, CrawlerFactory] = {}
        for source in sources:
            if source in base:
                out[source] = base[source]
            else:
                log.warning(
                    "bootstrap.unknown_source",
                    source=source,
                    available=sorted(base.keys()),
                )
        return out


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


async def _open_bootstrap_run(*, customer_id: str, source: str) -> int:
    """Insert a ``wiki_synthesis_runs`` row at status='running'.

    Migration 0044 added kind='bootstrap' to the CHECK constraint and a
    nullable ``source`` column. Bootstrap rows fill both; daily replay
    rows leave ``source`` NULL (unchanged behaviour).
    """
    async with with_tenant(customer_id) as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO wiki_synthesis_runs (customer_id, kind, stage, source, status)
            VALUES ($1, 'bootstrap', 'synthesis', $2, 'running')
            RETURNING run_id
            """,
            customer_id,
            source,
        )
    return int(row["run_id"])


async def _close_bootstrap_run(
    *,
    run_id: int,
    customer_id: str,
    source: str,
    status: str,
    pages_updated: int,
    pages_created: int,
    error: str | None,
) -> None:
    async with with_tenant(customer_id) as conn:
        await conn.execute(
            """
            UPDATE wiki_synthesis_runs
            SET finished_at = NOW(),
                status = $2,
                pages_updated = $3,
                pages_created = $4,
                error = $5
            WHERE run_id = $1
            """,
            run_id,
            status,
            pages_updated,
            pages_created,
            error,
        )
    log.info(
        "bootstrap.run_closed",
        customer=customer_id,
        source=source,
        run_id=run_id,
        status=status,
        pages_updated=pages_updated,
        pages_created=pages_created,
        error=error,
    )


async def _wipe_wiki_for_customer(customer_id: str) -> None:
    """Drop the customer's compiled-wiki rows so re-bootstrap starts clean.

    Wiped (one txn):
      - ``wiki_links``                 (no RLS — explicit WHERE)
      - ``wiki_timeline_entries``      (no RLS — explicit WHERE)
      - ``wiki_raw_data``              (no RLS — explicit WHERE)
      - ``documents`` rows with doc_class='compiled_wiki'
        (RLS — uses with_tenant inside the same txn so the policy sees
        ``app.current_customer_id``).

    NOT wiped:
      - ``documents`` rows with doc_class='manual_entry' (human-authored).
      - ``documents`` rows with doc_class='agent_artifact' (the auto-
        generated wiki index — bootstrap regenerates this naturally on
        the first commit, but leaving the prior version in place during
        the gap keeps the dashboard from flashing empty).
      - ``wiki_synthesis_queue`` (untouched — daily replay handles it
        via the bootstrap_absorbed marker; locked decision #3).

    Single ``with_tenant`` block so all four DELETEs land in one
    transaction. wiki_links / wiki_timeline_entries / wiki_raw_data
    don't have RLS policies, but the txn boundary still gives atomicity.
    """
    async with with_tenant(customer_id) as conn:
        await conn.execute(
            "DELETE FROM wiki_links WHERE customer_id = $1",
            customer_id,
        )
        await conn.execute(
            "DELETE FROM wiki_timeline_entries WHERE customer_id = $1",
            customer_id,
        )
        await conn.execute(
            "DELETE FROM wiki_raw_data WHERE customer_id = $1",
            customer_id,
        )
        # documents has RLS — the with_tenant GUC powers the policy. The
        # explicit WHERE customer_id is defense-in-depth.
        await conn.execute(
            """
            DELETE FROM documents
            WHERE customer_id = $1
              AND doc_class = 'compiled_wiki'
            """,
            customer_id,
        )


# ---------------------------------------------------------------------------
# Default factories
# ---------------------------------------------------------------------------


async def _resolve_no_bearer() -> str | None:
    """Default bearer resolver — returns ``None``. Used when the
    orchestrator is constructed without a real factory (Lane C: registry
    is empty so this never fires in production)."""
    return None


def _no_bearer_factory(_customer_id: str, _source: str) -> BearerResolver:
    """Bearer factory used when the caller didn't pass one. Real
    crawlers (Lane D) will inject a factory that calls into
    ``shared.backend_client.fetch_<source>_token``.
    """

    async def _resolver() -> str | None:
        return None

    # Annotate the callable so static type-checkers see a BearerResolver.
    return _resolver  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Convenience for the listener / route layer
# ---------------------------------------------------------------------------


def parse_trigger_payload(raw: str | bytes | dict[str, Any]) -> dict[str, Any]:
    """Normalize a NOTIFY payload into a kwargs dict for ``bootstrap``.

    Accepts either a JSON string (the format the route emits) or a
    pre-parsed dict (tests). Missing keys default to None / sane values.
    """
    import orjson

    if isinstance(raw, dict):
        data = raw
    else:
        if isinstance(raw, str):
            raw_bytes = raw.encode("utf-8") if raw else b"{}"
        else:
            raw_bytes = raw or b"{}"
        try:
            data = orjson.loads(raw_bytes)
        except orjson.JSONDecodeError:
            # Legacy callers may NOTIFY with just the customer_id as a
            # bare string; fall back to that shape so the listener stays
            # forward/backward-compatible.
            data = {"customer_id": (raw_bytes.decode("utf-8") or "").strip()}
    return {
        "customer_id": str(data.get("customer_id") or ""),
        "sources": data.get("sources"),
        "wipe_first": bool(data.get("wipe_first", True)),
        "reason": str(data.get("reason") or "bootstrap"),
    }


__all__ = [
    "BootstrapOrchestrator",
    "BootstrapResult",
    "CrawlerFactory",
    "parse_trigger_payload",
]


# Touch raw_conn so our import surface is honest about what's reachable
# without importing the listener module — keeps lint quiet on unused
# imports during incremental refactors.
_ = raw_conn
