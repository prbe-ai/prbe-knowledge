"""Nightly trigger — wakes the wiki-worker + refreshes cross-repo edges.

Runs as a fly machine schedule (the fly.wiki-cron.toml `[processes].cron`
entry runs at 02:00 UTC daily). The script does TWO things in order:

  A. Refresh cross-repo dependency edges. Re-enqueue ``initial_backfill``
     events for every repo that has been code-graph-extracted before.
     The downstream worker re-runs the cross_repo_deps pass for each
     repo, picking up new mentions / removed deps. We then poll the
     queue for drain and call ``regenerate_wiki_diagram`` per customer
     so the architecture diagram in the wiki index reflects the fresh
     edges. Important: this surgically replaces only the
     ```mermaid block``` in the index page — the intro paragraph and
     page list stay byte-identical. The full index is rewritten only
     when the wiki agent runs (i.e. when wiki content actually
     changes). Cost: ~$0.005/repo, content_hash cache makes the
     symbol extraction itself a ~no-op on unchanged files.

  B. Trigger nightly wiki synthesis. SELECT customer_ids with at least
     one pending wiki_synthesis_queue row AND
     ``preferences->>'wiki_generation_enabled' = 'true'``. Per
     customer: pg_notify('wiki_synthesize_pending', customer_id). The
     wiki-worker drains, the wiki-synthesis app writes pages, and
     wiki_agent's commit hook calls ``regenerate_wiki_index`` again at
     the end (so the very latest edges from any code-graph activity
     during the wiki drain also flow in).

Step A finishes synchronously before step B begins so the wiki agent's
end-of-drain index regen sees a stable edge set.

Also exposed as functions so tests can drive each step without
spinning up the whole process.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import asyncpg

from services.ingestion.code_graph.bridge import enqueue_initial_backfill
from services.ingestion.code_graph.cross_repo_deps import drain_reverify_dlq
from services.synthesis.diagram_renderer import regenerate_wiki_diagram
from shared.config import get_settings
from shared.constants import WIKI_PENDING_CHANNEL, SourceSystem
from shared.logging import configure_logging, get_logger

log = get_logger(__name__)


# How long to wait between code-graph queue-drain polls. The queue
# scans every 5s itself; a 30s poll cadence catches drain within one
# cycle of the worker without DDoS-ing the table.
_CODE_GRAPH_POLL_INTERVAL_SECONDS = 30

# Hard ceiling on time spent waiting for code-graph to drain per
# customer. A pathological customer (clone failures, GitHub API
# rate limits) shouldn't hold the cron forever — we move on after
# this and let the next nightly run pick up the slack. Sized at 30
# minutes to comfortably fit even a fresh full-extraction.
_CODE_GRAPH_DRAIN_TIMEOUT_SECONDS = 30 * 60


async def trigger_nightly_synthesis(dsn: str) -> int:
    """Fire pg_notify on `wiki_synthesize_pending` for every opted-in
    customer with pending rows. Returns the count of customers notified.
    """
    started_at = datetime.now(UTC)
    customer_ids: list[str] = []
    conn = await asyncpg.connect(dsn)
    try:
        rows = await conn.fetch(
            """
            SELECT DISTINCT q.customer_id
            FROM wiki_synthesis_queue q
            JOIN customers c ON c.customer_id = q.customer_id
            WHERE q.status = 'pending'
              AND c.preferences->>'wiki_generation_enabled' = 'true'
            """
        )
        customer_ids = [row["customer_id"] for row in rows]
        for customer_id in customer_ids:
            await conn.execute(
                "SELECT pg_notify($1, $2)",
                WIKI_PENDING_CHANNEL,
                customer_id,
            )
    finally:
        await conn.close()
    log.info(
        "nightly_trigger.fired",
        customers=len(customer_ids),
        elapsed_seconds=(datetime.now(UTC) - started_at).total_seconds(),
        started_at=started_at.isoformat(),
    )
    return len(customer_ids)


async def refresh_cross_repo_edges(dsn: str) -> dict[str, int]:
    """Step A: re-enqueue code-graph backfills for every known
    (customer, repo) pair, wait for each customer's queue to drain, and
    regenerate the wiki index so the architecture diagram picks up the
    fresh edges.

    Returns a small summary dict for logging:
      ``customers``     — total customers processed
      ``repos_enqueued`` — total repos re-enqueued across customers
      ``index_regens``   — count of customers whose index regen succeeded
      ``drain_timeouts`` — count of customers we gave up on (still in queue)
    """
    started_at = datetime.now(UTC)
    summary = {
        "customers": 0,
        "repos_enqueued": 0,
        "index_regens": 0,
        "drain_timeouts": 0,
    }
    conn = await asyncpg.connect(dsn)
    try:
        rows = await conn.fetch(
            """
            SELECT customer_id, ARRAY_AGG(DISTINCT repo) AS repos
            FROM code_repo_state
            GROUP BY customer_id
            """
        )
    finally:
        await conn.close()

    if not rows:
        log.info("cross_repo_refresh.no_customers")
        return summary

    sha = f"head:{started_at.strftime('%Y%m%dT%H%M%S')}"

    for row in rows:
        customer_id = row["customer_id"]
        repos: list[str] = list(row["repos"] or [])
        if not repos:
            continue
        summary["customers"] += 1

        # Look up the customer's GitHub installation token id once.
        token_id = await _resolve_github_token_id(dsn, customer_id)
        if token_id is None:
            log.warning(
                "cross_repo_refresh.no_token",
                customer=customer_id,
                repos=len(repos),
            )
            continue

        for repo in repos:
            ok = await enqueue_initial_backfill(
                customer_id=customer_id,
                repo=repo,
                head_sha=sha,
                integration_token_id=token_id,
                originating_source=SourceSystem.GITHUB,
            )
            if ok:
                summary["repos_enqueued"] += 1

        drained = await _wait_for_code_graph_drain(dsn, customer_id)
        if not drained:
            summary["drain_timeouts"] += 1
            log.warning(
                "cross_repo_refresh.drain_timeout",
                customer=customer_id,
                repos=len(repos),
            )
            # Skip the index regen — the edge set would be stale-mid-
            # backfill. Next nightly run will retry; if a customer
            # systematically times out, that's a separate ops alert.
            continue

        try:
            wrote = await regenerate_wiki_diagram(customer_id=customer_id)
            if wrote:
                summary["index_regens"] += 1
        except Exception as exc:
            log.warning(
                "cross_repo_refresh.diagram_regen_failed",
                customer=customer_id,
                error=str(exc),
                error_class=type(exc).__name__,
            )

    log.info(
        "cross_repo_refresh.done",
        elapsed_seconds=(datetime.now(UTC) - started_at).total_seconds(),
        **summary,
    )
    return summary


async def _resolve_github_token_id(dsn: str, customer_id: str) -> str | None:
    """Pull the most recent ``installation:*`` token id for the customer.

    Returns ``None`` if no installation row exists (customer never
    completed the GitHub App handshake) — the refresh skips that
    customer instead of erroring.
    """
    conn = await asyncpg.connect(dsn)
    try:
        row = await conn.fetchrow(
            """
            SELECT token_id
            FROM integration_tokens
            WHERE customer_id = $1
              AND source_system = 'github'
              AND scope LIKE 'installation:%'
            ORDER BY token_id DESC
            LIMIT 1
            """,
            customer_id,
        )
    finally:
        await conn.close()
    return str(row["token_id"]) if row else None


async def _wait_for_code_graph_drain(dsn: str, customer_id: str) -> bool:
    """Poll until this customer has zero pending+processing code-graph
    rows, or the timeout fires. Returns True on drain, False on timeout.
    """
    deadline = datetime.now(UTC).timestamp() + _CODE_GRAPH_DRAIN_TIMEOUT_SECONDS
    while datetime.now(UTC).timestamp() < deadline:
        conn = await asyncpg.connect(dsn)
        try:
            in_flight = await conn.fetchval(
                """
                SELECT count(*)
                FROM ingestion_queue
                WHERE customer_id = $1
                  AND source_system = 'code_graph'
                  AND status IN ('pending', 'processing')
                """,
                customer_id,
            )
        finally:
            await conn.close()
        if in_flight == 0:
            return True
        await asyncio.sleep(_CODE_GRAPH_POLL_INTERVAL_SECONDS)
    return False


async def _drain_dlq() -> None:
    """Replay verifier-failed pushes that were parked for retry.

    Lazy-imports the ingestion-layer file fetcher and token resolver so
    nightly_trigger keeps a tight import surface for the cron container.
    """
    import httpx

    from services.ingestion.code_graph.fetch import fetch_files_at_sha
    from shared.backend_client import fetch_github_installation_token

    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as http:

        async def resolve_token(
            customer_id: str, token_id: str | None
        ) -> str | None:
            # `token_id` was recorded at DLQ enqueue time but the backend
            # mints fresh installation tokens by customer_id; the id
            # serves only as a "do we have an installation at all"
            # marker. We mint a fresh token here so a stale one parked
            # in the DLQ doesn't fail the whole retry.
            del token_id
            try:
                token, _expires = await fetch_github_installation_token(
                    http, customer_id=customer_id
                )
                return token
            except Exception as exc:
                log.warning(
                    "nightly_trigger.dlq_token_resolve_failed",
                    customer=customer_id,
                    error=str(exc),
                )
                return None

        summary = await drain_reverify_dlq(
            fetch_files_at_sha=fetch_files_at_sha,
            resolve_token=resolve_token,
        )
    log.info("nightly_trigger.dlq_drain_summary", **summary)


async def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    log.info("nightly_trigger.start", environment=settings.environment)

    # Step 0: drain the cross-repo DLQ first. If Gemini is back up,
    # parked verifications get re-run before today's refresh adds new
    # edges, so the diagram reflects the correct edge state when step
    # A regenerates it.
    try:
        await _drain_dlq()
    except Exception as exc:
        log.warning(
            "nightly_trigger.dlq_drain_failed",
            error=str(exc),
            error_class=type(exc).__name__,
        )

    # Step A: refresh cross-repo edges first so that step B's wiki drain
    # ends with a regenerate_wiki_index call against an up-to-date edge
    # set. Failures here are advisory — log + continue to step B.
    try:
        await refresh_cross_repo_edges(settings.database_url)
    except Exception as exc:
        log.warning(
            "nightly_trigger.cross_repo_refresh_failed",
            error=str(exc),
            error_class=type(exc).__name__,
        )

    # Step B: existing nightly synthesis trigger.
    notified = await trigger_nightly_synthesis(settings.database_url)
    log.info("nightly_trigger.done", customers=notified)


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(main())
