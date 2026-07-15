"""GitHub Phase 2 fan-out discoverer.

After the broad-pass GitHub crawler completes, this discoverer queries
``list_installation_repos()`` and returns top-N repos by ``pushed_at``
(N = ``BACKFILL_MAX_TARGETS_PER_SOURCE``, default 30). Each repo
becomes a Phase 2 row with ``target='owner/repo'``; the existing
``BackfillWorker._claim_one()`` picks them up and dispatches a scoped
``GitHubBackfillAgent`` per repo.
"""

from __future__ import annotations

from typing import ClassVar

import httpx

from engine.shared.constants import BACKFILL_MAX_TARGETS_PER_SOURCE
from engine.shared.logging import get_logger
from kb.synthesis.api_clients.github import GitHubAPIClient, get_shared_bucket

log = get_logger(__name__)


class GitHubBackfillFanout:
    """Discovers GitHub repos worth a Phase 2 deep dive.

    Filters out archived + disabled repos (no signal there). Sorts by
    ``pushed_at desc`` (the API client already does this) and truncates
    at ``BACKFILL_MAX_TARGETS_PER_SOURCE``.
    """

    source: ClassVar[str] = "github"

    async def discover_targets(
        self,
        *,
        customer_id: str,
        bearer: str,
        http: httpx.AsyncClient,
    ) -> list[str]:
        try:
            # Reuse the shared per-customer bucket. The API client already
            # honors it; this just makes sure the discover call is part of
            # the same rate envelope as the Phase 2 agents that follow.
            bucket = get_shared_bucket(customer_id, "github")
            client = GitHubAPIClient(bearer=bearer, http=http, bucket=bucket)
            collected: list[tuple[str, str]] = []
            async for repo in client.list_installation_repos():
                if repo.get("archived") or repo.get("disabled"):
                    continue
                full_name = repo.get("full_name")
                if not isinstance(full_name, str) or not full_name:
                    continue
                pushed_at = str(repo.get("pushed_at") or "")
                collected.append((full_name, pushed_at))
        except Exception as exc:
            log.warning(
                "backfill.fanout_discover_failed",
                customer=customer_id,
                source=self.source,
                error=str(exc),
                error_class=type(exc).__name__,
            )
            return []

        # Defense in depth — list_installation_repos already returns
        # pushed_at desc, but sort again so a future API behavior change
        # doesn't silently break the cap selection.
        collected.sort(key=lambda r: r[1], reverse=True)
        return [full_name for full_name, _ in collected[:BACKFILL_MAX_TARGETS_PER_SOURCE]]
