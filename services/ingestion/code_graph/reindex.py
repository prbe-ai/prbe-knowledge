"""Manual code-graph reindex for one customer.

Mirrors `scripts/code_graph_backfill_existing.py`'s per-tenant logic but
exposes it as an awaitable function so an HTTP route can call it. The
script is the original, ssh-into-the-worker one-shot path; this module
is what the dashboard's "Reindex code graph" cog ultimately invokes via
the BFF.

Per-customer flow:
  1. Find the customer's active github `integration_tokens` row. The
     scope must be `installation:<id>` (PAT-based github tokens are not
     supported here, same as the script).
  2. Mint a fresh installation token via prbe-backend.
  3. List the repos the installation can read.
  4. For each repo, look up HEAD SHA on the default branch and call
     `enqueue_initial_backfill`. Skip archived repos.

Idempotent end-to-end: bridge dedupes on (customer_id, source_event_id),
which embeds the SHA, so re-running with the same HEAD SHA per repo is a
no-op.

Raises `ReindexNotConnected` if the customer has no active
installation-scoped github integration. Raises `GitHubAuthError` if the
backend token endpoint is unhealthy.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from services.ingestion.code_graph.bridge import enqueue_initial_backfill
from shared.backend_client import fetch_github_installation_token
from shared.constants import GITHUB_INSTALLATION_SCOPE_PREFIX, SourceSystem
from shared.db import raw_conn
from shared.logging import get_logger

log = get_logger(__name__)

_GITHUB_API = "https://api.github.com"
_GITHUB_PAGE_SIZE = 100
_HTTP_TIMEOUT_S = 30.0


class ReindexNotConnected(Exception):
    """Customer has no active installation-scoped github integration."""


@dataclass(slots=True, frozen=True)
class ReindexResult:
    enqueued: int
    skipped: int
    repos: list[str]


async def _resolve_installation_id(customer_id: str) -> str:
    """Active github integration_tokens row for this customer.

    Mirrors the script's `_list_tenants` filter: `source_system='github'`,
    `status='active'`, scope must be `installation:<id>`.
    """
    async with raw_conn() as conn:
        row = await conn.fetchrow(
            """
            SELECT scope
            FROM integration_tokens
            WHERE customer_id = $1
              AND source_system = 'github'
              AND status = 'active'
            ORDER BY token_id DESC
            LIMIT 1
            """,
            customer_id,
        )
    if row is None:
        raise ReindexNotConnected(
            f"no active github integration for customer {customer_id}"
        )
    scope = row["scope"] or ""
    if not scope.startswith(GITHUB_INSTALLATION_SCOPE_PREFIX):
        raise ReindexNotConnected(
            f"github integration for {customer_id} is not installation-scoped"
        )
    return scope[len(GITHUB_INSTALLATION_SCOPE_PREFIX):]


async def _list_installation_repos(
    http: httpx.AsyncClient, bearer: str
) -> list[dict]:
    repos: list[dict] = []
    page = 1
    while True:
        resp = await http.get(
            f"{_GITHUB_API}/installation/repositories",
            params={"per_page": _GITHUB_PAGE_SIZE, "page": page},
            headers={
                "Authorization": f"Bearer {bearer}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        resp.raise_for_status()
        body = resp.json()
        chunk = body.get("repositories", []) or []
        repos.extend(chunk)
        if len(chunk) < _GITHUB_PAGE_SIZE:
            break
        page += 1
    return repos


async def _resolve_head_sha(
    http: httpx.AsyncClient, bearer: str, full_name: str, default_branch: str
) -> str | None:
    resp = await http.get(
        f"{_GITHUB_API}/repos/{full_name}/git/ref/heads/{default_branch}",
        headers={
            "Authorization": f"Bearer {bearer}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    if resp.status_code == 404:
        log.warning(
            "code_graph.reindex.head_ref_404",
            repo=full_name,
            branch=default_branch,
        )
        return None
    resp.raise_for_status()
    body = resp.json()
    sha = (body.get("object") or {}).get("sha")
    return sha if isinstance(sha, str) else None


async def reindex_customer(customer_id: str) -> ReindexResult:
    """Enqueue initial-backfill events for every repo the customer's github
    installation can see at HEAD."""
    installation_id = await _resolve_installation_id(customer_id)

    enqueued = 0
    skipped = 0
    repos: list[str] = []

    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT_S) as http:
        bearer, _expires = await fetch_github_installation_token(
            http, customer_id=customer_id
        )
        raw_repos = await _list_installation_repos(http, bearer)
        log.info(
            "code_graph.reindex.repos_visible",
            customer=customer_id,
            installation=installation_id,
            repo_count=len(raw_repos),
        )

        for r in raw_repos:
            full_name = r.get("full_name")
            default_branch = r.get("default_branch")
            archived = bool(r.get("archived"))
            if not isinstance(full_name, str) or not isinstance(default_branch, str):
                skipped += 1
                continue
            if archived:
                log.info(
                    "code_graph.reindex.skip_archived",
                    customer=customer_id,
                    repo=full_name,
                )
                skipped += 1
                continue

            sha = await _resolve_head_sha(http, bearer, full_name, default_branch)
            if sha is None:
                skipped += 1
                continue

            await enqueue_initial_backfill(
                customer_id=customer_id,
                repo=full_name,
                head_sha=sha,
                # token_id is bigint; the downstream connector mints its own
                # installation token from customer_id (see codegraph.py
                # _normalize_initial_backfill), so we don't need to thread it.
                integration_token_id=None,
                originating_source=SourceSystem.GITHUB,
            )
            enqueued += 1
            repos.append(full_name)

    log.info(
        "code_graph.reindex.done",
        customer=customer_id,
        installation=installation_id,
        enqueued=enqueued,
        skipped=skipped,
    )
    return ReindexResult(enqueued=enqueued, skipped=skipped, repos=repos)
