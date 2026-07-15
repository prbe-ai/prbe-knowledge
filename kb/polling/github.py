"""GitHub source poller (PR E1).

For self-host customers we cannot receive GitHub webhooks (no public
URL); instead the polling scheduler invokes this poller on a tick. The
poller calls the GitHub REST API for the customer's repo, extracts
issues + PRs, and hands them to the scheduler shaped exactly like the
webhook payloads the existing handler at
``services/ingestion/handlers/github.py`` consumes — so the downstream
normalizer has one code path regardless of origin.

Resource: ``resource_id`` is ``"<owner>/<repo>"``.
Cursor: ISO-8601 UTC timestamp of the most-recent ``updated_at`` we
have seen for this resource. ``None`` on first poll means "fetch the
last 7 days".

Endpoint: ``GET /repos/{owner}/{repo}/issues?since=<ts>&state=all&direction=asc&per_page=100``
The ``/issues`` endpoint returns BOTH issues and pull requests; rows
that are PRs carry a ``pull_request`` sub-object. ``since`` filters on
``updated_at`` so this is the right shape for incremental polling
(matches the upstream contract documented at
https://docs.github.com/rest/issues/issues#list-repository-issues).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from services.ingestion.polling.base import BasePoller, PollResult, register_poller
from shared.backend_client import fetch_github_installation_token
from shared.constants import GITHUB_INSTALLATION_SCOPE_PREFIX, SourceSystem
from shared.exceptions import GitHubAuthError
from shared.logging import get_logger
from shared.tokens import load_token

log = get_logger(__name__)


_GITHUB_API = "https://api.github.com"
_PER_PAGE = 100
_HTTP_TIMEOUT_SECONDS = 30.0
_FIRST_POLL_LOOKBACK_DAYS = 7
# Safety cap on pages walked in a single tick. The scheduler retries the
# same cursor on the next tick if there are more pages — we don't want a
# misbehaving repo (or a freshly-onboarded one with months of history) to
# starve every other customer in the tick budget. 50 pages * 100 rows =
# 5,000 issues/PRs per tick is plenty of headroom for a sane repo.
_MAX_PAGES_PER_TICK = 50

# GitHub event-type strings, mirrored from the webhook handler so the
# normalizer's existing branch on ``X-GitHub-Event`` works unchanged.
_EVENT_PULL_REQUEST = "pull_request"
_EVENT_ISSUES = "issues"


class GitHubPoller(BasePoller):
    """Poll GitHub for new/updated issues + PRs in one ``owner/repo``.

    One poller instance is created per scheduler tick (the scheduler
    does ``GitHubPoller()`` with no args). All state — the bearer,
    the http client — lives inside :meth:`poll`.
    """

    source = SourceSystem.GITHUB

    async def poll(
        self,
        *,
        customer_id: str,
        resource_id: str,
        cursor: str | None,
    ) -> PollResult:
        owner_repo = _validate_resource_id(resource_id)
        if owner_repo is None:
            return PollResult(
                documents=[],
                next_cursor=None,
                error=f"invalid resource_id {resource_id!r}; expected '<owner>/<repo>'",
            )

        token = await load_token(customer_id, SourceSystem.GITHUB)
        if token is None:
            return PollResult(
                documents=[],
                next_cursor=None,
                error="no active github integration_tokens row for customer",
            )

        since_iso = cursor or _default_since_iso()

        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT_SECONDS) as http:
            try:
                bearer = await _resolve_bearer(
                    http, token_access=token.access_token,
                    token_scope=token.scope,
                    customer_id=customer_id,
                )
            except GitHubAuthError as exc:
                return PollResult(
                    documents=[],
                    next_cursor=None,
                    error=f"github auth: {exc}",
                )

            headers = {
                "Authorization": f"Bearer {bearer}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            }

            try:
                rows, fetch_error = await _fetch_issues_paginated(
                    http,
                    owner_repo=owner_repo,
                    since_iso=since_iso,
                    headers=headers,
                )
            except httpx.HTTPError as exc:
                return PollResult(
                    documents=[],
                    next_cursor=None,
                    error=f"github http: {type(exc).__name__}: {exc}",
                )

        if fetch_error is not None:
            return PollResult(documents=[], next_cursor=None, error=fetch_error)

        if not rows:
            # Empty page is the steady-state "nothing new"; the scheduler
            # leaves the existing cursor in place via the COALESCE in
            # ``advance_cursor`` when we pass next_cursor=None.
            return PollResult(documents=[], next_cursor=None)

        documents = _rows_to_webhook_documents(customer_id, owner_repo, rows)
        next_cursor = _max_updated_at(rows)
        return PollResult(documents=documents, next_cursor=next_cursor)


# --- helpers -----------------------------------------------------------------


def _validate_resource_id(resource_id: str) -> str | None:
    """Return the validated ``owner/repo`` string, or None if malformed.

    Accepts exactly one slash; both sides must be non-empty. GitHub's
    own repo-name rules are stricter but we lean conservative — the
    REST call will surface anything we wave through.
    """
    if not isinstance(resource_id, str):
        return None
    parts = resource_id.split("/")
    if len(parts) != 2:
        return None
    owner, repo = parts
    if not owner or not repo:
        return None
    return resource_id


def _default_since_iso() -> str:
    """ISO-8601 UTC timestamp ``_FIRST_POLL_LOOKBACK_DAYS`` ago.

    GitHub's ``since`` parameter wants RFC 3339; ``isoformat`` on a
    UTC datetime emits a compatible string when we replace ``+00:00``
    with ``Z`` (which GitHub accepts and is the canonical form).
    """
    ts = datetime.now(UTC) - timedelta(days=_FIRST_POLL_LOOKBACK_DAYS)
    # GitHub's API documents both forms; stick with the Z form to
    # match what we'll later parse out of webhook payloads.
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


async def _resolve_bearer(
    http: httpx.AsyncClient,
    *,
    token_access: str,
    token_scope: str | None,
    customer_id: str,
) -> str:
    """Return the bearer to send on GitHub API calls.

    Mirrors ``GitHubConnector._resolve_installation_bearer`` in
    ``services/ingestion/handlers/github.py``: if the stored token
    scope is ``installation:<id>``, mint a fresh App installation
    token via prbe-backend (the App private key only lives there);
    otherwise the stored access_token is a PAT/legacy bearer and is
    used verbatim.
    """
    scope = token_scope or ""
    if scope.startswith(GITHUB_INSTALLATION_SCOPE_PREFIX):
        bearer, _expires = await fetch_github_installation_token(
            http, customer_id=customer_id
        )
        return bearer
    return token_access


async def _fetch_issues_paginated(
    http: httpx.AsyncClient,
    *,
    owner_repo: str,
    since_iso: str,
    headers: dict[str, str],
) -> tuple[list[dict[str, Any]], str | None]:
    """Walk ``GET /repos/{owner_repo}/issues`` from ``since_iso`` forward.

    Returns ``(rows, error)``. On a non-empty error string the rows
    list is whatever we managed to collect before the failure — the
    caller treats any error as "stamp + don't advance cursor", so
    partial rows are discarded.
    """
    first_url = (
        f"{_GITHUB_API}/repos/{owner_repo}/issues"
        f"?since={since_iso}"
        f"&state=all"
        f"&direction=asc"
        f"&sort=updated"
        f"&per_page={_PER_PAGE}"
    )
    url: str | None = first_url
    rows: list[dict[str, Any]] = []
    pages = 0

    while url is not None and pages < _MAX_PAGES_PER_TICK:
        resp = await http.get(url, headers=headers)
        pages += 1

        if resp.status_code == 429 or (
            resp.status_code == 403
            and resp.headers.get("x-ratelimit-remaining") == "0"
        ):
            retry_after = resp.headers.get("retry-after")
            return [], f"github rate limited: status={resp.status_code} retry_after={retry_after}"

        if resp.status_code >= 500:
            return [], f"github upstream {resp.status_code}: {resp.text[:200]}"

        if resp.status_code == 404:
            return [], f"github repo not found: {owner_repo}"

        if resp.status_code >= 400:
            return [], f"github {resp.status_code}: {resp.text[:200]}"

        body = resp.json()
        if not isinstance(body, list):
            # The /issues endpoint always returns an array. Defensive
            # log + bail; treating it as an error keeps us from
            # advancing the cursor past data we never saw.
            return [], f"github unexpected response shape: {type(body).__name__}"

        for row in body:
            if isinstance(row, dict):
                rows.append(row)

        url = _next_link(resp)

    if pages >= _MAX_PAGES_PER_TICK and url is not None:
        log.info(
            "polling.github.page_cap_hit",
            owner_repo=owner_repo,
            pages=pages,
            collected_rows=len(rows),
        )

    return rows, None


def _next_link(resp: httpx.Response) -> str | None:
    """Parse GitHub's ``Link`` header for the ``rel="next"`` URL.

    Same shape as the helper in ``handlers/github.py``; reimplemented
    here so the poller has no cross-module private-helper dependency.
    """
    link_header = resp.headers.get("link") or resp.headers.get("Link")
    if not link_header:
        return None
    for raw_part in link_header.split(","):
        part = raw_part.strip()
        if 'rel="next"' not in part:
            continue
        if part.startswith("<") and ">" in part:
            return part.split(">", 1)[0][1:]
    return None


def _max_updated_at(rows: list[dict[str, Any]]) -> str | None:
    """Return the largest ``updated_at`` string in ``rows``, or None.

    ISO-8601 with the ``Z`` suffix sorts lexicographically the same as
    by timestamp, so a plain ``max`` over strings is safe here.
    Rows without an updated_at are skipped (the same row will resurface
    on the next tick if upstream eventually stamps one).
    """
    candidates = [
        row["updated_at"]
        for row in rows
        if isinstance(row.get("updated_at"), str) and row["updated_at"]
    ]
    if not candidates:
        return None
    return max(candidates)


def _rows_to_webhook_documents(
    customer_id: str,
    owner_repo: str,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Map ``GET /issues`` rows into ``WebhookEvent``-shaped dicts.

    The returned dicts mirror ``shared.models.WebhookEvent`` field for
    field (``customer_id``, ``source_system``, ``source_event_id``,
    ``received_at``, ``raw_payload``, ``headers``) so the scheduler's
    sink (wired in PR C) can lift them into ``WebhookEvent`` instances
    without remapping. This matches the shape the existing backfill
    emitter in ``handlers/github.py`` yields from the SAME REST
    endpoint, so the downstream normalizer
    (``GitHubConnector._normalize_pr`` / ``_normalize_issue``) sees
    identical inputs whether the row arrived via webhook, backfill,
    or this poller.

    Required fields inside ``raw_payload`` per the normalizer:

      * ``raw_payload["action"]`` — ``"opened"`` to match backfill.
      * ``raw_payload["repository"]["full_name"]`` — synthesized from
        ``owner_repo`` since /issues rows don't carry a full repository
        object.
      * ``raw_payload["pull_request"]`` *or* ``raw_payload["issue"]`` —
        the row itself.
    """
    owner, _repo = owner_repo.split("/", 1)
    documents: list[dict[str, Any]] = []
    for row in rows:
        number = row.get("number")
        updated_at = row.get("updated_at")
        if number is None or not updated_at:
            continue

        is_pr = isinstance(row.get("pull_request"), dict)
        repo_stub = _repository_stub(owner_repo, owner)

        if is_pr:
            raw_payload = {
                "action": "opened",
                "repository": repo_stub,
                "pull_request": row,
            }
            event_type = _EVENT_PULL_REQUEST
            source_event_id = f"pr:{owner_repo}:{number}:opened:{updated_at}"
        else:
            raw_payload = {
                "action": "opened",
                "repository": repo_stub,
                "issue": row,
            }
            event_type = _EVENT_ISSUES
            source_event_id = f"issue:{owner_repo}:{number}:opened:{updated_at}"

        documents.append(
            {
                "customer_id": customer_id,
                "source_system": SourceSystem.GITHUB.value,
                "source_event_id": source_event_id,
                "received_at": updated_at,
                "raw_payload": raw_payload,
                "headers": {"X-GitHub-Event": event_type},
            }
        )
    return documents


def _repository_stub(owner_repo: str, owner: str) -> dict[str, Any]:
    """Minimal ``repository`` block the webhook parsers need.

    The /issues endpoint embeds a ``repository_url`` per row but not a
    full repository object. ``full_name`` and ``owner.login`` are all
    the parsers in ``handlers/github.py`` actually require on this
    path; everything richer (visibility, default_branch, ...) lives on
    the repo-level events those parsers don't gate on here.
    """
    return {
        "full_name": owner_repo,
        "owner": {"login": owner},
    }


# Module-import-time registration. The scheduler reads from the
# registry at tick time so importing this module anywhere in the
# polling pod's boot path wires GitHub in.
register_poller(SourceSystem.GITHUB, GitHubPoller)


__all__ = ["GitHubPoller"]
