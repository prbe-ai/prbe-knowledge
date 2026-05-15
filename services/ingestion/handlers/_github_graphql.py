"""GitHub GraphQL v4 client for the backfill loop.

Scope is intentionally narrow: this module powers `GitHubConnector.backfill()`
only. The webhook path still uses REST (parse_webhook_event), and code_graph
fetches still use REST. Adding a second engine here lets backfill burn through
a tenant's history with parallel repo fan-out at a fraction of the REST request
budget.

Exposed surface:
- `BACKFILL_PULLS_QUERY` / `BACKFILL_ISSUES_QUERY`: GraphQL strings.
- `GITHUB_GRAPHQL_URL`: endpoint constant.
- `run_graphql(http, headers, query, variables)`: POSTs, honors 429 / 403+rate
  / `retry-after` / `x-ratelimit-reset` like the REST loop did, and proactively
  sleeps when the response body's `rateLimit.remaining` drops below the floor.
- `normalize_pr_node(node)` / `normalize_issue_node(node)`: map GraphQL camelCase
  shape to REST snake_case shape so the downstream normalizer (which only
  reads REST keys) keeps working without modification.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

from shared.logging import get_logger

log = get_logger(__name__)

GITHUB_GRAPHQL_URL = "https://api.github.com/graphql"

# Floor for body-reported remaining points before we proactively back off.
# GitHub gives 5000 points/hour to a typical installation; queries here cost
# ~3 points/page. 100 is roughly two minutes of headroom at the worst page
# cost, which is enough room for the reset window to roll forward.
_RATE_LIMIT_REMAINING_FLOOR = 100

# Bounded page sizes. PRs ship heavier nodes (comments/reviews/files nested),
# so keep them smaller than issues to stay under the 500k-node-cost ceiling.
_PR_PAGE_SIZE = 50
_ISSUE_PAGE_SIZE = 100


BACKFILL_PULLS_QUERY = (
    "query BackfillPulls($owner: String!, $name: String!, $cursor: String) {\n"
    "  repository(owner: $owner, name: $name) {\n"
    f"    pullRequests(first: {_PR_PAGE_SIZE}, after: $cursor, "
    "orderBy: {field: UPDATED_AT, direction: DESC}, "
    "states: [OPEN, CLOSED, MERGED]) {\n"
    "      pageInfo { endCursor hasNextPage }\n"
    "      nodes {\n"
    "        number title body state url\n"
    "        createdAt updatedAt closedAt mergedAt merged\n"
    "        changedFiles additions deletions\n"
    "        baseRefName headRefName\n"
    "        author { login }\n"
    "        labels(first: 30) { nodes { name } }\n"
    "        assignees(first: 10) { nodes { login } }\n"
    "        comments(first: 100) {\n"
    "          nodes { id body createdAt updatedAt author { login } }\n"
    "        }\n"
    "        reviews(first: 50) {\n"
    "          nodes {\n"
    "            id state body submittedAt author { login }\n"
    "            comments(first: 50) {\n"
    "              nodes { id body path createdAt author { login } }\n"
    "            }\n"
    "          }\n"
    "        }\n"
    "        files(first: 100) { nodes { path additions deletions } }\n"
    "        commits(first: 50) {\n"
    "          nodes {\n"
    "            commit {\n"
    "              oid message committedDate\n"
    "              author { name email user { login } }\n"
    "            }\n"
    "          }\n"
    "        }\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "  rateLimit { cost remaining resetAt }\n"
    "}\n"
)


BACKFILL_ISSUES_QUERY = (
    "query BackfillIssues($owner: String!, $name: String!, $cursor: String) {\n"
    "  repository(owner: $owner, name: $name) {\n"
    f"    issues(first: {_ISSUE_PAGE_SIZE}, after: $cursor, "
    "orderBy: {field: UPDATED_AT, direction: DESC}, filterBy: {}) {\n"
    "      pageInfo { endCursor hasNextPage }\n"
    "      nodes {\n"
    "        number title body state url\n"
    "        createdAt updatedAt closedAt\n"
    "        author { login }\n"
    "        labels(first: 30) { nodes { name } }\n"
    "        assignees(first: 10) { nodes { login } }\n"
    "        comments(first: 100) {\n"
    "          nodes { id body createdAt updatedAt author { login } }\n"
    "        }\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "  rateLimit { cost remaining resetAt }\n"
    "}\n"
)


async def run_graphql(
    http,
    auth_headers: dict[str, str],
    query: str,
    variables: dict[str, Any],
) -> dict[str, Any] | None:
    """POST a GraphQL query to api.github.com/graphql.

    Returns the `data` block on success. Returns None on a non-recoverable
    failure (the caller should advance state past the current page).

    Handles three retry/backoff cases:
    1. HTTP 429 or 403 with `x-ratelimit-remaining: 0` — sleep per
       retry-after or x-ratelimit-reset, then retry once.
    2. Response body reports `rateLimit.remaining < FLOOR` — proactively
       sleep until resetAt before returning. Lets downstream callers keep
       paging without burning their entire window.
    3. Body contains `errors` — log + return None.
    """
    import httpx

    payload = {"query": query, "variables": variables}

    for attempt in range(2):
        try:
            resp = await http.post(
                GITHUB_GRAPHQL_URL,
                headers=auth_headers,
                json=payload,
            )
        except httpx.HTTPError as exc:
            log.warning("github.graphql_http_error", error=str(exc))
            return None

        if resp.status_code == 429 or (
            resp.status_code == 403
            and resp.headers.get("x-ratelimit-remaining") == "0"
        ):
            await asyncio.sleep(_compute_retry_delay(resp))
            if attempt == 0:
                continue
            return None

        if resp.status_code != 200:
            log.warning(
                "github.graphql_non_200",
                status=resp.status_code,
                body=resp.text[:200],
            )
            return None

        body = resp.json()
        if not isinstance(body, dict):
            return None
        if body.get("errors"):
            log.warning("github.graphql_errors", errors=body.get("errors"))
            return None

        data = body.get("data")
        if not isinstance(data, dict):
            return None

        rate_limit = data.get("rateLimit") or {}
        remaining = rate_limit.get("remaining")
        reset_at = rate_limit.get("resetAt")
        if (
            isinstance(remaining, int)
            and remaining < _RATE_LIMIT_REMAINING_FLOOR
            and isinstance(reset_at, str)
        ):
            # Proactive backoff: sleep until the reset rolls. Cap at 60s so a
            # bogus resetAt can't park a backfill indefinitely.
            try:
                reset_dt = datetime.fromisoformat(reset_at.replace("Z", "+00:00"))
                delay = max(
                    int((reset_dt - datetime.now(UTC)).total_seconds()), 1
                )
            except ValueError:
                delay = 5
            await asyncio.sleep(min(delay, 60))

        return data

    return None


def _compute_retry_delay(resp) -> int:
    """Translate GitHub rate-limit response headers into a sleep duration."""
    retry_after = resp.headers.get("retry-after")
    if retry_after is not None:
        try:
            return max(int(retry_after), 1)
        except ValueError:
            return 5

    reset = resp.headers.get("x-ratelimit-reset")
    if reset:
        try:
            return max(int(float(reset)) - int(datetime.now(UTC).timestamp()), 1)
        except ValueError:
            return 5
    return 5


def normalize_pr_node(node: dict[str, Any]) -> dict[str, Any]:
    """Map a GraphQL PullRequest node to the REST `pull_request` shape.

    Keys mirror what `_normalize_pr` reads:
      number, title, body, state, html_url, created_at, updated_at, closed_at,
      merged_at, merged, changed_files, additions, deletions, user.login,
      base.ref, head.ref, labels[].name, assignees[].login.

    GraphQL `state` is uppercased and includes MERGED as a third value; REST
    only has open/closed (with `merged: true` flagging the MERGED case).
    """
    raw_state = (node.get("state") or "").upper()
    if raw_state == "MERGED":
        state = "closed"
        merged = True
    elif raw_state == "CLOSED":
        state = "closed"
        merged = bool(node.get("merged"))
    else:
        state = "open"
        merged = bool(node.get("merged"))

    return {
        "number": node.get("number"),
        "title": node.get("title") or "",
        "body": node.get("body") or "",
        "state": state,
        "html_url": node.get("url") or "",
        "created_at": node.get("createdAt"),
        "updated_at": node.get("updatedAt"),
        "closed_at": node.get("closedAt"),
        "merged_at": node.get("mergedAt"),
        "merged": merged,
        "changed_files": node.get("changedFiles"),
        "additions": node.get("additions"),
        "deletions": node.get("deletions"),
        "user": {"login": ((node.get("author") or {}).get("login")) or "unknown"},
        "base": {"ref": node.get("baseRefName")},
        "head": {"ref": node.get("headRefName")},
        "labels": [
            {"name": la.get("name")}
            for la in ((node.get("labels") or {}).get("nodes") or [])
            if isinstance(la, dict) and la.get("name")
        ],
        "assignees": [
            {"login": a.get("login")}
            for a in ((node.get("assignees") or {}).get("nodes") or [])
            if isinstance(a, dict) and a.get("login")
        ],
    }


def normalize_issue_node(node: dict[str, Any]) -> dict[str, Any]:
    """Map a GraphQL Issue node to the REST `issue` shape.

    Keys mirror what `_normalize_issue` reads:
      number, title, body, state, html_url, created_at, updated_at, closed_at,
      user.login, labels[].name, assignees[].login.

    GraphQL `state` is uppercased; REST is lowercase.
    """
    return {
        "number": node.get("number"),
        "title": node.get("title") or "",
        "body": node.get("body") or "",
        "state": (node.get("state") or "open").lower(),
        "html_url": node.get("url") or "",
        "created_at": node.get("createdAt"),
        "updated_at": node.get("updatedAt"),
        "closed_at": node.get("closedAt"),
        "user": {"login": ((node.get("author") or {}).get("login")) or "unknown"},
        "labels": [
            {"name": la.get("name")}
            for la in ((node.get("labels") or {}).get("nodes") or [])
            if isinstance(la, dict) and la.get("name")
        ],
        "assignees": [
            {"login": a.get("login")}
            for a in ((node.get("assignees") or {}).get("nodes") or [])
            if isinstance(a, dict) and a.get("login")
        ],
    }


__all__ = [
    "BACKFILL_ISSUES_QUERY",
    "BACKFILL_PULLS_QUERY",
    "GITHUB_GRAPHQL_URL",
    "normalize_issue_node",
    "normalize_pr_node",
    "run_graphql",
]
