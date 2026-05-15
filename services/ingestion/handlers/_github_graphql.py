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
# Commits and releases ship lighter nodes than PRs; bigger pages are safe.
_COMMIT_PAGE_SIZE = 100
_RELEASE_PAGE_SIZE = 100


# REST `pull_request_review.state` values we map GraphQL onto.
_REST_REVIEW_STATES = {
    "APPROVED": "approved",
    "CHANGES_REQUESTED": "changes_requested",
    "COMMENTED": "commented",
    "DISMISSED": "dismissed",
    "PENDING": "pending",
}


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
    "            id databaseId state body submittedAt author { login }\n"
    "          }\n"
    "        }\n"
    "        files(first: 100) { nodes { path additions deletions } }\n"
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


BACKFILL_COMMITS_QUERY = (
    "query BackfillCommits($owner: String!, $name: String!, $cursor: String) {\n"
    "  repository(owner: $owner, name: $name) {\n"
    "    defaultBranchRef {\n"
    "      name\n"
    "      target {\n"
    "        ... on Commit {\n"
    f"          history(first: {_COMMIT_PAGE_SIZE}, after: $cursor) {{\n"
    "            pageInfo { endCursor hasNextPage }\n"
    "            nodes {\n"
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


BACKFILL_RELEASES_QUERY = (
    "query BackfillReleases($owner: String!, $name: String!, $cursor: String) {\n"
    "  repository(owner: $owner, name: $name) {\n"
    f"    releases(first: {_RELEASE_PAGE_SIZE}, after: $cursor, "
    "orderBy: {field: CREATED_AT, direction: DESC}) {\n"
    "      pageInfo { endCursor hasNextPage }\n"
    "      nodes {\n"
    "        id databaseId tagName name body\n"
    "        createdAt publishedAt updatedAt\n"
    "        isDraft isPrerelease\n"
    "        author { login }\n"
    "        url\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "  rateLimit { cost remaining resetAt }\n"
    "}\n"
)


_RECOVERABLE_GRAPHQL_ERROR_TYPES = {"RATE_LIMITED", "MAX_NODE_LIMIT_EXCEEDED"}
# Up to 3 attempts total: initial + 2 retries with backoffs of 1s and 2s.
# Covers transient 5xx (502/503/504 common under load) and recoverable
# GraphQL `errors[]` blocks (RATE_LIMITED / MAX_NODE_LIMIT_EXCEEDED).
_TRANSIENT_BACKOFF_SECONDS = (1, 2)


async def run_graphql(
    http,
    auth_headers: dict[str, str],
    query: str,
    variables: dict[str, Any],
) -> dict[str, Any] | None:
    """POST a GraphQL query to api.github.com/graphql.

    Returns the `data` block on success. Returns None on a non-recoverable
    failure (the caller should advance state past the current page).

    Handles five retry/backoff cases:
    1. HTTP 429 or 403 with `x-ratelimit-remaining: 0` -- sleep per
       retry-after or x-ratelimit-reset, then retry once.
    2. HTTP 500/502/503/504 -- transient infra blip, retry with backoff
       (1s then 2s). Killing a repo phase on a single 502 silently drops
       the rest of its page stream.
    3. Response body reports `rateLimit.remaining < FLOOR` -- proactively
       sleep until resetAt before returning. Lets downstream callers keep
       paging without burning their entire window.
    4. Body contains `errors` with recoverable types
       (RATE_LIMITED / MAX_NODE_LIMIT_EXCEEDED) -- retry with backoff.
       If GitHub returned partial `data` alongside the error, surface it
       so the caller can emit those nodes (the embedded cursor stays
       valid for the next page).
    5. Body contains non-recoverable `errors` (FORBIDDEN, etc.) -- log
       and return None.
    """
    import httpx

    payload = {"query": query, "variables": variables}

    # max_attempts = 1 + len(backoff schedule); attempt 0 is the initial call.
    max_attempts = 1 + len(_TRANSIENT_BACKOFF_SECONDS)
    for attempt in range(max_attempts):
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
            if attempt + 1 < max_attempts:
                continue
            return None

        if resp.status_code in (500, 502, 503, 504):
            # Transient infra blip. Don't kill the phase on one bad page.
            if attempt + 1 < max_attempts:
                await asyncio.sleep(_TRANSIENT_BACKOFF_SECONDS[attempt])
                continue
            log.warning(
                "github.graphql_5xx_exhausted",
                status=resp.status_code,
                body=resp.text[:200],
            )
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

        errors = body.get("errors")
        data = body.get("data")

        if errors:
            error_types = {
                (e.get("type") or "").upper()
                for e in errors
                if isinstance(e, dict)
            }
            recoverable = bool(error_types & _RECOVERABLE_GRAPHQL_ERROR_TYPES)
            if recoverable and attempt + 1 < max_attempts:
                log.warning(
                    "github.graphql_recoverable_errors",
                    error_types=sorted(error_types),
                    has_partial_data=isinstance(data, dict),
                )
                await asyncio.sleep(_TRANSIENT_BACKOFF_SECONDS[attempt])
                # If partial data is present alongside a recoverable error,
                # surface it: the nodes are valid and their embedded cursor
                # advances the page stream. The retry covers the next page.
                if isinstance(data, dict):
                    return data
                continue
            log.warning("github.graphql_errors", errors=errors)
            if isinstance(data, dict):
                return data
            return None

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


def normalize_review_node(
    node: dict[str, Any], pr_html_url: str | None = None
) -> dict[str, Any]:
    """Map a GraphQL PullRequestReview node to the REST `review` shape.

    Keys mirror what `_normalize_review` reads:
      id, state, body, submitted_at, html_url, user.login.

    GraphQL `state` is uppercase (APPROVED / CHANGES_REQUESTED / COMMENTED /
    DISMISSED / PENDING); the REST equivalent is lowercase + snake_case.

    Prefers `databaseId` (REST numeric id) for `id`, falling back to the
    GraphQL global id (string) when absent. Without this, backfill would
    write review docs keyed on the string global-id while live webhooks
    write on the integer REST id, producing two pgvector rows per review.

    `html_url` is synthesized from the parent PR url + databaseId so the
    downstream Document's source_url is non-empty (GraphQL Review nodes
    don't carry their own url field).
    """
    raw_state = (node.get("state") or "").upper()
    rest_state = _REST_REVIEW_STATES.get(raw_state, raw_state.lower())
    rest_id = node.get("databaseId")
    if rest_id is None:
        rest_id = node.get("id")
    html_url = ""
    if pr_html_url and node.get("databaseId") is not None:
        html_url = f"{pr_html_url}#pullrequestreview-{node.get('databaseId')}"
    return {
        "id": rest_id,
        "state": rest_state,
        "body": node.get("body") or "",
        "submitted_at": node.get("submittedAt"),
        "html_url": html_url,
        "user": {"login": ((node.get("author") or {}).get("login")) or "unknown"},
    }


def normalize_commit_node(
    node: dict[str, Any], default_branch: str | None = None
) -> dict[str, Any]:
    """Map a GraphQL Commit history node to the push-event commit shape.

    Output matches `_rest_commit_to_push_commit` so the downstream
    `_normalize_push` path sees a uniform structure. `default_branch` is
    accepted for symmetry with the REST helper but isn't embedded here —
    the synthesized push `ref` is built by the caller.
    """
    del default_branch  # unused; kept for API symmetry with the REST helper
    author = node.get("author") or {}
    user = author.get("user") if isinstance(author, dict) else None
    username = user.get("login") if isinstance(user, dict) else None
    return {
        "id": node.get("oid") or "",
        "message": node.get("message") or "",
        "timestamp": node.get("committedDate") or "",
        "author": {
            "name": (author.get("name") if isinstance(author, dict) else "") or "",
            "email": (author.get("email") if isinstance(author, dict) else "") or "",
            "username": username,
        },
        # GraphQL Commit doesn't expose `url` on this query path; leave blank
        # rather than synthesize. _normalize_push reads url best-effort.
        "url": "",
        # File deltas are intentionally empty for backfilled commits, matching
        # the REST path's behaviour (avoids N+1 per-commit fetches).
        "added": [],
        "modified": [],
        "removed": [],
    }


def normalize_release_node(node: dict[str, Any]) -> dict[str, Any]:
    """Map a GraphQL Release node to the REST `release` shape.

    Keys mirror what `_normalize_release` reads:
      id, tag_name, name, body, published_at, created_at, updated_at,
      draft, prerelease, html_url, author.login.

    Prefers `databaseId` (REST numeric id) for the `id` field, falling back
    to the GraphQL global id when absent (older releases occasionally lack
    a databaseId in GraphQL responses).
    """
    rest_id = node.get("databaseId")
    if rest_id is None:
        rest_id = node.get("id")
    author = node.get("author") or {}
    author_login = author.get("login") if isinstance(author, dict) else None
    return {
        "id": rest_id,
        "tag_name": node.get("tagName") or "",
        "name": node.get("name") or "",
        "body": node.get("body") or "",
        "published_at": node.get("publishedAt"),
        "created_at": node.get("createdAt"),
        "updated_at": node.get("updatedAt"),
        "draft": bool(node.get("isDraft")),
        "prerelease": bool(node.get("isPrerelease")),
        "html_url": node.get("url") or "",
        "author": (
            {"login": author_login}
            if isinstance(author_login, str) and author_login
            else None
        ),
    }


__all__ = [
    "BACKFILL_COMMITS_QUERY",
    "BACKFILL_ISSUES_QUERY",
    "BACKFILL_PULLS_QUERY",
    "BACKFILL_RELEASES_QUERY",
    "GITHUB_GRAPHQL_URL",
    "normalize_commit_node",
    "normalize_issue_node",
    "normalize_pr_node",
    "normalize_release_node",
    "normalize_review_node",
    "run_graphql",
]
