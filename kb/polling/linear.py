"""Linear source poller (PR E3).

Self-host customers don't get Linear webhooks — the cluster polls Linear's
GraphQL API outbound on a cadence. The scheduler dispatches one tick at a
time per ``(customer_id, resource_id)`` cursor row; this poller does:

  1. Look up the customer's active ``integration_tokens.access_token`` row
     for ``source_system='linear'``.
  2. GraphQL POST to ``https://api.linear.app/graphql`` for issues whose
     ``updatedAt`` is greater than the stored cursor (or > 7d ago on the
     first tick).
  3. Walk every page via ``pageInfo.endCursor`` until exhausted.
  4. Shape each issue node into the same webhook-style envelope the
     existing ``LinearConnector`` normalizes (``type=Issue, action=create``)
     so the downstream pipeline doesn't need to branch on origin.
  5. Return a ``PollResult`` whose ``next_cursor`` is the max ``updatedAt``
     seen across the run (so the next tick continues from there).

The ``resource_id`` slot is ``"*"`` for customer-wide polling — one cursor
per customer regardless of how many Linear teams they have visibility on.
That matches the existing connector's backfill, which is team-agnostic too:
the API key sees whatever the issuing user sees, and we ingest all of it.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar

import httpx

from services.ingestion.polling.base import BasePoller, PollResult, register_poller
from shared.constants import SourceSystem
from shared.db import raw_conn
from shared.encryption import decrypt_token
from shared.logging import get_logger

log = get_logger(__name__)

# Linear GraphQL endpoint. Same host the connector's OAuth + backfill use.
_LINEAR_GRAPHQL_URL = "https://api.linear.app/graphql"

# Customer-wide cursor sentinel. The scheduler keys cursor rows by
# (customer_id, source, resource_id); Linear has no per-team poll cadence
# distinction, so a single "*" row owns the whole customer.
RESOURCE_ID_WILDCARD = "*"

# How far back to look on the very first poll (no stored cursor yet).
# Matches the existing connector's "recent issues" backfill horizon and
# keeps the initial sync bounded — a fresh self-host install doesn't pull
# years of issue history before the steady-state cadence kicks in.
_FIRST_POLL_LOOKBACK = timedelta(days=7)

# Page size for the GraphQL `first` argument. 50 mirrors the connector's
# backfill page size and keeps each tick well under Linear's complexity
# budget while still draining a busy workspace in a few ticks.
_PAGE_SIZE = 50

# Safety cap on pages walked per tick. If the cursor falls so far behind
# that we'd otherwise grind through hundreds of pages, bail with the
# best-effort next_cursor so other tenants get their turn. Subsequent
# ticks resume from where we left off.
_MAX_PAGES_PER_TICK = 20

# HTTP timeout for a single GraphQL call. Linear's p95 is well under this;
# the value is loose enough to absorb transient slowness without holding
# up the scheduler.
_HTTP_TIMEOUT_SECONDS = 30.0

# The GraphQL query. Two cursors are conceptually in play:
#   * ``updatedAt`` watermark — the persisted cursor; used as a `filter`
#     so we only pull issues edited since the last successful tick.
#   * ``after`` page cursor — Linear's opaque pageInfo.endCursor, only
#     used WITHIN a tick to walk pages.
# Field set mirrors the connector's backfill so the normalizer doesn't
# care whether a payload came from webhook, backfill, or poll.
_ISSUES_QUERY = """
query PollIssues($updatedAfter: DateTimeOrDuration, $first: Int!, $after: String) {
  issues(
    filter: { updatedAt: { gt: $updatedAfter } }
    orderBy: updatedAt
    first: $first
    after: $after
  ) {
    pageInfo { hasNextPage endCursor }
    nodes {
      id
      identifier
      title
      description
      url
      createdAt
      updatedAt
      state { name type }
      priority
      team { id key name }
      creator { id name email }
      assignee { id name email }
    }
  }
}
"""


class LinearPoller(BasePoller):
    """Pull issues updated since ``cursor`` from Linear's GraphQL API.

    Cursor format: ISO-8601 ``updatedAt`` timestamp (e.g.
    ``"2026-05-14T17:30:00+00:00"``). ``None`` on the first tick — the
    poller substitutes ``now - 7d`` so we don't ingest the workspace's
    entire history at install time.
    """

    source: ClassVar[SourceSystem] = SourceSystem.LINEAR

    async def poll(
        self,
        *,
        customer_id: str,
        resource_id: str,
        cursor: str | None,
    ) -> PollResult:
        api_key = await _load_linear_api_key(customer_id)
        if api_key is None:
            # No active token row — surface as a soft error so the cursor
            # stays as-is and the scheduler doesn't busy-loop the row.
            return PollResult(
                documents=[],
                error="no active linear integration_tokens row",
            )

        updated_after = _resolve_cursor_floor(cursor)

        documents: list[dict[str, Any]] = []
        max_updated_at: datetime | None = _parse_iso(cursor)
        page_after: str | None = None

        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT_SECONDS) as http:
            for _page_index in range(_MAX_PAGES_PER_TICK):
                page_result = await _fetch_page(
                    http,
                    api_key=api_key,
                    updated_after=updated_after,
                    after=page_after,
                )
                if page_result.error is not None:
                    # Hard failure — return whatever we already accumulated
                    # PLUS the error. The scheduler stamps the error onto
                    # the cursor row without advancing the watermark.
                    return PollResult(
                        documents=documents,
                        next_cursor=_iso_or_none(max_updated_at),
                        error=page_result.error,
                    )

                for node in page_result.nodes:
                    doc = _wrap_issue_as_webhook(node)
                    if doc is not None:
                        documents.append(doc)
                    node_updated_at = _parse_iso(node.get("updatedAt"))
                    if node_updated_at is not None and (
                        max_updated_at is None or node_updated_at > max_updated_at
                    ):
                        max_updated_at = node_updated_at

                if not page_result.has_next_page or not page_result.end_cursor:
                    break
                page_after = page_result.end_cursor
            else:
                log.warning(
                    "linear_poller.page_cap_reached",
                    customer_id=customer_id,
                    pages=_MAX_PAGES_PER_TICK,
                    documents=len(documents),
                )

        return PollResult(
            documents=documents,
            next_cursor=_iso_or_none(max_updated_at),
            error=None,
        )


register_poller(SourceSystem.LINEAR, LinearPoller)


# --- internals ---------------------------------------------------------------


class _PageResult:
    """One GraphQL page worth of nodes + paging metadata + soft error."""

    __slots__ = ("end_cursor", "error", "has_next_page", "nodes")

    def __init__(
        self,
        *,
        nodes: list[dict[str, Any]],
        has_next_page: bool,
        end_cursor: str | None,
        error: str | None,
    ) -> None:
        self.nodes = nodes
        self.has_next_page = has_next_page
        self.end_cursor = end_cursor
        self.error = error


async def _fetch_page(
    http: httpx.AsyncClient,
    *,
    api_key: str,
    updated_after: str,
    after: str | None,
) -> _PageResult:
    """One GraphQL POST. Returns either nodes+paging or a soft error string.

    Soft errors (non-200, ``errors[].message`` set, network blow-up) come
    back as ``_PageResult(error=...)`` so the caller can stamp the cursor
    row and move on instead of crashing the tick.
    """
    variables: dict[str, Any] = {"updatedAfter": updated_after, "first": _PAGE_SIZE}
    if after:
        variables["after"] = after

    try:
        resp = await http.post(
            _LINEAR_GRAPHQL_URL,
            headers={
                # Linear personal API keys are passed verbatim in the
                # Authorization header (no Bearer prefix). OAuth tokens
                # use Bearer; the connector stores both as the same
                # `access_token` value, so the caller's responsibility is
                # to store the right shape.
                "Authorization": api_key,
                "Content-Type": "application/json",
            },
            json={"query": _ISSUES_QUERY, "variables": variables},
        )
    except httpx.HTTPError as exc:
        return _PageResult(
            nodes=[],
            has_next_page=False,
            end_cursor=None,
            error=f"linear graphql http error: {exc}",
        )

    if resp.status_code != 200:
        return _PageResult(
            nodes=[],
            has_next_page=False,
            end_cursor=None,
            error=(
                f"linear graphql non-200 status={resp.status_code} "
                f"body={resp.text[:500]}"
            ),
        )

    try:
        body = resp.json()
    except ValueError as exc:
        return _PageResult(
            nodes=[],
            has_next_page=False,
            end_cursor=None,
            error=f"linear graphql malformed json: {exc}",
        )

    # Linear returns 200 with a non-empty `errors` array on GraphQL
    # validation failures (unknown field, complexity blown). Without this
    # branch the loop would silently see no `data.issues` and the tick
    # would mark the cursor advanced with 0 events enqueued — exactly the
    # failure mode that masked the connector's bad query for two installs
    # (see feedback in services/ingestion/handlers/linear.py).
    errors = body.get("errors") or []
    if errors:
        messages = [str(e.get("message") or e) for e in errors][:3]
        return _PageResult(
            nodes=[],
            has_next_page=False,
            end_cursor=None,
            error=f"linear graphql errors: {'; '.join(messages)}",
        )

    issues = ((body.get("data") or {}).get("issues") or {})
    nodes = issues.get("nodes") or []
    page_info = issues.get("pageInfo") or {}
    return _PageResult(
        nodes=list(nodes),
        has_next_page=bool(page_info.get("hasNextPage")),
        end_cursor=page_info.get("endCursor"),
        error=None,
    )


def _wrap_issue_as_webhook(node: dict[str, Any]) -> dict[str, Any] | None:
    """Shape one GraphQL issue node into the webhook envelope the existing
    ``LinearConnector.parse_webhook_event`` normalizes.

    The connector keys its dedupe on ``source_event_id = type:id:action:clock:hash``
    and the parse step requires ``data.id`` + a ``createdAt`` clock. Anything
    missing those is dropped here so the queue never sees a degenerate row.
    """
    issue_id = node.get("id")
    if not issue_id:
        return None
    # We don't know the org id from the issue node alone in this query
    # (the connector's backfill fetches it separately via the `viewer`
    # query). The normalizer accepts empty `organizationId` and still
    # builds a valid doc_id from `linear::issue:<id>`. A follow-up could
    # extend the GraphQL query with `... team { organization { id } }`
    # to populate it; leaving it empty here keeps PR E3 strictly limited
    # to the polling-cursor concern.
    return {
        "type": "Issue",
        "action": "create",
        "data": node,
        "organizationId": "",
        # Stable clock for the connector's source_event_id. Falls back to
        # createdAt if updatedAt is missing; both shouldn't be missing on
        # a live Linear node.
        "createdAt": node.get("updatedAt") or node.get("createdAt"),
        # Tag the envelope so downstream traces can tell poll-vs-webhook
        # ingest apart without parsing.
        "_origin": "poll",
    }


def _resolve_cursor_floor(cursor: str | None) -> str:
    """The ISO timestamp passed to GraphQL's ``filter.updatedAt.gt``.

    On the first tick (``cursor is None``) we substitute ``now - 7d`` so a
    fresh install doesn't drag the workspace's entire history through the
    queue. Subsequent ticks pass the stored cursor verbatim.
    """
    parsed = _parse_iso(cursor)
    if parsed is None:
        parsed = datetime.now(UTC) - _FIRST_POLL_LOOKBACK
    return parsed.isoformat()


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def _iso_or_none(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt is not None else None


async def _load_linear_api_key(customer_id: str) -> str | None:
    """Active Linear ``integration_tokens.access_token`` for this customer.

    Returns ``None`` if no row matches — caller surfaces that as a soft
    error so the cursor row sticks around (a token may be re-installed
    later) but doesn't burn the scheduler tick.
    """
    async with raw_conn() as conn:
        row = await conn.fetchrow(
            """
            SELECT access_token_encrypted
              FROM integration_tokens
             WHERE customer_id = $1
               AND source_system = $2
               AND status = 'active'
             ORDER BY token_id DESC
             LIMIT 1
            """,
            customer_id,
            SourceSystem.LINEAR.value,
        )
    if row is None:
        return None
    return decrypt_token(row["access_token_encrypted"])
