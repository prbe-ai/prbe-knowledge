"""Notion source poller.

Drives Notion ingestion for customer workspaces that opted into the
polling path (no live webhook subscription). One ``poll`` tick:

  1. Loads the customer's Notion OAuth token from ``integration_tokens``.
  2. Hits ``POST https://api.notion.com/v1/search`` filtered to
     ``object: "page"`` and sorted by ``last_edited_time`` ascending.
     Notion's search returns ALL pages — there's no native "since X"
     filter on this endpoint — so we paginate through the page-list
     and filter client-side against our cursor.
  3. Maps each surviving result into the same webhook-shaped payload
     the Notion connector's webhook handler produces (``type: page.created``
     + ``entity: {type, id, last_edited_time, workspace_id}`` +
     ``data: <raw page>``), so the downstream ingestion + normalizer
     doesn't need to branch on origin.
  4. Returns ``PollResult(documents, next_cursor=<max last_edited_time>)``.
     On a Notion API error envelope (``{"object": "error", ...}``) the
     result carries ``error`` and the scheduler stamps it without
     advancing the cursor.

Cursor semantics — the cursor is the latest Notion ``last_edited_time``
we've already enqueued (ISO-8601 string). First poll on a fresh
``resource_id`` has ``cursor=None``; we seed it to "7 days ago" so the
initial sweep doesn't drag in the entire workspace history. Subsequent
ticks pull the strict greater-than slice.

The ``resource_id`` here is the OAuth token's workspace_id (a single row
in ``ingestion_cursors`` per customer-workspace pair). Notion's search
API returns everything the integration has access to in that workspace
in one paginated walk — we don't fan out per-page.

NOTE: this poller intentionally does NOT hydrate page bodies. The
downstream ingestion path (the connector's ``fetch_supplementary``)
already pulls block content from /v1/blocks/{id}/children when it sees
the webhook envelope, so doing it here would duplicate the work. The
poller is a thin "what's changed since X?" feeder.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from services.ingestion.polling.base import (
    BasePoller,
    PollResult,
    register_poller,
)
from shared.constants import SourceSystem
from shared.tokens import load_token

logger = logging.getLogger(__name__)

# Endpoint + version pinned to match the connector's webhook handler.
# See services/ingestion/handlers/notion.py for the production hydration
# path that consumes the documents this poller emits.
_NOTION_SEARCH_URL = "https://api.notion.com/v1/search"
_NOTION_VERSION = "2022-06-28"

# How far back the first-ever poll for a resource reaches. Notion search
# returns every page the integration can see — without a floor, a fresh
# resource would drag in years of pages and dominate the ingestion queue
# for hours. 7 days mirrors the wider "polling backfill window" we use
# elsewhere; anything older is expected to come in via a separate
# backfill job, not the live poller.
_FIRST_POLL_LOOKBACK_DAYS = 7

# Notion's max page_size for /v1/search is 100. The framework expects
# pollers to drain their cursor's worth of work in one tick (the
# scheduler will re-tick when polled_at next ages past the threshold),
# but we cap total pages per tick so a workspace with thousands of
# stale edits doesn't blow the scheduler's per-tick latency budget.
_SEARCH_PAGE_SIZE = 100
_MAX_PAGES_PER_TICK = 50

# Per-call HTTP timeout. Notion's API is typically <500ms but their
# /v1/search can be slow on very large workspaces; 30s leaves headroom
# without letting a hung connection block the scheduler indefinitely.
_HTTP_TIMEOUT_SECONDS = 30.0


class NotionPoller(BasePoller):
    """Polling-mode ingestion for Notion pages.

    Stateless — the scheduler constructs a fresh instance per tick. All
    per-customer state (auth token, cursor) is passed into ``poll`` or
    fetched from the DB inside it.
    """

    source = SourceSystem.NOTION

    async def poll(
        self,
        *,
        customer_id: str,
        resource_id: str,
        cursor: str | None,
    ) -> PollResult:
        token = await load_token(customer_id, SourceSystem.NOTION)
        if token is None:
            # No token = nothing to poll. Surface as a soft error so the
            # operator dashboard shows "needs reauth" rather than the row
            # silently advancing.
            return PollResult(
                documents=[],
                next_cursor=None,
                error="no active integration_tokens row for notion",
            )

        cursor_dt = _parse_cursor(cursor)
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT_SECONDS) as http:
            return await self._poll_with_client(
                http=http,
                access_token=token.access_token,
                customer_id=customer_id,
                resource_id=resource_id,
                cursor_dt=cursor_dt,
            )

    async def _poll_with_client(
        self,
        *,
        http: httpx.AsyncClient,
        access_token: str,
        customer_id: str,
        resource_id: str,
        cursor_dt: datetime,
    ) -> PollResult:
        """Inner poll body parameterized on a pre-built httpx client.

        Split out so tests can inject an ``httpx.MockTransport`` client
        directly without monkeypatching ``httpx.AsyncClient.__init__``.
        """
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Notion-Version": _NOTION_VERSION,
            "Content-Type": "application/json",
        }

        documents: list[dict[str, Any]] = []
        max_edited_seen = cursor_dt
        next_page_cursor: str | None = None
        pages_walked = 0

        while pages_walked < _MAX_PAGES_PER_TICK:
            body_json: dict[str, Any] = {
                "filter": {"property": "object", "value": "page"},
                "sort": {
                    "timestamp": "last_edited_time",
                    "direction": "ascending",
                },
                "page_size": _SEARCH_PAGE_SIZE,
            }
            if next_page_cursor:
                body_json["start_cursor"] = next_page_cursor

            try:
                resp = await http.post(
                    _NOTION_SEARCH_URL,
                    headers=headers,
                    json=body_json,
                )
            except httpx.HTTPError as exc:
                logger.warning(
                    "notion.poll.http_error customer_id=%s resource=%s err=%s",
                    customer_id,
                    resource_id,
                    exc,
                )
                return PollResult(
                    documents=documents,
                    next_cursor=_format_cursor(max_edited_seen) if documents else None,
                    error=f"http error: {type(exc).__name__}: {exc}",
                )

            body = _safe_json(resp)
            if body is None:
                return PollResult(
                    documents=documents,
                    next_cursor=None,
                    error=f"http {resp.status_code} non-json body",
                )

            # Notion's documented error envelope: {"object": "error",
            # "status": 4xx, "code": "...", "message": "..."}. We surface
            # the message verbatim so operators can paste it into Notion's
            # support search.
            if body.get("object") == "error":
                return PollResult(
                    documents=documents,
                    next_cursor=None,
                    error=_format_notion_error(body, resp.status_code),
                )

            # A non-error non-200 (e.g. a proxy 502) — bail with the
            # documents we did collect; cursor advances iff we found
            # something usable.
            if resp.status_code != 200:
                return PollResult(
                    documents=documents,
                    next_cursor=_format_cursor(max_edited_seen) if documents else None,
                    error=f"http {resp.status_code}: {str(body)[:300]}",
                )

            results = body.get("results") or []
            new_docs, page_max = _extract_new_documents(
                results=results,
                cursor_dt=cursor_dt,
                workspace_id=resource_id,
            )
            documents.extend(new_docs)
            if page_max is not None and page_max > max_edited_seen:
                max_edited_seen = page_max

            pages_walked += 1
            if not body.get("has_more"):
                break
            next_page_cursor = body.get("next_cursor")
            if not next_page_cursor:
                break

        if pages_walked >= _MAX_PAGES_PER_TICK and body.get("has_more"):
            # We capped out mid-walk — log so operators can see when
            # _MAX_PAGES_PER_TICK is a bottleneck. The cursor advances
            # to the max we did see; the next tick picks up from there.
            logger.info(
                "notion.poll.page_cap customer_id=%s resource=%s pages=%d",
                customer_id,
                resource_id,
                pages_walked,
            )

        next_cursor = _format_cursor(max_edited_seen) if documents else None
        return PollResult(documents=documents, next_cursor=next_cursor, error=None)


# --- helpers ---------------------------------------------------------------


def _parse_cursor(cursor: str | None) -> datetime:
    """Decode the stored cursor into a tz-aware datetime.

    ``None`` → "_FIRST_POLL_LOOKBACK_DAYS_ ago". Anything unparseable
    falls back to the same lookback (a corrupt cursor shouldn't strand
    a customer at "nothing new ever") but logs a warning.
    """
    if cursor is None:
        return datetime.now(UTC) - timedelta(days=_FIRST_POLL_LOOKBACK_DAYS)
    try:
        text = cursor.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
    except (ValueError, AttributeError):
        logger.warning("notion.poll.bad_cursor cursor=%r — resetting to lookback", cursor)
        return datetime.now(UTC) - timedelta(days=_FIRST_POLL_LOOKBACK_DAYS)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _format_cursor(dt: datetime) -> str:
    """Stringify a cursor datetime in the same ISO-8601 shape Notion
    returns (trailing ``Z``), so round-tripping through the DB matches
    the upstream payload byte-for-byte."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parse_iso8601(value: Any) -> datetime | None:
    """Best-effort ISO-8601 parse for Notion's ``last_edited_time``.

    Returns ``None`` on anything we can't decode so the caller can skip
    the page entirely (a page with no parseable timestamp can't be
    compared against the cursor)."""
    if not isinstance(value, str) or not value:
        return None
    try:
        text = value.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _safe_json(resp: httpx.Response) -> dict[str, Any] | None:
    """Decode a response body to a dict, returning ``None`` on parse
    failure (non-JSON body, list at top level, etc.). Used so a wedged
    upstream proxy returning HTML can't crash the poller."""
    try:
        body = resp.json()
    except ValueError:
        return None
    return body if isinstance(body, dict) else None


def _format_notion_error(body: dict[str, Any], http_status: int) -> str:
    """Build a single error string from Notion's error envelope.

    Keeps the upstream code + message + http status visible so a
    rotated-token 401 looks distinct from a rate-limited 429 in the
    cursor row's ``last_error`` column."""
    code = body.get("code") or "unknown_code"
    message = body.get("message") or "unknown message"
    return f"notion error http={http_status} code={code}: {message}"[:1500]


def _extract_new_documents(
    *,
    results: list[Any],
    cursor_dt: datetime,
    workspace_id: str,
) -> tuple[list[dict[str, Any]], datetime | None]:
    """Filter a /v1/search results page against the cursor and shape the
    survivors into webhook-style payloads.

    Notion's API doesn't expose a server-side "since X" filter on this
    endpoint, so we do the comparison client-side. Pages with no
    parseable ``last_edited_time`` are dropped (logged at debug elsewhere
    — they'd produce a cursor-less event the connector can't dedupe).

    Returns ``(documents, max_edited_in_page)`` so the caller can update
    the running max across the paginated walk.
    """
    documents: list[dict[str, Any]] = []
    page_max: datetime | None = None
    for result in results:
        if not isinstance(result, dict):
            continue
        if result.get("object") != "page":
            continue
        last_edited = _parse_iso8601(result.get("last_edited_time"))
        if last_edited is None:
            continue
        if last_edited <= cursor_dt:
            continue
        page_id = result.get("id")
        if not isinstance(page_id, str) or not page_id:
            continue
        documents.append(_build_webhook_payload(result, workspace_id))
        if page_max is None or last_edited > page_max:
            page_max = last_edited
    return documents, page_max


def _build_webhook_payload(
    page: dict[str, Any],
    workspace_id: str,
) -> dict[str, Any]:
    """Wrap a raw Notion search result in the webhook envelope shape the
    connector's ``parse_webhook_event`` already accepts.

    Picked ``page.created`` (rather than ``page.content_updated``) for
    the same reason the connector's backfill does: the latter requires
    a ``data.updated_blocks`` list the search endpoint doesn't give us,
    and the normalizer treats both equivalently (re-fetches the entity
    + blocks fresh).
    """
    last_edited = page.get("last_edited_time") or ""
    page_id = page.get("id") or ""
    return {
        "type": "page.created",
        "entity": {
            "type": "page",
            "id": page_id,
            "last_edited_time": last_edited,
            "workspace_id": workspace_id,
        },
        "data": page,
        "workspace_id": workspace_id,
        "_source": "poller",
    }


# Self-register on import so ``services.ingestion.polling.notion`` only
# needs to be referenced in the scheduler's startup imports for the row
# dispatch to wire up.
register_poller(SourceSystem.NOTION, NotionPoller)


__all__ = ["NotionPoller"]
