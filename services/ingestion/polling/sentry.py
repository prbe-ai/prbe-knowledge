"""Sentry source poller (PR E5).

Per-tenant polling of Sentry's project-events endpoint for self-host
customers, where webhooks aren't reachable from inside the customer's
cluster. One ``SentryPoller`` instance is constructed per tick by the
scheduler; it reads the OAuth/auth token from ``integration_tokens``,
calls Sentry's REST API, and emits webhook-shaped document dicts that
match what the inbound ``SentryConnector`` webhook handler produces.

Endpoint contract
-----------------

``GET https://sentry.io/api/0/projects/{org_slug}/{project_slug}/events/``

  * First poll for a fresh resource: ``?statsPeriod=7d`` (one-week
    backfill window). We do NOT page through historical data here —
    deep backfill is the connector's ``backfill`` path, which is run
    separately during initial install. The poller's job is incremental
    forward-only catch-up.
  * Subsequent polls: ``?cursor=<cursor>`` where ``<cursor>`` came from
    the previous tick's ``PollResult.next_cursor``. Sentry uses Link-
    header cursor pagination — the existing webhook handler already
    has parsers for the ``rel="next"`` URL and its ``cursor=`` query
    param; we reuse those.

Cursor semantics
----------------

We persist Sentry's own cursor token rather than a ``dateCreated``
timestamp. Sentry's cursor encodes (timestamp, offset, direction) and
is the only durable handle that survives across page boundaries when
events share a millisecond. Using the cursor directly also keeps us
robust against clock skew between Sentry and the polling pod.

``resource_id`` shape: ``<org_slug>/<project_slug>``. Split with a
single ``/`` — Sentry slugs are URL-safe and never contain slashes
themselves.

Error handling
--------------

Soft failures (429, 5xx, transport errors) surface as
``PollResult(error=...)`` so the scheduler stamps the cursor row and
walks on. The cursor is NOT advanced on error — the next tick retries
from the same point.

Hard failures (4xx other than 429, missing token, missing token after
load) also return an error string; the polling-pod operator triages
via the ``ingestion_cursors.last_error`` field.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any, ClassVar

import httpx

from services.ingestion.polling.base import BasePoller, PollResult, register_poller
from shared.constants import SourceSystem
from shared.logging import get_logger
from shared.tokens import load_token

log = get_logger(__name__)


# --- Sentry endpoint + tuning constants ------------------------------------

_SENTRY_API_BASE = "https://sentry.io/api/0"

# Backfill window when a resource has no cursor yet. Matches the
# documented poller spec (PR E5): "first poll → statsPeriod=7d".
_FIRST_POLL_STATS_PERIOD = "7d"

# HTTP timeout per request. Sentry's events endpoint typically responds
# in <1s on small projects; 30s is generous headroom for noisy projects.
_HTTP_TIMEOUT_SECONDS = 30.0

# Sentry resource_id format: "<org_slug>/<project_slug>".
_RESOURCE_ID_RE = re.compile(r"^([^/]+)/([^/]+)$")

# Webhook-resource value the inbound handler keys off of for event
# payloads. Sentry's hook header would normally carry "event_alert"
# (or "error"); the poller emits "event_alert" so the normalizer's
# event path runs.
_WEBHOOK_RESOURCE_EVENT = "event_alert"


class SentryPoller(BasePoller):
    """Per-tenant Sentry events poller.

    One instance per scheduler tick (the scheduler does ``SentryPoller()``
    on every tick). State (auth token, HTTP client) is built fresh inside
    ``poll`` so concurrent ticks on different tenants never share a
    connection or a credential.
    """

    source: ClassVar[SourceSystem] = SourceSystem.SENTRY

    # Override for tests: pass a pre-built ``httpx.AsyncClient`` (with a
    # ``MockTransport``) to bypass live HTTP. Production leaves this None
    # and the poller builds a fresh client per tick.
    _http_client_factory: ClassVar[Any] = None

    async def poll(
        self,
        *,
        customer_id: str,
        resource_id: str,
        cursor: str | None,
    ) -> PollResult:
        org_slug, project_slug = _parse_resource_id(resource_id)
        if org_slug is None or project_slug is None:
            return PollResult(
                documents=[],
                error=f"invalid sentry resource_id: {resource_id!r} "
                "(expected '<org_slug>/<project_slug>')",
            )

        token = await load_token(customer_id, SourceSystem.SENTRY)
        if token is None or not token.access_token:
            return PollResult(
                documents=[],
                error="no active sentry integration token for tenant",
            )

        url, params = _build_request(org_slug, project_slug, cursor)
        auth_headers = {"Authorization": f"Bearer {token.access_token}"}

        client = self._build_http_client()
        try:
            try:
                resp = await client.get(url, params=params, headers=auth_headers)
            except httpx.HTTPError as exc:
                return PollResult(
                    documents=[],
                    error=f"sentry transport error: {type(exc).__name__}: {exc}",
                )

            if resp.status_code == 429:
                return PollResult(
                    documents=[], error=f"sentry rate-limited (429): {resp.text[:200]}"
                )
            if resp.status_code >= 500:
                return PollResult(
                    documents=[],
                    error=f"sentry {resp.status_code}: {resp.text[:200]}",
                )
            if resp.status_code >= 400:
                # 4xx other than 429 — auth / project-missing / etc.
                # Treat as a soft error so the cursor row is stamped and
                # an operator can triage; we don't want a permanently
                # broken resource to block the scheduler.
                return PollResult(
                    documents=[],
                    error=f"sentry {resp.status_code}: {resp.text[:200]}",
                )

            try:
                events = resp.json()
            except ValueError as exc:
                return PollResult(
                    documents=[],
                    error=f"sentry returned non-JSON body: {exc}",
                )
            if not isinstance(events, list):
                return PollResult(
                    documents=[],
                    error=f"sentry returned unexpected body type: {type(events).__name__}",
                )

            link_header = resp.headers.get("link") or ""
        finally:
            # Only close the client if we built it ourselves; tests inject
            # a client they manage.
            if self._http_client_factory is None:
                await client.aclose()

        next_cursor = _parse_next_cursor(link_header)
        documents = [
            _event_to_webhook_doc(ev, org_slug, project_slug)
            for ev in events
            if isinstance(ev, dict)
        ]
        # Drop any None entries from events we couldn't shape.
        documents = [d for d in documents if d is not None]

        return PollResult(documents=documents, next_cursor=next_cursor, error=None)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _build_http_client(self) -> httpx.AsyncClient:
        factory = type(self)._http_client_factory
        if factory is not None:
            return factory()
        return httpx.AsyncClient(timeout=_HTTP_TIMEOUT_SECONDS)


# --- helpers ---------------------------------------------------------------


def _parse_resource_id(resource_id: str) -> tuple[str | None, str | None]:
    """Split ``<org_slug>/<project_slug>`` into its two halves.

    Returns ``(None, None)`` if the shape doesn't match — the caller
    surfaces this as a soft error on the cursor row.
    """
    if not isinstance(resource_id, str) or not resource_id:
        return None, None
    m = _RESOURCE_ID_RE.match(resource_id)
    if m is None:
        return None, None
    return m.group(1), m.group(2)


def _build_request(
    org_slug: str, project_slug: str, cursor: str | None
) -> tuple[str, dict[str, str]]:
    """Compose the events-endpoint URL + query params for one tick.

    First poll → statsPeriod=7d (the backfill window). Subsequent polls
    → pass the cursor straight through to Sentry; ``statsPeriod`` is
    omitted so we follow the cursor's own time anchor.
    """
    url = f"{_SENTRY_API_BASE}/projects/{org_slug}/{project_slug}/events/"
    params: dict[str, str] = {}
    if cursor:
        params["cursor"] = cursor
    else:
        params["statsPeriod"] = _FIRST_POLL_STATS_PERIOD
    return url, params


def _parse_next_link(link_header: str) -> str | None:
    """Pull the URL out of a Link header's ``rel="next"`` slot.

    Sentry's Link header looks like::

      <https://sentry.io/...?cursor=abc>; rel="previous"; results="false"; cursor="0:0:1",
      <https://sentry.io/...?cursor=def>; rel="next";     results="true";  cursor="0:100:0"

    Returns ``None`` when there's no next page (``results="false"`` on
    the next slot, or no next slot at all).
    """
    if not link_header:
        return None
    for part in link_header.split(","):
        part = part.strip()
        if 'rel="next"' not in part:
            continue
        if 'results="false"' in part:
            return None
        if part.startswith("<") and ">" in part:
            return part.split(">", 1)[0][1:]
    return None


def _parse_next_cursor(link_header: str) -> str | None:
    """Extract the cursor= token from the next-rel link, if any.

    The cursor in the URL takes precedence over the duplicated
    ``cursor="..."`` link-header attribute — both should match, but the
    URL is the one Sentry's pagination contract is documented against.
    """
    url = _parse_next_link(link_header)
    if not url or "cursor=" not in url:
        return None
    return url.split("cursor=", 1)[1].split("&", 1)[0]


def _event_to_webhook_doc(
    event: dict[str, Any],
    org_slug: str,
    project_slug: str,
) -> dict[str, Any] | None:
    """Reshape a Sentry events-endpoint row into the webhook payload
    shape ``SentryConnector.parse_webhook_event`` consumes.

    Sentry's project events endpoint returns rows like::

      {"id": "abc123", "eventID": "abc123",
       "groupID": "9999", "dateCreated": "2026-05-14T...",
       "title": "...", "culprit": "...",
       "tags": [...], "entries": [...]}

    The inbound webhook for ``event_alert`` carries::

      {"action": "triggered",
       "data": {"event": {...}},
       "project": {"slug": "..."},
       "organization": {"slug": "..."}}

    We rebuild that shape so the downstream normalizer doesn't need a
    second code path. Events without an ``event_id`` AND no ``id`` are
    dropped — there's nothing to anchor a document on.
    """
    event_id = event.get("event_id") or event.get("eventID") or event.get("id")
    if not event_id:
        return None

    timestamp = (
        event.get("timestamp")
        or event.get("dateCreated")
        or event.get("dateReceived")
    )
    if not timestamp:
        # Stamp with poll-time so the doc still has a valid_from.
        timestamp = datetime.now(UTC).isoformat()

    # The webhook handler reads ``event.event_id`` (not the Sentry-side
    # snake-case variant). Mirror it here so the parse path lights up
    # without a special poller branch on the handler side.
    event_payload: dict[str, Any] = dict(event)
    event_payload.setdefault("event_id", event_id)
    event_payload.setdefault("timestamp", timestamp)
    # Carry the group anchor under both the camelCase and snake_case
    # spellings — different Sentry surfaces use different conventions.
    if "groupID" in event and "group_id" not in event_payload:
        event_payload["group_id"] = event["groupID"]

    return {
        "action": "triggered",
        "data": {"event": event_payload},
        "project": {"slug": project_slug},
        "organization": {"slug": org_slug},
        # Mark the payload as poller-sourced for downstream observability.
        # The webhook handler ignores unknown top-level keys.
        "_poller_source": "sentry",
    }


# Register at import time so the scheduler's get_poller() resolves us.
register_poller(SourceSystem.SENTRY, SentryPoller)


__all__ = ["SentryPoller"]
