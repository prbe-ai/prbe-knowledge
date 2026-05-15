"""Slack source poller (Phase 2 PR E2).

Pulls new messages from a single Slack channel since the previous tick's
cursor and emits webhook-shaped event dicts the existing Slack connector
already knows how to normalize.

Wiring per the framework contract:

  * ``source`` = ``SourceSystem.SLACK``
  * ``resource_id`` = a Slack channel id (e.g. ``"C0123456"``). One
    ingestion_cursors row per (customer, channel).
  * ``cursor`` = the most-recent message ``ts`` we've already ingested
    for this channel (Slack's monotonically-increasing
    ``"<seconds>.<microseconds>"`` string). The next ``conversations.history``
    call passes it as ``oldest`` with ``inclusive=false`` so we only pull
    strictly-newer messages.

First-poll cold start (``cursor is None``): pull the last 7 days from
the channel. The framework's seeding path (added later) will create an
ingestion_cursors row when a customer connects Slack; this poller
gracefully handles the case where that row exists with ``cursor_value
IS NULL``.

Document shape: each item in ``PollResult.documents`` mirrors what an
inbound Slack ``event_callback`` webhook would deliver, so the
existing ``SlackConnector.parse_webhook_event`` / ``normalize`` pipeline
consumes them without branching on origin. Concretely::

    {
        "team_id": "<workspace team id>",
        "type": "event_callback",
        "event": {
            "type": "message",
            "channel": "<resource_id>",
            "ts": "...",
            "text": "...",
            "user": "...",
            ...
        },
    }

Error handling: any Slack response with ``ok: false`` (including
``ratelimited``) is returned as ``PollResult.error`` so the scheduler
stamps the cursor row and moves on. The cursor itself is NOT advanced
on error — the next tick retries from the same ``oldest`` value, which
is idempotent because we filter by strict ``> cursor`` semantics.

Tests inject an ``httpx.MockTransport`` via ``SlackPoller.http_client_factory``
so the live token-decryption path is bypassed (see ``tests/test_polling_slack.py``).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar

import httpx

from services.ingestion.polling.base import BasePoller, PollResult, register_poller
from shared.constants import SourceSystem
from shared.logging import get_logger
from shared.models import IntegrationToken
from shared.tokens import load_token

log = get_logger(__name__)


# Slack API base. Lives in module scope so tests' MockTransport handlers
# can match against the same URL string the poller constructs.
SLACK_API = "https://slack.com/api"

# conversations.history page size. Slack documents a max of 1000; 200
# matches the production backfill walker (services/ingestion/handlers/slack.py)
# and keeps per-page parsing time bounded.
HISTORY_PAGE_LIMIT = 200

# Safety cap on pagination depth per tick. A polling tick should NOT
# attempt to drain a 7-day backlog in one shot — that's what backfill
# is for. If a channel returns more than ``MAX_PAGES_PER_TICK`` pages
# of new messages, we ingest what we have, advance the cursor, and let
# the next tick continue. With ``HISTORY_PAGE_LIMIT=200`` that's up to
# 2000 messages per tick per channel — plenty for normal traffic.
MAX_PAGES_PER_TICK = 10

# First-poll window: how far back to look on the first tick for a
# channel. 7 days matches the spec; longer histories are the backfill
# walker's job.
FIRST_POLL_WINDOW = timedelta(days=7)

# HTTP request timeout. Slack's published p99 for conversations.history
# is under 5s; 30s gives generous headroom without letting a hung
# request stall the scheduler.
HTTP_TIMEOUT_SECONDS = 30.0


# Type alias for the http-client factory hook. The poller defaults to
# ``httpx.AsyncClient`` (with the standard timeout); tests override
# this class attribute to inject an ``httpx.MockTransport`` without
# threading the client through poll()'s signature.
HttpClientFactory = Callable[[], httpx.AsyncClient]


def _default_http_factory() -> httpx.AsyncClient:
    """Production http client. Module-level so tests can monkeypatch
    cleanly without inheriting from the poller class."""
    return httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS)


def _seven_days_ago_ts() -> str:
    """Slack ``oldest`` for the first poll. Slack accepts a Unix-epoch
    string like ``"1704067200.000000"``; we serialize with microsecond
    precision so it sorts naturally against the per-message ``ts``."""
    cutoff = datetime.now(UTC) - FIRST_POLL_WINDOW
    return f"{cutoff.timestamp():.6f}"


def _max_ts(messages: list[dict[str, Any]]) -> str | None:
    """Largest ``ts`` in the batch, used as the next cursor.

    Slack returns messages newest-first per page, so the first message
    on the first page is the max; but pagination + multi-page merging
    means we just scan defensively. Comparing as strings is safe —
    ``ts`` is a fixed-width zero-padded decimal in practice, but we
    coerce to float for the comparison to be robust against any future
    width changes."""
    best: str | None = None
    best_f: float = -1.0
    for msg in messages:
        ts = msg.get("ts")
        if not isinstance(ts, str):
            continue
        try:
            t = float(ts)
        except ValueError:
            continue
        if t > best_f:
            best_f = t
            best = ts
    return best


class SlackPoller(BasePoller):
    """Per-channel Slack poller.

    Stateless across ticks beyond what's persisted in
    ``ingestion_cursors`` — each ``poll()`` call opens a fresh
    ``httpx.AsyncClient``, loads the customer's bot token, and either
    walks 7 days of history (cold start) or fetches everything strictly
    newer than the stored cursor.
    """

    source: ClassVar[SourceSystem] = SourceSystem.SLACK

    # Tests override this to inject a MockTransport-backed client. Keep
    # it a class attribute (not an __init__ arg) so the scheduler's
    # ``PollerCls()`` zero-arg construction continues to work.
    http_client_factory: ClassVar[HttpClientFactory] = staticmethod(_default_http_factory)

    async def poll(
        self,
        *,
        customer_id: str,
        resource_id: str,
        cursor: str | None,
    ) -> PollResult:
        token = await load_token(customer_id, SourceSystem.SLACK)
        if token is None:
            # No active token — surface as a soft error so the scheduler
            # stamps it. The cursor is preserved; once the customer
            # re-authorizes Slack the next tick picks up where this one
            # left off (or, if cursor is None, falls back to the 7-day
            # window).
            return PollResult(documents=[], error="missing_active_token")

        oldest = cursor if cursor else _seven_days_ago_ts()

        async with type(self).http_client_factory() as client:
            return await self._poll_channel(
                client=client,
                token=token,
                customer_id=customer_id,
                channel_id=resource_id,
                oldest=oldest,
            )

    async def _poll_channel(
        self,
        *,
        client: httpx.AsyncClient,
        token: IntegrationToken,
        customer_id: str,
        channel_id: str,
        oldest: str,
    ) -> PollResult:
        """Pull messages from a single channel, paginating up to
        ``MAX_PAGES_PER_TICK`` times. Returns the merged
        ``PollResult`` for the tick."""
        team_id = await _resolve_team_id(client, token.access_token)
        if team_id is None:
            # auth.test failed — surface a generic error. We do NOT
            # advance the cursor; next tick retries.
            return PollResult(documents=[], error="auth_test_failed")

        all_messages: list[dict[str, Any]] = []
        page_cursor: str | None = None
        for _ in range(MAX_PAGES_PER_TICK):
            body: dict[str, Any] = {
                "channel": channel_id,
                "oldest": oldest,
                "limit": HISTORY_PAGE_LIMIT,
                "inclusive": False,
            }
            if page_cursor:
                body["cursor"] = page_cursor

            try:
                resp = await client.post(
                    f"{SLACK_API}/conversations.history",
                    json=body,
                    headers={
                        "Authorization": f"Bearer {token.access_token}",
                        "Content-Type": "application/json; charset=utf-8",
                    },
                )
            except httpx.HTTPError as exc:
                log.warning(
                    "slack.poller.http_error",
                    customer=customer_id,
                    channel=channel_id,
                    error=type(exc).__name__,
                )
                return PollResult(
                    documents=[],
                    error=f"http_error: {type(exc).__name__}",
                )

            if resp.status_code != 200:
                log.warning(
                    "slack.poller.non_200",
                    customer=customer_id,
                    channel=channel_id,
                    status=resp.status_code,
                )
                return PollResult(
                    documents=[],
                    error=f"http_{resp.status_code}",
                )

            payload = resp.json()
            if not payload.get("ok"):
                err = payload.get("error") or "unknown_error"
                log.info(
                    "slack.poller.api_error",
                    customer=customer_id,
                    channel=channel_id,
                    error=err,
                )
                return PollResult(documents=[], error=str(err))

            page_messages = payload.get("messages") or []
            # Filter to real messages with content — skip joins / topic
            # changes / empty bot messages, mirroring the backfill walker.
            for msg in page_messages:
                if msg.get("type") != "message":
                    continue
                if not msg.get("text") and not msg.get("files"):
                    continue
                all_messages.append(msg)

            next_cursor = (payload.get("response_metadata") or {}).get("next_cursor")
            if not next_cursor:
                break
            page_cursor = next_cursor
        else:
            log.info(
                "slack.poller.page_cap_reached",
                customer=customer_id,
                channel=channel_id,
                pages=MAX_PAGES_PER_TICK,
            )

        if not all_messages:
            return PollResult(documents=[], next_cursor=None)

        documents = [
            _to_webhook_payload(team_id=team_id, channel_id=channel_id, message=msg)
            for msg in all_messages
        ]
        next_cursor = _max_ts(all_messages)
        return PollResult(documents=documents, next_cursor=next_cursor)


def _to_webhook_payload(
    *,
    team_id: str,
    channel_id: str,
    message: dict[str, Any],
) -> dict[str, Any]:
    """Wrap a raw conversations.history message in the same
    ``event_callback`` envelope an inbound Slack webhook would carry.

    The poller does NOT set ``user_profile`` — the normalizer's
    fetch_supplementary path resolves display names from the shared
    per-workspace cache, so we don't pay an extra users.info call per
    polled message. Cache misses degrade gracefully (chunks ship without
    the "<name>: " prefix on first sight, get the prefix on the next
    re-upsert after the name resolves)."""
    event: dict[str, Any] = {
        **message,
        "type": "message",
        "channel": channel_id,
    }
    return {
        "team_id": team_id,
        "type": "event_callback",
        "event": event,
    }


async def _resolve_team_id(client: httpx.AsyncClient, access_token: str) -> str | None:
    """One auth.test per tick. Slack's auth.test is a tier-4 method
    (~100/min) so the per-channel overhead is negligible. Returns None
    on any non-ok response; the caller surfaces that as an error."""
    try:
        resp = await client.post(
            f"{SLACK_API}/auth.test",
            headers={"Authorization": f"Bearer {access_token}"},
        )
    except httpx.HTTPError as exc:
        log.warning("slack.poller.auth_test_http_error", error=type(exc).__name__)
        return None
    if resp.status_code != 200:
        return None
    body = resp.json()
    if not body.get("ok"):
        return None
    team_id = body.get("team_id")
    return team_id if isinstance(team_id, str) and team_id else None


# Register with the scheduler at module import. The framework's
# idempotent-same-class guard means re-imports are safe.
register_poller(SourceSystem.SLACK, SlackPoller)
