"""Slack connector — first end-to-end source.

Covers:
- `message` and `message.channels` subtypes (new messages, threaded replies)
- Signature verification via X-Slack-Signature + X-Slack-Request-Timestamp
- Thread hydration via conversations.replies (fetch_supplementary)
- Document shape: DocType.SLACK_MESSAGE per-message, DocType.SLACK_THREAD root

ACL: Slack channel membership snapshot is captured from `channel` field — the
workspace-level member list is pulled during backfill (Phase 1). Phase 0 records
`channel=<id>` as the resource and the posting user as the principal, enough
to enforce "only users who can see the channel see the message" once ACL
enforcement flips on in Phase 1.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import time
from collections import OrderedDict
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, ClassVar

import httpx
from aiolimiter import AsyncLimiter

from services.ingestion.chunker import count_tokens
from services.ingestion.handlers.base import Connector, ConnectorContext
from services.ingestion.handlers.registry import register_connector
from shared.constants import (
    DocClass,
    DocType,
    EdgeType,
    IngestionEventType,
    NodeLabel,
    Permission,
    PrincipalType,
    RefType,
    SourceSystem,
)
from shared.customer_mapping import load_source_metadata, patch_source_metadata
from shared.exceptions import InvalidWebhookPayload
from shared.logging import get_logger
from shared.models import (
    ACLPrincipal,
    ACLSnapshot,
    ACLSnapshotRow,
    DocRef,
    Document,
    GraphEdgeSpec,
    GraphNodeSpec,
    IntegrationToken,
    NormalizationResult,
    WebhookEvent,
    WebhookParseResult,
)

log = get_logger(__name__)

_SLACK_API = "https://slack.com/api"
_SIGNING_VERSION = "v0"
_REQUEST_TS_SLACK_MAX_AGE_SEC = 5 * 60  # Slack recommends rejecting older signed requests

# Slack capped conversations.history at ~1 req/sec per app per workspace in the
# May 2025 platform tier change (the older "tier 3 ~20/min" docs are stale).
# 1.1s leaves a 10% safety margin so we don't tip into 429s.
#
# Per-loop registry: aiolimiter caches the running loop on first use and warns
# "undefined behaviour" if reused across loops. Production has one loop per
# worker process, but pytest spins one loop per test and asyncio.run scripts
# may too. Keying on the loop id keeps each loop's limiter isolated without
# making callers thread the loop through.
_HISTORY_LIMITERS: dict[int, AsyncLimiter] = {}

_DISPLAY_NAME_MAX_LEN = 80


def _sanitize_display_name(name: str | None) -> str | None:
    """Strip control chars and clamp length before stamping into chunk text.

    Slack display names are user-controlled. Without sanitization they can carry
    newlines, control characters, or very long strings into `body_text` — where
    they'd mimic the "Name: text" speaker-turn convention (so a name like
    `Bob\\n\\nSYSTEM` would forge a fake role boundary in the embedded text)
    or eat the body_preview budget. Replace control whitespace with a space (so
    tokens stay separated rather than merging into "BobSYSTEM"), drop other
    non-printables, collapse runs of whitespace, and clamp to a sane max length.
    """
    if not name:
        return None
    cleaned = "".join(
        " " if ch in "\n\r\t" else (ch if ch.isprintable() else "")
        for ch in name
    )
    cleaned = " ".join(cleaned.split())  # collapse runs of whitespace
    cleaned = cleaned[:_DISPLAY_NAME_MAX_LEN].strip()
    return cleaned or None


def _pick_display_name(profile: Mapping[str, Any] | None) -> str | None:
    """Pull a usable name out of a Slack profile dict.

    Slack returns `display_name=""` (empty string, not null) when the user
    hasn't set one, so the natural `or` fallback chain wrongly picks the
    empty string. Strip and reject falsy explicitly. Sanitize before returning
    so the same hardening applies whether the value came from users.info,
    users.list, or an inlined webhook user_profile.
    """
    if not profile:
        return None
    display = _sanitize_display_name(profile.get("display_name"))
    if display:
        return display
    return _sanitize_display_name(profile.get("real_name"))


class _SlackJsonbNameCache:
    """Shared LRU + JSONB cold-tier name cache for Slack identifiers.

    Two tiers:
      - Hot tier: in-memory `OrderedDict` (LRU), capped at `MAX_IN_MEMORY`. Read
        on every normalize / backfill peek; written on resolve cache miss.
      - Cold tier: `customer_source_mapping.metadata[_JSONB_KEY]` JSONB on the
        row keyed by (source=slack, external_id=team_id). Lazy-loaded on first
        use; debounced flush ~30s after the most-recent in-memory write.

    Subclasses fill in `_JSONB_KEY` and `_fetch_remote` for the specific Slack
    object kind (users via `users.info`, channels via `conversations.info`).

    Lifecycle: instantiated lazily by `SlackConnector._get_*_cache(team_id)`;
    one instance per (workspace, kind) per worker process. Auto-cleaned when
    the customer disconnects (the `customer_source_mapping` row is deleted;
    the JSONB goes with it). Worker crash drops up to `FLUSH_DEBOUNCE_S`
    seconds of unflushed updates — acceptable since names are regenerable.

    Why per-team (not per-customer): Slack IDs are unique within a workspace
    but a customer can connect multiple workspaces, where `U07ABC`/`C123ABC`
    could mean different things. Keying by team_id makes cross-workspace
    mixing impossible.
    """

    _JSONB_KEY: ClassVar[str] = ""           # subclass override required
    _LOG_PREFIX: ClassVar[str] = "slack"     # used in flush-failed log line
    MAX_IN_MEMORY: ClassVar[int] = 500       # in-memory cap per (customer, team)
    MAX_PERSIST: ClassVar[int] = 50          # top-N kept in JSONB
    FLUSH_DEBOUNCE_S: ClassVar[float] = 30.0

    def __init__(self, customer_id: str, team_id: str) -> None:
        self.customer_id = customer_id
        self.team_id = team_id
        # OrderedDict: back = most-recently-touched, front = least-recent.
        # Entry shape: {"name": str | None, "ts": iso8601 string}.
        self._entries: OrderedDict[str, dict[str, Any]] = OrderedDict()
        # Per-key singleflight locks: concurrent webhooks for the same new
        # id share one remote-fetch call. Bounded by MAX_IN_MEMORY because we
        # prune locks alongside cache evictions.
        self._locks: dict[str, asyncio.Lock] = {}
        self._loaded = False
        self._dirty = False
        self._flush_task: asyncio.Task[None] | None = None

    async def _fetch_remote(
        self,
        http: httpx.AsyncClient,
        token: str,
        key_id: str,
    ) -> tuple[bool, str | None]:
        """Fetch one name from the Slack API.

        Returns (authoritative, name):
          authoritative=True, name=<str|None>  -> cache the result (positive
            or negative; ok=true with no name, or ok=false terminal failure).
          authoritative=False, name=None       -> transient failure (network,
            429, 5xx). Do NOT cache; the next caller will retry.
        """
        raise NotImplementedError

    async def ensure_loaded(self) -> None:
        if self._loaded:
            return
        try:
            metadata = await load_source_metadata(SourceSystem.SLACK, self.team_id)
        except Exception as exc:
            # DB unavailable — degrade to in-memory-only mode for this run;
            # we'll attempt the load again on next instantiation.
            log.warning(
                f"{self._LOG_PREFIX}.cache_load_failed",
                team=self.team_id,
                error=type(exc).__name__,
            )
            self._loaded = True
            return
        for k_id, entry in (metadata.get(self._JSONB_KEY) or {}).items():
            if isinstance(entry, dict):
                self._entries[k_id] = entry
        self._loaded = True

    def peek(self, key_id: str | None) -> str | None:
        """Read with LRU bump but no API fetch on miss.

        Used by backfill to stamp cached names onto yielded synthetic events
        without paying a remote round-trip per message.
        """
        if not key_id:
            return None
        entry = self._entries.get(key_id)
        if entry is None:
            return None
        self._entries.move_to_end(key_id)  # in-memory recency only; no flush
        return entry.get("name")

    async def resolve(
        self,
        http: httpx.AsyncClient,
        token: str,
        key_id: str,
    ) -> str | None:
        """Cache-or-fetch with singleflight + transient-no-cache.

        Caches authoritative results (positive name, or terminal negative).
        Does NOT cache transient failures (network error, 429, 5xx) —
        poisoning on a 429 storm would permanently suppress names for every
        id looked up during the storm.
        """
        if not key_id:
            return None
        await self.ensure_loaded()
        if key_id in self._entries:
            self._entries.move_to_end(key_id)
            return self._entries[key_id].get("name")
        lock = self._locks.setdefault(key_id, asyncio.Lock())
        async with lock:
            if key_id in self._entries:
                self._entries.move_to_end(key_id)
                return self._entries[key_id].get("name")
            authoritative, name = await self._fetch_remote(http, token, key_id)
            if not authoritative:
                return None
            self._set(key_id, name)
            return name

    def _set(self, key_id: str, name: str | None) -> None:
        self._entries[key_id] = {
            "name": name,
            "ts": datetime.now(UTC).isoformat(),
        }
        self._entries.move_to_end(key_id)
        # Evict oldest until under the in-memory cap. Drop their locks too so
        # _locks doesn't outgrow _entries.
        while len(self._entries) > self.MAX_IN_MEMORY:
            evicted_key, _ = self._entries.popitem(last=False)
            self._locks.pop(evicted_key, None)
        self._mark_dirty()

    def _mark_dirty(self) -> None:
        self._dirty = True
        if self._flush_task is None or self._flush_task.done():
            self._flush_task = asyncio.create_task(self._debounced_flush())

    async def _debounced_flush(self) -> None:
        await asyncio.sleep(self.FLUSH_DEBOUNCE_S)
        if not self._dirty:
            return
        try:
            await self._flush_to_db()
            self._dirty = False
        except Exception as exc:
            # Best-effort; the next dirty mark will reschedule. Log so silent
            # flush failures are visible in operations.
            log.warning(
                f"{self._LOG_PREFIX}.cache_flush_failed",
                customer=self.customer_id,
                team=self.team_id,
                error=type(exc).__name__,
            )

    async def _flush_to_db(self) -> None:
        """Persist this worker's view, merged with whatever's already in JSONB.

        Without merge: each `patch_source_metadata` replaces the whole
        sub-object (top-level JSONB `||` semantics), so two workers flushing
        concurrently would clobber each other's contributions. With merge:
        read current state, take the most-recent entry per id across both
        views, cap at `MAX_PERSIST` by recency, write back. Cold-starting
        workers then load the union of all flushes, not just whoever wrote
        last.

        Race window: between read and write here, another worker could write
        and have its contributions overwritten. Acceptable trade — flushes
        are debounced 30s, concurrent flushes for the same team are rare,
        and a missed entry comes back on the next dirty mark anywhere. If we
        ever need stronger guarantees, the upgrade path is a SQL-side merge
        via `jsonb_set(metadata, '{<key>}', ... || $patch::jsonb)`.
        """
        try:
            persisted = await load_source_metadata(SourceSystem.SLACK, self.team_id)
        except Exception:
            # Read failed — fall back to write-only (last-writer-wins). The
            # exception will surface in the outer flush handler's logs.
            persisted = {}
        merged: dict[str, dict[str, Any]] = {}
        for source in (persisted.get(self._JSONB_KEY) or {}, dict(self._entries)):
            for k_id, entry in source.items():
                if not isinstance(entry, dict):
                    continue
                existing = merged.get(k_id)
                if existing is None or entry.get("ts", "") >= existing.get("ts", ""):
                    merged[k_id] = entry

        # Cap at MAX_PERSIST by recency (latest ts wins, ties resolved by
        # iteration order which is fine — both versions are equivalent).
        ranked = sorted(
            merged.items(),
            key=lambda kv: kv[1].get("ts", ""),
            reverse=True,
        )
        keep = dict(ranked[: self.MAX_PERSIST])
        await patch_source_metadata(
            SourceSystem.SLACK,
            self.team_id,
            patch={self._JSONB_KEY: keep},
        )

    async def flush_now(self) -> None:
        """Synchronous flush bypass for tests and shutdown hooks."""
        if self._dirty:
            await self._flush_to_db()
            self._dirty = False


class _SlackUserCache(_SlackJsonbNameCache):
    """Display-name cache for Slack user ids (`U…`) via `users.info`."""

    _JSONB_KEY: ClassVar[str] = "user_names"
    _LOG_PREFIX: ClassVar[str] = "slack.user"

    async def _fetch_remote(
        self,
        http: httpx.AsyncClient,
        token: str,
        key_id: str,
    ) -> tuple[bool, str | None]:
        try:
            resp = await http.get(
                f"{_SLACK_API}/users.info",
                params={"user": key_id},
                headers={"Authorization": f"Bearer {token}"},
            )
        except httpx.HTTPError as exc:
            log.warning(
                "slack.users_info_transient",
                user=key_id,
                error=type(exc).__name__,
            )
            return (False, None)
        if resp.status_code != 200:
            log.warning(
                "slack.users_info_non_200",
                user=key_id,
                status=resp.status_code,
            )
            return (False, None)
        body = resp.json()
        if body.get("ok"):
            return (True, _pick_display_name((body.get("user") or {}).get("profile")))
        return (True, None)  # authoritative negative (e.g. user_not_found)

    async def prime(self, http: httpx.AsyncClient, token: str) -> None:
        """Bulk-fill from Slack users.list paginated. Best-effort.

        Fills up to `MAX_IN_MEMORY` entries; the debounced flush captures the
        top `MAX_PERSIST` for the cold tier. HTTP failures or `ok=false` log
        and short-circuit — `resolve` will fall back to users.info on miss.
        """
        await self.ensure_loaded()
        cursor: str | None = None
        while True:
            params: dict[str, Any] = {"limit": 200}
            if cursor:
                params["cursor"] = cursor
            try:
                resp = await http.get(
                    f"{_SLACK_API}/users.list",
                    params=params,
                    headers={"Authorization": f"Bearer {token}"},
                )
            except httpx.HTTPError as exc:
                log.warning("slack.users_list_transient", error=type(exc).__name__)
                return
            if resp.status_code != 200:
                log.warning("slack.users_list_non_200", status=resp.status_code)
                return
            body = resp.json()
            if not body.get("ok"):
                log.warning("slack.users_list_not_ok", error=body.get("error"))
                return
            for u in body.get("members", []):
                uid = u.get("id")
                if not uid:
                    continue
                self._set(uid, _pick_display_name(u.get("profile")))
            cursor = (body.get("response_metadata") or {}).get("next_cursor") or None
            if not cursor:
                return


class _SlackChannelCache(_SlackJsonbNameCache):
    """Name cache for Slack channel ids (`C…`/`G…`) via `conversations.info`.

    `conversations.info` returns `{"channel": {"name": "engineering", ...}}`
    for both public and private channels under the `channels:read` and
    `groups:read` scopes the connector already requests at install time.

    Bulk-fill path is `prime_from_listing` because the backfill already
    paginates `conversations.list` to rank channels by member count — pulling
    names from that same response avoids a second walk through the workspace.
    """

    _JSONB_KEY: ClassVar[str] = "channel_names"
    _LOG_PREFIX: ClassVar[str] = "slack.channel"

    async def _fetch_remote(
        self,
        http: httpx.AsyncClient,
        token: str,
        key_id: str,
    ) -> tuple[bool, str | None]:
        try:
            resp = await http.get(
                f"{_SLACK_API}/conversations.info",
                params={"channel": key_id},
                headers={"Authorization": f"Bearer {token}"},
            )
        except httpx.HTTPError as exc:
            log.warning(
                "slack.conversations_info_transient",
                channel=key_id,
                error=type(exc).__name__,
            )
            return (False, None)
        if resp.status_code != 200:
            log.warning(
                "slack.conversations_info_non_200",
                channel=key_id,
                status=resp.status_code,
            )
            return (False, None)
        body = resp.json()
        if body.get("ok"):
            ch = body.get("channel") or {}
            return (True, _sanitize_display_name(ch.get("name")))
        # Authoritative negative (channel_not_found, missing_scope, etc.).
        return (True, None)

    async def prime_from_listing(
        self, listed: list[tuple[str, int, str | None]]
    ) -> None:
        """Bulk-fill from an already-paginated `conversations.list` result.

        Backfill already pulls (id, num_members, name) from conversations.list
        to rank hot channels first; reusing that data avoids a redundant API
        walk. Names with no value sanitize to None and still get cached so
        `resolve` short-circuits instead of firing conversations.info per
        message.
        """
        await self.ensure_loaded()
        for ch_id, _members, name in listed:
            if not ch_id:
                continue
            self._set(ch_id, _sanitize_display_name(name))


def _get_history_limiter() -> AsyncLimiter:
    import asyncio as _asyncio

    loop = _asyncio.get_running_loop()
    limiter = _HISTORY_LIMITERS.get(id(loop))
    if limiter is None:
        limiter = AsyncLimiter(1, 1.1)
        _HISTORY_LIMITERS[id(loop)] = limiter
    return limiter


@register_connector(SourceSystem.SLACK)
class SlackConnector(Connector):
    source_system: ClassVar[SourceSystem] = SourceSystem.SLACK
    display_name: ClassVar[str] = "Slack"

    def __init__(self, ctx: ConnectorContext) -> None:
        super().__init__(ctx)
        # One cache per Slack workspace per kind (keyed by team_id). Lives
        # for the lifetime of this connector instance, which is itself
        # one-per-process via Normalizer._connectors. Persists across
        # requests; flushed to JSONB on customer_source_mapping.metadata.
        self._caches: dict[str, _SlackUserCache] = {}
        self._channel_caches: dict[str, _SlackChannelCache] = {}

    def _get_cache(self, customer_id: str, team_id: str) -> _SlackUserCache:
        cache = self._caches.get(team_id)
        if cache is None:
            cache = _SlackUserCache(customer_id=customer_id, team_id=team_id)
            self._caches[team_id] = cache
        return cache

    def _get_channel_cache(
        self, customer_id: str, team_id: str
    ) -> _SlackChannelCache:
        cache = self._channel_caches.get(team_id)
        if cache is None:
            cache = _SlackChannelCache(customer_id=customer_id, team_id=team_id)
            self._channel_caches[team_id] = cache
        return cache

    # ------------------------------------------------------------------
    # 1. signature verification
    # ------------------------------------------------------------------

    def verify_signature(
        self,
        headers: Mapping[str, str],
        raw_body: bytes,
    ) -> bool:
        secret = self.settings.slack_signing_secret
        if secret is None:
            # Dev mode: accept unsigned payloads only when running locally.
            return self.settings.is_local

        ts = _header(headers, "x-slack-request-timestamp")
        sig = _header(headers, "x-slack-signature")
        if not ts or not sig:
            return False
        try:
            ts_int = int(ts)
        except ValueError:
            return False
        if abs(time.time() - ts_int) > _REQUEST_TS_SLACK_MAX_AGE_SEC:
            return False

        basestring = f"{_SIGNING_VERSION}:{ts}:".encode() + raw_body
        expected = (
            _SIGNING_VERSION
            + "="
            + hmac.new(
                secret.get_secret_value().encode(), basestring, hashlib.sha256
            ).hexdigest()
        )
        return hmac.compare_digest(expected, sig)

    # ------------------------------------------------------------------
    # 2. event parsing
    # ------------------------------------------------------------------

    def parse_webhook_event(
        self,
        customer_id: str,
        headers: Mapping[str, str],
        raw_payload: Mapping[str, Any],
    ) -> WebhookParseResult | None:
        # Slack sends a URL verification challenge on install — no event to ingest.
        if raw_payload.get("type") == "url_verification":
            return None

        event = raw_payload.get("event")
        if not isinstance(event, dict):
            raise InvalidWebhookPayload("slack payload missing 'event' dict")

        event_type = event.get("type")
        subtype = event.get("subtype")

        # Ignore ephemeral noise.
        if event_type in {"user_typing", "desktop_notification", "hello"}:
            return None
        if event_type != "message":
            return None

        channel = event.get("channel")
        if not channel:
            raise InvalidWebhookPayload("slack message missing channel")

        team_id = raw_payload.get("team_id")

        # Edits arrive as subtype=message_changed with the new body under
        # event.message and the prior under event.previous_message. The
        # stable message identity is the inner message's `ts`; event_ts is
        # the edit's own event timestamp, which we fold into source_event_id
        # so repeated edits of the same message don't collide on the UNIQUE
        # (customer_id, source_system, source_event_id) constraint.
        if subtype == "message_changed":
            inner = event.get("message")
            if not isinstance(inner, dict):
                raise InvalidWebhookPayload("message_changed missing 'message'")
            msg_ts = inner.get("ts")
            edited = inner.get("edited") or {}
            event_ts = event.get("event_ts") or edited.get("ts") or msg_ts
            if not msg_ts or not event_ts:
                raise InvalidWebhookPayload("slack message_changed missing ts/event_ts")
            return WebhookParseResult(
                source_event_id=f"{channel}:{msg_ts}:edit:{event_ts}",
                received_at=_ts_to_datetime(event_ts),
                event_kind=IngestionEventType.WEBHOOK,
                parse_hint={
                    "subtype": "message_changed",
                    "channel": channel,
                    "ts": msg_ts,
                    "thread_ts": inner.get("thread_ts"),
                    "team_id": team_id,
                },
            )

        # Deletes arrive as subtype=message_deleted with the original ts under
        # event.deleted_ts (and the full prior message under event.previous_message).
        if subtype == "message_deleted":
            previous = event.get("previous_message") or {}
            deleted_ts = event.get("deleted_ts") or previous.get("ts")
            event_ts = event.get("event_ts") or deleted_ts
            if not deleted_ts or not event_ts:
                raise InvalidWebhookPayload("slack message_deleted missing deleted_ts")
            return WebhookParseResult(
                source_event_id=f"{channel}:{deleted_ts}:delete:{event_ts}",
                received_at=_ts_to_datetime(event_ts),
                event_kind=IngestionEventType.WEBHOOK,
                parse_hint={
                    "subtype": "message_deleted",
                    "channel": channel,
                    "ts": deleted_ts,
                    "thread_ts": previous.get("thread_ts"),
                    "team_id": team_id,
                },
            )

        # Bot messages without text are noise (e.g. blocks-only interactive messages).
        if event.get("bot_id") and not event.get("text"):
            return None

        ts = event.get("ts")
        if not ts:
            raise InvalidWebhookPayload("slack message missing ts")

        # ts is monotonic per channel → globally unique with channel prefix.
        return WebhookParseResult(
            source_event_id=f"{channel}:{ts}",
            received_at=_ts_to_datetime(ts),
            event_kind=IngestionEventType.WEBHOOK,
            parse_hint={
                "subtype": subtype,
                "channel": channel,
                "ts": ts,
                "thread_ts": event.get("thread_ts"),
                "team_id": team_id,
            },
        )

    # ------------------------------------------------------------------
    # 3. hydration
    # ------------------------------------------------------------------

    async def fetch_supplementary(
        self,
        event: WebhookEvent,
        token: IntegrationToken | None,
    ) -> dict[str, Any]:
        # Thread replies aren't in the webhook body — fetch when we can.
        if token is None:
            return {}

        result: dict[str, Any] = {}

        msg = event.raw_payload.get("event", {})
        # message_changed/message_deleted nest the actual message under .message
        # / .previous_message. Pull the user from there when present so author
        # resolution covers edits/deletes too.
        inner = msg.get("message") if isinstance(msg.get("message"), dict) else None
        if inner is None and isinstance(msg.get("previous_message"), dict):
            inner = msg["previous_message"]
        author_msg = inner or msg

        # Resolve display name for the author when the webhook didn't inline a
        # user_profile (Slack sometimes does, sometimes doesn't). Skip for bot
        # messages — the resolver would just negative-cache the bot_id.
        user_id = author_msg.get("user")
        team_id = event.raw_payload.get("team_id") or msg.get("team")
        if user_id and team_id and not author_msg.get("user_profile"):
            cache = self._get_cache(event.customer_id, team_id)
            name = await cache.resolve(self.http, token.access_token, user_id)
            if name:
                result["user_profile"] = {"display_name": name}

        # Resolve channel name. Webhook bodies never carry it (only the id),
        # and the backfill prime won't have run for a workspace whose first
        # contact is a live message. Cache absorbs subsequent traffic so the
        # API call fires at most once per channel per worker.
        channel_id = msg.get("channel")
        if channel_id and team_id:
            ch_cache = self._get_channel_cache(event.customer_id, team_id)
            ch_name = await ch_cache.resolve(
                self.http, token.access_token, channel_id
            )
            if ch_name:
                result["channel_name"] = ch_name

        thread_ts = msg.get("thread_ts")
        channel = msg.get("channel")
        if not thread_ts or not channel:
            return result

        try:
            resp = await self.http.get(
                f"{_SLACK_API}/conversations.replies",
                params={"channel": channel, "ts": thread_ts, "limit": 50},
                headers={"Authorization": f"Bearer {token.access_token}"},
            )
        except Exception as exc:
            log.warning("slack.fetch_replies_failed", error=str(exc))
            return result

        if resp.status_code != 200:
            return result
        body = resp.json()
        if not body.get("ok"):
            return result
        result["replies"] = body.get("messages", [])
        return result

    # ------------------------------------------------------------------
    # 4. normalization
    # ------------------------------------------------------------------

    async def normalize(
        self,
        event: WebhookEvent,
        hydrated: Mapping[str, Any],
    ) -> NormalizationResult:
        outer = event.raw_payload.get("event", {})
        subtype = outer.get("subtype")
        is_edit = subtype == "message_changed"
        is_delete = subtype == "message_deleted"
        team_id = event.raw_payload.get("team_id", "")
        channel = outer.get("channel")

        # For edits, the authoritative message is under `message`. For deletes,
        # `previous_message` is what we had before. For plain messages, the
        # event itself is the message.
        if is_edit:
            msg = outer.get("message") or {}
        elif is_delete:
            msg = outer.get("previous_message") or {}
        else:
            msg = outer

        if not channel:
            channel = msg.get("channel")

        ts = msg.get("ts") or outer.get("ts") or outer.get("deleted_ts")
        if is_delete and not ts:
            ts = outer.get("deleted_ts")

        user = msg.get("user") or msg.get("bot_id") or "unknown"
        text = "" if is_delete else (msg.get("text") or "")
        thread_ts = msg.get("thread_ts")

        # Display-name stamping: hydrated value (webhook fetch_supplementary)
        # wins over msg-inline (some webhooks ship user_profile in-band) wins
        # over nothing. Critical: when no name is known, prefix is empty —
        # do NOT fall back to the raw U_ID (would pollute embeddings).
        display_name = _pick_display_name(
            hydrated.get("user_profile") or msg.get("user_profile")
        )
        # Channel name: fetch_supplementary sets channel_name from the cache
        # on the webhook path. Backfill stamps it via cache.peek() onto the
        # synthetic event's msg dict at `channel_name`. Missing => no
        # display_name on the Channel node; the JSONB-merge upsert
        # (graph_writer.py: properties || EXCLUDED.properties) lets a later
        # message with a resolved name fill it in without re-running anything.
        channel_name = _sanitize_display_name(
            hydrated.get("channel_name") or msg.get("channel_name")
        )
        body_text = "" if is_delete else (
            f"{display_name}: {text}" if display_name else text
        )

        if not channel or not ts:
            return NormalizationResult(skipped_reason="missing channel/ts after parse")

        doc_id = f"slack:{team_id}:{channel}:{ts}"
        source_url = self._permalink(team_id, channel, ts)
        created = _ts_to_datetime(ts)
        # Edits/deletes come in after the original — use the event's received_at
        # as the mutation clock so valid_from on the new version is monotonic.
        updated = event.received_at if (is_edit or is_delete) else created
        valid_from = updated

        if is_delete:
            # For deletes, body is empty (text already cleared above) and
            # deleted_at marks the tombstone. The content_hash MUST differ
            # from the prior live version's hash — otherwise the content-hash
            # no-op guard in _upsert_document would wrongly skip the delete.
            content_hash = _sha256(f"{doc_id}|__deleted__|{event.received_at.isoformat()}")
        else:
            # Hash on `body_text` (post-prefix) so a late-arriving display name
            # — webhook-only path that misses the cache on first sight then
            # resolves on retry — produces a different hash and re-upserts the
            # chunk with the name embedded. Without the prefix in the hash,
            # name-stamped vs raw versions would collide and the no-op guard
            # would drop the better one.
            content_hash = _sha256(
                f"{doc_id}|{body_text}|{','.join(sorted(_attachment_urls(msg)))}"
            )

        acl_principals = [
            ACLPrincipal(
                principal_type=PrincipalType.CHANNEL,
                principal_id=channel,
                permission=Permission.READ,
            )
        ]

        doc = Document(
            doc_id=doc_id,
            customer_id=event.customer_id,
            source_system=SourceSystem.SLACK,
            source_id=f"{channel}:{ts}",
            source_url=source_url,
            doc_class=DocClass.RAW_SOURCE,
            doc_type=(
                DocType.SLACK_THREAD if thread_ts == ts else DocType.SLACK_MESSAGE
            ),
            content_type="text/plain",
            content_hash=content_hash,
            title=_derive_title(text),
            body_preview=body_text[:280],
            body_size_bytes=len(body_text.encode("utf-8")),
            body_token_count=count_tokens(body_text),
            author_id=user,
            created_at=created,
            updated_at=updated,
            valid_from=valid_from,
            deleted_at=event.received_at if is_delete else None,
            ingested_at=datetime.now(UTC),
            parent_doc_id=(
                f"slack:{team_id}:{channel}:{thread_ts}"
                if thread_ts and thread_ts != ts
                else None
            ),
            acl=ACLSnapshot(principals=acl_principals, captured_at=event.received_at),
            metadata={
                "team_id": team_id,
                "channel_id": channel,
                "thread_ts": thread_ts,
                "edited": bool(msg.get("edited")) or is_edit,
                "deleted": is_delete,
                "reactions": msg.get("reactions", []),
            },
            body=body_text,
            doc_references=_references_from_text(text),
        )

        # Person properties: only attach name/display_name when the canonical_id
        # is a real Slack user ID (msg.user). When the canonical_id is a bot_id
        # or the "unknown" sentinel, the name belongs to a different identity
        # and would be misleading on the node.
        #
        # Both fields are populated: `name` is what retrieval reads
        # (`properties->>'name'` in graph_explore._node_title_expr +
        # retrievers/sql._entity_match_clause + the alnum/lowercased functional
        # indexes from migrations 0019/0022), and `display_name` is the legacy
        # alias that downstream UI code still references.
        person_props: dict[str, Any] = {"source_system": SourceSystem.SLACK.value}
        if msg.get("user") and display_name:
            person_props["name"] = display_name
            person_props["display_name"] = display_name

        channel_props: dict[str, Any] = {"team_id": team_id}
        if channel_name:
            # `name` is what retrieval reads — graph_explore._node_title_expr
            # surfaces it as the entity title, and retrievers/sql matches it
            # via the LOWER + alnum functional indexes (migrations 0019/0022).
            # Without `name`, the entity title falls back to canonical_id and
            # users see "C0B20FZSCUU" in results. `display_name` carries the
            # "#" prefix for UI renderers that distinguish channels from DMs.
            channel_props["name"] = channel_name
            channel_props["display_name"] = f"#{channel_name}"

        nodes = [
            GraphNodeSpec(
                label=NodeLabel.CHANNEL,
                canonical_id=channel,
                properties=channel_props,
            ),
            GraphNodeSpec(
                label=NodeLabel.PERSON,
                canonical_id=user,
                properties=person_props,
            ),
            GraphNodeSpec(
                label=NodeLabel.DOCUMENT,
                canonical_id=doc_id,
                properties={"doc_type": doc.doc_type},
            ),
        ]

        edges = [
            GraphEdgeSpec(
                edge_type=EdgeType.MEMBER_OF,
                from_label=NodeLabel.PERSON,
                from_canonical_id=user,
                to_label=NodeLabel.CHANNEL,
                to_canonical_id=channel,
                valid_from=created,
            ),
            GraphEdgeSpec(
                edge_type=EdgeType.AUTHORED,
                from_label=NodeLabel.PERSON,
                from_canonical_id=user,
                to_label=NodeLabel.DOCUMENT,
                to_canonical_id=doc_id,
                valid_from=created,
            ),
            # Document → Channel so list-pipeline entity filter
            # ("messages from #engineering") can find the doc.
            GraphEdgeSpec(
                edge_type=EdgeType.MEMBER_OF,
                from_label=NodeLabel.DOCUMENT,
                from_canonical_id=doc_id,
                to_label=NodeLabel.CHANNEL,
                to_canonical_id=channel,
                valid_from=created,
            ),
        ]

        acl_rows = [
            ACLSnapshotRow(
                source_system=SourceSystem.SLACK,
                principal_type=PrincipalType.CHANNEL,
                principal_id=channel,
                resource_type="slack.message",
                resource_id=f"{channel}:{ts}",
                permission=Permission.READ,
                valid_from=created,
            )
        ]

        return NormalizationResult(
            documents=[doc],
            graph_nodes=nodes,
            graph_edges=edges,
            acl_snapshots=acl_rows,
        )

    # ------------------------------------------------------------------
    # 5. OAuth install + exchange
    # ------------------------------------------------------------------

    def oauth_install_url(self, customer_id: str, redirect_uri: str) -> str:
        cid = self.settings.slack_client_id
        if not cid:
            from shared.exceptions import MissingSecret

            raise MissingSecret("SLACK_CLIENT_ID not configured")
        scopes = ",".join(
            [
                "channels:history",
                "channels:join",
                "channels:read",
                "groups:history",
                "groups:read",
                "users:read",
                "team:read",
            ]
        )
        return (
            "https://slack.com/oauth/v2/authorize"
            f"?client_id={cid}&scope={scopes}&redirect_uri={redirect_uri}"
            f"&state={customer_id}"
        )

    async def exchange_oauth_code(
        self,
        code: str | None,
        redirect_uri: str,
        extra_params: dict[str, str] | None = None,
    ) -> IntegrationToken:
        cid = self.settings.slack_client_id
        secret = self.settings.slack_client_secret
        if not cid or secret is None:
            from shared.exceptions import MissingSecret

            raise MissingSecret("SLACK_CLIENT_ID / SLACK_CLIENT_SECRET not configured")

        resp = await self.http.post(
            f"{_SLACK_API}/oauth.v2.access",
            data={
                "client_id": cid,
                "client_secret": secret.get_secret_value(),
                "redirect_uri": redirect_uri,
                "code": code,
            },
        )
        resp.raise_for_status()
        body = resp.json()
        if not body.get("ok"):
            from shared.exceptions import PermanentSourceError

            raise PermanentSourceError(f"slack oauth failed: {body.get('error')}")

        return IntegrationToken(
            customer_id="",
            source_system=SourceSystem.SLACK,
            access_token=body["access_token"],
            scope=body.get("scope"),
            webhook_secret=None,
        )

    # ------------------------------------------------------------------
    # 7. workspace identification
    # ------------------------------------------------------------------

    async def identify_workspaces(self, token: IntegrationToken):  # type: ignore[override]
        """Use Slack's `auth.test` to resolve team_id + team_name from the token.

        This is one API call per install; result is cached forever in
        customer_source_mapping (unless the customer re-installs under a
        different workspace).
        """
        try:
            resp = await self.http.post(
                f"{_SLACK_API}/auth.test",
                headers={"Authorization": f"Bearer {token.access_token}"},
            )
        except Exception as exc:
            log.warning("slack.auth_test_failed", error=str(exc))
            return []
        if resp.status_code != 200:
            return []
        body = resp.json()
        if not body.get("ok"):
            return []
        team_id = body.get("team_id")
        if not team_id:
            return []
        from shared.models import ExternalWorkspaceRef

        return [
            ExternalWorkspaceRef(
                external_id=team_id,
                external_name=body.get("team"),
                metadata={
                    "url": body.get("url"),
                    "bot_user_id": body.get("user_id"),
                    "bot_id": body.get("bot_id"),
                },
            )
        ]

    def extract_external_id_from_payload(self, headers, raw_payload):
        team_id = raw_payload.get("team_id")
        if not team_id and isinstance(raw_payload.get("team"), dict):
            team_id = raw_payload["team"].get("id")
        return str(team_id) if team_id else None

    # ------------------------------------------------------------------
    # 5. backfill
    # ------------------------------------------------------------------

    async def backfill(
        self,
        customer_id: str,
        token: IntegrationToken,
        cursor: str | None = None,
    ):
        """Historical Slack backfill - round-robin walk across channels.

        Each round fetches ONE page from each non-exhausted channel before
        starting the next round. Because conversations.history returns messages
        newest-first per page, after one round the user has the most recent
        ~200 messages of every channel ingested - the "newest-first across the
        workspace" UX win comes from the API contract, no separate phase needed.

        Rate limit: Slack capped conversations.history at ~1 req/sec per app
        per workspace (May 2025). Bottleneck is global, so a single shared
        token bucket (`_HISTORY_LIMITER`) is correct - parallel walkers would
        give zero throughput, only redistribute which channel gets the next
        page.

        Resumable: the cursor JSON encodes `{active: {ch_id: page_cursor}, done}`.
        `_decode_slack_cursor` migrates pre-rewrite cursors transparently so
        in-flight backfills survive deploy.

        Cursor in the event stream: real message events DO NOT carry `_cursor`
        (with N active channels the cursor can be ~50 bytes per channel; making
        every yielded event copy that to R2 is pure waste). End-of-round
        synthetic `_checkpoint` events carry the full cursor. Worst-case loss
        on crash = 1 round of progress; ON CONFLICT in ingestion_queue dedups
        on resume.
        """
        import asyncio as _asyncio
        import json as _json

        from shared.models import WebhookEvent

        state = _decode_slack_cursor(cursor)

        # On first run, auto-join every public channel so both backfill and
        # live webhooks see them. No-op without `channels:join` scope. Private
        # channels still need a manual `/invite @bot`.
        if cursor is None:
            await _join_all_public_channels(self.http, token, customer_id)

        # Resolve team_id first — needed both as the cache key AND stamped on
        # every yielded WebhookEvent's payload.team_id so the normalizer can
        # build canonical doc_ids without a second auth.test round-trip.
        team_id = await _auth_team_id(self.http, token.access_token) or "UNKNOWN"

        # Prime the per-workspace display-name cache once per backfill kickoff
        # so per-message normalize() can stamp "<name>: " into chunk text
        # without firing one users.info per message. On resume the JSONB cold
        # tier rehydrates the top-N hot users; the prime tops it up with the
        # full workspace member list.
        cache = self._get_cache(customer_id, team_id)
        await cache.prime(self.http, token.access_token)

        # Always enumerate channels - even on resume - so the round-robin
        # ranking has fresh num_members for every active channel. On a fresh
        # start, also seed `state.active`. This costs one paginated
        # conversations.list call per worker restart (~10-30s under the rate
        # cap on a 1000-channel workspace). Worth it for stable ranking.
        listed = await _list_channels(self.http, token.access_token)
        members_map: dict[str, int] = {ch_id: n for ch_id, n, _name in listed}

        # Prime the channel-name cache from the conversations.list payload
        # we already paid for. After this every Channel GraphNodeSpec produced
        # by normalize() can stamp `display_name=#<name>` without firing
        # conversations.info per message.
        ch_cache = self._get_channel_cache(customer_id, team_id)
        await ch_cache.prime_from_listing(listed)

        if cursor is None:
            state = {
                "active": {ch_id: None for ch_id, _, _name in listed},
                "done": [],
            }

        while state["active"]:
            # Sort once per round: hottest channels page first within the round
            # so #engineering's recent messages land before 499 dead channels'.
            # Members map is computed once per backfill() invocation so the
            # ordering is stable across rounds (channels with no entry default
            # to 0 -> sort last; covers channels deleted/archived since the
            # cursor was written).
            ranked = sorted(
                state["active"].keys(),
                key=lambda ch: members_map.get(ch, 0),
                reverse=True,
            )

            for ch_id in ranked:
                page_cursor = state["active"].get(ch_id)
                params: dict[str, Any] = {"channel": ch_id, "limit": 200}
                if page_cursor:
                    params["cursor"] = page_cursor

                try:
                    async with _get_history_limiter():
                        resp = await self.http.get(
                            f"{_SLACK_API}/conversations.history",
                            params=params,
                            headers={"Authorization": f"Bearer {token.access_token}"},
                        )
                except httpx.HTTPError as exc:
                    log.warning(
                        "slack.backfill_http_error", channel=ch_id, error=str(exc)
                    )
                    state["done"].append(ch_id)
                    state["active"].pop(ch_id, None)
                    continue

                if resp.status_code == 429:
                    # Workspace-global cap: pause OUTSIDE the limiter so other
                    # call sites also wait, then break the round so we don't
                    # immediately re-fire on the next channel under penalty.
                    retry_after = int(resp.headers.get("retry-after", "5"))
                    log.info(
                        "slack.backfill_429_pause",
                        retry_after_s=retry_after,
                        channel=ch_id,
                    )
                    await _asyncio.sleep(retry_after)
                    break

                if resp.status_code != 200:
                    state["done"].append(ch_id)
                    state["active"].pop(ch_id, None)
                    continue

                body = resp.json()
                if not body.get("ok"):
                    log.info(
                        "slack.backfill_channel_dropped",
                        channel=ch_id,
                        error=body.get("error"),
                    )
                    state["done"].append(ch_id)
                    state["active"].pop(ch_id, None)
                    continue

                for msg in body.get("messages", []):
                    if msg.get("type") != "message":
                        continue
                    if not msg.get("text") and not msg.get("files"):
                        continue
                    # Stamp cached display name onto the synthetic event so
                    # normalize() doesn't have to know whether this came from a
                    # webhook (where Slack sometimes inlines user_profile) or
                    # backfill (where it never does). Cache miss => no key
                    # written, normalize falls back gracefully.
                    cached_name = cache.peek(msg.get("user"))
                    cached_channel_name = ch_cache.peek(ch_id)
                    event_body: dict[str, Any] = {
                        **msg,
                        "type": "message",
                        "channel": ch_id,
                    }
                    if cached_name:
                        event_body["user_profile"] = {"display_name": cached_name}
                    if cached_channel_name:
                        event_body["channel_name"] = cached_channel_name
                    payload = {
                        "team_id": team_id,
                        "type": "event_callback",
                        "event": event_body,
                        # No `_cursor` here on purpose - see the cursor-bloat
                        # comment in the docstring. The runner advances cursor
                        # via the `_checkpoint` event yielded at end of round.
                    }
                    ts = msg.get("ts", "")
                    yield WebhookEvent(
                        customer_id=customer_id,
                        source_system=SourceSystem.SLACK,
                        source_event_id=f"{ch_id}:{ts}",
                        received_at=_ts_to_datetime(ts) if ts else datetime.now(UTC),
                        payload_s3_key="",
                        raw_payload=payload,
                        headers={},
                    )

                next_cursor = (body.get("response_metadata") or {}).get("next_cursor")
                if next_cursor:
                    state["active"][ch_id] = next_cursor
                else:
                    state["done"].append(ch_id)
                    state["active"].pop(ch_id, None)

            # End of round - emit a synthetic checkpoint so the runner persists
            # the cursor without us having to copy it onto every message event.
            # The runner's `_checkpoint` branch (backfill_runner.py:114) skips
            # the queue insert and just calls `_update_progress`.
            yield WebhookEvent(
                customer_id=customer_id,
                source_system=SourceSystem.SLACK,
                source_event_id=f"_checkpoint:{datetime.now(UTC).isoformat()}",
                received_at=datetime.now(UTC),
                payload_s3_key="",
                raw_payload={
                    "_checkpoint": True,
                    "_cursor": _json.dumps(state),
                },
                headers={},
            )

    # ------------------------------------------------------------------


# ---- helpers ---------------------------------------------------------------


_AUTO_JOIN_SCOPE = "channels:join"


async def _join_all_public_channels(
    http,
    token: IntegrationToken,
    customer_id: str,
) -> None:
    """Call conversations.join on every non-archived public channel the token can see.

    No-op if the token lacks channels:join scope. conversations.join is idempotent
    (already_in_channel is not an error). Respects 429 Retry-After; other
    per-channel failures are logged and the sweep continues.
    """
    import asyncio as _asyncio

    if not token.scope or _AUTO_JOIN_SCOPE not in token.scope:
        log.info("slack.auto_join.skipped_no_scope", customer=customer_id)
        return

    discovered = joined = already = errors = 0
    cursor: str | None = None
    while True:
        params: dict[str, Any] = {
            "types": "public_channel",
            "exclude_archived": "true",
            "limit": 1000,
        }
        if cursor:
            params["cursor"] = cursor
        resp = await http.get(
            f"{_SLACK_API}/conversations.list",
            params=params,
            headers={"Authorization": f"Bearer {token.access_token}"},
        )
        if resp.status_code == 429:
            await _asyncio.sleep(int(resp.headers.get("retry-after", "5")))
            continue
        if resp.status_code != 200:
            log.warning("slack.auto_join.list_failed", status=resp.status_code)
            break
        body = resp.json()
        if not body.get("ok"):
            log.warning("slack.auto_join.list_failed", error=body.get("error"))
            break

        for ch in body.get("channels", []):
            discovered += 1
            channel_id = ch.get("id")
            if not channel_id:
                continue
            if ch.get("is_member"):
                already += 1
                continue
            # Retry on 429; a single non-429 failure ends this channel's attempt.
            while True:
                jr = await http.post(
                    f"{_SLACK_API}/conversations.join",
                    data={"channel": channel_id},
                    headers={"Authorization": f"Bearer {token.access_token}"},
                )
                if jr.status_code == 429:
                    await _asyncio.sleep(int(jr.headers.get("retry-after", "5")))
                    continue
                break
            jbody = jr.json() if jr.status_code == 200 else {}
            if jr.status_code == 200 and jbody.get("ok"):
                joined += 1
            else:
                errors += 1
                log.warning(
                    "slack.auto_join.channel_failed",
                    channel=channel_id,
                    error=jbody.get("error") if jbody else f"http_{jr.status_code}",
                )

        cursor = (body.get("response_metadata") or {}).get("next_cursor") or None
        if not cursor:
            break

    log.info(
        "slack.auto_join.done",
        customer=customer_id,
        discovered=discovered,
        joined=joined,
        already_member=already,
        errors=errors,
    )


async def _list_channels(http, token: str) -> list[tuple[str, int, str | None]]:
    """Enumerate all channels the bot can see. Paginated.

    Returns [(channel_id, num_members, name), ...]. num_members lets the
    round-robin walker rank hot channels first so #engineering's recent
    messages don't sit behind 499 dead channels' first-page fetches.
    num_members is in conversations.list's default response shape per Slack
    docs (public AND private channels). Channels missing the field default
    to 0 -> sort last. `name` feeds the per-workspace channel-name cache so
    Channel GraphNodeSpecs get a `#<name>` display name without firing
    conversations.info per message.
    """
    channels: list[tuple[str, int, str | None]] = []
    cursor: str | None = None
    while True:
        params = {"types": "public_channel,private_channel", "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        resp = await http.get(
            f"{_SLACK_API}/conversations.list",
            params=params,
            headers={"Authorization": f"Bearer {token}"},
        )
        if resp.status_code != 200:
            break
        body = resp.json()
        if not body.get("ok"):
            break
        for ch in body.get("channels", []):
            if ch.get("id") and ch.get("is_member", True):
                channels.append(
                    (
                        ch["id"],
                        int(ch.get("num_members") or 0),
                        ch.get("name"),
                    )
                )
        cursor = (body.get("response_metadata") or {}).get("next_cursor") or None
        if not cursor:
            break
    return channels


async def _auth_team_id(http, token: str) -> str | None:
    resp = await http.post(
        f"{_SLACK_API}/auth.test",
        headers={"Authorization": f"Bearer {token}"},
    )
    if resp.status_code != 200:
        return None
    body = resp.json()
    if not body.get("ok"):
        return None
    return body.get("team_id")


def _decode_slack_cursor(cursor: str | None) -> dict[str, Any]:
    """Decode the persisted backfill cursor into the round-robin state shape.

    Shape: {"active": {channel_id: page_cursor_or_none, ...}, "done": [...]}.

    Migrates the pre-rewrite shape transparently to keep in-flight backfills
    moving on deploy without losing any channel: channels_remaining + the
    current channel both fold into `active`, and current_channel's
    history_cursor wins on collision so we never re-walk a channel from page 1.
    """
    import json as _json

    empty: dict[str, Any] = {"active": {}, "done": []}
    if not cursor:
        return empty
    try:
        data = _json.loads(cursor)
    except _json.JSONDecodeError:
        return empty
    if not isinstance(data, dict):
        return empty

    # New shape — passthrough with shallow copy + type coercion.
    if isinstance(data.get("active"), dict):
        return {
            "active": dict(data["active"]),
            "done": list(data.get("done", [])),
        }

    # Old shape — migrate. Order matters: seed channels_remaining first so the
    # current_channel write below can overwrite a duplicate entry without
    # losing its history_cursor.
    if "channels_remaining" in data or "current_channel" in data:
        active: dict[str, str | None] = {
            ch: None for ch in (data.get("channels_remaining") or []) if ch
        }
        cur = data.get("current_channel")
        if cur:
            active[cur] = data.get("history_cursor")
        return {"active": active, "done": []}

    return empty





def _header(headers: Mapping[str, str], name: str) -> str | None:
    for k, v in headers.items():
        if k.lower() == name.lower():
            return v
    return None


def _ts_to_datetime(ts: str) -> datetime:
    seconds = float(ts)
    return datetime.fromtimestamp(seconds, tz=UTC)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _permalink(team_id: str, channel: str, ts: str) -> str:
    # Slack permalinks don't need team-subdomain resolution for linking back.
    ts_part = ts.replace(".", "")
    return f"https://slack.com/archives/{channel}/p{ts_part}"


def _derive_title(text: str) -> str | None:
    if not text:
        return None
    first_line = text.splitlines()[0].strip()
    return first_line[:120] if first_line else None


def _attachment_urls(msg: Mapping[str, Any]) -> list[str]:
    urls: list[str] = []
    for f in msg.get("files", []) or []:
        url = f.get("url_private") or f.get("permalink")
        if url:
            urls.append(url)
    return urls


def _references_from_text(text: str) -> list[DocRef]:
    refs: list[DocRef] = []
    if not text:
        return refs
    for token in text.split():
        if token.startswith("<http") and token.endswith(">"):
            url = token.strip("<>").split("|", 1)[0]
            refs.append(DocRef(external_url=url, ref_type=RefType.LINKS_TO))
    return refs


# Bind _permalink onto the connector class so tests can reach it easily.
SlackConnector._permalink = staticmethod(_permalink)  # type: ignore[attr-defined]
