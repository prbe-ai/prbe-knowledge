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

import hashlib
import hmac
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, ClassVar

from services.ingestion.chunker import count_tokens
from services.ingestion.handlers.base import Connector
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


@register_connector(SourceSystem.SLACK)
class SlackConnector(Connector):
    source_system: ClassVar[SourceSystem] = SourceSystem.SLACK
    display_name: ClassVar[str] = "Slack"

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

        msg = event.raw_payload.get("event", {})
        thread_ts = msg.get("thread_ts")
        channel = msg.get("channel")
        if not thread_ts or not channel:
            return {}

        try:
            resp = await self.http.get(
                f"{_SLACK_API}/conversations.replies",
                params={"channel": channel, "ts": thread_ts, "limit": 50},
                headers={"Authorization": f"Bearer {token.access_token}"},
            )
        except Exception as exc:
            log.warning("slack.fetch_replies_failed", error=str(exc))
            return {}

        if resp.status_code != 200:
            return {}
        body = resp.json()
        if not body.get("ok"):
            return {}
        return {"replies": body.get("messages", [])}

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
            content_hash = _sha256(
                f"{doc_id}|{text}|{','.join(sorted(_attachment_urls(msg)))}"
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
            body_preview=text[:280],
            body_size_bytes=len(text.encode("utf-8")),
            body_token_count=count_tokens(text),
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
                "body": text,
                "team_id": team_id,
                "channel_id": channel,
                "thread_ts": thread_ts,
                "edited": bool(msg.get("edited")) or is_edit,
                "deleted": is_delete,
                "reactions": msg.get("reactions", []),
            },
            doc_references=_references_from_text(text),
        )

        nodes = [
            GraphNodeSpec(
                label=NodeLabel.CHANNEL,
                canonical_id=channel,
                properties={"team_id": team_id},
            ),
            GraphNodeSpec(
                label=NodeLabel.PERSON,
                canonical_id=user,
                properties={"source_system": SourceSystem.SLACK.value},
            ),
            GraphNodeSpec(
                label=NodeLabel.DOCUMENT,
                canonical_id=doc_id,
                properties={"doc_type": doc.doc_type.value},
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

    def oauth_install_url(
        self, customer_id: str, redirect_uri: str, state: str
    ) -> str:
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
            f"&state={state}"
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
                metadata={"url": body.get("url")},
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
        """Historical Slack backfill — paginated channel + message walk.

        Resumable via the `cursor` arg: an opaque JSON blob encoding which
        channel we're in and where in that channel's history we stopped.
        Yields synthetic WebhookEvents shaped exactly like live `message`
        events so the normalizer has one code path.

        Rate limits: Slack tier 3 (~20 req/min on conversations.history).
        We rely on httpx + source-returned Retry-After on 429 to back off;
        Slack's docs promise graceful degradation, not throttling kills.
        """
        import json as _json

        import httpx

        from shared.models import WebhookEvent

        state = _decode_slack_cursor(cursor)

        # 1. On first run, auto-join every public channel so both backfill and
        # live webhooks see them. No-op if the token lacks channels:join scope.
        # Private channels still require a manual `/invite @bot`.
        if cursor is None:
            await _join_all_public_channels(self.http, token, customer_id)

        # 2. Enumerate channels once if we don't have them yet.
        if not state["channels_remaining"] and state["current_channel"] is None:
            state["channels_remaining"] = await _list_channels(self.http, token.access_token)

        team_id = await _auth_team_id(self.http, token.access_token) or "UNKNOWN"

        while state["current_channel"] or state["channels_remaining"]:
            if state["current_channel"] is None:
                state["current_channel"] = state["channels_remaining"].pop(0)
                state["history_cursor"] = None

            channel = state["current_channel"]

            try:
                params = {"channel": channel, "limit": 200}
                if state["history_cursor"]:
                    params["cursor"] = state["history_cursor"]
                resp = await self.http.get(
                    f"{_SLACK_API}/conversations.history",
                    params=params,
                    headers={"Authorization": f"Bearer {token.access_token}"},
                )
            except httpx.HTTPError as exc:
                log.warning("slack.backfill_http_error", channel=channel, error=str(exc))
                # Move on to next channel rather than stalling the whole backfill.
                state["current_channel"] = None
                state["history_cursor"] = None
                continue

            if resp.status_code == 429:
                # Respect Retry-After (seconds). httpx won't sleep for us here.
                import asyncio as _asyncio

                retry_after = int(resp.headers.get("retry-after", "5"))
                await _asyncio.sleep(retry_after)
                continue

            if resp.status_code != 200:
                state["current_channel"] = None
                state["history_cursor"] = None
                continue

            body = resp.json()
            if not body.get("ok"):
                state["current_channel"] = None
                state["history_cursor"] = None
                continue

            for msg in body.get("messages", []):
                if msg.get("type") != "message":
                    continue
                # Skip messages without text (ephemeral, bot blocks-only).
                if not msg.get("text") and not msg.get("files"):
                    continue
                payload = {
                    "team_id": team_id,
                    "type": "event_callback",
                    "event": {
                        **msg,
                        "type": "message",
                        "channel": channel,
                    },
                    # Runner reads this to persist the cursor:
                    "_cursor": _json.dumps(state),
                }
                ts = msg.get("ts", "")
                yield WebhookEvent(
                    customer_id=customer_id,
                    source_system=SourceSystem.SLACK,
                    source_event_id=f"{channel}:{ts}",
                    received_at=_ts_to_datetime(ts) if ts else datetime.now(UTC),
                    payload_s3_key="",  # runner fills this in
                    raw_payload=payload,
                    headers={},
                )

            next_cursor = (body.get("response_metadata") or {}).get("next_cursor")
            if next_cursor:
                state["history_cursor"] = next_cursor
            else:
                # Channel exhausted — move on.
                state["current_channel"] = None
                state["history_cursor"] = None

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


async def _list_channels(http, token: str) -> list[str]:
    """Enumerate all channels the bot can see. Paginated."""
    channels: list[str] = []
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
                channels.append(ch["id"])
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


def _decode_slack_cursor(cursor: str | None) -> dict:
    import json as _json

    if not cursor:
        return {"channels_remaining": [], "current_channel": None, "history_cursor": None}
    try:
        return _json.loads(cursor)
    except _json.JSONDecodeError:
        # Corrupt cursor — start over.
        return {"channels_remaining": [], "current_channel": None, "history_cursor": None}





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
