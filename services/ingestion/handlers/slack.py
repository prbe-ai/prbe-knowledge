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
            inner = event.get("message") or {}
            msg_ts = inner.get("ts")
            event_ts = event.get("event_ts") or inner.get("edited", {}).get("ts")
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
            deleted_ts = event.get("deleted_ts") or (
                event.get("previous_message") or {}
            ).get("ts")
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
        team_id = event.raw_payload.get("team_id", "")
        channel = outer.get("channel")

        # For edits, the authoritative message is under `message`. For deletes,
        # `previous_message` is what we had before. For plain messages, the
        # event itself is the message.
        if subtype == "message_changed":
            msg = outer.get("message") or {}
        elif subtype == "message_deleted":
            msg = outer.get("previous_message") or {}
        else:
            msg = outer

        ts = msg.get("ts") or outer.get("ts") or outer.get("deleted_ts")
        if subtype == "message_deleted" and not ts:
            ts = outer.get("deleted_ts")

        user = msg.get("user") or msg.get("bot_id") or "unknown"
        text = msg.get("text") or ""
        thread_ts = msg.get("thread_ts")

        if not channel or not ts:
            return NormalizationResult(skipped_reason="missing channel/ts after parse")

        doc_id = f"slack:{team_id}:{channel}:{ts}"
        source_url = self._permalink(team_id, channel, ts)
        created = _ts_to_datetime(ts)

        # For deletes, body is empty and deleted_at marks the tombstone.
        # The content_hash MUST differ from the prior live version so the
        # normalizer writes a new version and marks old chunks stale via diff.
        deleted_at: datetime | None = None
        if subtype == "message_deleted":
            text = ""
            deleted_at = event.received_at
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
            updated_at=event.received_at,
            valid_from=event.received_at,
            deleted_at=deleted_at,
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
                "edited": bool(msg.get("edited")),
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

    def oauth_install_url(self, customer_id: str, redirect_uri: str) -> str:
        cid = self.settings.slack_client_id
        if not cid:
            from shared.exceptions import MissingSecret

            raise MissingSecret("SLACK_CLIENT_ID not configured")
        scopes = ",".join(
            [
                "channels:history",
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
        code: str,
        redirect_uri: str,
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

        # 1. Enumerate channels once if we don't have them yet.
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
