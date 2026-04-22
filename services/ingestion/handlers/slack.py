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
        if subtype in {"message_changed", "message_deleted"}:
            # We care, but they need special handling — Phase 0 ignores edits/deletes.
            return None
        if event_type != "message":
            return None
        if event.get("bot_id") and not event.get("text"):
            return None

        ts = event.get("ts")
        channel = event.get("channel")
        if not ts or not channel:
            raise InvalidWebhookPayload("slack message missing ts/channel")

        # ts is monotonic per channel → globally unique with channel prefix.
        source_event_id = f"{channel}:{ts}"
        received_at = _ts_to_datetime(ts)

        return WebhookParseResult(
            source_event_id=source_event_id,
            received_at=received_at,
            event_kind=IngestionEventType.WEBHOOK,
            parse_hint={
                "channel": channel,
                "ts": ts,
                "thread_ts": event.get("thread_ts"),
                "team_id": raw_payload.get("team_id"),
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
        msg = event.raw_payload.get("event", {})
        channel = msg.get("channel")
        ts = msg.get("ts")
        team_id = event.raw_payload.get("team_id", "")
        user = msg.get("user") or msg.get("bot_id") or "unknown"
        text = msg.get("text") or ""
        thread_ts = msg.get("thread_ts")

        if not channel or not ts:
            return NormalizationResult(skipped_reason="missing channel/ts after parse")

        doc_id = f"slack:{team_id}:{channel}:{ts}"
        source_url = self._permalink(team_id, channel, ts)
        created = _ts_to_datetime(ts)

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
            updated_at=created,
            valid_from=created,
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
    # 5. OAuth install (for completeness — real redirect wired in Tier 7)
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

    # ------------------------------------------------------------------


# ---- helpers ---------------------------------------------------------------


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
