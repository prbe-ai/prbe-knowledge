"""Notion connector — page + database ingestion.

Covers two inbound shapes (Phase 0 treats them uniformly):

1. Notion's official webhook payload (beta as of 2025): top-level `type`
   like `"page.updated"` + an `entity` dict with `{type, id}` and a `data`
   dict carrying `last_edited_time` / `last_edited_by`.

2. A synthetic push from our lightweight polling worker for customers whose
   Notion workspace doesn't yet have webhooks. Shape:
       {"customer_id", "resource_type": "page"|"database",
        "resource_id", "polled_at", "last_edited_time"?, ...}

Both collapse to the same WebhookEvent → fetch_supplementary → normalize path.

ACL: Notion's permissions model is partly opaque to internal integrations.
Phase 0 strategy (captured, not enforced):
  - Always emit a workspace-level ACL row (PrincipalType.WORKSPACE) so the
    default "anyone in this Notion workspace can read" rule is recorded.
  - If the page metadata exposes a `permissions` list (requires the
    integration's "Read user information" capability), emit a more specific
    USER/GROUP row per entry.
  - Record `parent_id` + `inherits: True` in the ACL row metadata so the
    Phase 1 enforcement layer can walk the parent chain at query time
    without re-reading every page.
"""

from __future__ import annotations

import hashlib
import hmac
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

_NOTION_API = "https://api.notion.com/v1"
_NOTION_VERSION = "2022-06-28"

# Webhook event types we persist.
_ACCEPTED_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "page.updated",
        "page.created",
        "database.updated",
        "database.created",
        # Deletes produce a tombstone document with deleted_at set; the chunk
        # diff in the normalizer marks all previously-live chunks stale.
        "page.deleted",
        "database.deleted",
    }
)

_DELETE_EVENT_TYPES: frozenset[str] = frozenset(
    {"page.deleted", "database.deleted"}
)

_DEFAULT_WORKSPACE_PRINCIPAL = "notion-default"

_NOTION_OAUTH_AUTHORIZE = "https://api.notion.com/v1/oauth/authorize"
_NOTION_OAUTH_TOKEN = "https://api.notion.com/v1/oauth/token"


# ---- source-shape discriminators -------------------------------------------


def _is_synthetic_poll(payload: Mapping[str, Any]) -> bool:
    """Synthetic polling pushes carry `resource_type` + `resource_id` and no `entity` dict."""
    return (
        "entity" not in payload
        and "resource_id" in payload
        and "resource_type" in payload
    )


def _is_notion_webhook(payload: Mapping[str, Any]) -> bool:
    """Notion official webhook: top-level `type` + `entity` object."""
    entity = payload.get("entity")
    return isinstance(entity, dict) and isinstance(payload.get("type"), str)


# ---- block → markdown -------------------------------------------------------


def _rich_text_to_plain(rich_text: list[dict[str, Any]] | None) -> str:
    if not rich_text:
        return ""
    parts: list[str] = []
    for span in rich_text:
        plain = span.get("plain_text")
        if plain:
            parts.append(plain)
    return "".join(parts)


def _extract_mentioned_user_ids(blocks: list[dict[str, Any]]) -> list[str]:
    """Walk block rich_text spans, collect Notion user ids from @mentions."""
    user_ids: list[str] = []
    for block in blocks:
        btype = block.get("type")
        payload = block.get(btype) if btype else None
        if not isinstance(payload, dict):
            continue
        rich = payload.get("rich_text") or []
        for span in rich:
            if span.get("type") != "mention":
                continue
            mention = span.get("mention") or {}
            if mention.get("type") == "user":
                uid = (mention.get("user") or {}).get("id")
                if uid and uid not in user_ids:
                    user_ids.append(uid)
    return user_ids


def blocks_to_markdown(blocks: list[dict[str, Any]]) -> str:
    """Convert a flat list of Notion blocks to a markdown-ish plain-text body.

    Handles: paragraph, heading_1/2/3, bulleted_list_item, numbered_list_item,
    to_do, code, quote. Unknown block types emit `[block:<type>]` so the
    downstream chunker still sees a stable placeholder.
    """
    lines: list[str] = []
    numbered_counter = 0
    for block in blocks:
        btype = block.get("type") or ""
        payload = block.get(btype)
        if not isinstance(payload, dict):
            lines.append(f"[block:{btype or 'unknown'}]")
            numbered_counter = 0
            continue

        text = _rich_text_to_plain(payload.get("rich_text"))

        if btype == "paragraph":
            lines.append(text)
            numbered_counter = 0
        elif btype == "heading_1":
            lines.append(f"# {text}")
            numbered_counter = 0
        elif btype == "heading_2":
            lines.append(f"## {text}")
            numbered_counter = 0
        elif btype == "heading_3":
            lines.append(f"### {text}")
            numbered_counter = 0
        elif btype == "bulleted_list_item":
            lines.append(f"- {text}")
            numbered_counter = 0
        elif btype == "numbered_list_item":
            numbered_counter += 1
            lines.append(f"{numbered_counter}. {text}")
        elif btype == "to_do":
            checked = payload.get("checked", False)
            box = "[x]" if checked else "[ ]"
            lines.append(f"- {box} {text}")
            numbered_counter = 0
        elif btype == "code":
            lang = payload.get("language") or ""
            lines.append(f"```{lang}\n{text}\n```")
            numbered_counter = 0
        elif btype == "quote":
            lines.append(f"> {text}")
            numbered_counter = 0
        else:
            lines.append(f"[block:{btype}]")
            numbered_counter = 0

    return "\n".join(line for line in lines if line is not None)


# ---- title extraction ------------------------------------------------------


def _title_from_properties(properties: Mapping[str, Any]) -> str | None:
    """Pull a title string out of a page's properties dict.

    Notion pages have exactly one property of type `title`, but it can be
    keyed as "title", "Name", or arbitrary. Databases also use rich_text
    "Name" in some shapes; fall back to scanning for the first title-typed
    property.
    """
    # Canonical path first
    for key in ("title", "Name", "Title"):
        prop = properties.get(key)
        if isinstance(prop, dict):
            rich = prop.get("title") or prop.get("rich_text")
            if isinstance(rich, list):
                val = _rich_text_to_plain(rich).strip()
                if val:
                    return val[:200]

    # Fallback: scan for any property with type=title
    for prop in properties.values():
        if not isinstance(prop, dict):
            continue
        if prop.get("type") == "title":
            val = _rich_text_to_plain(prop.get("title")).strip()
            if val:
                return val[:200]
    return None


# ---- connector -------------------------------------------------------------


@register_connector(SourceSystem.NOTION)
class NotionConnector(Connector):
    source_system: ClassVar[SourceSystem] = SourceSystem.NOTION
    display_name: ClassVar[str] = "Notion"

    # ------------------------------------------------------------------
    # 1. signature verification
    # ------------------------------------------------------------------

    def verify_signature(
        self,
        headers: Mapping[str, str],
        raw_body: bytes,
    ) -> bool:
        # Local dev always accepts.
        if self.settings.is_local:
            return True

        sig = _header(headers, "x-notion-signature")
        if sig is None:
            return False

        secret = self.settings.notion_client_secret
        if secret is None:
            return False

        expected = hmac.new(
            secret.get_secret_value().encode(),
            raw_body,
            hashlib.sha256,
        ).hexdigest()
        # Notion docs show the header as a hex digest; tolerate a `sha256=` prefix.
        candidate = sig.split("=", 1)[1] if sig.startswith("sha256=") else sig
        return hmac.compare_digest(expected, candidate)

    # ------------------------------------------------------------------
    # 2. event parsing
    # ------------------------------------------------------------------

    def parse_webhook_event(
        self,
        customer_id: str,
        headers: Mapping[str, str],
        raw_payload: Mapping[str, Any],
    ) -> WebhookParseResult | None:
        if _is_synthetic_poll(raw_payload):
            return self._parse_synthetic(raw_payload)
        if _is_notion_webhook(raw_payload):
            return self._parse_notion_webhook(raw_payload)
        raise InvalidWebhookPayload(
            "notion payload missing both 'entity' (webhook) and 'resource_id' (synthetic poll)"
        )

    def _parse_notion_webhook(
        self,
        raw_payload: Mapping[str, Any],
    ) -> WebhookParseResult | None:
        event_type = raw_payload.get("type")
        if event_type not in _ACCEPTED_EVENT_TYPES:
            # Unknown / ignored (deletes, user.*, workspace.*, verification ping).
            return None

        entity = raw_payload.get("entity")
        if not isinstance(entity, dict):
            raise InvalidWebhookPayload("notion webhook missing 'entity' dict")

        entity_type = entity.get("type")
        entity_id = entity.get("id")
        if entity_type not in {"page", "database"} or not entity_id:
            raise InvalidWebhookPayload(
                f"notion webhook has unsupported entity: type={entity_type!r} id={entity_id!r}"
            )

        data = raw_payload.get("data") or {}
        last_edited_time = (
            data.get("last_edited_time")
            or raw_payload.get("timestamp")
            or _utcnow_iso()
        )

        is_delete = event_type in _DELETE_EVENT_TYPES
        # Deletes get their own source_event_id suffix so a create/update/delete
        # sequence on the same resource doesn't collide on the UNIQUE constraint.
        tail = "delete" if is_delete else "edit"
        source_event_id = f"{entity_type}:{entity_id}:{tail}:{last_edited_time}"
        received_at = _parse_iso8601(raw_payload.get("timestamp") or last_edited_time)

        return WebhookParseResult(
            source_event_id=source_event_id,
            received_at=received_at,
            event_kind=IngestionEventType.WEBHOOK,
            parse_hint={
                "resource_type": entity_type,
                "resource_id": entity_id,
                "last_edited_time": last_edited_time,
                "workspace_id": raw_payload.get("workspace_id"),
                "source_event_shape": "notion_webhook",
                "event_type": event_type,
                "is_delete": is_delete,
            },
        )

    def _parse_synthetic(
        self,
        raw_payload: Mapping[str, Any],
    ) -> WebhookParseResult:
        resource_type = raw_payload.get("resource_type")
        resource_id = raw_payload.get("resource_id")
        if resource_type not in {"page", "database"} or not resource_id:
            raise InvalidWebhookPayload(
                "synthetic notion poll requires resource_type in {page,database} + resource_id"
            )
        last_edited_time = (
            raw_payload.get("last_edited_time") or raw_payload.get("polled_at") or _utcnow_iso()
        )
        received_at = _parse_iso8601(
            raw_payload.get("polled_at") or last_edited_time
        )
        source_event_id = f"{resource_type}:{resource_id}:{last_edited_time}"
        return WebhookParseResult(
            source_event_id=source_event_id,
            received_at=received_at,
            event_kind=IngestionEventType.SYNC,
            parse_hint={
                "resource_type": resource_type,
                "resource_id": resource_id,
                "last_edited_time": last_edited_time,
                "workspace_id": raw_payload.get("workspace_id"),
                "source_event_shape": "synthetic_poll",
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
        if token is None:
            return {}

        resource_type, resource_id = _resource_from_event(event)
        if not resource_type or not resource_id:
            return {}

        entity = await self._fetch_entity(resource_type, resource_id, token)
        if entity is None:
            return {}

        blocks: list[dict[str, Any]] = []
        # Pages carry content blocks. Databases carry schema only — no block tree.
        if resource_type == "page":
            blocks = await self._fetch_all_blocks(resource_id, token)

        body_markdown = blocks_to_markdown(blocks) if blocks else ""
        mentioned_user_ids = _extract_mentioned_user_ids(blocks)
        permissions = entity.get("permissions") or []

        return {
            "entity": entity,
            "body_markdown": body_markdown,
            "mentioned_user_ids": mentioned_user_ids,
            "permissions": permissions,
            "resource_type": resource_type,
        }

    async def _fetch_entity(
        self,
        resource_type: str,
        resource_id: str,
        token: IntegrationToken,
    ) -> dict[str, Any] | None:
        path = "pages" if resource_type == "page" else "databases"
        url = f"{_NOTION_API}/{path}/{resource_id}"
        try:
            resp = await self.http.get(url, headers=_auth_headers(token))
        except (httpx_error_types()) as exc:
            log.warning("notion.fetch_entity_failed", error=str(exc), id=resource_id)
            return None

        if resp.status_code != 200:
            log.warning(
                "notion.fetch_entity_non_200",
                status=resp.status_code,
                id=resource_id,
            )
            return None
        body = resp.json()
        return body if isinstance(body, dict) else None

    async def _fetch_all_blocks(
        self,
        page_id: str,
        token: IntegrationToken,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        cursor: str | None = None
        # Cap pages defensively — Phase 0 doesn't page arbitrarily large docs.
        for _ in range(20):
            params: dict[str, Any] = {"page_size": 100}
            if cursor:
                params["start_cursor"] = cursor
            try:
                resp = await self.http.get(
                    f"{_NOTION_API}/blocks/{page_id}/children",
                    params=params,
                    headers=_auth_headers(token),
                )
            except (httpx_error_types()) as exc:
                log.warning("notion.fetch_blocks_failed", error=str(exc), id=page_id)
                break
            if resp.status_code != 200:
                break
            body = resp.json()
            page = body.get("results") or []
            results.extend(page)
            if not body.get("has_more"):
                break
            cursor = body.get("next_cursor")
            if not cursor:
                break
        return results

    # ------------------------------------------------------------------
    # 4. normalization
    # ------------------------------------------------------------------

    async def normalize(
        self,
        event: WebhookEvent,
        hydrated: Mapping[str, Any],
    ) -> NormalizationResult:
        resource_type, resource_id = _resource_from_event(event)
        if not resource_type or not resource_id:
            return NormalizationResult(skipped_reason="notion event missing resource id")

        entity = hydrated.get("entity") or {}
        if not isinstance(entity, dict):
            entity = {}

        last_edited_time = (
            entity.get("last_edited_time")
            or _hint(event, "last_edited_time")
            or _utcnow_iso()
        )
        created_time = entity.get("created_time") or last_edited_time
        workspace_id = (
            _hint(event, "workspace_id")
            or entity.get("workspace_id")
            or _DEFAULT_WORKSPACE_PRINCIPAL
        )

        last_edited_by = (entity.get("last_edited_by") or {}).get("id") or "unknown"
        created_at = _parse_iso8601(created_time)
        updated_at = _parse_iso8601(last_edited_time)

        is_page = resource_type == "page"
        # _hint() only reads from raw_payload/data, not from parse_hint, so
        # we derive is_delete directly from the webhook event_type. Synthetic
        # polls never represent deletes (they're update-only by construction).
        raw_event_type = event.raw_payload.get("type")
        is_delete = raw_event_type in _DELETE_EVENT_TYPES
        doc_id = f"notion:{resource_type}:{resource_id}"
        doc_type = DocType.NOTION_PAGE if is_page else DocType.NOTION_DATABASE

        properties = entity.get("properties") or {}
        title = _title_from_properties(properties) or _title_from_database(entity)
        body = hydrated.get("body_markdown") or ""
        if not is_page and not body:
            body = _database_schema_summary(entity)

        source_url = entity.get("url") or _fallback_url(resource_id)
        deleted_at: datetime | None = None
        if is_delete:
            body = ""
            deleted_at = event.received_at
            content_hash = _sha256(
                f"{doc_id}|__deleted__|{event.received_at.isoformat()}"
            )
        else:
            content_hash = _sha256(
                f"{doc_id}|{last_edited_time}|{title or ''}|{body}"
            )

        parent = entity.get("parent") or {}
        parent_id = (
            parent.get("page_id")
            or parent.get("database_id")
            or parent.get("workspace")
        )

        mentioned_user_ids: list[str] = []
        raw_mentions = hydrated.get("mentioned_user_ids") or []
        if isinstance(raw_mentions, list):
            mentioned_user_ids = [str(u) for u in raw_mentions if u]

        permissions_raw = hydrated.get("permissions") or []
        acl_principals, acl_rows = _build_acl(
            resource_type=resource_type,
            resource_id=resource_id,
            workspace_id=str(workspace_id),
            parent_id=parent_id if isinstance(parent_id, str) else None,
            permissions=permissions_raw if isinstance(permissions_raw, list) else [],
            valid_from=updated_at,
        )

        doc = Document(
            doc_id=doc_id,
            customer_id=event.customer_id,
            source_system=SourceSystem.NOTION,
            source_id=f"{resource_type}:{resource_id}",
            source_url=source_url,
            doc_class=DocClass.RAW_SOURCE,
            doc_type=doc_type,
            content_type="text/markdown",
            content_hash=content_hash,
            title=title,
            body_preview=body[:280] if body else None,
            body_size_bytes=len(body.encode("utf-8")),
            body_token_count=count_tokens(body),
            author_id=last_edited_by,
            created_at=created_at,
            updated_at=updated_at,
            valid_from=updated_at,
            deleted_at=deleted_at,
            ingested_at=datetime.now(UTC),
            parent_doc_id=(
                f"notion:page:{parent.get('page_id')}"
                if parent.get("type") == "page_id" and parent.get("page_id")
                else (
                    f"notion:database:{parent.get('database_id')}"
                    if parent.get("type") == "database_id" and parent.get("database_id")
                    else None
                )
            ),
            acl=ACLSnapshot(principals=acl_principals, captured_at=event.received_at),
            metadata={
                "body": body,
                "workspace_id": workspace_id,
                "resource_type": resource_type,
                "properties": properties,
                "parent": parent,
                "archived": entity.get("archived", False),
                "mentioned_user_ids": mentioned_user_ids,
                "last_edited_time": last_edited_time,
                "hydrated": bool(entity),
            },
            doc_references=_references_from_parent(parent),
        )

        nodes: list[GraphNodeSpec] = [
            GraphNodeSpec(
                label=NodeLabel.DOCUMENT,
                canonical_id=doc_id,
                properties={
                    "doc_type": doc.doc_type.value,
                    "workspace_id": workspace_id,
                },
            ),
            GraphNodeSpec(
                label=NodeLabel.PERSON,
                canonical_id=last_edited_by,
                properties={"source_system": SourceSystem.NOTION.value},
            ),
        ]
        for uid in mentioned_user_ids:
            nodes.append(
                GraphNodeSpec(
                    label=NodeLabel.PERSON,
                    canonical_id=uid,
                    properties={"source_system": SourceSystem.NOTION.value},
                )
            )

        edges: list[GraphEdgeSpec] = [
            GraphEdgeSpec(
                edge_type=EdgeType.AUTHORED,
                from_label=NodeLabel.PERSON,
                from_canonical_id=last_edited_by,
                to_label=NodeLabel.DOCUMENT,
                to_canonical_id=doc_id,
                valid_from=updated_at,
            )
        ]
        for uid in mentioned_user_ids:
            edges.append(
                GraphEdgeSpec(
                    edge_type=EdgeType.MENTIONS,
                    from_label=NodeLabel.DOCUMENT,
                    from_canonical_id=doc_id,
                    to_label=NodeLabel.PERSON,
                    to_canonical_id=uid,
                    valid_from=updated_at,
                )
            )

        return NormalizationResult(
            documents=[doc],
            graph_nodes=nodes,
            graph_edges=edges,
            acl_snapshots=acl_rows,
        )

    # ------------------------------------------------------------------
    # 6. OAuth install + exchange
    # ------------------------------------------------------------------

    def oauth_install_url(
        self, customer_id: str, redirect_uri: str, state: str
    ) -> str:
        cid = self.settings.notion_client_id
        if not cid:
            from shared.exceptions import MissingSecret

            raise MissingSecret("NOTION_CLIENT_ID not configured")
        from urllib.parse import urlencode

        params = urlencode(
            {
                "client_id": cid,
                "redirect_uri": redirect_uri,
                "response_type": "code",
                "owner": "user",
                "state": state,
            }
        )
        return f"{_NOTION_OAUTH_AUTHORIZE}?{params}"

    async def exchange_oauth_code(
        self,
        code: str | None,
        redirect_uri: str,
        extra_params: dict[str, str] | None = None,
    ) -> IntegrationToken:
        from shared.exceptions import (
            InvalidWebhookPayload,
            MissingSecret,
            PermanentSourceError,
        )

        cid = self.settings.notion_client_id
        secret = self.settings.notion_client_secret
        if not cid or secret is None:
            raise MissingSecret(
                "NOTION_CLIENT_ID / NOTION_CLIENT_SECRET not configured"
            )
        if not code:
            raise InvalidWebhookPayload("notion oauth callback missing code")

        resp = await self.http.post(
            _NOTION_OAUTH_TOKEN,
            auth=(cid, secret.get_secret_value()),  # httpx handles Basic encoding
            headers={"Notion-Version": _NOTION_VERSION},
            json={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri,
            },
        )
        if resp.status_code != 200:
            raise PermanentSourceError(
                f"notion oauth failed: HTTP {resp.status_code} {resp.text[:200]}"
            )
        body = resp.json()
        return IntegrationToken(
            customer_id="",  # filled in by oauth/routes.py:callback
            source_system=SourceSystem.NOTION,
            access_token=body["access_token"],
            refresh_token=body.get("refresh_token"),
            scope=None,  # Notion uses per-integration capability checkboxes
            install_metadata={
                "workspace_id": body["workspace_id"],
                "workspace_name": body.get("workspace_name"),
                "workspace_icon": body.get("workspace_icon"),
                "bot_id": body.get("bot_id"),
                "owner": body.get("owner"),
            },
        )

    # ------------------------------------------------------------------
    # 7. workspace identification
    # ------------------------------------------------------------------

    async def identify_workspaces(self, token):  # type: ignore[override]
        """Notion's OAuth response includes workspace_id + workspace_name
        directly. Since we don't currently capture those during
        `exchange_oauth_code`, fall back to `/v1/users/me` which returns
        the bot's workspace affiliation via `bot.workspace_name` on
        recent API versions.
        """
        from shared.logging import get_logger
        from shared.models import ExternalWorkspaceRef

        lg = get_logger(__name__)
        try:
            resp = await self.http.get(
                "https://api.notion.com/v1/users/me",
                headers={
                    "Authorization": f"Bearer {token.access_token}",
                    "Notion-Version": "2022-06-28",
                },
            )
        except Exception as exc:
            lg.warning("notion.identify_workspaces_failed", error=str(exc))
            return []
        if resp.status_code != 200:
            return []
        body = resp.json()
        bot = body.get("bot") or {}
        ws_id = (
            bot.get("workspace_id")
            or bot.get("workspace")
            or body.get("workspace_id")
        )
        if not ws_id:
            return []
        return [
            ExternalWorkspaceRef(
                external_id=str(ws_id),
                external_name=bot.get("workspace_name"),
                metadata={"owner_user_id": (bot.get("owner") or {}).get("user", {}).get("id")},
            )
        ]

    def extract_external_id_from_payload(self, headers, raw_payload):
        # Notion official webhooks carry `workspace_id` at top level.
        # Synthetic poll shape includes it via the hint we produce in parse.
        wid = raw_payload.get("workspace_id")
        if not wid and isinstance(raw_payload.get("entity"), dict):
            wid = raw_payload["entity"].get("workspace_id")
        return str(wid) if wid else None

    # ------------------------------------------------------------------
    # backfill
    # ------------------------------------------------------------------

    async def backfill(
        self,
        customer_id: str,
        token,
        cursor: str | None = None,
    ):
        """Paginated `/search` → synthetic page.updated events.

        Emits events shaped like real Notion webhooks. The normalizer's
        `fetch_supplementary` path will hit Notion again for full block
        content, so we don't need to fully hydrate here — just enqueue
        the entity reference.
        """
        from shared.models import WebhookEvent

        workspace_id = await _fetch_workspace_id(self.http, token.access_token) or "unknown"
        next_cursor = cursor
        headers = {
            "Authorization": f"Bearer {token.access_token}",
            "Notion-Version": _NOTION_VERSION,
            "Content-Type": "application/json",
        }

        while True:
            body_json: dict[str, Any] = {"page_size": 50}
            if next_cursor:
                body_json["start_cursor"] = next_cursor

            try:
                resp = await self.http.post(
                    f"{_NOTION_API}/search", headers=headers, json=body_json
                )
            except Exception as exc:
                log.warning("notion.backfill_http_error", error=str(exc))
                return
            if resp.status_code != 200:
                return
            body = resp.json()
            for result in body.get("results", []):
                entity_type = result.get("object")  # "page" | "database"
                if entity_type not in {"page", "database"}:
                    continue
                payload = {
                    "type": f"{entity_type}.updated",
                    "entity": {
                        "type": entity_type,
                        "id": result.get("id"),
                        "last_edited_time": result.get("last_edited_time"),
                        "workspace_id": workspace_id,
                    },
                    "data": result,
                    "workspace_id": workspace_id,
                    "_cursor": next_cursor,
                }
                eid = result.get("id") or ""
                last_edited = result.get("last_edited_time") or ""
                yield WebhookEvent(
                    customer_id=customer_id,
                    source_system=SourceSystem.NOTION,
                    source_event_id=f"{entity_type}:{eid}:{last_edited}",
                    received_at=_parse_iso(last_edited) or datetime.now(UTC),
                    payload_s3_key="",
                    raw_payload=payload,
                    headers={},
                )
            if not body.get("has_more"):
                return
            next_cursor = body.get("next_cursor")
            if not next_cursor:
                return

    # ------------------------------------------------------------------


async def _fetch_workspace_id(http, token: str) -> str | None:
    try:
        resp = await http.get(
            f"{_NOTION_API}/users/me",
            headers={"Authorization": f"Bearer {token}", "Notion-Version": _NOTION_VERSION},
        )
    except Exception:
        return None
    if resp.status_code != 200:
        return None
    body = resp.json()
    bot = body.get("bot") or {}
    return bot.get("workspace_id") or bot.get("workspace") or body.get("workspace_id")


def _parse_iso(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


# ---- ACL construction ------------------------------------------------------


def _build_acl(
    *,
    resource_type: str,
    resource_id: str,
    workspace_id: str,
    parent_id: str | None,
    permissions: list[Any],
    valid_from: datetime,
) -> tuple[list[ACLPrincipal], list[ACLSnapshotRow]]:
    """Produce ACL principals (embedded in Document) + wide-row snapshots.

    Always includes the workspace-wide READ fallback. If `permissions` is
    non-empty we add USER/GROUP rows on top — more specific, but additive:
    Phase 0 captures, Phase 1 will enforce with parent-chain inheritance.
    """
    resource_type_label = f"notion.{resource_type}"

    principals: list[ACLPrincipal] = [
        ACLPrincipal(
            principal_type=PrincipalType.WORKSPACE,
            principal_id=workspace_id,
            permission=Permission.READ,
        )
    ]
    rows: list[ACLSnapshotRow] = [
        ACLSnapshotRow(
            source_system=SourceSystem.NOTION,
            principal_type=PrincipalType.WORKSPACE,
            principal_id=workspace_id,
            resource_type=resource_type_label,
            resource_id=resource_id,
            permission=Permission.READ,
            valid_from=valid_from,
            metadata={
                "parent_id": parent_id,
                "inherits": True,
                "source": "workspace_default",
            },
        )
    ]

    for entry in permissions:
        if not isinstance(entry, dict):
            continue
        ptype, pid = _classify_permission_entry(entry)
        if ptype is None or pid is None:
            continue
        permission = _role_to_permission(entry.get("role"))
        principals.append(
            ACLPrincipal(
                principal_type=ptype,
                principal_id=pid,
                permission=permission,
            )
        )
        rows.append(
            ACLSnapshotRow(
                source_system=SourceSystem.NOTION,
                principal_type=ptype,
                principal_id=pid,
                resource_type=resource_type_label,
                resource_id=resource_id,
                permission=permission,
                valid_from=valid_from,
                metadata={
                    "parent_id": parent_id,
                    "inherits": True,
                    "source": "page_permissions",
                    "role": entry.get("role"),
                },
            )
        )

    return principals, rows


def _classify_permission_entry(
    entry: Mapping[str, Any],
) -> tuple[PrincipalType | None, str | None]:
    # Notion exposes several shapes depending on integration capability.
    # Normalize the common ones here.
    etype = entry.get("type")
    if etype in {"user_permissions", "user"}:
        uid = entry.get("user_id") or (entry.get("user") or {}).get("id")
        return (PrincipalType.USER, uid) if uid else (None, None)
    if etype in {"group_permissions", "group"}:
        gid = entry.get("group_id") or (entry.get("group") or {}).get("id")
        return (PrincipalType.GROUP, gid) if gid else (None, None)
    # Fallback: look for explicit id fields without a typed wrapper.
    if entry.get("user_id"):
        return (PrincipalType.USER, entry["user_id"])
    if entry.get("group_id"):
        return (PrincipalType.GROUP, entry["group_id"])
    return (None, None)


def _role_to_permission(role: Any) -> Permission:
    role_str = str(role or "").lower()
    if role_str in {"editor", "full_access", "owner"}:
        return Permission.WRITE
    if role_str == "admin":
        return Permission.ADMIN
    return Permission.READ


# ---- helpers ---------------------------------------------------------------


def _header(headers: Mapping[str, str], name: str) -> str | None:
    for k, v in headers.items():
        if k.lower() == name.lower():
            return v
    return None


def _parse_iso8601(value: Any) -> datetime:
    """Parse a Notion ISO-8601 timestamp. Notion returns trailing `Z`.

    `datetime.fromisoformat` accepts `Z` directly on 3.12, but swap it for
    `+00:00` explicitly to stay portable across patches.
    """
    if not value:
        return datetime.now(UTC)
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    text = str(value).replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return datetime.now(UTC)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _resource_from_event(event: WebhookEvent) -> tuple[str | None, str | None]:
    payload = event.raw_payload
    if _is_notion_webhook(payload):
        entity = payload.get("entity") or {}
        return entity.get("type"), entity.get("id")
    if _is_synthetic_poll(payload):
        return payload.get("resource_type"), payload.get("resource_id")
    return None, None


def _hint(event: WebhookEvent, key: str) -> Any:
    # parse_hint isn't persisted on WebhookEvent, but the raw payload has
    # the same fields we put into the hint.
    payload = event.raw_payload
    if _is_synthetic_poll(payload):
        return payload.get(key)
    if _is_notion_webhook(payload):
        data = payload.get("data") or {}
        if key in data:
            return data[key]
        return payload.get(key)
    return None


def _auth_headers(token: IntegrationToken) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token.access_token}",
        "Notion-Version": _NOTION_VERSION,
    }


def _fallback_url(resource_id: str) -> str:
    # Strip dashes from Notion UUIDs for their canonical URL shape.
    slug = resource_id.replace("-", "")
    return f"https://www.notion.so/{slug}"


def _references_from_parent(parent: Mapping[str, Any]) -> list[DocRef]:
    ptype = parent.get("type")
    if ptype == "page_id" and parent.get("page_id"):
        parent_id = parent["page_id"]
        return [
            DocRef(
                doc_id=f"notion:page:{parent_id}",
                external_url=_fallback_url(parent_id),
                ref_type=RefType.LINKS_TO,
            )
        ]
    if ptype == "database_id" and parent.get("database_id"):
        parent_id = parent["database_id"]
        return [
            DocRef(
                doc_id=f"notion:database:{parent_id}",
                external_url=_fallback_url(parent_id),
                ref_type=RefType.LINKS_TO,
            )
        ]
    return []


def _title_from_database(entity: Mapping[str, Any]) -> str | None:
    # Databases expose title as a top-level `title` rich_text array.
    title = entity.get("title")
    if isinstance(title, list):
        val = _rich_text_to_plain(title).strip()
        if val:
            return val[:200]
    return None


def _database_schema_summary(entity: Mapping[str, Any]) -> str:
    """Flatten a database description + property schema into a plain-text blob."""
    parts: list[str] = []
    description = entity.get("description")
    if isinstance(description, list):
        desc_text = _rich_text_to_plain(description).strip()
        if desc_text:
            parts.append(desc_text)
    properties = entity.get("properties") or {}
    if isinstance(properties, dict) and properties:
        parts.append("Properties:")
        for name, prop in properties.items():
            ptype = prop.get("type") if isinstance(prop, dict) else "unknown"
            parts.append(f"- {name} ({ptype})")
    return "\n".join(parts)


def httpx_error_types() -> tuple[type[BaseException], ...]:
    """Tuple of httpx error classes we catch during hydration.

    Kept as a tiny helper so the catch-list is explicit — no bare `except Exception`.
    """
    import httpx

    return (httpx.HTTPError, httpx.InvalidURL, OSError)


__all__ = ["NotionConnector", "blocks_to_markdown"]
