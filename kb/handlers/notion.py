"""Notion connector — page + database + data_source ingestion.

Covers two inbound shapes (Phase 0 treats them uniformly):

1. Notion's official webhook payload (2026-03-11 API version): top-level
   `type` like `"page.content_updated"` / `"page.properties_updated"` /
   `"data_source.schema_updated"` + an `entity` dict `{type, id}` + a
   `data` dict whose shape varies per event type. Real Notion never puts
   `last_edited_time` in `data` — that field comes from the hydrated
   REST entity (`/v1/pages/{id}` / `/v1/databases/{id}` /
   `/v1/data_sources/{id}`).

   See https://developers.notion.com/reference/webhooks-events-delivery
   for the canonical event schema and ../reference/versioning for the
   API version semantics.

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
import json
from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar

from engine.ingest.chunker import count_tokens
from engine.ingest.handlers.base import Connector
from engine.ingest.handlers.registry import register_connector
from engine.shared.constants import (
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
from engine.shared.exceptions import InvalidWebhookPayload
from engine.shared.logging import get_logger
from engine.shared.models import (
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

# Pinned to 2026-03-11 — the Notion API version our webhook subscriptions
# are configured against. See https://developers.notion.com/reference/versioning
# for the version semantics. Practical implications for this connector:
#
#   • Pages: GET /v1/pages/{id} unchanged.
#   • Databases: GET /v1/databases/{id} now returns a container shape
#     {title, icon, cover, data_sources: [...]} — NO `properties`. The
#     row-schema moved to per-data-source records.
#   • Data sources: NEW endpoint GET /v1/data_sources/{id} carrying
#     {object: "data_source", id, properties, parent: {database_id, ...},
#     database_parent}. We hit this for `data_source.*` webhook events.
#   • Querying: POST /v1/databases/{id}/query is REMOVED. Row enumeration
#     for backfill must use POST /v1/data_sources/{id}/query instead —
#     see the TODO in `_iter_database_rows`.
#   • Field rename: `archived` → `in_trash` across pages/databases/blocks/
#     data sources.
#   • Block rename: `transcription` → `meeting_notes` (we render unknowns
#     as `[block:<type>]` placeholders so the rename is benign).
_NOTION_VERSION = "2026-03-11"

# Webhook event types we persist. Mirrors the 2025-09-03 Notion webhook
# spec — see https://developers.notion.com/reference/webhooks-events-delivery.
#
# `_DEFERRED_EVENT_TYPES` below names events Notion *does* emit but we
# don't ingest yet; keeping them in a named set (rather than letting them
# fall through to "unknown") makes the next maintainer's life easier when
# Notion adds another event type and "why is X being silently dropped"
# starts as a question instead of a bug.
_PAGE_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "page.created",
        "page.content_updated",
        "page.properties_updated",
        "page.moved",
        "page.locked",
        "page.unlocked",
        "page.deleted",
        "page.undeleted",
    }
)

_DATABASE_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "database.created",
        "database.deleted",
        "database.moved",
        "database.undeleted",
        # Deprecated in 2025-09-03 in favor of data_source.* — still emitted
        # to subscriptions configured on older API versions, so we keep them.
        "database.content_updated",
        "database.schema_updated",
    }
)

# data_source.* events were introduced in API version 2025-09-03 to replace
# the database.content_updated / database.schema_updated event surface.
# Subscriptions on the new API version emit these instead of (not in
# addition to) the deprecated database events. With `_NOTION_VERSION`
# bumped to 2026-03-11, hydration calls /v1/data_sources/{id} for these
# events and gets back the schema/properties directly.
_DATA_SOURCE_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "data_source.created",
        "data_source.content_updated",
        "data_source.schema_updated",
        "data_source.moved",
        "data_source.deleted",
        "data_source.undeleted",
    }
)

_ACCEPTED_EVENT_TYPES: frozenset[str] = (
    _PAGE_EVENT_TYPES | _DATABASE_EVENT_TYPES | _DATA_SOURCE_EVENT_TYPES
)

# Events Notion emits that we explicitly recognize but choose not to ingest
# yet. parse_webhook_event returns None for these; the ingestion entry
# point logs the receipt with `status: ignored`. Listed by name so a
# future spec change shows up as "added event type X" rather than as
# silent traffic that disappears.
_DEFERRED_EVENT_TYPES: frozenset[str] = frozenset(
    {
        # Comments require a separate fetch model (/v1/comments?block_id=)
        # and a different document shape; out of scope for the connector
        # as it stands today.
        "comment.created",
        "comment.updated",
        "comment.deleted",
    }
)

_DELETE_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "page.deleted",
        "database.deleted",
        "data_source.deleted",
    }
)

_DEFAULT_WORKSPACE_PRINCIPAL = "notion-default"

# Token-exchange endpoint. The public-facing /oauth/notion/install +
# /oauth/notion/callback live in prbe-backend's gateway; this endpoint
# is what the gateway POSTs to (via /api/oauth/notion/exchange) once it
# has a verified-state callback in hand.
_NOTION_OAUTH_TOKEN = "https://api.notion.com/v1/oauth/token"
_NOTION_API_BASE = "https://api.notion.com/v1"


def _expires_at_from_expires_in(expires_in: Any) -> datetime | None:
    """Notion returns `expires_in` as seconds-from-now (int) on token-rotated
    integrations. Convert to a UTC datetime so the standard
    `integration_tokens.expires_at` column can drive `list_tokens_expiring_within`
    for proactive refresh. Returns None when the field is absent (legacy
    long-lived token path) or malformed."""
    try:
        seconds = int(expires_in)
    except (TypeError, ValueError):
        return None
    if seconds <= 0:
        return None
    return datetime.now(UTC) + timedelta(seconds=seconds)


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
    """Walk block rich_text spans, collect Notion user ids from @mentions.

    Recurses into `_children` attached by `_fetch_all_blocks` so mentions
    nested inside toggles / columns / sub-bullets still surface.
    """
    user_ids: list[str] = []

    def _walk(items: list[dict[str, Any]]) -> None:
        for block in items:
            btype = block.get("type")
            payload = block.get(btype) if btype else None
            if isinstance(payload, dict):
                rich = payload.get("rich_text") or []
                for span in rich:
                    if span.get("type") != "mention":
                        continue
                    mention = span.get("mention") or {}
                    if mention.get("type") == "user":
                        uid = (mention.get("user") or {}).get("id")
                        if uid and uid not in user_ids:
                            user_ids.append(uid)
            children = block.get("_children")
            if isinstance(children, list):
                _walk(children)

    _walk(blocks)
    return user_ids


# Block types whose content lives entirely in `_children` (no own rich_text
# worth rendering). Used by `blocks_to_markdown` to suppress placeholder noise.
_CONTAINER_BLOCK_TYPES: frozenset[str] = frozenset(
    {"column_list", "column", "synced_block"}
)


def blocks_to_markdown(
    blocks: list[dict[str, Any]],
    *,
    _depth: int = 0,
) -> str:
    """Convert Notion blocks (with optional `_children`) to markdown-ish text.

    Handles: paragraph, heading_1/2/3, bulleted_list_item, numbered_list_item,
    to_do, code, quote, toggle, callout, child_page, child_database. Container
    blocks (column_list / column / synced_block) emit nothing themselves but
    descend into `_children`. Unknown leaf block types emit `[block:<type>]`
    so the downstream chunker still sees a stable placeholder.

    `_children` is populated by `_fetch_all_blocks` for any block whose Notion
    `has_children` flag is true. Without that recursion, content inside a
    toggle or column was silently dropped — that's the bulk of the "missing
    docs" the user reported on Notion ingestion.
    """
    indent = "  " * _depth
    lines: list[str] = []
    numbered_counter = 0
    for block in blocks:
        btype = block.get("type") or ""
        payload = block.get(btype)
        if not isinstance(payload, dict):
            lines.append(f"{indent}[block:{btype or 'unknown'}]")
            numbered_counter = 0
        else:
            text = _rich_text_to_plain(payload.get("rich_text"))

            if btype == "paragraph":
                lines.append(f"{indent}{text}")
                numbered_counter = 0
            elif btype == "heading_1":
                lines.append(f"{indent}# {text}")
                numbered_counter = 0
            elif btype == "heading_2":
                lines.append(f"{indent}## {text}")
                numbered_counter = 0
            elif btype == "heading_3":
                lines.append(f"{indent}### {text}")
                numbered_counter = 0
            elif btype == "bulleted_list_item":
                lines.append(f"{indent}- {text}")
                numbered_counter = 0
            elif btype == "numbered_list_item":
                numbered_counter += 1
                lines.append(f"{indent}{numbered_counter}. {text}")
            elif btype == "to_do":
                checked = payload.get("checked", False)
                box = "[x]" if checked else "[ ]"
                lines.append(f"{indent}- {box} {text}")
                numbered_counter = 0
            elif btype == "code":
                lang = payload.get("language") or ""
                lines.append(f"{indent}```{lang}\n{indent}{text}\n{indent}```")
                numbered_counter = 0
            elif btype == "quote":
                lines.append(f"{indent}> {text}")
                numbered_counter = 0
            elif btype == "toggle":
                lines.append(f"{indent}> {text}" if text else f"{indent}> ▼")
                numbered_counter = 0
            elif btype == "callout":
                lines.append(f"{indent}> {text}")
                numbered_counter = 0
            elif btype == "child_page":
                title = payload.get("title") or ""
                lines.append(f"{indent}[child_page: {title}]")
                numbered_counter = 0
            elif btype == "child_database":
                title = payload.get("title") or ""
                lines.append(f"{indent}[child_database: {title}]")
                numbered_counter = 0
            elif btype in _CONTAINER_BLOCK_TYPES:
                # No own line — children render inline at same depth.
                numbered_counter = 0
            else:
                lines.append(f"{indent}[block:{btype}]")
                numbered_counter = 0

        children = block.get("_children")
        if isinstance(children, list) and children:
            # Containers keep children at the same depth so multi-column /
            # synced content reads as flat prose; everything else nests.
            child_depth = _depth if btype in _CONTAINER_BLOCK_TYPES else _depth + 1
            child_md = blocks_to_markdown(children, _depth=child_depth)
            if child_md:
                lines.append(child_md)

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
    doc_type_prefix: ClassVar[str] = "notion."

    # ------------------------------------------------------------------
    # 1. signature verification
    # ------------------------------------------------------------------

    def verify_signature(
        self,
        headers: Mapping[str, str],
        raw_body: bytes,
    ) -> bool:
        """HMAC-SHA256 the body with the subscription's verification token.

        Per Notion's webhook docs, the signing key is the per-subscription
        `verification_token` (received in the one-time `{"verification_token":
        "..."}` handshake POST when the subscription is wired up), NOT the
        OAuth client secret. The token is stored in
        `notion_webhook_verification_token`.

        Local dev accepts unsigned to keep tunnels working. Prod requires
        a signed payload — return False (caller -> 401) when the token isn't
        configured yet so a misconfigured environment can't silently accept
        forged events.
        """
        if self.settings.is_local:
            return True

        sig = _header(headers, "x-notion-signature")
        if sig is None:
            return False

        token = self.settings.notion_webhook_verification_token
        if token is None:
            return False

        expected = hmac.new(
            token.get_secret_value().encode(),
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
        if event_type in _DEFERRED_EVENT_TYPES:
            # Recognized event type we haven't wired up yet (comments today).
            # Single info log so operators can see drops are intentional, not
            # a bug, and so a future "why isn't this ingesting" question
            # answers itself in `fly logs`.
            log.info(
                "notion.webhook_deferred",
                event_type=event_type,
                entity_id=(raw_payload.get("entity") or {}).get("id"),
            )
            return None
        if event_type not in _ACCEPTED_EVENT_TYPES:
            # Truly unknown event type. Skip silently — Notion may add new
            # event types without deprecation, and a noisy log on every
            # delivery is worse than a missed signal.
            return None

        entity = raw_payload.get("entity")
        if not isinstance(entity, dict):
            raise InvalidWebhookPayload("notion webhook missing 'entity' dict")

        entity_type = entity.get("type")
        entity_id = entity.get("id")
        if entity_type not in {"page", "database", "data_source"} or not entity_id:
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
        # last_edited_time is per-second, so rapid block edits would collapse;
        # add a payload fingerprint to disambiguate while keeping retries idempotent.
        tail = "delete" if is_delete else "edit"
        payload_fp = hashlib.sha256(
            json.dumps(data, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:16]
        source_event_id = (
            f"{entity_type}:{entity_id}:{tail}:{last_edited_time}:{payload_fp}"
        )
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
        # Pages carry content blocks. Databases (containers) and data
        # sources (schema records) don't have a block tree — skip the
        # block-fetch path for both.
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
        # 2026-03-11 splits the legacy "database" resource into two: a
        # container at /v1/databases/{id} (carries title/icon/cover +
        # `data_sources: [...]`) and per-data-source records at
        # /v1/data_sources/{id} (carry the row schema as `properties`).
        # Each event type ends up at the corresponding endpoint.
        path_map = {
            "page": "pages",
            "database": "databases",
            "data_source": "data_sources",
        }
        path = path_map.get(resource_type)
        if path is None:
            log.warning(
                "notion.fetch_entity_unknown_type",
                resource_type=resource_type,
                id=resource_id,
            )
            return None
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
                resource_type=resource_type,
                id=resource_id,
            )
            return None
        body = resp.json()
        return body if isinstance(body, dict) else None

    async def _fetch_all_blocks(
        self,
        page_id: str,
        token: IntegrationToken,
        *,
        _depth: int = 0,
        _budget: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch a page's blocks, recursing into nested children.

        Notion's `/blocks/{id}/children` only returns DIRECT children. Anything
        inside a toggle, column, callout, synced_block, or list item with
        sub-bullets is hidden behind `has_children=true` and a separate fetch.
        Skipping that recursion is the main reason the chunker was seeing
        "some chunks but not everything" — entire collapsed sections vanished.

        `child_page` / `child_database` blocks are deliberately *not* descended
        into here: they're independent ingestion roots that surface through
        their own events from `/search` (and database row enumeration during
        backfill). Recursing would duplicate documents.

        Two safety nets keep a pathological page from DoS'ing the worker:
          - `_depth` cap (Notion technically permits arbitrary nesting; >25
            levels in real workspaces is essentially nonexistent).
          - shared `_budget` block-count cap so a single page can't fan out
            into millions of nested blocks.
        """
        if _budget is None:
            _budget = [50_000]
        if _depth > 25:
            log.warning("notion.fetch_blocks_depth_cap", id=page_id)
            return []

        results: list[dict[str, Any]] = []
        cursor: str | None = None
        while _budget[0] > 0:
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
            for block in page:
                _budget[0] -= 1
                if _budget[0] < 0:
                    log.warning("notion.fetch_blocks_budget_exhausted", id=page_id)
                    break
                btype = block.get("type")
                if (
                    block.get("has_children")
                    and btype not in {"child_page", "child_database"}
                ):
                    child_id = block.get("id")
                    if child_id:
                        block["_children"] = await self._fetch_all_blocks(
                            child_id,
                            token,
                            _depth=_depth + 1,
                            _budget=_budget,
                        )
                results.append(block)
            if _budget[0] < 0:
                break
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

        # Synth corpus bypass: when hydration produced no entity (no live
        # OAuth fetch — synth has no real Notion token), fall back to the
        # entity inlined on the raw webhook payload. scripts/synth/output/
        # notion.py inlines properties.title and body_markdown there so the
        # corpus can be ingested without calling Notion's API. Real Notion
        # webhooks ship `entity` with only {type, id}; that shape doesn't
        # carry properties or body_markdown, so the title + body extraction
        # below produces the same nulls/empties prod sees today, and the
        # behavior is unchanged for real traffic.
        if not entity:
            raw_entity = event.raw_payload.get("entity")
            if isinstance(raw_entity, dict):
                entity = raw_entity

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
        # Synth corpus bypass: scripts/synth/output/notion.py inlines a
        # pre-rendered body_markdown on `entity` so the synth corpus can be
        # ingested without a live Notion OAuth token (the prod hydration path
        # would 401 against Notion's API). Real Notion webhooks NEVER set
        # entity.body_markdown, so this fallback is a no-op on prod traffic.
        body = (
            hydrated.get("body_markdown")
            or entity.get("body_markdown")
            or ""
        )
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
                "workspace_id": workspace_id,
                "resource_type": resource_type,
                "properties": properties,
                "parent": parent,
                # Notion renamed `archived` → `in_trash` in 2026-03-11.
                # Read `in_trash` first; fall back to `archived` for any
                # legacy webhook envelopes captured before the upgrade.
                "in_trash": (
                    entity.get("in_trash")
                    if entity.get("in_trash") is not None
                    else entity.get("archived", False)
                ),
                "mentioned_user_ids": mentioned_user_ids,
                "last_edited_time": last_edited_time,
                # `hydrated` reflects whether fetch_supplementary returned data —
                # NOT whether `entity` happens to be non-empty (a raw webhook
                # payload always has entity={type, id}, and the synth corpus
                # bypass populates entity from the raw payload too). Use the
                # presence of the hydrated dict itself as the signal.
                "hydrated": bool(hydrated),
            },
            body=body,
            doc_references=_references_from_parent(parent),
        )

        nodes: list[GraphNodeSpec] = [
            GraphNodeSpec(
                label=NodeLabel.DOCUMENT,
                canonical_id=doc_id,
                properties={
                    "doc_type": doc.doc_type,
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
    # 6. OAuth code-for-token exchange
    # ------------------------------------------------------------------
    # Public-facing /oauth/notion/install + /oauth/notion/callback live in
    # prbe-backend's gateway (api.prbe.ai). After backend verifies the
    # signed state, it POSTs (customer_id, code, redirect_uri, extra_params)
    # to /api/oauth/notion/exchange (admin_routes.py), which calls into
    # this method.

    async def exchange_oauth_code(
        self,
        code: str | None,
        redirect_uri: str,
        extra_params: dict[str, str] | None = None,
    ) -> IntegrationToken:
        from engine.shared.exceptions import (
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
        # access_token + workspace_id are documented non-null on a 200 response,
        # but guard explicitly so a malformed body becomes a 502 (PermanentSourceError)
        # rather than an unhandled KeyError → 500.
        try:
            access_token = body["access_token"]
            workspace_id = body["workspace_id"]
        except (KeyError, TypeError) as exc:
            raise PermanentSourceError(
                f"notion oauth response missing required fields: {exc}"
            ) from exc
        return IntegrationToken(
            customer_id="",  # filled in by /api/oauth/notion/exchange caller
            source_system=SourceSystem.NOTION,
            access_token=access_token,
            refresh_token=body.get("refresh_token"),
            # Notion returns `expires_in` only when the integration is
            # configured for token rotation (post-Sep-2024 feature). Without
            # rotation, tokens are long-lived; expires_at stays None and the
            # refresh cron leaves the row alone.
            expires_at=_expires_at_from_expires_in(body.get("expires_in")),
            scope=None,  # Notion uses per-integration capability checkboxes
            install_metadata={
                "workspace_id": workspace_id,
                "workspace_name": body.get("workspace_name"),
                "workspace_icon": body.get("workspace_icon"),
                "bot_id": body.get("bot_id"),
                "owner": body.get("owner"),
            },
        )

    async def exchange_refresh_token(
        self, token: IntegrationToken
    ) -> IntegrationToken:
        """Exchange the persisted refresh_token for a new access_token.

        Method name matches `scripts/cron_token_refresh.py`'s convention so
        Notion automatically participates in the existing refresh cron once
        a token with a refresh_token and `expires_at <= now+1h` exists.

        Notion's token rotation: when the integration is configured with
        rotated tokens, the exchange response includes refresh_token +
        expires_in (~1 hour for the access_token). Legacy long-lived
        integrations have no refresh_token; this method raises
        PermanentSourceError so the caller can flag auth_failed.
        """
        from engine.shared.exceptions import (
            MissingSecret,
            PermanentSourceError,
            TransientSourceError,
        )

        if not token.refresh_token:
            raise PermanentSourceError(
                "notion exchange_refresh_token called without a stored refresh_token"
                " — integration predates token rotation; reconnect required",
            )
        cid = self.settings.notion_client_id
        secret = self.settings.notion_client_secret
        if not cid or secret is None:
            raise MissingSecret(
                "NOTION_CLIENT_ID / NOTION_CLIENT_SECRET not configured"
            )
        resp = await self.http.post(
            _NOTION_OAUTH_TOKEN,
            auth=(cid, secret.get_secret_value()),
            headers={"Notion-Version": _NOTION_VERSION},
            json={
                "grant_type": "refresh_token",
                "refresh_token": token.refresh_token,
            },
        )
        if resp.status_code >= 500:
            raise TransientSourceError(
                f"notion /oauth/token (refresh) returned {resp.status_code}",
                status=resp.status_code,
                body=resp.text[:500],
            )
        if resp.status_code >= 400:
            raise PermanentSourceError(
                f"notion /oauth/token (refresh) returned {resp.status_code}",
                status=resp.status_code,
                body=resp.text[:500],
            )
        body = resp.json()
        new_access = body.get("access_token")
        if not new_access:
            raise PermanentSourceError(
                "notion refresh_token 200 but missing access_token",
                body=str(body)[:500],
            )
        return IntegrationToken(
            customer_id=token.customer_id,
            source_system=SourceSystem.NOTION,
            access_token=new_access,
            # Notion rotates refresh_tokens on every refresh per their docs;
            # fall back to the old one if the response omits it (defensive).
            refresh_token=body.get("refresh_token") or token.refresh_token,
            expires_at=_expires_at_from_expires_in(body.get("expires_in")),
        )

    async def verify_token_health(self, token: IntegrationToken) -> bool:
        """Liveness probe: GET `/v1/users/me` and return True on success.

        Returns False when Notion responds 401 (token revoked / invalidated
        out of band). Any other status — including transient 5xx — propagates
        as a raised exception so the caller can distinguish "definitely-bad"
        from "we-don't-know". Used by the periodic token-health cron to
        flip `integration_tokens.status` from `active` to `auth_failed`
        without waiting for the next webhook delivery to fail.

        Notion returns 401 with `{"code": "unauthorized"}` when the token
        is invalidated. We check status_code only; the body's `code` field
        is consistent across revocation modes (user uninstalled,
        integration deleted, admin revoked).
        """
        resp = await self.http.get(
            f"{_NOTION_API_BASE}/users/me",
            headers={
                "Authorization": f"Bearer {token.access_token}",
                "Notion-Version": _NOTION_VERSION,
            },
        )
        if resp.status_code == 401:
            return False
        if resp.status_code != 200:
            from engine.shared.exceptions import TransientSourceError

            raise TransientSourceError(
                f"notion users/me probe returned {resp.status_code}",
                status=resp.status_code,
                body=resp.text[:500],
            )
        return True

    # ------------------------------------------------------------------
    # 7. workspace identification
    # ------------------------------------------------------------------

    async def identify_workspaces(self, token):  # type: ignore[override]
        """Read workspace info captured during `exchange_oauth_code`.

        The Notion token-exchange response gives us `workspace_id` +
        `workspace_name` + `bot_id` directly, so we stash that in
        `token.install_metadata` and read it here — no second API call,
        no fallback cascade needed.

        Returns [] when `install_metadata` is None (e.g., a token loaded
        from DB; the field is Pydantic-transient). The exchange caller
        already handles that case by skipping `record_mapping` and
        logging a warning.
        """
        from engine.shared.logging import get_logger
        from engine.shared.models import ExternalWorkspaceRef

        lg = get_logger(__name__)
        meta = token.install_metadata or {}
        ws_id = meta.get("workspace_id")
        if not ws_id:
            lg.warning(
                "notion.identify_workspaces_no_install_metadata",
                customer=token.customer_id,
            )
            return []

        owner_user_id = (
            ((meta.get("owner") or {}).get("user") or {}).get("id")
        )
        return [
            ExternalWorkspaceRef(
                external_id=str(ws_id),
                external_name=meta.get("workspace_name"),
                metadata={
                    "bot_id": meta.get("bot_id"),
                    "owner_user_id": owner_user_id,
                    "workspace_icon": meta.get("workspace_icon"),
                },
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
        """Paginated `/search` → synthetic page.created / database.created events.

        Emits events shaped like real Notion webhooks. The normalizer's
        `fetch_supplementary` path will hit Notion again for full block
        content, so we don't need to fully hydrate here — just enqueue
        the entity reference.

        Picked `*.created` (rather than the more semantically precise
        `*.content_updated`) because the latter requires `data.updated_blocks`
        per the spec; backfill doesn't have that list.

        For each database we encounter, we additionally page through
        `databases/{id}/query` and yield a `page.created` event per row.
        Rows in Notion are pages with their own block trees; they're not
        returned by `/search` unless individually shared with the integration,
        so without this enumeration every database's contents are invisible.
        """
        from engine.shared.models import WebhookEvent

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
                eid = result.get("id") or ""
                last_edited = result.get("last_edited_time") or ""
                payload = {
                    "type": f"{entity_type}.created",
                    "entity": {
                        "type": entity_type,
                        "id": eid,
                        "last_edited_time": last_edited,
                        "workspace_id": workspace_id,
                    },
                    "data": result,
                    "workspace_id": workspace_id,
                    "_cursor": next_cursor,
                }
                yield WebhookEvent(
                    customer_id=customer_id,
                    source_system=SourceSystem.NOTION,
                    source_event_id=f"{entity_type}:{eid}:{last_edited}",
                    received_at=_parse_iso(last_edited) or datetime.now(UTC),
                    payload_s3_key="",
                    raw_payload=payload,
                    headers={},
                )

                if entity_type == "database" and eid:
                    async for row in self._iter_database_rows(eid, headers):
                        row_id = row.get("id") or ""
                        if not row_id:
                            continue
                        row_last_edited = row.get("last_edited_time") or ""
                        row_payload = {
                            "type": "page.created",
                            "entity": {
                                "type": "page",
                                "id": row_id,
                                "last_edited_time": row_last_edited,
                                "workspace_id": workspace_id,
                            },
                            "data": row,
                            "workspace_id": workspace_id,
                            "_parent_database_id": eid,
                        }
                        yield WebhookEvent(
                            customer_id=customer_id,
                            source_system=SourceSystem.NOTION,
                            source_event_id=f"page:{row_id}:{row_last_edited}",
                            received_at=_parse_iso(row_last_edited) or datetime.now(UTC),
                            payload_s3_key="",
                            raw_payload=row_payload,
                            headers={},
                        )

            if not body.get("has_more"):
                return
            next_cursor = body.get("next_cursor")
            if not next_cursor:
                return

    async def _iter_database_rows(
        self,
        database_id: str,
        headers: Mapping[str, str],
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield every page row in a database, paginating `databases/{id}/query`.

        Read-only POST with no filter/sort body returns all rows the integration
        can see. Rate-limit / 4xx responses end the iteration cleanly so a single
        bad database doesn't abort the entire backfill.

        TODO(2026-03-11): /v1/databases/{id}/query is removed in this API
        version. The replacement is a two-step lookup:

            1. GET /v1/databases/{id} → read `data_sources: [{id, name}]`
            2. POST /v1/data_sources/{ds_id}/query for each source

        Backfill is initial-sync only and isn't on the hot path for
        webhook ingestion; deferring this change until backfill is next
        run on a 2026-03-11-bound integration. Today, calling this
        method against the live API will return 404.
        """
        next_cursor: str | None = None
        while True:
            body_json: dict[str, Any] = {"page_size": 100}
            if next_cursor:
                body_json["start_cursor"] = next_cursor
            try:
                resp = await self.http.post(
                    f"{_NOTION_API}/databases/{database_id}/query",
                    headers=dict(headers),
                    json=body_json,
                )
            except Exception as exc:
                log.warning(
                    "notion.query_database_http_error",
                    error=str(exc),
                    id=database_id,
                )
                return
            if resp.status_code != 200:
                log.warning(
                    "notion.query_database_non_200",
                    status=resp.status_code,
                    id=database_id,
                )
                return
            body = resp.json()
            for row in body.get("results") or []:
                if row.get("object") == "page":
                    yield row
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
    """Flatten a database description + property schema into a plain-text blob.

    In 2026-03-11, GET /v1/databases/{id} returns a *container* shape:
    `{title, icon, cover, data_sources: [...]}` with NO top-level
    `properties`. The schema moved to each entry under `data_sources`.
    For data_source events the entity comes back from /v1/data_sources/{id}
    and DOES carry `properties` directly.

    We summarize what's available — description plus either the per-source
    list (database container shape) or the property schema (data source
    or legacy database shape).
    """
    parts: list[str] = []
    description = entity.get("description")
    if isinstance(description, list):
        desc_text = _rich_text_to_plain(description).strip()
        if desc_text:
            parts.append(desc_text)

    # Database container (2026-03-11): list nested data_sources by name.
    data_sources = entity.get("data_sources")
    if isinstance(data_sources, list) and data_sources:
        parts.append("Data sources:")
        for ds in data_sources:
            if not isinstance(ds, dict):
                continue
            ds_name = ds.get("name") or ds.get("id") or "(unnamed)"
            parts.append(f"- {ds_name}")

    # Data source (2026-03-11) or legacy database (≤2025-09-03): row schema.
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
