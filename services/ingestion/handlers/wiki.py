"""Wiki connector — curated team-knowledge pages.

Source shape: pages flow in through `PUT /api/wiki/pages/{wiki_type}/{slug}`
on the ingestion service (see `wiki_routes.py`). There is no external webhook;
the route synthesizes a `WebhookEvent` whose `raw_payload[WIKI_PAYLOAD_KEY]`
carries everything `normalize` needs:

    {"wiki_page": {
        "wiki_type": "runbook" | "decision" | "feature" | "service_card",
        "slug": "<a-z0-9->",
        "title": "...",
        "body": "<markdown>",
        "frontmatter": {...},                  # optional
        "doc_class": "manual_entry"|...,      # optional, default MANUAL_ENTRY
        "author_id": "...",                    # optional
        "compiled_from_doc_ids": [...],        # optional, COMPILED_WIKI only
        "is_delete": false,                    # optional, soft-delete tombstone
        "updated_at": "<ISO8601>",             # optional, defaults to now
    }}

Cross-references in the body use `[[...]]` syntax — see `wiki_links.py`.
Typed links (`[[Person: X]]`, `[[Service: Y]]`, ...) emit graph nodes/edges so
wiki pages connect to the same canonical entities the other connectors populate.
Plain `[[Page]]` links are recorded in `metadata.dangling_links` for a future
lint job; we do not resolve to other wiki documents during ingestion (would
require a DB call inside `normalize`, which the rest of the pipeline avoids).

ACL: workspace-readable for now (any caller authenticated as the customer can
read every wiki page). Per-page ACLs are a Phase 2 concern.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, ClassVar

from services.ingestion.chunker import count_tokens
from services.ingestion.handlers.base import Connector
from services.ingestion.handlers.registry import register_connector
from services.ingestion.wiki_links import WikiPageLink, parse_page_links
from shared.constants import (
    CompileTrigger,
    DocClass,
    DocType,
    EdgeType,
    IngestionEventType,
    NodeLabel,
    Permission,
    PrincipalType,
    SourceSystem,
)
from shared.exceptions import InvalidWebhookPayload
from shared.logging import get_logger
from shared.models import (
    ACLPrincipal,
    ACLSnapshot,
    ACLSnapshotRow,
    Document,
    GraphEdgeSpec,
    GraphNodeSpec,
    NormalizationResult,
    WebhookEvent,
    WebhookParseResult,
)

log = get_logger(__name__)

# The payload key inside `WebhookEvent.raw_payload` that carries the upload.
WIKI_PAYLOAD_KEY = "wiki_page"

# URL / payload `wiki_type` string -> DocType enum value.
WIKI_TYPE_TO_DOC_TYPE: dict[str, DocType] = {
    "service_card": DocType.WIKI_SERVICE_CARD,
    "decision": DocType.WIKI_DECISION,
    "feature": DocType.WIKI_FEATURE,
    "runbook": DocType.WIKI_RUNBOOK,
}

# `[[<kind>: ...]]` -> (graph node label, edge type emitted from the wiki page).
#
# `[[Person: X]]` resolves to NodeLabel.WIKI_PERSON (not PERSON) — wiki page
# bodies carry only a rendered name, not a canonical platform id (e.g. a Slack
# user_id or GitHub login). Merging into the canonical PERSON namespace at
# ingest time would create false-canonical nodes whenever the rendered name
# happens to match an existing PERSON canonical_id by accident. A future alias
# resolver can fold WIKI_PERSON into PERSON via fuzzy match. The other typed
# refs use canonical-shaped ids (slugs, repo names, ticket numbers) and slot
# straight into their canonical NodeLabel.
_LINK_NODE_MAP: dict[str, tuple[NodeLabel, EdgeType]] = {
    "person": (NodeLabel.WIKI_PERSON, EdgeType.MENTIONS),
    "service": (NodeLabel.SERVICE, EdgeType.DESCRIBES),
    "repo": (NodeLabel.REPO, EdgeType.DESCRIBES),
    "ticket": (NodeLabel.TICKET, EdgeType.DESCRIBES),
    "feature": (NodeLabel.FEATURE, EdgeType.DESCRIBES),
    "decision": (NodeLabel.DECISION, EdgeType.DESCRIBES),
}


@register_connector(SourceSystem.WIKI)
class WikiConnector(Connector):
    source_system: ClassVar[SourceSystem] = SourceSystem.WIKI
    display_name: ClassVar[str] = "Wiki"

    def verify_signature(
        self,
        headers: Mapping[str, str],
        raw_body: bytes,
    ) -> bool:
        # No external webhook surface. Uploads land on internal /api/wiki/*
        # which are gated by X-Internal-Knowledge-Key. Returning False keeps
        # /webhooks/wiki a hard 401 for anyone who tries.
        return False

    def parse_webhook_event(
        self,
        customer_id: str,
        headers: Mapping[str, str],
        raw_payload: Mapping[str, Any],
    ) -> WebhookParseResult | None:
        wiki = raw_payload.get(WIKI_PAYLOAD_KEY)
        if not isinstance(wiki, Mapping):
            raise InvalidWebhookPayload(f"wiki payload missing '{WIKI_PAYLOAD_KEY}' object")

        wiki_type = wiki.get("wiki_type")
        slug = wiki.get("slug")
        if wiki_type not in WIKI_TYPE_TO_DOC_TYPE:
            raise InvalidWebhookPayload(
                f"unsupported wiki_type {wiki_type!r}; expected one of "
                f"{sorted(WIKI_TYPE_TO_DOC_TYPE)}"
            )
        if not isinstance(slug, str) or not slug:
            raise InvalidWebhookPayload("wiki payload missing string slug")

        updated_at_iso = wiki.get("updated_at") or _utcnow_iso()
        is_delete = bool(wiki.get("is_delete"))

        # source_event_id pins (resource * revision * intent). The "delete" tag
        # disambiguates a delete from an edit at the same `updated_at`, matching
        # the convention notion.py uses for tombstones.
        tail = "delete" if is_delete else "edit"
        return WebhookParseResult(
            source_event_id=f"{wiki_type}:{slug}:{tail}:{updated_at_iso}",
            received_at=_parse_iso(updated_at_iso),
            event_kind=IngestionEventType.MANUAL,
            parse_hint={
                "wiki_type": wiki_type,
                "slug": slug,
                "updated_at": updated_at_iso,
                "is_delete": is_delete,
            },
        )

    async def fetch_supplementary(
        self,
        event: WebhookEvent,
        token: Any,
    ) -> Mapping[str, Any]:
        # The whole page lives in the raw payload — no second fetch.
        return {}

    async def normalize(
        self,
        event: WebhookEvent,
        hydrated: Mapping[str, Any],
    ) -> NormalizationResult:
        return build_normalization_result(event)


def build_normalization_result(event: WebhookEvent) -> NormalizationResult:
    """Pure transform from a wiki upload event to a NormalizationResult.

    Split out from `WikiConnector.normalize` so unit tests can construct a
    `WebhookEvent` and assert on the result without instantiating the
    connector + its httpx client.
    """
    payload = event.raw_payload.get(WIKI_PAYLOAD_KEY)
    if not isinstance(payload, Mapping):
        raise InvalidWebhookPayload(f"wiki event raw_payload missing '{WIKI_PAYLOAD_KEY}' object")

    wiki_type = payload["wiki_type"]
    slug = payload["slug"]
    if wiki_type not in WIKI_TYPE_TO_DOC_TYPE:
        raise InvalidWebhookPayload(f"unsupported wiki_type {wiki_type!r}")

    doc_type = WIKI_TYPE_TO_DOC_TYPE[wiki_type]
    doc_id = f"wiki:{wiki_type}:{slug}"
    source_id = f"{wiki_type}:{slug}"
    source_url = f"/wiki/{wiki_type}/{slug}"

    title = (payload.get("title") or _humanize(slug)).strip() or _humanize(slug)
    body = payload.get("body") or ""
    frontmatter = payload.get("frontmatter") or {}
    if not isinstance(frontmatter, Mapping):
        raise InvalidWebhookPayload("wiki frontmatter must be an object")

    doc_class_raw = payload.get("doc_class") or DocClass.MANUAL_ENTRY.value
    try:
        doc_class = DocClass(doc_class_raw)
    except ValueError as exc:
        raise InvalidWebhookPayload(f"unsupported doc_class {doc_class_raw!r}") from exc

    author_id = payload.get("author_id")
    compiled_from_doc_ids = payload.get("compiled_from_doc_ids")
    if compiled_from_doc_ids is not None and not isinstance(compiled_from_doc_ids, list):
        raise InvalidWebhookPayload("compiled_from_doc_ids must be a list of doc_id strings")

    is_delete = bool(payload.get("is_delete"))
    received_at = event.received_at

    if is_delete:
        body = ""
        deleted_at: datetime | None = received_at
        content_hash = _sha256(f"{doc_id}|__deleted__|{received_at.isoformat()}")
        links: list[WikiPageLink] = []
        dangling_links: list[str] = []
    else:
        deleted_at = None
        content_hash = _sha256(f"{doc_id}|{received_at.isoformat()}|{title}|{body}")
        links = parse_page_links(body)
        dangling_links = [link.raw for link in links if link.kind == "plain"]

    acl = ACLSnapshot(
        principals=[
            ACLPrincipal(
                principal_type=PrincipalType.WORKSPACE,
                principal_id=event.customer_id,
                permission=Permission.READ,
            )
        ],
        captured_at=received_at,
    )
    acl_rows = [
        ACLSnapshotRow(
            source_system=SourceSystem.WIKI,
            principal_type=PrincipalType.WORKSPACE,
            principal_id=event.customer_id,
            resource_type="wiki_page",
            resource_id=doc_id,
            permission=Permission.READ,
            valid_from=received_at,
        )
    ]

    metadata: dict[str, Any] = {
        "wiki_type": wiki_type,
        "slug": slug,
        "frontmatter": dict(frontmatter),
        "body": body,
        "dangling_links": dangling_links,
        "doc_class": doc_class.value,
    }

    compile_trigger: CompileTrigger | None = None
    compiled_at: datetime | None = None
    if doc_class == DocClass.COMPILED_WIKI:
        # An agent (re)compiled this page from sources. Default the trigger to
        # MANUAL when the caller didn't say otherwise — only the synthesize
        # cron will need to override this once it lands.
        trigger_raw = payload.get("compile_trigger") or CompileTrigger.MANUAL.value
        try:
            compile_trigger = CompileTrigger(trigger_raw)
        except ValueError as exc:
            raise InvalidWebhookPayload(f"unsupported compile_trigger {trigger_raw!r}") from exc
        compiled_at = received_at

    doc = Document(
        doc_id=doc_id,
        customer_id=event.customer_id,
        source_system=SourceSystem.WIKI,
        source_id=source_id,
        source_url=source_url,
        doc_class=doc_class,
        doc_type=doc_type,
        content_type="text/markdown",
        content_hash=content_hash,
        title=title,
        body_preview=body[:500] if body else None,
        body_size_bytes=len(body.encode("utf-8")),
        body_token_count=count_tokens(body),
        author_id=author_id,
        created_at=received_at,
        updated_at=received_at,
        valid_from=received_at,
        deleted_at=deleted_at,
        ingested_at=datetime.now(UTC),
        acl=acl,
        metadata=metadata,
        compiled_from_doc_ids=compiled_from_doc_ids,
        compile_trigger=compile_trigger,
        compiled_at=compiled_at,
    )

    nodes, edges = _build_graph(doc_id, doc_type, links, received_at)
    return NormalizationResult(
        documents=[doc],
        graph_nodes=nodes,
        graph_edges=edges,
        acl_snapshots=acl_rows,
    )


def _build_graph(
    doc_id: str,
    doc_type: DocType,
    links: list[WikiPageLink],
    valid_from: datetime,
) -> tuple[list[GraphNodeSpec], list[GraphEdgeSpec]]:
    nodes: list[GraphNodeSpec] = [
        GraphNodeSpec(
            label=NodeLabel.DOCUMENT,
            canonical_id=doc_id,
            properties={
                "doc_type": doc_type.value,
                "source_system": SourceSystem.WIKI.value,
            },
        ),
    ]
    seen: set[tuple[str, str]] = set()
    edges: list[GraphEdgeSpec] = []
    for link in links:
        mapped = _LINK_NODE_MAP.get(link.kind)
        if mapped is None:
            continue
        node_label, edge_type = mapped
        canonical = link.target.strip()
        if not canonical:
            continue
        key = (node_label.value, canonical)
        if key not in seen:
            seen.add(key)
            nodes.append(
                GraphNodeSpec(
                    label=node_label,
                    canonical_id=canonical,
                    properties={"source_system": SourceSystem.WIKI.value},
                )
            )
        edges.append(
            GraphEdgeSpec(
                edge_type=edge_type,
                from_label=NodeLabel.DOCUMENT,
                from_canonical_id=doc_id,
                to_label=node_label,
                to_canonical_id=canonical,
                valid_from=valid_from,
            )
        )
    return nodes, edges


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _parse_iso(value: str) -> datetime:
    # Accept both `Z` and `+HH:MM` suffixes; tolerate naive inputs by stamping UTC.
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(value)
    except ValueError as exc:
        raise InvalidWebhookPayload(f"wiki updated_at must be ISO-8601, got {value!r}") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def _humanize(slug: str) -> str:
    return slug.replace("-", " ").strip().title() or slug
