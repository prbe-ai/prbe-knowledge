"""Internal /api/wiki/pages/* endpoints.

Synchronous: a `PUT` blocks for the chunk + embed + persist round trip
(60-120s for a large body). Acceptable at MVP cadence — wiki uploads are
human-paced and the immediate "did my upload work + which links are
dangling" feedback is the whole point. A future phase can swap to async
+ status polling if it becomes a problem.

Routes:

    PUT    /api/wiki/pages/{wiki_type}/{slug}   upsert
    GET    /api/wiki/pages/{wiki_type}/{slug}   fetch live version
    DELETE /api/wiki/pages/{wiki_type}/{slug}   soft-delete (sets deleted_at)
    GET    /api/wiki/pages?type={wiki_type}     list (title, slug, updated_at)

Auth: every route depends on `verify_internal_knowledge_key` and reads tenant
from `X-Prbe-Customer`. The dashboard reaches these via the prbe-backend BFF
(separate session).
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

import orjson
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from pydantic import BaseModel, Field, field_validator

from services.ingestion.admin_routes import verify_internal_knowledge_key
from services.ingestion.handlers.wiki import (
    WIKI_PAYLOAD_KEY,
    WIKI_TYPE_TO_DOC_TYPE,
    build_normalization_result,
)
from services.ingestion.normalizer import Normalizer
from services.ingestion.wiki_links import parse_wiki_links
from shared.constants import DocClass, SourceSystem
from shared.db import with_tenant
from shared.exceptions import InvalidWebhookPayload
from shared.logging import get_logger
from shared.models import WebhookEvent

log = get_logger(__name__)

router = APIRouter(prefix="/api/wiki", tags=["wiki"])

# Slugs are URL components plus the source_id segment after the type prefix.
# Keep them lowercase + dash-only so /wiki/runbook/Foo and /wiki/runbook/foo
# can't both exist.
_SLUG_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$")
_MAX_BODY_BYTES = 1_048_576  # 1 MiB hard cap; bigger uploads must split into pages


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class WikiUpsertBody(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    body: str = Field(default="", max_length=_MAX_BODY_BYTES)
    frontmatter: dict[str, Any] = Field(default_factory=dict)
    doc_class: str = Field(default=DocClass.MANUAL_ENTRY.value)
    author_id: str | None = Field(default=None, max_length=128)
    compiled_from_doc_ids: list[str] | None = None
    compile_trigger: str | None = None
    updated_at: datetime | None = None

    @field_validator("doc_class")
    @classmethod
    def _doc_class_allowlist(cls, value: str) -> str:
        # Phase 1 gates the public route to MANUAL_ENTRY only. The future
        # synthesize cron will write COMPILED_WIKI pages by calling
        # `build_normalization_result` directly, not through this endpoint.
        if value != DocClass.MANUAL_ENTRY.value:
            raise ValueError(
                f"doc_class={value!r} not allowed via this endpoint; "
                "use 'manual_entry' for human uploads"
            )
        return value


class WikiLinkOut(BaseModel):
    raw: str
    kind: str
    target: str


class WikiUpsertResponse(BaseModel):
    doc_id: str
    source_url: str
    version: int
    chunk_count: int
    links: list[WikiLinkOut]
    dangling_links: list[str]


class WikiPageResponse(BaseModel):
    doc_id: str
    customer_id: str
    wiki_type: str
    slug: str
    title: str | None
    body: str
    frontmatter: dict[str, Any]
    doc_class: str
    author_id: str | None
    version: int
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None


class WikiListItem(BaseModel):
    wiki_type: str
    slug: str
    title: str | None
    updated_at: datetime
    version: int


class WikiListResponse(BaseModel):
    items: list[WikiListItem]
    count: int


class WikiDeleteResponse(BaseModel):
    doc_id: str
    deleted: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_customer(
    x_prbe_customer: str | None = Header(default=None, alias="X-Prbe-Customer"),
) -> str:
    if not x_prbe_customer:
        raise HTTPException(status_code=400, detail="missing X-Prbe-Customer")
    return x_prbe_customer


def _validate_wiki_type(wiki_type: str) -> str:
    if wiki_type not in WIKI_TYPE_TO_DOC_TYPE:
        raise HTTPException(
            status_code=400,
            detail=(
                f"unsupported wiki_type {wiki_type!r}; expected one of "
                f"{sorted(WIKI_TYPE_TO_DOC_TYPE)}"
            ),
        )
    return wiki_type


def _validate_slug(slug: str) -> str:
    if not _SLUG_RE.match(slug):
        raise HTTPException(
            status_code=400,
            detail="slug must match ^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$",
        )
    return slug


def _coerce_metadata(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, (str, bytes, bytearray)):
        return orjson.loads(raw)
    if isinstance(raw, dict):
        return raw
    return {}


def _build_event(
    customer_id: str,
    wiki_type: str,
    slug: str,
    *,
    title: str,
    body: str,
    frontmatter: dict[str, Any],
    doc_class: str,
    author_id: str | None,
    compiled_from_doc_ids: list[str] | None,
    compile_trigger: str | None,
    received_at: datetime,
    is_delete: bool,
) -> WebhookEvent:
    raw_payload = {
        WIKI_PAYLOAD_KEY: {
            "wiki_type": wiki_type,
            "slug": slug,
            "title": title,
            "body": body,
            "frontmatter": frontmatter,
            "doc_class": doc_class,
            "author_id": author_id,
            "compiled_from_doc_ids": compiled_from_doc_ids,
            "compile_trigger": compile_trigger,
            "is_delete": is_delete,
            "updated_at": received_at.isoformat(),
        }
    }
    tail = "delete" if is_delete else "edit"
    return WebhookEvent(
        customer_id=customer_id,
        source_system=SourceSystem.WIKI,
        source_event_id=f"{wiki_type}:{slug}:{tail}:{received_at.isoformat()}",
        received_at=received_at,
        payload_s3_key="",
        payload_s3_keys=[],
        raw_payload=raw_payload,
        headers={},
    )


async def _read_doc_version(customer_id: str, doc_id: str) -> int | None:
    async with with_tenant(customer_id) as conn:
        row = await conn.fetchrow(
            """
            SELECT version FROM documents
            WHERE customer_id = $1 AND doc_id = $2 AND valid_to IS NULL
            """,
            customer_id,
            doc_id,
        )
    return row["version"] if row else None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.put(
    "/pages/{wiki_type}/{slug}",
    response_model=WikiUpsertResponse,
    dependencies=[Depends(verify_internal_knowledge_key)],
)
async def upsert_wiki_page(
    wiki_type: str,
    slug: str,
    body: WikiUpsertBody,
    request: Request,
    customer_id: str = Depends(_require_customer),
) -> WikiUpsertResponse:
    wiki_type = _validate_wiki_type(wiki_type)
    slug = _validate_slug(slug)
    received_at = body.updated_at or datetime.now(UTC)

    event = _build_event(
        customer_id=customer_id,
        wiki_type=wiki_type,
        slug=slug,
        title=body.title,
        body=body.body,
        frontmatter=body.frontmatter,
        doc_class=body.doc_class,
        author_id=body.author_id,
        compiled_from_doc_ids=body.compiled_from_doc_ids,
        compile_trigger=body.compile_trigger,
        received_at=received_at,
        is_delete=False,
    )

    try:
        result = build_normalization_result(event)
    except InvalidWebhookPayload as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    normalizer = Normalizer(ctx=request.app.state.ctx, store=request.app.state.store)
    outcome = await normalizer._persist(customer_id, SourceSystem.WIKI, result)

    if not outcome.doc_ids:
        raise HTTPException(status_code=500, detail="wiki normalize produced no documents")

    doc = result.documents[0]
    parsed_links = parse_wiki_links(body.body)
    dangling = [link.raw for link in parsed_links if link.kind == "plain"]
    version = await _read_doc_version(customer_id, doc.doc_id) or doc.version

    log.info(
        "wiki.upserted",
        customer=customer_id,
        doc_id=doc.doc_id,
        version=version,
        chunk_count=outcome.chunk_count,
        added=outcome.added_chunk_count,
        reused=outcome.reused_chunk_count,
        removed=outcome.removed_chunk_count,
        dangling_count=len(dangling),
    )

    return WikiUpsertResponse(
        doc_id=doc.doc_id,
        source_url=doc.source_url,
        version=version,
        chunk_count=outcome.chunk_count,
        links=[
            WikiLinkOut(raw=link.raw, kind=link.kind, target=link.target) for link in parsed_links
        ],
        dangling_links=dangling,
    )


@router.get(
    "/pages/{wiki_type}/{slug}",
    response_model=WikiPageResponse,
    dependencies=[Depends(verify_internal_knowledge_key)],
)
async def get_wiki_page(
    wiki_type: str,
    slug: str,
    customer_id: str = Depends(_require_customer),
) -> WikiPageResponse:
    wiki_type = _validate_wiki_type(wiki_type)
    slug = _validate_slug(slug)
    doc_id = f"wiki:{wiki_type}:{slug}"

    async with with_tenant(customer_id) as conn:
        row = await conn.fetchrow(
            """
            SELECT doc_id, customer_id, version, source_id, title, metadata,
                   author_id, created_at, updated_at, deleted_at
            FROM documents
            WHERE customer_id = $1 AND doc_id = $2 AND valid_to IS NULL
            """,
            customer_id,
            doc_id,
        )
    if row is None or row["deleted_at"] is not None:
        raise HTTPException(status_code=404, detail="wiki page not found")

    metadata = _coerce_metadata(row["metadata"])
    return WikiPageResponse(
        doc_id=row["doc_id"],
        customer_id=row["customer_id"],
        wiki_type=metadata.get("wiki_type", wiki_type),
        slug=metadata.get("slug", slug),
        title=row["title"],
        body=metadata.get("body", ""),
        frontmatter=metadata.get("frontmatter", {}),
        doc_class=metadata.get("doc_class", DocClass.MANUAL_ENTRY.value),
        author_id=row["author_id"],
        version=row["version"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        deleted_at=row["deleted_at"],
    )


@router.delete(
    "/pages/{wiki_type}/{slug}",
    response_model=WikiDeleteResponse,
    dependencies=[Depends(verify_internal_knowledge_key)],
)
async def delete_wiki_page(
    wiki_type: str,
    slug: str,
    request: Request,
    customer_id: str = Depends(_require_customer),
) -> WikiDeleteResponse:
    wiki_type = _validate_wiki_type(wiki_type)
    slug = _validate_slug(slug)
    received_at = datetime.now(UTC)

    event = _build_event(
        customer_id=customer_id,
        wiki_type=wiki_type,
        slug=slug,
        title="",
        body="",
        frontmatter={},
        doc_class=DocClass.MANUAL_ENTRY.value,
        author_id=None,
        compiled_from_doc_ids=None,
        compile_trigger=None,
        received_at=received_at,
        is_delete=True,
    )

    try:
        result = build_normalization_result(event)
    except InvalidWebhookPayload as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    normalizer = Normalizer(ctx=request.app.state.ctx, store=request.app.state.store)
    outcome = await normalizer._persist(customer_id, SourceSystem.WIKI, result)
    doc_id = result.documents[0].doc_id
    log.info("wiki.deleted", customer=customer_id, doc_id=doc_id)
    return WikiDeleteResponse(doc_id=doc_id, deleted=bool(outcome.doc_ids))


@router.get(
    "/pages",
    response_model=WikiListResponse,
    dependencies=[Depends(verify_internal_knowledge_key)],
)
async def list_wiki_pages(
    type: str | None = Query(default=None, description="Optional wiki_type filter"),
    customer_id: str = Depends(_require_customer),
    limit: int = Query(default=100, ge=1, le=500),
) -> WikiListResponse:
    doc_type_filter: str | None = None
    if type is not None:
        _validate_wiki_type(type)
        doc_type_filter = WIKI_TYPE_TO_DOC_TYPE[type].value

    async with with_tenant(customer_id) as conn:
        if doc_type_filter:
            rows = await conn.fetch(
                """
                SELECT doc_id, source_id, title, version, updated_at, metadata
                FROM documents
                WHERE customer_id = $1
                  AND source_system = $2
                  AND doc_type = $3
                  AND valid_to IS NULL
                  AND deleted_at IS NULL
                ORDER BY updated_at DESC
                LIMIT $4
                """,
                customer_id,
                SourceSystem.WIKI.value,
                doc_type_filter,
                limit,
            )
        else:
            rows = await conn.fetch(
                """
                SELECT doc_id, source_id, title, version, updated_at, metadata
                FROM documents
                WHERE customer_id = $1
                  AND source_system = $2
                  AND valid_to IS NULL
                  AND deleted_at IS NULL
                ORDER BY updated_at DESC
                LIMIT $3
                """,
                customer_id,
                SourceSystem.WIKI.value,
                limit,
            )

    items: list[WikiListItem] = []
    for row in rows:
        metadata = _coerce_metadata(row["metadata"])
        wiki_type_value = metadata.get("wiki_type")
        slug_value = metadata.get("slug")
        if not wiki_type_value or not slug_value:
            # Defensive fallback: derive from source_id "<type>:<slug>" if metadata
            # was somehow missing the keys (older rows pre-this-feature).
            parts = row["source_id"].split(":", 1)
            if len(parts) == 2:
                wiki_type_value = wiki_type_value or parts[0]
                slug_value = slug_value or parts[1]
        items.append(
            WikiListItem(
                wiki_type=wiki_type_value or "unknown",
                slug=slug_value or row["source_id"],
                title=row["title"],
                updated_at=row["updated_at"],
                version=row["version"],
            )
        )
    return WikiListResponse(items=items, count=len(items))
