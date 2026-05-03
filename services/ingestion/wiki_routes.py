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

## Access model — who writes what doc_class

Public PUT/PATCH/DELETE accept `doc_class=manual_entry` ONLY (validated by
`_reject_non_manual_doc_class` on `WikiUpsertBody.doc_class`). Anything else
is rejected with 422.

`doc_class=compiled_wiki` writes are produced by the synthesis cron in
`services/synthesis/wiki_cron.py` (Phase 2). The cron does NOT call this
HTTP route; it builds a synthetic `WebhookEvent` and calls
`build_normalization_result` + `Normalizer._persist` directly — same code
path the route handlers use, just bypassing the FastAPI/HTTP boundary.
There is no internal-only route variant: the cron is in-process inside
the worker fly app, so there's nothing for HTTP to mediate. If a future
out-of-process synthesizer needs to write `compiled_wiki`, it should
either re-use the in-process path or get its own dedicated route — don't
loosen this validator.
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
    INDEX_SLUG,
    USER_AUTHORED_WIKI_TYPES,
    WIKI_PAYLOAD_KEY,
    WIKI_TYPE_TO_DOC_TYPE,
    build_normalization_result,
)
from services.ingestion.normalizer import (
    Normalizer,
    fetch_body_from_chunks_for_version,
    fetch_live_body_from_chunks,
)
from services.ingestion.wiki_links import parse_page_links
from shared.constants import DocClass, DocType, SourceSystem
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
    commit_message: str | None = Field(default=None, max_length=240)
    summary: str | None = Field(default=None, max_length=240)

    @field_validator("doc_class")
    @classmethod
    def _reject_non_manual_doc_class(cls, value: str) -> str:
        """Public route accepts MANUAL_ENTRY only.

        See module docstring "Access model" — `compiled_wiki` writes are
        produced by `services/synthesis/wiki_cron.py` calling
        `build_normalization_result` directly, in-process. The HTTP route
        is the human-authoring surface and stays narrowly scoped.
        """
        if value != DocClass.MANUAL_ENTRY.value:
            raise ValueError(
                f"doc_class={value!r} not allowed via this endpoint; "
                "use 'manual_entry' for human uploads"
            )
        return value


class WikiRevertBody(BaseModel):
    to_version: int = Field(ge=1)
    reason: str = Field(min_length=1, max_length=240)
    author_id: str | None = Field(default=None, max_length=128)


class WikiHistoryEntry(BaseModel):
    version: int
    updated_at: datetime
    author_id: str | None
    commit_message: str | None
    commit_author: str | None
    commit_run_id: str | int | None
    content_hash: str
    is_live: bool


class WikiHistoryResponse(BaseModel):
    doc_id: str
    entries: list[WikiHistoryEntry]


class WikiIndexEntry(BaseModel):
    wiki_type: str
    slug: str
    title: str | None
    summary: str | None
    updated_at: datetime
    version: int


class WikiIndexResponse(BaseModel):
    body: str
    entries: list[WikiIndexEntry]
    updated_at: datetime | None
    version: int | None


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
    """Validate the path wiki_type for human-author routes (PUT/GET/DELETE).

    The 'index' type is a singleton synthesized by the cron — never a valid
    target for human upload, so we reject it here. The cron writes the
    index by calling `build_normalization_result` directly.
    """
    if wiki_type not in USER_AUTHORED_WIKI_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"unsupported wiki_type {wiki_type!r}; expected one of "
                f"{sorted(USER_AUTHORED_WIKI_TYPES)}"
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
    commit_message: str | None = None,
    commit_author: str | None = None,
    summary: str | None = None,
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
            "commit_message": commit_message,
            "commit_author": commit_author,
            "summary": summary,
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
        commit_message=body.commit_message
        or (
            f"Manual upload by {body.author_id}"
            if body.author_id
            else "Manual upload"
        ),
        commit_author=body.author_id,
        summary=body.summary,
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
    parsed_links = parse_page_links(body.body)
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
        # Body lives in chunks, not metadata (storage-cleanup migration 0035).
        # Fetched inside the same with_tenant block so RLS on chunks sees the
        # tenant GUC.
        body = await fetch_live_body_from_chunks(conn, customer_id, doc_id)

    metadata = _coerce_metadata(row["metadata"])
    return WikiPageResponse(
        doc_id=row["doc_id"],
        customer_id=row["customer_id"],
        wiki_type=metadata.get("wiki_type", wiki_type),
        slug=metadata.get("slug", slug),
        title=row["title"],
        body=body,
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
        # The auto-generated index is exposed via /api/wiki/index, not in the
        # general list of user-authored pages.
        if wiki_type_value == "index":
            continue
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


# ---------------------------------------------------------------------------
# History + revert + index
# ---------------------------------------------------------------------------


@router.get(
    "/pages/{wiki_type}/{slug}/history",
    response_model=WikiHistoryResponse,
    dependencies=[Depends(verify_internal_knowledge_key)],
)
async def get_wiki_page_history(
    wiki_type: str,
    slug: str,
    customer_id: str = Depends(_require_customer),
) -> WikiHistoryResponse:
    """Every persisted version of a wiki page, newest first.

    Includes both the live version (valid_to IS NULL) and all closed-out
    historical versions. Each entry surfaces the commit metadata stamped at
    write time so consumers can render an audit trail.
    """
    wiki_type = _validate_wiki_type(wiki_type)
    slug = _validate_slug(slug)
    doc_id = f"wiki:{wiki_type}:{slug}"

    async with with_tenant(customer_id) as conn:
        rows = await conn.fetch(
            """
            SELECT version, updated_at, author_id, content_hash, valid_to,
                   metadata
            FROM documents
            WHERE customer_id = $1 AND doc_id = $2
            ORDER BY version DESC
            """,
            customer_id,
            doc_id,
        )

    if not rows:
        raise HTTPException(status_code=404, detail="wiki page not found")

    entries: list[WikiHistoryEntry] = []
    for row in rows:
        metadata = _coerce_metadata(row["metadata"])
        commit = metadata.get("commit") if isinstance(metadata, dict) else None
        if not isinstance(commit, dict):
            commit = {}
        entries.append(
            WikiHistoryEntry(
                version=row["version"],
                updated_at=row["updated_at"],
                author_id=row["author_id"],
                commit_message=commit.get("message"),
                commit_author=commit.get("author"),
                commit_run_id=commit.get("run_id"),
                content_hash=row["content_hash"],
                is_live=row["valid_to"] is None,
            )
        )
    return WikiHistoryResponse(doc_id=doc_id, entries=entries)


@router.post(
    "/pages/{wiki_type}/{slug}/revert",
    response_model=WikiUpsertResponse,
    dependencies=[Depends(verify_internal_knowledge_key)],
)
async def revert_wiki_page(
    wiki_type: str,
    slug: str,
    body: WikiRevertBody,
    request: Request,
    customer_id: str = Depends(_require_customer),
) -> WikiUpsertResponse:
    """Roll a wiki page back to a prior version.

    Reads version `to_version` from `documents`, copies its body + frontmatter
    + summary into a new write through `build_normalization_result`. The new
    row's commit metadata explains the revert. Original history is untouched.
    """
    wiki_type = _validate_wiki_type(wiki_type)
    slug = _validate_slug(slug)
    doc_id = f"wiki:{wiki_type}:{slug}"

    async with with_tenant(customer_id) as conn:
        target = await conn.fetchrow(
            """
            SELECT title, metadata
            FROM documents
            WHERE customer_id = $1 AND doc_id = $2 AND version = $3
            """,
            customer_id,
            doc_id,
            body.to_version,
        )
        if target is None:
            raise HTTPException(
                status_code=404,
                detail=f"version {body.to_version} not found for {doc_id}",
            )
        # Reconstruct the prior version's body from chunks. Chunks are
        # version-spanning via [first_seen_version, last_seen_version], so a
        # row participates in version V iff it covers V. Falls back to empty
        # when the prior version had no chunks (e.g. a delete tombstone) —
        # the caller's "revert" semantics on an empty page mean the new
        # version is empty too.
        target_body = await fetch_body_from_chunks_for_version(
            conn, customer_id, doc_id, body.to_version
        )

    target_meta = _coerce_metadata(target["metadata"])
    received_at = datetime.now(UTC)
    event = _build_event(
        customer_id=customer_id,
        wiki_type=wiki_type,
        slug=slug,
        title=target["title"] or "",
        body=target_body,
        frontmatter=target_meta.get("frontmatter") or {},
        doc_class=DocClass.MANUAL_ENTRY.value,
        author_id=body.author_id,
        compiled_from_doc_ids=None,
        compile_trigger=None,
        received_at=received_at,
        is_delete=False,
        commit_message=f"Revert to v{body.to_version}: {body.reason}",
        commit_author=body.author_id,
        summary=target_meta.get("summary"),
    )

    try:
        result = build_normalization_result(event)
    except InvalidWebhookPayload as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    normalizer = Normalizer(ctx=request.app.state.ctx, store=request.app.state.store)
    outcome = await normalizer._persist(customer_id, SourceSystem.WIKI, result)
    if not outcome.doc_ids:
        # The revert produced byte-identical content to the live version
        # (the user reverted to the version that was already live). Surface
        # that as a 200 with the existing version, not a 500.
        version = await _read_doc_version(customer_id, doc_id) or body.to_version
    else:
        version = await _read_doc_version(customer_id, doc_id) or result.documents[0].version

    log.info(
        "wiki.reverted",
        customer=customer_id,
        doc_id=doc_id,
        to_version=body.to_version,
        new_version=version,
    )

    doc = result.documents[0]
    parsed_links = parse_page_links(target_meta.get("body") or "")
    dangling = [link.raw for link in parsed_links if link.kind == "plain"]
    return WikiUpsertResponse(
        doc_id=doc.doc_id,
        source_url=doc.source_url,
        version=version,
        chunk_count=outcome.chunk_count,
        links=[
            WikiLinkOut(raw=link.raw, kind=link.kind, target=link.target)
            for link in parsed_links
        ],
        dangling_links=dangling,
    )


@router.get(
    "/index",
    response_model=WikiIndexResponse,
    dependencies=[Depends(verify_internal_knowledge_key)],
)
async def get_wiki_index(
    customer_id: str = Depends(_require_customer),
) -> WikiIndexResponse:
    """Return the auto-generated table of contents.

    The synthesis cron regenerates this at the end of each tick from the
    live set of wiki pages. If the cron has never run for this customer
    yet, return a deterministic fallback assembled from current page
    titles + summaries (so the dashboard always has something to show).
    """
    index_doc_id = f"wiki:index:{INDEX_SLUG}"

    async with with_tenant(customer_id) as conn:
        index_row = await conn.fetchrow(
            """
            SELECT version, updated_at, metadata
            FROM documents
            WHERE customer_id = $1 AND doc_id = $2 AND valid_to IS NULL
            """,
            customer_id,
            index_doc_id,
        )
        page_rows = await conn.fetch(
            """
            SELECT title, source_id, version, updated_at, metadata
            FROM documents
            WHERE customer_id = $1
              AND source_system = $2
              AND doc_type = ANY($3::text[])
              AND valid_to IS NULL
              AND deleted_at IS NULL
            ORDER BY updated_at DESC
            """,
            customer_id,
            SourceSystem.WIKI.value,
            [
                DocType.WIKI_SERVICE_CARD.value,
                DocType.WIKI_DECISION.value,
                DocType.WIKI_FEATURE.value,
                DocType.WIKI_RUNBOOK.value,
            ],
        )

    entries: list[WikiIndexEntry] = []
    for row in page_rows:
        metadata = _coerce_metadata(row["metadata"])
        wiki_type_value = metadata.get("wiki_type") or row["source_id"].split(":", 1)[0]
        slug_value = metadata.get("slug") or row["source_id"].split(":", 1)[-1]
        entries.append(
            WikiIndexEntry(
                wiki_type=wiki_type_value,
                slug=slug_value,
                title=row["title"],
                summary=metadata.get("summary"),
                updated_at=row["updated_at"],
                version=row["version"],
            )
        )

    if index_row is not None:
        index_meta = _coerce_metadata(index_row["metadata"])
        body_text = index_meta.get("body") or _render_index_body(entries)
        return WikiIndexResponse(
            body=body_text,
            entries=entries,
            updated_at=index_row["updated_at"],
            version=index_row["version"],
        )

    # Fallback: cron has not produced an index yet. Render deterministically.
    return WikiIndexResponse(
        body=_render_index_body(entries),
        entries=entries,
        updated_at=None,
        version=None,
    )


def _render_index_body(entries: list[WikiIndexEntry]) -> str:
    """Plain markdown TOC. Used when the cron-generated index is missing."""
    if not entries:
        return "# Wiki\n\nNo pages yet.\n"
    by_type: dict[str, list[WikiIndexEntry]] = {}
    for entry in entries:
        by_type.setdefault(entry.wiki_type, []).append(entry)
    parts = ["# Wiki", ""]
    for wiki_type in sorted(by_type):
        parts.append(f"## {wiki_type.replace('_', ' ').title()}")
        for entry in by_type[wiki_type]:
            title = entry.title or entry.slug
            summary = entry.summary or ""
            line = f"- [[{title}]] — {summary}" if summary else f"- [[{title}]]"
            parts.append(line)
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"
