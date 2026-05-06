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

import asyncio
import re
from datetime import UTC, datetime
from typing import Any

import asyncpg
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
from services.synthesis.crawlers import REGISTRY as BACKFILL_CRAWLER_REGISTRY
from shared.constants import (
    BACKFILL_CANCEL_DRAIN_TIMEOUT_SECONDS,
    INDEXABLE_WIKI_DOC_TYPES,
    WIKI_BACKFILL_CANCEL_CHANNEL,
    WIKI_BACKFILL_CHANNEL,
    WIKI_PENDING_CHANNEL,
    DocClass,
    SourceSystem,
)
from shared.db import raw_conn, with_tenant
from shared.exceptions import InvalidWebhookPayload
from shared.locks import advisory_lock_key
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
        or (f"Manual upload by {body.author_id}" if body.author_id else "Manual upload"),
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
            WikiLinkOut(raw=link.raw, kind=link.kind, target=link.target) for link in parsed_links
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
            [dt.value for dt in INDEXABLE_WIKI_DOC_TYPES],
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


# ---------------------------------------------------------------------------
# Synthesis trigger + status (manual wake from the dashboard button)
# ---------------------------------------------------------------------------


class SynthesisTriggerBody(BaseModel):
    reason: str | None = Field(default=None, max_length=240)


class SynthesisTriggerResponse(BaseModel):
    triggered: bool
    pending_events: int
    last_run_at: datetime | None


class SynthesisStatusResponse(BaseModel):
    pending_events: int
    triaged_events: int
    in_flight_events: int
    failed_events: int
    # v4: synthesis_skipped covers both verifier_rejected (legacy) and
    # the new agent skip_events tool. The dashboard surfaces both
    # together; tracking them apart wasn't useful in practice.
    synthesis_skipped_events: int
    # v4: rows DLQ'd by triage batch crash or wiki agent halt. Admin
    # reset (POST .../dlq/reset) flips them back to pending or triaged.
    dlq_count: int
    oldest_dlq_at: datetime | None
    last_run_at: datetime | None
    last_run_status: str | None
    last_run_pages_updated: int | None
    last_run_pages_created: int | None


class DlqResetBody(BaseModel):
    reason: str | None = Field(default=None, max_length=240)
    max_rows: int = Field(default=5000, ge=1, le=100_000)


class DlqResetResponse(BaseModel):
    reset_count: int
    triaged_reset: int
    pending_reset: int
    oldest_dlq_at: datetime | None


@router.post(
    "/synthesize/trigger",
    response_model=SynthesisTriggerResponse,
    dependencies=[Depends(verify_internal_knowledge_key)],
)
async def trigger_wiki_synthesis(
    body: SynthesisTriggerBody,
    customer_id: str = Depends(_require_customer),
) -> SynthesisTriggerResponse:
    """Wake the wiki-worker for this customer NOW.

    Fires `pg_notify(WIKI_PENDING_CHANNEL, customer_id)`. The wiki-worker
    fly app's NotifyListener wakes within seconds and starts triage; the
    wiki-synthesis app picks up the triaged rows the worker produces.

    Rate-limit + advisory-lock are enforced upstream in the BFF
    (prbe-backend), not here — this endpoint is the internal-keyed
    pass-through. Returns the pending count + last-run timestamp so the
    dashboard can render an accurate status badge.
    """
    async with raw_conn() as conn:
        pending_count = await conn.fetchval(
            """
            SELECT count(*)::bigint
            FROM wiki_synthesis_queue
            WHERE customer_id = $1 AND status = 'pending'
            """,
            customer_id,
        )
        last_run_row = await conn.fetchrow(
            """
            SELECT started_at FROM wiki_synthesis_runs
            WHERE customer_id = $1 AND stage = 'synthesis'
            ORDER BY started_at DESC
            LIMIT 1
            """,
            customer_id,
        )
        await conn.execute(
            "SELECT pg_notify($1, $2)",
            WIKI_PENDING_CHANNEL,
            customer_id,
        )
    log.info(
        "wiki.synthesize.trigger",
        customer=customer_id,
        pending=int(pending_count or 0),
        reason=body.reason,
    )
    return SynthesisTriggerResponse(
        triggered=True,
        pending_events=int(pending_count or 0),
        last_run_at=last_run_row["started_at"] if last_run_row else None,
    )


@router.get(
    "/synthesize/status",
    response_model=SynthesisStatusResponse,
    dependencies=[Depends(verify_internal_knowledge_key)],
)
async def get_wiki_synthesis_status(
    customer_id: str = Depends(_require_customer),
) -> SynthesisStatusResponse:
    """Counts + last-run summary for the dashboard status badge.

    No write side effects. Reads `wiki_synthesis_queue` for per-status
    counts and the latest synthesis-stage `wiki_synthesis_runs` row.
    The `stage` filter matters because the triage worker also opens
    its own run row per drain — those rows have
    pages_updated/pages_created=0 by design and would flap the
    dashboard if surfaced.

    v4 surfaces dlq_count + oldest_dlq_at so the dashboard can render
    a "drain stuck — N events need admin reset" banner.
    """
    async with raw_conn() as conn:
        counts = await conn.fetchrow(
            """
            SELECT
                COUNT(*) FILTER (WHERE status = 'pending') AS pending,
                COUNT(*) FILTER (WHERE status = 'triaged') AS triaged,
                COUNT(*) FILTER (
                    WHERE status IN ('triaging', 'synthesizing')
                ) AS in_flight,
                COUNT(*) FILTER (WHERE status = 'failed') AS failed,
                COUNT(*) FILTER (
                    WHERE status = 'synthesis_skipped'
                ) AS synthesis_skipped,
                COUNT(*) FILTER (WHERE status = 'dlq') AS dlq,
                MIN(dlq_at) FILTER (WHERE status = 'dlq') AS oldest_dlq_at
            FROM wiki_synthesis_queue
            WHERE customer_id = $1
            """,
            customer_id,
        )
        last_run = await conn.fetchrow(
            """
            SELECT started_at, status, pages_updated, pages_created
            FROM wiki_synthesis_runs
            WHERE customer_id = $1 AND stage = 'synthesis'
            ORDER BY started_at DESC
            LIMIT 1
            """,
            customer_id,
        )
    return SynthesisStatusResponse(
        pending_events=int(counts["pending"] or 0) if counts else 0,
        triaged_events=int(counts["triaged"] or 0) if counts else 0,
        in_flight_events=int(counts["in_flight"] or 0) if counts else 0,
        failed_events=int(counts["failed"] or 0) if counts else 0,
        synthesis_skipped_events=(int(counts["synthesis_skipped"] or 0) if counts else 0),
        dlq_count=int(counts["dlq"] or 0) if counts else 0,
        oldest_dlq_at=counts["oldest_dlq_at"] if counts else None,
        last_run_at=last_run["started_at"] if last_run else None,
        last_run_status=last_run["status"] if last_run else None,
        last_run_pages_updated=last_run["pages_updated"] if last_run else None,
        last_run_pages_created=last_run["pages_created"] if last_run else None,
    )


@router.post(
    "/synthesize/dlq/reset",
    response_model=DlqResetResponse,
    dependencies=[Depends(verify_internal_knowledge_key)],
)
async def reset_wiki_synthesis_dlq(
    body: DlqResetBody,
    customer_id: str = Depends(_require_customer),
) -> DlqResetResponse:
    """Flip DLQ rows back to pending / triaged for retry.

    Behavior: rows with dlq_reason starting 'agent.' (the wiki agent
    halted mid-drain) -> back to 'triaged' (they were already triaged
    when the agent saw them; just need another pass). Other rows
    (triage batch crashes -> dlq_reason starts 'triage.') -> back to
    'pending' (the triage layer needs to re-score).

    Capped by max_rows so an enormous backlog reset doesn't lock the
    queue table for minutes. Defaults to 5000; max 100k. attempts is
    reset to 0 so the per-row attempt cap doesn't immediately kick
    them straight back to 'failed'.

    Admin-gated upstream in the BFF (require_role('admin')); this
    endpoint is the internal-keyed pass-through.
    """
    async with raw_conn() as conn:
        # First pass: snapshot how many rows in each "would-reset" bucket
        # so we can return both counts to the caller.
        counts = await conn.fetchrow(
            """
            SELECT
                COUNT(*) FILTER (
                    WHERE dlq_reason LIKE 'agent.%'
                ) AS triaged_reset,
                COUNT(*) FILTER (
                    WHERE dlq_reason IS NULL
                       OR NOT (dlq_reason LIKE 'agent.%')
                ) AS pending_reset
            FROM wiki_synthesis_queue
            WHERE customer_id = $1 AND status = 'dlq'
            LIMIT $2
            """,
            customer_id,
            body.max_rows,
        )
        # Second pass: actually flip the rows.
        result = await conn.execute(
            """
            UPDATE wiki_synthesis_queue
            SET status = CASE
                  WHEN dlq_reason LIKE 'agent.%' THEN 'triaged'
                  ELSE 'pending'
                END,
                attempts = 0,
                dlq_reason = NULL,
                dlq_at = NULL
            WHERE queue_id IN (
                SELECT queue_id FROM wiki_synthesis_queue
                WHERE customer_id = $1 AND status = 'dlq'
                LIMIT $2
            )
            """,
            customer_id,
            body.max_rows,
        )
        # Read remaining oldest_dlq_at after the reset for the response.
        oldest = await conn.fetchval(
            """
            SELECT MIN(dlq_at)
            FROM wiki_synthesis_queue
            WHERE customer_id = $1 AND status = 'dlq'
            """,
            customer_id,
        )
    try:
        reset_count = int(result.split()[-1])
    except (ValueError, IndexError):
        reset_count = 0
    triaged_reset = int(counts["triaged_reset"] or 0) if counts else 0
    pending_reset = int(counts["pending_reset"] or 0) if counts else 0
    log.info(
        "wiki.synthesize.dlq_reset",
        customer=customer_id,
        reset_count=reset_count,
        triaged_reset=triaged_reset,
        pending_reset=pending_reset,
        reason=body.reason,
    )
    return DlqResetResponse(
        reset_count=reset_count,
        triaged_reset=triaged_reset,
        pending_reset=pending_reset,
        oldest_dlq_at=oldest,
    )


# ---------------------------------------------------------------------------
# Backfill trigger (per-source crawler agents — not the daily replay loop)
#
# Both /api/wiki/backfill/* and the legacy /api/wiki/bootstrap/* aliases
# share the same handler logic via _do_trigger_wiki_backfill /
# _do_get_wiki_backfill_status. The DB value `kind='bootstrap'` and the
# fly app name stay unchanged for now; only the public surface, Python
# identifiers, and structlog events are renamed.
# ---------------------------------------------------------------------------


class BackfillTriggerBody(BaseModel):
    """Request body for ``POST /api/wiki/backfill/trigger``.

    All three fields are optional. Defaults mirror the locked plan:
    every registered source, wipe-first re-backfill, generic reason.
    """

    sources: list[str] | None = Field(default=None)
    wipe_first: bool = Field(default=True)
    reason: str | None = Field(default=None, max_length=240)


class BackfillTriggerResponse(BaseModel):
    """Shape consumed by the BFF ``POST /api/wiki/backfill/trigger`` proxy.

    ``run_ids`` are the freshly-inserted ``wiki_synthesis_runs`` row IDs
    (one per requested source) at status='pending'. A worker on any
    machine claims them via ``FOR UPDATE SKIP LOCKED`` and runs each
    crawler in parallel.
    """

    triggered: bool = True
    run_ids: list[int]


class BackfillStatusResponse(BaseModel):
    """Aggregate over the most recent backfill "burst" for the customer.

    A burst = the runs opened by a single trigger (one per source). We
    detect it by anchoring on the most recent ``kind='bootstrap'`` row's
    ``started_at`` and including everything within 60 seconds before
    that. Empty payload when the customer has never backfilled.
    """

    in_progress: bool
    started_at: datetime | None
    sources_attempted: list[str]
    sources_succeeded: list[str]
    sources_failed: dict[str, str]
    pages_created: int
    pages_updated: int
    # Phase 2 per-target progress, keyed by source then target.
    # Example: ``{"github": {"prbe-ai/prbe-knowledge": "complete",
    # "prbe-ai/prbe-backend": "running", "prbe-ai/prbe-dashboard": "failed"}}``.
    # Phase 1 rows (target=None) appear in sources_attempted/succeeded/failed
    # above and DO NOT appear here. Empty dict when no Phase 2 fan-out has
    # been issued (e.g., before the GitHub crawler ships Phase 2, or for
    # sources that don't fan out).
    targets: dict[str, dict[str, str]] = {}


# Back-compat aliases. Existing test imports + the prbe-backend BFF
# response models still reference the old names.
BootstrapTriggerBody = BackfillTriggerBody
BootstrapTriggerResponse = BackfillTriggerResponse
BootstrapStatusResponse = BackfillStatusResponse


# In-flight states for the trigger route's pre-flight check.
_IN_FLIGHT_STATUSES: tuple[str, ...] = ("pending", "running")


async def _get_in_flight_runs(conn: asyncpg.Connection, customer_id: str) -> list[dict[str, Any]]:
    """Return every in-flight bootstrap run row for ``customer_id``.

    "In-flight" = ``status IN ('pending','running')``. Used by the
    trigger route to short-circuit with a 409 (or, when ``force=true``,
    to mark the in-flight rows ``cancelled`` and notify workers).
    """
    rows = await conn.fetch(
        """
        SELECT run_id, source, status, started_at
        FROM wiki_synthesis_runs
        WHERE customer_id = $1
          AND kind = 'bootstrap'
          AND status = ANY($2::text[])
        ORDER BY started_at ASC
        """,
        customer_id,
        list(_IN_FLIGHT_STATUSES),
    )
    return [dict(r) for r in rows]


async def _wipe_wiki_for_customer(conn: asyncpg.Connection, customer_id: str) -> None:
    """Drop the customer's compiled-wiki rows so re-bootstrap starts clean.

    Operates on the passed-in connection (which the trigger route holds
    inside its critical-section txn). Wipes:

      - ``wiki_links``                 (no RLS — explicit WHERE)
      - ``wiki_timeline_entries``      (no RLS — explicit WHERE)
      - ``wiki_raw_data``              (no RLS — explicit WHERE)
      - ``documents`` rows with doc_class='compiled_wiki'
        (RLS — but the trigger route binds ``app.current_customer_id``
        on this conn before calling).

    NOT wiped:
      - ``documents`` rows with doc_class='manual_entry' (human-authored).
      - ``documents`` rows with doc_class='agent_artifact' (the auto-
        generated wiki index — bootstrap regenerates this on first
        commit; leaving the prior version in place keeps the dashboard
        from flashing empty).
      - ``wiki_synthesis_queue`` (untouched — daily replay handles it
        via the bootstrap_absorbed marker; locked decision #3).
    """
    await conn.execute(
        "DELETE FROM wiki_links WHERE customer_id = $1",
        customer_id,
    )
    await conn.execute(
        "DELETE FROM wiki_timeline_entries WHERE customer_id = $1",
        customer_id,
    )
    await conn.execute(
        "DELETE FROM wiki_raw_data WHERE customer_id = $1",
        customer_id,
    )
    # documents has RLS — the GUC bound on this conn powers the policy.
    # The explicit WHERE customer_id is defense-in-depth.
    await conn.execute(
        """
        DELETE FROM documents
        WHERE customer_id = $1
          AND doc_class = 'compiled_wiki'
        """,
        customer_id,
    )


async def _insert_pending_runs(
    conn: asyncpg.Connection,
    *,
    customer_id: str,
    sources: list[str],
) -> dict[str, int]:
    """Insert one wiki_synthesis_runs row per source at status='pending'.

    Single batched INSERT so a partial failure can't leak orphaned
    rows. Returns ``{source: run_id}`` matching the input order.
    """
    if not sources:
        return {}
    rows = await conn.fetch(
        """
        INSERT INTO wiki_synthesis_runs
            (customer_id, kind, stage, source, status)
        SELECT $1, 'bootstrap', 'synthesis', s, 'pending'
        FROM unnest($2::text[]) AS s
        RETURNING run_id, source
        """,
        customer_id,
        sources,
    )
    return {row["source"]: int(row["run_id"]) for row in rows}


async def _do_trigger_wiki_backfill(
    body: BackfillTriggerBody,
    customer_id: str,
    force: bool,
) -> BackfillTriggerResponse:
    """Shared handler for both /backfill/trigger and the legacy
    /bootstrap/trigger alias.

    Inserts one ``wiki_synthesis_runs`` row per requested source at
    ``status='pending'`` and fires a payload-less ``pg_notify`` on
    ``WIKI_BACKFILL_CHANNEL`` as a wake hint. ``BackfillWorker``s on
    every wiki-backfill fly machine claim rows via
    ``FOR UPDATE SKIP LOCKED`` and run crawlers in parallel.
    """
    requested = body.sources
    if requested is None:
        sources = sorted(BACKFILL_CRAWLER_REGISTRY.keys())
    else:
        unknown = [s for s in requested if s not in BACKFILL_CRAWLER_REGISTRY]
        if unknown:
            raise HTTPException(
                status_code=400,
                detail=f"unknown sources: {sorted(unknown)}",
            )
        sources = list(requested)

    lock_key = advisory_lock_key("backfill-trigger", customer_id)
    cancelled_ids: list[int] = []

    # Critical section: hold the per-customer trigger lock for in-flight
    # check + cancel UPDATE + wipe + new pending insert. Workers claim
    # via FOR UPDATE SKIP LOCKED on the same table, so this lock has no
    # contention with crawls already in flight.
    async with with_tenant(customer_id) as conn:
        await conn.execute("SELECT pg_advisory_xact_lock($1)", lock_key)

        in_flight = await _get_in_flight_runs(conn, customer_id)
        if in_flight and not force:
            running_sources = [str(r["source"]) for r in in_flight if r["status"] == "running"]
            pending_sources = [str(r["source"]) for r in in_flight if r["status"] == "pending"]
            run_ids = [int(r["run_id"]) for r in in_flight]
            started_at = in_flight[0]["started_at"] if in_flight else None
            log.info(
                "wiki.backfill.trigger.in_flight",
                customer=customer_id,
                run_ids=run_ids,
                running=running_sources,
                pending=pending_sources,
            )
            # FastAPI/Pydantic doesn't natively serialize datetimes inside
            # HTTPException.detail; coerce to isoformat for transport.
            raise HTTPException(
                status_code=409,
                detail={
                    "status": "in_flight",
                    "run_ids": run_ids,
                    "started_at": started_at.isoformat() if started_at else None,
                    "sources_running": running_sources,
                    "sources_pending": pending_sources,
                },
            )

        if in_flight and force:
            cancelled_ids = [int(r["run_id"]) for r in in_flight]
            await conn.execute(
                """
                UPDATE wiki_synthesis_runs
                   SET status = 'cancelled',
                       finished_at = NOW(),
                       error = COALESCE(error, '')
                           || (CASE WHEN error IS NULL OR error = '' THEN ''
                                    ELSE ' | ' END)
                           || 'cancelled by force-trigger'
                 WHERE run_id = ANY($1::bigint[])
                """,
                cancelled_ids,
            )
            cancel_payload = orjson.dumps(
                {"customer_id": customer_id, "run_ids": cancelled_ids}
            ).decode("utf-8")
            await conn.execute(
                "SELECT pg_notify($1, $2)",
                WIKI_BACKFILL_CANCEL_CHANNEL,
                cancel_payload,
            )
            log.info(
                "wiki.backfill.trigger.cancel_fired",
                customer=customer_id,
                cancelled_run_ids=cancelled_ids,
            )
            # Cooperative drain: sleep so workers have a chance to
            # finish current API calls / commit in-flight pages. We
            # don't need an ack channel — the per-page advisory lock
            # in wiki_agent prevents post-wipe writes from a stuck
            # worker from clobbering new backfill output.
            await asyncio.sleep(BACKFILL_CANCEL_DRAIN_TIMEOUT_SECONDS)

        if body.wipe_first:
            await _wipe_wiki_for_customer(conn, customer_id)

        run_ids_map = await _insert_pending_runs(
            conn,
            customer_id=customer_id,
            sources=sources,
        )
        # Wake hint to any LISTENing worker. Empty payload — workers
        # claim rows via FOR UPDATE SKIP LOCKED, no per-NOTIFY routing
        # needed.
        await conn.execute(
            "SELECT pg_notify($1, '')",
            WIKI_BACKFILL_CHANNEL,
        )

    log.info(
        "wiki.backfill.trigger",
        customer=customer_id,
        sources=sources,
        wipe_first=body.wipe_first,
        reason=body.reason,
        force=force,
        run_ids=run_ids_map,
        cancelled_run_ids=cancelled_ids or None,
    )
    return BackfillTriggerResponse(
        triggered=True,
        run_ids=[run_ids_map[s] for s in sources],
    )


async def _do_get_wiki_backfill_status(customer_id: str) -> BackfillStatusResponse:
    """Shared handler for both /backfill/status and the legacy
    /bootstrap/status alias.
    """
    async with raw_conn() as conn:
        anchor = await conn.fetchrow(
            """
            SELECT started_at FROM wiki_synthesis_runs
            WHERE customer_id = $1 AND kind = 'bootstrap'
            ORDER BY started_at DESC
            LIMIT 1
            """,
            customer_id,
        )
        if anchor is None:
            return BackfillStatusResponse(
                in_progress=False,
                started_at=None,
                sources_attempted=[],
                sources_succeeded=[],
                sources_failed={},
                pages_created=0,
                pages_updated=0,
                targets={},
            )
        rows = await conn.fetch(
            """
            SELECT source, target, status, error, pages_created,
                   pages_updated, started_at
            FROM wiki_synthesis_runs
            WHERE customer_id = $1
              AND kind = 'bootstrap'
              AND started_at >= $2::timestamptz - INTERVAL '60 seconds'
              AND started_at <= NOW()
            ORDER BY started_at ASC
            """,
            customer_id,
            anchor["started_at"],
        )

    sources_attempted: list[str] = []
    sources_succeeded: list[str] = []
    sources_failed: dict[str, str] = {}
    # Phase 2 per-target progress. Keyed by source -> target -> status
    # (one of: pending, running, complete, partial, failed, cancelled).
    # Phase 1 rows (target=NULL) flow into sources_* above; Phase 2
    # rows (target set) flow into this dict only.
    targets: dict[str, dict[str, str]] = {}
    in_progress = False
    pages_created = 0
    pages_updated = 0
    started_at = rows[0]["started_at"] if rows else None
    for row in rows:
        source = row["source"] or ""
        target = row["target"]
        status = row["status"]
        # Both 'pending' (queued, not yet claimed) and 'running' (a
        # worker is actively crawling) count as in-progress so the
        # dashboard's "Backfilling..." pill stays up across the claim
        # boundary.
        if status in ("pending", "running"):
            in_progress = True
        pages_created += int(row["pages_created"] or 0)
        pages_updated += int(row["pages_updated"] or 0)
        if target is None:
            sources_attempted.append(source)
            if status in ("complete", "partial"):
                sources_succeeded.append(source)
            elif status == "failed":
                sources_failed[source] = row["error"] or "failed"
            elif status == "cancelled":
                sources_failed[source] = row["error"] or "cancelled"
        else:
            targets.setdefault(source, {})[target] = status

    return BackfillStatusResponse(
        in_progress=in_progress,
        started_at=started_at,
        sources_attempted=sources_attempted,
        sources_succeeded=sources_succeeded,
        sources_failed=sources_failed,
        pages_created=pages_created,
        pages_updated=pages_updated,
        targets=targets,
    )


_FORCE_QUERY = Query(
    default=False,
    description=(
        "Cancel any in-flight backfill for this customer and start a "
        "fresh one. Without force, an in-flight run returns 409."
    ),
)


@router.post(
    "/backfill/trigger",
    response_model=BackfillTriggerResponse,
    status_code=202,
    dependencies=[Depends(verify_internal_knowledge_key)],
)
async def trigger_wiki_backfill(
    body: BackfillTriggerBody,
    customer_id: str = Depends(_require_customer),
    force: bool = _FORCE_QUERY,
) -> BackfillTriggerResponse:
    """Enqueue a wiki backfill for ``customer_id``."""
    return await _do_trigger_wiki_backfill(body, customer_id, force)


@router.post(
    "/bootstrap/trigger",
    response_model=BackfillTriggerResponse,
    status_code=202,
    dependencies=[Depends(verify_internal_knowledge_key)],
)
async def trigger_wiki_bootstrap(
    body: BackfillTriggerBody,
    customer_id: str = Depends(_require_customer),
    force: bool = _FORCE_QUERY,
) -> BackfillTriggerResponse:
    """Legacy alias for ``POST /api/wiki/backfill/trigger``."""
    return await _do_trigger_wiki_backfill(body, customer_id, force)


@router.get(
    "/backfill/status",
    response_model=BackfillStatusResponse,
    dependencies=[Depends(verify_internal_knowledge_key)],
)
async def get_wiki_backfill_status(
    customer_id: str = Depends(_require_customer),
) -> BackfillStatusResponse:
    """Aggregate over the most-recent backfill burst for ``customer_id``."""
    return await _do_get_wiki_backfill_status(customer_id)


@router.get(
    "/bootstrap/status",
    response_model=BackfillStatusResponse,
    dependencies=[Depends(verify_internal_knowledge_key)],
)
async def get_wiki_bootstrap_status(
    customer_id: str = Depends(_require_customer),
) -> BackfillStatusResponse:
    """Legacy alias for ``GET /api/wiki/backfill/status``."""
    return await _do_get_wiki_backfill_status(customer_id)


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
