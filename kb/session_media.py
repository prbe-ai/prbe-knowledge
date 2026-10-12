"""Pasted images of captured coding-agent sessions: upload, promote, sweep, read.

A capture client that pastes an image into a session (Claude Code, Codex, pi,
Kimi Code, Capy) uploads its bytes straight to object storage, outside the
session batch, and names it in the batch by its SHA-256 (probe-events/1
`media_block.sha256`; fragment version 2's image part). Only while this
customer is advertised `accepts.media` (kb/session_receipts.py
`media_enabled`: SESSION_MEDIA_CUSTOMERS / SESSION_MEDIA_ALL).

    client --POST .../media {sha256, size_bytes, media_type}--> this route
           <-- presigned PUT for the STAGING key (15 min; signs length, type, sha256)
    client --PUT bytes--> raw/<src>/<customer>/media-staging/<session>/<sha256>
    client --batch naming the image--> receipt door (kb/session_receipts.py)
    worker pass (`promote_named_media`, never the request path)
           --server-side copy--> raw/<src>/<customer>/<session>/media/<sha256>.<ext>
    reader --GET .../media/<sha256>--> presigned GET (5 min, Content-Type pinned)

Why staging. Bytes land under the session's folder only once a batch naming
them was accepted, and only while the session is not deleted (the promotion
takes the session lock and asks `session_deletions` under it, like every other
writer: engine/shared/session_suppression.py). The copy finishes while that
lock is held, even when the pass is cancelled (`_to_completion`): a deletion
waiting on the lock lists the session's folder only after the copy landed. A
URL handed out before a deletion can still be used for 15 minutes; its bytes
stay in staging, where the deletion and then the 24-hour sweep remove them. A
batch that never comes leaves only a staging object, also swept.

Owed promotions. The receipt door stamps `named_at` on the image's row in the
same transaction that accepts a batch naming it (the worker stamps it too, for
an upload requested after its batch). From then on the promotion is owed: a
pass that could not promote (the staged bytes not there yet, the store down)
is retried by the hourly sweep, and the staged object is kept past the usual
24 hours, for PROMOTION_RETRY_S (7 days) after it was named. Then the sweep
gives up (`session_media.promotion_abandoned`) and the image keeps its
placeholder. The read route never promotes: copies stay off the request path.

Route contract (internal, the receipt door's auth: X-Internal-Knowledge-Key;
the tenant from X-Prbe-Customer, never the body)

    POST /api/sessions/{source}/{session_id}/media
        {"sha256": "<64 lowercase hex>", "size_bytes": <int >= 1>,
         "media_type": "image/jpeg" | "image/png" | "image/gif" | "image/webp"}
      200 {"status": "have"}       already in the session's folder; send nothing
      200 {"status": "upload", "method": "PUT", "url": "...",
           "headers": {"Content-Length", "Content-Type", "x-amz-checksum-sha256"},
           "expires_at": "<ISO 8601>"}
          PUT the bytes to `url` with exactly `headers`; anything else is refused
          by the store. Then send the batch that names the image.
      403 {"detail": {"reason": "media_disabled"}}
      409 {"detail": {"reason": "tenant_not_active", ...}} | "session capture source is disconnected"
      410 {"detail": {"reason": "session_deleted", ...}}
      413 {"detail": {"reason": "media_too_large", "max_bytes": 5368709120}}
      415 {"detail": {"reason": "media_type_unsupported", "media_types": [...]}}
      422 malformed source, session id (a UUID), sha256 or size_bytes
      429 {"detail": {"reason": "media_budget", "limit_bytes", "used_bytes", "window_s"}}
      503 {"detail": {"reason": "media_unavailable"}}   the table is not migrated yet

    GET /api/sessions/{source}/{session_id}/media/{sha256}
      200 {"url", "media_type", "size_bytes", "expires_at"}   a presigned GET whose
          response is served as `media_type`
      404 {"detail": {"reason": "media_not_found"}}   never uploaded, not promoted
          (yet), or gone
      410 {"detail": {"reason": "session_deleted", ...}}

The read is here, beside the upload, and not on retrieval beside
GET /trajectory: it needs the same key layout, deletion check and internal
auth research-os already sends to this service for the receipts read, and it
reads no document. The caller (research-os) checks the reader may see the
session; this route checks only the tenant.
"""

from __future__ import annotations

import asyncio
import base64
import re
from collections import Counter
from collections.abc import Awaitable, Iterable, Mapping
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any
from uuid import UUID

import asyncpg
import structlog
from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, StrictInt, StrictStr

from engine.ingest.atif.fold import media_path
from engine.ingest.atif.fragment import IMAGE_MEDIA_TYPES, PieceType
from engine.ingest.connectedness import is_source_connected
from engine.shared.config import get_settings
from engine.shared.constants import AGENT_SESSION_SOURCES, SourceSystem
from engine.shared.db import raw_conn, with_tenant
from engine.shared.exceptions import StorageNotFound
from engine.shared.session_suppression import (
    LEGAL_HOLD_VALUE_SQL,
    MEDIA_FOLDER,
    MEDIA_STAGING_FOLDER,
    deleted_sessions,
    is_own_folder_key,
    lock_session,
    media_staging_prefix,
    session_folder,
)
from engine.shared.storage import get_store
from engine.shared.tenant_status import refusal_for
from kb.admin_routes import verify_internal_knowledge_key
from kb.session_receipts import _SOURCES, media_enabled, refuse_deleted

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/sessions", dependencies=[Depends(verify_internal_knowledge_key)])

#: The largest image one upload URL is handed out for.
MEDIA_MAX_BYTES = 5 * 1024**3
#: How long an upload URL works.
UPLOAD_URL_TTL_S = 15 * 60
#: How long a read URL works: long enough to load, short enough not to share.
READ_URL_TTL_S = 5 * 60
#: A staged image older than this is swept, promoted or not; an unpromoted row
#: older than this is dropped -- unless an accepted batch named it within
#: PROMOTION_RETRY_S. Also the daily budget's window.
STAGING_TTL_S = 24 * 3600
#: How long a named, unpromoted image is retried (hourly, by the sweep) and its
#: staged bytes kept, from when an accepted batch named it. Covers a store
#: outage over a weekend; past it the image keeps its placeholder.
PROMOTION_RETRY_S = 7 * 24 * 3600
#: The sweep also deletes any image object of a session whose deletion was
#: requested this recently: a copy that a store finished after its client gave
#: up (a timeout) can land after the deletion listed the folder.
LATE_MEDIA_WINDOW_S = 2 * 24 * 3600

_SHA256_HEX = re.compile(r"[0-9a-f]{64}\Z")


class MediaRefusal(StrEnum):
    """`detail.reason` of a refused request; research-os matches on these."""

    DISABLED = "media_disabled"
    BUDGET = "media_budget"
    TOO_LARGE = "media_too_large"
    TYPE_UNSUPPORTED = "media_type_unsupported"
    NOT_FOUND = "media_not_found"
    UNAVAILABLE = "media_unavailable"


class Promotion(StrEnum):
    """What one promotion attempt did (a log field, counted per pass)."""

    PROMOTED = "promoted"
    #: Nothing staged under its key: the upload never finished, or the sweep
    #: took it. The image keeps its placeholder; a later pass tries again.
    MISSING = "missing"
    #: The staged object is not what the URL was signed for (size or digest):
    #: never promoted; the sweep removes it.
    MISMATCH = "mismatch"
    #: Already promoted, or its row is gone, by the time the lock was held.
    SKIPPED = "skipped"
    #: The session was deleted: nothing is written.
    DELETED = "deleted"
    #: The store or the database failed: logged; the sweep retries it.
    FAILED = "failed"


# --- keys ---------------------------------------------------------------------


def media_key(source: str, customer_id: str, session_id: str, sha256: str, media_type: str) -> str:
    """Where a promoted image lives: in the session's folder, at the path its
    trajectory's image part names (fold.media_path)."""
    return f"{session_folder(source, customer_id, session_id)}{media_path(sha256, media_type)}"


def staging_key(source: str, customer_id: str, session_id: str, sha256: str) -> str:
    return f"{media_staging_prefix(source, customer_id, session_id)}{sha256}"


def checksum_b64(sha256: str) -> str:
    """The SHA-256 as S3's `x-amz-checksum-sha256` carries it: base64 of the digest."""
    return base64.b64encode(bytes.fromhex(sha256)).decode()


# --- what a session's batches name ---------------------------------------------


def named_media(hydrated: Mapping[str, Any]) -> set[str]:
    """The SHA-256 of every image a session's batches name: a user event's
    image block with `sha256` (protocol 2, and a protocol-3 canary's events)
    and a fragment's image part (protocol 3, fragment version 2). Malformed
    values are ignored; whether a digest is the session's is the row's to say."""
    found: set[str] = set()
    for key in ("events", "canary_events"):
        for event in hydrated.get(key) or ():
            raw = event.get("raw") if isinstance(event, dict) else None
            message = raw.get("message") if isinstance(raw, dict) else None
            content = message.get("content") if isinstance(message, dict) else None
            if not (isinstance(raw, dict) and raw.get("type") == "user" and isinstance(content, list)):
                continue
            for block in content:
                if isinstance(block, dict) and block.get("type") == "image":
                    _add_digest(found, block.get("sha256"))
    for item in hydrated.get("fragments") or ():
        parts = item.get("parts") if isinstance(item, dict) else None
        for part in parts if isinstance(parts, list) else ():
            if isinstance(part, dict) and part.get("type") == PieceType.IMAGE.value:
                _add_digest(found, part.get("sha256"))
    return found


def _add_digest(found: set[str], value: Any) -> None:
    if isinstance(value, str) and _SHA256_HEX.match(value):
        found.add(value)


# --- the table ------------------------------------------------------------------


async def _pending(conn: Any, customer_id: str, source: str, session_id: str, shas: list[str]) -> list[str]:
    """Which of `shas` this session has an unpromoted upload for. No table
    yet (code ahead of migration 0149) reads as none, inside a savepoint so the
    caller's transaction survives."""
    try:
        async with conn.transaction():
            rows = await conn.fetch(
                "SELECT sha256 FROM session_media WHERE customer_id = $1 AND source_system = $2 "
                "AND session_id = $3 AND promoted_at IS NULL AND sha256 = ANY($4::text[]) "
                "ORDER BY sha256",
                customer_id,
                source,
                session_id,
                shas,
            )
    except asyncpg.UndefinedTableError:
        return []
    return [r["sha256"] for r in rows]


async def mark_named(
    conn: Any, customer_id: str, source: str, session_id: str, shas: Iterable[str]
) -> int:
    """Stamp `named_at` on this session's rows for `shas` (first naming wins):
    their promotion is owed from now on. The receipt door calls it inside the
    transaction that accepts the batch, so an accepted batch's images are never
    forgotten. 0 before migration 0149."""
    wanted = sorted({s for s in shas if isinstance(s, str) and _SHA256_HEX.match(s)})
    if not wanted:
        return 0
    try:
        async with conn.transaction():
            status = await conn.execute(
                "UPDATE session_media SET named_at = now() WHERE customer_id = $1 "
                "AND source_system = $2 AND session_id = $3 AND sha256 = ANY($4::text[]) "
                "AND named_at IS NULL",
                customer_id,
                source,
                session_id,
                wanted,
            )
    except asyncpg.UndefinedTableError:
        return 0
    return _affected(status)


def _affected(status: str) -> int:
    try:
        return int(status.rsplit(" ", 1)[1])
    except (IndexError, ValueError):
        return 0


async def session_media_count(conn: Any, customer_id: str, source: str, session_id: str) -> int:
    """This session's rows (a deletion's inventory); 0 before migration 0149."""
    try:
        async with conn.transaction():
            return int(
                await conn.fetchval(
                    "SELECT count(*) FROM session_media WHERE customer_id = $1 "
                    "AND source_system = $2 AND session_id = $3",
                    customer_id,
                    source,
                    session_id,
                )
                or 0
            )
    except asyncpg.UndefinedTableError:
        return 0


async def delete_session_media_rows(conn: Any, customer_id: str, source: str, session_id: str) -> int:
    """Remove this session's rows (a deletion, under its lock); 0 before
    migration 0149."""
    try:
        async with conn.transaction():
            status = await conn.execute(
                "DELETE FROM session_media WHERE customer_id = $1 AND source_system = $2 "
                "AND session_id = $3",
                customer_id,
                source,
                session_id,
            )
    except asyncpg.UndefinedTableError:
        return 0
    return _affected(status)


# --- promotion (the worker) ---------------------------------------------------------


async def promote_named_media(
    customer_id: str,
    source: str,
    session_id: str,
    named: Iterable[str],
    *,
    store: Any = None,
) -> dict[Promotion, int]:
    """Promote every staged image of this session that its batches name and
    that is not promoted yet. Idempotent; returns counts by `Promotion` outcome.

    Each image: under the session's lock, refused for a deleted session; the
    staged object must be exactly what its URL was signed for (size, and the
    SHA-256 the store reports, where it keeps one); then a server-side copy
    into the session's folder, `promoted_at`, commit, and the staged object is
    deleted. A missing staged object or a failing store is not the pass's
    error: logged, the image keeps its placeholder, and a later pass of the
    session or the hourly sweep (`named_at`, stamped here too) tries again.
    """
    wanted = sorted({s for s in named if isinstance(s, str) and _SHA256_HEX.match(s)})
    counts: Counter[Promotion] = Counter()
    if not wanted:
        return {}
    async with with_tenant(customer_id) as conn:
        # An upload asked for after the batch that names it has no stamp yet.
        await mark_named(conn, customer_id, source, session_id, wanted)
        pending = await _pending(conn, customer_id, source, session_id, wanted)
    if not pending:
        return {}
    store = store or get_store()
    bucket = await store.bucket_for(customer_id)
    for sha256 in pending:
        try:
            outcome = await _promote_one(store, bucket, customer_id, source, session_id, sha256)
        except Exception:
            log.warning(
                "session_media.promotion_failed",
                customer=customer_id,
                source=source,
                session_id=session_id,
                sha256=sha256,
                exc_info=True,
            )
            outcome = Promotion.FAILED
        counts[outcome] += 1
        if outcome == Promotion.DELETED:
            break
    log.info(
        "session_media.promotion",
        customer=customer_id,
        source=source,
        session_id=session_id,
        **{outcome.value: n for outcome, n in counts.items()},
    )
    return dict(counts)


async def _promote_one(
    store: Any, bucket: str, customer_id: str, source: str, session_id: str, sha256: str
) -> Promotion:
    staged = staging_key(source, customer_id, session_id, sha256)
    async with with_tenant(customer_id) as conn:
        await lock_session(conn, customer_id, source, session_id)
        if await deleted_sessions(conn, customer_id, source, [session_id]):
            log.info(
                "session_media.promotion_refused_deleted",
                customer=customer_id,
                source=source,
                session_id=session_id,
            )
            return Promotion.DELETED
        # FOR UPDATE: the sweep's DELETE of an old unpromoted row waits for
        # this, then sees it promoted and leaves it.
        row = await conn.fetchrow(
            "SELECT media_type, size_bytes FROM session_media WHERE customer_id = $1 "
            "AND source_system = $2 AND session_id = $3 AND sha256 = $4 "
            "AND promoted_at IS NULL FOR UPDATE",
            customer_id,
            source,
            session_id,
            sha256,
        )
        if row is None:
            return Promotion.SKIPPED
        head = await store.head(bucket, staged)
        if head is None:
            log.info(
                "session_media.staging_missing",
                customer=customer_id,
                source=source,
                session_id=session_id,
                sha256=sha256,
            )
            return Promotion.MISSING
        if head.size != row["size_bytes"] or (
            head.checksum_sha256 is not None and head.checksum_sha256 != checksum_b64(sha256)
        ):
            log.warning(
                "session_media.staging_mismatch",
                customer=customer_id,
                source=source,
                session_id=session_id,
                sha256=sha256,
                size=head.size,
                expected_size=row["size_bytes"],
                checksum_reported=head.checksum_sha256 is not None,
            )
            return Promotion.MISMATCH
        try:
            # To its end while the lock is held, cancelled or not: a cancelled
            # await would release the lock while the copy's thread still runs,
            # and a deletion could finish before the copy lands.
            await _to_completion(
                store.copy(
                    bucket,
                    staged,
                    media_key(source, customer_id, session_id, sha256, row["media_type"]),
                    content_type=row["media_type"],
                )
            )
        except StorageNotFound:
            # Swept between the HEAD and the copy.
            return Promotion.MISSING
        await conn.execute(
            "UPDATE session_media SET promoted_at = now() WHERE customer_id = $1 "
            "AND source_system = $2 AND session_id = $3 AND sha256 = $4",
            customer_id,
            source,
            session_id,
            sha256,
        )
    try:
        await store.delete(bucket, staged)
    except Exception:
        # The sweep removes it within a day.
        log.warning("session_media.staging_delete_failed", customer=customer_id, exc_info=True)
    return Promotion.PROMOTED


async def _to_completion[T](awaitable: Awaitable[T]) -> T:
    """Await `awaitable` to its end even if this task is cancelled meanwhile;
    then re-raise the cancellation. For work whose effect must not outlive the
    caller's lock: `asyncio.to_thread` cannot stop its thread, so a plain
    cancelled await returns (and the caller's transaction, with its advisory
    lock, ends) while the work goes on."""
    task = asyncio.ensure_future(awaitable)
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.cancelled():
                raise
            cancelled = True
        except Exception:
            break  # the task ended with it; result() raises it below
    if cancelled:
        if not task.cancelled():
            task.exception()  # retrieved: never "exception was never retrieved"
        raise asyncio.CancelledError
    return task.result()


# --- the sweep (scripts/cron_session_completer.py) -------------------------------------


async def sweep_media_staging(
    *, older_than_s: int = STAGING_TTL_S, dry_run: bool = False, store: Any = None
) -> dict[str, int]:
    """The hourly upkeep of staged images, for every customer not under legal
    hold. Returns counts.

    1. Retry every owed promotion (named within PROMOTION_RETRY_S, not
       promoted), under the same lock and deletion check as the worker's.
    2. Delete staged objects older than `older_than_s`, except those still
       owed. Found by LISTING, not by row: a URL handed out before a session was
       deleted can still be used after the deletion removed the session's rows
       and its staging folder, and nothing else records that object.
    3. Drop unpromoted rows as old, except those still owed; an owed row past
       PROMOTION_RETRY_S is given up (`session_media.promotion_abandoned`).
    4. Delete any image object of a session whose deletion was requested within
       LATE_MEDIA_WINDOW_S: a copy a store finished after its client timed out
       can land after the deletion listed the session's folder.

    `dry_run` counts and changes nothing (no retry either). One customer's
    failure is counted (`customers_failed`) and the rest are swept.
    """
    store = store or get_store()
    cutoff = datetime.now(UTC) - timedelta(seconds=older_than_s)
    async with raw_conn() as conn:
        customers = await conn.fetch(
            f"SELECT c.customer_id, {LEGAL_HOLD_VALUE_SQL} AS hold FROM customers c "
            "ORDER BY c.customer_id"
        )
    counts: Counter[str] = Counter()
    for customer in customers:
        customer_id = customer["customer_id"]
        if customer["hold"] is not None:
            # Evidence now: nothing of a held tenant is deleted.
            counts["customers_held"] += 1
            continue
        try:
            await _sweep_customer(store, customer_id, cutoff, counts, dry_run=dry_run)
        except Exception:
            counts["customers_failed"] += 1
            log.warning("session_media.sweep_failed", customer=customer_id, exc_info=True)
    log.info("session_media.swept", dry_run=dry_run, customers=len(customers), **counts)
    return dict(counts)


async def _owed(customer_id: str) -> list[Any]:
    """This customer's images named within PROMOTION_RETRY_S and not promoted."""
    try:
        async with with_tenant(customer_id) as conn:
            return list(
                await conn.fetch(
                    "SELECT source_system, session_id, sha256 FROM session_media "
                    "WHERE customer_id = $1 AND promoted_at IS NULL AND named_at IS NOT NULL "
                    "AND named_at > now() - make_interval(secs => $2) "
                    "ORDER BY source_system, session_id, sha256",
                    customer_id,
                    PROMOTION_RETRY_S,
                )
            )
    except asyncpg.UndefinedTableError:
        return []


async def _sweep_customer(
    store: Any, customer_id: str, cutoff: datetime, counts: Counter[str], *, dry_run: bool
) -> None:
    bucket = await store.bucket_for(customer_id)
    owed = await _owed(customer_id)
    if owed and not dry_run:
        by_session: dict[tuple[str, str], list[str]] = {}
        for r in owed:
            by_session.setdefault((r["source_system"], r["session_id"]), []).append(r["sha256"])
        for (source, session_id), shas in by_session.items():
            outcomes = await promote_named_media(
                customer_id, source, session_id, shas, store=store
            )
            for outcome, n in outcomes.items():
                counts[f"retry_{outcome.value}"] += n
        owed = await _owed(customer_id)
    counts["owed"] += len(owed)
    kept = {staging_key(r["source_system"], customer_id, r["session_id"], r["sha256"]) for r in owed}
    old: list[str] = []
    for source in sorted(s.value for s in AGENT_SESSION_SOURCES):
        prefix = f"raw/{source}/{customer_id}/{MEDIA_STAGING_FOLDER}/"
        old.extend(
            e.key
            for e in await store.list_entries(bucket, prefix)
            if e.last_modified < cutoff and e.key not in kept
        )
    old.extend(await _late_media_of_deleted_sessions(store, bucket, customer_id))
    counts["objects"] += len(old)
    if old and not dry_run:
        deleted, failed = await store.delete_named_keys(bucket, old)
        counts["objects_deleted"] += deleted
        counts["objects_failed"] += len(failed)
    rows, abandoned = await _expire_rows(customer_id, cutoff, dry_run=dry_run)
    counts["rows"] += rows
    counts["abandoned"] += abandoned
    if abandoned:
        log.warning(
            "session_media.promotion_abandoned",
            customer=customer_id,
            images=abandoned,
            retried_for_s=PROMOTION_RETRY_S,
            dry_run=dry_run,
        )


async def _late_media_of_deleted_sessions(store: Any, bucket: str, customer_id: str) -> list[str]:
    """Image objects (promoted or staged) of sessions whose deletion was
    requested within LATE_MEDIA_WINDOW_S. Nothing may write them, so any there
    landed after the deletion's sweep."""
    try:
        async with with_tenant(customer_id) as conn:
            deleted = await conn.fetch(
                "SELECT source_system, session_id FROM session_deletions WHERE customer_id = $1 "
                "AND requested_at > now() - make_interval(secs => $2)",
                customer_id,
                LATE_MEDIA_WINDOW_S,
            )
    except asyncpg.UndefinedTableError:
        return []
    keys: list[str] = []
    for r in deleted:
        source, session_id = r["source_system"], r["session_id"]
        folder = session_folder(source, customer_id, session_id)
        keys.extend(
            k
            for k in await store.list_keys(bucket, f"{folder}{MEDIA_FOLDER}")
            if is_own_folder_key(k, folder)
        )
        keys.extend(
            await store.list_keys(bucket, media_staging_prefix(source, customer_id, session_id))
        )
    return keys


async def _expire_rows(customer_id: str, cutoff: datetime, *, dry_run: bool) -> tuple[int, int]:
    """Drop unpromoted rows older than `cutoff` that are not owed any more.
    Returns (rows, of them owed ones given up)."""
    sql = (
        "FROM session_media WHERE customer_id = $1 AND promoted_at IS NULL AND created_at < $2 "
        "AND (named_at IS NULL OR named_at <= now() - make_interval(secs => $3))"
    )
    try:
        async with with_tenant(customer_id) as conn:
            if dry_run:
                rows = await conn.fetch(
                    f"SELECT named_at IS NOT NULL AS named {sql}",
                    customer_id,
                    cutoff,
                    PROMOTION_RETRY_S,
                )
            else:
                rows = await conn.fetch(
                    f"DELETE {sql} RETURNING named_at IS NOT NULL AS named",
                    customer_id,
                    cutoff,
                    PROMOTION_RETRY_S,
                )
    except asyncpg.UndefinedTableError:
        return 0, 0
    return len(rows), sum(1 for r in rows if r["named"])


# --- routes -----------------------------------------------------------------------------


class MediaUpload(BaseModel):
    sha256: StrictStr
    size_bytes: StrictInt
    media_type: StrictStr


def _refuse(status: int, reason: MediaRefusal, **fields: Any) -> HTTPException:
    return HTTPException(status, {"reason": reason.value, **fields})


def _identity(source: str, session_id: str, sha256: str) -> None:
    if source not in _SOURCES:
        raise HTTPException(422, "unsupported transcript source")
    try:
        UUID(session_id)
    except ValueError as exc:
        raise HTTPException(422, "malformed session id") from exc
    if not _SHA256_HEX.match(sha256):
        raise HTTPException(422, "sha256 must be 64 lowercase hex characters")


@router.post("/{source}/{session_id}/media")
async def request_upload(
    source: str,
    session_id: str,
    body: MediaUpload,
    x_prbe_customer: str = Header(...),
) -> dict[str, Any]:
    customer = x_prbe_customer
    _identity(source, session_id, body.sha256)
    if body.size_bytes < 1:
        raise HTTPException(422, "size_bytes must be at least 1")
    if body.media_type not in IMAGE_MEDIA_TYPES:
        raise _refuse(415, MediaRefusal.TYPE_UNSUPPORTED, media_types=sorted(IMAGE_MEDIA_TYPES))
    if body.size_bytes > MEDIA_MAX_BYTES:
        raise _refuse(413, MediaRefusal.TOO_LARGE, max_bytes=MEDIA_MAX_BYTES)
    if (refusal := await refusal_for(customer)) is not None:
        raise HTTPException(409, refusal)
    settings = get_settings()
    if not media_enabled(customer, settings):
        raise _refuse(403, MediaRefusal.DISABLED)
    if not await is_source_connected(customer, SourceSystem(source)):
        raise HTTPException(409, "session capture source is disconnected")
    store = get_store()
    bucket = await store.bucket_for(customer)
    final = media_key(source, customer, session_id, body.sha256, body.media_type)
    try:
        async with with_tenant(customer) as conn:
            await lock_session(conn, customer, source, session_id)
            await refuse_deleted(conn, customer, source, session_id)
            if await store.exists(bucket, final):
                return {"status": "have"}
            # One customer's uploads are budgeted one at a time: two sessions'
            # requests must not both fit under the last free bytes.
            await conn.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
                f"session-media-budget:{customer}",
            )
            used = int(
                await conn.fetchval(
                    "SELECT COALESCE(sum(size_bytes), 0) FROM session_media "
                    "WHERE customer_id = $1 AND created_at > now() - make_interval(secs => $2) "
                    "AND NOT (source_system = $3 AND session_id = $4 AND sha256 = $5)",
                    customer,
                    STAGING_TTL_S,
                    source,
                    session_id,
                    body.sha256,
                )
            )
            limit = settings.session_media_daily_bytes
            if used + body.size_bytes > limit:
                log.info(
                    "session_media.budget_refused",
                    customer=customer,
                    source=source,
                    used_bytes=used,
                    size_bytes=body.size_bytes,
                    limit_bytes=limit,
                )
                raise _refuse(
                    429, MediaRefusal.BUDGET, limit_bytes=limit, used_bytes=used, window_s=STAGING_TTL_S
                )
            # A repeat (an expired URL, a retry) restamps the row: it counts
            # once, and a promoted image whose copy went missing is staged again.
            await conn.execute(
                "INSERT INTO session_media (customer_id, source_system, session_id, sha256, "
                "size_bytes, media_type) VALUES ($1, $2, $3, $4, $5, $6) "
                "ON CONFLICT (customer_id, source_system, session_id, sha256) DO UPDATE SET "
                "size_bytes = EXCLUDED.size_bytes, media_type = EXCLUDED.media_type, "
                "created_at = now(), promoted_at = NULL",
                customer,
                source,
                session_id,
                body.sha256,
                body.size_bytes,
                body.media_type,
            )
    except asyncpg.UndefinedTableError as exc:
        raise _refuse(503, MediaRefusal.UNAVAILABLE) from exc
    put = store.presign_put(
        bucket,
        staging_key(source, customer, session_id, body.sha256),
        size=body.size_bytes,
        checksum_sha256=checksum_b64(body.sha256),
        content_type=body.media_type,
        expires_in=UPLOAD_URL_TTL_S,
    )
    return {
        "status": "upload",
        "method": put.method,
        "url": put.url,
        "headers": put.headers,
        "expires_at": put.expires_at.isoformat(),
    }


@router.get("/{source}/{session_id}/media/{sha256}")
async def read_media(
    source: str,
    session_id: str,
    sha256: str,
    x_prbe_customer: str = Header(...),
) -> dict[str, Any]:
    customer = x_prbe_customer
    _identity(source, session_id, sha256)
    try:
        async with with_tenant(customer) as conn:
            await refuse_deleted(conn, customer, source, session_id)
            row = await conn.fetchrow(
                "SELECT media_type, size_bytes FROM session_media WHERE customer_id = $1 "
                "AND source_system = $2 AND session_id = $3 AND sha256 = $4 "
                "AND promoted_at IS NOT NULL",
                customer,
                source,
                session_id,
                sha256,
            )
    except asyncpg.UndefinedTableError:
        row = None
    if row is None:
        raise _refuse(404, MediaRefusal.NOT_FOUND)
    store = get_store()
    bucket = await store.bucket_for(customer)
    key = media_key(source, customer, session_id, sha256, row["media_type"])
    if not await store.exists(bucket, key):
        raise _refuse(404, MediaRefusal.NOT_FOUND)
    got = store.presign_get(
        bucket, key, response_content_type=row["media_type"], expires_in=READ_URL_TTL_S
    )
    return {
        "url": got.url,
        "media_type": row["media_type"],
        "size_bytes": row["size_bytes"],
        "expires_at": got.expires_at.isoformat(),
    }
