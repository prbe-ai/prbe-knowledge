"""Granola connector — meeting notes ingestion via REST polling.

Granola does not offer webhooks (as of 2026-04). Integration is poll-only:

    services/ingestion/poller (Fly app `prbe-knowledge-poller`)
        │  every 5 min: re_enqueue_for_polling(customer, granola)
        │  on pg_notify('granola_refresh', customer): same
        ▼
    backfill_state.status = 'pending'
        │
        ▼
    BackfillWorker (in worker process, LISTENs same channel)
        │  claim row, call connector.backfill(customer, token, cursor)
        ▼
    GranolaConnector.backfill yields WebhookEvent per note
        │  paginates Granola's /v1/notes with cursor + created_after watermark
        ▼
    backfill_runner persists raw to R2, INSERTs ingestion_queue rows
        ▼
    Worker.process drains queue, calls GranolaConnector.normalize per note
        ▼
    documents + chunks + graph

Auth:
    Static API key copy-pasted from Granola desktop app → Settings → API.
    Personal tier sees only issuing user's notes + shared. Enterprise tier
    sees whole workspace. Tier is stored in `integration_tokens.scope` as
    "tier:personal" or "tier:enterprise".

Cursor format (opaque to backfill_runner — connector owns the schema):
    JSON {"watermark": "<ISO8601>", "page_cursor": "<granola opaque>" | null}
    - watermark: high-water `created_at` we've seen. Used as `created_after`
      query param on the next sync to do incremental polling.
    - page_cursor: Granola's opaque pagination cursor inside one tick.
      Reset to null between ticks; only meaningful within a single backfill run.

Rate limit:
    Granola enforces 5 rps / 25 in 5s burst. We sleep
    GRANOLA_REQUEST_INTERVAL_SECONDS between calls (~4 rps). The single-instance
    poller + single-instance worker keep us within budget without a distributed
    rate limiter.

ACL: meeting owner gets READ. Personal tier inherently scopes to the owner,
so Phase 0 stores owner-only ACL and lets Phase 1 enforcement decide what to
do with shared notes.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime
from typing import Any, ClassVar

import httpx

from services.ingestion.chunker import count_tokens
from services.ingestion.handlers.base import Connector
from services.ingestion.handlers.registry import register_connector
from shared.constants import (
    GRANOLA_REQUEST_INTERVAL_SECONDS,
    DocClass,
    DocType,
    EdgeType,
    NodeLabel,
    Permission,
    PrincipalType,
    SourceSystem,
)
from shared.exceptions import (
    InvalidWebhookPayload,
    PermanentSourceError,
    RateLimited,
    TransientSourceError,
)
from shared.logging import get_logger
from shared.models import (
    ACLPrincipal,
    ACLSnapshot,
    ACLSnapshotRow,
    Document,
    GraphEdgeSpec,
    GraphNodeSpec,
    IntegrationToken,
    NormalizationResult,
    WebhookEvent,
    WebhookParseResult,
)

log = get_logger(__name__)

_GRANOLA_API = "https://public-api.granola.ai/v1"
_PAGE_SIZE = 50
# Defensive cap on pages per backfill tick. A 5K-note backfill is 100 pages;
# we'd sustain that. Beyond 1000 pages something is looping — bail and let
# the next tick resume from the cursor we last persisted.
_MAX_PAGES_PER_TICK = 1000


@register_connector(SourceSystem.GRANOLA)
class GranolaConnector(Connector):
    source_system: ClassVar[SourceSystem] = SourceSystem.GRANOLA
    display_name: ClassVar[str] = "Granola"

    # ---- 1. signature verification ----------------------------------------
    #
    # Granola has no inbound webhooks. The /webhooks/granola route would 401
    # for any caller, but we never expose it because the poller drives
    # everything through backfill().

    def verify_signature(
        self,
        headers: Mapping[str, str],
        raw_body: bytes,
    ) -> bool:
        return False

    # ---- 2. event parsing -------------------------------------------------

    def parse_webhook_event(
        self,
        customer_id: str,
        headers: Mapping[str, str],
        raw_payload: Mapping[str, Any],
    ) -> WebhookParseResult | None:
        # Backfill events synthesized by this connector flow through the
        # worker without re-entering parse_webhook_event — they go straight
        # to normalize via the queue. So this should never be called in
        # practice. If something does call it (manual replay, future
        # webhook support), accept the synthesized shape.
        note = raw_payload.get("note")
        if not isinstance(note, dict) or not note.get("id"):
            raise InvalidWebhookPayload(
                "granola payload missing 'note' object with id"
            )
        note_id = str(note["id"])
        received_at = _parse_iso(note.get("created_at")) or datetime.now(UTC)
        return WebhookParseResult(
            source_event_id=note_id,
            received_at=received_at,
        )

    # ---- 3. hydration -----------------------------------------------------
    #
    # Backfill yields fully-hydrated events (we already called GET /notes/{id}
    # ?include=transcript when paginating). No second fetch needed.

    async def fetch_supplementary(
        self,
        event: WebhookEvent,
        token: IntegrationToken | None,
    ) -> dict[str, Any]:
        return {}

    # ---- 4. normalization -------------------------------------------------

    async def normalize(
        self,
        event: WebhookEvent,
        hydrated: Mapping[str, Any],
    ) -> NormalizationResult:
        note = event.raw_payload.get("note")
        if not isinstance(note, dict) or not note.get("id"):
            return NormalizationResult(
                skipped_reason="granola event missing note.id"
            )

        note_id = str(note["id"])
        title = (note.get("title") or "").strip() or None
        summary = (note.get("summary") or "").strip()
        owner = note.get("owner") or {}
        owner_email = (owner.get("email") or "").strip().lower() or "unknown"
        owner_name = owner.get("name") or owner_email

        created_at = _parse_iso(note.get("created_at")) or event.received_at
        # Granola's GET /notes/{id} doesn't document an updated_at field.
        # Use created_at for both source-side timestamps; Phase 0 accepts
        # this gap (tracked in TODOS.md — edited notes are silently missed).
        updated_at = created_at

        transcript = note.get("transcript") or []
        if not isinstance(transcript, list):
            transcript = []

        body = _compose_body(summary, transcript)
        body_size = len(body.encode("utf-8"))

        # content_hash makes re-polling free: same note → same hash → bitemporal
        # writer no-ops via its content_hash check.
        content_hash = _sha256(f"{note_id}|{summary}|{_transcript_digest(transcript)}")

        doc_id = f"granola:meeting:{note_id}"
        source_url = f"https://notes.granola.ai/d/{note_id}"

        acl_principals = [
            ACLPrincipal(
                principal_type=PrincipalType.USER,
                principal_id=owner_email,
                name=owner_name,
                permission=Permission.WRITE,
            )
        ]
        acl_rows = [
            ACLSnapshotRow(
                source_system=SourceSystem.GRANOLA,
                principal_type=PrincipalType.USER,
                principal_id=owner_email,
                resource_type="granola.meeting",
                resource_id=note_id,
                permission=Permission.WRITE,
                valid_from=updated_at,
                metadata={"role": "owner"},
            )
        ]

        doc = Document(
            doc_id=doc_id,
            customer_id=event.customer_id,
            source_system=SourceSystem.GRANOLA,
            source_id=note_id,
            source_url=source_url,
            doc_class=DocClass.RAW_SOURCE,
            doc_type=DocType.GRANOLA_MEETING,
            content_type="text/markdown",
            content_hash=content_hash,
            title=title,
            body_preview=summary[:280] if summary else None,
            body_size_bytes=body_size,
            body_token_count=count_tokens(body),
            author_id=owner_email,
            created_at=created_at,
            updated_at=updated_at,
            valid_from=updated_at,
            ingested_at=datetime.now(UTC),
            acl=ACLSnapshot(principals=acl_principals, captured_at=event.received_at),
            metadata={
                "body": body,
                "owner_email": owner_email,
                "owner_name": owner_name,
                "transcript_segments": len(transcript),
                "has_transcript": bool(transcript),
            },
        )

        nodes: list[GraphNodeSpec] = [
            GraphNodeSpec(
                label=NodeLabel.DOCUMENT,
                canonical_id=doc_id,
                properties={"doc_type": doc.doc_type.value},
            ),
            GraphNodeSpec(
                label=NodeLabel.PERSON,
                canonical_id=owner_email,
                properties={
                    "source_system": SourceSystem.GRANOLA.value,
                    "name": owner_name,
                },
            ),
        ]

        edges: list[GraphEdgeSpec] = [
            GraphEdgeSpec(
                edge_type=EdgeType.AUTHORED,
                from_label=NodeLabel.PERSON,
                from_canonical_id=owner_email,
                to_label=NodeLabel.DOCUMENT,
                to_canonical_id=doc_id,
                valid_from=updated_at,
            )
        ]

        return NormalizationResult(
            documents=[doc],
            graph_nodes=nodes,
            graph_edges=edges,
            acl_snapshots=acl_rows,
        )

    # ---- 5. backfill ------------------------------------------------------

    async def backfill(
        self,
        customer_id: str,
        token: IntegrationToken,
        cursor: str | None = None,
    ) -> AsyncIterator[WebhookEvent]:
        """Stream Granola notes since `cursor.watermark`, fully hydrated.

        Cursor flow:
          - cursor=None        → full backfill (created_after omitted)
          - cursor=<json>      → incremental (created_after=watermark)

        Each yielded event carries `_cursor` in its raw_payload so the
        backfill_runner persists progress. The watermark in `_cursor`
        advances as we see notes with newer created_at, so the next
        backfill resumes from where this one ended.
        """
        state = _decode_cursor(cursor)
        watermark: str | None = state.get("watermark")
        page_cursor: str | None = None  # reset across ticks

        headers = {
            "Authorization": f"Bearer {token.access_token}",
            "User-Agent": "prbe-knowledge/0.1",
        }

        pages = 0
        while pages < _MAX_PAGES_PER_TICK:
            pages += 1
            params: dict[str, Any] = {"limit": _PAGE_SIZE}
            if watermark:
                params["created_after"] = watermark
            if page_cursor:
                params["cursor"] = page_cursor

            list_body = await self._granola_get(
                f"{_GRANOLA_API}/notes", params=params, headers=headers
            )
            if list_body is None:
                return  # transient error — let next tick resume

            results = list_body.get("notes") or list_body.get("data") or []
            if not isinstance(results, list):
                results = []

            for note_summary in results:
                if not isinstance(note_summary, dict):
                    continue
                note_id = note_summary.get("id")
                if not note_id:
                    continue

                # Hydrate: GET /notes/{id}?include=transcript per note.
                # The list response doesn't include summary or transcript,
                # so we always need the per-note fetch.
                note = await self._granola_get(
                    f"{_GRANOLA_API}/notes/{note_id}",
                    params={"include": "transcript"},
                    headers=headers,
                )
                if note is None:
                    # Skip on transient — next tick will retry the cursor.
                    continue

                created_at_str = note.get("created_at") or ""
                received_at = _parse_iso(created_at_str) or datetime.now(UTC)

                # Advance watermark to the latest created_at we've seen.
                # Strict > so we don't get stuck re-polling notes with
                # identical timestamps (Granola IDs are unique within a tick).
                if created_at_str and (watermark is None or created_at_str > watermark):
                    watermark = created_at_str

                next_state = {"watermark": watermark, "page_cursor": None}
                payload = {
                    "note": note,
                    "_cursor": json.dumps(next_state),
                }
                yield WebhookEvent(
                    customer_id=customer_id,
                    source_system=SourceSystem.GRANOLA,
                    source_event_id=str(note_id),
                    received_at=received_at,
                    payload_s3_key="",
                    raw_payload=payload,
                    headers={},
                )

            # Pagination: cursor in body → next page.
            has_more = bool(list_body.get("hasMore") or list_body.get("has_more"))
            next_cursor = list_body.get("cursor") or list_body.get("next_cursor")
            if not has_more or not next_cursor:
                return
            page_cursor = str(next_cursor)

    # ---- 7. workspace identification --------------------------------------
    #
    # Granola doesn't expose a workspace identifier we can route webhooks by.
    # No-op since there are no webhooks to route.

    async def identify_workspaces(
        self, token: IntegrationToken
    ):  # type: ignore[override]
        return []

    # ---- internal helpers -------------------------------------------------

    async def _granola_get(
        self,
        url: str,
        *,
        params: Mapping[str, Any],
        headers: Mapping[str, str],
    ) -> dict[str, Any] | None:
        """One Granola GET with rate-limited spacing and explicit error mapping.

        Returns parsed JSON dict on 200, None on transient failure (caller
        should treat as "stop this tick, resume next time"). Raises on
        permanent failures (auth, 4xx that aren't 429).
        """
        await asyncio.sleep(GRANOLA_REQUEST_INTERVAL_SECONDS)
        try:
            resp = await self.http.get(url, params=params, headers=headers)
        except (httpx.HTTPError, OSError) as exc:
            log.warning("granola.http_error", url=url, error=str(exc))
            return None

        if resp.status_code == 200:
            try:
                body = resp.json()
            except ValueError:
                log.warning("granola.invalid_json", url=url)
                return None
            return body if isinstance(body, dict) else None

        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After", "1")
            raise RateLimited(
                f"granola 429: retry-after={retry_after}",
                url=url,
            )

        if resp.status_code in {401, 403}:
            raise PermanentSourceError(
                f"granola auth failure: {resp.status_code}",
                url=url,
                status=resp.status_code,
            )

        if resp.status_code >= 500:
            raise TransientSourceError(
                f"granola 5xx: {resp.status_code}",
                url=url,
                status=resp.status_code,
            )

        # Other 4xx: treat as permanent (bad request, deleted resource, etc.).
        # Log + skip rather than wedging the whole backfill on one bad note.
        log.warning(
            "granola.unexpected_status",
            url=url,
            status=resp.status_code,
            body=resp.text[:200],
        )
        return None


# ---- module helpers --------------------------------------------------------


def _decode_cursor(cursor: str | None) -> dict[str, Any]:
    """Parse the JSON cursor we encode in `_cursor`. Bad/empty → empty dict."""
    if not cursor:
        return {}
    try:
        decoded = json.loads(cursor)
    except (json.JSONDecodeError, TypeError):
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _parse_iso(value: Any) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _transcript_digest(transcript: list[dict[str, Any]]) -> str:
    """Stable text digest of transcript so identical re-fetches don't bump version.

    macOS notes give speaker.source ('microphone'/'speaker') with no diarization.
    iOS notes add diarization_label ('Speaker A' etc.). Either way the
    text concatenation is the source of truth for content equality.
    """
    if not transcript:
        return ""
    parts: list[str] = []
    for turn in transcript:
        if not isinstance(turn, dict):
            continue
        text = (turn.get("text") or "").strip()
        if not text:
            continue
        speaker = turn.get("speaker") or {}
        label = (
            speaker.get("diarization_label")
            or speaker.get("source")
            or "unknown"
        ) if isinstance(speaker, dict) else "unknown"
        parts.append(f"{label}: {text}")
    return "\n".join(parts)


def _compose_body(summary: str, transcript: list[dict[str, Any]]) -> str:
    """Human-readable doc body. Summary first, transcript second under a header.

    Goes into Document.metadata['body'] — the chunker pulls from there. Keeping
    summary first means short queries against this doc match summary text
    before transcript noise.
    """
    if not transcript:
        return summary
    transcript_text = _transcript_digest(transcript)
    if not transcript_text:
        return summary
    if not summary:
        return f"## Transcript\n{transcript_text}"
    return f"{summary}\n\n## Transcript\n{transcript_text}"


__all__ = ["GranolaConnector"]
