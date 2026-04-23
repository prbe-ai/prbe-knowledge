"""Per-queue-row ingestion pipeline.

Generic across all connectors. Pulls the raw payload from R2, asks the
right Connector to parse/hydrate/normalize, then persists everything.

Called from the worker per queue row. Idempotent: running the same payload
twice produces the same database state.

Persistence is content-addressable + bitemporal:

- Documents use `valid_to` to close out the prior live version on edit.
  `_upsert_document` detects no-op (content_hash match, not a delete)
  and returns False without bumping version.
- Chunks live at identity `(doc_id, content_hash)`. On re-ingest we diff
  (live chunks ⟵ new chunks) — reused chunks just have `last_seen_version`
  bumped (no embedding call), new chunks are embedded + inserted,
  removed chunks get `valid_to = NOW()`.

That's the bit that keeps embedding cost proportional to actual content
change rather than the size of the document.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import asyncpg
import orjson

from services.ingestion.chunker import ChunkPiece, chunk_text
from services.ingestion.graph_writer import upsert_edges, upsert_nodes
from services.ingestion.handlers.base import Connector, ConnectorContext
from services.ingestion.handlers.registry import build_connector
from shared.constants import (
    CHUNKER_VERSION,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    NORMALIZER_VERSION,
    SourceSystem,
)
from shared.db import get_pool, with_tenant
from shared.embeddings import Embedder, get_embedder
from shared.encryption import decrypt_token
from shared.exceptions import (
    DuplicateEventIgnored,
    NormalizationError,
    TenantIsolationError,
    UnsupportedEventType,
)
from shared.logging import get_logger
from shared.models import (
    Document,
    IntegrationToken,
    NormalizationResult,
    WebhookEvent,
)
from shared.storage import ObjectStore, get_store

log = get_logger(__name__)


@dataclass(slots=True)
class NormalizeOutcome:
    doc_ids: list[str]
    # Live chunks after the diff (added + reused). Not a version-count.
    chunk_count: int
    failed_chunk_count: int
    added_chunk_count: int = 0
    reused_chunk_count: int = 0
    removed_chunk_count: int = 0
    stale_doc_ids: list[str] = field(default_factory=list)


class Normalizer:
    def __init__(
        self,
        ctx: ConnectorContext,
        store: ObjectStore | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self._ctx = ctx
        self._store = store or get_store()
        self._embedder = embedder or get_embedder()
        self._connectors: dict[SourceSystem, Connector] = {}

    def _connector(self, source: SourceSystem) -> Connector:
        if source not in self._connectors:
            self._connectors[source] = build_connector(source, self._ctx)
        return self._connectors[source]

    async def process_queue_row(
        self,
        queue_id: int,
        customer_id: str,
        source_system: SourceSystem,
        source_event_id: str,
        payload_s3_key: str,
    ) -> NormalizeOutcome:
        log.info(
            "normalizer.start",
            queue_id=queue_id,
            customer=customer_id,
            source=source_system.value,
            event_id=source_event_id,
        )

        raw_bytes = await self._store.get(
            self._store.bucket_for(customer_id), payload_s3_key
        )
        raw_payload = orjson.loads(raw_bytes)
        headers = raw_payload.get("_headers", {})
        payload = raw_payload.get("payload", raw_payload)

        connector = self._connector(source_system)
        parse_result = connector.parse_webhook_event(customer_id, headers, payload)
        if parse_result is None:
            raise UnsupportedEventType(
                "connector returned None from parse_webhook_event",
                source=source_system.value,
            )

        event = WebhookEvent(
            customer_id=customer_id,
            source_system=source_system,
            source_event_id=parse_result.source_event_id,
            received_at=parse_result.received_at,
            payload_s3_key=payload_s3_key,
            raw_payload=payload,
            headers=headers,
        )

        token = await self._load_token(customer_id, source_system)
        hydrated = await connector.fetch_supplementary(event, token)
        result = await connector.normalize(event, hydrated)

        if result.is_empty:
            if result.skipped_reason:
                raise DuplicateEventIgnored(result.skipped_reason)
            raise NormalizationError("connector produced no documents and no reason")

        return await self._persist(customer_id, result)

    # ---- persistence --------------------------------------------------------

    async def _persist(
        self, customer_id: str, result: NormalizationResult
    ) -> NormalizeOutcome:
        doc_ids: list[str] = []
        total_live_chunks = 0
        total_added = 0
        total_reused = 0
        total_removed = 0
        total_failed = 0

        async with with_tenant(customer_id) as conn:
            # Nodes first, then edges — edges look up node_ids.
            node_ids = await upsert_nodes(conn, customer_id, result.graph_nodes)
            await upsert_edges(conn, customer_id, result.graph_edges, node_ids)
            await _insert_acl_snapshots(conn, customer_id, result.acl_snapshots)

            for doc in result.documents:
                persisted = await _upsert_document(conn, doc)
                if not persisted:
                    continue  # content_hash match on live version → no-op
                doc_ids.append(doc.doc_id)

                sync_outcome = await self._sync_chunks(conn, doc)
                total_live_chunks += sync_outcome.live
                total_added += sync_outcome.added
                total_reused += sync_outcome.reused
                total_removed += sync_outcome.removed
                total_failed += sync_outcome.failed

        log.info(
            "normalizer.done",
            customer=customer_id,
            docs=len(doc_ids),
            live_chunks=total_live_chunks,
            added=total_added,
            reused=total_reused,
            removed=total_removed,
            failed_chunks=total_failed,
        )
        return NormalizeOutcome(
            doc_ids=doc_ids,
            chunk_count=total_live_chunks,
            failed_chunk_count=total_failed,
            added_chunk_count=total_added,
            reused_chunk_count=total_reused,
            removed_chunk_count=total_removed,
        )

    async def _sync_chunks(
        self, conn: asyncpg.Connection, doc: Document
    ) -> _ChunkSyncOutcome:
        """Diff live chunks against freshly-chunked content.

        The whole point: don't re-embed what hasn't changed.
        """
        # Deleted docs: no body → chunks is empty → every live chunk gets closed out.
        if doc.deleted_at is not None:
            new_pieces: list[ChunkPiece] = []
        else:
            new_pieces = chunk_text(_stringify_body(doc))

        new_hashes: list[str] = [_chunk_hash(p.content) for p in new_pieces]
        new_by_hash: dict[str, ChunkPiece] = {h: p for h, p in zip(new_hashes, new_pieces, strict=True)}

        live_rows = await conn.fetch(
            """
            SELECT content_hash, chunker_version, chunk_index
            FROM chunks
            WHERE doc_id = $1 AND valid_to IS NULL
            """,
            doc.doc_id,
        )

        # A chunk produced by an older chunker version is never "reused" — its
        # hash space isn't comparable to naive-v1's, so we mark it stale up front
        # and let the diff treat it as removed.
        live_hashes: set[str] = set()
        stale_hashes: set[str] = set()
        for row in live_rows:
            if row["chunker_version"] == CHUNKER_VERSION:
                live_hashes.add(row["content_hash"])
            else:
                stale_hashes.add(row["content_hash"])

        new_hashes_set = set(new_hashes)
        reused_hashes = live_hashes & new_hashes_set
        added_hashes = new_hashes_set - live_hashes
        removed_hashes = (live_hashes - new_hashes_set) | stale_hashes

        # 1) Reuse: just bump last_seen_version. No embed call.
        if reused_hashes:
            await conn.execute(
                """
                UPDATE chunks
                SET last_seen_version = $1
                WHERE doc_id = $2 AND content_hash = ANY($3::text[]) AND valid_to IS NULL
                """,
                doc.version,
                doc.doc_id,
                list(reused_hashes),
            )

        # 2) Added: embed + insert. Position inputs by their order in `added`
        # and pair back via the embedder's `chunk_index` which equals input index.
        added_pieces: list[ChunkPiece] = [new_by_hash[h] for h in new_hashes if h in added_hashes]

        failed = 0
        if added_pieces:
            embeds = await self._embedder.embed_many([p.content for p in added_pieces])
            for match in embeds.embedded:
                if match.chunk_index < 0 or match.chunk_index >= len(added_pieces):
                    continue
                piece = added_pieces[match.chunk_index]
                await _insert_chunk(conn, doc, piece, match.embedding)
            for fail in embeds.failed:
                # Map back to the originating piece to record chunk_index on failure.
                if 0 <= fail.chunk_index < len(added_pieces):
                    piece = added_pieces[fail.chunk_index]
                else:
                    piece = None
                await _insert_failed_chunk(conn, doc, fail, piece)
                failed += 1

        # 3) Removed: mark stale at NOW().
        if removed_hashes:
            await conn.execute(
                """
                UPDATE chunks
                SET valid_to = NOW()
                WHERE doc_id = $1 AND content_hash = ANY($2::text[]) AND valid_to IS NULL
                """,
                doc.doc_id,
                list(removed_hashes),
            )

        return _ChunkSyncOutcome(
            added=len(added_pieces) - failed,
            reused=len(reused_hashes),
            removed=len(removed_hashes),
            failed=failed,
            live=len(reused_hashes) + (len(added_pieces) - failed),
        )

    # ---- helpers ------------------------------------------------------------

    async def _load_token(
        self, customer_id: str, source_system: SourceSystem
    ) -> IntegrationToken | None:
        async with get_pool().acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT access_token_encrypted, refresh_token_encrypted,
                       expires_at, scope, webhook_secret
                FROM integration_tokens
                WHERE customer_id = $1 AND source_system = $2 AND status = 'active'
                """,
                customer_id,
                source_system.value,
            )
        if row is None:
            return None
        return IntegrationToken(
            customer_id=customer_id,
            source_system=source_system,
            access_token=decrypt_token(row["access_token_encrypted"]),
            refresh_token=(
                decrypt_token(row["refresh_token_encrypted"])
                if row["refresh_token_encrypted"]
                else None
            ),
            expires_at=row["expires_at"],
            scope=row["scope"],
            webhook_secret=row["webhook_secret"],
        )


@dataclass(slots=True)
class _ChunkSyncOutcome:
    added: int
    reused: int
    removed: int
    failed: int
    live: int


# ---- SQL helpers (module-level so tests can unit-test them) ---------------


async def _upsert_document(conn: asyncpg.Connection, doc: Document) -> bool:
    """Insert a new document version if content changed.

    Returns True if a new (doc_id, version) row was written (and the prior
    live version was closed out), False if this payload matches the current
    live version's content_hash and isn't a delete (i.e. no-op).

    Mutates `doc.version` to the newly-written version so callers can use it
    when writing chunks.
    """
    existing = await conn.fetchrow(
        """
        SELECT version, content_hash
        FROM documents
        WHERE doc_id = $1 AND customer_id = $2 AND valid_to IS NULL
        LIMIT 1
        """,
        doc.doc_id,
        doc.customer_id,
    )

    if existing and existing["content_hash"] == doc.content_hash and doc.deleted_at is None:
        # Same content, not a delete → idempotent no-op.
        return False

    if existing is not None:
        # Close out the prior live version in the same transaction.
        await conn.execute(
            """
            UPDATE documents
            SET valid_to = NOW()
            WHERE doc_id = $1 AND version = $2 AND valid_to IS NULL
            """,
            doc.doc_id,
            existing["version"],
        )
        next_version = existing["version"] + 1
        # Link the new version to the one it supersedes, if not already.
        if doc.supersedes_doc_id is None:
            doc.supersedes_doc_id = doc.doc_id
    else:
        # Take the MAX in case there are pre-existing non-live versions (e.g.
        # from a previously tombstoned doc that's being re-created).
        max_version = await conn.fetchval(
            """
            SELECT COALESCE(MAX(version), 0)
            FROM documents
            WHERE doc_id = $1 AND customer_id = $2
            """,
            doc.doc_id,
            doc.customer_id,
        )
        next_version = int(max_version or 0) + 1

    await conn.execute(
        """
        INSERT INTO documents (
            doc_id, version, customer_id,
            source_system, source_id, source_url,
            doc_class, doc_type, content_type, language,
            content_hash, title, body_preview, body_size_bytes, body_token_count,
            author_id,
            created_at, updated_at, valid_from, valid_to, deleted_at, ingested_at,
            parent_doc_id, supersedes_doc_id,
            acl, metadata, entities, attachments, doc_references,
            normalizer_version
        )
        VALUES (
            $1, $2, $3,
            $4, $5, $6,
            $7, $8, $9, $10,
            $11, $12, $13, $14, $15,
            $16,
            $17, $18, $19, $20, $21, $22,
            $23, $24,
            $25::jsonb, $26::jsonb, $27::jsonb, $28::jsonb, $29::jsonb,
            $30
        )
        ON CONFLICT (doc_id, version) DO NOTHING
        """,
        doc.doc_id,
        next_version,
        doc.customer_id,
        doc.source_system.value,
        doc.source_id,
        doc.source_url,
        doc.doc_class.value,
        doc.doc_type.value,
        doc.content_type,
        doc.language,
        doc.content_hash,
        doc.title,
        doc.body_preview,
        doc.body_size_bytes,
        doc.body_token_count,
        doc.author_id,
        doc.created_at,
        doc.updated_at,
        doc.valid_from,
        doc.valid_to,
        doc.deleted_at,
        doc.ingested_at,
        doc.parent_doc_id,
        doc.supersedes_doc_id,
        _json(doc.acl.model_dump()),
        _json(doc.metadata),
        _json([e.model_dump() for e in doc.entities]),
        _json([a.model_dump() for a in doc.attachments]),
        _json([r.model_dump() for r in doc.doc_references]),
        NORMALIZER_VERSION,
    )
    doc.version = next_version
    return True


async def _insert_chunk(
    conn: asyncpg.Connection,
    doc: Document,
    piece: ChunkPiece,
    embedding: list[float],
) -> None:
    content_hash = _chunk_hash(piece.content)
    chunk_id = f"{doc.doc_id}:c_{content_hash[:16]}"
    await conn.execute(
        """
        INSERT INTO chunks (
            chunk_id, doc_id, customer_id,
            chunk_index, content, content_hash, token_count,
            embedding, embedding_model, embedding_dim, chunker_version,
            first_seen_version, last_seen_version
        )
        VALUES (
            $1, $2, $3,
            $4, $5, $6, $7,
            $8::halfvec, $9, $10, $11,
            $12, $12
        )
        ON CONFLICT (doc_id, content_hash) DO UPDATE
            SET last_seen_version = EXCLUDED.last_seen_version,
                valid_to = NULL
        """,
        chunk_id,
        doc.doc_id,
        doc.customer_id,
        piece.chunk_index,
        piece.content,
        content_hash,
        piece.token_count,
        _pg_vector(embedding),
        EMBEDDING_MODEL,
        EMBEDDING_DIM,
        CHUNKER_VERSION,
        doc.version,
    )


async def _insert_failed_chunk(
    conn: asyncpg.Connection, doc: Document, failed: Any, piece: ChunkPiece | None = None
) -> None:
    chunk_index = piece.chunk_index if piece is not None else failed.chunk_index
    await conn.execute(
        """
        INSERT INTO failed_chunks
            (customer_id, doc_id, doc_version, chunk_index, content_preview, error)
        VALUES ($1, $2, $3, $4, $5, $6)
        """,
        doc.customer_id,
        doc.doc_id,
        doc.version,
        chunk_index,
        failed.content_preview,
        failed.error,
    )


async def _insert_acl_snapshots(
    conn: asyncpg.Connection,
    customer_id: str,
    rows: list,
) -> None:
    if not rows:
        return
    for r in rows:
        await conn.execute(
            """
            INSERT INTO acl_snapshots (
                customer_id, source_system, principal_type, principal_id,
                resource_type, resource_id, permission, valid_from, valid_to, metadata
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb)
            """,
            customer_id,
            r.source_system.value,
            r.principal_type.value,
            r.principal_id,
            r.resource_type,
            r.resource_id,
            r.permission.value,
            r.valid_from,
            r.valid_to,
            _json(r.metadata),
        )


# ---- formatting helpers --------------------------------------------------


def _json(obj: Any) -> str:
    return orjson.dumps(obj, default=_json_default).decode("utf-8")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, datetime):
        if obj.tzinfo is None:
            obj = obj.replace(tzinfo=UTC)
        return obj.isoformat()
    if hasattr(obj, "value"):  # StrEnum
        return obj.value
    raise TypeError(f"not serializable: {type(obj)}")


def _pg_vector(vec: list[float]) -> str:
    """Format a float list into pgvector's textual literal '[1.0,2.0,...]'.

    halfvec accepts the same text input — the cast in the INSERT handles it.
    """
    return "[" + ",".join(f"{x:.7f}" for x in vec) + "]"


def _chunk_hash(content: str) -> str:
    """Stable content-hash used as the identity of a chunk within a doc."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _stringify_body(doc: Document) -> str:
    """Pull the embeddable body text off a document.

    Phase 0 convention: connectors put the full normalized body in
    `metadata["body"]`, since the documents table doesn't carry a body column
    (chunks is the source of truth for full text). body_preview is short
    enough that we don't want to chunk off that alone.
    """
    body = doc.metadata.get("body")
    if not body:
        # Fall back to title + preview so we still get a chunk for short entities.
        parts = [p for p in (doc.title, doc.body_preview) if p]
        body = "\n\n".join(parts) if parts else ""
    return str(body)


def ensure_tenant_bound(customer_id: str) -> None:
    if not customer_id:
        raise TenantIsolationError("customer_id required for normalize")
