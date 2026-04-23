"""Per-queue-row ingestion pipeline.

Generic across all connectors. Pulls the raw payload from R2, asks the
right Connector to parse/hydrate/normalize, then persists everything.

Persistence semantics:

- Docs are versioned. A new version bump closes out the prior version
  (sets `valid_to = NOW()`, `supersedes_doc_id = $doc_id`).
- Chunks are content-addressable. Identity is (doc_id, content_hash).
  On re-ingest of an edited doc, chunks are reconciled as a three-way diff:
    • reused (content still present) → bump `last_seen_version`, no embed call
    • added  (new content)           → insert row, embed
    • removed (no longer present)    → mark `valid_to = NOW()` (stale)
- Tombstone path: when a doc arrives with `deleted_at` set, skip chunking;
  the diff naturally marks every live chunk stale (empty-new-set).

Idempotency:
- Same `content_hash` on the doc → no-op (no version bump).
- Same chunk `content_hash` across versions → existing chunk row is reused.
- The UNIQUE constraint on `ingestion_queue` dedupes webhook redelivery.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import asyncpg
import orjson

from services.ingestion.chunker import chunk_text
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
    chunk_count: int          # chunks that are live (added + reused)
    added_chunk_count: int    # newly-embedded this run
    reused_chunk_count: int   # matched an existing row, no embed call
    removed_chunk_count: int  # marked stale this run
    failed_chunk_count: int


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
        total_live = 0
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
                    continue  # content_hash match → no new version, no chunk work
                doc_ids.append(doc.doc_id)

                # Deleted doc → no chunks in the new set. The diff below will
                # mark every previously-live chunk as removed.
                new_chunks = [] if doc.deleted_at else chunk_text(_stringify_body(doc))

                diff = await self._sync_chunks(conn, doc, new_chunks)
                total_live += diff.live_count
                total_added += diff.added_count
                total_reused += diff.reused_count
                total_removed += diff.removed_count
                total_failed += diff.failed_count

        log.info(
            "normalizer.done",
            customer=customer_id,
            docs=len(doc_ids),
            chunks_live=total_live,
            chunks_added=total_added,
            chunks_reused=total_reused,
            chunks_removed=total_removed,
            chunks_failed=total_failed,
        )
        return NormalizeOutcome(
            doc_ids=doc_ids,
            chunk_count=total_live,
            added_chunk_count=total_added,
            reused_chunk_count=total_reused,
            removed_chunk_count=total_removed,
            failed_chunk_count=total_failed,
        )

    async def _sync_chunks(
        self,
        conn: asyncpg.Connection,
        doc: Document,
        new_chunks: list,
    ) -> _ChunkDiffOutcome:
        """Three-way diff on chunks for a doc version bump.

        Matches new chunks to live rows by (doc_id, content_hash).
        Only unmatched chunks get embedded; stale rows are marked valid_to.
        """
        new_by_hash: dict[str, Any] = {}
        for piece in new_chunks:
            h = _chunk_hash(piece.content)
            # First occurrence wins in case the chunker produced duplicates.
            new_by_hash.setdefault(h, piece)

        # Pull currently-live chunks for this doc. If the chunker version
        # changed since the last write, we can't trust the hashes — force
        # a full re-embed by treating nothing as reused.
        live_rows = await conn.fetch(
            """
            SELECT chunk_id, content_hash, chunker_version
            FROM chunks
            WHERE doc_id = $1 AND customer_id = $2 AND valid_to IS NULL
            """,
            doc.doc_id,
            doc.customer_id,
        )
        live_by_hash: dict[str, asyncpg.Record] = {}
        stale_mismatched_chunker: list[str] = []
        for r in live_rows:
            if r["chunker_version"] != CHUNKER_VERSION:
                stale_mismatched_chunker.append(r["chunk_id"])
                continue
            live_by_hash[r["content_hash"]] = r

        # Mark chunks that were produced by a different chunker version as stale.
        # They can't be reliably diffed — the new chunker may have sliced content
        # differently so hashes aren't comparable across versions.
        if stale_mismatched_chunker:
            await conn.execute(
                "UPDATE chunks SET valid_to = NOW() WHERE chunk_id = ANY($1::text[])",
                stale_mismatched_chunker,
            )

        new_hashes = set(new_by_hash)
        live_hashes = set(live_by_hash)
        reused_hashes = new_hashes & live_hashes
        added_hashes = new_hashes - live_hashes
        removed_hashes = live_hashes - new_hashes

        # Bump last_seen_version on reused chunks — cheap, no embed call.
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

        # Mark removed chunks stale.
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

        # Embed and insert only the added chunks. The embedder indexes by
        # position in the input list (not by the chunker's chunk_index), so
        # we pair added_pieces with embed results positionally.
        added_pieces = [new_by_hash[h] for h in added_hashes]
        added_count = 0
        failed_count = 0

        if added_pieces:
            embeds = await self._embedder.embed_many(
                [p.content for p in added_pieces]
            )
            embed_by_input_idx = {e.chunk_index: e for e in embeds.embedded}
            failed_input_idxs = {f.chunk_index for f in embeds.failed}
            for input_idx, piece in enumerate(added_pieces):
                if input_idx in failed_input_idxs:
                    continue
                match = embed_by_input_idx.get(input_idx)
                if match is None:
                    continue
                await _insert_chunk(conn, doc, piece, match.embedding)
                added_count += 1
            for failed in embeds.failed:
                # failed.chunk_index is the input-list index into added_pieces;
                # translate back to the chunker's chunk_index for the audit row.
                if 0 <= failed.chunk_index < len(added_pieces):
                    orig = added_pieces[failed.chunk_index]
                    audit = type(failed)(
                        chunk_index=orig.chunk_index,
                        content_preview=failed.content_preview,
                        error=failed.error,
                    )
                    await _insert_failed_chunk(conn, doc, audit)
                else:
                    await _insert_failed_chunk(conn, doc, failed)
                failed_count += 1

        live_count = len(reused_hashes) + added_count
        return _ChunkDiffOutcome(
            live_count=live_count,
            added_count=added_count,
            reused_count=len(reused_hashes),
            removed_count=len(removed_hashes) + len(stale_mismatched_chunker),
            failed_count=failed_count,
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
class _ChunkDiffOutcome:
    live_count: int
    added_count: int
    reused_count: int
    removed_count: int
    failed_count: int


# ---- SQL helpers (module-level so tests can unit-test them) ---------------


async def _upsert_document(conn: asyncpg.Connection, doc: Document) -> bool:
    """Insert a new document version if content_hash changed.

    Side effects when a new version is written:
    - Prior live version (if any) has `valid_to = NOW()` and
      `supersedes_doc_id = doc.doc_id` set in the same transaction.
    - `doc.version` is mutated in place to the version number actually written.

    Returns True iff a new (doc_id, version) was written.
    """
    existing = await conn.fetchrow(
        """
        SELECT version, content_hash FROM documents
        WHERE doc_id = $1 AND customer_id = $2 AND valid_to IS NULL
        ORDER BY version DESC
        LIMIT 1
        """,
        doc.doc_id,
        doc.customer_id,
    )

    if existing and existing["content_hash"] == doc.content_hash and not doc.deleted_at:
        return False

    # There may be a prior version row with valid_to already set (from a
    # chain-of-edits history); we need the latest version number to bump.
    last_version = await conn.fetchval(
        """
        SELECT MAX(version) FROM documents
        WHERE doc_id = $1 AND customer_id = $2
        """,
        doc.doc_id,
        doc.customer_id,
    )
    next_version = (last_version or 0) + 1

    # Close out the prior live version in the same transaction. We only touch
    # valid_to here — supersedes_doc_id is reserved for chain-breaking scenarios
    # (source deletes item X, recreates with new source_id) where the connector
    # sets it explicitly. Within-doc_id version bumps don't need it because the
    # (doc_id, version) primary key already links the chain.
    if existing:
        await conn.execute(
            """
            UPDATE documents
            SET valid_to = NOW()
            WHERE doc_id = $1 AND version = $2 AND valid_to IS NULL
            """,
            doc.doc_id,
            existing["version"],
        )

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
    piece: Any,
    embedding: list[float],
) -> None:
    """Insert a chunk row. Identity is (doc_id, content_hash).

    Under normal operation the UNIQUE (doc_id, content_hash) shouldn't fire —
    the diff algorithm only routes to this function for chunks that aren't
    in the live set. ON CONFLICT handles the race where two workers ingest
    overlapping doc versions concurrently.
    """
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
    conn: asyncpg.Connection, doc: Document, failed: Any
) -> None:
    await conn.execute(
        """
        INSERT INTO failed_chunks
            (customer_id, doc_id, doc_version, chunk_index, content_preview, error)
        VALUES ($1, $2, $3, $4, $5, $6)
        """,
        doc.customer_id,
        doc.doc_id,
        doc.version,
        failed.chunk_index,
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


def _chunk_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


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
