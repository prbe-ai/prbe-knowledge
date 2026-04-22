"""Per-queue-row ingestion pipeline.

Generic across all connectors. Pulls the raw payload from R2, asks the
right Connector to parse/hydrate/normalize, then persists everything.

Called from the worker per queue row. Idempotent: running the same payload
twice produces the same database state (content_hash dedupes versions,
(doc_id, version) dedupes chunks).
"""

from __future__ import annotations

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
    chunk_count: int
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
        total_chunks = 0
        total_failed = 0

        async with with_tenant(customer_id) as conn:
            # Nodes first, then edges — edges look up node_ids.
            node_ids = await upsert_nodes(conn, customer_id, result.graph_nodes)
            await upsert_edges(conn, customer_id, result.graph_edges, node_ids)
            await _insert_acl_snapshots(conn, customer_id, result.acl_snapshots)

            for doc in result.documents:
                persisted = await _upsert_document(conn, doc)
                if not persisted:
                    continue  # content_hash match → no new version
                doc_ids.append(doc.doc_id)

                chunks = chunk_text(_stringify_body(doc))
                if not chunks:
                    continue

                embeds = await self._embedder.embed_many([c.content for c in chunks])

                for piece in chunks:
                    match = next(
                        (e for e in embeds.embedded if e.chunk_index == piece.chunk_index),
                        None,
                    )
                    if match is None:
                        continue
                    await _insert_chunk(conn, doc, piece, match.embedding)
                    total_chunks += 1

                for failed in embeds.failed:
                    await _insert_failed_chunk(conn, doc, failed)
                    total_failed += 1

        log.info(
            "normalizer.done",
            customer=customer_id,
            docs=len(doc_ids),
            chunks=total_chunks,
            failed_chunks=total_failed,
        )
        return NormalizeOutcome(
            doc_ids=doc_ids, chunk_count=total_chunks, failed_chunk_count=total_failed
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


# ---- SQL helpers (module-level so tests can unit-test them) ---------------


async def _upsert_document(conn: asyncpg.Connection, doc: Document) -> bool:
    """Insert a new document version if content_hash changed.

    Returns True if a new (doc_id, version) was written, False if this
    payload matches the latest version's content_hash (no-op).
    """
    existing = await conn.fetchrow(
        """
        SELECT version, content_hash FROM documents
        WHERE doc_id = $1 AND customer_id = $2
        ORDER BY version DESC
        LIMIT 1
        """,
        doc.doc_id,
        doc.customer_id,
    )

    if existing and existing["content_hash"] == doc.content_hash:
        return False

    next_version = (existing["version"] + 1) if existing else 1

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
    # Update the doc in place with the version we wrote so callers can use it.
    doc.version = next_version
    return True


async def _insert_chunk(
    conn: asyncpg.Connection,
    doc: Document,
    piece: Any,
    embedding: list[float],
) -> None:
    chunk_id = f"{doc.doc_id}:v{doc.version}:c{piece.chunk_index}"
    await conn.execute(
        """
        INSERT INTO chunks (
            chunk_id, doc_id, doc_version, customer_id,
            chunk_index, content, token_count,
            embedding, embedding_model, embedding_dim, chunker_version
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8::halfvec, $9, $10, $11)
        ON CONFLICT (chunk_id) DO NOTHING
        """,
        chunk_id,
        doc.doc_id,
        doc.version,
        doc.customer_id,
        piece.chunk_index,
        piece.content,
        piece.token_count,
        _pg_vector(embedding),
        EMBEDDING_MODEL,
        EMBEDDING_DIM,
        CHUNKER_VERSION,
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
