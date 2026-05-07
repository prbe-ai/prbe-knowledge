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
import re
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
from shared.customer_prefs import is_wiki_generation_enabled
from shared.db import get_pool, with_tenant
from shared.embeddings import Embedder, get_embedder
from shared.encryption import decrypt_token
from shared.exceptions import (
    DuplicateEventIgnored,
    NormalizationError,
    PrbeError,
    TenantIsolationError,
    UnsupportedEventType,
)
from shared.logging import get_logger
from shared.models import (
    METADATA_CHUNK_INDEX,
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
    quarantined_doc_ids: list[str] = field(default_factory=list)


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
        payload_s3_keys: list[str],
    ) -> NormalizeOutcome:
        """Process one queue row.

        For non-CC connectors, `payload_s3_keys` is single-element and the
        flow is unchanged: read the single payload, parse, fetch, normalize.

        For claude_code (after migration 0026), `payload_s3_keys` carries
        every batch's R2 key coalesced into the row. We read the FIRST
        key here for the parse_webhook_event metadata (any batch yields
        the same session_id, so any key is fine for that purpose). The
        connector's `fetch_supplementary` reads ALL keys from
        `event.payload_s3_keys` and merges events from every batch.
        """
        log.info(
            "normalizer.start",
            queue_id=queue_id,
            customer=customer_id,
            source=source_system.value,
            event_id=source_event_id,
            payload_count=len(payload_s3_keys),
        )

        if not payload_s3_keys:
            raise UnsupportedEventType(
                "queue row has no payload_s3_keys",
                source=source_system.value,
            )

        # Read the first (oldest) payload for parse_webhook_event metadata.
        # For non-CC connectors this is THE payload; for CC the connector
        # reads the full set in fetch_supplementary.
        first_key = payload_s3_keys[0]
        raw_bytes = await self._store.get(self._store.bucket_for(customer_id), first_key)
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
            payload_s3_key=first_key,  # legacy back-compat
            payload_s3_keys=payload_s3_keys,
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

        return await self._persist(customer_id, source_system, result)

    # ---- persistence --------------------------------------------------------

    async def _persist(
        self,
        customer_id: str,
        source_system: SourceSystem,
        result: NormalizationResult,
    ) -> NormalizeOutcome:
        doc_ids: list[str] = []
        quarantined: list[str] = []
        total_live_chunks = 0
        total_added = 0
        total_reused = 0
        total_removed = 0
        total_failed = 0

        # ---- Flatten standard + pre-chunked Documents into a single
        # ordered list of (doc, pre_chunked, pre_chunked_metadata) tuples.
        # Standard Documents go through chunk_text(doc.body); pre-chunked
        # Documents bypass the chunker — the connector owns chunking when
        # symbol/structural granularity matters (today only code-graph).
        all_docs: list[tuple[Document, list[ChunkPiece] | None, ChunkPiece | None]] = [
            (doc, None, None) for doc in result.documents
        ]
        for prechunked in result.documents_with_chunks:
            all_docs.append(
                (prechunked.document, prechunked.chunks, prechunked.metadata_chunk)
            )

        # ---- Storage guard: handlers must NOT stuff body into metadata.
        # body lives on the transient Document.body field; persisting it into
        # metadata jsonb doubles storage on every doc (the original
        # documents.metadata["body"] duplication bug). See migration 0035.
        for doc, pre_chunked, _ in all_docs:
            if "body" in doc.metadata:
                raise NormalizationError(
                    f"connector {source_system.value} set metadata['body'] on "
                    f"doc {doc.doc_id} — body must be passed via Document.body "
                    "(transient field). See services/ingestion/normalizer.py "
                    "_stringify_body."
                )
            # Pre-chunked Documents must NOT carry a body — the chunks list
            # is authoritative. Body + pre_chunked together is ambiguous and
            # likely a connector bug.
            if pre_chunked is not None and doc.body is not None:
                raise NormalizationError(
                    f"connector {source_system.value} provided both Document.body "
                    f"and pre-chunked pieces for doc {doc.doc_id}; pre-chunked "
                    "Documents must set body=None"
                )

        # ---- Phase A: pre-compute chunk plans WITHOUT holding a write txn.
        #
        # Each plan does a tiny read txn for the live-chunks SELECT under
        # tenant RLS, closes it, then calls `embed_many` outside any txn.
        # Doing this inside Phase B's transaction (the historical bug) held
        # row locks across the 60-120s OpenAI round trip, which caused
        # concurrent workers' `graph_nodes` upserts to hit the 30s
        # statement_timeout and DLQ. See incident 2026-04-29.
        #
        # Embedding-cost dedup against same-session work is enforced upstream
        # in `Worker._claim_one` (NOT EXISTS on session_key) — without that,
        # two batches of the same session would compute overlapping chunk
        # sets here and pay OpenAI twice for identical content_hashes.
        plans: list[_ChunkPlan] = []
        for doc, pre_chunked, pre_chunked_metadata in all_docs:
            plans.append(
                await self._plan_chunks(
                    customer_id, doc, pre_chunked, pre_chunked_metadata
                )
            )

        # ---- Phase B: ONE short write transaction. No external I/O between
        # BEGIN and COMMIT, so every lock held here is millisecond-scale.
        async with with_tenant(customer_id) as conn:
            # Nodes first, then edges — edges look up node_ids.
            node_ids = await upsert_nodes(
                conn, customer_id, result.graph_nodes, source_system.value
            )
            await upsert_edges(conn, customer_id, result.graph_edges, node_ids, source_system.value)
            await _insert_acl_snapshots(conn, customer_id, result.acl_snapshots)
            await _upsert_code_repo_state(conn, customer_id, result.code_repo_state_updates)

            for idx, ((doc, _, _), plan) in enumerate(
                zip(all_docs, plans, strict=True)
            ):
                # Per-doc SAVEPOINT: a deterministic-permanent error on one doc
                # (e.g. a malformed body) must not roll back already-good
                # siblings, otherwise the queue row stays poisoned forever.
                # Transient errors still bubble out so the worker retries the
                # whole row.
                sp = f"doc_{idx}"
                await conn.execute(f"SAVEPOINT {sp}")
                try:
                    persisted = await _upsert_document(conn, doc)
                    if persisted:
                        doc_ids.append(doc.doc_id)
                        sync_outcome = await _apply_chunk_plan(conn, doc, plan)
                        total_live_chunks += sync_outcome.live
                        total_added += sync_outcome.added
                        total_reused += sync_outcome.reused
                        total_removed += sync_outcome.removed
                        total_failed += sync_outcome.failed
                except PrbeError as exc:
                    if getattr(exc, "transient", False):
                        raise
                    await conn.execute(f"ROLLBACK TO SAVEPOINT {sp}")
                    quarantined.append(doc.doc_id)
                    log.warning(
                        "normalizer.doc_quarantined",
                        customer=customer_id,
                        doc_id=doc.doc_id,
                        error=str(exc),
                        error_class=type(exc).__name__,
                    )
                await conn.execute(f"RELEASE SAVEPOINT {sp}")

        # ---- Wiki-synthesis enqueue (no NOTIFY) -------------------------
        # After Phase B's commit, append one row per persisted doc into
        # wiki_synthesis_queue. Queue rows accumulate at status='pending'
        # silently during the day — synthesis is nightly-only, driven by
        # the wiki-cron fly app firing pg_notify('wiki_synthesize_pending')
        # at 02:00 UTC, plus a manual trigger endpoint for the dashboard.
        #
        # Pre-redesign this path also fired pg_notify for each persisted
        # doc, causing continuous daytime synthesis. Removed so the wiki
        # behaves like the slow-moving knowledge base it's supposed to be.
        #
        # Skip when the source IS the wiki (cron's own COMPILED_WIKI
        # writes must not feed back into its own queue). Skip CODE_GRAPH
        # too: code.symbol docs are deterministic AST extractions whose
        # body is a function signature + docstring — wiki synthesis would
        # burn LLM tokens trying to extract Decisions/Runbooks from them
        # and produce nothing. Also skip when the tenant has not opted
        # into wiki generation.
        if (
            source_system != SourceSystem.WIKI
            and source_system != SourceSystem.CODE_GRAPH
            and doc_ids
            and await is_wiki_generation_enabled(customer_id)
        ):
            try:
                # Use the in-memory documents that were just persisted —
                # they carry the source-side metadata extract_source_ts
                # needs (Slack ts, GitHub created_at, etc.). The DB only
                # stores body_preview/title/etc., not the raw source-side
                # event timestamp surface.
                persisted_docs = [
                    doc for doc in result.documents if doc.doc_id in set(doc_ids)
                ]
                await self._enqueue_wiki_synthesis(customer_id, persisted_docs)
            except Exception as exc:
                # Boundary swallow: ingestion already committed, queue
                # enqueue is best-effort. The nightly trigger picks up
                # anything we missed.
                log.warning(
                    "wiki.synthesis.enqueue_failed",
                    customer=customer_id,
                    doc_count=len(doc_ids),
                    error=str(exc),
                    error_class=type(exc).__name__,
                )

        # ---- Inferred-edges enqueue (best-effort) ---------------------------
        # After Phase B commits, append one row per persisted doc into
        # inferred_edges_queue. The side-queue worker drains this table
        # asynchronously: builds a bundle -> one LLM call -> upserts
        # INFERRED/AMBIGUOUS edges. Insert failure MUST NOT block main
        # ingestion -- wrap in try/except, log on failure, continue.
        #
        # Skip wiki source (no cross-source inference on compiled pages)
        # and skip if no docs were persisted this cycle.
        if source_system != SourceSystem.WIKI and doc_ids:
            try:
                await self._enqueue_inferred_edges(customer_id, doc_ids)
            except Exception as exc:
                log.warning(
                    "inferred_edges.enqueue_failed",
                    customer=customer_id,
                    doc_count=len(doc_ids),
                    error=str(exc),
                    error_class=type(exc).__name__,
                )

        log.info(
            "normalizer.done",
            customer=customer_id,
            docs=len(doc_ids),
            quarantined=len(quarantined),
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
            quarantined_doc_ids=quarantined,
        )

    async def _enqueue_wiki_synthesis(
        self,
        customer_id: str,
        docs: list[Document],
    ) -> None:
        """Append a wiki_synthesis_queue row per persisted doc.

        Idempotent on `(customer_id, doc_id, doc_version)` — a redelivered
        webhook that re-persists the same content (same version) won't
        double-enqueue. The doc_version comes from the DB rather than the
        in-memory Document because `_upsert_document` mutates `doc.version`
        in place; we re-read to be defensive against future callers that
        skip that mutation.

        Populates `source_ts` from per-source metadata via
        `extract_source_ts(doc)` so the wiki agent can read the day in
        time order. Falls back to documents.created_at when the connector
        didn't surface a parseable source-side timestamp.

        Does NOT fire pg_notify — synthesis is nightly-batch via the
        wiki-cron fly app, not realtime. See the comment block in
        `_persist` for the why.
        """
        from services.synthesis.source_ts import extract_source_ts

        if not docs:
            return
        doc_ids = [doc.doc_id for doc in docs]
        # Map doc_id -> source_ts for the parameterized join below.
        source_ts_by_doc_id: dict[str, datetime] = {
            doc.doc_id: extract_source_ts(doc) for doc in docs
        }
        # asyncpg doesn't accept Python dicts as parameters; pass two
        # parallel arrays and join by index.
        ts_doc_ids = list(source_ts_by_doc_id.keys())
        ts_values = [source_ts_by_doc_id[d] for d in ts_doc_ids]
        async with with_tenant(customer_id) as conn:
            await conn.execute(
                """
                INSERT INTO wiki_synthesis_queue
                    (customer_id, doc_id, doc_version, source_system,
                     doc_type, status, enqueued_at, source_ts)
                SELECT d.customer_id, d.doc_id, d.version, d.source_system,
                       d.doc_type, 'pending', NOW(), ts.source_ts
                FROM documents d
                JOIN unnest($2::text[], $3::timestamptz[])
                     AS ts(doc_id, source_ts)
                  ON ts.doc_id = d.doc_id
                WHERE d.customer_id = $1
                  AND d.doc_id = ANY($2::text[])
                  AND d.valid_to IS NULL
                ON CONFLICT (customer_id, doc_id, doc_version) DO NOTHING
                """,
                customer_id,
                ts_doc_ids,
                ts_values,
            )
        _ = doc_ids  # retained for potential future logging

    async def _enqueue_inferred_edges(
        self,
        customer_id: str,
        doc_ids: list[str],
    ) -> None:
        """Insert one inferred_edges_queue row per persisted doc_id.

        Uses ON CONFLICT DO NOTHING so re-delivers of the same doc are
        deduplicated at the (customer_id, anchor_doc_id, extractor_id) level.
        The queue has no UNIQUE constraint on those three columns by default
        (the design intentionally allows re-extraction on prompt version bumps)
        -- idempotence here is soft: we just avoid flooding the queue with
        identical rows from the same ingest run.

        CRITICAL: any exception from this method is caught by the caller
        (_persist) which logs it and continues. This must never block ingestion.
        """
        from services.ingestion.inferred_edges.prompts.v1 import PROMPT_VERSION

        if not doc_ids:
            return

        async with with_tenant(customer_id) as conn:
            await conn.executemany(
                """
                INSERT INTO inferred_edges_queue
                    (customer_id, anchor_doc_id, extractor_id)
                VALUES ($1, $2, $3)
                ON CONFLICT DO NOTHING
                """,
                [(customer_id, doc_id, PROMPT_VERSION) for doc_id in doc_ids],
            )

    async def _plan_chunks(
        self,
        customer_id: str,
        doc: Document,
        pre_chunked: list[ChunkPiece] | None = None,
        pre_chunked_metadata: ChunkPiece | None = None,
    ) -> _ChunkPlan:
        """Phase A: build a `_ChunkPlan` for one document — without holding
        a write transaction across the embed_many call.

        Reads live chunks under a short read txn (RLS needs the tenant GUC),
        closes the txn, then calls the embedder for added pieces outside any
        txn. The returned plan is applied later by `_apply_chunk_plan` inside
        the shared write txn.

        The whole point: don't re-embed what hasn't changed, AND don't hold
        graph_nodes/chunks row locks during the long OpenAI round trip.

        `pre_chunked` is the connector-controlled chunking path (today only
        code-graph file Documents). When provided, `chunk_text(doc.body)` is
        bypassed and these pieces are treated as authoritative; downstream
        diff/reuse semantics are unchanged. `pre_chunked_metadata` plays the
        same role as `_metadata_piece(doc)` for the metadata chunk slot.
        Both default to the standard chunker path for backwards compat.
        """
        # Deleted docs: no body → chunks is empty → every live chunk gets closed out.
        # The metadata chunk also disappears for deleted docs (joins the removed
        # set just like content chunks).
        if doc.deleted_at is not None:
            new_pieces: list[ChunkPiece] = []
            metadata_piece: ChunkPiece | None = None
        elif pre_chunked is not None:
            new_pieces = pre_chunked
            metadata_piece = pre_chunked_metadata
        else:
            new_pieces = chunk_text(_stringify_body(doc))
            metadata_piece = _metadata_piece(doc)

        new_hashes: list[str] = [_chunk_hash(p.content) for p in new_pieces]
        new_by_hash: dict[str, ChunkPiece] = {
            h: p for h, p in zip(new_hashes, new_pieces, strict=True)
        }

        # The metadata chunk participates in the same hash-based reuse machinery
        # as content chunks: same content_hash across versions → just bump
        # last_seen_version, no re-embed. Its `kind='metadata'` is what
        # distinguishes it at insert + read time. Tracked separately so we can
        # filter live_rows by kind to avoid colliding with content hashes.
        metadata_hash: str | None = _chunk_hash(metadata_piece.content) if metadata_piece else None

        # Read-only txn for the live-chunks lookup. RLS on `chunks` requires
        # the tenant GUC, which `with_tenant` sets at txn start. Closing this
        # txn before calling the embedder is the whole point of Phase A.
        async with with_tenant(customer_id) as conn:
            live_rows = await conn.fetch(
                """
                SELECT content_hash, chunker_version, chunk_index, kind
                FROM chunks
                WHERE customer_id = $1 AND doc_id = $2 AND valid_to IS NULL
                """,
                doc.customer_id,
                doc.doc_id,
            )

        # Diff content chunks and the (singular) metadata chunk against
        # separate live-row partitions. Filtering by kind avoids the case
        # where a content chunk and the metadata chunk happen to coincide
        # by content_hash (highly unlikely under sha256 but cheap to defend).
        live_content_hashes: set[str] = set()
        live_metadata_hashes: set[str] = set()
        stale_hashes: set[str] = set()
        for row in live_rows:
            if row["chunker_version"] != CHUNKER_VERSION:
                stale_hashes.add(row["content_hash"])
                continue
            if row["kind"] == "metadata":
                live_metadata_hashes.add(row["content_hash"])
            else:
                live_content_hashes.add(row["content_hash"])

        new_hashes_set = set(new_hashes)
        reused_hashes = live_content_hashes & new_hashes_set
        added_hashes = new_hashes_set - live_content_hashes
        removed_hashes = (live_content_hashes - new_hashes_set) | stale_hashes

        # Metadata-chunk diff: separate set, separate flow.
        metadata_reuse = metadata_hash is not None and metadata_hash in live_metadata_hashes
        metadata_add = metadata_piece is not None and metadata_hash not in live_metadata_hashes
        # Any live metadata-kind hash that doesn't match the new one (e.g.
        # title changed across doc versions) is stale and gets closed out.
        removed_metadata_hashes = {h for h in live_metadata_hashes if h != metadata_hash}
        if removed_metadata_hashes:
            removed_hashes |= removed_metadata_hashes

        # Build embed inputs: content pieces being added + the metadata piece
        # (if it's being added). Content pieces preserve their input order so
        # the chunker_index in the resulting chunks row matches what the
        # chunker emitted. The metadata piece is appended last with a
        # sentinel chunk_index (METADATA_CHUNK_INDEX) baked in by `_metadata_piece`.
        added_content_pieces: list[ChunkPiece] = [
            new_by_hash[h] for h in new_hashes if h in added_hashes
        ]
        embed_inputs: list[ChunkPiece] = list(added_content_pieces)
        metadata_input_index: int | None = None
        if metadata_add and metadata_piece is not None:
            metadata_input_index = len(embed_inputs)
            embed_inputs.append(metadata_piece)

        # Embed OUTSIDE any txn. This is the long I/O — 60-120s for big
        # sessions. Holding a write txn across it caused the 2026-04-29
        # incident.
        added_pieces: list[tuple[ChunkPiece, list[float], str]] = []
        failed_pieces: list[tuple[ChunkPiece | None, Any]] = []
        failed_count = 0
        if embed_inputs:
            embeds = await self._embedder.embed_many([p.content for p in embed_inputs])
            for match in embeds.embedded:
                if match.chunk_index < 0 or match.chunk_index >= len(embed_inputs):
                    continue
                piece = embed_inputs[match.chunk_index]
                kind = (
                    "metadata"
                    if metadata_input_index is not None
                    and match.chunk_index == metadata_input_index
                    else "content"
                )
                added_pieces.append((piece, match.embedding, kind))
            for fail in embeds.failed:
                # Map back to the originating piece to record chunk_index on failure.
                if 0 <= fail.chunk_index < len(embed_inputs):
                    fail_piece: ChunkPiece | None = embed_inputs[fail.chunk_index]
                else:
                    fail_piece = None
                failed_pieces.append((fail_piece, fail))
                # Don't count metadata-chunk failures against `failed` — that
                # counter feeds the chunk-sync outcome which is content-shaped.
                if metadata_input_index is None or fail.chunk_index != metadata_input_index:
                    failed_count += 1

        added_content_count = len(added_content_pieces) - failed_count
        return _ChunkPlan(
            reused_content_hashes=reused_hashes,
            reused_metadata_hash=metadata_hash if metadata_reuse else None,
            added_pieces=added_pieces,
            failed_pieces=failed_pieces,
            removed_hashes=removed_hashes,
            added_count=added_content_count,
            reused_count=len(reused_hashes),
            removed_count=len(removed_hashes),
            failed_count=failed_count,
            live_count=len(reused_hashes) + added_content_count,
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


@dataclass(slots=True)
class _ChunkPlan:
    """Pre-computed chunk diff + embeddings for one document.

    Phase A (`Normalizer._plan_chunks`) builds this without holding a write
    transaction — it does the live-chunks SELECT under a short read txn,
    closes the txn, then calls `embed_many` outside any txn. Phase B
    (`_apply_chunk_plan`) takes the plan and runs the three writes inside
    the single shared write txn that also covers nodes/edges/ACL/docs.

    The split exists because `embed_many` is a 60-120s OpenAI round trip
    for long sessions — holding row locks across it caused conflicting
    `graph_nodes` upserts from concurrent workers to time out and DLQ.
    See incident 2026-04-29.
    """

    # Content-chunk hashes that already exist live → just bump last_seen_version.
    reused_content_hashes: set[str] = field(default_factory=set)
    # Metadata chunk's hash if it already exists live (singleton). None if no
    # metadata chunk in the plan or if it's being added/changed.
    reused_metadata_hash: str | None = None
    # (piece, embedding, kind) triples to insert. Kind is "content" or "metadata".
    added_pieces: list[tuple[ChunkPiece, list[float], str]] = field(default_factory=list)
    # (piece_or_None, failed_record) tuples from embedding failures. Goes to
    # failed_chunks. Phase B writes these inside the per-doc savepoint.
    failed_pieces: list[tuple[ChunkPiece | None, Any]] = field(default_factory=list)
    # All hashes that should be marked valid_to=NOW() — content + metadata stale.
    removed_hashes: set[str] = field(default_factory=set)
    # Pre-computed counts for the outcome (computed in Phase A so Phase B's
    # `_apply_chunk_plan` is purely SQL — no derived bookkeeping).
    added_count: int = 0
    reused_count: int = 0
    removed_count: int = 0
    failed_count: int = 0
    live_count: int = 0


# ---- SQL helpers (module-level so tests can unit-test them) ---------------


# Bounded retry budget for the version-compute + INSERT loop. Five is high
# enough to ride out the worst observed concurrent-writer storm against a
# single doc_id (3-4 contenders) without spinning forever; an exhausted budget
# raises a transient error so the worker requeues the row instead of DLQing.
_UPSERT_DOC_MAX_RETRIES = 5


class _UpsertDocumentRaceExhausted(NormalizationError):
    """Concurrent writers kept winning the version race past the retry budget.

    Marked transient so the worker re-queues the row (the next attempt will
    see the latest version and slot in cleanly) rather than DLQing.
    """

    transient = True


async def _upsert_document(conn: asyncpg.Connection, doc: Document) -> bool:
    """Insert a new document version if content changed.

    Returns True if a new (doc_id, version) row was written (and the prior
    live version was closed out), False if this payload matches the current
    live version's content_hash and isn't a delete (i.e. no-op).

    Mutates `doc.version` to the newly-written version so callers can use it
    when writing chunks.

    `doc.coalesce_into_live` opts the doc into update-in-place semantics:
    when True AND a live version exists AND that live version is itself in a
    coalesce-eligible state (e.g. session_complete=False), we UPDATE the
    live row in place rather than opening a new SCD2 version. This kills
    the per-batch SCD2 amplification on long-running claude_code sessions.

    Wrapped in a bounded retry loop because two concurrent writers can both
    read `version=N`, both compute `N+1`, and one's INSERT silently no-ops
    via `ON CONFLICT DO NOTHING` — losing the second writer's content. On
    that conflict we re-read the live state inside the same `with_tenant`
    txn and retry with the freshly-bumped version. Read-committed isolation
    guarantees the conflicting writer's commit is visible to the next read.
    """
    for attempt in range(_UPSERT_DOC_MAX_RETRIES):
        existing = await conn.fetchrow(
            """
            SELECT version, content_hash, metadata
            FROM documents
            WHERE doc_id = $1 AND customer_id = $2 AND valid_to IS NULL
            LIMIT 1
            """,
            doc.doc_id,
            doc.customer_id,
        )

        if existing and existing["content_hash"] == doc.content_hash and doc.deleted_at is None:
            # Same content, not a delete → idempotent no-op. A retry can land
            # here too: a concurrent writer wrote OUR exact content first.
            # Preserve the live version on the in-memory doc so chunk writes
            # (which use doc.version) target the existing row.
            doc.version = existing["version"]
            return False

        # Coalesce-in-place path: while the live version is still in an
        # incomplete state (e.g. session_complete=False) and the incoming doc
        # asks for coalescing, UPDATE the live row in place. Same version,
        # refreshed content_hash + metadata + body_preview + token counts.
        # Chunk writes (called after this returns True) then diff against the
        # same version, so reused chunks just bump last_seen_version.
        if existing is not None and doc.coalesce_into_live and doc.deleted_at is None:
            existing_meta = _coerce_jsonb(existing["metadata"])
            prior_complete = bool(existing_meta.get("session_complete"))
            if not prior_complete:
                doc.version = existing["version"]
                await conn.execute(
                    """
                    UPDATE documents
                    SET content_hash = $3,
                        title = $4,
                        body_preview = $5,
                        body_size_bytes = $6,
                        body_token_count = $7,
                        updated_at = $8,
                        valid_from = $9,
                        metadata = $10::jsonb,
                        entities = $11::jsonb,
                        attachments = $12::jsonb,
                        doc_references = $13::jsonb,
                        ingested_at = $14
                    WHERE customer_id = $1 AND doc_id = $2 AND version = $15
                      AND valid_to IS NULL
                    """,
                    doc.customer_id,
                    doc.doc_id,
                    doc.content_hash,
                    doc.title,
                    doc.body_preview,
                    doc.body_size_bytes,
                    doc.body_token_count,
                    doc.updated_at,
                    doc.valid_from,
                    _json(doc.metadata),
                    _json([e.model_dump() for e in doc.entities]),
                    _json([a.model_dump() for a in doc.attachments]),
                    _json([r.model_dump() for r in doc.doc_references]),
                    doc.ingested_at,
                    existing["version"],
                )
                # Returning True so the caller proceeds to chunk diff against
                # the same version; chunk diff handles add/reuse/remove correctly.
                return True

        if existing is not None:
            # Close out the prior live version in the same transaction. Idempotent
            # under retry: a second pass against an already-closed row matches
            # zero rows and is a silent no-op.
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

        inserted = await conn.fetchval(
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
            ON CONFLICT (customer_id, doc_id, version) DO NOTHING
            RETURNING 1
            """,
            doc.doc_id,
            next_version,
            doc.customer_id,
            doc.source_system.value,
            doc.source_id,
            doc.source_url,
            doc.doc_class.value,
            doc.doc_type,
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
        if inserted is not None:
            doc.version = next_version
            return True

        # ON CONFLICT swallowed the write — a concurrent writer claimed this
        # (customer_id, doc_id, version) slot. Re-read on the next iteration
        # so we land on top of their bumped version.
        log.info(
            "upsert_document.retry",
            customer=doc.customer_id,
            doc_id=doc.doc_id,
            attempted_version=next_version,
            attempt=attempt + 1,
        )

    raise _UpsertDocumentRaceExhausted(
        "exhausted retry budget computing next document version under contention",
        customer=doc.customer_id,
        doc_id=doc.doc_id,
        retries=_UPSERT_DOC_MAX_RETRIES,
    )


async def _insert_chunk(
    conn: asyncpg.Connection,
    doc: Document,
    piece: ChunkPiece,
    embedding: list[float],
    kind: str = "content",
) -> None:
    content_hash = _chunk_hash(piece.content)
    # Distinct chunk_id prefix for metadata so a same-content_hash collision
    # between content and metadata (theoretical) doesn't ON CONFLICT-update
    # the wrong row.
    prefix = "m_" if kind == "metadata" else "c_"
    chunk_id = f"{doc.doc_id}:{prefix}{content_hash[:16]}"
    await conn.execute(
        """
        INSERT INTO chunks (
            chunk_id, doc_id, customer_id,
            chunk_index, content, content_hash, token_count,
            embedding, embedding_model, embedding_dim, chunker_version,
            first_seen_version, last_seen_version, kind
        )
        VALUES (
            $1, $2, $3,
            $4, $5, $6, $7,
            $8::halfvec, $9, $10, $11,
            $12, $12, $13
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
        kind,
    )


async def _insert_chunks_batch(
    conn: asyncpg.Connection,
    doc: Document,
    added_pieces: list[tuple[ChunkPiece, list[float], str]],
) -> None:
    """Batched counterpart to `_insert_chunk` — one INSERT for all pieces.

    Dedupes by content_hash before insert: the unique constraint is
    (doc_id, content_hash), and ON CONFLICT DO UPDATE can't touch the same
    target row twice in one statement. The prior loop handled duplicate
    hashes by letting the second iteration UPDATE the row inserted by the
    first; the batched form collapses duplicates upfront with last-wins
    semantics on the per-piece fields (chunk_index, content) — they're
    identical across same-hash entries in practice.
    """
    chunk_ids: list[str] = []
    chunk_indexes: list[int] = []
    contents: list[str] = []
    content_hashes: list[str] = []
    token_counts: list[int] = []
    embeddings: list[str] = []
    kinds: list[str] = []
    seen_hashes: set[str] = set()

    for piece, embedding, kind in added_pieces:
        content_hash = _chunk_hash(piece.content)
        if content_hash in seen_hashes:
            continue
        seen_hashes.add(content_hash)
        # See _insert_chunk for rationale on the prefix — keeps a metadata
        # chunk's chunk_id from colliding with a content chunk that happens
        # to share content_hash.
        prefix = "m_" if kind == "metadata" else "c_"
        chunk_ids.append(f"{doc.doc_id}:{prefix}{content_hash[:16]}")
        chunk_indexes.append(piece.chunk_index)
        contents.append(piece.content)
        content_hashes.append(content_hash)
        token_counts.append(piece.token_count)
        embeddings.append(_pg_vector(embedding))
        kinds.append(kind)

    if not chunk_ids:
        return

    await conn.execute(
        """
        INSERT INTO chunks (
            chunk_id, doc_id, customer_id,
            chunk_index, content, content_hash, token_count,
            embedding, embedding_model, embedding_dim, chunker_version,
            first_seen_version, last_seen_version, kind
        )
        SELECT
            chunk_id, $2, $3,
            chunk_index, content, content_hash, token_count,
            embedding::halfvec, $9, $10, $11,
            $12, $12, kind
        FROM unnest(
            $1::text[], $4::int[], $5::text[], $6::text[], $7::int[],
            $8::text[], $13::text[]
        ) AS t(chunk_id, chunk_index, content, content_hash, token_count,
               embedding, kind)
        ON CONFLICT (doc_id, content_hash) DO UPDATE
            SET last_seen_version = EXCLUDED.last_seen_version,
                valid_to = NULL
        """,
        chunk_ids,
        doc.doc_id,
        doc.customer_id,
        chunk_indexes,
        contents,
        content_hashes,
        token_counts,
        embeddings,
        EMBEDDING_MODEL,
        EMBEDDING_DIM,
        CHUNKER_VERSION,
        doc.version,
        kinds,
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


async def _apply_chunk_plan(
    conn: asyncpg.Connection, doc: Document, plan: _ChunkPlan
) -> _ChunkSyncOutcome:
    """Phase B: apply a pre-computed `_ChunkPlan` inside the write txn.

    No external I/O. Pure SQL: bump last_seen_version on reused, INSERT
    new chunks (with embeddings already computed in Phase A), record
    embedding failures, mark removed chunks stale.

    Counts are pre-computed in Phase A and passed through verbatim — the
    caller's outcome accumulators don't notice the split.
    """
    # 1) Reuse: bump last_seen_version. Combined: content reused + metadata reused.
    reused_for_bump: set[str] = set(plan.reused_content_hashes)
    if plan.reused_metadata_hash is not None:
        reused_for_bump.add(plan.reused_metadata_hash)
    if reused_for_bump:
        await conn.execute(
            """
            UPDATE chunks
            SET last_seen_version = $1
            WHERE customer_id = $2 AND doc_id = $3
              AND content_hash = ANY($4::text[]) AND valid_to IS NULL
            """,
            doc.version,
            doc.customer_id,
            doc.doc_id,
            list(reused_for_bump),
        )

    # 2) Added: insert with pre-computed embeddings. One INSERT for all
    #    chunks regardless of count — per-chunk round-trips inside Phase B
    #    were the next contention layer behind the per-node loop in
    #    upsert_nodes.
    if plan.added_pieces:
        await _insert_chunks_batch(conn, doc, plan.added_pieces)
    for piece, fail in plan.failed_pieces:
        await _insert_failed_chunk(conn, doc, fail, piece)

    # 3) Removed: mark stale at NOW().
    if plan.removed_hashes:
        await conn.execute(
            """
            UPDATE chunks
            SET valid_to = NOW()
            WHERE customer_id = $1 AND doc_id = $2
              AND content_hash = ANY($3::text[]) AND valid_to IS NULL
            """,
            doc.customer_id,
            doc.doc_id,
            list(plan.removed_hashes),
        )

    return _ChunkSyncOutcome(
        added=plan.added_count,
        reused=plan.reused_count,
        removed=plan.removed_count,
        failed=plan.failed_count,
        live=plan.live_count,
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
            ON CONFLICT ON CONSTRAINT acl_snapshots_assertion_unique DO NOTHING
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


async def _upsert_code_repo_state(
    conn: asyncpg.Connection,
    customer_id: str,
    rows: list,
) -> None:
    """Upsert code-graph per-file cache rows. No-op for non-code-graph runs.

    `code_repo_state` is keyed (customer_id, repo, file_path). The cache
    short-circuits re-extraction when content_hash + extractor_version match
    the prior run, so this is the cumulative source of truth for "what
    extraction state is current."
    """
    if not rows:
        return
    for r in rows:
        await conn.execute(
            """
            INSERT INTO code_repo_state
                (customer_id, repo, file_path, content_hash, language,
                 symbol_count, last_extracted_at, last_extractor_version)
            VALUES ($1, $2, $3, $4, $5, $6, NOW(), $7)
            ON CONFLICT (customer_id, repo, file_path) DO UPDATE SET
                content_hash = EXCLUDED.content_hash,
                language = EXCLUDED.language,
                symbol_count = EXCLUDED.symbol_count,
                last_extracted_at = NOW(),
                last_extractor_version = EXCLUDED.last_extractor_version
            """,
            customer_id,
            r.repo,
            r.file_path,
            r.content_hash,
            r.language,
            r.symbol_count,
            r.extractor_version,
        )


# ---- formatting helpers --------------------------------------------------


def _json(obj: Any) -> str:
    return orjson.dumps(obj, default=_json_default).decode("utf-8")


def _coerce_jsonb(raw: Any) -> dict[str, Any]:
    """Decode an asyncpg jsonb value into a dict.

    asyncpg returns jsonb as bytes/str (raw JSON) or as already-decoded dict
    depending on codec configuration. This util collapses both into a dict
    so callers don't have to branch.
    """
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, (bytes, bytearray, str)):
        decoded = orjson.loads(raw)
        if isinstance(decoded, dict):
            return decoded
    return {}


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


# METADATA_CHUNK_INDEX moved to shared.models so cross-module callers
# (code_graph pipeline) can reach for the same constant. Re-imported
# at module top for back-compat with existing references in this file.


# Strip opaque commit SHAs from github URLs before embedding — they
# tokenize into garbage and waste vector capacity. Captures `/commit/`
# followed by a 40-hex SHA, replaces with `/commit/`. Keeps the
# human-readable repo path intact.
_GITHUB_SHA_RE = re.compile(r"/commit/[0-9a-f]{40}")


def _strip_opaque_ids(url: str) -> str:
    """Remove SHAs and similar opaque IDs from a URL prior to embedding."""
    if not url:
        return url
    return _GITHUB_SHA_RE.sub("/commit/", url)


def _metadata_text(doc: Document) -> str:
    """Structured key:value text for the synthetic metadata chunk.

    Contains the human-readable doc fields that the search path needs to
    rank metadata-keyed queries on ("what's going on with prbe-backend"
    matches `repo:` line via embedding similarity / BM25).

    Excludes opaque IDs (doc_id, content_hash, SHAs in URLs) — they
    waste embedding capacity and never match real queries. Stable across
    re-ingestion of the same document state, so the resulting chunk's
    content_hash is idempotent under retries.

    Exception: `source_id` (the handler-supplied stable identifier — e.g.
    a session UUID or `issue:<uuid>`) lands as its own line so a query
    naming the full id can BM25-match the metadata chunk. Today's title
    only carries a short prefix (8-char UUID slice), so without this line
    the full id is not searchable lexically.
    """
    lines: list[str] = []
    if doc.title:
        lines.append(f"title: {doc.title}")
    lines.append(f"source: {doc.source_system.value}")
    if doc.source_id:
        lines.append(f"id: {doc.source_id}")
    if doc.author_id:
        lines.append(f"author: {doc.author_id}")
    # Co-authors (currently github commits via Co-authored-by trailers) get
    # one line per entry so they're indexed by BM25 and vector alongside the
    # primary author. Without this, a commit Mahit only co-authored never
    # surfaces for "Mahit's work" queries via the metadata-chunk path —
    # only via the graph retriever's AUTHORED edge, which requires the
    # router to extract a Person entity.
    co_authors = doc.metadata.get("co_authors") if isinstance(doc.metadata, dict) else None
    if isinstance(co_authors, list):
        for c in co_authors:
            if not isinstance(c, dict):
                continue
            email = c.get("email")
            name = c.get("name")
            if email and name:
                lines.append(f"co_author: {email} ({name})")
            elif email:
                lines.append(f"co_author: {email}")
    if doc.source_url:
        lines.append(f"url: {_strip_opaque_ids(doc.source_url)}")
    if doc.body_preview:
        # First line of the body gives ranking a content anchor without
        # bloating the metadata chunk to full chunk size.
        first_line = doc.body_preview.strip().splitlines()
        if first_line:
            lines.append(f"summary: {first_line[0][:200]}")
    return "\n".join(lines)


def _metadata_piece(doc: Document) -> ChunkPiece | None:
    """Build a ChunkPiece for the doc's metadata text. None if there's
    nothing useful to embed (e.g. no title and no body_preview)."""
    text = _metadata_text(doc)
    if not text.strip():
        return None
    from services.ingestion.chunker import count_tokens

    return ChunkPiece(
        chunk_index=METADATA_CHUNK_INDEX,
        content=text,
        token_count=count_tokens(text),
    )


def _stringify_body(doc: Document) -> str:
    """Pull the embeddable body text off a document.

    Connectors set `doc.body` (a transient, never-persisted field on Document)
    so the chunker can see the full text. The documents table itself only
    carries `body_preview`; the source of truth for full text is chunks.content.

    Historically connectors stuffed body into `metadata["body"]`, which
    duplicated ~440 MB of storage. That key is no longer written by any
    handler. The fallback to metadata["body"] below remains only for stray
    test fixtures or mid-deploy queue rows that predate the migration; it
    can be removed after the wiki/runbook/storage-cleanup migration drains.
    """
    if doc.body is not None and doc.body != "":
        return doc.body
    legacy = doc.metadata.get("body")
    if legacy:
        return str(legacy)
    # Fall back to title + preview so we still get a chunk for short entities.
    parts = [p for p in (doc.title, doc.body_preview) if p]
    return "\n\n".join(parts) if parts else ""


def ensure_tenant_bound(customer_id: str) -> None:
    if not customer_id:
        raise TenantIsolationError("customer_id required for normalize")


async def fetch_live_body_from_chunks(
    conn: asyncpg.Connection,
    customer_id: str,
    doc_id: str,
) -> str:
    """Reconstruct a document's live body by joining its content chunks.

    Replaces the ~440 MB documents.metadata['body'] duplication: chunks.content
    is the source of truth for full text, ordered by chunk_index. Filters to
    kind='content' so the synthetic per-doc metadata chunk doesn't bleed in.
    Returns an empty string when no live content chunks exist (e.g. deleted
    or never-chunked doc).

    Note: chunker overlap means string_agg can include some duplicated text
    between adjacent chunks (typically 10-15% inflation). That's harmless for
    LLM synthesis input but callers comparing exact byte equality vs the
    original source body need to keep that in mind.
    """
    body = await conn.fetchval(
        """
        SELECT string_agg(content, '' ORDER BY chunk_index)
        FROM chunks
        WHERE customer_id = $1
          AND doc_id = $2
          AND kind = 'content'
          AND valid_to IS NULL
        """,
        customer_id,
        doc_id,
    )
    return body or ""


async def fetch_body_from_chunks_for_version(
    conn: asyncpg.Connection,
    customer_id: str,
    doc_id: str,
    version: int,
) -> str:
    """Reconstruct a specific document version's body from chunks.

    Chunks span versions via [first_seen_version, last_seen_version]; a row
    is part of version V if first_seen_version <= V <= last_seen_version.
    Used by wiki history/revert flows where any prior version may need to
    be read back.
    """
    body = await conn.fetchval(
        """
        SELECT string_agg(content, '' ORDER BY chunk_index)
        FROM chunks
        WHERE customer_id = $1
          AND doc_id = $2
          AND kind = 'content'
          AND first_seen_version <= $3
          AND last_seen_version >= $3
        """,
        customer_id,
        doc_id,
        version,
    )
    return body or ""
