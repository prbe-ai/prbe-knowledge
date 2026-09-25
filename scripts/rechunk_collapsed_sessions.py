"""One-off: re-chunk the session transcripts the credential scrubber collapsed.

WHAT HAPPENED. From 2026-09-18 (#568) the normalizer scrubbed the whole session
body as one FIELD. The field scrubber replaces a value entirely when it holds
any `%XX`/`\\uXXXX` escape (or a NUL) plus a finding anywhere, so a long
transcript came back as the 10-character placeholder `<redacted>`. That chunked
into ONE chunk, and the chunk diff retired every real chunk the session had.
The fix (#587) scrubs the body as free text. It repairs a session the next time
the worker writes it with a CHANGED body, but an ended session never changes:
its `content_hash` is the hash of the unscrubbed body, so a re-queued pass is
skipped as unchanged and its chunks stay retired. This script is that repair.

SELECTION, per tenant (FORCE RLS: every read runs under the tenant GUC): the
live, undeleted `*.session` document of an agent-session source that has a live
content chunk equal to `<redacted>` and whose live content chunks hold under a
tenth of `body_size_bytes`. Measured on research 2026-09-23: 28 documents at
under 0.001 of their body, every other session document at 0.9-1.5.

PER DOCUMENT:

  1. Re-render the body from R2 exactly as the worker does (first readable
     payload -> `parse_webhook_event` -> `fetch_supplementary` over every key
     on the session's queue row -> `normalize`). The session is declared LIVE
     for this pass: the body renders identically either way, and only an
     ended pass runs extraction (mining, a paid LLM call). No other LLM call
     exists on this path.
  2. Only the body comes from R2. Everything else the chunks carry (title,
     preview, author, url, metadata, visibility, version) is taken from the
     live documents row, which this script never writes. The metadata chunk
     therefore re-renders identically and is reused.
  3. `_plan_chunks` -- the FIXED scrub, the chunker, the diff against live
     chunks, and on --write the normal embedder for every added chunk. A dry
     run swaps in a counting embedder: it reports how many texts --write would
     embed and calls nothing.
  4. On --write, ONE transaction: lock the live documents row, confirm the row
     and its live chunk set are exactly what was planned against, then
     `_apply_chunk_plan`. A document that changed is refused, never forced.
     Errors are isolated per document.

REFUSED (reported, not written):

  still_collapses   the scrubbed body is still the bare placeholder
  partial_render    the render is under half of `body_size_bytes`, or a key on
                    the queue row has no object in R2 (--allow-partial writes it)
  queue_row_active  the queue row is pending/processing: the worker will
                    rewrite this session itself, with the fixed scrubber
  embedding_failed  a chunk failed to embed; writing would lose it
  document_changed / chunks_changed   moved between plan and write; re-run

NEVER: writes `documents` or `ingestion_queue`, runs extraction, prints any
document text. Output is ids and counts, one JSON line per document.

Run inside the engine worker pod (R2 and DB credentials, `redactd` on PATH), on
an image that carries #587. On one that does not, the scrub still collapses and
every document is refused as `still_collapses`, so nothing is written:

    kubectl --context do-sfo3-probe-research -n research exec deploy/research-os-engine-worker -- \\
        python -m scripts.rechunk_collapsed_sessions --all-tenants
    kubectl --context do-sfo3-probe-research -n research exec deploy/research-os-engine-worker -- \\
        python -m scripts.rechunk_collapsed_sessions --customer probe \\
        --doc-id claude_code:probe:55abf56b-045f-40d6-9494-f46218c3aec7 --write
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import orjson

from engine.ingest.handlers.base import make_default_context
from engine.ingest.normalizer import Normalizer, _apply_chunk_plan, _ChunkPlan, _coerce_jsonb
from engine.shared import storage
from engine.shared.constants import AGENT_SESSION_SOURCES, SourceSystem, Visibility
from engine.shared.db import close_pool, get_pool, init_pool, with_tenant
from engine.shared.embeddings import (
    DocItem,
    EmbeddedChunk,
    EmbedResult,
    GeminiEmbedder,
    get_embedder_v2,
)
from engine.shared.exceptions import StorageNotFound
from engine.shared.models import Document, WebhookEvent
from engine.shared.session_signals import is_cron_marker_key
from engine.shared.tenant_status import ACTIVE_TENANTS_SQL

# THE CONNECTORS REGISTER ON IMPORT (engine/ingest/handlers/registry.py), and
# only the ingestion app and the worker import them at startup. Without this the
# re-render finds no connector for any session source: the first prod dry run
# raised HandlerNotFound for all 27 documents, while the tests passed because
# tests/conftest.py imports the package for them.
import kb.handlers  # noqa: F401  # isort: skip

PLACEHOLDER = "<redacted>"
#: Live content chunks holding under this fraction of the body is "collapsed".
COVERAGE_CEILING = 0.1
#: A render under this fraction of the stored body size is not written.
MIN_RENDER_FRACTION = 0.5

_SELECT_SQL = """
WITH candidates AS (
    SELECT DISTINCT c.doc_id
    FROM chunks c
    WHERE c.customer_id = $1
      AND c.valid_to IS NULL
      AND c.kind = 'content'
      AND c.content = $2
      AND ($5::text IS NULL OR c.doc_id = $5)
)
SELECT * FROM (
    SELECT d.doc_id, d.version, d.source_system, d.source_id, d.content_hash,
           d.body_size_bytes, d.title, d.body_preview, d.author_id, d.source_url,
           d.metadata, d.visibility,
           (SELECT coalesce(sum(length(c.content)), 0)
              FROM chunks c
             WHERE c.customer_id = d.customer_id AND c.doc_id = d.doc_id
               AND c.valid_to IS NULL AND c.kind = 'content') AS live_chars
    FROM candidates k
    JOIN documents d
      ON d.customer_id = $1 AND d.doc_id = k.doc_id
     AND d.valid_to IS NULL AND d.deleted_at IS NULL
    WHERE d.source_system = ANY($3::text[])
      AND d.doc_type LIKE '%.session'
) s
-- float8, or asyncpg infers int4 from body_size_bytes and sends 0.1 as 0.
WHERE s.live_chars < s.body_size_bytes * $4::float8
ORDER BY s.doc_id
"""


class _GapCountingStore:
    """The real store, except a missing object reads as empty and is counted.

    The connector skips an empty payload, so a gap shortens the render instead
    of failing it -- which is what lets `partial_render` measure a gap at all.
    Installed as the process store for the run: the connector reads R2 through
    `get_store()`, not through the normalizer.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.missing: set[str] = set()

    async def get(self, bucket: str, key: str) -> bytes:
        try:
            return await self._inner.get(bucket, key)
        except StorageNotFound:
            self.missing.add(key)
            return b""

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _CountingEmbedder:
    """Dry-run embedder: counts what --write would embed and calls nothing."""

    def __init__(self) -> None:
        self.texts = 0

    async def embed_documents(self, items: list[DocItem]) -> EmbedResult:
        self.texts += len(items)
        return EmbedResult(
            embedded=[EmbeddedChunk(chunk_index=i, embedding=[]) for i in range(len(items))],
            failed=[],
        )


class _RenderFailed(Exception):
    pass


@dataclass
class Summary:
    selected: int = 0
    written: int = 0
    would_write: int = 0
    refused: Counter[str] = field(default_factory=Counter)
    errors: Counter[str] = field(default_factory=Counter)
    embed_texts: int = 0
    embed_requests: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "event": "rechunk.summary",
            "selected": self.selected,
            "written": self.written,
            "would_write": self.would_write,
            "refused": dict(self.refused),
            "errors": dict(self.errors),
            "embed_texts": self.embed_texts,
            "embed_requests": self.embed_requests,
        }


def _embed_requests(texts: int) -> int:
    """Provider round trips for `texts` inputs: the embedder sends groups of 4."""
    return math.ceil(texts / GeminiEmbedder._SUBBATCH_GROUP_SIZE)


async def _tenants(customers: list[str] | None, all_tenants: bool) -> list[str]:
    if not all_tenants:
        return list(dict.fromkeys(customers or []))
    # `customers` is not row-secured: the readable tenant list is what makes
    # the per-tenant GUC loop possible (same as scripts/cron_chunk_retention).
    # ACTIVE tenants only: a held tenant is not re-rendered or re-embedded.
    async with get_pool().acquire() as conn:
        rows = await conn.fetch(ACTIVE_TENANTS_SQL)
    return [r["customer_id"] for r in rows]


async def select_collapsed(customer_id: str, doc_id: str | None = None) -> list[Any]:
    async with with_tenant(customer_id) as conn:
        return list(
            await conn.fetch(
                _SELECT_SQL,
                customer_id,
                PLACEHOLDER,
                [s.value for s in AGENT_SESSION_SOURCES],
                COVERAGE_CEILING,
                doc_id,
            )
        )


async def _queue_row(customer_id: str, source: str, session_id: str) -> Any:
    async with with_tenant(customer_id) as conn:
        rows = await conn.fetch(
            """
            SELECT queue_id, status, payload_s3_key, payload_s3_keys
            FROM ingestion_queue
            WHERE customer_id = $1 AND source_system = $2 AND source_event_id = $3
            """,
            customer_id,
            source,
            session_id,
        )
    if len(rows) != 1:
        raise _RenderFailed(f"expected one queue row, found {len(rows)}")
    return rows[0]


async def _render(
    normalizer: Normalizer, store: Any, customer_id: str, source: SourceSystem, keys: list[str]
) -> Document:
    """The worker's read -> parse -> fetch -> normalize, on the no-extraction branch."""
    bucket = await store.bucket_for(customer_id)
    first: tuple[str, dict[str, Any], Any] | None = None
    for key in keys:
        if is_cron_marker_key(key):
            continue
        raw = await store.get(bucket, key)
        if raw:
            envelope = orjson.loads(raw)
            first = (key, envelope.get("_headers", {}), envelope.get("payload", envelope))
            break
    if first is None:
        raise _RenderFailed("no readable payload on the queue row")
    first_key, headers, payload = first
    connector = normalizer._connector(source)
    parsed = connector.parse_webhook_event(customer_id, headers, payload)
    if parsed is None:
        raise _RenderFailed("parse_webhook_event returned None")
    event = WebhookEvent(
        customer_id=customer_id,
        source_system=source,
        source_event_id=parsed.source_event_id,
        received_at=parsed.received_at,
        payload_s3_key=first_key,
        payload_s3_keys=keys,
        raw_payload=payload,
        headers=headers,
    )
    token = await normalizer._load_token(customer_id, source)
    hydrated = dict(await connector.fetch_supplementary(event, token))
    # The body renders the same whether or not the session ended; only an ENDED
    # pass mines it, and mining is a paid LLM call this repair must not make.
    hydrated["session_complete"] = False
    hydrated["completed_by"] = None
    result = await connector.normalize(event, hydrated)
    if (
        result.extraction_outcome is not None
        or result.documents_with_chunks
        or len(result.documents) != 1
    ):
        raise _RenderFailed("normalize produced more than the session document")
    return result.documents[0]


async def rechunk_one(
    normalizer: Normalizer,
    store: _GapCountingStore,
    customer_id: str,
    row: Any,
    *,
    write: bool,
    allow_partial: bool,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "event": "rechunk.doc",
        "customer": customer_id,
        "doc_id": row["doc_id"],
        "version": row["version"],
        "body_bytes": row["body_size_bytes"],
        "live_chars": row["live_chars"],
    }
    source = SourceSystem(row["source_system"])
    queue = await _queue_row(customer_id, source.value, row["source_id"])
    keys = list(queue["payload_s3_keys"] or [])
    if not keys and queue["payload_s3_key"]:
        keys = [queue["payload_s3_key"]]
    out.update(queue_id=queue["queue_id"], queue_status=queue["status"], keys=len(keys))

    store.missing = set()
    doc = await _render(normalizer, store, customer_id, source, keys)
    if doc.doc_id != row["doc_id"]:
        raise _RenderFailed("rendered a different document id")
    body = doc.body or ""
    rendered_bytes = len(body.encode("utf-8"))
    out.update(
        missing_keys=len(store.missing),
        rendered_chars=len(body),
        rendered_bytes=rendered_bytes,
        # True when R2 reproduces exactly the body the live row was written from.
        hash_match=hashlib.sha256(body.encode("utf-8")).hexdigest() == row["content_hash"],
    )

    # Only the body comes from R2; the live row is authoritative for the rest.
    doc.version = row["version"]
    doc.title = row["title"]
    doc.body_preview = row["body_preview"]
    doc.author_id = row["author_id"]
    doc.source_url = row["source_url"]
    doc.metadata = _coerce_jsonb(row["metadata"])
    doc.visibility = Visibility(row["visibility"])

    plan: _ChunkPlan = await normalizer._plan_chunks(customer_id, doc)
    scrubbed = doc.body or ""
    embed_texts = len(plan.added_pieces) + len(plan.failed_pieces)
    out.update(
        scrubbed_chars=len(scrubbed),
        still_collapses=scrubbed.strip() == PLACEHOLDER,
        planned_chunks=plan.live_count,
        reused=plan.reused_count,
        retire=plan.removed_count,
        embed_texts=embed_texts,
        embed_requests=_embed_requests(embed_texts),
    )

    refusal: str | None = None
    if out["still_collapses"] or plan.live_count == 0:
        refusal = "still_collapses"
    elif not allow_partial and (
        store.missing or rendered_bytes < MIN_RENDER_FRACTION * (row["body_size_bytes"] or 0)
    ):
        refusal = "partial_render"
    elif queue["status"] in ("pending", "processing"):
        refusal = "queue_row_active"
    elif plan.failed_pieces:
        refusal = "embedding_failed"
    if refusal or not write:
        out["refused"] = refusal
        return out

    planned_against = plan.reused_content_hashes | plan.removed_hashes
    if plan.reused_metadata_hash is not None:
        planned_against = planned_against | {plan.reused_metadata_hash}
    async with with_tenant(customer_id) as conn:
        current = await conn.fetchrow(
            """
            SELECT version, content_hash FROM documents
            WHERE customer_id = $1 AND doc_id = $2 AND valid_to IS NULL
            FOR UPDATE
            """,
            customer_id,
            row["doc_id"],
        )
        if current is None or (current["version"], current["content_hash"]) != (
            row["version"],
            row["content_hash"],
        ):
            out["refused"] = "document_changed"
            return out
        live = {
            r["content_hash"]
            for r in await conn.fetch(
                "SELECT content_hash FROM chunks "
                "WHERE customer_id = $1 AND doc_id = $2 AND valid_to IS NULL",
                customer_id,
                row["doc_id"],
            )
        }
        if live != planned_against:
            out["refused"] = "chunks_changed"
            return out
        outcome = await _apply_chunk_plan(conn, doc, plan)
    out.update(refused=None, written=True, live_after=outcome.live)
    return out


async def run(
    *,
    customers: list[str] | None = None,
    all_tenants: bool = False,
    doc_id: str | None = None,
    write: bool = False,
    allow_partial: bool = False,
    store: Any = None,
    embedder: Any = None,
    emit: Callable[[dict[str, Any]], None] = lambda d: print(json.dumps(d, default=str)),
) -> Summary:
    if write and embedder is None:
        embedder = get_embedder_v2()
        if not embedder._gateway_url and not embedder._api_key:
            # Stub mode hashes text into fake vectors; writing them would
            # replace a blank session with one no query can find.
            raise SystemExit("embedder is in stub mode (no LLM_GATEWAY_URL / GOOGLE_API_KEY)")
    counting = _CountingEmbedder()
    gap_store = _GapCountingStore(store or storage.get_store())
    normalizer = Normalizer(
        make_default_context(),
        store=gap_store,
        embedder=embedder if write else counting,  # type: ignore[arg-type]
    )
    summary = Summary()
    previous_store = storage._store
    storage._store = gap_store  # type: ignore[assignment]
    try:
        for customer_id in await _tenants(customers, all_tenants):
            for row in await select_collapsed(customer_id, doc_id):
                summary.selected += 1
                try:
                    out = await rechunk_one(
                        normalizer, gap_store, customer_id, row,
                        write=write, allow_partial=allow_partial,
                    )
                except Exception as exc:  # isolated per document
                    # The class name only: an exception message can quote content.
                    out = {
                        "event": "rechunk.doc",
                        "customer": customer_id,
                        "doc_id": row["doc_id"],
                        "error": type(exc).__name__,
                    }
                    summary.errors[type(exc).__name__] += 1
                else:
                    if out.get("refused"):
                        summary.refused[out["refused"]] += 1
                    elif out.get("written"):
                        summary.written += 1
                    else:
                        summary.would_write += 1
                    if not out.get("refused"):
                        summary.embed_texts += out["embed_texts"]
                        summary.embed_requests += out["embed_requests"]
                emit(out)
    finally:
        storage._store = previous_store
    emit(summary.as_dict())
    return summary


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--customer", action="append", help="Tenant to repair (repeatable).")
    scope.add_argument("--all-tenants", action="store_true", help="Every ACTIVE tenant in `customers`.")
    parser.add_argument("--doc-id", default=None, help="Only this document.")
    parser.add_argument(
        "--write", action="store_true", help="Embed and apply. Without it: dry run, no writes."
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Write even when R2 is missing keys or the render is under half the stored size.",
    )
    return parser.parse_args(argv)


async def _main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    await init_pool()
    try:
        await run(
            customers=args.customer,
            all_tenants=args.all_tenants,
            doc_id=args.doc_id,
            write=args.write,
            allow_partial=args.allow_partial,
        )
    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(_main())
