"""Bounded index lookup for interactive search, without the gatherer.

Both retrievers own their SQL, tenant RLS, live-version, visibility and scope
predicates. This adapter only combines their document ranks; it does not run
grounding, query expansion, graph traversal or a generative model.
"""

from __future__ import annotations

import asyncio
import time
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from engine.retrieval.retrievers.bm25 import BM25Hit, bm25_search
from engine.retrieval.retrievers.vector import VectorHit, vector_search
from engine.shared.constants import MAX_REQUEST_SOURCE_KEYS, SourceSystem
from engine.shared.custom_ingest import is_valid_source_key
from engine.shared.db import with_tenant
from engine.shared.logging import get_logger
from engine.shared.models import (
    MatchProvenance,
    QueryChunk,
    QueryDocumentResult,
    RetrieveResponse,
    ScopeSpec,
)

log = get_logger(__name__)
CHANNEL_TIMEOUT_SECONDS = 5.0
RRF_CONSTANT = 60
#: Body chunks a result carries.
CHUNKS_PER_DOCUMENT = 2
Hit = VectorHit | BM25Hit


class DirectChannel(StrEnum):
    """The index reads this adapter can run. VECTOR embeds the query (a model
    call); BM25 is keyword-only, so `["bm25"]` alone is a lookup with no model
    call at all."""

    VECTOR = "vector"
    BM25 = "bm25"


class DirectRetrieveRequest(BaseModel):
    """Only filters that both direct retrievers enforce before ranking."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=20, ge=1, le=50)
    sources: list[SourceSystem] | None = None
    source_keys: list[str] | None = Field(default=None, max_length=MAX_REQUEST_SOURCE_KEYS)
    doc_types: list[str] | None = None
    scope: ScopeSpec | None = None
    # An unrequested channel is never called: `["bm25"]` must not embed.
    channels: list[DirectChannel] = Field(
        default_factory=lambda: [DirectChannel.VECTOR, DirectChannel.BM25], min_length=1
    )
    # BM25 only: restate sources / source_keys inside the pg_search query so
    # they narrow its candidate pool instead of post-filtering it (see
    # `bm25_search`). Vector already applies them before its LIMIT.
    index_side_doc_filters: bool = False

    @field_validator("channels")
    @classmethod
    def dedupe_channels(cls, value: list[DirectChannel]) -> list[DirectChannel]:
        return list(dict.fromkeys(value))

    @field_validator("source_keys")
    @classmethod
    def valid_source_keys(cls, value: list[str] | None) -> list[str] | None:
        # The ingest charset, 128 characters at most. No stored document has a
        # key outside it, and an oversized one would otherwise reach BM25's
        # index-side regex and cost the whole channel instead of a 422.
        if value and not all(is_valid_source_key(key) for key in value):
            raise ValueError("each source_key must match ^[a-z0-9][a-z0-9:_-]{0,127}$")
        return value

    @field_validator("query")
    @classmethod
    def trim_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be blank")
        return value


class DirectRetrieveResponse(RetrieveResponse):
    """Bounded recall is explicit even when chunk caps leave few documents."""

    truncated: bool = False


async def _live_docs(
    req: DirectRetrieveRequest, customer_id: str, doc_ids: list[str]
) -> set[tuple[str, int]]:
    """Recheck live versions before shipping content, including tombstones.

    Retrievers narrow their own candidate pools, but a deletion or move can
    commit between the parallel reads. A scope/read failure must fail closed.
    """
    if not doc_ids:
        return set()
    async with asyncio.timeout(1.0), with_tenant(customer_id) as conn:
        rows = await conn.fetch(
            """SELECT doc_id, version FROM documents
               WHERE customer_id=$1 AND doc_id=ANY($2::text[])
                 AND deleted_at IS NULL AND valid_to IS NULL AND visibility='approved'
                 AND ($3::text[] IS NULL OR source_system=ANY($3))
                 AND ($4::text[] IS NULL OR doc_type=ANY($4))
                 AND ($5::text[] IS NULL OR metadata->>'source_key'=ANY($5))
                 AND ($6::text IS NULL OR metadata->>'project_id'=$6)""",
            customer_id,
            doc_ids,
            req.sources or None,
            req.doc_types or None,
            req.source_keys or None,
            req.scope.project_id if req.scope else None,
        )
    return {(row["doc_id"], row["version"]) for row in rows}


async def retrieve_direct(req: DirectRetrieveRequest, customer_id: str) -> DirectRetrieveResponse:
    started = time.perf_counter()
    # Chunks compete within each retriever, documents compete in the result.
    # Over-fetch once to give long documents room without a repeat-query loop.
    filters = {
        "top_k": min(req.top_k * 4, 200),
        "sources": req.sources,
        "source_keys": req.source_keys,
        "doc_types": req.doc_types,
        "project_id": req.scope.project_id if req.scope else None,
    }
    lost: list[str] = []
    timings: dict[str, float] = {}
    # Looked up per call, not bound at import, so the module attributes stay
    # the seam tests patch.
    searches = {DirectChannel.VECTOR: vector_search, DirectChannel.BM25: bm25_search}

    async def channel(name: DirectChannel) -> list[Hit]:
        kwargs = filters
        # Only when asked, so a default request calls bm25_search exactly as
        # before this flag existed.
        if name is DirectChannel.BM25 and req.index_side_doc_filters:
            kwargs = {**kwargs, "index_side_doc_filters": True}
        # A result keeps CHUNKS_PER_DOCUMENT chunks, so without a cap one long
        # document matching everywhere (a live session) can fill every chunk
        # slot and leave a one-document answer. BM25-only for now: the
        # default two-channel request (the typeahead) stays exactly as it was.
        if name is DirectChannel.BM25 and req.channels == [DirectChannel.BM25]:
            kwargs = {**kwargs, "max_chunks_per_doc": CHUNKS_PER_DOCUMENT}
        before = time.perf_counter()
        try:
            async with asyncio.timeout(CHANNEL_TIMEOUT_SECONDS):
                return await searches[name](customer_id, req.query, **kwargs)
        except Exception as exc:
            lost.append(name.value)
            log.warning(
                "direct_search.channel_failed", channel=name.value, error=type(exc).__name__
            )
            return []
        finally:
            timings[name.value] = round((time.perf_counter() - before) * 1000, 1)

    # Folded in enum order whatever order the caller listed, so a document
    # both channels found keeps its vector hit as the representative, as it
    # did before channels were selectable.
    requested = [name for name in DirectChannel if name in req.channels]
    gathered = await asyncio.gather(*(channel(name) for name in requested))
    candidate_cap_reached = any(len(hits) >= filters["top_k"] for hits in gathered)
    try:
        live = await _live_docs(
            req, customer_id, list({h.doc_id for hits in gathered for h in hits})
        )
    except Exception as exc:
        log.warning("direct_search.live_lookup_failed", error=type(exc).__name__)
        live = set()
        lost.append("live_documents")
    docs: dict[str, Hit] = {}
    evidence: dict[str, list[MatchProvenance]] = {}
    chunks: dict[str, dict[str, Hit]] = {}
    chunk_evidence: dict[str, list[MatchProvenance]] = {}
    for channel_name, channel_hits in zip(requested, gathered, strict=True):
        name = channel_name.value
        hits = [h for h in channel_hits if (h.doc_id, h.doc_version) in live]
        seen: set[str] = set()
        for hit in hits:
            docs.setdefault(hit.doc_id, hit)
            chunks.setdefault(hit.doc_id, {}).setdefault(hit.chunk_id, hit)
            if hit.doc_id not in seen:
                seen.add(hit.doc_id)
                evidence.setdefault(hit.doc_id, []).append(
                    MatchProvenance(channel=name, rank=len(seen), score=hit.score)
                )
            chunk_evidence.setdefault(hit.chunk_id, []).append(
                MatchProvenance(channel=name, rank=len(seen), score=hit.score)
            )

    scores = {
        doc_id: sum(1 / (RRF_CONSTANT + match.rank) for match in matches)
        for doc_id, matches in evidence.items()
    }
    ordered = sorted(docs, key=lambda doc_id: (-scores[doc_id], doc_id))
    results = []
    for rank, doc_id in enumerate(ordered[: req.top_k], 1):
        hit = docs[doc_id]
        body_chunks = [chunk for chunk in chunks[doc_id].values() if chunk.kind != "metadata"]
        body_chunks.sort(
            key=lambda chunk: (
                -sum(1 / (RRF_CONSTANT + m.rank) for m in chunk_evidence[chunk.chunk_id]),
                chunk.chunk_id,
            )
        )
        selected = [
            QueryChunk(
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                score=1 / chunk_rank,
                rank_in_doc=chunk_rank,
                matched_via=chunk_evidence[chunk.chunk_id],
                retriever_scores={m.channel: m.score for m in chunk_evidence[chunk.chunk_id]},
            )
            for chunk_rank, chunk in enumerate(body_chunks[:CHUNKS_PER_DOCUMENT], 1)
        ]
        results.append(
            QueryDocumentResult(
                canonical_id=doc_id,
                doc_id=doc_id,
                doc_version=hit.doc_version,
                source_system=hit.source_system,
                source_url=hit.source_url,
                title=hit.title,
                author_id=hit.author_id,
                created_at=hit.created_at,
                updated_at=hit.updated_at,
                score=1 / rank,
                rank=rank,
                chunks=selected,
                chunk_count=len(selected),
                matched_via=evidence[doc_id],
                retriever_scores={
                    **{m.channel: m.score for m in evidence[doc_id]},
                    "rrf_fused": scores[doc_id],
                },
            )
        )
    return DirectRetrieveResponse(
        query=req.query,
        results=results,
        total_candidates=len(docs),
        truncated=candidate_cap_reached or len(docs) > req.top_k,
        router_hit_cache=False,
        trace_id=str(uuid4()),
        applied_mode="direct",
        applied_sources=req.sources,
        applied_doc_types=req.doc_types,
        applied_scope=req.scope.model_dump(mode="json", exclude_none=True) if req.scope else None,
        lost_channels=sorted(lost),
        degraded=bool(lost),
        degraded_reason="retrieval_channel_unavailable" if lost else None,
        timing_ms={**timings, "total": round((time.perf_counter() - started) * 1000, 1)},
    )
