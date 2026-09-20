"""Bounded index lookup for interactive search, without the gatherer.

Both retrievers own their SQL, tenant RLS, live-version, visibility and scope
predicates. This adapter only combines their document ranks; it does not run
grounding, query expansion, graph traversal or a generative model.
"""

from __future__ import annotations

import asyncio
import time
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from engine.retrieval.retrievers.bm25 import BM25Hit, bm25_search
from engine.retrieval.retrievers.vector import VectorHit, vector_search
from engine.shared.constants import MAX_REQUEST_SOURCE_KEYS, SourceSystem
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
Hit = VectorHit | BM25Hit


class DirectRetrieveRequest(BaseModel):
    """Only filters that both direct retrievers enforce before ranking."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=20, ge=1, le=50)
    sources: list[SourceSystem] | None = None
    source_keys: list[str] | None = Field(default=None, max_length=MAX_REQUEST_SOURCE_KEYS)
    doc_types: list[str] | None = None
    scope: ScopeSpec | None = None

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

    async def channel(name: str, search) -> list[Hit]:
        before = time.perf_counter()
        try:
            async with asyncio.timeout(CHANNEL_TIMEOUT_SECONDS):
                return await search(customer_id, req.query, **filters)
        except Exception as exc:
            lost.append(name)
            log.warning("direct_search.channel_failed", channel=name, error=type(exc).__name__)
            return []
        finally:
            timings[name] = round((time.perf_counter() - before) * 1000, 1)

    vector, bm25 = await asyncio.gather(
        channel("vector", vector_search), channel("bm25", bm25_search)
    )
    candidate_cap_reached = len(vector) >= filters["top_k"] or len(bm25) >= filters["top_k"]
    try:
        live = await _live_docs(req, customer_id, list({h.doc_id for h in [*vector, *bm25]}))
    except Exception as exc:
        log.warning("direct_search.live_lookup_failed", error=type(exc).__name__)
        live = set()
        lost.append("live_documents")
    vector = [h for h in vector if (h.doc_id, h.doc_version) in live]
    bm25 = [h for h in bm25 if (h.doc_id, h.doc_version) in live]
    docs: dict[str, Hit] = {}
    evidence: dict[str, list[MatchProvenance]] = {}
    chunks: dict[str, dict[str, Hit]] = {}
    chunk_evidence: dict[str, list[MatchProvenance]] = {}
    for name, hits in (("vector", vector), ("bm25", bm25)):
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
            for chunk_rank, chunk in enumerate(body_chunks[:2], 1)
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
