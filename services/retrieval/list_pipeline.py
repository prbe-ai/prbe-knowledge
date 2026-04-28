"""Deterministic list pipeline — SQL window/aggregate against `documents`.

When the router classifies a query as `mode=list` (sort or temporal intent
present, no topic entity), retrieval bypasses the vector + BM25 + graph
fusion entirely and runs a parameterized SQL query.

Three operations:
  - list:     ORDER BY <field> <dir> LIMIT N           → chunks (one per doc)
  - count:    COUNT(*)                                  → aggregation={count: N}
  - group_by: <key>, COUNT(*) GROUP BY <key>            → aggregation={groups: [...]}

ACL filtering still runs (no-op until P4 flips ENFORCE_ACL on, but the path
is exercised so the Phase-1 flip doesn't surprise anyone). Embedding-based
dedup is skipped — SQL returns one chunk per distinct doc_id by construction,
so there are no near-duplicates to collapse.
"""

from __future__ import annotations

import time

from services.retrieval.acl import filter_by_acl
from services.retrieval.retrievers.sql import sql_count, sql_group_by, sql_list
from services.retrieval.router import RouterOutput
from shared.constants import SourceSystem
from shared.logging import get_logger
from shared.models import (
    QueryChunk,
    QueryRequest,
    QueryResponse,
    TemporalSpec,
)

log = get_logger(__name__)


def _pick_sort(routed: RouterOutput) -> tuple[str, str]:
    """Default to (updated_at, desc). Honor router-extracted sort when present."""
    if routed.sort:
        field = routed.sort.get("field") or "updated_at"
        direction = routed.sort.get("direction") or "desc"
        if field not in ("created_at", "updated_at"):
            field = "updated_at"
        if direction not in ("asc", "desc"):
            direction = "desc"
        return field, direction
    return "updated_at", "desc"


def _author_ids_from_entities(routed: RouterOutput) -> list[str] | None:
    """Pull person canonical_ids out of the extracted entities for the
    SQL `author_id = ANY(...)` filter. Confidence threshold matches what
    apply_entity_filter uses by default — entities below 0.7 are noise."""
    ids = [
        e.canonical_id
        for e in routed.entities
        if e.entity_type == "person" and e.confidence >= 0.7 and e.canonical_id
    ]
    return ids or None


async def run_list(
    req: QueryRequest,
    customer_id: str,
    routed: RouterOutput,
    spec: TemporalSpec,
    temporal_meta: dict[str, object],
    sort_meta: dict[str, object] | None,
    extracted_entities: list[dict[str, object]],
    doc_types: list[str] | None,
    trace_id: str,
    timing: dict[str, float],
) -> QueryResponse:
    sources = [s.value for s in req.sources] if req.sources else None
    author_ids = _author_ids_from_entities(routed)
    operation = (routed.operation or "list").lower()
    if operation not in ("list", "count", "group_by"):
        operation = "list"

    aggregation: dict[str, object] | None = None
    chunks: list[QueryChunk] = []
    total_candidates = 0

    t_sql = time.perf_counter()

    if operation == "count":
        n = await sql_count(
            customer_id,
            sources=sources,
            doc_types=doc_types,
            author_ids=author_ids,
            temporal=spec,
        )
        aggregation = {"count": n}
        total_candidates = n

    elif operation == "group_by":
        key = (routed.group_by_key or "source_system").lower()
        if key not in ("source_system", "doc_type", "author_id"):
            key = "source_system"
        groups = await sql_group_by(
            customer_id,
            key=key,  # type: ignore[arg-type]
            top_k=req.top_k,
            sources=sources,
            doc_types=doc_types,
            author_ids=author_ids,
            temporal=spec,
        )
        aggregation = {"key": key, "groups": groups}
        total_candidates = len(groups)

    else:  # operation == "list"
        sort_field, sort_dir = _pick_sort(routed)
        hits = await sql_list(
            customer_id,
            top_k=req.top_k,
            sources=sources,
            doc_types=doc_types,
            author_ids=author_ids,
            sort_field=sort_field,  # type: ignore[arg-type]
            sort_direction=sort_dir,  # type: ignore[arg-type]
            temporal=spec,
        )
        # ACL still runs for shape consistency (no-op until ENFORCE_ACL flips).
        t_acl = time.perf_counter()
        hits = await filter_by_acl(customer_id, req.requesting_user_id, hits)
        timing["acl_ms"] = (time.perf_counter() - t_acl) * 1000

        total_candidates = len(hits)
        chunks = [
            QueryChunk(
                chunk_id=h.chunk_id,
                doc_id=h.doc_id,
                doc_version=h.doc_version,
                source_system=SourceSystem(h.source_system),
                source_url=h.source_url,
                title=h.title,
                content=h.content,
                created_at=h.created_at,
                updated_at=h.updated_at,
                score=h.score,
                rank=i + 1,
                retriever_scores={"sql": h.score},
            )
            for i, h in enumerate(hits[: req.top_k])
        ]

    timing["sql_ms"] = (time.perf_counter() - t_sql) * 1000

    return QueryResponse(
        query=req.query,
        chunks=chunks,
        total_candidates=total_candidates,
        router_hit_cache=False,
        applied_temporal=temporal_meta,
        applied_sort=sort_meta,
        applied_entity_filter=None,
        applied_mode="list",
        applied_doc_types=doc_types,
        extracted_entities=extracted_entities,
        aggregation=aggregation,
        timing_ms=timing,
        trace_id=trace_id,
    )
