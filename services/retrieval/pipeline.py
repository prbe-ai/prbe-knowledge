"""Retrieval dispatcher — splits semantic search vs deterministic listing.

Flow:
    /retrieve → authenticate → resolve customer_id → route_query (Haiku)
              → resolve temporal + doc_type
              → branch on routed.mode:
                    "list"   → list_pipeline.run_list  (SQL window/aggregate)
                    "search" → search_pipeline.run_search  (vec + bm25 + graph + RRF)
                    None/unknown → fall back to search

Defensive recheck: even when Haiku says `mode=list`, we re-verify the gate
locally. If a topic entity slipped through (Haiku misclassification under
the new schema), we route to search instead. Trust-but-verify the LLM.

The pipeline is split into two phases so the streaming endpoint
(/query/stream) can emit progress events between them:
    run_router_phase  → resolves entities + temporal + mode (Haiku call)
    run_search_phase  → executes the chosen retrieval pipeline
The non-streaming `run_retrieval` runs both back-to-back unchanged.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime

from fastapi import HTTPException

from services.retrieval.doc_type_resolver import resolve_doc_type_token
from services.retrieval.list_pipeline import run_list
from services.retrieval.router import (
    TOPIC_ENTITY_TYPES,
    RouterOutput,
    route_query,
)
from services.retrieval.search_pipeline import run_search
from services.retrieval.temporal import resolve_temporal
from shared.logging import get_logger
from shared.models import (
    QueryRequest,
    QueryResponse,
    TemporalMode,
    TemporalSpec,
)

log = get_logger(__name__)


def _gate_verify_list(routed: RouterOutput, spec: TemporalSpec) -> bool:
    """Local recheck of the mode=list gate.

    Returns True iff the Haiku output actually satisfies the gate:
        (sort non-null OR temporal non-default) AND no topic entity.

    A defense-in-depth: if Haiku misclassifies (e.g. emits mode=list with a
    `feature` entity present), the dispatcher catches it and falls back to
    search. The router prompt should already prevent this; we double-check
    here so a bad Haiku snapshot can't silently route topic queries to SQL.
    """
    has_temporal = spec.mode != TemporalMode.LATEST or any(
        v is not None for v in (spec.since, spec.until, spec.as_of)
    )
    has_sort = routed.sort is not None
    if not (has_temporal or has_sort):
        return False
    return not any(e.entity_type in TOPIC_ENTITY_TYPES for e in routed.entities)


@dataclass(slots=True)
class RouterPhaseResult:
    """Everything resolved before retrieval runs.

    Streaming callers consume this between the `refining` and `searching`
    stages so the UI can render extracted entities while retrieval is
    still working. Mode dispatch (list vs search) is also baked in here
    so the streaming endpoint doesn't have to re-derive it.
    """

    routed: RouterOutput
    spec: TemporalSpec
    temporal_meta: dict[str, object]
    sort_meta: dict[str, object] | None
    extracted_entities: list[dict[str, object]]
    doc_types: list[str] | None
    trace_id: str
    timing: dict[str, float]
    dispatch_mode: str  # "list" or "search" — already gate-verified


async def run_router_phase(req: QueryRequest, customer_id: str) -> RouterPhaseResult:
    """Run only the Haiku router + post-route resolution. No retrieval."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="empty query")

    trace_id = req.trace_id or f"q-{int(datetime.now().timestamp() * 1000)}"
    timing: dict[str, float] = {}

    t_router = time.perf_counter()
    routed = await route_query(customer_id, req.query)
    timing["router_ms"] = (time.perf_counter() - t_router) * 1000

    # Resolve temporal: caller's explicit (non-default) TemporalSpec wins.
    # Otherwise use Haiku-extracted symbolic temporal. Otherwise fall back
    # to the default LATEST. extraction_failed surfaces unresolvable
    # event anchors ("since the auth refactor") — agent decides what to do.
    now = datetime.now(UTC)
    inferred_spec, extraction_error = resolve_temporal(routed.temporal, now=now)
    caller_explicit = (
        req.temporal.mode != TemporalMode.LATEST
        or req.temporal.since is not None
        or req.temporal.until is not None
        or req.temporal.as_of is not None
    )
    if caller_explicit:
        spec = req.temporal
        temporal_source = "caller"
    elif inferred_spec is not None:
        spec = inferred_spec
        temporal_source = "inferred"
    elif extraction_error is not None:
        spec = TemporalSpec()
        temporal_source = "extraction_failed"
    else:
        spec = TemporalSpec()
        temporal_source = "default"

    raw_phrase = (routed.temporal or {}).get("raw_phrase") if routed.temporal else None
    temporal_meta: dict[str, object] = {
        "mode": spec.mode.value,
        "source": temporal_source,
        "raw_phrase": raw_phrase,
        "error": extraction_error,
    }
    if spec.mode == TemporalMode.CHANGED_BETWEEN:
        temporal_meta["since"] = spec.since.isoformat() if spec.since else None
        temporal_meta["until"] = spec.until.isoformat() if spec.until else None
        temporal_meta["basis"] = spec.time_basis
    elif spec.mode == TemporalMode.AS_OF:
        temporal_meta["as_of"] = spec.as_of.isoformat() if spec.as_of else None

    sort_meta: dict[str, object] | None = None
    if routed.sort:
        sort_meta = {
            "field": routed.sort.get("field"),
            "direction": routed.sort.get("direction"),
            "trigger_phrase": routed.sort.get("trigger_phrase"),
            "source": "inferred",
        }

    extracted_entities = [
        {
            "entity_type": e.entity_type,
            "canonical_id": e.canonical_id,
            "display_name": e.display_name,
            "confidence": e.confidence,
        }
        for e in routed.entities
    ]

    # Caller-provided doc_types always win. Otherwise resolve Haiku's token.
    doc_types: list[str] | None
    if req.doc_types:
        doc_types = list(req.doc_types)
    else:
        doc_types = resolve_doc_type_token(routed.doc_type, sources=req.sources)

    routed_mode = (routed.mode or "search").lower()
    dispatch_mode = (
        "list" if routed_mode == "list" and _gate_verify_list(routed, spec) else "search"
    )

    return RouterPhaseResult(
        routed=routed,
        spec=spec,
        temporal_meta=temporal_meta,
        sort_meta=sort_meta,
        extracted_entities=extracted_entities,
        doc_types=doc_types,
        trace_id=trace_id,
        timing=timing,
        dispatch_mode=dispatch_mode,
    )


async def run_search_phase(
    req: QueryRequest, customer_id: str, phase: RouterPhaseResult
) -> QueryResponse:
    """Run the chosen retrieval pipeline (list or search) given a router result."""
    if phase.dispatch_mode == "list":
        log.info(
            "query.dispatch",
            extra={
                "trace_id": phase.trace_id,
                "mode": "list",
                "doc_type": phase.routed.doc_type,
                "operation": phase.routed.operation,
                "router_mode": phase.routed.mode,
                "query_len": len(req.query),
            },
        )
        return await run_list(
            req=req,
            customer_id=customer_id,
            routed=phase.routed,
            spec=phase.spec,
            temporal_meta=phase.temporal_meta,
            sort_meta=phase.sort_meta,
            extracted_entities=phase.extracted_entities,
            doc_types=phase.doc_types,
            trace_id=phase.trace_id,
            timing=phase.timing,
        )

    log.info(
        "query.dispatch",
        extra={
            "trace_id": phase.trace_id,
            "mode": "search",
            "doc_type": phase.routed.doc_type,
            "router_mode": phase.routed.mode,
            "query_len": len(req.query),
        },
    )
    return await run_search(
        req=req,
        customer_id=customer_id,
        routed=phase.routed,
        spec=phase.spec,
        temporal_meta=phase.temporal_meta,
        sort_meta=phase.sort_meta,
        extracted_entities=phase.extracted_entities,
        doc_types=phase.doc_types,
        trace_id=phase.trace_id,
        timing=phase.timing,
    )


async def run_retrieval(req: QueryRequest, customer_id: str) -> QueryResponse:
    """Run the full retrieval pipeline. The single entry point for the
    /retrieve endpoint and (via /query) the synthesis layer."""
    phase = await run_router_phase(req, customer_id)
    return await run_search_phase(req, customer_id, phase)
