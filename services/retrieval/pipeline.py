"""Retrieval dispatcher — per-intent fan-out, RRF fusion, loud failure recovery.

Flow:
    /retrieve → authenticate → resolve customer_id → route_query (Haiku)
              → for each intent: resolve temporal + doc_type + gate
              → asyncio.gather all intents (isolated failures)
              → RRF-fuse surviving results
              → if all failed: loud log + fallback single-intent run

The pipeline is split into two phases so the streaming endpoint
(/query/stream) can emit progress events between them:
    run_router_phase  → resolves entities + temporal + mode per intent
    run_search_phase  → fan-out + fusion
The non-streaming `run_retrieval` runs both back-to-back unchanged.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from fastapi import HTTPException

if TYPE_CHECKING:
    from starlette.requests import Request

from services.retrieval.doc_type_resolver import resolve_doc_type_token
from services.retrieval.grounding import GroundingBundle, GroundingCandidate
from services.retrieval.list_pipeline import run_list
from services.retrieval.router import (
    TOPIC_ENTITY_TYPES,
    Intent,
    RouterOutput,
    route_query,
)
from services.retrieval.search_pipeline import run_search
from services.retrieval.temporal import resolve_temporal
from shared.constants import HAIKU_MODEL
from shared.logging import get_logger
from shared.models import (
    IntentAggregation,
    MatchProvenance,
    QueryRequest,
    QueryResponse,
    TemporalMode,
    TemporalSpec,
)

log = get_logger(__name__)

_RRF_K = 60


def _bundle_to_jsonable(b: GroundingBundle) -> dict:
    """Convert a GroundingBundle to a JSON-serializable dict.

    GroundingCandidate.last_seen_at is a datetime — isoformat it so
    json.dumps survives. Called in run_router_phase before stashing on
    request.state.grounding_bundle.
    """
    def _candidate(c: GroundingCandidate) -> dict:
        return {
            "entity_type": c.entity_type,
            "canonical_id": c.canonical_id,
            "display_name": c.display_name,
            "last_seen_at": c.last_seen_at.isoformat() if c.last_seen_at else None,
            "match_source": c.match_source,
        }

    return {
        "candidates": [_candidate(c) for c in b.candidates],
        "bare_id_matches": [_candidate(c) for c in b.bare_id_matches],
        "connected_sources": list(b.connected_sources),
        "timing_ms": b.timing_ms,
    }


def _gate_verify_list(intent: Intent, spec: TemporalSpec) -> bool:
    """Local recheck of the mode=list gate.

    Returns True iff the intent output actually satisfies the gate:
        (sort non-null OR temporal non-default) AND no topic entity.

    A defense-in-depth: if Haiku misclassifies (e.g. emits mode=list with a
    `feature` entity present), the dispatcher catches it and falls back to
    search. The router prompt should already prevent this; we double-check
    here so a bad Haiku snapshot can't silently route topic queries to SQL.
    """
    has_temporal = spec.mode != TemporalMode.LATEST or any(
        v is not None for v in (spec.since, spec.until, spec.as_of)
    )
    has_sort = intent.sort is not None
    if not (has_temporal or has_sort):
        return False
    return not any(e.entity_type in TOPIC_ENTITY_TYPES for e in intent.entities)


@dataclass(slots=True)
class ResolvedIntent:
    """Per-intent state resolved from the router output.

    Carries everything downstream retrieval needs — temporal spec,
    sort/entity/doc-type metadata, dispatch mode, and pre-computed
    temporal_meta dict for the streaming progress event.
    """

    intent: Intent
    spec: TemporalSpec
    sort_meta: dict[str, object] | None
    extracted_entities: list[dict[str, object]]
    doc_types: list[str] | None
    dispatch_mode: str  # "list" or "search" — already gate-verified
    temporal_meta: dict[str, object]


@dataclass(slots=True)
class RouterPhaseResult:
    """Everything resolved before retrieval runs.

    Streaming callers consume this between the `refining` and `searching`
    stages so the UI can render extracted entities while retrieval is
    still working. Per-intent state is in `resolved_intents`; the first
    entry is used for the streaming `entities` event until Task 6 emits
    the full union.
    """

    routed: RouterOutput
    resolved_intents: list[ResolvedIntent]
    trace_id: str
    timing: dict[str, float]


def _resolve_one_intent(
    req: QueryRequest,
    intent: Intent,
    *,
    now: datetime,
) -> ResolvedIntent:
    """Resolve a single intent to its full retrieval parameters."""
    inferred_spec, extraction_error = resolve_temporal(intent.temporal, now=now)
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

    raw_phrase = (intent.temporal or {}).get("raw_phrase") if intent.temporal else None
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
    if intent.sort:
        sort_meta = {
            "field": intent.sort.get("field"),
            "direction": intent.sort.get("direction"),
            "trigger_phrase": intent.sort.get("trigger_phrase"),
            "source": "inferred",
        }

    extracted_entities = [
        {
            "entity_type": e.entity_type,
            "canonical_id": e.canonical_id,
            "display_name": e.display_name,
            "confidence": e.confidence,
        }
        for e in intent.entities
    ]

    # Caller-provided doc_types always win. Otherwise resolve Haiku's token.
    doc_types: list[str] | None
    if req.doc_types:
        doc_types = list(req.doc_types)
    else:
        doc_types = resolve_doc_type_token(intent.doc_type, sources=req.sources)

    routed_mode = (intent.mode or "search").lower()
    dispatch_mode = (
        "list" if routed_mode == "list" and _gate_verify_list(intent, spec) else "search"
    )

    return ResolvedIntent(
        intent=intent,
        spec=spec,
        sort_meta=sort_meta,
        extracted_entities=extracted_entities,
        doc_types=doc_types,
        dispatch_mode=dispatch_mode,
        temporal_meta=temporal_meta,
    )


async def run_router_phase(
    req: QueryRequest,
    customer_id: str,
    request: Request | None = None,
) -> RouterPhaseResult:
    """Run only the Haiku router + post-route resolution. No retrieval.

    When `request` is provided (FastAPI Request), populates these state fields
    for the telemetry middleware:
        grounding_bundle, router_raw, intents_count, router_model,
        cache_tokens, failure_recovered (initial value from router failure).
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="empty query")

    trace_id = req.trace_id or f"q-{int(datetime.now().timestamp() * 1000)}"
    timing: dict[str, float] = {}

    t_router = time.perf_counter()
    routed = await route_query(customer_id, req.query)
    timing["router_ms"] = (time.perf_counter() - t_router) * 1000

    now = datetime.now(UTC)
    resolved_intents = [
        _resolve_one_intent(req, intent, now=now) for intent in routed.intents
    ]

    if request is not None:
        try:
            request.state.grounding_bundle = _bundle_to_jsonable(routed.grounding_bundle)
        except Exception:
            log.warning("pipeline.grounding_bundle_serialize_failed")
            request.state.grounding_bundle = None
        request.state.router_raw = routed.router_raw
        request.state.intents_count = len(routed.intents)
        request.state.router_model = HAIKU_MODEL
        request.state.cache_tokens = routed.cache_tokens
        # failure_recovered: True when the router emitted a fallback intent.
        # Reads RouterOutput.fallback_used directly — covers Haiku timeouts,
        # parse errors, empty intents[] payloads, and post-parse exceptions.
        request.state.failure_recovered = routed.fallback_used

    return RouterPhaseResult(
        routed=routed,
        resolved_intents=resolved_intents,
        trace_id=trace_id,
        timing=timing,
    )


async def _run_one_intent(
    req: QueryRequest,
    customer_id: str,
    resolved: ResolvedIntent,
    intent_idx: int,
    trace_id: str,
    timing: dict[str, float],
) -> QueryResponse:
    """Dispatch a single resolved intent to list or search pipeline."""
    if resolved.dispatch_mode == "list":
        log.info(
            "query.dispatch",
            trace_id=trace_id,
            intent_idx=intent_idx,
            mode="list",
            doc_type=resolved.intent.doc_type,
            operation=resolved.intent.operation,
            query_len=len(req.query),
        )
        return await run_list(
            req=req,
            customer_id=customer_id,
            intent=resolved.intent,
            spec=resolved.spec,
            temporal_meta=resolved.temporal_meta,
            sort_meta=resolved.sort_meta,
            extracted_entities=resolved.extracted_entities,
            doc_types=resolved.doc_types,
            trace_id=trace_id,
            timing=timing,
            intent_idx=intent_idx,
        )

    log.info(
        "query.dispatch",
        trace_id=trace_id,
        intent_idx=intent_idx,
        mode="search",
        doc_type=resolved.intent.doc_type,
        query_len=len(req.query),
    )
    return await run_search(
        req=req,
        customer_id=customer_id,
        intent=resolved.intent,
        spec=resolved.spec,
        temporal_meta=resolved.temporal_meta,
        sort_meta=resolved.sort_meta,
        extracted_entities=resolved.extracted_entities,
        doc_types=resolved.doc_types,
        trace_id=trace_id,
        timing=timing,
        intent_idx=intent_idx,
    )


def fuse_intent_results(
    per_intent: list[QueryResponse],
    *,
    top_k: int,
) -> QueryResponse:
    """RRF-fuse results across intents.

    - Deduplicates by `canonical_id`, unions `matched_via` provenance.
    - New-shape per-intent `aggregations: list[IntentAggregation]` are extended.
    - `total_candidates` is summed.
    - Returns the first surviving response's metadata fields (applied_temporal,
      applied_sort, etc.) for backwards compatibility. Task 6 will emit
      the full union.

    Known gap: legacy `aggregation: dict | None` from list_pipeline is
    inherited from `per_intent[0]` only. Multi-intent fan-outs where a
    non-first intent emits an aggregation will lose that payload unless
    list_pipeline is updated to write into the new `aggregations` slot.
    Tracked for Task 6 telemetry pass.
    """
    if not per_intent:
        raise ValueError("fuse_intent_results called with empty list")

    if len(per_intent) == 1:
        r = per_intent[0]
        return r.model_copy(update={"results": r.results[:top_k]})

    # RRF across intents: score = sum of 1/(K + rank_i) per intent that
    # returned this doc. Docs appearing in multiple intents bubble up.
    rrf_scores: dict[str, float] = {}
    # Map canonical_id → list of QueryResult across intents (pick first for body)
    first_result: dict[str, object] = {}
    # Union matched_via: canonical_id → list[MatchProvenance]
    provenance_union: dict[str, list[MatchProvenance]] = {}

    for resp in per_intent:
        for result in resp.results:
            cid = result.canonical_id
            rank = result.rank  # 1-indexed
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (_RRF_K + rank)
            if cid not in first_result:
                first_result[cid] = result
            if cid not in provenance_union:
                provenance_union[cid] = []
            provenance_union[cid].extend(result.matched_via)

    # Sort by descending RRF score, then canonical_id for determinism.
    ranked = sorted(rrf_scores.keys(), key=lambda c: (-rrf_scores[c], c))
    ranked = ranked[:top_k]

    # Build final results — clone each result with unified provenance + rank.
    fused_results = []
    for new_rank, cid in enumerate(ranked, start=1):
        base = first_result[cid]
        fused = base.model_copy(
            update={
                "rank": new_rank,
                "matched_via": provenance_union[cid],
            }
        )
        fused_results.append(fused)

    # Aggregate: collect from all intents (they're disjoint by intent_idx).
    all_aggs: list[IntentAggregation] = []
    for resp in per_intent:
        all_aggs.extend(resp.aggregations)

    total_candidates = sum(r.total_candidates for r in per_intent)
    base_resp = per_intent[0]

    return base_resp.model_copy(
        update={
            "results": fused_results,
            "total_candidates": total_candidates,
            "aggregations": all_aggs,
        }
    )


async def run_search_phase(
    req: QueryRequest,
    customer_id: str,
    phase: RouterPhaseResult,
    request: Request | None = None,
) -> QueryResponse:
    """Fan out to all resolved intents, fuse surviving results.

    When `request` is provided (FastAPI Request), populates:
        intent_dispatch (list of per-intent timing/result dicts)
        failure_recovered (True when all intents fail, OR-ed with router failure)
    """
    top_k = req.top_k

    # Wrap each _run_one_intent call to capture per-intent timing.
    # Each intent gets its OWN copy of phase.timing — search_pipeline /
    # list_pipeline mutate `timing` in place, and a shared dict across
    # concurrent intents under asyncio.gather corrupts per-stage timings
    # (last writer wins). The copy isolates per-intent timing breakdowns;
    # high-level latency is captured separately in intent_dispatch.
    async def _timed_run(
        idx: int, resolved: ResolvedIntent
    ) -> tuple[int, QueryResponse | BaseException, float, BaseException | None]:
        t0 = time.perf_counter()
        try:
            resp = await _run_one_intent(
                req,
                customer_id,
                resolved,
                intent_idx=idx,
                trace_id=phase.trace_id,
                timing=dict(phase.timing),
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return idx, resp, elapsed_ms, None
        except asyncio.CancelledError:
            # Re-raise so the outer gather correctly propagates client
            # disconnects. Swallowing CancelledError would let subtasks
            # continue holding DB connections after the request aborted.
            raise
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return idx, exc, elapsed_ms, exc

    timed_coros = [
        _timed_run(i, resolved) for i, resolved in enumerate(phase.resolved_intents)
    ]
    raw_results: list[tuple[int, QueryResponse | BaseException, float, BaseException | None]] = (
        await asyncio.gather(*timed_coros)
    )

    surviving: list[QueryResponse] = []
    for idx, resp, _elapsed_ms, _exc in raw_results:
        if isinstance(resp, BaseException):
            log.warning(
                "pipeline.intent_failed",
                trace_id=phase.trace_id,
                customer_id=customer_id,
                intent_idx=idx,
                error=str(resp),
                error_type=type(resp).__name__,
            )
        else:
            surviving.append(resp)

    all_failed = not surviving

    if all_failed:
        # Every intent failed — loud log, then one last chance with the raw query.
        log.error(
            "pipeline.full_failure_recovered",
            trace_id=phase.trace_id,
            customer_id=customer_id,
            query=req.query,
        )

    # Build intent_dispatch telemetry from timed results (before fallback).
    intent_dispatch = [
        {
            "intent_idx": idx,
            "mode": phase.resolved_intents[idx].dispatch_mode,
            "latency_ms": round(elapsed_ms, 3),
            "result_count": len(resp.results) if not isinstance(resp, BaseException) else None,
            "error_class": type(resp).__name__ if isinstance(resp, BaseException) else None,
        }
        for idx, resp, elapsed_ms, _exc in raw_results
    ]

    if request is not None:
        request.state.intent_dispatch = intent_dispatch
        # OR-in the all-intents-fail signal (may already be True from router failure)
        if all_failed:
            request.state.failure_recovered = True

    if all_failed:
        fallback_intent = Intent(query_text=req.query, mode="search", confidence=0.0)
        fallback_resolved = ResolvedIntent(
            intent=fallback_intent,
            spec=TemporalSpec(),
            sort_meta=None,
            extracted_entities=[],
            doc_types=None,
            dispatch_mode="search",
            temporal_meta={
                "mode": TemporalMode.LATEST.value,
                "source": "default",
                "raw_phrase": None,
                "error": None,
            },
        )
        try:
            fallback_resp = await _run_one_intent(
                req,
                customer_id,
                fallback_resolved,
                intent_idx=0,
                trace_id=phase.trace_id,
                timing=phase.timing,
            )
            surviving = [fallback_resp]
        except Exception as exc:
            # Both the initial fan-out AND the fallback failed. Surface
            # an unambiguous flag so alerts keyed on `full_failure` page
            # someone — `failure_recovered=True` alone would suggest the
            # fallback succeeded, suppressing the alert.
            if request is not None:
                request.state.full_failure = True
            log.error(
                "pipeline.full_failure_unrecoverable",
                trace_id=phase.trace_id,
                customer_id=customer_id,
                query=req.query,
                error=str(exc),
            )
            raise HTTPException(status_code=500, detail="retrieval unavailable") from exc

    return fuse_intent_results(surviving, top_k=top_k)


async def run_retrieval(
    req: QueryRequest,
    customer_id: str,
    request: Request | None = None,
) -> QueryResponse:
    """Run the full retrieval pipeline. The single entry point for the
    /retrieve endpoint and (via /query) the synthesis layer."""
    phase = await run_router_phase(req, customer_id, request=request)
    return await run_search_phase(req, customer_id, phase, request=request)
