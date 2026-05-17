"""Retrieval pipeline — gatherer-only architecture (Phase 2 cutover).

The gatherer agent IS the pipeline. This module is now a thin shim:

    /retrieve → run_retrieval → run_gatherer (Fireworks gpt-oss-120B)
                                    ↓
                              GathererOutput → QueryResponse

The streaming endpoint (`/query/stream`) still calls `run_router_phase`
+ `run_search_phase` separately so it can emit progress SSE events
between grounding and the agent loop. Those two are now compat shims:
`run_router_phase` runs the grounding step and returns a synthetic
single-intent `RouterPhaseResult`; `run_search_phase` runs the agent
loop and adapts to `QueryResponse`.

Deleted from the pre-cutover pipeline:
- `fuse_intent_results` (RRF fusion — no fusion in the gatherer model)
- `_run_one_intent` (per-intent dispatch — gatherer is single-loop)
- `_resolve_one_intent`, `_gate_verify_list`, `_timed_run` (mode-gating
  + per-intent timing wrappers, all subsumed by the agent)

Plan: docs/specs/agentic-search.md, section "Phase 2 — Build + ship".
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from fastapi import HTTPException

if TYPE_CHECKING:
    from starlette.requests import Request

from services.retrieval.agent.loop import run_gatherer
from services.retrieval.grounding import GroundingBundle, GroundingCandidate
from services.retrieval.router import (
    Intent,
    RouterOutput,
    _build_bundle_with_token_fallback,
    _fallback_intent,
)
from shared.constants import SEARCH_AGENT_INFERENCE_MODEL
from shared.logging import get_logger
from shared.models import (
    QueryRequest,
    QueryResponse,
    TemporalMode,
    TemporalSpec,
)

log = get_logger(__name__)


def _bundle_to_jsonable(b: GroundingBundle) -> dict:
    """Convert a GroundingBundle to a JSON-serializable dict.

    Used by the query_traces middleware to persist the grounding bundle
    into Postgres JSONB. Kept compatible with the pre-cutover shape so
    the middleware needs no changes.
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


@dataclass(slots=True)
class ResolvedIntent:
    """Compat shape for the streaming endpoint.

    Pre-cutover this was one entry per Haiku-emitted intent. Post-cutover
    the gatherer is single-intent at the harness level (it does its own
    multi-query splits internally via `parallel_multi_query`). The
    streaming endpoint always sees exactly one ResolvedIntent so its
    "entities" SSE event still renders.
    """

    intent: Intent
    spec: TemporalSpec
    sort_meta: dict[str, object] | None
    extracted_entities: list[dict[str, object]]
    doc_types: list[str] | None
    dispatch_mode: str  # always "search" post-cutover
    temporal_meta: dict[str, object]


@dataclass(slots=True)
class RouterPhaseResult:
    """Compat shape for the streaming endpoint."""

    routed: RouterOutput
    resolved_intents: list[ResolvedIntent]
    trace_id: str
    timing: dict[str, float]


async def run_router_phase(
    req: QueryRequest,
    customer_id: str,
    request: Request | None = None,
) -> RouterPhaseResult:
    """Grounding-only compat shim.

    Pre-cutover this called Haiku to extract intents. Post-cutover it
    runs `_build_bundle_with_token_fallback` to extract the grounded
    entity bag and wraps a synthetic single-intent payload so the
    streaming endpoint's `entities` SSE event has something to render.

    Telemetry semantics preserved: writes the bundle + intent count +
    router_model to `request.state.*` so the query_traces middleware
    persists matching rows pre/post cutover.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="empty query")

    trace_id = req.trace_id or f"q-{int(datetime.now().timestamp() * 1000)}"
    timing: dict[str, float] = {}

    t_grounding = time.perf_counter()
    try:
        bundle = await _build_bundle_with_token_fallback(customer_id, req.query)
    except Exception as exc:
        log.warning(
            "pipeline.grounding_failed",
            customer_id=customer_id,
            trace_id=trace_id,
            error=str(exc),
        )
        bundle = GroundingBundle()
    timing["grounding_ms"] = (time.perf_counter() - t_grounding) * 1000

    # One synthetic intent — captures the query text + grounded entities
    # the streaming endpoint surfaces to consumers. The agent doesn't
    # consume this shape; it reads the grounding bundle directly.
    fallback = _fallback_intent(req.query)
    routed = RouterOutput(
        intents=[fallback],
        grounding_bundle=bundle,
        router_raw={},
        cache_tokens=None,
        fallback_used=False,
    )

    extracted_entities: list[dict[str, object]] = [
        {
            "entity_type": c.entity_type,
            "canonical_id": c.canonical_id,
            "display_name": c.display_name,
            "confidence": 1.0,
        }
        for c in (list(bundle.candidates) + list(bundle.bare_id_matches))
    ]
    resolved = ResolvedIntent(
        intent=fallback,
        spec=TemporalSpec(),
        sort_meta=None,
        extracted_entities=extracted_entities,
        doc_types=None,
        dispatch_mode="search",
        temporal_meta={
            "mode": TemporalMode.LATEST.value,
            "source": "default",
            "raw_phrase": None,
            "error": None,
        },
    )

    if request is not None:
        try:
            request.state.grounding_bundle = _bundle_to_jsonable(bundle)
        except Exception:
            log.warning("pipeline.grounding_bundle_serialize_failed")
            request.state.grounding_bundle = None
        request.state.router_raw = {}
        request.state.intents_count = 1
        request.state.router_model = SEARCH_AGENT_INFERENCE_MODEL
        request.state.cache_tokens = None
        request.state.failure_recovered = False

    return RouterPhaseResult(
        routed=routed,
        resolved_intents=[resolved],
        trace_id=trace_id,
        timing=timing,
    )


async def run_search_phase(
    req: QueryRequest,
    customer_id: str,
    phase: RouterPhaseResult,  # noqa: ARG001 — kept for streaming-endpoint signature compat
    request: Request | None = None,
) -> QueryResponse:
    """Agent loop. Reads the query, runs the gatherer, returns a QueryResponse.

    Ignores `phase` other than the trace_id continuity (the agent
    re-runs grounding internally — they're cheap and keep the agent
    self-contained for testability).
    """
    return await run_gatherer(req, customer_id, request=request)


async def run_retrieval(
    req: QueryRequest,
    customer_id: str,
    request: Request | None = None,
) -> QueryResponse:
    """Run the full retrieval pipeline.

    Post-cutover this is `run_gatherer`. The two-phase split survives
    only for the streaming endpoint, which calls
    `run_router_phase` + `run_search_phase` separately to emit SSE
    progress events between grounding and the agent loop.
    """
    return await run_gatherer(req, customer_id, request=request)
