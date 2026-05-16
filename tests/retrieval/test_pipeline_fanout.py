"""pipeline.py — per-intent resolution + fan-out + fusion."""

from __future__ import annotations

from datetime import UTC
from datetime import datetime as _dt
from unittest.mock import AsyncMock, patch

import pytest

from services.retrieval.grounding import GroundingBundle
from services.retrieval.pipeline import (
    fuse_intent_results,
    run_retrieval,
    run_router_phase,
)
from services.retrieval.router import Intent, RouterEntity, RouterOutput
from shared.models import (
    IntentAggregation,
    MatchProvenance,
    QueryDocumentResult,
    QueryRequest,
    QueryResponse,
)


def _mk_router_output(intents):
    return RouterOutput(intents=intents, grounding_bundle=GroundingBundle(), router_raw={})


def _mk_doc(canonical_id: str, rank: int, intent_idx: int, score: float = 1.0):
    # NOTE: node_type literal is "Document" (capital D); see shared/models.py.
    return QueryDocumentResult(
        canonical_id=canonical_id,
        doc_id=canonical_id,
        doc_version=1,
        score=score,
        rank=rank,
        matched_via=[
            MatchProvenance(channel="vector", rank=rank, score=score, intent_idx=intent_idx)
        ],
        chunks=[],
        source_system="github",
        source_url=f"https://example.com/{canonical_id}",
        created_at=_dt(2026, 1, 1, tzinfo=UTC),
        updated_at=_dt(2026, 1, 1, tzinfo=UTC),
    )


def _mk_response(results, agg=None):
    return QueryResponse(
        query="x",
        results=results,
        total_candidates=len(results),
        router_hit_cache=False,
        aggregations=agg or [],
        trace_id="t-test",
    )


# ---- Resolution ---------------------------------------------------------


@pytest.mark.integration
async def test_run_router_phase_single_intent(seeded_customer, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from shared.config import get_settings as _gs

    _gs.cache_clear()

    fake = _mk_router_output(
        [
            Intent(
                query_text="show recent commits",
                mode="list",
                confidence=0.9,
                sort={"field": "updated_at", "direction": "desc"},
            )
        ]
    )
    with patch(
        "services.retrieval.pipeline.route_query", new=AsyncMock(return_value=fake)
    ):
        phase = await run_router_phase(
            QueryRequest(query="show recent commits"),
            seeded_customer.customer_id,
        )

    assert len(phase.resolved_intents) == 1
    assert phase.resolved_intents[0].dispatch_mode == "list"


@pytest.mark.integration
async def test_run_router_phase_per_intent_gate(seeded_customer, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from shared.config import get_settings as _gs

    _gs.cache_clear()

    fake = _mk_router_output(
        [
            Intent(
                query_text="commits last week",
                mode="list",
                confidence=0.9,
                sort={"field": "updated_at", "direction": "desc"},
                temporal={"since": {"kind": "rel", "offset_days": -7}, "basis": "source"},
            ),
            Intent(
                query_text="auth refactor status",
                mode="list",
                confidence=0.7,
                entities=[
                    RouterEntity(
                        entity_type="feature",
                        canonical_id="auth-refactor",
                        display_name="auth refactor",
                        confidence=0.9,
                    )
                ],
                sort={"field": "updated_at", "direction": "desc"},
            ),
        ]
    )
    with patch(
        "services.retrieval.pipeline.route_query", new=AsyncMock(return_value=fake)
    ):
        phase = await run_router_phase(
            QueryRequest(query="commits last week and auth refactor status"),
            seeded_customer.customer_id,
        )

    assert phase.resolved_intents[0].dispatch_mode == "list"
    assert phase.resolved_intents[1].dispatch_mode == "search"


# ---- Fusion ------------------------------------------------------------


def test_fuse_single_intent_passthrough():
    r = _mk_response([_mk_doc("A", 1, 0), _mk_doc("B", 2, 0), _mk_doc("C", 3, 0)])
    fused = fuse_intent_results([r], top_k=10)
    assert [d.canonical_id for d in fused.results] == ["A", "B", "C"]


def test_fuse_two_intents_overlap_bubbles_to_top():
    r1 = _mk_response([_mk_doc("A", 1, 0), _mk_doc("B", 2, 0)])
    r2 = _mk_response([_mk_doc("B", 1, 1), _mk_doc("C", 2, 1)])
    fused = fuse_intent_results([r1, r2], top_k=10)
    assert fused.results[0].canonical_id == "B"


def test_fuse_appends_aggregations():
    r1 = _mk_response([_mk_doc("A", 1, 0)])
    r2 = _mk_response(
        [], agg=[IntentAggregation(intent_idx=1, operation="count", payload={"count": 8})]
    )
    fused = fuse_intent_results([r1, r2], top_k=10)
    assert [d.canonical_id for d in fused.results] == ["A"]
    assert len(fused.aggregations) == 1
    assert fused.aggregations[0].payload == {"count": 8}


def test_fuse_provenance_unions_per_doc():
    r1 = _mk_response([_mk_doc("B", 1, 0)])
    r2 = _mk_response([_mk_doc("B", 1, 1)])
    fused = fuse_intent_results([r1, r2], top_k=10)
    intent_idxs = {p.intent_idx for p in fused.results[0].matched_via}
    assert intent_idxs == {0, 1}


def test_fuse_respects_top_k():
    r = _mk_response([_mk_doc(f"D{i}", i + 1, 0) for i in range(10)])
    fused = fuse_intent_results([r], top_k=3)
    assert len(fused.results) == 3


# ---- Fan-out + failure handling ----------------------------------------


@pytest.mark.integration
async def test_run_search_phase_per_intent_failure_isolated(
    seeded_customer, monkeypatch
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from shared.config import get_settings as _gs

    _gs.cache_clear()

    fake = _mk_router_output(
        [
            Intent(query_text="boom", mode="search", confidence=0.9),
            Intent(query_text="works", mode="search", confidence=0.9),
        ]
    )

    async def maybe_boom(*args, **kwargs):
        intent = kwargs.get("intent")
        intent_idx = kwargs.get("intent_idx", 0)
        if intent.query_text == "boom":
            raise RuntimeError("simulated retriever failure")
        return _mk_response([_mk_doc("survivor", 1, intent_idx)])

    # structlog bypasses pytest's caplog by default — patch log.warning
    # directly to assert on the per-intent-failure event name.
    import services.retrieval.pipeline as pipeline_mod
    with (
        patch(
            "services.retrieval.pipeline.route_query", new=AsyncMock(return_value=fake)
        ),
        patch("services.retrieval.pipeline.run_search", side_effect=maybe_boom),
        patch.object(pipeline_mod.log, "warning") as mock_warning,
    ):
        resp = await run_retrieval(QueryRequest(query="x"), seeded_customer.customer_id)
    assert any(r.canonical_id == "survivor" for r in resp.results)
    # Lock the event name so alert rules keyed on it stay wired up.
    assert any(
        call.args and call.args[0] == "pipeline.intent_failed"
        for call in mock_warning.call_args_list
    )


@pytest.mark.integration
async def test_run_search_phase_full_failure_falls_back_loudly(
    seeded_customer, monkeypatch
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from shared.config import get_settings as _gs

    _gs.cache_clear()

    fake = _mk_router_output(
        [
            Intent(query_text="boom1", mode="search", confidence=0.9),
            Intent(query_text="boom2", mode="search", confidence=0.9),
        ]
    )

    boom_count = {"n": 0}

    async def maybe_boom(*args, **kwargs):
        boom_count["n"] += 1
        if boom_count["n"] <= 2:
            raise RuntimeError("simulated")
        return _mk_response([_mk_doc("fallback", 1, 0)])

    # structlog bypasses pytest's caplog by default — patch log.error
    # directly to assert on the full-failure event name.
    import services.retrieval.pipeline as pipeline_mod
    with (
        patch(
            "services.retrieval.pipeline.route_query", new=AsyncMock(return_value=fake)
        ),
        patch("services.retrieval.pipeline.run_search", side_effect=maybe_boom),
        patch.object(pipeline_mod.log, "error") as mock_error,
    ):
        resp = await run_retrieval(QueryRequest(query="raw query"), seeded_customer.customer_id)

    # Verify the corrected event name: pipeline.full_failure_recovered
    assert any(
        call.args and call.args[0] == "pipeline.full_failure_recovered"
        for call in mock_error.call_args_list
    )
    assert any(r.canonical_id == "fallback" for r in resp.results)


# ---- request.state telemetry derivation --------------------------------

class _FakeState:
    """Stand-in for FastAPI Request.state — accepts any attribute assignment."""


class _FakeRequest:
    """Minimal stand-in for FastAPI Request — only exposes `.state`."""

    def __init__(self) -> None:
        self.state = _FakeState()


@pytest.mark.integration
async def test_run_retrieval_sets_failure_recovered_on_router_failure(
    seeded_customer, monkeypatch
):
    """When route_query returns router_raw == {} (Haiku fallback path),
    run_router_phase must set request.state.failure_recovered = True so the
    middleware writes it to the trace row."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from shared.config import get_settings as _gs

    _gs.cache_clear()

    fake = RouterOutput(
        intents=[Intent(query_text="anything", mode="search", confidence=0.0)],
        grounding_bundle=GroundingBundle(),
        router_raw={},
        fallback_used=True,  # pipeline reads this for failure_recovered telemetry
    )

    async def stub_search(*args, **kwargs):
        return _mk_response([_mk_doc("ok", 1, 0)])

    request = _FakeRequest()
    with (
        patch("services.retrieval.pipeline.route_query", new=AsyncMock(return_value=fake)),
        patch("services.retrieval.pipeline.run_search", side_effect=stub_search),
    ):
        await run_retrieval(
            QueryRequest(query="anything"), seeded_customer.customer_id, request=request
        )

    assert request.state.failure_recovered is True
    assert request.state.intents_count == 1
    assert request.state.router_model is not None


@pytest.mark.integration
async def test_run_retrieval_sets_failure_recovered_on_all_intents_fail(
    seeded_customer, monkeypatch
):
    """When every intent raises in the gather, run_search_phase ORs
    request.state.failure_recovered to True even if the router itself succeeded."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from shared.config import get_settings as _gs

    _gs.cache_clear()

    fake = _mk_router_output(
        [
            Intent(query_text="boom1", mode="search", confidence=0.9),
            Intent(query_text="boom2", mode="search", confidence=0.9),
        ]
    )

    boom_count = {"n": 0}

    async def maybe_boom(*args, **kwargs):
        boom_count["n"] += 1
        if boom_count["n"] <= 2:
            raise RuntimeError("simulated")
        return _mk_response([_mk_doc("fallback", 1, 0)])

    request = _FakeRequest()
    with (
        patch("services.retrieval.pipeline.route_query", new=AsyncMock(return_value=fake)),
        patch("services.retrieval.pipeline.run_search", side_effect=maybe_boom),
    ):
        await run_retrieval(
            QueryRequest(query="boom"), seeded_customer.customer_id, request=request
        )

    assert request.state.failure_recovered is True
    assert request.state.intent_dispatch is not None
    assert len(request.state.intent_dispatch) == 2
    assert all(d["error_class"] == "RuntimeError" for d in request.state.intent_dispatch)
