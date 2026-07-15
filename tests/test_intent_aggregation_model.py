"""Pydantic shape tests for IntentAggregation + new MatchProvenance.intent_idx."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from engine.shared.models import IntentAggregation, MatchProvenance, QueryResponse


def test_match_provenance_intent_idx_defaults_to_zero():
    mp = MatchProvenance(channel="vector", rank=1, score=0.9)
    assert mp.intent_idx == 0


def test_match_provenance_intent_idx_explicit():
    mp = MatchProvenance(channel="graph", rank=2, score=0.5, intent_idx=3)
    assert mp.intent_idx == 3


def test_intent_aggregation_count_payload():
    agg = IntentAggregation(intent_idx=1, operation="count", payload={"count": 42})
    assert agg.operation == "count"
    assert agg.payload == {"count": 42}


def test_intent_aggregation_group_by_payload():
    agg = IntentAggregation(
        intent_idx=2, operation="group_by",
        payload={"groups": [{"key": "mahit", "count": 12}]},
    )
    assert agg.operation == "group_by"


def test_intent_aggregation_rejects_invalid_operation():
    with pytest.raises(ValidationError):
        IntentAggregation(intent_idx=0, operation="list", payload={})


def test_query_response_aggregations_defaults_empty():
    resp = QueryResponse(query="x", results=[], total_candidates=0, router_hit_cache=False, trace_id="t-1")
    assert resp.aggregations == []


def test_query_response_aggregations_populated():
    resp = QueryResponse(
        query="x", results=[], total_candidates=0, router_hit_cache=False, trace_id="t-2",
        aggregations=[IntentAggregation(intent_idx=0, operation="count", payload={"count": 8})],
    )
    assert len(resp.aggregations) == 1
    assert resp.aggregations[0].payload["count"] == 8
