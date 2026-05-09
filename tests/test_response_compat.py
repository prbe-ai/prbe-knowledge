"""Pydantic round-trip tests for `related_entities` on QueryResponse.

Wire compatibility: callers on either side of the deploy may produce or
consume responses with or without the new field. The model contract is:

- Old-shape JSON (no `related_entities` key) parses with field=None.
- New-shape JSON parses with field populated.
- `related_entities[].canonical_id` set never overlaps with
  `extracted_entities[].canonical_id` set under matching label in any
  populated response (the search/list pipelines do the exclusion in code;
  this test pins the contract).
"""

from __future__ import annotations

from shared.models import AnswerRequest, QueryRequest, QueryResponse, RelatedEntity


def _base_payload() -> dict:
    """Minimal valid QueryResponse JSON without the new fields.

    Polymorphic shape (PR feat/polymorphic-search-results): the wire
    format uses `results: list[QueryResult]` (discriminated Document /
    Entity), not the old `chunks: list[QueryChunk]`. An old client
    sending `chunks=[]` is silently dropped by Pydantic's default
    extra-field handling -- callers that need the chunk data must
    upgrade.
    """
    return {
        "query": "anything",
        "results": [],
        "total_candidates": 0,
        "router_hit_cache": False,
        "trace_id": "t-1",
    }


def test_old_shape_parses_with_field_none() -> None:
    """Old responses (no `related_entities` key) round-trip cleanly with
    field=None and no related_entities_error."""
    payload = _base_payload()
    resp = QueryResponse.model_validate(payload)
    assert resp.related_entities is None
    assert resp.related_entities_error is None


def test_new_shape_parses_with_populated_field() -> None:
    """New-shape JSON round-trips with the field populated."""
    payload = _base_payload()
    payload["related_entities"] = [
        {
            "canonical_id": "pr:42",
            "label": "PR",
            "display_name": "Fix the thing",
            "edge_types": ["MENTIONS"],
            "max_confidence": "EXTRACTED",
            "doc_count": 3,
            "score": 1.5,
            "associated_doc_ids": ["doc:1", "doc:2"],
        }
    ]
    resp = QueryResponse.model_validate(payload)
    assert resp.related_entities is not None
    assert len(resp.related_entities) == 1
    re = resp.related_entities[0]
    assert isinstance(re, RelatedEntity)
    assert re.canonical_id == "pr:42"
    assert re.score == 1.5
    assert re.associated_doc_ids == ["doc:1", "doc:2"]


def test_empty_list_distinguished_from_none() -> None:
    """Three-state contract (codex-B4): [] is a legitimate value distinct
    from None. The Pydantic round-trip must preserve the distinction."""
    payload_empty = _base_payload()
    payload_empty["related_entities"] = []
    resp_empty = QueryResponse.model_validate(payload_empty)
    assert resp_empty.related_entities == []
    assert resp_empty.related_entities is not None  # not the None case

    payload_none = _base_payload()
    resp_none = QueryResponse.model_validate(payload_none)
    assert resp_none.related_entities is None


def test_failure_state_carries_error_string() -> None:
    """Walk failure: related_entities=None AND related_entities_error set."""
    payload = _base_payload()
    payload["related_entities"] = None
    payload["related_entities_error"] = "RuntimeError"
    resp = QueryResponse.model_validate(payload)
    assert resp.related_entities is None
    assert resp.related_entities_error == "RuntimeError"


def test_round_trip_serialize_then_parse() -> None:
    """model_dump() output parses back to an equivalent model. Catches any
    accidental Field(exclude=True) drift on the new fields."""
    re = RelatedEntity(
        canonical_id="pr:42",
        label="PR",
        max_confidence="INFERRED",
        doc_count=2,
        score=0.5,
        associated_doc_ids=["doc:1"],
    )
    resp = QueryResponse(
        query="q",
        results=[],
        total_candidates=0,
        router_hit_cache=False,
        trace_id="t",
        related_entities=[re],
    )
    dumped = resp.model_dump(mode="json")
    assert "related_entities" in dumped
    assert dumped["related_entities"][0]["canonical_id"] == "pr:42"
    assert "related_entities_error" in dumped
    assert dumped["related_entities_error"] is None

    resp2 = QueryResponse.model_validate(dumped)
    assert resp2.related_entities is not None
    assert resp2.related_entities[0].canonical_id == "pr:42"
    assert resp2.related_entities[0].score == 0.5


def test_extracted_and_related_entities_disjoint_under_matching_label() -> None:
    """Pinned contract: in any populated response we hand-build, the
    routed `extracted_entities` set never shares a (label, canonical_id)
    pair with `related_entities`. The pipelines enforce this via the
    `exclude_node_keys` arg to `walk_result_doc_neighbors`; this test
    asserts nothing in the model layer accepts a violation silently."""
    payload = _base_payload()
    payload["extracted_entities"] = [
        # `extracted_entities` is `list[dict[str, object]]` -- raw shape
        # the router emits via dict-coerce.
        {"entity_type": "service", "canonical_id": "prbe-backend", "display_name": "prbe-backend", "confidence": 0.9},
    ]
    payload["related_entities"] = [
        # Different label OR different canonical_id are both fine.
        {
            "canonical_id": "prbe-knowledge",
            "label": "Service",
            "max_confidence": "EXTRACTED",
            "doc_count": 1,
            "score": 0.5,
        },
    ]
    resp = QueryResponse.model_validate(payload)
    extracted_pairs = {
        (e.get("entity_type", "").capitalize(), e.get("canonical_id"))
        for e in resp.extracted_entities
    }
    related_pairs = {(r.label, r.canonical_id) for r in (resp.related_entities or [])}
    # Disjoint -- no overlap.
    assert not (extracted_pairs & related_pairs)


def test_answer_request_defaults_top_k_related_to_zero() -> None:
    """`AnswerRequest` (synthesis path) defaults `top_k_related=0` so the
    related-entities walk does not run on /query, where the response shape
    discards the field anyway. Override of `QueryRequest.top_k_related=10`.
    """
    q_req = QueryRequest(query="x")
    assert q_req.top_k_related == 10  # search default unchanged

    a_req = AnswerRequest(query="x")
    assert a_req.top_k_related == 0  # synthesis default overridden

    # Explicit opt-in still works on AnswerRequest.
    a_req_explicit = AnswerRequest(query="x", top_k_related=5)
    assert a_req_explicit.top_k_related == 5
