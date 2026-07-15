"""Pydantic round-trip tests for the polymorphic QueryResult discriminated
union (PR feat/polymorphic-search-results).

The wire contract:
- `QueryResponse.results: list[QueryResult]` is a discriminated union.
- Each entry carries `node_type: "Document"` or `node_type: "Entity"`.
- `model_validate` routes to the right subclass; `model_dump(mode="json")`
  preserves the discriminator field so a round-trip is lossless.
- Mixing both variants in one list is supported (the search pipeline
  emits Documents and Entities side-by-side in the same response).
"""

from __future__ import annotations

from datetime import UTC, datetime

from engine.shared.constants import SourceSystem
from engine.shared.models import (
    GraphEvidence,
    MatchProvenance,
    QueryChunk,
    QueryDocumentResult,
    QueryEntityResult,
    QueryResponse,
)


def _now() -> datetime:
    return datetime(2026, 4, 28, 12, 0, tzinfo=UTC)


def _doc_payload(doc_id: str = "github:foo/bar:pr:1") -> dict:
    return {
        "node_type": "Document",
        "canonical_id": doc_id,
        "doc_id": doc_id,
        "doc_version": 1,
        "source_system": "github",
        "source_url": f"https://example/{doc_id}",
        "title": "example",
        "author_id": "alice",
        "created_at": _now().isoformat(),
        "updated_at": _now().isoformat(),
        "score": 0.9,
        "rank": 1,
        "matched_via": [
            {"channel": "vector", "rank": 1, "score": 0.9},
        ],
        "chunks": [
            {
                "chunk_id": "c0",
                "content": "hello",
                "score": 0.9,
                "rank_in_doc": 1,
                "retriever_scores": {"vector": 0.9},
                "graph_evidence": [],
            }
        ],
        "chunk_count": 1,
        "retriever_scores": {"vector": 0.9},
    }


def _entity_payload(canonical_id: str = "prbe-backend") -> dict:
    return {
        "node_type": "Entity",
        "canonical_id": canonical_id,
        "label": "Service",
        "display_name": "prbe-backend",
        "score": 1.5,
        "rank": 2,
        "properties": {"name": "prbe-backend", "team": "platform"},
        "attached_doc_ids": ["doc:1", "doc:2"],
        "edge_types": ["MENTIONS", "OWNS"],
        "doc_count": 7,
        "matched_via": [
            {"channel": "graph", "rank": 1, "score": 0.9},
        ],
    }


def _base_payload(results: list[dict]) -> dict:
    return {
        "query": "anything",
        "results": results,
        "total_candidates": len(results),
        "router_hit_cache": False,
        "trace_id": "t-1",
    }


# ---------------------------------------------------------------------------
# Discriminated union routing
# ---------------------------------------------------------------------------


def test_document_node_type_parses_to_document_result() -> None:
    payload = _base_payload([_doc_payload()])
    resp = QueryResponse.model_validate(payload)
    assert len(resp.results) == 1
    first = resp.results[0]
    assert isinstance(first, QueryDocumentResult)
    assert first.doc_id == "github:foo/bar:pr:1"
    assert first.source_system == SourceSystem.GITHUB
    assert first.chunk_count == 1
    assert first.chunks[0].chunk_id == "c0"


def test_entity_node_type_parses_to_entity_result() -> None:
    payload = _base_payload([_entity_payload()])
    resp = QueryResponse.model_validate(payload)
    assert len(resp.results) == 1
    first = resp.results[0]
    assert isinstance(first, QueryEntityResult)
    assert first.label == "Service"
    assert first.display_name == "prbe-backend"
    assert first.doc_count == 7
    assert first.attached_doc_ids == ["doc:1", "doc:2"]


def test_mixed_results_list_supports_both_variants() -> None:
    """The search pipeline emits Documents AND Entities in the same list;
    the discriminator routes each to the right subclass."""
    payload = _base_payload([_doc_payload(), _entity_payload()])
    resp = QueryResponse.model_validate(payload)
    assert len(resp.results) == 2
    assert isinstance(resp.results[0], QueryDocumentResult)
    assert isinstance(resp.results[1], QueryEntityResult)


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_round_trip_preserves_discriminator() -> None:
    """model_dump(mode='json') -> model_validate must preserve `node_type`."""
    payload = _base_payload([_doc_payload(), _entity_payload()])
    resp = QueryResponse.model_validate(payload)
    dumped = resp.model_dump(mode="json")
    assert dumped["results"][0]["node_type"] == "Document"
    assert dumped["results"][1]["node_type"] == "Entity"

    reparsed = QueryResponse.model_validate(dumped)
    assert isinstance(reparsed.results[0], QueryDocumentResult)
    assert isinstance(reparsed.results[1], QueryEntityResult)
    # Field-level equality on a representative subset.
    assert reparsed.results[0].doc_id == resp.results[0].canonical_id
    assert reparsed.results[1].canonical_id == resp.results[1].canonical_id


def test_round_trip_preserves_match_provenance_inferred_edge() -> None:
    """An inferred-edge MatchProvenance carries 5 channel-specific fields
    (anchor_doc_id, edge_type, confidence, why) that must survive round trip.
    """
    doc_payload = _doc_payload()
    doc_payload["matched_via"] = [
        {
            "channel": "inferred_edge",
            "rank": 1,
            "score": 0.25,
            "anchor_doc_id": "primary:doc",
            "edge_type": "DISCUSSES",
            "confidence": "INFERRED",
            "why": "Both docs cover the auth refactor decision.",
        }
    ]
    payload = _base_payload([doc_payload])
    resp = QueryResponse.model_validate(payload)
    prov = resp.results[0].matched_via[0]
    assert prov.channel == "inferred_edge"
    assert prov.anchor_doc_id == "primary:doc"
    assert prov.edge_type == "DISCUSSES"
    assert prov.confidence == "INFERRED"
    assert prov.why == "Both docs cover the auth refactor decision."

    dumped = resp.model_dump(mode="json")
    assert dumped["results"][0]["matched_via"][0]["why"] == prov.why


def test_round_trip_preserves_graph_evidence_list_per_chunk() -> None:
    """A chunk reached via N seeds carries N GraphEvidence entries -- the
    list must round-trip with all entries preserved."""
    doc_payload = _doc_payload()
    doc_payload["chunks"][0]["graph_evidence"] = [
        {"edge_type": "MENTIONS", "confidence": "EXTRACTED", "via_entity": "Repo:foo"},
        {"edge_type": "AUTHORED", "confidence": "INFERRED", "via_entity": "Person:bar"},
    ]
    payload = _base_payload([doc_payload])
    resp = QueryResponse.model_validate(payload)
    chunk = resp.results[0].chunks[0]
    assert len(chunk.graph_evidence) == 2
    assert chunk.graph_evidence[0].via_entity == "Repo:foo"
    assert chunk.graph_evidence[1].confidence == "INFERRED"


# ---------------------------------------------------------------------------
# Construction-side parity with parsing-side
# ---------------------------------------------------------------------------


def test_constructed_models_dump_with_node_type_field() -> None:
    """Building a QueryDocumentResult / QueryEntityResult in Python (not via
    JSON) and then dumping must include the discriminator -- this catches
    any drift if `node_type` ever stopped being a default literal."""
    doc = QueryDocumentResult(
        canonical_id="d1",
        doc_id="d1",
        doc_version=1,
        source_system=SourceSystem.GITHUB,
        source_url="https://example/d1",
        title="t",
        created_at=_now(),
        updated_at=_now(),
        score=0.5,
        rank=1,
    )
    entity = QueryEntityResult(
        canonical_id="e1",
        label="Service",
        score=0.5,
        rank=2,
    )
    assert doc.model_dump()["node_type"] == "Document"
    assert entity.model_dump()["node_type"] == "Entity"


def test_chunk_has_no_doc_level_fields() -> None:
    """QueryChunk no longer carries doc_id, source_system, title, etc.
    Those fields live on the parent QueryDocumentResult. Constructing
    a QueryChunk with just chunk-level fields succeeds; it has no
    `doc_id` attribute."""
    chunk = QueryChunk(
        chunk_id="c0",
        content="body",
        score=0.5,
        rank_in_doc=1,
        graph_evidence=[
            GraphEvidence(edge_type="MENTIONS", confidence="EXTRACTED", via_entity="X"),
        ],
    )
    assert chunk.chunk_id == "c0"
    assert not hasattr(chunk, "doc_id")
    assert not hasattr(chunk, "source_system")
    assert not hasattr(chunk, "title")
    assert chunk.graph_evidence[0].via_entity == "X"


def test_match_provenance_inferred_fields_default_none() -> None:
    """For non-inferred-edge channels, the inferred-edge-specific fields
    on MatchProvenance default to None."""
    prov = MatchProvenance(channel="vector", rank=1, score=0.9)
    assert prov.anchor_doc_id is None
    assert prov.edge_type is None
    assert prov.confidence is None
    assert prov.why is None
