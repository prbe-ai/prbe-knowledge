"""The MCP response-compaction layer must not strip caller-critical signals.

`compact_search` runs on every `search_knowledge` response and drops a
deny-list of diagnostic fields before the caller sees the payload. Three
top-level fields are deliberately exempt because an agent needs them to
decide whether to trust or re-run:

  * `confidence_breakdown` — pre-existing router signal.
  * `degraded`            — added with the degradation-visibility work.
  * `degraded_reason`     — says WHICH action to take (retry vs narrow).

The failure mode this guards is silent and total: if any of those names lands in
`_TOP_LEVEL_DROP`, the field vanishes from every response and a degraded
search becomes byte-identical to a healthy one again. There is no error, no
log, and no other test that would notice — which is the exact bug the
`degraded` flag exists to fix, reintroduced one deny-list entry later.
"""

from __future__ import annotations

import pytest

from engine.mcp.clients._responses import _TOP_LEVEL_DROP, apply_detail, compact_search

# Fields that must survive compaction. Add to this list, never remove.
_MUST_SURVIVE = ("degraded", "degraded_reason", "confidence_breakdown")


def _payload() -> dict:
    return {
        "query": "outage",
        "results": [],
        "total_candidates": 0,
        "router_hit_cache": False,
        "trace_id": "t-1",
        "degraded": True,
        "degraded_reason": "provider_error_prefanout_fallback",
        "confidence_breakdown": {"EXTRACTED": 0, "INFERRED": 3, "AMBIGUOUS": 0},
        # A field that SHOULD be stripped, so this test also proves
        # compaction is actually running rather than passing everything.
        "timing_ms": {"total": 1.0},
    }


def test_critical_fields_are_not_in_the_deny_list() -> None:
    """Tripwire on the deny-list itself, independent of any payload."""
    for field in _MUST_SURVIVE:
        assert field not in _TOP_LEVEL_DROP, (
            f"{field!r} was added to _TOP_LEVEL_DROP — this silently removes it "
            "from every MCP response. See this module's docstring."
        )


def test_degraded_survives_compaction() -> None:
    out = compact_search(_payload())
    assert out["degraded"] is True
    assert out["degraded_reason"] == "provider_error_prefanout_fallback"


def test_healthy_response_still_carries_the_flag() -> None:
    """`degraded: false` must be present, not omitted.

    An absent key and a false key are the same thing to a careless caller,
    but only one of them is distinguishable from an old server that predates
    the field.
    """
    payload = _payload() | {"degraded": False, "degraded_reason": None}
    out = compact_search(payload)
    assert out["degraded"] is False
    assert out["degraded_reason"] is None


def test_compaction_still_strips_diagnostics() -> None:
    """Negative control: proves the deny-list is applied at all."""
    out = compact_search(_payload())
    assert "timing_ms" not in out


# ---------------------------------------------------------------------------
# Redundancy collapse + detail profiles (the token-lean work).
#
# Every rule below was measured on live production responses before being
# written: canonical_id == doc_id on 69/83 docs, chunk retriever_scores == {}
# on 69/69 chunks, chunk matched_via identical to the doc's on 19/69. Under
# the ~20KB byte budget those repeats were paid for with dropped tail chunks
# (9/10 measured responses truncated), so this is recall, not just tokens.
# ---------------------------------------------------------------------------


def _doc_payload() -> dict:
    boiler_via = [
        {
            "channel": "vector",
            "rank": 1,
            "score": 1.0,
            "intent_idx": 0,
            "anchor_doc_id": None,
            "edge_type": None,
            "confidence": None,
            "why": None,
        }
    ]
    edge_via = [
        {
            "channel": "graph",
            "rank": 2,
            "score": 0.8,
            "intent_idx": 0,
            "anchor_doc_id": "github:acme:pr:1",
            "edge_type": "resolves",
            "confidence": "INFERRED",
            "why": "PR resolves the ticket the query names",
        }
    ]
    return {
        "query": "q",
        "trace_id": "t-1",
        "degraded": False,
        "degraded_reason": None,
        "total_candidates": 2,
        "confidence_breakdown": {"EXTRACTED": 1, "INFERRED": 1, "AMBIGUOUS": 0},
        "results": [
            {
                "node_type": "Document",
                "doc_id": "github:acme:pr:1",
                "canonical_id": "github:acme:pr:1",  # identical -> collapsed
                "title": "fix the thing",
                "source_system": "github",
                "source_url": "https://github.com/acme/pr/1",
                "score": 0.9,
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-02T00:00:00Z",
                "author_id": "someone",
                "chunk_count": 2,
                "matched_via": boiler_via,
                "chunks": [
                    {
                        "content": "the evidence",
                        "score": 0.9,
                        "chunk_id": "c1",
                        "rank_in_doc": 0,
                        "retriever_scores": {},
                        "why_relevant": "",
                        "graph_evidence": [],
                        "matched_via": boiler_via,  # == doc's -> collapsed
                    },
                    {
                        "content": "own trail",
                        "score": 0.7,
                        "retriever_scores": {},
                        "why_relevant": "names the exact symbol",
                        "graph_evidence": [{"edge_type": "resolves"}],
                        "matched_via": edge_via,  # differs -> kept
                    },
                ],
            },
            {
                "node_type": "Entity",
                "canonical_id": "code_graph:acme:svc",
                "label": "svc",
                "score": 1.0,
                "rank": 1,
            },
        ],
    }


def test_compaction_collapses_measured_repeats() -> None:
    doc = compact_search(_doc_payload())["results"][0]
    # Identical canonical_id is a repeat of doc_id, not a second identity.
    assert "canonical_id" not in doc
    assert doc["doc_id"] == "github:acme:pr:1"
    first, second = doc["chunks"]
    # The chunk that inherited the doc's provenance verbatim drops its copy...
    assert "matched_via" not in first
    # ...and empty diagnostic fields go with it.
    for gone in ("retriever_scores", "why_relevant", "graph_evidence", "chunk_id"):
        assert gone not in first
    # A chunk that matched its own way keeps its own (leaned) trail, and
    # populated rationale always passes.
    assert second["matched_via"][0]["edge_type"] == "resolves"
    assert second["why_relevant"] == "names the exact symbol"
    # graph_evidence is VERBOSE-ONLY (2026-08-31): measured at ~42% of a
    # live response after the graph channel lit up. confidence_breakdown
    # still says it exists; verbose=True returns the trails.
    assert "graph_evidence" not in second
    # Null-valued provenance keys are stripped; populated ones are untouched.
    assert doc["matched_via"] == [
        {"channel": "vector", "rank": 1, "score": 1.0, "intent_idx": 0}
    ]


def test_canonical_id_survives_when_it_differs() -> None:
    payload = _doc_payload()
    payload["results"][0]["canonical_id"] = "linear:acme:ticket:9"
    doc = compact_search(payload)["results"][0]
    assert doc["canonical_id"] == "linear:acme:ticket:9"


def test_detail_evidence_drops_audit_metadata_but_keeps_the_handles() -> None:
    out = apply_detail(compact_search(_doc_payload()), "evidence")
    doc = out["results"][0]
    for gone in ("created_at", "updated_at", "author_id"):
        assert gone not in doc
    # Boilerplate provenance goes; the citation/drill-down handles stay.
    assert "matched_via" not in doc
    for kept in ("doc_id", "title", "source_url", "score", "chunk_count"):
        assert kept in doc
    # Chunk content is the point of "evidence" — untouched, edge-provenance
    # kept on the chunk that has it.
    assert doc["chunks"][0]["content"] == "the evidence"
    assert doc["chunks"][1]["matched_via"][0]["why"]


def test_detail_ids_returns_identity_without_content() -> None:
    out = apply_detail(compact_search(_doc_payload()), "ids")
    doc = out["results"][0]
    assert doc == {
        "node_type": "Document",
        "doc_id": "github:acme:pr:1",
        "title": "fix the thing",
        "source_system": "github",
        "score": 0.9,
        # The weight signal: 1 span vs 12 is a different triage answer.
        "chunk_count": 2,
    }


def test_detail_full_is_the_compact_shape_unchanged() -> None:
    compacted = compact_search(_doc_payload())
    assert apply_detail(compacted, "full") is compacted


def test_detail_never_touches_the_envelope_and_entities_pass_through() -> None:
    """Profiles rewrite `results` rows only. A leaner detail must never make a
    degraded or truncated answer look healthier — the same failure mode the
    _TOP_LEVEL_DROP tripwire above guards, one layer down."""
    import copy

    compacted = compact_search(_doc_payload())
    compacted["degraded"] = True
    compacted["degraded_reason"] = "provider_error_prefanout_fallback"
    compacted["truncated"] = True
    # Deep snapshot: apply_detail shares sub-objects with its input, so
    # comparing out[key] == compacted[key] would compare an object with
    # itself and pass even if a profile mutated it in place.
    expected = copy.deepcopy(compacted)
    for detail in ("ids", "evidence", "full"):
        out = apply_detail(compacted, detail)
        for key, value in expected.items():
            if key == "results":
                continue
            assert out[key] == value, (detail, key)
        # Entities are small and self-describing: identical at every detail.
        assert out["results"][1] == expected["results"][1], detail
    # And the input itself was never mutated by any profile.
    assert compacted == expected


def test_detail_rejects_unknown_values_loudly() -> None:
    with pytest.raises(ValueError, match="ids, evidence, full"):
        apply_detail({"results": []}, "eviednce")


def test_detail_literal_matches_the_vocabulary() -> None:
    """The tool schema's Literal and the runtime tuple must be the same set:
    the Literal is what clients see and what FastMCP validates pre-handler,
    VALID_DETAILS is what apply_detail enforces — a fourth profile added to
    one and not the other would validate in one layer and 422 in the next."""
    from typing import get_args

    from engine.mcp.clients._responses import VALID_DETAILS
    from engine.mcp.server import DetailMode

    assert tuple(get_args(DetailMode)) == tuple(VALID_DETAILS)


def test_compact_query_applies_the_same_collapse_and_keeps_synthesis_fields() -> None:
    """query_knowledge shares _compact_result, so its evidence rows changed
    with this diff too — pin the collapse on that path, and that the
    synthesized fields ride through untouched."""
    from engine.mcp.clients._responses import compact_query

    payload = {
        **_doc_payload(),
        "answer": "the answer",
        "citations": [{"doc_id": "github:acme:pr:1"}],
        "insufficient_context": False,
        "model": "m",
    }
    out = compact_query(payload)
    assert out["answer"] == "the answer"
    assert out["citations"] == [{"doc_id": "github:acme:pr:1"}]
    assert out["insufficient_context"] is False
    doc = out["results"][0]
    # Same collapse rules as search: canonical_id repeat gone, inherited
    # chunk provenance gone, empty diagnostics gone — and the doc keeps the
    # audit metadata the search DEFAULT drops (query has no detail param;
    # its rows match search at detail="full").
    assert "canonical_id" not in doc
    assert doc["author_id"] == "someone"
    first = doc["chunks"][0]
    assert "matched_via" not in first
    assert "retriever_scores" not in first


def test_boilerplate_trims_2026_08_31() -> None:
    """Constant/empty fields agents never act on are shed; populated
    values pass. Envelope honesty fields are untouched."""
    payload = _doc_payload()
    payload["aggregations"] = []
    payload["related_entities_error"] = None
    payload["related_entities"] = [
        {
            "canonical_id": "person:alice",
            "label": "Person",
            "display_name": "Alice",
            "edge_types": [],
            "max_confidence": "EXTRACTED",
            "doc_count": 1,
            "score": 1.0,
            "associated_doc_ids": [],
            "member_count": 1,
            "member_sources": [],
        }
    ]
    payload["degraded"] = False
    payload["results"][0]["chunks"][0]["score"] = 1.0
    payload["results"][1].update(
        {"matched_via": [], "properties": {}, "attached_doc_ids": [],
         "edge_types": [], "doc_count": 0}
    )
    out = compact_search(payload)
    assert "aggregations" not in out
    assert "related_entities_error" not in out
    assert out["related_entities"] == [
        {"canonical_id": "person:alice", "label": "Person", "display_name": "Alice"}
    ]
    assert out["degraded"] is False  # honesty fields pass untouched
    chunk = out["results"][0]["chunks"][0]
    assert "score" not in chunk  # the gatherer-path constant 1.0
    entity = out["results"][1]
    for gone in ("matched_via", "properties", "attached_doc_ids",
                 "edge_types", "doc_count"):
        assert gone not in entity
    assert entity["canonical_id"] == "code_graph:acme:svc"


def test_populated_values_survive_the_boilerplate_trims() -> None:
    payload = _doc_payload()
    payload["aggregations"] = [{"count": 3}]
    payload["related_entities_error"] = "walk timed out"
    payload["results"][0]["chunks"][0]["score"] = 0.83
    payload["results"][1].update({"doc_count": 4, "edge_types": ["AUTHORED"]})
    out = compact_search(payload)
    assert out["aggregations"] == [{"count": 3}]
    assert out["related_entities_error"] == "walk timed out"
    assert out["results"][0]["chunks"][0]["score"] == 0.83
    assert out["results"][1]["doc_count"] == 4
    assert out["results"][1]["edge_types"] == ["AUTHORED"]
