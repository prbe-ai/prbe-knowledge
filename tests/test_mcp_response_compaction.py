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
    # populated diagnostics always pass.
    assert second["matched_via"][0]["edge_type"] == "resolves"
    assert second["why_relevant"] == "names the exact symbol"
    assert second["graph_evidence"] == [{"edge_type": "resolves"}]
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
    }


def test_detail_full_is_the_compact_shape_unchanged() -> None:
    compacted = compact_search(_doc_payload())
    assert apply_detail(compacted, "full") is compacted


def test_detail_never_touches_the_envelope_and_entities_pass_through() -> None:
    """Profiles rewrite `results` rows only. A leaner detail must never make a
    degraded or truncated answer look healthier — the same failure mode the
    _TOP_LEVEL_DROP tripwire above guards, one layer down."""
    compacted = compact_search(_doc_payload())
    compacted["degraded"] = True
    compacted["degraded_reason"] = "provider_error_prefanout_fallback"
    compacted["truncated"] = True
    for detail in ("ids", "evidence", "full"):
        out = apply_detail(compacted, detail)
        for key, value in compacted.items():
            if key == "results":
                continue
            assert out[key] == value, (detail, key)
        # Entities are small and self-describing: identical at every detail.
        assert out["results"][1] == compacted["results"][1], detail


def test_detail_rejects_unknown_values_loudly() -> None:
    with pytest.raises(ValueError, match="ids, evidence, full"):
        apply_detail({"results": []}, "eviednce")
