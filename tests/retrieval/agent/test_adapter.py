"""Adapter tests: GathererOutput → QueryResponse projection.

Focus: the chain-data projection (`graph_evidence` per chunk +
aggregated `confidence_breakdown` + `related_entities` from curated
entities). The legacy doc/entity grouping is covered indirectly by
test_loop.py end-to-end runs.
"""

from __future__ import annotations

from services.retrieval.agent.adapter import (
    _build_doc_to_graph_evidence,
    to_query_response,
)
from services.retrieval.agent.models import (
    GatheredChunk,
    GatheredEntity,
    GathererNotes,
    GathererOutput,
)


def _ge(doc_id: str, content: str = "...", matched_via: list[str] | None = None) -> GatheredChunk:
    return GatheredChunk(
        doc_id=doc_id,
        chunk_id=f"{doc_id}:c0",
        content=content,
        matched_via=matched_via or ["inferred_edge"],
        why_relevant="surfaced via test",
    )


# ============================================================
# _build_doc_to_graph_evidence — prefanout → doc_id index
# ============================================================

def test_evidence_index_groups_by_linked_doc_id() -> None:
    """Inferred-edge hits index by their `doc_id` (the LINKED doc, not
    the anchor). One linked doc can have evidence from multiple anchors
    (separate edges originating from different source docs)."""
    prefanout = {"sub_queries": [{
        "query": "q",
        "vector": [], "bm25": [], "graph": [],
        "inferred_edge": [
            {
                "doc_id": "linear:org:issue:abc",
                "anchor_doc_id": "github:pr:72",
                "edge_type": "motivates_pr",
                "confidence": "INFERRED",
                "why": "ticket asks for the proxy that pr72 builds",
            },
            {
                "doc_id": "linear:org:issue:abc",
                "anchor_doc_id": "github:pr:73",
                "edge_type": "related_to",
                "confidence": "INFERRED",
                "why": "fix builds on the same enrichment path",
            },
        ],
    }]}
    idx = _build_doc_to_graph_evidence(prefanout)
    assert list(idx) == ["linear:org:issue:abc"]
    evidence = idx["linear:org:issue:abc"]
    assert len(evidence) == 2
    anchors = {e.via_entity for e in evidence}
    assert anchors == {"github:pr:72", "github:pr:73"}
    reasons = {e.reason for e in evidence}
    assert any("proxy that pr72 builds" in (r or "") for r in reasons)


def test_evidence_index_dedupes_repeat_anchor_edge() -> None:
    """Cross-sub_query overlap can surface the same `(anchor, edge_type)`
    pair twice for one linked doc. We keep first occurrence only — same
    discipline as `_format_inferred_chains` in loop.py."""
    hit = {
        "doc_id": "linear:org:issue:abc",
        "anchor_doc_id": "github:pr:72",
        "edge_type": "motivates_pr",
        "confidence": "INFERRED",
        "why": "first occurrence",
    }
    dup = dict(hit, why="duplicate occurrence — should be skipped")
    idx = _build_doc_to_graph_evidence({"sub_queries": [
        {"query": "q1", "vector": [], "bm25": [], "graph": [], "inferred_edge": [hit]},
        {"query": "q2", "vector": [], "bm25": [], "graph": [], "inferred_edge": [dup]},
    ]})
    assert len(idx["linear:org:issue:abc"]) == 1
    assert idx["linear:org:issue:abc"][0].reason == "first occurrence"


def test_evidence_index_skips_hits_missing_doc_or_anchor() -> None:
    """Defensive: a hit missing either endpoint can't form a valid
    chain. Skip silently rather than building a degenerate entry."""
    idx = _build_doc_to_graph_evidence({"sub_queries": [{
        "query": "q", "vector": [], "bm25": [], "graph": [],
        "inferred_edge": [
            {"doc_id": "", "anchor_doc_id": "a"},  # empty doc_id
            {"doc_id": "linked", "anchor_doc_id": ""},  # empty anchor
            {"doc_id": "linked", "anchor_doc_id": "a", "edge_type": "ok",
             "confidence": "INFERRED"},
        ],
    }]})
    assert "linked" in idx
    assert len(idx["linked"]) == 1


def test_evidence_index_empty_when_no_prefanout() -> None:
    assert _build_doc_to_graph_evidence(None) == {}
    assert _build_doc_to_graph_evidence({}) == {}
    assert _build_doc_to_graph_evidence({"sub_queries": []}) == {}


# ============================================================
# to_query_response — end-to-end chain projection
# ============================================================

def test_chunks_carry_graph_evidence_from_prefanout() -> None:
    """The agent emits a chunk for `linear:org:issue:abc`; the prefanout
    has an inferred-edge hit pointing PR #72 → that doc with a `why`
    string. The adapter projects the `why` onto every chunk of that
    doc as `graph_evidence`, so MCP consumers see the chain rationale
    instead of a flat doc list."""
    gathered = GathererOutput(
        entities=[],
        chunks=[
            _ge("linear:org:issue:abc"),
        ],
        gatherer_notes=GathererNotes(),
    )
    prefanout = {"sub_queries": [{
        "query": "q",
        "vector": [], "bm25": [], "graph": [],
        "inferred_edge": [{
            "doc_id": "linear:org:issue:abc",
            "anchor_doc_id": "github:pr:72",
            "edge_type": "motivates_pr",
            "confidence": "INFERRED",
            "why": "linear ticket asks for the proxy that pr72 builds",
        }],
    }]}
    resp = to_query_response(
        query="why was pr72 built",
        gathered=gathered,
        trace_id="t-1",
        timing_ms={},
        prefanout=prefanout,
    )
    docs = [r for r in resp.results if r.canonical_id == "linear:org:issue:abc"]
    assert len(docs) == 1
    chunk = docs[0].chunks[0]
    assert len(chunk.graph_evidence) == 1
    ge = chunk.graph_evidence[0]
    assert ge.edge_type == "motivates_pr"
    assert ge.confidence == "INFERRED"
    assert ge.via_entity == "github:pr:72"
    assert ge.reason and "proxy that pr72 builds" in ge.reason


def test_confidence_breakdown_aggregates_across_chunks() -> None:
    """Top-level `confidence_breakdown` is the count of evidence by
    tier across every chunk in the response. Mirrors the legacy
    fusion-path counter so MCP filters keyed on confidence keep working."""
    prefanout = {"sub_queries": [{
        "query": "q",
        "vector": [], "bm25": [], "graph": [],
        "inferred_edge": [
            {"doc_id": "d1", "anchor_doc_id": "a1", "edge_type": "e1",
             "confidence": "INFERRED", "why": "w1"},
            {"doc_id": "d2", "anchor_doc_id": "a2", "edge_type": "e2",
             "confidence": "INFERRED", "why": "w2"},
        ],
    }]}
    gathered = GathererOutput(
        entities=[],
        chunks=[_ge("d1"), _ge("d2")],
        gatherer_notes=GathererNotes(),
    )
    resp = to_query_response(
        query="q", gathered=gathered, trace_id="t", timing_ms={}, prefanout=prefanout,
    )
    assert resp.confidence_breakdown == {"EXTRACTED": 0, "INFERRED": 2, "AMBIGUOUS": 0}


def test_no_prefanout_leaves_graph_evidence_empty() -> None:
    """Backward compat: empty/no prefanout (no-LLM short-circuit,
    harness passthrough) yields graph_evidence=[] per chunk and a
    zero confidence_breakdown — the legacy behaviour before this PR."""
    gathered = GathererOutput(
        entities=[],
        chunks=[_ge("d1")],
        gatherer_notes=GathererNotes(),
    )
    resp = to_query_response(
        query="q", gathered=gathered, trace_id="t", timing_ms={}, prefanout=None,
    )
    docs = [r for r in resp.results if r.canonical_id == "d1"]
    assert docs[0].chunks[0].graph_evidence == []
    assert resp.confidence_breakdown == {"EXTRACTED": 0, "INFERRED": 0, "AMBIGUOUS": 0}


def test_related_entities_populated_from_gathered_entities() -> None:
    """`related_entities` is the dashboard / MCP crawl-candidate field.
    Before this PR the adapter left it None — the dashboard's related-
    entities panel rendered empty. Now we project gathered entities
    into it (best-effort scoring; future PR can swap to a proper
    1-hop walker)."""
    gathered = GathererOutput(
        entities=[
            GatheredEntity(
                canonical_id="pr-72",
                label="PR",
                properties={"name": "feat(phase4)", "url": "https://github.com/..."},
                why_relevant="the answer entity",
            ),
        ],
        chunks=[],
        gatherer_notes=GathererNotes(),
    )
    resp = to_query_response(
        query="q", gathered=gathered, trace_id="t", timing_ms={}, prefanout=None,
    )
    assert resp.related_entities is not None
    assert len(resp.related_entities) == 1
    re = resp.related_entities[0]
    assert re.canonical_id == "pr-72"
    assert re.label == "PR"
    assert re.display_name == "feat(phase4)"
    assert re.max_confidence == "EXTRACTED"


def test_related_entities_label_falls_back_to_canonical_prefix() -> None:
    """Empty `label` (Cerebras provider drift — schema-tolerant default
    on GatheredEntity is ""). The dashboard panel renders the label
    column; without a fallback it shows blank. We derive from the
    canonical_id namespace prefix as a last resort."""
    gathered = GathererOutput(
        entities=[GatheredEntity(canonical_id="feature:gh:org/repo#42", label="")],
        chunks=[],
        gatherer_notes=GathererNotes(),
    )
    resp = to_query_response(
        query="q", gathered=gathered, trace_id="t", timing_ms={}, prefanout=None,
    )
    assert resp.related_entities[0].label == "Feature"


def test_related_entities_none_when_agent_emitted_no_entities() -> None:
    """The model contract distinguishes `None` (not requested / walk
    failed) from `[]` (requested, no neighbors). With zero gathered
    entities we emit None — that's the "this query didn't surface any
    crawl candidates" signal, not the "panel disabled" signal."""
    gathered = GathererOutput(
        entities=[], chunks=[_ge("d1")], gatherer_notes=GathererNotes(),
    )
    resp = to_query_response(
        query="q", gathered=gathered, trace_id="t", timing_ms={}, prefanout=None,
    )
    assert resp.related_entities is None
