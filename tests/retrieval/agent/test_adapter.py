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

async def test_chunks_carry_graph_evidence_from_prefanout() -> None:
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
    resp = await to_query_response(
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


async def test_confidence_breakdown_aggregates_across_chunks() -> None:
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
    resp = await to_query_response(
        query="q", gathered=gathered, trace_id="t", timing_ms={}, prefanout=prefanout,
    )
    assert resp.confidence_breakdown == {"EXTRACTED": 0, "INFERRED": 2, "AMBIGUOUS": 0}


async def test_no_prefanout_leaves_graph_evidence_empty() -> None:
    """Backward compat: empty/no prefanout (no-LLM short-circuit,
    harness passthrough) yields graph_evidence=[] per chunk and a
    zero confidence_breakdown — the legacy behaviour before this PR."""
    gathered = GathererOutput(
        entities=[],
        chunks=[_ge("d1")],
        gatherer_notes=GathererNotes(),
    )
    resp = await to_query_response(
        query="q", gathered=gathered, trace_id="t", timing_ms={}, prefanout=None,
    )
    docs = [r for r in resp.results if r.canonical_id == "d1"]
    assert docs[0].chunks[0].graph_evidence == []
    assert resp.confidence_breakdown == {"EXTRACTED": 0, "INFERRED": 0, "AMBIGUOUS": 0}


async def test_related_entities_populated_from_gathered_entities() -> None:
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
    resp = await to_query_response(
        query="q", gathered=gathered, trace_id="t", timing_ms={}, prefanout=None,
    )
    assert resp.related_entities is not None
    assert len(resp.related_entities) == 1
    re = resp.related_entities[0]
    assert re.canonical_id == "pr-72"
    assert re.label == "PR"
    assert re.display_name == "feat(phase4)"
    assert re.max_confidence == "EXTRACTED"


async def test_related_entities_label_falls_back_to_canonical_prefix() -> None:
    """Empty `label` (Cerebras provider drift — schema-tolerant default
    on GatheredEntity is ""). The dashboard panel renders the label
    column; without a fallback it shows blank. We derive from the
    canonical_id namespace prefix as a last resort."""
    gathered = GathererOutput(
        entities=[GatheredEntity(canonical_id="feature:gh:org/repo#42", label="")],
        chunks=[],
        gatherer_notes=GathererNotes(),
    )
    resp = await to_query_response(
        query="q", gathered=gathered, trace_id="t", timing_ms={}, prefanout=None,
    )
    assert resp.related_entities[0].label == "Feature"


async def test_enrichment_merges_db_edges_with_prefanout(monkeypatch) -> None:
    """Live shape from the "what is multi-granola" trace: agent emits
    Linear + Notion + Slack docs that came via vector/BM25 (no
    inferred_edge channel hit). The graph has INFERRED edges between
    every pair; without the post-hoc enrichment graph_evidence stays
    empty on all of them and the chain panel hides. With enrichment
    the chain hops surface so the panel renders. Patch the DB function
    to avoid needing a live Postgres in the test."""
    gathered = GathererOutput(
        entities=[],
        chunks=[
            _ge("linear:org:issue:abc"),
            _ge("notion:page:xyz"),
            _ge("slack:T1:C1:1.0"),
        ],
        gatherer_notes=GathererNotes(),
    )

    from shared.models import GraphEvidence as GE
    async def fake_enrich(customer_id, doc_ids):
        # Two INFERRED edges between the curated docs.
        return {
            "notion:page:xyz": [
                GE(edge_type="DISCUSSES", confidence="INFERRED",
                   via_entity="linear:org:issue:abc",
                   reason="Notion describes the implementation plan in the Linear ticket"),
            ],
            "slack:T1:C1:1.0": [
                GE(edge_type="DISCUSSES", confidence="INFERRED",
                   via_entity="linear:org:issue:abc",
                   reason="Slack thread discusses the Linear ticket's plan"),
            ],
        }
    monkeypatch.setattr(
        "services.retrieval.agent.adapter._enrich_graph_evidence_from_result_set",
        fake_enrich,
    )

    resp = await to_query_response(
        query="what is multi-granola",
        gathered=gathered,
        trace_id="t",
        timing_ms={},
        prefanout=None,
        customer_id="cust-test",  # opts into enrichment
    )

    # Both linked docs now carry graph_evidence even though prefanout
    # had nothing for them.
    notion = next(r for r in resp.results if getattr(r, "doc_id", "") == "notion:page:xyz")
    slack = next(r for r in resp.results if getattr(r, "doc_id", "") == "slack:T1:C1:1.0")
    assert len(notion.chunks[0].graph_evidence) == 1
    assert notion.chunks[0].graph_evidence[0].reason and "Linear ticket" in notion.chunks[0].graph_evidence[0].reason
    assert len(slack.chunks[0].graph_evidence) == 1
    # And the aggregated breakdown counts both.
    assert resp.confidence_breakdown["INFERRED"] == 2


async def test_enrichment_skipped_when_no_customer_id(monkeypatch) -> None:
    """`customer_id=None` is the no-LLM / harness-passthrough path —
    enrichment is opt-in to keep those paths DB-free."""
    called = {"n": 0}
    async def fake_enrich(customer_id, doc_ids):
        called["n"] += 1
        return {}
    monkeypatch.setattr(
        "services.retrieval.agent.adapter._enrich_graph_evidence_from_result_set",
        fake_enrich,
    )
    gathered = GathererOutput(
        entities=[],
        chunks=[_ge("d1"), _ge("d2")],
        gatherer_notes=GathererNotes(),
    )
    await to_query_response(
        query="q", gathered=gathered, trace_id="t", timing_ms={},
        prefanout=None, customer_id=None,
    )
    assert called["n"] == 0


async def test_enrichment_skipped_when_only_one_doc(monkeypatch) -> None:
    """No point querying the DB if only one doc is in the result set —
    the enrichment function returns early but the harness shouldn't even
    make the call. Pin that contract."""
    called = {"n": 0}
    async def fake_enrich(customer_id, doc_ids):
        called["n"] += 1
        return {}
    monkeypatch.setattr(
        "services.retrieval.agent.adapter._enrich_graph_evidence_from_result_set",
        fake_enrich,
    )
    gathered = GathererOutput(
        entities=[], chunks=[_ge("only-doc")], gatherer_notes=GathererNotes(),
    )
    await to_query_response(
        query="q", gathered=gathered, trace_id="t", timing_ms={},
        prefanout=None, customer_id="cust-test",
    )
    assert called["n"] == 0


async def test_enrichment_dedupes_against_prefanout_evidence(monkeypatch) -> None:
    """If the same `(anchor, edge_type)` hop already came through via
    prefanout's inferred_edge channel, the post-hoc enrichment must
    NOT double-render it on the chunk."""
    gathered = GathererOutput(
        entities=[],
        chunks=[_ge("d1"), _ge("d2")],
        gatherer_notes=GathererNotes(),
    )
    prefanout = {"sub_queries": [{
        "query": "q",
        "vector": [], "bm25": [], "graph": [],
        "inferred_edge": [{
            "doc_id": "d2",
            "anchor_doc_id": "d1",
            "edge_type": "DISCUSSES",
            "confidence": "INFERRED",
            "why": "from prefanout",
        }],
    }]}
    from shared.models import GraphEvidence as GE
    async def fake_enrich(customer_id, doc_ids):
        # Same (anchor, edge_type) tuple — should NOT re-add to d2.
        return {
            "d2": [GE(edge_type="DISCUSSES", confidence="INFERRED",
                     via_entity="d1", reason="from db enrichment")]
        }
    monkeypatch.setattr(
        "services.retrieval.agent.adapter._enrich_graph_evidence_from_result_set",
        fake_enrich,
    )
    resp = await to_query_response(
        query="q", gathered=gathered, trace_id="t", timing_ms={},
        prefanout=prefanout, customer_id="cust-test",
    )
    d2 = next(r for r in resp.results if getattr(r, "doc_id", "") == "d2")
    # First-wins dedup — prefanout entry is kept, enrichment one is dropped.
    assert len(d2.chunks[0].graph_evidence) == 1
    assert d2.chunks[0].graph_evidence[0].reason == "from prefanout"


async def test_query_root_doc_id_picks_top_ranked_document() -> None:
    """The dashboard's chain-graph viz pins a `query_root` node so
    everything else lays out radially around it. The adapter must
    surface a deterministic root — top-ranked Document — instead of
    making the frontend guess via `results[0]`. Pin that contract."""
    gathered = GathererOutput(
        entities=[],
        chunks=[
            _ge("github:prbe-ai/prbe-knowledge:pr:328"),
            _ge("claude_code:probe-founders:abc-123"),
        ],
        gatherer_notes=GathererNotes(),
    )
    resp = await to_query_response(
        query="why was pr328 created",
        gathered=gathered,
        trace_id="t",
        timing_ms={},
        prefanout=None,
    )
    assert resp.query_root_doc_id == "github:prbe-ai/prbe-knowledge:pr:328"


async def test_query_root_doc_id_falls_back_to_top_entity() -> None:
    """Entity-only result (grounding resolved the entity but agent
    didn't emit a doc chunk): use the top entity's canonical_id so
    the chain panel still has something to anchor on."""
    gathered = GathererOutput(
        entities=[
            GatheredEntity(canonical_id="pr-328", label="PR"),
            GatheredEntity(canonical_id="repo-x", label="Repo"),
        ],
        chunks=[],
        gatherer_notes=GathererNotes(),
    )
    resp = await to_query_response(
        query="pr 328", gathered=gathered, trace_id="t", timing_ms={}, prefanout=None,
    )
    assert resp.query_root_doc_id == "pr-328"


async def test_query_root_doc_id_none_for_empty_result() -> None:
    """Empty result set → None. Frontend hides the root pin instead of
    falling back to a stale value."""
    gathered = GathererOutput(
        entities=[], chunks=[], gatherer_notes=GathererNotes(),
    )
    resp = await to_query_response(
        query="q", gathered=gathered, trace_id="t", timing_ms={}, prefanout=None,
    )
    assert resp.query_root_doc_id is None


async def test_related_entities_none_when_agent_emitted_no_entities() -> None:
    """The model contract distinguishes `None` (not requested / walk
    failed) from `[]` (requested, no neighbors). With zero gathered
    entities we emit None — that's the "this query didn't surface any
    crawl candidates" signal, not the "panel disabled" signal."""
    gathered = GathererOutput(
        entities=[], chunks=[_ge("d1")], gatherer_notes=GathererNotes(),
    )
    resp = await to_query_response(
        query="q", gathered=gathered, trace_id="t", timing_ms={}, prefanout=None,
    )
    assert resp.related_entities is None
