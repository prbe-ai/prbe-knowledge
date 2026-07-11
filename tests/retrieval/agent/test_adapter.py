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


async def test_top_k_related_zero_disables_gatherer_projection() -> None:
    gathered = GathererOutput(
        entities=[GatheredEntity(canonical_id="pr-72", label="PR")],
        chunks=[],
        gatherer_notes=GathererNotes(),
    )

    resp = await to_query_response(
        query="q",
        gathered=gathered,
        trace_id="t",
        timing_ms={},
        top_k_related=0,
    )

    assert resp.related_entities is None


async def test_top_k_related_caps_gatherer_projection() -> None:
    gathered = GathererOutput(
        entities=[
            GatheredEntity(canonical_id="pr-72", label="PR"),
            GatheredEntity(canonical_id="pr-73", label="PR"),
        ],
        chunks=[],
        gatherer_notes=GathererNotes(),
    )

    resp = await to_query_response(
        query="q",
        gathered=gathered,
        trace_id="t",
        timing_ms={},
        top_k_related=1,
    )

    assert resp.related_entities is not None
    assert [entity.canonical_id for entity in resp.related_entities] == ["pr-72"]


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


async def test_enrichment_fires_on_single_doc_result(monkeypatch) -> None:
    """Live trace 2026-05-19: Cerebras agent curation collapsed
    identical-query reruns from 5 docs to 2 docs. Under the old `>= 2`
    threshold a 2-doc curated set (1 Document + 1 Entity) skipped
    enrichment → confidence_breakdown.INFERRED dropped from 10 to 0 →
    chain panel rendered empty.

    Threshold now fires on >=1 doc — the enrichment function itself
    handles the SELECT shape (edges where AT LEAST ONE endpoint is in
    the curated set) so a single result doc still gets its chain
    neighbors projected. Pin the new contract."""
    called = {"n": 0, "doc_ids": []}

    async def fake_enrich(customer_id, doc_ids):
        called["n"] += 1
        called["doc_ids"] = list(doc_ids)
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
    assert called["n"] == 1
    assert called["doc_ids"] == ["only-doc"]


async def test_enrichment_carries_via_entity_title_through_to_response(monkeypatch) -> None:
    """The new `via_entity_title` field on GraphEvidence lets the
    dashboard chain-graph render the OTHER endpoint of an edge with a
    human-readable label even when that doc isn't itself in the curated
    result set. This test pins that the field flows from the
    enrichment query down through to the QueryChunk shape consumers
    actually read."""
    gathered = GathererOutput(
        entities=[],
        chunks=[_ge("linear:org:issue:abc")],
        gatherer_notes=GathererNotes(),
    )

    from shared.models import GraphEvidence as GE

    async def fake_enrich(customer_id, doc_ids):
        return {
            "linear:org:issue:abc": [
                GE(
                    edge_type="DISCUSSES",
                    confidence="INFERRED",
                    via_entity="slack:T1:C1:1.0",
                    via_entity_title="multi-granola deploy order is **strict**...",
                    reason="The Slack thread discusses the Multi-Granola implementation plan.",
                ),
            ],
        }

    monkeypatch.setattr(
        "services.retrieval.agent.adapter._enrich_graph_evidence_from_result_set",
        fake_enrich,
    )

    resp = await to_query_response(
        query="multi-granola",
        gathered=gathered,
        trace_id="t",
        timing_ms={},
        prefanout=None,
        customer_id="cust-test",
    )
    linear = next(
        r for r in resp.results if getattr(r, "doc_id", "") == "linear:org:issue:abc"
    )
    ev = linear.chunks[0].graph_evidence[0]
    assert ev.via_entity == "slack:T1:C1:1.0"
    assert ev.via_entity_title == "multi-granola deploy order is **strict**..."


async def test_enrichment_carries_neighbor_metadata_through_to_response(monkeypatch) -> None:
    """`via_entity_source_system` / `_created_at` / `_url` flow from
    the enrichment query through to the synthesizer-visible shape so
    the synthesis LLM can order chain hops chronologically and cite
    by source. Without these the LLM saw an opaque canonical_id and
    refused chronology questions (verified empty answer 2026-05-20).
    """
    from datetime import UTC, datetime

    from shared.models import GraphEvidence as GE

    fixed_ts = datetime(2026, 5, 5, 16, 6, 56, tzinfo=UTC)

    async def fake_enrich(customer_id, doc_ids):
        return {
            "linear:org:issue:abc": [
                GE(
                    edge_type="DISCUSSES",
                    confidence="INFERRED",
                    via_entity="slack:T1:C1:1.0",
                    via_entity_title="thread 1: Mahit raises Granola sync gap",
                    via_entity_source_system="slack",
                    via_entity_created_at=fixed_ts,
                    via_entity_url="https://slack.com/archives/C1/p10",
                    reason="The Slack thread raises the original gap.",
                ),
            ],
        }

    monkeypatch.setattr(
        "services.retrieval.agent.adapter._enrich_graph_evidence_from_result_set",
        fake_enrich,
    )

    gathered = GathererOutput(
        entities=[],
        chunks=[_ge("linear:org:issue:abc")],
        gatherer_notes=GathererNotes(),
    )
    resp = await to_query_response(
        query="multi-granola timeline",
        gathered=gathered,
        trace_id="t",
        timing_ms={},
        prefanout=None,
        customer_id="cust-test",
    )
    linear = next(
        r for r in resp.results if getattr(r, "doc_id", "") == "linear:org:issue:abc"
    )
    ev = linear.chunks[0].graph_evidence[0]
    assert ev.via_entity_source_system == "slack"
    assert ev.via_entity_created_at == fixed_ts
    assert ev.via_entity_url == "https://slack.com/archives/C1/p10"


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
            _ge("claude_code:acme:abc-123"),
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


# ---------------------------------------------------------------------------
# display_name enrichment from graph_nodes.properties.name
# ---------------------------------------------------------------------------


async def test_display_name_enriched_from_graph_nodes_lookup(monkeypatch) -> None:
    """The gatherer rarely populates GatheredEntity.properties; without
    this enrichment the response's display_name falls back to canonical_id
    and the dashboard renders opaque Slack IDs ("C0B20FZSCUU") instead of
    resolved names ("engineering"). The adapter batches a single DB query
    keyed by canonical_id, stamps the resolved `properties.name` into both
    `related_entities[].display_name` and `extracted_entities[].display_name`."""
    captured: dict[str, object] = {}

    async def fake_lookup(customer_id, canonical_ids):
        captured["customer_id"] = customer_id
        captured["canonical_ids"] = sorted(canonical_ids)
        return {
            "C0B1T8PPK0D": "engineering",
            "U0ARLAD3B2B": "Richard Wei",
        }

    monkeypatch.setattr(
        "services.retrieval.agent.adapter._fetch_entity_names_from_graph",
        fake_lookup,
    )

    gathered = GathererOutput(
        entities=[
            # Gatherer-emitted shape: label carries a readable string, but
            # properties is empty (Cerebras provider drift). Pre-fix this
            # surfaced display_name=canonical_id in the response.
            GatheredEntity(canonical_id="C0B1T8PPK0D", label="engineering channel", properties={}),
            GatheredEntity(canonical_id="U0ARLAD3B2B", label="Richard Wei", properties={}),
            # Cache-miss case: not in the lookup result, falls back to
            # gatherer label rather than canonical_id.
            GatheredEntity(canonical_id="U0NOTINDEX", label="Some Person", properties={}),
        ],
        chunks=[],
        gatherer_notes=GathererNotes(),
    )

    resp = await to_query_response(
        query="q", gathered=gathered, trace_id="t", timing_ms={},
        prefanout=None, customer_id="cust-1",
    )

    assert captured["customer_id"] == "cust-1"
    assert captured["canonical_ids"] == ["C0B1T8PPK0D", "U0ARLAD3B2B", "U0NOTINDEX"]

    related_by_id = {e.canonical_id: e for e in resp.related_entities}
    assert related_by_id["C0B1T8PPK0D"].display_name == "engineering"
    assert related_by_id["U0ARLAD3B2B"].display_name == "Richard Wei"
    # Cache miss → falls back to the gatherer's label (still readable),
    # not the raw canonical_id.
    assert related_by_id["U0NOTINDEX"].display_name == "Some Person"

    extracted_by_id = {e["canonical_id"]: e for e in resp.extracted_entities}
    assert extracted_by_id["C0B1T8PPK0D"]["display_name"] == "engineering"
    assert extracted_by_id["U0ARLAD3B2B"]["display_name"] == "Richard Wei"
    assert extracted_by_id["U0NOTINDEX"]["display_name"] == "Some Person"


async def test_display_name_enrichment_prefers_db_over_emitted_properties(
    monkeypatch,
) -> None:
    """When BOTH the DB lookup AND the gatherer's emitted properties have
    a name, the DB wins. The DB reflects current ingestion state; the
    gatherer's emitted name is a token from the LLM's working memory and
    may be stale or hallucinated."""

    async def fake_lookup(customer_id, canonical_ids):
        return {"C0B1T8PPK0D": "engineering"}

    monkeypatch.setattr(
        "services.retrieval.agent.adapter._fetch_entity_names_from_graph",
        fake_lookup,
    )

    gathered = GathererOutput(
        entities=[
            GatheredEntity(
                canonical_id="C0B1T8PPK0D",
                label="Channel",
                properties={"name": "stale-old-name"},
            ),
        ],
        chunks=[],
        gatherer_notes=GathererNotes(),
    )
    resp = await to_query_response(
        query="q", gathered=gathered, trace_id="t", timing_ms={},
        prefanout=None, customer_id="cust-1",
    )
    assert resp.related_entities[0].display_name == "engineering"


async def test_display_name_enrichment_skipped_when_no_customer_id(monkeypatch) -> None:
    """`customer_id=None` is the test / harness-passthrough path —
    no DB available, so no enrichment. Falls back to the gatherer's
    emitted properties / label / canonical_id chain (pre-PR behavior)."""
    calls = {"n": 0}

    async def fake_lookup(customer_id, canonical_ids):
        calls["n"] += 1
        return {}

    monkeypatch.setattr(
        "services.retrieval.agent.adapter._fetch_entity_names_from_graph",
        fake_lookup,
    )

    gathered = GathererOutput(
        entities=[
            GatheredEntity(
                canonical_id="C0B1T8PPK0D",
                label="engineering channel",
                properties={"name": "engineering"},
            ),
        ],
        chunks=[],
        gatherer_notes=GathererNotes(),
    )
    resp = await to_query_response(
        query="q", gathered=gathered, trace_id="t", timing_ms={},
        prefanout=None, customer_id=None,
    )
    assert calls["n"] == 0, "enrichment must not fire without customer_id"
    # Falls back to gatherer's emitted properties.name.
    assert resp.related_entities[0].display_name == "engineering"


async def test_display_name_enrichment_handles_db_failure_gracefully(monkeypatch) -> None:
    """DB outage / query timeout returns {} — response still renders
    with the pre-enrichment fallback chain. The dashboard sees readable
    text (the gatherer-emitted label) rather than a 500."""
    from services.retrieval.agent import adapter as adapter_mod

    async def boom(customer_id, canonical_ids):
        # Simulate the underlying with_tenant path raising. The wrapper
        # logs and returns {} so callers fall back to the chain.
        return {}

    monkeypatch.setattr(adapter_mod, "_fetch_entity_names_from_graph", boom)

    gathered = GathererOutput(
        entities=[GatheredEntity(canonical_id="C0NEW", label="brand-new-channel", properties={})],
        chunks=[],
        gatherer_notes=GathererNotes(),
    )
    resp = await to_query_response(
        query="q", gathered=gathered, trace_id="t", timing_ms={},
        prefanout=None, customer_id="cust-1",
    )
    # No DB hit — fall back to the gatherer-emitted label rather than
    # raw canonical_id.
    assert resp.related_entities[0].display_name == "brand-new-channel"
