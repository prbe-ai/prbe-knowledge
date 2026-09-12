"""E5: surprise ordering is an exploration posture, not the default.

The graph channel sorted every query by a per-edge surprise score. Surprise
deliberately rewards the speculative edge (AMBIGUOUS carries a 1.5x weight
against EXTRACTED's 1.0) and penalizes hub-to-hub links -- which is the ranking
you want for "anything else I should know about X" and the exact opposite of
what you want for "what is the status of PRB-17".
"""

from __future__ import annotations

import inspect

from engine.retrieval.retrievers.graph import GraphHit, graph_search, order_graph_hits


def _hit(chunk_id: str, confidence: str, score: float) -> GraphHit:
    from datetime import UTC, datetime

    now = datetime.now(UTC)
    return GraphHit(
        chunk_id=chunk_id,
        doc_id=f"slack:{chunk_id}",
        doc_version=1,
        source_system="slack",
        source_url="",
        title=None,
        content="body",
        created_at=now,
        updated_at=now,
        score=score,
        via_entity="e1",
        confidence=confidence,
    )


def _sort_as_module_does(hits: list[GraphHit], *, discovery: bool) -> list[str]:
    """Run the REAL ordering. `order_graph_hits` is the function `graph_search`
    calls, extracted so this exercises production code rather than a copy of it
    that cannot fail when production changes."""
    ordered = list(hits)
    order_graph_hits(ordered, discovery=discovery)
    return [h.chunk_id for h in ordered]


def test_graph_search_takes_a_discovery_flag() -> None:
    assert "discovery" in inspect.signature(graph_search).parameters


def test_discovery_ranks_the_surprising_edge_first() -> None:
    hits = [
        _hit("canonical", "EXTRACTED", score=1.0),
        _hit("speculative", "AMBIGUOUS", score=6.0),
    ]
    assert _sort_as_module_does(hits, discovery=True)[0] == "speculative"


def test_a_direct_lookup_ranks_the_deterministic_edge_first() -> None:
    """The default. Surprise would put `speculative` first on a query whose
    answer is the deterministically-matched edge into a hub."""
    hits = [
        _hit("canonical", "EXTRACTED", score=1.0),
        _hit("speculative", "AMBIGUOUS", score=6.0),
    ]
    assert _sort_as_module_does(hits, discovery=False)[0] == "canonical"


def test_confidence_tiers_order_extracted_then_inferred_then_ambiguous() -> None:
    hits = [
        _hit("amb", "AMBIGUOUS", score=9.0),
        _hit("inf", "INFERRED", score=5.0),
        _hit("ext", "EXTRACTED", score=1.0),
    ]
    assert _sort_as_module_does(hits, discovery=False) == ["ext", "inf", "amb"]


def test_surprise_still_breaks_ties_inside_a_tier() -> None:
    hits = [
        _hit("low", "EXTRACTED", score=1.0),
        _hit("high", "EXTRACTED", score=4.0),
    ]
    assert _sort_as_module_does(hits, discovery=False) == ["high", "low"]


def test_ordering_is_deterministic_on_a_full_tie() -> None:
    """MCP surfaces retriever_scores; jitter on identical queries reads as a
    ranking change that never happened."""
    hits = [_hit("b", "EXTRACTED", 1.0), _hit("a", "EXTRACTED", 1.0)]
    assert _sort_as_module_does(hits, discovery=False) == ["a", "b"]
    assert _sort_as_module_does(hits, discovery=True) == ["a", "b"]
