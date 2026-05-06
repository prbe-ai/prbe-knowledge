"""Unit tests for the cross-file qualifier promotion."""

from __future__ import annotations

from services.ingestion.code_graph.qualifier import promote_single_match
from services.ingestion.code_graph.types import CodeEdge, ExtractResult, Symbol
from shared.constants import EdgeType, NodeLabel


def _sym(qname: str, kind: NodeLabel = NodeLabel.FUNCTION) -> Symbol:
    return Symbol(
        qualified_name=qname,
        kind=kind,
        file_path="x.py",
        def_line=1,
        end_line=1,
        source_snippet="",
    )


def _edge(
    to_qname: str,
    *,
    candidates: list[str] | None = None,
    ambiguous: bool = True,
) -> CodeEdge:
    return CodeEdge(
        edge_type=EdgeType.CALLS,
        from_qname="caller",
        to_qname=to_qname,
        ambiguous=ambiguous,
        target_candidates=candidates if candidates is not None else [to_qname],
    )


def test_single_match_promotes_to_resolved() -> None:
    r1 = ExtractResult(symbols=[_sym("foo.bar.baz")])
    r2 = ExtractResult(edges=[_edge("baz")])
    promote_single_match([r1, r2])
    edge = r2.edges[0]
    assert edge.ambiguous is False
    assert edge.to_qname == "foo.bar.baz"
    assert edge.target_candidates == []


def test_zero_match_stays_ambiguous() -> None:
    r1 = ExtractResult(symbols=[_sym("other.name")])
    r2 = ExtractResult(edges=[_edge("nonexistent")])
    promote_single_match([r1, r2])
    edge = r2.edges[0]
    assert edge.ambiguous is True
    assert edge.to_qname == "nonexistent"


def test_multi_match_stays_ambiguous_with_refreshed_candidates() -> None:
    """When the qualifier sees multiple symbols matching the candidate's
    tail, it leaves AMBIGUOUS but refreshes target_candidates with the
    full match list so PR-B's promoter has better signal."""
    r1 = ExtractResult(
        symbols=[
            _sym("foo.bar.baz"),
            _sym("other.module.baz"),
        ]
    )
    r2 = ExtractResult(edges=[_edge("baz")])
    promote_single_match([r1, r2])
    edge = r2.edges[0]
    assert edge.ambiguous is True
    assert sorted(edge.target_candidates) == sorted(
        ["foo.bar.baz", "other.module.baz"]
    )


def test_does_not_touch_already_resolved_edges() -> None:
    r1 = ExtractResult(symbols=[_sym("foo.bar.baz")])
    r2 = ExtractResult(
        edges=[
            CodeEdge(
                edge_type=EdgeType.CALLS,
                from_qname="x",
                to_qname="explicit.target",
                ambiguous=False,
            )
        ]
    )
    promote_single_match([r1, r2])
    assert r2.edges[0].to_qname == "explicit.target"
    assert r2.edges[0].ambiguous is False


def test_handles_empty_inputs() -> None:
    assert promote_single_match([]) == []
    assert promote_single_match([ExtractResult()]) == [ExtractResult()]
