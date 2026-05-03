"""Unit tests for graph-retriever confidence filtering + bundle construction."""

from __future__ import annotations

from datetime import UTC, datetime

from services.retrieval.retrievers.graph import (
    CODE_GRAPH_LABELS,
    GraphHit,
    passes_confidence_filter,
)
from services.retrieval.search_pipeline import _build_bundles


def _hit(
    *,
    chunk_id: str = "c1",
    via_entity: str = "org/repo:Foo",
    via_label: str = "Class",
    confidence: str | None = "EXTRACTED",
    edge_type: str | None = "DEFINED_IN",
) -> GraphHit:
    now = datetime.now(UTC)
    return GraphHit(
        chunk_id=chunk_id,
        doc_id="doc1",
        doc_version=1,
        source_system="code_graph",
        source_url="",
        title=None,
        content="",
        created_at=now,
        updated_at=now,
        score=1.0,
        via_entity=via_entity,
        edge_type=edge_type,
        confidence=confidence,
        via_label=via_label,
    )


# ---- confidence filter ----------------------------------------------------


def test_passes_confidence_filter_default_drops_ambiguous() -> None:
    """Default min_confidence='INFERRED' drops AMBIGUOUS only."""
    assert passes_confidence_filter("EXTRACTED", "INFERRED") is True
    assert passes_confidence_filter("INFERRED", "INFERRED") is True
    assert passes_confidence_filter("AMBIGUOUS", "INFERRED") is False


def test_passes_confidence_filter_extracted_only() -> None:
    """Strict callers pass 'EXTRACTED' to drop both INFERRED and AMBIGUOUS."""
    assert passes_confidence_filter("EXTRACTED", "EXTRACTED") is True
    assert passes_confidence_filter("INFERRED", "EXTRACTED") is False
    assert passes_confidence_filter("AMBIGUOUS", "EXTRACTED") is False


def test_passes_confidence_filter_none_accepts_everything() -> None:
    """Debug callers pass None to include all tiers."""
    assert passes_confidence_filter("EXTRACTED", None) is True
    assert passes_confidence_filter("INFERRED", None) is True
    assert passes_confidence_filter("AMBIGUOUS", None) is True


def test_passes_confidence_filter_treats_null_confidence_as_extracted() -> None:
    """Edges from before the migration come back NULL → treat as EXTRACTED."""
    assert passes_confidence_filter(None, "INFERRED") is True
    assert passes_confidence_filter(None, "EXTRACTED") is True


# ---- bundle construction --------------------------------------------------


def test_build_bundles_groups_by_via_entity() -> None:
    hits = [
        _hit(chunk_id="c1", via_entity="org/repo:Foo", via_label="Class"),
        _hit(chunk_id="c2", via_entity="org/repo:Foo", via_label="Class"),
        _hit(chunk_id="c3", via_entity="org/repo:Bar", via_label="Function"),
    ]
    bundles = _build_bundles(hits)
    assert bundles is not None
    assert len(bundles) == 2
    by_seed = {b.seed_entity: b for b in bundles}
    assert sorted(by_seed["org/repo:Foo"].related_chunk_ids) == ["c1", "c2"]
    assert by_seed["org/repo:Bar"].related_chunk_ids == ["c3"]


def test_build_bundles_returns_none_when_no_code_graph_seeds() -> None:
    """Non-code-graph seeds (Repo, Person, Channel) don't trigger bundles."""
    hits = [
        _hit(via_entity="prbe-backend", via_label="Repo"),
        _hit(via_entity="alice", via_label="Person"),
    ]
    assert _build_bundles(hits) is None


def test_build_bundles_returns_none_for_empty_hits() -> None:
    assert _build_bundles([]) is None


def test_build_bundles_confidence_breakdown_counts_tiers() -> None:
    hits = [
        _hit(chunk_id="c1", confidence="EXTRACTED"),
        _hit(chunk_id="c2", confidence="EXTRACTED"),
        _hit(chunk_id="c3", confidence="INFERRED"),
    ]
    bundles = _build_bundles(hits)
    assert bundles is not None
    assert bundles[0].confidence_breakdown == {
        "EXTRACTED": 2,
        "INFERRED": 1,
        "AMBIGUOUS": 0,
    }


def test_build_bundles_treats_null_confidence_as_extracted() -> None:
    """Pre-migration edges (NULL confidence) bucket into EXTRACTED."""
    hits = [
        _hit(chunk_id="c1", confidence=None),
        _hit(chunk_id="c2", confidence=None),
    ]
    bundles = _build_bundles(hits)
    assert bundles is not None
    assert bundles[0].confidence_breakdown["EXTRACTED"] == 2


def test_code_graph_labels_constant_completeness() -> None:
    """Constants stay in sync with the spec's symbol NodeLabel set."""
    assert {
        "Function",
        "Method",
        "Class",
        "Module",
        "Symbol",
    } == CODE_GRAPH_LABELS
