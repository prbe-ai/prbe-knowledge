"""Unit tests for graph-retriever confidence filtering.

The bundle-construction tests that previously lived here were removed when
QueryBundle was dropped from QueryResponse (the polymorphic search-result
shape replaced bundles with `QueryDocumentResult.matched_via` provenance
on per-doc results -- see PR feat/polymorphic-search-results).
"""

from __future__ import annotations

from services.retrieval.retrievers.graph import (
    CODE_GRAPH_LABELS,
    passes_confidence_filter,
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


def test_code_graph_labels_constant_completeness() -> None:
    """Constants stay in sync with the spec's symbol NodeLabel set."""
    assert {
        "Function",
        "Method",
        "Class",
        "Module",
        "Symbol",
    } == CODE_GRAPH_LABELS
