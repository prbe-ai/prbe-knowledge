"""NormalizeOutcome must propagate requires_investigation from the
NormalizationResult so the worker can dispatch investigations."""
from __future__ import annotations

from services.ingestion.normalizer import NormalizeOutcome


def test_outcome_default_flag_is_false() -> None:
    o = NormalizeOutcome(doc_ids=[], chunk_count=0, failed_chunk_count=0)
    assert o.requires_investigation is False


def test_outcome_flag_settable_true() -> None:
    o = NormalizeOutcome(
        doc_ids=["pd:incident:T-001"], chunk_count=1, failed_chunk_count=0,
        requires_investigation=True,
    )
    assert o.requires_investigation is True
