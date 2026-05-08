"""Unit tests for classifier failure semantics: LLM-failure vs all-coincidence."""

from __future__ import annotations

from pathlib import Path

import pytest

from services.ingestion.code_graph.cross_repo_deps import (
    CandidateMatch,
    ClassifierUnavailable,
    classify_with_llm,
)


@pytest.mark.asyncio
async def test_classify_returns_empty_list_when_all_coincidence(monkeypatch) -> None:
    """When the LLM successfully says every candidate is COINCIDENCE,
    classify_with_llm returns [] (legit empty), NOT raises."""

    async def fake_call_classifier_llm(**kwargs):
        # Successful classify, all coincidences (real=False)
        return [
            {"number": 1, "real": False, "reason": "doc cross-link"},
            {"number": 2, "real": False, "reason": "user-facing setup"},
        ]

    monkeypatch.setattr(
        "services.ingestion.code_graph.cross_repo_deps._call_classifier_llm",
        fake_call_classifier_llm,
    )

    result = await classify_with_llm(
        source_repo="prbe-ai/example",
        candidates=[
            CandidateMatch(file_path="a.py", line_number=1, snippet="x", candidate_target="prbe-ai/other-1"),
            CandidateMatch(file_path="b.py", line_number=1, snippet="y", candidate_target="prbe-ai/other-2"),
        ],
        target_dir=Path("/tmp"),
    )
    assert result == []


@pytest.mark.asyncio
async def test_classify_raises_when_llm_returns_none(monkeypatch) -> None:
    """When _call_classifier_llm returns None (LLM failure), classify_with_llm
    raises ClassifierUnavailable instead of returning []."""

    async def fake_call_classifier_llm(**kwargs):
        return None

    monkeypatch.setattr(
        "services.ingestion.code_graph.cross_repo_deps._call_classifier_llm",
        fake_call_classifier_llm,
    )

    with pytest.raises(ClassifierUnavailable):
        await classify_with_llm(
            source_repo="prbe-ai/example",
            candidates=[
                CandidateMatch(
                    file_path="a.py",
                    line_number=1,
                    snippet="x",
                    candidate_target="prbe-ai/other",
                ),
            ],
            target_dir=Path("/tmp"),
        )


@pytest.mark.asyncio
async def test_classify_returns_empty_for_no_candidates() -> None:
    """The empty-input early-return path; no LLM call needed."""
    result = await classify_with_llm(
        source_repo="prbe-ai/example",
        candidates=[],
        target_dir=Path("/tmp"),
    )
    assert result == []
