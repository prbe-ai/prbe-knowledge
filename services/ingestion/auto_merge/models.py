"""Pydantic schemas for the AutoMergeAnalyzer's Cerebras judge call.

`response_format=AutoMergeVerdict` is constrained-decoded by Cerebras
gpt-oss-120b via the LiteLLM proxy — same pattern as
services/retrieval/agent/models.py:EntityExtraction.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AutoMergeVerdict(BaseModel):
    """LLM's verdict on whether a primary entity is the same as one of N candidates.

    `primary_canonical_id` MUST be the canonical_id of one of the candidates
    when `verdict='duplicate'`; null when `verdict='unique'`. `confidence`
    levels gate the downstream action:
      - 'high' AND `auto_merge_execute` → POST entity-clusters merge
      - 'high' AND NOT `auto_merge_execute` → insert into entity_merge_suggestions
      - 'medium' or 'low' → insert into entity_merge_suggestions
      - 'unique' → no action
    """

    model_config = ConfigDict(extra="forbid")

    verdict: Literal["unique", "duplicate"]
    primary_canonical_id: str | None = Field(
        default=None,
        max_length=512,
        description=(
            "Canonical_id of the existing entity the new entity duplicates. "
            "MUST be one of the candidates' canonical_ids; null when verdict=unique."
        ),
    )
    confidence: Literal["high", "medium", "low"] | None = Field(
        default=None,
        description=(
            "Required when verdict=duplicate. 'high' = clear textual/property "
            "overlap (shared email, shared name, shared identifier). 'medium' = "
            "thematic overlap, plausible but not certain. 'low' = weak hint."
        ),
    )
    rationale: str = Field(
        ...,
        max_length=240,
        description="One-sentence reason (~30 words). Name the shared signal.",
    )
