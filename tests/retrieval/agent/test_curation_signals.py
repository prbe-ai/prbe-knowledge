"""E5: the gatherer is told how to judge VALIDITY, and given the two facts it
needs to do it.

Relevance and validity are different questions, and the prompt only ever asked
the first one. These pin the curation rule, the `age_days` / `origin` render it
reads, and the surprise ordering that used to apply to every query.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from engine.retrieval.agent.prompt import build_system_prompt
from engine.retrieval.agent.tools import _age_days, _hit_origin, _hit_to_chunk_dict


class _Hit:
    def __init__(self, updated_at=None, metadata=None) -> None:
        self.chunk_id = "c1"
        self.doc_id = "slack:d1"
        self.doc_version = 3
        self.source_system = "slack"
        self.source_url = "https://example.test/d1"
        self.title = "a title"
        self.content = "body"
        self.score = 1.0
        self.created_at = datetime(2026, 1, 1, tzinfo=UTC)
        self.updated_at = updated_at
        self.author_id = None
        self.metadata = metadata


# ------------------------------------------------------------------ the rule

def test_the_prompt_tells_the_gatherer_to_keep_both_sides_of_a_disagreement() -> None:
    prompt = build_system_prompt(datetime.now(UTC))
    assert "RELEVANCE AND VALIDITY ARE DIFFERENT QUESTIONS" in prompt
    # Keep both, don't pick a winner.
    assert "keep BOTH" in prompt
    # Channel order carries no truth.
    assert "CHANNEL ORDER IS NOT A VALIDITY SIGNAL" in prompt
    # A generated summary is weaker than the human note it summarizes.
    assert "weaker evidence than the human note" in prompt


# ---------------------------------------------------------------- age_days

def test_age_days_is_whole_days_from_the_documents_own_clock() -> None:
    assert _age_days(datetime.now(UTC) - timedelta(days=7)) == 7
    assert _age_days((datetime.now(UTC) - timedelta(days=2)).isoformat()) == 2


def test_an_unknown_clock_renders_as_absent_never_as_zero() -> None:
    """"Unknown age" and "written today" are opposite claims. Defaulting the
    first to the second is how a stale document reads as current."""
    assert _age_days(None) is None
    assert _age_days("not a timestamp") is None
    rendered = _hit_to_chunk_dict(_Hit(updated_at=None), "vector")
    assert "age_days" not in rendered


def test_age_days_rides_on_every_rendered_hit() -> None:
    rendered = _hit_to_chunk_dict(_Hit(updated_at=datetime.now(UTC) - timedelta(days=30)), "vector")
    assert rendered["age_days"] == 30


# ------------------------------------------------------------------ origin

def test_origin_is_carried_when_the_document_says_and_omitted_when_it_does_not() -> None:
    assert _hit_origin(_Hit(metadata={"origin": "generated"})) == "generated"
    assert _hit_origin(_Hit(metadata={"origin": "human"})) == "human"
    # Never invent an attribution.
    assert _hit_origin(_Hit(metadata={})) is None
    assert _hit_origin(_Hit(metadata=None)) is None
    assert _hit_origin(_Hit(metadata={"origin": "something-else"})) is None
    assert "origin" not in _hit_to_chunk_dict(_Hit(), "vector")
