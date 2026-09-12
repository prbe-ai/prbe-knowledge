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
    """A stand-in for a retriever hit.

    NOTE the `origin` attribute and the test below that pins it against the
    REAL dataclasses. The first version of this file gave the fake a
    `metadata` dict, which no retriever hit has ever had -- so `_hit_origin`
    returned None for every hit in production while these tests stayed green.
    A fake is only evidence about production if production has the same shape.
    """

    def __init__(self, updated_at=None, origin=None) -> None:
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
        self.origin = origin


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
    assert _hit_origin(_Hit(origin="generated")) == "generated"
    assert _hit_origin(_Hit(origin="human")) == "human"
    # Never invent an attribution.
    assert _hit_origin(_Hit(origin=None)) is None
    assert _hit_origin(_Hit(origin="something-else")) is None
    assert "origin" not in _hit_to_chunk_dict(_Hit(), "vector")


def test_every_retriever_hit_actually_has_the_field_this_reads() -> None:
    """The guard for the bug this file shipped with.

    `_hit_origin` read `hit.metadata`, and NO retriever hit has ever had that
    attribute -- so it returned None for every hit in production while the
    test above passed, because the fake had the attribute the real objects
    lack. Asserting against the real dataclasses is the only version of this
    test that can fail when the field moves.
    """
    import dataclasses

    from engine.retrieval.retrievers.bm25 import BM25Hit
    from engine.retrieval.retrievers.graph import GraphHit
    from engine.retrieval.retrievers.vector import VectorHit

    for hit_type in (VectorHit, BM25Hit, GraphHit):
        names = {f.name for f in dataclasses.fields(hit_type)}
        assert "origin" in names, (
            f"{hit_type.__name__} has no `origin` field, so _hit_origin "
            f"returns None for every hit it produces"
        )


def test_every_channel_selects_origin_from_the_document() -> None:
    """A field on the dataclass that no query populates is the same bug one
    layer down: the attribute exists, it is always None, and nothing fails."""
    import inspect

    from engine.retrieval.retrievers import bm25, graph, vector

    for module in (vector, bm25, graph):
        source = inspect.getsource(module)
        assert "metadata->>'origin'" in source, (
            f"{module.__name__} never selects origin, so its hits carry None"
        )
        assert "origin_of(r)" in source, (
            f"{module.__name__} selects origin but never puts it on the hit"
        )
