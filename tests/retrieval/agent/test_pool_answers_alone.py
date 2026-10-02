"""The pool's fused order answers ONLY when no selector answered.

A selector's answer (the gatherer's picks, Jev's ranking) ships exactly as it
is, however few documents it holds: nothing tops it up from the raw pool
(Richard, 2026-10-02). When nothing was selected -- the `floor` selector, Jev
unavailable, a gatherer that timed out -- the pool's fused order IS the
answer, capped at `_RESPONSE_DOCS` and tagged `recall_floor`.
"""

from __future__ import annotations

import pytest

from engine.retrieval.agent.loop import (
    _RESPONSE_DOCS,
    RecallFloorOutcome,
    _answer_from_pool,
    _pool_answers,
)
from engine.retrieval.agent.models import GatheredChunk, GathererNotes, GathererOutput


def _chunk(doc_id: str) -> GatheredChunk:
    return GatheredChunk(
        doc_id=doc_id,
        chunk_id=f"{doc_id}#0",
        content=f"body of {doc_id}",
        matched_via=["vector"],
        why_relevant="a model read this and chose it",
    )


def _gathered(n_chunks: int) -> GathererOutput:
    return GathererOutput(
        chunks=[_chunk(f"slack:doc{i}") for i in range(n_chunks)],
        gatherer_notes=GathererNotes(),
    )


def _pool(*doc_ids: str) -> dict:
    return {
        "sub_queries": [
            {
                "vector": [
                    {
                        "doc_id": d,
                        "chunk_id": f"{d}#0",
                        "content": f"pool body of {d}",
                        "title": d,
                        "source_system": "slack",
                        "source_url": f"https://example.test/{d}",
                    }
                    for d in doc_ids
                ]
            }
        ]
    }


# --------------------------------------------------------------- the decision

@pytest.mark.parametrize("n_chunks", [1, 3, 9])
def test_a_selector_answer_is_never_topped_up(n_chunks: int) -> None:
    """REGRESSION for the old recall floor, which appended pool docs until the
    answer held ten. A selector that picked three delivers three."""
    gathered = _gathered(n_chunks)
    outcome = _answer_from_pool(
        gathered, _pool(*(f"slack:new{i}" for i in range(20))), status="ok"
    )
    assert outcome.appended == 0
    assert outcome.reason == "selector_answered"
    assert [c.doc_id for c in gathered.chunks] == [f"slack:doc{i}" for i in range(n_chunks)]


def test_a_partial_jev_answer_is_not_topped_up_either() -> None:
    gathered = _gathered(4)
    outcome = _answer_from_pool(gathered, _pool("slack:new1", "slack:new2"), status="jev_partial")
    assert outcome.appended == 0
    assert len(gathered.chunks) == 4


def test_an_id_lookup_answer_is_its_pins_alone() -> None:
    """The adapter delivers the pins; the pool must not pad around them."""
    answers, reason = _pool_answers(_gathered(0), status="id_lookup_short_circuit")
    assert (answers, reason) == (False, "id_pins_answer")


@pytest.mark.parametrize("status", ["ok", "jev_unavailable", "loop_timeout", "schema_violation"])
def test_with_nothing_selected_the_pool_answers(status: str) -> None:
    """`floor` (status ok, nothing picked), Jev unavailable, a gatherer that
    failed: the pool's fused order is the whole answer, not a fill."""
    gathered = _gathered(0)
    outcome = _answer_from_pool(gathered, _pool("slack:a", "slack:b", "slack:c"), status=status)
    assert outcome.reason == "no_selection"
    assert outcome.appended == 3
    assert [c.doc_id for c in gathered.chunks] == ["slack:a", "slack:b", "slack:c"]


def test_the_pool_answer_is_capped_not_padded() -> None:
    gathered = _gathered(0)
    outcome = _answer_from_pool(
        gathered, _pool(*(f"slack:p{i}" for i in range(_RESPONSE_DOCS + 5))), status="ok"
    )
    assert outcome.appended == _RESPONSE_DOCS == len(gathered.chunks)


def test_an_empty_pool_answers_nothing() -> None:
    gathered = _gathered(0)
    outcome = _answer_from_pool(gathered, None, status="zero_recall_short_circuit")
    assert outcome == RecallFloorOutcome(appended=0, reason="no_selection", rejected=0, unexamined=0)
    assert gathered.chunks == []


# --------------------------------------------------------------- the tagging

def test_pool_answers_are_tagged_recall_floor_not_the_channel() -> None:
    """A consumer must be able to tell a passage a selector read and chose from
    one that merely ranked well in the raw pool."""
    gathered = _gathered(0)
    _answer_from_pool(gathered, _pool("slack:new1", "slack:new2"), status="jev_unavailable")
    assert len(gathered.chunks) == 2
    for chunk in gathered.chunks:
        assert chunk.harness_appended
        assert chunk.matched_via == ["recall_floor"]
        assert chunk.why_relevant == ""


# --------------------------------------------------------------- the counts

def test_rejected_and_unexamined_are_counted_separately() -> None:
    """A pool doc the selector SAW and declined is curation working; one it
    never saw is a budget or batch miss."""
    gathered = _gathered(1)
    outcome = _answer_from_pool(
        gathered,
        _pool("slack:seen_and_dropped", "slack:never_shown"),
        status="ok",
        examined_doc_ids={"slack:doc0", "slack:seen_and_dropped"},
    )
    assert outcome.rejected == 1
    assert outcome.unexamined == 1


def test_without_a_rendered_set_everything_reads_as_unexamined() -> None:
    """No render happened, so nothing was rejected. Claiming rejections there
    would credit curation that never ran."""
    gathered = _gathered(1)
    outcome = _answer_from_pool(
        gathered, _pool("slack:a", "slack:b"), status="ok", examined_doc_ids=None
    )
    assert outcome.rejected == 0
    assert outcome.unexamined == 2
