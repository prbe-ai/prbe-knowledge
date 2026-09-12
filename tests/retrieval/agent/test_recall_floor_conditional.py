"""E4: the recall floor is a DECISION, not a reflex.

The floor supplies ~88% of the chunks a consumer receives, which makes the
gatherer's curation nearly invisible in the delivered result. These tests pin
the three things that change that: when the floor fires, what it tags the
chunks it appends, and the two counts that let an A/B tell a well-curating
gatherer from one that was never shown the candidates.
"""

from __future__ import annotations

import pytest

from engine.retrieval.agent.loop import (
    RecallFloorOutcome,
    _backfill_recall_floor,
    _recall_floor_should_backfill,
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


def _gathered(n_chunks: int, confidence: str | None = "high") -> GathererOutput:
    notes = GathererNotes()
    # None models a `_coerce_lenient` recovery where the grade never survived
    # the parse -- the shape the fail-open rule exists for.
    notes.confidence = confidence  # type: ignore[assignment]
    return GathererOutput(
        chunks=[_chunk(f"slack:doc{i}") for i in range(n_chunks)],
        gatherer_notes=notes,
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

@pytest.mark.parametrize(
    ("confidence", "n_chunks", "expected"),
    [
        ("high", 8, False),     # confident AND substantial -> trust the curation
        ("medium", 8, True),    # not high -> top it up
        ("low", 8, True),
        ("high", 2, True),      # confident but thin -> top it up
        (None, 8, True),        # ABSENT -> a parse recovery, fail OPEN
    ],
)
def test_conditional_mode_backfills_only_when_the_answer_is_thin(
    confidence: str | None, n_chunks: int, expected: bool
) -> None:
    should, reason = _recall_floor_should_backfill(
        _gathered(n_chunks, confidence), mode="conditional"
    )
    assert should is expected, reason


def test_absent_confidence_fails_open_rather_than_reading_as_high() -> None:
    """A missing confidence is a SCHEMA recovery, not a verdict.

    ~2% of emits come back off-schema and are rebuilt by `_coerce_lenient`;
    the grade is the field most often lost. Reading that as "high" would let
    the thinnest answers be exactly the ones that skip the floor.
    """
    should, reason = _recall_floor_should_backfill(
        _gathered(20, confidence=None), mode="conditional"
    )
    assert should is True
    assert reason == "confidence_absent"


def test_always_mode_is_unchanged_by_any_of_it() -> None:
    """REGRESSION. `always` is the shipped behaviour and the default; a
    confident, substantial answer still gets topped up under it."""
    should, reason = _recall_floor_should_backfill(
        _gathered(8, "high"), mode="always"
    )
    assert should is True
    assert reason == "mode_always"


# --------------------------------------------------------------- the tagging

def test_backfilled_chunks_are_tagged_recall_floor_not_the_channel() -> None:
    """The tag is the whole point: a consumer must be able to tell a passage a
    model read and chose from one that merely ranked well in the raw pool."""
    gathered = _gathered(1, "high")
    outcome = _backfill_recall_floor(
        gathered, _pool("slack:new1", "slack:new2"), mode="always"
    )
    assert outcome.appended == 2
    appended = [c for c in gathered.chunks if c.harness_appended]
    assert len(appended) == 2
    for chunk in appended:
        assert chunk.matched_via == ["recall_floor"], (
            "a backfilled chunk claiming `vector` would assert that a model "
            "surfaced it on purpose, which is exactly what did not happen"
        )
        assert chunk.why_relevant == ""
    # The gatherer's own chunk is untouched.
    curated = [c for c in gathered.chunks if not c.harness_appended]
    assert curated[0].matched_via == ["vector"]


def test_conditional_mode_skips_and_says_why() -> None:
    gathered = _gathered(8, "high")
    before = len(gathered.chunks)
    outcome = _backfill_recall_floor(
        gathered, _pool("slack:new1", "slack:new2"), mode="conditional"
    )
    assert outcome == RecallFloorOutcome(
        appended=0, reason="gatherer_sufficient", rejected=0, unexamined=2
    )
    assert len(gathered.chunks) == before, "skipping must not mutate the result"


# --------------------------------------------------------------- the counts

def test_rejected_and_unexamined_are_counted_separately() -> None:
    """The A/B is graded on these, not on backfill share.

    A pool doc the gatherer SAW and declined is curation working. A pool doc
    that never fit the render budget is a budget ceiling no prompt can lift.
    Both inflate a backfill count identically.
    """
    gathered = _gathered(1, "high")
    outcome = _backfill_recall_floor(
        gathered,
        _pool("slack:seen_and_dropped", "slack:never_shown"),
        mode="conditional",
        examined_doc_ids={"slack:doc0", "slack:seen_and_dropped"},
    )
    assert outcome.rejected == 1
    assert outcome.unexamined == 1


def test_without_a_rendered_set_everything_reads_as_unexamined() -> None:
    """No render happened (the id-lookup short circuit), so nothing was
    rejected. Claiming rejections there would credit curation that never ran."""
    gathered = _gathered(1, "high")
    outcome = _backfill_recall_floor(
        gathered, _pool("slack:a", "slack:b"), mode="always", examined_doc_ids=None
    )
    assert outcome.rejected == 0
    assert outcome.unexamined == 2


def test_floor_already_met_is_distinguishable_from_a_conditional_skip() -> None:
    gathered = _gathered(12, "medium")
    outcome = _backfill_recall_floor(gathered, _pool("slack:new"), mode="conditional")
    assert outcome.appended == 0
    assert outcome.reason == "floor_already_met"
