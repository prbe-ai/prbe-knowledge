"""E4: the dedupe module, re-attached.

`dedupe()` has shipped with zero callers since the agentic cutover, so the
same Slack cross-post reached consumers three times. `dedupe_gathered` is the
version the loop calls: an exact content pass first (free), cosine only over
chunks whose embedding is loaded.
"""

from __future__ import annotations

from engine.retrieval.dedup import content_key, dedupe_gathered


class _Chunk:
    def __init__(self, chunk_id: str, content: str | None, tag: str = "") -> None:
        self.chunk_id = chunk_id
        self.content = content
        self.tag = tag


#: Long enough to clear MIN_EXACT_DEDUPE_CHARS; two docs carrying this
#: verbatim copied it from one another.
_PASSAGE = (
    "The reconciler compares our recomputed source content hash against the "
    "engine enumeration and re-enqueues anything that drifted, which is why a "
    "projection version bump re-pushes every document of that type without a "
    "bespoke backfill script having to exist at all for it. "
)


def test_rewrapped_text_is_the_same_content() -> None:
    """Byte equality would miss most real duplicates: the same passage arriving
    through two channels differs by rewrapping far more often than by a word."""
    assert content_key(_PASSAGE) == content_key(_PASSAGE.replace(" ", "  "))
    assert content_key(_PASSAGE) == content_key(_PASSAGE.upper())


def test_a_short_line_two_documents_share_is_not_a_duplicate() -> None:
    """This stage runs CROSS-DOC, so a false collapse costs a whole citable
    source. Two docs both saying "Approved." did not copy each other."""
    assert content_key("Approved.") is None
    kept, dropped = dedupe_gathered(
        [_Chunk("1", "Approved."), _Chunk("2", "Approved."), _Chunk("3", "Approved.")]
    )
    assert dropped == 0
    assert len(kept) == 3


def test_empty_content_is_not_a_duplicate_of_every_other_empty_chunk() -> None:
    assert content_key("") is None
    assert content_key("   \n ") is None
    kept, dropped = dedupe_gathered([_Chunk("1", ""), _Chunk("2", None)])
    assert dropped == 0
    assert len(kept) == 2


def test_the_first_copy_survives_which_is_the_curated_one() -> None:
    """Order here is DELIVERY order -- gatherer picks first, harness backfill
    after -- so keeping the first means the copy carrying a `why_relevant`
    line always beats the raw pool copy of the same passage."""
    curated = _Chunk("c1", _PASSAGE, tag="curated")
    backfilled = _Chunk("c2", _PASSAGE.replace(" ", "  "), tag="backfilled")
    kept, dropped = dedupe_gathered([curated, backfilled])
    assert dropped == 1
    assert [c.tag for c in kept] == ["curated"]


def test_cosine_drops_a_near_duplicate_only_when_both_embeddings_are_loaded() -> None:
    a = _Chunk("a", "first text")
    b = _Chunk("b", "different text entirely")
    embeddings = {"a": [1.0, 0.0, 0.0], "b": [0.999, 0.01, 0.0]}
    kept, dropped = dedupe_gathered([a, b], embeddings)
    assert dropped == 1
    assert [c.chunk_id for c in kept] == ["a"]

    # Same pair, no embeddings loaded: unjudgeable is not the same as duplicate.
    kept, dropped = dedupe_gathered([_Chunk("a", "first text"), _Chunk("b", "different text entirely")])
    assert dropped == 0


def test_distinct_content_is_left_alone() -> None:
    chunks = [_Chunk(str(i), _PASSAGE + f" Variant number {i}.") for i in range(5)]
    kept, dropped = dedupe_gathered(chunks)
    assert dropped == 0
    assert len(kept) == 5
