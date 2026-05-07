"""Unit tests for the chunk-overlap dedup helper used to reassemble
document bodies from the chunks table.

The chunker emits 512-token windows with 64-token overlap, so adjacent
chunks share a tail/head region. ``_dedupe_chunk_overlap`` strips that
overlap so reassembled bodies (used by wiki render + diagram refresh)
don't render duplicated text — most visibly Mermaid edge lines that
appear twice when a chunk seam lands inside a code block.
"""

from __future__ import annotations

from services.ingestion.normalizer import (
    _MAX_CHUNK_OVERLAP_CHARS,
    _MIN_CHUNK_OVERLAP_CHARS,
    _dedupe_chunk_overlap,
)


def test_dedupe_no_overlap() -> None:
    prev = "hello world"
    curr = "goodnight moon"
    assert _dedupe_chunk_overlap(prev, curr) == "goodnight moon"


def test_dedupe_strips_real_overlap() -> None:
    prev = "abcdefghij1234567890"
    curr = "1234567890wxyz"
    assert _dedupe_chunk_overlap(prev, curr) == "wxyz"


def test_dedupe_below_min_keeps_curr() -> None:
    # 5-char shared region — below _MIN_CHUNK_OVERLAP_CHARS (=10).
    # The function must NOT strip it; small incidental matches are
    # likely to be coincidence (e.g. trailing newline + heading
    # repeat across unrelated chunks), not real chunker overlap.
    assert _MIN_CHUNK_OVERLAP_CHARS == 10
    prev = "aaa12345"
    curr = "12345bbb"
    assert _dedupe_chunk_overlap(prev, curr) == "12345bbb"


def test_dedupe_above_max_caps_search() -> None:
    # The shared region is 700 chars but the function only searches
    # up to _MAX_CHUNK_OVERLAP_CHARS (=600). Construct prev so its
    # last 700 chars equal the first 700 of curr; the function must
    # find a match at exactly 600 chars (the cap) and strip that
    # many — no more.
    assert _MAX_CHUNK_OVERLAP_CHARS == 600
    tail = "x" * 700
    prev = "PREFIX" + tail
    curr = tail + "SUFFIX"
    result = _dedupe_chunk_overlap(prev, curr)
    # Cap is enforced: at most 600 chars stripped.
    stripped = len(curr) - len(result)
    assert stripped <= _MAX_CHUNK_OVERLAP_CHARS
    assert stripped >= _MIN_CHUNK_OVERLAP_CHARS
    # Result still contains the unique tail of curr.
    assert result.endswith("SUFFIX")
    # And, concretely, the function strips exactly 600 (the cap)
    # because curr[:600] == prev[-600:] holds.
    assert stripped == _MAX_CHUNK_OVERLAP_CHARS


def test_dedupe_realistic_chunker_overlap() -> None:
    # Production shape: prev ENDS with the overlap region, curr
    # BEGINS with the same overlap region. This is exactly the
    # pattern that caused duplicate Mermaid arrows in the wiki
    # architecture diagram before the fix.
    overlap = (
        "  prbe_dashboard --> prbe_orchestrator\n"
        "  prbe_knowledge --> prbe_backend\n"
    )
    prev = "earlier content\n" + overlap
    curr = overlap + "later content"
    result = _dedupe_chunk_overlap(prev, curr)
    assert result == "later content"
