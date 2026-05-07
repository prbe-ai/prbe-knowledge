"""Unit tests for the token-based chunk overlap dedup helper.

Note: ``_skip_chunk_overlap_tokens`` is exercised against the real
chunker (`chunk_text`) to prove that reassembly recovers the original
body for multi-chunk inputs. That's the test that matters; the unit
cases below are scaffolding.
"""

from __future__ import annotations

from services.ingestion.chunker import _enc, chunk_text
from services.ingestion.normalizer import _skip_chunk_overlap_tokens


def test_no_token_overlap_returns_curr_unchanged() -> None:
    out = _skip_chunk_overlap_tokens(
        "completely unrelated text one",
        "completely unrelated text two",
    )
    assert out == "completely unrelated text two"


def test_real_chunker_pair_strips_overlap_exactly() -> None:
    """The end-to-end test: chunk a multi-chunk body, reassemble with the
    helper, confirm it equals the original (modulo tokenizer round-trip)."""
    enc = _enc()
    edges = [f"  node_{i:03d} --> node_{i + 1:03d}" for i in range(200)]
    original = "header line\n" + "\n".join(edges) + "\nfooter line\n"

    pieces = chunk_text(original)
    assert len(pieces) >= 2, f"expected multi-chunk text, got {len(pieces)}"

    reassembled = pieces[0].content
    for piece in pieces[1:]:
        reassembled += _skip_chunk_overlap_tokens(reassembled, piece.content)

    expected = enc.decode(enc.encode(original, disallowed_special=()))
    assert reassembled == expected


def test_production_shape_seam_no_crash() -> None:
    """The exact strings that broke the character-based dedup. These two
    weren't produced by the same chunker run, so token overlap may be 0.
    Just verify the function returns a sane string and doesn't crash."""
    prev = "prbe_knowledge -->|one-way|"
    curr = "be_cc_tap_plugin\n  prbe_dashboard -->|one-way| prbe_codex_tap_plugin"
    out = _skip_chunk_overlap_tokens(prev, curr)
    # Output must be a suffix of curr (possibly equal to curr).
    assert curr.endswith(out)


def test_synthesized_token_overlap_strips() -> None:
    """Construct a clear N-token overlap and prove it's stripped."""
    enc = _enc()
    full = (
        "alpha bravo charlie delta echo foxtrot golf hotel india juliet "
        "kilo lima mike november oscar papa quebec romeo sierra tango"
    )
    tokens = enc.encode(full, disallowed_special=())
    assert len(tokens) >= 8
    overlap_n = 4
    mid = len(tokens) // 2
    prev = enc.decode(tokens[: mid + overlap_n])
    curr = enc.decode(tokens[mid:])
    out = _skip_chunk_overlap_tokens(prev, curr)
    expected = enc.decode(tokens[mid + overlap_n:])
    assert out == expected
