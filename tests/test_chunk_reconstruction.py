"""Regression tests for strict standard-chunker source reconstruction."""

from __future__ import annotations

import random
import string
from itertools import pairwise

import pytest

from engine.ingest.chunker import chunk_text
from engine.shared.chunk_reconstruction import (
    DEFAULT_CHUNK_OVERLAP,
    chunk_encoding,
    reconstruct_chunk_text,
)


def _round_trip(text: str) -> str:
    enc = chunk_encoding()
    return enc.decode(enc.encode(text, disallowed_special=()))


def _exact_overlap_size(previous: str, current: str) -> int:
    enc = chunk_encoding()
    previous_tokens = enc.encode(previous, disallowed_special=())
    current_tokens = enc.encode(current, disallowed_special=())
    return next(
        (
            size
            for size in range(DEFAULT_CHUNK_OVERLAP, 0, -1)
            if previous_tokens[-size:] == current_tokens[:size]
        ),
        0,
    )


def test_unicode_replacement_boundaries_round_trip_exactly() -> None:
    original = "\n".join(
        f"😀 café 漢字 row {index}: " + ("x" * ((index % 17) + 1))
        for index in range(1000)
    )
    pieces = chunk_text(original)
    assert len(pieces) == 33
    assert sum(
        _exact_overlap_size(previous.content, current.content) == 0
        for previous, current in pairwise(pieces)
    ) == 16

    reconstructed = reconstruct_chunk_text(piece.content for piece in pieces)

    assert reconstructed == _round_trip(original)
    assert "\ufffd" not in reconstructed


def test_deterministic_ascii_punctuation_fuzz_round_trips() -> None:
    rng = random.Random(20260803)
    alphabet = string.ascii_letters + string.digits + string.punctuation + " "
    original = "\n".join(
        "".join(rng.choice(alphabet) for _ in range(rng.randint(20, 120)))
        for _ in range(2000)
    )
    pieces = chunk_text(original)
    # Pin that this corpus exercises the character-border fallback, not only
    # the direct 64-token fast path.
    assert any(
        _exact_overlap_size(previous.content, current.content) == 0
        for previous, current in pairwise(pieces)
    )

    reconstructed = reconstruct_chunk_text(piece.content for piece in pieces)

    assert reconstructed == _round_trip(original)


def test_seeded_unicode_property_repro_round_trips_short_reencoded_seams() -> None:
    rng = random.Random(0)
    alphabet = [*"abc xyz\n", "😀", "🚀", "é", "漢", "字", "λ", "🧪", "\u0301"]
    original = "".join(rng.choice(alphabet) for _ in range(18_000))
    pieces = chunk_text(original)
    assert len(pieces) == 48
    assert any(
        0 < _exact_overlap_size(previous.content, current.content) < DEFAULT_CHUNK_OVERLAP
        for previous, current in pairwise(pieces)
    )

    reconstructed = reconstruct_chunk_text(piece.content for piece in pieces)

    assert reconstructed == _round_trip(original)


def test_short_repeated_word_boundary_keeps_legacy_separator() -> None:
    assert reconstruct_chunk_text(["alpha", "alpha beta"]) == "alpha\n\nalpha beta"


@pytest.mark.parametrize("overlap_tokens", range(1, DEFAULT_CHUNK_OVERLAP))
def test_disabled_overlap_preserves_one_to_63_token_prechunked_boundaries(
    overlap_tokens: int,
) -> None:
    enc = chunk_encoding()
    tokens = enc.encode(
        " ".join(f"legacy_token_{index}" for index in range(300)),
        disallowed_special=(),
    )
    previous = enc.decode(tokens[:100])
    current = enc.decode(tokens[100 - overlap_tokens : 180])

    assert reconstruct_chunk_text(
        [previous, current],
        expected_overlap_tokens=None,
    ) == f"{previous}\n\n{current}"
