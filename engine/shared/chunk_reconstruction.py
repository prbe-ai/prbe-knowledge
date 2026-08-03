"""Token-aware reconstruction of persisted document chunks.

Standard ingestion stores overlapping token windows so retrieval has enough
context around chunk boundaries. Full-source readers must remove that overlap,
while pre-chunked sources whose rows do not overlap retain the historic
double-newline separator between rows.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import tiktoken

DEFAULT_CHUNK_OVERLAP = 64
_MAX_BOUNDARY_REPLACEMENT_CHARS = 3
_REPLACEMENT_CHAR = "\ufffd"

_encoding: tiktoken.Encoding | None = None


@dataclass(frozen=True, slots=True)
class ChunkLineSpan:
    """One raw chunk's inclusive line range in the reconstructed document."""

    line_start: int
    line_end: int


def chunk_encoding() -> tiktoken.Encoding:
    """Return the tokenizer shared by chunking and reconstruction."""
    global _encoding
    if _encoding is None:
        _encoding = tiktoken.get_encoding("cl100k_base")
    return _encoding


def _split_chunk_overlap_tokens(
    previous: str,
    current: str,
    *,
    max_overlap_tokens: int = DEFAULT_CHUNK_OVERLAP,
) -> tuple[str | None, str]:
    """Return the decoded overlap prefix, if any, and current's new suffix."""
    enc = chunk_encoding()
    previous_tokens = enc.encode(previous, disallowed_special=())
    current_tokens = enc.encode(current, disallowed_special=())
    upper = min(
        len(previous_tokens),
        len(current_tokens),
        max_overlap_tokens,
    )
    for overlap_size in range(upper, 0, -1):
        if previous_tokens[-overlap_size:] == current_tokens[:overlap_size]:
            return (
                enc.decode(current_tokens[:overlap_size]),
                enc.decode(current_tokens[overlap_size:]),
            )
    return None, current


def _suffix_prefix_border_lengths(previous: str, current: str) -> list[int]:
    """Return character-border lengths shared by previous's end/current's start.

    Uses the KMP prefix table so even large pre-chunked rows stay linear rather
    than repeatedly slicing and comparing every possible border.
    """
    max_length = min(len(previous), len(current))
    pattern = current[:max_length]
    if not pattern:
        return []

    prefix_lengths = [0] * len(pattern)
    for index in range(1, len(pattern)):
        matched = prefix_lengths[index - 1]
        while matched and pattern[index] != pattern[matched]:
            matched = prefix_lengths[matched - 1]
        if pattern[index] == pattern[matched]:
            matched += 1
        prefix_lengths[index] = matched

    matched = 0
    suffix = previous[-max_length:]
    for index, char in enumerate(suffix):
        while matched and char != pattern[matched]:
            matched = prefix_lengths[matched - 1]
        if char == pattern[matched]:
            matched += 1
        if matched == len(pattern) and index != len(suffix) - 1:
            matched = prefix_lengths[matched - 1]

    borders: list[int] = []
    while matched:
        borders.append(matched)
        matched = prefix_lengths[matched - 1]
    return borders


def _split_standard_chunk_overlap(
    previous: str,
    current: str,
    *,
    expected_overlap_tokens: int,
) -> tuple[str | None, str, int]:
    """Split one standard-chunker seam without matching legacy coincidences.

    A direct full-token match is the common path. Re-encoding decoded chunks
    can lose the signature when a token window bisects UTF-8 or punctuation,
    so a bounded character-border fallback handles seams whose consumed current
    prefix still encodes to approximately the configured overlap. Short exact
    re-encode matches do not veto this fallback: real
    Unicode seams can collapse to as little as one matching token after lossy
    boundary decoding. Callers must disable overlap detection for known
    pre-chunked provenance instead of inferring provenance from content.

    Returns ``(overlap_prefix, new_suffix, trailing_previous_chars_to_trim)``.
    Trimming repairs a trailing U+FFFD emitted when the previous chunk ended in
    the middle of a Unicode character; the current chunk supplies the intact
    character in ``new_suffix``.
    """
    if expected_overlap_tokens < 1:
        raise ValueError("expected_overlap_tokens must be >= 1")

    enc = chunk_encoding()
    previous_tokens = enc.encode(previous, disallowed_special=())
    current_tokens = enc.encode(current, disallowed_special=())
    if (
        len(previous_tokens) < expected_overlap_tokens
        or len(current_tokens) < expected_overlap_tokens
    ):
        return None, current, 0

    exact_overlap = next(
        (
            overlap_size
            for overlap_size in range(expected_overlap_tokens, 0, -1)
            if previous_tokens[-overlap_size:] == current_tokens[:overlap_size]
        ),
        0,
    )
    if exact_overlap == expected_overlap_tokens:
        return (
            enc.decode(current_tokens[:expected_overlap_tokens]),
            enc.decode(current_tokens[expected_overlap_tokens:]),
            0,
        )
    trailing_replacements = len(previous) - len(previous.rstrip(_REPLACEMENT_CHAR))
    leading_replacements = len(current) - len(current.lstrip(_REPLACEMENT_CHAR))
    if (
        trailing_replacements > _MAX_BOUNDARY_REPLACEMENT_CHARS
        or leading_replacements > _MAX_BOUNDARY_REPLACEMENT_CHARS
    ):
        return None, current, 0

    previous_for_match = (
        previous[:-trailing_replacements] if trailing_replacements else previous
    )
    current_for_match = current[leading_replacements:]
    target_chars = len(enc.decode(current_tokens[:expected_overlap_tokens]))
    candidates: list[tuple[int, int, int, int]] = []
    for border_length in _suffix_prefix_border_lengths(
        previous_for_match,
        current_for_match,
    ):
        consumed_chars = leading_replacements + border_length
        consumed_tokens = len(
            enc.encode(current[:consumed_chars], disallowed_special=())
        )
        if (
            expected_overlap_tokens - _MAX_BOUNDARY_REPLACEMENT_CHARS
            <= consumed_tokens
            <= expected_overlap_tokens
        ):
            candidates.append(
                (
                    abs(consumed_chars - target_chars),
                    abs(consumed_tokens - expected_overlap_tokens),
                    -border_length,
                    border_length,
                )
            )

    if not candidates:
        return None, current, 0
    border_length = min(candidates)[-1]
    consumed_chars = leading_replacements + border_length
    return (
        current[:consumed_chars],
        current[consumed_chars:],
        trailing_replacements,
    )


def strip_chunk_overlap_tokens(
    previous: str,
    current: str,
    *,
    max_overlap_tokens: int = DEFAULT_CHUNK_OVERLAP,
) -> str:
    """Return ``current`` without its leading token overlap with ``previous``.

    Character-level dedup is unreliable because token-window boundaries can
    decode asymmetrically in the middle of a word. Compare encoded token
    suffixes/prefixes instead and remove the longest overlap up to the standard
    chunker overlap. Unrelated or manually pre-chunked rows are returned
    unchanged.
    """
    _, suffix = _split_chunk_overlap_tokens(
        previous,
        current,
        max_overlap_tokens=max_overlap_tokens,
    )
    return suffix


def reconstruct_chunk_text_with_spans(
    contents: Iterable[str],
    *,
    non_overlap_separator: str = "\n\n",
    expected_overlap_tokens: int | None = DEFAULT_CHUNK_OVERLAP,
) -> tuple[str, list[ChunkLineSpan]]:
    """Reassemble chunks and map each raw row into reconstructed line space."""
    iterator = iter(contents)
    try:
        previous = next(iterator)
    except StopIteration:
        return "", []

    first_line_count = len(previous.splitlines()) or 1
    spans = [ChunkLineSpan(line_start=1, line_end=first_line_count)]
    parts = [previous]
    newline_count = previous.count("\n")

    for current in iterator:
        if expected_overlap_tokens is None:
            overlap_prefix, suffix, trim_previous_chars = None, current, 0
        else:
            overlap_prefix, suffix, trim_previous_chars = _split_standard_chunk_overlap(
                previous,
                current,
                expected_overlap_tokens=expected_overlap_tokens,
            )
        if overlap_prefix is None:
            parts.extend((non_overlap_separator, current))
            line_start = newline_count + non_overlap_separator.count("\n") + 1
            newline_count += non_overlap_separator.count("\n") + current.count("\n")
        else:
            if trim_previous_chars:
                parts[-1] = parts[-1][:-trim_previous_chars]
            parts.append(suffix)
            line_start = newline_count + 1 - overlap_prefix.count("\n")
            newline_count += suffix.count("\n")

        line_count = len(current.splitlines()) or 1
        spans.append(
            ChunkLineSpan(
                line_start=line_start,
                line_end=line_start + line_count - 1,
            )
        )
        previous = current

    return "".join(parts), spans


def reconstruct_chunk_text(
    contents: Iterable[str],
    *,
    non_overlap_separator: str = "\n\n",
    expected_overlap_tokens: int | None = DEFAULT_CHUNK_OVERLAP,
) -> str:
    """Reassemble adjacent chunk bodies without duplicating token overlap.

    Rows produced by the standard chunker concatenate directly after their
    shared token window is removed. When two adjacent rows have no token
    overlap, retain ``non_overlap_separator`` so pre-chunked and legacy rows
    preserve the source endpoints' historic rendering.
    """
    text, _ = reconstruct_chunk_text_with_spans(
        contents,
        non_overlap_separator=non_overlap_separator,
        expected_overlap_tokens=expected_overlap_tokens,
    )
    return text
