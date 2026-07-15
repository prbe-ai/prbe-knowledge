"""Tests for kb.code_graph.chunking.split_symbol_body.

Covers:
  - Small symbol -> 1 chunk, primary header preserved (existing behavior)
  - Oversized symbol -> N chunks, each <= MAX_SYMBOL_CHUNK_TOKENS,
    primary header on window 0, continuation on windows 1+
  - Empty body -> single header-only chunk (symbol stays in index)
  - Indivisible giant block -> falls back to token-window splitter
  - Real production-shape fixture: 30KB-ish module-scope class never
    emits a chunk above MAX_SYMBOL_CHUNK_TOKENS
"""

from engine.ingest.chunker import count_tokens
from engine.shared.constants import MAX_SYMBOL_CHUNK_TOKENS
from kb.code_graph.chunking import (
    _split_into_blocks,
    split_symbol_body,
)

PRIMARY = (
    "# prbe-ai/prbe-knowledge · services/foo.py · "
    "FooClass.bar (function) · L42-58\n"
)
CONT = "# FooClass.bar (cont.)\n"


def test_small_body_returns_single_chunk_with_primary_header() -> None:
    body = "def bar(self):\n    return self.x + 1"
    chunks = split_symbol_body(
        body,
        primary_header=PRIMARY,
        continuation_header=CONT,
        max_tokens=MAX_SYMBOL_CHUNK_TOKENS,
    )
    assert len(chunks) == 1
    assert chunks[0].startswith(PRIMARY)
    assert body in chunks[0]


def test_empty_body_returns_header_only_so_symbol_stays_in_index() -> None:
    chunks = split_symbol_body(
        "",
        primary_header=PRIMARY,
        continuation_header=CONT,
        max_tokens=MAX_SYMBOL_CHUNK_TOKENS,
    )
    assert len(chunks) == 1
    # rstrip the trailing newline; a header-only chunk still surfaces in
    # search even though it has no body.
    assert chunks[0] == PRIMARY.rstrip("\n")


def test_whitespace_only_body_returns_header_only() -> None:
    chunks = split_symbol_body(
        "   \n\n  \t  \n",
        primary_header=PRIMARY,
        continuation_header=CONT,
        max_tokens=MAX_SYMBOL_CHUNK_TOKENS,
    )
    assert len(chunks) == 1
    assert chunks[0] == PRIMARY.rstrip("\n")


def test_oversized_body_splits_with_correct_headers() -> None:
    # Build a body with many blank-line-separated blocks, each ~50 tokens,
    # totaling well over MAX_SYMBOL_CHUNK_TOKENS so it must split.
    block = "x = " + " ".join(["foo"] * 30) + "\n# comment line"
    body = "\n\n".join([block] * 40)
    assert count_tokens(body) > MAX_SYMBOL_CHUNK_TOKENS * 2

    chunks = split_symbol_body(
        body,
        primary_header=PRIMARY,
        continuation_header=CONT,
        max_tokens=MAX_SYMBOL_CHUNK_TOKENS,
    )
    assert len(chunks) >= 2

    assert chunks[0].startswith(PRIMARY)
    for c in chunks[1:]:
        assert c.startswith(CONT), (
            f"Window 1+ must use continuation header, got: {c[:80]!r}"
        )

    # Every chunk under the cap (with small slack for header-vs-body
    # accounting; a packer rounding error of a few tokens is acceptable
    # but emitting a 30KB chunk is not).
    for c in chunks:
        assert count_tokens(c) <= MAX_SYMBOL_CHUNK_TOKENS + 16, (
            f"chunk over budget: {count_tokens(c)} tokens"
        )


def test_indivisible_giant_block_falls_back_to_token_window() -> None:
    # Single line longer than MAX_SYMBOL_CHUNK_TOKENS — no blank-line
    # boundaries to split on. Must fall back to the prose token-window
    # splitter rather than emit one giant chunk.
    body = " ".join(["alpha"] * (MAX_SYMBOL_CHUNK_TOKENS * 2))
    chunks = split_symbol_body(
        body,
        primary_header=PRIMARY,
        continuation_header=CONT,
        max_tokens=MAX_SYMBOL_CHUNK_TOKENS,
    )
    assert len(chunks) >= 2
    for c in chunks:
        assert count_tokens(c) <= MAX_SYMBOL_CHUNK_TOKENS + 16


def test_continuation_header_carries_no_repo_path_tokens() -> None:
    # Regression guard: BM25 over-fire mitigation depends on continuation
    # headers NOT carrying repo/file/path tokens. If anyone ever changes
    # the production caller to pass primary_header twice, this test fails.
    body = "\n\n".join(["block " + " ".join(["x"] * 100)] * 30)
    chunks = split_symbol_body(
        body,
        primary_header=PRIMARY,
        continuation_header=CONT,
        max_tokens=MAX_SYMBOL_CHUNK_TOKENS,
    )
    assert len(chunks) >= 2
    for c in chunks[1:]:
        # Continuation chunks must not contain repo or file path tokens.
        # If they do, BM25 will fire 4x harder per query and re-introduce
        # the problem the 0.3x demote was tuned to fix (commit 7745043c).
        assert "prbe-knowledge" not in c
        assert "services/foo.py" not in c


def test_split_into_blocks_preserves_block_contents() -> None:
    body = "alpha\nbeta\n\ngamma\n\n\ndelta"
    blocks = _split_into_blocks(body)
    assert blocks == ["alpha\nbeta", "gamma", "delta"]


def test_split_into_blocks_handles_no_blank_lines() -> None:
    body = "alpha\nbeta\ngamma"
    blocks = _split_into_blocks(body)
    assert blocks == ["alpha\nbeta\ngamma"]


def test_split_into_blocks_strips_trailing_blanks() -> None:
    body = "alpha\n\nbeta\n\n\n\n"
    blocks = _split_into_blocks(body)
    assert blocks == ["alpha", "beta"]


def test_signature_only_body_fits_in_one_chunk() -> None:
    # Realistic case: stub function or `pass` body — symbol.signature
    # falls back when source_snippet is missing. Should always fit.
    body = "def bar(self, x: int) -> int: ..."
    chunks = split_symbol_body(
        body,
        primary_header=PRIMARY,
        continuation_header=CONT,
        max_tokens=MAX_SYMBOL_CHUNK_TOKENS,
    )
    assert len(chunks) == 1


def test_max_tokens_validation() -> None:
    import pytest

    with pytest.raises(ValueError, match="max_tokens"):
        split_symbol_body(
            "anything",
            primary_header=PRIMARY,
            continuation_header=CONT,
            max_tokens=0,
        )


def test_regression_30kb_symbol_never_emits_oversized_chunk() -> None:
    # Production fixture shape: large module-scope class with many methods.
    # Pre-fix, this would land as a single 6000+ token chunk and blow the
    # Probe MCP 25KB tool-result cap. Post-fix every chunk must fit budget.
    method = (
        "    def {name}(self, x):\n"
        "        # docstring describing what this method does\n"
        "        result = self.process(x)\n"
        "        if result is None:\n"
        "            raise ValueError('no result')\n"
        "        return result\n"
    )
    body = "class GiantClass:\n" + "\n".join(
        method.format(name=f"method_{i}") for i in range(80)
    )
    assert count_tokens(body) > 1500, "fixture should be genuinely oversized"

    chunks = split_symbol_body(
        body,
        primary_header=PRIMARY,
        continuation_header=CONT,
        max_tokens=MAX_SYMBOL_CHUNK_TOKENS,
    )
    assert len(chunks) >= 2
    for c in chunks:
        assert count_tokens(c) <= MAX_SYMBOL_CHUNK_TOKENS + 16, (
            f"chunk over budget by {count_tokens(c) - MAX_SYMBOL_CHUNK_TOKENS}"
        )
