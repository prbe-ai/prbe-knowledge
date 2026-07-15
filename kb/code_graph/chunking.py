"""Code-aware splitter for oversized code_graph symbol bodies.

Background
----------
`_build_file_document_with_symbol_chunks` originally emitted one ChunkPiece
per symbol with no size cap. Production traces showed module-scope classes
and large handlers landing as single 7-30KB chunks. That broke two things:

  1. Probe MCP `search_knowledge` responses with code_graph hits exceeded
     the 25KB tool-result envelope (one query: 196KB / 12 docs / 7 files).
  2. BM25 ranking over outsized chunks fires harder than over prose chunks
     of the same query — the underlying `0.3x` code_graph score multiplier
     in commit 7745043c was a ranking-layer band-aid for an
     ingestion-layer mismatch.

This module is the ingestion-layer fix: cap each emitted chunk at
`MAX_SYMBOL_CHUNK_TOKENS` (== `DEFAULT_CHUNK_TOKENS` for retrieval
scale parity with prose), splitting oversized bodies into windows
that respect line boundaries.

Header strategy
---------------
Window 0 carries the full original header — `# {repo} · {file} · {qname}
({kind}) · L<def>-<end>` — so retrieval still ranks repo-qualified
queries correctly.

Windows 1+ carry only `# {qname} (cont.)` (no repo/file/path tokens) to
avoid amplifying BM25 firing on identifier tokens. A 4-window symbol that
duplicated the full header on every window would have inflated BM25 hits
on `prbe-backend / file.py` ~4x relative to a same-size prose chunk,
re-introducing the symptom the demote was tuned to fix.

Splitter algorithm
------------------
  1. If `header + body` fits in budget: emit one chunk.
  2. Else, split body on blank-line boundaries (preserves logical blocks).
  3. If any single block still exceeds budget: fall back to `chunk_text()`
     (the prose token-window splitter) for that block. Code with no
     blank-line structure (minified JS, single-line lambda) degrades
     to token-window slicing rather than failing.
  4. Pack blocks into windows greedily, up to `max_tokens` each.
  5. Empty / whitespace-only body: emit `[primary_header]` so the symbol
     stays in the index instead of disappearing.
"""

from __future__ import annotations

from services.ingestion.chunker import chunk_text, count_tokens


def split_symbol_body(
    body: str,
    *,
    primary_header: str,
    continuation_header: str,
    max_tokens: int,
) -> list[str]:
    """Return one or more chunk-content strings for a symbol body.

    Args:
        body: the symbol source snippet (or signature/qname fallback).
        primary_header: full header for window 0 (carries repo/file/qname).
        continuation_header: lighter header for windows 1+ (qname-only).
        max_tokens: soft target per chunk (== `MAX_SYMBOL_CHUNK_TOKENS`).
            The token-window fallback enforces this strictly; the
            blank-line packer respects it for normal cases.

    Returns:
        A non-empty list of chunk-content strings. Each fits under
        `max_tokens` whenever the input has any usable line structure;
        unsplittable inputs fall back to the prose chunker which is
        bounded by `MAX_INPUT_TOKENS`.

        Element 0 always starts with `primary_header`.
        Elements 1+ start with `continuation_header`.
    """
    if max_tokens < 1:
        raise ValueError("max_tokens must be >= 1")

    if not body or not body.strip():
        # Empty body: keep the symbol in the index via header-only chunk.
        # Strip trailing newline so downstream tokenization is stable.
        return [primary_header.rstrip("\n")]

    primary_tokens = count_tokens(primary_header)
    cont_tokens = count_tokens(continuation_header)

    # Fast path: whole symbol fits in one chunk.
    if primary_tokens + count_tokens(body) <= max_tokens:
        return [primary_header + body]

    # Pack body blocks into windows. Block boundaries are blank lines —
    # respects function/class/statement structure for typical code.
    blocks = _split_into_blocks(body)
    block_budget_w0 = max(1, max_tokens - primary_tokens)
    block_budget_wn = max(1, max_tokens - cont_tokens)

    windows: list[list[str]] = [[]]  # window -> list of block strings
    window_tokens: list[int] = [0]
    is_first_window = True

    def _budget() -> int:
        return block_budget_w0 if is_first_window else block_budget_wn

    for block in blocks:
        block_tok = count_tokens(block)

        # An indivisible block bigger than budget: fall back to the
        # prose token-window splitter for THIS block. Each fallback
        # piece becomes its own window. Avoids losing the block but
        # also avoids emitting chunks that exceed embedding context.
        if block_tok > _budget():
            # Flush any in-progress window first.
            if windows[-1]:
                windows.append([])
                window_tokens.append(0)
                is_first_window = False
            for piece in chunk_text(
                block,
                chunk_tokens=_budget(),
                overlap=0,
            ):
                windows[-1].append(piece.content)
                window_tokens[-1] = count_tokens(piece.content)
                windows.append([])
                window_tokens.append(0)
                is_first_window = False
            # Trailing empty window from the last append; trim later.
            continue

        # Normal block: pack into current window if it fits, else open new.
        if window_tokens[-1] + block_tok > _budget():
            windows.append([])
            window_tokens.append(0)
            is_first_window = False
        windows[-1].append(block)
        window_tokens[-1] += block_tok

    # Drop trailing empty window (from the indivisible-block flush path).
    if windows and not windows[-1]:
        windows.pop()

    # Render: window 0 with primary header, windows 1+ with continuation.
    result: list[str] = []
    for idx, blocks_in_window in enumerate(windows):
        if not blocks_in_window:
            continue
        header = primary_header if idx == 0 else continuation_header
        body_text = "\n\n".join(blocks_in_window)
        result.append(header + body_text)

    # Defensive fallback: if every window ended up empty (shouldn't happen
    # given the empty-body guard above, but keep the symbol in the index).
    if not result:
        return [primary_header.rstrip("\n")]

    return result


def _split_into_blocks(body: str) -> list[str]:
    """Split body on blank-line boundaries, preserving block contents.

    Returns a list of non-empty block strings. Lines within a block are
    joined with `\n`; blocks are intended to be re-joined with `\n\n`.
    Single-line bodies and bodies with no blank lines return as one block.
    """
    blocks: list[list[str]] = [[]]
    for line in body.splitlines():
        if line.strip() == "":
            if blocks[-1]:
                blocks.append([])
        else:
            blocks[-1].append(line)
    # Drop trailing empty block.
    if blocks and not blocks[-1]:
        blocks.pop()
    return ["\n".join(b) for b in blocks if b]
