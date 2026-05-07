"""Naive token-based chunker.

Phase 0 strategy: fixed token window with overlap, using the cl100k_base
tokenizer (correct for OpenAI text-embedding-3-large). Structural chunking
(respect headings, code blocks, thread boundaries) is Phase 1+.
"""

from __future__ import annotations

import tiktoken

from shared.constants import CHUNKER_VERSION

# ChunkPiece moved to shared.models so cross-module contracts
# (NormalizationResult.documents_with_chunks) can reference it without
# the shared→services layering violation. Re-exported here for
# backwards-compatible imports.
from shared.models import ChunkPiece

DEFAULT_CHUNK_TOKENS = 512
DEFAULT_CHUNK_OVERLAP = 64
MAX_INPUT_TOKENS = 8191  # OpenAI embedding-3-large hard ceiling (8192 is exclusive)


_encoding: tiktoken.Encoding | None = None


def _enc() -> tiktoken.Encoding:
    global _encoding
    if _encoding is None:
        _encoding = tiktoken.get_encoding("cl100k_base")
    return _encoding


def chunk_text(
    text: str,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[ChunkPiece]:
    """Split text into overlapping token windows.

    Empty text returns an empty list — we don't persist zero-content chunks.
    Windows cap at MAX_INPUT_TOKENS even if chunk_tokens is set higher.
    """
    if not text or not text.strip():
        return []
    if chunk_tokens < 1:
        raise ValueError("chunk_tokens must be >= 1")
    if overlap < 0 or overlap >= chunk_tokens:
        raise ValueError("overlap must be in [0, chunk_tokens)")

    chunk_tokens = min(chunk_tokens, MAX_INPUT_TOKENS)

    enc = _enc()
    # Treat any literal special-token strings (e.g. "<|endoftext|>") as plain
    # text — user content can contain them and they must not crash the worker.
    tokens = enc.encode(text, disallowed_special=())
    if not tokens:
        return []

    stride = chunk_tokens - overlap
    pieces: list[ChunkPiece] = []
    for idx, start in enumerate(range(0, len(tokens), stride)):
        window = tokens[start : start + chunk_tokens]
        if not window:
            break
        content = enc.decode(window)
        pieces.append(
            ChunkPiece(chunk_index=idx, content=content, token_count=len(window))
        )
        if start + chunk_tokens >= len(tokens):
            break
    return pieces


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_enc().encode(text, disallowed_special=()))


def chunker_version() -> str:
    return CHUNKER_VERSION
