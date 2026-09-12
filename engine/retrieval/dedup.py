"""Cross-doc deduplication.

Two chunks whose embeddings have cosine similarity > DEDUP_COSINE_THRESHOLD
are near-duplicates -- common with Slack cross-posts, Notion mirrors, etc.
Drop the lower-ranked copy.

TWO STAGES, cheapest first. `dedupe_gathered` runs an exact content-hash pass
over every chunk (free, catches the mirror-and-cross-post case that motivated
this module), then the cosine pass only over chunks whose embedding is
actually loaded. Cosine here is pure Python and O(n^2 * d): at 10 chunks and
1,536 dimensions that is ~100 vector ops, fine; it is NOT fine over a
pre-fan-out pool, which is why the caller passes the FINAL chunk set and the
loop records `agent.dedupe_ms` in `timing_ms`. If that number's p50 passes
50 ms, move the inner loop to numpy or drop the cosine stage -- the exact pass
carries most of the value on its own.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any

from engine.shared.constants import DEDUP_COSINE_THRESHOLD


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


_WS_RUN = re.compile(r"\s+")

#: Minimum normalized length before identical text counts as a DUPLICATE.
#:
#: Two different documents sharing a long passage copied it from each other.
#: Two different documents both saying "Approved." did not -- that is a short
#: string colliding by chance, and collapsing them destroys a real citation
#: apiece. Because this stage runs cross-DOC (that is the module's purpose:
#: Slack cross-posts, Notion mirrors), a false collapse costs a whole source,
#: not a redundant chunk. So the exact stage only judges passages long enough
#: that identity implies copying; below the line, chunks are passed through and
#: the cosine stage handles them if embeddings are loaded.
MIN_EXACT_DEDUPE_CHARS = 200


def content_key(text: str | None) -> str | None:
    """Hash of a chunk's content, whitespace-normalized.

    Normalized because the same passage arriving through two channels differs
    by rewrapping far more often than by a word -- an exact byte match would
    miss most real duplicates and the whole cheap stage with it.

    Returns None when the text is empty OR shorter than
    `MIN_EXACT_DEDUPE_CHARS`; the caller treats None as "not comparable"
    rather than as a duplicate of every other None.
    """
    if not text:
        return None
    normalized = _WS_RUN.sub(" ", text).strip().lower()
    if len(normalized) < MIN_EXACT_DEDUPE_CHARS:
        return None
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def dedupe(
    hits: list[Any],
    embeddings: dict[str, list[float]],
    threshold: float = DEDUP_COSINE_THRESHOLD,
) -> list[Any]:
    """Remove near-duplicates (cosine > threshold) keeping the higher-ranked hit.

    `hits` is already sorted by descending score. `embeddings` maps chunk_id -> vector.
    Hits without an embedding entry are passed through (they can't be compared).
    """
    kept: list[Any] = []
    kept_vecs: list[list[float]] = []
    for hit in hits:
        vec = embeddings.get(hit.chunk_id)
        if vec is None:
            kept.append(hit)
            continue
        is_dup = any(cosine(vec, kv) > threshold for kv in kept_vecs)
        if is_dup:
            continue
        kept.append(hit)
        kept_vecs.append(vec)
    return kept


def dedupe_gathered(
    chunks: list[Any],
    embeddings: dict[str, list[float]] | None = None,
    threshold: float = DEDUP_COSINE_THRESHOLD,
) -> tuple[list[Any], int]:
    """Drop duplicate chunks from a gatherer result, keeping the FIRST of each.

    Order is the delivery order, which is the gatherer's own curation followed
    by any harness backfill -- so "first" means a curated chunk always survives
    a raw pool copy of the same passage, never the other way round. That is the
    whole reason this runs after the backfill rather than inside it.

    Returns `(kept, dropped_count)`. A chunk with no content and no embedding is
    never dropped: unjudgeable is not the same as duplicate.
    """
    kept: list[Any] = []
    seen_content: set[str] = set()
    kept_vecs: list[list[float]] = []
    dropped = 0
    embeddings = embeddings or {}
    for chunk in chunks:
        key = content_key(getattr(chunk, "content", None))
        if key is not None:
            if key in seen_content:
                dropped += 1
                continue
            seen_content.add(key)
        vec = embeddings.get(getattr(chunk, "chunk_id", None) or "")
        if vec is not None:
            if any(cosine(vec, kv) > threshold for kv in kept_vecs):
                dropped += 1
                continue
            kept_vecs.append(vec)
        kept.append(chunk)
    return kept, dropped
