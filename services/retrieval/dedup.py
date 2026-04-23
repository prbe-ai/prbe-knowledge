"""Cross-doc deduplication.

Two chunks whose embeddings have cosine similarity > DEDUP_COSINE_THRESHOLD
are near-duplicates — common with Slack cross-posts, Notion mirrors, etc.
Drop the lower-ranked copy.
"""

from __future__ import annotations

import math
from typing import Any

from shared.constants import DEDUP_COSINE_THRESHOLD


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


def dedupe(
    hits: list[Any],
    embeddings: dict[str, list[float]],
    threshold: float = DEDUP_COSINE_THRESHOLD,
) -> list[Any]:
    """Remove near-duplicates (cosine > threshold) keeping the higher-ranked hit.

    `hits` is already sorted by descending score. `embeddings` maps chunk_id → vector.
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
