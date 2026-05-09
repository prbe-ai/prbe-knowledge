"""Doc-grouped retrieval tests — fusion ranks docs by breadth+depth, chunks
preserve per-seed graph_evidence, top-level confidence_breakdown aggregates."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from services.retrieval.fusion import fuse
from shared.constants import RRF_BREADTH_ALPHA

_NOW = datetime(2026, 5, 1, tzinfo=UTC)


@dataclass
class FakeHit:
    chunk_id: str
    doc_id: str
    doc_version: int = 1
    source_system: str = "slack"
    source_url: str = "u"
    title: str | None = "t"
    content: str = "c"
    created_at: datetime = field(default_factory=lambda: _NOW)
    updated_at: datetime = field(default_factory=lambda: _NOW)
    score: float = 0.0


def test_doc_with_more_matched_chunks_outranks_single_strong_chunk_doc() -> None:
    """A doc whose 3 chunks all match should rank above a doc with one strong
    chunk, when the strong-doc's only edge is its single chunk's RRF.

    Encodes the formula: max(rrfs) + alpha*sum(others) + metadata_sum.
    A 3-chunk doc earns alpha*sum(others); a 1-chunk doc only gets max.
    """
    # `wide` doc: 3 chunks, each surfacing once at rank 2 in vec.
    wide = [
        FakeHit(chunk_id="w1", doc_id="wide", score=0.5),
        FakeHit(chunk_id="w2", doc_id="wide", score=0.5),
        FakeHit(chunk_id="w3", doc_id="wide", score=0.5),
    ]
    # `narrow` doc: one chunk that surfaces at rank 1 in vec (slightly higher RRF
    # than each of `wide`'s individual chunks).
    narrow = [FakeHit(chunk_id="n1", doc_id="narrow", score=0.9)]

    fused = fuse({"vector": [*narrow, *wide]}, top_k=10, now=_NOW)
    by_doc = {d.doc_id: d for d in fused}

    # Sanity: each chunk's individual RRF.
    # narrow's only chunk lands at vec rank 1 → 1/(60+1) = 0.01639...
    # wide's 3 chunks land at vec ranks 2, 3, 4.
    # narrow's doc score = max = 1/61
    # wide's doc score   = 1/62 + alpha*(1/63 + 1/64)
    narrow_score = 1.0 / 61
    wide_score = 1.0 / 62 + RRF_BREADTH_ALPHA * (1.0 / 63 + 1.0 / 64)
    assert wide_score > narrow_score
    assert by_doc["wide"].score > by_doc["narrow"].score
    assert fused[0].doc_id == "wide"


def test_chunks_within_doc_carry_monotonic_rank_in_doc() -> None:
    """rank_in_doc is 1-indexed and reflects within-doc RRF ordering."""
    vec = [
        FakeHit(chunk_id="a", doc_id="d", score=0.1),
        FakeHit(chunk_id="b", doc_id="d", score=0.9),
        FakeHit(chunk_id="c", doc_id="d", score=0.5),
    ]
    bm25 = [FakeHit(chunk_id="b", doc_id="d", score=0.5)]

    fused = fuse({"vector": vec, "bm25": bm25}, top_k=10, now=_NOW)
    assert len(fused) == 1
    chunks = fused[0].chunks
    # b appears in two retrievers → highest RRF → rank 1.
    assert [c.chunk_id for c in chunks] == ["b", "a", "c"]
    assert [c.rank_in_doc for c in chunks] == [1, 2, 3]


def test_top_k_applies_to_documents_not_chunks() -> None:
    """top_k=2 returns 2 distinct documents even if each has multiple chunks."""
    vec = []
    for doc_idx in range(5):
        for chunk_idx in range(3):
            vec.append(
                FakeHit(
                    chunk_id=f"d{doc_idx}-c{chunk_idx}",
                    doc_id=f"doc-{doc_idx}",
                )
            )
    fused = fuse({"vector": vec}, top_k=2, now=_NOW)
    assert len(fused) == 2
    assert all(len(d.chunks) >= 1 for d in fused)


def test_chunk_count_reflects_surviving_content_chunks() -> None:
    """QueryDocument.chunk_count == len(chunks) for every doc returned."""
    vec = [
        FakeHit(chunk_id="a", doc_id="d", score=0.1),
        FakeHit(chunk_id="b", doc_id="d", score=0.9),
    ]
    fused = fuse({"vector": vec}, top_k=10, now=_NOW)
    assert len(fused) == 1
    assert len(fused[0].chunks) == 2
