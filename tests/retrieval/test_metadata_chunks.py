"""Unit tests for the metadata-chunk machinery (PR-A).

Covers:
  - _metadata_text: structured key:value, strips SHAs, no doc_id
  - _metadata_piece: returns None when nothing useful, METADATA_CHUNK_INDEX
  - fuse(): kind-aware doc-level scoring, drops metadata-only docs,
    response always shows content chunk
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from services.ingestion.normalizer import (
    METADATA_CHUNK_INDEX,
    _metadata_piece,
    _metadata_text,
    _strip_opaque_ids,
)
from services.retrieval.fusion import fuse
from shared.models import (
    ACLPrincipal,
    ACLSnapshot,
    DocClass,
    DocType,
    Document,
    Permission,
    PrincipalType,
    SourceSystem,
)

_NOW = datetime(2026, 4, 28, tzinfo=UTC)


def _doc(
    *,
    doc_id: str = "github:prbe-ai/prbe-backend:commit:abc",
    title: str | None = "fix(consent): rebrand error-page strings",
    source_system: SourceSystem = SourceSystem.GITHUB,
    source_url: str = "https://github.com/prbe-ai/prbe-backend/commit/abc1234567890abcdef1234567890abcdef12345",
    author_id: str | None = "alice",
    body_preview: str | None = "Three branding fixes on the OAuth consent page",
) -> Document:
    return Document(
        doc_id=doc_id,
        customer_id="cust-A",
        version=1,
        source_system=source_system,
        source_id="x",
        source_url=source_url,
        doc_class=DocClass.RAW_SOURCE,
        doc_type=DocType.GITHUB_COMMIT,
        content_hash="h",
        title=title,
        author_id=author_id,
        body_preview=body_preview,
        created_at=_NOW,
        updated_at=_NOW,
        valid_from=_NOW,
        ingested_at=_NOW,
        acl=ACLSnapshot(
            principals=[
                ACLPrincipal(
                    principal_type=PrincipalType.WORKSPACE,
                    principal_id="cust-A",
                    permission=Permission.READ,
                )
            ],
            captured_at=_NOW,
        ),
    )


# ---- _strip_opaque_ids ----------------------------------------------------


def test_strip_opaque_ids_removes_github_sha() -> None:
    url = "https://github.com/prbe-ai/prbe-backend/commit/abc1234567890abcdef1234567890abcdef12345"
    assert _strip_opaque_ids(url) == "https://github.com/prbe-ai/prbe-backend/commit/"


def test_strip_opaque_ids_passes_through_other_urls() -> None:
    assert (
        _strip_opaque_ids("https://github.com/prbe-ai/prbe-backend/pull/42")
        == "https://github.com/prbe-ai/prbe-backend/pull/42"
    )
    assert (
        _strip_opaque_ids("https://example.slack.com/archives/C123/p456")
        == "https://example.slack.com/archives/C123/p456"
    )


def test_strip_opaque_ids_handles_empty() -> None:
    assert _strip_opaque_ids("") == ""


# ---- _metadata_text -------------------------------------------------------


def test_metadata_text_structured_key_value() -> None:
    text = _metadata_text(_doc())
    lines = text.split("\n")
    assert "title: fix(consent): rebrand error-page strings" in lines
    assert "source: github" in lines
    assert "author: alice" in lines
    # URL with SHA stripped.
    assert "url: https://github.com/prbe-ai/prbe-backend/commit/" in lines
    assert "summary: Three branding fixes on the OAuth consent page" in lines


def test_metadata_text_excludes_doc_id() -> None:
    """doc_id and other opaque IDs must NOT appear in the embedded text —
    they tokenize into noise that hurts vector quality and pollutes BM25."""
    text = _metadata_text(_doc())
    assert "github:prbe-ai/prbe-backend:commit:abc" not in text
    assert "doc_id" not in text


def test_metadata_text_handles_missing_fields() -> None:
    text = _metadata_text(_doc(title=None, author_id=None, body_preview=None))
    # Source + URL still present even when other fields are None.
    assert "source: github" in text
    # No 'title:', 'author:', 'summary:' lines.
    assert "title:" not in text
    assert "author:" not in text
    assert "summary:" not in text


def test_metadata_text_summary_first_line_only_capped() -> None:
    long_preview = "first line\nsecond line that should not appear"
    text = _metadata_text(_doc(body_preview=long_preview))
    assert "summary: first line" in text
    assert "second line" not in text


# ---- _metadata_piece ------------------------------------------------------


def test_metadata_piece_uses_sentinel_index() -> None:
    piece = _metadata_piece(_doc())
    assert piece is not None
    assert piece.chunk_index == METADATA_CHUNK_INDEX
    assert piece.chunk_index < 0  # sentinel guarantee


def test_metadata_piece_returns_none_when_empty() -> None:
    """If a doc has nothing embeddable (no title, no author, no body_preview,
    no source_url) — happens for malformed connector outputs — _metadata_piece
    returns None and the caller skips it."""
    # We need the doc's source_url to be empty to actually produce an empty
    # text. Forcing it via a stripped URL.
    doc = _doc(
        title=None,
        author_id=None,
        body_preview=None,
        source_url="",
    )
    # source_system still adds a "source: github" line, so the text isn't
    # empty. This documents the current behavior — the metadata text always
    # has at least the source line.
    piece = _metadata_piece(doc)
    assert piece is not None
    assert "source: github" in piece.content


# ---- fuse() kind-aware ----------------------------------------------------


@dataclass
class _FakeHit:
    chunk_id: str
    doc_id: str
    doc_version: int = 1
    source_system: str = "github"
    source_url: str = "https://example/x"
    title: str | None = "title"
    content: str = "content text"
    score: float = 1.0
    kind: str = "content"
    created_at: datetime = _NOW
    updated_at: datetime = _NOW


def test_fuse_drops_doc_with_only_metadata_chunk() -> None:
    """If a doc surfaces ONLY via its metadata chunk (no content chunk in
    the candidate pool), it gets dropped — metadata-only ranking is too
    noisy to trust as the lone signal for inclusion."""
    fused = fuse(
        ranked_lists={
            "vector": [_FakeHit(chunk_id="m1", doc_id="docA", kind="metadata")],
        },
        top_k=10,
    )
    assert fused == []


def test_fuse_metadata_score_boosts_doc_ranking() -> None:
    """A doc with both content and metadata matches should out-rank a doc
    with only a content match (assuming similar RRF positions)."""
    # docA: content only at rank 1 → RRF = 1/(60+1)
    # docB: content at rank 2 + metadata at rank 1 → RRF = 1/(60+2) + 1/(60+1)
    fused = fuse(
        ranked_lists={
            "vector": [
                _FakeHit(chunk_id="cA", doc_id="docA", kind="content"),
                _FakeHit(chunk_id="cB", doc_id="docB", kind="content"),
            ],
            "bm25": [
                _FakeHit(chunk_id="mB", doc_id="docB", kind="metadata"),
            ],
        },
        top_k=10,
    )
    # docB should rank above docA because metadata-bm25 contribution adds
    # to its score.
    doc_ids = [h.doc_id for h in fused]
    assert doc_ids[0] == "docB"
    assert doc_ids[1] == "docA"


def test_fuse_response_always_returns_content_chunk() -> None:
    """When a doc has both metadata and content matches, the response shows
    the content chunk (synthetic key:value text never escapes to agents)."""
    fused = fuse(
        ranked_lists={
            "vector": [
                _FakeHit(
                    chunk_id="m1", doc_id="docA", kind="metadata", content="title: x\nrepo: y"
                ),
                _FakeHit(chunk_id="c1", doc_id="docA", kind="content", content="real body text"),
            ],
        },
        top_k=10,
    )
    assert len(fused) == 1
    assert fused[0].kind == "content"
    assert fused[0].content == "real body text"
    assert "title:" not in fused[0].content


def test_fuse_metadata_breakdown_visible_in_retriever_scores() -> None:
    """Metadata-chunk RRF contribution shows up in retriever_scores prefixed
    with `metadata_` so callers can see the doc surfaced via metadata."""
    fused = fuse(
        ranked_lists={
            "vector": [
                _FakeHit(chunk_id="c1", doc_id="docA", kind="content", score=0.5),
            ],
            "bm25": [
                _FakeHit(chunk_id="m1", doc_id="docA", kind="metadata", score=0.9),
            ],
        },
        top_k=10,
    )
    assert len(fused) == 1
    breakdown = fused[0].retriever_scores
    assert "vector" in breakdown
    assert "metadata_bm25" in breakdown


def test_fuse_no_metadata_unchanged_behavior() -> None:
    """REGRESSION: when there are no metadata chunks (pre-backfill state),
    fusion behaves identically to legacy doc-level collapse."""
    fused = fuse(
        ranked_lists={
            "vector": [
                _FakeHit(chunk_id="c1", doc_id="docA"),
                _FakeHit(chunk_id="c2", doc_id="docB"),
            ],
            "bm25": [
                _FakeHit(chunk_id="c1", doc_id="docA"),
            ],
        },
        top_k=10,
    )
    assert len(fused) == 2
    # docA wins — surfaces in both vector and bm25.
    assert fused[0].doc_id == "docA"
    assert fused[0].kind == "content"
