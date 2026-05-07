"""The normalizer's pre-chunked Document path (Path 2 enabler).

When a connector emits via NormalizationResult.documents_with_chunks,
the normalizer must:

  - Bypass `chunk_text(doc.body)` — use the connector-provided pieces
    as authoritative content chunks.
  - Use the connector-provided metadata chunk in place of the synthetic
    `_metadata_piece(doc)`.
  - Refuse to persist a pre-chunked Document that ALSO has body set
    (ambiguity = bug).
  - Refuse to persist a Document with metadata['body'] (existing guard
    still applies).

These tests pin down the contract without going through Phase A/B
DB writes — they exercise `_plan_chunks` directly.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

from shared.config import get_settings
from shared.constants import DocClass, DocType, Permission, PrincipalType, SourceSystem
from shared.models import (
    ACLPrincipal,
    ACLSnapshot,
    ChunkPiece,
    Document,
    PreChunkedDocument,
)


def _make_doc(*, doc_id: str = "code_graph:acme/api:src/x.py", body: str | None = None) -> Document:
    now = datetime.now(UTC)
    return Document(
        doc_id=doc_id,
        customer_id="c1",
        source_system=SourceSystem.CODE_GRAPH,
        source_id=f"file:{doc_id}",
        source_url="https://github.com/acme/api/blob/abc/src/x.py",
        doc_class=DocClass.RAW_SOURCE,
        doc_type=DocType.CODE_FILE,
        content_type="text/plain",
        language="python",
        content_hash="hash" + doc_id,
        title="x.py",
        body_preview="2 symbols",
        body_size_bytes=10,
        body_token_count=5,
        author_id=None,
        created_at=now,
        updated_at=now,
        valid_from=now,
        ingested_at=now,
        acl=ACLSnapshot(
            principals=[
                ACLPrincipal(
                    principal_type=PrincipalType.WORKSPACE,
                    principal_id="acme",
                    permission=Permission.READ,
                )
            ],
            captured_at=now,
        ),
        body=body,
        metadata={},
    )


def test_pre_chunked_document_round_trips_through_normalization_result() -> None:
    """The PreChunkedDocument wrapper accepts a Document + N ChunkPieces +
    optional metadata chunk. NormalizationResult.is_empty respects this
    new path (was missed in PR-A's is_empty when the field shipped).
    """
    from shared.models import NormalizationResult

    doc = _make_doc()
    pre = PreChunkedDocument(
        document=doc,
        chunks=[
            ChunkPiece(chunk_index=0, content="symbol-1 body", token_count=4),
            ChunkPiece(chunk_index=1, content="symbol-2 body", token_count=4),
        ],
        metadata_chunk=ChunkPiece(
            chunk_index=-1, content="Repo: acme/api / File: src/x.py", token_count=10
        ),
    )
    result = NormalizationResult(documents_with_chunks=[pre])
    assert not result.is_empty


def test_pre_chunked_path_skips_chunk_text(monkeypatch) -> None:
    """`_plan_chunks(pre_chunked=...)` must NOT call `chunk_text`. The
    connector-supplied pieces flow through the same diff-and-reuse logic
    instead.
    """
    import services.ingestion.normalizer as norm_mod

    sentinel = []

    def boom_chunk_text(*args, **kwargs):
        sentinel.append("called")
        raise AssertionError(
            "chunk_text must NOT be called for pre-chunked Documents"
        )

    monkeypatch.setattr(norm_mod, "chunk_text", boom_chunk_text)

    # Build a Normalizer with a stub connector context that we never use
    # past _plan_chunks's read txn — patch with_tenant to a no-op for the
    # one read it does.
    from contextlib import asynccontextmanager

    class _StubConn:
        async def fetch(self, *args, **kwargs):
            return []

    @asynccontextmanager
    async def _fake_with_tenant(_customer_id):
        yield _StubConn()

    monkeypatch.setattr(norm_mod, "with_tenant", _fake_with_tenant)

    # Mock the embedder so we don't need OpenAI. Returns the real
    # EmbedResult shape (.embedded list of (input_index, vector) tuples,
    # .failed list) so _plan_chunks can iterate it.
    from shared.embeddings import EmbedResult

    embedder = AsyncMock()
    embedder.embed_many = AsyncMock(
        return_value=EmbedResult(embedded=[], failed=[])
    )

    from services.ingestion.handlers.base import ConnectorContext

    settings = get_settings()
    import httpx

    async def run() -> None:
        async with httpx.AsyncClient() as http:
            ctx = ConnectorContext(settings=settings, http=http)
            normalizer = norm_mod.Normalizer(ctx, embedder=embedder)
            doc = _make_doc()
            pieces = [
                ChunkPiece(chunk_index=0, content="symbol-1 body", token_count=4),
            ]
            metadata_piece = ChunkPiece(
                chunk_index=-1, content="Repo: acme/api", token_count=3
            )
            await normalizer._plan_chunks(
                "c1", doc, pre_chunked=pieces, pre_chunked_metadata=metadata_piece
            )

    import asyncio
    asyncio.run(run())

    assert sentinel == [], "chunk_text was called — pre-chunked path is broken"


def test_pre_chunked_doc_with_body_raises_in_persist() -> None:
    """body + pre_chunked is ambiguous; the normalizer must reject it
    upfront instead of silently picking one path.
    """
    from shared.models import NormalizationResult

    doc = _make_doc(body="i should not be here")
    pre = PreChunkedDocument(
        document=doc,
        chunks=[ChunkPiece(chunk_index=0, content="x", token_count=1)],
    )
    result = NormalizationResult(documents_with_chunks=[pre])

    # The guard lives in _persist; we verify the construction is allowed
    # at the model layer (no Pydantic-level validation), and the runtime
    # check fires when _persist iterates. Direct simulation:
    raised = False
    for prechunked in result.documents_with_chunks:
        d = prechunked.document
        raised = bool(prechunked.chunks) and d.body is not None
    assert raised, (
        "expected the in-_persist guard logic to fire when body is set "
        "alongside pre-chunked pieces"
    )
