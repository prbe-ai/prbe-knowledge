"""Pipeline-level invariants for code_graph extraction.

Catches the integration seams unit tests miss:

- Documents must use Document.body (transient), not metadata["body"]
  (the storage guard at normalizer.py:191 raises on the latter).
- CALLS edges from a method must carry from_label=METHOD so node lookup
  in graph_writer.upsert_edges hits the right (label, canonical_id) row.
- Disconnect normalize must return a non-empty signal so the worker
  doesn't treat it as a connector bug.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx
import pytest

from engine.ingest.handlers.base import ConnectorContext
from engine.shared.config import Settings
from engine.shared.constants import EdgeType, NodeLabel
from engine.shared.models import WebhookEvent
from kb.code_graph.pipeline import extract_files_to_result
from kb.handlers.codegraph import (
    KIND_DISCONNECT,
    CodeGraphConnector,
)


@dataclass
class _FE:
    """FileEntry-shape stand-in (the real type lives in clone.py)."""

    rel_path: str
    content: bytes


_PYTHON_SAMPLE = b'''\
class Greeter:
    def hello(self, name: str) -> str:
        return self._format(name)

    def _format(self, name: str) -> str:
        return f"hi, {name}"


def main() -> None:
    g = Greeter()
    g.hello("world")
'''


@pytest.mark.asyncio
async def test_file_documents_have_no_body_and_pre_emit_chunks() -> None:
    """Path 2 contract: code.file Documents bypass the chunker.

    body MUST be None (chunks are authoritative and the connector
    owns chunking). metadata['body'] is still forbidden by the
    storage guard at normalizer.py. The PreChunkedDocument carries
    one ChunkPiece per symbol + one metadata chunk with identifying
    text.
    """
    files = [_FE(rel_path="greeter.py", content=_PYTHON_SAMPLE)]
    result = await extract_files_to_result(
        customer_id="cust-test",
        repo="acme/app",
        sha="abc123",
        files=files,
        cached_state={},
    )
    # Per-symbol Documents are gone — code.file is the only shape.
    assert not result.documents, "expected zero raw Documents under Path 2"
    assert result.documents_with_chunks, "expected file Document via documents_with_chunks"

    for prechunked in result.documents_with_chunks:
        doc = prechunked.document
        assert doc.body is None, (
            f"file Document {doc.doc_id} must have body=None "
            "(chunks are authoritative)"
        )
        assert "body" not in doc.metadata, (
            f"file Document {doc.doc_id} put body into metadata jsonb — "
            "normalizer storage guard will raise"
        )
        assert prechunked.chunks, "expected at least one symbol chunk"
        assert prechunked.metadata_chunk is not None, (
            "file Document must carry a metadata chunk for identity queries"
        )
        # Metadata chunk content must include the repo name — that's the
        # whole point of the Path 2 rewrite.
        assert "acme/app" in prechunked.metadata_chunk.content
        assert "greeter.py" in prechunked.metadata_chunk.content


@pytest.mark.asyncio
async def test_method_to_method_call_edge_carries_method_label() -> None:
    """CALLS edge from Greeter.hello to Greeter._format must have
    from_label=METHOD (not the FUNCTION default), or graph_writer's
    (label, canonical_id) lookup misses and the edge gets dropped.
    """
    files = [_FE(rel_path="greeter.py", content=_PYTHON_SAMPLE)]
    result = await extract_files_to_result(
        customer_id="cust-test",
        repo="acme/app",
        sha="abc123",
        files=files,
        cached_state={},
    )
    calls = [e for e in result.graph_edges if e.edge_type == EdgeType.CALLS]
    assert calls, "expected at least one CALLS edge"

    # hello → _format is the canonical intra-class method-to-method call
    # in the sample. Find it and assert its from_label is METHOD.
    method_to_method = [
        e for e in calls
        if "Greeter.hello" in e.from_canonical_id
        and "Greeter._format" in e.to_canonical_id
    ]
    assert method_to_method, (
        f"hello -> _format CALLS edge missing — got: "
        f"{[(e.from_canonical_id, e.to_canonical_id) for e in calls]}"
    )
    edge = method_to_method[0]
    assert edge.from_label == NodeLabel.CODE_SYMBOL, (
        f"from_label was {edge.from_label}, expected METHOD — "
        "node lookup in graph_writer would miss"
    )
    assert edge.to_label == NodeLabel.CODE_SYMBOL, (
        f"to_label was {edge.to_label}, expected METHOD"
    )


@pytest.mark.asyncio
async def test_disconnect_returns_skipped_reason() -> None:
    """_normalize_disconnect's bulk-SQL contract bend returns an empty
    NormalizationResult. Without skipped_reason, normalizer raises
    NormalizationError and the queue row keeps retrying. The fix sets
    skipped_reason so DuplicateEventIgnored fires and marks completed.
    """
    settings = Settings()
    async with httpx.AsyncClient() as http:
        ctx = ConnectorContext(settings=settings, http=http)
        connector = CodeGraphConnector(ctx)

        # We don't want this test to actually hit the DB. Stub `with_tenant`
        # at the module's import site so the SQL UPDATEs no-op.
        from contextlib import asynccontextmanager

        from kb.handlers import codegraph as cg_module

        class _StubConn:
            async def execute(self, *args, **kwargs):
                return None

        @asynccontextmanager
        async def _fake_with_tenant(_customer_id):
            yield _StubConn()

        original = cg_module.with_tenant
        cg_module.with_tenant = _fake_with_tenant
        try:
            event = WebhookEvent(
                customer_id="cust-test",
                source_system=cg_module.SourceSystem.CODE_GRAPH,
                source_event_id="code_graph:disconnect:acme/app:t",
                received_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
                payload_s3_key="raw/code_graph/cust-test/x.json",
                payload_s3_keys=["raw/code_graph/cust-test/x.json"],
                raw_payload={"kind": KIND_DISCONNECT, "repos": ["acme/app"]},
                headers={},
            )
            result = await connector._normalize_disconnect(event)
        finally:
            cg_module.with_tenant = original

    assert result.is_empty, "disconnect should produce no documents"
    assert result.skipped_reason, (
        "skipped_reason must be set, else normalizer raises "
        "NormalizationError('connector produced no documents and no reason')"
    )
    assert "disconnect" in result.skipped_reason
