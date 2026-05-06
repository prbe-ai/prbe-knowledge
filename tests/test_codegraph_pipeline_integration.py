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

from services.ingestion.code_graph.pipeline import extract_files_to_result
from services.ingestion.handlers.base import ConnectorContext
from services.ingestion.handlers.codegraph import (
    KIND_DISCONNECT,
    CodeGraphConnector,
)
from shared.config import Settings
from shared.constants import EdgeType, NodeLabel
from shared.models import WebhookEvent


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
async def test_documents_use_body_field_not_metadata_body() -> None:
    """Storage guard at normalizer.py:191 raises on metadata['body'].

    pipeline must put the source body on the transient Document.body,
    not into metadata jsonb. Regression test for the rebase-induced
    body-field bug.
    """
    files = [_FE(rel_path="greeter.py", content=_PYTHON_SAMPLE)]
    result = await extract_files_to_result(
        customer_id="cust-test",
        repo="acme/app",
        sha="abc123",
        files=files,
        cached_state={},
    )
    assert result.documents, "expected symbol Documents from greeter.py"
    for doc in result.documents:
        assert "body" not in doc.metadata, (
            f"doc {doc.doc_id} put body into metadata jsonb — "
            "normalizer storage guard will raise on persist"
        )
        assert doc.body, f"doc {doc.doc_id} missing transient body"


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
    assert edge.from_label == NodeLabel.METHOD, (
        f"from_label was {edge.from_label}, expected METHOD — "
        "node lookup in graph_writer would miss"
    )
    assert edge.to_label == NodeLabel.METHOD, (
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

        from services.ingestion.handlers import codegraph as cg_module

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
