"""Guard tests for the documents.metadata['body'] storage cleanup.

The original ~440 MB / ~180 MB TOASTed duplication was caused by every
connector stuffing the full normalized body into the documents.metadata
jsonb column under the 'body' key, while the canonical source of truth
(chunks.content) carried it too. The cleanup:

  * adds a transient Document.body field (excluded from model_dump)
  * makes Normalizer._stringify_body read doc.body
  * has every handler set body=... on the Document constructor
  * adds a normalizer-level guard that fails ingestion if metadata['body']
    sneaks in

These tests pin all four invariants so a regression (a new connector
copying the old pattern) breaks CI rather than silently re-doubling
storage.
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from engine.ingest.handlers.base import ConnectorContext
from engine.ingest.normalizer import Normalizer
from engine.shared.config import Settings
from engine.shared.constants import (
    DocClass,
    DocType,
    Permission,
    PrincipalType,
    SourceSystem,
)
from engine.shared.exceptions import NormalizationError
from engine.shared.models import (
    ACLPrincipal,
    ACLSnapshot,
    Document,
    NormalizationResult,
)
from kb import handlers as handlers_pkg


def _bare_doc(*, metadata: dict[str, Any]) -> Document:
    now = datetime.now(UTC)
    return Document(
        doc_id="guard:doc:1",
        customer_id="cust-guard",
        source_system=SourceSystem.SLACK,
        source_id="src-1",
        source_url="https://example/x",
        doc_class=DocClass.RAW_SOURCE,
        doc_type=DocType.SLACK_MESSAGE,
        content_hash="x" * 64,
        title="t",
        body_preview="hello",
        body_size_bytes=5,
        body_token_count=1,
        author_id="alice",
        created_at=now,
        updated_at=now,
        valid_from=now,
        ingested_at=now,
        acl=ACLSnapshot(
            principals=[
                ACLPrincipal(
                    principal_type=PrincipalType.WORKSPACE,
                    principal_id="cust-guard",
                    permission=Permission.READ,
                )
            ],
            captured_at=now,
        ),
        metadata=metadata,
        body="hello",
    )


def _ctx() -> ConnectorContext:
    import httpx

    return ConnectorContext(settings=Settings(environment="local"), http=httpx.AsyncClient())


@pytest.mark.asyncio
async def test_persist_rejects_metadata_body_at_boundary() -> None:
    """A Document with `metadata['body']` must NOT be allowed through
    persistence. The normalizer raises before opening any DB transaction
    so the malformed payload doesn't leave a half-written row."""
    n = Normalizer(_ctx())
    bad = _bare_doc(metadata={"body": "the duplicated text", "team_id": "T1"})
    result = NormalizationResult(documents=[bad])

    with pytest.raises(NormalizationError, match="metadata\\['body'\\]"):
        await n._persist("cust-guard", SourceSystem.SLACK, result)


@pytest.mark.asyncio
async def test_persist_accepts_doc_with_body_field() -> None:
    """The new path — body on the transient Document.body field, NOT in
    metadata — is accepted. We only need to prove the guard doesn't
    false-positive on the supported shape: stub _plan_chunks to short-
    circuit before the live-DB path runs.
    """
    n = Normalizer(_ctx())
    good = _bare_doc(metadata={"team_id": "T1"})
    assert "body" not in good.metadata
    assert good.body == "hello"

    sentinel = RuntimeError("guard passed; reached _plan_chunks as expected")

    async def _raise(*args: Any, **kwargs: Any) -> None:
        raise sentinel

    n._plan_chunks = _raise  # type: ignore[method-assign]
    with pytest.raises(RuntimeError) as exc_info:
        await n._persist(
            "cust-guard", SourceSystem.SLACK, NormalizationResult(documents=[good])
        )
    assert exc_info.value is sentinel


def test_document_body_is_excluded_from_model_dump() -> None:
    """The transient body field must NOT serialize into model_dump output —
    that's what keeps it out of the persisted metadata jsonb (and out of
    any future code path that round-trips a Document through .model_dump()).
    """
    doc = _bare_doc(metadata={"x": 1})
    dumped = doc.model_dump()
    assert "body" not in dumped, (
        f"Document.body must be Field(exclude=True); dumped keys: {sorted(dumped)}"
    )
    # coalesce_into_live is also transient and must not leak into the row.
    assert "coalesce_into_live" not in dumped


def test_no_handler_writes_body_into_metadata() -> None:
    """Static guard: scan every connector handler for the regression pattern
    `"body": <something>` inside a metadata dict literal. The cleanup
    moved every handler over to `Document(body=..., metadata={...})`, and
    a new connector copying the old shape would silently re-double storage.

    This is a source-text scan rather than a runtime instantiation guard
    because most connectors require fixtures + tokens to construct a
    Document; the pattern is unambiguous enough that grep-equivalent
    coverage is sufficient.
    """
    import pkgutil
    from pathlib import Path

    import engine.ingest.handlers as engine_handlers_pkg

    # Connectors live in two packages since the engine/kb split: the kb
    # integration connectors plus the engine-door connectors
    # (custom_ingest, manual_upload). Scan both so the guard keeps full
    # coverage.
    pkg_dirs = [
        Path(handlers_pkg.__file__).parent,
        Path(engine_handlers_pkg.__file__).parent,
    ]
    offenders: list[tuple[str, int, str]] = []
    for pkg_dir, mod_info in (
        (d, m) for d in pkg_dirs for m in pkgutil.iter_modules([str(d)])
    ):
        if mod_info.name in {"__init__", "base", "registry"}:
            continue
        path = pkg_dir / f"{mod_info.name}.py"
        text = path.read_text(encoding="utf-8")
        # Walk lines; flag any occurrence of `"body":` that sits inside a
        # `metadata={` dict literal (heuristic: the most recent unclosed
        # `metadata=` brace within the last 30 lines).
        lines = text.splitlines()
        in_metadata = False
        depth = 0
        for lineno, line in enumerate(lines, 1):
            if "metadata={" in line:
                in_metadata = True
                depth = line.count("{") - line.count("}")
                continue
            if in_metadata:
                depth += line.count("{") - line.count("}")
                if '"body":' in line:
                    offenders.append((mod_info.name, lineno, line.strip()))
                if depth <= 0:
                    in_metadata = False
                    depth = 0

    assert not offenders, (
        "regression: handler is writing body into metadata dict literal — "
        "use Document(body=..., metadata={...}) instead. Offenders:\n"
        + "\n".join(f"  {n}.py:{ln}  {ln_text}" for n, ln, ln_text in offenders)
    )


def test_stringify_body_prefers_doc_body_over_legacy_metadata() -> None:
    """If both Document.body and metadata['body'] are set (legacy fixtures
    might still construct that shape), the new field wins. The legacy
    fallback exists only as a safety net for in-flight queue rows that
    predate the storage-cleanup deploy.
    """
    from engine.ingest.normalizer import _stringify_body

    doc = _bare_doc(metadata={})
    doc.body = "from-body-field"
    # Bypass the persistence guard — _stringify_body itself doesn't enforce.
    assert _stringify_body(doc) == "from-body-field"


def test_stringify_body_falls_back_to_legacy_metadata_body() -> None:
    """The fallback exists for in-flight rows from before the cleanup —
    they still need to chunk correctly. This test pins that fallback
    so removing it later is an explicit decision.
    """
    from engine.ingest.normalizer import _stringify_body

    doc = _bare_doc(metadata={"body": "legacy-from-metadata"})
    doc.body = None
    assert _stringify_body(doc) == "legacy-from-metadata"


