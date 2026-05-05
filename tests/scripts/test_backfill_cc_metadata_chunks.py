"""Unit tests for scripts/backfill_cc_metadata_chunks.

The atomic close-old / insert-new flow is exercised via DB integration
elsewhere (the chunker live-version invariants are tested in
tests/test_chunk_diff.py). Here we cover the script-specific surface:

- _doc_from_row reconstructs a Document the normalizer's _metadata_text
  helper can read.
- The new-text hash differs once an identity field appears, so the
  script's same-hash skip path doesn't accidentally swallow real changes.
"""

from __future__ import annotations

from datetime import UTC, datetime

from scripts.backfill_cc_metadata_chunks import _doc_from_row
from services.ingestion.normalizer import _chunk_hash, _metadata_text
from shared.constants import DocClass, DocType, SourceSystem


def _row(
    metadata: dict,
    *,
    title: str = "Claude Code session 82861aa0",
    source_system: SourceSystem = SourceSystem.CLAUDE_CODE,
) -> dict:
    """Build an in-memory documents row dict matching the SELECT shape."""
    now = datetime.now(UTC)
    prefix = source_system.value
    return {
        "doc_id": f"{prefix}:cust-1:82861aa0",
        "customer_id": "cust-1",
        "version": 1,
        "source_system": source_system.value,
        "source_id": "82861aa0",
        "source_url": "https://prbe.ai/dashboard/agent-sessions/82861aa0",
        "doc_class": DocClass.RAW_SOURCE.value,
        "doc_type": DocType.CLAUDE_CODE_SESSION.value,
        "content_type": "application/json",
        "content_hash": "deadbeef" * 8,
        "title": title,
        "body_preview": "USER: hi",
        "body_size_bytes": 8,
        "body_token_count": 0,
        "author_id": "11111111-1111-4111-8111-111111111111",
        "created_at": now,
        "updated_at": now,
        "valid_from": now,
        "ingested_at": now,
        "metadata": metadata,
    }


def test_doc_from_row_yields_document_metadata_text_can_read() -> None:
    row = _row(metadata={"agent": "claude_code", "device_id": "dev-1"})
    doc = _doc_from_row(row)

    text = _metadata_text(doc)
    # Title appears verbatim in metadata text, as does the source.
    assert "title: Claude Code session 82861aa0" in text
    assert "source: claude_code" in text
    # author and url present.
    assert "author: 11111111-1111-4111-8111-111111111111" in text
    assert "url: https://prbe.ai/dashboard/agent-sessions/82861aa0" in text


def test_doc_from_row_accepts_codex_source() -> None:
    row = _row(
        metadata={"agent": "codex", "device_id": "dev-1"},
        title="Richard Wei's (richard@prbe.ai) Codex session 82861aa0 (Richards-MacBook-Pro.local)",
        source_system=SourceSystem.CODEX,
    )
    doc = _doc_from_row(row)

    text = _metadata_text(doc)
    assert "title: Richard Wei's (richard@prbe.ai) Codex session 82861aa0" in text
    assert "source: codex" in text


def test_metadata_text_hash_differs_when_identity_added() -> None:
    """If the migration rewrites title to include name/email/hostname,
    _metadata_text(doc) MUST produce a new hash so the script's skip
    path doesn't no-op past a real change."""
    plain = _doc_from_row(_row(metadata={}))
    enriched = _doc_from_row(
        _row(
            metadata={},
            title="Richard Wei's (richard@prbe.ai) Claude Code session 82861aa0 (Richards-Macbook-Pro)",
        )
    )
    assert _chunk_hash(_metadata_text(plain)) != _chunk_hash(_metadata_text(enriched))


def test_codex_metadata_text_hash_differs_when_identity_added() -> None:
    plain = _doc_from_row(
        _row(
            metadata={"agent": "codex"},
            title="Codex session 82861aa0",
            source_system=SourceSystem.CODEX,
        )
    )
    enriched = _doc_from_row(
        _row(
            metadata={"agent": "codex"},
            title="Richard Wei's (richard@prbe.ai) Codex session 82861aa0 (Richards-MacBook-Pro.local)",
            source_system=SourceSystem.CODEX,
        )
    )
    assert _chunk_hash(_metadata_text(plain)) != _chunk_hash(_metadata_text(enriched))


def test_doc_from_row_handles_missing_optional_fields() -> None:
    """Some legacy docs lack body_preview / source_url; the dict-row
    reconstruction must not blow up on Nones."""
    row = _row(metadata={})
    row["body_preview"] = None
    row["source_url"] = None
    doc = _doc_from_row(row)
    # _metadata_text tolerates missing body_preview and empty url.
    text = _metadata_text(doc)
    assert "title:" in text
    # No "url:" line because source_url is empty/falsy.
    assert "url:" not in text
