from __future__ import annotations

import io
import zipfile
from datetime import UTC, datetime

import httpx
import pytest

from services.ingestion.handlers.base import ConnectorContext
from services.ingestion.handlers.manual_upload import ManualUploadConnector
from services.ingestion.manual_uploads import (
    MAX_DOCX_DECOMPRESSED_BYTES,
    ManualUploadParseError,
    parse_manual_upload,
)
from shared.config import get_settings
from shared.constants import DocClass, DocType, SourceSystem
from shared.models import WebhookEvent


def test_parse_manual_upload_plain_text() -> None:
    parsed = parse_manual_upload("notes.txt", "text/plain", b"hello\nworld")

    assert parsed.filename == "notes.txt"
    assert parsed.text == "hello\nworld"
    assert parsed.doc_type == DocType.MANUAL_UPLOAD_TEXT


def test_parse_manual_upload_rejects_binary_without_text() -> None:
    with pytest.raises(ManualUploadParseError):
        parse_manual_upload("image.png", "image/png", b"\x00\x01\x02\x03" * 100)


def test_parse_manual_upload_rejects_pdf_until_parser_exists() -> None:
    with pytest.raises(ManualUploadParseError):
        parse_manual_upload(
            "brief.pdf",
            "application/pdf",
            b"%PDF-1.7\n1 0 obj\n<< /Type /Catalog >>\nendobj",
        )


def test_parse_manual_upload_docx() -> None:
    body = io.BytesIO()
    with zipfile.ZipFile(body, "w") as archive:
        archive.writestr(
            "word/document.xml",
            """
            <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
              <w:body>
                <w:p><w:r><w:t>Manual plan</w:t></w:r></w:p>
                <w:p><w:r><w:t>Index this text.</w:t></w:r></w:p>
              </w:body>
            </w:document>
            """,
        )

    parsed = parse_manual_upload(
        "plan.docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        body.getvalue(),
    )

    assert parsed.doc_type == DocType.MANUAL_UPLOAD_DOCX
    assert "Manual plan" in parsed.text
    assert "Index this text." in parsed.text


def test_parse_manual_upload_docx_rejects_decompression_bomb() -> None:
    body = io.BytesIO()
    huge = "A" * (MAX_DOCX_DECOMPRESSED_BYTES + 1)
    with zipfile.ZipFile(body, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "word/document.xml",
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            f"<w:body><w:p><w:r><w:t>{huge}</w:t></w:r></w:p></w:body></w:document>",
        )

    with pytest.raises(ManualUploadParseError):
        parse_manual_upload(
            "bomb.docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            body.getvalue(),
        )


def test_parse_manual_upload_docx_rejects_xml_entities() -> None:
    body = io.BytesIO()
    with zipfile.ZipFile(body, "w") as archive:
        archive.writestr(
            "word/document.xml",
            '<?xml version="1.0"?>'
            '<!DOCTYPE r [<!ENTITY a "aaaa"><!ENTITY b "&a;&a;&a;&a;">]>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            "<w:body><w:p><w:r><w:t>&b;</w:t></w:r></w:p></w:body></w:document>",
        )

    with pytest.raises(ManualUploadParseError):
        parse_manual_upload(
            "entities.docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            body.getvalue(),
        )


@pytest.mark.asyncio
async def test_manual_upload_connector_normalize() -> None:
    uploaded_at = datetime(2026, 5, 3, tzinfo=UTC)
    async with httpx.AsyncClient() as client:
        connector = ManualUploadConnector(
            ConnectorContext(settings=get_settings(), http=client)
        )
        result = await connector.normalize(
            WebhookEvent(
                customer_id="cust_1",
                source_system=SourceSystem.MANUAL_UPLOAD,
                source_event_id="manual-1",
                received_at=uploaded_at,
                raw_payload={
                    "upload_id": "manual-1",
                    "filename": "runbook.md",
                    "content_type": "text/markdown",
                    "file_size_bytes": 42,
                    "file_sha256": "abc123",
                    "uploaded_by": "user_1",
                    "uploaded_at": uploaded_at.isoformat(),
                    "extracted_text": "# Runbook\n\nRestart the worker.",
                    "parse_engine": "plain-text",
                    "doc_type": DocType.MANUAL_UPLOAD_MARKDOWN.value,
                    "doc_id": "manual_upload:manual-1",
                },
            ),
            {},
        )

    assert len(result.documents) == 1
    doc = result.documents[0]
    assert doc.source_system == SourceSystem.MANUAL_UPLOAD
    assert doc.doc_class == DocClass.MANUAL_ENTRY
    assert doc.doc_type == DocType.MANUAL_UPLOAD_MARKDOWN
    assert doc.body == "# Runbook\n\nRestart the worker."
    assert doc.metadata["original_deleted_after_ingest"] is True
