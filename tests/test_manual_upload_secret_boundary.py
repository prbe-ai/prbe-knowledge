"""Manual multipart uploads must be inspected before the first object write."""

import hashlib
import io
import json
import zipfile
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import UploadFile
from starlette.datastructures import Headers
from starlette.requests import Request

from kb import ingestion_app as app

KEY = "ghp_" + hashlib.sha256(b"synthetic manual upload regression").hexdigest()[:36]


def docx(text, *, hidden=None):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "word/document.xml",
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            "<w:body><w:p><w:r><w:t>" + text + "</w:t></w:r></w:p></w:body></w:document>",
        )
        if hidden is not None:
            archive.writestr("docProps/custom.xml", hidden)
    return stream.getvalue()


@pytest.fixture
def boundary(monkeypatch):
    puts = []
    rows = []
    queued = []

    class Store:
        async def bucket_for(self, customer):
            return "synthetic-bucket"

        async def ensure_bucket(self, bucket):
            pass

        async def put(self, bucket, key, body, **kwargs):
            puts.append((key, body))

        async def delete(self, bucket, key):
            pass

    async def insert(**kwargs):
        rows.append(kwargs)

    async def enqueue(**kwargs):
        queued.append(kwargs)
        return True

    monkeypatch.setattr(app, "_verify_internal_key", lambda request: None)
    monkeypatch.setattr(
        app, "get_ingestion_killswitch", AsyncMock(return_value=SimpleNamespace(enabled=True))
    )
    monkeypatch.setattr(app, "_insert_manual_upload_row", insert)
    monkeypatch.setattr(app, "_enqueue", enqueue)
    request = Request(
        {
            "type": "http",
            "headers": [(b"x-note", KEY.encode())],
            "app": SimpleNamespace(state=SimpleNamespace(store=Store())),
        }
    )

    async def invoke(body, name="synthetic.txt", content_type="text/plain"):
        upload = UploadFile(
            file=io.BytesIO(body),
            filename=name,
            size=len(body),
            headers=Headers({"content-type": content_type}),
        )
        return await app.create_manual_uploads(
            request,
            [upload],
            uploaded_by=KEY,
            x_trace_id="synthetic-trace",
            x_prbe_customer="synthetic-tenant",
        )

    return invoke, puts, rows, queued


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ["text", "docx", "docx_hidden", "opaque"])
async def test_unsafe_manual_original_never_reaches_first_put(boundary, shape):
    invoke, puts, rows, queued = boundary
    body = KEY.encode()
    name = "synthetic.txt"
    if shape == "docx":
        body, name = docx(KEY), "synthetic.docx"
    if shape == "docx_hidden":
        body, name = docx("benign", hidden=KEY), "synthetic.docx"
    if shape == "opaque":
        body, name = b"\x00\xff\x01" * 100, "synthetic.bin"
    response = await invoke(body, name)
    assert not puts and not queued
    assert KEY not in response.body.decode()
    assert KEY not in json.dumps(rows, default=str)


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ["text", "docx"])
async def test_clean_manual_original_is_preserved_and_metadata_scrubbed(boundary, shape):
    invoke, puts, rows, queued = boundary
    body = b"benign research notes"
    name = "notes-" + KEY + ".txt"
    if shape == "docx":
        body, name = docx("benign research notes"), "notes-" + KEY + ".docx"
    response = await invoke(body, name)
    assert len(puts) == 2 and len(queued) == 1
    assert puts[0][1] == body
    assert KEY not in puts[0][0] and KEY.encode() not in puts[1][1]
    assert KEY not in response.body.decode() and KEY not in json.dumps(rows, default=str)


@pytest.mark.asyncio
async def test_parser_failure_is_generic_and_precedes_staging(boundary, monkeypatch):
    invoke, puts, rows, queued = boundary
    from engine.ingest.manual_uploads import ManualUploadParseError

    def fail(*args):
        raise ManualUploadParseError("synthetic diagnostic " + KEY)

    monkeypatch.setattr(app, "parse_manual_upload", fail)
    response = await invoke(b"benign research notes")
    assert not puts and not queued
    assert KEY not in response.body.decode() and KEY not in json.dumps(rows, default=str)


@pytest.mark.asyncio
async def test_manual_scanner_failure_cannot_create_objects_or_rows(boundary, monkeypatch):
    invoke, puts, rows, queued = boundary
    from engine.ingest import payload_redaction
    from engine.shared.exceptions import ScanUnavailable

    def fail(*args):
        raise ScanUnavailable("synthetic unavailable")

    monkeypatch.setattr(payload_redaction, "redact_documents", fail)
    with pytest.raises(ScanUnavailable):
        await invoke(b"benign research notes")
    assert not puts and not rows and not queued


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_phase", ["request", "file", "extracted", "envelope"])
async def test_every_scan_phase_precedes_manual_persistence(boundary, monkeypatch, failed_phase):
    from engine.shared.exceptions import ScanUnavailable

    invoke, puts, rows, queued = boundary
    original = app.redact_payload_async

    async def scan(value):
        if isinstance(value, str):
            phase = "extracted"
        elif "extracted_text" in value:
            phase = "envelope"
        elif "headers" in value:
            phase = "request"
        else:
            phase = "file"
        if phase == failed_phase:
            raise ScanUnavailable("synthetic unavailable")
        return await original(value)

    monkeypatch.setattr(app, "redact_payload_async", scan)
    with pytest.raises(ScanUnavailable):
        await invoke(b"benign research notes")
    assert not puts and not rows and not queued


@pytest.mark.asyncio
async def test_extracted_credentials_cannot_admit_original(boundary, monkeypatch):
    from dataclasses import replace

    invoke, puts, rows, queued = boundary
    original = app.parse_manual_upload

    def parse(*args):
        return replace(original(*args), text=KEY)

    monkeypatch.setattr(app, "parse_manual_upload", parse)
    response = await invoke(b"benign research notes")
    assert not puts and not queued
    assert KEY not in response.body.decode() and KEY not in json.dumps(rows, default=str)


@pytest.mark.asyncio
@pytest.mark.parametrize("member_kind", ["opaque", "encrypted"])
async def test_partial_manual_inspection_warns_caller_and_preserves_original(boundary, member_kind):
    from structlog.testing import capture_logs

    invoke, puts, rows, queued = boundary
    body = docx("benign research notes", hidden=b"\x00\xff\x01" * 101)
    if member_kind == "encrypted":
        # A flagged encrypted attachment cannot be opened by the inspector.
        # The readable document member remains valid for text extraction.
        raw = bytearray(body)
        with zipfile.ZipFile(io.BytesIO(body)) as archive:
            member = archive.getinfo("docProps/custom.xml")
        raw[member.header_offset + 6] |= 1
        offset = 0
        while True:
            offset = raw.find(b"PK\x01\x02", offset)
            if offset < 0:
                break
            name_length = int.from_bytes(raw[offset + 28 : offset + 30], "little")
            if raw[offset + 46 : offset + 46 + name_length] == b"docProps/custom.xml":
                raw[offset + 8] |= 1
                break
            offset += 4
        body = bytes(raw)
    with capture_logs() as logs:
        response = await invoke(body, "synthetic.docx")
    result = json.loads(response.body)
    assert result["accepted"] == 1
    assert len(puts) == 2 and len(queued) == 1
    assert puts[0][1] == body
    uploaded = result["uploads"][0]
    assert len(rows) == 1 and rows[0]["status"] == "queued"
    assert uploaded["fully_inspected"] is False
    assert uploaded["warnings"] and uploaded["warning"]
    assert result["warnings"] and response.headers["x-probe-inspection-warning"]
    assert response.headers["x-probe-inspection-warning"] == uploaded["warning"]
    assert uploaded["warning"].startswith("WARNING:")
    assert any(row.get("event") == "manual_upload.inspection_incomplete" for row in logs)
    stored = json.loads(puts[1][1])["payload"]
    assert stored["fully_inspected"] is False and stored["warnings"]


@pytest.mark.asyncio
async def test_manual_readable_credential_after_opaque_member_still_blocks(boundary):
    invoke, puts, rows, queued = boundary
    stream = io.BytesIO(docx("benign research notes", hidden=b"\x00\xff\x01" * 101))
    with zipfile.ZipFile(stream, "a", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("word/comments.xml", KEY)
    response = await invoke(stream.getvalue(), "synthetic.docx")
    assert json.loads(response.body)["accepted"] == 0
    assert not puts and not queued
    assert KEY not in response.body.decode()
    assert KEY not in json.dumps(rows, default=str)


@pytest.mark.asyncio
async def test_fully_inspected_manual_upload_does_not_emit_warning(boundary):
    invoke, puts, rows, queued = boundary
    response = await invoke(b"benign research notes")
    result = json.loads(response.body)
    assert len(puts) == 2 and len(rows) == 1 and len(queued) == 1
    assert result["uploads"][0]["fully_inspected"] is True
    assert result["uploads"][0]["warnings"] == []
    assert result["warnings"] == []
    assert "x-probe-inspection-warning" not in response.headers
