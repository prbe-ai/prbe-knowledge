"""Synthetic regressions: content cannot bypass the persistence scrubber."""

import base64
import hashlib
import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from engine.ingest import normalizer, secret_redaction
from engine.shared.constants import SourceSystem
from engine.shared.exceptions import ScanUnavailable
from engine.shared.models import ACLSnapshot, Document

KEY = "ghp_" + "8a29Df63bC17eA94fE61dB82aC03eF75dA19"


@pytest.mark.parametrize("report", [b"{}", b'[{"RuleID":"github-pat"}]', b"[null]"])
def test_non_verdict_cli_reports_fail_closed(monkeypatch, report):
    proc = SimpleNamespace(returncode=0, stdout=report, stderr=b"")
    monkeypatch.setattr(secret_redaction.subprocess, "run", lambda *a, **k: proc)
    with pytest.raises(ScanUnavailable):
        secret_redaction._scan_once("synthetic-scanner", KEY.encode())


@pytest.mark.asyncio
async def test_document_metadata_and_urls_are_clean_at_sql_boundary(monkeypatch):
    inserts = []
    embeddings = []

    class Conn:
        async def fetch(self, *args):
            return []

        async def fetchrow(self, *args):
            return None

        async def fetchval(self, sql, *args):
            if "INSERT INTO documents" in sql:
                inserts.append(args)
                return 1
            return 0

        async def execute(self, *args):
            return "OK"

    conn = Conn()

    @asynccontextmanager
    async def tenant(*args):
        yield conn

    class Embedder:
        async def embed_documents(self, docs):
            embeddings.extend(d.content for d in docs)
            return SimpleNamespace(
                embedded=[
                    SimpleNamespace(chunk_index=i, embedding=[0.0]) for i in range(len(docs))
                ],
                failed=[],
            )

    monkeypatch.setattr(normalizer, "with_tenant", tenant)
    n = object.__new__(normalizer.Normalizer)
    n._embedder = Embedder()
    now = datetime.now(UTC)
    doc = Document(
        doc_id="audit:synthetic",
        customer_id="audit-local",
        source_system=SourceSystem.CUSTOM_INGEST,
        source_id="audit",
        source_url="https://example.invalid/?token=" + KEY,
        doc_type="audit",
        content_hash="audit",
        title="title " + KEY,
        body="body " + KEY,
        body_preview="preview " + KEY,
        author_id=KEY,
        metadata={"details": KEY, "config": {"password": "Harbor7!"}, "eos_token": "</s>"},
        entities=[
            {
                "entity_type": "person",
                "canonical_id": "synthetic-person",
                "display_name": KEY,
                "attributes": {"config": {"password": "Harbor7!"}},
            }
        ],
        attachments=[
            {
                "kind": "file",
                "url": "https://example.invalid/" + KEY,
                "metadata": {"config": {"password": "Harbor7!"}},
            }
        ],
        doc_references=[
            {"external_url": "https://example.invalid/?token=" + KEY, "ref_type": "links_to"}
        ],
        created_at=now,
        updated_at=now,
        valid_from=now,
        ingested_at=now,
        acl=ACLSnapshot(principals=[], captured_at=now),
    )
    await n._plan_chunks("audit-local", doc)
    await normalizer._upsert_document(conn, doc)
    assert embeddings and inserts
    assert all(KEY not in t for t in embeddings)
    args = inserts[0]
    assert all(KEY not in str(x) for x in args)
    assert all("Harbor7!" not in str(x) for x in args)
    assert "</s>" in args[25]
    assert doc.entities[0].entity_type.value == "person"
    assert doc.attachments[0].kind.value == "file"
    assert doc.doc_references[0].ref_type.value == "links_to"


@pytest.mark.parametrize(
    "finding",
    [
        None,
        {},
        {"rule": "rule"},
        {"secret": "synthetic"},
        {"rule": "rule", "secret": "synthetic", "line": True},
        {"rule": "rule", "secret": "synthetic", "line": -1},
        {"rule": {}, "secret": "synthetic", "line": 1},
        {"rule": "rule", "secret": 5, "line": 1},
    ],
)
def test_malformed_daemon_findings_fail_closed(monkeypatch, finding):
    from engine.ingest.redactd import RedactdSupervisor

    supervisor = RedactdSupervisor()
    monkeypatch.setattr(
        supervisor, "_request_locked", lambda request: {"ok": True, "findings": [[finding]]}
    )
    with pytest.raises(ScanUnavailable):
        supervisor.scan(["synthetic content"])


@pytest.mark.parametrize("ok", ["true", 1, ["true"]])
def test_non_boolean_daemon_success_is_not_a_verdict(monkeypatch, ok):
    from engine.ingest.redactd import RedactdSupervisor

    supervisor = RedactdSupervisor()
    monkeypatch.setattr(supervisor, "_request_locked", lambda request: {"ok": ok, "findings": [[]]})
    with pytest.raises(ScanUnavailable):
        supervisor.scan(["synthetic content"])


def test_sensitive_dictionary_key_collisions_preserve_every_value():
    from engine.ingest.payload_redaction import redact_payload

    other = "ghp_" + hashlib.sha256(b"other synthetic key").hexdigest()[:36]
    clean = redact_payload({KEY: "first benign payload", other: "second benign payload"})
    assert sorted(clean.values()) == ["first benign payload", "second benign payload"]
    assert KEY not in json.dumps(clean) and other not in json.dumps(clean)


@pytest.mark.asyncio
@pytest.mark.parametrize("placement", ["content", "metadata"])
@pytest.mark.parametrize("form", ["short_password", "encoded"])
async def test_prechunked_pieces_scrub_before_embedding(monkeypatch, placement, form):
    from engine.shared.models import ChunkPiece

    value = "Harbor7!"
    text = "password=" + value
    if form == "encoded":
        value = base64.b64encode(KEY.encode()).decode()
        text = "copied " + value
    embeddings = []

    class Conn:
        async def fetch(self, *args):
            return []

    @asynccontextmanager
    async def tenant(*args):
        yield Conn()

    class Embedder:
        async def embed_documents(self, docs):
            embeddings.extend(d.content for d in docs)
            return SimpleNamespace(
                embedded=[
                    SimpleNamespace(chunk_index=i, embedding=[0.0]) for i in range(len(docs))
                ],
                failed=[],
            )

    monkeypatch.setattr(normalizer, "with_tenant", tenant)
    n = object.__new__(normalizer.Normalizer)
    n._embedder = Embedder()
    now = datetime.now(UTC)
    doc = Document(
        doc_id="audit:prechunked",
        customer_id="audit-local",
        source_system=SourceSystem.CUSTOM_INGEST,
        source_id="audit",
        source_url="https://example.invalid/audit",
        doc_type="audit",
        content_hash="audit",
        created_at=now,
        updated_at=now,
        valid_from=now,
        ingested_at=now,
        acl=ACLSnapshot(principals=[], captured_at=now),
    )
    piece = ChunkPiece(chunk_index=0, content=text, token_count=8)
    plan = await n._plan_chunks(
        "audit-local",
        doc,
        [piece] if placement == "content" else [],
        piece if placement == "metadata" else None,
    )
    assert embeddings
    assert all(value not in text for text in embeddings)
    assert piece.content == text  # input pieces are immutable evidence for callers
    assert len(plan.added_pieces) == 1
    stored_piece = plan.added_pieces[0][0]
    assert stored_piece.chunk_index == 0 and stored_piece.token_count == 8


def test_scanner_batches_by_aggregate_bytes_with_real_daemon():
    """Each input fits the scanner; their aggregate must use separate requests."""
    text = " " * (5 * 1024 * 1024) + KEY
    results = secret_redaction.scan_many([text, text])
    assert len(results) == 2
    assert all(any(value == KEY for _rule, value, _line in found) for found in results)


@pytest.mark.parametrize(
    "failure", ["cli_stderr", "cli_spawn", "daemon_rejection", "daemon_restart"]
)
def test_scanner_diagnostics_never_expose_credentials(monkeypatch, failure):
    from structlog.testing import capture_logs

    from engine.ingest.redactd import RedactdSupervisor

    def fail(*args, **kwargs):
        raise OSError("synthetic diagnostic " + KEY)

    if failure.startswith("cli"):
        if failure == "cli_stderr":
            proc = SimpleNamespace(
                returncode=1, stdout=b"", stderr=("synthetic diagnostic " + KEY).encode()
            )
            monkeypatch.setattr(secret_redaction.subprocess, "run", lambda *a, **k: proc)
        else:
            monkeypatch.setattr(secret_redaction.subprocess, "run", fail)
        def invoke():
            return secret_redaction._scan_once("synthetic-scanner", b"benign")
    else:
        supervisor = RedactdSupervisor()
        if failure == "daemon_rejection":
            monkeypatch.setattr(
                supervisor, "_request_locked", lambda req: {"ok": False, "error": KEY}
            )
        else:
            monkeypatch.setattr(supervisor, "_request_locked", fail)
            monkeypatch.setattr(supervisor, "_restart_locked", lambda: None)
        def invoke():
            return supervisor.scan(["benign"])
    with capture_logs() as logs, pytest.raises(ScanUnavailable) as caught:
        invoke()
    assert KEY not in str(caught.value)
    assert KEY not in json.dumps(logs, default=str)


@pytest.mark.asyncio
@pytest.mark.parametrize("secret_id", [False, True])
async def test_webhook_trace_and_parsed_event_id_never_persist_raw(monkeypatch, secret_id):
    import json
    from unittest.mock import AsyncMock

    from starlette.requests import Request

    from kb import ingestion_app as app
    logs, puts, queued = [], [], []
    class Logger:
        def info(self, *args, **kwargs): logs.append(kwargs)
    class Store:
        async def bucket_for(self, customer): return "test"
        async def ensure_bucket(self, bucket): pass
        async def put(self, bucket, key, body): puts.append((key, body.decode()))
    class Connector:
        def verify_signature(self, *args): return True
        def parse_webhook_event(self, customer, headers, payload):
            return SimpleNamespace(source_event_id=payload['id'], received_at=datetime.now(UTC), parse_hint={})
    async def enqueue(**kwargs):
        queued.append(kwargs)
        return True
    monkeypatch.setattr(app, 'get_settings', lambda: SimpleNamespace(internal_knowledge_api_key=None, default_customer_id='test'))
    monkeypatch.setattr(app, 'get_ingestion_killswitch', AsyncMock(return_value=SimpleNamespace(enabled=True)))
    monkeypatch.setattr(app, 'refusal_for', AsyncMock(return_value=None))
    monkeypatch.setattr(app, 'get_connector_class', lambda source: Connector)
    monkeypatch.setattr(app, 'build_connector', lambda *args: Connector())
    monkeypatch.setattr(app, 'bind_trace', lambda value: logs.append({'bound': value}))
    monkeypatch.setattr(app, 'log', Logger())
    monkeypatch.setattr(app, '_enqueue', enqueue)
    raw = json.dumps({'id': KEY if secret_id else 'event-1', 'description': 'benign'}).encode()
    async def receive(): return {'type': 'http.request', 'body': raw, 'more_body': False}
    request = Request({'type':'http','headers':[(b'x-trace-id', KEY.encode())],
        'app':SimpleNamespace(state=SimpleNamespace(ctx=None, store=Store()))}, receive)
    if secret_id:
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc:
            await app.webhook('linear', request, x_trace_id=KEY, x_prbe_customer=None)
        assert exc.value.status_code == 422
        assert not puts and not queued
    else:
        response = await app.webhook('linear', request, x_trace_id=KEY, x_prbe_customer=None)
        assert response.status_code == 200
        assert puts and queued
    assert KEY not in json.dumps({'puts':puts, 'queue':queued, 'logs':logs})


@pytest.mark.parametrize('lookup', ["os.environ['ANTHROPIC_API_KEY']", "os.getenv('ANTHROPIC_API_KEY')"])
@pytest.mark.parametrize('literal_suffix', ['', ' + "FabricatedCredential42!"'])
def test_combined_ingress_scrubber_preserves_only_complete_environment_references(lookup, literal_suffix):
    from engine.ingest.payload_redaction import redact_payload

    expression = lookup + literal_suffix
    payload = {'content': 'api_key = ' + expression, 'api_key': expression}
    clean = redact_payload(payload)
    if not literal_suffix:
        assert clean == payload
    else:
        assert 'FabricatedCredential42!' not in json.dumps(clean)


def _transcript_with_one_encoded_credential(filler_lines: int = 600) -> str:
    """A long session body: one BENIGN percent-escape, one plain credential and
    one ENCODED credential, each on its own line, far apart.

    The shape that erased 28 whole transcripts on research (2026-09-18 on):
    `scrub_string` decodes the WHOLE value when any `%XX`/`\\uXXXX` appears and,
    if the decoded view changes anywhere, returns `<redacted>` for all of it.
    """
    lines = ["session start: opened https://example.invalid/docs/a%20b.md"]
    lines += [
        f"turn {i}: we discussed chunk retirement in the normalizer at length."
        for i in range(filler_lines)
    ]
    lines.insert(filler_lines // 2, f"user pasted: export GITHUB_TOKEN={KEY}")
    lines.append("callback https://example.invalid/login?next=%2Fhome%3Fpassword%3DHarbor7%21")
    lines.append("final-marker-line: the session kept going after every finding")
    return "\n".join(lines)


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["body", "prechunked"])
async def test_one_encoded_finding_costs_its_line_not_the_whole_body(monkeypatch, path):
    """A finding anywhere in a multi-line body must not replace the entire body.

    Before the fix the body came back as the 10-character placeholder, chunked
    into ONE `<redacted>` chunk, and the chunk diff retired every live content
    chunk of the session (prod: 832 chunks, 1.39M chars, in one pass).
    """
    from engine.shared.models import ChunkPiece

    text = _transcript_with_one_encoded_credential()
    embeddings = []

    class Conn:
        async def fetch(self, *args):
            return []

    @asynccontextmanager
    async def tenant(*args):
        yield Conn()

    class Embedder:
        async def embed_documents(self, docs):
            embeddings.extend(d.content for d in docs)
            return SimpleNamespace(
                embedded=[
                    SimpleNamespace(chunk_index=i, embedding=[0.0]) for i in range(len(docs))
                ],
                failed=[],
            )

    monkeypatch.setattr(normalizer, "with_tenant", tenant)
    n = object.__new__(normalizer.Normalizer)
    n._embedder = Embedder()
    now = datetime.now(UTC)
    doc = Document(
        doc_id="claude_code:audit-local:synthetic-session",
        customer_id="audit-local",
        source_system=SourceSystem.CLAUDE_CODE,
        source_id="synthetic-session",
        source_url="https://example.invalid/session",
        doc_type="claude_code.session",
        content_hash="audit",
        body=text if path == "body" else None,
        created_at=now,
        updated_at=now,
        valid_from=now,
        ingested_at=now,
        acl=ACLSnapshot(principals=[], captured_at=now),
    )
    prechunked = (
        [ChunkPiece(chunk_index=0, content=text, token_count=8)] if path == "prechunked" else None
    )
    plan = await n._plan_chunks("audit-local", doc, prechunked)

    stored = "\n".join(piece.content for piece, _emb, kind in plan.added_pieces if kind == "content")
    # The body survived: its far end, its middle and its benign escape are all there.
    assert "final-marker-line" in stored
    assert f"turn {300 + 1}:" in stored
    assert "a%20b.md" in stored
    # Both credentials are still gone -- plain and percent-encoded.
    joined = "\n".join(embeddings)
    assert KEY not in joined
    assert "Harbor7" not in joined
    # And only the encoded credential's own line was given up for it.
    assert stored.count("<redacted>") <= 2


def _too_deep(text: str = "curl https://example.test/?q=a b", levels: int = 6) -> str:
    """Text percent-encoded more levels deep than the scrubber will inspect."""
    from urllib.parse import quote

    for _ in range(levels):
        text = quote(text)
    return text


def test_the_fixture_is_deeper_than_the_scrubber_inspects():
    from engine.ingest._credential_redaction import default_scrub

    with pytest.raises(ValueError, match="nesting limit"):
        default_scrub({"text": _too_deep()})


def test_a_value_too_deep_to_inspect_costs_only_itself():
    """It failed the whole ingest request with a 500, retried forever by the tap:
    every later batch of that session queued behind it, uncaptured."""
    from engine.ingest.payload_redaction import redact_payload

    token = "ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
    payload = {
        "session_id": "s1",
        "events": [
            {"line_no": 0, "raw": {"message": {"content": _too_deep()}}},
            {"line_no": 1, "raw": {"message": {"content": "plain words stay"}}},
            {"line_no": 2, "raw": {"message": {"content": f"token {token}"}}},
        ],
    }
    clean = redact_payload(payload)
    assert clean["session_id"] == "s1"
    assert clean["events"][0]["raw"]["message"]["content"] == "<redacted>"
    assert clean["events"][1]["raw"]["message"]["content"] == "plain words stay"
    assert token not in json.dumps(clean)


def test_a_too_deep_line_costs_only_its_line():
    from engine.ingest.payload_redaction import redact_payload, redact_texts

    text = "first line\n" + _too_deep() + "\nlast line"
    assert redact_payload({"body": text})["body"] == "first line\n<redacted>\nlast line"
    assert redact_texts([text]) == ["first line\n<redacted>\nlast line"]


def test_a_too_deep_key_is_replaced_and_its_value_kept():
    from engine.ingest.payload_redaction import redact_payload

    clean = redact_payload({_too_deep(): "value", "other": 1})
    assert clean == {"<redacted>": "value", "other": 1}


def test_a_credential_container_is_still_dropped_whole():
    """The fallback walks only what `default_scrub` would have walked."""
    from engine.ingest.payload_redaction import redact_payload

    clean = redact_payload({"credentials": {"note": _too_deep()}, "text": _too_deep()})
    assert clean == {"credentials": "<redacted>", "text": "<redacted>"}
