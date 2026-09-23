"""The collapsed-session repair re-chunks from R2 and writes nothing else.

Seeds the production shape on a real Postgres: a session the worker indexed
healthily, then collapsed the way #568 did it (every content chunk retired, one
live `<redacted>` chunk left, the documents row untouched). The script must
restore exactly the healthy chunk set without writing the documents row or the
queue row, without mining, and without printing any of the transcript.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime

import orjson
import pytest

from engine.ingest.handlers.base import make_default_context
from engine.ingest.normalizer import Normalizer, _chunk_hash
from engine.shared import db as db_module
from engine.shared import storage
from engine.shared.constants import EMBEDDING_V2_DIM, SourceSystem
from engine.shared.embeddings import EmbeddedChunk, EmbedResult
from engine.shared.exceptions import StorageNotFound
from engine.shared.session_signals import cron_marker_key
from scripts import rechunk_collapsed_sessions as rechunk

C = "rechunk-collapsed-cust"
SID = "0a1b2c3d-4e5f-4a6b-8c7d-9e0f1a2b3c4d"
DOC_ID = f"claude_code:{C}:{SID}"
KEY = "ghp_" + "8a29Df63bC17eA94fE61dB82aC03eF75dA19"
MARKER = "unique-transcript-marker"


class FakeStore:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    async def bucket_for(self, customer_id: str) -> str:
        return "test-bucket"

    async def get(self, bucket: str, key: str) -> bytes:
        try:
            return self.objects[key]
        except KeyError:
            raise StorageNotFound(key) from None


class Embedder:
    def __init__(self) -> None:
        self.texts = 0

    async def embed_documents(self, items):
        self.texts += len(items)
        return EmbedResult(
            embedded=[
                EmbeddedChunk(chunk_index=i, embedding=[0.0] * EMBEDDING_V2_DIM)
                for i in range(len(items))
            ],
            failed=[],
        )


def _envelope(batch_seq: int, lines: list[str]) -> bytes:
    return orjson.dumps(
        {
            "_headers": {},
            "payload": {
                "device_id": "test-device",
                "session_id": SID,
                "batch_seq": batch_seq,
                "cwd": None,
                "events": [
                    {
                        "line_no": batch_seq * 1000 + i,
                        "employee_id": "emp-rechunk",
                        "raw": {"role": "user", "content": text},
                    }
                    for i, text in enumerate(lines)
                ],
            },
            "received_at": datetime.now(UTC).isoformat(),
        }
    )


def _transcript_batches() -> list[list[str]]:
    """Three batches; one benign escape and one credential, far apart."""
    batches = [
        [f"{MARKER} turn {b}.{i}: we walked through the chunk diff and the retire path."
         for i in range(60)]
        for b in range(3)
    ]
    batches[0][0] = "opened https://example.invalid/docs/a%20b.md to read the plan"
    batches[2][30] = f"export GITHUB_TOKEN={KEY}"
    return batches


@pytest.fixture
async def collapsed(live_db, monkeypatch):
    """A healthy session, then the #568 collapse applied to its chunks."""
    store = FakeStore()
    keys = []
    for seq, lines in enumerate(_transcript_batches()):
        key = f"raw/claude_code/{C}/2026/09/10/{SID}:{seq}.json"
        store.objects[key] = _envelope(seq, lines)
        keys.append(key)
    monkeypatch.setattr(storage, "_store", store)

    async def no_mining(**kwargs):
        raise AssertionError("the repair must never mine a session")

    monkeypatch.setattr("kb.handlers.claude_code._ext.extract_units_from_session", no_mining)

    async with db_module.raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers(customer_id, display_name, api_key_hash) "
            "VALUES ($1, 'r', 'r-hash') ON CONFLICT DO NOTHING",
            C,
        )
        queue_id = await conn.fetchval(
            """
            INSERT INTO ingestion_queue
                (customer_id, source_system, source_event_id, payload_s3_key,
                 payload_s3_keys, status, priority, version)
            VALUES ($1, 'claude_code', $2, ($3::text[])[1], $3::text[], 'done', 60, 3)
            RETURNING queue_id
            """,
            C, SID, keys,
        )

    # The worker's own healthy pass, with the fixed scrubber.
    await Normalizer(make_default_context(), store=store, embedder=Embedder()).process_queue_row(
        queue_id, C, SourceSystem.CLAUDE_CODE, SID, keys
    )

    async with db_module.raw_conn() as conn:
        healthy = {
            r["content_hash"]
            for r in await conn.fetch(
                "SELECT content_hash FROM chunks WHERE customer_id=$1 AND doc_id=$2 "
                "AND valid_to IS NULL AND kind='content'",
                C, DOC_ID,
            )
        }
        assert len(healthy) > 5
        # The session then ENDED (the sweep's marker is its newest key), so a
        # worker pass over this row would mine it -- the repair must not.
        await conn.execute(
            "UPDATE ingestion_queue SET payload_s3_keys = payload_s3_keys || $2::text "
            "WHERE queue_id = $1",
            queue_id, cron_marker_key("claude_code", C, SID),
        )
        # What #568 did: retire every content chunk, leave one `<redacted>`.
        await conn.execute(
            "UPDATE chunks SET valid_to = now(), last_seen_version = 0 "
            "WHERE customer_id=$1 AND doc_id=$2 AND kind='content' AND valid_to IS NULL",
            C, DOC_ID,
        )
        await conn.execute(
            """
            INSERT INTO chunks (chunk_id, doc_id, customer_id, chunk_index, content,
                content_hash, token_count, chunker_version, first_seen_version,
                last_seen_version, kind, embedding_v2, embedding_v2_model,
                embedding_v2_dim, visibility, title)
            SELECT doc_id || ':c_collapsed', doc_id, customer_id, 0, $3, $4, 3,
                chunker_version, 1, 1, 'content', embedding_v2, embedding_v2_model,
                embedding_v2_dim, visibility, title
            FROM chunks WHERE customer_id=$1 AND doc_id=$2 AND kind='content' LIMIT 1
            """,
            C, DOC_ID, rechunk.PLACEHOLDER, _chunk_hash(rechunk.PLACEHOLDER),
        )
        doc_row = dict(await conn.fetchrow(
            "SELECT version, content_hash, updated_at, body_size_bytes, metadata "
            "FROM documents WHERE customer_id=$1 AND doc_id=$2 AND valid_to IS NULL",
            C, DOC_ID,
        ))
        queue_row = dict(await conn.fetchrow(
            "SELECT status, version, payload_s3_keys, completed_at FROM ingestion_queue "
            "WHERE queue_id=$1", queue_id,
        ))
    return {"store": store, "keys": keys, "healthy": healthy, "doc": doc_row,
            "queue": queue_row, "queue_id": queue_id}


async def _live_content(conn) -> set[str]:
    return {
        r["content_hash"]
        for r in await conn.fetch(
            "SELECT content_hash FROM chunks WHERE customer_id=$1 AND doc_id=$2 "
            "AND valid_to IS NULL AND kind='content'",
            C, DOC_ID,
        )
    }


def _no_text(lines: list[dict]) -> None:
    printed = json.dumps(lines)
    assert KEY not in printed and MARKER not in printed and "a%20b" not in printed


async def test_dry_run_plans_the_full_body_and_writes_nothing(collapsed):
    lines: list[dict] = []
    summary = await rechunk.run(customers=[C], store=collapsed["store"], emit=lines.append)

    (doc,) = [line for line in lines if line["event"] == "rechunk.doc"]
    assert doc["doc_id"] == DOC_ID and doc["refused"] is None
    assert doc["hash_match"] is True and doc["missing_keys"] == 0
    assert doc["still_collapses"] is False
    assert doc["planned_chunks"] == len(collapsed["healthy"])
    # Every content chunk is embedded; the metadata chunk re-renders identically.
    assert doc["embed_texts"] == len(collapsed["healthy"])
    assert summary.would_write == 1 and summary.written == 0
    _no_text(lines)
    async with db_module.raw_conn() as conn:
        assert await _live_content(conn) == {_chunk_hash(rechunk.PLACEHOLDER)}


async def test_write_restores_the_healthy_chunks_and_nothing_else(collapsed):
    embedder = Embedder()
    lines: list[dict] = []
    summary = await rechunk.run(
        customers=[C], write=True, store=collapsed["store"], embedder=embedder, emit=lines.append
    )

    assert summary.written == 1, lines
    assert embedder.texts == len(collapsed["healthy"])
    _no_text(lines)
    async with db_module.raw_conn() as conn:
        assert await _live_content(conn) == collapsed["healthy"]
        contents = [r["content"] for r in await conn.fetch(
            "SELECT content FROM chunks WHERE customer_id=$1 AND doc_id=$2 AND valid_to IS NULL",
            C, DOC_ID,
        )]
        assert not any(KEY in c for c in contents)
        assert any(MARKER in c for c in contents)
        doc_row = dict(await conn.fetchrow(
            "SELECT version, content_hash, updated_at, body_size_bytes, metadata "
            "FROM documents WHERE customer_id=$1 AND doc_id=$2 AND valid_to IS NULL",
            C, DOC_ID,
        ))
        assert doc_row == collapsed["doc"]
        assert await conn.fetchval(
            "SELECT count(*) FROM documents WHERE customer_id=$1", C
        ) == 1  # no unit documents: nothing was mined
        queue_row = dict(await conn.fetchrow(
            "SELECT status, version, payload_s3_keys, completed_at FROM ingestion_queue "
            "WHERE queue_id=$1", collapsed["queue_id"],
        ))
        assert queue_row == collapsed["queue"]

    # Idempotent: once repaired, the selection no longer finds it.
    again: list[dict] = []
    assert (await rechunk.run(customers=[C], store=collapsed["store"], emit=again.append)).selected == 0


@pytest.mark.parametrize("case", ["missing_keys", "queue_row_active", "still_collapses"])
async def test_refusals_write_nothing(collapsed, case, monkeypatch):
    if case == "still_collapses":
        # An image without #587: the body scrub still returns the placeholder.
        async def old_scrub(texts):
            return [rechunk.PLACEHOLDER for _ in texts]

        monkeypatch.setattr("engine.ingest.normalizer.redact_texts_async", old_scrub)
        expected = "still_collapses"
    elif case == "missing_keys":
        for key in collapsed["keys"][1:]:
            del collapsed["store"].objects[key]
        expected = "partial_render"
    else:
        async with db_module.raw_conn() as conn:
            await conn.execute(
                "UPDATE ingestion_queue SET status='pending' WHERE queue_id=$1",
                collapsed["queue_id"],
            )
        expected = "queue_row_active"
    lines: list[dict] = []
    summary = await rechunk.run(
        customers=[C], write=True, store=collapsed["store"], embedder=Embedder(), emit=lines.append
    )
    (doc,) = [line for line in lines if line["event"] == "rechunk.doc"]
    assert doc["refused"] == expected and summary.written == 0
    if case == "missing_keys":
        assert doc["missing_keys"] == 2
    async with db_module.raw_conn() as conn:
        assert await _live_content(conn) == {_chunk_hash(rechunk.PLACEHOLDER)}


def test_importing_the_script_registers_the_session_connectors() -> None:
    """In a FRESH interpreter: this suite's conftest imports `kb.handlers`
    for every test, which is exactly how the script shipped without it."""
    import subprocess
    import sys

    code = (
        "import scripts.rechunk_collapsed_sessions\n"
        "from engine.ingest.handlers.registry import list_registered\n"
        "print(sorted(str(s) for s in list_registered()))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=120
    )
    for source in ("claude_code", "codex", "pi"):
        assert source in out.stdout, out.stdout
