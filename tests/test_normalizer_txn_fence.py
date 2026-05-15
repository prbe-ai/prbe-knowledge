"""Regression test for incident 2026-04-29.

The Normalizer's `_persist` used to call `embedder.embed_many` while a
`with_tenant` write transaction was open. For long Claude Code sessions
that meant 60-120s row locks, which caused concurrent workers' graph_nodes
upserts to hit the 30s db_statement_timeout and DLQ — 9 production rows
went to DLQ before the fix.

The fix splits `_persist` into Phase A (no write txn, embed outside any
asyncpg connection) and Phase B (one short write txn). This test pins
that property: while `embed_many` is in flight, the asyncpg pool must
have zero acquired connections — i.e. nothing is sitting in a transaction
holding locks.

If someone later moves the embed call back inside `with_tenant`, this
test fires before the change ships.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest

from services.ingestion.handlers.base import ConnectorContext
from services.ingestion.handlers.slack import SlackConnector  # noqa: F401 — registers
from services.ingestion.normalizer import Normalizer
from shared import db as db_module
from shared.config import Settings
from shared.constants import EMBEDDING_DIM, SourceSystem
from shared.embeddings import EmbeddedChunk, EmbedResult

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "slack" / "message_simple.json"


@dataclass
class _StubStore:
    blobs: dict[tuple[str, str], bytes]

    async def bucket_for(self, customer_id: str) -> str:
        return f"test-bucket-{customer_id}"

    async def get(self, bucket: str, key: str) -> bytes:
        return self.blobs[(bucket, key)]


def _wrap_payload(payload: dict[str, Any]) -> bytes:
    return json.dumps({"_headers": {}, "payload": payload}).encode("utf-8")


def _base_payload() -> dict[str, Any]:
    return copy.deepcopy(json.loads(FIXTURE_PATH.read_text()))


async def _seed_customer(customer_id: str) -> None:
    async with db_module.raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO customers (customer_id, display_name, api_key_hash)
            VALUES ($1, 'test', 'test-hash')
            ON CONFLICT DO NOTHING
            """,
            customer_id,
        )


class _ConnectionFenceEmbedder:
    """Embedder that records the asyncpg pool's in-flight connection count
    each time `embed_documents` is invoked.

    The fence: in-flight count must be 0 during the embed call. Anything else
    means a `with_tenant` (or raw_conn) block is holding a connection across
    the Gemini round trip — the exact bug from 2026-04-29.
    """

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self.calls: list[int] = []
        self._dim = dim

    async def embed_documents(self, items: list) -> EmbedResult:
        pool = db_module.get_pool()
        # asyncpg.Pool exposes get_size (total connections owned) and
        # get_idle_size (those not currently checked out). Their difference
        # is the number of connections held by code right now. Phase A
        # embeds AFTER closing its read txn, so this should be 0.
        in_flight = pool.get_size() - pool.get_idle_size()
        self.calls.append(in_flight)
        # Return zero-vector matches so the rest of the pipeline succeeds.
        return EmbedResult(
            embedded=[
                EmbeddedChunk(chunk_index=i, embedding=[0.0] * self._dim)
                for i in range(len(items))
            ],
            failed=[],
        )


@pytest.mark.asyncio
async def test_embed_many_runs_with_no_db_connection_in_flight(live_db) -> None:
    """Phase A must close its read txn BEFORE calling embed_many.

    Pin the txn-fence: pool.in_flight must be 0 throughout each embed call.
    """
    customer_id = "cust-txn-fence"
    await _seed_customer(customer_id)

    store = _StubStore(blobs={})
    embedder = _ConnectionFenceEmbedder()
    settings = Settings(environment="local")
    ctx = ConnectorContext(settings=settings, http=httpx.AsyncClient())
    normalizer = Normalizer(ctx, store=store, embedder=embedder)  # type: ignore[arg-type]

    payload = _base_payload()
    bucket = await store.bucket_for(customer_id)
    key = f"raw/slack/{customer_id}/evt-fence-1.json"
    store.blobs[(bucket, key)] = _wrap_payload(payload)

    await normalizer.process_queue_row(
        queue_id=1,
        customer_id=customer_id,
        source_system=SourceSystem.SLACK,
        source_event_id="evt-fence-1",
        payload_s3_keys=[key],
    )

    assert embedder.calls, (
        "embed_documents should have been called at least once for the slack message"
    )
    assert all(c == 0 for c in embedder.calls), (
        "asyncpg connection in flight during embed_documents — Phase A txn fence is "
        f"broken. Per-call in_flight counts: {embedder.calls!r}. The embed "
        "call must run AFTER `with_tenant` exits, not inside it."
    )
