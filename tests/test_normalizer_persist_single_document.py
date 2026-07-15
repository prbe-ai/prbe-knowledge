"""Tests for the public Normalizer.persist_single_document entry.

Verifies one-shot persistence: a Document → row in `documents` + chunks
in `chunks`, with the source_system and doc_type stamped exactly as the
caller passed them (no override or downgrade).

Live Postgres + minio required (DATABASE_URL + R2_* set in conftest).
"""
from __future__ import annotations

import hashlib
import os
import uuid
from datetime import UTC, datetime

import asyncpg
import httpx
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
from engine.shared.db import raw_conn
from engine.shared.models import ACLPrincipal, ACLSnapshot, Document

pytestmark = pytest.mark.asyncio


def _new_customer_id() -> str:
    return f"persist-test-{uuid.uuid4().hex[:8]}"


@pytest.fixture
async def customer_id(live_db: None):
    """Isolated customer row; pool is already up via live_db."""
    cid = _new_customer_id()
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers (customer_id, display_name, api_key_hash, r2_bucket) "
            "VALUES ($1, $2, $3, $4) ON CONFLICT (customer_id) DO NOTHING",
            cid, f"persist test {cid}", "h", f"b-{cid}",
        )
    yield cid
    # live_db teardown truncates everything; explicit cleanup is defensive only.
    async with raw_conn() as conn:
        await conn.execute("DELETE FROM chunks WHERE customer_id = $1", cid)
        await conn.execute("DELETE FROM documents WHERE customer_id = $1", cid)
        await conn.execute("DELETE FROM customers WHERE customer_id = $1", cid)


@pytest.fixture
async def normalizer(live_db: None) -> Normalizer:
    settings = Settings(environment="local")
    ctx = ConnectorContext(settings=settings, http=httpx.AsyncClient())
    return Normalizer(ctx)


def _make_doc(
    customer_id: str,
    *,
    doc_id: str = "pd:investigation:T-001:v1",
    source_system: SourceSystem = SourceSystem.PAGERDUTY,
    doc_type: str = DocType.INCIDENT_INVESTIGATION,
) -> Document:
    now = datetime.now(UTC)
    body = "## Hypothesis\n\nThe checkout-svc DB pool exhausted because of a runaway query introduced in the v1.42 deploy. Recent deploys + Sentry stack traces point at `select_top_orders` in `apps/checkout/queries.py`.\n\n## Evidence\n\n- 2 deploys to checkout-svc in the last 24h (github:pr:1234, github:pr:1240)\n- 18 open Sentry issues mentioning `connection pool` since the deploy\n- Slack #incidents thread `slack:msg:abc123`\n"
    content_hash = hashlib.sha256(body.encode()).hexdigest()
    return Document(
        doc_id=doc_id,
        customer_id=customer_id,
        source_system=source_system,
        source_id=doc_id,
        source_url="",
        doc_class=DocClass.AGENT_ARTIFACT,
        doc_type=doc_type,
        content_type="text/markdown",
        content_hash=content_hash,
        title="Investigation: Checkout DB pool exhausted",
        body=body,
        body_preview=body[:280],
        body_size_bytes=len(body.encode("utf-8")),
        body_token_count=400,
        created_at=now,
        updated_at=now,
        valid_from=now,
        ingested_at=now,
        acl=ACLSnapshot(
            principals=[
                ACLPrincipal(
                    principal_type=PrincipalType.WORKSPACE,
                    principal_id=customer_id,
                    permission=Permission.READ,
                ),
            ],
            captured_at=now,
        ),
        parent_doc_id="pd:incident:T-001",
    )


async def test_persist_writes_document_and_chunks(
    customer_id: str, normalizer: Normalizer,
) -> None:
    doc = _make_doc(customer_id)
    outcome = await normalizer.persist_single_document(customer_id, doc)
    assert outcome.doc_ids == [doc.doc_id]
    assert outcome.chunk_count >= 1
    assert outcome.failed_chunk_count == 0

    dsn = os.environ["DATABASE_URL"]
    conn = await asyncpg.connect(dsn=dsn)
    try:
        row = await conn.fetchrow(
            "SELECT doc_id, source_system::text AS source_system, "
            "       doc_type::text AS doc_type, doc_class::text AS doc_class, "
            "       title, parent_doc_id "
            "FROM documents WHERE customer_id = $1 AND doc_id = $2 "
            "AND valid_to IS NULL",
            customer_id, doc.doc_id,
        )
        assert row is not None
        assert row["source_system"] == "pagerduty"
        assert row["doc_type"] == "incident.investigation"
        assert row["doc_class"] == "agent_artifact"
        assert row["title"] == "Investigation: Checkout DB pool exhausted"
        assert row["parent_doc_id"] == "pd:incident:T-001"

        chunk_count = await conn.fetchval(
            "SELECT COUNT(*) FROM chunks WHERE customer_id = $1 AND doc_id = $2",
            customer_id, doc.doc_id,
        )
        assert chunk_count >= 1
    finally:
        await conn.close()


async def test_persist_stamps_incident_io_source_system_correctly(
    customer_id: str, normalizer: Normalizer,
) -> None:
    """Confirm incident.io investigations are stamped as `incident_io`,
    not coerced to `custom_ingest`."""
    doc = _make_doc(
        customer_id,
        doc_id="iio:investigation:T-002:v1",
        source_system=SourceSystem.INCIDENT_IO,
    )
    await normalizer.persist_single_document(customer_id, doc)

    dsn = os.environ["DATABASE_URL"]
    conn = await asyncpg.connect(dsn=dsn)
    try:
        row = await conn.fetchrow(
            "SELECT source_system::text AS source_system FROM documents "
            "WHERE customer_id = $1 AND doc_id = $2 AND valid_to IS NULL",
            customer_id, doc.doc_id,
        )
        assert row["source_system"] == "incident_io"
    finally:
        await conn.close()


async def test_persist_emits_chunks_searchable_by_doc_type(
    customer_id: str, normalizer: Normalizer,
) -> None:
    """Filter docs by doc_type='incident.investigation' should return
    the persisted doc — confirms retrieval-parity claim."""
    doc = _make_doc(customer_id)
    await normalizer.persist_single_document(customer_id, doc)

    dsn = os.environ["DATABASE_URL"]
    conn = await asyncpg.connect(dsn=dsn)
    try:
        rows = await conn.fetch(
            "SELECT doc_id FROM documents WHERE customer_id = $1 "
            "AND doc_type = 'incident.investigation' AND valid_to IS NULL",
            customer_id,
        )
        doc_ids = {r["doc_id"] for r in rows}
        assert doc.doc_id in doc_ids
    finally:
        await conn.close()
