from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest
from httpx import ASGITransport

from services.ingestion.handlers.base import ConnectorContext, make_default_context
from services.ingestion.handlers.custom_ingest import CustomIngestConnector
from services.ingestion.normalizer import Normalizer
from shared.config import Settings, get_settings
from shared.constants import DocType, Permission, PrincipalType, SourceSystem
from shared.custom_ingest import CustomIngestEnvelope, document_content_hash, source_event_id
from shared.db import raw_conn
from shared.embeddings import reset_embedder
from shared.models import WebhookEvent
from shared.storage import reset_store

CUSTOMER = "cust-custom-ingest"
INTERNAL_KEY = "test-internal-key-32bytes-padding-padding"


def _payload(body: str = "Payments API timeout") -> dict:
    return {
        "source_key": "acme_internal_incidents",
        "batch_id": "batch-1",
        "documents": [
            {
                "id": "incident-123",
                "type": "incident",
                "title": "Payments API timeout",
                "body": body,
                "url": "https://internal.example/incidents/123",
                "author": {
                    "id": "jane@example.com",
                    "name": "Jane Doe",
                    "email": "jane@example.com",
                },
                "created_at": "2026-05-05T18:10:00Z",
                "updated_at": "2026-05-05T19:30:00Z",
                "metadata": {"service": "payments", "severity": "p1"},
                "acl": [
                    {
                        "type": "user",
                        "id": "jane@example.com",
                        "permission": "read",
                    }
                ],
            }
        ],
    }


def _headers() -> dict[str, str]:
    return {
        "content-type": "application/json",
        "x-internal-knowledge-key": INTERNAL_KEY,
        "x-prbe-customer": CUSTOMER,
    }


@pytest.fixture(autouse=True)
def _patch_env(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> None:
    monkeypatch.setenv(
        "TOKEN_ENCRYPTION_KEY",
        settings.token_encryption_key.get_secret_value(),
    )
    monkeypatch.setenv("INTERNAL_KNOWLEDGE_API_KEY", INTERNAL_KEY)
    monkeypatch.setenv("ENVIRONMENT", "local")
    reset_embedder()
    reset_store()
    get_settings.cache_clear()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_custom_ingest_route_requires_internal_key() -> None:
    from services.ingestion.main import app

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await client.post(
            "/api/custom-ingest/documents",
            json=_payload(),
            headers={"content-type": "application/json", "x-prbe-customer": CUSTOMER},
        )

    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_custom_ingest_route_validates_source_key() -> None:
    from services.ingestion.main import app

    body = _payload()
    body["source_key"] = "Bad Source Key"
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await client.post(
            "/api/custom-ingest/documents",
            json=body,
            headers=_headers(),
        )

    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_custom_ingest_route_rejects_reserved_metadata_body() -> None:
    from services.ingestion.main import app

    body = _payload()
    body["documents"][0]["metadata"]["body"] = "do not duplicate body here"
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await client.post(
            "/api/custom-ingest/documents",
            json=body,
            headers=_headers(),
        )

    assert resp.status_code == 422


def test_custom_ingest_hash_includes_timestamps() -> None:
    first = CustomIngestEnvelope.model_validate(_payload())
    second_payload = _payload()
    second_payload["documents"][0]["updated_at"] = "2026-05-05T20:00:00Z"
    second = CustomIngestEnvelope.model_validate(second_payload)

    first_hash = document_content_hash(first.source_key, first.documents[0])
    second_hash = document_content_hash(second.source_key, second.documents[0])

    assert first_hash != second_hash
    assert (
        source_event_id(first, first.documents[0], first_hash)
        != source_event_id(second, second.documents[0], second_hash)
    )


@pytest.mark.asyncio
async def test_custom_ingest_route_enqueues_and_dedupes_retries(
    live_db,
) -> None:
    from services.ingestion.main import app

    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO customers (customer_id, display_name, api_key_hash)
            VALUES ($1, 'Custom Ingest', 'x')
            ON CONFLICT DO NOTHING
            """,
            CUSTOMER,
        )

    transport = ASGITransport(app=app)
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://t") as client,
        app.router.lifespan_context(app),
    ):
        first = await client.post(
            "/api/custom-ingest/documents",
            json=_payload(),
            headers=_headers(),
        )
        second = await client.post(
            "/api/custom-ingest/documents",
            json=_payload(),
            headers=_headers(),
        )

    assert first.status_code == 202, first.text
    assert first.json()["accepted"] == 1
    assert first.json()["duplicates"] == 0
    assert second.status_code == 202, second.text
    assert second.json()["accepted"] == 0
    assert second.json()["duplicates"] == 1

    async with raw_conn() as conn:
        queue_count = await conn.fetchval(
            """
            SELECT COUNT(*)
            FROM ingestion_queue
            WHERE customer_id = $1 AND source_system = $2
            """,
            CUSTOMER,
            SourceSystem.CUSTOM_INGEST.value,
        )

    assert queue_count == 1


@pytest.mark.asyncio
async def test_custom_ingest_connector_normalization() -> None:
    async with httpx.AsyncClient() as client:
        connector = CustomIngestConnector(
            ConnectorContext(settings=get_settings(), http=client)
        )
        event = WebhookEvent(
            customer_id=CUSTOMER,
            source_system=SourceSystem.CUSTOM_INGEST,
            source_event_id="acme_internal_incidents:doc:hash",
            received_at=datetime(2026, 5, 5, 19, 31, tzinfo=UTC),
            raw_payload={
                "source_key": "acme_internal_incidents",
                "batch_id": "batch-1",
                "source_event_id": "acme_internal_incidents:doc:hash",
                "content_hash": "abc123",
                "received_at": "2026-05-05T19:31:00Z",
                "document": _payload()["documents"][0],
            },
        )
        result = await connector.normalize(event, {})

    assert len(result.documents) == 1
    doc = result.documents[0]
    assert doc.doc_id == f"custom_ingest:{CUSTOMER}:acme_internal_incidents:incident-123"
    assert doc.source_system == SourceSystem.CUSTOM_INGEST
    assert doc.source_id == "acme_internal_incidents:incident-123"
    assert doc.doc_type == DocType.CUSTOM_DOCUMENT
    assert doc.metadata["source_key"] == "acme_internal_incidents"
    assert doc.metadata["custom_document_type"] == "incident"
    assert "acl" not in doc.metadata
    assert doc.author_id == "jane@example.com"
    assert doc.body == "Payments API timeout"
    assert "body" not in doc.metadata
    assert {
        (principal.principal_type, principal.principal_id, principal.permission)
        for principal in doc.acl.principals
    } == {
        (PrincipalType.WORKSPACE, CUSTOMER, Permission.READ),
        (PrincipalType.WORKSPACE, CUSTOMER, Permission.WRITE),
    }
    assert {
        (row.principal_type, row.principal_id, row.permission)
        for row in result.acl_snapshots
    } == {
        (PrincipalType.WORKSPACE, CUSTOMER, Permission.READ),
        (PrincipalType.WORKSPACE, CUSTOMER, Permission.WRITE),
    }


@pytest.mark.asyncio
async def test_custom_ingest_idempotency_versioning_and_chunks(
    live_db,
) -> None:
    from services.ingestion.main import app

    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO customers (customer_id, display_name, api_key_hash)
            VALUES ($1, 'Custom Ingest', 'x')
            ON CONFLICT DO NOTHING
            """,
            CUSTOMER,
        )

    transport = ASGITransport(app=app)
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://t") as client,
        app.router.lifespan_context(app),
    ):
        first = await client.post(
            "/api/custom-ingest/documents",
            json=_payload("Payments API timeout in checkout"),
            headers=_headers(),
        )
        assert first.status_code == 202, first.text
        first_queue_id = await _latest_queue_id()
        await _process_queue_id(first_queue_id)

        retry = await client.post(
            "/api/custom-ingest/documents",
            json=_payload("Payments API timeout in checkout"),
            headers=_headers(),
        )
        assert retry.status_code == 202, retry.text
        assert retry.json()["duplicates"] == 1

        changed = await client.post(
            "/api/custom-ingest/documents",
            json=_payload("Payments API timeout resolved after pool resize"),
            headers=_headers(),
        )
        assert changed.status_code == 202, changed.text
        changed_queue_id = await _latest_queue_id()
        assert changed_queue_id != first_queue_id
        await _process_queue_id(changed_queue_id)

    async with raw_conn() as conn:
        doc_versions = await conn.fetchval(
            """
            SELECT COUNT(*)
            FROM documents
            WHERE customer_id = $1 AND source_system = $2
            """,
            CUSTOMER,
            SourceSystem.CUSTOM_INGEST.value,
        )
        live_version = await conn.fetchval(
            """
            SELECT version
            FROM documents
            WHERE customer_id = $1 AND source_system = $2 AND valid_to IS NULL
            """,
            CUSTOMER,
            SourceSystem.CUSTOM_INGEST.value,
        )
        live_body = await conn.fetchval(
            """
            SELECT string_agg(c.content, '' ORDER BY c.chunk_index)
            FROM chunks c
            JOIN documents d ON d.customer_id = c.customer_id AND d.doc_id = c.doc_id
            WHERE d.customer_id = $1
              AND d.source_system = $2
              AND d.valid_to IS NULL
              AND c.valid_to IS NULL
              AND c.kind = 'content'
            """,
            CUSTOMER,
            SourceSystem.CUSTOM_INGEST.value,
        )

    assert doc_versions == 2
    assert live_version == 2
    assert "Payments API timeout resolved after pool resize" in live_body


async def _latest_queue_id() -> int:
    async with raw_conn() as conn:
        queue_id = await conn.fetchval(
            """
            SELECT queue_id
            FROM ingestion_queue
            WHERE customer_id = $1 AND source_system = $2
            ORDER BY queue_id DESC
            LIMIT 1
            """,
            CUSTOMER,
            SourceSystem.CUSTOM_INGEST.value,
        )
    assert isinstance(queue_id, int)
    return queue_id


async def _process_queue_id(queue_id: int) -> None:
    async with raw_conn() as conn:
        row = await conn.fetchrow(
            """
            SELECT source_event_id, payload_s3_keys
            FROM ingestion_queue
            WHERE queue_id = $1
            """,
            queue_id,
        )
    assert row is not None
    ctx = make_default_context()
    try:
        normalizer = Normalizer(ctx)
        await normalizer.process_queue_row(
            queue_id=queue_id,
            customer_id=CUSTOMER,
            source_system=SourceSystem.CUSTOM_INGEST,
            source_event_id=row["source_event_id"],
            payload_s3_keys=list(row["payload_s3_keys"]),
        )
    finally:
        await ctx.http.aclose()
