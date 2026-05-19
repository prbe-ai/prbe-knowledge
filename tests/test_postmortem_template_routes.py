"""Route tests for customer-postmortem-templates.

GET / PUT / GET-effective. The internal-key check is enforced on
every endpoint. PUT validates body/path consistency and surfaces
upsert_override's ValueError as 422.
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from services.ingestion.main import app
from shared.config import Settings, get_settings
from shared.db import close_pool, init_pool, raw_conn

pytestmark = pytest.mark.asyncio


_INTERNAL_KEY = "test-template-key"
_CUSTOMER = f"tmpl-test-{uuid.uuid4().hex[:8]}"


@pytest.fixture(autouse=True)
def _patch_internal_key(monkeypatch) -> None:
    monkeypatch.setenv("INTERNAL_KNOWLEDGE_API_KEY", _INTERNAL_KEY)
    get_settings.cache_clear()


@pytest_asyncio.fixture
async def client(live_db: None, settings: Settings) -> AsyncClient:
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers (customer_id, display_name, api_key_hash, r2_bucket) "
            "VALUES ($1, $2, $3, $4) ON CONFLICT (customer_id) DO NOTHING",
            _CUSTOMER, f"tmpl {_CUSTOMER}", "h", f"b-{_CUSTOMER}",
        )

    await close_pool()
    async with (
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac,
        app.router.lifespan_context(app),
    ):
        yield ac
    await init_pool(settings)


def _hdrs() -> dict:
    return {
        "x-internal-knowledge-key": _INTERNAL_KEY,
        "content-type": "application/json",
    }


async def _seed_approved_doc(customer_id: str, doc_id: str) -> None:
    """Insert an approved document so the doc_ref override resolves."""
    now = datetime.now(UTC)
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO documents (
                doc_id, version, customer_id,
                source_system, source_id, source_url,
                doc_class, doc_type, content_type,
                content_hash, body_size_bytes, body_token_count,
                created_at, updated_at, valid_from, ingested_at,
                acl, metadata, entities, attachments, doc_references,
                normalizer_version, visibility
            )
            VALUES (
                $1, 1, $2,
                'wiki', $1, '',
                'compiled_wiki', 'wiki.runbook', 'text/markdown',
                $3, 0, 0,
                $4, $4, $4, $4,
                '{"principals":[],"captured_at":"2026-05-17T00:00:00Z"}'::jsonb,
                '{}'::jsonb, '[]'::jsonb, '[]'::jsonb, '[]'::jsonb,
                'v1', 'approved'
            )
            ON CONFLICT (customer_id, doc_id, version) DO NOTHING
            """,
            doc_id, customer_id, f"hash-{doc_id}", now,
        )


async def test_get_returns_null_when_no_override(client) -> None:
    r = await client.get(
        f"/api/customer-postmortem-templates/{_CUSTOMER}",
        headers=_hdrs(),
    )
    assert r.status_code == 200
    # FastAPI serializes Optional[TemplateRow] as `null` JSON when absent.
    assert r.json() is None


async def test_put_inline_then_get_returns_it(client) -> None:
    body = {
        "customer_id": _CUSTOMER,
        "mode": "inline",
        "body_markdown": "## Summary\n## Root cause\n## Timeline\n",
        "ref_doc_id": None,
    }
    p = await client.put(
        f"/api/customer-postmortem-templates/{_CUSTOMER}",
        headers=_hdrs(),
        json=body,
    )
    assert p.status_code == 200, p.text
    assert p.json()["mode"] == "inline"

    g = await client.get(
        f"/api/customer-postmortem-templates/{_CUSTOMER}",
        headers=_hdrs(),
    )
    assert g.status_code == 200
    data = g.json()
    assert data is not None
    assert data["mode"] == "inline"
    assert data["body_markdown"].startswith("## Summary")


async def test_effective_returns_default_when_no_override(client) -> None:
    r = await client.get(
        f"/api/customer-postmortem-templates/{_CUSTOMER}/effective",
        headers=_hdrs(),
    )
    assert r.status_code == 200
    data = r.json()
    assert data["source"] == "default"
    assert len(data["body_markdown"]) > 0


async def test_effective_returns_inline_override(client) -> None:
    body = {
        "customer_id": _CUSTOMER,
        "mode": "inline",
        "body_markdown": "### Custom template\nfor this customer\n",
        "ref_doc_id": None,
    }
    await client.put(
        f"/api/customer-postmortem-templates/{_CUSTOMER}",
        headers=_hdrs(),
        json=body,
    )
    r = await client.get(
        f"/api/customer-postmortem-templates/{_CUSTOMER}/effective",
        headers=_hdrs(),
    )
    assert r.status_code == 200
    data = r.json()
    assert data["source"] == "inline_override"
    assert data["body_markdown"].startswith("### Custom template")


async def test_put_doc_ref_with_missing_doc_returns_422(client) -> None:
    body = {
        "customer_id": _CUSTOMER,
        "mode": "doc_ref",
        "body_markdown": None,
        "ref_doc_id": "wiki:runbook:does-not-exist",
    }
    r = await client.put(
        f"/api/customer-postmortem-templates/{_CUSTOMER}",
        headers=_hdrs(),
        json=body,
    )
    assert r.status_code == 422, r.text
    assert "not readable" in r.json()["detail"]


async def test_put_doc_ref_with_existing_doc_succeeds(client) -> None:
    doc_id = "wiki:runbook:checkout-template"
    await _seed_approved_doc(_CUSTOMER, doc_id)
    body = {
        "customer_id": _CUSTOMER,
        "mode": "doc_ref",
        "body_markdown": None,
        "ref_doc_id": doc_id,
    }
    r = await client.put(
        f"/api/customer-postmortem-templates/{_CUSTOMER}",
        headers=_hdrs(),
        json=body,
    )
    assert r.status_code == 200, r.text
    assert r.json()["mode"] == "doc_ref"
    assert r.json()["ref_doc_id"] == doc_id


async def test_put_path_body_customer_id_mismatch_returns_400(client) -> None:
    body = {
        "customer_id": "different-customer",
        "mode": "inline",
        "body_markdown": "## Template body\n",
        "ref_doc_id": None,
    }
    r = await client.put(
        f"/api/customer-postmortem-templates/{_CUSTOMER}",
        headers=_hdrs(),
        json=body,
    )
    assert r.status_code == 400
    assert "does not match" in r.json()["detail"]


async def test_bad_internal_key_returns_401(client) -> None:
    r = await client.get(
        f"/api/customer-postmortem-templates/{_CUSTOMER}",
        headers={"x-internal-knowledge-key": "wrong"},
    )
    assert r.status_code == 401
