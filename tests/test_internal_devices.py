"""Internal /api/devices/* endpoints (X-Internal-Knowledge-Key gated)."""
from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport

from services.ingestion.main import app
from shared.config import Settings, get_settings
from shared.db import close_pool, init_pool, raw_conn

CUSTOMER = "dev-test-cust"
EMPLOYEE = "emp-1"
DEVICE = "dev-1"
TOKEN_PLAINTEXT = "secret-XYZ"
TOKEN_HASH = hashlib.sha256(TOKEN_PLAINTEXT.encode("utf-8")).hexdigest()


def _hash(plaintext: str) -> str:
    return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()


@pytest.fixture(autouse=True)
def _patch_internal_key(monkeypatch) -> None:
    monkeypatch.setenv("INTERNAL_KNOWLEDGE_API_KEY", "test-internal-key")
    get_settings.cache_clear()  # type: ignore[attr-defined]


@pytest_asyncio.fixture
async def client(live_db: None, settings: Settings) -> AsyncIterator[httpx.AsyncClient]:
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers(customer_id, display_name, api_key_hash) VALUES ($1, 'd', 'd-hash') ON CONFLICT DO NOTHING",
            CUSTOMER,
        )

    await close_pool()
    transport = ASGITransport(app=app)
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://t") as c,
        app.router.lifespan_context(app),
    ):
        yield c
    await init_pool(settings)


def _hdr() -> dict[str, str]:
    return {"X-Internal-Knowledge-Key": "test-internal-key"}


@pytest.mark.asyncio
async def test_register_device_persists_row_and_mapping(client: httpx.AsyncClient) -> None:
    resp = await client.post(
        "/api/devices/register",
        json={
            "customer_id": CUSTOMER,
            "employee_id": EMPLOYEE,
            "device_id": DEVICE,
            "token_hash": TOKEN_HASH,
            "os": "macos",
            "hostname": "mahits-mbp",
        },
        headers=_hdr(),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body == {"customer_id": CUSTOMER, "device_id": DEVICE, "status": "active"}

    async with raw_conn() as conn:
        row = await conn.fetchrow(
            "SELECT webhook_secret, status, device_metadata FROM integration_tokens "
            "WHERE customer_id=$1 AND source_system='claude_code' AND device_id=$2",
            CUSTOMER, DEVICE,
        )
        mapping = await conn.fetchrow(
            "SELECT external_id, external_name FROM customer_source_mapping "
            "WHERE customer_id=$1 AND source_system='claude_code' AND external_id=$2",
            CUSTOMER, DEVICE,
        )
    assert row is not None
    assert row["webhook_secret"] == TOKEN_HASH
    assert row["status"] == "active"
    assert mapping is not None
    assert mapping["external_name"] == "mahits-mbp"


@pytest.mark.asyncio
async def test_register_requires_internal_knowledge_key(client: httpx.AsyncClient) -> None:
    resp = await client.post(
        "/api/devices/register",
        json={
            "customer_id": CUSTOMER, "employee_id": EMPLOYEE, "device_id": DEVICE,
            "token_hash": TOKEN_HASH,
        },
        # No X-Internal-Knowledge-Key
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_verify_token_resolves_to_device_claims(client: httpx.AsyncClient) -> None:
    await client.post(
        "/api/devices/register",
        json={
            "customer_id": CUSTOMER, "employee_id": EMPLOYEE, "device_id": DEVICE,
            "token_hash": TOKEN_HASH, "hostname": "h",
        },
        headers=_hdr(),
    )
    resp = await client.post(
        "/api/devices/verify-token",
        json={"token_hash": TOKEN_HASH},
        headers=_hdr(),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["customer_id"] == CUSTOMER
    assert body["device_id"] == DEVICE
    assert body["employee_id"] == EMPLOYEE
    assert body["status"] == "active"


@pytest.mark.asyncio
async def test_verify_token_unknown_hash_is_401(client: httpx.AsyncClient) -> None:
    resp = await client.post(
        "/api/devices/verify-token",
        json={"token_hash": _hash("nope")},
        headers=_hdr(),
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_verify_token_after_revoke_is_401(client: httpx.AsyncClient) -> None:
    await client.post(
        "/api/devices/register",
        json={
            "customer_id": CUSTOMER, "employee_id": EMPLOYEE, "device_id": DEVICE,
            "token_hash": TOKEN_HASH,
        },
        headers=_hdr(),
    )
    await client.post(
        f"/api/devices/{DEVICE}/revoke",
        json={"customer_id": CUSTOMER},
        headers=_hdr(),
    )
    resp = await client.post(
        "/api/devices/verify-token",
        json={"token_hash": TOKEN_HASH},
        headers=_hdr(),
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_heartbeat_updates_device_metadata(client: httpx.AsyncClient) -> None:
    await client.post(
        "/api/devices/register",
        json={
            "customer_id": CUSTOMER, "employee_id": EMPLOYEE, "device_id": DEVICE,
            "token_hash": TOKEN_HASH,
        },
        headers=_hdr(),
    )
    resp = await client.post(
        f"/api/devices/{DEVICE}/heartbeat",
        json={"customer_id": CUSTOMER},
        headers=_hdr(),
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["updated"] is True

    async with raw_conn() as conn:
        meta = await conn.fetchval(
            "SELECT device_metadata FROM integration_tokens "
            "WHERE customer_id=$1 AND source_system='claude_code' AND device_id=$2",
            CUSTOMER, DEVICE,
        )
    import orjson
    if isinstance(meta, (str, bytes, bytearray)):
        meta = orjson.loads(meta)
    assert "last_heartbeat_at" in meta


@pytest.mark.asyncio
async def test_heartbeat_unknown_device_is_404(client: httpx.AsyncClient) -> None:
    resp = await client.post(
        f"/api/devices/{DEVICE}/heartbeat",
        json={"customer_id": CUSTOMER},
        headers=_hdr(),
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_revoke_marks_device_revoked(client: httpx.AsyncClient) -> None:
    await client.post(
        "/api/devices/register",
        json={
            "customer_id": CUSTOMER, "employee_id": EMPLOYEE, "device_id": DEVICE,
            "token_hash": TOKEN_HASH,
        },
        headers=_hdr(),
    )
    resp = await client.post(
        f"/api/devices/{DEVICE}/revoke",
        json={"customer_id": CUSTOMER},
        headers=_hdr(),
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["updated"] is True

    async with raw_conn() as conn:
        status = await conn.fetchval(
            "SELECT status FROM integration_tokens "
            "WHERE customer_id=$1 AND source_system='claude_code' AND device_id=$2",
            CUSTOMER, DEVICE,
        )
    assert status == "revoked"


@pytest.mark.asyncio
async def test_list_devices_returns_per_device_rows(client: httpx.AsyncClient) -> None:
    for n in range(2):
        await client.post(
            "/api/devices/register",
            json={
                "customer_id": CUSTOMER,
                "employee_id": f"emp-{n}",
                "device_id": f"dev-{n}",
                "token_hash": _hash(f"t-{n}"),
                "hostname": f"laptop-{n}",
            },
            headers=_hdr(),
        )
    resp = await client.get(
        "/api/devices",
        params={"customer_id": CUSTOMER},
        headers=_hdr(),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    ids = sorted(d["device_id"] for d in body["devices"])
    assert ids == ["dev-0", "dev-1"]


@pytest.mark.asyncio
async def test_re_register_same_device_resets_status_and_merges_metadata(
    client: httpx.AsyncClient,
) -> None:
    await client.post(
        "/api/devices/register",
        json={
            "customer_id": CUSTOMER, "employee_id": EMPLOYEE, "device_id": DEVICE,
            "token_hash": TOKEN_HASH, "hostname": "old-host",
        },
        headers=_hdr(),
    )
    await client.post(
        f"/api/devices/{DEVICE}/revoke",
        json={"customer_id": CUSTOMER},
        headers=_hdr(),
    )
    new_hash = _hash("rotated")
    resp = await client.post(
        "/api/devices/register",
        json={
            "customer_id": CUSTOMER, "employee_id": EMPLOYEE, "device_id": DEVICE,
            "token_hash": new_hash, "hostname": "new-host",
        },
        headers=_hdr(),
    )
    assert resp.status_code == 200, resp.text

    verify = await client.post(
        "/api/devices/verify-token",
        json={"token_hash": new_hash},
        headers=_hdr(),
    )
    assert verify.status_code == 200, verify.text
    assert verify.json()["status"] == "active"
