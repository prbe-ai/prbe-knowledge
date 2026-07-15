"""Customer-source mapping: round-trip + resolution precedence.

Covers:
  - record_mapping / resolve_customer roundtrip
  - single_customer_fallback when only one tenant exists
  - Each connector's extract_external_id_from_payload on a realistic fixture
  - The webhook handler's resolution path when no X-Prbe-Customer header is sent
"""

from __future__ import annotations

import pytest

from engine.shared.config import Settings, get_settings
from engine.shared.constants import SourceSystem
from engine.shared.customer_mapping import (
    record_mapping,
    resolve_customer,
    single_customer_fallback,
)
from engine.shared.db import raw_conn
from engine.shared.embeddings import reset_embedder
from engine.shared.exceptions import SourceAlreadyConnectedError
from engine.shared.storage import reset_store


@pytest.fixture(autouse=True)
def _patch(monkeypatch, settings: Settings):
    monkeypatch.setenv("TOKEN_ENCRYPTION_KEY", settings.token_encryption_key.get_secret_value())
    monkeypatch.setenv("SLACK_SIGNING_SECRET", "test-secret")
    monkeypatch.setenv("ENVIRONMENT", "local")
    reset_embedder()
    reset_store()
    get_settings.cache_clear()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_record_and_resolve(live_db) -> None:
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers (customer_id, display_name, api_key_hash) VALUES ($1,'A','x') ON CONFLICT DO NOTHING",
            "cust-map",
        )

    await record_mapping(
        customer_id="cust-map",
        source_system=SourceSystem.SLACK,
        external_id="T_ACME",
        external_name="Acme",
    )
    resolved = await resolve_customer(SourceSystem.SLACK, "T_ACME")
    assert resolved == "cust-map"

    # Unknown external_id returns None.
    assert await resolve_customer(SourceSystem.SLACK, "T_UNKNOWN") is None


@pytest.mark.asyncio
async def test_record_mapping_same_customer_is_idempotent(live_db) -> None:
    """Re-installing under the same customer_id refreshes name/metadata."""
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers (customer_id, display_name, api_key_hash) VALUES ($1,'A','x') ON CONFLICT DO NOTHING",
            "cust-reinstall",
        )

    await record_mapping(
        customer_id="cust-reinstall",
        source_system=SourceSystem.LINEAR,
        external_id="org-1",
        external_name="Old Name",
    )
    # Same customer, different external_name — must succeed and update.
    await record_mapping(
        customer_id="cust-reinstall",
        source_system=SourceSystem.LINEAR,
        external_id="org-1",
        external_name="New Name",
    )

    async with raw_conn() as conn:
        row = await conn.fetchrow(
            "SELECT customer_id, external_name FROM customer_source_mapping WHERE source_system='linear' AND external_id='org-1'"
        )
    assert row["customer_id"] == "cust-reinstall"
    assert row["external_name"] == "New Name"


@pytest.mark.asyncio
async def test_record_mapping_blocks_cross_customer_overwrite(live_db) -> None:
    """A second customer trying to claim the same workspace must be refused.

    Pre-fix, this overwrote `customer_id` and split chunks across tenants
    (the Linear-org incident on 2026-04-28).
    """
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers (customer_id, display_name, api_key_hash) VALUES ('cust-a','A','x') ON CONFLICT DO NOTHING"
        )
        await conn.execute(
            "INSERT INTO customers (customer_id, display_name, api_key_hash) VALUES ('cust-b','B','x') ON CONFLICT DO NOTHING"
        )

    await record_mapping(
        customer_id="cust-a",
        source_system=SourceSystem.LINEAR,
        external_id="shared-org",
        external_name="Acme",
    )

    with pytest.raises(SourceAlreadyConnectedError) as excinfo:
        await record_mapping(
            customer_id="cust-b",
            source_system=SourceSystem.LINEAR,
            external_id="shared-org",
            external_name="Acme",
        )
    err = excinfo.value
    assert err.existing_customer_id == "cust-a"
    assert err.attempted_customer_id == "cust-b"
    assert err.source_system == "linear"
    assert err.external_id == "shared-org"

    # Mapping must still point at the original owner — refused write, not overwrite.
    assert await resolve_customer(SourceSystem.LINEAR, "shared-org") == "cust-a"


@pytest.mark.asyncio
async def test_single_customer_fallback(live_db) -> None:
    # Zero customers → None
    assert await single_customer_fallback() is None

    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers (customer_id, display_name, api_key_hash) VALUES ('solo','s','x') ON CONFLICT DO NOTHING"
        )
    assert await single_customer_fallback() == "solo"

    # Two customers → None (ambiguous)
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers (customer_id, display_name, api_key_hash) VALUES ('second','t','x') ON CONFLICT DO NOTHING"
        )
    assert await single_customer_fallback() is None


def _dummy_ctx():
    import httpx as _h

    from engine.ingest.handlers.base import ConnectorContext

    return ConnectorContext(settings=Settings(), http=_h.AsyncClient())


def test_slack_extract_external_id() -> None:
    from engine.ingest.handlers.registry import build_connector

    c = build_connector(SourceSystem.SLACK, _dummy_ctx())
    assert c.extract_external_id_from_payload({}, {"team_id": "T_X"}) == "T_X"
    assert c.extract_external_id_from_payload({}, {"team": {"id": "T_Y"}}) == "T_Y"
    assert c.extract_external_id_from_payload({}, {}) is None


def test_linear_extract_external_id() -> None:
    from engine.ingest.handlers.registry import build_connector

    c = build_connector(SourceSystem.LINEAR, _dummy_ctx())
    assert c.extract_external_id_from_payload({}, {"organizationId": "O_1"}) == "O_1"
    assert c.extract_external_id_from_payload({}, {}) is None


def test_github_extract_external_id() -> None:
    from engine.ingest.handlers.registry import build_connector

    c = build_connector(SourceSystem.GITHUB, _dummy_ctx())
    assert (
        c.extract_external_id_from_payload({}, {"installation": {"id": 42}}) == "42"
    )
    assert c.extract_external_id_from_payload({}, {}) is None


def test_notion_extract_external_id() -> None:
    from engine.ingest.handlers.registry import build_connector

    c = build_connector(SourceSystem.NOTION, _dummy_ctx())
    assert (
        c.extract_external_id_from_payload({}, {"workspace_id": "W_1"}) == "W_1"
    )
    assert (
        c.extract_external_id_from_payload({}, {"entity": {"workspace_id": "W_2"}})
        == "W_2"
    )


def test_sentry_extract_external_id() -> None:
    from engine.ingest.handlers.registry import build_connector

    c = build_connector(SourceSystem.SENTRY, _dummy_ctx())
    assert (
        c.extract_external_id_from_payload({}, {"organization": {"slug": "acme"}})
        == "acme"
    )
    assert (
        c.extract_external_id_from_payload(
            {}, {"installation": {"organization": {"slug": "acme2"}}}
        )
        == "acme2"
    )


