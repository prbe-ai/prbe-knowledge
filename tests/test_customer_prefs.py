"""Fail-closed semantics for shared.customer_prefs.is_wiki_generation_enabled.

These tests bypass the DB and patch `raw_conn` directly — the reader's
contract is "any non-True input → False" and that contract is what
keeps wiki synthesis off for tenants who haven't opted in. The DB-level
behavior (real customers row, real JSONB column) is exercised by
tests/test_normalizer_wiki_enqueue.py and tests/synthesis/test_wiki_cron.py.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest

from shared import customer_prefs


def _patch_raw_conn(monkeypatch, fetchval_return) -> MagicMock:
    """Patch shared.db.raw_conn so fetchval returns the given value."""
    fetchval = AsyncMock(return_value=fetchval_return)
    conn = MagicMock()
    conn.fetchval = fetchval

    @asynccontextmanager
    async def fake_raw_conn():
        yield conn

    monkeypatch.setattr(customer_prefs, "raw_conn", fake_raw_conn)
    return fetchval


@pytest.mark.asyncio
async def test_returns_true_when_key_explicitly_true(monkeypatch) -> None:
    _patch_raw_conn(monkeypatch, {"wiki_generation_enabled": True})
    assert await customer_prefs.is_wiki_generation_enabled("c1") is True


@pytest.mark.asyncio
async def test_returns_false_when_key_missing(monkeypatch) -> None:
    _patch_raw_conn(monkeypatch, {})
    assert await customer_prefs.is_wiki_generation_enabled("c1") is False


@pytest.mark.asyncio
async def test_returns_false_when_key_explicitly_false(monkeypatch) -> None:
    _patch_raw_conn(monkeypatch, {"wiki_generation_enabled": False})
    assert await customer_prefs.is_wiki_generation_enabled("c1") is False


@pytest.mark.asyncio
async def test_returns_false_for_truthy_non_bool_value(monkeypatch) -> None:
    """Strings like '1', 'true' must NOT be coerced to True — only the
    real bool counts. Otherwise a botched PATCH that stored the string
    'true' silently flips a tenant on."""
    _patch_raw_conn(monkeypatch, {"wiki_generation_enabled": "true"})
    assert await customer_prefs.is_wiki_generation_enabled("c1") is False


@pytest.mark.asyncio
async def test_returns_false_when_customer_missing(monkeypatch) -> None:
    _patch_raw_conn(monkeypatch, None)
    assert await customer_prefs.is_wiki_generation_enabled("nope") is False


@pytest.mark.asyncio
async def test_returns_false_for_blank_customer_id(monkeypatch) -> None:
    """Reader must short-circuit before opening a connection — guards
    against a caller that confused an empty tenant id with 'no filter'.
    """
    fetchval = _patch_raw_conn(monkeypatch, {"wiki_generation_enabled": True})
    assert await customer_prefs.is_wiki_generation_enabled("") is False
    fetchval.assert_not_called()


@pytest.mark.asyncio
async def test_returns_false_on_db_error(monkeypatch) -> None:
    @asynccontextmanager
    async def fake_raw_conn():
        raise RuntimeError("pool dead")
        yield  # pragma: no cover

    monkeypatch.setattr(customer_prefs, "raw_conn", fake_raw_conn)
    assert await customer_prefs.is_wiki_generation_enabled("c1") is False


@pytest.mark.asyncio
async def test_parses_string_jsonb(monkeypatch) -> None:
    """asyncpg returns JSONB as a str unless a codec is registered;
    the reader must handle both."""
    _patch_raw_conn(monkeypatch, '{"wiki_generation_enabled": true}')
    assert await customer_prefs.is_wiki_generation_enabled("c1") is True


@pytest.mark.asyncio
async def test_returns_false_on_malformed_json_string(monkeypatch) -> None:
    _patch_raw_conn(monkeypatch, "{not json")
    assert await customer_prefs.is_wiki_generation_enabled("c1") is False
