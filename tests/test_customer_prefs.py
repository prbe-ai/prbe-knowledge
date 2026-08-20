"""Fail-soft semantics for shared.customer_prefs readers.

These tests bypass the DB and patch `raw_conn` directly — the reader's
contract is "any unreadable or unexpected input → the declared
fallback", never a raise into the caller's hot path.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest

from engine.shared import customer_prefs


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


# ---- code_graph_indexed_branch ------------------------------------------


@pytest.mark.asyncio
async def test_branch_falls_back_to_default_when_no_overrides(monkeypatch) -> None:
    _patch_raw_conn(monkeypatch, {})
    branch = await customer_prefs.code_graph_indexed_branch(
        "c1", "acme/api", "main"
    )
    assert branch == "main"


@pytest.mark.asyncio
async def test_branch_returns_override_when_set(monkeypatch) -> None:
    _patch_raw_conn(
        monkeypatch,
        {"code_graph_branch_overrides": {"acme/api": "develop"}},
    )
    branch = await customer_prefs.code_graph_indexed_branch(
        "c1", "acme/api", "main"
    )
    assert branch == "develop"


@pytest.mark.asyncio
async def test_branch_falls_back_when_repo_not_in_overrides(monkeypatch) -> None:
    _patch_raw_conn(
        monkeypatch,
        {"code_graph_branch_overrides": {"acme/other": "release"}},
    )
    branch = await customer_prefs.code_graph_indexed_branch(
        "c1", "acme/api", "main"
    )
    assert branch == "main"


@pytest.mark.asyncio
async def test_branch_falls_back_for_blank_inputs(monkeypatch) -> None:
    """Reader short-circuits on empty customer or repo — neither makes
    sense as 'no filter' here, both should return the caller's default.
    """
    fetchval = _patch_raw_conn(
        monkeypatch,
        {"code_graph_branch_overrides": {"acme/api": "develop"}},
    )
    assert (
        await customer_prefs.code_graph_indexed_branch("", "acme/api", "main")
        == "main"
    )
    assert (
        await customer_prefs.code_graph_indexed_branch("c1", "", "main") == "main"
    )
    fetchval.assert_not_called()


@pytest.mark.asyncio
async def test_branch_falls_back_on_db_error(monkeypatch) -> None:
    @asynccontextmanager
    async def fake_raw_conn():
        raise RuntimeError("pool dead")
        yield  # pragma: no cover

    monkeypatch.setattr(customer_prefs, "raw_conn", fake_raw_conn)
    branch = await customer_prefs.code_graph_indexed_branch(
        "c1", "acme/api", "main"
    )
    assert branch == "main"


@pytest.mark.asyncio
async def test_branch_falls_back_on_malformed_overrides_shape(monkeypatch) -> None:
    """If overrides is anything other than dict[str, str], fall back."""
    _patch_raw_conn(
        monkeypatch,
        {"code_graph_branch_overrides": ["not", "a", "dict"]},
    )
    branch = await customer_prefs.code_graph_indexed_branch(
        "c1", "acme/api", "main"
    )
    assert branch == "main"


@pytest.mark.asyncio
async def test_branch_parses_string_jsonb(monkeypatch) -> None:
    _patch_raw_conn(
        monkeypatch,
        '{"code_graph_branch_overrides": {"acme/api": "develop"}}',
    )
    branch = await customer_prefs.code_graph_indexed_branch(
        "c1", "acme/api", "main"
    )
    assert branch == "develop"
