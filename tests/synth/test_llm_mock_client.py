"""Tests for MockLlmClient replay and record modes."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from scripts.synth.llm.base import LlmRequest, LlmResponse, Provider
from scripts.synth.llm.cache import cache_key
from scripts.synth.llm.fixtures import FixtureStore
from scripts.synth.llm.mock_client import FixtureNotFoundError, MockLlmClient


class _Schema(BaseModel):
    answer: str


def _make_req() -> LlmRequest:
    return LlmRequest(model="claude-opus-4-7", system="sys", prompt="classify this")


@pytest.mark.asyncio
async def test_replay_returns_canned_response(tmp_path: Path) -> None:
    store = FixtureStore(root=tmp_path)
    req = _make_req()
    key = cache_key(Provider.ANTHROPIC, req)
    store.record(Provider.ANTHROPIC, key, {"text": "canned text"})
    client = MockLlmClient(store=store, mode="replay")
    resp = await client.generate(req)
    assert isinstance(resp, LlmResponse)
    assert resp.text == "canned text"


@pytest.mark.asyncio
async def test_replay_raises_fixture_not_found(tmp_path: Path) -> None:
    store = FixtureStore(root=tmp_path)
    client = MockLlmClient(store=store, mode="replay")
    req = _make_req()
    with pytest.raises(FixtureNotFoundError, match="--record-llm"):
        await client.generate(req)


@pytest.mark.asyncio
async def test_record_mode_forwards_and_writes(tmp_path: Path) -> None:
    store = FixtureStore(root=tmp_path)
    real = MagicMock()
    real.generate = AsyncMock(return_value=LlmResponse(text="real response"))
    client = MockLlmClient(store=store, mode="record", real_client=real)
    req = _make_req()
    resp = await client.generate(req)
    assert resp.text == "real response"
    # fixture should now be written
    key = cache_key(Provider.ANTHROPIC, req)
    loaded = store.load(Provider.ANTHROPIC, key)
    assert loaded == {"text": "real response"}


@pytest.mark.asyncio
async def test_record_mode_requires_real_client(tmp_path: Path) -> None:
    store = FixtureStore(root=tmp_path)
    with pytest.raises(ValueError, match="real_client"):
        MockLlmClient(store=store, mode="record", real_client=None)


@pytest.mark.asyncio
async def test_close_delegates_to_real_client_if_present(tmp_path: Path) -> None:
    store = FixtureStore(root=tmp_path)
    real = MagicMock()
    real.generate = AsyncMock(return_value=LlmResponse(text="x"))
    real.close = AsyncMock()
    client = MockLlmClient(store=store, mode="record", real_client=real)
    await client.close()
    real.close.assert_called_once()
