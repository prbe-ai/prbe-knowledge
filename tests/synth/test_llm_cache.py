"""Tests for PromptCache key derivation and round-trip storage."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.synth.llm.base import LlmRequest, Provider
from scripts.synth.llm.cache import PromptCache, cache_key


def _req(**kwargs) -> LlmRequest:
    defaults = dict(model="claude-opus-4-7", system="sys", prompt="hello", max_tokens=2048, temperature=0.0)
    defaults.update(kwargs)
    return LlmRequest(**defaults)


def test_cache_key_is_hex_string() -> None:
    key = cache_key(Provider.ANTHROPIC, _req())
    assert isinstance(key, str)
    assert len(key) == 64  # sha256 hex digest


def test_cache_key_same_for_same_input() -> None:
    req = _req()
    assert cache_key(Provider.ANTHROPIC, req) == cache_key(Provider.ANTHROPIC, req)


def test_cache_key_changes_on_provider() -> None:
    req = _req()
    assert cache_key(Provider.ANTHROPIC, req) != cache_key(Provider.GEMINI, req)


def test_cache_key_changes_on_prompt() -> None:
    assert cache_key(Provider.ANTHROPIC, _req(prompt="a")) != cache_key(Provider.ANTHROPIC, _req(prompt="b"))


def test_cache_key_changes_on_temperature() -> None:
    assert cache_key(Provider.ANTHROPIC, _req(temperature=0.0)) != cache_key(Provider.ANTHROPIC, _req(temperature=0.5))


def test_cache_key_changes_on_schema() -> None:
    req = _req()
    assert cache_key(Provider.ANTHROPIC, req, schema_json='{"type":"object"}') != cache_key(Provider.ANTHROPIC, req, schema_json=None)


@pytest.mark.asyncio
async def test_prompt_cache_round_trip(tmp_path: Path) -> None:
    cache = PromptCache(root=tmp_path)
    req = _req()
    await cache.put(Provider.ANTHROPIC, req, None, {"text": "hello"})
    result = await cache.get(Provider.ANTHROPIC, req, None)
    assert result == {"text": "hello"}


@pytest.mark.asyncio
async def test_prompt_cache_miss_returns_none(tmp_path: Path) -> None:
    cache = PromptCache(root=tmp_path)
    req = _req(prompt="never stored")
    result = await cache.get(Provider.ANTHROPIC, req, None)
    assert result is None
