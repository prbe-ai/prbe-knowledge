"""Tests for Plan 3 CLI flags: --mock-llm, --no-llm-cache, --record-llm, and provider routing."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.synth.cli import (
    CachingLlmClient,
    LlmClientConfig,
    build_llm_clients,
)
from scripts.synth.llm.anthropic_client import AnthropicClient
from scripts.synth.llm.base import Provider
from scripts.synth.llm.mock_client import MockLlmClient


def _make_llm_cfg(**kwargs) -> dict:
    # Mirrors scripts/synth/cli.py:_LLM_DEFAULTS. Keep in sync if the
    # production defaults shift (last bump: 2026-05-19 validator_model →
    # gemini-3.5-flash after the A/B sweep in eval_3_5_flash_sweep.py).
    defaults = {
        "planner_model": "claude-opus-4-7",
        "writer_model": "claude-sonnet-4-6",
        "validator_model": "gemini-3.5-flash",
    }
    defaults.update(kwargs)
    return defaults


def test_mock_llm_flag_produces_mock_client(tmp_path: Path) -> None:
    """--mock-llm → all 3 clients are MockLlmClient instances. No API keys needed."""
    cfg = LlmClientConfig(
        llm_cfg=_make_llm_cfg(),
        mock_llm=True,
        no_llm_cache=False,
        record_llm=False,
        fixture_root=tmp_path,
    )
    clients = build_llm_clients(cfg)
    assert isinstance(clients.planner_client, MockLlmClient)
    assert isinstance(clients.writer_client, MockLlmClient)
    assert isinstance(clients.validator_client, MockLlmClient)


def test_no_llm_cache_flag_bypasses_caching_wrapper(tmp_path: Path, monkeypatch) -> None:
    """--no-llm-cache → planner_client is the raw inner client, NOT a CachingLlmClient."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    cfg = LlmClientConfig(
        llm_cfg=_make_llm_cfg(),
        mock_llm=False,
        no_llm_cache=True,
        record_llm=False,
        cache_root=tmp_path,
    )
    with patch("scripts.synth.cli.AnthropicClient") as mock_ac:
        mock_ac.return_value = MagicMock(spec=AnthropicClient)
        clients = build_llm_clients(cfg)
    assert not isinstance(clients.planner_client, CachingLlmClient)


def test_default_wraps_in_caching_client(tmp_path: Path, monkeypatch) -> None:
    """Default (no flags) → planner_client IS wrapped in CachingLlmClient."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    cfg = LlmClientConfig(
        llm_cfg=_make_llm_cfg(),
        mock_llm=False,
        no_llm_cache=False,
        record_llm=False,
        cache_root=tmp_path,
    )
    with patch("scripts.synth.cli.AnthropicClient") as mock_ac:
        mock_ac.return_value = MagicMock(spec=AnthropicClient)
        clients = build_llm_clients(cfg)
    assert isinstance(clients.planner_client, CachingLlmClient)


def test_record_llm_flag_raises_if_api_key_absent(tmp_path: Path, monkeypatch) -> None:
    """--record-llm without env keys → RuntimeError (fail loudly, do not silently skip)."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    cfg = LlmClientConfig(
        llm_cfg=_make_llm_cfg(),
        mock_llm=False,
        no_llm_cache=False,
        record_llm=True,
        fixture_root=tmp_path,
    )
    with pytest.raises(RuntimeError, match=r"ANTHROPIC_API_KEY|GOOGLE_API_KEY"):
        build_llm_clients(cfg)


def test_gemini_model_prefix_records_gemini_provider(tmp_path: Path) -> None:
    """llm.planner_model='gemini-2.5-pro' → planner_provider == Provider.GEMINI."""
    cfg = LlmClientConfig(
        llm_cfg=_make_llm_cfg(planner_model="gemini-2.5-pro"),
        mock_llm=True,
        no_llm_cache=False,
        record_llm=False,
        fixture_root=tmp_path,
    )
    clients = build_llm_clients(cfg)
    assert isinstance(clients.planner_client, MockLlmClient)
    assert clients.planner_provider == Provider.GEMINI


def test_anthropic_model_prefix_records_anthropic_provider(tmp_path: Path) -> None:
    """llm.planner_model='claude-opus-4-7' → planner_provider == Provider.ANTHROPIC."""
    cfg = LlmClientConfig(
        llm_cfg=_make_llm_cfg(planner_model="claude-opus-4-7"),
        mock_llm=True,
        no_llm_cache=False,
        record_llm=False,
        fixture_root=tmp_path,
    )
    clients = build_llm_clients(cfg)
    assert clients.planner_provider == Provider.ANTHROPIC


def test_record_llm_flag_raises_for_gemini_missing_key(tmp_path: Path, monkeypatch) -> None:
    """--record-llm with all-Gemini config + missing GOOGLE_API_KEY → RuntimeError matching GOOGLE_API_KEY."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    cfg = LlmClientConfig(
        llm_cfg=_make_llm_cfg(
            planner_model="gemini-2.5-pro",
            writer_model="gemini-2.5-pro",
            validator_model="gemini-2.5-pro",
        ),
        mock_llm=False,
        no_llm_cache=False,
        record_llm=True,
        fixture_root=tmp_path,
    )
    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
        build_llm_clients(cfg)
