"""Tests for FixtureStore path derivation, load, and record."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.synth.llm.base import Provider
from scripts.synth.llm.fixtures import FixtureStore


def test_path_for_derives_correct_path(tmp_path: Path) -> None:
    store = FixtureStore(root=tmp_path)
    p = store.path_for(Provider.ANTHROPIC, "abc123")
    assert p == tmp_path / "anthropic" / "abc123.json"


def test_load_returns_none_on_miss(tmp_path: Path) -> None:
    store = FixtureStore(root=tmp_path)
    assert store.load(Provider.ANTHROPIC, "nonexistent") is None


def test_record_creates_file(tmp_path: Path) -> None:
    store = FixtureStore(root=tmp_path)
    store.record(Provider.ANTHROPIC, "key1", {"text": "hello"})
    p = tmp_path / "anthropic" / "key1.json"
    assert p.exists()
    assert json.loads(p.read_text()) == {"text": "hello"}


def test_load_returns_recorded_data(tmp_path: Path) -> None:
    store = FixtureStore(root=tmp_path)
    store.record(Provider.GEMINI, "key2", {"label": "pos", "score": 0.9})
    result = store.load(Provider.GEMINI, "key2")
    assert result == {"label": "pos", "score": 0.9}
