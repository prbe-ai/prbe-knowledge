"""Disk-backed KV cache for repo signals + worldmodels.

The cache key is a string; values are arbitrary JSON-serializable dicts.
Cache hits return the same value byte-for-byte; misses return None."""

from __future__ import annotations

from pathlib import Path

from scripts.synth.cache import DiskCache


def test_get_returns_none_on_miss(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path)
    assert cache.get("nope") is None


def test_put_then_get_roundtrips(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path)
    value = {"a": 1, "b": [2, 3], "c": "ok"}
    cache.put("key1", value)
    assert cache.get("key1") == value


def test_keys_with_slashes_are_safe(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path)
    cache.put("github.com/x/y@abcd1234", {"sha": "abcd1234"})
    assert cache.get("github.com/x/y@abcd1234") == {"sha": "abcd1234"}


def test_invalidate_removes_entry(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path)
    cache.put("k", {"v": 1})
    cache.invalidate("k")
    assert cache.get("k") is None
