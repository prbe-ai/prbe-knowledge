"""Shared fixtures for the synthesis suite.

WHY THIS EXISTS: `get_settings` is an lru_cache. Several tests here set
`GOOGLE_API_KEY` / `LLM_GATEWAY_URL` via monkeypatch and then call
`get_settings.cache_clear()` so the new env is picked up -- but nothing clears
it on the way out. monkeypatch rolls the environment variable back; it cannot
roll back the `Settings` object already cached against it. So the cache leaks a
key-present Settings into whatever runs next.

That was invisible while `test_index_renderer.py` was the last argument in the
pytest invocation and while nothing in CI ran it at all. Both changed: it is now
gated in `.github/workflows/tests.yml`, and the next file appended after it
would run against a Settings reporting a Google key that its own environment
does not have. A no-key-fallback assertion there would then pass or fail purely
on argument order, which is the kind of failure that gets diagnosed as flake.

Clearing on both sides of every test removes the ordering dependency.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from engine.shared.config import get_settings


@pytest.fixture(autouse=True)
def _reset_settings_cache() -> Iterator[None]:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
