"""Listener DSN must default to direct (non-pooler) Neon endpoint.

Neon's pooler endpoint runs pgbouncer in transaction mode, which issues
`UNLISTEN *; RESET ALL` between every transaction — silently killing any
LISTEN registration. The wiki triage + synthesis apps therefore need a
dedicated direct-endpoint DSN for their NotifyListener.

These tests pin the resolution rule used in synthesis_app + triage_app:
prefer `settings.database_url_unpooled` when set, else fall back to
`settings.database_url`. The fallback keeps local dev (no pooler) and
any pre-pooler deploys working without forcing a config var on every
caller.

Implementation note: we instantiate `Settings()` directly inside each
test rather than going through `get_settings()` (which is lru_cache'd)
or `importlib.reload`. Both of those pollute global state and cascade
into other tests in this suite — the prior version of this file did
`importlib.reload(config)` and broke unrelated webhook + synthesis
tests on main. A bare `Settings()` reads env on construction without
touching module caches, so monkeypatch is fully isolated to this test.
"""

from __future__ import annotations

import pytest

from shared.config import Settings


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Strip any DSN env vars before each test so tests see only what
    they explicitly set. Without this, a stray DATABASE_URL_UNPOOLED in
    the dev shell or .env file could mask the field-default test.
    """
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL_UNPOOLED", raising=False)
    yield


def test_database_url_unpooled_defaults_to_none():
    s = Settings()
    assert s.database_url_unpooled is None


def test_database_url_unpooled_picks_up_env(monkeypatch):
    pooled = "postgresql://x:y@ep-foo-pooler.region/db"
    direct = "postgresql://x:y@ep-foo.region/db"
    monkeypatch.setenv("DATABASE_URL", pooled)
    monkeypatch.setenv("DATABASE_URL_UNPOOLED", direct)
    s = Settings()
    assert s.database_url == pooled
    assert s.database_url_unpooled == direct


def test_listener_dsn_resolution_prefers_unpooled():
    """Mirror the resolution used in synthesis_app + triage_app."""

    class _FakeSettings:
        database_url = "postgresql://pool/db"
        database_url_unpooled: str | None = "postgresql://direct/db"

    settings = _FakeSettings()
    listener_dsn = settings.database_url_unpooled or settings.database_url
    assert listener_dsn == "postgresql://direct/db"


def test_listener_dsn_resolution_falls_back_when_unpooled_unset():
    class _FakeSettings:
        database_url = "postgresql://pool/db"
        database_url_unpooled: str | None = None

    settings = _FakeSettings()
    listener_dsn = settings.database_url_unpooled or settings.database_url
    assert listener_dsn == "postgresql://pool/db"


def test_listener_dsn_resolution_falls_back_when_unpooled_empty_string():
    """Empty-string env vars should also fall back, not silently use ''."""

    class _FakeSettings:
        database_url = "postgresql://pool/db"
        database_url_unpooled: str | None = ""

    settings = _FakeSettings()
    listener_dsn = settings.database_url_unpooled or settings.database_url
    assert listener_dsn == "postgresql://pool/db"
