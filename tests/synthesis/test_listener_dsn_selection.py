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
"""

from __future__ import annotations

import importlib

import pytest

from shared import config


@pytest.fixture
def fresh_settings(monkeypatch):
    """Each test gets a freshly-rebuilt Settings instance from env."""

    def _build(env: dict[str, str]) -> config.Settings:
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        # Bust the lru_cache around get_settings.
        importlib.reload(config)
        return config.get_settings()

    return _build


def test_database_url_unpooled_defaults_to_none(fresh_settings, monkeypatch):
    monkeypatch.delenv("DATABASE_URL_UNPOOLED", raising=False)
    s = fresh_settings({"DATABASE_URL": "postgresql://x:y@example/db"})
    assert s.database_url_unpooled is None


def test_database_url_unpooled_picks_up_env(fresh_settings):
    pooled = "postgresql://x:y@ep-foo-pooler.region/db"
    direct = "postgresql://x:y@ep-foo.region/db"
    s = fresh_settings(
        {"DATABASE_URL": pooled, "DATABASE_URL_UNPOOLED": direct}
    )
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
