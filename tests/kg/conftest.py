"""Fixtures scoped to the kg test package.

The repo's top-level ``tests/conftest.py`` provides a ``live_db`` fixture
that initializes the module-level pool and truncates tables. Some kg tests
need a single asyncpg connection or two independent connections from the
pool (for concurrency tests of the advisory lock). Those fixtures live
here, scoped to ``tests/kg/`` so the additions don't impact existing tests.
"""

from __future__ import annotations

import socket
from collections.abc import AsyncIterator

import asyncpg
import pytest
import pytest_asyncio

from shared import db as db_module


def _postgres_reachable(host: str = "localhost", port: int = 5432) -> bool:
    """Probe whether a TCP listener is up at host:port.

    Used to skip live-DB tests when Postgres isn't running locally
    (CI without docker, or a fresh machine). Cheap; no DB handshake.
    """
    try:
        with socket.create_connection((host, port), timeout=0.25):
            return True
    except OSError:
        return False


# Module-level skip marker for live-DB fixtures so any test that requests
# them is skipped cleanly when Postgres isn't reachable. The advisory-lock
# unit tests that don't need a DB build their own minimal fakes and are
# unaffected.
_LIVE_DB_AVAILABLE = _postgres_reachable()
_LIVE_DB_SKIP_REASON = (
    "requires live Postgres at localhost:5432 "
    "(start with `docker compose up -d`)"
)


@pytest_asyncio.fixture
async def live_db_conn(live_db: None) -> AsyncIterator[asyncpg.Connection]:
    """Yield a single pooled asyncpg connection.

    Depends on the top-level ``live_db`` fixture for pool init + truncate.
    """
    async with db_module.get_pool().acquire() as conn:
        yield conn


@pytest_asyncio.fixture
async def pg_pool(live_db: None) -> AsyncIterator[asyncpg.Pool]:
    """Yield the module-level asyncpg pool itself.

    Concurrency tests need two independent connections; the pool is the
    canonical place to get them. Depends on ``live_db`` for init + cleanup.
    """
    yield db_module.get_pool()


def pytest_collection_modifyitems(config, items):  # type: ignore[no-untyped-def]
    """Auto-skip tests that depend on live-DB fixtures when Postgres is down.

    Looks at each test's fixturenames; if any name is in the live-DB set
    and Postgres isn't reachable, attach a skip marker. Lets the unit
    tests (no-transaction error, empty-customer-id error) keep running
    and passing without a DB.
    """
    if _LIVE_DB_AVAILABLE:
        return
    live_db_fixture_names = {"live_db", "live_db_conn", "pg_pool"}
    skip_marker = pytest.mark.skip(reason=_LIVE_DB_SKIP_REASON)
    for item in items:
        if live_db_fixture_names.intersection(getattr(item, "fixturenames", ())):
            item.add_marker(skip_marker)
