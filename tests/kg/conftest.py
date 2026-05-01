"""Fixtures scoped to the kg test package.

The repo's top-level ``tests/conftest.py`` provides a ``live_db`` fixture
that initializes the module-level pool and truncates tables. Some kg tests
need a single asyncpg connection or two independent connections from the
pool (for concurrency tests of the advisory lock). Those fixtures live
here, scoped to ``tests/kg/`` so the additions don't impact existing tests.

Shared API-test fixtures (``kg_app``, ``headers_for``, ``seeded_classes``,
plus the ``_seed_*`` / ``_cleanup`` helpers that back them) also live here
so both ``test_api_read.py`` and ``test_api_write.py`` consume the same
seeding / auth pattern. Keeping them in one place avoids drift between the
two routers' test suites.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import socket
from collections.abc import AsyncIterator

import asyncpg
import pytest
import pytest_asyncio
import structlog
from fastapi import FastAPI

from shared import db as db_module
from shared.config import Settings, get_settings
from shared.db import raw_conn

# Module-local internal key. Same pattern as tests/test_query_auth.py.
INTERNAL_KEY = "test-internal-knowledge-key"


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


@pytest.fixture
def _route_structlog_to_stdlib() -> None:
    """Route structlog output through stdlib logging so ``caplog`` sees it.

    The production ``configure_logging`` uses ``make_filtering_bound_logger``,
    which writes through ``structlog``'s own logger factory and bypasses
    stdlib logging entirely — meaning ``caplog`` would catch nothing. For
    tests, swap to ``structlog.stdlib.BoundLogger`` + ``LoggerFactory`` so
    each ``log.warning(...)`` becomes a stdlib ``LogRecord`` that ``caplog``
    can assert on.

    Non-autouse — only tests that assert on captured log output should
    request it. Other tests stay on the production structlog config so
    we don't perturb the global state of every kg test run. Lifted here
    from ``test_classifier_embedding.py`` / ``test_classifier_tiebreaker.py``
    once a third file (``test_traversal_expand.py``) needed the same
    behavior; consolidating avoids file-by-file copy drift.
    """
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False,
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
    live_db_fixture_names = {"live_db", "live_db_conn", "pg_pool", "seeded_classes"}
    skip_marker = pytest.mark.skip(reason=_LIVE_DB_SKIP_REASON)
    for item in items:
        if live_db_fixture_names.intersection(getattr(item, "fixturenames", ())):
            item.add_marker(skip_marker)


# ---------------------------------------------------------------------------
# Shared API-test fixtures — used by both test_api_read.py and test_api_write.py.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_settings(monkeypatch, settings: Settings) -> None:
    """Set the internal-key env var so the auth dep accepts our test header."""
    monkeypatch.setenv("ENVIRONMENT", "local")
    monkeypatch.setenv("INTERNAL_KNOWLEDGE_API_KEY", INTERNAL_KEY)
    get_settings.cache_clear()


@pytest.fixture
def kg_app() -> FastAPI:
    """Tiny FastAPI app with the KG read + write routers mounted at /kg.

    Built per-test so the test-suite never carries app state between cases.
    No lifespan: the live_db fixture handles pool init/close.
    """
    from services.kg.api.read import router as read_router
    from services.kg.api.write import router as write_router

    app = FastAPI()
    app.include_router(read_router, prefix="/kg")
    app.include_router(write_router, prefix="/kg")
    return app


@pytest.fixture
def headers_for():
    """Return a callable that builds auth headers for a given customer_id.

    Uses the X-Internal-Knowledge-Key + X-Prbe-Customer path (same as
    service-to-service callers). No DB roundtrip needed — convenient
    for tests that just want to assert RLS filtering.
    """

    def _make(customer_id: str) -> dict[str, str]:
        return {
            "X-Internal-Knowledge-Key": INTERNAL_KEY,
            "X-Prbe-Customer": customer_id,
        }

    return _make


async def _seed_customer(customer_id: str) -> None:
    api_key_hash = hashlib.sha256(secrets.token_urlsafe(32).encode()).hexdigest()
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO customers (customer_id, display_name, api_key_hash, status)
            VALUES ($1, $2, $3, 'active')
            ON CONFLICT (customer_id) DO NOTHING
            """,
            customer_id,
            f"{customer_id} display",
            api_key_hash,
        )


async def _seed_class(
    *,
    customer_id: str,
    class_id: str,
    description: str = "401 from upstream after JWT refresh",
    body: str = "## When this fires\n401 ...",
) -> None:
    """Insert a minimal valid kg_classes row.

    Frontmatter shape mirrors the Pydantic ``Frontmatter`` model so the
    GET-by-id handler can parse it back into a ``BugClass``.
    """
    frontmatter = {
        "id": class_id,
        "type": "bug-class",
        "description": description,
        "signature": {
            "must_match": ["status_code == 401"],
            "embedding_seed": "jwt refresh expired clock-skew",
        },
        "related": {
            "analogous_to": [],
            "overlaps_with": [],
            "often_confused_with": [],
            "regressed_by": [],
        },
        "context_sources": [],
        "evidence": {
            "match_count": 0,
            "last_updated": None,
            "recent_refinements": [],
        },
    }
    async with raw_conn() as conn:
        # Set the GUC inline so RLS lets the INSERT through (RLS USING is
        # checked on writes too because FORCE is enabled and there's no
        # bypass role here).
        await conn.execute(
            "SELECT set_config('app.current_customer_id', $1, true)",
            customer_id,
        )
        await conn.execute(
            """
            INSERT INTO kg_classes (customer_id, class_id, frontmatter, body)
            VALUES ($1, $2, $3::jsonb, $4)
            """,
            customer_id,
            class_id,
            json.dumps(frontmatter),
            body,
        )


async def _cleanup(customer_ids: list[str]) -> None:
    """Remove rows we inserted; live_db's TRUNCATE_SQL doesn't cover kg_*."""
    async with raw_conn() as conn:
        for cid in customer_ids:
            await conn.execute(
                "DELETE FROM kg_classes WHERE customer_id = $1", cid
            )
            await conn.execute(
                "DELETE FROM customers WHERE customer_id = $1", cid
            )


@pytest_asyncio.fixture
async def seeded_classes(live_db: None) -> AsyncIterator[None]:
    """Seed two tenants with known classes; clean up afterward.

    tA gets two classes (so list-view tests have >1 row); tB gets none
    (so the cross-tenant test has a clean negative case).
    """
    customers = ["tA", "tB"]
    try:
        for cid in customers:
            await _seed_customer(cid)
        await _seed_class(customer_id="tA", class_id="auth-401-jwt-refresh")
        await _seed_class(
            customer_id="tA",
            class_id="db-timeout-replica-lag",
            description="DB timeout on replica lag",
            body="## When this fires\nreplica lag exceeded ...",
        )
        yield None
    finally:
        await _cleanup(customers)
