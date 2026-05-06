"""Tests for ``BootstrapOrchestrator``.

Covers the orchestrator's contract:
  - parallel crawler launch (asyncio.gather)
  - per-source failure isolation
  - wipe_first vs wipe_skipped DELETE semantics
  - wiki_synthesis_runs row open + close per crawler

Each test uses a tiny mock crawler subclass that subclasses
``BootstrapAgent`` and sleeps for a configurable duration to simulate
work. Real crawlers (Lane D's GitHub) plug into the same lifecycle.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
import pytest_asyncio

from services.synthesis.bootstrap_orchestrator import (
    BootstrapOrchestrator,
    BootstrapResult,
)
from services.synthesis.crawlers.base import (
    BootstrapAgent,
    BootstrapAgentResult,
)
from shared.config import Settings, get_settings
from shared.db import raw_conn

# ---------------------------------------------------------------------------
# Mock crawler — base for the test fixtures
# ---------------------------------------------------------------------------


class _MockCrawler(BootstrapAgent):
    """Tiny BootstrapAgent that sleeps then returns a fake result.

    Tests configure ``sleep_seconds`` so they can assert parallel
    execution (two crawlers running at the same time finish in
    ~max(durations), not sum(durations)).
    """

    sleep_seconds: float = 0.0
    pages_created_value: int = 0
    pages_updated_value: int = 0
    items_processed_value: int = 0
    raise_on_run: BaseException | None = None

    def system_prompt(self) -> str:
        return "mock"

    def source_api_tools(self) -> list[dict[str, Any]]:
        return []

    async def dispatch_source_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        return {}

    async def run(self) -> BootstrapAgentResult:
        started = datetime.now(UTC)
        if self.sleep_seconds:
            await asyncio.sleep(self.sleep_seconds)
        if self.raise_on_run is not None:
            raise self.raise_on_run
        return BootstrapAgentResult(
            source=self.source,
            customer_id=self.customer_id,
            run_id=self.run_id,
            pages_created=self.pages_created_value,
            pages_updated=self.pages_updated_value,
            items_processed=self.items_processed_value,
            started_at=started,
            finished_at=datetime.now(UTC),
        )


def _make_factory(
    *,
    source: str,
    sleep_seconds: float = 0.0,
    pages_created: int = 0,
    pages_updated: int = 0,
    raise_on_run: BaseException | None = None,
):
    """Build a factory closure that produces a fresh _MockCrawler with
    the requested per-test config knobs."""

    def factory(**kwargs: Any) -> BootstrapAgent:
        cls = type(
            f"_Mock_{source}",
            (_MockCrawler,),
            {
                "source": source,
                "sleep_seconds": sleep_seconds,
                "pages_created_value": pages_created,
                "pages_updated_value": pages_updated,
                "raise_on_run": raise_on_run,
            },
        )
        return cls(**kwargs)

    return factory


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


CUSTOMER = "bootstrap-test-cust"


@pytest_asyncio.fixture
async def seeded_customer(live_db: None) -> AsyncIterator[None]:
    """Insert the test customer + grab a fresh settings instance."""
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers(customer_id, display_name, api_key_hash) "
            "VALUES ($1, 'bootstrap-test', 'h') ON CONFLICT DO NOTHING",
            CUSTOMER,
        )
    yield None


@pytest_asyncio.fixture
async def http_client() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient() as client:
        yield client


def _settings() -> Settings:
    get_settings.cache_clear()  # type: ignore[attr-defined]
    return Settings()


# ---------------------------------------------------------------------------
# Parallel launch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_orchestrator_launches_crawlers_in_parallel(
    seeded_customer: None, http_client: httpx.AsyncClient
) -> None:
    """Two crawlers, each sleeping 0.5s. Wall-clock < 0.9s proves parallel
    execution (sequential would take >=1.0s)."""
    factories = {
        "alpha": _make_factory(source="alpha", sleep_seconds=0.5),
        "beta": _make_factory(source="beta", sleep_seconds=0.5),
    }
    orch = BootstrapOrchestrator(settings=_settings(), http=http_client)
    started = time.monotonic()
    result = await orch.bootstrap(
        customer_id=CUSTOMER,
        sources=["alpha", "beta"],
        wipe_first=False,
        crawler_factories=factories,
    )
    elapsed = time.monotonic() - started
    assert elapsed < 0.9, f"crawlers ran sequentially? elapsed={elapsed}"
    assert sorted(result.sources_succeeded) == ["alpha", "beta"]
    assert result.sources_failed == {}
    assert len(result.per_source) == 2


# ---------------------------------------------------------------------------
# Per-source failure isolation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_orchestrator_per_source_failure_isolation(
    seeded_customer: None, http_client: httpx.AsyncClient
) -> None:
    """One crawler raises, the other succeeds. Result splits the two
    correctly: succeeded list has the survivor; failed dict has the
    crasher's error."""
    factories = {
        "good": _make_factory(source="good", sleep_seconds=0.05),
        "bad": _make_factory(source="bad", raise_on_run=RuntimeError("boom from bad")),
    }
    orch = BootstrapOrchestrator(settings=_settings(), http=http_client)
    result = await orch.bootstrap(
        customer_id=CUSTOMER,
        sources=["good", "bad"],
        wipe_first=False,
        crawler_factories=factories,
    )
    assert result.sources_succeeded == ["good"]
    assert "bad" in result.sources_failed
    assert "boom from bad" in result.sources_failed["bad"]
    assert "RuntimeError" in result.sources_failed["bad"]
    # Both per-source records are populated even though one crashed.
    by_source = {r.source: r for r in result.per_source}
    assert by_source["good"].error is None
    assert by_source["bad"].error is not None


# ---------------------------------------------------------------------------
# wipe_first
# ---------------------------------------------------------------------------


async def _seed_wiki_rows() -> None:
    """Seed one wiki_links row, one wiki_timeline_entries row, and one
    documents row of doc_class='compiled_wiki' for the test customer."""
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO wiki_links
                (customer_id, src_wiki_type, src_slug,
                 dst_wiki_type, dst_slug, link_type, link_source)
            VALUES ($1, 'service_card', 'auth', 'person', 'maison',
                    'works_at', 'markdown')
            """,
            CUSTOMER,
        )
        await conn.execute(
            """
            INSERT INTO wiki_timeline_entries
                (customer_id, wiki_type, slug, entry_date, source, summary)
            VALUES ($1, 'decision', 'rollback-policy', '2026-05-04',
                    'github', 'PR merged')
            """,
            CUSTOMER,
        )
        await conn.execute(
            """
            INSERT INTO wiki_raw_data
                (customer_id, wiki_type, slug, source, source_ref, data)
            VALUES ($1, 'decision', 'rollback-policy', 'github', 'pr:1',
                    '{"id":1}'::jsonb)
            """,
            CUSTOMER,
        )
        # documents row: an actual compiled_wiki page. Mirrors the
        # NOT-NULL columns from db/schema.sql:77 — source_url, acl,
        # created_at/updated_at/valid_from must all be provided.
        await conn.execute(
            """
            INSERT INTO documents
                (doc_id, customer_id, version, source_system, source_id,
                 source_url, doc_type, doc_class, title, body_preview,
                 body_token_count, content_hash, acl,
                 created_at, updated_at, valid_from)
            VALUES ($1, $2, 1, 'wiki', 'service_card:auth',
                    '/wiki/service_card/auth', 'wiki.service_card',
                    'compiled_wiki', 'Auth', 'Auth service', 0, 'hash',
                    '{"viewers":[]}'::jsonb,
                    NOW(), NOW(), NOW())
            """,
            f"wiki:service_card:auth-{CUSTOMER}",
            CUSTOMER,
        )


async def _row_counts() -> dict[str, int]:
    async with raw_conn() as conn:
        wl = await conn.fetchval("SELECT COUNT(*) FROM wiki_links WHERE customer_id = $1", CUSTOMER)
        wte = await conn.fetchval(
            "SELECT COUNT(*) FROM wiki_timeline_entries WHERE customer_id = $1",
            CUSTOMER,
        )
        wrd = await conn.fetchval(
            "SELECT COUNT(*) FROM wiki_raw_data WHERE customer_id = $1",
            CUSTOMER,
        )
        docs = await conn.fetchval(
            """
            SELECT COUNT(*) FROM documents
            WHERE customer_id = $1 AND doc_class = 'compiled_wiki'
            """,
            CUSTOMER,
        )
    return {
        "wiki_links": int(wl or 0),
        "wiki_timeline_entries": int(wte or 0),
        "wiki_raw_data": int(wrd or 0),
        "documents": int(docs or 0),
    }


@pytest.mark.asyncio
async def test_orchestrator_wipe_first(
    seeded_customer: None, http_client: httpx.AsyncClient
) -> None:
    await _seed_wiki_rows()
    pre = await _row_counts()
    assert all(v == 1 for v in pre.values()), pre

    factories = {"alpha": _make_factory(source="alpha")}
    orch = BootstrapOrchestrator(settings=_settings(), http=http_client)
    result = await orch.bootstrap(
        customer_id=CUSTOMER,
        sources=["alpha"],
        wipe_first=True,
        crawler_factories=factories,
    )
    assert result.wiped is True
    post = await _row_counts()
    assert post == {
        "wiki_links": 0,
        "wiki_timeline_entries": 0,
        "wiki_raw_data": 0,
        "documents": 0,
    }


@pytest.mark.asyncio
async def test_orchestrator_wipe_skipped(
    seeded_customer: None, http_client: httpx.AsyncClient
) -> None:
    await _seed_wiki_rows()
    factories = {"alpha": _make_factory(source="alpha")}
    orch = BootstrapOrchestrator(settings=_settings(), http=http_client)
    result = await orch.bootstrap(
        customer_id=CUSTOMER,
        sources=["alpha"],
        wipe_first=False,
        crawler_factories=factories,
    )
    assert result.wiped is False
    post = await _row_counts()
    # Untouched.
    assert post == {
        "wiki_links": 1,
        "wiki_timeline_entries": 1,
        "wiki_raw_data": 1,
        "documents": 1,
    }


# ---------------------------------------------------------------------------
# wiki_synthesis_runs row creation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_orchestrator_creates_run_rows(
    seeded_customer: None, http_client: httpx.AsyncClient
) -> None:
    """Each crawler gets its own row at kind='bootstrap' with source set
    + status flips correctly based on outcome."""
    factories = {
        "winner": _make_factory(source="winner", pages_created=2, pages_updated=3),
        "loser": _make_factory(source="loser", raise_on_run=ValueError("kaboom")),
    }
    orch = BootstrapOrchestrator(settings=_settings(), http=http_client)
    result = await orch.bootstrap(
        customer_id=CUSTOMER,
        sources=["winner", "loser"],
        wipe_first=False,
        crawler_factories=factories,
    )
    assert "winner" in result.sources_succeeded
    assert "loser" in result.sources_failed

    async with raw_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT source, kind, stage, status, pages_updated, pages_created,
                   error
            FROM wiki_synthesis_runs
            WHERE customer_id = $1
            ORDER BY source
            """,
            CUSTOMER,
        )
    by_source = {row["source"]: row for row in rows}
    assert set(by_source.keys()) == {"loser", "winner"}
    assert by_source["winner"]["kind"] == "bootstrap"
    assert by_source["winner"]["stage"] == "synthesis"
    assert by_source["winner"]["status"] == "complete"
    assert by_source["winner"]["pages_updated"] == 3
    assert by_source["winner"]["pages_created"] == 2
    assert by_source["winner"]["error"] is None
    assert by_source["loser"]["status"] == "failed"
    assert by_source["loser"]["error"] is not None
    assert "kaboom" in (by_source["loser"]["error"] or "")


@pytest.mark.asyncio
async def test_orchestrator_unknown_source_dropped_silently(
    seeded_customer: None, http_client: httpx.AsyncClient
) -> None:
    """Sources passed but not in the factory map are dropped with a
    warning; they don't crash the run, they don't show up in
    sources_attempted."""
    factories = {"alpha": _make_factory(source="alpha")}
    orch = BootstrapOrchestrator(settings=_settings(), http=http_client)
    result: BootstrapResult = await orch.bootstrap(
        customer_id=CUSTOMER,
        sources=["alpha", "ghost"],
        wipe_first=False,
        crawler_factories=factories,
    )
    assert result.sources_attempted == ["alpha"]
    assert result.sources_succeeded == ["alpha"]


@pytest.mark.asyncio
async def test_orchestrator_no_sources_is_noop(
    seeded_customer: None, http_client: httpx.AsyncClient
) -> None:
    """Empty factory map = empty result. wipe still runs if requested."""
    await _seed_wiki_rows()
    orch = BootstrapOrchestrator(settings=_settings(), http=http_client)
    result = await orch.bootstrap(
        customer_id=CUSTOMER,
        sources=[],
        wipe_first=True,
        crawler_factories={},
    )
    assert result.sources_attempted == []
    assert result.sources_succeeded == []
    assert result.sources_failed == {}
    assert result.wiped is True
    post = await _row_counts()
    assert post["wiki_links"] == 0
