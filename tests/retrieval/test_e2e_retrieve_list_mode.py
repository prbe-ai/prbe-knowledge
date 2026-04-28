"""End-to-end /retrieve test for the deterministic list path.

Reproduces the original bug ("3 most recent github commits returns PRs
mixed with commits and the actually-newest commits are missing") and
verifies the fix: with mode=list dispatched, the result is exactly the
N newest commits, sorted by date, no PR leakage.

Mocks Haiku at the AsyncAnthropic boundary so no network call is made;
the SQL path itself runs against the real test DB.
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from httpx import ASGITransport

from shared.config import Settings, get_settings
from shared.db import close_pool, raw_conn
from shared.embeddings import reset_embedder
from shared.storage import reset_store

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _patch_settings(monkeypatch, settings: Settings) -> None:
    monkeypatch.setenv("TOKEN_ENCRYPTION_KEY", settings.token_encryption_key.get_secret_value())
    monkeypatch.setenv("ENVIRONMENT", "local")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    reset_embedder()
    reset_store()
    get_settings.cache_clear()  # type: ignore[attr-defined]


def _tool_use_resp(payload: dict) -> SimpleNamespace:
    return SimpleNamespace(
        content=[SimpleNamespace(type="tool_use", name="route_query", input=payload)]
    )


async def _seed_customer(customer_id: str) -> str:
    api_key = secrets.token_urlsafe(32)
    h = hashlib.sha256(api_key.encode()).hexdigest()
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO customers (customer_id, display_name, api_key_hash)
            VALUES ($1, 'test', $2)
            ON CONFLICT (customer_id) DO UPDATE SET api_key_hash = EXCLUDED.api_key_hash
            """,
            customer_id,
            h,
        )
    return api_key


async def _seed_doc(
    customer_id: str,
    doc_id: str,
    *,
    source_system: str,
    doc_type: str,
    title: str,
    updated_at: datetime,
) -> None:
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO documents (
                doc_id, version, customer_id,
                source_system, source_id, source_url,
                doc_class, doc_type, content_type,
                content_hash, title, body_size_bytes, body_token_count,
                created_at, updated_at, valid_from, ingested_at,
                acl
            ) VALUES (
                $1, 1, $2,
                $3, $4, $5,
                'raw_source', $6, 'text/plain',
                $7, $8, 10, 0,
                $9, $9, $9, $9,
                '{}'::jsonb
            )
            """,
            doc_id,
            customer_id,
            source_system,
            doc_id + "-src",
            f"https://example/{doc_id}",
            doc_type,
            f"hash-{doc_id}",
            title,
            updated_at,
        )
        await conn.execute(
            """
            INSERT INTO chunks (
                chunk_id, doc_id, customer_id,
                chunk_index, content, content_hash, token_count,
                embedding, first_seen_version, last_seen_version
            ) VALUES (
                $1, $2, $3, 0, $4, $5, 5,
                array_fill(0::real, ARRAY[3072])::halfvec,
                1, 1
            )
            ON CONFLICT (doc_id, content_hash) DO NOTHING
            """,
            f"{doc_id}:c0:v1",
            doc_id,
            customer_id,
            title + " body",
            f"chash-{doc_id}",
        )


async def _post(body: dict, headers: dict) -> httpx.Response:
    from services.retrieval.main import app

    await close_pool()
    async with (
        httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c,
        app.router.lifespan_context(app),
    ):
        return await c.post("/retrieve", json=body, headers=headers)


async def test_three_most_recent_commits_returns_only_commits_sorted(live_db) -> None:
    """The bug we're fixing: '3 most recent github commits' must return
    the 3 newest commits in date order, not PRs."""
    api_key = await _seed_customer("cust-1")
    base = datetime(2026, 4, 28, tzinfo=UTC)

    # 5 commits, descending timestamps.
    for i in range(5):
        await _seed_doc(
            "cust-1",
            f"github:repo:commit:{i}",
            source_system="github",
            doc_type="github.commit",
            title=f"commit {i}",
            updated_at=base - timedelta(minutes=i),
        )
    # 2 PRs, slightly newer than the newest commit. These would beat
    # the commits on a sort-by-date over the unfiltered table — they
    # MUST NOT show up in the result.
    await _seed_doc(
        "cust-1",
        "github:repo:pr:1",
        source_system="github",
        doc_type="github.pull_request",
        title="PR-1 must not appear",
        updated_at=base + timedelta(minutes=1),
    )
    await _seed_doc(
        "cust-1",
        "github:repo:pr:2",
        source_system="github",
        doc_type="github.pull_request",
        title="PR-2 must not appear",
        updated_at=base + timedelta(minutes=2),
    )

    haiku_payload = {
        "entities": [
            {
                "entity_type": "repo",
                "canonical_id": "github",
                "display_name": "GitHub",
                "confidence": 0.9,
            }
        ],
        "expansions": [],
        "temporal": None,
        "sort": {"field": "updated_at", "direction": "desc", "trigger_phrase": "most recent"},
        "mode": "list",
        "doc_type": "commit",
        "operation": "list",
        "group_by_key": None,
    }

    with patch("services.retrieval.router.AsyncAnthropic") as mock_client_cls:
        mock_client_cls.return_value.messages.create = AsyncMock(
            return_value=_tool_use_resp(haiku_payload)
        )
        resp = await _post(
            {"query": "3 most recent github commits", "top_k": 3},
            {"Authorization": f"Bearer {api_key}"},
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["applied_mode"] == "list"
    assert body["applied_doc_types"] == ["github.commit"]
    chunks = body["chunks"]
    assert len(chunks) == 3
    # Newest first, all commits, no PR leakage.
    titles = [c["title"] for c in chunks]
    assert titles == ["commit 0", "commit 1", "commit 2"]
    for c in chunks:
        assert "PR" not in (c["title"] or "")


async def test_haiku_misclassifies_topic_query_as_list_dispatcher_falls_back(
    live_db,
) -> None:
    """Defense-in-depth: if Haiku emits mode=list but the entities include
    a topic type (feature/decision/error_group), the dispatcher's local
    gate-recheck must fall back to search."""
    api_key = await _seed_customer("cust-2")
    base = datetime(2026, 4, 28, tzinfo=UTC)
    # Seed a single doc so search returns SOMETHING (no embedding model
    # actually runs — we mocked Anthropic, vector path will use the dummy
    # zero embedding which still returns docs at distance 1.0).
    await _seed_doc(
        "cust-2",
        "doc:1",
        source_system="github",
        doc_type="github.commit",
        title="some commit",
        updated_at=base,
    )

    haiku_payload = {
        "entities": [
            {
                "entity_type": "feature",  # TOPIC — should force search
                "canonical_id": "auth",
                "display_name": "auth",
                "confidence": 0.85,
            }
        ],
        "expansions": [],
        "temporal": None,
        "sort": {"field": "updated_at", "direction": "desc", "trigger_phrase": "most recent"},
        "mode": "list",  # Haiku said list (incorrectly!) — gate must catch this
        "doc_type": "commit",
        "operation": "list",
        "group_by_key": None,
    }

    with patch("services.retrieval.router.AsyncAnthropic") as mock_client_cls:
        mock_client_cls.return_value.messages.create = AsyncMock(
            return_value=_tool_use_resp(haiku_payload)
        )
        resp = await _post(
            {"query": "most recent commits about auth", "top_k": 3},
            {"Authorization": f"Bearer {api_key}"},
        )

    assert resp.status_code == 200
    body = resp.json()
    # Local gate recheck dropped the misclassification — went to search.
    assert body["applied_mode"] == "search"


async def test_router_fallback_routes_to_search(live_db) -> None:
    """REGRESSION: existing semantic queries (where the router emits
    nothing — e.g. router timeout or empty key) keep going through the
    semantic pipeline. Ensures the file split + dispatcher didn't break
    today's behavior on a typical 'what is X' query."""
    api_key = await _seed_customer("cust-3")
    base = datetime(2026, 4, 28, tzinfo=UTC)
    await _seed_doc(
        "cust-3",
        "doc:1",
        source_system="github",
        doc_type="github.commit",
        title="any commit",
        updated_at=base,
    )

    # Empty payload — like the router's RouterTimeout / RouterParseError fallback.
    haiku_payload = {
        "entities": [],
        "expansions": [],
        "temporal": None,
        "sort": None,
        "mode": "search",
        "doc_type": None,
        "operation": None,
        "group_by_key": None,
    }
    with patch("services.retrieval.router.AsyncAnthropic") as mock_client_cls:
        mock_client_cls.return_value.messages.create = AsyncMock(
            return_value=_tool_use_resp(haiku_payload)
        )
        resp = await _post(
            {"query": "what is the auth middleware", "top_k": 5},
            {"Authorization": f"Bearer {api_key}"},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["applied_mode"] == "search"
    # Semantic chunks list shape is the same as before the PR.
    assert isinstance(body["chunks"], list)
    assert body["aggregation"] is None


async def test_count_query_returns_aggregation(live_db) -> None:
    api_key = await _seed_customer("cust-4")
    base = datetime(2026, 4, 28, tzinfo=UTC)
    for i in range(3):
        await _seed_doc(
            "cust-4",
            f"github:repo:commit:{i}",
            source_system="github",
            doc_type="github.commit",
            title=f"c{i}",
            updated_at=base - timedelta(minutes=i),
        )

    haiku_payload = {
        "entities": [],
        "expansions": [],
        "temporal": None,
        "sort": {"field": "updated_at", "direction": "desc", "trigger_phrase": "this week"},
        "mode": "list",
        "doc_type": "commit",
        "operation": "count",
        "group_by_key": None,
    }
    with patch("services.retrieval.router.AsyncAnthropic") as mock_client_cls:
        mock_client_cls.return_value.messages.create = AsyncMock(
            return_value=_tool_use_resp(haiku_payload)
        )
        resp = await _post(
            {"query": "how many commits this week", "top_k": 10},
            {"Authorization": f"Bearer {api_key}"},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["applied_mode"] == "list"
    assert body["chunks"] == []  # count query, no chunks
    assert body["aggregation"] == {"count": 3}
