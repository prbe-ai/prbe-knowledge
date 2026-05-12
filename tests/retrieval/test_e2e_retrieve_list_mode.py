"""End-to-end /retrieve test for the deterministic list path.

Reproduces the original bug ("3 most recent github commits returns PRs
mixed with commits and the actually-newest commits are missing") and
verifies the fix: with mode=list dispatched, the result is exactly the
N newest commits, sorted by date, no PR leakage.

Mocks Haiku at the ``shared.llm_tools.acompletion`` boundary so no
network call is made; the SQL path itself runs against the real test DB.
(Pre-Phase-0b this mocked ``services.retrieval.router.AsyncAnthropic``;
that import is gone now — the router goes through ``shared.llm`` via the
``forced_tool_call`` helper.)
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import orjson
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
    monkeypatch.delenv("LLM_GATEWAY_URL", raising=False)
    reset_embedder()
    reset_store()
    get_settings.cache_clear()  # type: ignore[attr-defined]


def _tool_use_resp(payload: dict) -> SimpleNamespace:
    """LiteLLM-shaped response carrying a single forced tool call.

    Matches what ``shared.llm.acompletion`` returns to
    ``shared.llm_tools.forced_tool_call`` after the Phase-0b migration:
    ``choices[0].message.tool_calls[0].function.{name, arguments}`` where
    ``arguments`` is a JSON string.
    """
    func = SimpleNamespace(
        name="route_query",
        arguments=orjson.dumps(payload).decode("utf-8"),
    )
    call = SimpleNamespace(type="function", function=func)
    message = SimpleNamespace(content=None, tool_calls=[call])
    choice = SimpleNamespace(message=message, finish_reason="tool_calls")
    return SimpleNamespace(choices=[choice], usage=None)


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
    author_id: str | None = None,
) -> None:
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO documents (
                doc_id, version, customer_id,
                source_system, source_id, source_url,
                doc_class, doc_type, content_type,
                content_hash, title, body_size_bytes, body_token_count,
                author_id,
                created_at, updated_at, valid_from, ingested_at,
                acl
            ) VALUES (
                $1, 1, $2,
                $3, $4, $5,
                'raw_source', $6, 'text/plain',
                $7, $8, 10, 0,
                $9,
                $10, $10, $10, $10,
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
            author_id,
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

    # Haiku for "3 most recent github commits" extracts sort + doc_type but
    # no narrowing entity (github is a source-system mention, not a repo
    # canonical_id). With no entity, the list path skips the graph entity
    # filter and returns N most-recent commits.
    haiku_payload = {
        "entities": [],
        "expansions": [],
        "temporal": None,
        "sort": {"field": "updated_at", "direction": "desc", "trigger_phrase": "most recent"},
        "mode": "list",
        "doc_type": "commit",
        "operation": "list",
        "group_by_key": None,
    }

    # Phase-0b: router goes through shared.llm; mock at the helper's boundary.
    with patch(
        "shared.llm_tools.acompletion",
        AsyncMock(return_value=_tool_use_resp(haiku_payload)),
    ):
        resp = await _post(
            {"query": "3 most recent github commits", "top_k": 3},
            {"Authorization": f"Bearer {api_key}"},
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["applied_mode"] == "list"
    assert body["applied_doc_types"] == ["github.commit"]
    # Polymorphic shape: list pipeline returns Document results; the title
    # lives on the Document itself, not on the nested chunk
    # (PR feat/polymorphic-search-results).
    docs = [r for r in body["results"] if r["node_type"] == "Document"]
    assert len(docs) == 3
    # Newest first, all commits, no PR leakage.
    titles = [d["title"] for d in docs]
    assert titles == ["commit 0", "commit 1", "commit 2"]
    for d in docs:
        assert "PR" not in (d["title"] or "")


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

    # Phase-0b: router goes through shared.llm; mock at the helper's boundary.
    with patch(
        "shared.llm_tools.acompletion",
        AsyncMock(return_value=_tool_use_resp(haiku_payload)),
    ):
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
    # Phase-0b: router goes through shared.llm; mock at the helper's boundary.
    with patch(
        "shared.llm_tools.acompletion",
        AsyncMock(return_value=_tool_use_resp(haiku_payload)),
    ):
        resp = await _post(
            {"query": "what is the auth middleware", "top_k": 5},
            {"Authorization": f"Bearer {api_key}"},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["applied_mode"] == "search"
    # Polymorphic results list shape (PR feat/polymorphic-search-results).
    assert isinstance(body["results"], list)
    assert body["aggregation"] is None


async def test_author_id_surfaces_on_list_chunks(live_db) -> None:
    """REGRESSION: a Mahit commit must self-identify on the wire.

    Seeds two commits — one by Mahit, one by an anonymous "unknown" author.
    The list-path response must carry `author_id` per chunk: "mahit" for
    his row, null (not the literal "unknown") for the anonymous one.
    """
    api_key = await _seed_customer("cust-author-list")
    base = datetime(2026, 4, 28, tzinfo=UTC)
    await _seed_doc(
        "cust-author-list",
        "github:repo:commit:by-mahit",
        source_system="github",
        doc_type="github.commit",
        title="mahit's commit",
        updated_at=base,
        author_id="mahit",
    )
    await _seed_doc(
        "cust-author-list",
        "github:repo:commit:no-author",
        source_system="github",
        doc_type="github.commit",
        title="anonymous commit",
        updated_at=base - timedelta(minutes=1),
        author_id="unknown",
    )

    haiku_payload = {
        "entities": [],
        "expansions": [],
        "temporal": None,
        "sort": {"field": "updated_at", "direction": "desc", "trigger_phrase": "most recent"},
        "mode": "list",
        "doc_type": "commit",
        "operation": "list",
        "group_by_key": None,
    }
    # Phase-0b: router goes through shared.llm; mock at the helper's boundary.
    with patch(
        "shared.llm_tools.acompletion",
        AsyncMock(return_value=_tool_use_resp(haiku_payload)),
    ):
        resp = await _post(
            {"query": "recent commits", "top_k": 10},
            {"Authorization": f"Bearer {api_key}"},
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    # author_id moved from chunk-level (old QueryChunk) to doc-level
    # (QueryDocumentResult) per the polymorphic shape.
    by_doc = {
        d["doc_id"]: d
        for d in body["results"]
        if d["node_type"] == "Document"
    }
    assert by_doc["github:repo:commit:by-mahit"]["author_id"] == "mahit"
    assert by_doc["github:repo:commit:no-author"]["author_id"] is None


async def test_author_id_surfaces_on_search_chunks(live_db) -> None:
    """Search-path version of the regression: vector + BM25 + graph fusion
    must all preserve author_id on the way to QueryChunk. If any retriever
    forgets to SELECT d.author_id, the field will be None for chunks that
    only that retriever surfaced — this catches that."""
    api_key = await _seed_customer("cust-author-search")
    base = datetime(2026, 4, 28, tzinfo=UTC)
    await _seed_doc(
        "cust-author-search",
        "github:repo:commit:mahit-search",
        source_system="github",
        doc_type="github.commit",
        title="mahit shipped the auth fix",
        updated_at=base,
        author_id="mahit",
    )

    # Empty router payload → search mode (vector + BM25 + graph + fusion).
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
    # Phase-0b: router goes through shared.llm; mock at the helper's boundary.
    with patch(
        "shared.llm_tools.acompletion",
        AsyncMock(return_value=_tool_use_resp(haiku_payload)),
    ):
        resp = await _post(
            {"query": "auth fix", "top_k": 5},
            {"Authorization": f"Bearer {api_key}"},
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["applied_mode"] == "search"
    docs = [r for r in body["results"] if r["node_type"] == "Document"]
    assert len(docs) >= 1
    target = next(
        (d for d in docs if d["doc_id"] == "github:repo:commit:mahit-search"),
        None,
    )
    assert target is not None, "seeded commit did not surface in search results"
    assert target["author_id"] == "mahit"


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
    # Phase-0b: router goes through shared.llm; mock at the helper's boundary.
    with patch(
        "shared.llm_tools.acompletion",
        AsyncMock(return_value=_tool_use_resp(haiku_payload)),
    ):
        resp = await _post(
            {"query": "how many commits this week", "top_k": 10},
            {"Authorization": f"Bearer {api_key}"},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["applied_mode"] == "list"
    assert body["results"] == []  # count query, no per-doc results
    assert body["aggregation"] == {"count": 3}
