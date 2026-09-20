"""Interactive recall uses both actual retriever interfaces, never a gatherer."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from pydantic import ValidationError

from engine.retrieval import direct
from engine.retrieval.retrievers.bm25 import BM25Hit
from engine.retrieval.retrievers.vector import VectorHit


@pytest.fixture(autouse=True)
def live_documents(monkeypatch):
    async def live(_req, _customer, doc_ids):
        return {(doc_id, 1) for doc_id in doc_ids}

    monkeypatch.setattr(direct, "_live_docs", live)


def hit(doc_id, *, channel="vector", chunk="0", score=0.8, kind="content"):
    cls = VectorHit if channel == "vector" else BM25Hit
    return cls(
        chunk_id=f"{doc_id}#{chunk}",
        doc_id=doc_id,
        doc_version=1,
        source_system="custom_ingest",
        source_url=f"https://example.test/{doc_id}",
        title=doc_id,
        content=f"evidence for {doc_id}",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
        score=score,
        kind=kind,
    )


async def test_combines_keyword_and_semantic_recall_without_comparing_scores(monkeypatch):
    vector = AsyncMock(return_value=[hit("both", score=0.01), hit("concept", score=0.001)])
    bm25 = AsyncMock(
        return_value=[hit("keyword", channel="bm25", score=1000), hit("both", channel="bm25")]
    )
    monkeypatch.setattr(direct, "vector_search", vector)
    monkeypatch.setattr(direct, "bm25_search", bm25)
    project = str(uuid4())
    req = direct.DirectRetrieveRequest(
        query="hidden reasoning",
        top_k=3,
        sources=["custom_ingest"],
        source_keys=["experiments"],
        doc_types=["custom.experiment.run"],
        scope={"project_id": project},
    )
    response = await direct.retrieve_direct(req, "tenant-a")
    assert [doc.doc_id for doc in response.results] == ["both", "keyword", "concept"]
    assert {m.channel for m in response.results[0].matched_via} == {"vector", "bm25"}
    assert not response.lost_channels and not response.degraded
    for search in (vector, bm25):
        search.assert_awaited_once_with(
            "tenant-a", "hidden reasoning", top_k=12, sources=["custom_ingest"],
            source_keys=["experiments"], doc_types=["custom.experiment.run"], project_id=project,
        )
    assert response.applied_scope == {"project_id": project}


async def test_long_documents_get_one_vote_per_channel_and_metadata_is_not_preview(monkeypatch):
    monkeypatch.setattr(
        direct, "vector_search", AsyncMock(return_value=[
            hit("long", chunk="meta", kind="metadata"),
            *[hit("long", chunk=str(i)) for i in range(15)],
            hit("other"),
        ]),
    )
    monkeypatch.setattr(
        direct, "bm25_search", AsyncMock(return_value=[hit("other", channel="bm25")]),
    )
    response = await direct.retrieve_direct(direct.DirectRetrieveRequest(query="concept"), "tenant")
    assert [doc.doc_id for doc in response.results] == ["other", "long"]
    assert response.results[0].matched_via[0].rank == 2
    assert len(response.results[1].matched_via) == 1
    assert len(response.results[1].chunks) == 2
    assert all(not chunk.chunk_id.endswith("#meta") for chunk in response.results[1].chunks)


@pytest.mark.parametrize("failed", ["vector", "bm25"])
async def test_channel_failure_preserves_other_channel_and_is_explicit(monkeypatch, failed):
    for name in ("vector", "bm25"):
        search = AsyncMock(
            side_effect=TimeoutError("index busy") if name == failed else None,
            return_value=[hit("survivor", channel=name)],
        )
        monkeypatch.setattr(direct, f"{name}_search", search)
    response = await direct.retrieve_direct(direct.DirectRetrieveRequest(query="concept"), "tenant")
    assert [doc.doc_id for doc in response.results] == ["survivor"]
    assert response.degraded and response.lost_channels == [failed]


async def test_direct_endpoint_authenticates_and_never_enters_agentic_pipeline(monkeypatch):
    from engine.retrieval import main
    from engine.shared.config import Settings

    monkeypatch.setattr(
        "engine.retrieval.auth.get_settings",
        lambda: Settings(internal_knowledge_api_key="test-secret"),
    )
    monkeypatch.setattr(main, "run_retrieval", AsyncMock(side_effect=AssertionError("gatherer")))
    vector = AsyncMock(return_value=[hit("semantic")])
    monkeypatch.setattr(direct, "vector_search", vector)
    monkeypatch.setattr(direct, "bm25_search", AsyncMock(return_value=[]))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=main.app), base_url="http://test",
    ) as client:
        response = await client.post("/retrieve/direct", json={"query": "concept"})
        assert response.status_code == 401
        vector.assert_not_awaited()
        response = await client.post(
            "/retrieve/direct", json={"query": "concept"},
            headers={"X-Internal-Knowledge-Key": "test-secret", "X-Prbe-Customer": "tenant-a"},
        )
    assert response.status_code == 200, response.text
    assert response.json()["applied_mode"] == "direct"
    assert vector.await_args.args[0] == "tenant-a"


async def test_stale_versions_deleted_and_out_of_scope_hits_are_dropped(monkeypatch):
    monkeypatch.setattr(direct, "vector_search", AsyncMock(return_value=[hit("live"), hit("gone")]))
    monkeypatch.setattr(direct, "bm25_search", AsyncMock(return_value=[hit("changed", channel="bm25")]))
    monkeypatch.setattr(direct, "_live_docs", AsyncMock(return_value={("live", 1), ("changed", 2)}))
    response = await direct.retrieve_direct(direct.DirectRetrieveRequest(query="concept"), "tenant")
    assert [doc.doc_id for doc in response.results] == ["live"]


async def test_live_lookup_failure_never_ships_unverified_content(monkeypatch):
    monkeypatch.setattr(direct, "vector_search", AsyncMock(return_value=[hit("unverified")]))
    monkeypatch.setattr(direct, "bm25_search", AsyncMock(return_value=[]))
    monkeypatch.setattr(direct, "_live_docs", AsyncMock(side_effect=TimeoutError("pool busy")))
    response = await direct.retrieve_direct(direct.DirectRetrieveRequest(query="concept"), "tenant")
    assert response.results == []
    assert response.degraded and response.lost_channels == ["live_documents"]


async def test_empty_optional_scope_does_not_become_a_literal_none_project(monkeypatch):
    vector = AsyncMock(return_value=[])
    monkeypatch.setattr(direct, "vector_search", vector)
    monkeypatch.setattr(direct, "bm25_search", AsyncMock(return_value=[]))
    response = await direct.retrieve_direct(
        direct.DirectRetrieveRequest(query="concept", scope={}), "tenant"
    )
    assert vector.await_args.kwargs["project_id"] is None
    assert not response.degraded


async def test_chunk_pool_cap_is_reported_even_when_only_one_document_survives(monkeypatch):
    monkeypatch.setattr(direct, "vector_search", AsyncMock(return_value=[
        hit("long", chunk=str(i)) for i in range(8)
    ]))
    monkeypatch.setattr(direct, "bm25_search", AsyncMock(return_value=[]))
    response = await direct.retrieve_direct(direct.DirectRetrieveRequest(query="concept", top_k=2), "tenant")
    assert len(response.results) == 1
    assert response.truncated


async def test_cancellation_propagates_and_cancels_both_index_reads(monkeypatch):
    import asyncio

    started = {channel: asyncio.Event() for channel in ("vector", "bm25")}
    cancelled = set()

    def search(channel):
        async def wait(*_args, **_kwargs):
            started[channel].set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.add(channel)
        return wait

    for name in ("vector", "bm25"):
        monkeypatch.setattr(direct, f"{name}_search", search(name))
    task = asyncio.create_task(direct.retrieve_direct(direct.DirectRetrieveRequest(query="x"), "tenant"))
    await asyncio.wait_for(asyncio.gather(*(event.wait() for event in started.values())), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled == {"vector", "bm25"}


@pytest.mark.parametrize("payload", [
    {"query": " "}, {"query": "x", "top_k": 51}, {"query": "x", "top_k": 0},
    {"query": "x", "include_drafts": True}, {"query": "x", "customer_id": "other"},
])
def test_direct_request_is_bounded_and_cannot_assert_identity_or_draft_access(payload):
    with pytest.raises(ValidationError):
        direct.DirectRetrieveRequest(**payload)
