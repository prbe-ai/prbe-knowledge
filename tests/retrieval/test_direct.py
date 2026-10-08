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


async def test_bm25_only_never_calls_the_vector_channel(monkeypatch):
    """`["bm25"]` is the zero-model-call lookup: vector_search embeds the query,
    so it must not even be invoked, let alone have its result ignored."""
    vector = AsyncMock(side_effect=AssertionError("vector_search embeds the query"))
    bm25 = AsyncMock(return_value=[hit("keyword", channel="bm25")])
    monkeypatch.setattr(direct, "vector_search", vector)
    monkeypatch.setattr(direct, "bm25_search", bm25)
    response = await direct.retrieve_direct(
        direct.DirectRetrieveRequest(query="concept", channels=["bm25"]), "tenant"
    )
    vector.assert_not_called()
    bm25.assert_awaited_once()
    assert [doc.doc_id for doc in response.results] == ["keyword"]
    assert [m.channel for m in response.results[0].matched_via] == ["bm25"]
    assert set(response.timing_ms) == {"bm25", "total"}
    assert not response.lost_channels and not response.degraded


async def test_default_channels_run_both_retrievers(monkeypatch):
    vector = AsyncMock(return_value=[])
    bm25 = AsyncMock(return_value=[])
    monkeypatch.setattr(direct, "vector_search", vector)
    monkeypatch.setattr(direct, "bm25_search", bm25)
    req = direct.DirectRetrieveRequest(query="concept")
    assert req.channels == [direct.DirectChannel.VECTOR, direct.DirectChannel.BM25]
    assert req.index_side_doc_filters is False
    await direct.retrieve_direct(req, "tenant")
    vector.assert_awaited_once()
    bm25.assert_awaited_once()


def test_channels_must_name_at_least_one_known_channel():
    for channels in ([], ["graph"]):
        with pytest.raises(ValidationError):
            direct.DirectRetrieveRequest(query="x", channels=channels)


def test_duplicate_channels_collapse_in_order():
    req = direct.DirectRetrieveRequest(query="x", channels=["bm25", "vector", "bm25"])
    assert req.channels == [direct.DirectChannel.BM25, direct.DirectChannel.VECTOR]
    assert direct.DirectRetrieveRequest(query="x", channels=["bm25", "bm25"]).channels == [
        direct.DirectChannel.BM25
    ]


async def test_listing_order_does_not_change_the_representative_hit(monkeypatch):
    """Hits fold in a fixed order, so `["bm25", "vector"]` answers exactly as
    the default does."""
    monkeypatch.setattr(
        direct, "vector_search", AsyncMock(return_value=[hit("both", score=0.01)])
    )
    monkeypatch.setattr(
        direct, "bm25_search", AsyncMock(return_value=[hit("both", channel="bm25", score=9)])
    )
    default = await direct.retrieve_direct(direct.DirectRetrieveRequest(query="x"), "t")
    reordered = await direct.retrieve_direct(
        direct.DirectRetrieveRequest(query="x", channels=["bm25", "vector"]), "t"
    )
    for response in (default, reordered):
        assert [m.channel for m in response.results[0].matched_via] == ["vector", "bm25"]


async def test_bm25_only_failure_is_reported_as_the_lost_channel(monkeypatch):
    monkeypatch.setattr(
        direct, "vector_search", AsyncMock(side_effect=AssertionError("not requested"))
    )
    monkeypatch.setattr(direct, "bm25_search", AsyncMock(side_effect=TimeoutError("busy")))
    response = await direct.retrieve_direct(
        direct.DirectRetrieveRequest(query="concept", channels=["bm25"]), "tenant"
    )
    assert response.results == []
    assert response.lost_channels == ["bm25"]
    assert response.degraded
    assert response.degraded_reason == "retrieval_channel_unavailable"


async def test_candidate_cap_counts_only_requested_channels(monkeypatch):
    monkeypatch.setattr(
        direct, "bm25_search", AsyncMock(return_value=[
            hit("long", channel="bm25", chunk=str(i)) for i in range(8)
        ]),
    )
    monkeypatch.setattr(
        direct, "vector_search", AsyncMock(side_effect=AssertionError("not requested"))
    )
    response = await direct.retrieve_direct(
        direct.DirectRetrieveRequest(query="concept", top_k=2, channels=["bm25"]), "tenant"
    )
    assert len(response.results) == 1
    assert response.truncated


async def test_index_side_doc_filters_reaches_bm25_and_never_vector(monkeypatch):
    vector = AsyncMock(return_value=[])
    bm25 = AsyncMock(return_value=[])
    monkeypatch.setattr(direct, "vector_search", vector)
    monkeypatch.setattr(direct, "bm25_search", bm25)
    req = direct.DirectRetrieveRequest(
        query="concept", sources=["custom_ingest"], source_keys=["experiments"],
        index_side_doc_filters=True,
    )
    await direct.retrieve_direct(req, "tenant")
    assert bm25.await_args.kwargs["index_side_doc_filters"] is True
    assert bm25.await_args.kwargs["sources"] == ["custom_ingest"]
    assert "index_side_doc_filters" not in vector.await_args.kwargs


async def test_index_side_doc_filters_off_calls_bm25_exactly_as_before(monkeypatch):
    """The flag is passed only when set, so every existing request reaches
    bm25_search with the arguments it always had."""
    bm25 = AsyncMock(return_value=[])
    monkeypatch.setattr(direct, "vector_search", AsyncMock(return_value=[]))
    monkeypatch.setattr(direct, "bm25_search", bm25)
    await direct.retrieve_direct(
        direct.DirectRetrieveRequest(query="concept", sources=["custom_ingest"]), "tenant"
    )
    assert "index_side_doc_filters" not in bm25.await_args.kwargs


@pytest.mark.parametrize("key", ["a" * 129, "Experiments", "-leading", "has space", ""])
def test_source_keys_outside_the_ingest_charset_are_refused(key):
    """One oversized or malformed key is a 422, not a lost BM25 channel: the
    index-side scope compiles every key into a regex, and no stored document
    carries a key outside this charset anyway."""
    with pytest.raises(ValidationError):
        direct.DirectRetrieveRequest(query="x", source_keys=["experiments", key])


def test_source_keys_inside_the_ingest_charset_are_accepted():
    keys = ["experiments", "team_notes", "workspace:1d155c9c-4f05-4707-98a7-f69763c171e0",
            "a" * 128]
    assert direct.DirectRetrieveRequest(query="x", source_keys=keys).source_keys == keys


async def test_bm25_only_caps_each_document_at_the_chunks_a_result_shows(monkeypatch):
    """One document matching everywhere must not take every chunk slot."""
    bm25 = AsyncMock(return_value=[])
    monkeypatch.setattr(direct, "vector_search", AsyncMock(side_effect=AssertionError("no")))
    monkeypatch.setattr(direct, "bm25_search", bm25)
    await direct.retrieve_direct(
        direct.DirectRetrieveRequest(query="concept", channels=["bm25"]), "tenant"
    )
    assert bm25.await_args.kwargs["max_chunks_per_doc"] == direct.CHUNKS_PER_DOCUMENT == 2


@pytest.mark.parametrize("channels", [None, ["bm25", "vector"], ["vector", "bm25"]])
async def test_two_channel_requests_send_bm25_no_document_cap(monkeypatch, channels):
    """The typeahead's request (both channels) reaches bm25_search unchanged."""
    bm25 = AsyncMock(return_value=[])
    monkeypatch.setattr(direct, "vector_search", AsyncMock(return_value=[]))
    monkeypatch.setattr(direct, "bm25_search", bm25)
    extra = {} if channels is None else {"channels": channels}
    await direct.retrieve_direct(direct.DirectRetrieveRequest(query="concept", **extra), "t")
    assert "max_chunks_per_doc" not in bm25.await_args.kwargs
