"""The `floor` and `jev` result selectors, and the one-time query rewrite.

Two layers:

  * `engine/retrieval/agent/jev.py` -- pool, batching, the HTTP call's failure
    shapes, ranking. Driven through an httpx MockTransport: no key, no network.
  * `run_gatherer` end to end with each selector, on the suite's stubbed
    grounding / extraction / pre-fan-out. The load-bearing checks are that
    `floor` and `jev` NEVER call the gatherer LLM, that Jev's order survives
    to the response, and that every Jev failure degrades to the recall floor
    instead of failing the search.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from engine.retrieval.agent import jev
from engine.retrieval.agent import loop as L
from engine.retrieval.agent.models import EntityExtraction
from engine.retrieval.grounding import GroundingBundle
from engine.shared.models import QueryRequest


def _hit(doc: str, *, chunk: str | None = None, content: str = "body", title: str = ""):
    return {
        "doc_id": doc,
        "chunk_id": chunk or f"{doc}#0",
        "content": content,
        "source_system": "github",
        "title": title or doc,
        "score": 0.5,
    }


def _pf(vector=(), bm25=(), graph=()):
    return {"sub_queries": [{
        "query": "q", "grounded_entities": [],
        "vector": list(vector), "bm25": list(bm25),
        "graph": list(graph), "inferred_edge": [],
    }]}


# =====================================================================
# jev.py -- pure parts
# =====================================================================

def test_pool_dedupes_by_chunk_and_skips_unscoreable_hits():
    pf = _pf(
        vector=[_hit("d1", content="first"), {"doc_id": "d9", "chunk_id": "d9#0"}],
        bm25=[_hit("d1", content="second"), "junk"],
    )
    pool = jev.pool_chunks(pf)
    assert list(pool) == ["d1#0"]
    assert pool["d1#0"]["content"] == "first"


def test_a_chunk_reports_every_channel_that_retrieved_it():
    pf = _pf(vector=[_hit("d1")], bm25=[_hit("d1")], graph=[_hit("d2")])
    assert jev.channels_by_chunk(pf) == {"d1#0": ["vector", "bm25"], "d2#0": ["graph"]}


def test_questions_cost_tokens_as_well_as_state():
    assert jev.estimate_tokens(0, 100) == 3500


def test_batches_fit_the_budget_and_lose_nothing():
    pool = {f"c{i}": _hit(f"d{i}", chunk=f"c{i}", content="x" * 5000) for i in range(60)}
    batches = jev.batch_pool(pool, query="q")
    assert len(batches) > 1
    flat = [c for b in batches for c in b]
    assert sorted(flat) == sorted(pool) and len(flat) == len(set(flat))


def test_one_enormous_chunk_is_truncated_not_dropped():
    pool = {"big": _hit("d1", chunk="big", content="x" * 10_000_000)}
    assert jev.batch_pool(pool, query="q") == [["big"]]
    assert len(jev._body(pool["big"])) == jev.JEV_MAX_CHUNK_CHARS


def test_rank_scores_a_document_by_its_best_chunk_and_counts_documents():
    pool = {
        "d1#0": _hit("d1", chunk="d1#0"),
        "d1#3": _hit("d1", chunk="d1#3"),
        "d2#0": _hit("d2", chunk="d2#0"),
        "d3#0": _hit("d3", chunk="d3#0"),
    }
    ranked = jev.rank_documents(
        pool, {"d1#0": 0.1, "d1#3": 0.9, "d2#0": 0.5}, limit=10
    )
    assert [(r.doc_id, r.chunk_id) for r in ranked] == [("d1", "d1#3"), ("d2", "d2#0")]
    # d3 was never scored: absent, not ranked as zero
    assert all(r.doc_id != "d3" for r in ranked)


def test_rank_ties_keep_retrieval_order():
    pool = {f"d{i}#0": _hit(f"d{i}") for i in range(4)}
    ranked = jev.rank_documents(pool, {k: 0.5 for k in pool}, limit=3)
    assert [r.doc_id for r in ranked] == ["d0", "d1", "d2"]


@pytest.mark.parametrize("best,want", [(None, "low"), (0.1, "low"), (0.39, "low"),
                                       (0.4, "medium"), (0.69, "medium"), (0.7, "high")])
def test_confidence_follows_the_phase0_answerability_cut(best, want):
    assert jev.confidence_for(best) == want


def test_near_copy():
    assert jev.near_copy("why did the auth refactor break login", "why did auth refactor break login")
    assert not jev.near_copy("auth refactor", "session token rotation in the gateway")


# =====================================================================
# jev.py -- the HTTP call, through a MockTransport
# =====================================================================

def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _ok(ids, value=0.8):
    return httpx.Response(200, json={
        "model": "jev-1.13.0",
        "usage": {"input_tokens": 100},
        "answers": {i: {"type": "noul", "noul": value} for i in ids},
    })


@pytest.mark.asyncio
async def test_score_pool_sends_the_pinned_model_and_one_noul_per_chunk():
    seen = {}

    def handler(req):
        body = json.loads(req.content)
        seen.update(body)
        assert req.headers["Authorization"] == "Bearer k"
        return _ok(body["questions"])

    pool = {f"c{i}": _hit(f"d{i}", chunk=f"c{i}") for i in range(3)}
    res = await jev.score_pool("q", pool, api_key="k", client=_client(handler))
    assert seen["model"] == jev.JEV_MODEL
    assert set(seen["questions"]) == set(pool)
    assert all(q["type"] == "noul" for q in seen["questions"].values())
    assert res.scores == {k: 0.8 for k in pool} and res.requests == 1


@pytest.mark.asyncio
async def test_no_key_raises_so_the_caller_falls_back():
    with pytest.raises(jev.JevError):
        await jev.score_pool("q", {"c": _hit("d")}, api_key="")


@pytest.mark.asyncio
async def test_over_cap_halves_and_retries():
    def handler(req):
        qs = json.loads(req.content)["questions"]
        if len(qs) > 2:
            return httpx.Response(400, json={"detail": {"error_type": "max_tokens_exceeded"}})
        return _ok(qs)

    pool = {f"c{i}": _hit(f"d{i}", chunk=f"c{i}") for i in range(8)}
    res = await jev.score_pool("q", pool, api_key="k", client=_client(handler))
    assert set(res.scores) == set(pool)
    assert res.splits >= 3


@pytest.mark.asyncio
async def test_a_missing_answer_is_absent_not_zero():
    def handler(req):
        return _ok(["c0"])

    pool = {"c0": _hit("d0", chunk="c0"), "c1": _hit("d1", chunk="c1")}
    res = await jev.score_pool("q", pool, api_key="k", client=_client(handler))
    assert "c1" not in res.scores


@pytest.mark.asyncio
async def test_every_batch_failing_raises_with_a_readable_reason():
    def handler(req):
        raise httpx.ReadTimeout("", request=req)  # httpx timeouts stringify EMPTY

    with pytest.raises(jev.JevError) as e:
        await jev.score_pool("q", {"c": _hit("d")}, api_key="k", client=_client(handler))
    assert "ReadTimeout" in str(e.value)


@pytest.mark.asyncio
async def test_one_failed_batch_keeps_the_others():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        qs = json.loads(req.content)["questions"]
        if "c0" in qs:
            return httpx.Response(503, text="busy")
        return _ok(qs)

    pool = {f"c{i}": _hit(f"d{i}", chunk=f"c{i}", content="x" * 40_000) for i in range(4)}
    res = await jev.score_pool("q", pool, api_key="k", client=_client(handler))
    assert "c0" not in res.scores and res.scores
    assert any("http_503" in e for e in res.errors)


# =====================================================================
# run_gatherer end to end
# =====================================================================

POOL = _pf(
    vector=[_hit(f"v{i}", title=f"vec {i}") for i in range(12)],
    bm25=[_hit("v3"), _hit("b1", title="bm25 only")],
)


@pytest.fixture(autouse=True)
def _stubs(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(L, "_no_llm_configured", lambda: False)
    monkeypatch.setattr(L, "_build_bundle_with_token_fallback",
                        AsyncMock(return_value=GroundingBundle()))
    monkeypatch.setattr(L, "extract_entities_with_llm",
                        AsyncMock(return_value=EntityExtraction()))

    async def _all_live(customer_id, doc_ids, **kwargs):  # type: ignore[no-untyped-def]
        return {d: True for d in doc_ids}

    monkeypatch.setattr("engine.retrieval.agent.adapter._scope_verdicts", _all_live)
    monkeypatch.setattr(L, "execute_search", AsyncMock(return_value=json.loads(json.dumps(POOL))))


def _req(**kw) -> QueryRequest:
    return QueryRequest(query="why did the auth refactor break login", **kw)


def _state():
    return SimpleNamespace(state=SimpleNamespace())


def _scored(mapping):
    """A stand-in `jev.score_pool` returning fixed chunk scores."""
    async def fake(query, pool, *, api_key, client=None):
        out = jev.ScoreResult(requests=1, input_tokens=10)
        out.scores = {c: mapping.get(c, 0.05) for c in pool}
        return out
    return fake


@pytest.mark.asyncio
async def test_floor_never_calls_the_gatherer_llm():
    req = _req(selector="floor")
    fr = _state()
    with patch.object(L, "acompletion", new=AsyncMock()) as llm:
        resp = await L.run_gatherer(req, customer_id="c1", request=fr)
    llm.assert_not_called()
    assert len(resp.results) == 10
    assert fr.state.router_model == "recall_floor"
    assert fr.state.gatherer_status == "ok"
    assert resp.degraded is False


@pytest.mark.asyncio
async def test_jev_ranking_reaches_the_response_in_order():
    fr = _state()
    scores = {"v7#0": 0.95, "b1#0": 0.9, "v2#0": 0.8}
    with patch.object(L, "acompletion", new=AsyncMock()) as llm, \
         patch.object(jev, "score_pool", new=_scored(scores)):
        resp = await L.run_gatherer(_req(selector="jev"), customer_id="c1", request=fr)
    llm.assert_not_called()  # best 0.95 >= 0.4: no rewrite, no LLM at all
    assert [r.doc_id for r in resp.results[:3]] == ["v7", "b1", "v2"]
    assert fr.state.router_model == jev.JEV_MODEL
    assert fr.state.gatherer_status == "ok"


@pytest.mark.asyncio
async def test_jev_picks_carry_their_real_channels_not_recall_floor():
    fr = _state()
    with patch.object(jev, "score_pool", new=_scored({"v3#0": 0.99})):
        resp = await L.run_gatherer(_req(selector="jev"), customer_id="c1", request=fr)
    top = resp.results[0]
    assert top.doc_id == "v3"
    channels = {m.channel for m in top.matched_via}
    assert {"vector", "bm25"} <= channels and "recall_floor" not in channels


@pytest.mark.asyncio
@pytest.mark.parametrize("exc", [jev.JevError("no key"), TimeoutError()])
async def test_any_jev_failure_degrades_to_the_floor_never_fails(exc):
    async def boom(*a, **k):
        raise exc

    fr = _state()
    with patch.object(jev, "score_pool", new=boom):
        resp = await L.run_gatherer(_req(selector="jev"), customer_id="c1", request=fr)
    assert fr.state.gatherer_status == "jev_unavailable"
    assert resp.degraded is True
    assert len(resp.results) == 10  # the floor answered


def _llm_says(text):
    return AsyncMock(return_value=SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")],
        usage=None,
    ))


@pytest.mark.asyncio
async def test_weak_scores_trigger_exactly_one_rewrite_that_widens_the_pool(monkeypatch):
    extra = _pf(vector=[_hit("new1", title="found by rewrite")])
    search = AsyncMock(side_effect=[json.loads(json.dumps(POOL)), extra])
    monkeypatch.setattr(L, "execute_search", search)
    fr = _state()
    scores = {"new1#0": 0.9}  # everything in the first pool scores 0.05
    with patch.object(L, "acompletion", new=_llm_says("session token rotation gateway")) as llm, \
         patch.object(jev, "score_pool", new=_scored(scores)):
        resp = await L.run_gatherer(_req(selector="jev"), customer_id="c1", request=fr)
    assert llm.await_count == 1
    assert search.await_count == 2
    assert resp.results[0].doc_id == "new1"


@pytest.mark.asyncio
async def test_a_near_copy_rewrite_does_not_search_again(monkeypatch):
    search = AsyncMock(return_value=json.loads(json.dumps(POOL)))
    monkeypatch.setattr(L, "execute_search", search)
    with patch.object(L, "acompletion", new=_llm_says("why did the auth refactor break the login")), \
         patch.object(jev, "score_pool", new=_scored({})):
        await L.run_gatherer(_req(selector="jev"), customer_id="c1", request=_state())
    assert search.await_count == 1


@pytest.mark.asyncio
async def test_a_rewrite_that_finds_the_same_documents_stops(monkeypatch):
    search = AsyncMock(side_effect=[json.loads(json.dumps(POOL)), json.loads(json.dumps(POOL))])
    monkeypatch.setattr(L, "execute_search", search)
    captured = {}
    real = L._stash_for_trace_persist

    def spy(request, **kw):
        captured["state"] = kw["state"]
        return real(request, **kw)

    monkeypatch.setattr(L, "_stash_for_trace_persist", spy)
    with patch.object(L, "acompletion", new=_llm_says("session token rotation gateway")), \
         patch.object(jev, "score_pool", new=_scored({})):
        await L.run_gatherer(_req(selector="jev"), customer_id="c1", request=_state())
    assert captured["state"].selection["rewrite"]["outcome"] == "repeat_results"


@pytest.mark.asyncio
async def test_an_empty_pool_gets_its_one_rewrite_then_gives_up(monkeypatch):
    empty = {"sub_queries": [{"query": "q", "grounded_entities": [], "vector": [],
                              "bm25": [], "graph": [], "inferred_edge": []}]}
    search = AsyncMock(side_effect=[json.loads(json.dumps(empty)), json.loads(json.dumps(empty))])
    monkeypatch.setattr(L, "execute_search", search)
    fr = _state()
    with patch.object(L, "acompletion", new=_llm_says("login flow authentication")) as llm:
        resp = await L.run_gatherer(_req(selector="jev"), customer_id="c1", request=fr)
    assert llm.await_count == 1 and search.await_count == 2
    assert fr.state.gatherer_status == "zero_recall_short_circuit"
    assert resp.results == []


@pytest.mark.asyncio
async def test_an_empty_pool_rescued_by_the_rewrite_is_scored(monkeypatch):
    empty = {"sub_queries": [{"query": "q", "grounded_entities": [], "vector": [],
                              "bm25": [], "graph": [], "inferred_edge": []}]}
    found = _pf(vector=[_hit("auth1")])
    monkeypatch.setattr(L, "execute_search",
                        AsyncMock(side_effect=[json.loads(json.dumps(empty)), found]))
    fr = _state()
    with patch.object(L, "acompletion", new=_llm_says("authentication service")) as llm, \
         patch.object(jev, "score_pool", new=_scored({"auth1#0": 0.1})):
        resp = await L.run_gatherer(_req(selector="jev"), customer_id="c1", request=fr)
    # The rewrite was spent on the empty pool: the weak score after it must
    # NOT buy a second one.
    assert llm.await_count == 1
    assert resp.results[0].doc_id == "auth1"


def test_explicit_selector_beats_rollout_and_default(monkeypatch):
    monkeypatch.setattr(L, "SEARCH_SELECTOR_JEV_CUSTOMERS", frozenset({"c1"}))
    monkeypatch.setattr(L, "SEARCH_SELECTOR_DEFAULT", "floor")
    assert L._resolve_selector(_req(), "c1") == "jev"
    assert L._resolve_selector(_req(), "c2") == "floor"
    assert L._resolve_selector(_req(selector="gatherer"), "c1") == "gatherer"


def test_an_unknown_default_falls_back_to_the_gatherer(monkeypatch):
    monkeypatch.setattr(L, "SEARCH_SELECTOR_DEFAULT", "typo")
    assert L._resolve_selector(_req(), "c2") == "gatherer"


# =====================================================================
# extraction on Jev: shadow + sampled apply
# =====================================================================

from engine.retrieval.agent.models import SearchOptions  # noqa: E402


def _choice_resp(sort="recency", sconf=0.9, cls="pull_requests", cconf=0.95):
    return httpx.Response(200, json={"model": "jev-1.13.0", "answers": {
        "sort": {"type": "choice", "choice": sort, "confidence": sconf},
        "doc_class": {"type": "choice", "choice": cls, "confidence": cconf},
    }})


@pytest.mark.asyncio
async def test_extract_options_maps_the_class_to_real_doc_types():
    c = await jev.extract_options("latest PRs", api_key="k",
                                  client=_client(lambda r: _choice_resp()))
    assert (c.sort, c.doc_class, c.doc_types) == ("recency", "pull_requests", ["github.pull_request"])


@pytest.mark.asyncio
async def test_an_unknown_choice_degrades_to_the_safe_default():
    c = await jev.extract_options("q", api_key="k",
                                  client=_client(lambda r: _choice_resp(sort="sideways", cls="made_up")))
    assert c.sort == "relevance" and c.doc_class == "no_class" and c.doc_types is None


@pytest.mark.asyncio
async def test_a_malformed_extraction_answer_raises_jev_error():
    bad = lambda r: httpx.Response(200, json={"answers": {}})  # noqa: E731
    with pytest.raises(jev.JevError):
        await jev.extract_options("q", api_key="k", client=_client(bad))


def _gpt(sort="relevance", doc_types=None):
    return EntityExtraction(search_options=SearchOptions(sort=sort, doc_types=doc_types))


def _jc(sort="recency", cls="pull_requests", cconf=0.95):
    return jev.ExtractionChoice(sort=sort, sort_confidence=0.9, doc_class=cls,
                                doc_types=jev.DOC_CLASSES[cls], class_confidence=cconf,
                                elapsed_ms=1.0)


def test_shadow_mode_logs_the_disagreement_and_changes_nothing(monkeypatch):
    monkeypatch.setattr(L, "SEARCH_EXTRACTION_JEV_APPLY_RATE", 0.0)
    out, rec = L._merge_jev_extraction(_gpt(), _jc(), customer_id="c", trace_id="t")
    assert out.search_options.sort == "relevance" and out.search_options.doc_types is None
    assert rec["applied"] is False and rec["sort_agree"] is False and rec["doc_types_agree"] is False


def test_the_sampled_arm_uses_jev_for_sort_and_a_confident_class(monkeypatch):
    monkeypatch.setattr(L, "SEARCH_EXTRACTION_JEV_APPLY_RATE", 1.0)
    out, rec = L._merge_jev_extraction(_gpt(), _jc(), customer_id="c", trace_id="t")
    assert rec["applied"] is True
    assert out.search_options.sort == "recency"
    assert out.search_options.doc_types == ["github.pull_request"]


def test_an_unsure_class_is_left_off_even_in_the_sampled_arm(monkeypatch):
    # A wrong class zeroes the pool; below the confidence floor, no filter.
    monkeypatch.setattr(L, "SEARCH_EXTRACTION_JEV_APPLY_RATE", 1.0)
    out, _ = L._merge_jev_extraction(_gpt(doc_types=["linear.issue"]), _jc(cconf=0.52),
                                     customer_id="c", trace_id="t")
    assert out.search_options.doc_types is None


def test_entities_and_sub_queries_always_stay_gpt_oss(monkeypatch):
    monkeypatch.setattr(L, "SEARCH_EXTRACTION_JEV_APPLY_RATE", 1.0)
    gpt = EntityExtraction(sub_queries=["a", "b"])
    out, _ = L._merge_jev_extraction(gpt, _jc(), customer_id="c", trace_id="t")
    assert out.sub_queries == ["a", "b"]


def test_no_jev_answer_leaves_extraction_untouched():
    out, rec = L._merge_jev_extraction(_gpt(sort="recency"), None, customer_id="c", trace_id="t")
    assert out.search_options.sort == "recency" and rec == {"jev": None, "applied": False}


def test_sampling_is_deterministic_per_trace():
    hits = [L._in_sample(f"t{i}", 0.3) for i in range(2000)]
    assert hits == [L._in_sample(f"t{i}", 0.3) for i in range(2000)]
    assert 0.25 < sum(hits) / len(hits) < 0.35
    assert not L._in_sample("x", 0.0) and L._in_sample("x", 1.0)


@pytest.mark.asyncio
async def test_extraction_on_jev_runs_only_for_jev_searches(monkeypatch):
    called = AsyncMock(return_value=None)
    monkeypatch.setattr(L, "_jev_extraction_or_none", called)
    with patch.object(jev, "score_pool", new=_scored({"v1#0": 0.9})):
        await L.run_gatherer(_req(selector="jev"), customer_id="c1", request=_state())
    assert called.await_count == 1
    called.reset_mock()
    await L.run_gatherer(_req(selector="floor"), customer_id="c1", request=_state())
    assert called.await_count == 0
