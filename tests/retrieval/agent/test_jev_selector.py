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
async def test_one_failed_batch_keeps_the_others(monkeypatch):
    monkeypatch.setattr(jev, "JEV_TOKEN_BUDGET", 24_000)
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        qs = json.loads(req.content)["questions"]
        if "c0" in qs:
            return httpx.Response(503, text="busy")
        return _ok(qs)

    pool = {f"c{i}": _hit(f"d{i}", chunk=f"c{i}", content="x" * 40_000) for i in range(4)}
    assert len(jev.batch_pool(pool, query="q")) >= 2  # precondition, not luck
    res = await jev.score_pool("q", pool, api_key="k", client=_client(handler))
    assert "c0" not in res.scores and res.scores and res.partial
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
    # The end-to-end tests run as tenant "c1"; allow it to use jev, give the
    # plane a (fake) key so `jev` resolves, and NEVER let a test reach the real
    # service: the extraction shadow is stubbed out and the breaker reset.
    monkeypatch.setattr(L, "SEARCH_SELECTOR_JEV_ALLOWED", frozenset({"c1"}))
    from engine.shared.config import get_settings

    monkeypatch.setattr(get_settings(), "typesafe_api_key", "test-key-not-real")
    monkeypatch.setattr(L, "_jev_extraction_or_none", AsyncMock(return_value=None))
    jev.BREAKER.success()
    jev.EXTRACT_BREAKER.success()
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
@pytest.mark.parametrize("exc", [jev.JevError("no key"), TimeoutError(), RuntimeError("an untyped surprise")])
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
    monkeypatch.setattr(L, "SEARCH_SELECTOR_JEV_ALLOWED", frozenset({"c1"}))
    monkeypatch.setattr(L, "SEARCH_SELECTOR_JEV_CUSTOMERS", frozenset({"c1"}))
    monkeypatch.setattr(L, "SEARCH_SELECTOR_DEFAULT", "floor")
    assert L._resolve_selector(_req(), "c1") == "jev"
    assert L._resolve_selector(_req(), "c2") == "floor"
    assert L._resolve_selector(_req(selector="gatherer"), "c1") == "gatherer"


def test_jev_is_never_honoured_for_a_tenant_not_allowed(monkeypatch):
    """`jev` sends passages to an outside company: an explicit request, a
    rollout entry, or a jev default must all be refused for a tenant that is
    not on the allow-list."""
    monkeypatch.setattr(L, "SEARCH_SELECTOR_JEV_ALLOWED", frozenset({"probe"}))
    monkeypatch.setattr(L, "SEARCH_SELECTOR_JEV_CUSTOMERS", frozenset({"other"}))
    monkeypatch.setattr(L, "SEARCH_SELECTOR_DEFAULT", "floor")
    assert L._resolve_selector(_req(selector="jev"), "other") == "floor"
    assert L._resolve_selector(_req(), "other") == "floor"
    monkeypatch.setattr(L, "SEARCH_SELECTOR_DEFAULT", "jev")
    assert L._resolve_selector(_req(), "other") == "floor"
    assert L._resolve_selector(_req(), "probe") == "jev"
    monkeypatch.setattr(L, "SEARCH_SELECTOR_JEV_ALLOWED", frozenset({"*"}))
    assert L._resolve_selector(_req(), "other") == "jev"


def test_vendor_error_text_is_never_logged():
    resp = httpx.Response(422, json={"detail": {"error_type": "bad", "input": "SECRET PASSAGE"}})
    assert jev._error_type(resp) == "bad"
    assert jev._error_type(httpx.Response(500, text="SECRET PASSAGE")) == "unparseable"


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


def test_an_unsure_jev_class_keeps_the_extractors_own_filter(monkeypatch):
    # A wrong class zeroes the pool. Below the confidence floor Jev's class is
    # not applied -- and gpt-oss's own decision stands rather than being dropped.
    monkeypatch.setattr(L, "SEARCH_EXTRACTION_JEV_APPLY_RATE", 1.0)
    out, _ = L._merge_jev_extraction(_gpt(doc_types=["linear.issue"]), _jc(cconf=0.52),
                                     customer_id="c", trace_id="t")
    assert out.search_options.doc_types == ["linear.issue"]
    out, _ = L._merge_jev_extraction(_gpt(doc_types=None), _jc(cconf=0.52),
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


# =====================================================================
# review follow-ups: budgets, breaker, partial scoring, every failure branch
# =====================================================================


def _capture_state(monkeypatch):
    captured = {}
    real = L._stash_for_trace_persist

    def spy(request, **kw):
        captured["state"] = kw["state"]
        return real(request, **kw)

    monkeypatch.setattr(L, "_stash_for_trace_persist", spy)
    return captured


@pytest.mark.asyncio
async def test_a_hanging_jev_is_cut_by_its_timeout_and_the_floor_answers(monkeypatch):
    import asyncio
    import time

    async def hang(*a, **k):
        await asyncio.sleep(10)

    monkeypatch.setattr(L, "JEV_SELECTION_TIMEOUT_SECONDS", 0.05)
    fr = _state()
    t0 = time.perf_counter()
    with patch.object(jev, "score_pool", new=hang):
        resp = await L.run_gatherer(_req(selector="jev"), customer_id="c1", request=fr)
    assert time.perf_counter() - t0 < 2.0
    assert fr.state.gatherer_status == "jev_unavailable"
    assert fr.state.router_model == "recall_floor"  # the floor answered; say so
    assert len(resp.results) == 10


@pytest.mark.asyncio
async def test_partial_scoring_is_degraded_and_the_floor_fills_the_rest(monkeypatch):
    async def half(query, pool, *, api_key, client=None):
        out = jev.ScoreResult(requests=1)
        keys = list(pool)
        out.scores = {c: 0.9 for c in keys[: len(keys) // 2]}
        out.errors = ["http_503:unknown"]
        return out

    fr = _state()
    with patch.object(jev, "score_pool", new=half):
        resp = await L.run_gatherer(_req(selector="jev"), customer_id="c1", request=fr)
    assert fr.state.gatherer_status == "jev_partial"
    assert resp.degraded is True
    assert len(resp.results) == 10
    floor_filled = [r for r in resp.results
                    if any(m.channel == "recall_floor" for m in r.matched_via)]
    assert floor_filled, "the floor must fill the slots Jev could not rank"


@pytest.mark.asyncio
async def test_no_key_means_the_floor_as_a_healthy_answer(monkeypatch):
    from engine.shared.config import get_settings

    monkeypatch.setattr(get_settings(), "typesafe_api_key", "")
    fr = _state()
    resp = await L.run_gatherer(_req(selector="jev"), customer_id="c1", request=fr)
    assert fr.state.gatherer_status == "ok" and resp.degraded is False
    assert fr.state.router_model == "recall_floor"


@pytest.mark.asyncio
async def test_rescore_failure_after_a_rewrite_keeps_the_first_pass(monkeypatch):
    extra = _pf(vector=[_hit("new1")])
    monkeypatch.setattr(L, "execute_search",
                        AsyncMock(side_effect=[json.loads(json.dumps(POOL)), extra]))
    calls = {"n": 0}

    async def flaky(query, pool, *, api_key, client=None):
        calls["n"] += 1
        if calls["n"] > 1:
            raise jev.JevError("down")
        out = jev.ScoreResult(requests=1)
        out.scores = {c: 0.1 for c in pool}
        return out

    cap = _capture_state(monkeypatch)
    fr = _state()
    with patch.object(L, "acompletion", new=_llm_says("session token rotation gateway")), \
         patch.object(jev, "score_pool", new=flaky):
        resp = await L.run_gatherer(_req(selector="jev"), customer_id="c1", request=fr)
    # The first pass stands; the rewrite's new passage went unscored, so the
    # answer is honestly partial rather than `ok`.
    assert fr.state.gatherer_status == "jev_partial" and resp.results
    assert cap["state"].selection["rewrite"]["rescore_error"] == "JevError"


@pytest.mark.asyncio
async def test_a_refanout_error_is_recorded_not_raised(monkeypatch):
    monkeypatch.setattr(L, "execute_search",
                        AsyncMock(side_effect=[json.loads(json.dumps(POOL)), RuntimeError("db")]))
    cap = _capture_state(monkeypatch)
    with patch.object(L, "acompletion", new=_llm_says("session token rotation gateway")), \
         patch.object(jev, "score_pool", new=_scored({})):
        resp = await L.run_gatherer(_req(selector="jev"), customer_id="c1", request=_state())
    assert resp.results
    assert cap["state"].selection["rewrite"]["outcome"] == "refanout_error:RuntimeError"


@pytest.mark.asyncio
async def test_a_rewrite_llm_error_is_recorded_not_raised(monkeypatch):
    from engine.shared.llm import LLMError

    cap = _capture_state(monkeypatch)
    with patch.object(L, "acompletion", new=AsyncMock(side_effect=LLMError("gateway 503"))), \
         patch.object(jev, "score_pool", new=_scored({})):
        resp = await L.run_gatherer(_req(selector="jev"), customer_id="c1", request=_state())
    rec = cap["state"].selection["rewrite"]
    assert resp.results and rec["outcome"] == "no_rewrite" and "LLMError" in rec["error"]
    assert "ms" in rec


@pytest.mark.asyncio
async def test_rewrite_disabled_never_calls_the_llm_or_searches_twice(monkeypatch):
    monkeypatch.setattr(L, "SEARCH_REWRITE_ENABLED", False)
    search = AsyncMock(return_value=json.loads(json.dumps(POOL)))
    monkeypatch.setattr(L, "execute_search", search)
    with patch.object(L, "acompletion", new=AsyncMock()) as llm, \
         patch.object(jev, "score_pool", new=_scored({})):
        await L.run_gatherer(_req(selector="jev"), customer_id="c1", request=_state())
    llm.assert_not_called()
    assert search.await_count == 1


@pytest.mark.asyncio
async def test_no_rewrite_when_the_stage_budget_is_nearly_spent(monkeypatch):
    monkeypatch.setattr(L, "SEARCH_REWRITE_MIN_BUDGET_SECONDS", 10_000.0)
    cap = _capture_state(monkeypatch)
    with patch.object(L, "acompletion", new=AsyncMock()) as llm, \
         patch.object(jev, "score_pool", new=_scored({})):
        await L.run_gatherer(_req(selector="jev"), customer_id="c1", request=_state())
    llm.assert_not_called()
    assert cap["state"].selection["rewrite"]["outcome"] == "no_budget"


@pytest.mark.asyncio
async def test_non_gatherer_selectors_render_no_prompt(monkeypatch):
    built = AsyncMock()
    monkeypatch.setattr(L, "_build_user_message", built)
    cap = _capture_state(monkeypatch)
    await L.run_gatherer(_req(selector="floor"), customer_id="c1", request=_state())
    built.assert_not_called()
    assert cap["state"].messages == []


@pytest.mark.asyncio
async def test_the_extraction_shadow_never_makes_a_search_wait(monkeypatch):
    import asyncio
    import time

    async def slow(query):
        await asyncio.sleep(5)

    monkeypatch.setattr(L, "_jev_extraction_or_none", slow)
    t0 = time.perf_counter()
    with patch.object(jev, "score_pool", new=_scored({"v1#0": 0.9})):
        await L.run_gatherer(_req(selector="jev"), customer_id="c1", request=_state())
    assert time.perf_counter() - t0 < 2.0


@pytest.mark.asyncio
async def test_the_breaker_opens_after_repeated_failures_and_skips_jev(monkeypatch):
    monkeypatch.setattr(jev, "JEV_BREAKER_FAILURES", 2)
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        return httpx.Response(503, text="down")

    pool = {"c": _hit("d")}
    for _ in range(2):
        with pytest.raises(jev.JevError):
            await jev.score_pool("q", pool, api_key="k", client=_client(handler))
    n = calls["n"]
    with pytest.raises(jev.JevError, match="breaker_open"):
        await jev.score_pool("q", pool, api_key="k", client=_client(handler))
    assert calls["n"] == n  # the open breaker made no request
    jev.BREAKER.success()
    jev.EXTRACT_BREAKER.success()


@pytest.mark.asyncio
@pytest.mark.parametrize("resp", [
    lambda r: httpx.Response(200, json={"answers": {"sort": "recency", "doc_class": "no_class"}}),
    lambda r: httpx.Response(200, json={"answers": {"sort": {"choice": "recency", "confidence": "high"},
                                                     "doc_class": {"choice": "no_class"}}}),
    lambda r: httpx.Response(500, text="boom"),
])
async def test_every_extraction_failure_is_a_jev_error(resp):
    with pytest.raises(jev.JevError):
        await jev.extract_options("q", api_key="k", client=_client(resp))
    jev.BREAKER.success()
    jev.EXTRACT_BREAKER.success()


@pytest.mark.asyncio
async def test_a_malformed_200_costs_one_batch_not_the_pool(monkeypatch):
    monkeypatch.setattr(jev, "JEV_TOKEN_BUDGET", 24_000)

    def handler(req):
        qs = json.loads(req.content)["questions"]
        if "c0" in qs:
            return httpx.Response(200, json={"answers": "not a dict", "usage": {"input_tokens": "x"}})
        return _ok(qs)

    pool = {f"c{i}": _hit(f"d{i}", chunk=f"c{i}", content="x" * 40_000) for i in range(4)}
    res = await jev.score_pool("q", pool, api_key="k", client=_client(handler))
    assert res.scores and "c0" not in res.scores and "malformed_answer" in res.errors


@pytest.mark.asyncio
async def test_splitting_stops_at_the_depth_limit():
    def handler(req):
        return httpx.Response(400, json={"detail": {"error_type": "max_tokens_exceeded"}})

    with pytest.raises(jev.JevError):
        await jev.score_pool("q", {"c": _hit("d")}, api_key="k", client=_client(handler))
    pool = {f"c{i}": _hit(f"d{i}", chunk=f"c{i}") for i in range(32)}
    out = jev.ScoreResult()
    await jev._score_batch(_client(handler), "k", "q", pool, list(pool), out)
    assert out.splits <= 2 ** jev.JEV_MAX_SPLIT_DEPTH - 1
    assert out.scores == {}
    jev.BREAKER.success()
    jev.EXTRACT_BREAKER.success()


def test_hits_without_a_doc_id_are_not_scoreable():
    pf = _pf(vector=[{"chunk_id": "x#0", "content": "orphan"}, _hit("d1")])
    assert list(jev.pool_chunks(pf)) == ["d1#0"]


def test_every_doc_class_maps_to_a_real_doc_type():
    from engine.shared.constants import DocType

    known = {d.value for d in DocType} | {"codex.session", "pi.session"}
    for cls, types in jev.DOC_CLASSES.items():
        for t in types or []:
            assert t in known, (cls, t)
    assert set(jev._CLASS_QUESTION["criteria"]) == set(jev.DOC_CLASSES)


def test_the_request_literal_and_the_selector_constant_agree():
    from typing import get_args

    from engine.shared.constants import SEARCH_SELECTOR_VALUES
    from engine.shared.models import QueryRequest

    ann = QueryRequest.model_fields["selector"].annotation
    literal = next(a for a in get_args(ann) if get_args(a))
    assert set(get_args(literal)) == set(SEARCH_SELECTOR_VALUES)


@pytest.mark.asyncio
async def test_invalid_probabilities_are_not_scores():
    def handler(req):
        return httpx.Response(200, json={"answers": {
            "c0": {"noul": "NaN"},
            "c1": {"noul": 1.7}, "c2": {"noul": True}, "c3": {"noul": 0.4},
            "stranger": {"noul": 0.99},
        }})

    pool = {f"c{i}": _hit(f"d{i}", chunk=f"c{i}") for i in range(4)}
    res = await jev.score_pool("q", pool, api_key="k", client=_client(handler))
    assert res.scores == {"c3": 0.4}


@pytest.mark.asyncio
async def test_a_200_that_skips_answers_is_partial_not_ok(monkeypatch):
    async def skippy(query, pool, *, api_key, client=None):
        out = jev.ScoreResult(requests=1)
        out.scores = {c: 0.9 for c in list(pool)[:3]}  # no errors, just gaps
        return out

    fr = _state()
    with patch.object(jev, "score_pool", new=skippy):
        resp = await L.run_gatherer(_req(selector="jev"), customer_id="c1", request=fr)
    assert fr.state.gatherer_status == "jev_partial" and len(resp.results) == 10


@pytest.mark.asyncio
async def test_a_new_passage_in_a_seen_document_is_kept(monkeypatch):
    same_doc_new_chunk = _pf(vector=[_hit("v1", chunk="v1#9", content="the actual answer")])
    monkeypatch.setattr(L, "execute_search",
                        AsyncMock(side_effect=[json.loads(json.dumps(POOL)), same_doc_new_chunk]))
    with patch.object(L, "acompletion", new=_llm_says("session token rotation gateway")), \
         patch.object(jev, "score_pool", new=_scored({"v1#9": 0.95})):
        resp = await L.run_gatherer(_req(selector="jev"), customer_id="c1", request=_state())
    assert resp.results[0].doc_id == "v1"
    assert any(c.chunk_id == "v1#9" for c in resp.results[0].chunks)


@pytest.mark.asyncio
async def test_floor_answers_on_a_plane_with_no_llm_configured(monkeypatch):
    monkeypatch.setattr(L, "_no_llm_configured", lambda: True)
    fr = _state()
    resp = await L.run_gatherer(_req(selector="floor"), customer_id="c1", request=fr)
    assert len(resp.results) == 10 and fr.state.gatherer_status == "ok"


@pytest.mark.asyncio
async def test_a_slow_rewrite_keeps_the_first_pass(monkeypatch):
    import asyncio

    async def slow_search(*a, **k):
        if slow_search.calls:
            await asyncio.sleep(10)
        slow_search.calls += 1
        return json.loads(json.dumps(POOL))
    slow_search.calls = 0

    monkeypatch.setattr(L, "execute_search", slow_search)
    monkeypatch.setattr(L, "SEARCH_AGENT_LOOP_TIMEOUT_SECONDS", 30.0)
    monkeypatch.setattr(L, "SEARCH_REWRITE_MIN_BUDGET_SECONDS", 1.0)
    monkeypatch.setattr(L, "JEV_SELECTION_TIMEOUT_SECONDS", 29.8)
    cap = _capture_state(monkeypatch)
    fr = _state()
    with patch.object(L, "acompletion", new=_llm_says("session token rotation gateway")), \
         patch.object(jev, "score_pool", new=_scored({"v2#0": 0.2})):
        resp = await L.run_gatherer(_req(selector="jev"), customer_id="c1", request=fr)
    assert cap["state"].selection["rewrite"]["outcome"] == "timeout"
    assert resp.results[0].doc_id == "v2"  # the first pass stood


@pytest.mark.asyncio
async def test_partial_answers_force_the_floor_even_in_conditional_mode(monkeypatch):
    from engine.shared.config import get_settings

    monkeypatch.setattr(get_settings(), "recall_floor_conditional_enabled", True)

    async def half(query, pool, *, api_key, client=None):
        out = jev.ScoreResult(requests=1)
        out.scores = {c: 0.9 for c in list(pool)[: len(pool) // 2]}
        out.errors = ["http_503:unknown"]
        return out

    fr = _state()
    with patch.object(jev, "score_pool", new=half):
        resp = await L.run_gatherer(
            _req(selector="jev", recall_floor_mode="conditional"), customer_id="c1", request=fr
        )
    assert fr.state.gatherer_status == "jev_partial"
    assert len(resp.results) == 10


@pytest.mark.asyncio
async def test_a_selection_timeout_counts_toward_the_breaker(monkeypatch):
    import asyncio

    async def hang(*a, **k):
        await asyncio.sleep(10)

    monkeypatch.setattr(L, "JEV_SELECTION_TIMEOUT_SECONDS", 0.02)
    before = jev.BREAKER.failures
    with patch.object(jev, "score_pool", new=hang):
        await L.run_gatherer(_req(selector="jev"), customer_id="c1", request=_state())
    assert jev.BREAKER.failures == before + 1
    jev.BREAKER.success()
    jev.EXTRACT_BREAKER.success()


@pytest.mark.asyncio
async def test_each_request_keeps_the_pool_wait_bound():
    seen = {}

    def handler(req):
        seen.update(req.extensions.get("timeout") or {})
        return _ok(json.loads(req.content)["questions"])

    await jev.score_pool("q", {"c": _hit("d")}, api_key="k", client=_client(handler))
    assert seen.get("pool") == jev.JEV_POOL_WAIT_SECONDS


@pytest.mark.asyncio
async def test_partial_answers_leave_the_floor_room_inside_top_k(monkeypatch):
    async def most(query, pool, *, api_key, client=None):
        out = jev.ScoreResult(requests=1)
        keys = list(pool)
        out.scores = {c: 0.9 for c in keys[: int(len(keys) * 0.7)]}
        out.errors = ["http_503:unknown"]
        return out

    fr = _state()
    with patch.object(jev, "score_pool", new=most):
        resp = await L.run_gatherer(_req(selector="jev", top_k=5), customer_id="c1", request=fr)
    assert len(resp.results) == 5
    assert any(any(m.channel == "recall_floor" for m in r.matched_via) for r in resp.results), \
        "a partial answer must leave the floor at least one of the slots the caller will see"


@pytest.mark.asyncio
async def test_extraction_successes_do_not_mask_a_scoring_outage(monkeypatch):
    monkeypatch.setattr(jev, "JEV_BREAKER_FAILURES", 2)
    jev.BREAKER.success()
    jev.EXTRACT_BREAKER.success()
    jev.EXTRACT_BREAKER.success()
    ok_extract = lambda r: _choice_resp()  # noqa: E731
    down = lambda r: httpx.Response(503, text="down")  # noqa: E731
    for _ in range(2):
        await jev.extract_options("q", api_key="k", client=_client(ok_extract))
        with pytest.raises(jev.JevError):
            await jev.score_pool("q", {"c": _hit("d")}, api_key="k", client=_client(down))
    assert jev.BREAKER.is_open()
    assert not jev.EXTRACT_BREAKER.is_open()
    jev.BREAKER.success()
    jev.EXTRACT_BREAKER.success()


def test_the_shipped_defaults_are_the_rollout_step():
    """Pins what production runs: the floor for everyone, Jev for `probe` only.
    Read from the constants module, not the test-pinned loop attribute."""
    from engine.shared import constants as C

    assert C.SEARCH_SELECTOR_DEFAULT == "floor"
    assert C.SEARCH_SELECTOR_JEV_CUSTOMERS == frozenset({"probe"})
    assert C.SEARCH_SELECTOR_JEV_ALLOWED == frozenset({"probe"})
