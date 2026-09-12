"""Request-level TEMPORAL scope threads everywhere; fetch tools serve the
version the spec selects; the response gate has two failure postures (E1 /
E3 / E6 of the retrieval validity plan).

`QueryRequest.temporal` was accepted, enum-validated and never read on the
gatherer path -- every channel was hard-wired to `TemporalSpec()` and
`applied_temporal` was never set (the field was stranded when the Phase 2
gatherer cutover deleted the list pipeline that used it). `fetch_doc` and
`fetch_chunk_window` filtered only `visibility`, so for the 30 days a dead
version's chunks are retained an edited document paged out old and new text
interleaved. Pinned the way `sources` and `project_id` are: signatures and
predicates first, then live rows.
"""

from __future__ import annotations

import inspect
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from engine.retrieval.agent import adapter as adapter_mod
from engine.retrieval.agent import loop as loop_mod
from engine.retrieval.agent.adapter import (
    _enforce_scope_on_chunks,
    _scope_verdicts,
    to_query_response,
)
from engine.retrieval.agent.loop import LoopState, _backfill_recall_floor, _execute_tool_call
from engine.retrieval.agent.models import GatheredChunk, GathererNotes, GathererOutput
from engine.retrieval.agent.tools import (
    _coerce_temporal,
    execute_fetch_chunk_window,
    execute_fetch_doc,
    execute_search,
    execute_subgraph,
)
from engine.retrieval.temporal import applied_temporal_meta, live_version_join
from engine.shared import db as db_module
from engine.shared.models import QueryRequest, TemporalMode, TemporalSpec

# ============================================================
# 1. The wire
# ============================================================


def test_execute_search_and_every_content_tool_accept_temporal() -> None:
    for fn in (execute_search, execute_fetch_doc, execute_fetch_chunk_window, execute_subgraph):
        assert "temporal" in inspect.signature(fn).parameters, fn.__name__


def test_loop_state_carries_the_request_temporal_spec() -> None:
    assert "request_temporal" in LoopState.__dataclass_fields__
    assert LoopState(customer_id="c", trace_id="t", query="q").request_temporal.mode == TemporalMode.LATEST


def test_adapter_accepts_temporal_and_min_confidence() -> None:
    params = inspect.signature(to_query_response).parameters
    assert "temporal" in params and "min_confidence" in params
    assert "temporal" in inspect.signature(_scope_verdicts).parameters


def test_removed_request_knobs_are_gone() -> None:
    """`entity_match_threshold` and `requesting_user_id` fed only the deleted
    list pipeline. Sending them is ignored (extras are not forbidden across
    two independently deployed repos), never honoured."""
    req = QueryRequest(query="q", entity_match_threshold=0.5, requesting_user_id="u")
    assert not hasattr(req, "entity_match_threshold")
    assert not hasattr(req, "requesting_user_id")


def test_coerce_temporal_accepts_spec_dict_and_none() -> None:
    as_of = datetime(2026, 1, 1, tzinfo=UTC)
    spec = TemporalSpec(mode=TemporalMode.AS_OF, as_of=as_of)
    assert _coerce_temporal(None).mode == TemporalMode.LATEST
    assert _coerce_temporal(spec) is spec
    assert _coerce_temporal(spec.model_dump(mode="json")) == spec


def test_live_version_join_names_both_aliases() -> None:
    assert live_version_join("d", "c") == (
        "AND d.version BETWEEN c.first_seen_version AND c.last_seen_version"
    )


# ============================================================
# 2. The loop injects a non-default spec and strips a model-supplied one
# ============================================================


def _tool_call(name: str, arguments: dict) -> SimpleNamespace:
    return SimpleNamespace(
        id="call_1", function=SimpleNamespace(name=name, arguments=json.dumps(arguments))
    )


async def test_loop_injects_a_non_latest_spec_and_strips_the_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatch = AsyncMock(return_value={"ok": True})
    monkeypatch.setattr(loop_mod, "dispatch_tool_call", dispatch)
    as_of = datetime(2026, 1, 1, tzinfo=UTC)
    state = LoopState(
        customer_id="c1", trace_id="t", query="q",
        request_temporal=TemporalSpec(mode=TemporalMode.AS_OF, as_of=as_of),
    )
    await _execute_tool_call(
        state, _tool_call("fetch_doc", {"doc_id": "d", "temporal": {"mode": "latest"}})
    )
    sent = dispatch.await_args.kwargs["arguments"]
    assert sent["temporal"]["mode"] == "as_of"
    assert sent["temporal"]["as_of"].startswith("2026-01-01")


async def test_loop_sends_no_temporal_for_the_default_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LATEST is what every tool assumes; an always-present dict would only
    churn the trace shape."""
    dispatch = AsyncMock(return_value={"ok": True})
    monkeypatch.setattr(loop_mod, "dispatch_tool_call", dispatch)
    state = LoopState(customer_id="c1", trace_id="t", query="q")
    await _execute_tool_call(state, _tool_call("fetch_doc", {"doc_id": "d", "temporal": {"mode": "all"}}))
    assert "temporal" not in dispatch.await_args.kwargs["arguments"]


# ============================================================
# 3. The response echoes what it ran under
# ============================================================


def test_applied_temporal_meta_default_and_request() -> None:
    assert applied_temporal_meta(TemporalSpec(), source="default") == {
        "mode": "latest", "source": "default",
    }
    as_of = datetime(2026, 1, 1, tzinfo=UTC)
    meta = applied_temporal_meta(TemporalSpec(mode=TemporalMode.AS_OF, as_of=as_of), source="request")
    assert meta["mode"] == "as_of" and meta["source"] == "request"
    assert meta["as_of"].startswith("2026-01-01") and meta["time_basis"] == "source"


async def test_response_echoes_applied_temporal_and_min_confidence() -> None:
    gathered = GathererOutput(chunks=[], gatherer_notes=GathererNotes())
    resp = await to_query_response(query="q", gathered=gathered, trace_id="t", timing_ms={}, status=None)
    assert resp.applied_temporal == {"mode": "latest", "source": "default"}
    assert resp.applied_min_confidence is None
    as_of = datetime(2026, 1, 1, tzinfo=UTC)
    resp = await to_query_response(
        query="q", gathered=gathered, trace_id="t", timing_ms={}, status=None,
        temporal=TemporalSpec(mode=TemporalMode.AS_OF, as_of=as_of),
        temporal_from_request=True, min_confidence="EXTRACTED",
    )
    assert resp.applied_temporal["mode"] == "as_of" and resp.applied_temporal["source"] == "request"
    assert resp.applied_min_confidence == "EXTRACTED"


def test_doc_version_rides_the_chunk_to_the_response() -> None:
    """The adapter used to hard-code doc_version=1 on every result."""
    c = GatheredChunk(doc_id="d1", chunk_id="d1:c0", content="x", doc_version=3,
                      matched_via=["vector"], why_relevant="w")
    assert c.doc_version == 3
    with pytest.raises(ValueError):
        GatheredChunk(doc_id="d1", chunk_id="d1:c0", content="x", doc_version=0,
                      matched_via=["vector"], why_relevant="w")


async def test_response_reports_the_real_doc_version(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _all_live(customer_id, doc_ids, **kwargs):  # type: ignore[no-untyped-def]
        return {d: True for d in doc_ids}

    monkeypatch.setattr(adapter_mod, "_scope_verdicts", _all_live)
    gathered = GathererOutput(
        chunks=[GatheredChunk(doc_id="d1", chunk_id="d1:c0", content="x", doc_version=7,
                              matched_via=["vector"], why_relevant="w")],
        gatherer_notes=GathererNotes(),
    )
    resp = await to_query_response(query="q", gathered=gathered, trace_id="t", timing_ms={}, status=None)
    assert resp.results[0].doc_version == 7


# ============================================================
# 4. The recall floor marks what it appends
# ============================================================


def test_backfill_marks_every_chunk_it_appends() -> None:
    prefanout = {"sub_queries": [{
        "query": "q", "grounded_entities": [],
        "vector": [{"doc_id": f"doc:{i}", "chunk_id": f"doc:{i}:c0", "score": 0.5, "source_system": "github",
                    "title": "t", "content": "c", "updated_at": datetime.now(UTC).isoformat()}
                   for i in range(3)],
        "bm25": [], "graph": [], "inferred_edge": [],
    }]}
    gathered = GathererOutput(chunks=[], gatherer_notes=GathererNotes())
    appended = _backfill_recall_floor(gathered, prefanout)
    assert appended == 3 and len(gathered.chunks) == 3
    assert all(c.harness_appended for c in gathered.chunks)


# ============================================================
# 5. The gate: two failure postures, and AS_OF admits a retired version
# ============================================================


def _chunk(doc_id: str) -> GatheredChunk:
    return GatheredChunk(doc_id=doc_id, chunk_id=f"{doc_id}:c0", content="...",
                         matched_via=["vector"], why_relevant="t")


async def test_unscoped_gate_keeps_chunks_and_degrades_when_the_db_is_away(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _boom(*a, **k):  # type: ignore[no-untyped-def]
        raise RuntimeError("pool not initialized")

    monkeypatch.setattr(adapter_mod, "_scope_verdicts", _boom)
    gathered = GathererOutput(chunks=[_chunk("doc-a")], gatherer_notes=GathererNotes())
    ok = await _enforce_scope_on_chunks("c1", gathered, source_keys=None, doc_types=None, trace_id="t")
    assert ok is False and [c.doc_id for c in gathered.chunks] == ["doc-a"]
    resp = await to_query_response(
        query="q", gathered=gathered, trace_id="t", timing_ms={}, status="ok", customer_id="c1"
    )
    assert resp.degraded is True and resp.degraded_reason == "scope_check_unavailable"
    assert [r.doc_id for r in resp.results] == ["doc-a"]


async def test_scoped_gate_fails_when_the_db_is_away(monkeypatch: pytest.MonkeyPatch) -> None:
    """The check IS the scope: a scoped request must not answer with rows it
    could not verify."""
    async def _boom(*a, **k):  # type: ignore[no-untyped-def]
        raise RuntimeError("pool not initialized")

    monkeypatch.setattr(adapter_mod, "_scope_verdicts", _boom)
    gathered = GathererOutput(chunks=[_chunk("doc-a")], gatherer_notes=GathererNotes())
    with pytest.raises(RuntimeError):
        await _enforce_scope_on_chunks(
            "c1", gathered, source_keys=None, doc_types=None, trace_id="t", project_id="proj-a"
        )


async def test_an_already_degraded_status_is_not_overwritten_by_the_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _boom(*a, **k):  # type: ignore[no-untyped-def]
        raise RuntimeError("down")

    monkeypatch.setattr(adapter_mod, "_scope_verdicts", _boom)
    gathered = GathererOutput(chunks=[_chunk("doc-a")], gatherer_notes=GathererNotes())
    resp = await to_query_response(
        query="q", gathered=gathered, trace_id="t", timing_ms={}, status="loop_timeout", customer_id="c1"
    )
    assert resp.degraded_reason == "loop_timeout"


_INSERT_DOC = """
INSERT INTO documents (
    doc_id, version, customer_id, source_system, source_id, source_url,
    doc_class, doc_type, content_type, content_hash, title, body_preview,
    body_size_bytes, body_token_count, created_at, updated_at, valid_from, valid_to,
    ingested_at, acl, metadata
) VALUES (
    $1, $2::int, $3, 'custom_ingest', $1, '/x/' || $1,
    'raw_source', 'custom.experiment.run', 'text/plain', 'h-' || $1 || '-' || ($2::int)::text,
    $1, 'body', 4, 1, $4, $4, $4, $5, $4, '{}'::jsonb, '{}'::jsonb
)
"""
_INSERT_CHUNK = """
INSERT INTO chunks (
    chunk_id, doc_id, customer_id, chunk_index, content, content_hash, token_count, kind,
    embedding, first_seen_version, last_seen_version, valid_from, valid_to
) VALUES (
    $1, $2, $3, 0, $4, 'ch-' || $1, 3, 'content',
    array_fill(0::real, ARRAY[3072])::halfvec, $5::int, $6::int, $7, $8
)
"""


async def _seed_two_versions(customer_id: str) -> tuple[datetime, datetime]:
    """v1 valid [t0, t1), v2 valid [t1, now). Returns (t_before_edit, t1)."""
    now = datetime.now(UTC)
    t0 = now - timedelta(days=10)
    t1 = now - timedelta(days=5)
    async with db_module.raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers (customer_id, display_name, api_key_hash) "
            "VALUES ($1, $1, 'h') ON CONFLICT DO NOTHING",
            customer_id,
        )
        await conn.execute(_INSERT_DOC, "edited", 1, customer_id, t0, t1)
        await conn.execute(_INSERT_DOC, "edited", 2, customer_id, t1, None)
        await conn.execute(_INSERT_CHUNK, "edited:v1", "edited", customer_id, "old text", 1, 1, t0, t1)
        await conn.execute(_INSERT_CHUNK, "edited:v2", "edited", customer_id, "new text", 2, 2, t1, None)
    return t0 + timedelta(days=1), t1


async def test_fetch_doc_serves_the_live_version_only(live_db: None) -> None:
    """The regression: both versions' chunks came back at colliding
    chunk_index values with no marker for the model."""
    cid = "test-cust-temporal-fetch"
    await _seed_two_versions(cid)
    page = await execute_fetch_doc(cid, doc_id="edited")
    assert [c["content"] for c in page["chunks"]] == ["new text"]
    assert page["chunks"][0].get("doc_version") in (2, None)


async def test_fetch_doc_serves_the_version_valid_as_of(live_db: None) -> None:
    cid = "test-cust-temporal-fetch-asof"
    before_edit, _ = await _seed_two_versions(cid)
    spec = {"mode": "as_of", "as_of": before_edit.isoformat()}
    page = await execute_fetch_doc(cid, doc_id="edited", temporal=spec)
    assert [c["content"] for c in page["chunks"]] == ["old text"]


async def test_fetch_chunk_window_stays_within_the_live_version(live_db: None) -> None:
    cid = "test-cust-temporal-window"
    await _seed_two_versions(cid)
    win = await execute_fetch_chunk_window(cid, chunk_id="edited:v2", before=3, after=3)
    assert [c["content"] for c in win["chunks"]] == ["new text"]
    # the dead chunk is not a valid window target under LATEST
    dead = await execute_fetch_chunk_window(cid, chunk_id="edited:v1", before=3, after=3)
    assert dead["chunks"] == []


async def test_scope_verdicts_honour_as_of(live_db: None) -> None:
    """An AS_OF request correctly selects a since-retired version; the gate
    must not drop it for not being current today -- and under LATEST the
    same doc is live via v2."""
    cid = "test-cust-temporal-verdicts"
    before_edit, _ = await _seed_two_versions(cid)
    latest = await _scope_verdicts(cid, ["edited", "ghost"], source_keys=None, doc_types=None)
    assert latest == {"edited": True}
    as_of = await _scope_verdicts(
        cid, ["edited"], source_keys=None, doc_types=None,
        temporal=TemporalSpec(mode=TemporalMode.AS_OF, as_of=before_edit),
    )
    assert as_of == {"edited": True}
    long_ago = await _scope_verdicts(
        cid, ["edited"], source_keys=None, doc_types=None,
        temporal=TemporalSpec(mode=TemporalMode.AS_OF, as_of=datetime(2000, 1, 1, tzinfo=UTC)),
    )
    assert long_ago == {}  # no version existed then: unverifiable, dropped


async def test_unscoped_gate_drops_an_invented_chunk_id(live_db: None) -> None:
    """Nothing in _coerce_lenient checks emitted ids against the pool or the
    DB; the live-row gate is the only thing that does, so it runs unscoped."""
    cid = "test-cust-temporal-invented"
    await _seed_two_versions(cid)
    gathered = GathererOutput(chunks=[_chunk("edited"), _chunk("made-up-doc")], gatherer_notes=GathererNotes())
    ok = await _enforce_scope_on_chunks(cid, gathered, source_keys=None, doc_types=None, trace_id="t")
    assert ok is True and [c.doc_id for c in gathered.chunks] == ["edited"]


# ============================================================
# 6. Review fixes: parameter binding, suppressed lanes, harness-owned version
# ============================================================


async def test_changed_between_window_binds_only_what_it_references(live_db: None) -> None:
    """The regression: CHANGED_BETWEEN puts BOTH parameters in `doc_sql` and
    NONE in `chunk_sql`, so a query that bound them and used only the chunk
    half handed asyncpg two arguments nothing referenced -- every
    CHANGED_BETWEEN window fetch died on the bind."""
    cid = "test-cust-temporal-changed-between"
    _, edited_at = await _seed_two_versions(cid)
    spec = {
        "mode": "changed_between",
        "since": (edited_at - timedelta(days=1)).isoformat(),
        "until": (edited_at + timedelta(days=1)).isoformat(),
    }
    win = await execute_fetch_chunk_window(cid, chunk_id="edited:v2", before=2, after=2, temporal=spec)
    assert [c["content"] for c in win["chunks"]] == ["new text"]
    # Outside the window the document did not change: no rows, still no bind error.
    stale = {
        "mode": "changed_between",
        "since": "2000-01-01T00:00:00+00:00",
        "until": "2000-01-02T00:00:00+00:00",
    }
    assert (await execute_fetch_chunk_window(
        cid, chunk_id="edited:v2", before=2, after=2, temporal=stale
    ))["chunks"] == []


async def test_changed_between_fetch_doc_binds_cleanly(live_db: None) -> None:
    cid = "test-cust-temporal-changed-between-doc"
    _, edited_at = await _seed_two_versions(cid)
    spec = {
        "mode": "changed_between",
        "since": (edited_at - timedelta(days=1)).isoformat(),
        "until": (edited_at + timedelta(days=1)).isoformat(),
    }
    page = await execute_fetch_doc(cid, doc_id="edited", temporal=spec)
    assert [c["content"] for c in page["chunks"]] == ["new text"]


async def test_window_reports_the_version_it_read(live_db: None) -> None:
    cid = "test-cust-temporal-window-version"
    await _seed_two_versions(cid)
    win = await execute_fetch_chunk_window(cid, chunk_id="edited:v2", before=1, after=1)
    assert win["chunks"][0]["doc_version"] == 2


async def test_historical_requests_suppress_the_inferred_edge_lane() -> None:
    """inferred_edge_search reads CURRENT versions and CURRENT edges; pairing
    today's inference with an as-of chunk set would date the rationale wrong,
    so the lane is omitted and RECORDED as lost rather than silently thinned."""
    from engine.retrieval.channel_health import begin_request, lost_channels

    begin_request()
    out = await execute_search(
        "c1", queries=["q"], temporal={"mode": "as_of", "as_of": "2026-01-01T00:00:00+00:00"}
    )
    assert "inferred_edge" in lost_channels()
    assert out["sub_queries"][0]["inferred_edge"] == []


async def test_subgraph_drops_edge_enrichment_on_a_historical_spec(live_db: None) -> None:
    cid = "test-cust-temporal-subgraph"
    await _seed_two_versions(cid)
    out = await execute_subgraph(
        cid, anchor_canonical_id="edited", include_inferred=True,
        temporal={"mode": "as_of", "as_of": "2026-01-01T00:00:00+00:00"},
    )
    # The node walk still runs; only the current-only inferred edges are gone.
    assert out.get("inferred_edges", out.get("outbound_inferred_edges", [])) == []


async def test_fetch_doc_drops_edge_enrichment_on_a_historical_spec(live_db: None) -> None:
    cid = "test-cust-temporal-fetchdoc-edges"
    await _seed_two_versions(cid)
    page = await execute_fetch_doc(
        cid, doc_id="edited", with_inferred_edges=True, with_evidence=True,
        temporal={"mode": "as_of", "as_of": "2026-01-01T00:00:00+00:00"},
    )
    assert page.get("outbound_inferred_edges", []) == []
    assert page.get("evidence_by_edge_id", {}) == {}


def test_doc_version_is_harness_owned_never_the_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """A model-emitted version would be reported to the caller as the
    document's real version. The channels know it; the model does not."""
    from engine.retrieval.agent.loop import LoopState, _coerce_lenient

    state = LoopState(customer_id="c1", trace_id="t", query="q")
    state.prefanout = {"sub_queries": [{
        "query": "q", "grounded_entities": [],
        "vector": [{"doc_id": "d1", "chunk_id": "d1:c0", "content": "c", "score": 0.5,
                    "source_system": "github", "title": "t", "doc_version": 9}],
        "bm25": [], "graph": [], "inferred_edge": [],
    }]}
    out = _coerce_lenient(
        {"chunks": [{"doc_id": "d1", "chunk_id": "d1:c0", "content": "c",
                     "why_relevant": "w", "matched_via": ["vector"], "doc_version": 999}]},
        state=state,
    )
    assert out["chunks"][0]["doc_version"] == 9


def test_backfilled_chunks_carry_the_version_they_were_read_from() -> None:
    prefanout = {"sub_queries": [{
        "query": "q", "grounded_entities": [],
        "vector": [{"doc_id": "doc:1", "chunk_id": "doc:1:c0", "score": 0.5,
                    "source_system": "github", "title": "t", "content": "c",
                    "doc_version": 4, "updated_at": datetime.now(UTC).isoformat()}],
        "bm25": [], "graph": [], "inferred_edge": [],
    }]}
    gathered = GathererOutput(chunks=[], gatherer_notes=GathererNotes())
    assert _backfill_recall_floor(gathered, prefanout) == 1
    assert gathered.chunks[0].doc_version == 4


async def test_applied_temporal_source_reflects_what_the_caller_sent() -> None:
    """QueryRequest.temporal has a default factory, so a non-null value proves
    nothing: without model_fields_set every ordinary request claimed
    `source: request` and the echo said nothing at all."""
    gathered = GathererOutput(chunks=[], gatherer_notes=GathererNotes())
    resp = await to_query_response(
        query="q", gathered=gathered, trace_id="t", timing_ms={}, status=None,
        temporal=TemporalSpec(), temporal_from_request=False,
    )
    assert resp.applied_temporal["source"] == "default"
    assert "temporal" not in QueryRequest(query="q").model_fields_set
    assert "temporal" in QueryRequest(query="q", temporal={"mode": "all"}).model_fields_set


async def test_min_confidence_reaches_the_graph_channel(monkeypatch: pytest.MonkeyPatch) -> None:
    """It is the graph channel's floor, not just an evidence filter: an
    EXTRACTED request used to get inferred-only neighbours because the value
    never left the adapter."""
    from engine.retrieval.agent import tools as tools_mod

    seen: dict = {}

    async def _graph(**kwargs):  # type: ignore[no-untyped-def]
        seen.update(kwargs)
        return []

    monkeypatch.setattr(tools_mod, "_graph", _graph)
    await execute_search(
        "c1", queries=["q"], entity_ids=[{"entity_type": "issue", "canonical_id": "x"}],
        min_confidence="EXTRACTED",
    )
    assert seen.get("min_confidence") == "EXTRACTED"


def test_every_kwarg_the_loop_sends_is_one_the_callee_accepts() -> None:
    """The regression, and the blind spot that let it reach production.

    `run_gatherer` calls `execute_search` and `to_query_response` with long
    keyword lists. Every test in this suite patches `execute_search` with an
    AsyncMock, which accepts ANY keyword -- so a kwarg the real function does
    not take passes the whole suite and then 500s on the first live request.
    That is exactly what happened: `temporal_from_request` belongs to the
    response builder, an edit added it to the prefanout call too, and
    `/retrieve` answered 500 for every query until it was caught by hand.

    So this reads the SOURCE of the two call sites and checks each keyword
    against the real signature. No mock can absorb it.
    """
    import ast
    import inspect
    import pathlib

    from engine.retrieval.agent.adapter import to_query_response
    from engine.retrieval.agent.tools import execute_search

    targets = {
        "execute_search": set(inspect.signature(execute_search).parameters),
        "to_query_response": set(inspect.signature(to_query_response).parameters),
    }
    source = pathlib.Path(inspect.getsourcefile(execute_search)).parent / "loop.py"
    tree = ast.parse(source.read_text())
    checked = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if name not in targets:
            continue
        accepted = targets[name]
        for kw in node.keywords:
            if kw.arg is None:  # **kwargs splat
                continue
            assert kw.arg in accepted, (
                f"loop.py line {node.lineno}: {name}() is called with "
                f"`{kw.arg}=`, which it does not accept"
            )
            checked += 1
    assert checked > 20, f"expected to check many keywords, checked {checked}"
