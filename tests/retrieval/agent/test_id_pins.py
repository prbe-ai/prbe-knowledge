"""The id-pins lane: pin arithmetic, adapter survival, pure-lookup gate.

Three layers, matching where each guarantee lives:
  * resolve_pins — one slot per identifier, best doc, cap arithmetic
    (incl. the top_k=1 case the outside voice flagged as a zero-slot bug).
  * adapter — a pinned doc SURVIVES the model dropping it, dedupes when
    kept, and the explicit top_k budget holds with pins first.
  * loop — the pure-lookup gate skips BOTH LLM calls only when every
    detected id resolved AND the residual is empty; one unresolved id
    keeps the full loop.
"""

from __future__ import annotations

from datetime import UTC, datetime

from engine.retrieval.agent.adapter import to_query_response
from engine.retrieval.agent.models import (
    GatheredChunk,
    GathererNotes,
    GathererOutput,
    is_degraded,
)
from engine.retrieval.retrievers.id_lookup import IdLookupHit, resolve_pins
from engine.shared.identifiers import DetectedIdentifier
from engine.shared.models import QueryDocumentResult

_TS = datetime(2026, 8, 31, tzinfo=UTC)


def _hit(cid: str, doc_id: str, ts: datetime = _TS) -> IdLookupHit:
    return IdLookupHit(
        chunk_id=f"ck-{doc_id}",
        doc_id=doc_id,
        doc_version=1,
        source_system="linear",
        source_url=f"https://x/{cid}",
        title=f"title {cid}",
        content=f"content for {cid}",
        created_at=ts,
        updated_at=ts,
        score=1.0,
        matched_canonical_id=cid,
    )


def _det(*cids: str) -> list[DetectedIdentifier]:
    return [DetectedIdentifier(kind="ticket", canonical_id=c) for c in cids]


# ---- resolve_pins -----------------------------------------------------------


def test_one_pin_per_identifier_best_doc_first() -> None:
    newer = datetime(2026, 8, 30, tzinfo=UTC)
    older = datetime(2026, 8, 1, tzinfo=UTC)
    hits = sorted(
        [_hit("PRB-17", "doc-old", older), _hit("PRB-17", "doc-new", newer)],
        key=lambda h: (h.matched_canonical_id, -h.updated_at.timestamp(), h.doc_id),
    )
    pins, unresolved, overflow = resolve_pins(_det("PRB-17"), hits, top_k=8)
    assert [p.doc_id for p in pins] == ["doc-new"]
    assert unresolved == set()
    assert overflow == set()


def test_unresolved_ids_reported_and_multi_entity_pins() -> None:
    hits = [_hit("PRB-17", "doc-a"), _hit("PRB-99", "doc-b")]
    pins, unresolved, overflow = resolve_pins(
        _det("PRB-17", "PRB-99", "GONE-1"), hits, top_k=8
    )
    assert [p.matched_canonical_id for p in pins] == ["PRB-17", "PRB-99"]
    assert unresolved == {"GONE-1"}
    assert overflow == set()


def test_cap_is_half_top_k_but_never_zero() -> None:
    hits = [_hit(f"T-{i}", f"doc-{i}") for i in range(10)]
    det = _det(*[f"T-{i}" for i in range(10)])
    pins, _, overflow = resolve_pins(det, hits, top_k=8)
    assert len(pins) == 4  # max(1, 8 // 2)
    # Review F5: ids that resolved but lost their slot to the cap are a
    # third category — reported, so the caller can block the fast path.
    assert overflow == {f"T-{i}" for i in range(4, 10)}
    pins1, _, overflow1 = resolve_pins(det, hits, top_k=1)
    assert len(pins1) == 1  # the outside-voice F4 integer-division case
    assert overflow1 == {f"T-{i}" for i in range(1, 10)}


def test_two_ids_same_doc_pin_once() -> None:
    hits = [_hit("PRB-17", "doc-x"), _hit("uuid-1", "doc-x")]
    pins, unresolved, overflow = resolve_pins(
        _det("PRB-17", "uuid-1"), hits, top_k=8
    )
    assert [p.doc_id for p in pins] == ["doc-x"]
    assert unresolved == set()
    # Represented-by-the-same-doc is NOT overflow: both ids are in the answer.
    assert overflow == set()


# ---- adapter guarantee ------------------------------------------------------


def _gathered(*doc_ids: str) -> GathererOutput:
    return GathererOutput(
        chunks=[
            GatheredChunk(
                doc_id=d,
                chunk_id=f"ck-{d}",
                content=f"body {d}",
                source_system="github",
            )
            for d in doc_ids
        ],
        entities=[],
        gatherer_notes=GathererNotes(confidence="high"),
    )


def _doc_ids(resp) -> list[str]:
    return [
        r.doc_id for r in resp.results if isinstance(r, QueryDocumentResult)
    ]


async def test_model_dropped_pin_is_reinserted_first() -> None:
    """THE critical regression class: the 21% nondeterministic selector
    drops the exact match; the adapter must make that impossible."""
    resp = await to_query_response(
        query="q",
        gathered=_gathered("doc-model-kept"),
        trace_id="t",
        timing_ms={},
        status="ok",
        id_pins=[_hit("PRB-17", "doc-pinned")],
        top_k=8,
    )
    ids = _doc_ids(resp)
    assert ids[0] == "doc-pinned"
    assert "doc-model-kept" in ids


async def test_kept_pin_dedupes_to_front_with_id_lookup_provenance() -> None:
    resp = await to_query_response(
        query="q",
        gathered=_gathered("doc-pinned", "doc-other"),
        trace_id="t",
        timing_ms={},
        status="ok",
        id_pins=[_hit("PRB-17", "doc-pinned")],
        top_k=8,
    )
    ids = _doc_ids(resp)
    assert ids[0] == "doc-pinned"
    assert ids.count("doc-pinned") == 1
    pinned = next(
        r
        for r in resp.results
        if isinstance(r, QueryDocumentResult) and r.doc_id == "doc-pinned"
    )
    assert "id_lookup" in [m.channel for m in pinned.matched_via]
    # Review F3: the model's curated chunk answers the query; the pin must
    # move the doc to the front WITHOUT replacing that chunk with the
    # lookup's header chunk.
    assert pinned.chunks[0].content == "body doc-pinned"


async def test_pin_merge_does_not_mutate_the_stashed_gathered() -> None:
    """Review F7: the loop stashes the GathererOutput by reference for the
    post-flush trace persist; the pin merge must land on a copy so traces
    keep the gatherer's true output."""
    gathered = _gathered("doc-model-kept")
    before = [(c.doc_id, tuple(c.matched_via)) for c in gathered.chunks]
    await to_query_response(
        query="q",
        gathered=gathered,
        trace_id="t",
        timing_ms={},
        status="ok",
        id_pins=[_hit("PRB-17", "doc-pinned")],
        top_k=8,
    )
    after = [(c.doc_id, tuple(c.matched_via)) for c in gathered.chunks]
    assert after == before


async def test_top_k_slice_keeps_ranks_contiguous() -> None:
    """Review F8: the budget applies before rank assignment, so surviving
    docs are ranked 1..top_k and entity rows continue the sequence with no
    hole."""
    from engine.retrieval.agent.models import GatheredEntity

    gathered = _gathered("a", "b", "c", "d", "e")
    gathered.entities = [GatheredEntity(canonical_id="ent-1", label="Person")]
    resp = await to_query_response(
        query="q",
        gathered=gathered,
        trace_id="t",
        timing_ms={},
        status="ok",
        top_k=2,
    )
    ranks = [r.rank for r in resp.results]
    assert ranks == [1, 2, 3]  # 2 docs + 1 entity, contiguous


async def test_top_k_budget_holds_with_pins_first() -> None:
    resp = await to_query_response(
        query="q",
        gathered=_gathered("a", "b", "c", "d"),
        trace_id="t",
        timing_ms={},
        status="ok",
        id_pins=[_hit("PRB-17", "doc-pinned")],
        top_k=3,
    )
    ids = _doc_ids(resp)
    assert len(ids) == 3
    assert ids[0] == "doc-pinned"


async def test_no_pins_no_top_k_is_byte_identical_behavior() -> None:
    resp = await to_query_response(
        query="q",
        gathered=_gathered("a", "b"),
        trace_id="t",
        timing_ms={},
        status="ok",
    )
    assert _doc_ids(resp) == ["a", "b"]


# ---- status semantics -------------------------------------------------------


def test_id_lookup_short_circuit_is_not_degraded() -> None:
    assert not is_degraded("id_lookup_short_circuit")
    assert is_degraded("loop_timeout")


async def test_inferred_pin_carries_resolution_note_not_exact_claim() -> None:
    """Review: an expansion the user never typed must not be presented as
    an exact match."""
    hit = _hit("ce09c43", "doc-commit")
    hit.resolution_note = "Uniquely resolved partial identifier: ce09c43"
    resp = await to_query_response(
        query="q",
        gathered=_gathered(),
        trace_id="t",
        timing_ms={},
        status="ok",
        id_pins=[hit],
        top_k=8,
    )
    pinned = next(
        r
        for r in resp.results
        if isinstance(r, QueryDocumentResult) and r.doc_id == "doc-commit"
    )
    why = pinned.chunks[0].why_relevant
    assert "Exact identifier match" not in why
    assert "Uniquely resolved" in why
