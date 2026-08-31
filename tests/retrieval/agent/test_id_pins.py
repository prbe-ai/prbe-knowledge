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
    pins, unresolved = resolve_pins(_det("PRB-17"), hits, top_k=8)
    assert [p.doc_id for p in pins] == ["doc-new"]
    assert unresolved == set()


def test_unresolved_ids_reported_and_multi_entity_pins() -> None:
    hits = [_hit("PRB-17", "doc-a"), _hit("PRB-99", "doc-b")]
    pins, unresolved = resolve_pins(
        _det("PRB-17", "PRB-99", "GONE-1"), hits, top_k=8
    )
    assert [p.matched_canonical_id for p in pins] == ["PRB-17", "PRB-99"]
    assert unresolved == {"GONE-1"}


def test_cap_is_half_top_k_but_never_zero() -> None:
    hits = [_hit(f"T-{i}", f"doc-{i}") for i in range(10)]
    det = _det(*[f"T-{i}" for i in range(10)])
    pins, _ = resolve_pins(det, hits, top_k=8)
    assert len(pins) == 4  # max(1, 8 // 2)
    pins1, _ = resolve_pins(det, hits, top_k=1)
    assert len(pins1) == 1  # the outside-voice F4 integer-division case


def test_two_ids_same_doc_pin_once() -> None:
    hits = [_hit("PRB-17", "doc-x"), _hit("uuid-1", "doc-x")]
    pins, unresolved = resolve_pins(_det("PRB-17", "uuid-1"), hits, top_k=8)
    assert [p.doc_id for p in pins] == ["doc-x"]
    assert unresolved == set()


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
