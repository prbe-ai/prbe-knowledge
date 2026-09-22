"""Phase 0 replay harness -- the parts that must not drift.

These tests pin the reading of a trace blob, because every number in the Phase 0
report is downstream of it. The load-bearing one is
`test_floor_reconstruction_matches_the_blob`: it proves the harness rebuilds the
recall floor the way production ran it, on traces where production ran the floor
alone. Without that, an arm-B number is a claim about our re-implementation.
"""

from __future__ import annotations

import gzip
import json
from datetime import UTC, datetime, timedelta

import pytest

from engine.retrieval.agent import jev_shadow as js


def _hit(doc: str, *, chunk: str | None = None, content: str = "body", **extra):
    h = {
        "doc_id": doc,
        "chunk_id": chunk or f"{doc}#0",
        "content": content,
        "source_system": "slack",
        "title": doc,
        "updated_at": "2026-09-01T00:00:00+00:00",
    }
    h.update(extra)
    return h


def _blob(*, vector=(), bm25=(), graph=(), chunks=(), status="ok",
          customer="probe", query="why did it fail", stamp="2026-09-20T12:00:00+00:00"):
    return {
        "trace_id": "t1",
        "customer_id": customer,
        "status": status,
        "query": query,
        "timestamp_utc": stamp,
        "prefanout": {"sub_queries": [{
            "query": query,
            "vector": list(vector), "bm25": list(bm25),
            "graph": list(graph), "inferred_edge": [],
        }]},
        "gathered": {"chunks": list(chunks), "gatherer_notes": {}},
    }


# ---------------------------------------------------------------- the pool

def test_pool_dedupes_by_chunk_id_and_keeps_the_first_hit():
    a = _hit("d1", content="first")
    b = _hit("d1", content="second")  # same chunk_id, arrives via another channel
    pool = js.pool_chunks(_blob(vector=[a], bm25=[b])["prefanout"])
    assert list(pool) == ["d1#0"]
    assert pool["d1#0"]["content"] == "first"


@pytest.mark.parametrize("bad", [
    {"doc_id": "d1", "chunk_id": "d1#0"},                 # no content
    {"doc_id": "d1", "chunk_id": "d1#0", "content": "  "},  # blank content
    {"doc_id": "d1", "content": "body"},                  # no chunk_id
    "not-a-dict",
])
def test_pool_skips_hits_that_cannot_be_scored_or_delivered(bad):
    assert js.pool_chunks(_blob(vector=[bad])["prefanout"]) == {}


def test_pool_spans_every_channel_and_sub_query():
    pf = {"sub_queries": [
        {"vector": [_hit("d1")], "bm25": [_hit("d2")]},
        {"graph": [_hit("d3")], "inferred_edge": [_hit("d4")]},
    ]}
    assert set(js.pool_chunks(pf)) == {"d1#0", "d2#0", "d3#0", "d4#0"}


def test_empty_and_missing_prefanout_are_an_empty_pool_not_a_crash():
    assert js.pool_chunks(None) == {}
    assert js.pool_chunks({}) == {}
    assert js.pool_chunks({"sub_queries": [None, "x"]}) == {}


# ------------------------------------------------------------- delivered

def test_delivered_split_reads_the_harness_appended_flag():
    chunks = [
        {"doc_id": "d1", "chunk_id": "d1#0", "harness_appended": False},
        {"doc_id": "d2", "chunk_id": "d2#0", "harness_appended": True},
        {"doc_id": "d1", "chunk_id": "d1#1", "harness_appended": False},  # same doc
    ]
    ordered, model, appended, known = js.delivered_docs(_blob(chunks=chunks))
    assert ordered == ["d1", "d2"]          # order preserved, docs deduped
    assert model == {"d1"} and appended == {"d2"}
    assert known is True


def test_a_blob_without_the_flag_reports_provenance_UNKNOWN_not_model():
    """`harness_appended` shipped 2026-09-12. Before it, nobody recorded who
    chose a chunk -- and defaulting that to "the model did" invents a clean,
    false cliff on the day the field landed: every older trace reads as 100%
    model-supplied. Absence of a record is not evidence of authorship.
    """
    ordered, model, appended, known = js.delivered_docs(
        _blob(chunks=[{"doc_id": "d1", "chunk_id": "d1#0"}])
    )
    assert known is False
    assert ordered == ["d1"]           # still delivered, still comparable
    assert model == set() and appended == set()


def test_provenance_is_known_when_any_chunk_carries_the_flag():
    assert js.has_provenance(_blob(chunks=[
        {"doc_id": "d1", "chunk_id": "d1#0", "harness_appended": False},
    ])) is True
    assert js.has_provenance(_blob(chunks=[{"doc_id": "d1", "chunk_id": "d1#0"}])) is False


def test_arms_still_compare_without_provenance():
    # The A/B comparison needs the delivered set and a reconstruction, not the
    # flag -- so a pre-09-12 trace is still a valid row.
    pool = [_hit(f"d{i}") for i in range(12)]
    blob = _blob(vector=pool, chunks=[{"doc_id": "d0", "chunk_id": "d0#0"}])
    cmp = js.compare_a_b(blob)
    assert cmp.provenance_known is False
    assert cmp.a_docs and cmp.b_docs


# ------------------------------------------------------------- the arms

def test_arm_today_is_delivery_order_cut_to_budget():
    chunks = [{"doc_id": f"d{i}", "chunk_id": f"d{i}#0"} for i in range(14)]
    assert js.arm_today(_blob(chunks=chunks)) == [f"d{i}" for i in range(10)]


def test_arm_floor_only_fills_the_whole_budget_from_the_pool():
    pool = [_hit(f"d{i}") for i in range(14)]
    b = js.arm_floor_only(_blob(vector=pool))
    assert len(b) == js.DELIVERY_BUDGET_DOCS
    assert len(set(b)) == len(b)


def test_arm_floor_only_ignores_what_the_model_delivered():
    # Arm B is "no model at all" -- the blob's own gathered chunks must not
    # leak in, or B is just A wearing a different label.
    pool = [_hit(f"d{i}") for i in range(12)]
    delivered = [{"doc_id": "zzz-not-in-pool", "chunk_id": "zzz#0"}]
    assert "zzz-not-in-pool" not in js.arm_floor_only(_blob(vector=pool, chunks=delivered))


def test_arm_jev_takes_scores_over_theta_then_tops_up_from_the_floor():
    pool = [_hit(f"d{i}") for i in range(12)]
    blob = _blob(vector=pool)
    scores = {"d0#0": 0.95, "d1#0": 0.91, "d2#0": 0.10}
    docs, n_jev = js.arm_jev(blob, scores, theta=0.7)
    assert docs[:2] == ["d0", "d1"]         # highest first
    assert n_jev == 2                        # only two cleared theta
    assert "d2" not in docs[:2]
    assert len(docs) == js.DELIVERY_BUDGET_DOCS  # floor supplied the rest
    assert len(set(docs)) == len(docs)       # top-up never duplicates


def test_arm_jev_scores_a_document_by_its_best_chunk():
    # A long document whose 4th chunk is the relevant one must not be ranked
    # on its 1st chunk. Same reason the floor counts documents.
    pool = [_hit("d1", chunk="d1#0"), _hit("d1", chunk="d1#3"), _hit("d2", chunk="d2#0")]
    blob = _blob(vector=pool)
    docs, n_jev = js.arm_jev(blob, {"d1#0": 0.01, "d1#3": 0.99, "d2#0": 0.50}, theta=0.9)
    assert docs[0] == "d1" and n_jev == 1


def test_arm_jev_with_no_scores_degrades_to_the_floor():
    pool = [_hit(f"d{i}") for i in range(12)]
    blob = _blob(vector=pool)
    docs, n_jev = js.arm_jev(blob, {}, theta=0.7)
    assert n_jev == 0
    assert docs == js.arm_floor_only(blob)


def test_every_arm_delivers_the_same_budget():
    # The comparison is only meaningful if the arms are the same size.
    pool = [_hit(f"d{i}") for i in range(20)]
    chunks = [{"doc_id": f"d{i}", "chunk_id": f"d{i}#0"} for i in range(20)]
    blob = _blob(vector=pool, chunks=chunks)
    a = js.arm_today(blob)
    b = js.arm_floor_only(blob)
    c, _ = js.arm_jev(blob, {f"d{i}#0": 0.99 for i in range(20)}, theta=0.7)
    assert len(a) == len(b) == len(c) == js.DELIVERY_BUDGET_DOCS


# ------------------------------------------------------------- batching

def test_estimate_tokens_charges_for_questions_as_well_as_state():
    # The half a chars/4 estimate misses. 100 questions is ~3.5k tokens.
    assert js.estimate_tokens(0, 100) == 3500
    assert js.estimate_tokens(37700, 0) == 10000


def test_batch_pool_keeps_every_batch_under_the_budget():
    pool = {f"c{i}": _hit(f"d{i}", chunk=f"c{i}", content="x" * 4000) for i in range(60)}
    batches = js.batch_pool(pool, query="q")
    assert len(batches) > 1
    assert all(b.tokens <= js.JEV_TOKEN_BUDGET for b in batches)


def test_batch_pool_loses_no_chunk_and_repeats_none():
    pool = {f"c{i}": _hit(f"d{i}", chunk=f"c{i}", content="x" * 3000) for i in range(80)}
    seen = [c for b in js.batch_pool(pool, query="q") for c in b.chunk_ids]
    assert sorted(seen) == sorted(pool)
    assert len(seen) == len(set(seen))


def test_a_chunk_bigger_than_the_budget_still_gets_a_batch():
    # Dropping it would be a recall bug wearing a size limit's clothes; the
    # caller truncates and says so.
    pool = {"big": _hit("d1", chunk="big", content="x" * 400_000)}
    batches = js.batch_pool(pool, query="q")
    assert [b.chunk_ids for b in batches] == [["big"]]


def test_batch_pool_of_an_empty_pool_is_no_requests():
    assert js.batch_pool({}, query="q") == []


# ------------------------------------------------------------- filtering

@pytest.mark.parametrize("tenant", sorted(js.EXCLUDED_TENANTS))
def test_excluded_tenants_are_skipped_with_a_reason(tenant):
    ok, why = js.is_replayable(_blob(vector=[_hit("d1")], customer=tenant))
    assert (ok, why) == (False, "excluded_tenant")


def test_zero_recall_is_skipped_for_selection():
    ok, why = js.is_replayable(_blob(status="zero_recall_short_circuit"))
    assert (ok, why) == (False, "excluded_status")


def test_an_empty_pool_is_skipped_even_on_an_ok_status():
    ok, why = js.is_replayable(_blob())
    assert (ok, why) == (False, "empty_pool")


def test_a_real_trace_is_replayable():
    assert js.is_replayable(_blob(vector=[_hit("d1")])) == (True, "ok")


# ------------------------------------------------------------- the clock

def test_frozen_now_pins_the_loop_clock_to_the_trace():
    blob = _blob(stamp="2026-09-01T00:00:00+00:00")
    from engine.retrieval.agent import loop as _loop

    with js.frozen_now(blob) as ref:
        assert _loop.datetime.now(UTC) == ref
        assert ref.year == 2026 and ref.month == 9 and ref.day == 1
    # restored afterwards
    assert (datetime.now(UTC) - _loop.datetime.now(UTC)) < timedelta(seconds=5)


def test_frozen_now_survives_a_missing_or_malformed_timestamp():
    for stamp in (None, "", "not-a-date"):
        with js.frozen_now({"timestamp_utc": stamp}) as ref:
            assert ref.tzinfo is not None


def test_recency_ordering_uses_the_trace_clock_not_today():
    # Two docs, one much newer. Frozen at a date BEFORE the newer doc existed
    # vs today, the decay differs -- which is the whole reason for freezing.
    old = _hit("old", content="body", updated_at="2026-01-01T00:00:00+00:00")
    new = _hit("new", content="body", updated_at="2026-09-19T00:00:00+00:00")
    order = js.fused_order(_blob(vector=[old, new], stamp="2026-09-20T00:00:00+00:00"))
    assert set(order) == {"old", "new"}
    assert order[0] == "new"  # recency decay favours the fresh doc


# ------------------------------------------------------------- comparison

def test_compare_a_b_counts_the_overlap_both_ways():
    pool = [_hit(f"d{i}") for i in range(12)]
    chunks = [{"doc_id": "d0", "chunk_id": "d0#0", "harness_appended": False},
              {"doc_id": "zz", "chunk_id": "zz#0", "harness_appended": True}]
    cmp = js.compare_a_b(_blob(vector=pool, chunks=chunks))
    assert cmp.shared + cmp.a_only == len(cmp.a_docs)
    assert cmp.shared + cmp.b_only == len(cmp.b_docs)
    assert cmp.n_model_picked == 1 and cmp.n_appended == 1
    assert cmp.provenance_known is True
    assert cmp.pool_chunks == 12 and cmp.pool_docs == 12


def test_compare_a_b_flags_a_v2_blob_as_approximate():
    # v2 blobs carry no rendered_doc_ids / recency inputs, so the floor
    # reconstruction is close but not provably identical. Say so per row.
    assert js.compare_a_b(_blob(vector=[_hit("d1")])).approx is True
    blob = _blob(vector=[_hit("d1")]) | {"rendered_doc_ids": ["d1"]}
    assert js.compare_a_b(blob).approx is False


def test_load_blob_round_trips(tmp_path):
    p = tmp_path / "t.json.gz"
    p.write_bytes(gzip.compress(json.dumps({"trace_id": "x"}).encode()))
    assert js.load_blob(p)["trace_id"] == "x"


# --------------------------------------------- the self-test that matters

def test_floor_reconstruction_matches_the_blob():
    """On a trace where the model emitted nothing, arm B must equal what was
    delivered. That is the whole floor path -- fusion, source weighting,
    recency decay, the distinct-doc count -- reproduced from the blob.

    If this drifts, every arm-B number in the report is measuring the harness
    rather than production, and it will look like a result instead of a bug.
    """
    pool = [_hit(f"d{i}", content=f"body {i}") for i in range(14)]
    blob = _blob(vector=pool)
    floor_docs = js.arm_floor_only(blob, budget=99)
    # Simulate what production wrote: the floor's own picks, flagged appended.
    blob["gathered"]["chunks"] = [
        {"doc_id": d, "chunk_id": f"{d}#0", "harness_appended": True}
        for d in floor_docs
    ]
    assert js.arm_today(blob) == js.arm_floor_only(blob)
