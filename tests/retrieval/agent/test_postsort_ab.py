"""Pins the post-sort A/B harness: the ported research-os transforms, both arms,
the metrics, and the parity replay against a synthetic trace."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "postsort_ab", Path(__file__).resolve().parents[3] / "scripts" / "jev_shadow" / "postsort_ab.py"
)
ab = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(ab)

CUST = "probe"
RUN = "custom_ingest:probe:experiments:run:1111"
FILE = "custom_ingest:probe:artifacts:file:2222"
GH = "github:prbe-ai/miles:commit:abc"
T1 = "claude_code:probe:sess-1"
T2 = "codex:probe:sess-2"
T3 = "pi:probe:sess-3"
D1 = "custom_ingest:probe:session_digests:session_digest:claude_code:sess-1"


def _sources(docs):
    return {d: ab.source_of(d) for d in docs}


def test_identity_parsers_match_research_os():
    assert ab.transcript_identity(T1, CUST) == ("claude_code", "sess-1")
    assert ab.transcript_identity(T2, CUST) == ("codex", "sess-2")
    assert ab.transcript_identity("claude_code:other:sess-1", CUST) is None
    assert ab.transcript_identity("claude_code:sess-1", CUST) is None
    assert ab.digest_identity(D1, CUST) == ("claude_code", "sess-1")
    assert ab.digest_identity(RUN, CUST) is None
    assert ab.session_behind(D1, CUST) == ab.session_behind(T1, CUST)
    assert ab.session_behind(GH, CUST) is None


def test_partition_is_a_stable_split_not_a_slide():
    engine = [T1, T2, GH, RUN, T3, FILE]
    assert ab.partition(engine, _sources(engine)) == [GH, RUN, FILE, T1, T2, T3]
    # no transcripts, or only transcripts: identity
    assert ab.partition([GH, RUN], _sources([GH, RUN])) == [GH, RUN]
    assert ab.partition([T1, T2], _sources([T1, T2])) == [T1, T2]


def test_source_prefers_hit_source_system_then_prefix():
    assert ab.source_of(T1, {"source_system": "claude_code"}) == "claude_code"
    assert ab.source_of(T1, {"source_system": None}) == "claude_code"
    assert ab.is_transcript(GH) is False
    assert ab.is_transcript(D1) is False  # a digest is custom_ingest: not demoted


def test_dedupe_swaps_transcript_into_digest_slot():
    # digest first on score, its transcript later: transcript takes the digest's slot
    assert ab.dedupe_by_session([D1, RUN, T1], CUST) == [T1, RUN]
    # transcript first: digest dropped
    assert ab.dedupe_by_session([T1, RUN, D1], CUST) == [T1, RUN]
    # two different sessions both stay
    assert ab.dedupe_by_session([T1, T2], CUST) == [T1, T2]


def test_arms_differ_only_when_both_kinds_present():
    engine = [T1, GH, T2, RUN]
    src = _sources(engine)
    assert ab.arm_a(engine, src, CUST, 10) == [GH, RUN, T1, T2]
    assert ab.arm_b(engine, CUST, 10) == [T1, GH, T2, RUN]
    assert ab.arm_a(engine, src, CUST, 2) == [GH, RUN]
    assert ab.arm_b(engine, CUST, 2) == [T1, GH]
    # dedupe runs AFTER the partition in arm A, as in production
    engine2 = [D1, GH, T1]
    assert ab.arm_a(engine2, _sources(engine2), CUST, 10) == [T1, GH]
    assert ab.arm_b(engine2, CUST, 10) == [T1, GH]


def test_make_record_flags_affected_and_keeps_bodies():
    bodies = {d: {"title": "t", "content": "body " * 400, "source_system": ab.source_of(d)} for d in (T1, GH)}
    rec = ab.make_record(trace_id="q", origin="live", customer_id=CUST, day="2026-09-23", query="x",
                         engine_docs=[T1, GH], bodies=bodies)
    assert rec["affected"] is True
    assert rec["arms"]["A"]["10"] == [GH, T1] and rec["arms"]["B"]["10"] == [T1, GH]
    assert len(rec["bodies"][T1]) <= ab.BODY_LIMIT + 10
    rec2 = ab.make_record(trace_id="q2", origin="live", customer_id=CUST, day=None, query="x",
                          engine_docs=[GH], bodies=bodies)
    assert rec2["affected"] is False


def test_metrics():
    rel = {"a": 1, "b": 0, "c": 1}
    assert ab.prec_at(["a", "b", "c"], rel, 2) == 0.5
    assert ab.mrr(["b", "a", "c"], rel) == 0.5
    assert ab.mrr(["b"], {"b": 0}) == 0.0
    assert math.isclose(ab.ndcg_at(["a", "c", "b"], rel, 3, ["a", "b", "c"]), 1.0)
    worse = ab.ndcg_at(["b", "a", "c"], rel, 3, ["a", "b", "c"])
    assert worse is not None and worse < 1.0
    assert ab.ndcg_at(["b"], {"b": 0}, 1, ["b"]) is None
    w, l, t, p1, p2 = ab.sign_test([1, 1, 1, 0, -1])
    assert (w, l, t) == (3, 1, 1) and 0 < p1 < 1 and p2 <= 1
    raw, kp = ab.kappa([(True, True), (False, False), (True, False), (False, False)])
    assert raw == 0.75


def test_parity_replay_on_a_synthetic_live_blob():
    blob = {
        "customer_id": CUST,
        "selector": "jev",
        "status": "ok",
        "query": "q",
        "selection": {"ranked": [[T1, 0.9], [GH, 0.5], [RUN, 0.4]]},
        "gathered": {"chunks": [
            {"doc_id": T1, "chunk_id": T1 + ":c0"},
            {"doc_id": GH, "chunk_id": GH},
            {"doc_id": RUN, "chunk_id": RUN + ":c0"},
        ]},
        "prefanout": {"sub_queries": [{"vector": [
            {"chunk_id": T1 + ":c0", "doc_id": T1, "content": "x", "source_system": "claude_code"},
            {"chunk_id": GH, "doc_id": GH, "content": "y", "source_system": "github"},
            {"chunk_id": RUN + ":c0", "doc_id": RUN, "content": "z", "source_system": "custom_ingest"},
        ]}]},
    }
    eng = ab.engine_list(blob)
    assert eng == [T1, GH, RUN] == ab.jev_ranked(blob)
    bodies = ab.doc_bodies(blob)
    sources = {d: ab.source_of(d, bodies[d]) for d in eng}
    assert ab.arm_a(eng, sources, CUST, 20) == [GH, RUN, T1]  # what /v1/search delivers today
    assert ab.arm_a(eng, sources, CUST, 2) == [GH, RUN]  # MCP-style cut drops Jev's #1


def test_parse_takes_the_final_yes_no_token():
    assert ab._parse_yes_no("YES") is True
    assert ab._parse_yes_no("No.") is False
    assert ab._parse_yes_no("The query asks whether X. The document covers it, so YES") is True
    assert ab._parse_yes_no("**NO** -- it is a Nostradamus quote") is False  # 'Nostradamus' is not a token
    assert ab._parse_yes_no("It does not say.") is None
    assert ab._parse_yes_no("") is None


def test_kind_and_tier_classification():
    assert ab.kind_of(RUN) == "run" and ab.tier_of(RUN) == 0
    assert ab.kind_of("custom_ingest:probe:experiments:paper:1") == "paper"
    assert ab.kind_of(FILE) == "file" and ab.tier_of(FILE) == 2
    assert ab.kind_of(GH) == "gh_commit" and ab.tier_of(GH) == 2
    assert ab.tier_of("github:o/r:pull_request:12") == 1 and ab.tier_of("github:o/r:pr:12") == 1
    assert ab.kind_of(T1) == "transcript" and ab.tier_of(T1) == 3
    assert ab.kind_of(D1) == "digest" and ab.tier_of(D1) == 3
    assert ab.tier_of("code_graph:x:y") == 2


def test_arm_c_penalty_only_decides_near_ties():
    engine = [T1, GH, RUN]
    p = {T1: 0.91, GH: 0.55, RUN: 0.53}
    # strong Jev judgment survives; commit vs run near-tie flips to the run
    assert ab.arm_c(engine, p, [0, 0.02, 0.05, 0.08], CUST, 10) == [T1, RUN, GH]
    # zero penalties == engine order
    assert ab.arm_c(engine, p, [0, 0, 0, 0], CUST, 10) == engine
    # a transcript barely ahead of a run drops behind it
    assert ab.arm_c([T1, RUN], {T1: 0.60, RUN: 0.58}, [0, 0.02, 0.05, 0.08], CUST, 10) == [RUN, T1]
