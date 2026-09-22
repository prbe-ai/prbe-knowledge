"""Trace blob v3 -- the three fields an offline study could not ask v2 for.

Each test here exists because its absence cost a real measurement:

  * the emit arguments were never recorded, so "the model named nothing" and
    "the parser dropped what it named" were indistinguishable in 3,048 traces;
  * `query_traces.grounding_bundle` was NULL on every gatherer row, so the
    candidate set the extractor chose from was unrecoverable;
  * the recall floor decays by age with a PER-SOURCE half-life, so a replay
    run later reorders the pool unless the blob carries the clock inputs.
"""

from __future__ import annotations

import json

from engine.retrieval.agent.loop import LoopState, _parse_terminal_args
from engine.retrieval.agent.trace_blob import (
    TRACE_BLOB_SCHEMA_VERSION,
    build_trace_blob,
)
from engine.retrieval.grounding import (
    GroundingBundle,
    GroundingCandidate,
    bundle_to_jsonable,
)


def _state(**kw) -> LoopState:
    return LoopState(customer_id="c", trace_id="t", query="q", **kw)


def _emit_args(**over) -> str:
    args = {
        "chunks": [{"doc_id": "slack:d1", "chunk_id": "slack:d1#0", "content": "body"}],
        "gatherer_notes": {"confidence": "high"},
    }
    args.update(over)
    return json.dumps(args)


# ------------------------------------------------------- terminal_raw

def test_the_raw_emit_is_captured_on_the_normal_path():
    st = _state()
    _parse_terminal_args(_emit_args(), st)
    assert st.terminal_raw is not None
    assert "slack:d1" in st.terminal_raw


def test_the_raw_emit_is_captured_even_when_the_parse_drops_every_chunk():
    """The whole point. A chunk with neither doc_id nor chunk_id is discarded by
    `_coerce_lenient`, so `gathered` records nothing -- and without the raw
    string that is indistinguishable from a model that emitted nothing at all.
    """
    st = _state()
    out = _parse_terminal_args(
        json.dumps({"chunks": [{"content": "a passage with no citation"}]}), st
    )
    assert out is not None and out.chunks == []      # the parse dropped it
    assert "a passage with no citation" in (st.terminal_raw or "")  # the blob keeps it


def test_the_raw_emit_is_captured_when_the_json_is_unparseable():
    st = _state()
    assert _parse_terminal_args('{"chunks": [{"doc_id": "d1"', st) is None
    assert st.terminal_raw.startswith('{"chunks"')


def test_a_dict_argument_is_captured_too():
    # Some providers hand back an already-decoded object rather than a string.
    st = _state()
    _parse_terminal_args({"chunks": [], "gatherer_notes": {}}, st)
    assert st.terminal_raw and "chunks" in st.terminal_raw


def test_a_runaway_emit_is_capped_not_stored_whole():
    st = _state()
    _parse_terminal_args(json.dumps({"chunks": [], "pad": "x" * 500_000}), st)
    assert len(st.terminal_raw) == 262_144


def test_capture_is_optional_so_callers_without_state_still_parse():
    assert _parse_terminal_args(_emit_args(), None) is not None


# --------------------------------------------------------- the blob

def test_v3_carries_the_capture_fields():
    bundle = GroundingBundle(
        candidates=[GroundingCandidate(
            entity_type="repo", canonical_id="prbe-knowledge",
            display_name="prbe knowledge", match_source="trgm", last_seen_at=None,
        )],
    )
    st = _state(
        terminal_raw='{"chunks":[]}',
        grounding_json=bundle_to_jsonable(bundle),
        extraction_json={"entities": [], "search_options": {"sort": "recency"}},
        request_recency_half_life_days=14.0,
    )
    st.rendered_doc_ids = {"slack:d2", "slack:d1"}
    blob = build_trace_blob(
        state=st, gathered=None, status="ok", timing={}, query="q",
        customer_id="c", trace_id="t", model="m",
    )
    assert blob["schema_version"] == 3
    assert blob["terminal_raw"] == '{"chunks":[]}'
    assert blob["grounding"]["candidates"][0]["canonical_id"] == "prbe-knowledge"
    assert blob["extraction"]["search_options"]["sort"] == "recency"
    # Sorted, so two runs of the same trace produce byte-identical blobs.
    assert blob["rendered_doc_ids"] == ["slack:d1", "slack:d2"]
    assert blob["request_recency_half_life_days"] == 14.0
    assert blob["request_recall_floor_mode"] == "always"


def test_the_pre_loop_failure_path_still_has_every_v3_key():
    # The nightly analyzer keys off a stable schema; a missing key on the
    # failure path is how a digest starts throwing on exactly the traces that
    # failed.
    blob = build_trace_blob(
        state=None, gathered=None, status=None, timing={}, query="q",
        customer_id="c", trace_id="t", model="m",
    )
    for k in ("terminal_raw", "grounding", "extraction", "rendered_doc_ids",
              "request_recency_half_life_days", "request_recall_floor_mode"):
        assert k in blob, k


def test_the_blob_is_json_serializable_with_capture_present():
    st = _state(terminal_raw="{}", grounding_json={"candidates": []},
                extraction_json={"entities": []})
    blob = build_trace_blob(state=st, gathered=None, status="ok", timing={},
                            query="q", customer_id="c", trace_id="t", model="m")
    json.dumps(blob)  # raises if anything non-serializable crept in


def test_version_was_bumped_with_the_shape():
    assert TRACE_BLOB_SCHEMA_VERSION == 3


# ------------------------------------------- the shared bundle serializer

def test_pipeline_and_the_gatherer_share_one_serializer():
    """`pipeline` imports `agent.loop`, so the gatherer cannot import back from
    it -- which is why this helper lives in `grounding.py`. A second copy would
    drift, and the query_traces middleware writes this shape into JSONB.
    """
    from engine.retrieval import pipeline

    assert pipeline._bundle_to_jsonable is bundle_to_jsonable


def test_the_serialized_bundle_keeps_its_stored_shape():
    b = GroundingBundle(
        candidates=[GroundingCandidate(
            entity_type="person", canonical_id="mahit", display_name="Mahit",
            match_source="tsvector", last_seen_at=None,
        )],
        connected_sources=["slack", "github"],
    )
    out = bundle_to_jsonable(b)
    assert set(out) == {"candidates", "bare_id_matches", "connected_sources", "timing_ms"}
    assert out["candidates"][0]["match_source"] == "tsvector"
    assert out["candidates"][0]["last_seen_at"] is None
    json.dumps(out)
