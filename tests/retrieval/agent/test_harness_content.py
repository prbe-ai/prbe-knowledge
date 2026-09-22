"""The gatherer names chunks; the harness supplies their text.

Before this, `GatheredChunk.content` was required, so the model re-typed every
passage it picked -- slow (output tokens) and lossy: `_coerce_lenient` dropped a
chunk whose retype came back empty, so a correctly NAMED chunk could vanish from
the answer. The stored text now fills `content` whenever the harness retrieved
the chunk; only a chunk it never saw still needs the model's text.
"""

from __future__ import annotations

from engine.retrieval.agent.loop import LoopState, _coerce_lenient
from engine.retrieval.agent.models import GathererOutput


def _state():
    st = LoopState(customer_id="c", trace_id="t", query="q")
    st.prefanout = {"sub_queries": [{
        "vector": [{"doc_id": "slack:d1", "chunk_id": "slack:d1#0",
                    "content": "the stored passage, verbatim", "title": "d1",
                    "source_system": "slack"}],
        "bm25": [], "graph": [], "inferred_edge": [],
    }]}
    return st


def _chunks(*chunks):
    return _coerce_lenient({"chunks": list(chunks), "gatherer_notes": {}}, _state())["chunks"]


def test_a_named_pool_chunk_with_no_text_is_kept_with_the_stored_text():
    out = _chunks({"doc_id": "slack:d1", "chunk_id": "slack:d1#0"})
    assert [c["content"] for c in out] == ["the stored passage, verbatim"]


def test_a_fumbled_retype_is_replaced_by_the_stored_text():
    out = _chunks({"doc_id": "slack:d1", "chunk_id": "slack:d1#0",
                   "content": "the stord pasage [@200] verbtim"})
    assert out[0]["content"] == "the stored passage, verbatim"


def test_a_chunk_the_harness_never_saw_still_needs_its_text():
    assert _chunks({"doc_id": "slack:zz", "chunk_id": "slack:zz#0"}) == []
    out = _chunks({"doc_id": "slack:zz", "chunk_id": "slack:zz#0", "summary": "from fetch_doc"})
    assert out[0]["content"] == "from fetch_doc"


def test_the_emit_schema_no_longer_requires_content():
    schema = GathererOutput.model_json_schema()
    chunk = schema["$defs"]["GatheredChunk"]
    assert "content" not in chunk.get("required", [])
    assert {"doc_id", "chunk_id"} <= set(chunk.get("required", []))


def test_the_prompt_tells_the_model_not_to_retype():
    from datetime import UTC, datetime

    from engine.retrieval.agent.prompt import build_system_prompt

    text = build_system_prompt(datetime.now(UTC))
    assert "Do NOT" in text and "copy the passage text into `content`" in text
