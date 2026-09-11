"""The ruler, and the spans a model reads off it.

A model cannot count characters, so the harness prints the positions into the
text it shows and the model reads one off. Everything here is arithmetic:
labels at known offsets, pointers pulled into range, and the one measurement
arithmetic cannot make -- whether the chosen window is on topic -- logged
rather than asserted.
"""

from __future__ import annotations

import json

from engine.retrieval.agent.loop import _dump_prefanout, _ruled_payload, resolve_span, ruler
from engine.shared.constants import (
    SEARCH_AGENT_RULER_MIN_CHARS,
    SEARCH_AGENT_RULER_STRIDE,
    SEARCH_AGENT_SPAN_MAX_LEN,
    SEARCH_AGENT_SPAN_MIN_LEN,
)

_BODY = "Corrected current-state conclusion after deployment verification. "


def _long(chars: int = 2000) -> str:
    return (_BODY * (chars // len(_BODY) + 1))[:chars]


def test_labels_land_on_exact_multiples_of_the_stride() -> None:
    text = _long()
    out = ruler(text)
    for start in range(0, len(text), SEARCH_AGENT_RULER_STRIDE):
        assert f"[@{start}]" in out


def test_a_label_is_followed_by_the_characters_it_names() -> None:
    """THE contract: [@N] means `content[N:]`, on the stored string."""
    text = _long()
    out = ruler(text)
    for start in (0, 200, 1000):
        marker = f"[@{start}]"
        assert out[out.index(marker) + len(marker) :].startswith(text[start : start + 40])


def test_stripping_the_labels_returns_the_original_byte_for_byte() -> None:
    text = _long()
    stripped = ruler(text)
    for start in range(0, len(text), SEARCH_AGENT_RULER_STRIDE):
        stripped = stripped.replace(f"[@{start}]", "", 1)
    assert stripped == text


def test_a_short_chunk_gets_no_ruler() -> None:
    text = "x" * (SEARCH_AGENT_RULER_MIN_CHARS - 1)
    assert ruler(text) == text


def test_the_ruler_is_deterministic() -> None:
    """The cached prompt prefix depends on it: same text, same bytes."""
    text = _long()
    assert ruler(text) == ruler(text)


def test_only_chunk_bodies_are_ruled() -> None:
    payload = {
        "sub_queries": [
            {
                "vector": [{"chunk_id": "c1", "doc_id": "d", "content": _long()}],
                "graph": [{"doc_id": "d", "title": "a title, not a body"}],
            }
        ],
        "query": _long(),
    }
    out = _ruled_payload(payload)
    assert "[@200]" in out["sub_queries"][0]["vector"][0]["content"]
    assert "[@" not in out["query"], "only a chunk body is a thing to point into"
    assert out["sub_queries"][0]["graph"][0]["title"] == "a title, not a body"


def test_what_is_costed_is_what_is_sent() -> None:
    """The budget measures the serializer's output, labels included."""
    payload = {"sub_queries": [{"vector": [{"chunk_id": "c", "doc_id": "d", "content": _long()}]}]}
    dumped = _dump_prefanout(payload)
    assert "[@200]" in dumped
    assert json.loads(dumped)["sub_queries"][0]["vector"][0]["content"].startswith("[@0]")


# -- spans --------------------------------------------------------------------


def test_a_span_inside_the_content_is_kept_as_asked() -> None:
    span, outcome = resolve_span(_long(), 400, 300)
    assert span == (400, 300)
    assert outcome == "valid"


def test_a_start_past_the_end_is_clamped_not_rejected() -> None:
    text = _long(500)
    span, outcome = resolve_span(text, 9_000, 300)
    assert outcome == "clamped"
    assert span is not None and span[0] < len(text)


def test_a_length_past_the_end_is_trimmed_to_what_exists() -> None:
    text = _long(500)
    span, outcome = resolve_span(text, 400, 800)
    assert span == (400, 100)
    assert outcome == "clamped"


def test_a_tiny_length_is_raised_to_the_floor() -> None:
    span, _ = resolve_span(_long(), 0, 5)
    assert span == (0, SEARCH_AGENT_SPAN_MIN_LEN)


def test_a_huge_length_is_capped() -> None:
    span, _ = resolve_span(_long(5000), 0, 99_999)
    assert span == (0, SEARCH_AGENT_SPAN_MAX_LEN)


def test_a_missing_or_malformed_pointer_is_no_pointer() -> None:
    for start, length in ((None, 200), (10, None), ("10", 200), (True, 200), (10, False)):
        span, outcome = resolve_span(_long(), start, length)
        assert span is None and outcome == "missing"


def test_an_empty_chunk_has_nothing_to_point_at() -> None:
    assert resolve_span("", 0, 200) == (None, "missing")


def test_the_resolved_span_indexes_the_stored_string() -> None:
    """What the harness returns must be usable as `content[start:start+len]`."""
    text = _long()
    span, _ = resolve_span(text, 640, 200)
    assert span is not None
    start, length = span
    assert text[start : start + length]
    assert len(text[start : start + length]) == length


# -- the span must index the text the ruler labelled, not the model's retype ---


def _loop_state(prefanout: dict) -> object:
    from engine.retrieval.agent.loop import LoopState

    state = LoopState(customer_id="t", trace_id="q-1", query="deployment verification")
    state.prefanout = prefanout
    return state


def _gathered(chunk_id: str, content: str, start: int | None, length: int | None):
    from engine.retrieval.agent.models import GatheredChunk, GathererOutput

    return GathererOutput(
        entities=[],
        chunks=[
            GatheredChunk(
                doc_id="d1",
                chunk_id=chunk_id,
                content=content,
                start=start,
                len=length,
            )
        ],
    )


def _prefanout(chunk_id: str, content: str) -> dict:
    return {"sub_queries": [{"vector": [{"chunk_id": chunk_id, "doc_id": "d1", "content": content}]}]}


def test_a_pointed_chunk_is_restored_to_the_text_its_offsets_describe() -> None:
    """THE bug this guards: the model's `content` is not what it read.

    Verbatim only 12% of the time, an excerpt 59%, reworded 29% -- so a span
    read off the ruler and applied to the model's own rendering lands nowhere
    near the sentence it meant, and against a short paraphrase it clamps to a
    one-character window.
    """
    from engine.retrieval.agent.loop import _resolve_spans

    stored = _long(2000)
    gathered = _gathered("c1", "a short paraphrase of the chunk", 1200, 300)
    _resolve_spans(gathered, _loop_state(_prefanout("c1", stored)), query="deployment")
    chunk = gathered.chunks[0]
    assert chunk.content == stored, "restored to the string the ruler labelled"
    assert (chunk.start, chunk.len) == (1200, 300)
    assert len(chunk.content[chunk.start : chunk.start + chunk.len]) == 300


def test_a_chunk_whose_stored_text_is_gone_keeps_its_own_and_loses_the_span() -> None:
    """A pointer into a string it does not describe is worse than no pointer."""
    from engine.retrieval.agent.loop import _resolve_spans

    gathered = _gathered("c-unknown", "the model's own text", 1200, 300)
    _resolve_spans(gathered, _loop_state(_prefanout("c1", _long())), query="deployment")
    chunk = gathered.chunks[0]
    assert chunk.content == "the model's own text"
    assert chunk.start is None and chunk.len is None


def test_an_unpointed_chunk_is_left_entirely_alone() -> None:
    """Content fidelity is a separate change; this one only touches pointers."""
    from engine.retrieval.agent.loop import _resolve_spans

    gathered = _gathered("c1", "the model's own text", None, None)
    _resolve_spans(gathered, _loop_state(_prefanout("c1", _long())), query="deployment")
    assert gathered.chunks[0].content == "the model's own text"


def test_no_chunk_is_ever_dropped() -> None:
    """#370 removed chunks whose lookup missed and search went empty (#371)."""
    from engine.retrieval.agent.loop import _resolve_spans

    gathered = _gathered("c-unknown", "kept", 10, 200)
    _resolve_spans(gathered, _loop_state({}), query="x")
    assert len(gathered.chunks) == 1


# -- the plumbing must never reach a reader ------------------------------------


def test_a_copied_label_is_taken_back_out_of_emitted_content() -> None:
    """The prompt says never to copy a label. A prompt is advice.

    If one is copied anyway it lands as `[@200]` litter in a user-visible search
    preview -- the one place this feature must not show its own plumbing.
    """
    from engine.retrieval.agent.loop import _parse_terminal_args

    out = _parse_terminal_args(
        {
            "entities": [],
            "chunks": [
                {
                    "doc_id": "d1",
                    "chunk_id": "c1",
                    "content": "[@0]Retain the deployed 0.5 MPP[@200] preprocessing.",
                }
            ],
        }
    )
    assert out is not None
    assert out.chunks[0].content == "Retain the deployed 0.5 MPP preprocessing."


def test_stripping_leaves_real_text_alone() -> None:
    """`[@2]` is a pandoc citation, not our plumbing.

    The bracket-at-number shape alone is not proof the harness wrote it, so a
    label at a multiple of the stride has to be present before anything is
    removed -- otherwise a search preview quietly loses a real citation.
    """
    from engine.retrieval.agent.loop import strip_ruler

    for text in ("array[@2] is not a label", "see [@17] and [@33]", "a@b.com", "", "prose"):
        assert strip_ruler(text) == text


def test_stripping_fires_once_a_real_label_proves_the_ruler() -> None:
    from engine.retrieval.agent.loop import strip_ruler

    # [@200] is ours; [@17] beside it goes too, because this text is ruled.
    assert strip_ruler("[@200]a [@17]b") == "a b"
