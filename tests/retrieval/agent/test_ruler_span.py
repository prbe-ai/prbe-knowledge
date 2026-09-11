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
