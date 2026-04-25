"""Unit tests for the synthesis layer — provider mocked, no network."""

from __future__ import annotations

import pytest

from services.retrieval.synthesis import (
    SynthesisChunk,
    SynthesisError,
    _extract_citations,
    _format_user_prompt,
    _parse_response,
    normalize_citations_in_answer,
    synthesize,
)

_NOW_ISO = "2026-04-24T00:00:00+00:00"


def _chunk(idx: int, content: str = "x", title: str | None = None) -> SynthesisChunk:
    return SynthesisChunk(
        chunk_id=f"chunk-{idx}",
        title=title,
        content=content,
        source_system="slack",
        source_url=f"https://example.com/{idx}",
        updated_at=_NOW_ISO,
    )


def test_format_user_prompt_indexes_chunks() -> None:
    chunks = [_chunk(1, content="alpha", title="T1"), _chunk(2, content="beta")]
    out = _format_user_prompt("what's up?", chunks)
    assert "[chunk:1]" in out
    assert "[chunk:2]" in out
    assert "T1" in out
    assert "alpha" in out and "beta" in out
    assert "what's up?" in out


def test_format_user_prompt_truncates_long_chunks() -> None:
    long = "a" * 4000
    out = _format_user_prompt("q", [_chunk(1, content=long)])
    assert "…" in out
    assert len(out) < 4000  # well under raw chunk size


def test_parse_response_handles_plain_json() -> None:
    parsed = _parse_response('{"answer": "hi [chunk:1]", "insufficient_context": false}')
    assert parsed["answer"] == "hi [chunk:1]"
    assert parsed["insufficient_context"] is False


def test_parse_response_strips_markdown_fence() -> None:
    raw = '```json\n{"answer": "ok [chunk:1]", "insufficient_context": false}\n```'
    parsed = _parse_response(raw)
    assert parsed["answer"] == "ok [chunk:1]"


def test_parse_response_falls_back_for_non_json() -> None:
    """When the model returns prose despite the prompt, we render the prose
    rather than 502 the caller. Citation extraction still works on
    [chunk:N] tags inside the text downstream.
    """
    parsed = _parse_response("Klavis shipped Tuesday [chunk:1].")
    assert parsed["answer"] == "Klavis shipped Tuesday [chunk:1]."
    assert parsed["insufficient_context"] is False


def test_parse_response_falls_back_when_answer_key_missing() -> None:
    parsed = _parse_response('{"foo": "bar"}')
    assert parsed["answer"] == '{"foo": "bar"}'
    assert parsed["insufficient_context"] is False


def test_parse_response_infers_insufficient_context_from_prose() -> None:
    parsed = _parse_response(
        "I cannot answer this question from the provided chunks."
    )
    assert parsed["insufficient_context"] is True


def test_parse_response_strips_anthropic_prefill_double_brace_safely() -> None:
    """When prefill `{` succeeds and Claude returns valid JSON we still parse
    the JSON path. The `{` prefix doesn't break us.
    """
    parsed = _parse_response('{"answer": "ok [chunk:1]", "insufficient_context": false}')
    assert parsed["answer"] == "ok [chunk:1]"


def test_extract_citations_dedupes_and_drops_out_of_range() -> None:
    chunks = [_chunk(1), _chunk(2), _chunk(3)]
    answer = "fact one [chunk:1]. fact two [chunk:2][chunk:1]. invalid [chunk:9]."
    out = _extract_citations(answer, chunks)
    assert out == [
        {"index": 1, "chunk_id": "chunk-1"},
        {"index": 2, "chunk_id": "chunk-2"},
    ]


def test_extract_citations_empty_when_no_tags() -> None:
    chunks = [_chunk(1)]
    assert _extract_citations("plain prose with no citations", chunks) == []


def test_extract_citations_accepts_bare_chunk_n() -> None:
    """Gemini ignores the [bracket] format sometimes — extractor must
    still recognize bare `chunk:1`."""
    chunks = [_chunk(1), _chunk(2)]
    answer = "Granola was added chunk:1. Backfill via poller chunk:2."
    out = _extract_citations(answer, chunks)
    assert {c["index"] for c in out} == {1, 2}


def test_extract_citations_uses_declared_list() -> None:
    """When the model also returns `citations_used: [...]`, indices in that
    list are added even if no inline tag references them."""
    chunks = [_chunk(1), _chunk(2), _chunk(3)]
    out = _extract_citations(
        "answer text with [chunk:1].", chunks, declared=[1, 2, 3]
    )
    assert {c["index"] for c in out} == {1, 2, 3}


def test_normalize_citations_brackets_bare_tags() -> None:
    text = "Granola integration chunk:1 and the poller chunk:2."
    assert (
        normalize_citations_in_answer(text)
        == "Granola integration [chunk:1] and the poller [chunk:2]."
    )


def test_normalize_citations_idempotent_on_already_bracketed() -> None:
    text = "Already done [chunk:1]."
    assert normalize_citations_in_answer(text) == "Already done [chunk:1]."


@pytest.mark.asyncio
async def test_synthesize_short_circuits_on_empty_chunks() -> None:
    result = await synthesize(
        "anything", [], model="anthropic/claude-sonnet-4-6"
    )
    assert result.insufficient_context is True
    assert result.citations == []
    assert result.model == "anthropic/claude-sonnet-4-6"


@pytest.mark.asyncio
async def test_synthesize_rejects_unknown_model() -> None:
    with pytest.raises(SynthesisError) as exc_info:
        await synthesize(
            "q", [_chunk(1)], model="bogus/model-name"
        )
    assert "unsupported synthesis model" in str(exc_info.value)


@pytest.mark.asyncio
async def test_synthesize_anthropic_path(monkeypatch) -> None:
    """End-to-end via mocked Anthropic dispatch — verifies prompt + citation
    extraction against a realistic JSON response.
    """
    captured: dict = {}

    async def fake_call(provider_name, *, system, user, model, max_tokens):
        captured["provider"] = provider_name
        captured["model"] = model
        captured["max_tokens"] = max_tokens
        captured["user"] = user
        return (
            '{"answer": "Klavis shipped on Tuesday [chunk:1]. '
            'It uses MCP [chunk:2].", "insufficient_context": false}'
        )

    monkeypatch.setattr("services.retrieval.synthesis._dispatch", fake_call)
    chunks = [
        _chunk(1, content="Klavis went live Tuesday"),
        _chunk(2, content="Built on top of MCP"),
    ]
    result = await synthesize(
        "what is klavis?",
        chunks,
        model="anthropic/claude-sonnet-4-6",
        max_tokens=200,
    )
    assert captured["provider"] == "anthropic"
    assert captured["model"] == "claude-sonnet-4-6"
    assert captured["max_tokens"] == 200
    assert "what is klavis?" in captured["user"]
    assert "[chunk:1]" in captured["user"]
    assert result.insufficient_context is False
    assert {c["chunk_id"] for c in result.citations} == {"chunk-1", "chunk-2"}


@pytest.mark.asyncio
async def test_synthesize_routes_openai(monkeypatch) -> None:
    captured: dict = {}

    async def fake_call(provider_name, *, system, user, model, max_tokens):
        captured["provider"] = provider_name
        captured["model"] = model
        return '{"answer": "x [chunk:1]", "insufficient_context": false}'

    monkeypatch.setattr("services.retrieval.synthesis._dispatch", fake_call)
    await synthesize(
        "q", [_chunk(1)], model="openai/gpt-4o-mini",
    )
    assert captured["provider"] == "openai"
    assert captured["model"] == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_synthesize_routes_google(monkeypatch) -> None:
    captured: dict = {}

    async def fake_call(provider_name, *, system, user, model, max_tokens):
        captured["provider"] = provider_name
        captured["model"] = model
        return '{"answer": "x [chunk:1]", "insufficient_context": false}'

    monkeypatch.setattr("services.retrieval.synthesis._dispatch", fake_call)
    await synthesize(
        "q", [_chunk(1)], model="google/gemini-2.5-flash",
    )
    assert captured["provider"] == "google"
    assert captured["model"] == "gemini-2.5-flash"


@pytest.mark.asyncio
async def test_synthesize_propagates_insufficient_context(monkeypatch) -> None:
    async def fake_call(provider_name, *, system, user, model, max_tokens):
        return (
            '{"answer": "I cannot answer this from the available chunks.", '
            '"insufficient_context": true}'
        )

    monkeypatch.setattr("services.retrieval.synthesis._dispatch", fake_call)
    result = await synthesize(
        "obscure question",
        [_chunk(1, content="totally unrelated")],
        model="anthropic/claude-sonnet-4-6",
    )
    assert result.insufficient_context is True
