"""Unit tests for the synthesis layer — provider mocked, no network."""

from __future__ import annotations

import pytest

from services.retrieval.synthesis import (
    SynthesisChunk,
    SynthesisError,
    _extract_citations,
    _format_user_prompt,
    _parse_response,
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


def test_parse_response_raises_on_garbage() -> None:
    with pytest.raises(SynthesisError):
        _parse_response("not json at all")


def test_parse_response_raises_when_answer_missing() -> None:
    with pytest.raises(SynthesisError):
        _parse_response('{"foo": "bar"}')


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
