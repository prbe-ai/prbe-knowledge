"""Unit tests for the synthesis layer — provider mocked, no network."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from services.retrieval.synthesis import (
    ANSWER_SCHEMA,
    StreamDelta,
    StreamFinal,
    SynthesisChunk,
    SynthesisError,
    _extract_citations,
    _fallback_parse_text,
    _format_user_prompt,
    _strip_keys_recursive,
    normalize_citations_in_answer,
    synthesize,
    synthesize_stream,
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


# ---------------------------------------------------------------------------
# Schema sanity
# ---------------------------------------------------------------------------


def test_answer_schema_has_strict_shape() -> None:
    """OpenAI strict json_schema mode requires additionalProperties:false
    and every property in the required list."""
    assert ANSWER_SCHEMA["additionalProperties"] is False
    props = set(ANSWER_SCHEMA["properties"].keys())
    required = set(ANSWER_SCHEMA["required"])
    assert props == required == {"answer", "citations_used", "insufficient_context"}


def test_strip_keys_recursive_removes_at_every_level() -> None:
    """Google's response_schema rejects `additionalProperties`. The Google
    adapter strips it before the call; verify the helper handles nesting.
    """
    nested = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "x": {"type": "string", "additionalProperties": False},
            "y": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "z": {"type": "array", "items": {"type": "object", "additionalProperties": False}},
                },
            },
        },
    }
    out = _strip_keys_recursive(nested, ("additionalProperties",))
    # No additionalProperties anywhere in the resulting tree.
    def _walk(node: object) -> None:
        if isinstance(node, dict):
            assert "additionalProperties" not in node
            for v in node.values():
                _walk(v)
        elif isinstance(node, list):
            for v in node:
                _walk(v)

    _walk(out)
    # Original schema unchanged (deep-copy-style).
    assert "additionalProperties" in nested


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Fallback parser (when a provider returns free-form text)
# ---------------------------------------------------------------------------


def test_fallback_parse_handles_plain_json_text() -> None:
    parsed = _fallback_parse_text(
        '{"answer": "hi [chunk:1]", "insufficient_context": false}'
    )
    assert parsed["answer"] == "hi [chunk:1]"
    assert parsed["insufficient_context"] is False


def test_fallback_parse_strips_markdown_fence() -> None:
    raw = '```json\n{"answer": "ok [chunk:1]", "insufficient_context": false}\n```'
    parsed = _fallback_parse_text(raw)
    assert parsed["answer"] == "ok [chunk:1]"


def test_fallback_parse_wraps_prose_as_answer() -> None:
    parsed = _fallback_parse_text("Klavis shipped Tuesday [chunk:1].")
    assert parsed["answer"] == "Klavis shipped Tuesday [chunk:1]."
    assert parsed["insufficient_context"] is False
    assert parsed["citations_used"] == []


def test_fallback_parse_wraps_unrelated_json_as_answer() -> None:
    parsed = _fallback_parse_text('{"foo": "bar"}')
    # No "answer" key in the parsed json → fall through to wrap-as-text path.
    assert parsed["answer"] == '{"foo": "bar"}'


def test_fallback_parse_infers_insufficient_context() -> None:
    parsed = _fallback_parse_text(
        "I cannot answer this question from the provided chunks."
    )
    assert parsed["insufficient_context"] is True


def test_fallback_parse_handles_empty() -> None:
    parsed = _fallback_parse_text("")
    assert parsed["insufficient_context"] is True
    assert "no content" in parsed["answer"].lower()


# ---------------------------------------------------------------------------
# Citation extraction + normalization
# ---------------------------------------------------------------------------


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
    chunks = [_chunk(1), _chunk(2)]
    answer = "Granola was added chunk:1. Backfill via poller chunk:2."
    out = _extract_citations(answer, chunks)
    assert {c["index"] for c in out} == {1, 2}


def test_extract_citations_uses_declared_list() -> None:
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


# ---------------------------------------------------------------------------
# Synthesize end-to-end with mocked _dispatch (returns dict now, not str)
# ---------------------------------------------------------------------------


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
        await synthesize("q", [_chunk(1)], model="bogus/model-name")
    assert "unsupported synthesis model" in str(exc_info.value)


@pytest.mark.asyncio
async def test_synthesize_anthropic_path(monkeypatch) -> None:
    """Mocked dispatch returns a structured dict (the new contract).
    Verifies prompt content + citation extraction.
    """
    captured: dict = {}

    async def fake_call(provider_name, *, system, user, model, max_tokens):
        captured["provider"] = provider_name
        captured["model"] = model
        captured["max_tokens"] = max_tokens
        captured["user"] = user
        return {
            "answer": "Klavis shipped on Tuesday [chunk:1]. It uses MCP [chunk:2].",
            "citations_used": [1, 2],
            "insufficient_context": False,
        }

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
async def test_synthesize_routes_haiku(monkeypatch) -> None:
    captured: dict = {}

    async def fake_call(provider_name, *, system, user, model, max_tokens):
        captured["provider"] = provider_name
        captured["model"] = model
        return {
            "answer": "x [chunk:1]",
            "citations_used": [1],
            "insufficient_context": False,
        }

    monkeypatch.setattr("services.retrieval.synthesis._dispatch", fake_call)
    await synthesize("q", [_chunk(1)], model="anthropic/claude-haiku-4-5-20251001")
    assert captured["provider"] == "anthropic"
    assert captured["model"] == "claude-haiku-4-5-20251001"


@pytest.mark.asyncio
async def test_synthesize_propagates_insufficient_context(monkeypatch) -> None:
    async def fake_call(provider_name, *, system, user, model, max_tokens):
        return {
            "answer": "I cannot answer this from the available chunks.",
            "citations_used": [],
            "insufficient_context": True,
        }

    monkeypatch.setattr("services.retrieval.synthesis._dispatch", fake_call)
    result = await synthesize(
        "obscure question",
        [_chunk(1, content="totally unrelated")],
        model="anthropic/claude-sonnet-4-6",
    )
    assert result.insufficient_context is True


@pytest.mark.asyncio
async def test_synthesize_normalizes_bare_citations(monkeypatch) -> None:
    """Provider returned bare `chunk:1`. synthesize() should canonicalize
    to [chunk:1] before returning."""
    async def fake_call(provider_name, *, system, user, model, max_tokens):
        return {
            "answer": "Klavis shipped chunk:1 and uses MCP chunk:2.",
            "citations_used": [1, 2],
            "insufficient_context": False,
        }

    monkeypatch.setattr("services.retrieval.synthesis._dispatch", fake_call)
    chunks = [_chunk(1), _chunk(2)]
    result = await synthesize(
        "what is klavis?",
        chunks,
        model="anthropic/claude-haiku-4-5-20251001",
    )
    assert "[chunk:1]" in result.answer
    assert "[chunk:2]" in result.answer
    assert "chunk:1." not in result.answer  # bare form was rewritten


# ---------------------------------------------------------------------------
# synthesize_stream — Anthropic streaming path
# ---------------------------------------------------------------------------


class _FakeStream:
    """Mimics the async-context-manager returned by AsyncAnthropic.messages.stream.

    Yields each entry of `texts` as a separate text-stream chunk so we can
    verify synthesize_stream emits one StreamDelta per chunk.
    """

    def __init__(self, texts: list[str]) -> None:
        self._texts = texts

    async def __aenter__(self) -> _FakeStream:
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    @property
    def text_stream(self):
        async def _gen():
            for t in self._texts:
                yield t

        return _gen()


class _FakeMessages:
    def __init__(self, texts: list[str]) -> None:
        self._texts = texts
        self.stream_calls: list[dict] = []

    def stream(self, **kwargs: object) -> _FakeStream:
        self.stream_calls.append(kwargs)
        return _FakeStream(self._texts)


class _FakeAsyncAnthropic:
    """Stand-in for anthropic.AsyncAnthropic. Captures construction kwargs and
    proxies messages.stream to a controllable fake.
    """

    last_instance: _FakeAsyncAnthropic | None = None

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.init_args = args
        self.init_kwargs = kwargs
        # texts is patched in by the test via `_FakeAsyncAnthropic.next_texts`.
        texts = getattr(_FakeAsyncAnthropic, "next_texts", [""])
        self.messages = _FakeMessages(texts)
        _FakeAsyncAnthropic.last_instance = self


@pytest.mark.asyncio
async def test_synthesize_stream_short_circuits_on_empty_chunks() -> None:
    """No chunks → no model call → single StreamFinal flagged insufficient."""
    events = []
    async for evt in synthesize_stream(
        "anything", [], model="anthropic/claude-sonnet-4-6"
    ):
        events.append(evt)
    assert len(events) == 1
    assert isinstance(events[0], StreamFinal)
    assert events[0].insufficient_context is True
    assert events[0].citations == []


@pytest.mark.asyncio
async def test_synthesize_stream_rejects_unknown_model() -> None:
    with pytest.raises(SynthesisError) as exc_info:
        async for _ in synthesize_stream(
            "q", [_chunk(1)], model="bogus/model-name"
        ):
            pass
    assert "unsupported synthesis model" in str(exc_info.value)


@pytest.mark.asyncio
async def test_synthesize_stream_rejects_non_anthropic_model(monkeypatch) -> None:
    """OpenAI / Google models would be valid for synthesize() but not yet
    supported by synthesize_stream — must fail loudly rather than silently
    use Anthropic. Patches SYNTHESIS_MODELS to register a fake openai entry
    so the provider-check branch fires (today's allowlist is Anthropic-only).
    """
    monkeypatch.setitem(
        __import__("services.retrieval.synthesis", fromlist=["SYNTHESIS_MODELS"])
        .SYNTHESIS_MODELS,
        "openai/gpt-4o-mini",
        "openai",
    )
    with pytest.raises(SynthesisError) as exc_info:
        async for _ in synthesize_stream(
            "q", [_chunk(1)], model="openai/gpt-4o-mini"
        ):
            pass
    assert "streaming synthesis only supports Anthropic" in str(exc_info.value)


@pytest.mark.asyncio
async def test_synthesize_stream_yields_deltas_then_final(monkeypatch) -> None:
    """Happy path: streaming chunks arrive as StreamDelta events; the
    accumulated text is parsed for citations and emitted as StreamFinal.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from shared.config import get_settings

    get_settings.cache_clear()  # type: ignore[attr-defined]

    _FakeAsyncAnthropic.next_texts = [
        "Klavis ",
        "shipped Tuesday [chunk:1]. ",
        "It uses MCP [chunk:2].",
    ]

    chunks = [
        _chunk(1, content="Klavis went live Tuesday"),
        _chunk(2, content="Built on top of MCP"),
    ]

    with patch("anthropic.AsyncAnthropic", _FakeAsyncAnthropic):
        events = []
        async for evt in synthesize_stream(
            "what is klavis?",
            chunks,
            model="anthropic/claude-sonnet-4-6",
            max_tokens=200,
        ):
            events.append(evt)

    deltas = [e for e in events if isinstance(e, StreamDelta)]
    finals = [e for e in events if isinstance(e, StreamFinal)]
    assert len(deltas) == 3
    assert [d.text for d in deltas] == _FakeAsyncAnthropic.next_texts
    assert len(finals) == 1
    final = finals[0]
    assert final.insufficient_context is False
    assert final.model == "anthropic/claude-sonnet-4-6"
    assert {c["chunk_id"] for c in final.citations} == {"chunk-1", "chunk-2"}
    assert "[chunk:1]" in final.answer

    # Verify max_tokens + system prompt routed to the SDK.
    assert _FakeAsyncAnthropic.last_instance is not None
    call_kwargs = _FakeAsyncAnthropic.last_instance.messages.stream_calls[0]
    assert call_kwargs["model"] == "claude-sonnet-4-6"
    assert call_kwargs["max_tokens"] == 200


@pytest.mark.asyncio
async def test_synthesize_stream_strips_insufficient_sentinel(monkeypatch) -> None:
    """Model-emitted <<INSUFFICIENT>> sentinel sets the flag and is stripped
    from the final answer text.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    from shared.config import get_settings

    get_settings.cache_clear()  # type: ignore[attr-defined]

    _FakeAsyncAnthropic.next_texts = [
        "<<INSUFFICIENT>>\n",
        "The chunks don't mention Klavis at all.",
    ]

    with patch("anthropic.AsyncAnthropic", _FakeAsyncAnthropic):
        events = []
        async for evt in synthesize_stream(
            "what is klavis?",
            [_chunk(1, content="totally unrelated")],
            model="anthropic/claude-sonnet-4-6",
        ):
            events.append(evt)

    final = next(e for e in events if isinstance(e, StreamFinal))
    assert final.insufficient_context is True
    assert "<<INSUFFICIENT>>" not in final.answer
    assert final.answer.startswith("The chunks don't mention")
