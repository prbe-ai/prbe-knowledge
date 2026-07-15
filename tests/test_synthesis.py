"""Unit tests for the synthesis layer — provider mocked, no network."""

from __future__ import annotations

import pytest

from engine.retrieval.synthesis import (
    ANSWER_SCHEMA,
    StreamDelta,
    StreamFinal,
    SynthesisChunk,
    SynthesisError,
    _build_streaming_system_prompt,
    _build_system_prompt,
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


def test_format_user_prompt_renders_chain_section_when_graph_evidence_present() -> None:
    """`graph_evidence` on a chunk is the chain rationale (anchor doc +
    edge_type + `why`). Without rendering it in the user prompt the LLM
    answer drops the chain — saw this live on "why was PR 72 created":
    the Linear ticket was in the chunks but the synthesized answer said
    "no explicit ticket triggered the PR" because the edge rationale
    wasn't in its prompt. Lock the rendering so we don't regress."""
    from engine.shared.models import GraphEvidence

    chunk = SynthesisChunk(
        chunk_id="c-linear",
        title="linear tickets not enriched",
        content="customers report empty enrichment on linear tickets",
        source_system="linear",
        source_url="https://linear.app/x",
        updated_at=_NOW_ISO,
        graph_evidence=[
            GraphEvidence(
                edge_type="motivates_pr",
                confidence="INFERRED",
                via_entity="github:prbe-ai/prbe-backend:pr:72",
                reason="PR #72 implements the proxy that fixes the 502s causing linear tickets to not be enriched",
            )
        ],
    )
    out = _format_user_prompt("why was pr72 created", [chunk])
    assert "CHAIN:" in out
    assert "github:prbe-ai/prbe-backend:pr:72" in out
    assert "motivates_pr" in out
    assert "INFERRED" in out
    assert "implements the proxy that fixes" in out


def test_format_user_prompt_truncates_long_chain_reasons() -> None:
    """Per-edge `reason` cap protects the synthesizer's token budget when
    the LLM-derived `why` string is long. The cap lives on the rendering
    side, not the wire — full `why` survives in graph_evidence for
    other consumers (dashboard chain panel, MCP graph_evidence field)."""
    from engine.shared.models import GraphEvidence

    long_why = "a" * 5000
    chunk = SynthesisChunk(
        chunk_id="c-x", title=None, content="...", source_system="github",
        source_url="x", updated_at=_NOW_ISO,
        graph_evidence=[GraphEvidence(
            edge_type="e", confidence="INFERRED", via_entity="anchor", reason=long_why,
        )],
    )
    out = _format_user_prompt("q", [chunk])
    # Rendered output stays manageable — the long reason is truncated.
    chain_section_size = out.split("CHAIN:", 1)[1] if "CHAIN:" in out else ""
    assert len(chain_section_size) < 1000


def test_format_user_prompt_skips_chain_section_when_no_evidence() -> None:
    """Vector/BM25-only chunks have empty graph_evidence — the CHAIN
    section is omitted entirely to keep the prompt tight on queries
    where there's no chain to render."""
    out = _format_user_prompt("q", [_chunk(1, content="body")])
    assert "CHAIN:" not in out


def test_system_prompt_carries_source_preference_rule() -> None:
    """The synthesis LLM was returning empty answers when an agent-
    session chunk (`claude_code:*`) landed as the primary doc and the
    chunk content was meta-commentary about Probe rather than factual
    content. The system prompt now instructs the LLM to treat
    authoritative source docs (linear/notion/slack/github/wiki) as
    primary truth and use session transcripts as supporting context
    only. Verified 2026-05-20 against the multi-granola chronology
    query — when curation lands on the live debugging session, the
    prompt should keep the LLM from refusing to answer."""
    from datetime import UTC, datetime

    sp = _build_system_prompt(datetime(2026, 5, 20, tzinfo=UTC))
    assert "Source-preference rule" in sp
    # Anchors LLM on the chunk-header `source:` field, not body-text guessing
    assert "`source:`" in sp
    # Authoritative + session source-system identifiers both named
    assert "claude_code" in sp and "codex" in sp
    assert "linear" in sp and "notion" in sp and "slack" in sp
    # Explicit session-query exception
    assert "EXPLICITLY asking about" in sp or "explicitly asking about" in sp.lower()
    # Don't-refuse-on-session-meta-text clause
    assert "Session meta-text" in sp
    # All-sessions fallback (no authoritative present)
    assert "NO authoritative" in sp or "no authoritative" in sp.lower()


def test_streaming_system_prompt_matches_nonstreaming_rule() -> None:
    """Both prompt variants must carry identical Source-preference rule
    text. Pre-fix the streaming variant dropped two clauses (the
    explicit-session-query exception and the chunk-#1-fallback), which
    would have over-suppressed legitimate session-focused queries on
    the dashboard streaming path. Pin that the shared rule constant
    flows through to both prompts identically."""
    from datetime import UTC, datetime

    now = datetime(2026, 5, 20, tzinfo=UTC)
    sp_nonstream = _build_system_prompt(now)
    sp_stream = _build_streaming_system_prompt(now)
    # Both must carry the same anchor signals
    for marker in (
        "Source-preference rule",
        "`source:`",
        "claude_code",
        "codex",
        "linear",
        "notion",
        "EXPLICITLY asking about",
        "Session meta-text",
        "NO authoritative",
    ):
        assert marker in sp_nonstream, f"non-streaming prompt missing: {marker}"
        assert marker in sp_stream, f"streaming prompt missing: {marker}"


def test_format_user_prompt_renders_neighbor_metadata_for_chronology() -> None:
    """The synthesis LLM previously refused chronology queries
    (answer="" on "reconstruct the multi-granola timeline") because
    chain neighbors were rendered as opaque canonical_ids with no
    date/source/url. The renderer now surfaces title + source +
    created_at (date precision) + url alongside the edge metadata so
    the LLM has the temporal + provenance grounding it needs."""
    from datetime import UTC, datetime

    from engine.shared.models import GraphEvidence

    chunk = SynthesisChunk(
        chunk_id="c-linear",
        title="Multi-Granola — End-to-End Implementation Plan",
        content="PRB-18 plan body...",
        source_system="linear",
        source_url="https://linear.app/prbe/issue/PRB-18",
        updated_at=_NOW_ISO,
        graph_evidence=[
            GraphEvidence(
                edge_type="DISCUSSES",
                confidence="INFERRED",
                via_entity="slack:T0:C0:1778043682.343959",
                via_entity_title="thread 1: Mahit raises Granola sync gap",
                via_entity_source_system="slack",
                via_entity_created_at=datetime(2026, 5, 4, 17, 0, 0, tzinfo=UTC),
                via_entity_url="https://slack.com/archives/C0/p1778043682343959",
                reason="The Slack thread raises the original gap.",
            ),
        ],
    )
    out = _format_user_prompt("reconstruct the multi-granola timeline", [chunk])
    # Title is the human-readable display
    assert "thread 1: Mahit raises Granola sync gap" in out
    # Source + created-date + url all appear so the LLM can order + cite
    assert "slack" in out
    assert "2026-05-04" in out
    assert "https://slack.com/archives/C0/p1778043682343959" in out
    # Canonical_id is still present as a stable handle
    assert "slack:T0:C0:1778043682.343959" in out


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


def test_normalize_citations_splits_multi_chunk_brackets() -> None:
    # Gemini in particular emits `[chunk:1, 5, 7]` when the prompt asks for
    # "one or more inline citations". Without splitting, _CITATION_RE eats
    # `[chunk:1` and leaves `, 5, 7]` dangling in the rendered answer.
    text = "Granola is shipped [chunk:1, 5, 7] and on prod [chunk:2, 4]."
    assert normalize_citations_in_answer(text) == (
        "Granola is shipped [chunk:1][chunk:5][chunk:7] and on prod "
        "[chunk:2][chunk:4]."
    )


def test_normalize_citations_handles_multi_chunk_with_whitespace() -> None:
    text = "Spaced [chunk: 1 ,  5 , 7 ]."
    assert normalize_citations_in_answer(text) == (
        "Spaced [chunk:1][chunk:5][chunk:7]."
    )


def test_extract_citations_finds_all_in_multi_chunk() -> None:
    chunks = [_chunk(1), _chunk(2), _chunk(3), _chunk(4), _chunk(5)]
    answer = normalize_citations_in_answer("fact [chunk:1, 3, 5].")
    out = _extract_citations(answer, chunks)
    assert {c["index"] for c in out} == {1, 3, 5}


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

    monkeypatch.setattr("engine.retrieval.synthesis._dispatch", fake_call)
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

    monkeypatch.setattr("engine.retrieval.synthesis._dispatch", fake_call)
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

    monkeypatch.setattr("engine.retrieval.synthesis._dispatch", fake_call)
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

    monkeypatch.setattr("engine.retrieval.synthesis._dispatch", fake_call)
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
# synthesize_stream — LiteLLM-routed streaming path (Phase-0b chunk D)
# ---------------------------------------------------------------------------
#
# `synthesize_stream` now routes through `shared.llm.acompletion(stream=True)`
# regardless of provider. LiteLLM normalizes Anthropic + Google streaming into
# one OpenAI-shaped async-iterator of chunks: each chunk exposes
# `chunk.choices[0].delta.content` (string or None) and the final chunk may
# carry `chunk.usage`. The test fakes that contract.


class _FakeDelta:
    def __init__(self, content: str | None) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str | None) -> None:
        self.delta = _FakeDelta(content)


class _FakeChunk:
    def __init__(self, content: str | None, usage: object | None = None) -> None:
        self.choices = [_FakeChoice(content)]
        if usage is not None:
            self.usage = usage


class _FakeUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


async def _fake_chunk_iter(texts: list[str], with_final_usage: bool = True):
    """Yield one OpenAI-shaped chunk per text, then optionally one trailing
    empty-delta chunk carrying `.usage` (matches LiteLLM's wire shape — the
    last chunk has no content delta but does carry token accounting).
    """
    for t in texts:
        yield _FakeChunk(content=t)
    if with_final_usage:
        yield _FakeChunk(
            content=None,
            usage=_FakeUsage(prompt_tokens=10, completion_tokens=20),
        )


class _AcompletionRecorder:
    """Stand-in for `shared.llm.acompletion` that records call kwargs and
    returns a controllable async-chunk iterator on each invocation.
    """

    def __init__(self, texts: list[str]) -> None:
        self.texts = texts
        self.calls: list[dict[str, object]] = []

    async def __call__(self, **kwargs: object):
        self.calls.append(kwargs)
        return _fake_chunk_iter(self.texts)


@pytest.mark.asyncio
async def test_synthesize_stream_short_circuits_on_empty_chunks() -> None:
    """No chunks -> no model call -> single StreamFinal flagged insufficient."""
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
async def test_synthesize_stream_rejects_unsupported_provider(monkeypatch) -> None:
    """Streaming today supports anthropic + google. OpenAI (or any other
    provider in SYNTHESIS_MODELS) must fail loudly rather than silently
    falling through to one of the supported branches. Patches
    SYNTHESIS_MODELS to register a fake openai entry so the provider-check
    branch fires.
    """
    monkeypatch.setitem(
        __import__("engine.retrieval.synthesis", fromlist=["SYNTHESIS_MODELS"])
        .SYNTHESIS_MODELS,
        "openai/gpt-4o-mini",
        "openai",
    )
    with pytest.raises(SynthesisError) as exc_info:
        async for _ in synthesize_stream(
            "q", [_chunk(1)], model="openai/gpt-4o-mini"
        ):
            pass
    assert "streaming synthesis only supports Anthropic and Google" in str(
        exc_info.value
    )


@pytest.mark.asyncio
async def test_synthesize_stream_yields_deltas_then_final_anthropic(
    monkeypatch,
) -> None:
    """Happy path on the Anthropic provider: streaming chunks arrive as
    StreamDelta events; the accumulated text is parsed for citations and
    emitted as StreamFinal. Verifies the LiteLLM call is invoked with the
    right model prefix and messages shape.
    """
    fake = _AcompletionRecorder(
        texts=[
            "Klavis ",
            "shipped Tuesday [chunk:1]. ",
            "It uses MCP [chunk:2].",
        ]
    )
    monkeypatch.setattr("engine.retrieval.synthesis.shared_llm.acompletion", fake)

    chunks = [
        _chunk(1, content="Klavis went live Tuesday"),
        _chunk(2, content="Built on top of MCP"),
    ]

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
    assert [d.text for d in deltas] == fake.texts
    assert len(finals) == 1
    final = finals[0]
    assert final.insufficient_context is False
    assert final.model == "anthropic/claude-sonnet-4-6"
    assert {c["chunk_id"] for c in final.citations} == {"chunk-1", "chunk-2"}
    assert "[chunk:1]" in final.answer

    # LiteLLM call shape: provider-prefixed model id (split from the
    # SYNTHESIS_MODELS key), OpenAI-style messages, stream=True,
    # max_tokens forwarded. No reasoning_effort on Anthropic.
    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["model"] == "anthropic/claude-sonnet-4-6"
    assert call["stream"] is True
    assert call["max_tokens"] == 200
    assert "reasoning_effort" not in call
    messages = call["messages"]
    assert isinstance(messages, list) and len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "what is klavis?" in messages[1]["content"]


@pytest.mark.asyncio
async def test_synthesize_stream_yields_deltas_then_final_google(
    monkeypatch,
) -> None:
    """Google streaming path goes through the same LiteLLM call, with the
    Gemini-specific `reasoning_effort="none"` knob (the chunk-D replacement
    for the legacy `thinking_config: {thinking_budget: 0}`).
    """
    fake = _AcompletionRecorder(
        texts=["Gemini ", "answered [chunk:1]."]
    )
    monkeypatch.setattr("engine.retrieval.synthesis.shared_llm.acompletion", fake)

    chunks = [_chunk(1, content="some context")]

    events = []
    async for evt in synthesize_stream(
        "q?",
        chunks,
        model="google/gemini-3-flash-preview",
        max_tokens=150,
    ):
        events.append(evt)

    deltas = [e for e in events if isinstance(e, StreamDelta)]
    assert [d.text for d in deltas] == ["Gemini ", "answered [chunk:1]."]

    assert len(fake.calls) == 1
    call = fake.calls[0]
    # SYNTHESIS_MODELS keys use `google/` (internal dispatch tag) but
    # the LiteLLM model id translates to `gemini/<id>` — that's the
    # only prefix LiteLLM routes via AI Studio API-key auth. `google/`
    # and bare ids both go to Vertex AI (full GCP creds required), so
    # the translation in synthesize_stream is load-bearing.
    assert call["model"] == "gemini/gemini-3-flash-preview"
    assert call["stream"] is True
    assert call["max_tokens"] == 150
    # reasoning_effort="none" is the LiteLLM-canonical replacement for
    # `thinking_config: {thinking_budget: 0}`. On Gemini 3+ it maps to
    # `thinking_level="minimal"` (true budget=0 is unavailable on 3.x).
    assert call["reasoning_effort"] == "none"


@pytest.mark.asyncio
async def test_synthesize_stream_strips_insufficient_sentinel(monkeypatch) -> None:
    """Model-emitted <<INSUFFICIENT>> sentinel sets the flag and is stripped
    from the final answer text.
    """
    fake = _AcompletionRecorder(
        texts=[
            "<<INSUFFICIENT>>\n",
            "The chunks don't mention Klavis at all.",
        ]
    )
    monkeypatch.setattr("engine.retrieval.synthesis.shared_llm.acompletion", fake)

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


@pytest.mark.asyncio
async def test_synthesize_stream_wraps_litellm_error(monkeypatch) -> None:
    """`shared.llm.LLMError` raised during streaming surfaces as
    SynthesisError so the retrieval route handler can convert to 502.
    """
    from engine.shared.llm import LLMError

    async def raising_acompletion(**_kwargs: object):
        raise LLMError("upstream 429", status_code=429, provider="anthropic")

    monkeypatch.setattr(
        "engine.retrieval.synthesis.shared_llm.acompletion",
        raising_acompletion,
    )

    with pytest.raises(SynthesisError) as exc_info:
        async for _ in synthesize_stream(
            "q",
            [_chunk(1)],
            model="anthropic/claude-sonnet-4-6",
        ):
            pass
    assert "streaming api error" in str(exc_info.value)


@pytest.mark.asyncio
async def test_synthesize_stream_tolerates_none_content_chunks(
    monkeypatch,
) -> None:
    """LiteLLM yields a trailing chunk with `delta.content=None` carrying
    `.usage` for the final accounting. The streamer must skip it (no empty
    StreamDelta) and still emit exactly one StreamFinal.
    """
    fake = _AcompletionRecorder(texts=["Only ", "two ", "real ", "deltas."])
    monkeypatch.setattr("engine.retrieval.synthesis.shared_llm.acompletion", fake)

    events = []
    async for evt in synthesize_stream(
        "q",
        [_chunk(1, content="x")],
        model="anthropic/claude-sonnet-4-6",
    ):
        events.append(evt)

    deltas = [e for e in events if isinstance(e, StreamDelta)]
    finals = [e for e in events if isinstance(e, StreamFinal)]
    # Four real text chunks + one usage-only chunk -> still 4 deltas + 1 final.
    assert len(deltas) == 4
    assert len(finals) == 1
