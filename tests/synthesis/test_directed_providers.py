"""Unit tests for the DirectedPhrasesProvider abstraction in
`services/synthesis/providers.py`.

Covers:
  - factory dispatch by model name (Anthropic vs Gemini, unknown raises)
  - _AnthropicDirectedPhrases parsing (happy + 3 error paths) — via
    LiteLLM-shaped response mocking after Phase-0b chunk C
  - _GeminiDirectedPhrases parsing (delegates to _gemini_call_json)
  - _coerce_phrases (length cap, max-count cap, type coercion)
  - _thinking_budget_for (per-model rule)

DB and live LLM calls are NOT exercised here. End-to-end coverage lives
in tests/synthesis/test_directed_phrases.py (orchestrator) and
tests/retrieval/test_search_pipeline_directed.py (retrieval e2e).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import orjson
import pytest

from engine.shared.constants import (
    HAIKU_MODEL,
    MAX_DIRECTED_PHRASE_CHARS,
    MAX_DIRECTED_VECTORS_PER_DOC,
)
from kb.synthesis.providers import (
    DirectedPhrasesParseError,
    _AnthropicDirectedPhrases,
    _coerce_phrases,
    _GeminiDirectedPhrases,
    _thinking_budget_for,
    get_directed_phrases_provider,
)

# ---- _coerce_phrases (rule-based, no LLM) --------------------------------


def test_coerce_phrases_happy_path() -> None:
    out = _coerce_phrases(["alpha", "beta", "  gamma  "])
    assert out == ["alpha", "beta", "gamma"]


def test_coerce_phrases_drops_empty_and_non_strings() -> None:
    out = _coerce_phrases(["alpha", "", "  ", 42, None, "beta"])
    assert out == ["alpha", "beta"]


def test_coerce_phrases_drops_oversize() -> None:
    long = "x" * (MAX_DIRECTED_PHRASE_CHARS + 10)
    out = _coerce_phrases(["short", long, "also short"])
    assert "short" in out
    assert "also short" in out
    assert long not in out


def test_coerce_phrases_caps_count() -> None:
    out = _coerce_phrases([f"phrase {i}" for i in range(MAX_DIRECTED_VECTORS_PER_DOC + 5)])
    assert len(out) == MAX_DIRECTED_VECTORS_PER_DOC


def test_coerce_phrases_rejects_non_list() -> None:
    with pytest.raises(DirectedPhrasesParseError):
        _coerce_phrases("not a list")  # type: ignore[arg-type]
    with pytest.raises(DirectedPhrasesParseError):
        _coerce_phrases({"phrases": ["alpha"]})  # type: ignore[arg-type]


# ---- _thinking_budget_for ------------------------------------------------


def test_thinking_budget_zero_for_flash_and_flash_lite() -> None:
    # The eval at scripts/eval_directed_phrases.py confirmed Flash and
    # Flash Lite both work with budget=0; pin that contract.
    assert _thinking_budget_for("gemini-3-flash-preview") == 0
    assert _thinking_budget_for("gemini-3.1-flash-lite") == 0
    assert _thinking_budget_for("gemini-flash-lite") == 0


def test_thinking_budget_nonzero_for_pro() -> None:
    # Pro REJECTS budget=0 (NeonDbError-style 400 in the eval). Must
    # carry slack.
    assert _thinking_budget_for("gemini-3.1-pro-preview") > 0
    assert _thinking_budget_for("gemini-3-pro") > 0


# ---- _AnthropicDirectedPhrases ------------------------------------------


def _litellm_tool_response(tool_name: str, payload: dict) -> SimpleNamespace:
    """LiteLLM-shaped response with one forced tool call."""
    func = SimpleNamespace(
        name=tool_name,
        arguments=orjson.dumps(payload).decode("utf-8"),
    )
    call = SimpleNamespace(type="function", function=func)
    message = SimpleNamespace(content=None, tool_calls=[call])
    choice = SimpleNamespace(message=message, finish_reason="tool_calls")
    return SimpleNamespace(choices=[choice], usage=None)


def _litellm_no_tool_call_response() -> SimpleNamespace:
    """LiteLLM-shaped response with no tool_calls (model emitted text only)."""
    message = SimpleNamespace(content="oops, no tool call", tool_calls=None)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], usage=None)


@pytest.mark.asyncio
async def test_anthropic_directed_happy_path(monkeypatch) -> None:
    fake = AsyncMock(
        return_value=_litellm_tool_response(
            "record_directed_phrases",
            {"phrases": ["alpha", "beta", "gamma"]},
        )
    )
    monkeypatch.setattr("engine.shared.llm_tools.acompletion", fake)
    provider = _AnthropicDirectedPhrases(model=HAIKU_MODEL)
    out = await provider.generate(page_title="t", page_body="b")
    assert out == ["alpha", "beta", "gamma"]


@pytest.mark.asyncio
async def test_anthropic_directed_no_tool_use_block_raises(monkeypatch) -> None:
    fake = AsyncMock(return_value=_litellm_no_tool_call_response())
    monkeypatch.setattr("engine.shared.llm_tools.acompletion", fake)
    provider = _AnthropicDirectedPhrases(model=HAIKU_MODEL)
    with pytest.raises(DirectedPhrasesParseError, match=r"no .* tool_use block"):
        await provider.generate(page_title="t", page_body="b")


@pytest.mark.asyncio
async def test_anthropic_directed_input_not_a_dict_raises(monkeypatch) -> None:
    """LiteLLM's `function.arguments` is required to be a JSON-string
    representing an object. A non-object payload should raise the
    same parse error as the SDK-shaped path."""
    func = SimpleNamespace(
        name="record_directed_phrases",
        arguments=orjson.dumps(["not", "a", "dict"]).decode("utf-8"),
    )
    call = SimpleNamespace(type="function", function=func)
    message = SimpleNamespace(content=None, tool_calls=[call])
    choice = SimpleNamespace(message=message, finish_reason="tool_calls")
    response = SimpleNamespace(choices=[choice], usage=None)
    fake = AsyncMock(return_value=response)
    monkeypatch.setattr("engine.shared.llm_tools.acompletion", fake)
    provider = _AnthropicDirectedPhrases(model=HAIKU_MODEL)
    with pytest.raises(DirectedPhrasesParseError):
        await provider.generate(page_title="t", page_body="b")


@pytest.mark.asyncio
async def test_anthropic_directed_phrases_not_a_list_raises(monkeypatch) -> None:
    # _coerce_phrases rejects non-list payloads with the same exception.
    fake = AsyncMock(
        return_value=_litellm_tool_response(
            "record_directed_phrases",
            {"phrases": "not a list"},
        )
    )
    monkeypatch.setattr("engine.shared.llm_tools.acompletion", fake)
    provider = _AnthropicDirectedPhrases(model=HAIKU_MODEL)
    with pytest.raises(DirectedPhrasesParseError, match="not a list"):
        await provider.generate(page_title="t", page_body="b")


# ---- _GeminiDirectedPhrases ----------------------------------------------


@pytest.mark.asyncio
async def test_gemini_directed_happy_path() -> None:
    """`_gemini_call_json` is the only thing that talks to the network.
    Patch it; verify _GeminiDirectedPhrases passes correct args + parses
    the returned dict.
    """
    with patch(
        "kb.synthesis.providers._gemini_call_json",
        new=AsyncMock(return_value={"phrases": ["alpha", "beta"]}),
    ) as mock_call:
        provider = _GeminiDirectedPhrases(model="gemini-3-flash-preview")
        out = await provider.generate(page_title="t", page_body="b")
    assert out == ["alpha", "beta"]
    # Call args carry the model + flattened prompt.
    kwargs = mock_call.await_args.kwargs
    assert kwargs["model"] == "gemini-3-flash-preview"
    assert kwargs["schema"]["properties"]["phrases"]["type"] == "array"


@pytest.mark.asyncio
async def test_gemini_directed_underlying_failure_wraps_as_parse_error() -> None:
    """A network/SDK error from `_gemini_call_json` propagates as
    `DirectedPhrasesParseError` so callers can match a single exception
    type regardless of provider. Pins the 'silent partial state' fix
    from the directed_phrases.persist orchestrator.
    """
    with patch(
        "kb.synthesis.providers._gemini_call_json",
        new=AsyncMock(side_effect=RuntimeError("gemini 5xx")),
    ):
        provider = _GeminiDirectedPhrases(model="gemini-3-flash-preview")
        with pytest.raises(DirectedPhrasesParseError, match=r"gemini.*call failed"):
            await provider.generate(page_title="t", page_body="b")


# ---- factory dispatch ----------------------------------------------------


def test_factory_dispatch_gemini_flash() -> None:
    provider = get_directed_phrases_provider(model_override="gemini-3-flash-preview")
    assert isinstance(provider, _GeminiDirectedPhrases)
    # The canonical id is what gets sent to the API. Pin it so a future
    # alias-table edit doesn't silently route flash traffic elsewhere.
    assert provider._model == "gemini-3-flash-preview"


def test_factory_resolves_short_aliases_to_canonical_ids() -> None:
    """REGRESSION for the dead-branch alias bug.
    Before fix: factory used `name if name.startswith("gemini") else <fallback>`
    which sent the alias string verbatim as the API model id (4xx in prod).
    """
    # Flash short aliases:
    for alias in ("gemini-flash", "gemini-3-flash"):
        provider = get_directed_phrases_provider(model_override=alias)
        assert isinstance(provider, _GeminiDirectedPhrases)
        assert provider._model == "gemini-3-flash-preview", (
            f"alias {alias!r} must resolve to canonical id"
        )
    # Flash-Lite short aliases:
    for alias in ("gemini-flash-lite",):
        provider = get_directed_phrases_provider(model_override=alias)
        assert isinstance(provider, _GeminiDirectedPhrases)
        assert provider._model == "gemini-3.1-flash-lite"


def test_factory_dispatch_gemini_flash_lite_alias() -> None:
    # Explicit alias lets a future Flash -> Flash Lite flip be a one-line
    # constants.py change with no provider edit.
    provider = get_directed_phrases_provider(
        model_override="gemini-3.1-flash-lite"
    )
    assert isinstance(provider, _GeminiDirectedPhrases)
    assert provider._model == "gemini-3.1-flash-lite"


def test_factory_dispatch_anthropic_constructs_provider_without_client() -> None:
    """Post-Phase-0b: the Anthropic directed provider no longer needs a
    caller-supplied AsyncAnthropic. Every call routes through
    `shared.llm.acompletion`, which picks up `ANTHROPIC_API_KEY` from
    the env or `LLM_GATEWAY_URL`. The factory just returns a provider
    instance — no lazy client construction, no api-key check at
    factory time.
    """
    provider = get_directed_phrases_provider(model_override="haiku")
    assert isinstance(provider, _AnthropicDirectedPhrases)


def test_factory_dispatch_anthropic_with_client_passes_through() -> None:
    """A caller-supplied `anthropic_client` is accepted (legacy shape)
    and stored on the provider for any future inspection — but it's
    no longer consumed. Pins the call-site backward-compat contract.
    """
    sentinel = object()
    provider = get_directed_phrases_provider(
        anthropic_client=sentinel, model_override="haiku"
    )
    assert isinstance(provider, _AnthropicDirectedPhrases)
    assert provider._client is sentinel


def test_factory_dispatch_unknown_model_raises_loudly() -> None:
    """Loud failure at deploy time beats a silent fallback that ships
    the wrong provider for weeks unnoticed."""
    with pytest.raises(ValueError, match="unknown DIRECTED_PHRASES_MODEL"):
        get_directed_phrases_provider(model_override="some-unreleased-model")


# ---- Regression tests for missing-key + parse failures ------------------


@pytest.mark.asyncio
async def test_gemini_directed_missing_phrases_key_raises_not_silent_empty(
    monkeypatch,
) -> None:
    """REGRESSION for the silent-prior-row-purge bug.
    Before fix: payload.get("phrases", []) returned [] on key drift, the
    orchestrator treated this as "successful zero phrases" and called
    _reconcile_llm with keep=[], DELETING all prior LLM rows. A prompt
    drift to {"trigger_phrases": ...} would silently wipe every doc's
    directed_vectors with no warning.

    After fix: missing key raises DirectedPhrasesParseError, which the
    orchestrator catches and flips llm_failed=True, preserving prior rows.
    """
    # Simulate Gemini returning a different key (prompt/schema drift).
    with patch(
        "kb.synthesis.providers._gemini_call_json",
        new=AsyncMock(return_value={"trigger_phrases": ["a", "b"]}),
    ):
        provider = _GeminiDirectedPhrases(model="gemini-3-flash-preview")
        with pytest.raises(DirectedPhrasesParseError, match=r"missing 'phrases' key"):
            await provider.generate(page_title="t", page_body="b")


@pytest.mark.asyncio
async def test_anthropic_directed_missing_phrases_key_raises_not_silent_empty(
    monkeypatch,
) -> None:
    """Symmetric to the Gemini missing-key test. Anthropic providers go
    through the same _coerce_phrases path and should also raise rather
    than silently return [].
    """
    fake = AsyncMock(
        return_value=_litellm_tool_response(
            "record_directed_phrases",
            {"some_other_key": ["a"]},  # missing 'phrases'
        )
    )
    monkeypatch.setattr("engine.shared.llm_tools.acompletion", fake)
    provider = _AnthropicDirectedPhrases(model=HAIKU_MODEL)
    with pytest.raises(DirectedPhrasesParseError, match=r"missing 'phrases' key"):
        await provider.generate(page_title="t", page_body="b")
